"""Bounded storage admission, continuous checks, and owned-process cancellation.

Only application writers using this guard are bounded. Concurrent external writes
and arbitrary installers require operating-system quotas for a hard guarantee.
No GPU sampling or automatic deletion of unrelated/partial downloads occurs here.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid

GIB = 2**30


class ResourceStop(RuntimeError):
    """A job must stop without claiming completion; safe to serialize in evidence."""
    def __init__(self, reason, *, operation=None, volumes=()):
        self.reason, self.operation, self.volumes = reason, operation, list(volumes)
        super().__init__(reason)

    def to_dict(self):
        return dict(status="resource_stopped", code="storage_reserve", reason=self.reason,
                    operation=self.operation, volumes=self.volumes)


@dataclass(frozen=True)
class Volume:
    id: str
    path: str
    kind: str = "filesystem"
    resolution: str = "resolved"


def _existing_parent(path):
    path = Path(path).expanduser().resolve()
    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return path, parent


@lru_cache(maxsize=1)
def wsl_backing_volume():
    """Resolve the current WSL distro VHD's host drive, with an explicit override.

    If registry lookup is unavailable, the C: fallback is marked conservative,
    never described as an observed physical mapping. Installations with a moved
    distro should set OPIUM_WSL_BACKING_VOLUME if Windows interop is disabled.
    """
    override = os.environ.get("OPIUM_WSL_BACKING_VOLUME")
    if override:
        path = Path(override).resolve()
        if not re.fullmatch(r"/mnt/[a-zA-Z]", str(path)):
            raise ValueError("OPIUM_WSL_BACKING_VOLUME must name a /mnt/<drive> mount")
        return str(path), "configured"
    distro = os.environ.get("WSL_DISTRO_NAME")
    if not distro and not Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists():
        return None
    executable = shutil.which("powershell.exe")
    fallback = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not executable and fallback.is_file():
        executable = str(fallback)
    if executable:
        # The script is constant; no shell interpolation of user/distro names.
        script = ("Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss | "
                  "ForEach-Object { Get-ItemProperty $_.PSPath } | "
                  "Select-Object DistributionName,BasePath | ConvertTo-Json -Compress")
        try:
            raw = subprocess.run([executable, "-NoProfile", "-NonInteractive", "-Command", script],
                                 capture_output=True, text=True, timeout=3, check=True)
            rows = json.loads(raw.stdout.lstrip("\ufeff"))
            rows = rows if isinstance(rows, list) else [rows]
            for row in rows:
                if row.get("DistributionName") == distro:
                    match = re.match(r"^(?:\\\\\?\\)?([A-Za-z]):", row.get("BasePath", ""))
                    if match:
                        return f"/mnt/{match[1].lower()}", "registry"
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    if Path("/mnt/c").exists():
        return "/mnt/c", "conservative_fallback"
    raise ResourceStop("Cannot resolve the WSL backing volume; configure OPIUM_WSL_BACKING_VOLUME")


def resolve_volumes(path, *, backing=None):
    """Return every capacity pool consumed by a write, deduplicated by identity."""
    resolved, existing = _existing_parent(path)
    drive = re.match(r"^/mnt/([A-Za-z])(?:/|$)", str(resolved))
    if drive:
        letter = drive[1].lower()
        if not Path(f"/mnt/{letter}").is_dir():
            raise ResourceStop(f"Destination drive /mnt/{letter} is unavailable")
        return (Volume(f"windows:{letter}", f"/mnt/{letter}", "host_volume"),)
    if os.name == "nt":
        drive = resolved.drive.casefold()
        return (Volume(f"windows:{drive}", str(resolved.anchor), "host_volume"),)
    volumes = [Volume(f"device:{existing.stat().st_dev}", str(existing), "filesystem")]
    # A mounted external Linux device is not a growing WSL root virtual disk.
    root_device = Path("/").stat().st_dev
    if existing.stat().st_dev == root_device:
        host = backing if backing is not None else wsl_backing_volume()
        if host:
            host_path, resolution = host
            letter = Path(host_path).name.lower()
            volumes.append(Volume(f"windows:{letter}", str(host_path), "wsl_backing", resolution))
    return tuple(volumes)


def disk_telemetry(path):
    _, existing = _existing_parent(path)
    usage = shutil.disk_usage(existing)
    return dict(free_bytes=usage.free, total_bytes=usage.total)


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class ResourceGuard:
    """One reservation ledger for all controlled writers in a service process.

    reserve() claims planned final AND temporary bytes. before_write()/written()
    bound individual writes and release consumed reservation after the physical
    write. A reservation is outstanding capacity, not a promise against external
    writers. Call check(force=True) during blocked child jobs via supervise_process.
    """
    def __init__(self, destinations=None, *, reserve_bytes=10 * GIB,
                 emergency_bytes=1024**2, max_write_bytes=4 * 1024**2,
                 interval_seconds=1.0, forbid_large_c_writes=True,
                 large_write_bytes=64 * 1024**2, telemetry=disk_telemetry,
                 resolver=resolve_volumes, clock=time.monotonic):
        self.reserve_bytes = _bytes(reserve_bytes, "reserve_bytes")
        self.emergency_bytes = _bytes(emergency_bytes, "emergency_bytes")
        self.max_write_bytes = _bytes(max_write_bytes, "max_write_bytes")
        if not self.max_write_bytes or not 0 < interval_seconds <= 60:
            raise ValueError("Use a positive chunk limit and a 0–60 second check interval")
        self.forbid_large_c_writes = bool(forbid_large_c_writes)
        self.large_write_bytes = _bytes(large_write_bytes, "large_write_bytes")
        self.interval_seconds, self.telemetry, self.resolver, self.clock = interval_seconds, telemetry, resolver, clock
        self.destinations = {name: str(Path(path).resolve()) for name, path in (destinations or {}).items()}
        self.lock = threading.RLock()
        self.reservations, self.volumes, self.samples = {}, {}, {}
        self.trend = deque(maxlen=60)
        self.last_check, self.stopped = float("-inf"), None
        for path in self.destinations.values():
            self._resolve(path)

    def _resolve(self, path):
        resolved = str(Path(path).resolve())
        pools = tuple(self.resolver(resolved))
        if not pools:
            raise ValueError("A destination must resolve to a capacity pool")
        for pool in pools:
            self.volumes.setdefault(pool.id, pool)
        return resolved, tuple(dict.fromkeys(pool.id for pool in pools))

    def _totals(self):
        totals = {key: 0 for key in self.volumes}
        for paths in self.reservations.values():
            for entry in paths.values():
                for pool in entry["volumes"]:
                    totals[pool] = totals.get(pool, 0) + entry["remaining"]
        return totals

    def reserve(self, operation, writes):
        """Atomically admit a named job; writes maps destinations to byte estimates."""
        if not isinstance(operation, str) or not operation or not isinstance(writes, dict) or not writes:
            raise ValueError("A named operation and nonempty writes mapping are required")
        with self.lock:
            if operation in self.reservations:
                raise ValueError("Operation already reserved")
            paths = {}
            for path, count in writes.items():
                count = _bytes(count, "planned_bytes")
                resolved, pools = self._resolve(path)
                if resolved in paths:
                    paths[resolved]["remaining"] += count
                else:
                    paths[resolved] = dict(volumes=pools, remaining=count)
            c_bytes = sum(entry["remaining"] for entry in paths.values() if "windows:c" in entry["volumes"])
            if self.forbid_large_c_writes and c_bytes >= self.large_write_bytes:
                raise ResourceStop("Large managed writes cannot use C: or its WSL backing disk; choose another volume", operation=operation)
            self.reservations[operation] = paths
            try:
                self.check(force=True, operation=operation)
            except BaseException:
                self.reservations.pop(operation, None)
                raise
        return operation

    def preflight(self, writes):
        """Check whole-job estimates before reserving a bounded streaming burst."""
        name = f"preflight-{uuid.uuid4().hex}"
        self.reserve(name, writes)
        self.release(name)
        return self.check(force=True)

    def update_remaining(self, operation, writes):
        """Reconcile observed remaining download bytes; never increase a claim.

        Only trustworthy adapter progress may consume a reservation. A stalled
        library that provides no progress should use whole-job preflight followed
        by a bounded burst reservation and supervise_process() monitoring.
        """
        with self.lock:
            changes = []
            for path, value in writes.items():
                value = _bytes(value, "remaining_bytes")
                entry = self._entry(operation, path)
                if value > entry["remaining"]:
                    raise ValueError("Remaining bytes may not increase without new admission")
                changes.append((entry, value))
            for entry, value in changes:
                entry["remaining"] = value
            return self.check(force=True, operation=operation)

    def check(self, *, force=False, operation=None):
        with self.lock:
            now = self.clock()
            if not force and now - self.last_check < self.interval_seconds:
                return self.status()
            totals, rows = self._totals(), []
            for identifier, volume in self.volumes.items():
                try:
                    sample = self.telemetry(volume.path)
                    free = _bytes(sample["free_bytes"], "free_bytes")
                    total = _bytes(sample["total_bytes"], "total_bytes")
                except (OSError, KeyError, ValueError) as exc:
                    raise ResourceStop(f"Storage telemetry unavailable for {volume.path}: {exc}", operation=operation) from exc
                row = dict(asdict(volume), free_bytes=free, total_bytes=total,
                           outstanding_bytes=totals[identifier], reserve_bytes=self.reserve_bytes,
                           emergency_bytes=self.emergency_bytes)
                row["available_bytes"] = free - totals[identifier] - self.reserve_bytes - self.emergency_bytes
                rows.append(row)
            self.last_check = now
            self.samples = {row["id"]: row for row in rows}
            self.trend.append(dict(time=now, free_bytes={r["id"]: r["free_bytes"] for r in rows}))
            breached = [row for row in rows if row["available_bytes"] < 0]
            if breached:
                error = ResourceStop("Storage reserve would be breached; controlled writes stopped", operation=operation, volumes=breached)
                self.stopped = error.to_dict()
                raise error
            self.stopped = None
            return self.status()

    def _entry(self, operation, path):
        resolved = str(Path(path).resolve())
        try:
            return self.reservations[operation][resolved]
        except KeyError as exc:
            raise ValueError("Write has no reservation for this operation/destination") from exc

    def before_write(self, operation, path, count):
        count = _bytes(count, "write_bytes")
        if count > self.max_write_bytes:
            raise ValueError("Write exceeds the guarded chunk size")
        with self.lock:
            if count > self._entry(operation, path)["remaining"]:
                raise ResourceStop("Write exceeds its admitted reservation", operation=operation)
            return self.check(force=True, operation=operation)

    def written(self, operation, path, count):
        count = _bytes(count, "written_bytes")
        with self.lock:
            entry = self._entry(operation, path)
            if count > entry["remaining"]:
                raise ValueError("Written bytes exceed remaining reservation")
            entry["remaining"] -= count

    def release(self, operation):
        with self.lock:
            self.reservations.pop(operation, None)

    @contextmanager
    def write(self, path, count, *, operation=None):
        """Guard a bounded write, using a short-lived reservation when omitted."""
        own = operation is None
        operation = operation or f"write-{uuid.uuid4().hex}"
        if own:
            self.reserve(operation, {path: count})
        try:
            self.before_write(operation, path, count)
            yield
            self.written(operation, path, count)
        finally:
            if own:
                self.release(operation)

    def status(self):
        with self.lock:
            return dict(destinations=self.destinations.copy(), reserve_bytes=self.reserve_bytes,
                        emergency_bytes=self.emergency_bytes, max_write_bytes=self.max_write_bytes,
                        interval_seconds=self.interval_seconds, forbid_large_c_writes=self.forbid_large_c_writes,
                        large_write_bytes=self.large_write_bytes, volumes=list(self.samples.values()),
                        reservations={name: sum(v["remaining"] for v in paths.values()) for name, paths in self.reservations.items()},
                        trend=list(self.trend), stop=self.stopped)


class EmergencyMetadata:
    """Owned, physically preallocated metadata headroom, released only on abort.

    allocate() writes actual bytes (not a sparse truncate). finalize() frees that
    allowance before an exclusive error-record write. It never overwrites a run.
    Allocate one slot per concurrently active job and account for it in reserve().
    """
    def __init__(self, directory, *, allowance_bytes=64 * 1024):
        self.directory = Path(directory).resolve()
        self.allowance_bytes = _bytes(allowance_bytes, "allowance_bytes")
        if not self.allowance_bytes or self.allowance_bytes > 4 * 1024**2:
            raise ValueError("Emergency metadata allowance must be 1 byte–4 MiB")
        self.path = self.directory / f".emergency-{uuid.uuid4().hex}"

    def allocate(self, guard=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = b"\0" * self.allowance_bytes
        if guard:
            with guard.write(self.directory, len(payload)):
                self._allocate(payload)
        else:
            self._allocate(payload)
        return self

    def _allocate(self, payload):
        created = False
        try:
            with self.path.open("xb") as stream:
                created = True
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            if created:
                self.path.unlink(missing_ok=True)
            raise

    def release(self):
        self.path.unlink(missing_ok=True)

    def finalize(self, error):
        payload = (json.dumps(error.to_dict() if isinstance(error, ResourceStop) else error,
                              allow_nan=False, ensure_ascii=False) + "\n").encode()
        if len(payload) > self.allowance_bytes:
            raise ValueError("Emergency record exceeds preallocated allowance")
        self.release()
        destination = self.directory / f"resource-stop-{uuid.uuid4().hex}.json"
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return destination


def cancel_owned_process(process, *, grace_seconds=2.0, process_group=False):
    """Terminate, then kill only the supplied owned child (or its owned group)."""
    if process.poll() is not None:
        return process.returncode
    if process_group and os.name != "nt":
        if os.getpgid(process.pid) != process.pid:
            raise ValueError("Refusing cancellation of a process group not owned by this child")
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        if process_group and os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        return process.wait(timeout=grace_seconds)


def supervise_process(process, guard, *, cancel_event=None, operation=None,
                      emergency=None, poll_seconds=.25, grace_seconds=2.0,
                      process_group=False, on_sample=None):
    """Poll a live child even while its downloader blocks; cancel on reserve loss.

    Caller owns Popen and stdout/stderr draining. Use start_new_session=True when
    process_group=True. Child concurrency/write bursts must be bounded by its
    adapter; reserve their maximum outstanding chunk bytes in the admission plan.
    This supervisor preserves partial artifacts and does not pretend to intercept
    unrestricted third-party writes. It never claims success after an abort.
    """
    if not 0 < poll_seconds <= 5:
        raise ValueError("Supervisor polling must be within 0–5 seconds")
    try:
        while process.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                cancel_owned_process(process, grace_seconds=grace_seconds, process_group=process_group)
                return dict(status="cancelled", returncode=process.returncode)
            sample = guard.check(force=True, operation=operation)
            if on_sample:
                on_sample(sample)
            if cancel_event is None:
                time.sleep(poll_seconds)
            else:
                cancel_event.wait(poll_seconds)
        # A final capacity check closes the gap at child completion.
        guard.check(force=True, operation=operation)
        return dict(status="complete" if process.returncode == 0 else "failed", returncode=process.returncode)
    except ResourceStop as exc:
        cancel_owned_process(process, grace_seconds=grace_seconds, process_group=process_group)
        result = exc.to_dict()
        result["returncode"] = process.returncode
        if emergency:
            try:
                result["record"] = str(emergency.finalize(exc))
            except OSError as record_error:
                result["record_error"] = str(record_error)
        return result
