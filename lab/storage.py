"""Portable append-only run records and explicit storage preflight."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import gzip
import os
import re
import shutil
import subprocess
import threading
import uuid

from . import __version__

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix="run"):
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


_WRITE_GUARD = ContextVar("opium_write_guard", default=None)


@contextmanager
def managed_writes(guard):
    """Apply an injected guard to writes in this worker/job context only."""
    token = _WRITE_GUARD.set(guard)
    try:
        yield
    finally:
        _WRITE_GUARD.reset(token)


def current_write_guard():
    return _WRITE_GUARD.get()


class _GuardedWriter:
    """Bound actual write calls, including seek-back ZIP directory updates."""
    def __init__(self, stream, path, guard, operation):
        self.stream, self.path, self.guard, self.operation = stream, path, guard, operation

    def write(self, data):
        view = memoryview(data).cast("B")
        count = 0
        for offset in range(0, len(view), self.guard.max_write_bytes):
            chunk = view[offset:offset+self.guard.max_write_bytes]
            self.guard.before_write(self.operation, self.path, len(chunk))
            wrote = self.stream.write(chunk)
            if wrote != len(chunk):
                raise OSError("Incomplete guarded storage write")
            self.guard.written(self.operation, self.path, wrote)
            count += wrote
        return count

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextmanager
def guarded_open(path, *, mode="wb", planned_bytes, guard=None):
    """Admit the whole write, then check/release actual bounded chunks."""
    if mode not in {"wb", "xb", "ab"}:
        raise ValueError("Guarded file mode must be wb, xb, or ab")
    if type(planned_bytes) is not int or planned_bytes < 0:
        raise ValueError("Guarded write requires a nonnegative byte estimate")
    guard = guard if guard is not None else current_write_guard()
    path = Path(path)
    operation = "storage-write-" + uuid.uuid4().hex
    if guard:
        guard.reserve(operation, {path: planned_bytes})
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode) as stream:
            yield _GuardedWriter(stream, path, guard, operation) if guard else stream
    finally:
        if guard:
            guard.release(operation)


def guarded_bytes(path, data, *, mode="wb", guard=None):
    """Write existing bytes through bounded chunks; never silently truncate on refusal."""
    data = memoryview(data).cast("B")
    with guarded_open(path, mode=mode, planned_bytes=len(data), guard=guard) as stream:
        stream.write(data)


def atomic_json(path, value, *, guard=None, ensure_ascii=False):
    path = Path(path)
    raw = (json.dumps(value, indent=2, allow_nan=False, ensure_ascii=ensure_ascii) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        guarded_bytes(tmp, raw, mode="xb", guard=guard)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def guarded_npz(path, *, guard=None, **arrays):
    """Write numeric calibration arrays without a large unguarded NumPy burst."""
    import numpy as np
    values = {key: np.asanyarray(value) for key, value in arrays.items()}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError("Calibration archives cannot contain object/pickle arrays")
    estimate = sum(value.nbytes for value in values.values()) + (len(values)+1)*65536
    with guarded_open(path, planned_bytes=estimate, guard=guard) as stream:
        np.savez(stream, **values)


def guarded_copyfile(source, destination, *, guard=None):
    source, destination = Path(source), Path(destination)
    selected = guard if guard is not None else current_write_guard()
    chunk_size = selected.max_write_bytes if selected else 4*1024**2
    with source.open("rb") as incoming:
        with guarded_open(destination, planned_bytes=source.stat().st_size, guard=selected) as outgoing:
            while chunk := incoming.read(chunk_size):
                outgoing.write(chunk)
    return str(destination)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def disk_info(path):
    path = Path(path).resolve()
    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    return dict(path=str(path), free_bytes=usage.free, total_bytes=usage.total,
                free_gib=round(usage.free / 2**30, 2))


def storage_info(data_dir, cache_dir):
    result = dict(data=disk_info(data_dir), cache=disk_info(cache_dir), reserve_gib=10)
    if Path("/mnt/c").exists():
        result["windows_c"] = disk_info("/mnt/c")
        result["note"] = "WSL virtual free space can exceed backing Windows capacity. Large files use the selected data/cache drive."
    return result


def source_manifest():
    sources = sorted((ROOT / "lab").rglob("*.py")) + [ROOT / "self_admin_protocol.py"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(commit=commit, dirty=dirty, source_sha256=hashes)


def preflight(path, required_bytes=0, reserve_bytes=10 * 2**30):
    info = disk_info(path)
    if required_bytes < 0 or info["free_bytes"] - required_bytes < reserve_bytes:
        raise ValueError(f"Storage reserve would be breached at {info['path']}: {info['free_gib']} GiB free; need {required_bytes / 2**30:.1f} GiB plus {reserve_bytes / 2**30:.1f} GiB reserve.")
    # A write into Linux root may grow a WSL virtual disk on C:. Be conservative
    # when its backing volume cannot be inferred from the destination itself.
    resolved = str(Path(path).resolve())
    if Path("/mnt/c").exists() and not resolved.startswith("/mnt/"):
        host = disk_info("/mnt/c")
        if host["free_bytes"] - required_bytes < reserve_bytes:
            raise ValueError("C: lacks the free-space reserve for possible WSL growth. Select a data/cache directory on another drive.")
    return info


class Store:
    def __init__(self, data_dir, historical=(), guard=None):
        self.guard = guard
        self.emergency = {}
        self.root = Path(data_dir).resolve()
        self.runs = self.root / "runs"
        self.calibrations = self.root / "calibrations"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.calibrations.mkdir(parents=True, exist_ok=True)
        self.historical = [Path(p).resolve() for p in historical]
        self.lock = threading.RLock()

    def run_path(self, identifier, writable=False):
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError("Invalid run identifier")
        candidates = [self.runs] if writable else [self.runs, *self.historical]
        for base in candidates:
            path = (base / identifier).resolve()
            if path.parent != base.resolve():
                continue  # Never follow a run-directory symlink outside its evidence root.
            if path.is_dir() or writable:
                return path
        raise FileNotFoundError("Run not found")

    def create(self, mode, config, parent=None, *, source_identity=None):
        identifier = new_id()
        path = self.run_path(identifier, writable=True)
        path.mkdir(exist_ok=False)
        self.prepare_emergency(identifier)
        atomic_json(path / "manifest.json", dict(id=identifier, mode=mode,
                    status="queued", config=config, created_at=utc_now(), parent=parent,
                    format_version=2, software="opium-bench/" + __version__, source=source_manifest() if source_identity is None else source_identity), guard=self.guard)
        return identifier, path

    def prepare_emergency(self, identifier):
        """Preallocate metadata outside exported evidence before a managed job."""
        if self.guard and identifier not in self.emergency:
            from .resources import EmergencyMetadata
            self.run_path(identifier, writable=True)
            self.emergency[identifier] = EmergencyMetadata(self.root / "resource-headroom" / identifier).allocate(self.guard)

    def release_emergency(self, identifier):
        slot = self.emergency.pop(identifier, None)
        if slot:
            slot.release()
            try:
                slot.directory.rmdir()
            except OSError:
                pass

    @staticmethod
    def resource_stop_record(path):
        records = sorted(Path(path).glob("resource-stop-*.json"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
        for record in records:
            if record.is_symlink():
                continue
            value = read_json(record)
            if isinstance(value, dict) and value.get("status") == "resource_stopped":
                return value
        return None

    def resource_stop(self, identifier, error, summary=None):
        """Write one bounded emergency record without replacing prior evidence.

        Catalog/replay overlay the explicit terminal state from this record. The
        saved manifest and last complete checkpoint remain byte-for-byte intact.
        """
        with self.lock:
            path = self.run_path(identifier, writable=True)
            existing = self.resource_stop_record(path)
            if existing:
                self.release_emergency(identifier)
                return existing
            slot = self.emergency.pop(identifier, None)
            if not slot:
                raise OSError("No preallocated metadata slot for this run; preserve supervisor resource-stop evidence")
            detail = error.to_dict() if hasattr(error, "to_dict") else dict(error) if isinstance(error, dict) else {"reason":str(error)}
            record = dict(status="resource_stopped", termination="storage_reserve", resource_stop=detail,
                          run_id=identifier, time=utc_now(), summary=summary or {})
            if len(json.dumps(record, ensure_ascii=False, allow_nan=False).encode()) + 1 > slot.allowance_bytes:
                record["summary"] = {key: (summary or {}).get(key) for key in ("tokens", "actions", "assigned", "submitted", "correct")}
            target = slot.finalize(record)
            target.rename(path / target.name)
            return record

    def append(self, identifier, event):
        with self.lock:
            path = self.run_path(identifier, writable=True)
            raw = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            guarded_bytes(path / "events.jsonl", raw, mode="ab", guard=self.guard)

    def update(self, identifier, **values):
        with self.lock:
            if values.get("status") == "resource_stopped" and self.guard:
                summary = values.get("summary", {})
                return self.resource_stop(identifier, summary.get("resource_stop", {"reason":"storage_reserve"}), summary)
            path = self.run_path(identifier, writable=True)
            old = read_json(path / "manifest.json", {})
            old.update(values)
            atomic_json(path / "manifest.json", old, guard=self.guard)
            if "summary" in values:
                atomic_json(path / "summary.json", values["summary"], guard=self.guard)
            if values.get("status") in {"complete", "failed", "stopped", "cancelled", "aborted"}:
                self.release_emergency(identifier)

    def catalog(self):
        rows, seen = [], set()
        for base in [self.runs, *self.historical]:
            if not base.exists():
                continue
            for path in base.iterdir():
                if not path.is_dir() or not ID.fullmatch(path.name) or path.name in seen:
                    continue
                if path.resolve().parent != base.resolve():
                    continue
                # Local records take precedence over a bundled copy, matching run_path.
                seen.add(path.name)
                m = read_json(path / "manifest.json")
                if not m:
                    continue
                resource_stop = self.resource_stop_record(path)
                rows.append(dict(id=path.name, mode=m.get("mode", "historical"),
                    status="resource_stopped" if resource_stop else m.get("status", "unknown"), created_at=m.get("created_at", m.get("started_utc", "")),
                    config=m.get("config", {}), summary=dict(resource_stop.get("summary", {}), termination="storage_reserve",resource_stop=resource_stop["resource_stop"]) if resource_stop else read_json(path / "summary.json", m.get("summary", {})),
                    historical=base != self.runs, imported=(path / "_portable").is_dir(), model=m.get("model", {})))
        return sorted(rows, key=lambda x: x["created_at"], reverse=True)

    def calibration_catalog(self):
        rows = []
        for path in self.calibrations.iterdir():
            m = read_json(path / "calibration.json") if path.is_dir() and not path.is_symlink() and not (path / "calibration.json").is_symlink() else None
            if m:
                rows.append(dict(m, id=path.name, name=m.get("name", path.name)))
        return rows

    def calibration_path(self, identifier):
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError("Select a valid calibration")
        path = self.calibrations / identifier
        if path.is_symlink() or (path / "calibration.json").is_symlink() or not (path / "calibration.json").is_file():
            raise ValueError("Calibration not found; calibrate the loaded model first")
        return path

    def read_run(self, identifier):
        path = self.run_path(identifier)
        manifest = read_json(path / "manifest.json", {})
        events = []
        event_file = path / "events.jsonl"
        compressed = path / "events.jsonl.gz"
        has_events = event_file.exists() or compressed.exists()
        if has_events:
            opener, target = (open, event_file) if event_file.exists() else (gzip.open, compressed)
            with opener(target, "rt", encoding="utf-8") as stream:
                lines = stream.readlines()
            for line in lines:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass  # Incomplete final write after an interrupted process.
        parent_events = self.read_parent_events(path, manifest)
        resource_stop = self.resource_stop_record(path)
        summary = read_json(path / "summary.json", {})
        if resource_stop:
            manifest = dict(manifest, status="resource_stopped", resource_stop=resource_stop["resource_stop"])
            summary = dict(resource_stop.get("summary", {}), termination="storage_reserve", resource_stop=resource_stop["resource_stop"])
        return dict(id=identifier, manifest=manifest,
                    summary=summary, events=events,
                    parent_events=parent_events, conversation=read_json(path / "conversation.json", []),
                    historical=not has_events, imported=(path / "_portable").is_dir())

    @staticmethod
    def read_parent_events(path, manifest=None):
        """Read the immutable replay prefix, including earlier branch generations.

        The immediate parent prefix keeps its checkpoint-bound hash. Earlier
        prefixes have a separate manifest hash and never enter child accounting.
        """
        path = Path(path)
        manifest = manifest if manifest is not None else read_json(path / "manifest.json", {})
        rows = []
        for name, checksum in (
            ("ancestor-events.jsonl.gz", (manifest.get("replay_ancestry") or {}).get("sha256")),
            ("parent-events.jsonl.gz", (manifest.get("parent") or {}).get("parent_prefix_sha256")),
        ):
            prefix = path / name
            if not checksum:
                if prefix.exists() or prefix.is_symlink():
                    raise ValueError("Replay event prefix has no integrity record")
                continue
            if prefix.is_symlink() or not prefix.is_file():
                raise ValueError("Replay event prefix is missing or not an ordinary file")
            with gzip.open(prefix, "rb") as stream:
                raw = stream.read(128 * 1024**2 + 1)
            if len(raw) > 128 * 1024**2 or hashlib.sha256(raw).hexdigest() != checksum:
                raise ValueError("Parent event prefix integrity check failed")
            rows.extend(dict(json.loads(line), inherited=True) for line in raw.splitlines())
        return rows
