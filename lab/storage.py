"""Portable append-only run records and explicit storage preflight."""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix="run"):
    return f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


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
    def __init__(self, data_dir, historical=()):
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

    def create(self, mode, config, parent=None):
        identifier = new_id()
        path = self.run_path(identifier, writable=True)
        path.mkdir(exist_ok=False)
        atomic_json(path / "manifest.json", dict(id=identifier, mode=mode,
                    status="queued", config=config, created_at=utc_now(), parent=parent,
                    format_version=2, software="opium-bench/0.2.0", source=source_manifest()))
        return identifier, path

    def append(self, identifier, event):
        with self.lock:
            path = self.run_path(identifier, writable=True)
            with (path / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def update(self, identifier, **values):
        with self.lock:
            path = self.run_path(identifier, writable=True)
            old = read_json(path / "manifest.json", {})
            old.update(values)
            atomic_json(path / "manifest.json", old)
            if "summary" in values:
                atomic_json(path / "summary.json", values["summary"])

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
                rows.append(dict(id=path.name, mode=m.get("mode", "historical"),
                    status=m.get("status", "unknown"), created_at=m.get("created_at", m.get("started_utc", "")),
                    config=m.get("config", {}), summary=read_json(path / "summary.json", m.get("summary", {})),
                    historical=base != self.runs, imported=(path / "_portable").is_dir(), model=m.get("model", {})))
        return sorted(rows, key=lambda x: x["created_at"], reverse=True)

    def calibration_catalog(self):
        rows = []
        for path in self.calibrations.iterdir():
            m = read_json(path / "calibration.json") if path.is_dir() else None
            if m:
                rows.append(dict(m, id=path.name, name=m.get("name", path.name)))
        return rows

    def calibration_path(self, identifier):
        if not isinstance(identifier, str) or not ID.fullmatch(identifier):
            raise ValueError("Select a valid calibration")
        path = self.calibrations / identifier
        if not (path / "calibration.json").is_file():
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
        parent_events = []
        parent_file = path / "parent-events.jsonl.gz"
        if parent_file.is_file() and not parent_file.is_symlink():
            with gzip.open(parent_file, "rb") as stream:
                raw = stream.read(128 * 1024**2 + 1)
            if len(raw) > 128 * 1024**2 or hashlib.sha256(raw).hexdigest() != (manifest.get("parent") or {}).get("parent_prefix_sha256"):
                raise ValueError("Parent event prefix integrity check failed")
            parent_events = [dict(json.loads(line), inherited=True) for line in raw.splitlines()]
        return dict(id=identifier, manifest=manifest,
                    summary=read_json(path / "summary.json", {}), events=events,
                    parent_events=parent_events, conversation=read_json(path / "conversation.json", []),
                    historical=not has_events, imported=(path / "_portable").is_dir())
