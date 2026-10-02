"""Bounded, JSON-only metadata and inert evidence bundle portability.

Bundles never contain model weights or executable checkpoints. Imported source
snapshots/HTML are inert evidence: callers must render them as text or construct a
safe report, never serve imported HTML as a trusted application page. Checkpoint
presence is not permission to resume; the lifecycle adapter must validate every
model/runtime/calibration identity and boundary state first.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import ast
import base64
import ctypes
import errno
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import uuid
import zipfile

FORMAT = "opium-bench/evidence-bundle"
VERSION = 1
SUPPORTED_RUN_VERSIONS = {1, 2, 3}
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
ALLOWED_SUFFIXES = {".json", ".jsonl", ".gz", ".npz", ".md", ".txt", ".csv", ".html", ".svg", ".png", ".py"}
RESERVED_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


@dataclass(frozen=True)
class Limits:
    compressed_bytes: int = 128 * 1024**2
    expanded_bytes: int = 256 * 1024**2
    member_bytes: int = 64 * 1024**2
    members: int = 512
    json_depth: int = 64
    json_nodes: int = 1_000_000

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in self.__dict__.values()):
            raise ValueError("Import limits must be positive integers")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _encode(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def _json(raw, limits):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON object key")
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("Invalid or overly nested UTF-8 JSON") from exc
    _shape(value, limits)
    return value


def _shape(value, limits):
    count, stack = 0, [(value, 0)]
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > limits.json_depth or count > limits.json_nodes:
            raise ValueError("JSON exceeds structural bounds")
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("Nonfinite JSON number")
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("Unsupported JSON value")


def _safe_name(name):
    if not isinstance(name, str) or len(name) > 240 or "\\" in name or "\0" in name:
        raise ValueError("Unsafe bundle path")
    path = PurePosixPath(name)
    if not name or name.startswith("/") or str(path) != name:
        raise ValueError("Unsafe bundle path")
    for part in path.parts:
        if part in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part.endswith((".", " ")):
            raise ValueError("Unsafe bundle path")
        if part.split(".")[0].casefold() in RESERVED_NAMES:
            raise ValueError("Reserved platform filename")
    return path


def _identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("Invalid run identifier")
    _safe_name(value)
    return value


def _run_manifest(value):
    if not isinstance(value, dict):
        raise ValueError("Run manifest must be a JSON object")
    version = value.get("format_version", 1)
    if type(version) is not int or version not in SUPPORTED_RUN_VERSIONS:
        raise ValueError("Unsupported run format version")


def _npz_member_name(name):
    """Validate inert NPY keys without treating them as extracted filenames.

    Schema-2 extraction stores '<integer layer>:<pooling>' keys inside NPZ.
    These members are read as numeric array streams, never written to paths.
    Outer bundle names always retain the stricter filesystem rules.
    """
    match = re.fullmatch(r"(0|[1-9][0-9]{0,3}):(final|mean|span)\.npy", name)
    if match and int(match[1]) <= 4096:
        return
    _safe_name(name)


def _zip_members(archive, limits, *, numeric_archive=False):
    infos = archive.infolist()
    if len(infos) > limits.members:
        raise ValueError("Too many bundle members")
    seen, total = set(), 0
    for info in infos:
        (_npz_member_name if numeric_archive else _safe_name)(info.filename)
        folded = info.filename.casefold()
        if folded in seen:
            raise ValueError("Duplicate or case-colliding bundle member")
        seen.add(folded)
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in {0, stat.S_IFREG}):
            raise ValueError("Only ordinary file members are permitted")
        if info.flag_bits & 1:
            raise ValueError("Encrypted bundle members are unsupported")
        if info.file_size > limits.member_bytes:
            raise ValueError("Bundle member exceeds expanded size limit")
        total += info.file_size
        if total > limits.expanded_bytes:
            raise ValueError("Bundle exceeds expanded size limit")
    return infos


def _validate_npz(raw, limits):
    """Check numeric-only NPY headers without importing NumPy or unpickling."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for info in _zip_members(archive, limits, numeric_archive=True):
                if not info.filename.endswith(".npy"):
                    raise ValueError("Only numeric NPY arrays belong in calibration archives")
                with archive.open(info) as stream:
                    prefix = stream.read(12)
                    if prefix[:6] != b"\x93NUMPY" or prefix[6:8] not in {b"\x01\x00", b"\x02\x00", b"\x03\x00"}:
                        raise ValueError("Invalid numeric array header")
                    old = prefix[6] == 1
                    length = struct.unpack("<H" if old else "<I", prefix[8:10 if old else 12])[0]
                    if length > 64 * 1024:
                        raise ValueError("Numeric array header too large")
                    header = (prefix[10:] if old else b"") + stream.read(length - (2 if old else 0))
                    value = ast.literal_eval(header.decode("latin1"))
                    dtype, shape = value["descr"], value["shape"]
                    if not isinstance(dtype, str) or not re.fullmatch(r"[<>=|][?biufc][0-9]+", dtype):
                        raise ValueError("Object/structured arrays are not portable checkpoints")
                    if not isinstance(shape, tuple) or any(type(n) is not int or n < 0 for n in shape):
                        raise ValueError("Invalid numeric array shape")
                    size = int(re.search(r"[0-9]+$", dtype)[0])
                    if math.prod(shape) * size + (10 if old else 12) + length != info.file_size:
                        raise ValueError("Numeric array shape does not match byte length")
    except (zipfile.BadZipFile, SyntaxError, KeyError, TypeError, EOFError, UnicodeError) as exc:
        raise ValueError("Invalid numeric calibration archive") from exc


def _validate_file(name, raw, limits):
    suffix = PurePosixPath(name).suffix
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Bundle contains unsupported files or model weights")
    if suffix == ".npz":
        _validate_npz(raw, limits)
    if suffix == ".gz":
        if not name.endswith((".jsonl.gz", ".json.gz")):
            raise ValueError("Only compressed JSON/JSONL evidence is supported")
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                expanded = stream.read(limits.member_bytes + 1)
        except (OSError, EOFError) as exc:
            raise ValueError("Invalid compressed JSONL") from exc
        if len(expanded) > limits.member_bytes:
            raise ValueError("Nested compressed evidence exceeds expanded limit")
        return _validate_file(name[:-3], expanded, limits)
    if suffix == ".json":
        value = _json(raw, limits)
        if PurePosixPath(name).name == "manifest.json":
            _run_manifest(value)
        if PurePosixPath(name).name == "checkpoint.json":
            version = value.get("schema_version", value.get("format_version", 1)) if isinstance(value, dict) else None
            if type(version) is not int or version != 1:
                raise ValueError("Unsupported checkpoint version")
        if PurePosixPath(name).name == "calibration.json":
            if not isinstance(value, dict) or type(value.get("schema_version", 1)) is not int or value.get("schema_version", 1) not in {1, 2}:
                raise ValueError("Unsupported calibration schema version")
    elif suffix == ".jsonl":
        # Preserve a crashed run's incomplete final record, explicitly as partial.
        lines = raw.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                item = _json(line, limits)
                if not isinstance(item, dict):
                    raise ValueError("Evidence event must be an object")
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not raw.endswith(b"\n"):
                    return False
                raise
    return True


def _expanded_weight(name, raw, limits):
    """Count nested compression as well as stored bytes against one total cap."""
    if name.endswith(".npz"):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            return len(raw) + sum(info.file_size for info in _zip_members(archive, limits, numeric_archive=True))
    if name.endswith((".jsonl.gz", ".json.gz")):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            expanded = stream.read(limits.member_bytes + 1)
        if len(expanded) > limits.member_bytes:
            raise ValueError("Nested compressed evidence exceeds expanded limit")
        return len(raw) + len(expanded)
    return len(raw)


def _check_total(files, limits):
    if len(files) > limits.members or sum(_expanded_weight(name, raw, limits) for name, raw in files.items()) > limits.expanded_bytes:
        raise ValueError("Evidence exceeds combined expanded size/member limit")


def _read_source(source, limit):
    if isinstance(source, bytes):
        raw = source
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise ValueError("Import source is not a bounded regular file")
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
    elif hasattr(source, "read"):
        raw = source.read(limit + 1)
    else:
        raise ValueError("Import source must be bytes, a regular file, or a binary stream")
    if not isinstance(raw, bytes) or len(raw) > limit:
        raise ValueError("Import exceeds input size limit")
    return raw


def _collect(directory, prefix, limits):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Evidence root must be an ordinary directory")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlinks are not portable evidence")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Only ordinary evidence files can be exported")
        name = str(PurePosixPath(prefix) / path.relative_to(directory).as_posix())
        _safe_name(name)
        if path.stat().st_size > limits.member_bytes:
            raise ValueError("Export member exceeds size limit")
        raw = _read_source(path, limits.member_bytes)
        _validate_file(name, raw, limits)
        if name.casefold() in {existing.casefold() for existing in files}:
            raise ValueError("Case-colliding source filenames")
        files[name] = raw
        _check_total(files, limits)
    return files


def _write_bytes(path, raw, guard=None, operation=None, reservation_path=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    chunk = guard.max_write_bytes if guard else 1024**2
    with path.open("xb") as stream:
        for start in range(0, len(raw), chunk):
            part = raw[start:start + chunk]
            if guard:
                guard.before_write(operation, reservation_path, len(part))
            stream.write(part)
            if guard:
                guard.written(operation, reservation_path, len(part))
        stream.flush()
        os.fsync(stream.fileno())


def export_bundle(run_dir, destination, *, calibration_dir=None, guard=None, limits=Limits()):
    """Write a new ZIP with byte-preserved evidence, hashes, and no model weights.

    Completed/boundary snapshots should be exported: callers must serialize live
    writes. Existing destinations are never overwritten, even on a race.
    """
    run_dir, destination = Path(run_dir), Path(destination)
    identifier = _identifier(run_dir.name)
    files = _collect(run_dir, f"run/{identifier}", limits)
    manifest_name = f"run/{identifier}/manifest.json"
    if manifest_name not in files:
        raise ValueError("A run manifest is required")
    manifest = _json(files[manifest_name], limits)
    if manifest.get("id", identifier) != identifier:
        raise ValueError("Run directory and manifest identifiers differ")
    calibration = None
    if calibration_dir:
        calibration_dir = Path(calibration_dir)
        calibration = _identifier(calibration_dir.name)
        files.update(_collect(calibration_dir, f"calibration/{calibration}", limits))
    _check_total(files, limits)
    if len(files) + 1 > limits.members or sum(map(len, files.values())) > limits.expanded_bytes:
        raise ValueError("Export exceeds bundle limits")
    bundle = dict(format=FORMAT, version=VERSION, run_id=identifier, calibration_id=calibration,
                  weights_included=False, provenance=dict(manifest_sha256=_sha(files[manifest_name])),
                  files={name: dict(size=len(raw), sha256=_sha(raw)) for name, raw in files.items()})
    # Bounded in memory before any destination mutation. Compressed size is also
    # checked; the disk guard still checks each physical output chunk.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bundle.json", _encode(bundle))
        for name, raw in files.items():
            archive.writestr(name, raw)
    payload = buffer.getvalue()
    if len(payload) > limits.compressed_bytes:
        raise ValueError("Export exceeds compressed size limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Export destination already exists")
    operation = f"export-{uuid.uuid4().hex}"
    temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    if guard:
        guard.reserve(operation, {destination.parent: len(payload)})
    try:
        _write_bytes(temporary, payload, guard, operation, destination.parent)
        # Hard linking gives no-replace installation; temporary and destination
        # are on the same filesystem. No fallback may silently overwrite.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        if guard:
            guard.release(operation)
    return dict(path=str(destination.resolve()), run_id=identifier, size=len(payload), sha256=_sha(payload), version=VERSION)


def _eligibility(files, prefix, partial):
    conversation = files.get(f"{prefix}/conversation.json")
    checkpoint_paths = [name for name in files if name.startswith(prefix + "/") and (PurePosixPath(name).name == "checkpoint.json" or "/checkpoints/" in name)]
    candidate = bool(conversation and checkpoint_paths and not partial)
    return dict(replay_only=True, resume_eligible=False, checkpoint_candidate=candidate,
                reason="A local lifecycle validator must verify the full boundary and model/runtime/calibration identity" if candidate else "No complete validated boundary checkpoint in this import")


def _windows_directory_move(source, destination):
    """drvfs lacks renameat2 flags; Windows Directory.Move refuses replacement."""
    powershell = shutil.which("powershell.exe")
    fallback = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not powershell and fallback.is_file():
        powershell = str(fallback)
    if not powershell or not shutil.which("wslpath"):
        raise OSError(errno.ENOTSUP, "Atomic drvfs import requires Windows PowerShell interop")
    paths = []
    for path in (source, destination):
        result = subprocess.run(["wslpath", "-w", str(Path(path).resolve())], check=True,
                                capture_output=True, text=True, timeout=5)
        paths.append(base64.b64encode(result.stdout.strip().encode("utf-8")).decode("ascii"))
    # Paths are opaque base64 literals; no shell interpolation of filenames.
    script = ("$ErrorActionPreference='Stop';"
              f"$src=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{paths[0]}'));"
              f"$dst=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{paths[1]}'));"
              "try{[IO.Directory]::Move($src,$dst)}catch{[Console]::Error.WriteLine($_.Exception.Message);exit 1}")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        result = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired as exc:
        # The owned PowerShell is killed by subprocess.run. Reinspect the final
        # filesystem before reporting an uncertain timeout as installation failure.
        if Path(destination).is_dir() and not Path(source).exists():
            return
        raise OSError(errno.EIO, "Atomic drvfs import timed out") from exc
    if result.returncode:
        if Path(destination).exists():
            raise FileExistsError(errno.EEXIST, "Run identifier conflict", str(destination))
        raise OSError(errno.EIO, "Atomic drvfs import failed: " + result.stderr.strip()[:300])


def _rename_no_replace(source, destination):
    """Atomic installation without replacing even an empty racing directory."""
    if os.name == "nt":
        os.rename(source, destination)  # Windows fails if destination exists.
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
                return
            code = ctypes.get_errno()
            if code in {errno.EINVAL, errno.ENOTSUP, errno.ENOSYS} and str(Path(destination).resolve()).startswith("/mnt/"):
                return _windows_directory_move(source, destination)
            raise OSError(code, os.strerror(code), str(destination))
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renamex_np", None)
        if rename is not None:
            rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if rename(os.fsencode(source), os.fsencode(destination), 4) == 0:
                return
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
    raise OSError(errno.ENOTSUP, "Atomic no-replace directory installation is unavailable")


def _install(files, bundle, raw_hash, runs_dir, *, guard, limits, source_format, partial=False, reserved_ids=()):
    identifier = _identifier(bundle["run_id"])
    if identifier.casefold() in {name.casefold() for name in reserved_ids}:
        raise FileExistsError("A run with this identifier is already available; evidence is never shadowed")
    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    destination = runs_dir / identifier
    # Case-fold conflicts also fail on case-sensitive hosts for portability.
    if any(path.name.casefold() == identifier.casefold() for path in runs_dir.iterdir()):
        raise FileExistsError("A run with this identifier already exists; evidence is never overwritten")
    prefix = f"run/{identifier}"
    eligibility = _eligibility(files, prefix, partial)
    metadata = dict(format="opium-bench/import-record", version=1, source_format=source_format,
                    source_sha256=raw_hash, imported_at=datetime.now(timezone.utc).isoformat(),
                    source_bundle=bundle, partial_events=partial, **eligibility)
    target_files = {}
    for name, raw in files.items():
        if name.startswith(prefix + "/"):
            relative = name[len(prefix) + 1:]
        elif name.startswith("calibration/"):
            relative = "_portable/" + name
        else:
            raise ValueError("Bundle file is outside its declared run/calibration")
        target_files[relative] = raw
    target_files[f"_portable/import-{uuid.uuid4().hex}.json"] = _encode(metadata)
    total = sum(map(len, target_files.values()))
    operation = f"import-{uuid.uuid4().hex}"
    if guard:
        guard.reserve(operation, {runs_dir: total})
    staging = None
    lock = runs_dir / f".import-{identifier}.lock"
    locked = False
    try:
        # Exclusive lock protects cooperating installers. The final destination
        # check plus rename preserves a preexisting directory on all platforms.
        with lock.open("xb"):
            locked = True
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Run identifier conflict")
        staging = Path(tempfile.mkdtemp(prefix=".import-stage-", dir=runs_dir))
        for name, raw in target_files.items():
            _safe_name(name)
            _write_bytes(staging / name, raw, guard, operation, runs_dir)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Run identifier conflict")
        _rename_no_replace(staging, destination)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        if locked:
            lock.unlink(missing_ok=True)
        if guard:
            guard.release(operation)
    return dict(id=identifier, path=str(destination), imported=True, partial_events=partial,
                source_sha256=raw_hash, embedded_calibration=bundle.get("calibration_id"), **eligibility)


def import_bundle(source, runs_dir, *, guard=None, limits=Limits(), reserved_ids=()):
    raw = _read_source(source, limits.compressed_bytes)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = _zip_members(archive, limits)
            names = {info.filename for info in infos}
            if "bundle.json" not in names:
                raise ValueError("Missing bundle manifest")
            bundle = _json(archive.read("bundle.json"), limits)
            if not isinstance(bundle, dict) or bundle.get("format") != FORMAT or type(bundle.get("version")) is not int or bundle.get("version") != VERSION:
                raise ValueError("Unsupported evidence bundle version")
            identifier = _identifier(bundle.get("run_id"))
            if bundle.get("weights_included") is not False:
                raise ValueError("Evidence bundles must exclude model weights")
            expected = bundle.get("files")
            if not isinstance(expected, dict) or set(expected) != names - {"bundle.json"}:
                raise ValueError("Bundle file inventory mismatch")
            calibration = bundle.get("calibration_id")
            if calibration is not None:
                _identifier(calibration)
            files, partial = {}, False
            for name, descriptor in expected.items():
                if not name.startswith(f"run/{identifier}/") and not (calibration and name.startswith(f"calibration/{calibration}/")):
                    raise ValueError("Bundle path does not match declared identifiers")
                content = archive.read(name)
                if not isinstance(descriptor, dict) or type(descriptor.get("size")) is not int or descriptor["size"] != len(content) or descriptor.get("sha256") != _sha(content):
                    raise ValueError("Evidence size/hash mismatch")
                partial |= not _validate_file(name, content, limits)
                files[name] = content
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError("Invalid or unsupported evidence ZIP") from exc
    _check_total(files, limits)
    manifest_name = f"run/{identifier}/manifest.json"
    if manifest_name not in files:
        raise ValueError("Missing run manifest")
    manifest = _json(files[manifest_name], limits)
    if manifest.get("id", identifier) != identifier:
        raise ValueError("Run and manifest identifiers differ")
    if bundle.get("provenance", {}).get("manifest_sha256") != _sha(files[manifest_name]):
        raise ValueError("Source manifest provenance hash mismatch")
    return _install(files, bundle, _sha(raw), runs_dir, guard=guard, limits=limits, source_format=FORMAT, partial=partial, reserved_ids=reserved_ids)


def import_json_export(source, runs_dir, *, guard=None, limits=Limits(), reserved_ids=()):
    """Import the existing /api/runs/<id>/export JSON as replay evidence.

    Missing full tool conversation/checkpoint is never reconstructed from token
    text. The source bytes are retained even for legacy prototype exports.
    """
    raw = _encode(source) if isinstance(source, dict) else _read_source(source, limits.member_bytes)
    if len(raw) > limits.member_bytes:
        raise ValueError("JSON export exceeds size limit")
    value = _json(raw, limits)
    if not isinstance(value, dict):
        raise ValueError("Export must be a JSON object")
    manifest, summary, events = value.get("manifest"), value.get("summary", {}), value.get("events", [])
    _run_manifest(manifest)
    identifier = _identifier(value.get("id", manifest.get("id")))
    if manifest.get("id", identifier) != identifier or not isinstance(summary, dict) or not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        raise ValueError("Invalid exported run structure")
    prefix = f"run/{identifier}"
    files = {f"{prefix}/manifest.json": _encode(manifest), f"{prefix}/summary.json": _encode(summary),
             f"{prefix}/events.jsonl": b"".join(json.dumps(e, ensure_ascii=False, allow_nan=False).encode() + b"\n" for e in events),
             f"{prefix}/_portable/source-export.json": raw}
    if "conversation" in value:
        messages = value["conversation"]
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("Conversation must be an array of message objects")
        files[f"{prefix}/conversation.json"] = _encode(messages)
    bundle = dict(format="opium-bench/json-export", version=1, run_id=identifier, weights_included=False,
                  provenance=dict(manifest_sha256=_sha(files[f"{prefix}/manifest.json"])),
                  files={name: dict(size=len(content), sha256=_sha(content)) for name, content in files.items()})
    if sum(map(len, files.values())) > limits.expanded_bytes:
        raise ValueError("Expanded JSON import exceeds size limit")
    return _install(files, bundle, _sha(raw), runs_dir, guard=guard, limits=limits, source_format=bundle["format"], reserved_ids=reserved_ids)
