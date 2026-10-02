"""Bounded boundary-file reads and explicit event-prefix provenance."""
from pathlib import Path
import gzip
import hashlib
import json
import re

from .checkpoints import MAX_BYTES, validate

NAME = re.compile(r"^turn-(\d{5})-event-(\d{9})\.json\.gz$")


def list_boundaries(directory):
    root = Path(directory).resolve()
    rows = []
    for file in (root / "checkpoints").glob("*.json.gz"):
        match = NAME.fullmatch(file.name)
        if match and not file.is_symlink() and file.resolve().parent == root / "checkpoints":
            rows.append(dict(id=file.name, turns=int(match[1]), event_cutoff=int(match[2])))
    return sorted(rows, key=lambda row: (row["turns"], row["event_cutoff"]), reverse=True)


def read_boundary(directory, identifier="latest"):
    root = Path(directory).resolve()
    if identifier == "latest":
        file, opener = root / "checkpoint.json", open
    elif isinstance(identifier, str) and NAME.fullmatch(identifier):
        file, opener = root / "checkpoints" / identifier, gzip.open
    else:
        raise ValueError("Choose a saved completed-turn boundary")
    if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
        raise ValueError("Saved boundary was not found")
    with opener(file, "rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("Saved boundary exceeds its size limit")
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Saved boundary is not valid JSON") from exc
    return validate(result)


def event_prefix(directory, count):
    """Preserve exactly the original JSONL bytes through the captured cutoff."""
    if type(count) is not int or not 0 <= count <= 10**7:
        raise ValueError("Checkpoint has no valid event cutoff")
    root = Path(directory).resolve()
    path = root / "events.jsonl"
    opener = open
    if not path.exists():
        path, opener = root / "events.jsonl.gz", gzip.open
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise ValueError("Parent event record was not found")
    total, lines = 0, []
    with opener(path, "rb") as stream:
        for _ in range(count):
            line = stream.readline(MAX_BYTES + 1)
            if not line.endswith(b"\n"):
                raise ValueError("Parent event prefix is incomplete")
            total += len(line)
            if total > 128 * 1024**2:
                raise ValueError("Parent prefix exceeds the portable evidence limit")
            try:
                if not isinstance(json.loads(line), dict):
                    raise ValueError("Expected event object")
            except ValueError as exc:
                raise ValueError("Parent event prefix is malformed") from exc
            lines.append(line)
    raw = b"".join(lines)
    return raw, hashlib.sha256(raw).hexdigest()
