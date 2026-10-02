"""Manually invoked compatibility checks against the published 0.2 release.

The fixture records existing evidence and model-visible protocol semantics.
It is a frozen input, not a cache to regenerate when implementation changes.
"""
import hashlib
import json
from pathlib import Path


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def verify_legacy_contract(root=None, include_archives=True):
    from .protocol import TaskEnvironment
    from .service import normalize_config
    from run_lab_study import expand

    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    contract = json.loads((root / "tests/fixtures/legacy_release_contract.json").read_text())
    if contract.get("contract_version") != 1:
        raise ValueError("Unsupported compatibility contract version")
    failures = []
    counts = {"protected_files": 0, "protocols": 0, "recipes": 0, "visible_payloads": 0}

    def compare(label, actual, expected):
        if actual != expected:
            failures.append(label)

    if include_archives:
        for name, expected in contract["protected_files"].items():
            path = (root / name).resolve()
            if root not in path.parents or not path.is_file():
                failures.append(f"Missing or unsafe protected evidence: {name}")
                continue
            compare(f"Changed protected evidence: {name}", hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            counts["protected_files"] += 1
    for name, expected in contract["protocols"].items():
        path = root / name
        raw = path.read_bytes()
        compare(f"Changed protocol file: {name}", hashlib.sha256(raw).hexdigest(), expected["file_sha256"])
        episodes = expand(json.loads(raw))
        compare(f"Changed protocol expansion: {name}", canonical_hash(episodes), expected["expanded_sha256"])
        compare(f"Changed episode count: {name}", len(episodes), expected["episodes"])
        counts["protocols"] += 1
    for name, expected in contract["recipes"].items():
        compare(f"Changed unversioned recipe: {name}", canonical_hash(normalize_config({"id": name})), expected)
        counts["recipes"] += 1
    for case in contract["visible_payloads"]:
        env = TaskEnvironment(case["family"], 3, case["seed"], True, True)
        payload = {"tools": env.tools,
                   "system": env.system_prompt(20, 4096, case["thinking"], case["grammar"]),
                   "task": env.task_prompt()}
        label = "/".join(str(case[key]) for key in ("family", "seed", "thinking", "grammar"))
        compare(f"Changed legacy model-visible payload: {label}", canonical_hash(payload), case["sha256"])
        counts["visible_payloads"] += 1
    return {"release_commit": contract["release_commit"], "checked": counts,
            "passed": not failures, "failures": failures}
