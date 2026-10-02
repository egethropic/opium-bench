#!/usr/bin/env python3
"""Read-only first-run check; no installation, download, model load or inference.

Package inspection uses the selected worker Python without importing packages.
Only --check-cuda imports torch and queries availability; no tensors are created.
An OK result is an environment check, not model/adapter or kernel validation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True
from lab.resources import ResourceGuard, ResourceStop

ROOT = Path(__file__).resolve().parent
PROFILES = {"4b": "requirements.txt", "27b": "requirements-27b.txt"}
PROBE = r'''
import importlib.metadata as md, json, platform, sys
request = json.loads(sys.argv[1])
packages = {}
for name in request["packages"]:
    try: packages[name] = md.version(name)
    except md.PackageNotFoundError: packages[name] = None
result = dict(executable=sys.executable, prefix=sys.prefix, python=platform.python_version(),
              python_minor=list(sys.version_info[:2]), implementation=platform.python_implementation(),
              system=platform.system(), machine=platform.machine(), packages=packages,
              cuda=dict(status="not_checked"))
if request["cuda"]:
    try:
        import torch
        result["cuda"] = dict(status="checked", available=bool(torch.cuda.is_available()),
            device_count=torch.cuda.device_count(), torch_cuda_version=torch.version.cuda,
            cxx11_abi=getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None),
            torch_version=torch.__version__, tensor_allocation_performed=False)
    except Exception as exc:
        result["cuda"] = dict(status="error", error=type(exc).__name__ + ": " + str(exc))
print(json.dumps(result, allow_nan=False))
'''


def requirements(path):
    """Read this repository's exact pins and pinned wheel URL; fail on new syntax."""
    rows = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pin = re.fullmatch(r"([\w.-]+)==([\w.+-]+)", line)
        wheel = re.fullmatch(r"([\w.-]+)\s+@\s+(https://\S+)", line)
        if pin:
            rows.append(dict(package=pin[1], expected=pin[2], source="exact_pin"))
        elif wheel:
            url = urlsplit(wheel[2])
            filename = unquote(url.path.rsplit("/", 1)[-1])
            parts = filename.split("-")
            normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
            if (len(parts) < 5 or not filename.endswith(".whl") or
                    normalize(parts[0]) != normalize(wheel[1]) or
                    not re.fullmatch(r"sha256=[a-f0-9]{64}", url.fragment)):
                raise ValueError(f"Unsupported pinned wheel at {path}:{lineno}")
            # Upstream's wheel filename has a CUDA/ABI suffix but its actual
            # distribution METADATA reports the public version (1.7.0).
            rows.append(dict(package=wheel[1], expected=parts[1].split("+", 1)[0], source="pinned_wheel",
                             wheel_filename_version=parts[1],
                             wheel_sha256=url.fragment.split("=", 1)[1],
                             provenance_check="Public metadata version only; wheel build identity and installed bytes are not authenticated"))
        else:
            raise ValueError(f"Unsupported requirement syntax at {path}:{lineno}: {line}")
    return rows


def matching_version(installed, expected):
    # Exact public pins permit a local build tag (e.g. torch 2.8.0+cu128).
    # A wheel containing an explicit local tag must match that tag too.
    return installed is not None and (installed == expected if "+" in expected else installed.split("+", 1)[0] == expected)


def probe_python(python, packages, check_cuda=False):
    selected = os.fspath(python)
    executable = shutil.which(selected) if os.sep not in selected else os.path.abspath(os.path.expanduser(selected))
    if not executable:
        raise ValueError(f"Selected Python is unavailable: {selected}")
    # Do not resolve executable symlinks: that would silently bypass a venv.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTORCH_NVML_BASED_CUDA_CHECK="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PIP_NO_INDEX="1")
    result = subprocess.run([executable, "-B", "-c", PROBE,
                             json.dumps(dict(packages=packages, cuda=check_cuda))],
                            capture_output=True, text=True, timeout=30, env=env, shell=False)
    if result.returncode:
        raise ValueError(f"Selected Python probe exited {result.returncode}: {result.stderr[-2000:]}")
    try:
        observed = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError("Selected Python returned unexpected probe output") from exc
    observed["requested_executable"] = selected
    if result.stderr.strip():
        observed["stderr"] = result.stderr[-2000:]
    return observed


def inspect_runtime(*, python, profile, data_dir, cache_dir, temp_dir=None,
                    check_cuda=False, guard_factory=ResourceGuard, probe=probe_python):
    requirement_path = ROOT / PROFILES[profile]
    pins = requirements(requirement_path)
    destinations = {name: str(Path(value).expanduser().resolve()) for name, value in
                    (("data", data_dir), ("cache", cache_dir), ("temp", temp_dir or Path(data_dir) / "tmp"))}
    report = dict(kind="opium-bench/runtime-check", schema_version=1, profile=profile,
                  requirements_file=str(requirement_path), destinations=destinations,
                  read_only=True, issues=[], limitations=[
                      "Does not test model compatibility, calibration, GPU memory capacity or kernel correctness.",
                      "Storage is a current reserve check, without a model/download/install size reservation.",
                      "Metadata version matches do not authenticate installed package contents."])
    try:
        observed = probe(python, [row["package"] for row in pins], check_cuda)
        report["interpreter"] = {k: v for k, v in observed.items() if k not in {"packages", "cuda"}}
        report["packages"] = [{**row, "installed": observed["packages"].get(row["package"]),
                               "status": "match" if matching_version(observed["packages"].get(row["package"]), row["expected"])
                               else "missing" if observed["packages"].get(row["package"]) is None else "version_mismatch"}
                              for row in pins]
        report["issues"].extend(f"{row['package']}: {row['status']} (expected {row['expected']}, installed {row['installed']})"
                                for row in report["packages"] if row["status"] != "match")
        if observed["python_minor"] != [3, 12] or observed["implementation"] != "CPython":
            report["issues"].append("Reference worker requires CPython 3.12")
        if observed["system"] != "Linux" or observed["machine"] != "x86_64":
            report["issues"].append("Reference worker requires Linux/WSL x86_64; this platform is not validated")
        report["cuda"] = observed["cuda"]
        if check_cuda:
            cuda = observed["cuda"]
            if cuda.get("status") != "checked" or not cuda.get("available"):
                report["issues"].append("CUDA is unavailable or its explicit check failed")
            if cuda.get("torch_cuda_version") != "12.8":
                report["issues"].append("Reference PyTorch CUDA build requires CUDA 12.8")
            if profile == "27b" and cuda.get("cxx11_abi") is not True:
                report["issues"].append("Pinned 27B causal-conv1d wheel requires the C++11 ABI")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["issues"].append(str(exc))
    try:
        guard = guard_factory(destinations)
        report["storage"] = guard.check(force=True)
    except ResourceStop as exc:
        report["storage"] = exc.to_dict()
        report["issues"].append(str(exc))
    except (OSError, ValueError) as exc:
        report["storage"] = dict(status="unavailable", error=str(exc))
        report["issues"].append(str(exc))
    report["status"] = "issues_found" if report["issues"] else "checks_passed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Worker interpreter; venv symlinks are preserved")
    parser.add_argument("--profile", choices=PROFILES, default="4b")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("OPIUM_DATA_DIR", ROOT / "data")))
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache root; defaults to HF_HOME or data-dir/hf-cache")
    parser.add_argument("--temp-dir", type=Path, help="Worker temporary destination; defaults to data-dir/tmp")
    parser.add_argument("--check", action="store_true", help="Run the read-only check (also the default)")
    parser.add_argument("--check-cuda", action="store_true", help="Explicitly import torch and query CUDA availability; no tensors/model")
    args = parser.parse_args(argv)
    report = inspect_runtime(python=args.python, profile=args.profile, data_dir=args.data_dir,
                             cache_dir=args.cache_dir or os.environ.get("HF_HOME", args.data_dir / "hf-cache"),
                             temp_dir=args.temp_dir, check_cuda=args.check_cuda)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "checks_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
