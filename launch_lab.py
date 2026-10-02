#!/usr/bin/env python3
"""Launch the local Opium Bench. The web server itself needs only Python."""
import argparse
import os
from pathlib import Path
import sys

from lab.service import LabService
from lab.server import create_server
from lab.storage import ROOT


def defaults():
    workspace_runtime = ROOT.parent.parent / "work" / "runtime"
    if workspace_runtime.exists():
        runtime = workspace_runtime.resolve()
        return runtime / "lab-data", runtime / "hf-cache", workspace_runtime / "venv" / "bin" / "python"
    base = ROOT / "data"
    return base, base / "hf-cache", Path(sys.executable)


def main():
    data, cache, python = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("OPIUM_DATA_DIR", data))
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("HF_HOME", cache))
    parser.add_argument("--python", type=Path, default=python, help="Python with the GPU runtime installed")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be 1024–65535")
    service = LabService(args.data_dir, args.cache_dir, args.python)
    server = create_server(service, port=args.port)
    print(f"Opium Bench: http://localhost:{args.port}\nData: {service.store.root}\nModel cache: {service.cache_dir}", flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        service.close()
        server.server_close()


if __name__ == "__main__":
    main()
