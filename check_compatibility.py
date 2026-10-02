#!/usr/bin/env python3
"""Verify published evidence bytes and legacy protocol behavior locally."""
import json
from lab.compatibility import verify_legacy_contract


if __name__ == "__main__":
    result = verify_legacy_contract()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
