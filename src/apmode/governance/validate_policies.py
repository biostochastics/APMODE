# SPDX-License-Identifier: GPL-2.0-or-later
"""CI validation hook for gate policy files.

Usage: python -m apmode.governance.validate_policies policies/
Validates all .json files in the given directory against the GatePolicy schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from apmode.governance.policy import GatePolicy


def validate_policy_file(path: Path) -> list[str]:
    """Validate a single policy JSON file. Returns list of error messages."""
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"{path}: JSON parse error: {e}"]

    try:
        GatePolicy.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{path}: {loc}: {err['msg']}")

    return errors


def main(policy_dir: str) -> int:
    """Validate all policy files in a directory. Returns exit code."""
    policy_path = Path(policy_dir)
    if not policy_path.is_dir():
        print(f"Error: {policy_dir} is not a directory", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    json_files = list(policy_path.glob("*.json"))
    if not json_files:
        print(f"Warning: no .json files found in {policy_dir}", file=sys.stderr)
        return 0

    for f in sorted(json_files):
        errors = validate_policy_file(f)
        all_errors.extend(errors)

    if all_errors:
        for err in all_errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1

    print(f"OK: {len(json_files)} policy file(s) validated successfully")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <policy_dir>", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
