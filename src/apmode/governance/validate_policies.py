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
    """Validate a single policy JSON file.

    The ``policies/`` directory hosts two distinct schemas:

    - **GatePolicy** (submission.json / discovery.json / optimization.json)
      — discriminated by the top-level ``lane`` field.
    - **ProfilerPolicy** (profiler.json) — discriminated by
      ``policy_id`` starting with ``profiler/``. Its schema lives in
      :mod:`apmode.data.policy` as a dataclass; we duck-type the
      structural contract here rather than import the dataclass so the
      CI hook has no heavy dependency.

    Unknown shapes are reported so a new policy cannot slip through
    silently.
    """
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [f"{path}: JSON parse error: {e}"]

    if not isinstance(data, dict):
        return [f"{path}: top-level JSON must be an object"]

    if "lane" in data:
        try:
            GatePolicy.model_validate(data)
        except ValidationError as e:
            errors.extend(
                f"{path}: {'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
    elif isinstance(data.get("policy_id"), str) and data["policy_id"].startswith("profiler/"):
        required = {
            "policy_id",
            "policy_version",
            "schema_version",
            "manifest_schema_version",
        }
        missing = sorted(required - data.keys())
        if missing:
            errors.append(f"{path}: ProfilerPolicy missing required fields: {missing}")
    else:
        errors.append(
            f"{path}: unrecognized policy schema — expected top-level 'lane' "
            "(GatePolicy) or 'policy_id' starting with 'profiler/' (ProfilerPolicy)"
        )

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
