#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Single-source README values from the codebase.

Reads authoritative values (test count, declared package version, policy
versions, transform count, CLI-command count, dataset-registry count) from
the source tree and rewrites the regions between named markers in
``README.md`` + ``CLAUDE.md``:

    <!-- apmode:AUTO:<key> -->VALUE<!-- apmode:/AUTO:<key> -->

Usage:
    uv run python scripts/sync_readme.py [--check]

``--check`` exits 1 if any value is out of date — suitable for CI.

Keys emitted:

    version           Declared package version (from pyproject ``fallback-version``
                      + the ``[project.scripts]`` version-file if present, or the
                      nearest annotated tag with a rc suffix). We prefer the
                      changelog's most recent ``## [X.Y.Z]`` header because the
                      vcs-generated ``_version.py`` carries dev-revision suffixes
                      that are noisy to show to end users.
    version_tag       ``vX.Y.Z`` form for the badge.
    tests             Output of ``pytest --collect-only -q`` tail line.
    tests_nonlive     Collected tests minus those marked ``live`` (best-effort via
                      ``-m 'not live'``); falls back to the raw count when the
                      deselect machinery errors.
    policy_gate       Gate-policy schema version (from ``policies/submission.json``
                      — all three lanes must match or the script errors).
    policy_profiler   Profiler policy version (``policies/profiler.json``).
    profiler_manifest Profiler manifest schema version (int).
    transforms        Count of ``FormularTransform`` members (``dsl/transforms.py`` +
                      ``dsl/prior_transforms.py``).
    cli_cmds          Count of ``@app.command`` decorators in ``cli.py``.
    datasets          Count of ``DatasetInfo(...)`` entries in ``data/datasets.py``.
    backends          Count of non-dunder files under ``src/apmode/backends/``
                      (bayes/ lives outside).

Exit codes: 0 success, 1 drift (``--check``) or tool failure.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CLAUDE_MD = ROOT / "CLAUDE.md"
POLICY_DIR = ROOT / "policies"
CLI_FILE = ROOT / "src/apmode/cli.py"
DATASETS_FILE = ROOT / "src/apmode/data/datasets.py"
TRANSFORMS_FILES = [
    ROOT / "src/apmode/dsl/transforms.py",
    ROOT / "src/apmode/dsl/prior_transforms.py",
]
CHANGELOG = ROOT / "CHANGELOG.md"
BACKENDS_DIR = ROOT / "src/apmode/backends"

MARKER_RE = re.compile(
    r"<!--\s*apmode:AUTO:(?P<key>[a-z0-9_]+)\s*-->"
    r"(?P<body>.*?)"
    r"<!--\s*apmode:/AUTO:(?P<close>[a-z0-9_]+)\s*-->",
    re.DOTALL,
)


def _shell(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def collect_tests() -> tuple[str, str]:
    """Return (total, non_live) test counts as strings."""
    try:
        out = _shell(["uv", "run", "pytest", "tests/", "--collect-only", "-q"])
    except subprocess.CalledProcessError as e:
        print(f"pytest collect failed: {e.output[-400:]}", file=sys.stderr)
        return ("?", "?")
    m = re.search(r"(\d+)\s+tests? collected", out)
    total = m.group(1) if m else "?"
    try:
        out2 = _shell(["uv", "run", "pytest", "tests/", "--collect-only", "-q", "-m", "not live"])
        m2 = re.search(r"(\d+)/\d+ tests collected", out2) or re.search(
            r"(\d+)\s+tests? collected", out2
        )
        nonlive = m2.group(1) if m2 else total
    except subprocess.CalledProcessError:
        nonlive = total
    return (total, nonlive)


def collect_version() -> tuple[str, str]:
    """Pull the canonical release version from the most recent CHANGELOG header."""
    text = CHANGELOG.read_text()
    m = re.search(r"^##\s+\[(?P<v>[0-9]+\.[0-9]+\.[0-9]+(?:-[a-z0-9]+)?)\]", text, re.M)
    if not m:
        return ("0.0.0", "v0.0.0")
    v = m.group("v")
    return (v, f"v{v}")


def collect_policy_versions() -> tuple[str, str, int]:
    """Return (gate_policy_version, profiler_policy_version, manifest_schema_version)."""
    gate_versions = set()
    for lane in ("submission", "discovery", "optimization"):
        pj = json.loads((POLICY_DIR / f"{lane}.json").read_text())
        gate_versions.add(pj["policy_version"])
    if len(gate_versions) != 1:
        raise SystemExit(
            f"Gate policy versions diverge across lanes: {sorted(gate_versions)}. "
            f"Fix policies/*.json before syncing the README."
        )
    profiler = json.loads((POLICY_DIR / "profiler.json").read_text())
    return (
        gate_versions.pop(),
        profiler["policy_version"],
        int(profiler["manifest_schema_version"]),
    )


def collect_transform_count() -> int:
    count = 0
    for f in TRANSFORMS_FILES:
        count += len(re.findall(r"^class\s+[A-Z]\w*\(BaseModel\):", f.read_text(), re.M))
    return count


def collect_cli_command_count() -> int:
    return len(re.findall(r"^@app\.command\(", CLI_FILE.read_text(), re.M))


def collect_dataset_count() -> int:
    return CLI_FILE.read_text().count("DatasetInfo(") + DATASETS_FILE.read_text().count(
        "DatasetInfo("
    )  # DatasetInfo only appears in datasets.py but the OR preserves safety


def collect_backend_count() -> int:
    return sum(
        1
        for p in BACKENDS_DIR.iterdir()
        if p.is_file() and p.suffix == ".py" and not p.name.startswith("_")
    )


def build_values() -> dict[str, str]:
    tests_total, tests_nonlive = collect_tests()
    version, version_tag = collect_version()
    gate_ver, profiler_ver, manifest_ver = collect_policy_versions()
    return {
        "version": version,
        "version_tag": version_tag,
        "tests": tests_total,
        "tests_nonlive": tests_nonlive,
        "policy_gate": gate_ver,
        "policy_profiler": profiler_ver,
        "profiler_manifest": str(manifest_ver),
        "transforms": str(collect_transform_count()),
        "cli_cmds": str(collect_cli_command_count()),
        "datasets": str(DATASETS_FILE.read_text().count("DatasetInfo(")),
        "backends": str(collect_backend_count()),
    }


def replace_markers(text: str, values: dict[str, str]) -> tuple[str, list[tuple[str, str, str]]]:
    changes: list[tuple[str, str, str]] = []

    def repl(m: re.Match[str]) -> str:
        key, body, close = m.group("key"), m.group("body"), m.group("close")
        if key != close:
            raise SystemExit(f"Marker mismatch: open={key} close={close}")
        new = values.get(key, f"???:{key}")
        if body != new:
            changes.append((key, body, new))
        return f"<!-- apmode:AUTO:{key} -->{new}<!-- apmode:/AUTO:{key} -->"

    return MARKER_RE.sub(repl, text), changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if README/CLAUDE.md is out of date instead of writing.",
    )
    args = parser.parse_args()

    values = build_values()
    print("Resolved values:")
    for k, v in values.items():
        print(f"  {k:20s} = {v}")

    any_drift = False
    for path in (README, CLAUDE_MD):
        if not path.exists():
            continue
        text = path.read_text()
        new, changes = replace_markers(text, values)
        if changes:
            any_drift = True
            print(f"\n{path.name}: {len(changes)} drift(s)")
            for key, old, newv in changes:
                old_s = old.replace("\n", "\\n")
                print(f"  {key}: {old_s!r} → {newv!r}")
            if not args.check:
                path.write_text(new)
                print(f"  → wrote {path.name}")
    if args.check and any_drift:
        print("\nDrift detected. Run without --check to update.", file=sys.stderr)
        return 1
    if not any_drift:
        print("\nNo drift. README is in sync with source of truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
