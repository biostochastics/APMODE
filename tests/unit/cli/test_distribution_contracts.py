# SPDX-License-Identifier: GPL-2.0-or-later
"""Release/documentation contracts that are easy to break without source tests."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[3]


def test_wheel_includes_runtime_policies() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["hatch"]["version"]["fallback-version"] == "0.7.0-rc1"
    assert "matplotlib>=3.8,<3.11" in config["project"]["optional-dependencies"]["bayesian"]
    force_include = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["policies"] == "apmode/policies"

    sdist_include = config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert "/src" in sdist_include
    assert "/policies" in sdist_include
    assert not any("node_modules" in entry or "_cloned" in entry for entry in sdist_include)


def test_ci_uses_current_benchmark_paths_and_excludes_external_r_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "tests/unit/benchmarks/test_benchmark_suite_a.py" in workflow
    assert "tests/unit/benchmarks/test_benchmark_simulation.py" in workflow
    assert "not requires_r" in workflow
    assert "uv build --wheel" in workflow


def test_colab_notebook_uses_current_public_api() -> None:
    notebook = json.loads(
        (ROOT / "notebooks" / "colab_node_quickstart.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    for stale_name in (
        "ingest_csv",
        "profile_dataset",
        "apmode.data.nca",
        "NODERunner",
        "assess_fidelity",
        "distill_node_candidate",
    ):
        assert stale_name not in source
    assert "ingest_nonmem_csv" in source
    assert "NodeBackendRunner" in source
    assert "result.distillation" in source
    assert "uv pip install --system --all-extras -e ." in source
    assert "uv sync --all-extras" not in source


def test_public_docs_do_not_link_to_ignored_prd() -> None:
    public_files = [
        ROOT / "README.md",
        ROOT / "docs" / "adr" / "0004-cross-paradigm-simulation-metric-circularity.md",
        ROOT / "docs-site" / "content" / "docs" / "guide" / "concepts" / "evidence-manifest.mdx",
    ]
    for path in public_files:
        assert "PRD_APMODE_v0.3.md" not in path.read_text(encoding="utf-8")


def test_docs_view_source_url_points_into_docs_site() -> None:
    page = (ROOT / "docs-site" / "app" / "docs" / "[[...slug]]" / "page.tsx").read_text(
        encoding="utf-8"
    )
    assert "/docs-site/content/docs/${page.path}" in page

    package = json.loads((ROOT / "docs-site" / "package.json").read_text(encoding="utf-8"))
    assert package["scripts"]["build"] == "next build --webpack"


def test_internal_markdown_links_resolve() -> None:
    """Keep repository docs and rendered MDX links off dead local targets."""
    roots = [
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "docs",
        ROOT / "docs-site" / "content" / "docs",
    ]
    files: list[Path] = []
    for root in roots:
        files.extend(
            [root]
            if root.is_file()
            else [path for path in root.rglob("*") if path.suffix.lower() in {".md", ".mdx"}]
        )

    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    docs_site_root = ROOT / "docs-site" / "content" / "docs"
    broken: list[str] = []
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in link_pattern.finditer(line):
                raw_target = match.group(1).split("#", 1)[0].split("?", 1)[0]
                target = unquote(raw_target.strip(" <>"))
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                if target.startswith("/docs") and docs_site_root in path.parents:
                    base = docs_site_root / target.removeprefix("/docs").lstrip("/")
                elif target.startswith("/"):
                    continue
                else:
                    base = path.parent / target
                candidates = (
                    base,
                    base.with_suffix(".md"),
                    base.with_suffix(".mdx"),
                    base / "index.md",
                    base / "index.mdx",
                )
                if not any(candidate.exists() for candidate in candidates):
                    broken.append(f"{path.relative_to(ROOT)}:{line_number}: {target}")

    assert not broken, "Broken internal links:\n" + "\n".join(broken)
