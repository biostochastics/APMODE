# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Bayesian artifact → File entity projection.

Covers the regex extension for ``sampler_config``, ``posterior_summary``,
and ``posterior_draws`` introduced alongside the new emitter methods
(plan Tasks 10-12). The older ``prior_manifest``, ``simulation_protocol``,
``mcmc_diagnostics``, and legacy ``draws`` shapes must still project
identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apmode.bundle.rocrate.entities.bayesian import add_bayesian_artifacts


def _seed(bundle: Path, filenames: list[str]) -> None:
    bdir = bundle / "bayesian"
    bdir.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        if name.endswith(".json"):
            (bdir / name).write_text(json.dumps({"stub": name}) + "\n")
        else:
            (bdir / name).write_bytes(b"PAR1\x00\x00parquet-sentinel")


def _root_graph() -> list[dict[str, Any]]:
    return [{"@id": "./", "@type": "Dataset"}]


def test_projects_new_bayesian_artifact_shapes(tmp_path: Path) -> None:
    """The extended regex must pick up the three new emitter artifacts."""
    _seed(
        tmp_path,
        [
            "cand001_prior_manifest.json",
            "cand001_simulation_protocol.json",
            "cand001_mcmc_diagnostics.json",
            "cand001_sampler_config.json",
            "cand001_posterior_summary.parquet",
            "cand001_posterior_draws.parquet",
            "cand001_draws.parquet",  # legacy sidecar still accepted
        ],
    )
    graph = _root_graph()
    added = add_bayesian_artifacts(graph, tmp_path)
    # Every seeded file must project into the graph.
    assert len(added) == 7
    added_ids = set(added)
    for expected in (
        "bayesian/cand001_prior_manifest.json",
        "bayesian/cand001_simulation_protocol.json",
        "bayesian/cand001_mcmc_diagnostics.json",
        "bayesian/cand001_sampler_config.json",
        "bayesian/cand001_posterior_summary.parquet",
        "bayesian/cand001_posterior_draws.parquet",
        "bayesian/cand001_draws.parquet",
    ):
        assert expected in added_ids


def test_ignores_unmatched_filenames(tmp_path: Path) -> None:
    """Files that don't match the Bayesian suffix set are left alone."""
    _seed(
        tmp_path,
        [
            "cand001_prior_manifest.json",
            "cand001_random_noise.json",  # unknown kind
            "not_a_candidate_at_all.json",
        ],
    )
    graph = _root_graph()
    added = add_bayesian_artifacts(graph, tmp_path)
    assert added == ["bayesian/cand001_prior_manifest.json"]


def test_encoding_format_selects_parquet_mime(tmp_path: Path) -> None:
    _seed(tmp_path, ["cand001_posterior_draws.parquet"])
    graph = _root_graph()
    add_bayesian_artifacts(graph, tmp_path)
    file_entity = next(
        e for e in graph if e.get("@id") == "bayesian/cand001_posterior_draws.parquet"
    )
    assert file_entity["encodingFormat"] == "application/vnd.apache.parquet"


def test_absent_bayesian_dir_is_noop(tmp_path: Path) -> None:
    graph = _root_graph()
    assert add_bayesian_artifacts(graph, tmp_path) == []
    assert graph == _root_graph()
