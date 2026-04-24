# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for :func:`apmode.bayes.harness.build_loo_summary` (plan Task 18).

The helper computes PSIS-LOO via ``arviz.loo`` over a Stan-style
``InferenceData`` and projects onto the bundle's :class:`LOOSummary`
shape. Failure modes (missing ``log_lik`` group, arviz unavailable) must
return ``status="not_computed"`` with the captured reason — Gate 1
Bayesian treats that as a warning, never a hard fail.

Tests cover:

* Happy path on a synthetic ``arviz.from_dict``-built ``InferenceData``
  with a well-behaved ``log_lik`` group: ``status="computed"``,
  ``elpd_loo`` finite, Pareto-k counts populated.
* Bin labels match the Vehtari 2017 bands (``good`` / ``ok`` / ``bad`` /
  ``very_bad``).
* Missing ``log_lik`` variable → ``status="not_computed"`` with a reason.
* Bundle emitter writes the summary to
  ``bayesian/{cid}_loo_summary.json`` and the on-disk JSON round-trips
  back through :class:`LOOSummary`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from apmode.bayes.harness import _bin_pareto_k, build_loo_summary  # noqa: E402
from apmode.bundle.emitter import BundleEmitter  # noqa: E402
from apmode.bundle.models import LOOSummary  # noqa: E402


def _synth_inference_data(
    n_chains: int = 4,
    n_draws: int = 200,
    n_obs: int = 50,
    *,
    include_log_lik: bool = True,
    seed: int = 7,
) -> object:
    """Build a tiny InferenceData with optional ``log_lik`` for arviz.loo.

    log_lik values are drawn from N(-1, 0.5) — typical magnitude for a
    well-behaved likelihood under tight priors. Per-chain noise keeps
    arviz.loo from short-circuiting on a degenerate sample.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, dict[str, np.ndarray]] = {
        "posterior": {"theta": rng.standard_normal(size=(n_chains, n_draws))},
    }
    if include_log_lik:
        data["log_likelihood"] = {
            "log_lik": rng.normal(loc=-1.0, scale=0.5, size=(n_chains, n_draws, n_obs)),
        }
    return az.from_dict(data)


# --- Happy path ----------------------------------------------------------


def test_loo_summary_on_well_behaved_idata() -> None:
    idata = _synth_inference_data()
    summary = build_loo_summary(idata, candidate_id="cand001")
    assert summary["candidate_id"] == "cand001"
    assert summary["status"] == "computed"
    assert summary["elpd_loo"] is not None
    assert np.isfinite(summary["elpd_loo"])
    assert summary["se_elpd_loo"] is not None
    assert summary["p_loo"] is not None
    assert summary["n_observations"] == 50
    # Pareto-k counts must sum to n_observations
    assert sum(summary["k_counts"].values()) == 50


def test_pareto_k_bin_labels_cover_four_bands() -> None:
    summary = build_loo_summary(_synth_inference_data(), candidate_id="cand001")
    assert set(summary["k_counts"].keys()) == {"good", "ok", "bad", "very_bad"}


def test_bin_pareto_k_bucketing_matches_arviz_bands() -> None:
    arr = np.array([0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5])
    counts = _bin_pareto_k(arr)
    # good: x <= 0.5 → {0.0, 0.4, 0.5} = 3
    # ok: 0.5 < x <= 0.7 → {0.6, 0.7} = 2
    # bad: 0.7 < x <= 1.0 → {0.8, 1.0} = 2
    # very_bad: x > 1.0 → {1.5} = 1
    assert counts == {"good": 3, "ok": 2, "bad": 2, "very_bad": 1}


# --- Skip path -----------------------------------------------------------


def test_missing_log_lik_returns_not_computed() -> None:
    idata = _synth_inference_data(include_log_lik=False)
    summary = build_loo_summary(idata, candidate_id="cand001")
    assert summary["status"] == "not_computed"
    assert summary["reason"] is not None
    assert "log_lik" in summary["reason"] or "skipped" in summary["reason"]


# --- Emitter round-trip --------------------------------------------------


def test_emitter_writes_loo_summary(tmp_path: Path) -> None:
    em = BundleEmitter(base_dir=tmp_path, run_id="run_loo")
    em.initialize()
    summary = LOOSummary(
        candidate_id="cand001",
        status="computed",
        elpd_loo=-100.0,
        se_elpd_loo=12.3,
        p_loo=4.5,
        pareto_k_max=0.42,
        n_observations=50,
        k_counts={"good": 48, "ok": 2, "bad": 0, "very_bad": 0},
    )
    path = em.write_loo_summary(summary)
    assert path.name == "cand001_loo_summary.json"
    body = json.loads(path.read_text())
    assert body["status"] == "computed"
    assert body["k_counts"] == {"good": 48, "ok": 2, "bad": 0, "very_bad": 0}
    # Round-trip through the model — schema enforcement
    restored = LOOSummary.model_validate(body)
    assert restored.elpd_loo == -100.0


def test_emitter_writes_not_computed_payload(tmp_path: Path) -> None:
    em = BundleEmitter(base_dir=tmp_path, run_id="run_loo")
    em.initialize()
    summary = LOOSummary(
        candidate_id="cand001",
        status="not_computed",
        reason="log_lik missing in generated quantities",
    )
    path = em.write_loo_summary(summary)
    body = json.loads(path.read_text())
    assert body["status"] == "not_computed"
    assert body["reason"].startswith("log_lik")
    # Numeric fields stay None — schema makes that explicit
    assert body["elpd_loo"] is None
    assert body["k_counts"] == {}
