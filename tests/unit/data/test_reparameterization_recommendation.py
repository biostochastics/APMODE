# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the advisory reparameterization artifact (plan Task 25).

APMODE never switches parameterization automatically — divergences must
surface to the operator via a dedicated JSON artifact, and the Gate 1
Bayesian check routes through it. This module covers:

* ``build_reparameterization_recommendation`` escalates to
  ``switch_to_non_centered`` above the 5% divergence-fraction threshold.
* Low-rate divergences recommend ``refit_with_higher_adapt_delta``.
* Tree-depth saturations alone (zero divergences) still emit the
  advisory, pointing at adapt_delta / max_treedepth rather than the
  parameterization.
* A clean fit (no divergences, no saturations) returns ``None`` so the
  emitter skips the artifact.
* The bundle emitter writes to
  ``bayesian/{candidate_id}_reparameterization_recommendation.json`` and
  rejects unsafe candidate ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bayes.harness import build_reparameterization_recommendation
from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import ReparameterizationRecommendation


def _cfg(chains: int = 4, sampling: int = 1000, adapt_delta: float = 0.8) -> dict[str, object]:
    return {"chains": chains, "sampling": sampling, "adapt_delta": adapt_delta}


def _diag(n_divergent: int = 0, n_max_treedepth: int = 0) -> dict[str, object]:
    return {
        "rhat_max": 1.0,
        "ess_bulk_min": 500.0,
        "ess_tail_min": 500.0,
        "n_divergent": n_divergent,
        "n_max_treedepth": n_max_treedepth,
    }


# --- Helper logic --------------------------------------------------------


def test_clean_fit_returns_none() -> None:
    rec = build_reparameterization_recommendation(_diag(), _cfg(), candidate_id="cand001")
    assert rec is None


def test_high_divergence_rate_recommends_non_centered() -> None:
    # 4 chains * 1000 samples = 4000. 5% threshold = 200. 250 > 200 → switch.
    rec = build_reparameterization_recommendation(
        _diag(n_divergent=250), _cfg(), candidate_id="cand001"
    )
    assert rec is not None
    assert rec["recommended_action"] == "switch_to_non_centered"
    assert rec["divergence_count"] == 250
    assert 0.06 <= rec["divergence_fraction"] <= 0.07
    assert "funnel" in rec["rationale"].lower() or "non-centered" in rec["rationale"]


def test_low_divergence_rate_recommends_higher_adapt_delta() -> None:
    # 10 divergences / 4000 = 0.25% — well below threshold
    rec = build_reparameterization_recommendation(
        _diag(n_divergent=10), _cfg(), candidate_id="cand001"
    )
    assert rec is not None
    assert rec["recommended_action"] == "refit_with_higher_adapt_delta"
    assert rec["divergence_count"] == 10
    assert "adapt_delta" in rec["rationale"]


def test_treedepth_only_recommends_higher_adapt_delta() -> None:
    rec = build_reparameterization_recommendation(
        _diag(n_max_treedepth=5), _cfg(), candidate_id="cand001"
    )
    assert rec is not None
    assert rec["recommended_action"] == "refit_with_higher_adapt_delta"
    assert rec["divergence_count"] == 0
    assert rec["max_treedepth_count"] == 5
    assert "tree-depth" in rec["rationale"].lower() or "treedepth" in rec["rationale"].lower()


def test_candidate_id_is_stamped_on_payload() -> None:
    rec = build_reparameterization_recommendation(
        _diag(n_divergent=1), _cfg(), candidate_id="cand_abc"
    )
    assert rec is not None
    assert rec["candidate_id"] == "cand_abc"


def test_divergence_fraction_capped_at_one() -> None:
    """Edge case: divergences > total_iter shouldn't produce fraction > 1."""
    rec = build_reparameterization_recommendation(
        _diag(n_divergent=5000), _cfg(), candidate_id="cand001"
    )
    assert rec is not None
    # Not strictly capped in the helper — we cap via the model's le=1.0 — so
    # the raw fraction here can be > 1.0 but the pydantic model will reject.
    # Assert the helper reports the raw value; the downstream emitter refuses.
    assert rec["divergence_fraction"] > 1.0


# --- Bundle emitter ------------------------------------------------------


def test_emitter_writes_artifact_under_bayesian_dir(tmp_path: Path) -> None:
    em = BundleEmitter(base_dir=tmp_path, run_id="run_reparam")
    em.initialize()
    rec = ReparameterizationRecommendation(
        candidate_id="cand001",
        divergence_count=250,
        divergence_fraction=0.0625,
        max_treedepth_count=0,
        recommended_action="switch_to_non_centered",
        rationale="Divergence rate above threshold.",
    )
    path = em.write_reparameterization_recommendation(rec)
    assert path.name == "cand001_reparameterization_recommendation.json"
    assert path.parent.name == "bayesian"
    body = json.loads(path.read_text())
    assert body["recommended_action"] == "switch_to_non_centered"
    assert body["candidate_id"] == "cand001"
    assert body["divergence_count"] == 250


def test_emitter_rejects_unsafe_candidate_id(tmp_path: Path) -> None:
    em = BundleEmitter(base_dir=tmp_path, run_id="run_reparam")
    em.initialize()
    rec = ReparameterizationRecommendation(
        candidate_id="../escape",
        divergence_count=1,
        divergence_fraction=0.0002,
        max_treedepth_count=0,
        recommended_action="refit_with_higher_adapt_delta",
        rationale="tests",
    )
    with pytest.raises(ValueError, match=r"unsafe characters"):
        em.write_reparameterization_recommendation(rec)


def test_model_rejects_fraction_above_one() -> None:
    with pytest.raises(ValueError, match=r"less than or equal to 1"):
        ReparameterizationRecommendation(
            candidate_id="cand001",
            divergence_count=1,
            divergence_fraction=1.5,
            recommended_action="refit_with_higher_adapt_delta",
            rationale="bad",
        )


def test_model_rejects_negative_divergence_count() -> None:
    with pytest.raises(ValueError, match=r"greater than or equal to 0"):
        ReparameterizationRecommendation(
            candidate_id="cand001",
            divergence_count=-1,
            divergence_fraction=0.0,
            recommended_action="refit_with_higher_adapt_delta",
            rationale="bad",
        )
