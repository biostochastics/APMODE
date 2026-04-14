# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for cross-paradigm ranking (PRD SS4.3.1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    VPCSummary,
)
from apmode.governance.gates import evaluate_gate3
from apmode.governance.policy import GatePolicy
from apmode.governance.ranking import (
    compute_npe,
    compute_vpc_concordance,
    is_cross_paradigm,
    rank_cross_paradigm,
)

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


def _make_result(
    *,
    model_id: str = "test_model",
    backend: str = "nlmixr2",
    bic: float = 170.0,
    cwres_mean: float = 0.01,
    cwres_sd: float = 1.0,
    vpc_coverage: dict[str, float] | None = None,
) -> BackendResult:
    return BackendResult(
        model_id=model_id,
        backend=backend,  # type: ignore[arg-type]
        converged=True,
        ofv=150.0,
        aic=160.0,
        bic=bic,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=5.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=70.0, category="structural"),
        },
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=200,
            minimization_status="successful",
            wall_time_seconds=45.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(
                cwres_mean=cwres_mean,
                cwres_sd=cwres_sd,
                outlier_fraction=0.02,
            ),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage=vpc_coverage or {"p5": 0.92, "p50": 0.95, "p95": 0.91},
                n_bins=10,
                prediction_corrected=False,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=45.0,
        backend_versions={"nlmixr2": "2.1.2"},
        initial_estimate_source="nca",
    )


def _load_policy(lane: str) -> GatePolicy:
    data = json.loads((POLICY_DIR / f"{lane}.json").read_text())
    return GatePolicy.model_validate(data)


class TestIsCrossParadigm:
    """Detect whether survivors span multiple backends."""

    def test_single_backend(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2")
        r2 = _make_result(model_id="m2", backend="nlmixr2")
        assert is_cross_paradigm([r1, r2]) is False

    def test_mixed_backends(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2")
        r2 = _make_result(model_id="m2", backend="jax_node")
        assert is_cross_paradigm([r1, r2]) is True


class TestVPCConcordance:
    """VPC coverage concordance metric."""

    def test_perfect_coverage(self) -> None:
        r = _make_result(vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90})
        score = compute_vpc_concordance(r)
        assert score == pytest.approx(1.0)

    def test_good_coverage(self) -> None:
        r = _make_result(vpc_coverage={"p5": 0.92, "p50": 0.95, "p95": 0.91})
        score = compute_vpc_concordance(r)
        assert 0.9 < score <= 1.0

    def test_poor_coverage(self) -> None:
        r = _make_result(vpc_coverage={"p5": 0.50, "p50": 0.60, "p95": 0.55})
        score = compute_vpc_concordance(r)
        assert score < 0.7

    def test_no_vpc_returns_zero(self) -> None:
        r = _make_result()
        r.diagnostics.vpc = None
        score = compute_vpc_concordance(r)
        assert score == 0.0


class TestNPE:
    """Normalized prediction error metric."""

    def test_ideal_cwres(self) -> None:
        r = _make_result(cwres_mean=0.0, cwres_sd=1.0)
        npe = compute_npe(r)
        assert npe == pytest.approx(0.0, abs=1e-10)

    def test_biased_mean(self) -> None:
        r = _make_result(cwres_mean=0.5, cwres_sd=1.0)
        npe = compute_npe(r)
        assert npe > 0.0

    def test_inflated_sd(self) -> None:
        r = _make_result(cwres_mean=0.0, cwres_sd=1.5)
        npe = compute_npe(r)
        assert npe > 0.0


class TestRankCrossParadigm:
    """Full cross-paradigm ranking."""

    def test_ranks_by_composite_score(self) -> None:
        r1 = _make_result(
            model_id="classical",
            backend="nlmixr2",
            bic=160.0,
            cwres_mean=0.01,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.91, "p50": 0.93, "p95": 0.90},
        )
        r2 = _make_result(
            model_id="node",
            backend="jax_node",
            bic=155.0,
            cwres_mean=0.1,
            cwres_sd=1.2,
            vpc_coverage={"p5": 0.85, "p50": 0.88, "p95": 0.82},
        )
        result = rank_cross_paradigm([r1, r2])

        assert result.is_cross_paradigm is True
        assert result.qualified_comparison is True
        assert len(result.ranked_candidates) == 2
        assert result.backends_compared == ["jax_node", "nlmixr2"]

    def test_single_backend_not_cross(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2")
        r2 = _make_result(model_id="m2", backend="nlmixr2")
        result = rank_cross_paradigm([r1, r2])
        assert result.is_cross_paradigm is False

    def test_empty_survivors(self) -> None:
        result = rank_cross_paradigm([])
        assert result.is_cross_paradigm is False
        assert result.ranked_candidates == []


class TestGate3CrossParadigm:
    """Gate 3 uses cross-paradigm ranking for mixed backends."""

    def test_within_paradigm_uses_bic(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2", bic=170.0)
        r2 = _make_result(model_id="m2", backend="nlmixr2", bic=160.0)
        policy = _load_policy("submission")
        g3, ranked = evaluate_gate3([r1, r2], policy)

        assert g3.gate_name == "within_paradigm_ranking"
        assert ranked[0].bic == 160.0
        # Should have ranking_method check
        method = next(c for c in g3.checks if c.check_id == "ranking_method")
        assert method.observed == "within_paradigm_bic"

    def test_cross_paradigm_uses_simulation_metrics(self) -> None:
        r1 = _make_result(model_id="classical", backend="nlmixr2", bic=170.0)
        r2 = _make_result(model_id="node", backend="jax_node", bic=165.0)
        policy = _load_policy("discovery")
        g3, ranked = evaluate_gate3([r1, r2], policy)

        assert g3.gate_name == "cross_paradigm_ranking"
        assert len(ranked) == 2
        method = next(c for c in g3.checks if c.check_id == "ranking_method")
        assert method.observed == "cross_paradigm_simulation_based"

    def test_qualified_comparison_flag(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2")
        r2 = _make_result(model_id="m2", backend="jax_node")
        policy = _load_policy("discovery")
        g3, _ranked = evaluate_gate3([r1, r2], policy)

        qc = next(c for c in g3.checks if c.check_id == "qualified_comparison")
        assert qc.passed is True
        assert qc.observed is True

    def test_nlpd_not_used_cross_paradigm(self) -> None:
        """NLPD/BIC is not the primary metric for cross-paradigm."""
        r1 = _make_result(model_id="m1", backend="nlmixr2", bic=200.0)
        r2 = _make_result(
            model_id="m2",
            backend="jax_node",
            bic=150.0,
            cwres_mean=0.5,
            cwres_sd=1.5,
            vpc_coverage={"p5": 0.50, "p50": 0.60, "p95": 0.55},
        )
        policy = _load_policy("discovery")
        g3, ranked = evaluate_gate3([r1, r2], policy)

        # m2 has lower BIC but much worse VPC/NPE
        # Cross-paradigm should not simply pick lowest BIC
        assert g3.gate_name == "cross_paradigm_ranking"
        # The classical model should rank better due to better VPC/NPE
        assert ranked[0].candidate_id == "m1"
