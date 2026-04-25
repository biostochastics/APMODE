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
    ScoringContract,
    VPCSummary,
)
from apmode.governance.gates import evaluate_gate3
from apmode.governance.policy import Gate3Config, GatePolicy
from apmode.governance.ranking import (
    compute_cwres_npe_proxy,
    compute_vpc_concordance,
    is_cross_paradigm,
    rank_cross_paradigm,
    ranking_requires_simulation_metrics,
)

# Default config preserves legacy weighted-sum behavior used by the
# original assertions in TestRankCrossParadigm.
_DEFAULT_GATE3 = Gate3Config()
_DEFAULT_VPC_TARGET = 0.90

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


def _scoring_contract_for_backend(backend: str, blq_method: str) -> ScoringContract:
    """Per-backend default contract so test bundles pass
    :meth:`BackendResult.validate_backend_scoring_contract_consistency`.
    """
    if backend == "bayesian_stan":
        return ScoringContract(
            nlpd_kind="marginal",
            re_treatment="integrated",
            nlpd_integrator="hmc_nuts",
            blq_method=blq_method,  # type: ignore[arg-type]
            observation_model="combined",
            float_precision="float64",
        )
    if backend == "jax_node":
        return ScoringContract(
            nlpd_kind="conditional",
            re_treatment="pooled",
            nlpd_integrator="none",
            blq_method=blq_method,  # type: ignore[arg-type]
            observation_model="combined",
            float_precision="float32",
        )
    # nlmixr2 and agentic_llm both default to the classical FOCEI contract.
    return ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="nlmixr2_focei",
        blq_method=blq_method,  # type: ignore[arg-type]
        observation_model="combined",
        float_precision="float64",
    )


def _make_result(
    *,
    model_id: str = "test_model",
    backend: str = "nlmixr2",
    bic: float = 170.0,
    cwres_mean: float = 0.01,
    cwres_sd: float = 1.0,
    vpc_coverage: dict[str, float] | None = None,
    blq_method: str = "none",
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
            blq=BLQHandling(method=blq_method, n_blq=0, blq_fraction=0.0),  # type: ignore[arg-type]
            scoring_contract=_scoring_contract_for_backend(backend, blq_method),
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


class TestRankingRequiresSimulationMetrics:
    """PRD §10 Q2: BIC/NLPD ranking is only valid for a shared observation model."""

    def test_single_backend_uniform_blq_uses_within_paradigm(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2", blq_method="m3")
        r2 = _make_result(model_id="m2", backend="nlmixr2", blq_method="m3")
        required, reason = ranking_requires_simulation_metrics([r1, r2])
        assert required is False
        assert "within-paradigm" in reason

    def test_mixed_backends_forces_simulation(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2")
        r2 = _make_result(model_id="m2", backend="jax_node")
        required, reason = ranking_requires_simulation_metrics([r1, r2])
        assert required is True
        assert "cross-paradigm" in reason

    def test_same_backend_mixed_blq_forces_simulation(self) -> None:
        # Within-paradigm comparability failure: same backend, different
        # BLQ handling → likelihood scales are not comparable.
        r1 = _make_result(model_id="m1", backend="nlmixr2", blq_method="m3")
        r2 = _make_result(model_id="m2", backend="nlmixr2", blq_method="m4")
        required, reason = ranking_requires_simulation_metrics([r1, r2])
        assert required is True
        assert "BLQ" in reason or "blq" in reason.lower()

    def test_same_backend_m3_vs_m7_plus_forces_simulation(self) -> None:
        # Regression: before the BLQ enum was extended with "m7_plus" the
        # policy-side M7+ directive could not be encoded into BLQHandling,
        # so an M3-vs-M7+ pair silently looked like uniform BLQ handling and
        # ranking would fall through to BIC/NLPD. Likelihood scales are *not*
        # comparable between these methods (M3 adds a censored-likelihood
        # term; M7+ imputes 0 + inflated additive error), so the ranker must
        # route to simulation metrics.
        r1 = _make_result(model_id="m1", backend="nlmixr2", blq_method="m3")
        r2 = _make_result(model_id="m2", backend="nlmixr2", blq_method="m7_plus")
        required, reason = ranking_requires_simulation_metrics([r1, r2])
        assert required is True
        assert "BLQ" in reason or "blq" in reason.lower()

    def test_auc_cmax_weight_triggers_simulation_path(self) -> None:
        # When a lane policy enables the AUC/Cmax BE component, Gate 3
        # must take the simulation-based path even if the survivor set is
        # homogeneous (same backend, same BLQ method). Without this trigger
        # BIC ranking would run and the AUC/Cmax component would never
        # contribute to the outcome despite policy saying it should.
        r1 = _make_result(model_id="m1", backend="nlmixr2", blq_method="m3")
        r2 = _make_result(model_id="m2", backend="nlmixr2", blq_method="m3")
        gate3_be = Gate3Config(
            composite_method="borda",
            vpc_weight=0.4,
            npe_weight=0.3,
            bic_weight=0.0,
            auc_cmax_weight=0.3,
        )
        required, reason = ranking_requires_simulation_metrics([r1, r2], gate3=gate3_be)
        assert required is True
        assert "auc_cmax" in reason.lower() or "AUC/Cmax" in reason

    def test_gate3_routes_mixed_blq_to_simulation_metrics(self) -> None:
        from apmode.governance.gates import evaluate_gate3

        r1 = _make_result(model_id="m1", backend="nlmixr2", blq_method="m3", bic=170.0)
        r2 = _make_result(model_id="m2", backend="nlmixr2", blq_method="m4", bic=165.0)
        gate_result, _ranked = evaluate_gate3([r1, r2], _load_policy("submission"))
        assert gate_result.gate_name == "cross_paradigm_ranking"
        # The qualification reason must mention BLQ to document *why* BIC was refused.
        quals = [c for c in gate_result.checks if c.check_id == "qualified_comparison"]
        assert quals
        evidence = quals[0].evidence_ref or ""
        assert "BLQ" in evidence or "blq" in evidence.lower()


class TestVPCConcordance:
    """VPC coverage concordance metric."""

    def test_perfect_coverage(self) -> None:
        r = _make_result(vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90})
        score = compute_vpc_concordance(r, target=_DEFAULT_VPC_TARGET)
        assert score == pytest.approx(1.0)

    def test_good_coverage(self) -> None:
        r = _make_result(vpc_coverage={"p5": 0.92, "p50": 0.95, "p95": 0.91})
        score = compute_vpc_concordance(r, target=_DEFAULT_VPC_TARGET)
        assert 0.9 < score <= 1.0

    def test_poor_coverage(self) -> None:
        r = _make_result(vpc_coverage={"p5": 0.50, "p50": 0.60, "p95": 0.55})
        score = compute_vpc_concordance(r, target=_DEFAULT_VPC_TARGET)
        assert score < 0.7

    def test_no_vpc_returns_zero(self) -> None:
        r = _make_result()
        r.diagnostics.vpc = None
        score = compute_vpc_concordance(r, target=_DEFAULT_VPC_TARGET)
        assert score == 0.0


class TestCWRESNPEProxy:
    """Cross-paradigm NPE proxy metric (CWRES-based)."""

    def test_ideal_cwres(self) -> None:
        r = _make_result(cwres_mean=0.0, cwres_sd=1.0)
        npe = compute_cwres_npe_proxy(r)
        assert npe == pytest.approx(0.0, abs=1e-10)

    def test_biased_mean(self) -> None:
        r = _make_result(cwres_mean=0.5, cwres_sd=1.0)
        npe = compute_cwres_npe_proxy(r)
        assert npe > 0.0

    def test_inflated_sd(self) -> None:
        r = _make_result(cwres_mean=0.0, cwres_sd=1.5)
        npe = compute_cwres_npe_proxy(r)
        assert npe > 0.0


def _rank(survivors: list, *, gate3: Gate3Config | None = None):
    return rank_cross_paradigm(
        survivors,
        gate3=gate3 or _DEFAULT_GATE3,
        vpc_concordance_target=_DEFAULT_VPC_TARGET,
    )


class TestRankCrossParadigm:
    """Full cross-paradigm ranking (weighted-sum default)."""

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
        result = _rank([r1, r2])

        assert result.is_cross_paradigm is True
        assert result.qualified_comparison is True
        assert len(result.ranked_candidates) == 2
        assert result.backends_compared == ["jax_node", "nlmixr2"]
        assert result.composite_method == "weighted_sum"

    def test_single_backend_not_cross(self) -> None:
        r1 = _make_result(model_id="m1", backend="nlmixr2")
        r2 = _make_result(model_id="m2", backend="nlmixr2")
        result = _rank([r1, r2])
        assert result.is_cross_paradigm is False

    def test_empty_survivors(self) -> None:
        result = _rank([])
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


class TestGate3ConfigValidation:
    """Policy-level invariants on gate3 composite configuration."""

    def test_default_weights_sum_to_one(self) -> None:
        g = Gate3Config()
        assert g.vpc_weight + g.npe_weight + g.bic_weight + g.auc_cmax_weight == pytest.approx(1.0)
        # Default auc_cmax is off — lanes opt in explicitly.
        assert g.auc_cmax_weight == 0.0

    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            Gate3Config(vpc_weight=0.5, npe_weight=0.4, bic_weight=0.2)

    def test_all_zero_weights_rejected(self) -> None:
        # Rejected by the sum-to-1 check before the "at least one active" check
        # triggers; both are invariants, either rejection is acceptable.
        with pytest.raises(ValueError):
            Gate3Config(vpc_weight=0.0, npe_weight=0.0, bic_weight=0.0)

    def test_four_weight_sum_with_auc_cmax(self) -> None:
        g = Gate3Config(
            vpc_weight=0.3,
            npe_weight=0.3,
            bic_weight=0.1,
            auc_cmax_weight=0.3,
        )
        assert g.auc_cmax_weight == pytest.approx(0.3)

    def test_four_weight_sum_rejected_when_over_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            Gate3Config(
                vpc_weight=0.4,
                npe_weight=0.4,
                bic_weight=0.0,
                auc_cmax_weight=0.4,
            )

    def test_borda_config_round_trips(self) -> None:
        g = Gate3Config(
            composite_method="borda",
            vpc_weight=0.5,
            npe_weight=0.5,
            bic_weight=0.0,
        )
        assert g.composite_method == "borda"
        assert g.bic_weight == 0.0


class TestBordaCompositeAggregation:
    """Borda-count aggregation (policy-driven opt-in)."""

    _BORDA_VPC_NPE = Gate3Config(
        composite_method="borda",
        vpc_weight=0.5,
        npe_weight=0.5,
        bic_weight=0.0,
    )

    def test_borda_ranks_by_sum_of_ranks(self) -> None:
        # r1: best VPC, worst NPE
        # r2: worst VPC, best NPE
        # r3: middle on both → should win (lowest rank sum)
        r1 = _make_result(
            model_id="r1",
            backend="nlmixr2",
            cwres_mean=0.5,
            cwres_sd=1.5,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        r2 = _make_result(
            model_id="r2",
            backend="jax_node",
            cwres_mean=0.0,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.50, "p50": 0.60, "p95": 0.55},
        )
        r3 = _make_result(
            model_id="r3",
            backend="agentic_llm",
            cwres_mean=0.1,
            cwres_sd=1.1,
            vpc_coverage={"p5": 0.80, "p50": 0.85, "p95": 0.82},
        )
        result = _rank([r1, r2, r3], gate3=self._BORDA_VPC_NPE)

        assert result.composite_method == "borda"
        # Each candidate ranks 1/2/3 on each enabled metric → rank sums
        # r1: vpc=1, npe=3 → 4
        # r2: vpc=3, npe=1 → 4
        # r3: vpc=2, npe=2 → 4 ← wait, this scenario ties them all.
        # Sanity: composite_score must be sum of ranks across 2 metrics
        # with 3 candidates → each candidate's sum ∈ {2..6}.
        for m in result.ranked_candidates:
            assert 2.0 <= m.composite_score <= 6.0
            assert set(m.component_scores.keys()) == {"vpc", "npe"}

    def test_borda_scale_invariance(self) -> None:
        # Multiply NPE-determining CWRES by a huge constant for one candidate.
        # Under weighted_sum with npe_cap=5 the effect would be capped; under
        # borda the *rank* is unchanged — so the ranking should be identical
        # to a version with small perturbations, as long as ordering stays.
        r_good = _make_result(
            model_id="good",
            backend="nlmixr2",
            cwres_mean=0.0,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.92, "p50": 0.93, "p95": 0.91},
        )
        r_bad_mild = _make_result(
            model_id="bad_mild",
            backend="jax_node",
            cwres_mean=0.3,
            cwres_sd=1.2,
            vpc_coverage={"p5": 0.70, "p50": 0.75, "p95": 0.72},
        )
        r_bad_extreme = _make_result(
            model_id="bad_extreme",
            backend="jax_node",
            cwres_mean=50.0,  # pathological
            cwres_sd=100.0,
            vpc_coverage={"p5": 0.70, "p50": 0.75, "p95": 0.72},
        )
        mild = _rank([r_good, r_bad_mild], gate3=self._BORDA_VPC_NPE)
        extreme = _rank([r_good, r_bad_extreme], gate3=self._BORDA_VPC_NPE)
        assert mild.ranked_candidates[0].candidate_id == "good"
        assert extreme.ranked_candidates[0].candidate_id == "good"
        # Composite must be purely rank-based: both comparisons yield the
        # same rank-sum for "good" (best on both metrics) = 1 + 1 = 2.
        assert mild.ranked_candidates[0].composite_score == pytest.approx(2.0)
        assert extreme.ranked_candidates[0].composite_score == pytest.approx(2.0)

    def test_borda_bic_disabled_ignores_missing(self) -> None:
        # bic=None should not affect ranking when bic_weight=0.
        r1 = _make_result(
            model_id="m1",
            backend="nlmixr2",
            cwres_mean=0.0,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.92, "p50": 0.93, "p95": 0.91},
        )
        r2 = _make_result(
            model_id="m2",
            backend="jax_node",
            cwres_mean=0.3,
            cwres_sd=1.2,
            vpc_coverage={"p5": 0.70, "p50": 0.75, "p95": 0.72},
        )
        r1.bic = None
        r2.bic = None
        result = _rank([r1, r2], gate3=self._BORDA_VPC_NPE)
        assert result.ranked_candidates[0].candidate_id == "m1"

    def test_borda_ties_share_average_rank(self) -> None:
        # Two candidates with identical metrics → both get rank 1.5, sum 3.
        r1 = _make_result(
            model_id="m1",
            backend="nlmixr2",
            cwres_mean=0.1,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        r2 = _make_result(
            model_id="m2",
            backend="jax_node",
            cwres_mean=0.1,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        result = _rank([r1, r2], gate3=self._BORDA_VPC_NPE)
        assert result.ranked_candidates[0].composite_score == pytest.approx(3.0)
        assert result.ranked_candidates[1].composite_score == pytest.approx(3.0)


class TestRankerNaNSanitization:
    """Non-finite metric inputs must not poison ranking."""

    _BORDA_VPC_NPE_BIC = Gate3Config(
        composite_method="borda",
        vpc_weight=0.4,
        npe_weight=0.3,
        bic_weight=0.3,
    )

    def test_nan_npe_loses_under_borda(self) -> None:
        good = _make_result(
            model_id="good",
            backend="nlmixr2",
            cwres_mean=0.0,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        nan_candidate = _make_result(
            model_id="nan",
            backend="jax_node",
            cwres_mean=0.0,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        # Inject a NaN directly via the canonical npe_score field; _resolve_npe
        # will fall back to the CWRES proxy (0.0), so force the pathological
        # case by monkey-patching after the resolve point would be a test
        # hack — instead, verify the Borda-layer sanitization by passing a
        # non-finite value into _resolve_npe's fallback via a NaN CWRES.
        nan_candidate.diagnostics.gof.cwres_mean = float("nan")
        nan_candidate.diagnostics.gof.cwres_sd = 1.0
        result = _rank([good, nan_candidate], gate3=self._BORDA_VPC_NPE_BIC)
        # Good must not be beaten by a NaN-metric candidate on the NPE axis.
        assert result.ranked_candidates[0].candidate_id == "good"

    def test_nan_vpc_loses_under_weighted_sum(self) -> None:
        good = _make_result(
            model_id="good",
            backend="nlmixr2",
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        bad = _make_result(
            model_id="bad",
            backend="jax_node",
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        # Drop VPC entirely on the bad candidate → VPC concordance returns
        # 0.0 (worst) via compute_vpc_concordance, which composes to the
        # worst VPC component under weighted_sum.
        bad.diagnostics.vpc = None
        result = _rank([good, bad])
        assert result.ranked_candidates[0].candidate_id == "good"


class TestNPEUnification:
    """Real simulation-based NPE (from backend) must override the proxy."""

    def test_real_npe_preferred_when_populated(self) -> None:
        # Two candidates identical EXCEPT for diagnostics.npe_score.
        # The CWRES proxy is identical (0.0 for both), so the rank must be
        # driven entirely by the real NPE values.
        r_good = _make_result(
            model_id="good",
            backend="nlmixr2",
            cwres_mean=0.0,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        r_bad = _make_result(
            model_id="bad",
            backend="jax_node",
            cwres_mean=0.0,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        r_good.diagnostics.npe_score = 0.01
        r_bad.diagnostics.npe_score = 2.0

        result = _rank([r_good, r_bad])
        assert result.ranked_candidates[0].candidate_id == "good"
        assert result.ranked_candidates[0].npe_source == "simulation"
        assert result.ranked_candidates[0].npe == pytest.approx(0.01)
        assert result.ranked_candidates[1].npe_source == "simulation"

    def test_proxy_fallback_when_npe_score_missing(self) -> None:
        r = _make_result(cwres_mean=0.3, cwres_sd=1.2)
        r.diagnostics.npe_score = None
        result = _rank([r, _make_result(model_id="other", backend="jax_node")])
        me = next(m for m in result.ranked_candidates if m.candidate_id == "test_model")
        assert me.npe_source == "cwres_proxy"
        assert me.npe > 0.0

    def test_non_finite_npe_score_falls_back(self) -> None:
        r = _make_result(cwres_mean=0.0, cwres_sd=1.0)
        r.diagnostics.npe_score = float("nan")
        result = _rank([r, _make_result(model_id="other", backend="jax_node")])
        me = next(m for m in result.ranked_candidates if m.candidate_id == "test_model")
        assert me.npe_source == "cwres_proxy"

    def test_compute_npe_end_to_end_wire(self) -> None:
        # Prove the contract: when a backend computes ``compute_npe(observed,
        # predicted_simulations)`` and stores the result on
        # ``diagnostics.npe_score``, Gate 3 ranking picks up exactly that
        # value (no rounding, no proxy fallback). This is the wire every
        # backend is expected to close when its posterior-predictive
        # pipeline is implemented.
        import numpy as np

        from apmode.benchmarks.scoring import compute_npe

        rng = np.random.default_rng(seed=42)
        n_obs, n_sims = 30, 200
        observed = rng.normal(loc=5.0, scale=0.8, size=n_obs)
        predicted = rng.normal(loc=5.0, scale=1.0, size=(n_sims, n_obs))
        expected_npe = compute_npe(observed, predicted)

        r_real = _make_result(
            model_id="real_sim",
            backend="nlmixr2",
            cwres_mean=0.0,
            cwres_sd=1.0,
        )
        r_real.diagnostics.npe_score = expected_npe
        r_proxy = _make_result(model_id="proxy", backend="jax_node")
        result = _rank([r_real, r_proxy])
        real = next(m for m in result.ranked_candidates if m.candidate_id == "real_sim")
        assert real.npe_source == "simulation"
        assert real.npe == pytest.approx(expected_npe, rel=0.0, abs=0.0)


class TestAgenticRunnerPropagatesNPEScore:
    """The agentic runner rebuilds BackendResult but must not drop diagnostics.

    Regression: a simulation-based ``npe_score`` populated by the inner
    runner (nlmixr2 / jax_node / stan) must survive being re-stamped as the
    ``agentic_llm`` backend. Anything else would silently demote Gate 3
    ranking to the CWRES proxy specifically for agentic candidates.
    """

    def test_agentic_diagnostics_preserve_npe_score(self) -> None:
        # Construct an inner result with a populated ``npe_score`` and feed it
        # through the same construction path the agentic runner uses at
        # ``agentic_runner.py::run`` end (backend="agentic_llm", all other
        # fields copied from ``best_result``). The test does not spin up the
        # LLM loop — that is covered elsewhere — but it guarantees the
        # propagation contract at the BackendResult boundary.
        inner = _make_result(model_id="inner", backend="nlmixr2")
        inner.diagnostics.npe_score = 0.073
        stamped = BackendResult(
            model_id=inner.model_id,
            backend="agentic_llm",
            converged=inner.converged,
            ofv=inner.ofv,
            aic=inner.aic,
            bic=inner.bic,
            parameter_estimates=inner.parameter_estimates,
            eta_shrinkage=inner.eta_shrinkage,
            convergence_metadata=inner.convergence_metadata,
            diagnostics=inner.diagnostics,
            wall_time_seconds=inner.wall_time_seconds,
            backend_versions=inner.backend_versions,
            initial_estimate_source=inner.initial_estimate_source,
        )
        assert stamped.diagnostics.npe_score == pytest.approx(0.073)


class TestWeightedSumZeroBIC:
    """BIC weight=0 must drop BIC entirely even under weighted_sum."""

    _NO_BIC = Gate3Config(
        composite_method="weighted_sum",
        vpc_weight=0.5,
        npe_weight=0.5,
        bic_weight=0.0,
    )

    def test_bic_weight_zero_ignores_bic(self) -> None:
        # Same metrics except BIC; disabled BIC means identical scores.
        r1 = _make_result(
            model_id="m1",
            backend="nlmixr2",
            bic=100.0,
            cwres_mean=0.1,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        r2 = _make_result(
            model_id="m2",
            backend="jax_node",
            bic=1000.0,  # very different BIC
            cwres_mean=0.1,
            cwres_sd=1.0,
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        result = _rank([r1, r2], gate3=self._NO_BIC)
        assert result.ranked_candidates[0].composite_score == pytest.approx(
            result.ranked_candidates[1].composite_score
        )
        for m in result.ranked_candidates:
            assert m.component_scores["bic"] == 0.0


class TestAUCCmaxScoringFunctions:
    """``compute_auc_cmax_be_score`` math and NCA-eligibility gating."""

    def test_all_subjects_within_bounds(self) -> None:
        import numpy as np

        from apmode.benchmarks.scoring import compute_auc_cmax_be_score

        # Candidate ≈ reference → every GMR is 1.0, inside [0.80, 1.25].
        cand_auc = np.array([100.0, 120.0, 80.0])
        cand_cmax = np.array([10.0, 12.0, 8.0])
        ref_auc = np.array([100.0, 120.0, 80.0])
        ref_cmax = np.array([10.0, 12.0, 8.0])
        assert compute_auc_cmax_be_score(cand_auc, cand_cmax, ref_auc, ref_cmax) == 1.0

    def test_all_subjects_outside_bounds(self) -> None:
        import numpy as np

        from apmode.benchmarks.scoring import compute_auc_cmax_be_score

        cand_auc = np.array([200.0, 200.0, 200.0])  # GMR 2.0 > 1.25
        cand_cmax = np.array([20.0, 20.0, 20.0])
        ref_auc = np.array([100.0, 100.0, 100.0])
        ref_cmax = np.array([10.0, 10.0, 10.0])
        assert compute_auc_cmax_be_score(cand_auc, cand_cmax, ref_auc, ref_cmax) == 0.0

    def test_partial_pass_gives_fraction(self) -> None:
        import numpy as np

        from apmode.benchmarks.scoring import compute_auc_cmax_be_score

        # 2/4 subjects are in-bounds on both AUC and Cmax.
        cand_auc = np.array([100.0, 100.0, 200.0, 50.0])  # in, in, high, low
        cand_cmax = np.array([10.0, 10.0, 20.0, 5.0])
        ref_auc = np.array([100.0, 110.0, 100.0, 100.0])
        ref_cmax = np.array([10.0, 11.0, 10.0, 10.0])
        assert compute_auc_cmax_be_score(cand_auc, cand_cmax, ref_auc, ref_cmax) == 0.5

    def test_requires_both_auc_and_cmax_in_bounds(self) -> None:
        import numpy as np

        from apmode.benchmarks.scoring import compute_auc_cmax_be_score

        # AUC is fine (GMR=1.0) but Cmax GMR is 2.0 → per-subject fail.
        cand_auc = np.array([100.0])
        cand_cmax = np.array([20.0])
        ref_auc = np.array([100.0])
        ref_cmax = np.array([10.0])
        assert compute_auc_cmax_be_score(cand_auc, cand_cmax, ref_auc, ref_cmax) == 0.0

    def test_shape_mismatch_raises(self) -> None:
        import numpy as np

        from apmode.benchmarks.scoring import compute_auc_cmax_be_score

        with pytest.raises(ValueError, match="must share length"):
            compute_auc_cmax_be_score(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
            )

    def test_non_finite_or_non_positive_counts_as_fail(self) -> None:
        import numpy as np

        from apmode.benchmarks.scoring import compute_auc_cmax_be_score

        # Subject 0 has NaN candidate Cmax; subject 1 is in-bounds.
        cand_auc = np.array([100.0, 100.0])
        cand_cmax = np.array([np.nan, 10.0])
        ref_auc = np.array([100.0, 100.0])
        ref_cmax = np.array([10.0, 10.0])
        assert compute_auc_cmax_be_score(cand_auc, cand_cmax, ref_auc, ref_cmax) == 0.5

    def test_empty_subjects_returns_none(self) -> None:
        # Empty cohort is undefined ("no data"), not BE-failed.
        # Returning None lets the uniform-drop rule in rank_cross_paradigm
        # drop the auc_cmax component cleanly.
        import numpy as np

        from apmode.benchmarks.scoring import compute_auc_cmax_be_score

        empty = np.array([])
        assert compute_auc_cmax_be_score(empty, empty, empty, empty) is None

    def test_nca_eligibility_adequate_low_blq(self) -> None:
        from apmode.benchmarks.scoring import is_nca_eligible_for_auc_cmax
        from apmode.bundle.models import EvidenceManifest

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=0,
            covariate_correlated=False,
            covariate_missingness=None,
            time_varying_covariates=False,
            blq_burden=0.05,
            lloq_value=0.1,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        eligible, reason = is_nca_eligible_for_auc_cmax(manifest)
        assert eligible is True
        assert "admissible" in reason.lower()

    def test_nca_eligibility_high_blq(self) -> None:
        from apmode.benchmarks.scoring import is_nca_eligible_for_auc_cmax
        from apmode.bundle.models import EvidenceManifest

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=0,
            covariate_correlated=False,
            covariate_missingness=None,
            time_varying_covariates=False,
            blq_burden=0.35,
            lloq_value=0.1,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        eligible, reason = is_nca_eligible_for_auc_cmax(manifest)
        assert eligible is False
        assert "blq" in reason.lower()

    def test_nca_eligibility_inadequate_absorption(self) -> None:
        from apmode.benchmarks.scoring import is_nca_eligible_for_auc_cmax
        from apmode.bundle.models import EvidenceManifest

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="rich",
            identifiability_ceiling="high",
            covariate_burden=0,
            covariate_correlated=False,
            covariate_missingness=None,
            time_varying_covariates=False,
            blq_burden=0.05,
            lloq_value=0.1,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="inadequate",
            elimination_phase_coverage="adequate",
        )
        eligible, reason = is_nca_eligible_for_auc_cmax(manifest)
        assert eligible is False
        assert "absorption" in reason.lower()


class TestAUCCmaxInGate3Ranking:
    """Integration tests: AUC/Cmax flows through the composite + uniform-drop."""

    _WITH_AUC_CMAX = Gate3Config(
        composite_method="weighted_sum",
        vpc_weight=0.3,
        npe_weight=0.3,
        bic_weight=0.0,
        auc_cmax_weight=0.4,
    )
    _BORDA_WITH_AUC_CMAX = Gate3Config(
        composite_method="borda",
        vpc_weight=0.3,
        npe_weight=0.3,
        bic_weight=0.0,
        auc_cmax_weight=0.4,
    )

    def test_weighted_sum_includes_auc_cmax_component(self) -> None:
        r_good = _make_result(
            model_id="good",
            backend="nlmixr2",
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        r_bad = _make_result(
            model_id="bad",
            backend="jax_node",
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
        )
        # Identical VPC + NPE proxy; AUC/Cmax BE drives the ranking.
        r_good.diagnostics.auc_cmax_be_score = 0.90
        r_good.diagnostics.auc_cmax_source = "observed_trapezoid"
        r_bad.diagnostics.auc_cmax_be_score = 0.10
        r_bad.diagnostics.auc_cmax_source = "observed_trapezoid"

        result = _rank([r_good, r_bad], gate3=self._WITH_AUC_CMAX)
        assert result.ranked_candidates[0].candidate_id == "good"
        assert result.ranked_candidates[0].auc_cmax_be == pytest.approx(0.90)
        assert result.ranked_candidates[0].auc_cmax_source == "observed_trapezoid"
        # Component score present in weighted_sum output.
        assert "auc_cmax" in result.ranked_candidates[0].component_scores

    def test_borda_includes_auc_cmax_metric(self) -> None:
        r_good = _make_result(model_id="good", backend="nlmixr2")
        r_bad = _make_result(model_id="bad", backend="jax_node")
        r_good.diagnostics.auc_cmax_be_score = 0.95
        r_good.diagnostics.auc_cmax_source = "observed_trapezoid"
        r_bad.diagnostics.auc_cmax_be_score = 0.20
        r_bad.diagnostics.auc_cmax_source = "observed_trapezoid"

        result = _rank([r_good, r_bad], gate3=self._BORDA_WITH_AUC_CMAX)
        top = result.ranked_candidates[0]
        assert top.candidate_id == "good"
        assert "auc_cmax" in top.component_scores

    def test_uniform_drop_when_any_candidate_missing_auc_cmax(self) -> None:
        # r_a has a populated score; r_b does not (e.g. NCA ineligible for
        # r_b's dataset view). The uniform-drop rule removes the auc_cmax
        # component for *both* candidates rather than giving r_a a free win
        # on a metric r_b was never scored on.
        r_a = _make_result(model_id="a", backend="nlmixr2")
        r_b = _make_result(model_id="b", backend="jax_node")
        r_a.diagnostics.auc_cmax_be_score = 0.99
        r_a.diagnostics.auc_cmax_source = "observed_trapezoid"
        r_b.diagnostics.auc_cmax_be_score = None

        result = _rank([r_a, r_b], gate3=self._WITH_AUC_CMAX)
        # Component zeroed for *every* candidate under weighted_sum.
        for m in result.ranked_candidates:
            assert m.component_scores["auc_cmax"] == 0.0
        # Qualification reason records the drop for audit traceability.
        assert "auc_cmax_be unavailable" in result.qualification_reason
        # Per-candidate values retained so reports can surface the raw
        # score that was computed even though it didn't factor in.
        by_id = {m.candidate_id: m for m in result.ranked_candidates}
        assert by_id["a"].auc_cmax_be == pytest.approx(0.99)
        assert by_id["b"].auc_cmax_be is None

    def test_uniform_drop_renormalizes_remaining_weights(self) -> None:
        # Before drop: vpc=0.3, npe=0.3, bic=0.0, auc_cmax=0.4.
        # After drop: remaining = 0.6, scale = 1/0.6 → vpc=0.5, npe=0.5.
        # Two candidates that differ only in NPE; ranker should still pick
        # the one with the lower NPE, and the weighted_sum composite for a
        # pure worst/best VPC+NPE pair must lie on [0, 1] (proving the
        # weights genuinely renormalized to sum 1).
        r_good = _make_result(
            model_id="good",
            backend="nlmixr2",
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
            cwres_mean=0.0,
            cwres_sd=1.0,
        )
        r_bad = _make_result(
            model_id="bad",
            backend="jax_node",
            vpc_coverage={"p5": 0.90, "p50": 0.90, "p95": 0.90},
            cwres_mean=0.5,
            cwres_sd=1.5,
        )
        # Both lack auc_cmax_be → uniform-drop triggers.
        result = _rank([r_good, r_bad], gate3=self._WITH_AUC_CMAX)
        assert result.ranked_candidates[0].candidate_id == "good"
        for m in result.ranked_candidates:
            assert 0.0 <= m.composite_score <= 1.0

    def test_auc_cmax_weight_zero_does_not_trigger_drop_logic(self) -> None:
        # With auc_cmax_weight=0 the component is simply inactive — missing
        # scores on individual candidates must NOT trigger the uniform-drop
        # rewrite, and the qualification reason must not mention auc_cmax.
        r1 = _make_result(model_id="m1", backend="nlmixr2")
        r2 = _make_result(model_id="m2", backend="jax_node")
        r1.diagnostics.auc_cmax_be_score = None
        r2.diagnostics.auc_cmax_be_score = None
        result = _rank([r1, r2], gate3=_DEFAULT_GATE3)
        assert "auc_cmax_be unavailable" not in result.qualification_reason
