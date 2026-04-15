# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the policy-driven missing-data pipeline.

Covers:
  - ``resolve_directive``: every branch of the policy decision table
  - ``aggregate_stability``: pooling, rank_stability, sign consistency
  - ``SearchSpace.apply_directive``: BLQ method mapping
  - ``_check_imputation_stability`` gate check (convergence + rank thresholds)
"""

from __future__ import annotations

import pytest

from apmode.bundle.models import (
    CovariateSpec,
    EvidenceManifest,
    ImputationStabilityEntry,
    MissingDataDirective,
)
from apmode.data.missing_data import (
    OMEGA_POOLING_CAVEATS,
    build_stability_manifest,
    resolve_directive,
)
from apmode.governance.policy import MissingDataPolicy
from apmode.search.candidates import SearchSpace
from apmode.search.stability import PerImputationFit, aggregate_stability


def _manifest(
    *,
    cov_spec: CovariateSpec | None = None,
    time_varying: bool = False,
    blq_burden: float = 0.0,
    lloq: float | None = None,
    correlated: bool = False,
) -> EvidenceManifest:
    return EvidenceManifest(
        route_certainty="confirmed",
        absorption_complexity="simple",
        nonlinear_clearance_signature=False,
        richness_category="moderate",
        identifiability_ceiling="medium",
        covariate_burden=1,
        covariate_correlated=correlated,
        covariate_missingness=cov_spec,
        time_varying_covariates=time_varying,
        blq_burden=blq_burden,
        lloq_value=lloq,
        protocol_heterogeneity="single-study",
        absorption_phase_coverage="adequate",
        elimination_phase_coverage="adequate",
    )


class TestResolveDirective:
    def test_no_missingness_uses_exclude(self) -> None:
        d = resolve_directive(MissingDataPolicy(), _manifest())
        assert d.covariate_method == "exclude"
        assert d.m_imputations is None
        assert d.blq_method == "M7+"  # default policy, 0% burden

    def test_time_varying_prefers_frem(self) -> None:
        cov = CovariateSpec(pattern="MAR", fraction_incomplete=0.05, strategy="MI-PMM")
        d = resolve_directive(MissingDataPolicy(), _manifest(cov_spec=cov, time_varying=True))
        assert d.covariate_method == "FREM"
        assert d.m_imputations is None
        assert any("time-varying" in r.lower() for r in d.rationale)

    def test_high_missingness_prefers_frem(self) -> None:
        cov = CovariateSpec(pattern="MAR", fraction_incomplete=0.45, strategy="MI-PMM")
        d = resolve_directive(MissingDataPolicy(), _manifest(cov_spec=cov))
        assert d.covariate_method == "FREM"

    def test_medium_missingness_prefers_frem_over_large_m(self) -> None:
        # Between mi_pmm_max_missingness (0.30) and frem_preferred_above (0.30 default);
        # set policy with a gap to hit that branch.
        policy = MissingDataPolicy(
            mi_pmm_max_missingness=0.10,
            frem_preferred_above=0.40,
        )
        cov = CovariateSpec(pattern="MAR", fraction_incomplete=0.20, strategy="MI-PMM")
        d = resolve_directive(policy, _manifest(cov_spec=cov))
        assert d.covariate_method == "FREM"

    def test_low_missingness_uses_mi_pmm(self) -> None:
        cov = CovariateSpec(pattern="MAR", fraction_incomplete=0.10, strategy="MI-PMM")
        policy = MissingDataPolicy(m_imputations=7)
        d = resolve_directive(policy, _manifest(cov_spec=cov))
        assert d.covariate_method == "MI-PMM"
        assert d.m_imputations == 7

    def test_correlated_covariates_use_missforest(self) -> None:
        cov = CovariateSpec(pattern="MAR", fraction_incomplete=0.10, strategy="MI-PMM")
        d = resolve_directive(MissingDataPolicy(), _manifest(cov_spec=cov, correlated=True))
        assert d.covariate_method == "MI-missRanger"

    def test_blq_threshold_selects_m3(self) -> None:
        d = resolve_directive(MissingDataPolicy(), _manifest(blq_burden=0.25))
        assert d.blq_method == "M3"

    def test_blq_force_m3_override(self) -> None:
        d = resolve_directive(MissingDataPolicy(blq_force_m3=True), _manifest())
        assert d.blq_method == "M3"

    def test_blq_below_threshold_uses_m7plus(self) -> None:
        d = resolve_directive(MissingDataPolicy(), _manifest(blq_burden=0.05))
        assert d.blq_method == "M7+"


class TestStabilityManifest:
    def test_mi_injects_omega_caveats(self) -> None:
        cov = CovariateSpec(pattern="MAR", fraction_incomplete=0.10, strategy="MI-PMM")
        directive = resolve_directive(MissingDataPolicy(), _manifest(cov_spec=cov))
        manifest = build_stability_manifest(directive, entries=[])
        assert manifest.omega_pooling_caveats == list(OMEGA_POOLING_CAVEATS)

    def test_frem_omits_omega_caveats(self) -> None:
        cov = CovariateSpec(pattern="MAR", fraction_incomplete=0.45, strategy="FREM")
        directive = resolve_directive(MissingDataPolicy(), _manifest(cov_spec=cov))
        manifest = build_stability_manifest(directive, entries=[])
        assert manifest.omega_pooling_caveats == []


class TestAggregateStability:
    def test_convergence_rate_uses_m_not_fits(self) -> None:
        fits = [
            PerImputationFit(imputation_idx=0, candidate_id="c1", converged=True, bic=100.0),
            PerImputationFit(imputation_idx=1, candidate_id="c1", converged=True, bic=101.0),
        ]
        # m=5 means 3 imputations are missing — candidate effectively converged on 2/5.
        entries = aggregate_stability(fits, m=5)
        e = next(e for e in entries if e.candidate_id == "c1")
        assert e.convergence_rate == pytest.approx(0.4)

    def test_rank_stability_counts_top_k_appearances(self) -> None:
        # Two candidates across 3 imputations; c1 always wins.
        fits: list[PerImputationFit] = []
        for i in range(3):
            fits.append(
                PerImputationFit(imputation_idx=i, candidate_id="c1", converged=True, bic=100.0)
            )
            fits.append(
                PerImputationFit(imputation_idx=i, candidate_id="c2", converged=True, bic=200.0)
            )
        entries = aggregate_stability(fits, m=3, top_k=1)
        by_id = {e.candidate_id: e for e in entries}
        assert by_id["c1"].rank_stability == pytest.approx(1.0)
        assert by_id["c2"].rank_stability == pytest.approx(0.0)

    def test_pooled_ofv_is_mean_of_converged(self) -> None:
        fits = [
            PerImputationFit(imputation_idx=0, candidate_id="c1", converged=True, ofv=100.0),
            PerImputationFit(imputation_idx=1, candidate_id="c1", converged=True, ofv=200.0),
            PerImputationFit(imputation_idx=2, candidate_id="c1", converged=False, ofv=None),
        ]
        entries = aggregate_stability(fits, m=3)
        assert entries[0].pooled_ofv == pytest.approx(150.0)

    def test_sign_consistency(self) -> None:
        fits = [
            PerImputationFit(
                imputation_idx=0,
                candidate_id="c1",
                converged=True,
                covariate_effect_signs={"WT_on_CL": 1.0},
            ),
            PerImputationFit(
                imputation_idx=1,
                candidate_id="c1",
                converged=True,
                covariate_effect_signs={"WT_on_CL": 1.0},
            ),
            PerImputationFit(
                imputation_idx=2,
                candidate_id="c1",
                converged=True,
                covariate_effect_signs={"WT_on_CL": -1.0},
            ),
        ]
        entries = aggregate_stability(fits, m=3)
        assert entries[0].covariate_sign_consistency["WT_on_CL"] == pytest.approx(2 / 3)


class TestSearchSpaceApplyDirective:
    def test_m3_forces_blq_method(self) -> None:
        manifest = _manifest(blq_burden=0.25, lloq=0.5)
        space = SearchSpace.from_manifest(manifest)
        directive = MissingDataDirective(covariate_method="exclude", blq_method="M3")
        updated = space.apply_directive(directive, manifest)
        assert updated.force_blq_method == "m3"
        assert updated.lloq_value == 0.5
        assert updated.blq_strategy == "M3"

    def test_m7plus_clears_force_blq(self) -> None:
        manifest = _manifest(blq_burden=0.25, lloq=0.5)
        space = SearchSpace.from_manifest(manifest)
        # from_manifest set force_blq_method="m3" because burden>0.20
        assert space.force_blq_method == "m3"
        directive = MissingDataDirective(covariate_method="exclude", blq_method="M7+")
        updated = space.apply_directive(directive, manifest)
        assert updated.force_blq_method is None
        assert updated.blq_strategy == "M7+"
        # Original space is not mutated.
        assert space.force_blq_method == "m3"


class TestGate1ImputationStability:
    """Exercise the imputation_stability check via evaluate_gate1."""

    def _result_and_policy(self) -> tuple[object, object]:
        from tests.unit.test_gates import _load_policy, _make_backend_result

        return _make_backend_result(), _load_policy("discovery")

    def test_no_directive_passes(self) -> None:
        from apmode.governance.gates import evaluate_gate1

        result, policy = self._result_and_policy()
        g1 = evaluate_gate1(result, policy)  # type: ignore[arg-type]
        check = next(c for c in g1.checks if c.check_id == "imputation_stability")
        assert check.passed is True
        assert check.observed == "not_applicable"

    def test_low_convergence_fails(self) -> None:
        from apmode.governance.gates import evaluate_gate1

        result, policy = self._result_and_policy()
        directive = MissingDataDirective(
            covariate_method="MI-PMM",
            m_imputations=5,
            blq_method="M7+",
            imputation_stability_penalty=0.0,
        )
        stability = ImputationStabilityEntry(
            candidate_id=result.model_id,  # type: ignore[attr-defined]
            convergence_rate=0.3,
            rank_stability=1.0,
        )
        g1 = evaluate_gate1(
            result,
            policy,
            stability=stability,
            directive=directive,  # type: ignore[arg-type]
        )
        check = next(c for c in g1.checks if c.check_id == "imputation_stability")
        assert check.passed is False

    def test_penalty_sets_rank_threshold(self) -> None:
        from apmode.governance.gates import evaluate_gate1

        result, policy = self._result_and_policy()
        directive = MissingDataDirective(
            covariate_method="MI-PMM",
            m_imputations=5,
            blq_method="M7+",
            imputation_stability_penalty=0.5,
        )
        # rank_stability=0.4 < (1 - 0.5)=0.5 → fail
        unstable = ImputationStabilityEntry(
            candidate_id=result.model_id,  # type: ignore[attr-defined]
            convergence_rate=1.0,
            rank_stability=0.4,
        )
        g1 = evaluate_gate1(
            result,
            policy,
            stability=unstable,
            directive=directive,  # type: ignore[arg-type]
        )
        check = next(c for c in g1.checks if c.check_id == "imputation_stability")
        assert check.passed is False

        # rank_stability=0.6 > 0.5 → pass
        stable = ImputationStabilityEntry(
            candidate_id=result.model_id,  # type: ignore[attr-defined]
            convergence_rate=1.0,
            rank_stability=0.6,
        )
        g1 = evaluate_gate1(
            result,
            policy,
            stability=stable,
            directive=directive,  # type: ignore[arg-type]
        )
        check = next(c for c in g1.checks if c.check_id == "imputation_stability")
        assert check.passed is True


# --- Rubin pooling tests -------------------------------------------------


class TestRubinPool:
    """Rubin (1987) pooling of per-imputation scalar parameter estimates."""

    def test_single_imputation_returns_estimate_unchanged(self) -> None:
        from apmode.search.stability import rubin_pool

        pooled, within, between, total, dof = rubin_pool([5.0], [0.5])
        assert pooled == pytest.approx(5.0)
        assert within == pytest.approx(0.25)  # SE²
        assert between == pytest.approx(0.0)
        assert total == pytest.approx(0.25)
        assert dof == float("inf")

    def test_two_imputations_decomposes_variance(self) -> None:
        from apmode.search.stability import rubin_pool

        pooled, within, between, total, _ = rubin_pool([5.0, 5.4], [0.3, 0.3])
        assert pooled == pytest.approx(5.2)
        # Within: mean of SE² = 0.09
        assert within == pytest.approx(0.09)
        # Between: sample var([5.0, 5.4]) = 0.08 (n-1 denominator)
        assert between == pytest.approx(0.08)
        # Total: Ū + (1 + 1/2)*B = 0.09 + 1.5*0.08 = 0.21
        assert total == pytest.approx(0.21)

    def test_none_ses_zero_within_variance(self) -> None:
        from apmode.search.stability import rubin_pool

        pooled, within, between, total, _ = rubin_pool([5.0, 5.4, 4.8], [None, None, None])
        assert pooled == pytest.approx(5.0666, rel=1e-3)
        assert within == pytest.approx(0.0)
        # Only between-imputation variance contributes to total.
        assert total == pytest.approx((1 + 1 / 3) * between)

    def test_misaligned_inputs_raise(self) -> None:
        from apmode.search.stability import rubin_pool

        with pytest.raises(ValueError, match="must align"):
            rubin_pool([1.0, 2.0], [0.1])

    def test_aggregate_populates_pooled_parameters(self) -> None:
        """aggregate_stability must surface Rubin-pooled params when fits carry (est, se)."""
        fits = [
            PerImputationFit(
                imputation_idx=0,
                candidate_id="c1",
                converged=True,
                bic=100.0,
                parameter_estimates={"CL": (5.0, 0.3), "V": (50.0, 2.0)},
            ),
            PerImputationFit(
                imputation_idx=1,
                candidate_id="c1",
                converged=True,
                bic=101.0,
                parameter_estimates={"CL": (5.2, 0.3), "V": (51.0, 2.0)},
            ),
            PerImputationFit(
                imputation_idx=2,
                candidate_id="c1",
                converged=True,
                bic=100.5,
                parameter_estimates={"CL": (4.9, 0.3), "V": (49.0, 2.0)},
            ),
        ]
        from apmode.search.stability import aggregate_stability

        entries = aggregate_stability(fits, m=3)
        assert entries[0].pooled_parameters
        cl = entries[0].pooled_parameters["CL"]
        assert cl["pooled_estimate"] == pytest.approx((5.0 + 5.2 + 4.9) / 3)
        assert cl["within_var"] == pytest.approx(0.09)
        assert cl["total_var"] > cl["within_var"]  # between adds variance
        assert cl["dof"] > 0
