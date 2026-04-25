# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the shared posterior-predictive diagnostics helper.

Covers:

- Shape / invariant validation on :class:`SubjectSimulation` inputs.
- Deterministic math for AUC/Cmax/NPE on a hand-crafted 1-cmt case.
- Per-subject NCA eligibility gating (floor + fraction → None).
- VPC coverage dict shape and schema compatibility with
  :class:`VPCSummary` validator.
- Atomicity contract — all three diagnostics derive from the same
  simulation matrix and reflect consistent medians.
- Integration with the Gate 3 ranker: a populated
  :class:`PredictiveSummaryBundle` lets
  :func:`_resolve_npe` / :func:`_resolve_auc_cmax` pick up the
  simulation-based values instead of the CWRES / None fallback.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from apmode.backends.predictive_summary import (
    PredictiveSummaryBundle,
    SubjectSimulation,
    build_predictive_diagnostics,
)
from apmode.benchmarks.scoring import (
    compute_auc_cmax_be_score,
    is_nca_eligible_per_subject,
)
from apmode.bundle.models import (
    BackendResult as BR,
)
from apmode.bundle.models import (
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    NCASubjectDiagnostic,
    ParameterEstimate,
)
from apmode.governance.policy import Gate3Config
from apmode.governance.ranking import rank_cross_paradigm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eligible_diagnostic(subject_id: str) -> NCASubjectDiagnostic:
    """Admissible NCA QC record for a subject — ``excluded=False``."""
    return NCASubjectDiagnostic(
        subject_id=subject_id,
        tmax=1.0,
        cmax=10.0,
        cl=5.0,
        v=50.0,
        ka=1.0,
        kel=0.1,
        auc_last=100.0,
        auc_inf=110.0,
        auc_extrap_fraction=0.09,
        lambda_z_adj_r2=0.98,
        lambda_z_n_points=5,
        span_ratio=1.2,
        excluded=False,
    )


def _excluded_diagnostic(subject_id: str, reason: str = "auc_extrap>20%") -> NCASubjectDiagnostic:
    """Ineligible NCA QC record — ``excluded=True`` with a reason."""
    return NCASubjectDiagnostic(
        subject_id=subject_id,
        excluded=True,
        excluded_reason=reason,
    )


def _constant_subject(
    subject_id: str,
    *,
    times: np.ndarray,
    observed: np.ndarray,
    simulated_constant: float,
    n_sims: int = 100,
    eligible: bool = True,
) -> SubjectSimulation:
    """Subject whose sim matrix is a constant (deterministic math baseline)."""
    sims = np.full((n_sims, times.shape[0]), simulated_constant, dtype=float)
    diag = _eligible_diagnostic(subject_id) if eligible else _excluded_diagnostic(subject_id)
    return SubjectSimulation(
        subject_id=subject_id,
        t_observed=times,
        observed_dv=observed,
        sims_at_observed=sims,
        nca_diagnostic=diag,
    )


def _policy_with_floors(
    *,
    min_eligible: int = 8,
    min_eligible_fraction: float = 0.5,
    vpc_n_bins: int = 5,
) -> Gate3Config:
    """Gate3Config with predictable eligibility floors + small bin count."""
    return Gate3Config(
        composite_method="weighted_sum",
        vpc_weight=0.5,
        npe_weight=0.5,
        bic_weight=0.0,
        auc_cmax_weight=0.0,
        auc_cmax_nca_min_eligible=min_eligible,
        auc_cmax_nca_min_eligible_fraction=min_eligible_fraction,
        vpc_n_bins=vpc_n_bins,
    )


# ---------------------------------------------------------------------------
# Eligibility helper unit tests
# ---------------------------------------------------------------------------


class TestIsNCAEligiblePerSubject:
    def test_admissible_subject(self) -> None:
        diag = _eligible_diagnostic("s1")
        eligible, reason = is_nca_eligible_per_subject(diag)
        assert eligible is True
        assert "admissible" in reason.lower()

    def test_excluded_with_reason(self) -> None:
        diag = _excluded_diagnostic("s1", reason="span_ratio<1")
        eligible, reason = is_nca_eligible_per_subject(diag)
        assert eligible is False
        assert "span_ratio<1" in reason

    def test_excluded_without_reason_falls_back(self) -> None:
        diag = NCASubjectDiagnostic(subject_id="s1", excluded=True)
        eligible, reason = is_nca_eligible_per_subject(diag)
        assert eligible is False
        assert "nca qc" in reason.lower()


# ---------------------------------------------------------------------------
# compute_auc_cmax_be_score — eligibility mask + floors
# ---------------------------------------------------------------------------


class TestComputeAUCCmaxBEWithEligibilityMask:
    def test_default_no_mask_preserves_legacy_return(self) -> None:
        """Backward compat: no mask, no floors → float, mean over all subjects."""
        cand_auc = np.array([100.0, 100.0, 100.0])
        cand_cmax = np.array([10.0, 10.0, 10.0])
        ref_auc = np.array([100.0, 100.0, 100.0])
        ref_cmax = np.array([10.0, 10.0, 10.0])
        score = compute_auc_cmax_be_score(cand_auc, cand_cmax, ref_auc, ref_cmax)
        assert score == 1.0

    def test_mask_drops_ineligible_from_denominator(self) -> None:
        """Mask-drop: n_eligible is denominator, not n_total."""
        # 3 subjects: S0 BE-pass eligible, S1 BE-fail eligible, S2 BE-pass ineligible.
        cand_auc = np.array([100.0, 200.0, 100.0])  # S1 GMR=2.0 fails
        cand_cmax = np.array([10.0, 20.0, 10.0])
        ref_auc = np.array([100.0, 100.0, 100.0])
        ref_cmax = np.array([10.0, 10.0, 10.0])
        mask = np.array([True, True, False])  # S2 ineligible
        score = compute_auc_cmax_be_score(
            cand_auc, cand_cmax, ref_auc, ref_cmax, eligible_mask=mask
        )
        # Wins: S0 (in-bounds). Denominator: 2 eligible. Expect 0.5.
        assert score == 0.5

    def test_below_min_eligible_floor_returns_none(self) -> None:
        cand_auc = np.array([100.0, 100.0, 100.0])
        cand_cmax = np.array([10.0, 10.0, 10.0])
        ref_auc = np.array([100.0, 100.0, 100.0])
        ref_cmax = np.array([10.0, 10.0, 10.0])
        mask = np.array([True, False, False])  # 1 eligible
        assert (
            compute_auc_cmax_be_score(
                cand_auc,
                cand_cmax,
                ref_auc,
                ref_cmax,
                eligible_mask=mask,
                min_eligible=8,
            )
            is None
        )

    def test_below_min_eligible_fraction_returns_none(self) -> None:
        # 4 eligible out of 10 → 0.4 < 0.5 floor → None
        cand = np.full(10, 100.0)
        ref = np.full(10, 100.0)
        mask = np.array(
            [True, True, True, True, False, False, False, False, False, False],
            dtype=bool,
        )
        assert (
            compute_auc_cmax_be_score(
                cand,
                cand,
                ref,
                ref,
                eligible_mask=mask,
                min_eligible=1,
                min_eligible_fraction=0.5,
            )
            is None
        )

    def test_mask_length_mismatch_raises(self) -> None:
        cand = np.array([100.0, 100.0])
        ref = np.array([100.0, 100.0])
        with pytest.raises(ValueError, match="eligible_mask length"):
            compute_auc_cmax_be_score(cand, cand, ref, ref, eligible_mask=np.array([True]))

    def test_empty_inputs_with_floor_returns_none(self) -> None:
        empty = np.array([])
        assert compute_auc_cmax_be_score(empty, empty, empty, empty, min_eligible=1) is None

    def test_empty_inputs_returns_none(self) -> None:
        # Empty cohort is undefined, not BE-failed — returning None lets
        # the uniform-drop rule remove the component.
        empty = np.array([])
        assert compute_auc_cmax_be_score(empty, empty, empty, empty) is None


# ---------------------------------------------------------------------------
# SubjectSimulation shape validation inside build_predictive_diagnostics
# ---------------------------------------------------------------------------


class TestBuildShapeValidation:
    def _baseline(self) -> list[SubjectSimulation]:
        times = np.array([0.5, 1.0, 2.0, 4.0])
        obs = np.array([5.0, 8.0, 6.0, 3.0])
        return [
            _constant_subject(f"s{i}", times=times, observed=obs, simulated_constant=6.0)
            for i in range(10)
        ]

    def test_empty_subject_list_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_predictive_diagnostics([], policy=_policy_with_floors())

    def test_sims_not_2d_raises(self) -> None:
        times = np.array([0.5, 1.0, 2.0])
        obs = np.array([5.0, 8.0, 6.0])
        bad = SubjectSimulation(
            subject_id="s0",
            t_observed=times,
            observed_dv=obs,
            sims_at_observed=np.zeros(3),  # 1D
        )
        with pytest.raises(ValueError, match="must be 2D"):
            build_predictive_diagnostics([bad], policy=_policy_with_floors())

    def test_sims_shape_mismatch_raises(self) -> None:
        times = np.array([0.5, 1.0, 2.0])
        obs = np.array([5.0, 8.0, 6.0])
        bad = SubjectSimulation(
            subject_id="s0",
            t_observed=times,
            observed_dv=obs,
            sims_at_observed=np.zeros((100, 4)),  # 4 != 3
        )
        with pytest.raises(ValueError, match="inconsistent with n_obs"):
            build_predictive_diagnostics([bad], policy=_policy_with_floors())

    def test_observed_dv_shape_mismatch_raises(self) -> None:
        times = np.array([0.5, 1.0, 2.0])
        obs = np.array([5.0, 8.0])  # length 2 not 3
        bad = SubjectSimulation(
            subject_id="s0",
            t_observed=times,
            observed_dv=obs,
            sims_at_observed=np.zeros((100, 3)),
        )
        with pytest.raises(ValueError, match="observed_dv shape"):
            build_predictive_diagnostics([bad], policy=_policy_with_floors())

    def test_n_sims_mismatch_across_subjects_raises(self) -> None:
        times = np.array([0.5, 1.0, 2.0])
        obs = np.array([5.0, 8.0, 6.0])
        s0 = SubjectSimulation(
            subject_id="s0",
            t_observed=times,
            observed_dv=obs,
            sims_at_observed=np.zeros((50, 3)),
        )
        s1 = SubjectSimulation(
            subject_id="s1",
            t_observed=times,
            observed_dv=obs,
            sims_at_observed=np.zeros((100, 3)),
        )
        with pytest.raises(ValueError, match="n_sims="):
            build_predictive_diagnostics([s0, s1], policy=_policy_with_floors())


# ---------------------------------------------------------------------------
# spec -> compute_npe error_model wire (mavoglurant ng/mL inflation fix)
# ---------------------------------------------------------------------------


def _spec_with_obs(observation: object) -> object:
    """Build a minimal one-cmt oral DSLSpec carrying the given observation."""
    from apmode.dsl.ast_models import (
        IIV,
        DSLSpec,
        FirstOrder,
        LinearElim,
        OneCmt,
    )

    return DSLSpec(
        model_id="test_npe_spec_0001",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=10.0),
        elimination=LinearElim(CL=1.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=observation,  # type: ignore[arg-type]
    )


class TestNPEObservationErrorModelWire:
    """spec.observation must drive compute_npe's residual scaling.

    Without this wire, the rc8 raw-MedAE path inflates NPE by ~3 OoMs
    on ng/mL-scaled fixtures (mavoglurant: AMT in mg, DV in ng/mL,
    median DV 89, max 1730) vs mg/L fixtures (theo / warfarin) even
    when the underlying model fit quality is comparable. The
    dimensionless rescaling collapses the cross-fixture skew so the
    Phase-1 ``fraction_beats_literature_median`` gate compares
    methodology, not units.
    """

    def _baseline_subjects(self, dv_scale: float) -> list[SubjectSimulation]:
        """6 subjects, 5 obs each, 50% under-prediction (sim = obs / 2)."""
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        rng = np.random.default_rng(42)
        out: list[SubjectSimulation] = []
        for sid in range(6):
            obs = dv_scale * (5 + rng.uniform(0, 5, size=5))
            sims = np.full((50, 5), 0.5) * obs[None, :]  # constant 50% under
            out.append(
                SubjectSimulation(
                    subject_id=f"s{sid}",
                    t_observed=times,
                    observed_dv=obs,
                    sims_at_observed=sims,
                    nca_diagnostic=_eligible_diagnostic(f"s{sid}"),
                )
            )
        return out

    def test_proportional_npe_is_dimensionless(self) -> None:
        from apmode.dsl.ast_models import Proportional

        # Same shape of residuals, two different DV scales (mg/L vs
        # ng/mL = 1000x). With proportional error_model the NPE values
        # must agree to within numerical noise (geometric residual is
        # invariant under units).
        spec = _spec_with_obs(Proportional(sigma_prop=0.2))
        npe_mg = build_predictive_diagnostics(
            self._baseline_subjects(dv_scale=1.0),
            policy=_policy_with_floors(),
            spec=spec,
        ).npe_score
        npe_ng = build_predictive_diagnostics(
            self._baseline_subjects(dv_scale=1000.0),
            policy=_policy_with_floors(),
            spec=spec,
        ).npe_score
        assert npe_mg == pytest.approx(npe_ng, rel=0.05), (
            f"Proportional NPE must be dimensionless: mg-scale={npe_mg}, "
            f"ng-scale={npe_ng}, ratio={npe_ng / npe_mg:.3f}"
        )
        # And the dimensionless value must be ~0.5 (the constant under-
        # prediction fraction we built into the synthetic data).
        assert 0.4 < npe_mg < 0.6

    def test_additive_default_preserves_rc8_scale_dependence(self) -> None:
        # When ``spec`` is omitted (or carries an Additive observation
        # model), behaviour must remain bit-identical to the rc8
        # raw-MedAE path. This pins backwards-compat for any caller that
        # has not yet threaded the spec through.
        npe_mg = build_predictive_diagnostics(
            self._baseline_subjects(dv_scale=1.0),
            policy=_policy_with_floors(),
        ).npe_score
        npe_ng = build_predictive_diagnostics(
            self._baseline_subjects(dv_scale=1000.0),
            policy=_policy_with_floors(),
        ).npe_score
        # Raw-MedAE on ng-scale data is ~1000x the mg-scale value (the
        # SAME bug the proportional wire fixes).
        assert npe_ng / npe_mg == pytest.approx(1000.0, rel=0.05)

    def test_combined_uses_combined_scaling(self) -> None:
        from apmode.dsl.ast_models import Combined

        spec = _spec_with_obs(Combined(sigma_prop=0.1, sigma_add=0.5))
        npe = build_predictive_diagnostics(
            self._baseline_subjects(dv_scale=1.0),
            policy=_policy_with_floors(),
            spec=spec,
        ).npe_score
        # Combined denominator sqrt(obs² + 1²) ≈ obs for obs ≫ 1, so
        # the NPE collapses to ~the proportional value (~0.5) on this
        # mg-scale data; just pin it within a sane range to catch
        # accidental wire-up to additive (which would be ~3.5).
        assert 0.3 < npe < 0.7, f"combined-scale NPE outside expected band: {npe}"

    def test_blq_wraps_underlying_error_model(self) -> None:
        from apmode.dsl.ast_models import BLQM3

        # BLQ_M3 with proportional underlying model must use the
        # proportional residual scaling, not additive — otherwise the
        # M3 likelihood and the predictive NPE scoring diverge.
        spec = _spec_with_obs(BLQM3(loq_value=0.05, error_model="proportional"))
        npe = build_predictive_diagnostics(
            self._baseline_subjects(dv_scale=1.0),
            policy=_policy_with_floors(),
            spec=spec,
        ).npe_score
        assert 0.4 < npe < 0.6, (
            f"BLQ-wrapped proportional must produce dimensionless NPE; got {npe}"
        )


# ---------------------------------------------------------------------------
# Deterministic math
# ---------------------------------------------------------------------------


class TestBuildPredictiveDiagnosticsDeterministic:
    """Hand-computable inputs validate AUC/Cmax/NPE/VPC math."""

    def _policy(self) -> Gate3Config:
        return _policy_with_floors(min_eligible=8, vpc_n_bins=3)

    def _uniform_cohort(
        self, *, sim_value: float, observed_value: float, n_subjects: int = 10
    ) -> list[SubjectSimulation]:
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        obs = np.full(times.shape, observed_value)
        return [
            _constant_subject(
                f"s{i}",
                times=times,
                observed=obs,
                simulated_constant=sim_value,
                n_sims=50,
            )
            for i in range(n_subjects)
        ]

    def test_sim_equals_observed_gives_npe_zero(self) -> None:
        cohort = self._uniform_cohort(sim_value=5.0, observed_value=5.0)
        bundle = build_predictive_diagnostics(cohort, policy=self._policy())
        assert bundle.npe_score == pytest.approx(0.0)

    def test_sim_equals_observed_gives_be_pass_one(self) -> None:
        cohort = self._uniform_cohort(sim_value=5.0, observed_value=5.0)
        bundle = build_predictive_diagnostics(cohort, policy=self._policy())
        # Identical candidate / NCA → GMR 1.0 for all subjects → BE 1.0.
        assert bundle.auc_cmax_be_score == pytest.approx(1.0)
        assert bundle.auc_cmax_source == "observed_trapezoid"
        assert bundle.n_subjects_nca_eligible == bundle.n_subjects_total == 10

    def test_constant_factor_keeps_cmax_within_goalpost(self) -> None:
        # Candidate is 90% of observed — GMR 0.9 inside [0.80, 1.25] → BE=1.0
        cohort = self._uniform_cohort(sim_value=4.5, observed_value=5.0)
        bundle = build_predictive_diagnostics(cohort, policy=self._policy())
        assert bundle.auc_cmax_be_score == pytest.approx(1.0)

    def test_severe_bias_fails_every_subject(self) -> None:
        # Candidate 50% of observed — GMR 0.5 outside goalpost for all subjects.
        cohort = self._uniform_cohort(sim_value=2.5, observed_value=5.0)
        bundle = build_predictive_diagnostics(cohort, policy=self._policy())
        assert bundle.auc_cmax_be_score == pytest.approx(0.0)

    def test_vpc_coverage_keys_match_percentiles(self) -> None:
        cohort = self._uniform_cohort(sim_value=5.0, observed_value=5.0)
        bundle = build_predictive_diagnostics(cohort, policy=self._policy())
        assert set(bundle.vpc.coverage.keys()) == {"p5", "p50", "p95"}
        assert bundle.vpc.percentiles == [5.0, 50.0, 95.0]

    def test_npe_is_non_negative(self) -> None:
        cohort = self._uniform_cohort(sim_value=7.0, observed_value=3.0)
        bundle = build_predictive_diagnostics(cohort, policy=self._policy())
        assert bundle.npe_score >= 0.0


# ---------------------------------------------------------------------------
# Per-subject NCA eligibility — floor behavior
# ---------------------------------------------------------------------------


class TestEligibilityFloors:
    def _mixed_cohort(self, *, n_total: int, n_eligible: int) -> list[SubjectSimulation]:
        times = np.array([0.0, 1.0, 2.0, 3.0])
        obs = np.array([5.0, 5.0, 5.0, 5.0])
        cohort: list[SubjectSimulation] = []
        for i in range(n_total):
            cohort.append(
                _constant_subject(
                    f"s{i}",
                    times=times,
                    observed=obs,
                    simulated_constant=5.0,
                    eligible=(i < n_eligible),
                    n_sims=30,
                )
            )
        return cohort

    def test_below_absolute_floor_drops_score(self) -> None:
        cohort = self._mixed_cohort(n_total=20, n_eligible=5)
        policy = _policy_with_floors(min_eligible=8, min_eligible_fraction=0.0)
        bundle = build_predictive_diagnostics(cohort, policy=policy)
        assert bundle.auc_cmax_be_score is None
        assert bundle.auc_cmax_source is None
        # Audit fields still populated even when score dropped.
        assert bundle.n_subjects_nca_eligible == 5
        assert bundle.n_subjects_total == 20

    def test_below_fraction_floor_drops_score(self) -> None:
        cohort = self._mixed_cohort(n_total=20, n_eligible=9)  # 9/20 = 0.45
        policy = _policy_with_floors(min_eligible=1, min_eligible_fraction=0.5)
        bundle = build_predictive_diagnostics(cohort, policy=policy)
        assert bundle.auc_cmax_be_score is None

    def test_both_floors_met_keeps_score(self) -> None:
        cohort = self._mixed_cohort(n_total=20, n_eligible=12)
        policy = _policy_with_floors(min_eligible=8, min_eligible_fraction=0.5)
        bundle = build_predictive_diagnostics(cohort, policy=policy)
        assert bundle.auc_cmax_be_score is not None
        assert bundle.auc_cmax_source == "observed_trapezoid"

    def test_backend_without_nca_diagnostics_drops_score(self) -> None:
        # Backend emits no per-subject NCA diagnostics — every subject is
        # treated as ineligible; score drops atomically.
        times = np.array([0.0, 1.0, 2.0])
        obs = np.array([5.0, 5.0, 5.0])
        sims = np.full((30, 3), 5.0)
        cohort = [
            SubjectSimulation(
                subject_id=f"s{i}",
                t_observed=times,
                observed_dv=obs,
                sims_at_observed=sims,
                nca_diagnostic=None,
            )
            for i in range(12)
        ]
        bundle = build_predictive_diagnostics(cohort, policy=_policy_with_floors())
        assert bundle.auc_cmax_be_score is None
        assert bundle.auc_cmax_source is None
        assert bundle.n_subjects_nca_eligible == 0


# ---------------------------------------------------------------------------
# Schema + integration
# ---------------------------------------------------------------------------


class TestBundleSchema:
    """Validator bounds mirror DiagnosticBundle — wire the fields 1-1."""

    def _valid_bundle(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "vpc": {
                "percentiles": [5.0, 50.0, 95.0],
                "coverage": {"p5": 0.9, "p50": 0.9, "p95": 0.9},
                "n_bins": 10,
                "prediction_corrected": False,
            },
            "pit_calibration": {
                "probability_levels": [0.05, 0.50, 0.95],
                "calibration": {"p5": 0.05, "p50": 0.50, "p95": 0.95},
                "n_observations": 96,
                "n_subjects": 12,
                "aggregation": "subject_robust",
            },
            "npe_score": 0.5,
            "auc_cmax_be_score": 0.75,
            "auc_cmax_source": "observed_trapezoid",
            "n_subjects_total": 12,
            "n_subjects_nca_eligible": 10,
        }
        base.update(overrides)
        return base

    def test_non_negative_npe_enforced(self) -> None:
        with pytest.raises(ValidationError):
            PredictiveSummaryBundle.model_validate(self._valid_bundle(npe_score=-0.1))

    def test_auc_cmax_in_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            PredictiveSummaryBundle.model_validate(self._valid_bundle(auc_cmax_be_score=1.5))

    def test_frozen_bundle_rejects_mutation(self) -> None:
        bundle = PredictiveSummaryBundle.model_validate(self._valid_bundle())
        with pytest.raises(ValidationError):
            bundle.npe_score = 42.0  # type: ignore[misc]


class TestRankerIntegration:
    """A populated PredictiveSummaryBundle feeds the ranker without proxy fallback."""

    def _result_from_bundle(
        self,
        *,
        model_id: str,
        backend: str,
        bundle: PredictiveSummaryBundle,
    ) -> BR:
        from apmode.bundle.models import ScoringContract

        if backend == "jax_node":
            contract = ScoringContract(
                nlpd_kind="conditional",
                re_treatment="pooled",
                nlpd_integrator="none",
                blq_method="none",
                observation_model="combined",
                float_precision="float32",
            )
        elif backend == "bayesian_stan":
            contract = ScoringContract(
                nlpd_kind="marginal",
                re_treatment="integrated",
                nlpd_integrator="hmc_nuts",
                blq_method="none",
                observation_model="combined",
                float_precision="float64",
            )
        else:
            contract = ScoringContract(
                nlpd_kind="marginal",
                re_treatment="integrated",
                nlpd_integrator="nlmixr2_focei",
                blq_method="none",
                observation_model="combined",
                float_precision="float64",
            )
        diagnostics = DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.05, cwres_sd=1.0, outlier_fraction=0.02),
            vpc=bundle.vpc,
            identifiability=IdentifiabilityFlags(
                condition_number=10.0,
                profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            npe_score=bundle.npe_score,
            auc_cmax_be_score=bundle.auc_cmax_be_score,
            auc_cmax_source=bundle.auc_cmax_source,
            scoring_contract=contract,
        )
        return BR(
            model_id=model_id,
            backend=backend,  # type: ignore[arg-type]
            converged=True,
            ofv=100.0,
            aic=110.0,
            bic=120.0,
            parameter_estimates={
                "CL": ParameterEstimate(name="CL", estimate=5.0, category="structural"),
            },
            eta_shrinkage={"CL": 0.05},
            convergence_metadata=ConvergenceMetadata(
                method="saem",
                converged=True,
                iterations=200,
                minimization_status="successful",
                wall_time_seconds=10.0,
            ),
            diagnostics=diagnostics,
            wall_time_seconds=10.0,
            backend_versions={"nlmixr2": "2.1.5"},
            initial_estimate_source="nca",
        )

    def test_cross_paradigm_ranker_uses_simulation_npe(self) -> None:
        times = np.array([0.0, 1.0, 2.0, 3.0])
        obs = np.array([5.0, 5.0, 5.0, 5.0])
        cohort = [
            _constant_subject(
                f"s{i}", times=times, observed=obs, simulated_constant=5.0, n_sims=40
            )
            for i in range(12)
        ]
        policy = _policy_with_floors(min_eligible=8, vpc_n_bins=3)
        bundle = build_predictive_diagnostics(cohort, policy=policy)

        # Populate two results (different backends) from the same bundle so
        # simulation metrics are forced.
        r1 = self._result_from_bundle(model_id="m1", backend="nlmixr2", bundle=bundle)
        r2 = self._result_from_bundle(model_id="m2", backend="jax_node", bundle=bundle)

        ranking = rank_cross_paradigm([r1, r2], gate3=policy, vpc_concordance_target=0.90)
        assert ranking.is_cross_paradigm is True
        for rc in ranking.ranked_candidates:
            assert rc.npe_source == "simulation"
            assert rc.npe == pytest.approx(bundle.npe_score)
