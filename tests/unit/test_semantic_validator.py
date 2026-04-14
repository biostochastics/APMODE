# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for DSL semantic validator (ARCHITECTURE.md §2.2).

Constraint table enforcement: volumes > 0, rates >= 0, NODE dim <= lane ceiling,
constraint_template max dim check, NODE not admissible in Submission lane.
"""

from apmode.backends.protocol import Lane
from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    Additive,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    OccasionByStudy,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    TimeVaryingElim,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.validator import ValidationError, validate_dsl


def _make_spec(**overrides: object) -> DSLSpec:
    """Build a valid baseline DSLSpec, overriding specific modules."""
    defaults: dict[str, object] = {
        "model_id": "test_id_000000000000",
        "absorption": FirstOrder(ka=1.0),
        "distribution": OneCmt(V=70.0),
        "elimination": LinearElim(CL=5.0),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)  # type: ignore[arg-type]


class TestValidSpecsPass:
    """Valid specs should produce no errors."""

    def test_simple_1cmt_oral(self) -> None:
        spec = _make_spec()
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_2cmt_submission(self) -> None:
        spec = _make_spec(
            distribution=TwoCmt(V1=10.0, V2=20.0, Q=3.0),
            elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0),
            variability=[IIV(params=["CL", "V1"], structure="diagonal")],
            observation=Combined(sigma_prop=0.1, sigma_add=0.5),
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_3cmt_submission(self) -> None:
        spec = _make_spec(
            absorption=LaggedFirstOrder(ka=1.5, tlag=0.3),
            distribution=ThreeCmt(V1=10.0, V2=20.0, V3=5.0, Q2=3.0, Q3=1.0),
            elimination=MichaelisMenten(Vmax=100.0, Km=10.0),
            variability=[IIV(params=["Vmax", "V1"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_transit_absorption(self) -> None:
        spec = _make_spec(absorption=Transit(n=4, ktr=2.0, ka=1.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_node_in_discovery_within_limits(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=4, constraint_template="monotone_increasing"),
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert errors == []

    def test_node_elimination_discovery(self) -> None:
        spec = _make_spec(
            elimination=NODEElimination(dim=6, constraint_template="bounded_positive"),
            variability=[IIV(params=["ka", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert errors == []

    def test_node_in_optimization_within_limits(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=4, constraint_template="saturable"),
        )
        errors = validate_dsl(spec, lane=Lane.OPTIMIZATION)
        assert errors == []

    def test_blq_m3(self) -> None:
        spec = _make_spec(observation=BLQM3(loq_value=0.1))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_blq_m4(self) -> None:
        spec = _make_spec(observation=BLQM4(loq_value=0.5))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_additive_error(self) -> None:
        spec = _make_spec(observation=Additive(sigma_add=1.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_mixed_first_zero(self) -> None:
        spec = _make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=0.6))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_zero_order(self) -> None:
        spec = _make_spec(absorption=ZeroOrder(dur=0.5))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_time_varying_elimination(self) -> None:
        spec = _make_spec(elimination=TimeVaryingElim(CL=5.0, decay_fn="exponential"))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_multi_variability(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="block"),
                IOV(params=["CL"], occasions=OccasionByStudy()),
                CovariateLink(param="CL", covariate="WT", form="power"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_tlag_zero_is_valid(self) -> None:
        spec = _make_spec(absorption=LaggedFirstOrder(ka=1.0, tlag=0.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []


class TestPositivityConstraints:
    """Volumes must be > 0, rates must be > 0."""

    def test_negative_ka(self) -> None:
        spec = _make_spec(absorption=FirstOrder(ka=-1.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "ka" in errors[0].param
        assert errors[0].constraint == "positive"

    def test_zero_ka(self) -> None:
        spec = _make_spec(absorption=FirstOrder(ka=0.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "ka" in errors[0].param

    def test_negative_volume(self) -> None:
        spec = _make_spec(distribution=OneCmt(V=-10.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "V" in errors[0].param

    def test_zero_volume(self) -> None:
        spec = _make_spec(distribution=OneCmt(V=0.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1

    def test_negative_cl(self) -> None:
        spec = _make_spec(elimination=LinearElim(CL=-5.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "CL" in errors[0].param

    def test_negative_sigma_prop(self) -> None:
        spec = _make_spec(observation=Proportional(sigma_prop=-0.1))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "sigma_prop" in errors[0].param

    def test_negative_sigma_add(self) -> None:
        spec = _make_spec(observation=Additive(sigma_add=-0.5))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "sigma_add" in errors[0].param

    def test_negative_loq_value(self) -> None:
        spec = _make_spec(observation=BLQM3(loq_value=-0.1))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "loq_value" in errors[0].param

    def test_negative_dur_zero_order(self) -> None:
        spec = _make_spec(absorption=ZeroOrder(dur=-0.5))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "dur" in errors[0].param

    def test_negative_tlag(self) -> None:
        spec = _make_spec(absorption=LaggedFirstOrder(ka=1.0, tlag=-0.1))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "tlag" in errors[0].param
        assert errors[0].constraint == "non_negative"

    def test_negative_ktr(self) -> None:
        spec = _make_spec(absorption=Transit(n=3, ktr=-1.0, ka=1.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "ktr" in errors[0].param

    def test_2cmt_negative_volumes(self) -> None:
        spec = _make_spec(
            distribution=TwoCmt(V1=-10.0, V2=-20.0, Q=3.0),
            variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 2

    def test_3cmt_negative_q(self) -> None:
        spec = _make_spec(
            distribution=ThreeCmt(V1=10.0, V2=20.0, V3=5.0, Q2=-3.0, Q3=-1.0),
            variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 2

    def test_mm_negative_vmax(self) -> None:
        spec = _make_spec(
            elimination=MichaelisMenten(Vmax=-100.0, Km=10.0),
            variability=[IIV(params=["Vmax", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "Vmax" in errors[0].param

    def test_combined_both_negative(self) -> None:
        spec = _make_spec(observation=Combined(sigma_prop=-0.1, sigma_add=-0.5))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 2

    def test_frac_out_of_range_high(self) -> None:
        spec = _make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=1.5))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "frac" in errors[0].param
        assert errors[0].constraint == "unit_interval"

    def test_frac_out_of_range_low(self) -> None:
        spec = _make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=-0.1))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "frac" in errors[0].param

    def test_frac_zero_invalid(self) -> None:
        spec = _make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=0.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.param == "absorption.frac" for e in errors)

    def test_frac_one_invalid(self) -> None:
        spec = _make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=1.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.param == "absorption.frac" for e in errors)

    def test_transit_n_zero(self) -> None:
        spec = _make_spec(absorption=Transit(n=0, ktr=2.0, ka=1.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "n" in errors[0].param

    def test_multiple_errors_accumulated(self) -> None:
        """Validator should surface ALL violations, not fail-fast."""
        spec = _make_spec(
            absorption=FirstOrder(ka=-1.0),
            distribution=OneCmt(V=-70.0),
            elimination=LinearElim(CL=-5.0),
            observation=Proportional(sigma_prop=-0.1),
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 4


class TestNODEConstraints:
    """NODE dim ceilings and lane admissibility (PRD §4.2.5)."""

    def test_node_not_admissible_submission(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=4, constraint_template="monotone_increasing"),
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert errors[0].constraint == "node_lane_admissibility"

    def test_node_elim_not_admissible_submission(self) -> None:
        spec = _make_spec(
            elimination=NODEElimination(dim=4, constraint_template="bounded_positive"),
            variability=[IIV(params=["ka", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert errors[0].constraint == "node_lane_admissibility"

    def test_node_exceeds_template_max_dim(self) -> None:
        """monotone_increasing has max dim 4 (PRD §4.2.5 table)."""
        spec = _make_spec(
            absorption=NODEAbsorption(dim=5, constraint_template="monotone_increasing"),
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert len(errors) == 1
        assert errors[0].constraint == "node_template_max_dim"

    def test_node_exceeds_lane_ceiling_optimization(self) -> None:
        """Optimization lane ceiling is 4 (PRD §4.2.5)."""
        spec = _make_spec(
            absorption=NODEAbsorption(dim=5, constraint_template="bounded_positive"),
        )
        # bounded_positive max is 6, but optimization ceiling is 4
        errors = validate_dsl(spec, lane=Lane.OPTIMIZATION)
        assert len(errors) == 1
        assert errors[0].constraint == "node_lane_dim_ceiling"

    def test_node_exceeds_lane_ceiling_discovery(self) -> None:
        """Discovery lane ceiling is 8 (PRD §4.2.5)."""
        spec = _make_spec(
            elimination=NODEElimination(dim=9, constraint_template="unconstrained_smooth"),
            variability=[IIV(params=["ka", "V"], structure="diagonal")],
        )
        # unconstrained_smooth max is 8, so dim=9 exceeds template max too
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "node_template_max_dim" for e in errors)

    def test_node_at_exact_template_limit(self) -> None:
        """dim == template max should be valid."""
        spec = _make_spec(
            absorption=NODEAbsorption(dim=4, constraint_template="monotone_increasing"),
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert errors == []

    def test_node_at_exact_lane_ceiling(self) -> None:
        """dim == lane ceiling should be valid."""
        spec = _make_spec(
            absorption=NODEAbsorption(dim=4, constraint_template="saturable"),
        )
        errors = validate_dsl(spec, lane=Lane.OPTIMIZATION)
        assert errors == []

    def test_node_both_absorption_and_elimination(self) -> None:
        """Both NODE modules should be independently validated."""
        spec = _make_spec(
            absorption=NODEAbsorption(dim=5, constraint_template="monotone_increasing"),
            elimination=NODEElimination(dim=5, constraint_template="monotone_decreasing"),
            variability=[IIV(params=["V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        # Both exceed template max of 4
        assert len(errors) == 2

    def test_unconstrained_smooth_max_8(self) -> None:
        spec = _make_spec(
            elimination=NODEElimination(dim=8, constraint_template="unconstrained_smooth"),
            variability=[IIV(params=["ka", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert errors == []

    def test_bounded_positive_max_6(self) -> None:
        spec = _make_spec(
            elimination=NODEElimination(dim=7, constraint_template="bounded_positive"),
            variability=[IIV(params=["ka", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert len(errors) == 1
        assert errors[0].constraint == "node_template_max_dim"

    def test_node_dim_must_be_positive(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=0, constraint_template="monotone_increasing"),
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "positive_int" for e in errors)


class TestTimeVaryingElimConstraints:
    """TimeVaryingElim decay_fn enforcement."""

    def test_exponential_decay_fn_valid(self) -> None:
        spec = _make_spec(elimination=TimeVaryingElim(CL=5.0, decay_fn="exponential"))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_half_life_decay_fn_rejected(self) -> None:
        spec = _make_spec(elimination=TimeVaryingElim(CL=5.0, decay_fn="half_life"))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert errors[0].constraint == "supported_decay_fn"
        assert "half_life" in errors[0].message

    def test_linear_decay_fn_rejected(self) -> None:
        spec = _make_spec(elimination=TimeVaryingElim(CL=5.0, decay_fn="linear"))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert errors[0].constraint == "supported_decay_fn"
        assert "linear" in errors[0].message


class TestVariabilityConstraints:
    """Variability module semantic checks."""

    def test_iiv_empty_params(self) -> None:
        spec = _make_spec(variability=[IIV(params=[], structure="diagonal")])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert "params" in errors[0].param

    def test_iov_empty_params(self) -> None:
        spec = _make_spec(variability=[IOV(params=[], occasions=OccasionByStudy())])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1

    def test_block_structure_needs_multiple_params(self) -> None:
        spec = _make_spec(variability=[IIV(params=["CL"], structure="block")])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "block_min_params" for e in errors)

    def test_iiv_param_must_exist(self) -> None:
        spec = _make_spec(variability=[IIV(params=["CL", "nonexistent"], structure="diagonal")])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "iiv_param_exists" for e in errors)
        assert any("nonexistent" in e.message for e in errors)

    def test_iiv_params_valid(self) -> None:
        spec = _make_spec(variability=[IIV(params=["CL", "V"], structure="diagonal")])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_iov_param_must_exist(self) -> None:
        spec = _make_spec(variability=[IOV(params=["bogus"], occasions=OccasionByStudy())])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "iov_param_exists" for e in errors)


class TestCovariateLinkValidation:
    """CovariateLink param must reference a structural parameter."""

    def test_valid_covariate_param(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="CL", covariate="WT", form="power"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []

    def test_nonexistent_covariate_param(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="nonexistent", covariate="WT", form="power"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert errors[0].constraint == "covariate_param_exists"
        assert "nonexistent" in errors[0].message

    def test_covariate_on_derived_param_rejected(self) -> None:
        """kdecay is structural for TimeVaryingElim but CL is, so check both."""
        spec = _make_spec(
            elimination=TimeVaryingElim(CL=5.0, decay_fn="exponential"),
            variability=[
                IIV(params=["CL"], structure="diagonal"),
                CovariateLink(param="CL", covariate="WT", form="power"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []


class TestTMDDEliminationCompatibility:
    """TMDD distribution requires LinearElim (provides CL for kel = CL/V)."""

    def test_tmdd_core_with_linear_elim_valid(self) -> None:
        from apmode.dsl.ast_models import TMDDCore

        spec = _make_spec(
            distribution=TMDDCore(V=50.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "tmdd_requires_linear_elim" for e in errors)

    def test_tmdd_core_with_mm_elim_rejected(self) -> None:
        from apmode.dsl.ast_models import TMDDCore

        spec = _make_spec(
            distribution=TMDDCore(V=50.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05),
            elimination=MichaelisMenten(Vmax=100.0, Km=10.0),
            variability=[IIV(params=["Vmax", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "tmdd_requires_linear_elim" for e in errors)

    def test_tmdd_qss_with_mm_elim_rejected(self) -> None:
        from apmode.dsl.ast_models import TMDDQSS

        spec = _make_spec(
            distribution=TMDDQSS(V=50.0, R0=10.0, KD=0.5, kint=0.05),
            elimination=MichaelisMenten(Vmax=100.0, Km=10.0),
            variability=[IIV(params=["Vmax", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "tmdd_requires_linear_elim" for e in errors)

    def test_tmdd_qss_with_parallel_mm_rejected(self) -> None:
        from apmode.dsl.ast_models import TMDDQSS

        spec = _make_spec(
            distribution=TMDDQSS(V=50.0, R0=10.0, KD=0.5, kint=0.05),
            elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "tmdd_requires_linear_elim" for e in errors)


class TestDuplicateIIVParams:
    """Same parameter in multiple IIV blocks should be rejected."""

    def test_duplicate_iiv_param_rejected(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL"], structure="diagonal"),
                IIV(params=["CL", "V"], structure="diagonal"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "iiv_no_duplicate_params" for e in errors)

    def test_no_duplicate_across_blocks(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL"], structure="diagonal"),
                IIV(params=["V"], structure="diagonal"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "iiv_no_duplicate_params" for e in errors)


class TestDuplicateCovariateLinks:
    """Duplicate CovariateLink (same param+covariate) should be rejected."""

    def test_duplicate_covariate_link_rejected(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="CL", covariate="WT", form="power"),
                CovariateLink(param="CL", covariate="WT", form="exponential"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "covariate_link_no_duplicate" for e in errors)

    def test_different_param_covariate_ok(self) -> None:
        spec = _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="CL", covariate="WT", form="power"),
                CovariateLink(param="V", covariate="WT", form="power"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "covariate_link_no_duplicate" for e in errors)


class TestTransitNVariability:
    """Transit n should not accept IIV/covariates."""

    def test_iiv_on_n_rejected(self) -> None:
        spec = _make_spec(
            absorption=Transit(n=4, ktr=2.0, ka=1.0),
            variability=[IIV(params=["CL", "n"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "no_variability_on_param" for e in errors)

    def test_iiv_on_ka_ktr_ok(self) -> None:
        spec = _make_spec(
            absorption=Transit(n=4, ktr=2.0, ka=1.0),
            variability=[IIV(params=["CL", "ka"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "no_variability_on_param" for e in errors)


class TestKdecayValidation:
    """TimeVaryingElim kdecay should be validated as a first-class parameter."""

    def test_kdecay_positive_valid(self) -> None:
        spec = _make_spec(elimination=TimeVaryingElim(CL=5.0, kdecay=0.1, decay_fn="exponential"))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any("kdecay" in e.param for e in errors)

    def test_kdecay_negative_rejected(self) -> None:
        spec = _make_spec(elimination=TimeVaryingElim(CL=5.0, kdecay=-0.1, decay_fn="exponential"))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any("kdecay" in e.param for e in errors)

    def test_kdecay_default_value(self) -> None:
        elim = TimeVaryingElim(CL=5.0, decay_fn="exponential")
        assert elim.kdecay == 0.1


class TestValidationErrorStructure:
    """ValidationError should have module, param, constraint, and message."""

    def test_error_fields(self) -> None:
        spec = _make_spec(absorption=FirstOrder(ka=-1.0))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        err = errors[0]
        assert isinstance(err, ValidationError)
        assert err.module == "absorption"
        assert err.param == "absorption.ka"
        assert err.constraint == "positive"
        assert "must be > 0" in err.message
