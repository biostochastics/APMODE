# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for the DSL prior AST (src/apmode/dsl/priors.py).

Covers: construction, target classification, parameterization-schema
validation, mixture-weight invariants, FDA-gated justification rules.
"""

from __future__ import annotations

import pytest

from apmode.dsl.priors import (
    BetaPrior,
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    HistoricalBorrowingPrior,
    InvGammaPrior,
    LKJPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
    PriorSpec,
    classify_target,
    default_corr_prior,
    default_covariate_prior,
    default_iiv_prior,
    default_residual_prior,
    default_structural_prior,
    validate_prior_family,
    validate_priors,
)


class TestPriorFamilies:
    def test_normal_requires_positive_sigma(self) -> None:
        NormalPrior(mu=0.0, sigma=1.0)
        with pytest.raises(ValueError):
            NormalPrior(mu=0.0, sigma=0.0)
        with pytest.raises(ValueError):
            NormalPrior(mu=0.0, sigma=-1.0)

    def test_half_cauchy_requires_positive_scale(self) -> None:
        HalfCauchyPrior(scale=1.0)
        with pytest.raises(ValueError):
            HalfCauchyPrior(scale=0.0)

    def test_lkj_requires_positive_eta(self) -> None:
        LKJPrior(eta=2.0)
        with pytest.raises(ValueError):
            LKJPrior(eta=0.0)

    def test_gamma_and_inv_gamma_positive(self) -> None:
        GammaPrior(alpha=2.0, beta=1.0)
        InvGammaPrior(alpha=2.0, beta=1.0)
        with pytest.raises(ValueError):
            GammaPrior(alpha=0.0, beta=1.0)
        with pytest.raises(ValueError):
            InvGammaPrior(alpha=2.0, beta=0.0)

    def test_beta_positive(self) -> None:
        BetaPrior(alpha=2.0, beta=3.0)
        with pytest.raises(ValueError):
            BetaPrior(alpha=0.0, beta=1.0)


class TestMixturePrior:
    def test_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="weights must sum"):
            MixturePrior(
                components=[NormalPrior(mu=0.0, sigma=1.0), NormalPrior(mu=1.0, sigma=1.0)],
                weights=[0.5, 0.6],
            )

    def test_weights_and_components_same_length(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            MixturePrior(
                components=[NormalPrior(mu=0.0, sigma=1.0), NormalPrior(mu=1.0, sigma=1.0)],
                weights=[0.5, 0.3, 0.2],
            )

    def test_weights_non_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            MixturePrior(
                components=[NormalPrior(mu=0.0, sigma=1.0), NormalPrior(mu=1.0, sigma=1.0)],
                weights=[1.5, -0.5],
            )

    def test_at_least_two_components(self) -> None:
        with pytest.raises(ValueError):
            MixturePrior(
                components=[NormalPrior(mu=0.0, sigma=1.0)],
                weights=[1.0],
            )

    def test_two_component_normal_mixture(self) -> None:
        m = MixturePrior(
            components=[
                NormalPrior(mu=0.0, sigma=1.0),
                NormalPrior(mu=2.0, sigma=0.5),
            ],
            weights=[0.8, 0.2],
        )
        assert len(m.components) == 2
        assert sum(m.weights) == pytest.approx(1.0)


class TestHistoricalBorrowingPrior:
    def test_requires_historical_refs(self) -> None:
        with pytest.raises(ValueError):
            HistoricalBorrowingPrior(map_mean=1.0, map_sd=0.3)  # no historical_refs

    def test_robust_weight_in_bounds(self) -> None:
        HistoricalBorrowingPrior(
            map_mean=1.0, map_sd=0.3, robust_weight=0.2, historical_refs=["trial_x"]
        )
        with pytest.raises(ValueError):
            HistoricalBorrowingPrior(
                map_mean=1.0, map_sd=0.3, robust_weight=1.5, historical_refs=["trial_x"]
            )


class TestPriorSpecJustification:
    def test_weakly_informative_needs_no_justification(self) -> None:
        PriorSpec(
            target="CL",
            family=NormalPrior(mu=0.0, sigma=2.0),
            source="weakly_informative",
        )

    def test_historical_data_requires_justification(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty justification"):
            PriorSpec(
                target="CL",
                family=NormalPrior(mu=1.0, sigma=0.3),
                source="historical_data",
                historical_refs=["trial_a"],
            )

    def test_historical_data_requires_refs(self) -> None:
        with pytest.raises(ValueError, match="historical_refs"):
            PriorSpec(
                target="CL",
                family=NormalPrior(mu=1.0, sigma=0.3),
                source="historical_data",
                justification="MAP from phase 2 trial",
            )

    def test_expert_elicitation_requires_justification(self) -> None:
        with pytest.raises(ValueError):
            PriorSpec(
                target="CL",
                family=NormalPrior(mu=0.0, sigma=2.0),
                source="expert_elicitation",
            )

    def test_meta_analysis_requires_justification(self) -> None:
        with pytest.raises(ValueError):
            PriorSpec(
                target="CL",
                family=NormalPrior(mu=0.0, sigma=2.0),
                source="meta_analysis",
            )


class TestClassifyTarget:
    def test_structural_param(self) -> None:
        assert classify_target("CL", {"CL", "V"}) == "structural"
        assert classify_target("ka", {"CL", "V", "ka"}) == "structural"

    def test_iiv_sd(self) -> None:
        assert classify_target("omega_CL", {"CL"}) == "iiv_sd"

    def test_iov_sd(self) -> None:
        assert classify_target("omega_iov_CL", {"CL"}) == "iov_sd"

    def test_residual_sd(self) -> None:
        assert classify_target("sigma_prop", {"CL"}) == "residual_sd"
        assert classify_target("sigma_add", {"CL"}) == "residual_sd"

    def test_corr_iiv(self) -> None:
        assert classify_target("corr_iiv", {"CL"}) == "corr_iiv"

    def test_covariate(self) -> None:
        assert classify_target("beta_CL_WT", {"CL"}) == "covariate"

    def test_unknown_target(self) -> None:
        assert classify_target("foobar", {"CL", "V"}) is None


class TestValidatePriorFamily:
    def test_structural_accepts_normal(self) -> None:
        assert validate_prior_family("structural", NormalPrior(mu=0, sigma=1)) is None

    def test_structural_rejects_halfcauchy(self) -> None:
        err = validate_prior_family("structural", HalfCauchyPrior(scale=1))
        assert err is not None and "HalfCauchy" in err

    def test_iiv_sd_accepts_half_family(self) -> None:
        assert validate_prior_family("iiv_sd", HalfNormalPrior(sigma=1)) is None
        assert validate_prior_family("iiv_sd", HalfCauchyPrior(scale=1)) is None
        assert validate_prior_family("iiv_sd", GammaPrior(alpha=2, beta=1)) is None

    def test_iiv_sd_rejects_normal(self) -> None:
        err = validate_prior_family("iiv_sd", NormalPrior(mu=0, sigma=1))
        assert err is not None

    def test_corr_iiv_requires_lkj(self) -> None:
        assert validate_prior_family("corr_iiv", LKJPrior(eta=2)) is None
        err = validate_prior_family("corr_iiv", NormalPrior(mu=0, sigma=1))
        assert err is not None

    def test_covariate_accepts_normal(self) -> None:
        assert validate_prior_family("covariate", NormalPrior(mu=0, sigma=1)) is None


class TestValidatePriors:
    def test_empty_list_valid(self) -> None:
        assert validate_priors([], {"CL", "V"}) == []

    def test_detects_unknown_target(self) -> None:
        priors = [PriorSpec(target="UNKNOWN", family=NormalPrior(mu=0, sigma=1))]
        errors = validate_priors(priors, {"CL", "V"})
        assert len(errors) == 1 and "does not match any known pattern" in errors[0]

    def test_detects_duplicate_target(self) -> None:
        priors = [
            PriorSpec(target="CL", family=NormalPrior(mu=0, sigma=1)),
            PriorSpec(target="CL", family=NormalPrior(mu=1, sigma=1)),
        ]
        errors = validate_priors(priors, {"CL"})
        assert any("Duplicate" in e for e in errors)

    def test_detects_bad_family_target_pair(self) -> None:
        priors = [PriorSpec(target="CL", family=HalfCauchyPrior(scale=1))]
        errors = validate_priors(priors, {"CL"})
        assert len(errors) == 1 and "HalfCauchy" in errors[0]

    def test_accepts_valid_mixed_priors(self) -> None:
        priors = [
            PriorSpec(target="CL", family=NormalPrior(mu=1.0, sigma=0.3)),
            PriorSpec(target="omega_CL", family=HalfCauchyPrior(scale=0.5)),
            PriorSpec(target="sigma_prop", family=HalfNormalPrior(sigma=0.2)),
            PriorSpec(target="beta_CL_WT", family=NormalPrior(mu=0.75, sigma=0.5)),
        ]
        assert validate_priors(priors, {"CL"}) == []


class TestDefaults:
    def test_default_structural_prior(self) -> None:
        p = default_structural_prior(log_init=1.5)
        assert isinstance(p, NormalPrior)
        assert p.mu == 1.5
        assert p.sigma == 2.0

    def test_default_iiv_is_half_cauchy(self) -> None:
        assert isinstance(default_iiv_prior(), HalfCauchyPrior)

    def test_default_residual_scale(self) -> None:
        p = default_residual_prior(init=0.1)
        assert isinstance(p, HalfCauchyPrior)
        assert p.scale == 0.1

    def test_default_corr_is_lkj(self) -> None:
        assert isinstance(default_corr_prior(), LKJPrior)

    def test_default_covariate_power(self) -> None:
        p = default_covariate_prior(form="power")
        assert isinstance(p, NormalPrior)
        assert p.mu == 0.75

    def test_default_covariate_other(self) -> None:
        p = default_covariate_prior(form="exponential")
        assert p.mu == 0.0 and p.sigma == 1.0


class TestPriorImmutability:
    def test_prior_spec_frozen(self) -> None:
        p = PriorSpec(target="CL", family=NormalPrior(mu=0, sigma=1))
        with pytest.raises(ValueError):
            p.target = "V"  # type: ignore[misc]

    def test_normal_prior_frozen(self) -> None:
        p = NormalPrior(mu=0, sigma=1)
        with pytest.raises(ValueError):
            p.mu = 1.0  # type: ignore[misc]


class TestMixtureOfLogNormalAndNormal:
    """Cover the LogNormal | Normal mixture case used for historical borrowing."""

    def test_two_component_heterogeneous_mixture(self) -> None:
        m = MixturePrior(
            components=[
                LogNormalPrior(mu=1.5, sigma=0.3),
                NormalPrior(mu=0.0, sigma=2.0),
            ],
            weights=[0.8, 0.2],
        )
        assert len(m.components) == 2
