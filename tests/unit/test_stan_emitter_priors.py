# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for Bayesian prior emission in stan_emitter.

Verifies user-declared priors on spec.priors correctly override defaults
for structural params, IIV omega, residual sigma, and covariate betas.
Also checks mixture/historical-borrowing compilation to log_sum_exp form.
"""

from __future__ import annotations

from apmode.dsl.ast_models import (
    IIV,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
    TwoCmt,
)
from apmode.dsl.priors import (
    HalfCauchyPrior,
    HalfNormalPrior,
    HistoricalBorrowingPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
    PriorSpec,
)
from apmode.dsl.stan_emitter import emit_stan


def _one_cmt(priors: list[PriorSpec] | None = None) -> DSLSpec:
    return DSLSpec(
        model_id="test_model",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=20.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.3),
        priors=priors or [],
    )


def _two_cmt_with_covariate(priors: list[PriorSpec] | None = None) -> DSLSpec:
    return DSLSpec(
        model_id="test_model_cov",
        absorption=FirstOrder(ka=1.0),
        distribution=TwoCmt(V1=20.0, V2=30.0, Q=5.0),
        elimination=LinearElim(CL=5.0),
        variability=[
            IIV(params=["CL", "V1"], structure="diagonal"),
            CovariateLink(param="CL", covariate="WT", form="power"),
        ],
        observation=Combined(sigma_prop=0.2, sigma_add=0.1),
        priors=priors or [],
    )


class TestDefaultPriors:
    def test_structural_default_is_weak_normal(self) -> None:
        stan = emit_stan(_one_cmt())
        assert "log_CL ~ normal(0, 2)" in stan or "log_CL ~ normal(" in stan

    def test_iiv_default_is_half_cauchy(self) -> None:
        stan = emit_stan(_one_cmt())
        assert "omega_CL ~ cauchy(0, 1)" in stan
        assert "omega_V ~ cauchy(0, 1)" in stan

    def test_etas_non_centered(self) -> None:
        stan = emit_stan(_one_cmt())
        assert "to_vector(eta_raw) ~ std_normal()" in stan

    def test_sigma_default(self) -> None:
        stan = emit_stan(_one_cmt())
        assert "sigma_prop ~ cauchy(" in stan


class TestUserStructuralPriors:
    def test_normal_prior_override(self) -> None:
        priors = [PriorSpec(target="CL", family=NormalPrior(mu=1.5, sigma=0.3))]
        stan = emit_stan(_one_cmt(priors))
        assert "log_CL ~ normal(1.500000, 0.300000);" in stan
        # V keeps default
        assert "log_V ~ normal(" in stan

    def test_lognormal_on_log_scale_becomes_normal(self) -> None:
        """LogNormal(mu, sigma) on stan variable log_CL must emit as normal(mu, sigma).

        This is mathematically correct: if theta ~ LogNormal(mu, sigma), then
        log(theta) ~ Normal(mu, sigma). Since our Stan variable IS log_CL,
        we sample from Normal on the log-scale variable.
        """
        priors = [PriorSpec(target="CL", family=LogNormalPrior(mu=1.5, sigma=0.3))]
        stan = emit_stan(_one_cmt(priors))
        assert "log_CL ~ normal(1.500000, 0.300000);" in stan
        assert "log_CL ~ lognormal(" not in stan  # must NOT double-log


class TestUserIIVPriors:
    def test_half_normal_override(self) -> None:
        priors = [PriorSpec(target="omega_CL", family=HalfNormalPrior(sigma=0.5))]
        stan = emit_stan(_one_cmt(priors))
        assert "omega_CL ~ normal(0, 0.500000);" in stan
        # omega_V keeps default half-cauchy
        assert "omega_V ~ cauchy(0, 1)" in stan


class TestUserResidualPriors:
    def test_sigma_prop_override(self) -> None:
        priors = [PriorSpec(target="sigma_prop", family=HalfCauchyPrior(scale=0.1))]
        stan = emit_stan(_one_cmt(priors))
        assert "sigma_prop ~ cauchy(0, 0.100000);" in stan

    def test_combined_error_both_sigmas_overridable(self) -> None:
        priors = [
            PriorSpec(target="sigma_prop", family=HalfCauchyPrior(scale=0.1)),
            PriorSpec(target="sigma_add", family=HalfNormalPrior(sigma=0.05)),
        ]
        stan = emit_stan(_two_cmt_with_covariate(priors))
        assert "sigma_prop ~ cauchy(0, 0.100000);" in stan
        assert "sigma_add ~ normal(0, 0.050000);" in stan


class TestUserCovariatePriors:
    def test_covariate_beta_override(self) -> None:
        priors = [
            PriorSpec(target="beta_CL_WT", family=NormalPrior(mu=0.75, sigma=0.25)),
        ]
        stan = emit_stan(_two_cmt_with_covariate(priors))
        assert "beta_CL_WT ~ normal(0.750000, 0.250000);" in stan


class TestMixturePriorEmission:
    def test_log_sum_exp_form(self) -> None:
        mixture = MixturePrior(
            components=[
                NormalPrior(mu=1.5, sigma=0.3),
                NormalPrior(mu=0.0, sigma=2.0),
            ],
            weights=[0.8, 0.2],
        )
        priors = [PriorSpec(target="CL", family=mixture)]
        stan = emit_stan(_one_cmt(priors))
        assert "target += log_sum_exp" in stan
        assert "normal_lpdf(log_CL | 1.500000, 0.300000)" in stan
        assert "normal_lpdf(log_CL | 0.000000, 2.000000)" in stan

    def test_log_weights_computed_correctly(self) -> None:
        import math

        mixture = MixturePrior(
            components=[
                NormalPrior(mu=1.0, sigma=0.5),
                NormalPrior(mu=0.0, sigma=2.0),
            ],
            weights=[0.7, 0.3],
        )
        priors = [PriorSpec(target="CL", family=mixture)]
        stan = emit_stan(_one_cmt(priors))
        # Log weights should match math.log(0.7) and math.log(0.3) to 6 decimal places
        expected_log_07 = f"{math.log(0.7):.6f}"
        expected_log_03 = f"{math.log(0.3):.6f}"
        assert expected_log_07 in stan
        assert expected_log_03 in stan


class TestHistoricalBorrowingEmission:
    def test_compiles_to_two_component_mixture(self) -> None:
        """Robust MAP per Schmidli 2014 compiles to log_sum_exp with MAP + weak."""
        import math

        prior = HistoricalBorrowingPrior(
            map_mean=1.5,
            map_sd=0.3,
            robust_weight=0.2,
            historical_refs=["phase2_trial"],
        )
        spec = _one_cmt(
            [
                PriorSpec(
                    target="CL",
                    family=prior,
                    source="historical_data",
                    justification="phase2 posterior",
                    historical_refs=["phase2_trial"],
                )
            ]
        )
        stan = emit_stan(spec)
        # MAP component Normal(1.5, 0.3) present
        assert "normal_lpdf(log_CL | 1.500000, 0.300000)" in stan
        # Weak component Normal(0, 2) present
        assert "normal_lpdf(log_CL | 0.000000, 2.000000)" in stan
        # Weights: log(0.8) for MAP, log(0.2) for weak
        assert f"{math.log(0.8):.6f}" in stan
        assert f"{math.log(0.2):.6f}" in stan


class TestPriorsAreCompilableOutputShape:
    """Smoke test: the emitted Stan program is structurally valid."""

    def test_starts_with_generated_marker(self) -> None:
        stan = emit_stan(_one_cmt())
        assert stan.startswith("// APMODE generated Stan model:")

    def test_has_all_required_blocks(self) -> None:
        stan = emit_stan(_one_cmt())
        for block in [
            "data {",
            "transformed data {",
            "parameters {",
            "transformed parameters {",
            "model {",
            "generated quantities {",
        ]:
            assert block in stan

    def test_priors_dont_break_log_lik_generation(self) -> None:
        priors = [PriorSpec(target="CL", family=NormalPrior(mu=1.5, sigma=0.3))]
        stan = emit_stan(_one_cmt(priors))
        assert "log_lik" in stan  # Required for LOO-CV
