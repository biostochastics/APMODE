# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Stan codegen emitter (stan_emitter.py) + lowering test suite."""

from __future__ import annotations

import re

import pytest

from apmode.dsl.ast_models import (
    BLQM3,
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
    NODEAbsorption,
    OccasionByStudy,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    Transit,
    TwoCmt,
)
from apmode.dsl.stan_emitter import emit_stan

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_spec(
    absorption: object | None = None,
    distribution: object | None = None,
    elimination: object | None = None,
    variability: list[object] | None = None,
    observation: object | None = None,
    model_id: str = "test_model",
) -> DSLSpec:
    return DSLSpec(
        model_id=model_id,
        absorption=absorption or FirstOrder(ka=1.5),  # type: ignore[arg-type]
        distribution=distribution or OneCmt(V=70),  # type: ignore[arg-type]
        elimination=elimination or LinearElim(CL=5),  # type: ignore[arg-type]
        variability=variability or [],  # type: ignore[arg-type]
        observation=observation or Proportional(sigma_prop=0.15),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Basic emission
# ---------------------------------------------------------------------------


class TestStanEmitterBasic:
    """Basic Stan program generation."""

    def test_emits_stan_string(self) -> None:
        code = emit_stan(_make_spec())
        assert isinstance(code, str)
        assert len(code) > 100

    def test_has_required_blocks(self) -> None:
        code = emit_stan(_make_spec())
        assert "data {" in code
        assert "transformed data {" in code
        assert "parameters {" in code
        assert "transformed parameters {" in code
        assert "model {" in code
        assert "generated quantities {" in code

    def test_model_id_in_comment(self) -> None:
        code = emit_stan(_make_spec(model_id="my_model"))
        assert "my_model" in code

    def test_no_functions_block_for_linear(self) -> None:
        """Linear 1-cmt should use analytical solution, no functions block."""
        code = emit_stan(_make_spec())
        assert "functions {" not in code

    def test_functions_block_for_mm(self) -> None:
        """MM elimination needs ODE, should have functions block."""
        code = emit_stan(_make_spec(elimination=MichaelisMenten(Vmax=100, Km=10)))
        assert "functions {" in code
        assert "ode_rhs" in code


# ---------------------------------------------------------------------------
# Structural parameter emission
# ---------------------------------------------------------------------------


class TestStructuralParams:
    def test_1cmt_oral_linear(self) -> None:
        code = emit_stan(_make_spec())
        assert "log_ka" in code
        assert "log_V" in code
        assert "log_CL" in code

    def test_2cmt_linear(self) -> None:
        code = emit_stan(_make_spec(distribution=TwoCmt(V1=50, V2=80, Q=10)))
        assert "log_V1" in code
        assert "log_V2" in code
        assert "log_Q" in code

    def test_3cmt_linear(self) -> None:
        code = emit_stan(_make_spec(distribution=ThreeCmt(V1=50, V2=80, V3=100, Q2=10, Q3=5)))
        assert "log_Q2" in code
        assert "log_Q3" in code

    def test_mm_elim(self) -> None:
        code = emit_stan(_make_spec(elimination=MichaelisMenten(Vmax=100, Km=10)))
        assert "log_Vmax" in code
        assert "log_Km" in code

    def test_parallel_elim(self) -> None:
        code = emit_stan(_make_spec(elimination=ParallelLinearMM(CL=3, Vmax=100, Km=10)))
        assert "log_CL" in code
        assert "log_Vmax" in code

    def test_lagged_absorption(self) -> None:
        code = emit_stan(_make_spec(absorption=LaggedFirstOrder(ka=1.5, tlag=0.5)))
        assert "log_tlag" in code
        assert "tlag_i" in code

    def test_transit_absorption(self) -> None:
        code = emit_stan(_make_spec(absorption=Transit(n=3, ktr=2, ka=1)))
        assert "log_n" in code
        assert "log_ktr" in code


# ---------------------------------------------------------------------------
# Observation model emission
# ---------------------------------------------------------------------------


class TestObservationModel:
    def test_proportional(self) -> None:
        code = emit_stan(_make_spec(observation=Proportional(sigma_prop=0.15)))
        assert "sigma_prop" in code
        assert "lognormal" in code

    def test_additive(self) -> None:
        code = emit_stan(_make_spec(observation=Additive(sigma_add=0.5)))
        assert "sigma_add" in code
        # Should use normal, not lognormal
        assert re.search(r"dv\s*~\s*normal", code) is not None

    def test_combined(self) -> None:
        code = emit_stan(_make_spec(observation=Combined(sigma_prop=0.1, sigma_add=0.5)))
        assert "sigma_prop" in code
        assert "sigma_add" in code


# ---------------------------------------------------------------------------
# IIV emission
# ---------------------------------------------------------------------------


class TestIIVEmission:
    def test_diagonal_iiv(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[IIV(params=["CL", "V"], structure="diagonal")]  # type: ignore[list-item]
            )
        )
        assert "omega_CL" in code
        assert "omega_V" in code
        assert "eta_raw" in code

    def test_eta_in_transformed_params(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[IIV(params=["CL"], structure="diagonal")]  # type: ignore[list-item]
            )
        )
        assert "omega_CL * eta_raw" in code


# ---------------------------------------------------------------------------
# Covariate emission
# ---------------------------------------------------------------------------


class TestCovariateEmission:
    def test_power_covariate(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[
                    CovariateLink(param="CL", covariate="WT", form="power")  # type: ignore[list-item]
                ]
            )
        )
        assert "beta_CL_WT" in code
        assert "WT" in code
        assert "70" in code  # reference weight

    def test_categorical_covariate(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[
                    CovariateLink(param="CL", covariate="SEX", form="categorical")  # type: ignore[list-item]
                ]
            )
        )
        assert "beta_CL_SEX" in code


# ---------------------------------------------------------------------------
# Initial estimates as informative priors
# ---------------------------------------------------------------------------


class TestInitialEstimates:
    def test_priors_centered_on_estimates(self) -> None:
        code = emit_stan(
            _make_spec(),
            initial_estimates={"ka": 1.5, "V": 70, "CL": 5},
        )
        # log(1.5) ≈ 0.405
        assert "log_ka ~ normal(0.40" in code
        # log(70) ≈ 4.248
        assert "log_V ~ normal(4.24" in code

    def test_default_priors_without_estimates(self) -> None:
        code = emit_stan(_make_spec())
        # Without estimates, use N(0, 2)
        assert "log_ka ~ normal(0, 2)" in code


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestStanEmitterErrors:
    def test_rejects_node_modules(self) -> None:
        with pytest.raises(NotImplementedError, match="NODE"):
            emit_stan(
                _make_spec(
                    absorption=NODEAbsorption(dim=3, constraint_template="bounded_positive")
                )
            )

    def test_rejects_maturation_covariate(self) -> None:
        with pytest.raises(NotImplementedError, match=r"[Mm]aturation"):
            emit_stan(
                _make_spec(
                    variability=[
                        CovariateLink(  # type: ignore[list-item]
                            param="CL", covariate="AGE", form="maturation"
                        )
                    ]
                )
            )


# ---------------------------------------------------------------------------
# IOV emission
# ---------------------------------------------------------------------------


class TestIOVEmission:
    def test_iov_produces_valid_stan(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[IOV(params=["CL"], occasions=OccasionByStudy())]  # type: ignore[list-item]
            )
        )
        assert "N_occ" in code
        assert "occ" in code
        assert "omega_iov_CL" in code
        assert "eta_iov_raw" in code

    def test_iov_with_iiv(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="diagonal"),  # type: ignore[list-item]
                    IOV(params=["CL"], occasions=OccasionByStudy()),  # type: ignore[list-item]
                ]
            )
        )
        assert "omega_CL" in code
        assert "omega_iov_CL" in code
        assert "eta_raw" in code
        assert "eta_iov_raw" in code


# ---------------------------------------------------------------------------
# BLQ emission
# ---------------------------------------------------------------------------


class TestBLQEmission:
    def test_blq_m3_produces_valid_stan(self) -> None:
        code = emit_stan(_make_spec(observation=BLQM3(loq_value=0.1)))
        assert "cens" in code
        assert "loq" in code
        assert "normal_lcdf" in code
        assert "target +=" in code

    def test_blq_m4_produces_valid_stan(self) -> None:
        from apmode.dsl.ast_models import BLQM4

        code = emit_stan(_make_spec(observation=BLQM4(loq_value=0.05)))
        assert "cens" in code
        assert "log_diff_exp" in code

    def test_blq_m3_combined_error(self) -> None:
        code = emit_stan(
            _make_spec(
                observation=BLQM3(
                    loq_value=0.1,
                    error_model="combined",
                    sigma_prop=0.1,
                    sigma_add=0.5,
                )
            )
        )
        assert "sigma_prop" in code
        assert "sigma_add" in code

    def test_blq_log_lik_has_censoring(self) -> None:
        code = emit_stan(_make_spec(observation=BLQM3(loq_value=0.1)))
        assert "cens[n] == 0" in code


# ---------------------------------------------------------------------------
# Generated quantities
# ---------------------------------------------------------------------------


class TestGeneratedQuantities:
    def test_log_lik_for_loo(self) -> None:
        code = emit_stan(_make_spec())
        assert "log_lik" in code
        assert "lognormal_lpdf" in code


# ---------------------------------------------------------------------------
# Per-backend lowering test suite: nlmixr2 vs Stan cross-validation
# ---------------------------------------------------------------------------


class TestCrossBackendLowering:
    """Validate that both emitters accept the same DSLSpec inputs and
    produce structurally consistent output for each model class."""

    SPECS = [
        ("1cmt_oral_linear", _make_spec()),
        (
            "1cmt_oral_linear_iiv",
            _make_spec(
                variability=[IIV(params=["CL", "V"], structure="diagonal")]  # type: ignore[list-item]
            ),
        ),
        ("2cmt_oral_linear", _make_spec(distribution=TwoCmt(V1=50, V2=80, Q=10))),
        ("1cmt_mm_elim", _make_spec(elimination=MichaelisMenten(Vmax=100, Km=10))),
        ("1cmt_parallel_mm", _make_spec(elimination=ParallelLinearMM(CL=3, Vmax=100, Km=10))),
        ("1cmt_lagged", _make_spec(absorption=LaggedFirstOrder(ka=1.5, tlag=0.5))),
        ("1cmt_transit", _make_spec(absorption=Transit(n=3, ktr=2, ka=1))),
        (
            "1cmt_covariate",
            _make_spec(
                variability=[
                    IIV(params=["CL"], structure="diagonal"),  # type: ignore[list-item]
                    CovariateLink(param="CL", covariate="WT", form="power"),  # type: ignore[list-item]
                ]
            ),
        ),
        ("combined_error", _make_spec(observation=Combined(sigma_prop=0.1, sigma_add=0.5))),
    ]

    @pytest.mark.parametrize("name,spec", SPECS, ids=[s[0] for s in SPECS])
    def test_both_emitters_accept_spec(self, name: str, spec: DSLSpec) -> None:
        """Both nlmixr2 and Stan emitters accept the same DSLSpec."""
        from apmode.dsl.nlmixr2_emitter import emit_nlmixr2

        r_code = emit_nlmixr2(spec)
        stan_code = emit_stan(spec)
        assert len(r_code) > 50, f"nlmixr2 output too short for {name}"
        assert len(stan_code) > 100, f"Stan output too short for {name}"

    @pytest.mark.parametrize("name,spec", SPECS, ids=[s[0] for s in SPECS])
    def test_structural_params_present_in_both(self, name: str, spec: DSLSpec) -> None:
        """Both emitters declare the same structural parameters."""
        from apmode.dsl.nlmixr2_emitter import emit_nlmixr2

        r_code = emit_nlmixr2(spec)
        stan_code = emit_stan(spec)
        for param in spec.structural_param_names():
            assert param in r_code or f"l{param}" in r_code, f"nlmixr2 missing {param} for {name}"
            assert f"log_{param}" in stan_code, f"Stan missing log_{param} for {name}"
