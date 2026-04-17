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
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    OccasionByStudy,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    Transit,
    TwoCmt,
    ZeroOrder,
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
    def test_iov_raises_not_implemented(self) -> None:
        """IOV etas are declared but not applied in Stan — reject until fixed."""
        with pytest.raises(NotImplementedError, match="IOV"):
            emit_stan(
                _make_spec(
                    variability=[IOV(params=["CL"], occasions=OccasionByStudy())]  # type: ignore[list-item]
                )
            )

    def test_iov_with_iiv_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="IOV"):
            emit_stan(
                _make_spec(
                    variability=[
                        IIV(params=["CL", "V"], structure="diagonal"),  # type: ignore[list-item]
                        IOV(params=["CL"], occasions=OccasionByStudy()),  # type: ignore[list-item]
                    ]
                )
            )


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


class TestStanUnsupportedAbsorption:
    """Stan emitter should reject unsupported absorption types in ODE mode."""

    def test_zero_order_raises(self) -> None:
        spec = _make_spec(absorption=ZeroOrder(dur=0.5))
        with pytest.raises(NotImplementedError, match="ZeroOrder"):
            emit_stan(spec)

    def test_mixed_first_zero_raises(self) -> None:
        spec = _make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=0.6))
        with pytest.raises(NotImplementedError, match="MixedFirstZero"):
            emit_stan(spec)


# ---------------------------------------------------------------------------
# IVBolus — no depot compartment (W0 baseline for C1)
# ---------------------------------------------------------------------------


class TestIVBolusODE:
    """IVBolus dosing must not emit a depot compartment.

    With IV bolus, dose enters the central compartment directly. The Stan
    emitter must:

    1. Not alias ``y[1]`` as ``depot``.
    2. Not emit absorption-rate ODE terms (``dydt[1] = -ka * depot``).
    3. Not reference ``ka`` anywhere (IVBolus has no ``ka`` field and
       ``structural_param_names()`` does not include ``ka`` for IVBolus).
    4. Declare one fewer ODE state than the oral equivalent.
    """

    def test_ivbolus_onecmt_mm_has_no_depot_alias(self) -> None:
        """IVBolus + OneCmt + MM: central is y[1], no depot alias."""
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=OneCmt(V=70),
            elimination=MichaelisMenten(Vmax=100, Km=10),
        )
        code = emit_stan(spec)
        assert "real depot = y[1];" not in code, (
            "IVBolus must not alias y[1] as depot — dose enters central directly"
        )

    def test_ivbolus_onecmt_mm_has_no_absorption_term(self) -> None:
        """IVBolus + OneCmt + MM: no -ka*depot term in ODE."""
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=OneCmt(V=70),
            elimination=MichaelisMenten(Vmax=100, Km=10),
        )
        code = emit_stan(spec)
        assert "ka * depot" not in code, "IVBolus has no absorption phase"
        assert "-ka * depot" not in code, "IVBolus has no absorption phase"

    def test_ivbolus_onecmt_mm_does_not_reference_undefined_ka(self) -> None:
        """IVBolus has no ka parameter; emitted code must not reference ka tokens.

        ``structural_param_names()`` returns no ``ka`` for IVBolus, so any ``ka``
        reference in the ODE body is an undefined-variable error in Stan.
        """
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=OneCmt(V=70),
            elimination=MichaelisMenten(Vmax=100, Km=10),
        )
        code = emit_stan(spec)
        # Structural params should not include ka
        assert "log_ka" not in code, "IVBolus has no ka parameter"
        # The ODE body should not have bare `ka` as a standalone identifier
        assert not re.search(r"\bka\b", code), "IVBolus must not reference ka"

    def test_ivbolus_twocmt_mm_state_count(self) -> None:
        """IVBolus + TwoCmt + MM: 2 states (central, peripheral) not 3."""
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=TwoCmt(V1=50, V2=80, Q=10),
            elimination=MichaelisMenten(Vmax=100, Km=10),
        )
        code = emit_stan(spec)
        # No phantom depot alias
        assert "real depot = y[1];" not in code
        # Central is y[1], peripheral is y[2]
        assert "real centr = y[1];" in code
        assert "real periph = y[2];" in code

    def test_ivbolus_onecmt_linear_has_no_depot_alias(self) -> None:
        """IVBolus + OneCmt + Linear: analytical path may skip ODE entirely,
        but if an ODE is emitted (e.g. in a future refactor), it must
        still respect the no-depot invariant.
        """
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=OneCmt(V=70),
            elimination=LinearElim(CL=5),
        )
        code = emit_stan(spec)
        assert "real depot = y[1];" not in code
        assert not re.search(r"\bka\b", code), "IVBolus + Linear must not reference ka anywhere"

    def test_ivbolus_parallel_mm_has_no_absorption(self) -> None:
        """IVBolus + ParallelLinearMM: still no depot, no ka."""
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=OneCmt(V=70),
            elimination=ParallelLinearMM(CL=3, Vmax=100, Km=10),
        )
        code = emit_stan(spec)
        assert "real depot = y[1];" not in code
        assert not re.search(r"\bka\b", code)

    def test_oral_firstorder_still_has_depot(self) -> None:
        """Control: oral FirstOrder absorption still emits a depot compartment."""
        spec = _make_spec(
            absorption=FirstOrder(ka=1.5),
            distribution=OneCmt(V=70),
            elimination=MichaelisMenten(Vmax=100, Km=10),
        )
        code = emit_stan(spec)
        assert "real depot = y[1];" in code
        assert "ka * depot" in code


# ---------------------------------------------------------------------------
# Stan identifier sanitization
# ---------------------------------------------------------------------------


class TestStanIdentifierSanitization:
    """Covariate/parameter names must pass Stan identifier rules.

    Pydantic catches most syntactic violations at AST construction time;
    the emitter's ``_sanitize_stan_name`` catches reserved-word collisions
    and double-underscore suffixes.
    """

    def test_rejects_dotted_covariate_name_at_ast(self) -> None:
        """R-style ``WT.baseline`` is rejected at Pydantic construction."""
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="pattern"):
            CovariateLink(param="CL", covariate="WT.baseline", form="power")

    def test_rejects_leading_digit_covariate_at_ast(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="pattern"):
            CovariateLink(param="CL", covariate="1WT", form="power")

    def test_rejects_stan_reserved_keyword_at_emit(self) -> None:
        """Keywords pass Pydantic's regex but are rejected at emission."""
        spec = _make_spec(
            variability=[CovariateLink(param="CL", covariate="data", form="power")]  # type: ignore[list-item]
        )
        with pytest.raises(ValueError, match="reserved"):
            emit_stan(spec)

    def test_rejects_double_underscore_suffix_at_emit(self) -> None:
        spec = _make_spec(
            variability=[IIV(params=["CL__"], structure="diagonal")]  # type: ignore[list-item]
        )
        with pytest.raises(ValueError, match="double underscore"):
            emit_stan(spec)
