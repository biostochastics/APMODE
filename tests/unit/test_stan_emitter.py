# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Stan codegen emitter (stan_emitter.py) + lowering test suite."""

from __future__ import annotations

import importlib.util
import re
from typing import TYPE_CHECKING

import pytest

from apmode.dsl.ast_models import (
    BLQM3,
    IIV,
    IOV,
    TMDDQSS,
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
    ObservationEndpoint,
    OccasionByStudy,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.priors import LKJPrior, build_prior_spec
from apmode.dsl.stan_emitter import emit_stan

if TYPE_CHECKING:
    from pathlib import Path

    from syrupy.assertion import SnapshotAssertion

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_spec(
    absorption: object | None = None,
    distribution: object | None = None,
    elimination: object | None = None,
    variability: list[object] | None = None,
    covariates: list[object] | None = None,
    observation: object | None = None,
    priors: list[object] | None = None,
    model_id: str = "test_model",
) -> DSLSpec:
    return DSLSpec(
        model_id=model_id,
        absorption=absorption or FirstOrder(),  # type: ignore[arg-type]
        distribution=distribution or OneCmt(),  # type: ignore[arg-type]
        elimination=elimination or LinearElim(),  # type: ignore[arg-type]
        variability=variability or [],  # type: ignore[arg-type]
        covariates=covariates or [],  # type: ignore[arg-type]
        observation=observation or Proportional(sigma_prop=0.15),  # type: ignore[arg-type]
        priors=priors or [],  # type: ignore[arg-type]
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
        code = emit_stan(_make_spec(elimination=MichaelisMenten()))
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
        code = emit_stan(_make_spec(distribution=TwoCmt()))
        assert "log_V1" in code
        assert "log_V2" in code
        assert "log_Q" in code

    def test_3cmt_linear(self) -> None:
        code = emit_stan(_make_spec(distribution=ThreeCmt()))
        assert "log_Q2" in code
        assert "log_Q3" in code

    def test_mm_elim(self) -> None:
        code = emit_stan(_make_spec(elimination=MichaelisMenten()))
        assert "log_Vmax" in code
        assert "log_Km" in code

    def test_parallel_elim(self) -> None:
        code = emit_stan(_make_spec(elimination=ParallelLinearMM()))
        assert "log_CL" in code
        assert "log_Vmax" in code

    def test_lagged_absorption(self) -> None:
        code = emit_stan(_make_spec(absorption=LaggedFirstOrder()))
        assert "log_tlag" in code
        assert "tlag_i" in code

    def test_transit_absorption(self) -> None:
        code = emit_stan(_make_spec(absorption=Transit(n=3)))
        assert "log_n" not in code
        assert "log_ktr" in code
        assert "real transit_1 = y[1];" in code
        assert "dydt[3] = ktr * transit_2 - ktr * transit_3;" in code
        assert "dydt[4] = ktr * transit_3 - ka * depot;" in code


# ---------------------------------------------------------------------------
# Observation model emission
# ---------------------------------------------------------------------------


class TestObservationModel:
    def test_proportional(self) -> None:
        # #1: proportional likelihood is now Normal(f, sigma_prop * f) to
        # match the nlmixr2 ``cp ~ prop(prop.sd)`` emission and stay
        # internally consistent with the BLQ M3/M4 proportional path.
        # Lognormal is no longer emitted on this route.
        code = emit_stan(_make_spec(observation=Proportional(sigma_prop=0.15)))
        assert "sigma_prop" in code
        assert "lognormal" not in code
        assert "normal(f[n], sigma_prop * f[n])" in code

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
                covariates=[
                    CovariateLink(  # type: ignore[list-item]
                        param="CL", covariate="WT", form="power", theta=0.75, ref=70.0
                    )
                ]
            )
        )
        assert "beta_CL_WT" in code
        assert "WT" in code
        assert "70" in code  # reference weight

    def test_categorical_covariate(self) -> None:
        code = emit_stan(
            _make_spec(
                covariates=[
                    CovariateLink(  # type: ignore[list-item]
                        param="CL", covariate="SEX", form="categorical", reference="M"
                    )
                ]
            )
        )
        assert "beta_CL_SEX" in code

    def test_maturation_covariate(self) -> None:
        """Maturation (Hill/Emax) covariate lowers to Stan and mirrors the
        nlmixr2 back-transform ``log(cov^hill / (cov^hill + TM50^hill))``
        term with matching ``beta_``/``TM50_`` parameter naming."""
        code = emit_stan(
            _make_spec(
                covariates=[
                    CovariateLink(  # type: ignore[list-item]
                        param="CL", covariate="AGE", form="maturation", tm50=45.0, hill=3.0
                    )
                ]
            )
        )
        # Hill exponent coefficient + TM50 half-maturation parameter declared.
        assert "real beta_CL_AGE;" in code
        assert "real<lower=0> TM50_CL_AGE;" in code
        # Back-transform term is mathematically identical to nlmixr2's
        # log(AGE^beta / (AGE^beta + TM50^beta)) additive log-domain term.
        assert (
            " + log(AGE[i]^beta_CL_AGE / (AGE[i]^beta_CL_AGE + TM50_CL_AGE^beta_CL_AGE))"
        ) in code
        # Priors: beta centered on the Hill starting value, TM50 log-normal
        # centered on the tm50 starting value.
        assert "beta_CL_AGE ~ normal(3.0, 0.5);" in code
        assert "TM50_CL_AGE ~ lognormal(3.8067, 0.5);" in code


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

    def test_rejects_multi_analyte_observations(self) -> None:
        """P1.7: multi-analyte observations: blocks are a Phase 2 gap for Stan."""
        spec = _make_spec().model_copy(
            update={
                "observations": {
                    "plasma": ObservationEndpoint(
                        name="plasma",
                        dvid=1,
                        prediction="C_central",
                        error=Proportional(sigma_prop=0.15),
                    ),
                }
            }
        )
        with pytest.raises(NotImplementedError, match="observations"):
            emit_stan(spec)


# ---------------------------------------------------------------------------
# IOV emission
# ---------------------------------------------------------------------------


def _assert_no_unused_declarations(code: str, declared_names: list[str]) -> None:
    """Every name in `declared_names` must appear at least once besides its declaration."""
    for name in declared_names:
        assert code.count(name) >= 2, f"{name!r} declared but never reused in emitted program"


class TestIOVEmission:
    def test_iov_declares_occasion_data(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[IOV(params=["CL"], occasions=OccasionByStudy())]  # type: ignore[list-item]
            )
        )
        assert "int<lower=1> N_occ;" in code
        assert "array[N] int<lower=1,upper=N_occ> occ;" in code

    def test_iov_declares_omega_and_eta_raw(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[IOV(params=["CL"], occasions=OccasionByStudy())]  # type: ignore[list-item]
            )
        )
        assert "real<lower=0> omega_iov_CL;" in code
        assert "matrix[N_subjects * N_occ, 1] eta_iov_raw;" in code
        assert "to_vector(eta_iov_raw) ~ std_normal();" in code

    def test_iov_back_transform_contains_occasion_term_only_for_targeted_param(self) -> None:
        """CL carries IOV, V does not — only CL's back-transform should reference it."""
        code = emit_stan(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="diagonal"),  # type: ignore[list-item]
                    IOV(params=["CL"], occasions=OccasionByStudy()),  # type: ignore[list-item]
                ]
            )
        )
        assert "array[N_occ] real CL_i;" in code
        assert "omega_iov_CL * eta_iov_raw[(i - 1) * N_occ + occ_k, 1]" in code
        assert "real V_i = exp(log_V + omega_V * eta_raw[i, 2]);" in code
        assert "array[N_occ] real V_i;" not in code

    def test_iov_no_declared_but_unused_identifiers(self) -> None:
        code = emit_stan(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="diagonal"),  # type: ignore[list-item]
                    IOV(params=["CL", "V"], occasions=OccasionByStudy()),  # type: ignore[list-item]
                ]
            )
        )
        _assert_no_unused_declarations(code, ["omega_iov_CL", "omega_iov_V", "eta_iov_raw"])

    def test_iov_forces_ode_path_even_for_analytically_solvable_model(self) -> None:
        """1-cmt oral + linear elimination is normally analytical; IOV can't be
        represented by the closed-form superposition solution, so it must
        route through the ODE solver instead of silently ignoring occasions."""
        code = emit_stan(
            _make_spec(
                variability=[IOV(params=["CL"], occasions=OccasionByStudy())]  # type: ignore[list-item]
            )
        )
        assert "functions {" in code
        assert "ode_rk45" in code

    def test_iov_with_covariate_and_multiple_params_threads_occasion_index(self) -> None:
        """IOV on CL and V combined with a covariate on CL: occasion indexing
        must not collide with eta/covariate expression building."""
        code = emit_stan(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="diagonal"),  # type: ignore[list-item]
                    IOV(params=["CL", "V"], occasions=OccasionByStudy()),  # type: ignore[list-item]
                ],
                covariates=[
                    CovariateLink(  # type: ignore[call-arg]
                        param="CL", covariate="WT", form="power", ref=70.0, theta=0.75
                    )
                ],
            )
        )
        assert (
            "CL_i[occ_k] = exp(log_CL + omega_CL * eta_raw[i, 1]"
            " + beta_CL_WT * log(WT[i] / 70.0)"
            " + omega_iov_CL * eta_iov_raw[(i - 1) * N_occ + occ_k, 1]);" in code
        )
        assert (
            "V_i[occ_k] = exp(log_V + omega_V * eta_raw[i, 2]"
            " + omega_iov_V * eta_iov_raw[(i - 1) * N_occ + occ_k, 2]);" in code
        )
        assert "theta_i[3] = CL_i[occ[n]];" in code
        assert "theta_i[2] = V_i[occ[n]];" in code
        _assert_no_unused_declarations(code, ["omega_iov_CL", "omega_iov_V", "eta_iov_raw"])


# ---------------------------------------------------------------------------
# LKJ / corr_iiv (correlated IIV deferred; prior must not be silently dropped)
# ---------------------------------------------------------------------------


class TestLKJCorrIIVRejection:
    def test_lkj_prior_on_corr_iiv_raises_dedicated_error(self) -> None:
        spec = _make_spec(
            variability=[IIV(params=["CL", "V"], structure="diagonal")],  # type: ignore[list-item]
            priors=[build_prior_spec(target="corr_iiv", family=LKJPrior(eta=2.0))],
        )
        with pytest.raises(NotImplementedError, match=r"[Ll]kj|corr_iiv"):
            emit_stan(spec)

    def test_lkj_error_message_names_corr_iiv_specifically(self) -> None:
        """The rejection must not be confused with the generic block-IIV message."""
        spec = _make_spec(
            variability=[IIV(params=["CL", "V"], structure="diagonal")],  # type: ignore[list-item]
            priors=[build_prior_spec(target="corr_iiv", family=LKJPrior(eta=2.0))],
        )
        with pytest.raises(NotImplementedError) as excinfo:
            emit_stan(spec)
        assert "corr_iiv" in str(excinfo.value)

    def test_dead_lkj_corr_codegen_path_is_gone(self) -> None:
        """No emitted program may contain a non-Cholesky lkj_corr( call —
        the old centered-parameterization branch was unreachable dead code
        and has been removed; full correlated-IIV support (Cholesky-factor
        lkj_corr_cholesky) is a deferred follow-up, not emitted here either."""
        specs = [
            _make_spec(),
            _make_spec(variability=[IIV(params=["CL", "V"], structure="diagonal")]),  # type: ignore[list-item]
            _make_spec(
                variability=[IOV(params=["CL"], occasions=OccasionByStudy())]  # type: ignore[list-item]
            ),
            _make_spec(observation=BLQM3(loq_value=0.1)),
        ]
        for spec in specs:
            code = emit_stan(spec)
            assert "lkj_corr(" not in code
            assert "lkj_corr_cholesky(" not in code


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
        # #1: log_lik now uses normal_lpdf with proportional variance so
        # the LOO/WAIC decomposition matches the sampling statement.
        code = emit_stan(_make_spec())
        assert "log_lik" in code
        assert "normal_lpdf(dv[n] | f[n], sigma_prop * f[n])" in code


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
        ("2cmt_oral_linear", _make_spec(distribution=TwoCmt())),
        ("1cmt_mm_elim", _make_spec(elimination=MichaelisMenten())),
        ("1cmt_parallel_mm", _make_spec(elimination=ParallelLinearMM())),
        ("1cmt_lagged", _make_spec(absorption=LaggedFirstOrder())),
        ("1cmt_transit", _make_spec(absorption=Transit(n=3))),
        (
            "1cmt_covariate",
            _make_spec(
                variability=[
                    IIV(params=["CL"], structure="diagonal"),  # type: ignore[list-item]
                ],
                covariates=[
                    CovariateLink(  # type: ignore[list-item]
                        param="CL", covariate="WT", form="power", theta=0.75, ref=70.0
                    ),
                ],
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
        spec = _make_spec(absorption=ZeroOrder())
        with pytest.raises(NotImplementedError, match="ZeroOrder"):
            emit_stan(spec)

    def test_mixed_first_zero_raises(self) -> None:
        spec = _make_spec(absorption=MixedFirstZero())
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
            distribution=OneCmt(),
            elimination=MichaelisMenten(),
        )
        code = emit_stan(spec)
        assert "real depot = y[1];" not in code, (
            "IVBolus must not alias y[1] as depot — dose enters central directly"
        )

    def test_ivbolus_onecmt_mm_has_no_absorption_term(self) -> None:
        """IVBolus + OneCmt + MM: no -ka*depot term in ODE."""
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=OneCmt(),
            elimination=MichaelisMenten(),
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
            distribution=OneCmt(),
            elimination=MichaelisMenten(),
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
            distribution=TwoCmt(),
            elimination=MichaelisMenten(),
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
            distribution=OneCmt(),
            elimination=LinearElim(),
        )
        code = emit_stan(spec)
        assert "real depot = y[1];" not in code
        assert not re.search(r"\bka\b", code), "IVBolus + Linear must not reference ka anywhere"

    def test_ivbolus_parallel_mm_has_no_absorption(self) -> None:
        """IVBolus + ParallelLinearMM: still no depot, no ka."""
        spec = _make_spec(
            absorption=IVBolus(),
            distribution=OneCmt(),
            elimination=ParallelLinearMM(),
        )
        code = emit_stan(spec)
        assert "real depot = y[1];" not in code
        assert not re.search(r"\bka\b", code)

    def test_oral_firstorder_still_has_depot(self) -> None:
        """Control: oral FirstOrder absorption still emits a depot compartment."""
        spec = _make_spec(
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=MichaelisMenten(),
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
            covariates=[
                CovariateLink(  # type: ignore[list-item]
                    param="CL", covariate="data", form="power", theta=0.75, ref=70.0
                )
            ]
        )
        with pytest.raises(ValueError, match="reserved"):
            emit_stan(spec)

    def test_rejects_double_underscore_suffix_at_emit(self) -> None:
        spec = _make_spec(
            variability=[IIV(params=["CL__"], structure="diagonal")]  # type: ignore[list-item]
        )
        with pytest.raises(ValueError, match="double underscore"):
            emit_stan(spec)


# ---------------------------------------------------------------------------
# Golden master (fast, no cmdstan)
# ---------------------------------------------------------------------------


class TestStanGoldenMasterNoIOV:
    """Pins the IOV-free codegen path, which shares `has_iov`/`iov_params`/
    `_theta_ref` machinery with the IOV path in `_emit_transformed_parameters_block`
    and `_emit_ode_solve`; this snapshot fails fast if a change to that shared
    machinery alters output for specs with no IOV."""

    def test_1cmt_oral_linear_no_iov(self, snapshot: SnapshotAssertion) -> None:
        code = emit_stan(_make_spec())
        assert code == snapshot


# ---------------------------------------------------------------------------
# cmdstanpy compile smoke tests
# ---------------------------------------------------------------------------
#
# The fast structural/string assertions above cannot catch stanc's
# type/dimension checks (e.g. array-vs-real indexing mismatches, unbalanced
# braces, malformed builtin-function calls). This is a small, deliberately
# sampled set of real compiles covering the highest-risk IOV code paths;
# it is not a substitute for the fast tests, which remain the bulk of
# coverage.


def _cmdstan_available() -> bool:
    if importlib.util.find_spec("cmdstanpy") is None:
        return False
    import cmdstanpy

    try:
        cmdstanpy.cmdstan_path()
    except Exception:
        return False
    return True


def _compile_stan(code: str, tmp_path: Path, name: str) -> None:
    import cmdstanpy

    stan_file = tmp_path / f"{name}.stan"
    stan_file.write_text(code)
    cmdstanpy.CmdStanModel(stan_file=str(stan_file), force_compile=True)


@pytest.mark.slow
@pytest.mark.skipif(
    not _cmdstan_available(),
    reason="cmdstanpy/cmdstan toolchain not installed",
)
class TestStanCompiles:
    def test_baseline_no_iov_no_block_iiv_compiles(self, tmp_path: Path) -> None:
        """Regression control: plain analytical 1-cmt oral model."""
        code = emit_stan(_make_spec())
        _compile_stan(code, tmp_path, "baseline")

    def test_iov_on_single_structural_param_compiles(self, tmp_path: Path) -> None:
        code = emit_stan(
            _make_spec(
                variability=[IOV(params=["CL"], occasions=OccasionByStudy())]  # type: ignore[list-item]
            )
        )
        _compile_stan(code, tmp_path, "iov_single")

    def test_iov_multiple_params_with_covariate_compiles(self, tmp_path: Path) -> None:
        code = emit_stan(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="diagonal"),  # type: ignore[list-item]
                    IOV(params=["CL", "V"], occasions=OccasionByStudy()),  # type: ignore[list-item]
                ],
                covariates=[
                    CovariateLink(  # type: ignore[call-arg]
                        param="CL", covariate="WT", form="power", ref=70.0, theta=0.75
                    )
                ],
            )
        )
        _compile_stan(code, tmp_path, "iov_multi_covariate")

    def test_iov_with_tmdd_qss_direct_reference_path_compiles(self, tmp_path: Path) -> None:
        """Highest-risk combination: TMDDQSS reads its volume/KD directly
        (not via the packed theta array), so IOV on KD exercises the
        `_theta_ref` substitution at a call site distinct from theta
        packing — the path most likely to break silently."""
        code = emit_stan(
            _make_spec(
                absorption=IVBolus(),
                distribution=TMDDQSS(),
                variability=[IOV(params=["KD"], occasions=OccasionByStudy())],  # type: ignore[list-item]
            )
        )
        _compile_stan(code, tmp_path, "iov_tmdd_qss")
