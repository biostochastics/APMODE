# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for nlmixr2 lowering emitter (ARCHITECTURE.md §2.2).

DSL AST → R code strings for nlmixr2. Tests verify that the emitter produces
valid nlmixr2 model code for all classical module combinations.
NODE modules are Phase 2 and should raise NotImplementedError.
"""

import pytest

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
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
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2


def _make_spec(**overrides: object) -> DSLSpec:
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


class TestEmitModelFunction:
    """The emitter should produce a complete nlmixr2 model function."""

    def test_returns_string(self) -> None:
        r_code = emit_nlmixr2(_make_spec())
        assert isinstance(r_code, str)

    def test_contains_function_wrapper(self) -> None:
        r_code = emit_nlmixr2(_make_spec())
        assert "function()" in r_code
        assert "model({" in r_code

    def test_contains_ini_block(self) -> None:
        r_code = emit_nlmixr2(_make_spec())
        assert "ini({" in r_code

    def test_contains_model_block(self) -> None:
        r_code = emit_nlmixr2(_make_spec())
        assert "model({" in r_code


class TestAbsorptionEmission:
    """Test R code emission for each absorption module type."""

    def test_first_order(self) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=FirstOrder(ka=1.0)))
        assert "ka" in r_code
        # linCmt() handles linear 1-cmt models; ODE form uses depot explicitly
        assert "linCmt" in r_code or "depot" in r_code

    def test_zero_order(self) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=ZeroOrder(dur=0.5)))
        assert "dur" in r_code

    def test_lagged_first_order(self) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=LaggedFirstOrder(ka=1.5, tlag=0.3)))
        assert "ka" in r_code
        assert "tlag" in r_code
        assert "alag" in r_code.lower() or "tlag" in r_code

    def test_transit(self) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=Transit(n=4, ktr=2.0, ka=1.0)))
        assert "ktr" in r_code
        assert "mtt" in r_code  # rxode2 transit() uses mean transit time
        assert "transit" in r_code.lower()

    def test_mixed_first_zero(self) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=0.6)))
        assert "ka" in r_code
        assert "dur" in r_code
        assert "frac" in r_code or "F1" in r_code


class TestDistributionEmission:
    def test_one_cmt(self) -> None:
        r_code = emit_nlmixr2(_make_spec(distribution=OneCmt(V=70.0)))
        assert "V" in r_code
        # Should define central compartment concentration
        assert "central" in r_code.lower() or "cp" in r_code.lower()

    def test_two_cmt(self) -> None:
        r_code = emit_nlmixr2(_make_spec(distribution=TwoCmt(V1=10.0, V2=20.0, Q=3.0)))
        assert "V1" in r_code
        assert "V2" in r_code
        assert "Q" in r_code

    def test_three_cmt(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(distribution=ThreeCmt(V1=10.0, V2=20.0, V3=5.0, Q2=3.0, Q3=1.0))
        )
        assert "V1" in r_code
        assert "V2" in r_code
        assert "V3" in r_code
        assert "Q2" in r_code
        assert "Q3" in r_code


class TestEliminationEmission:
    def test_linear(self) -> None:
        r_code = emit_nlmixr2(_make_spec(elimination=LinearElim(CL=5.0)))
        assert "CL" in r_code

    def test_michaelis_menten(self) -> None:
        r_code = emit_nlmixr2(_make_spec(elimination=MichaelisMenten(Vmax=100.0, Km=10.0)))
        assert "Vmax" in r_code
        assert "Km" in r_code

    def test_parallel_linear_mm(self) -> None:
        r_code = emit_nlmixr2(_make_spec(elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0)))
        assert "CL" in r_code
        assert "Vmax" in r_code
        assert "Km" in r_code

    def test_time_varying(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(elimination=TimeVaryingElim(CL=5.0, decay_fn="exponential"))
        )
        assert "CL" in r_code


class TestObservationEmission:
    def test_proportional(self) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=Proportional(sigma_prop=0.1)))
        assert "prop" in r_code.lower() or "sigma_prop" in r_code

    def test_additive(self) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=Additive(sigma_add=1.0)))
        assert "add" in r_code.lower() or "sigma_add" in r_code

    def test_combined(self) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=Combined(sigma_prop=0.1, sigma_add=0.5)))
        # Should have both error model components
        assert "prop" in r_code.lower() or "sigma_prop" in r_code
        assert "add" in r_code.lower() or "sigma_add" in r_code

    def test_blq_m3(self) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=BLQM3(loq_value=0.1)))
        # BLQ M3: censoring is data-driven (CENS column), not model syntax
        assert "LLOQ" in r_code or "M3" in r_code or "CENS" in r_code or "0.1" in r_code

    def test_blq_m4(self) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=BLQM4(loq_value=0.5)))
        # BLQ M4: censoring with LIMIT column
        assert "LLOQ" in r_code or "M4" in r_code or "LIMIT" in r_code or "0.5" in r_code


class TestVariabilityEmission:
    def test_iiv_diagonal(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(variability=[IIV(params=["CL", "V"], structure="diagonal")])
        )
        assert "eta.CL" in r_code or "eta_CL" in r_code or "eta.cl" in r_code.lower()
        assert "eta.V" in r_code or "eta_V" in r_code or "eta.v" in r_code.lower()

    def test_iiv_block(self) -> None:
        r_code = emit_nlmixr2(_make_spec(variability=[IIV(params=["CL", "V"], structure="block")]))
        # nlmixr2 uses block() or + for correlated etas
        block_present = "block" in r_code.lower() or "+" in r_code
        assert block_present

    def test_iov(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(variability=[IOV(params=["CL"], occasions=OccasionByStudy())])
        )
        assert "occ" in r_code.lower() or "iov" in r_code.lower()

    def test_covariate_link_power(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                variability=[
                    IIV(params=["CL"], structure="diagonal"),
                    CovariateLink(param="CL", covariate="WT", form="power"),
                ]
            )
        )
        assert "WT" in r_code

    def test_covariate_link_exponential(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                variability=[
                    IIV(params=["CL"], structure="diagonal"),
                    CovariateLink(param="CL", covariate="CRCL", form="exponential"),
                ]
            )
        )
        assert "CRCL" in r_code

    def test_covariate_link_linear(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                variability=[
                    IIV(params=["V"], structure="diagonal"),
                    CovariateLink(param="V", covariate="AGE", form="linear"),
                ]
            )
        )
        assert "AGE" in r_code

    def test_multi_variability(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="block"),
                    IOV(params=["CL"], occasions=OccasionByStudy()),
                    CovariateLink(param="CL", covariate="WT", form="power"),
                ]
            )
        )
        assert "WT" in r_code


class TestInitialEstimates:
    """The ini() block should use parameter values from the DSL as initial estimates."""

    def test_ka_initial_estimate(self) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=FirstOrder(ka=1.5)))
        # Should contain ka <- 1.5 or similar in ini block
        assert "1.5" in r_code

    def test_cl_initial_estimate(self) -> None:
        r_code = emit_nlmixr2(_make_spec(elimination=LinearElim(CL=3.7)))
        assert "3.7" in r_code

    def test_v_initial_estimate(self) -> None:
        r_code = emit_nlmixr2(_make_spec(distribution=OneCmt(V=42.0)))
        assert "42" in r_code


class TestInitialEstimateOverrides:
    """emit_nlmixr2 with initial_estimates overrides DSLSpec values in ini block."""

    def test_full_override(self) -> None:
        spec = _make_spec()
        r_code = emit_nlmixr2(spec, initial_estimates={"CL": 8.0, "V": 50.0, "ka": 2.5})
        assert "log(8.0)" in r_code
        assert "log(50.0)" in r_code
        assert "log(2.5)" in r_code

    def test_partial_override(self) -> None:
        spec = _make_spec()
        r_code = emit_nlmixr2(spec, initial_estimates={"CL": 8.0})
        assert "log(8.0)" in r_code  # CL overridden
        assert "log(70.0)" in r_code  # V from spec
        assert "log(1.0)" in r_code  # ka from spec

    def test_none_means_spec_values(self) -> None:
        spec = _make_spec()
        r_code_default = emit_nlmixr2(spec)
        r_code_none = emit_nlmixr2(spec, initial_estimates=None)
        assert r_code_default == r_code_none

    def test_two_cmt_override(self) -> None:
        spec = _make_spec(distribution=TwoCmt(V1=50.0, V2=80.0, Q=10.0))
        r_code = emit_nlmixr2(spec, initial_estimates={"V1": 60.0, "Q": 15.0})
        assert "log(60.0)" in r_code  # V1 overridden
        assert "log(80.0)" in r_code  # V2 from spec
        assert "log(15.0)" in r_code  # Q overridden

    def test_mm_elim_override(self) -> None:
        spec = _make_spec(elimination=MichaelisMenten(Vmax=100.0, Km=10.0))
        r_code = emit_nlmixr2(spec, initial_estimates={"Vmax": 200.0})
        assert "log(200.0)" in r_code
        assert "log(10.0)" in r_code  # Km from spec


class TestTMDDEmission:
    def test_tmdd_core(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                distribution=TMDDCore(V=50.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05),
            )
        )
        assert "R0" in r_code
        assert "kon" in r_code
        assert "koff" in r_code
        assert "kint" in r_code
        # Mager & Jusko 2001: ksyn = kdeg * R0
        assert "ksyn" in r_code
        assert "kdeg" in r_code

    def test_tmdd_qss(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                distribution=TMDDQSS(V=50.0, R0=10.0, KD=0.5, kint=0.05),
            )
        )
        assert "R0" in r_code
        assert "KD" in r_code
        assert "kint" in r_code
        # Gibiansky et al. 2008: QSS uses KSS
        assert "KSS" in r_code or "Cfree" in r_code


class TestNODENotSupported:
    """NODE modules are Phase 2 — nlmixr2 emitter should refuse them."""

    def test_node_absorption_raises(self) -> None:
        spec = _make_spec(
            absorption=NODEAbsorption(dim=4, constraint_template="monotone_increasing"),
        )
        with pytest.raises(NotImplementedError, match="NODE"):
            emit_nlmixr2(spec)

    def test_node_elimination_raises(self) -> None:
        spec = _make_spec(
            elimination=NODEElimination(dim=4, constraint_template="bounded_positive"),
        )
        with pytest.raises(NotImplementedError, match="NODE"):
            emit_nlmixr2(spec)


class TestBLQComposition:
    """BLQ M3/M4 should compose with user-specified residual error models."""

    def test_blq_m3_default_proportional(self) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=BLQM3(loq_value=0.5)))
        assert "prop.sd <- 0.1" in r_code
        assert "cp ~ prop(prop.sd)" in r_code

    def test_blq_m3_additive(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(observation=BLQM3(loq_value=0.5, error_model="additive", sigma_add=1.0))
        )
        assert "add.sd <- 1.0" in r_code
        assert "cp ~ add(add.sd)" in r_code
        assert "prop.sd" not in r_code

    def test_blq_m3_combined(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                observation=BLQM3(
                    loq_value=0.5,
                    error_model="combined",
                    sigma_prop=0.2,
                    sigma_add=0.8,
                )
            )
        )
        assert "prop.sd <- 0.2" in r_code
        assert "add.sd <- 0.8" in r_code
        assert "cp ~ prop(prop.sd) + add(add.sd)" in r_code

    def test_blq_m4_proportional_custom_sigma(self) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=BLQM4(loq_value=1.0, sigma_prop=0.25)))
        assert "prop.sd <- 0.25" in r_code

    def test_blq_m4_combined(self) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                observation=BLQM4(
                    loq_value=1.0,
                    error_model="combined",
                    sigma_prop=0.15,
                    sigma_add=0.5,
                )
            )
        )
        assert "prop.sd <- 0.15" in r_code
        assert "add.sd <- 0.5" in r_code
        assert "BLQ M4" in r_code


class TestRoundtrip:
    """A compiled DSL spec should produce R code that contains all parameter names."""

    def test_all_structural_params_present(self) -> None:
        spec = _make_spec(
            absorption=LaggedFirstOrder(ka=1.5, tlag=0.3),
            distribution=TwoCmt(V1=10.0, V2=20.0, Q=3.0),
            elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0),
        )
        r_code = emit_nlmixr2(spec)
        for param in spec.structural_param_names():
            assert param in r_code, f"Parameter {param} not found in R code"
