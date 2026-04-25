# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for v0.7 SOTA absorption extension (ADR-0003).

Covers Erlang, ParallelFirstOrder, and SumIG (k=2):
- AST construction
- Grammar parsing
- Validator constraints (Erlang n cap, MT ordering, disposition gate,
  lane admissibility, k cap)
- Transform validation + apply (ConvertTransitToErlang, AddParallelRoute,
  SetSumIGComponents)
- nlmixr2 emitter lowering (smoke + content checks)
- Stan emitter rejection (Stan support deferred to v0.7.1)
- Property test: SumIG closed-form input rate integrates to ~1 on (0, ∞)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from apmode.backends.protocol import Lane

if TYPE_CHECKING:
    from lark import Lark
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    Erlang,
    FirstOrder,
    LinearElim,
    OneCmt,
    ParallelFirstOrder,
    Proportional,
    SumIG,
    Transit,
)
from apmode.dsl.grammar import load_grammar
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.dsl.priors import NormalPrior, PriorSpec
from apmode.dsl.stan_emitter import emit_stan
from apmode.dsl.transforms import (
    AddParallelRoute,
    ConvertTransitToErlang,
    SetSumIGComponents,
    apply_transform,
    validate_transform,
)
from apmode.dsl.validator import validate_dsl


@pytest.fixture
def parser() -> Lark:
    return load_grammar()


def _base_spec(absorption: object) -> DSLSpec:
    return DSLSpec(
        model_id="t",
        absorption=absorption,  # type: ignore[arg-type]
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _disposition_fixed_spec(absorption: object) -> DSLSpec:
    """Spec with fixed-external priors on CL and V (unblocks SumIG k>=2)."""
    return DSLSpec(
        model_id="t",
        absorption=absorption,  # type: ignore[arg-type]
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        priors=[
            PriorSpec(
                target="CL",
                target_kind="structural",
                family=NormalPrior(mu=math.log(5.0), sigma=0.001),
                source="fixed_external",
            ),
            PriorSpec(
                target="V",
                target_kind="structural",
                family=NormalPrior(mu=math.log(70.0), sigma=0.001),
                source="fixed_external",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# AST construction
# ---------------------------------------------------------------------------


class TestASTConstruction:
    def test_erlang_basic(self) -> None:
        e = Erlang(n=3, ktr=1.5)
        assert e.n == 3
        assert e.ktr == 1.5
        assert e.type == "Erlang"

    def test_parallel_first_order_basic(self) -> None:
        p = ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=0.6)
        assert p.ka1 == 1.5
        assert p.ka2 == 0.5
        assert p.frac == 0.6

    def test_sumig_basic(self) -> None:
        s = SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
        assert s.k == 2
        assert s.MT_1 < s.MT_2

    def test_structural_param_names_erlang(self) -> None:
        spec = _base_spec(Erlang(n=3, ktr=1.5))
        names = spec.structural_param_names()
        assert "ktr" in names
        assert "n" not in names  # n is structural-integer, not for IIV
        assert "ka" not in names  # no terminal ka

    def test_structural_param_names_parallel(self) -> None:
        spec = _base_spec(ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=0.6))
        names = spec.structural_param_names()
        assert {"ka1", "ka2", "frac"}.issubset(names)

    def test_structural_param_names_sumig(self) -> None:
        spec = _base_spec(SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6))
        names = spec.structural_param_names()
        assert {"MT_1", "MT_2", "RD2_1", "RD2_2", "weight_1"}.issubset(names)


# ---------------------------------------------------------------------------
# Grammar parsing
# ---------------------------------------------------------------------------


class TestGrammarParse:
    def test_parse_erlang(self, parser: Lark) -> None:
        src = """
        model {
            absorption: Erlang(n=3, ktr=1.5)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        assert parser.parse(src) is not None

    def test_parse_parallel_first_order(self, parser: Lark) -> None:
        src = """
        model {
            absorption: ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=0.6)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        assert parser.parse(src) is not None

    def test_parse_sumig(self, parser: Lark) -> None:
        src = """
        model {
            absorption: SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        assert parser.parse(src) is not None

    def test_parse_iv_bolus(self, parser: Lark) -> None:
        # ADR-0003: closes pre-existing gap (IVBolus was in AST but not grammar)
        src = """
        model {
            absorption: IVBolus()
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        assert parser.parse(src) is not None


# ---------------------------------------------------------------------------
# Validator constraints
# ---------------------------------------------------------------------------


class TestValidatorErlang:
    def test_n_must_be_positive_int(self) -> None:
        spec = _base_spec(Erlang(n=0, ktr=1.0))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "positive_int" for e in errors)

    def test_n_capped_at_7(self) -> None:
        spec = _base_spec(Erlang(n=8, ktr=1.0))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "erlang_max_n" for e in errors)

    def test_ktr_must_be_positive(self) -> None:
        spec = _base_spec(Erlang(n=3, ktr=0.0))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "positive" and "ktr" in e.message for e in errors)

    def test_valid_erlang_passes(self) -> None:
        spec = _base_spec(Erlang(n=3, ktr=1.5))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert errors == []

    def test_erlang_admissible_in_submission(self) -> None:
        spec = _base_spec(Erlang(n=3, ktr=1.5))
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert errors == []


class TestValidatorParallelFirstOrder:
    def test_ka1_positive(self) -> None:
        spec = _base_spec(ParallelFirstOrder(ka1=0.0, ka2=0.5, frac=0.5))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "positive" and "ka1" in e.message for e in errors)

    def test_ka2_positive(self) -> None:
        spec = _base_spec(ParallelFirstOrder(ka1=1.5, ka2=-0.1, frac=0.5))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "positive" and "ka2" in e.message for e in errors)

    def test_frac_unit_interval(self) -> None:
        spec = _base_spec(ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=1.0))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "unit_interval" for e in errors)

    def test_admissible_all_lanes(self) -> None:
        spec = _base_spec(ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=0.6))
        for lane in [Lane.SUBMISSION, Lane.DISCOVERY, Lane.OPTIMIZATION]:
            assert validate_dsl(spec, lane=lane) == []


class TestValidatorSumIG:
    def test_k_capped_at_2_in_v07(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=3, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "sumig_k_range" for e in errors)

    def test_mt_ordering_enforced(self) -> None:
        spec = _disposition_fixed_spec(
            # MT_1 >= MT_2 — label-switching guard should fire
            SumIG(k=2, MT_1=6.0, MT_2=2.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "sumig_mt_ordering" for e in errors)

    def test_disposition_fixed_required_for_k_ge_2(self) -> None:
        # No fixed-external priors → disposition gate fires
        spec = _base_spec(SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6))
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "sumig_disposition_fixed" for e in errors)

    def test_disposition_fixed_unblocks_sumig(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        # No disposition_fixed error
        assert not any(e.constraint == "sumig_disposition_fixed" for e in errors)

    def test_inadmissible_in_submission(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "lane_absorption_admissibility" for e in errors)

    def test_admissible_in_discovery_optimization(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
        )
        for lane in [Lane.DISCOVERY, Lane.OPTIMIZATION]:
            errors = validate_dsl(spec, lane=lane)
            assert errors == [], f"unexpected errors in {lane}: {errors}"

    def test_weight_unit_interval(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=1.0)
        )
        errors = validate_dsl(spec, lane=Lane.DISCOVERY)
        assert any(e.constraint == "unit_interval" for e in errors)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestConvertTransitToErlang:
    def test_validate_requires_transit(self) -> None:
        spec = _base_spec(FirstOrder(ka=1.0))
        t = ConvertTransitToErlang(n=3)
        errors = validate_transform(spec, t)
        assert errors and "Transit" in errors[0]

    def test_apply_drops_terminal_ka(self) -> None:
        spec = _base_spec(Transit(n=4, ktr=2.0, ka=1.0))
        t = ConvertTransitToErlang(n=3)
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, Erlang)
        assert new_spec.absorption.n == 3
        assert new_spec.absorption.ktr == 2.0  # ktr inherited from Transit

    def test_apply_prunes_stale_iiv_on_ka(self) -> None:
        # Transit had IIV on ka; Erlang has no ka, so IIV must be pruned
        spec = DSLSpec(
            model_id="t",
            absorption=Transit(n=4, ktr=2.0, ka=1.0),
            distribution=OneCmt(V=70.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "ka", "ktr"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        new_spec = apply_transform(spec, ConvertTransitToErlang(n=3))
        iiv = next(v for v in new_spec.variability if isinstance(v, IIV))
        assert "ka" not in iiv.params
        assert "ktr" in iiv.params
        assert "CL" in iiv.params


class TestAddParallelRoute:
    def test_validate_requires_first_order(self) -> None:
        spec = _base_spec(Transit(n=4, ktr=2.0, ka=1.0))
        t = AddParallelRoute(ka2=0.3, frac=0.6)
        errors = validate_transform(spec, t)
        assert errors and "FirstOrder" in errors[0]

    def test_apply_creates_parallel(self) -> None:
        spec = _base_spec(FirstOrder(ka=1.5))
        t = AddParallelRoute(ka2=0.3, frac=0.6)
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, ParallelFirstOrder)
        assert new_spec.absorption.ka1 == 1.5
        assert new_spec.absorption.ka2 == 0.3
        assert new_spec.absorption.frac == 0.6


class TestSetSumIGComponents:
    def test_validate_requires_sumig(self) -> None:
        spec = _base_spec(FirstOrder(ka=1.0))
        t = SetSumIGComponents(MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.5)
        errors = validate_transform(spec, t)
        assert errors and "SumIG" in errors[0]

    def test_validate_mt_ordering(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.5)
        )
        t = SetSumIGComponents(MT_1=6.0, MT_2=2.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.5)
        errors = validate_transform(spec, t)
        assert errors and "MT_1" in errors[0]

    def test_apply_updates_components(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.5)
        )
        t = SetSumIGComponents(MT_1=1.5, MT_2=8.0, RD2_1=0.3, RD2_2=1.2, weight_1=0.7)
        new_spec = apply_transform(spec, t)
        assert isinstance(new_spec.absorption, SumIG)
        assert new_spec.absorption.MT_1 == 1.5
        assert new_spec.absorption.MT_2 == 8.0
        assert new_spec.absorption.weight_1 == 0.7


# ---------------------------------------------------------------------------
# nlmixr2 emitter content checks (golden snapshots are separate)
# ---------------------------------------------------------------------------


class TestEmitterContent:
    def test_erlang_emits_explicit_chain(self) -> None:
        spec = _base_spec(Erlang(n=3, ktr=1.5))
        code = emit_nlmixr2(spec)
        # Explicit chain — not rxode2 transit() (ADR-0003 D2)
        assert "d/dt(E1)" in code
        assert "d/dt(E2)" in code
        assert "d/dt(E3)" in code
        assert "transit(" not in code
        assert "ktr * E3" in code  # influx to central from last compartment
        # No terminal ka
        assert "lka" not in code
        assert " ka " not in code

    def test_parallel_emits_two_depots(self) -> None:
        spec = _base_spec(ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=0.6))
        code = emit_nlmixr2(spec)
        assert "d/dt(depot_fast)" in code
        assert "d/dt(depot_slow)" in code
        assert "ka1 * depot_fast + ka2 * depot_slow" in code
        assert "f(depot_fast) <- frac" in code
        assert "f(depot_slow) <- 1 - frac" in code

    def test_sumig_emits_closed_form(self) -> None:
        spec = _disposition_fixed_spec(
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6)
        )
        code = emit_nlmixr2(spec)
        # Closed-form, not deconvolution macros
        assert "ig_1 <- sqrt(RD2_1" in code
        assert "ig_2 <- sqrt(RD2_2" in code
        assert "weight_1 * ig_1 + weight_2 * ig_2" in code
        # t-safety guard
        assert "_t_safe" in code
        # Dose-scaled influx
        assert "amt * sumig_input" in code
        # Positive-difference parameterisation for MT_2
        assert "delta_MT_2" in code
        assert "MT_2 <- MT_1 + delta_MT_2" in code


# ---------------------------------------------------------------------------
# Stan emitter rejects new variants (deferred to v0.7.1)
# ---------------------------------------------------------------------------


class TestStanRejection:
    @pytest.mark.parametrize(
        "absorption",
        [
            Erlang(n=3, ktr=1.5),
            ParallelFirstOrder(ka1=1.5, ka2=0.5, frac=0.6),
            SumIG(k=2, MT_1=2.0, MT_2=6.0, RD2_1=0.5, RD2_2=1.0, weight_1=0.6),
        ],
    )
    def test_stan_rejects_v07_absorption(self, absorption: object) -> None:
        spec = _base_spec(absorption)
        with pytest.raises(NotImplementedError, match="nlmixr2 backend"):
            emit_stan(spec)


# ---------------------------------------------------------------------------
# Property test: SumIG closed-form input rate integrates to 1
# ---------------------------------------------------------------------------


class TestSumIGIntegrationProperty:
    """The IG density ∫₀^∞ IG(t; MT, RD2) dt = 1 by construction.

    A weighted sum with weights summing to 1 must also integrate to 1.
    Verifies the closed-form expression we emit is the correct IG density.
    """

    @pytest.mark.parametrize(
        ("MT_1", "MT_2", "RD2_1", "RD2_2", "w1"),
        [
            (2.0, 6.0, 0.5, 1.0, 0.6),
            (1.5, 8.0, 0.3, 1.2, 0.4),
            (3.0, 12.0, 0.8, 0.6, 0.5),
        ],
    )
    def test_sumig_density_integrates_to_one(
        self,
        MT_1: float,
        MT_2: float,
        RD2_1: float,
        RD2_2: float,
        w1: float,
    ) -> None:
        # Reproduce the emitter's closed-form expression numerically.
        # Trapezoidal integration over [eps, 100*max(MT)] with 100k points
        # gives ample precision for the IG density.
        t_max = 100.0 * max(MT_1, MT_2)
        t = np.linspace(1e-6, t_max, 100_000)

        def ig(t_arr: np.ndarray, MT: float, RD2: float) -> np.ndarray:
            return np.sqrt(RD2 / (2 * np.pi * t_arr**3)) * np.exp(
                -RD2 * (t_arr - MT) ** 2 / (2 * MT**2 * t_arr)
            )

        rate = w1 * ig(t, MT_1, RD2_1) + (1 - w1) * ig(t, MT_2, RD2_2)
        integral = np.trapezoid(rate, t)
        assert math.isclose(integral, 1.0, abs_tol=0.01), (
            f"∫ sumig_input dt = {integral} (expected ≈ 1.0)"
        )

    def test_sumig_input_nonnegative(self) -> None:
        # The IG density is non-negative by construction; product with
        # non-negative weights stays non-negative.
        t = np.linspace(1e-6, 100.0, 10_000)
        MT_1, MT_2 = 2.0, 6.0
        RD2_1, RD2_2 = 0.5, 1.0
        w1 = 0.6

        def ig(t_arr: np.ndarray, MT: float, RD2: float) -> np.ndarray:
            return np.sqrt(RD2 / (2 * np.pi * t_arr**3)) * np.exp(
                -RD2 * (t_arr - MT) ** 2 / (2 * MT**2 * t_arr)
            )

        rate = w1 * ig(t, MT_1, RD2_1) + (1 - w1) * ig(t, MT_2, RD2_2)
        assert (rate >= 0).all()
