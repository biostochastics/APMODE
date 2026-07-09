# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for apmode.dsl.equations (Formular sharpening plan §4 Phase 2, P2.3).

Covers:
- One test per absorption variant against a OneCmt+LinearElim base
- TMDDCore / TMDDQSS: elimination module is ignored (no Vmax/Km leak)
- SumIG: algebraic (not differential) equation for the input rate
- ZeroOrder: notes mention the infusion-duration mechanism, no fake depot ODE
- TimeVaryingElim: the three decay_fn variants render distinctly
- NODE modules: NotImplementedError, mirroring the nlmixr2 emitter's gate
"""

from __future__ import annotations

import pytest
import sympy

from apmode.dsl.ast_models import (
    TMDDQSS,
    DSLSpec,
    Erlang,
    ExperimentalFlags,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    OneCmt,
    ParallelFirstOrder,
    Proportional,
    SumIG,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.equations import build_equations, render_equations


def _base_spec(absorption: object, *, distribution: object | None = None) -> DSLSpec:
    initial = {"V": 10.0, "CL": 5.0}
    for name, default in (
        ("ka", 1.0),
        ("dur", 2.0),
        ("tlag", 0.5),
        ("ktr", 1.0),
        ("frac", 0.5),
        ("ka1", 1.5),
        ("ka2", 0.5),
        ("MT_1", 1.0),
        ("MT_2", 2.0),
        ("RD2_1", 0.1),
        ("RD2_2", 0.1),
        ("weight_1", 0.5),
    ):
        initial[name] = default
    return DSLSpec(
        model_id="test",
        absorption=absorption,
        distribution=distribution if distribution is not None else OneCmt(),
        elimination=LinearElim(),
        variability=[],
        observation=Proportional(sigma_prop=0.1),
        initial=initial,
    )


# ---------------------------------------------------------------------------
# One test per absorption variant (OneCmt + LinearElim base)
# ---------------------------------------------------------------------------


def test_ivbolus_no_depot_ode() -> None:
    system = build_equations(_base_spec(IVBolus()))
    # Only the central-compartment ODE; no depot state anywhere.
    assert len(system.odes) == 1
    assert not any("depot" in str(eq) for eq in system.odes)
    assert any("IVBolus" in note for note in system.notes)
    assert system.observation_eq is not None


def test_first_order_has_depot_and_central_odes() -> None:
    system = build_equations(_base_spec(FirstOrder()))
    assert len(system.odes) == 2
    rendered = render_equations(system)
    assert "ka" in rendered
    assert "depot" in rendered
    assert "centr" in rendered
    assert not system.algebraic


def test_zero_order_no_depot_ode_but_has_note() -> None:
    system = build_equations(_base_spec(ZeroOrder()))
    # Only the central-compartment ODE — no synthetic depot ODE.
    assert len(system.odes) == 1
    assert not any("depot" in str(eq) for eq in system.odes)
    assert any("infusion" in note.lower() and "dur" in note for note in system.notes)


def test_lagged_first_order_has_alag_note_and_depot_ode() -> None:
    system = build_equations(_base_spec(LaggedFirstOrder()))
    assert len(system.odes) == 2
    assert any("alag" in note for note in system.notes)


def test_transit_has_opaque_transit_function_and_note() -> None:
    system = build_equations(_base_spec(Transit(n=3)))
    # One depot ODE plus one central ODE (Transit chain is rxode2's builtin,
    # not an explicit multi-compartment chain).
    assert len(system.odes) == 2
    rendered = render_equations(system)
    assert "transit" in rendered
    assert any("gamma-interpolated" in note for note in system.notes)


def test_mixed_first_zero_has_two_depot_odes() -> None:
    system = build_equations(_base_spec(MixedFirstZero()))
    assert len(system.odes) == 3  # depot_fo, depot_zo, centr
    rendered = render_equations(system)
    assert "depot_fo" in rendered
    assert "depot_zo" in rendered


def test_erlang_chain_length_matches_n() -> None:
    system = build_equations(_base_spec(Erlang(n=4)))
    # 4 chain compartments + 1 central
    assert len(system.odes) == 5
    all_odes = " ".join(str(eq) for eq in system.odes)
    for i in range(1, 5):
        assert f"E{i}" in all_odes
    # render_equations still runs cleanly (pretty-printer subscripts E1..E4).
    assert render_equations(system).strip()


def test_parallel_first_order_has_two_depot_odes() -> None:
    system = build_equations(_base_spec(ParallelFirstOrder()))
    assert len(system.odes) == 3  # depot_fast, depot_slow, centr
    rendered = render_equations(system)
    assert "depot_fast" in rendered
    assert "depot_slow" in rendered


def test_sumig_produces_algebraic_not_differential_equation() -> None:
    system = build_equations(_base_spec(SumIG(k=2)))
    # No differential equation for the SumIG compartment itself — only
    # the central-compartment ODE.
    assert len(system.odes) == 1
    assert len(system.algebraic) == 1
    rendered = render_equations(system)
    assert "I(t)" in rendered
    assert any("closed-form" in note for note in system.notes)


def test_node_absorption_raises_not_implemented() -> None:
    spec = _base_spec(
        NODEAbsorption(dim=2, constraint_template="bounded_positive"),
    )
    spec = spec.model_copy(update={"experimental": ExperimentalFlags(node=True)})
    with pytest.raises(NotImplementedError):
        build_equations(spec)


# ---------------------------------------------------------------------------
# TMDD: elimination module is ignored
# ---------------------------------------------------------------------------


def test_tmdd_core_ignores_elimination_module() -> None:
    spec = DSLSpec(
        model_id="tmdd_core",
        absorption=FirstOrder(),
        distribution=TMDDCore(),
        elimination=MichaelisMenten(),  # deliberately incompatible per validator
        variability=[],
        observation=Proportional(sigma_prop=0.1),
        initial={
            "ka": 1.0,
            "V": 10.0,
            "R0": 1.0,
            "kon": 0.1,
            "koff": 0.01,
            "kint": 0.05,
            "Vmax": 10.0,
            "Km": 1.0,
        },
    )
    system = build_equations(spec)
    rendered = render_equations(system)
    assert "Vmax" not in rendered
    assert "Km" not in rendered
    assert "kel" in rendered
    assert any("ignores spec.elimination" in note for note in system.notes)
    assert any("MichaelisMenten" in note for note in system.notes)


def test_tmdd_qss_ignores_elimination_module() -> None:
    spec = DSLSpec(
        model_id="tmdd_qss",
        absorption=FirstOrder(),
        distribution=TMDDQSS(),
        elimination=MichaelisMenten(),  # deliberately incompatible per validator
        variability=[],
        observation=Proportional(sigma_prop=0.1),
        initial={
            "ka": 1.0,
            "V": 10.0,
            "R0": 1.0,
            "KD": 1.0,
            "kint": 0.05,
            "Vmax": 10.0,
            "Km": 1.0,
        },
    )
    system = build_equations(spec)
    rendered = render_equations(system)
    assert "Vmax" not in rendered
    assert "Km" not in rendered
    assert "kel" in rendered
    assert "Cfree" in rendered
    assert any("ignores spec.elimination" in note for note in system.notes)
    # Central compartment is Atot/Rtot, never "centr", for TMDD states.
    assert "Atot" in rendered
    assert "Rtot" in rendered


def test_tmdd_core_states_present() -> None:
    system = build_equations(_base_spec(FirstOrder(), distribution=TMDDCore()))
    rendered = render_equations(system)
    for state in ("centr", "R", "RC"):
        assert state in rendered
    assert system.observation_eq is not None


def test_node_elimination_raises_not_implemented() -> None:
    spec = DSLSpec(
        model_id="node_elim",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=NODEElimination(dim=2, constraint_template="saturable"),
        variability=[],
        observation=Proportional(sigma_prop=0.1),
        initial={"ka": 1.0, "V": 10.0},
        experimental=ExperimentalFlags(node=True),
    )
    with pytest.raises(NotImplementedError):
        build_equations(spec)


# ---------------------------------------------------------------------------
# TimeVaryingElim: three decay_fn variants render distinctly
# ---------------------------------------------------------------------------


def test_time_varying_elim_decay_variants_render_distinctly() -> None:
    str_by_fn: dict[str, str] = {}
    for decay_fn in ("exponential", "half_life", "linear"):
        spec = DSLSpec(
            model_id=f"tv_{decay_fn}",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=TimeVaryingElim(decay_fn=decay_fn),  # type: ignore[arg-type]
            variability=[],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "V": 10.0, "CL": 5.0, "kdecay": 0.1},
        )
        system = build_equations(spec)
        str_by_fn[decay_fn] = str(system.odes[-1])
        # render_equations still runs cleanly for every variant.
        assert render_equations(system).strip()

    assert len(set(str_by_fn.values())) == 3
    assert "exp(" in str_by_fn["exponential"]
    assert "kdecay" in str_by_fn["half_life"]
    assert "Max(" in str_by_fn["linear"]


# ---------------------------------------------------------------------------
# Distribution: TwoCmt / ThreeCmt peripheral states
# ---------------------------------------------------------------------------


def test_two_cmt_has_peripheral_state() -> None:
    system = build_equations(_base_spec(FirstOrder(), distribution=TwoCmt()))
    all_odes = " ".join(str(eq) for eq in system.odes)
    assert "periph" in all_odes
    assert "Q" in all_odes


def test_three_cmt_has_two_peripheral_states() -> None:
    system = build_equations(_base_spec(FirstOrder(), distribution=ThreeCmt()))
    all_odes = " ".join(str(eq) for eq in system.odes)
    assert "periph1" in all_odes
    assert "periph2" in all_odes


# ---------------------------------------------------------------------------
# Rendering sanity
# ---------------------------------------------------------------------------


def test_render_equations_returns_nonempty_string_with_sections() -> None:
    system = build_equations(_base_spec(FirstOrder()))
    rendered = render_equations(system)
    assert "Differential equations:" in rendered
    assert "Observation / prediction:" in rendered
    assert isinstance(rendered, str)
    assert rendered.strip()


def test_odes_are_sympy_eq_instances() -> None:
    system = build_equations(_base_spec(FirstOrder()))
    for eq in system.odes:
        assert isinstance(eq, sympy.Eq)
