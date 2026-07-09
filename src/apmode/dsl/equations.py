# SPDX-License-Identifier: GPL-2.0-or-later
"""Symbolic ODE-system view over a compiled ``DSLSpec``.

This module is **non-authoritative and read-only**: it exists purely so a
human reviewing ``apmode formular explain --equations`` can see the
mathematical structure a spec compiles to. The R/Stan emitters
(:mod:`apmode.dsl.nlmixr2_emitter`, :mod:`apmode.dsl.stan_emitter`) remain
the ground truth for actual numerical execution — this is a one-way
mirror (emitters → equations.py understanding), never the reverse. A
stale or buggy ``equations.py`` cannot affect backend code generation:
nothing in the emitter modules imports from here.

Design follows the pattern used by pharmpy's ``CompartmentalSystem``: each
compartment's RHS is accumulated as a named sum of rate terms from a
structured per-module builder, then wrapped in a single
``sympy.Eq(sympy.Derivative(sympy.Function(name)(t), t), rhs)`` — never by
string-templating R/Stan source and re-parsing it with ``sympify``.

Every branch below is a line-for-line symbolic translation of
:func:`apmode.dsl.nlmixr2_emitter._emit_ode_dynamics`,
:func:`apmode.dsl.nlmixr2_emitter._elimination_rate_expr`,
:func:`apmode.dsl.nlmixr2_emitter._emit_tmdd_core_odes`, and
:func:`apmode.dsl.nlmixr2_emitter._emit_tmdd_qss_odes` — same compartment
names (``depot``, ``centr``, ``periph``, ``periph1``, ``periph2``,
``E1..En``, ``depot_fo``, ``depot_fast``/``depot_slow``,
``Atot``, ``Rtot``, ``R``, ``RC``), same rate expressions. Non-obvious
mirroring decisions (documented at the relevant branch below):

- ``SumIG`` absorption has no differential equation of its own — its
  contribution is a closed-form analytical input rate ``I(t)`` that
  integrates to 1, represented as an ALGEBRAIC equation feeding the
  central-compartment ODE's influx term (exactly as the emitter's
  ``sumig_input`` R line does).
- ``ZeroOrder`` absorption has no ``d/dt(depot)`` either — it is an
  infusion-duration constraint (``dur(centr) <- dur``), noted rather than
  faked as an ODE.
- ``TMDDCore``/``TMDDQSS`` distribution modules completely ignore
  ``spec.elimination`` — the emitter never calls
  ``_elimination_rate_expr`` in those branches; ``kel``/``kint`` are the
  only elimination-like terms. This holds even for a spec whose
  ``elimination`` field is not ``LinearElim`` (the semantic validator
  requires ``LinearElim`` for TMDD specs via
  ``FrmCode.AST_TMDD_REQUIRES_LINEAR_ELIM``, but that is a compatibility
  gate on the *declared* module, not evidence the dynamics read Vmax/Km
  from a MichaelisMenten/ParallelLinearMM/TimeVarying module — they never
  do, regardless of what is declared).
- NODE modules (``NODEAbsorption``/``NODEElimination``) have no closed-form
  symbolic representation and raise ``NotImplementedError``, mirroring
  :func:`apmode.dsl.nlmixr2_emitter.emit_nlmixr2`'s own
  ``spec.has_node_modules()`` guard.
- Whenever :func:`apmode.dsl._emitter_utils.needs_ode` returns ``False``
  for the spec (e.g. plain ``FirstOrder``/``OneCmt``/``LinearElim``), the
  real emitter takes the analytical ``linCmt()`` shortcut
  (:func:`apmode.dsl.nlmixr2_emitter._emit_lincmt_dynamics`) and
  ``_emit_ode_dynamics`` is never called at all — no ``d/dt()`` states
  are emitted in the generated R. ``build_equations`` always synthesizes
  the equivalent explicit ODE system regardless (there is no separate
  "closed-form" rendering), and appends a note to that effect on the
  returned :class:`EquationSystem` for exactly this case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy
from sympy import Derivative, Eq, Function, Max, exp, pi, sqrt, symbols

from apmode.dsl._emitter_utils import needs_ode
from apmode.dsl.ast_models import (
    TMDDQSS,
    DSLSpec,
    Erlang,
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
    ParallelLinearMM,
    SumIG,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)

# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquationSystem:
    """Symbolic representation of a compiled spec's dynamics.

    ``odes`` holds one :class:`sympy.Eq` per differential-equation state
    (``Eq(Derivative(state(t), t), rhs)``); ``algebraic`` holds closed-form
    / non-differential relations (SumIG's input rate, TMDD's QSS
    algebraic solves, initial conditions); ``observation_eq`` is the
    primary prediction variable's defining equation (nlmixr2's ``cp``),
    or ``None`` if the spec uses a module this view cannot represent;
    ``notes`` documents non-obvious composition decisions (see module
    docstring).
    """

    odes: list[sympy.Eq] = field(default_factory=list)
    algebraic: list[sympy.Eq] = field(default_factory=list)
    observation_eq: sympy.Eq | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Symbol / state helpers
# ---------------------------------------------------------------------------

t = sympy.Symbol("t", nonnegative=True)


def _state(name: str) -> sympy.Expr:
    """Return ``name(t)`` — an undefined function of time for compartment ``name``."""
    return Function(name)(t)


def _deriv(name: str) -> sympy.Expr:
    return Derivative(_state(name), t)


def _ode(name: str, rhs: sympy.Expr) -> sympy.Eq:
    return Eq(_deriv(name), rhs)


# ---------------------------------------------------------------------------
# Elimination rate expression (mirrors ``_elimination_rate_expr`` exactly)
# ---------------------------------------------------------------------------


def _elimination_rate_expr_sym(elim_mod: object, cmt: sympy.Expr, vol: sympy.Symbol) -> sympy.Expr:
    """Symbolic translation of ``nlmixr2_emitter._elimination_rate_expr``.

    ``cmt`` is the state expression (e.g. ``centr(t)``), ``vol`` the
    volume symbol (``V`` or ``V1``) — same argument shape as the R
    helper this mirrors.
    """
    CL, Vmax, Km, kdecay = symbols("CL Vmax Km kdecay")
    if isinstance(elim_mod, LinearElim):
        return CL / vol * cmt
    if isinstance(elim_mod, MichaelisMenten):
        return Vmax * (cmt / vol) / (Km + cmt / vol)
    if isinstance(elim_mod, ParallelLinearMM):
        return CL / vol * cmt + Vmax * (cmt / vol) / (Km + cmt / vol)
    if isinstance(elim_mod, TimeVaryingElim):
        if elim_mod.decay_fn == "half_life":
            return CL / (1 + kdecay * t) / vol * cmt
        if elim_mod.decay_fn == "linear":
            return Max(CL * (1 - kdecay * t), 0) / vol * cmt
        return CL * exp(-kdecay * t) / vol * cmt
    if isinstance(elim_mod, NODEElimination):  # pragma: no cover — gated earlier
        msg = "NODEElimination has no closed-form symbolic representation"
        raise NotImplementedError(msg)
    msg = f"unhandled EliminationModule variant: {elim_mod!r}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Absorption builder
# ---------------------------------------------------------------------------


@dataclass
class _AbsorptionResult:
    odes: list[sympy.Eq]
    algebraic: list[sympy.Eq]
    influx: sympy.Expr | None
    notes: list[str]


def _build_absorption(abs_mod: object) -> _AbsorptionResult:
    """Build the absorption-compartment ODEs/algebraics and the influx term.

    ``influx`` is the expression fed into the central-compartment ODE
    (``None`` when the dose enters the central compartment through a
    non-ODE mechanism — IVBolus routing or a ZeroOrder infusion), mirroring
    the emitter's ``_abs_influx`` string, which is likewise falsy (``""``)
    in exactly those two cases.
    """
    odes: list[sympy.Eq] = []
    algebraic: list[sympy.Eq] = []
    notes: list[str] = []
    ka, ktr = symbols("ka ktr")

    if isinstance(abs_mod, IVBolus):
        notes.append(
            "IVBolus: dose enters the central compartment directly via CMT=1 "
            "event routing — no depot compartment, no absorption-compartment ODE."
        )
        return _AbsorptionResult(odes, algebraic, None, notes)

    if isinstance(abs_mod, FirstOrder):
        odes.append(_ode("depot", -ka * _state("depot")))
        return _AbsorptionResult(odes, algebraic, ka * _state("depot"), notes)

    if isinstance(abs_mod, ZeroOrder):
        notes.append(
            "ZeroOrder: zero-order infusion at rate D/dur into the central "
            "compartment for 0 < t < dur (rxode2 `dur(<cmt>) <- dur` infusion "
            "mechanism) — no explicit depot ODE, no influx term on the central "
            "compartment's ODE (the mass enters via the infusion event itself)."
        )
        return _AbsorptionResult(odes, algebraic, None, notes)

    if isinstance(abs_mod, LaggedFirstOrder):
        notes.append(
            "LaggedFirstOrder: alag(depot) = tlag delays dosing events into "
            "depot by tlag before this ODE applies — an event-time shift, not "
            "a differential term."
        )
        odes.append(_ode("depot", -ka * _state("depot")))
        return _AbsorptionResult(odes, algebraic, ka * _state("depot"), notes)

    if isinstance(abs_mod, Transit):
        n_val = sympy.Integer(abs_mod.n)
        mtt = (n_val + 1) / ktr
        transit_input = Function("transit")(n_val, mtt, t)
        notes.append(
            "Transit: `transit(n, mtt)` is rxode2's built-in gamma-interpolated "
            "transit-compartment cascade (Savic et al. 2007), not a closed-form "
            "expression this view reproduces — shown here as the opaque "
            "function transit(n, mtt, t) with mtt = (n + 1) / ktr."
        )
        odes.append(_ode("depot", transit_input - ka * _state("depot")))
        return _AbsorptionResult(odes, algebraic, ka * _state("depot"), notes)

    if isinstance(abs_mod, MixedFirstZero):
        notes.append(
            "MixedFirstZero: f(depot_fo) = frac routes the first-order "
            "fraction to the depot; f(centr) = 1 - frac with dur(centr) = dur "
            "routes the zero-order fraction directly into central as an "
            "event-level infusion, not an ODE term."
        )
        odes.append(_ode("depot_fo", -ka * _state("depot_fo")))
        influx = ka * _state("depot_fo")
        return _AbsorptionResult(odes, algebraic, influx, notes)

    if isinstance(abs_mod, Erlang):
        n = abs_mod.n
        for i in range(1, n + 1):
            if i == 1:
                odes.append(_ode(f"E{i}", -ktr * _state(f"E{i}")))
            else:
                odes.append(_ode(f"E{i}", ktr * _state(f"E{i - 1}") - ktr * _state(f"E{i}")))
        return _AbsorptionResult(odes, algebraic, ktr * _state(f"E{n}"), notes)

    if isinstance(abs_mod, ParallelFirstOrder):
        ka1, ka2 = symbols("ka1 ka2")
        notes.append(
            "ParallelFirstOrder: f(depot_fast) = frac, f(depot_slow) = 1 - frac "
            "(bioavailability split) are dosing-event-level, not ODE terms."
        )
        odes.append(_ode("depot_fast", -ka1 * _state("depot_fast")))
        odes.append(_ode("depot_slow", -ka2 * _state("depot_slow")))
        influx = ka1 * _state("depot_fast") + ka2 * _state("depot_slow")
        return _AbsorptionResult(odes, algebraic, influx, notes)

    if isinstance(abs_mod, SumIG):
        MT_1, MT_2, RD2_1, RD2_2, weight_1, SUMIG_DOSE = symbols(
            "MT_1 MT_2 RD2_1 RD2_2 weight_1 SUMIG_DOSE"
        )
        weight_2 = 1 - weight_1
        ig_1 = sqrt(RD2_1 / (2 * pi * t**3)) * exp(-RD2_1 * (t - MT_1) ** 2 / (2 * MT_1**2 * t))
        ig_2 = sqrt(RD2_2 / (2 * pi * t**3)) * exp(-RD2_2 * (t - MT_2) ** 2 / (2 * MT_2**2 * t))
        input_rate = Function("I")(t)
        algebraic.append(Eq(input_rate, weight_1 * ig_1 + weight_2 * ig_2))
        notes.append(
            "SumIG has no differential equation for its own compartment — "
            "I(t) is a closed-form Sum-of-Inverse-Gaussians input rate "
            "(Csajka 2005; Weiss & Wegner 2022) that integrates to 1 over "
            "(0, infinity); the numerical _t_safe floor-guard near t=0 "
            "(a stability detail, not a modeling choice) is omitted here. "
            "Single-dose only; multi-dose superposition is not represented "
            "by this equation view."
        )
        return _AbsorptionResult(odes, algebraic, SUMIG_DOSE * input_rate, notes)

    if isinstance(abs_mod, NODEAbsorption):  # pragma: no cover — gated earlier
        msg = "NODEAbsorption has no closed-form symbolic representation"
        raise NotImplementedError(msg)

    msg = f"unhandled AbsorptionModule variant: {abs_mod!r}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# TMDD builders (mirror ``_emit_tmdd_core_odes`` / ``_emit_tmdd_qss_odes``)
# ---------------------------------------------------------------------------


def _build_tmdd_core(
    influx: sympy.Expr | None, elim_mod: object
) -> tuple[list[sympy.Eq], list[sympy.Eq], sympy.Eq]:
    V, R0, kon, koff, kint = symbols("V R0 kon koff kint")
    centr, R, RC = _state("centr"), _state("R"), _state("RC")
    influx_expr = influx if influx is not None else sympy.Integer(0)

    L = Function("L")(t)
    elim = Function("elim")(t)
    kdeg, ksyn = symbols("kdeg ksyn")

    algebraic = [
        Eq(L, centr / V),
        Eq(elim, _elimination_rate_expr_sym(elim_mod, centr, V)),
        Eq(kdeg, koff),
        Eq(ksyn, kdeg * R0),
        Eq(R.subs(t, 0), R0),
    ]
    odes = [
        _ode("centr", influx_expr - elim - kon * L * R * V + koff * RC * V),
        _ode("R", ksyn - kdeg * R - kon * L * R + koff * RC),
        _ode("RC", kon * L * R - koff * RC - kint * RC),
    ]
    observation_eq = Eq(Function("Cp")(t), centr / V)
    return odes, algebraic, observation_eq


def _build_tmdd_qss(
    influx: sympy.Expr | None, elim_mod: object
) -> tuple[list[sympy.Eq], list[sympy.Eq], sympy.Eq]:
    V, R0, KD, kint = symbols("V R0 KD kint")
    Atot, Rtot = _state("Atot"), _state("Rtot")
    influx_expr = influx if influx is not None else sympy.Integer(0)

    KSS = sympy.Symbol("KSS")
    Ctot, Cfree, Rfree, RC = (
        Function("Ctot")(t),
        Function("Cfree")(t),
        Function("Rfree")(t),
        Function("RC")(t),
    )
    elim = Function("elim")(t)
    kdeg, ksyn = symbols("kdeg ksyn")

    algebraic = [
        Eq(KSS, KD),
        Eq(Ctot, Atot / V),
        Eq(
            Cfree,
            sympy.Rational(1, 2)
            * ((Ctot - Rtot - KSS) + sqrt((Ctot - Rtot - KSS) ** 2 + 4 * KSS * Ctot)),
        ),
        Eq(Rfree, Rtot * KSS / (KSS + Cfree)),
        Eq(RC, Ctot - Cfree),
        Eq(elim, _elimination_rate_expr_sym(elim_mod, Cfree * V, V)),
        Eq(kdeg, kint),
        Eq(ksyn, kdeg * R0),
        Eq(Atot.subs(t, 0), 0),
        Eq(Rtot.subs(t, 0), R0),
    ]
    odes = [
        _ode("Atot", influx_expr - elim - kint * RC * V),
        _ode("Rtot", ksyn - kdeg * Rfree - kint * RC),
    ]
    observation_eq = Eq(Function("Cp")(t), Cfree)
    return odes, algebraic, observation_eq


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def build_equations(spec: DSLSpec) -> EquationSystem:
    """Build the symbolic ODE system for ``spec``, mirroring the nlmixr2 emitter.

    Raises ``NotImplementedError`` for NODE modules — see module
    docstring. Never mutates or imports from any emitter module.
    """
    if spec.has_node_modules():
        msg = (
            "Symbolic equations view does not support NODE modules "
            "(no closed-form representation; NODE uses a JAX/Diffrax hybrid "
            "backend)."
        )
        raise NotImplementedError(msg)

    abs_result = _build_absorption(spec.absorption)
    odes: list[sympy.Eq] = list(abs_result.odes)
    algebraic: list[sympy.Eq] = list(abs_result.algebraic)
    notes: list[str] = list(abs_result.notes)
    influx = abs_result.influx

    if not needs_ode(spec):
        notes.append(
            "apmode.dsl._emitter_utils.needs_ode(spec) is False for this "
            "spec (linear elimination, non-TMDD distribution, closed-form "
            "absorption): the real nlmixr2_emitter output takes the "
            "analytical linCmt() shortcut (`cp <- linCmt()`, no d/dt() "
            "states at all) and never calls _emit_ode_dynamics. The "
            "explicit ODE system below is a mathematically equivalent "
            "symbolic view for inspection, not the form the backend "
            "actually integrates."
        )

    dist_mod = spec.distribution
    elim_mod = spec.elimination

    if isinstance(dist_mod, (TMDDCore, TMDDQSS)):
        notes.append(
            f"TMDD distribution ({type(dist_mod).__name__}) applies the "
            f"declared elimination module ({type(elim_mod).__name__}) to "
            "free-drug amount, matching the nlmixr2 and Stan emitters."
        )
        if isinstance(dist_mod, TMDDCore):
            tmdd_odes, tmdd_algebraic, observation_eq = _build_tmdd_core(influx, elim_mod)
        else:
            tmdd_odes, tmdd_algebraic, observation_eq = _build_tmdd_qss(influx, elim_mod)
        odes.extend(tmdd_odes)
        algebraic.extend(tmdd_algebraic)
        return EquationSystem(
            odes=odes, algebraic=algebraic, observation_eq=observation_eq, notes=notes
        )

    V, V1, V2, V3, Q, Q2, Q3 = symbols("V V1 V2 V3 Q Q2 Q3")
    centr = _state("centr")
    influx_expr = influx if influx is not None else None

    if isinstance(dist_mod, OneCmt):
        elim_expr = _elimination_rate_expr_sym(elim_mod, centr, V)
        rhs = (influx_expr - elim_expr) if influx_expr is not None else -elim_expr
        odes.append(_ode("centr", rhs))
        observation_eq = Eq(Function("Cp")(t), centr / V)
        return EquationSystem(
            odes=odes, algebraic=algebraic, observation_eq=observation_eq, notes=notes
        )

    if isinstance(dist_mod, TwoCmt):
        periph = _state("periph")
        elim_expr = _elimination_rate_expr_sym(elim_mod, centr, V1)
        inter = -Q / V1 * centr + Q / V2 * periph
        rhs = (
            (influx_expr - elim_expr + inter) if influx_expr is not None else (-elim_expr + inter)
        )
        odes.append(_ode("centr", rhs))
        odes.append(_ode("periph", Q / V1 * centr - Q / V2 * periph))
        observation_eq = Eq(Function("Cp")(t), centr / V1)
        return EquationSystem(
            odes=odes, algebraic=algebraic, observation_eq=observation_eq, notes=notes
        )

    if isinstance(dist_mod, ThreeCmt):
        periph1, periph2 = _state("periph1"), _state("periph2")
        elim_expr = _elimination_rate_expr_sym(elim_mod, centr, V1)
        inter = -Q2 / V1 * centr + Q2 / V2 * periph1 - Q3 / V1 * centr + Q3 / V3 * periph2
        rhs = (
            (influx_expr - elim_expr + inter) if influx_expr is not None else (-elim_expr + inter)
        )
        odes.append(_ode("centr", rhs))
        odes.append(_ode("periph1", Q2 / V1 * centr - Q2 / V2 * periph1))
        odes.append(_ode("periph2", Q3 / V1 * centr - Q3 / V3 * periph2))
        observation_eq = Eq(Function("Cp")(t), centr / V1)
        return EquationSystem(
            odes=odes, algebraic=algebraic, observation_eq=observation_eq, notes=notes
        )

    msg = f"unhandled DistributionModule variant: {dist_mod!r}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_equations(system: EquationSystem) -> str:
    """Render an :class:`EquationSystem` as readable terminal text via ``sympy.pretty``.

    Unicode ASCII-art (not LaTeX) — well-suited to terminal output. This is
    a display helper only; it has no bearing on the (immutable) sympy
    objects it renders.
    """
    sections: list[str] = []

    if system.odes:
        block = ["Differential equations:", ""]
        for eq in system.odes:
            block.append(sympy.pretty(eq, use_unicode=True))
            block.append("")
        sections.append("\n".join(block).rstrip())

    if system.algebraic:
        block = ["Algebraic relations:", ""]
        for eq in system.algebraic:
            block.append(sympy.pretty(eq, use_unicode=True))
            block.append("")
        sections.append("\n".join(block).rstrip())

    if system.observation_eq is not None:
        block = [
            "Observation / prediction:",
            "",
            sympy.pretty(system.observation_eq, use_unicode=True),
        ]
        sections.append("\n".join(block))

    if system.notes:
        block = ["Notes:"]
        block.extend(f"  - {note}" for note in system.notes)
        sections.append("\n".join(block))

    return "\n\n".join(sections) + "\n"


__all__ = ["EquationSystem", "build_equations", "render_equations"]
