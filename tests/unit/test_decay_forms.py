# SPDX-License-Identifier: GPL-2.0-or-later
"""All three TimeVaryingElim decay forms (plan §4 / #9).

Validator tests live in ``test_semantic_validator.py``. This module covers:
  * nlmixr2 emitter's ODE RHS for each decay form
  * Stan emitter's ODE RHS for each decay form
  * Pydantic Literal whitelist at the AST boundary

Exact math:
  exponential:  CL(t) = CL * exp(-kdecay * t)
  half_life:    CL(t) = CL / (1 + kdecay * t)          (kdecay = ln(2)/t_half)
  linear:       CL(t) = max(CL * (1 - kdecay * t), 0)  (floored at 0)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from apmode.dsl.ast_models import (
    IIV,
    Additive,
    DSLSpec,
    FirstOrder,
    OneCmt,
    TimeVaryingElim,
)
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.dsl.stan_emitter import emit_stan


def _spec(decay: str) -> DSLSpec:
    return DSLSpec(
        model_id=f"tv_{decay}",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=70.0),
        elimination=TimeVaryingElim(CL=5.0, kdecay=0.1, decay_fn=decay),  # type: ignore[arg-type]
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=Additive(sigma_add=0.5),
    )


class TestDecayFormLiterals:
    """The Pydantic Literal is the authoritative whitelist."""

    def test_exponential_accepted(self) -> None:
        TimeVaryingElim(CL=5.0, decay_fn="exponential")

    def test_half_life_accepted(self) -> None:
        TimeVaryingElim(CL=5.0, decay_fn="half_life")

    def test_linear_accepted(self) -> None:
        TimeVaryingElim(CL=5.0, decay_fn="linear")

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TimeVaryingElim(CL=5.0, decay_fn="weibull")  # type: ignore[arg-type]


class TestNlmixr2EmitterPerDecayForm:
    """Each decay form lowers to a distinct R ODE RHS expression."""

    def test_exponential_rhs(self) -> None:
        code = emit_nlmixr2(_spec("exponential"))
        # Must include the exponential form and not the other two.
        assert "CL * exp(-kdecay * t)" in code
        assert "1 + kdecay * t" not in code
        assert "max(CL * (1 - kdecay * t), 0)" not in code

    def test_half_life_rhs(self) -> None:
        code = emit_nlmixr2(_spec("half_life"))
        assert "CL / (1 + kdecay * t)" in code
        assert "exp(-kdecay * t)" not in code

    def test_linear_rhs(self) -> None:
        code = emit_nlmixr2(_spec("linear"))
        # R uses max(..., 0) as the floor — must be literally present so
        # the compiled nlmixr2 model never takes negative clearance.
        assert "max(CL * (1 - kdecay * t), 0)" in code
        assert "exp(-kdecay * t)" not in code


class TestStanEmitterPerDecayForm:
    """Each decay form lowers to a distinct Stan ODE RHS expression."""

    def test_exponential_rhs(self) -> None:
        code = emit_stan(_spec("exponential"))
        assert "CL * exp(-kdecay * t)" in code
        assert "fmax(" not in code or "fmax(CL * (1 - kdecay * t), 0.0)" not in code

    def test_half_life_rhs(self) -> None:
        code = emit_stan(_spec("half_life"))
        assert "CL / (1 + kdecay * t)" in code

    def test_linear_rhs(self) -> None:
        """Stan uses `fmax(..., 0.0)` to floor the clearance at zero."""
        code = emit_stan(_spec("linear"))
        assert "fmax(CL * (1 - kdecay * t), 0.0)" in code


class TestDecayMathematicalInvariants:
    """The three forms produce distinct decay trajectories."""

    @staticmethod
    def _evaluate(decay: str, cl: float, kdecay: float, t: float) -> float:
        import math

        if decay == "exponential":
            return cl * math.exp(-kdecay * t)
        if decay == "half_life":
            return cl / (1.0 + kdecay * t)
        if decay == "linear":
            return max(cl * (1.0 - kdecay * t), 0.0)
        raise ValueError(f"unknown decay form: {decay}")

    def test_all_forms_identical_at_t_equals_zero(self) -> None:
        cl, kdecay = 5.0, 0.1
        vals = {
            d: self._evaluate(d, cl, kdecay, 0.0) for d in ("exponential", "half_life", "linear")
        }
        assert all(v == pytest.approx(cl, rel=1e-12) for v in vals.values())

    def test_forms_diverge_at_positive_t(self) -> None:
        cl, kdecay = 5.0, 0.1
        exp_v = self._evaluate("exponential", cl, kdecay, 5.0)
        hl_v = self._evaluate("half_life", cl, kdecay, 5.0)
        lin_v = self._evaluate("linear", cl, kdecay, 5.0)
        assert exp_v != pytest.approx(hl_v, rel=1e-6)
        assert hl_v != pytest.approx(lin_v, rel=1e-6)
        # sanity: positive and bounded above by CL
        for v in (exp_v, hl_v, lin_v):
            assert 0.0 <= v <= cl

    def test_linear_floors_at_zero(self) -> None:
        """`linear` past its zero-crossing must NOT go negative."""
        cl, kdecay = 5.0, 0.5
        # zero-crossing at t = 1/kdecay = 2.0; evaluate at t=10 (well past)
        val = self._evaluate("linear", cl, kdecay, 10.0)
        assert val == pytest.approx(0.0, abs=1e-12)

    def test_half_life_never_reaches_zero(self) -> None:
        """`half_life` is asymptotic — should stay strictly positive."""
        cl, kdecay = 5.0, 0.5
        for t in (10.0, 100.0, 1000.0):
            v = self._evaluate("half_life", cl, kdecay, t)
            assert v > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
