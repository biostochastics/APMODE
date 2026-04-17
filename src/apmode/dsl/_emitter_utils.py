# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared helpers used by multiple DSL emitters (nlmixr2, Stan, FREM).

Any change in the "does this spec need an ODE?" decision must happen in
exactly one place: ``needs_ode`` below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmode.dsl.ast_models import (
    TMDDQSS,
    MichaelisMenten,
    MixedFirstZero,
    ParallelLinearMM,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    ZeroOrder,
)

if TYPE_CHECKING:
    from apmode.dsl.ast_models import DSLSpec


def needs_ode(spec: DSLSpec) -> bool:
    """Whether the spec requires ODE-form emission (vs analytical/linCmt).

    ODE is needed when elimination is non-linear (MM/parallel/time-varying),
    when distribution is TMDD, or when absorption uses transit/mixed/zero-order.
    Otherwise both emitters can use the analytical (linCmt / closed-form)
    path for better numerical stability and speed.
    """
    if isinstance(spec.elimination, (MichaelisMenten, ParallelLinearMM, TimeVaryingElim)):
        return True
    if isinstance(spec.distribution, (TMDDCore, TMDDQSS)):
        return True
    return isinstance(spec.absorption, (Transit, MixedFirstZero, ZeroOrder))
