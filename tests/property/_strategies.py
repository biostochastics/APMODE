# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared Hypothesis strategies for the property test suite.

Single home for the strategies that were previously copy-pasted across
sibling property modules:

* DSL numeric strategies + the shared ``OBSERVATIONS`` template list used by
  both ``test_dsl_property.py`` and ``test_compiler_roundtrip.py``. (The
  ``ABSORPTIONS``/``DISTRIBUTIONS``/``ELIMINATIONS`` lists intentionally stay
  local to each module: the grammar-only test uses bare template strings while
  the full-pipeline test uses ``(template, param_names)`` tuples, so they are
  not the same object and must not be merged.)
* Prior ``(target, family)`` strategies shared by
  ``test_prior_spec_property.py`` and ``test_dsl_priors_grammar_property.py``.
* The LORO-CV synthetic-data generator shared by the LORO property tests.

This module is deliberately *not* ``test_``-prefixed so pytest does not try to
collect its strategy factories as test functions; the property test modules
import from it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st

from apmode.dsl.priors import (
    HalfCauchyPrior,
    HalfNormalPrior,
    LKJPrior,
    LogNormalPrior,
    NormalPrior,
    PriorFamily,
    TargetKind,
)

# --- DSL numeric strategies ---------------------------------------------------

# Shared residual-error observation templates. Both the grammar-only property
# test and the full-pipeline roundtrip test format these identically.
OBSERVATIONS: list[str] = [
    "Proportional(sigma_prop={v})",
    "Additive(sigma_add={v})",
    "Combined(sigma_prop={v}, sigma_add={v2})",
    "BLQ_M3(loq_value={v})",
    "BLQ_M4(loq_value={v})",
]


def pos_float() -> st.SearchStrategy[float]:
    """Positive calibration value in ``[0.01, 1000]`` (no NaN/inf)."""
    return st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)


def pos_int() -> st.SearchStrategy[int]:
    """Small positive integer (e.g. transit-compartment count)."""
    return st.integers(min_value=1, max_value=20)


# --- Prior strategies ---------------------------------------------------------


def prior_mu() -> st.SearchStrategy[float]:
    return st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)


def prior_pos_scale() -> st.SearchStrategy[float]:
    return st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)


@st.composite
def valid_target_and_family(draw: st.DrawFn) -> tuple[str, PriorFamily, TargetKind]:
    """Draw a ``(target, family, target_kind)`` triple valid under
    ``apmode.dsl.priors._VALID_FAMILIES``, restricted to the families the
    Formular grammar renderer supports (Normal, LogNormal, HalfNormal,
    HalfCauchy, LKJ).
    """
    kind: TargetKind = draw(
        st.sampled_from(["structural", "iiv_sd", "residual_sd", "corr_iiv", "covariate"])
    )

    family: PriorFamily
    if kind == "structural":
        target = draw(st.sampled_from(["CL", "V", "ka"]))
        family = draw(
            st.one_of(
                st.builds(NormalPrior, mu=prior_mu(), sigma=prior_pos_scale()),
                st.builds(LogNormalPrior, mu=prior_mu(), sigma=prior_pos_scale()),
            )
        )
    elif kind == "iiv_sd":
        target = "omega_CL"
        family = draw(
            st.one_of(
                st.builds(HalfNormalPrior, sigma=prior_pos_scale()),
                st.builds(HalfCauchyPrior, scale=prior_pos_scale()),
            )
        )
    elif kind == "residual_sd":
        target = draw(st.sampled_from(["sigma_prop", "sigma_add"]))
        family = draw(st.builds(HalfNormalPrior, sigma=prior_pos_scale()))
    elif kind == "corr_iiv":
        target = "corr_iiv"
        family = draw(st.builds(LKJPrior, eta=prior_pos_scale()))
    else:  # covariate
        target = "beta_CL_WT"
        family = draw(st.builds(NormalPrior, mu=prior_mu(), sigma=prior_pos_scale()))

    return target, family, kind


# --- LORO-CV data generator ---------------------------------------------------


@st.composite
def pk_data_with_regimens(
    draw: st.DrawFn,
    min_groups: int = 3,
    max_groups: int = 6,
) -> pd.DataFrame:
    """Generate synthetic canonical PK data with a variable number of unique
    dose (regimen) groups. Each subject receives exactly one dose, so the
    number of distinct ``AMT`` values among ``EVID==1`` rows equals the number
    of regimen groups ``loro_cv_splits`` will discover.
    """
    n_groups = draw(st.integers(min_value=min_groups, max_value=max_groups))
    n_per_group = draw(st.integers(min_value=2, max_value=8))
    doses = sorted(
        draw(
            st.lists(
                st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                min_size=n_groups,
                max_size=n_groups,
                unique=True,
            )
        )
    )

    rows: list[dict[str, object]] = []
    subject_id = 1
    for dose in doses:
        for _ in range(n_per_group):
            rows.append(
                {
                    "NMID": subject_id,
                    "TIME": 0.0,
                    "DV": 0.0,
                    "EVID": 1,
                    "AMT": dose,
                    "MDV": 1,
                }
            )
            for t in [1.0, 4.0, 12.0]:
                rows.append(
                    {
                        "NMID": subject_id,
                        "TIME": t,
                        "DV": float(np.random.default_rng(subject_id).lognormal(0, 1)),
                        "EVID": 0,
                        "AMT": 0.0,
                        "MDV": 0,
                    }
                )
            subject_id += 1

    return pd.DataFrame(rows)
