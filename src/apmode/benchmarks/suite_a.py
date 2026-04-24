# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite A: Synthetic Recovery scenarios (PRD §5).

Seven scenarios with known ground truth for structure/parameter recovery testing.
Each function returns a DSLSpec with realistic PK parameter values.

Scenarios:
  A1: 1-cmt oral, first-order absorption, linear elimination
  A2: 2-cmt IV, parallel linear + MM elimination
  A3: Transit absorption (n=3), 1-cmt, linear elimination
  A4: 1-cmt oral, Michaelis-Menten elimination
  A5: TMDD quasi-steady-state (SC mAb)
  A6: 1-cmt oral, allometric WT + categorical renal covariates on CL
  A7: 2-cmt, NODE nonlinear absorption (ground truth: saturable Michaelis-Menten)
  A8: 1-cmt oral, time-varying CL (diurnal) + allometric CRCL covariate
"""

from __future__ import annotations

from apmode.dsl.ast_models import (
    IIV,
    TMDDQSS,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LinearElim,
    MichaelisMenten,
    NODEAbsorption,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    Transit,
    TwoCmt,
)


def scenario_a1() -> DSLSpec:
    """A1: 1-cmt oral, first-order absorption, linear elimination.

    Simplest PK model. Tests correct structure identification
    and parameter recovery for standard oral dosing.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a1",
        absorption=FirstOrder(ka=1.5),
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
    )


def scenario_a2() -> DSLSpec:
    """A2: 2-cmt IV bolus, parallel linear + Michaelis-Menten elimination.

    Tests compartment count recovery and nonlinear clearance detection.
    IV bolus approximated via large ka (100 h^-1) since the DSL lacks a
    dedicated IV bolus module. This approximation produces near-instantaneous
    absorption but may show slight bias in very early timepoints (<0.1h).
    """
    return DSLSpec(
        model_id="suite_a_scenario_a2",
        absorption=FirstOrder(ka=100.0),  # large ka approximates IV bolus
        distribution=TwoCmt(V1=50.0, V2=80.0, Q=10.0),
        elimination=ParallelLinearMM(CL=3.0, Vmax=100.0, Km=10.0),
        variability=[IIV(params=["CL", "V1", "Vmax"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.5),
    )


def scenario_a3() -> DSLSpec:
    """A3: Transit absorption (n=3), 1-cmt, linear elimination.

    Tests transit chain detection and transit number recovery.
    Savic et al. (2007) transit compartment model.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a3",
        absorption=Transit(n=3, ktr=2.0, ka=1.0),
        distribution=OneCmt(V=60.0),
        elimination=LinearElim(CL=4.0),
        variability=[IIV(params=["CL", "V", "ktr"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.12),
    )


def scenario_a4() -> DSLSpec:
    """A4: 1-cmt oral, Michaelis-Menten elimination.

    Tests nonlinear clearance detection. At typical doses,
    Vmax/Km gives apparent CL of ~10 L/h at low concentrations,
    saturating at higher concentrations.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a4",
        absorption=FirstOrder(ka=1.2),
        distribution=OneCmt(V=65.0),
        elimination=MichaelisMenten(Vmax=80.0, Km=8.0),
        variability=[IIV(params=["Vmax", "V", "ka"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.3),
    )


def scenario_a5() -> DSLSpec:
    """A5: TMDD quasi-steady-state (subcutaneous monoclonal antibody).

    Tests TMDD identification vs. 2-compartment confusion. The QSS
    model (Gibiansky 2008) produces nonlinear PK: target-mediated
    clearance dominates at low concentrations and saturates at high
    concentrations. Classical 2-cmt can confuse the target-mediated
    disposition phase with peripheral distribution.

    Typical mAb PK: slow SC absorption, low volume, low clearance.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a5",
        absorption=FirstOrder(ka=0.02),
        distribution=TMDDQSS(V=3.5, R0=10.0, KD=1.0, kint=0.03),
        elimination=LinearElim(CL=0.015),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
    )


def scenario_a6() -> DSLSpec:
    """A6: 1-cmt oral with mixed covariate effects.

    Tests covariate structure recovery. Ground truth has:
      - Allometric scaling: CL * (WT/70)^0.75, V * (WT/70)
      - Categorical effect: CL * 0.6 for renally impaired subjects

    The DSLSpec records the covariate links; R simulation applies the
    actual covariate model to generate data with these effects embedded.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a6",
        absorption=FirstOrder(ka=1.5),
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[
            IIV(params=["CL", "V", "ka"], structure="diagonal"),
            CovariateLink(param="CL", covariate="WT", form="power"),
            CovariateLink(param="V", covariate="WT", form="power"),
            CovariateLink(param="CL", covariate="RENAL", form="categorical"),
        ],
        observation=Proportional(sigma_prop=0.12),
    )


def scenario_a7() -> DSLSpec:
    """A7: 2-cmt with NODE nonlinear absorption.

    Ground truth is saturable (Michaelis-Menten) absorption — a nonlinear
    process not representable by any classical absorption module in the DSL.
    Tests whether the hybrid NODE can recover the absorption shape, and
    whether the symbolic surrogate approximation is pharmacokinetically
    equivalent (AUC/Cmax within 80-125% GMR).

    R simulation uses Vmax_abs/Km_abs absorption; APMODE sees NODEAbsorption.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a7",
        absorption=NODEAbsorption(dim=4, constraint_template="bounded_positive"),
        distribution=TwoCmt(V1=50.0, V2=80.0, Q=10.0),
        elimination=LinearElim(CL=4.0),
        variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.3),
    )


def scenario_a8() -> DSLSpec:
    """A8: 1-cmt oral with time-varying CL and CRCL covariate.

    Ground truth in the R simulator is
    ``CL(t, CRCL) = CL0 * (CRCL / 90)^theta * exp(-delta * t / 24)``. The
    DSL captures the static allometric CRCL effect via a power CovariateLink;
    the diurnal damping (``exp(-delta * t / 24)``) is not expressible in the
    current DSL and is recorded in ``A8_COVARIATE_MODEL_NOTES`` for Suite A
    comparison measurements — i.e., fit quality vs ground truth is an
    APMODE output, not a DSL capability claim.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a8",
        absorption=FirstOrder(ka=1.822),
        distribution=OneCmt(V=29.964),
        elimination=LinearElim(CL=4.482),
        variability=[
            IIV(params=["CL", "V"], structure="diagonal"),
            CovariateLink(param="CL", covariate="CRCL", form="power"),
        ],
        observation=Proportional(sigma_prop=0.10),
    )


# Reference parameter values (ground truth for recovery testing).
# Keys match structural_param_names() for each scenario's DSLSpec.
REFERENCE_PARAMS: dict[str, dict[str, float]] = {
    "A1": {"ka": 1.5, "V": 70.0, "CL": 5.0},
    "A2": {"ka": 100.0, "V1": 50.0, "V2": 80.0, "Q": 10.0, "CL": 3.0, "Vmax": 100.0, "Km": 10.0},
    "A3": {"n": 3.0, "ktr": 2.0, "ka": 1.0, "V": 60.0, "CL": 4.0},
    "A4": {"ka": 1.2, "V": 65.0, "Vmax": 80.0, "Km": 8.0},
    "A5": {"ka": 0.02, "V": 3.5, "R0": 10.0, "KD": 1.0, "kint": 0.03, "CL": 0.015},
    "A6": {"ka": 1.5, "V": 70.0, "CL": 5.0},
    # A7: mechanistic params only (NODE absorption weights are not named structural params)
    "A7": {"V1": 50.0, "V2": 80.0, "Q": 10.0, "CL": 4.0},
    "A8": {"ka": 1.822, "V": 29.964, "CL": 4.482},
}

# Ground truth absorption parameters for A7 (not in DSLSpec structural params,
# but needed for R simulation and surrogate fidelity testing).
A7_ABSORPTION_TRUTH: dict[str, float] = {
    "Vmax_abs": 50.0,  # mg/h, saturable absorption Vmax
    "Km_abs": 20.0,  # mg, saturable absorption Km
}

# Ground truth covariate-model parameters for A8 that are not currently
# expressible as DSL primitives. ``theta_crcl`` is the static CRCL allometric
# exponent (captured in the DSL via ``CovariateLink(form="power")``), while
# ``delta_diurnal`` is a time-dependent damping rate (``exp(-delta * t / 24)``)
# with no DSL primitive today. The value is recorded here so Suite A fit
# comparisons can distinguish DSL-approximation bias from estimation error.
A8_COVARIATE_MODEL_NOTES: dict[str, float] = {
    "theta_crcl": 0.75,  # allometric exponent, CRCL/90 reference
    "delta_diurnal": 0.15,  # diurnal damping per 24 h (not in DSL today)
}

# All scenario factories for iteration
SCENARIOS: dict[str, type[None]] = {}  # populated below for type safety

ALL_SCENARIOS = [
    ("A1", scenario_a1),
    ("A2", scenario_a2),
    ("A3", scenario_a3),
    ("A4", scenario_a4),
    ("A5", scenario_a5),
    ("A6", scenario_a6),
    ("A7", scenario_a7),
    ("A8", scenario_a8),
]
