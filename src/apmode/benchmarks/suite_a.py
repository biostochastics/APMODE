# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite A: Synthetic Recovery scenarios (PRD S5).

Four scenarios with known ground truth for structure/parameter recovery testing.
Each function returns a DSLSpec with realistic PK parameter values.

Scenarios:
  A1: 1-cmt oral, first-order absorption, linear elimination
  A2: 2-cmt IV, parallel linear + MM elimination
  A3: Transit absorption (n=3), 1-cmt, linear elimination
  A4: 1-cmt oral, Michaelis-Menten elimination
"""

from __future__ import annotations

from apmode.dsl.ast_models import (
    IIV,
    Combined,
    DSLSpec,
    FirstOrder,
    LinearElim,
    MichaelisMenten,
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


# Reference parameter values (ground truth for recovery testing).
# Keys match structural_param_names() for each scenario's DSLSpec.
REFERENCE_PARAMS: dict[str, dict[str, float]] = {
    "A1": {"ka": 1.5, "V": 70.0, "CL": 5.0},
    "A2": {"ka": 100.0, "V1": 50.0, "V2": 80.0, "Q": 10.0, "CL": 3.0, "Vmax": 100.0, "Km": 10.0},
    "A3": {"n": 3.0, "ktr": 2.0, "ka": 1.0, "V": 60.0, "CL": 4.0},
    "A4": {"ka": 1.2, "V": 65.0, "Vmax": 80.0, "Km": 8.0},
}

# All scenario factories for iteration
SCENARIOS: dict[str, type[None]] = {}  # populated below for type safety

ALL_SCENARIOS = [
    ("A1", scenario_a1),
    ("A2", scenario_a2),
    ("A3", scenario_a3),
    ("A4", scenario_a4),
]
