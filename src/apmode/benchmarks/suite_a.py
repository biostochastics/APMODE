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
  A8: 1-cmt oral, monotonic autoinduction of CL + allometric CRCL covariate
"""

from __future__ import annotations

import csv
from pathlib import Path

from apmode.dsl.ast_models import (
    IIV,
    TMDDQSS,
    Combined,
    CovariateLink,
    DSLSpec,
    ExperimentalFlags,
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
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
        initial={"ka": 1.5, "V": 70.0, "CL": 5.0},
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
        absorption=FirstOrder(),  # large ka approximates IV bolus
        distribution=TwoCmt(),
        elimination=ParallelLinearMM(),
        # Q (inter-compartmental clearance) has simulated BSV (omega=0.04 in
        # reference_params.json) and must be estimable here too, or
        # eta-recovery scoring has nothing to compare it against.
        variability=[IIV(params=["CL", "V1", "Q", "Vmax"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.5),
        initial={
            "ka": 100.0,
            "V1": 50.0,
            "V2": 80.0,
            "Q": 10.0,
            "CL": 3.0,
            "Vmax": 100.0,
            "Km": 10.0,
        },
    )


def scenario_a3() -> DSLSpec:
    """A3: Transit absorption (n=3), 1-cmt, linear elimination.

    Tests transit chain detection and transit number recovery.
    Savic et al. (2007) transit compartment model.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a3",
        absorption=Transit(n=3),
        distribution=OneCmt(),
        elimination=LinearElim(),
        # ka has simulated BSV (omega=0.04 in reference_params.json)
        # alongside ktr; Savic et al. 2007 retain IIV on ka when adding
        # transit-compartment IIV rather than dropping it, and it must be
        # estimable here too or eta-recovery scoring has nothing to
        # compare it against.
        variability=[IIV(params=["CL", "V", "ktr", "ka"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.12),
        initial={"ktr": 2.0, "ka": 1.0, "V": 60.0, "CL": 4.0},
    )


def scenario_a4() -> DSLSpec:
    """A4: 1-cmt oral, Michaelis-Menten elimination.

    Tests nonlinear clearance detection. At typical doses,
    Vmax/Km gives apparent CL of ~10 L/h at low concentrations,
    saturating at higher concentrations.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a4",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=MichaelisMenten(),
        variability=[IIV(params=["Vmax", "V", "ka"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.3),
        initial={"ka": 1.2, "V": 65.0, "Vmax": 80.0, "Km": 8.0},
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
        absorption=FirstOrder(),
        distribution=TMDDQSS(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
        initial={"ka": 0.02, "V": 3.5, "R0": 10.0, "KD": 1.0, "kint": 0.03, "CL": 0.015},
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
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        covariates=[
            CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0),
            CovariateLink(param="V", covariate="WT", form="power", theta=1.0, ref=70.0),
            CovariateLink(param="CL", covariate="RENAL", form="categorical", reference="normal"),
        ],
        observation=Proportional(sigma_prop=0.12),
        initial={"ka": 1.5, "V": 70.0, "CL": 5.0},
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
        distribution=TwoCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V1"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.3),
        initial={"V1": 50.0, "V2": 80.0, "Q": 10.0, "CL": 4.0},
        # Suite A NODE scenario (Phase 0 P0.8): explicit opt-in required
        # since no emitter has a working NODE backend yet.
        experimental=ExperimentalFlags(node=True),
    )


def scenario_a8() -> DSLSpec:
    """A8: 1-cmt oral with monotonic CL autoinduction and CRCL covariate.

    Ground truth in the R simulator is
    ``CL(t, CRCL) = CL0 * (CRCL / 90)^theta * exp(-delta * t / 24)``. The
    DSL captures the static allometric CRCL effect via a power
    CovariateLink; the monotonic autoinduction term
    (``exp(-delta * t / 24)``) has no DSL primitive and is recorded in
    ``A8_COVARIATE_MODEL_NOTES``.

    Expected misspecification bias
    ------------------------------
    Because APMODE fits a static ``CL`` against a monotonically-decaying
    truth, the recovered point estimate approaches the time-average of
    ``CL(t)`` over the observation window. Over 0-48 h with
    ``CL0 = 4.482`` and ``delta = 0.15``, the analytical time-average
    ``CL0 * (24/(delta * 48)) * (1 - exp(-delta * 48/24)) = 3.872``, so
    the static-target recovery bias is ``-13.66%``. Benchmark tooling
    must compare recovery to this time-averaged target, not to the raw
    ``CL0 = 4.482``. The scenario is therefore a *DSL-capability* test
    (can APMODE detect the residual pattern as a covariate-vs-time
    misspecification?), not a pure parameter-recovery test like A1-A6.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a8",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        # ka has simulated BSV (omega=0.09 in reference_params.json) — this
        # is separate from the intentional autoinduction misspecification
        # documented above (which is about the fixed-effect CL trajectory,
        # not ka's between-subject variability) and must be estimable here
        # too or eta-recovery scoring has nothing to compare it against.
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        covariates=[
            CovariateLink(param="CL", covariate="CRCL", form="power", theta=0.75, ref=90.0),
        ],
        observation=Proportional(sigma_prop=0.10),
        initial={"ka": 1.822, "V": 29.964, "CL": 4.482},
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
# ``delta_autoind`` is a monotonic autoinduction rate (``exp(-delta * t / 24)``)
# with no DSL primitive today. The value is recorded here so Suite A fit
# comparisons can distinguish DSL-approximation bias from estimation error.
A8_COVARIATE_MODEL_NOTES: dict[str, float] = {
    "theta_crcl": 0.75,  # allometric exponent, CRCL/90 reference
    "delta_autoind": 0.15,  # autoinduction rate per 24 h (not in DSL today)
    # Expected downward bias on static-CL recovery over 0-48 h at delta=0.15:
    "time_averaged_CL_over_48h": 3.872,
    "static_target_bias_pct": -13.66,
}

# All scenario factories for iteration
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


# Filename stems produced by benchmarks/suite_a/simulate_all.R. The R
# simulator writes ``<stem>.csv`` for single-replicate runs and
# ``<stem>_repNN.csv`` when invoked with n_replicates > 1. Each CSV has a
# matching ``<stem>[_repNN]_eta.csv`` with per-subject eta draws.
SCENARIO_FILENAME_STEMS: dict[str, str] = {
    "A1": "a1_1cmt_oral_linear",
    "A2": "a2_2cmt_iv_parallel_mm",
    "A3": "a3_transit_1cmt_linear",
    "A4": "a4_1cmt_oral_mm",
    "A5": "a5_tmdd_qss",
    "A6": "a6_1cmt_covariates",
    "A7": "a7_2cmt_node_absorption",
    "A8": "a8_1cmt_autoind_covariate",
}


def scenario_dataset_paths(
    suite_dir: Path, scenario_id: str, *, include_eta: bool = False
) -> list[Path]:
    """Enumerate all dataset CSVs for a scenario, sorted by replicate index.

    Args:
        suite_dir: Directory containing the Suite A output CSVs
            (typically ``benchmarks/suite_a/``).
        scenario_id: One of ``A1``..``A8``.
        include_eta: When True, also include ``<stem>[_repNN]_eta.csv``.

    Returns:
        A list of ``Path`` objects. Single-replicate runs return
        ``[<stem>.csv]`` (or ``[<stem>.csv, <stem>_eta.csv]``);
        multi-replicate runs return the ``_repNN`` variants in ascending
        replicate order.
    """
    from re import compile as _re_compile

    if scenario_id not in SCENARIO_FILENAME_STEMS:
        raise KeyError(f"Unknown scenario id: {scenario_id!r}")
    stem = SCENARIO_FILENAME_STEMS[scenario_id]
    suite_dir = Path(suite_dir)

    single = suite_dir / f"{stem}.csv"
    rep_pattern = _re_compile(rf"^{stem}_rep(\d+)\.csv$")
    rep_matches = sorted(
        (
            (int(m.group(1)), p)
            for p in suite_dir.iterdir()
            if p.is_file() and (m := rep_pattern.match(p.name))
        ),
        key=lambda pair: pair[0],
    )

    if rep_matches:
        data_paths = [p for _, p in rep_matches]
    elif single.exists():
        data_paths = [single]
    else:
        return []

    if not include_eta:
        return data_paths

    result: list[Path] = []
    for p in data_paths:
        result.append(p)
        eta = p.with_name(p.stem + "_eta.csv")
        if eta.exists():
            result.append(eta)
    return result


def load_reference_eta(eta_csv_path: Path) -> dict[str, dict[str, float]]:
    """Parse a Suite A ``*_eta.csv`` ground-truth file into the
    ``{subject_id: {param_name: value}}`` shape :func:`score_eta_recovery`
    (``apmode.benchmarks.scoring``) expects.

    The simulator (``benchmarks/suite_a/simulate_all.R``) writes one row per
    subject with an ``NMID`` column plus one ``eta.<PARAM>`` column per
    random effect (e.g. ``"NMID","eta.ka","eta.V","eta.CL"``). The
    ``"eta."`` prefix is stripped and the subject id is cast to ``str`` to
    match the naming convention ``r/harness.R``'s ``per_subject_eta``
    extraction settled on (see the comment above that block) — both sides
    key by stringified subject id and bare parameter name (``CL``, not
    ``eta.CL``), so the two dicts line up without any translation layer at
    the ``score_eta_recovery`` call site.

    Args:
        eta_csv_path: Path to a ``<stem>[_repNN]_eta.csv`` file as produced
            by ``scenario_dataset_paths(..., include_eta=True)``.

    Returns:
        ``{subject_id: {param_name: value}}``, e.g.
        ``{"1": {"ka": 0.126, "V": 0.083, "CL": 0.214}}``.
    """
    eta_csv_path = Path(eta_csv_path)
    result: dict[str, dict[str, float]] = {}
    with eta_csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return result
        id_col = "NMID" if "NMID" in reader.fieldnames else reader.fieldnames[0]
        eta_cols = [c for c in reader.fieldnames if c.startswith("eta.")]
        for row in reader:
            subject_id = str(row[id_col])
            result[subject_id] = {col[len("eta.") :]: float(row[col]) for col in eta_cols}
    return result


def suite_a_manifest(suite_dir: Path) -> dict[str, list[Path]]:
    """Return ``{scenario_id: [dataset_paths]}`` for every A1..A8 scenario.

    Missing scenarios map to an empty list. Downstream harnesses iterate
    over ``manifest[scenario_id]`` to fit across replicates without
    knowing whether the simulator was invoked in single- or
    multi-replicate mode.
    """
    suite_dir = Path(suite_dir)
    return {
        scenario_id: scenario_dataset_paths(suite_dir, scenario_id)
        for scenario_id in SCENARIO_FILENAME_STEMS
    }
