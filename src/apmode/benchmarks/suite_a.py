# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite A: Synthetic Recovery scenarios (PRD §5).

Scenarios with known ground truth for structure/parameter recovery testing.
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
  A9: Erlang absorption (n=3 explicit chain), 1-cmt, linear elimination
  A10: Parallel first-order absorption (two simultaneous depots), 1-cmt, linear
  A11: Mixed first-order + zero-order absorption, 1-cmt, linear elimination
  A12: Standalone zero-order absorption, 1-cmt, linear elimination
  A13: Sum-of-two-Inverse-Gaussians (SumIG, k=2) absorption, 1-cmt, linear
  A14: Three-compartment IV bolus, linear elimination
  A15: TMDD full binding model (Mager & Jusko 2001), 3-arm dose-ranging design
  A16: 1-cmt oral, time-varying (exponential decay) elimination, unconfounded
  A17: 1-cmt oral, block-structured IIV (CL-V correlation) + additive error
  A18: 1-cmt oral, inter-occasion variability (IOV) on ka across 3 occasions
  A19: 1-cmt oral, maturation-form (Hill/sigmoid) covariate on CL
  A20a/A20b: 1-cmt oral, elevated-LLOQ BLQ (shared ground truth, M3 vs M4)
  A21: TMDD QSS, multi-analyte observation (free drug + total target)
"""

from __future__ import annotations

import csv
from pathlib import Path

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
    Erlang,
    ExperimentalFlags,
    FirstOrder,
    IVBolus,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    ObservationEndpoint,
    OccasionByDoseEpoch,
    OneCmt,
    ParallelFirstOrder,
    ParallelLinearMM,
    Proportional,
    SumIG,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.priors import LogNormalPrior, PriorSpec


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


def scenario_a9() -> DSLSpec:
    """A9: Erlang absorption (n=3 explicit chain), 1-cmt, linear elimination.

    Distinct from A3's Transit (rxode2 ``transit(n, mtt)``, gamma
    interpolation + terminal ka): Erlang lowers to an explicit
    n-compartment chain with the last state feeding centr directly, no
    terminal ka (ADR-0003 D2).
    """
    return DSLSpec(
        model_id="suite_a_scenario_a9",
        absorption=Erlang(n=3),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ktr"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.12),
        initial={"ktr": 2.0, "V": 65.0, "CL": 4.5},
    )


def scenario_a10() -> DSLSpec:
    """A10: Parallel first-order absorption (two simultaneous depots), 1-cmt.

    Two SIMULTANEOUS first-order depots (fast ka1, slow ka2) with a
    bioavailability-fraction split, distinct from A11's mixed first+zero-
    order mechanism. ``frac`` is fixed (no IIV) for identifiability.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a10",
        absorption=ParallelFirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ka1", "ka2"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.10),
        initial={"ka1": 2.0, "ka2": 0.3, "frac": 0.6, "V": 60.0, "CL": 4.0},
    )


def scenario_a11() -> DSLSpec:
    """A11: Mixed first-order + zero-order absorption, 1-cmt, linear elimination.

    A first-order depot and a separate zero-order depot (modeled duration)
    both feed centr, split by a fixed bioavailability fraction.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a11",
        absorption=MixedFirstZero(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ka", "dur"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.10),
        initial={"ka": 1.5, "dur": 3.0, "frac": 0.55, "V": 60.0, "CL": 4.0},
    )


def scenario_a12() -> DSLSpec:
    """A12: Standalone zero-order absorption, 1-cmt, linear elimination.

    Dose enters centr directly at a constant rate over ``dur`` hours
    (matrix-controlled-release-style oral, or constant-rate extravascular
    input) via rxode2's modeled-duration infusion mechanism.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a12",
        absorption=ZeroOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "dur"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.12),
        initial={"dur": 4.0, "V": 55.0, "CL": 4.5},
    )


def scenario_a13() -> DSLSpec:
    """A13: Sum-of-two-Inverse-Gaussians (SumIG, k=2) absorption, 1-cmt.

    Closed-form analytical input rate (Csajka 2005; Weiss & Wegner 2022),
    single-dose only (v0.7 limitation, ADR-0003 D4). Per SumIG's own
    identifiability note (ADR-0003 D5), disposition (CL/V) is kept fixed
    (no IIV) here -- only the absorption-shape parameter MT_1 carries BSV.
    ``k>=2`` requires disposition to be marked fixed-external in the
    validator's cross-module check (ADR-0003 D5) -- the ``priors`` below
    are the spec-side signal for that (as opposed to an
    ``EvidenceManifest.disposition_fixed`` flag set at dispatch time).
    """
    return DSLSpec(
        model_id="suite_a_scenario_a13",
        absorption=SumIG(k=2),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["MT_1"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
        priors=[
            PriorSpec(
                target="CL",
                family=LogNormalPrior(mu=1.386294, sigma=0.01),  # log(4.0)
                source="fixed_external",
                justification="Benchmark ground truth: CL fixed at the R-simulated "
                "population value per SumIG(k=2) disposition-fixed identifiability "
                "constraint (ADR-0003 D5).",
            ),
            PriorSpec(
                target="V",
                family=LogNormalPrior(mu=3.688879, sigma=0.01),  # log(40.0)
                source="fixed_external",
                justification="Benchmark ground truth: V fixed at the R-simulated "
                "population value per SumIG(k=2) disposition-fixed identifiability "
                "constraint (ADR-0003 D5).",
            ),
        ],
        initial={
            "MT_1": 0.5,
            "MT_2": 3.5,
            "RD2_1": 0.3,
            "RD2_2": 2.0,
            "weight_1": 0.55,
            "V": 40.0,
            "CL": 4.0,
        },
    )


def scenario_a14() -> DSLSpec:
    """A14: Three-compartment IV bolus, linear elimination.

    Dose routes directly to centr (IVBolus, no depot). Sampling in the R
    ground truth extends to 120h with several points beyond 24h so the deep
    third compartment (V3/Q3) is identifiable and the fit cannot collapse
    to an apparent 2-cmt model.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a14",
        absorption=IVBolus(),
        distribution=ThreeCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V1", "Q2", "Q3"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
        initial={"V1": 5.0, "V2": 15.0, "V3": 100.0, "Q2": 8.0, "Q3": 1.5, "CL": 5.0},
    )


def scenario_a15() -> DSLSpec:
    """A15: TMDD full binding model (Mager & Jusko 2001), dose-ranging design.

    Distinct from A5's QSS approximation. The R ground truth uses a 3-arm
    dose-ranging design (low/mid/high, ~20 subjects/arm) spanning
    sub-saturating to saturating target exposure so kon/koff are
    identifiable from the shape of the nonlinear-clearance transition — a
    single dose level cannot separate linear from target-mediated
    clearance. Binding-kinetic parameters (kon, koff, kint) carry no IIV.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a15",
        absorption=FirstOrder(),
        distribution=TMDDCore(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "R0", "ka"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.15),
        initial={
            "ka": 0.02,
            "V": 3.0,
            "R0": 20.0,
            "kon": 0.1,
            "koff": 0.1,
            "kint": 0.03,
            "CL": 0.01,
        },
    )


def scenario_a16() -> DSLSpec:
    """A16: 1-cmt oral, time-varying (exponential decay) elimination, unconfounded.

    ``CL(t) = CL * exp(-kdecay * t)`` with NO covariate attached to CL,
    unlike A8's covariate-confounded autoinduction — isolates whether
    structure search attributes the CL drift to time-varying kinetics
    rather than defaulting to a covariate explanation. R ground truth uses
    repeated (QD x 14 days) dosing so decay is separable from distribution.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a16",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=TimeVaryingElim(decay_fn="exponential"),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.10),
        initial={"ka": 1.2, "V": 50.0, "CL": 5.0, "kdecay": 0.0015},
    )


def scenario_a17() -> DSLSpec:
    """A17: 1-cmt oral, block-structured IIV (CL-V correlation) + additive error.

    Genuine positive CL-V correlation (~0.4) combined with an Additive
    (not proportional) residual error model. Variability structure and
    observation error are orthogonal DSL axes, so combining them in one
    spec does not confound either one's estimability.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a17",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[
            IIV(params=["CL", "V"], structure="block"),
            IIV(params=["ka"], structure="diagonal"),
        ],
        observation=Additive(sigma_add=0.5),
        initial={"ka": 1.3, "V": 12.0, "CL": 4.5},
    )


def scenario_a18() -> DSLSpec:
    """A18: 1-cmt oral, inter-occasion variability (IOV) on ka across 3 occasions.

    Three dosing occasions spaced a week apart (negligible carryover given
    the elimination half-life at these CL/V values). The R ground truth's
    occasion-indexing data column is named ``OCC`` (rxode2 itself requires
    the internal simulation column to be literally ``occ`` lowercase;
    verified empirically -- see ``simulate_all.R::sim_A18``).
    """
    return DSLSpec(
        model_id="suite_a_scenario_a18",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[
            IIV(params=["CL", "V", "ka"], structure="diagonal"),
            IOV(params=["ka"], occasions=OccasionByDoseEpoch(column="OCC")),
        ],
        observation=Proportional(sigma_prop=0.10),
        initial={"ka": 1.2, "V": 55.0, "CL": 4.0},
    )


def scenario_a19() -> DSLSpec:
    """A19: 1-cmt oral, maturation-form (Hill/sigmoid) covariate on CL.

    ``CL(PNA) = CL0 * PNA^hill / (PNA^hill + TM50^hill)``, postnatal age
    (PNA, weeks) drawn from uniform(2, 40) in the R ground truth so the
    cohort meaningfully straddles TM50=15 (populating the sigmoid's steep
    region, not just its flat asymptotes).
    """
    return DSLSpec(
        model_id="suite_a_scenario_a19",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        covariates=[
            CovariateLink(param="CL", covariate="PNA", form="maturation", tm50=15.0, hill=3.0),
        ],
        observation=Proportional(sigma_prop=0.12),
        initial={"ka": 1.0, "V": 25.0, "CL": 6.0},
    )


def scenario_a20a() -> DSLSpec:
    """A20a: 1-cmt oral, elevated-LLOQ BLQ via M3 (left-censoring).

    Shares the SAME simulated CSV ground truth as :func:`scenario_a20b`
    (elevated LLOQ=0.12 vs. A2/A4/A5's ~0.01-0.02, chosen for a ~15-30%
    BLQ fraction) — M3 vs. M4 differ only in the emitted likelihood, not
    the raw M3-style data encoding used by the R simulator. Scored as a
    paired benchmark unit (A20a/A20b) against identical ground truth, not
    as two independent recovery tests.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a20a",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=BLQM3(loq_value=0.12, error_model="proportional", sigma_prop=0.15),
        initial={"ka": 1.5, "V": 70.0, "CL": 6.0},
    )


def scenario_a20b() -> DSLSpec:
    """A20b: 1-cmt oral, elevated-LLOQ BLQ via M4 (censoring, positive constraint).

    See :func:`scenario_a20a` docstring — shares the same ground truth CSV.
    """
    return DSLSpec(
        model_id="suite_a_scenario_a20b",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V", "ka"], structure="diagonal")],
        observation=BLQM4(loq_value=0.12, error_model="proportional", sigma_prop=0.15),
        initial={"ka": 1.5, "V": 70.0, "CL": 6.0},
    )


def scenario_a21() -> DSLSpec:
    """A21: TMDD QSS, multi-analyte observation (free drug + total target).

    Same structural values as A5. The DSL's only multi-analyte-eligible
    pairing -- ``DSLSpec.known_prediction_variables()`` only exposes
    ``"C_target_total"`` (the Rtot state) under TMDDQSS; there is no
    metabolite/parent-child compartment topology in the DSL yet. DVID=1
    is free-drug concentration (``C_central``), DVID=2 is total-target
    concentration (``C_target_total``).
    """
    return DSLSpec(
        model_id="suite_a_scenario_a21",
        absorption=FirstOrder(),
        distribution=TMDDQSS(),
        elimination=LinearElim(),
        variability=[
            IIV(params=["CL", "V", "R0", "KD", "kint", "ka"], structure="diagonal"),
        ],
        observation=Proportional(sigma_prop=0.15),
        observations={
            "free_drug": ObservationEndpoint(
                name="free_drug",
                dvid=1,
                prediction="C_central",
                error=Proportional(sigma_prop=0.15),
            ),
            "total_target": ObservationEndpoint(
                name="total_target",
                dvid=2,
                prediction="C_target_total",
                error=Proportional(sigma_prop=0.12),
            ),
        },
        initial={"ka": 0.02, "V": 3.5, "R0": 10.0, "KD": 1.0, "kint": 0.03, "CL": 0.015},
    )


# Reference parameter values (ground truth for recovery testing).
# Keys match structural_param_names() for each scenario's DSLSpec.
REFERENCE_PARAMS: dict[str, dict[str, float]] = {
    "A1": {"ka": 1.5, "V": 70.0, "CL": 5.0},
    "A2": {"ka": 100.0, "V1": 50.0, "V2": 80.0, "Q": 10.0, "CL": 3.0, "Vmax": 100.0, "Km": 10.0},
    "A3": {"ktr": 2.0, "ka": 1.0, "V": 60.0, "CL": 4.0},
    "A4": {"ka": 1.2, "V": 65.0, "Vmax": 80.0, "Km": 8.0},
    "A5": {"ka": 0.02, "V": 3.5, "R0": 10.0, "KD": 1.0, "kint": 0.03, "CL": 0.015},
    "A6": {"ka": 1.5, "V": 70.0, "CL": 5.0},
    # A7: mechanistic params only (NODE absorption weights are not named structural params)
    "A7": {"V1": 50.0, "V2": 80.0, "Q": 10.0, "CL": 4.0},
    "A8": {"ka": 1.822, "V": 29.964, "CL": 4.482},
    # Erlang.n is structural (set by transform, not estimated) — unlike
    # Transit.n, it is not in structural_param_names(); only ktr is.
    "A9": {"ktr": 2.0, "V": 65.0, "CL": 4.5},
    "A10": {"ka1": 2.0, "ka2": 0.3, "frac": 0.6, "V": 60.0, "CL": 4.0},
    "A11": {"ka": 1.5, "dur": 3.0, "frac": 0.55, "V": 60.0, "CL": 4.0},
    "A12": {"dur": 4.0, "V": 55.0, "CL": 4.5},
    "A13": {
        "MT_1": 0.5,
        "MT_2": 3.5,
        "RD2_1": 0.3,
        "RD2_2": 2.0,
        "weight_1": 0.55,
        "V": 40.0,
        "CL": 4.0,
    },
    "A14": {"V1": 5.0, "V2": 15.0, "V3": 100.0, "Q2": 8.0, "Q3": 1.5, "CL": 5.0},
    "A15": {
        "ka": 0.02,
        "V": 3.0,
        "R0": 20.0,
        "kon": 0.1,
        "koff": 0.1,
        "kint": 0.03,
        "CL": 0.01,
    },
    "A16": {"ka": 1.2, "V": 50.0, "CL": 5.0, "kdecay": 0.0015},
    "A17": {"ka": 1.3, "V": 12.0, "CL": 4.5},
    "A18": {"ka": 1.2, "V": 55.0, "CL": 4.0},
    "A19": {"ka": 1.0, "V": 25.0, "CL": 6.0},
    "A20a": {"ka": 1.5, "V": 70.0, "CL": 6.0},
    "A20b": {"ka": 1.5, "V": 70.0, "CL": 6.0},
    "A21": {"ka": 0.02, "V": 3.5, "R0": 10.0, "KD": 1.0, "kint": 0.03, "CL": 0.015},
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
    ("A9", scenario_a9),
    ("A10", scenario_a10),
    ("A11", scenario_a11),
    ("A12", scenario_a12),
    ("A13", scenario_a13),
    ("A14", scenario_a14),
    ("A15", scenario_a15),
    ("A16", scenario_a16),
    ("A17", scenario_a17),
    ("A18", scenario_a18),
    ("A19", scenario_a19),
    # A20a/A20b: paired benchmark unit sharing one simulated ground-truth
    # CSV (a20_1cmt_oral_blq_elevated_lloq.csv) -- see scenario_a20a
    # docstring. Two DSLSpec factories so both BLQ_M3 and BLQ_M4 close
    # their respective capability tags against identical ground truth.
    ("A20a", scenario_a20a),
    ("A20b", scenario_a20b),
    ("A21", scenario_a21),
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
    "A9": "a9_erlang_absorption",
    "A10": "a10_parallel_first_order",
    "A11": "a11_mixed_first_zero_absorption",
    "A12": "a12_zero_order_absorption",
    "A13": "a13_sum_ig_absorption",
    "A14": "a14_3cmt_iv_bolus",
    "A15": "a15_tmdd_core_dosearms",
    "A16": "a16_time_varying_elim_unconfounded",
    "A17": "a17_block_iiv_additive_error",
    "A18": "a18_iov_occasions",
    "A19": "a19_maturation_covariate",
    # A20a/A20b share one ground-truth CSV -- see ALL_SCENARIOS comment.
    "A20a": "a20_1cmt_oral_blq_elevated_lloq",
    "A20b": "a20_1cmt_oral_blq_elevated_lloq",
    "A21": "a21_tmdd_qss_multi_analyte",
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
