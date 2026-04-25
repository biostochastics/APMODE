# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite C: Methodology validation vs established literature models.

Phase 1 (v0.6 scope) evaluates APMODE against *published, peer-reviewed*
reference parameterizations on real clinical PK datasets. The primary
metric is ``fraction-beats-literature-median``:

    fraction-beats-literature-median >= 60%

Phase 2 (head-to-head vs a blinded human-expert panel) is gated on
external collaborator coordination and **is not in v0.6 scope**. The
plan-task rename (plan Task 38) reframed the suite to match the evidence
this release actually ships: the literature anchor is a fixed baseline;
the expert panel baseline lands with Phase 2.

Methodology (Phase 1):
  1. For each dataset d, load the published literature model fixture.
  2. Score the literature model using the same harness applied to APMODE.
  3. APMODE "wins" on dataset d if NPE_APMODE <= NPE_literature * (1 - delta).
  4. Fraction-beats-literature-median = (#datasets APMODE wins) / total.

Pre-registered evaluation protocol:
  - Split: subject-level 5-fold cross-validation
  - Primary metric: NPE (median absolute prediction error from posterior
    predictive simulations on held-out subjects)
  - Win margin (delta): 0.02 (APMODE must beat the literature NPE by >= 2%)
  - Bootstrap CI: 80% CI of (NPE_APMODE - NPE_literature) < 0

Secondary outcomes:
  - NLPD gap vs literature (qualified: same observation model only)
  - Structure agreement with published model
  - Wall-clock time-to-model (APMODE measured)
  - Prediction interval calibration (50/80/90/95% PI coverage)

Initial Suite C datasets (Phase 1):
  C1: Mavoglurant (nlmixr2data) — 222 subjects, oral 2-CMT
  C2: Gentamicin IOV (DDMoRe) — 205 neonates, IOV challenge
  C3: Eleveld Propofol (OpenTCI) — 1033 subjects, IV infusion 3-CMT

Literature baseline strategy:
  - Published models serve as *reference comparators* ("literature anchor")
  - Phase 2 (future): blinded expert panel working under same time budget
"""

from __future__ import annotations

from apmode.benchmarks.models import (
    BenchmarkCase,
    ExpectedStructure,
    PublishedModel,
    SplitStrategy,
)

# ---------------------------------------------------------------------------
# Evaluation protocol constants
# ---------------------------------------------------------------------------

DEFAULT_SPLIT = SplitStrategy(
    method="subject_level_kfold",
    n_folds=5,
    seed=20260414,
)

WIN_MARGIN_DELTA: float = 0.02  # APMODE must beat literature NPE by >= 2%
BOOTSTRAP_CI_LEVEL: float = 0.80  # 80% CI for NPE gap
# Minimum literature comparators for a reliable Phase-1 median. Phase 2's
# blinded expert panel (post v0.6) reuses this floor as its N_experts.
MIN_LITERATURE_COUNT: int = 3
# Backwards-compatible alias: some callers still import MIN_EXPERT_COUNT.
# Scheduled for removal in v0.7 once the Suite C orchestration lands.
MIN_EXPERT_COUNT: int = MIN_LITERATURE_COUNT

# NOTE on literature_models: Each case currently provides 1 published model
# as a *reference anchor* (absolute baseline). Phase 2 will add a blinded
# expert panel (N >= MIN_LITERATURE_COUNT) working under the same time
# budget — that panel is assembled per evaluation round, not hard-coded.
# If a panel is unavailable, scoring falls back to comparing AIC/BIC/OFV
# against the published literature anchor.


# ---------------------------------------------------------------------------
# C1: Mavoglurant (primary Suite C dataset)
# ---------------------------------------------------------------------------

MAVOGLURANT_PUBLISHED = PublishedModel(
    citation=(
        "Wendling T, Ogungbenro K, Pigeolet E, Dumitras S, Woessner R, "
        "Aarons L. (2015) Pharm Res 32(5):1764-1778. "
        "doi:10.1007/s11095-014-1574-1"
    ),
    # Wendling fit sumIG absorption; the DSL collapses to first-order ka and
    # the simplified IR-cohort fit is the comparison the Phase-1 fixture
    # uses (see benchmarks/suite_c/mavoglurant_wendling_2015.yaml). Sourced
    # directly from that fixture's ``reference_params`` so the
    # PublishedModel record stays in lockstep with the live runner anchor.
    structure="2-CMT oral, first-order absorption, linear elimination",
    n_parameters=7,
    estimation_method="SAEM",
    tool="nlmixr2 3.0.0",
    key_estimates={
        "ka": 0.5,
        "V1": 25.0,
        "V2": 80.0,
        "Q": 5.0,
        "CL": 4.0,
    },
)

CASE_C1_MAVOGLURANT = BenchmarkCase(
    case_id="c1_mavoglurant",
    suite="C",
    dataset_id="nlmixr2data_mavoglurant",
    description=(
        "Mavoglurant literature comparison: 222 subjects, oral 2-CMT. "
        "Primary Suite C dataset — freely available, adequate complexity. "
        "Published nlmixr2 model available as anchor baseline."
    ),
    lane="submission",
    policy_file="submission.json",
    split_strategy=DEFAULT_SPLIT,
    expected_structure=ExpectedStructure(
        distribution="TwoCmt",
        elimination="Linear",
        n_compartments=2,
    ),
    literature_models=[MAVOGLURANT_PUBLISHED],
    win_margin=WIN_MARGIN_DELTA,
    ci_cadence="quarterly",
)


# ---------------------------------------------------------------------------
# C2: Gentamicin IOV (CC0 license)
# ---------------------------------------------------------------------------

GENTAMICIN_PUBLISHED = PublishedModel(
    citation="Germovsek et al. (2017) AAC 61(8):e00481-17",
    structure="1-CMT IV, linear elimination, IOV on CL, allometric WT",
    n_parameters=8,
    estimation_method="FOCE-I",
    tool="NONMEM 7.3",
    key_estimates={
        "CL": 0.065,  # L/h/kg (typical neonatal gentamicin)
        "V": 0.45,  # L/kg
    },
)

CASE_C2_GENTAMICIN = BenchmarkCase(
    case_id="c2_gentamicin_iov",
    suite="C",
    dataset_id="ddmore_gentamicin",
    description=(
        "Gentamicin IOV literature comparison: 205 neonates, 2788 obs. "
        "IOV on CL is the primary challenge — algorithmic stress test. "
        "Published NONMEM model (Germovsek 2017) as reference anchor. "
        "CC0 license, fully open."
    ),
    lane="submission",
    policy_file="submission.json",
    split_strategy=DEFAULT_SPLIT,
    literature_models=[GENTAMICIN_PUBLISHED],
    win_margin=WIN_MARGIN_DELTA,
    ci_cadence="quarterly",
)


# ---------------------------------------------------------------------------
# C3: Eleveld Propofol (large IV infusion)
# ---------------------------------------------------------------------------

PROPOFOL_PUBLISHED = PublishedModel(
    citation="Eleveld et al. (2018) Br J Anaesth 120(5):942-959",
    structure="3-CMT IV infusion, allometric scaling, age + opioid covariates",
    n_parameters=15,
    estimation_method="NONMEM FOCE-I",
    tool="NONMEM 7.4",
    key_estimates={
        "V1": 6.28,  # L (typical adult)
        "V2": 25.5,
        "V3": 273.0,
        "CL": 1.89,  # L/min
        "Q2": 1.29,
        "Q3": 0.836,
    },
    ofv=None,  # Not reported in summary
)

CASE_C3_PROPOFOL = BenchmarkCase(
    case_id="c3_eleveld_propofol",
    suite="C",
    dataset_id="opentci_propofol",
    description=(
        "Eleveld propofol literature comparison: 1033 subjects, 15433 PK obs. "
        "3-CMT IV infusion with rich covariates. The published Eleveld 2018 "
        "model is the gold standard — tests whether APMODE can match or "
        "improve on an expert model developed over multiple years. "
        "Requires IV infusion DSL support."
    ),
    lane="discovery",
    policy_file="discovery.json",
    split_strategy=DEFAULT_SPLIT,
    expected_structure=ExpectedStructure(
        n_compartments=3,
    ),
    literature_models=[PROPOFOL_PUBLISHED],
    win_margin=WIN_MARGIN_DELTA,
    ci_cadence="quarterly",
)


# ---------------------------------------------------------------------------
# All Suite C cases
# ---------------------------------------------------------------------------

ALL_CASES: list[BenchmarkCase] = [
    CASE_C1_MAVOGLURANT,
    CASE_C2_GENTAMICIN,
    CASE_C3_PROPOFOL,
]

# Phase 3 priority order (mavoglurant first — no access barriers)
PRIORITY_ORDER: list[str] = [
    "c1_mavoglurant",
    "c2_gentamicin_iov",
    "c3_eleveld_propofol",
]

__all__ = [
    "ALL_CASES",
    "CASE_C1_MAVOGLURANT",
    "CASE_C2_GENTAMICIN",
    "CASE_C3_PROPOFOL",
    "DEFAULT_SPLIT",
    "MIN_EXPERT_COUNT",  # deprecated alias, removed in v0.7
    "MIN_LITERATURE_COUNT",
    "PRIORITY_ORDER",
    "WIN_MARGIN_DELTA",
]
