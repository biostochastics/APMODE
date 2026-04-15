# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite C: Expert head-to-head comparison (PRD §5, Phase 3).

Evaluates APMODE's workflow competitiveness against human pharmacometricians
on real clinical PK datasets. The primary metric is:

    fraction-beats-median-expert >= 60%

Methodology:
  1. For each dataset d, obtain K expert final models (K=3-5).
  2. Score each expert model using the same harness applied to APMODE.
  3. Compute the median expert score for that dataset.
  4. APMODE "wins" on dataset d if NPE_APMODE <= MedianExpertNPE * (1 - delta).
  5. Fraction-beats-median = (#datasets APMODE wins) / total.

Pre-registered evaluation protocol:
  - Split: subject-level 5-fold cross-validation
  - Primary metric: NPE (median absolute prediction error from posterior
    predictive simulations on held-out subjects)
  - Win margin (delta): 0.02 (APMODE must beat expert median by >= 2%)
  - Bootstrap CI: 80% CI of (NPE_APMODE - median_expert) < 0

Secondary outcomes:
  - NLPD gap vs best expert (qualified: same observation model only)
  - Structure agreement with published model
  - Wall-clock time-to-model (APMODE measured; experts self-report)
  - Prediction interval calibration (50/80/90/95% PI coverage)

Initial Suite C datasets (Phase 3):
  C1: Mavoglurant (nlmixr2data) — 222 subjects, oral 2-CMT
  C2: Gentamicin IOV (DDMoRe) — 205 neonates, IOV challenge
  C3: Eleveld Propofol (OpenTCI) — 1033 subjects, IV infusion 3-CMT

Expert baseline strategy:
  - Published models serve as *reference comparators* ("published anchor")
  - Primary baseline: blinded expert panel working under same time budget
  - Experts deliver: structural model, final estimates, BLQ handling, time spent
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

WIN_MARGIN_DELTA: float = 0.02  # APMODE must beat expert median by >= 2%
BOOTSTRAP_CI_LEVEL: float = 0.80  # 80% CI for NPE gap
MIN_EXPERT_COUNT: int = 3  # Minimum experts for reliable median

# NOTE on expert models: Each case currently provides 1 published model as a
# *reference anchor* (absolute baseline). The primary expert baseline is a
# blinded expert panel (N >= MIN_EXPERT_COUNT) working under the same time
# budget — this panel is assembled per evaluation round, not hard-coded.
# If the panel is unavailable, scoring falls back to comparing AIC/BIC/OFV
# against the published anchor. See evaluate_expert_comparison() guard.


# ---------------------------------------------------------------------------
# C1: Mavoglurant (primary Suite C dataset)
# ---------------------------------------------------------------------------

MAVOGLURANT_PUBLISHED = PublishedModel(
    citation="nlmixr2data package / Novartis",
    structure="2-CMT oral, first-order absorption, linear elimination",
    n_parameters=7,
    estimation_method="SAEM",
    tool="nlmixr2 3.0.0",
    key_estimates={},  # To be populated from nlmixr2 fitting
)

CASE_C1_MAVOGLURANT = BenchmarkCase(
    case_id="c1_mavoglurant",
    suite="C",
    dataset_id="nlmixr2data_mavoglurant",
    description=(
        "Mavoglurant expert comparison: 222 subjects, oral 2-CMT. "
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
    expert_models=[MAVOGLURANT_PUBLISHED],
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
        "Gentamicin IOV expert comparison: 205 neonates, 2788 obs. "
        "IOV on CL is the primary challenge — algorithmic stress test. "
        "Published NONMEM model (Germovsek 2017) as reference anchor. "
        "CC0 license, fully open."
    ),
    lane="submission",
    policy_file="submission.json",
    split_strategy=DEFAULT_SPLIT,
    expert_models=[GENTAMICIN_PUBLISHED],
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
        "Eleveld propofol expert comparison: 1033 subjects, 15433 PK obs. "
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
    expert_models=[PROPOFOL_PUBLISHED],
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
    "MIN_EXPERT_COUNT",
    "PRIORITY_ORDER",
    "WIN_MARGIN_DELTA",
]
