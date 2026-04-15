# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite B Extended: Real-data anchors with controlled perturbations.

Extends the original Suite B (B1-B3) with real-data scenarios (B4-B9):

  B4: Theophylline NODE anchor — continuity with Bräm 2023
  B5: Mavoglurant BLQ robustness — inject 25%/40% BLQ censoring
  B6: Mavoglurant outlier resilience — inject 5% extreme outliers
  B7: Mavoglurant sparse absorption — remove pre-Tmax samples
  B8: Mavoglurant null covariate FP rate — add 5 random covariates
  B9: Gentamicin IOV challenge — inter-occasion variability on CL

Architecture:
  - Real datasets enter as *anchors* for controlled perturbations
  - One dataset can generate many cases via different recipes
  - Each case has expected assertions (dispatch, FP rate, rank stability)
"""

from __future__ import annotations

from apmode.benchmarks.models import (
    BenchmarkCase,
    ExpectedStructure,
    PerturbationRecipe,
    PerturbationType,
    SplitStrategy,
)

# ---------------------------------------------------------------------------
# B4: Theophylline NODE anchor (Bräm 2023 continuity)
# ---------------------------------------------------------------------------

CASE_B4_THEO_NODE = BenchmarkCase(
    case_id="b4_theophylline_node",
    suite="B",
    dataset_id="nlmixr2data_theophylline",
    description=(
        "Theophylline NODE anchor: reproduces Bräm 2023 qualitative behavior. "
        "1-CMT first-order absorption → NODE should recover shape. "
        "Small dataset (n=12) — sanity check, not primary claim."
    ),
    lane="discovery",
    policy_file="discovery.json",
    expected_structure=ExpectedStructure(
        absorption="FirstOrder",
        distribution="OneCmt",
        elimination="Linear",
        n_compartments=1,
    ),
    reference_params={"ka": 1.5, "V": 0.5, "CL": 0.04},  # per-kg estimates
    ci_cadence="nightly",
)


# ---------------------------------------------------------------------------
# B5: Mavoglurant BLQ robustness
# ---------------------------------------------------------------------------

CASE_B5_MAVO_BLQ25 = BenchmarkCase(
    case_id="b5_mavoglurant_blq25",
    suite="B",
    dataset_id="nlmixr2data_mavoglurant",
    description=(
        "Mavoglurant with 25% BLQ censoring. Tests rank stability and "
        "parameter bias under moderate BLQ burden."
    ),
    lane="submission",
    policy_file="submission.json",
    perturbations=[
        PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.25,
            seed=753849,
        ),
    ],
    expected_structure=ExpectedStructure(
        distribution="TwoCmt",
        elimination="Linear",
        n_compartments=2,
    ),
    ci_cadence="nightly",
)

CASE_B5_MAVO_BLQ40 = BenchmarkCase(
    case_id="b5_mavoglurant_blq40",
    suite="B",
    dataset_id="nlmixr2data_mavoglurant",
    description=(
        "Mavoglurant with 40% BLQ censoring. Stress test: heavy BLQ "
        "burden should not collapse structural selection."
    ),
    lane="submission",
    policy_file="submission.json",
    perturbations=[
        PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.40,
            seed=43,
        ),
    ],
    expected_structure=ExpectedStructure(
        distribution="TwoCmt",
        n_compartments=2,
    ),
    ci_cadence="weekly",
)


# ---------------------------------------------------------------------------
# B6: Mavoglurant outlier resilience
# ---------------------------------------------------------------------------

CASE_B6_MAVO_OUTLIERS = BenchmarkCase(
    case_id="b6_mavoglurant_outliers5",
    suite="B",
    dataset_id="nlmixr2data_mavoglurant",
    description=(
        "Mavoglurant with 5% outlier injection (5x magnitude). "
        "Tests outlier detection and rank stability."
    ),
    lane="submission",
    policy_file="submission.json",
    perturbations=[
        PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_OUTLIERS,
            outlier_fraction=0.05,
            outlier_magnitude=5.0,
            seed=44,
        ),
    ],
    expected_structure=ExpectedStructure(
        distribution="TwoCmt",
        n_compartments=2,
    ),
    ci_cadence="nightly",
)


# ---------------------------------------------------------------------------
# B7: Mavoglurant sparse absorption
# ---------------------------------------------------------------------------

CASE_B7_MAVO_SPARSE_ABS = BenchmarkCase(
    case_id="b7_mavoglurant_sparse_absorption",
    suite="B",
    dataset_id="nlmixr2data_mavoglurant",
    description=(
        "Mavoglurant with absorption-phase samples removed (TIME < 1h). "
        "Tests profiler absorption_phase_coverage detection and dispatch."
    ),
    lane="discovery",
    policy_file="discovery.json",
    perturbations=[
        PerturbationRecipe(
            perturbation_type=PerturbationType.REMOVE_ABSORPTION_SAMPLES,
            absorption_time_cutoff=1.0,
            seed=45,
        ),
    ],
    expected_dispatch_excludes=["jax_node"],  # Sparse absorption → no NODE
    ci_cadence="nightly",
)


# ---------------------------------------------------------------------------
# B8: Mavoglurant null covariate false positive rate
# ---------------------------------------------------------------------------

CASE_B8_MAVO_NULL_COV = BenchmarkCase(
    case_id="b8_mavoglurant_null_covariates",
    suite="B",
    dataset_id="nlmixr2data_mavoglurant",
    description=(
        "Mavoglurant with 5 null random covariates. Tests covariate "
        "false positive rate — none should be selected (target: ≤10% FPR)."
    ),
    lane="submission",
    policy_file="submission.json",
    perturbations=[
        PerturbationRecipe(
            perturbation_type=PerturbationType.ADD_NULL_COVARIATES,
            null_covariate_n=5,
            seed=46,
        ),
    ],
    expected_structure=ExpectedStructure(
        distribution="TwoCmt",
        n_compartments=2,
    ),
    ci_cadence="weekly",
)


# ---------------------------------------------------------------------------
# B9: Gentamicin IOV challenge
# ---------------------------------------------------------------------------

CASE_B9_GENTA_IOV = BenchmarkCase(
    case_id="b9_gentamicin_iov",
    suite="B",
    dataset_id="ddmore_gentamicin",
    description=(
        "Gentamicin neonatal IOV dataset (Germovsek 2017). "
        "Tests IOV estimation on CL with multiple occasions per subject. "
        "Known algorithmic challenge: many tools estimate IOV poorly."
    ),
    lane="discovery",
    policy_file="discovery.json",
    split_strategy=SplitStrategy(
        method="subject_level_kfold",
        n_folds=5,
        seed=47,
    ),
    ci_cadence="weekly",
)


# ---------------------------------------------------------------------------
# All extended cases
# ---------------------------------------------------------------------------

ALL_EXTENDED_CASES: list[BenchmarkCase] = [
    CASE_B4_THEO_NODE,
    CASE_B5_MAVO_BLQ25,
    CASE_B5_MAVO_BLQ40,
    CASE_B6_MAVO_OUTLIERS,
    CASE_B7_MAVO_SPARSE_ABS,
    CASE_B8_MAVO_NULL_COV,
    CASE_B9_GENTA_IOV,
]

# Nightly subset
NIGHTLY_CASES: list[BenchmarkCase] = [
    c for c in ALL_EXTENDED_CASES if c.ci_cadence in ("per_pr", "nightly")
]

__all__ = ["ALL_EXTENDED_CASES", "NIGHTLY_CASES"]
