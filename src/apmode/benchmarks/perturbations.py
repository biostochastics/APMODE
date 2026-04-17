# SPDX-License-Identifier: GPL-2.0-or-later
"""Data perturbation functions for Suite B benchmark cases.

Pure functions: (DataFrame, PerturbationRecipe) -> (DataFrame, manifest_dict).
Each perturbation is deterministic given the recipe seed.

Observation identification uses ``(EVID == 0) & (MDV == 0)`` per NONMEM
convention — ``EVID=0, MDV=1`` denotes a missing/dropped sample that must
not be perturbed.

Perturbation types:
  - inject_blq: Censor low DV values (M3-style: CENS=1, DV=LLOQ)
  - remove_absorption_samples: Drop early timepoints
  - add_null_covariates: Add random covariates uncorrelated with PK
  - inject_outliers: Replace random DV values with extreme values
  - sparsify: Randomly drop observations to target obs/subject
  - add_protocol_pooling: Assign STUDY_ID with per-protocol design variation
  - add_occasion_labels: Add OCCASION column for IOV testing
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from apmode.benchmarks.models import PerturbationRecipe, PerturbationType

__all__ = ["apply_perturbation", "apply_perturbations"]


def _obs_mask(df: pd.DataFrame) -> pd.Series:
    """Identify true observation rows: EVID==0 AND MDV==0."""
    mask = df["EVID"] == 0
    if "MDV" in df.columns:
        mask = mask & (df["MDV"] == 0)
    return mask


def apply_perturbation(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a single perturbation to a canonical NONMEM-style DataFrame.

    Returns (perturbed_df, perturbation_manifest).
    The manifest documents exactly what changed for auditability.
    """
    rng = np.random.default_rng(recipe.seed)
    df = df.copy()

    dispatch = {
        PerturbationType.INJECT_BLQ: _inject_blq,
        PerturbationType.REMOVE_ABSORPTION_SAMPLES: _remove_absorption_samples,
        PerturbationType.ADD_NULL_COVARIATES: _add_null_covariates,
        PerturbationType.INJECT_OUTLIERS: _inject_outliers,
        PerturbationType.SPARSIFY: _sparsify,
        PerturbationType.ADD_PROTOCOL_POOLING: _add_protocol_pooling,
        PerturbationType.ADD_OCCASION_LABELS: _add_occasion_labels,
        # #26: the four stress-surface perturbations are declared on
        # the enum so Suite C recipes can request them today. The
        # concrete simulators wrap the canonical dataset with new
        # structural dynamics and therefore require coupling to the
        # DSL/forward-solve path — tracked in PRD §10 / Suite C.
        # Until they land the dispatch raises NotImplementedError so a
        # request to stress-test is never silently replaced by a no-op.
        PerturbationType.SCALE_BSV_VARIANCES: _scale_bsv_variances,
        PerturbationType.SATURATE_CLEARANCE: _saturate_clearance,
        PerturbationType.TMDD: _tmdd_perturbation,
        PerturbationType.FLIP_FLOP: _flip_flop_perturbation,
    }

    fn = dispatch.get(recipe.perturbation_type)
    if fn is None:
        msg = f"Unknown perturbation type: {recipe.perturbation_type}"
        raise ValueError(msg)

    return fn(df, recipe, rng)


def apply_perturbations(
    df: pd.DataFrame,
    recipes: list[PerturbationRecipe],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply multiple perturbations sequentially.

    Returns (perturbed_df, list_of_manifests).
    """
    manifests: list[dict[str, Any]] = []
    for recipe in recipes:
        df, manifest = apply_perturbation(df, recipe)
        manifests.append(manifest)
    return df, manifests


# ---------------------------------------------------------------------------
# Perturbation implementations
# ---------------------------------------------------------------------------


def _scale_bsv_variances(
    _df: pd.DataFrame,
    _recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — inflate/contract subject-level DV variance."""
    msg = (
        "scale_bsv_variances is declared on PerturbationType but the "
        "transform has not been implemented yet (PRD §10, Suite C). "
        "Refusing to silently run a no-op on a requested stress test."
    )
    raise NotImplementedError(msg)


def _saturate_clearance(
    _df: pd.DataFrame,
    _recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — re-simulate DV with Michaelis-Menten clearance."""
    msg = (
        "saturate_clearance is declared on PerturbationType but the "
        "transform has not been implemented yet (PRD §10, Suite C). "
        "Refusing to silently run a no-op on a requested stress test."
    )
    raise NotImplementedError(msg)


def _tmdd_perturbation(
    _df: pd.DataFrame,
    _recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — introduce target-mediated disposition dynamics."""
    msg = (
        "tmdd perturbation is declared on PerturbationType but the "
        "transform has not been implemented yet (PRD §10, Suite C). "
        "Refusing to silently run a no-op on a requested stress test."
    )
    raise NotImplementedError(msg)


def _flip_flop_perturbation(
    _df: pd.DataFrame,
    _recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — swap ka/ke roles to produce flip-flop kinetics."""
    msg = (
        "flip_flop perturbation is declared on PerturbationType but the "
        "transform has not been implemented yet (PRD §10, Suite C). "
        "Refusing to silently run a no-op on a requested stress test."
    )
    raise NotImplementedError(msg)


def _inject_blq(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Censor low DV values using M3-style convention.

    Sets CENS=1 and DV=LLOQ for censored observations (nlmixr2/NONMEM M3).
    Does NOT set DV=0 — that is ambiguous and biases M3 likelihood.
    """
    obs = _obs_mask(df)
    obs_dv = df.loc[obs, "DV"]

    fraction = recipe.blq_fraction or 0.25
    if recipe.lloq is not None:
        lloq = recipe.lloq
    else:
        positive_dv = obs_dv[obs_dv > 0]
        lloq = float(np.quantile(positive_dv, fraction)) if len(positive_dv) > 0 else 1.0

    blq_mask = obs & (df["DV"] < lloq)
    n_censored = int(blq_mask.sum())

    # M3 convention: DV = LLOQ, CENS = 1 (censored below)
    df.loc[blq_mask, "DV"] = lloq
    df["CENS"] = 0
    df.loc[blq_mask, "CENS"] = 1
    df["BLQ_FLAG"] = 0
    df.loc[blq_mask, "BLQ_FLAG"] = 1
    df["LLOQ"] = lloq

    actual_fraction = n_censored / int(obs.sum()) if obs.sum() > 0 else 0.0

    return df, {
        "perturbation": "inject_blq",
        "target_fraction": fraction,
        "actual_fraction": actual_fraction,
        "lloq": lloq,
        "n_censored": n_censored,
        "n_observations": int(obs.sum()),
        "method": "M3",
    }


def _remove_absorption_samples(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove observation rows before the absorption time cutoff."""
    cutoff = recipe.absorption_time_cutoff or 2.0
    obs = _obs_mask(df)
    remove_mask = obs & (df["TIME"] < cutoff)
    n_removed = int(remove_mask.sum())

    df = pd.DataFrame(df[~remove_mask]).reset_index(drop=True)

    return df, {
        "perturbation": "remove_absorption_samples",
        "time_cutoff": cutoff,
        "n_removed": n_removed,
    }


def _add_null_covariates(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add random covariates uncorrelated with PK parameters.

    Used to test covariate false positive rate: none of these should
    be selected by the search algorithm.
    """
    names = list(recipe.null_covariate_names)
    n_extra = recipe.null_covariate_n

    for i in range(n_extra):
        names.append(f"NULL_COV_{i + 1}")

    n_subjects = int(df["NMID"].nunique())
    subject_ids = df["NMID"].unique()

    for cov_name in names:
        values = rng.normal(0, 1, size=n_subjects)
        subject_values: dict[Any, float] = dict(zip(subject_ids, values, strict=False))
        df[cov_name] = df["NMID"].map(subject_values)

    return df, {
        "perturbation": "add_null_covariates",
        "covariate_names": names,
        "n_covariates": len(names),
    }


def _inject_outliers(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace random DV values with extreme outlier values.

    For DV near zero, uses an additive offset to avoid zero*magnitude=zero.
    """
    obs = _obs_mask(df)
    obs_indices = df.index[obs].to_numpy()
    fraction = recipe.outlier_fraction or 0.05
    n_outliers = max(1, int(len(obs_indices) * fraction))

    outlier_indices = rng.choice(obs_indices, size=n_outliers, replace=False)
    original_values = df.loc[outlier_indices, "DV"].copy()

    # Multiplicative for non-zero, additive fallback for near-zero DV
    dv_vals = df.loc[outlier_indices, "DV"]
    median_dv = float(df.loc[obs, "DV"].median()) if obs.any() else 1.0
    additive_offset = max(median_dv, 1.0) * recipe.outlier_magnitude
    df.loc[outlier_indices, "DV"] = np.where(
        dv_vals.abs() > 1e-6,
        dv_vals * recipe.outlier_magnitude,
        additive_offset,
    )

    return df, {
        "perturbation": "inject_outliers",
        "target_fraction": fraction,
        "n_outliers": n_outliers,
        "magnitude": recipe.outlier_magnitude,
        "outlier_indices": outlier_indices.tolist(),
        "original_values": original_values.tolist(),
    }


def _sparsify(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Randomly drop observations to reach target obs/subject.

    Preserves dosing events (EVID=1) and at least one observation per subject.
    """
    target = recipe.target_obs_per_subject or 3
    obs = _obs_mask(df)
    dose_rows = df[~obs | (df["EVID"] == 1)]
    obs_df = df[obs]

    keep_indices: list[int] = []
    for _, group in obs_df.groupby("NMID"):
        n = len(group)
        n_keep = min(n, max(1, target))
        kept = rng.choice(group.index.to_numpy(), size=n_keep, replace=False)
        keep_indices.extend(kept.tolist())

    sparse_obs = obs_df.loc[keep_indices]
    df_out = (
        pd.concat([dose_rows, sparse_obs])
        .drop_duplicates()
        .sort_values(by=["NMID", "TIME", "EVID"], ascending=[True, True, False])
        .reset_index(drop=True)
    )

    original_obs = int(obs.sum())
    final_obs = len(sparse_obs)

    return df_out, {
        "perturbation": "sparsify",
        "target_obs_per_subject": target,
        "original_observations": original_obs,
        "final_observations": final_obs,
        "fraction_retained": final_obs / original_obs if original_obs > 0 else 0.0,
    }


def _add_protocol_pooling(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign STUDY_ID labels to simulate multi-study pooled analysis.

    Tests whether the pipeline handles protocol heterogeneity correctly.
    Optionally varies sampling schedules and LLOQ per protocol.

    Real pooled analyses struggle with heterogeneous designs (different
    sampling windows, assays). This scenario tests the orchestrator's
    robustness to protocol differences.
    """
    n_protocols = recipe.n_protocols
    subject_ids = df["NMID"].unique()
    n_subjects = len(subject_ids)

    # #38: the docstring claims "balanced allocation" but the rc8
    # implementation used ``rng.integers(...)`` which is uniform-random
    # — with n_subjects=20, n_protocols=4 you routinely see 3/8/5/4.
    # Use block randomisation: build a shuffled pool of exactly
    # ceil(n_subjects / n_protocols) repeats of each label and assign
    # the first n_subjects entries. This matches the "balanced" claim
    # and makes Suite C results reproducible across platforms.
    assignments_per_label = -(-n_subjects // n_protocols)  # ceil div
    pool = np.tile(np.arange(1, n_protocols + 1), assignments_per_label)[:n_subjects]
    rng.shuffle(pool)
    protocol_assignments = pool
    subject_to_protocol: dict[Any, int] = dict(
        zip(subject_ids, protocol_assignments.tolist(), strict=False)
    )
    df["STUDY_ID"] = df["NMID"].map(subject_to_protocol)

    # Optionally vary sampling schedules per protocol
    n_dropped_obs = 0
    if recipe.vary_sampling:
        all_drop: list[int] = []
        for protocol_id in range(1, n_protocols + 1):
            obs = _obs_mask(df)
            proto_mask = (df["STUDY_ID"] == protocol_id) & obs
            keep_fraction = 0.5 + 0.5 * rng.random()  # 50-100% of timepoints
            proto_indices = df.index[proto_mask].to_numpy()
            n_keep = max(1, int(len(proto_indices) * keep_fraction))
            n_drop = max(0, len(proto_indices) - n_keep)
            if n_drop > 0:
                drop_indices = rng.choice(proto_indices, size=n_drop, replace=False)
                all_drop.extend(drop_indices.tolist())
        n_dropped_obs = len(all_drop)
        df = df.drop(index=all_drop).reset_index(drop=True)

    # Optionally vary LLOQ per protocol
    per_protocol_lloq: dict[int, float] = {}
    if recipe.vary_lloq:
        base_lloq = recipe.lloq or 0.1
        for protocol_id in range(1, n_protocols + 1):
            # Each protocol has a different assay sensitivity (0.5x to 2x)
            factor = 0.5 + 1.5 * rng.random()
            per_protocol_lloq[protocol_id] = base_lloq * factor
        df["PROTOCOL_LLOQ"] = df["STUDY_ID"].map(per_protocol_lloq)

    subjects_per_protocol: dict[int, int] = {}
    for pid in range(1, n_protocols + 1):
        subjects_per_protocol[pid] = int((df["STUDY_ID"] == pid).any())

    return df, {
        "perturbation": "add_protocol_pooling",
        "n_protocols": n_protocols,
        "n_subjects": n_subjects,
        "subjects_per_protocol": {
            str(k): int(v)
            for k, v in zip(
                range(1, n_protocols + 1),
                [int((protocol_assignments == pid).sum()) for pid in range(1, n_protocols + 1)],
                strict=True,
            )
        },
        "vary_sampling": recipe.vary_sampling,
        "vary_lloq": recipe.vary_lloq,
        "n_dropped_obs": n_dropped_obs,
        "per_protocol_lloq": {str(k): v for k, v in per_protocol_lloq.items()},
    }


def _add_occasion_labels(
    df: pd.DataFrame,
    _recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add OCCASION column based on dosing events (vectorized).

    Sorts by (NMID, TIME, EVID descending) first to handle unsorted input.
    Each dose event (EVID==1, AMT>0) increments the occasion counter.
    """
    # Sort to ensure correct temporal ordering
    df = df.sort_values(by=["NMID", "TIME", "EVID"], ascending=[True, True, False]).reset_index(
        drop=True
    )

    # Vectorized: cumulative sum of dose indicators within each subject
    dose_indicator = (df["EVID"] == 1).astype(int)
    if "AMT" in df.columns:
        dose_indicator = dose_indicator & (df["AMT"] > 0).astype(int)

    df["OCCASION"] = df.groupby("NMID")[dose_indicator.name].transform(
        lambda _x: dose_indicator.loc[_x.index].cumsum()
    )
    # Ensure minimum occasion is 1
    df["OCCASION"] = df["OCCASION"].clip(lower=1)

    max_occ = int(df["OCCASION"].max())

    return df, {
        "perturbation": "add_occasion_labels",
        "n_occasions": max_occ,
    }
