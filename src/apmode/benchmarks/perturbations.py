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
        PerturbationType.INJECT_COVARIATE_MISSINGNESS: _inject_covariate_missingness,
        # The four stress-surface perturbations below are *data-side*
        # transforms that imprint a structural-misspecification
        # signature onto observed DV without re-running the forward
        # solve. The manifest documents each transform so a reviewer
        # can attribute the resulting fit shift to the perturbation
        # rather than to genuine dataset behaviour.
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
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — inflate/contract subject-level DV variance.

    Each subject's observed DV vector is multiplied by a single log-normal
    scalar drawn with sigma = ``bsv_scale_factor``. This preserves the
    geometric mean of DV across the population (the multiplier has zero
    log-mean) but inflates between-subject variance — the structural
    fingerprint of misspecified omega. Pure data-side transform: no
    forward solve, no parameter fit, manifest documents the per-subject
    multipliers for audit.
    """
    sigma = recipe.bsv_scale_factor
    if sigma is None:
        msg = "bsv_scale_factor required for SCALE_BSV_VARIANCES"
        raise ValueError(msg)

    obs = _obs_mask(df)
    subject_ids = df["NMID"].unique()
    log_multipliers = rng.normal(loc=0.0, scale=sigma, size=len(subject_ids))
    multipliers = np.exp(log_multipliers)
    subject_to_mult: dict[Any, float] = dict(zip(subject_ids, multipliers.tolist(), strict=True))
    factor = df["NMID"].map(subject_to_mult).astype(float)
    df.loc[obs, "DV"] = (df.loc[obs, "DV"].astype(float) * factor.loc[obs]).astype(float)

    return df, {
        "perturbation": "scale_bsv_variances",
        "bsv_scale_factor": float(sigma),
        "n_subjects": len(subject_ids),
        "multiplier_geomean": float(np.exp(float(log_multipliers.mean()))),
        "multiplier_geosd": (
            float(np.exp(float(log_multipliers.std(ddof=1)))) if len(log_multipliers) > 1 else 1.0
        ),
    }


def _saturate_clearance(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — bend DV to look like Michaelis-Menten clearance.

    Applies a saturable correction factor to observed DV so that high
    concentrations are inflated relative to low concentrations — the
    structural fingerprint a 1-cmt linear fit would miss when truth is
    saturable elimination. The correction is the reciprocal of the
    unitless saturation efficiency at the observed DV:

        DV_perturbed = DV * (1 + DV / saturation_km)

    capped at ``saturation_vmax`` (interpreted here as a concentration
    ceiling, so the perturbation cannot drive observations beyond a
    physiologically plausible level). This is a data-side stress test
    that produces an MM-shape signature without re-simulating; the
    manifest records the parameters so reviewers can distinguish the
    perturbation contribution from any genuine MM behaviour in the
    underlying dataset.
    """
    km = recipe.saturation_km
    vmax = recipe.saturation_vmax
    if km is None or vmax is None:
        msg = "saturation_km and saturation_vmax required for SATURATE_CLEARANCE"
        raise ValueError(msg)

    obs = _obs_mask(df)
    dv = df.loc[obs, "DV"].astype(float)
    perturbed = dv * (1.0 + dv / km)
    perturbed = perturbed.clip(upper=vmax)
    df.loc[obs, "DV"] = perturbed

    return df, {
        "perturbation": "saturate_clearance",
        "saturation_km": float(km),
        "saturation_vmax": float(vmax),
        "n_observations_perturbed": int(obs.sum()),
        "max_observed_perturbed_dv": float(perturbed.max()) if not perturbed.empty else 0.0,
    }


def _tmdd_perturbation(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — depress low-DV samples to mimic TMDD QSS.

    The QSS approximation of target-mediated drug disposition produces
    nonlinear PK at low concentrations: target binding sequesters drug
    and the observed (free) concentration is reduced. The signature on
    a measured-total assay is the opposite — saturation at high
    concentrations and depression at low concentrations relative to
    linear PK. We approximate by subtracting a saturable target-binding
    term:

        DV_perturbed = max(0, DV - tmdd_r0 * DV / (DV + tmdd_kss))

    so at DV ≪ tmdd_kss roughly ``tmdd_r0`` units of drug are bound and
    removed; at DV ≫ tmdd_kss the binding saturates and the depression
    becomes negligible. ``tmdd_r0`` should be set to a small fraction of
    the baseline DV scale so the transform is a stress signature, not a
    near-zero observation cliff. Pure data-side transform.
    """
    kss = recipe.tmdd_kss
    r0 = recipe.tmdd_r0
    if kss is None or r0 is None:
        msg = "tmdd_kss and tmdd_r0 required for TMDD"
        raise ValueError(msg)

    obs = _obs_mask(df)
    dv = df.loc[obs, "DV"].astype(float)
    bound = r0 * dv / (dv + kss)
    perturbed = (dv - bound).clip(lower=0.0)
    df.loc[obs, "DV"] = perturbed

    return df, {
        "perturbation": "tmdd",
        "tmdd_kss": float(kss),
        "tmdd_r0": float(r0),
        "n_observations_perturbed": int(obs.sum()),
        "mean_bound_fraction": (
            float((bound / dv.replace(0.0, np.nan)).mean()) if not dv.empty else 0.0
        ),
    }


def _flip_flop_perturbation(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    _rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """#26 stress surface — apply a flip-flop time-decay signature.

    True flip-flop kinetics arise when the absorption rate is slower
    than elimination, causing the apparent terminal half-life to
    reflect ``ka`` rather than ``ke``. Without re-simulating, we apply
    a time-dependent multiplicative bias to DV that lengthens the
    apparent terminal phase by a factor controlled by
    ``flip_flop_ke_ratio``:

        DV_perturbed(t) = DV(t) * exp(-(flip_flop_ke_ratio - 1)
                                       * flip_flop_ka * max(0, t - 1/ka))

    For ``flip_flop_ke_ratio < 1`` the terminal phase is stretched
    (the flip-flop signature); for > 1 it is compressed. ``flip_flop_ka``
    controls when the post-absorption regime begins. The early phase is
    untouched (factor ≈ 1) so absorption-phase samples remain
    informative for ka recovery. Pure data-side transform.
    """
    ka = recipe.flip_flop_ka
    ratio = recipe.flip_flop_ke_ratio
    if ka is None or ratio is None:
        msg = "flip_flop_ka and flip_flop_ke_ratio required for FLIP_FLOP"
        raise ValueError(msg)

    obs = _obs_mask(df)
    t = df.loc[obs, "TIME"].astype(float)
    delay = 1.0 / ka
    elapsed = (t - delay).clip(lower=0.0)
    log_factor = -(ratio - 1.0) * ka * elapsed
    factor = np.exp(log_factor)
    df.loc[obs, "DV"] = df.loc[obs, "DV"].astype(float) * factor

    return df, {
        "perturbation": "flip_flop",
        "flip_flop_ka": float(ka),
        "flip_flop_ke_ratio": float(ratio),
        "n_observations_perturbed": int(obs.sum()),
        "post_absorption_threshold_h": float(delay),
    }


def _inject_covariate_missingness(
    df: pd.DataFrame,
    recipe: PerturbationRecipe,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """PRD §5 — inject MCAR covariate missingness at the subject level.

    Drops a fraction of subject-level covariate values to NaN. Operates
    one decision per subject (not per row) so the missingness pattern is
    realistic for a population PK design where covariates are recorded
    once at enrolment and then carried forward. ``covariate_missingness_columns``
    selects which columns to perturb; if empty, all columns that are
    constant within subject and not part of the canonical NONMEM schema
    (``NMID``/``ID``/``TIME``/``DV``/``EVID``/``AMT``/``MDV``/``CENS``/
    ``BLQ_FLAG``/``LLOQ``/``DVID``/``OCCASION``/``STUDY_ID``) are
    treated as candidate covariates.
    """
    fraction = recipe.covariate_missingness_fraction
    if fraction is None:
        msg = "covariate_missingness_fraction required for INJECT_COVARIATE_MISSINGNESS"
        raise ValueError(msg)

    canonical_cols = {
        "NMID",
        "ID",
        "TIME",
        "DV",
        "EVID",
        "AMT",
        "MDV",
        "CENS",
        "BLQ_FLAG",
        "LLOQ",
        "DVID",
        "OCCASION",
        "STUDY_ID",
        "PROTOCOL_LLOQ",
    }
    if recipe.covariate_missingness_columns:
        cov_columns: list[str] = [
            c for c in recipe.covariate_missingness_columns if c in df.columns
        ]
    else:
        cov_columns = []
        for col in df.columns:
            if col in canonical_cols:
                continue
            constant_within_subject = df.groupby("NMID")[col].nunique(dropna=False).max() <= 1
            if bool(constant_within_subject):
                cov_columns.append(col)

    n_dropped_total = 0
    per_column_drops: dict[str, int] = {}
    subject_ids = df["NMID"].unique()
    n_subjects = len(subject_ids)
    n_drop = round(n_subjects * fraction)

    for col in cov_columns:
        drop_subjects = rng.choice(subject_ids, size=n_drop, replace=False)
        mask = df["NMID"].isin(drop_subjects)
        n_rows = int(mask.sum())
        n_dropped_total += n_rows
        per_column_drops[col] = n_rows
        df.loc[mask, col] = np.nan

    return df, {
        "perturbation": "inject_covariate_missingness",
        "covariate_missingness_fraction": float(fraction),
        "covariate_columns": cov_columns,
        "n_subjects_dropped_per_column": int(n_drop),
        "n_rows_dropped_total": int(n_dropped_total),
        "per_column_n_rows_dropped": per_column_drops,
    }


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

    # Recompute subject counts from the *current* frame so the manifest
    # reflects post-vary_sampling state (dropping rows can leave a
    # protocol with zero retained observations on a small dataset).
    subjects_per_protocol_post: dict[int, int] = {
        pid: int(df.loc[df["STUDY_ID"] == pid, "NMID"].nunique())
        for pid in range(1, n_protocols + 1)
    }

    return df, {
        "perturbation": "add_protocol_pooling",
        "n_protocols": n_protocols,
        "n_subjects": n_subjects,
        "subjects_per_protocol_assigned": {
            str(pid): int((protocol_assignments == pid).sum()) for pid in range(1, n_protocols + 1)
        },
        "subjects_per_protocol_post_drop": {
            str(pid): subjects_per_protocol_post[pid] for pid in range(1, n_protocols + 1)
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

    # Vectorized: cumulative sum of dose indicators within each subject.
    # Group on the dose-flag column directly (not on EVID) so the transform's
    # source column always matches the indicator semantics — even when AMT
    # filtering changes which rows count as a dose. Earlier revisions grouped
    # on ``dose_indicator.name`` (which equals "EVID") and then re-indexed
    # via ``.loc`` inside the lambda; that worked only because the lambda
    # discarded the EVID values, and would break under any refactor that
    # disturbed ``dose_indicator.name``.
    dose_flag = (df["EVID"] == 1).astype(int)
    if "AMT" in df.columns:
        dose_flag = dose_flag & (df["AMT"] > 0).astype(int)
    df["_DOSE_FLAG"] = dose_flag.astype(int)
    df["OCCASION"] = df.groupby("NMID")["_DOSE_FLAG"].cumsum()
    df = df.drop(columns=["_DOSE_FLAG"])
    # Ensure minimum occasion is 1
    df["OCCASION"] = df["OCCASION"].clip(lower=1)

    max_occ = int(df["OCCASION"].max())

    return df, {
        "perturbation": "add_occasion_labels",
        "n_occasions": max_occ,
    }
