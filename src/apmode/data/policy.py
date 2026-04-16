# SPDX-License-Identifier: GPL-2.0-or-later
"""Profiler policy loader.

Reads  and exposes a typed, frozen ``ProfilerPolicy`` dataclass.
The profiler module consumes constants through this object rather than
hardcoded floats so bundle consumers can:

  1. Replay a run exactly by checking out ``policy_sha256`` and
     ``policy_id`` from the emitted EvidenceManifest.
  2. Tune thresholds per deployment without touching Python source.
  3. Audit every dispatch decision against a versioned artifact.

The JSON file is the source of truth. Out-of-sync named constants in
``profiler.py`` trigger a test failure via
``tests/unit/test_profiler_policy_consistency.py``.

This module is kept pure (no side effects on import other than reading
the JSON once). The resolved policy is cached in a module-level
``_POLICY`` variable; ``reload_policy`` forces a re-read (useful only
in tests).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_POLICY_PATH = Path(__file__).resolve().parents[3] / "policies" / "profiler.json"


@dataclass(frozen=True)
class ProfilerPolicy:
    """Frozen, typed view of ``policies/profiler.json``.

    Fields mirror the JSON structure with snake_case names. When adding
    a new threshold to the JSON, add the matching field here AND update
    ``tests/unit/test_profiler_policy_consistency.py``.
    """

    policy_id: str
    policy_version: str
    policy_sha256: str
    schema_version: int
    manifest_schema_version: int

    # Covariate
    covariate_correlation_threshold_abs_r: float
    covariate_missingness_full_information_cutoff: float

    # Nonlinear clearance
    mm_curvature_ratio: float
    tmdd_curvature_ratio: float
    compartmentality_curvature_ratio: float
    terminal_monoexp_r2_linear_threshold: float

    # Smith 2000 dose-proportionality
    smith_theta_low: float
    smith_theta_high: float
    smith_min_dose_levels: int
    smith_min_dose_ratio: float

    # Huang 2025 lambda_z
    lambdaz_min_points: int
    lambdaz_tolerance: float
    lambdaz_phoenix_constraint: bool
    lambdaz_adj_r2_threshold: float

    # Steady-state
    ss_n_half_lives_required: int
    ss_n_doses_alt: int
    ss_interval_tolerance: float
    ss_dose_tolerance: float
    ss_min_doses: int

    # Shape detection
    multi_peak_fraction_threshold: float
    lag_signature_fraction_threshold: float
    lag_early_conc_fraction: float
    lag_early_time_percentile: int
    peak_prominence_range_fraction: float
    peak_prominence_cmax_floor: float
    peak_min_distance_intervals: float

    # Subject quality
    min_subjects_for_median: int
    min_concs_for_profile: int
    min_subjects_for_dynamic_range: int
    min_obs_per_subject_rich: int
    min_obs_per_subject_moderate: int
    absorption_coverage_min_pre_tmax: float
    elimination_coverage_min_post_tmax: float

    # Error model
    blq_m3_trigger: float
    dynamic_range_proportional: float
    high_cv_ceiling: float
    lloq_cmax_combined: float
    terminal_log_mad_combined: float
    narrow_range_additive: float
    low_cv_additive_ceiling: float

    # NODE readiness
    node_discovery_min_subjects: int
    node_optimization_min_subjects: int
    node_discovery_min_median_samples: int
    node_optimization_min_median_samples: int
    node_discovery_budget: int
    node_optimization_budget: int

    # Flip-flop
    flip_flop_ka_lambdaz_ratio_likely: float
    flip_flop_ka_lambdaz_ratio_possible: float
    # Quality guards for flip-flop classification (advisory — Richardson 2025
    # stricter than routine λz; terminal fits below this threshold cannot
    # support a "likely" flip-flop call).
    flip_flop_quality_adj_r2_min: float
    flip_flop_quality_min_npts: int

    # TAD consistency
    tad_in_window_fraction_clean: float

    # Protocol heterogeneity
    protocol_heterogeneity_obs_per_subject_cv_threshold: float

    # DVID filter
    pk_dvid_allowlist: frozenset[str]
    dvid_fail_open_when_no_match: bool


def _sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_from_path(path: Path) -> ProfilerPolicy:
    raw = json.loads(path.read_text())
    cov = raw["covariate"]
    nlc = raw["nonlinear_clearance"]
    smith = raw["smith_2000_dose_proportionality"]
    lz = raw["huang_2025_lambda_z"]
    ss = raw["steady_state"]
    shape = raw["shape_detection"]
    sq = raw["subject_quality"]
    err = raw["error_model"]
    node = raw["node_readiness"]
    ff = raw["flip_flop"]
    tad = raw["tad_consistency"]
    dvid = raw["dvid_filter"]
    ph = raw.get("protocol_heterogeneity", {})
    return ProfilerPolicy(
        policy_id=raw["policy_id"],
        policy_version=raw["policy_version"],
        policy_sha256=_sha256_of_file(path),
        schema_version=int(raw["schema_version"]),
        manifest_schema_version=int(raw["manifest_schema_version"]),
        covariate_correlation_threshold_abs_r=float(cov["correlation_threshold_abs_r"]),
        covariate_missingness_full_information_cutoff=float(
            cov["missingness_full_information_cutoff"]
        ),
        mm_curvature_ratio=float(nlc["mm_curvature_ratio"]),
        tmdd_curvature_ratio=float(nlc["tmdd_curvature_ratio"]),
        compartmentality_curvature_ratio=float(nlc["compartmentality_curvature_ratio"]),
        terminal_monoexp_r2_linear_threshold=float(nlc["terminal_monoexp_r2_linear_threshold"]),
        smith_theta_low=float(smith["theta_low"]),
        smith_theta_high=float(smith["theta_high"]),
        smith_min_dose_levels=int(smith["min_dose_levels"]),
        smith_min_dose_ratio=float(smith["min_dose_ratio"]),
        lambdaz_min_points=int(lz["min_points"]),
        lambdaz_tolerance=float(lz["tolerance"]),
        lambdaz_phoenix_constraint=bool(lz["phoenix_constraint"]),
        lambdaz_adj_r2_threshold=float(lz["adj_r2_threshold"]),
        ss_n_half_lives_required=int(ss["n_half_lives_required"]),
        ss_n_doses_alt=int(ss["n_doses_alt"]),
        ss_interval_tolerance=float(ss["interval_tolerance"]),
        ss_dose_tolerance=float(ss["dose_tolerance"]),
        ss_min_doses=int(ss["min_doses"]),
        multi_peak_fraction_threshold=float(shape["multi_peak_fraction_threshold"]),
        lag_signature_fraction_threshold=float(shape["lag_signature_fraction_threshold"]),
        lag_early_conc_fraction=float(shape["lag_early_conc_fraction"]),
        lag_early_time_percentile=int(shape["lag_early_time_percentile"]),
        peak_prominence_range_fraction=float(shape["peak_prominence_range_fraction"]),
        peak_prominence_cmax_floor=float(shape["peak_prominence_cmax_floor"]),
        peak_min_distance_intervals=float(shape["peak_min_distance_intervals"]),
        min_subjects_for_median=int(sq["min_subjects_for_median"]),
        min_concs_for_profile=int(sq["min_concs_for_profile"]),
        min_subjects_for_dynamic_range=int(sq["min_subjects_for_dynamic_range"]),
        min_obs_per_subject_rich=int(sq["min_obs_per_subject_rich"]),
        min_obs_per_subject_moderate=int(sq["min_obs_per_subject_moderate"]),
        absorption_coverage_min_pre_tmax=float(sq["absorption_coverage_min_pre_tmax"]),
        elimination_coverage_min_post_tmax=float(sq["elimination_coverage_min_post_tmax"]),
        blq_m3_trigger=float(err["blq_m3_trigger"]),
        dynamic_range_proportional=float(err["dynamic_range_proportional"]),
        high_cv_ceiling=float(err["high_cv_ceiling"]),
        lloq_cmax_combined=float(err["lloq_cmax_combined"]),
        terminal_log_mad_combined=float(err["terminal_log_mad_combined"]),
        narrow_range_additive=float(err["narrow_range_additive"]),
        low_cv_additive_ceiling=float(err["low_cv_additive_ceiling"]),
        node_discovery_min_subjects=int(node["min_subjects"]["discovery"]),
        node_optimization_min_subjects=int(node["min_subjects"]["optimization"]),
        node_discovery_min_median_samples=int(node["min_median_samples"]["discovery"]),
        node_optimization_min_median_samples=int(node["min_median_samples"]["optimization"]),
        node_discovery_budget=int(node["dim_budget"]["discovery"]),
        node_optimization_budget=int(node["dim_budget"]["optimization"]),
        flip_flop_ka_lambdaz_ratio_likely=float(ff["ka_lambdaz_ratio_likely"]),
        flip_flop_ka_lambdaz_ratio_possible=float(ff["ka_lambdaz_ratio_possible"]),
        flip_flop_quality_adj_r2_min=float(ff.get("quality_adj_r2_min", 0.85)),
        flip_flop_quality_min_npts=int(ff.get("quality_min_npts", 4)),
        tad_in_window_fraction_clean=float(tad["in_window_fraction_clean"]),
        protocol_heterogeneity_obs_per_subject_cv_threshold=float(
            ph.get("obs_per_subject_cv_threshold", 0.5)
        ),
        pk_dvid_allowlist=frozenset(str(x).lower() for x in dvid["pk_dvid_allowlist"]),
        dvid_fail_open_when_no_match=bool(dvid["fail_open_when_no_match"]),
    )


_POLICY: ProfilerPolicy | None = None


def get_policy() -> ProfilerPolicy:
    """Return the cached profiler policy, loading it on first call."""
    global _POLICY
    if _POLICY is None:
        _POLICY = _load_from_path(_POLICY_PATH)
    return _POLICY


def reload_policy() -> ProfilerPolicy:
    """Force a re-read of ``policies/profiler.json`` (tests only)."""
    global _POLICY
    _POLICY = _load_from_path(_POLICY_PATH)
    return _POLICY
