# SPDX-License-Identifier: GPL-2.0-or-later
"""Profiler-policy consistency tests.

Guards against drift between the JSON source of truth
(``policies/profiler.json``) and the Python derivatives:

  1. ``ProfilerPolicy`` dataclass fields (``apmode/data/policy.py``)
  2. Module-level ``_POLICY.*`` constants in ``apmode/data/profiler.py``
  3. AST-level check: the drift-prone heuristic functions may not contain
     bare numeric literals (except an allowlist of numerical-stability
     floors and loop bounds).

Background
----------
Pre-v2.1 the profiler accumulated bare literals (``median_r2 >= 0.85``,
``median_samples >= 8``, ``dv_cv_percent < 30.0``, …) that mirrored
values already present in the policy JSON but were not actually sourced
from it. Policy tuning silently had no effect. This test file locks the
contract down so any future drift fails CI before it lands.

If a new policy field is added:

  * add it to ``policies/profiler.json``;
  * add the matching ``ProfilerPolicy`` field in ``apmode/data/policy.py``
    with a loader entry in ``_load_from_path``;
  * add a derived module-level constant in ``apmode/data/profiler.py``
    next to its siblings; and
  * use the constant at every call site instead of a bare literal.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from apmode.data import profiler as profiler_module
from apmode.data.policy import ProfilerPolicy, _load_from_path, reload_policy

_REPO_ROOT = Path(__file__).resolve().parents[2]
_POLICY_PATH = _REPO_ROOT / "policies" / "profiler.json"
_PROFILER_PATH = _REPO_ROOT / "src" / "apmode" / "data" / "profiler.py"


# ---------------------------------------------------------------------------
# 1. JSON ↔ dataclass completeness
# ---------------------------------------------------------------------------


def _flatten_json(data: dict[str, object], prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in data.items():
        if key.startswith("_") or key in {"$schema", "description"}:
            continue
        full = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            out.update(_flatten_json(value, full))
        else:
            out[full] = value
    return out


# Mapping from JSON dotted key → ProfilerPolicy field name. One entry per
# active policy setting; fields here are REQUIRED to be both loaded into
# the dataclass AND consumed somewhere in the profiler (test 3 below).
_JSON_TO_DATACLASS: dict[str, str] = {
    "covariate.correlation_threshold_abs_r": "covariate_correlation_threshold_abs_r",
    "covariate.missingness_full_information_cutoff": (
        "covariate_missingness_full_information_cutoff"
    ),
    "nonlinear_clearance.mm_curvature_ratio": "mm_curvature_ratio",
    "nonlinear_clearance.tmdd_curvature_ratio": "tmdd_curvature_ratio",
    "nonlinear_clearance.compartmentality_curvature_ratio": "compartmentality_curvature_ratio",
    "nonlinear_clearance.terminal_monoexp_r2_linear_threshold": (
        "terminal_monoexp_r2_linear_threshold"
    ),
    "smith_2000_dose_proportionality.theta_low": "smith_theta_low",
    "smith_2000_dose_proportionality.theta_high": "smith_theta_high",
    "smith_2000_dose_proportionality.min_dose_levels": "smith_min_dose_levels",
    "smith_2000_dose_proportionality.min_dose_ratio": "smith_min_dose_ratio",
    "huang_2025_lambda_z.min_points": "lambdaz_min_points",
    "huang_2025_lambda_z.tolerance": "lambdaz_tolerance",
    "huang_2025_lambda_z.phoenix_constraint": "lambdaz_phoenix_constraint",
    "huang_2025_lambda_z.adj_r2_threshold": "lambdaz_adj_r2_threshold",
    "steady_state.n_half_lives_required": "ss_n_half_lives_required",
    "steady_state.n_doses_alt": "ss_n_doses_alt",
    "steady_state.interval_tolerance": "ss_interval_tolerance",
    "steady_state.dose_tolerance": "ss_dose_tolerance",
    "steady_state.min_doses": "ss_min_doses",
    "shape_detection.multi_peak_fraction_threshold": "multi_peak_fraction_threshold",
    "shape_detection.lag_signature_fraction_threshold": "lag_signature_fraction_threshold",
    "shape_detection.lag_early_conc_fraction": "lag_early_conc_fraction",
    "shape_detection.lag_early_time_percentile": "lag_early_time_percentile",
    "shape_detection.peak_prominence_range_fraction": "peak_prominence_range_fraction",
    "shape_detection.peak_prominence_cmax_floor": "peak_prominence_cmax_floor",
    "shape_detection.peak_min_distance_intervals": "peak_min_distance_intervals",
    "subject_quality.min_subjects_for_median": "min_subjects_for_median",
    "subject_quality.min_concs_for_profile": "min_concs_for_profile",
    "subject_quality.min_subjects_for_dynamic_range": "min_subjects_for_dynamic_range",
    "subject_quality.min_obs_per_subject_rich": "min_obs_per_subject_rich",
    "subject_quality.min_obs_per_subject_moderate": "min_obs_per_subject_moderate",
    "subject_quality.absorption_coverage_min_pre_tmax": "absorption_coverage_min_pre_tmax",
    "subject_quality.elimination_coverage_min_post_tmax": "elimination_coverage_min_post_tmax",
    "error_model.blq_m3_trigger": "blq_m3_trigger",
    "error_model.dynamic_range_proportional": "dynamic_range_proportional",
    "error_model.high_cv_ceiling": "high_cv_ceiling",
    "error_model.lloq_cmax_combined": "lloq_cmax_combined",
    "error_model.terminal_log_mad_combined": "terminal_log_mad_combined",
    "error_model.narrow_range_additive": "narrow_range_additive",
    "error_model.low_cv_additive_ceiling": "low_cv_additive_ceiling",
    "node_readiness.min_subjects.discovery": "node_discovery_min_subjects",
    "node_readiness.min_subjects.optimization": "node_optimization_min_subjects",
    "node_readiness.min_median_samples.discovery": "node_discovery_min_median_samples",
    "node_readiness.min_median_samples.optimization": "node_optimization_min_median_samples",
    "node_readiness.dim_budget.discovery": "node_discovery_budget",
    "node_readiness.dim_budget.optimization": "node_optimization_budget",
    "flip_flop.ka_lambdaz_ratio_likely": "flip_flop_ka_lambdaz_ratio_likely",
    "flip_flop.ka_lambdaz_ratio_possible": "flip_flop_ka_lambdaz_ratio_possible",
    "flip_flop.quality_adj_r2_min": "flip_flop_quality_adj_r2_min",
    "flip_flop.quality_min_npts": "flip_flop_quality_min_npts",
    "tad_consistency.in_window_fraction_clean": "tad_in_window_fraction_clean",
    "protocol_heterogeneity.obs_per_subject_cv_threshold": (
        "protocol_heterogeneity_obs_per_subject_cv_threshold"
    ),
    "dvid_filter.fail_open_when_no_match": "dvid_fail_open_when_no_match",
}


def test_every_active_json_field_is_loaded_into_dataclass() -> None:
    """Every non-metadata JSON leaf must map into a ProfilerPolicy field."""
    raw = json.loads(_POLICY_PATH.read_text())
    flat = _flatten_json(raw)
    # Non-threshold metadata plus list/allowlist fields have bespoke loading
    # paths and are validated separately by other tests.
    skip_prefixes = ("policy_id", "policy_version", "schema_version", "manifest_schema_version")
    skip_keys = {"dvid_filter.pk_dvid_allowlist"}
    missing: list[str] = []
    for key in flat:
        if any(key.startswith(p) for p in skip_prefixes):
            continue
        if key in skip_keys:
            continue
        if key not in _JSON_TO_DATACLASS:
            missing.append(key)
    assert not missing, (
        f"JSON policy fields not mapped into ProfilerPolicy: {missing}. "
        "Add an entry to _JSON_TO_DATACLASS and the dataclass/loader."
    )


def test_every_mapped_field_exists_on_dataclass() -> None:
    field_names = {f.name for f in fields(ProfilerPolicy)}
    unknown = [f for f in _JSON_TO_DATACLASS.values() if f not in field_names]
    assert not unknown, f"_JSON_TO_DATACLASS lists fields not on ProfilerPolicy: {unknown}"


def test_loader_values_match_json() -> None:
    raw = json.loads(_POLICY_PATH.read_text())
    flat = _flatten_json(raw)
    policy = _load_from_path(_POLICY_PATH)
    mismatches: list[str] = []
    for json_key, dc_field in _JSON_TO_DATACLASS.items():
        json_val = flat[json_key]
        dc_val = getattr(policy, dc_field)
        # bools inherit from int in Python; coerce by type to avoid
        # 1 == True passing when the schema says bool
        if isinstance(dc_val, bool) != isinstance(json_val, bool):
            mismatches.append(f"{json_key}: type mismatch ({type(json_val)} vs {type(dc_val)})")
            continue
        if (
            float(dc_val) != float(json_val)
            if isinstance(dc_val, (int, float))
            else dc_val != json_val
        ):
            mismatches.append(f"{json_key}: JSON={json_val!r} vs dataclass={dc_val!r}")
    assert not mismatches, "\n".join(mismatches)


# ---------------------------------------------------------------------------
# 2. Module-level constants derived from _POLICY
# ---------------------------------------------------------------------------

# Each entry: (profiler constant name, ProfilerPolicy field name). All entries
# must satisfy ``getattr(profiler, name) == getattr(policy, field)``.
_CONSTANTS_TO_POLICY: list[tuple[str, str]] = [
    ("_COVARIATE_CORRELATION_THRESHOLD", "covariate_correlation_threshold_abs_r"),
    ("_NONLINEAR_MM_CURVATURE_RATIO", "mm_curvature_ratio"),
    ("_COMPARTMENTALITY_CURVATURE_RATIO", "compartmentality_curvature_ratio"),
    ("_NONLINEAR_TMDD_CURVATURE_RATIO", "tmdd_curvature_ratio"),
    ("_MULTI_PEAK_FRACTION_THRESHOLD", "multi_peak_fraction_threshold"),
    ("_LAG_SIGNATURE_FRACTION_THRESHOLD", "lag_signature_fraction_threshold"),
    ("_BLQ_COVARIATE_MISSINGNESS_CUTOFF", "covariate_missingness_full_information_cutoff"),
    ("_TERMINAL_MONOEXP_R2_LINEAR_THRESHOLD", "terminal_monoexp_r2_linear_threshold"),
    ("_TERMINAL_FIT_MIN_POINTS", "lambdaz_min_points"),
    ("_PEAK_PROMINENCE_RANGE_FRACTION", "peak_prominence_range_fraction"),
    ("_PEAK_PROMINENCE_CMAX_FLOOR", "peak_prominence_cmax_floor"),
    ("_PEAK_MIN_DISTANCE_INTERVALS", "peak_min_distance_intervals"),
    ("_LAMBDAZ_ADJ_R2_THRESHOLD", "lambdaz_adj_r2_threshold"),
    ("_MIN_OBS_PER_SUBJECT_RICH", "min_obs_per_subject_rich"),
    ("_MIN_OBS_PER_SUBJECT_MODERATE", "min_obs_per_subject_moderate"),
    ("_ABSORPTION_COVERAGE_MIN_PRE_TMAX", "absorption_coverage_min_pre_tmax"),
    ("_ELIMINATION_COVERAGE_MIN_POST_TMAX", "elimination_coverage_min_post_tmax"),
    ("_NODE_DISCOVERY_MIN_SUBJECTS", "node_discovery_min_subjects"),
    ("_NODE_DISCOVERY_MIN_MEDIAN_SAMPLES", "node_discovery_min_median_samples"),
    ("_NODE_DISCOVERY_BUDGET", "node_discovery_budget"),
    ("_NODE_OPTIMIZATION_MIN_SUBJECTS", "node_optimization_min_subjects"),
    ("_NODE_OPTIMIZATION_MIN_MEDIAN_SAMPLES", "node_optimization_min_median_samples"),
    ("_NODE_OPTIMIZATION_BUDGET", "node_optimization_budget"),
    ("_FLIP_FLOP_KA_LAMBDAZ_RATIO_POSSIBLE", "flip_flop_ka_lambdaz_ratio_possible"),
    ("_FLIP_FLOP_QUALITY_ADJ_R2_MIN", "flip_flop_quality_adj_r2_min"),
    ("_FLIP_FLOP_QUALITY_MIN_NPTS", "flip_flop_quality_min_npts"),
    (
        "_PROTOCOL_HETEROGENEITY_CV_THRESHOLD",
        "protocol_heterogeneity_obs_per_subject_cv_threshold",
    ),
    ("_LOW_CV_ADDITIVE_CEILING", "low_cv_additive_ceiling"),
]


@pytest.mark.parametrize("const_name,policy_field", _CONSTANTS_TO_POLICY)
def test_profiler_constant_matches_policy(const_name: str, policy_field: str) -> None:
    # Reload the policy so a mid-session override (e.g. from another test
    # monkey-patching profiler_policy_path) does not taint this invariant.
    policy = reload_policy()
    const_value = getattr(profiler_module, const_name)
    policy_value = getattr(policy, policy_field)
    assert const_value == policy_value, (
        f"profiler.{const_name} ({const_value!r}) drifts from "
        f"ProfilerPolicy.{policy_field} ({policy_value!r})"
    )


# ---------------------------------------------------------------------------
# 3. AST-level drift guard: no bare numeric literals in drift-prone functions
# ---------------------------------------------------------------------------

# Allowlist: these constant values are legitimately "not policy" — numerical
# stability floors, loop bounds, pure array indices, and small enum-like ints
# (e.g. empty/sentinel counts). Updating the allowlist is a deliberate design
# step and should be rare; prefer adding a new policy field.
_LITERAL_ALLOWLIST: set[float] = {
    # pure counts / loop bounds / array indices
    0,
    1,
    2,
    3,
    4,
    5,
    # small integer thresholds that are structural, not policy (eg "need at
    # least N raw samples for any fit at all" is a math requirement, not a
    # tunable policy knob).
    -1,
    # display conversions fraction→percent; not a threshold.
    100,
    100.0,
    # numerical-stability floors
    1e-6,
    1e-10,
}

# Functions whose body must contain no off-allowlist numeric literals.
_GUARDED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "_compute_node_dim_budget",
        "_classify_richness",
        "_assess_identifiability",
        "_assess_absorption_coverage",
        "_assess_elimination_coverage",
        "_assess_flip_flop_risk",
        "_assess_protocol_heterogeneity",
        "recommend_error_model",
    }
)


def _collect_numeric_literals(node: ast.AST) -> list[tuple[int, float]]:
    """Walk *node* and return (lineno, value) for every off-allowlist literal.

    We treat ``ast.Constant`` with numeric ``value`` as a literal. Boolean
    constants ``True``/``False`` are excluded (``isinstance(x, bool)`` is
    True even though they subclass int, and they are not magic numbers).
    """
    hits: list[tuple[int, float]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Constant):
            continue
        value = child.value
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        if value in _LITERAL_ALLOWLIST:
            continue
        hits.append((child.lineno, float(value)))
    return hits


def test_drift_prone_functions_have_no_bare_literals() -> None:
    source = _PROFILER_PATH.read_text()
    tree = ast.parse(source)
    offenders: dict[str, list[tuple[int, float]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in _GUARDED_FUNCTIONS:
            continue
        hits = _collect_numeric_literals(node)
        # The function docstring may be present as ast.Constant (str) but is
        # filtered out above. Compound literals inside type annotations in
        # the signature — e.g. ``int = 3`` — would still trip. Accept
        # default values on parameters as they are part of the signature,
        # not the implementation: walk only the body.
        body_hits = [
            (ln, v)
            for (ln, v) in hits
            if any(ln >= stmt.lineno for stmt in node.body if hasattr(stmt, "lineno"))
        ]
        if body_hits:
            offenders[node.name] = body_hits
    assert not offenders, (
        "Bare numeric literals found in drift-prone functions "
        "(add a policy field or import an existing _POLICY.* constant):\n"
        + "\n".join(
            f"  {func}:  " + ", ".join(f"line {ln} value={v}" for ln, v in offenders[func])
            for func in offenders
        )
    )


# ---------------------------------------------------------------------------
# 4. Policy-file self-consistency
# ---------------------------------------------------------------------------


def test_policy_version_matches_policy_id() -> None:
    raw = json.loads(_POLICY_PATH.read_text())
    pid = raw.get("policy_id", "")
    version = raw.get("policy_version", "")
    match = re.match(r"profiler/v([0-9.]+)$", pid)
    assert match, f"policy_id must match 'profiler/v<version>', got {pid!r}"
    assert match.group(1) == version, (
        f"policy_id {pid!r} and policy_version {version!r} disagree — bump both together."
    )


def test_profiler_refinement_plan_exists() -> None:
    """The profiler JSON references ``docs/PROFILER_REFINEMENT_PLAN.md`` for
    derivation/citation; ensure the file is present so the link is not
    dangling."""
    plan = _REPO_ROOT / "docs" / "PROFILER_REFINEMENT_PLAN.md"
    assert plan.is_file(), (
        f"{plan.relative_to(_REPO_ROOT)} is referenced from "
        "policies/profiler.json but does not exist."
    )
