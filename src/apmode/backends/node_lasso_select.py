# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE-LASSO structural selection in derivative space (Bräm 2025).

Replaces ad-hoc ``curve_fit`` distillation with LASSO selection over a candidate
library. The trained NODE's neural RHS is evaluated directly over the observed
input support (per compartment; no ODE re-solving beyond the initial NODE fit),
and a penalized regression selects which PMX candidate shapes reconstruct the
learned rate law (Bräm et al. 2025, "Automated Pharmacometric Model Development by
Leveraging Low-Dimensional Neural ODEs and LASSO Regression", CPT:PSP,
doi:10.1002/psp4.70285).

Correctness pipeline (validated design):
  1. Standardize candidate columns so L1 selects on model quality, not raw scale.
  2. Group near-duplicate (collinear grid) columns so a coefficient is not split
     arbitrarily across near-identical shapes.
  3. LassoLarsIC(criterion='bic') for an initial support and a BIC score.
  4. Stability selection: bootstrap the derivative rows ``n_bootstrap`` times,
     refit LassoLarsIC each, keep terms selected in >= ``stability_threshold`` of
     bootstraps — guards against derivative-noise + collinear-shape instability.
  5. Refit UNPENALIZED (OLS) on the selected standardized support — LASSO shrinks
     coefficients, so the final estimate must be the unpenalized refit.
  6. De-standardize coefficients back to raw shape scale for export.
  7. Partition selected terms into dimensionally DSL-mappable (record
     coefficients) vs non-mappable (record, reject) — never silently mismapped.
     Because the NODE output is a first-order coefficient, only its constant
     intercept currently maps to ``LinearElim``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsIC

from apmode.backends.node_candidate_library import (
    CandidateColumn,
    standardize_columns,
)
from apmode.backends.node_ode import HybridPKODE  # noqa: TC001 — used at runtime
from apmode.bundle.models import LassoSelectionResult

__all__ = [
    "LassoSelectionResult",
    "derivative_data_from_node",
    "select_structure",
]

# Below this |coefficient| a LASSO/OLS term is treated as pruned (not selected).
_COEF_PRUNE_ABS = 1e-8
# Columns whose absolute correlation exceeds this are grouped as near-duplicates.
_DUPLICATE_CORR = 0.999


def derivative_data_from_node(
    model: HybridPKODE,
    *,
    n_points: int = 200,
    input_range: tuple[float, float] | None = None,
    time_point: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the trained NODE sub-function RHS over its input support.

    Mirrors :func:`node_distillation.visualize_sub_function`: the NODE takes a
    ``[concentration, time]`` input and returns a scalar rate-law value. This
    sweeps ``concentration`` over ``input_range`` (default ``(0.01, 100.0)``) at a
    fixed ``time_point`` and returns ``(inputs, derivatives)`` as numpy arrays,
    where ``inputs`` is the state/input support ``I`` and ``derivatives`` is the
    NODE-evaluated rate ``dY`` — the LASSO regression target in derivative space.
    """
    lo, hi = input_range if input_range is not None else (0.01, 100.0)
    inputs = np.linspace(lo, hi, n_points, dtype=np.float64)
    derivatives = np.empty(n_points, dtype=np.float64)
    for i, conc in enumerate(inputs):
        out = model.node(jnp.array([float(conc), time_point]))
        derivatives[i] = float(np.asarray(out).squeeze())
    return inputs, derivatives


def _group_near_duplicates(design: np.ndarray) -> list[int]:
    """Return representative column indices, one per near-duplicate group.

    Columns with pairwise absolute correlation above ``_DUPLICATE_CORR`` are
    unioned into a group; the lowest-index member represents the group. Grouping
    stops the selector from arbitrarily splitting a single coefficient across
    highly collinear grid points (Bräm 2025 raises this as a stability concern).
    """
    n_cols = design.shape[1]
    if n_cols <= 1:
        return list(range(n_cols))

    stds = design.std(axis=0)
    parent = list(range(n_cols))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(design, rowvar=False)
    for i in range(n_cols):
        if stds[i] <= 1e-12:
            continue
        for j in range(i + 1, n_cols):
            if stds[j] <= 1e-12:
                continue
            if abs(float(corr[i, j])) >= _DUPLICATE_CORR:
                union(i, j)

    representatives = sorted({find(i) for i in range(n_cols)})
    return representatives


def _fit_support(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit LassoLarsIC(bic) and return the boolean selected-column mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = LassoLarsIC(criterion="bic")
        model.fit(design, target)
    coefs = np.asarray(model.coef_, dtype=np.float64)
    return np.abs(coefs) > _COEF_PRUNE_ABS


def _fit_bic(design: np.ndarray, target: np.ndarray) -> float:
    """Return the minimum BIC of a LassoLarsIC(bic) fit over its LARS path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = LassoLarsIC(criterion="bic")
        model.fit(design, target)
    criterion = np.asarray(model.criterion_, dtype=np.float64)
    if criterion.size == 0:
        return float("nan")
    return float(np.min(criterion))


def select_structure(
    inputs: np.ndarray,
    derivatives: np.ndarray,
    library: list[CandidateColumn],
    *,
    n_bootstrap: int = 100,
    stability_threshold: float = 0.6,
    rng_seed: int = 0,
) -> LassoSelectionResult:
    """Select the structural replacement for a NODE sub-function via LASSO.

    ``inputs``/``derivatives`` are the NODE derivative data (from
    :func:`derivative_data_from_node`); ``library`` is the candidate library from
    :func:`node_candidate_library.build_candidate_library`. Returns a sealed
    :class:`LassoSelectionResult` whose ``coefficients`` are raw-scale,
    de-standardized, unpenalized-refit estimates for dimensionally DSL-mappable
    terms only. The unpenalized intercept is exported as ``constant``;
    concentration-linear, exponential, and Emax shapes are non-mappable
    first-order coefficients and are recorded in ``rejected_nonmappable``.
    Determinism is guaranteed by the seeded ``numpy.random.Generator`` built
    from ``rng_seed``.
    """
    target = np.asarray(derivatives, dtype=np.float64).ravel()
    rng = np.random.default_rng(rng_seed)

    if not library or target.size == 0:
        return LassoSelectionResult(bic=None, derivative_r_squared=0.0)

    if not bool(np.all(np.isfinite(target))):
        msg = "NODE derivative data must contain only finite values."
        raise ValueError(msg)

    # A constant NODE output is exactly the dimensionally valid LinearElim case.
    # The constant candidate standardizes to zero, so handle the unpenalized
    # intercept explicitly rather than asking LASSO to select an all-zero column.
    if float(np.std(target)) <= 1e-12:
        constant = float(np.mean(target))
        if abs(constant) <= _COEF_PRUNE_ABS:
            return LassoSelectionResult(bic=None, derivative_r_squared=0.0)
        return LassoSelectionResult(
            selected_terms=["constant"],
            coefficients={"constant": constant},
            selection_frequency={"constant": 1.0},
            bic=None,
            derivative_r_squared=1.0,
        )

    full_design, _means, full_stds = standardize_columns(library)

    # Group near-duplicate columns; the selector only sees representatives.
    rep_idx = _group_near_duplicates(full_design)
    design = full_design[:, rep_idx]
    stds = full_stds[rep_idx]
    names = [library[j].name for j in rep_idx]
    modules = [library[j].mappable_module for j in rep_idx]
    n_rep = len(rep_idx)

    # Initial fit: BIC score (reported) plus a baseline support.
    bic = _fit_bic(design, target)

    # Stability selection over bootstrapped rows.
    n_rows = target.shape[0]
    counts = np.zeros(n_rep, dtype=np.int64)
    effective_boot = max(int(n_bootstrap), 1)
    for _ in range(effective_boot):
        rows = rng.integers(0, n_rows, size=n_rows)
        boot_design = design[rows]
        boot_target = target[rows]
        # A degenerate resample (all identical rows) yields a zero-variance target;
        # skip it rather than let LassoLarsIC choke on a constant response.
        if float(np.std(boot_target)) <= 1e-12:
            continue
        try:
            mask = _fit_support(boot_design, boot_target)
        except (ValueError, np.linalg.LinAlgError):
            continue
        counts += mask.astype(np.int64)

    frequency = counts.astype(np.float64) / float(effective_boot)
    selection_frequency = {names[j]: float(frequency[j]) for j in range(n_rep)}

    selected = [j for j in range(n_rep) if frequency[j] >= stability_threshold]

    if not selected:
        return LassoSelectionResult(
            selection_frequency=selection_frequency,
            bic=bic,
            derivative_r_squared=0.0,
        )

    # Unpenalized OLS refit on the selected standardized support (LASSO shrinks;
    # the exported estimate must be the unbiased refit).
    sub_design = design[:, selected]
    aug = np.column_stack([np.ones(n_rows, dtype=np.float64), sub_design])
    beta, _resid, _rank, _sv = np.linalg.lstsq(aug, target, rcond=None)
    intercept = float(beta[0])
    slopes_std = beta[1:]

    prediction = aug @ beta
    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    selected_terms: list[str] = []
    coefficients: dict[str, float] = {}
    rejected_nonmappable: list[str] = []
    for local, j in enumerate(selected):
        name = names[j]
        selected_terms.append(name)
        # De-standardize: raw coefficient on the un-standardized column.
        raw_coef = float(slopes_std[local]) / float(stds[j])
        if modules[j] is not None:
            coefficients[name] = raw_coef
        else:
            rejected_nonmappable.append(name)

    # The OLS intercept is an implicit, unpenalized constant rate coefficient.
    # It is the only shape currently dimensionally equivalent to a Formular
    # elimination module, so export it explicitly when non-negligible.
    if abs(intercept) > _COEF_PRUNE_ABS:
        selected_terms.insert(0, "constant")
        coefficients["constant"] = intercept
        selection_frequency["constant"] = 1.0

    return LassoSelectionResult(
        selected_terms=selected_terms,
        coefficients=coefficients,
        rejected_nonmappable=rejected_nonmappable,
        selection_frequency=selection_frequency,
        bic=bic,
        derivative_r_squared=round(r_squared, 6),
    )
