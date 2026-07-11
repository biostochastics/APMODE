# SPDX-License-Identifier: GPL-2.0-or-later
"""Candidate function library for NODE-LASSO structural selection (Bräm 2025).

Builds a design matrix of typical pharmacometric (PMX) rate-law shapes evaluated
over the input support of a trained NODE sub-function. A LASSO regression in
derivative space then selects which shapes explain the learned rate law, replacing
ad-hoc ``curve_fit`` distillation (Bräm et al. 2025, "Automated Pharmacometric
Model Development by Leveraging Low-Dimensional Neural ODEs and LASSO Regression",
CPT:PSP, doi:10.1002/psp4.70285).

Each candidate column carries the Formular DSL module it maps to (or ``None`` when
non-mappable), so the selector can reject non-compilable terms rather than silently
mismapping them — the DSL-moat invariant. Columns are standardized before L1
penalization so LASSO selects on model quality, not raw numeric scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CandidateColumn",
    "build_candidate_library",
    "standardize_columns",
]

# Grid family sizes double as the LASSO parameter-count penalty per family
# (Bräm 2025 §2.3): a family that costs more free parameters must earn its place.
_FAMILY_PARAM_COUNT: dict[str, int] = {
    "constant": 1,
    "linear": 1,
    "exponential": 2,
    "emax": 3,
}


@dataclass(frozen=True)
class CandidateColumn:
    """One evaluated candidate shape in the NODE-LASSO design matrix.

    ``name`` is a stable identifier (family + grid params) used as the LASSO term
    label. ``family`` is the PMX shape. ``params`` holds the grid parameters that
    generated the column (e.g. ``{"u": 1.5}`` for exponential, ``{"h": 1.0,
    "ue50": 4.2}`` for Emax). ``mappable_module`` is the Formular DSL elimination
    module the column maps to, or ``None`` when the shape has no dimensionally
    equivalent DSL module and must be rejected. ``values`` is the column evaluated
    over the input grid. The NODE output multiplies central amount and is therefore
    a first-order coefficient: only a constant coefficient maps directly to
    ``LinearElim``. A concentration-linear or Emax coefficient does not.
    """

    name: str
    family: Literal["constant", "linear", "exponential", "emax"]
    params: dict[str, float]
    mappable_module: str | None
    values: np.ndarray

    @property
    def param_count(self) -> int:
        """Free-parameter count of the candidate's family (LASSO penalty weight)."""
        return _FAMILY_PARAM_COUNT[self.family]


def _default_ue50_grid(inputs: np.ndarray) -> np.ndarray:
    """Derive an Emax EC50 grid from the positive input range.

    Uses interior quantiles of the strictly-positive inputs so that EC50 candidates
    sit where the NODE actually saw curvature; falls back to a linspace over the
    positive range when too few positive samples exist.
    """
    positive = inputs[inputs > 0.0]
    if positive.size >= 4:
        grid = np.quantile(positive, [0.1, 0.25, 0.5, 0.75, 0.9])
        grid = np.unique(grid[grid > 0.0])
        if grid.size >= 1:
            return grid.astype(np.float64)
    hi = float(np.max(inputs)) if inputs.size else 1.0
    hi = hi if hi > 0.0 else 1.0
    return np.linspace(hi / 8.0, hi, 4, dtype=np.float64)


def build_candidate_library(
    inputs: np.ndarray,
    *,
    u_exp_grid: Sequence[float] | None = None,
    h_grid: Sequence[float] | None = None,
    ue50_grid: Sequence[float] | None = None,
) -> list[CandidateColumn]:
    """Build the PMX candidate library evaluated over the NODE input support.

    ``inputs`` is the 1D array of NN input values ``I`` (the state/input support
    from the NODE derivative data). Returns candidate columns for four families
    (Bräm 2025 §2.3):

    - ``constant``: ``f(I) = 1`` (maps to ``LinearElim`` via ``CL/V``).
    - ``linear``: ``f(I) = I`` (not mappable: this is a concentration-dependent
      first-order coefficient, not linear clearance).
    - ``exponential``: ``f(I) = exp(u * I)`` for ``u`` in ``u_exp_grid`` (default
      ``linspace(-2, 2, 9)`` with the ``u≈0`` point dropped, since it duplicates
      the constant column). Non-mappable -> ``mappable_module=None``.
    - ``emax`` (Hill): ``f(I) = I**h / (ue50**h + I**h)`` for ``h`` in ``h_grid``
      (default ``[0.5, 1, 2, 4]``) and ``ue50`` in ``ue50_grid`` (default derived
      from the positive input range). All are non-mappable. Formular's MM
      coefficient is ``Vmax / (V * (Km + C))``, not ``C / (Km + C)``.
    """
    grid = np.asarray(inputs, dtype=np.float64).ravel()
    n = grid.shape[0]
    cols: list[CandidateColumn] = []

    # constant f(I) = 1 -> folds into LinearElim CL offset
    cols.append(
        CandidateColumn(
            name="constant",
            family="constant",
            params={},
            mappable_module="LinearElim",
            values=np.ones(n, dtype=np.float64),
        )
    )

    # linear f(I) = I is a concentration-dependent first-order coefficient.
    cols.append(
        CandidateColumn(
            name="linear",
            family="linear",
            params={},
            mappable_module=None,
            values=grid.copy(),
        )
    )

    # exponential f(I) = exp(u * I) -> non-mappable
    u_values = np.linspace(-2.0, 2.0, 9) if u_exp_grid is None else np.asarray(u_exp_grid, float)
    for u in u_values:
        if abs(float(u)) < 1e-9:
            # exp(0 * I) == constant column; skip the duplicate.
            continue
        # Clip the exponent so a wide input support (e.g. amounts up to 1e2) with
        # |u| up to 2 cannot overflow to +inf and poison standardization.
        exponent = np.clip(float(u) * grid, -50.0, 50.0)
        cols.append(
            CandidateColumn(
                name=f"exp_u={float(u):+.3f}",
                family="exponential",
                params={"u": float(u)},
                mappable_module=None,
                values=np.exp(exponent),
            )
        )

    # Emax/Hill is not the per-unit coefficient implied by Michaelis-Menten.
    h_values = [0.5, 1.0, 2.0, 4.0] if h_grid is None else [float(h) for h in h_grid]
    ue50_values = (
        _default_ue50_grid(grid) if ue50_grid is None else np.asarray(ue50_grid, dtype=np.float64)
    )
    # Emax is defined for I >= 0; clamp negatives to 0 so I**h is real-valued.
    grid_pos = np.clip(grid, 0.0, None)
    for h in h_values:
        for ue50 in ue50_values:
            ue50_f = float(ue50)
            if ue50_f <= 0.0:
                continue
            num = np.power(grid_pos, h)
            values = num / (ue50_f**h + num)
            cols.append(
                CandidateColumn(
                    name=f"emax_h={h:.3f}_ue50={ue50_f:.4g}",
                    family="emax",
                    params={"h": h, "ue50": ue50_f},
                    mappable_module=None,
                    values=values.astype(np.float64),
                )
            )

    return cols


def standardize_columns(
    cols: list[CandidateColumn],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize candidate columns for L1 penalization.

    Returns ``(design, means, stds)`` where ``design[:, j] = (values_j - mean_j) /
    std_j``. Zero-variance columns (e.g. the constant column) get ``std = 1`` so no
    divide-by-zero occurs; their mean is still subtracted, yielding an all-zero
    column that LASSO harmlessly ignores. Coefficients recovered on this scale must
    be de-standardized (divide by ``std_j``) before export to raw shape scale.
    """
    if not cols:
        empty = np.empty((0, 0), dtype=np.float64)
        return empty, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    matrix = np.column_stack([c.values.astype(np.float64) for c in cols])
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    safe_stds = np.where(stds > 1e-12, stds, 1.0)
    design = (matrix - means) / safe_stds
    return design, means, safe_stds
