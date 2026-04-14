# SPDX-License-Identifier: GPL-2.0-or-later
"""Functional distillation for NODE interpretability (PRD SS4.2.4).

Three components — NOT SHAP:
  1. Learned sub-function visualization: plot NODE clearance/absorption
     over the observed concentration/time range.
  2. Parametric surrogate fitting: fit a classical parametric form
     (e.g., Michaelis-Menten) to the NODE-learned function.
  3. Fidelity quantification: AUC/Cmax 80-125% GMR bioequivalence
     between NODE and parametric surrogate predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
import numpy as np

from apmode.backends.node_ode import HybridPKODE  # noqa: TC001 — used at runtime


@dataclass(frozen=True)
class SurrogateResult:
    """Result of fitting a parametric surrogate to NODE output."""

    surrogate_type: str  # e.g. "michaelis_menten", "linear", "power"
    params: dict[str, float]
    residual_ss: float  # sum of squared residuals
    r_squared: float


@dataclass(frozen=True)
class FidelityResult:
    """Bioequivalence fidelity between NODE and surrogate."""

    auc_gmr: float  # geometric mean ratio of AUC
    cmax_gmr: float  # geometric mean ratio of Cmax
    auc_pass: bool  # within 80-125%
    cmax_pass: bool  # within 80-125%
    overall_pass: bool  # both pass


@dataclass
class DistillationReport:
    """Full distillation result for a NODE candidate."""

    candidate_id: str
    node_position: Literal["absorption", "elimination"]
    sub_function_x: list[float] = field(default_factory=list)
    sub_function_y: list[float] = field(default_factory=list)
    surrogate: SurrogateResult | None = None
    fidelity: FidelityResult | None = None


def visualize_sub_function(
    model: HybridPKODE,
    *,
    n_points: int = 100,
    conc_range: tuple[float, float] = (0.01, 100.0),
    time_point: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Evaluate the NODE sub-function over a concentration range.

    Returns (x_values, y_values) where x is concentration and y is the
    NODE output (rate law value).
    """
    x_vals = np.linspace(conc_range[0], conc_range[1], n_points).tolist()
    y_vals: list[float] = []

    for conc in x_vals:
        inp = jnp.array([conc, time_point])
        out = model.node(inp)
        y_vals.append(float(out.squeeze()))

    return x_vals, y_vals


def fit_parametric_surrogate(
    x_vals: list[float],
    y_vals: list[float],
) -> SurrogateResult:
    """Fit a parametric surrogate to NODE sub-function output.

    Tries Michaelis-Menten (Vmax*x/(Km+x)) and linear (a*x+b) forms,
    returns the better fit.
    """
    x = np.array(x_vals)
    y = np.array(y_vals)

    # Total sum of squares (shared by all fits)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 0 else 0.0

    # Linear fit: y = a*x + b
    if len(x) >= 2:
        _coeffs = np.polyfit(x, y, 1)
        a_lin, b_lin = float(_coeffs[0]), float(_coeffs[1])
        y_lin = a_lin * x + b_lin
        ss_lin = float(np.sum((y - y_lin) ** 2))
        r2_lin = 1.0 - ss_lin / ss_tot if ss_tot > 0 else 0.0
    else:
        a_lin, b_lin, ss_lin, r2_lin = 0.0, 0.0, float("inf"), 0.0

    # Michaelis-Menten fit: y = Vmax*x/(Km+x) via nonlinear least-squares
    from scipy.optimize import curve_fit  # type: ignore[import-untyped]

    def _mm_fn(x: np.ndarray, vmax: float, km: float) -> np.ndarray:
        return vmax * x / (km + x)

    try:
        popt, _ = curve_fit(
            _mm_fn,
            x,
            y,
            p0=[float(np.max(y)), float(np.median(x))],
            bounds=([0, 0], [np.inf, np.inf]),
            maxfev=5000,
        )
        vmax, km = float(popt[0]), float(popt[1])
        y_mm = _mm_fn(x, vmax, km)
        ss_mm = float(np.sum((y - y_mm) ** 2))
        r2_mm = 1.0 - ss_mm / ss_tot if ss_tot > 0 else 0.0
    except (RuntimeError, ValueError):
        vmax, km, ss_mm, r2_mm = 0.0, 0.0, float("inf"), 0.0

    # Choose the better fit
    if r2_mm > r2_lin and vmax > 0 and km > 0:
        return SurrogateResult(
            surrogate_type="michaelis_menten",
            params={"Vmax": round(vmax, 4), "Km": round(km, 4)},
            residual_ss=round(ss_mm, 6),
            r_squared=round(r2_mm, 4),
        )
    return SurrogateResult(
        surrogate_type="linear",
        params={"slope": round(a_lin, 6), "intercept": round(b_lin, 6)},
        residual_ss=round(ss_lin, 6),
        r_squared=round(r2_lin, 4),
    )


def quantify_fidelity(
    x_vals: list[float],
    node_y: list[float],
    surrogate: SurrogateResult,
) -> FidelityResult:
    """Quantify fidelity via AUC/Cmax 80-125% GMR bioequivalence.

    Compares integrated exposure (AUC) and peak (Cmax) between NODE
    and surrogate predictions over the concentration range.
    """
    x = np.array(x_vals)
    y_node = np.array(node_y)

    # Evaluate surrogate
    if surrogate.surrogate_type == "michaelis_menten":
        vmax = surrogate.params["Vmax"]
        km = surrogate.params["Km"]
        y_surr = vmax * x / (km + x)
    else:
        slope = surrogate.params["slope"]
        intercept = surrogate.params["intercept"]
        y_surr = slope * x + intercept

    # AUC (trapezoidal)
    auc_node = float(np.trapezoid(np.maximum(y_node, 0), x))
    auc_surr = float(np.trapezoid(np.maximum(y_surr, 0), x))

    # Cmax
    cmax_node = float(np.max(np.abs(y_node)))
    cmax_surr = float(np.max(np.abs(y_surr)))

    # GMR
    auc_gmr = auc_surr / auc_node if auc_node > 1e-10 else 0.0
    cmax_gmr = cmax_surr / cmax_node if cmax_node > 1e-10 else 0.0

    auc_pass = 0.80 <= auc_gmr <= 1.25
    cmax_pass = 0.80 <= cmax_gmr <= 1.25

    return FidelityResult(
        auc_gmr=round(auc_gmr, 4),
        cmax_gmr=round(cmax_gmr, 4),
        auc_pass=auc_pass,
        cmax_pass=cmax_pass,
        overall_pass=auc_pass and cmax_pass,
    )


def distill(
    model: HybridPKODE,
    candidate_id: str,
) -> DistillationReport:
    """Full distillation pipeline for a NODE candidate.

    1. Visualize sub-function
    2. Fit parametric surrogate
    3. Quantify fidelity
    """
    node_position = model.config.node_position

    x_vals, y_vals = visualize_sub_function(model)
    surrogate = fit_parametric_surrogate(x_vals, y_vals)
    fidelity = quantify_fidelity(x_vals, y_vals, surrogate)

    return DistillationReport(
        candidate_id=candidate_id,
        node_position=node_position,
        sub_function_x=x_vals,
        sub_function_y=y_vals,
        surrogate=surrogate,
        fidelity=fidelity,
    )
