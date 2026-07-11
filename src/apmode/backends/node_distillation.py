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

from typing import Literal

import jax.numpy as jnp
import numpy as np

from apmode.backends.node_ode import HybridPKODE  # noqa: TC001 — used at runtime

# The distillation result types are sealed bundle artifacts, defined once in
# apmode.bundle.models and re-exported here so existing import sites keep working
# (same constructor kwargs / attribute access as the former dataclasses).
from apmode.bundle.models import DistillationReport, FidelityResult, SurrogateResult
from apmode.dsl.ast_models import (
    AbsorptionModule,
    DSLSpec,
    EliminationModule,
    FirstOrder,
    IVBolus,
    LinearElim,
    Metadata,
    MichaelisMenten,
    OneCmt,
    Proportional,
)

__all__ = [
    "DistillationReport",
    "FidelityResult",
    "SurrogateResult",
    "distill",
    "distillation_passes_fidelity",
    "fit_parametric_surrogate",
    "quantify_fidelity",
    "surrogate_to_formular",
    "visualize_sub_function",
]


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
    from scipy.optimize import curve_fit

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


def distillation_passes_fidelity(
    report: DistillationReport,
    *,
    min_r_squared: float = 0.8,
) -> bool:
    """Whether a distilled surrogate is faithful enough to promote into Gate 3.

    Requires both an AUC/Cmax bioequivalence pass (``fidelity.overall_pass``)
    and a surrogate goodness-of-fit ``r_squared >= min_r_squared``. A report
    lacking either a surrogate or a fidelity result never passes.
    """
    if report.surrogate is None or report.fidelity is None:
        return False
    return report.fidelity.overall_pass and report.surrogate.r_squared >= min_r_squared


def surrogate_to_formular(
    surrogate: SurrogateResult,
    node_position: Literal["absorption", "elimination"],
    *,
    model_id: str,
    mechanistic_params: dict[str, float],
    reference_conc: float = 1.0,
    fidelity: FidelityResult | None = None,
    source_candidate_id: str | None = None,
) -> DSLSpec:
    """Promote a fitted NODE elimination surrogate to a classical ``DSLSpec``.

    The keystone of functional distillation's loop closure: a NODE-discovered
    elimination sub-function, once approximated by a parametric surrogate, is
    emitted as an ordinary classical spec so it can be re-fit through the
    nlmixr2 backend and ranked in Gate 3 as a genuine (BIC/NLPD-comparable)
    classical candidate. The returned spec carries no NODE modules.

    Mapping (elimination position only): ``linear`` -> :class:`LinearElim` with
    ``CL = ke_ref * V`` where ``ke_ref = slope * reference_conc + intercept``
    (the surrogate is a per-unit rate on the central amount; ``reference_conc``
    selects the concentration at which the effective first-order rate is read —
    ``1.0`` by default, or pass the fitted subjects' median concentration);
    ``michaelis_menten`` -> :class:`MichaelisMenten` carrying the fitted
    ``Vmax``/``Km``. Absorption-position surrogates are a separate follow-up and
    raise :class:`NotImplementedError`.

    Provenance (source candidate, surrogate family, fidelity GMRs) is recorded
    in ``metadata`` (fingerprint-excluded), so two distilled specs differing
    only in provenance still fingerprint identically.
    """
    if node_position != "elimination":
        msg = (
            "surrogate_to_formular maps only elimination-position surrogates "
            "(the two existing surrogate forms are elimination rate-laws); "
            "absorption-position distillation is a separate follow-up."
        )
        raise NotImplementedError(msg)

    volume = mechanistic_params["V"]
    ka = mechanistic_params.get("ka")
    absorption: AbsorptionModule = FirstOrder() if ka is not None else IVBolus()
    initial: dict[str, float] = {"V": volume}
    if ka is not None:
        initial["ka"] = ka

    elimination: EliminationModule
    if surrogate.surrogate_type == "linear":
        elimination = LinearElim()
        ke_ref = surrogate.params["slope"] * reference_conc + surrogate.params["intercept"]
        initial["CL"] = max(ke_ref, 1e-6) * volume
    else:  # "michaelis_menten" (Literal exhausts the two surrogate families)
        elimination = MichaelisMenten()
        initial["Vmax"] = surrogate.params["Vmax"]
        initial["Km"] = surrogate.params["Km"]

    context = (
        f"Distilled from NODE candidate {source_candidate_id}; "
        f"surrogate={surrogate.surrogate_type}, R^2={surrogate.r_squared:.3f}"
    )
    if fidelity is not None:
        context += f"; AUC GMR={fidelity.auc_gmr}, Cmax GMR={fidelity.cmax_gmr}"

    return DSLSpec(
        model_id=model_id,
        absorption=absorption,
        distribution=OneCmt(),
        elimination=elimination,
        variability=[],
        observation=Proportional(sigma_prop=0.1),
        initial=initial,
        metadata=Metadata(
            intent="functional_distillation",
            context_of_use=context,
            version="distilled",
        ),
    )
