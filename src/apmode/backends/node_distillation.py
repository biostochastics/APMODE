# SPDX-License-Identifier: GPL-2.0-or-later
"""Functional distillation for NODE interpretability (PRD SS4.2.4).

Three components — NOT SHAP:
  1. Learned sub-function visualization: plot NODE clearance/absorption
     over the observed concentration/time range.
  2. Parametric surrogate fitting: fit a classical parametric form
     (e.g., Michaelis-Menten) to the NODE-learned function.
  3. Fidelity quantification: concentration-time AUC/Cmax 80-125% ratio
     agreement between NODE and the actually promotable surrogate model.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from apmode.backends.node_candidate_library import (
    build_candidate_library,
    standardize_columns,
)
from apmode.backends.node_lasso_select import (
    derivative_data_from_node,
    select_structure,
)
from apmode.backends.node_ode import HybridPKODE  # noqa: TC001 — used at runtime

# The distillation result types are sealed bundle artifacts, defined once in
# apmode.bundle.models and re-exported here so existing import sites keep working
# (same constructor kwargs / attribute access as the former dataclasses).
from apmode.bundle.models import (
    DistillationReport,
    FidelityResult,
    LassoDistillationReport,
    LassoSelectionResult,
    SurrogateResult,
)
from apmode.dsl.ast_models import (
    AbsorptionModule,
    DSLSpec,
    EliminationModule,
    FirstOrder,
    IVBolus,
    LinearElim,
    Metadata,
    OneCmt,
    Proportional,
)

_CONSTANT_SLOPE_ABS_TOL = 1e-8


def _failed_fidelity() -> FidelityResult:
    """Return the canonical fail-closed exposure-fidelity result."""
    return FidelityResult(
        auc_gmr=0.0,
        cmax_gmr=0.0,
        auc_pass=False,
        cmax_pass=False,
        overall_pass=False,
    )


def _constant_rate_from_surrogate(surrogate: SurrogateResult) -> float | None:
    """Return a positive constant NODE coefficient, or ``None`` if not one.

    ``HybridPKODE`` multiplies the NODE output by central amount. Therefore a
    Formular ``LinearElim`` is equivalent only to a concentration-independent
    coefficient ``CL / V``. Evaluating a nonzero slope at one reference
    concentration changes the model and is not a valid dimensional mapping.
    """
    if surrogate.surrogate_type != "linear":
        return None
    try:
        slope = float(surrogate.params["slope"])
        intercept = float(surrogate.params["intercept"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not np.isfinite(slope)
        or not np.isfinite(intercept)
        or abs(slope) > _CONSTANT_SLOPE_ABS_TOL
        or intercept <= 0.0
    ):
        return None
    return intercept


__all__ = [
    "DistillationReport",
    "FidelityResult",
    "LassoDistillationReport",
    "LassoSelectionResult",
    "SurrogateResult",
    "distill",
    "distill_via_lasso",
    "distillation_passes_fidelity",
    "fit_parametric_surrogate",
    "lasso_fidelity",
    "lasso_result_to_formular",
    "quantify_fidelity",
    "quantify_timecourse_fidelity",
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
        r2_lin = 1.0 - ss_lin / ss_tot if ss_tot > 1e-12 else (1.0 if ss_lin <= 1e-12 else 0.0)
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
        r2_mm = 1.0 - ss_mm / ss_tot if ss_tot > 1e-12 else (1.0 if ss_mm <= 1e-12 else 0.0)
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
    """Legacy rate-law *shape* agreement over a concentration grid.

    These integrals/maxima are not pharmacokinetic exposure AUC/Cmax and this
    helper is no longer used to authorize promotion. Production distillation
    uses :func:`quantify_timecourse_fidelity`, which solves concentration-time
    trajectories before applying the 80--125% exposure rule.
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


def quantify_timecourse_fidelity(
    model: HybridPKODE,
    surrogate: SurrogateResult,
    *,
    reference_conc: float | None = None,
    dose: float = 100.0,
    duration_hours: float = 24.0,
    n_points: int = 200,
) -> FidelityResult:
    """Compare NODE and promoted-surrogate concentration-time AUC/Cmax.

    The only dimensionally valid mapping currently implemented by
    :func:`surrogate_to_formular` is a positive *constant* elimination NODE
    coefficient, mapped as ``CL = coefficient * V``. A concentration-linear
    coefficient and the fitted Emax/MM-shaped coefficient are not Formular
    ``LinearElim``/``MichaelisMenten`` laws and fail closed.

    ``reference_conc`` is retained only for API compatibility; it is
    intentionally ignored because evaluating a nonconstant rate at one
    concentration is the invalid mapping this function prevents.
    """
    if (
        model.config.node_position != "elimination"
        or model.n_cmt != 1
        or _constant_rate_from_surrogate(surrogate) is None
        or dose <= 0.0
        or duration_hours <= 0.0
        or n_points < 3
    ):
        return _failed_fidelity()

    # Deliberately do not derive a rate from this value. Keep the assignment so
    # static checkers and readers see that the compatibility argument is unused.
    _ = reference_conc

    times = np.linspace(0.0, duration_hours, n_points)
    node_y0 = np.zeros(3 if model.n_cmt == 2 else 2, dtype=float)
    node_y0[0] = dose
    try:
        node_states = np.asarray(
            model.solve(jnp.asarray(node_y0), jnp.asarray(times[1:])),
            dtype=float,
        )
    except Exception:
        return _failed_fidelity()
    node_conc = np.concatenate(([0.0], node_states[:, 1] / float(model.V)))
    if not bool(np.all(np.isfinite(node_conc))):
        return _failed_fidelity()

    constant_rate = _constant_rate_from_surrogate(surrogate)
    assert constant_rate is not None  # validated above
    ka = float(model.ka)
    volume = float(model.V)
    if not np.isfinite(ka) or not np.isfinite(volume) or ka <= 0.0 or volume <= 0.0:
        return _failed_fidelity()

    from scipy.integrate import solve_ivp

    def _promoted_rhs(_t: float, y: np.ndarray) -> np.ndarray:
        depot, central = float(y[0]), float(y[1])
        absorbed = ka * depot
        return np.asarray((-absorbed, absorbed - constant_rate * central), dtype=float)

    promoted = solve_ivp(
        _promoted_rhs,
        (0.0, duration_hours),
        np.asarray((dose, 0.0), dtype=float),
        t_eval=times,
        rtol=1e-7,
        atol=1e-9,
    )
    if not promoted.success:
        return _failed_fidelity()
    surrogate_conc = promoted.y[1] / volume
    if not bool(np.all(np.isfinite(surrogate_conc))):
        return _failed_fidelity()
    auc_node = float(np.trapezoid(np.maximum(node_conc, 0.0), times))
    auc_surrogate = float(np.trapezoid(np.maximum(surrogate_conc, 0.0), times))
    cmax_node = float(np.max(np.maximum(node_conc, 0.0)))
    cmax_surrogate = float(np.max(np.maximum(surrogate_conc, 0.0)))
    auc_gmr = auc_surrogate / auc_node if auc_node > 1e-12 else 0.0
    cmax_gmr = cmax_surrogate / cmax_node if cmax_node > 1e-12 else 0.0
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
    *,
    reference_conc: float = 1.0,
) -> DistillationReport:
    """Full distillation pipeline for a NODE candidate.

    1. Visualize sub-function
    2. Fit parametric surrogate
    3. Quantify fidelity
    """
    node_position = model.config.node_position

    x_vals, y_vals = visualize_sub_function(model)
    surrogate = fit_parametric_surrogate(x_vals, y_vals)
    fidelity = quantify_timecourse_fidelity(
        model,
        surrogate,
        reference_conc=reference_conc,
    )

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
    if _constant_rate_from_surrogate(report.surrogate) is None:
        # The NODE output is a first-order coefficient. Only a constant positive
        # coefficient is equivalent to Formular's CL/V linear elimination law.
        return False
    return report.fidelity.overall_pass and report.surrogate.r_squared >= min_r_squared


def surrogate_to_formular(
    surrogate: SurrogateResult,
    node_position: Literal["absorption", "elimination"],
    *,
    model_id: str,
    mechanistic_params: dict[str, float],
    reference_conc: float | None = None,
    fidelity: FidelityResult | None = None,
    source_candidate_id: str | None = None,
) -> DSLSpec:
    """Promote a fitted NODE elimination surrogate to a classical ``DSLSpec``.

    The keystone of functional distillation's loop closure: a NODE-discovered
    elimination sub-function, once approximated by a parametric surrogate, is
    emitted as an ordinary classical spec so it can be re-fit through the
    nlmixr2 backend and ranked in Gate 3 as a genuine (BIC/NLPD-comparable)
    classical candidate. The returned spec carries no NODE modules.

    Mapping (elimination position only): a ``linear`` surrogate with zero slope
    is a constant first-order coefficient and maps to :class:`LinearElim` with
    ``CL = intercept * V``. A nonzero concentration slope is not linear
    elimination and is rejected; it is never frozen at a reference concentration.
    A fitted ``michaelis_menten`` curve is not directly promotable because its
    output is a per-unit NODE rate while Formular's MM module expects an
    amount/time capacity. Absorption-position surrogates are also a separate
    follow-up and raise :class:`NotImplementedError`.

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

    _ = reference_conc  # retained for compatibility; never used to freeze a rate
    volume = float(mechanistic_params["V"])
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("mechanistic volume V must be finite and positive")
    if "V2" in mechanistic_params or "Q" in mechanistic_params:
        raise ValueError("one-compartment distillation cannot promote a two-compartment NODE")
    ka = mechanistic_params.get("ka")
    if ka is not None and (not np.isfinite(float(ka)) or float(ka) <= 0.0):
        raise ValueError("mechanistic absorption rate ka must be finite and positive")
    absorption: AbsorptionModule = FirstOrder() if ka is not None else IVBolus()
    initial: dict[str, float] = {"V": volume}
    if ka is not None:
        initial["ka"] = ka

    elimination: EliminationModule
    if surrogate.surrogate_type == "linear":
        constant_rate = _constant_rate_from_surrogate(surrogate)
        if constant_rate is None:
            raise ValueError(
                "cannot promote a concentration-dependent NODE coefficient as "
                "LinearElim; only a positive zero-slope coefficient equals CL/V"
            )
        elimination = LinearElim()
        initial["CL"] = constant_rate * volume
    else:  # "michaelis_menten" (Literal exhausts the two surrogate families)
        msg = (
            "cannot promote a fitted Michaelis-Menten NODE rate curve directly: "
            "its Vmax is a per-unit rate, while Formular MichaelisMenten expects "
            "an amount/time capacity"
        )
        raise ValueError(msg)

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


def lasso_fidelity(
    inputs: np.ndarray,
    derivatives: np.ndarray,
    result: LassoSelectionResult,
) -> dict[str, float | None]:
    """Derivative-space fidelity of a NODE-LASSO selected structure (Bräm 2025).

    Reconstructs the unpenalized OLS refit on the selected candidate support over
    the NODE-derivative grid and returns ``{"derivative_r_squared": R^2,
    "derivative_mare": MARE}``. R^2 is the primary fidelity metric (identical to
    :attr:`LassoSelectionResult.derivative_r_squared` since the refit is
    deterministic for the same ``inputs``); MARE is the mean absolute relative
    error over grid points with a non-negligible target. No integrated-trajectory
    metric is fabricated — this stays entirely in derivative space, which is the
    only space the NODE RHS was evaluated in.

    A refit here (rather than echoing the stored R^2) also recovers the OLS
    intercept, which the exported raw-scale ``coefficients`` fold away — so MARE
    reflects the true reconstruction, not a coefficient-only approximation.
    """
    target = np.asarray(derivatives, dtype=np.float64).ravel()
    if target.size == 0 or not result.selected_terms:
        return {"derivative_r_squared": 0.0, "derivative_mare": None}

    library = build_candidate_library(np.asarray(inputs, dtype=np.float64).ravel())
    design, _means, _stds = standardize_columns(library)
    name_to_idx = {col.name: i for i, col in enumerate(library)}
    selected = [name_to_idx[t] for t in result.selected_terms if t in name_to_idx]
    if not selected:
        return {"derivative_r_squared": 0.0, "derivative_mare": None}

    aug = np.column_stack([np.ones(target.shape[0], dtype=np.float64), design[:, selected]])
    beta, _resid, _rank, _sv = np.linalg.lstsq(aug, target, rcond=None)
    prediction = aug @ beta

    ss_res = float(np.sum((target - prediction) ** 2))
    ss_tot = float(np.sum((target - float(np.mean(target))) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else (1.0 if ss_res <= 1e-12 else 0.0)

    denom = np.abs(target)
    mask = denom > 1e-12
    if bool(np.any(mask)):
        mare = float(np.mean(np.abs(prediction[mask] - target[mask]) / denom[mask]))
    else:
        mare = None

    return {
        "derivative_r_squared": round(r_squared, 6),
        "derivative_mare": round(mare, 6) if mare is not None else None,
    }


def lasso_result_to_formular(
    result: LassoSelectionResult,
    node_position: Literal["absorption", "elimination"],
    *,
    model_id: str,
    mechanistic_params: dict[str, float],
    reference_conc: float | None = None,
) -> DSLSpec:
    """Promote a fully-mappable NODE-LASSO elimination structure to a ``DSLSpec``.

    The LASSO analogue of :func:`surrogate_to_formular`: instead of a single
    ``curve_fit`` surrogate, the promoted structure comes from the LASSO-selected,
    DSL-mappable candidate terms (Bräm 2025). For the implemented elimination
    NODE, the neural output multiplies central amount and is a first-order rate
    coefficient. Therefore the only currently valid mapping is a selected
    positive ``constant`` -> :class:`LinearElim` with ``CL = constant * V``.

    A ``linear`` concentration term is concentration-dependent clearance, not
    LinearElim. An Emax term ``C/(Km+C)`` is not Michaelis-Menten either: the MM
    coefficient that multiplies amount is ``Vmax/(V*(Km+C))``. Those shapes fail
    promotion closed until an exact DSL mapping is implemented and validated.

    ``result.coefficients`` must be non-empty and ``result.rejected_nonmappable``
    must be empty — callers (:func:`distill_via_lasso`) enforce this so a
    partially non-mappable structure is never emitted. Absorption-position
    promotion is a separate follow-up and raises :class:`NotImplementedError`.
    """
    if node_position != "elimination":
        msg = (
            "lasso_result_to_formular maps only elimination-position structures; "
            "absorption-position distillation is a separate follow-up."
        )
        raise NotImplementedError(msg)
    if not result.coefficients:
        msg = "lasso_result_to_formular requires at least one mappable selected term."
        raise ValueError(msg)
    if result.rejected_nonmappable:
        msg = (
            "lasso_result_to_formular refuses a structure with non-mappable selected "
            f"terms {result.rejected_nonmappable}; promotion must be skipped upstream."
        )
        raise ValueError(msg)

    if set(result.coefficients) != {"constant"}:
        msg = (
            "lasso_result_to_formular can promote only a constant NODE rate "
            "coefficient; concentration-linear and Emax terms are not "
            "dimensionally equivalent to LinearElim or MichaelisMenten"
        )
        raise ValueError(msg)

    _ = reference_conc  # compatibility only; freezing a rate at C is forbidden
    volume = float(mechanistic_params["V"])
    constant_rate = float(result.coefficients["constant"])
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("mechanistic volume V must be finite and positive")
    if not np.isfinite(constant_rate) or constant_rate <= 0.0:
        raise ValueError("constant NODE rate coefficient must be finite and positive")
    if "V2" in mechanistic_params or "Q" in mechanistic_params:
        raise ValueError("one-compartment distillation cannot promote a two-compartment NODE")
    ka = mechanistic_params.get("ka")
    if ka is not None and (not np.isfinite(float(ka)) or float(ka) <= 0.0):
        raise ValueError("mechanistic absorption rate ka must be finite and positive")
    absorption: AbsorptionModule = FirstOrder() if ka is not None else IVBolus()
    initial: dict[str, float] = {"V": volume}
    if ka is not None:
        initial["ka"] = ka

    elimination: EliminationModule = LinearElim()
    initial["CL"] = constant_rate * volume
    chosen = f"LinearElim (CL={initial['CL']:.4g})"

    context = (
        f"Distilled from NODE via LASSO selection (Bräm 2025); "
        f"selected={result.selected_terms}, module={chosen}, "
        f"derivative R^2={result.derivative_r_squared:.3f}"
    )

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


def distill_via_lasso(
    model: HybridPKODE,
    candidate_id: str,
    *,
    node_position: str | None = None,
    n_bootstrap: int = 100,
    stability_threshold: float = 0.6,
    rng_seed: int = 0,
    reference_conc: float | None = None,
    min_derivative_r_squared: float = 0.8,
) -> LassoDistillationReport:
    """Distil a trained NODE via LASSO structural selection (Bräm 2025).

    Replaces the ad-hoc ``curve_fit`` path (:func:`distill`) with derivative-space
    LASSO selection over a PMX candidate library:

    1. Evaluate the NODE RHS over its input support (:func:`derivative_data_from_node`).
    2. Build the candidate library (:func:`build_candidate_library`).
    3. Select structure via stability-selected LASSO (:func:`select_structure`).
    4. Measure derivative-space fidelity (:func:`lasso_fidelity`).
    5. Promote to a classical ``DSLSpec`` **only** when the selected support is
       the dimensionally valid positive constant coefficient, derivative-space
       R² clears ``min_derivative_r_squared``, and concentration-time AUC/Cmax
       fidelity passes. A non-mappable selection is recorded (on
       ``selection.rejected_nonmappable``) and promotion is skipped, never a
       crash and never a mismap — the DSL-moat invariant.

    ``node_position`` defaults to the model's configured position; a non-elimination
    position is reported but not promoted (absorption promotion is a follow-up).
    """
    resolved = node_position if node_position is not None else model.config.node_position
    if resolved not in ("absorption", "elimination"):
        msg = f"node_position must be 'absorption' or 'elimination', got {resolved!r}."
        raise ValueError(msg)
    position: Literal["absorption", "elimination"] = (
        "elimination" if resolved == "elimination" else "absorption"
    )

    inputs, derivatives = derivative_data_from_node(model)
    library = build_candidate_library(inputs)
    result = select_structure(
        inputs,
        derivatives,
        library,
        n_bootstrap=n_bootstrap,
        stability_threshold=stability_threshold,
        rng_seed=rng_seed,
    )
    fidelity = lasso_fidelity(inputs, derivatives, result)

    promoted = False
    promoted_model_id: str | None = None
    promoted_spec: DSLSpec | None = None
    timecourse_fidelity: FidelityResult | None = None
    fully_mappable = (
        set(result.coefficients) == {"constant"}
        and not result.rejected_nonmappable
        and result.derivative_r_squared >= min_derivative_r_squared
    )
    if position == "elimination" and fully_mappable:
        constant_rate = float(result.coefficients["constant"])
        constant_surrogate = SurrogateResult(
            surrogate_type="linear",
            params={"slope": 0.0, "intercept": constant_rate},
            residual_ss=0.0,
            r_squared=result.derivative_r_squared,
        )
        timecourse_fidelity = quantify_timecourse_fidelity(
            model,
            constant_surrogate,
            reference_conc=reference_conc,
        )

    if (
        position == "elimination"
        and fully_mappable
        and timecourse_fidelity is not None
        and timecourse_fidelity.overall_pass
    ):
        mech: dict[str, float] = {"V": float(model.V), "ka": float(model.ka)}
        promoted_model_id = f"{candidate_id}_lasso_distilled"
        promoted_spec = lasso_result_to_formular(
            result,
            "elimination",
            model_id=promoted_model_id,
            mechanistic_params=mech,
            reference_conc=reference_conc,
        )
        promoted = True

    return LassoDistillationReport(
        candidate_id=candidate_id,
        node_position=position,
        selection=result,
        promoted=promoted,
        promoted_model_id=promoted_model_id,
        promoted_spec=promoted_spec,
        derivative_mare=fidelity["derivative_mare"],
        timecourse_fidelity=timecourse_fidelity,
    )
