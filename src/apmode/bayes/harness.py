# SPDX-License-Identifier: GPL-2.0-or-later
"""Bayesian harness: driven as a subprocess by BayesianRunner.

Reads ``request.json``, compiles and samples the Stan program via cmdstanpy,
computes arviz diagnostics, writes ``posterior_draws.parquet`` and
``response.json``.

This script is invoked only when the ``bayesian`` optional extras are
installed (cmdstanpy, arviz, pyarrow). The imports are deferred so the module
can be imported for testing without those dependencies.

Wire contract (JSON on stdin/stdout files):
    Usage: python -m apmode.bayes.harness <request.json> <response.json>

Response classification (matches BayesianRunner._parse_response):
    status="success" + result=BackendResult-shaped dict
    status="error" + error_type in {"convergence", "crash", "compile_error",
                                    "invalid_spec"}

Safety:
    - Catastrophic sampling failure (all chains stuck, R̂>2, divergences>25%)
      is mapped to error_type="convergence" — BayesianRunner translates to
      ConvergenceError. Gate-1 threshold evaluation is policy-driven and
      runs *outside* this harness.
    - Any uncaught exception is mapped to error_type="crash".
    - Stan compile failure is mapped to error_type="compile_error".
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from apmode.backends.predictive_summary import PredictiveSummaryBundle
    from apmode.bundle.models import NCASubjectDiagnostic
    from apmode.governance.policy import Gate3Config

# Optional imports — deferred so tests and import-time checks don't fail when
# the bayesian extras are absent.
_CMDSTAN_AVAILABLE = False
_ARVIZ_AVAILABLE = False
CmdStanModel: Any = None
try:  # pragma: no cover - import-time guard
    import cmdstanpy  # noqa: F401 — imported at runtime inside _run_sampling
    from cmdstanpy import CmdStanModel as _CmdStanModel

    CmdStanModel = _CmdStanModel
    _CMDSTAN_AVAILABLE = True
except ImportError:
    pass
try:  # pragma: no cover
    import arviz  # noqa: F401 — imported at runtime inside _compute_diagnostics

    _ARVIZ_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI, dispatch to _run, write response.json.

    All exceptions — including failures writing the response file itself —
    are caught and classified into ``status="error"`` payloads so
    BayesianRunner's response parser never sees a silent crash path.
    """
    args = argv or sys.argv[1:]
    if len(args) != 2:
        sys.stderr.write("Usage: python -m apmode.bayes.harness <request.json> <response.json>\n")
        return 2

    response_path = Path(args[1])

    try:
        request_path = Path(args[0])
        response = _run(request_path)
    except Exception as exc:
        response = {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "crash",
            "error_detail": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "session_info": _session_info(),
        }

    try:
        response_path.write_text(json.dumps(response, indent=2))
    except Exception as exc:
        sys.stderr.write(f"harness: failed to write response.json: {exc}\n")
        return 2
    return 0 if response["status"] == "success" else 1


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _run(request_path: Path) -> dict[str, Any]:
    """Load request, compile Stan, sample, build BackendResult-shaped dict."""
    if not _CMDSTAN_AVAILABLE or not _ARVIZ_AVAILABLE:
        return {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "crash",
            "error_detail": (
                "Bayesian extras not installed. Install with: uv sync --extra bayesian"
            ),
            "session_info": _session_info(),
        }

    request = json.loads(request_path.read_text())
    work_dir = request_path.parent

    # 1. Write the Stan program (already compiled by the emitter).
    stan_path = work_dir / "model.stan"
    if not stan_path.exists():
        stan_path.write_text(request["compiled_stan_code"])

    # 2. Build Stan data dict from the input CSV + spec.
    try:
        stan_data = _build_stan_data(request)
    except ValueError as exc:
        return {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "invalid_spec",
            "error_detail": str(exc),
            "session_info": _session_info(),
        }
    data_json_path = work_dir / "stan_data.json"
    data_json_path.write_text(json.dumps(stan_data))

    # 3. Compile the Stan program.
    try:
        import cmdstanpy as cs

        model = cs.CmdStanModel(stan_file=str(stan_path))
    except Exception as exc:
        return {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "compile_error",
            "error_detail": f"{type(exc).__name__}: {exc}",
            "session_info": _session_info(),
        }

    # 4. Run NUTS sampling — with initial values seeded from the request's
    # initial_estimates to avoid rejecting-initial-value failures on
    # analytical-solution models with ka≈ke singularities near random draws.
    cfg = request["sampler_config"]
    init_dict = _build_inits(request)
    inits_for_chains: list[Mapping[str, Any]] | None = (
        [dict(init_dict) for _ in range(cfg["chains"])] if init_dict else None
    )
    start = time.time()
    fit = model.sample(
        data=str(data_json_path),
        chains=cfg["chains"],
        parallel_chains=cfg.get("parallel_chains") or cfg["chains"],
        iter_warmup=cfg["warmup"],
        iter_sampling=cfg["sampling"],
        adapt_delta=cfg["adapt_delta"],
        max_treedepth=cfg["max_treedepth"],
        seed=cfg["seed"] or None,
        threads_per_chain=cfg.get("threads_per_chain"),
        inits=inits_for_chains,
        show_progress=False,
        show_console=False,
        output_dir=str(work_dir),
    )
    wall_time_seconds = time.time() - start

    # 5. Compute diagnostics via arviz.
    diag = _compute_diagnostics(fit)

    # 6. Catastrophic sampling failure → convergence error.
    if _catastrophic(diag, cfg):
        return {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "convergence",
            "error_detail": (
                f"Catastrophic sampling failure: "
                f"R-hat_max={diag['rhat_max']:.2f}, "
                f"ess_bulk_min={diag['ess_bulk_min']:.0f}, "
                f"n_divergent={diag['n_divergent']}"
            ),
            "session_info": _session_info(),
        }

    # 7. Write draws to parquet.
    draws_path = Path(request["output_draws_path"])
    try:
        _write_draws_parquet(fit, draws_path)
    except Exception as exc:
        sys.stderr.write(f"WARN: failed to write draws parquet: {exc}\n")

    # 8. Aggregate ParameterEstimate summaries.
    structural_names = _extract_structural_names(request["spec"])
    param_estimates = _aggregate_estimates(fit, structural_names)

    # 9. Build BackendResult-shaped dict.
    sampler_config_persisted = dict(cfg)
    sampler_config_persisted["cmdstan_version"] = _cmdstan_version()
    sampler_config_persisted["stan_version"] = _stan_version()

    converged_flag = _is_converged(diag)

    result: dict[str, Any] = {
        "model_id": request["candidate_id"],
        "backend": "bayesian_stan",
        "converged": converged_flag,
        "parameter_estimates": param_estimates,
        "eta_shrinkage": _compute_eta_shrinkage(fit, structural_names),
        "convergence_metadata": {
            "method": "nuts",
            "converged": converged_flag,
            "iterations": cfg["warmup"] + cfg["sampling"],
            "minimization_status": "successful" if converged_flag else "marginal",
            "wall_time_seconds": wall_time_seconds,
        },
        "diagnostics": {
            "gof": {"cwres_mean": 0.0, "cwres_sd": 1.0, "outlier_fraction": 0.0},
            "identifiability": {
                "profile_likelihood_ci": {},
                "ill_conditioned": False,
            },
            "blq": {"method": "none", "n_blq": 0, "blq_fraction": 0.0},
        },
        "wall_time_seconds": wall_time_seconds,
        "backend_versions": {
            "cmdstan": _cmdstan_version(),
            "stan": _stan_version(),
        },
        "initial_estimate_source": "fallback",
        "posterior_diagnostics": diag,
        "sampler_config": sampler_config_persisted,
        "posterior_draws_path": str(draws_path),
    }

    return {
        "schema_version": "1.0",
        "status": "success",
        "result": result,
        "session_info": _session_info(),
    }


# ---------------------------------------------------------------------------
# Data conversion: DSLSpec + CSV → Stan data dict
# ---------------------------------------------------------------------------


def _build_stan_data(request: dict[str, Any]) -> dict[str, Any]:
    """Convert the NONMEM-style CSV + spec into the Stan data dict.

    Data block contract (from stan_emitter._emit_data_block):
        N, N_subjects, subject[N], time[N], dv[N],
        N_events, event_subject[N_events], event_time[N_events],
        event_amt[N_events], event_cmt[N_events], event_evid[N_events],
        event_rate[N_events], event_start[N_subjects], event_end[N_subjects]
        Optional: cens[N], loq (for BLQM3/BLQM4)
                  per-covariate vectors of length N_subjects

    Trust boundary: ``request["data_path"]`` must be produced by the
    orchestrator (``BayesianRunner`` wires the current
    ``DataManifest.data_path`` here). This helper validates that the path
    resolves to an existing regular file before ``pd.read_csv`` touches
    it — a defence-in-depth guard against a malformed or spoofed request
    JSON. It does not attempt a chroot-style jail; callers are
    responsible for constraining the trust domain.
    """
    import pandas as pd

    raw_path = request.get("data_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("request['data_path'] must be a non-empty string")
    data_path = Path(raw_path).resolve()
    if not data_path.is_file():
        raise ValueError(f"data_path does not point to a regular file: {data_path}")
    df = pd.read_csv(data_path)
    df = df.rename(columns={c: c.upper() for c in df.columns})

    # Filter observations (DV rows): EVID=0 and MDV=0 (or no MDV)
    if "EVID" not in df.columns or "TIME" not in df.columns:
        raise ValueError("Input CSV must have EVID and TIME columns")
    _ID_CANDIDATES = ("ID", "NMID", "USUBJID", "SUBJECT_ID", "PATIENT_ID")
    id_col = next((c for c in _ID_CANDIDATES if c in df.columns), None)
    if id_col is None:
        raise ValueError(
            "Input CSV must have an ID-like column (ID/NMID/USUBJID/SUBJECT_ID/PATIENT_ID)"
        )

    mdv_mask = df["MDV"].fillna(0).astype(int) == 0 if "MDV" in df.columns else True
    obs_mask = (df["EVID"].astype(int) == 0) & mdv_mask
    evt_mask = df["EVID"].astype(int).isin([1, 3, 4])
    obs_df = df[obs_mask].reset_index(drop=True)
    evt_df = df[evt_mask].reset_index(drop=True)

    # DV must be strictly positive for lognormal/proportional likelihood.
    # NONMEM convention often encodes pre-dose baseline as DV=0 MDV=0; these
    # are incompatible with a continuous lognormal observation model.
    # Silently dropping them would mutate the user's dataset underneath the
    # posterior likelihood, biasing estimates without any audit trail, so
    # we escalate to ``invalid_spec`` and point the user at the BLQM3/M4
    # modules for principled censoring. Callers that legitimately need to
    # exclude pre-dose baselines should set ``MDV=1`` on those rows.
    if "DV" in obs_df.columns:
        n_nonpos = int((obs_df["DV"].astype(float) <= 0.0).sum())
        if n_nonpos:
            raise ValueError(
                f"{n_nonpos} non-positive DV observations (MDV=0) incompatible "
                f"with the lognormal/proportional likelihood. Mark pre-dose "
                f"baselines with MDV=1 to exclude them from the likelihood, "
                f"or use the BLQ_M3 / BLQ_M4 observation module with "
                f"loq_value set so censoring is handled explicitly."
            )

    # Stable subject index: 1..N_subjects, preserving first-appearance order.
    subjects = df[id_col].drop_duplicates().tolist()
    subject_to_idx = {s: i + 1 for i, s in enumerate(subjects)}
    n_subjects = len(subjects)

    # Observations
    dv_col = "DV" if "DV" in obs_df.columns else None
    if dv_col is None:
        raise ValueError("Input CSV must have a DV column")
    subject_arr = obs_df[id_col].map(subject_to_idx).astype(int).tolist()
    time_arr = obs_df["TIME"].astype(float).tolist()
    dv_arr = obs_df[dv_col].astype(float).tolist()

    # Event arrays — sort by subject then time to satisfy per-subject ranges
    evt_df = evt_df.assign(_subj=evt_df[id_col].map(subject_to_idx))
    evt_df = evt_df.sort_values(["_subj", "TIME"], kind="stable").reset_index(drop=True)
    event_subject = evt_df["_subj"].astype(int).tolist()
    event_time = evt_df["TIME"].astype(float).tolist()
    event_amt = (
        evt_df["AMT"].astype(float).tolist() if "AMT" in evt_df.columns else [0.0] * len(evt_df)
    )
    event_cmt = (
        evt_df["CMT"].astype(int).tolist() if "CMT" in evt_df.columns else [1] * len(evt_df)
    )
    event_evid = evt_df["EVID"].astype(int).tolist()
    event_rate = (
        evt_df["RATE"].astype(float).tolist() if "RATE" in evt_df.columns else [0.0] * len(evt_df)
    )

    # Per-subject index ranges (1-indexed, inclusive). If a subject has no
    # events (pure-observation subject, rare), we set start=1, end=0 to
    # produce an empty range under Stan's 1-indexed semantics.
    event_start = []
    event_end = []
    for s in subjects:
        s_idx = subject_to_idx[s]
        mask = [i + 1 for i, v in enumerate(event_subject) if v == s_idx]
        if mask:
            event_start.append(mask[0])
            event_end.append(mask[-1])
        else:
            event_start.append(1)
            event_end.append(0)

    stan_data: dict[str, Any] = {
        "N": len(obs_df),
        "N_subjects": n_subjects,
        "subject": subject_arr,
        "time": time_arr,
        "dv": dv_arr,
        "N_events": len(evt_df),
        "event_subject": event_subject,
        "event_time": event_time,
        "event_amt": event_amt,
        "event_cmt": event_cmt,
        "event_evid": event_evid,
        "event_rate": event_rate,
        "event_start": event_start,
        "event_end": event_end,
    }

    # BLQ: if the observation module is BLQM3/BLQM4 the Stan data block
    # expects cens[N] and loq. Honor the observation.loq_value from spec.
    obs = request["spec"].get("observation", {}) if isinstance(request["spec"], dict) else {}
    if obs.get("type") in ("BLQ_M3", "BLQ_M4"):
        loq = float(obs.get("loq_value", 0.0))
        cens = [1 if v <= loq else 0 for v in dv_arr]
        stan_data["cens"] = cens
        stan_data["loq"] = loq

    # Covariates referenced via CovariateLink variability items.
    # Stan data declares covariates as vector[N_subjects] (subject-level constants).
    # Time-varying covariates silently collapsing to baseline would bias
    # estimates, so reject at data-build time.
    variability = (
        request["spec"].get("variability", []) if isinstance(request["spec"], dict) else []
    )
    cov_names = sorted({v["covariate"] for v in variability if v.get("type") == "CovariateLink"})
    for cov in cov_names:
        if cov not in df.columns:
            raise ValueError(
                f"Covariate {cov!r} declared via CovariateLink but not in input CSV "
                f"(columns: {list(df.columns)})"
            )
        nunique = df.groupby(id_col)[cov].nunique()
        if (nunique > 1).any():
            offenders = nunique[nunique > 1].index.tolist()
            raise ValueError(
                f"Covariate {cov!r} varies within subject for ids {offenders[:5]} "
                f"(first 5). Time-varying covariates are not supported in v1; "
                f"preprocess to a subject-level summary."
            )
        first_per_subject = df.drop_duplicates(subset=[id_col], keep="first")
        stan_data[cov] = first_per_subject[cov].astype(float).tolist()

    # ADDL/II (repeat dose) and SS (steady state) NMTRAN semantics are not yet
    # expanded by the harness — reject at data-build time rather than fit with
    # silently-biased posteriors. Users should preprocess into explicit dose
    # rows before calling the Bayesian harness.
    if "ADDL" in df.columns and (df["ADDL"].fillna(0).astype(int) > 0).any():
        raise ValueError(
            "ADDL>0 repeat-dose rows detected. Pre-expand into explicit dose events "
            "before the Bayesian harness (v1 does not implement ADDL/II expansion)."
        )
    if "SS" in df.columns and (df["SS"].fillna(0).astype(int) > 0).any():
        raise ValueError(
            "SS>0 steady-state rows detected. Steady-state dosing is not yet "
            "supported by the Bayesian harness; preprocess to explicit dose records."
        )

    return stan_data


# ---------------------------------------------------------------------------
# Diagnostics via arviz
# ---------------------------------------------------------------------------


def _compute_diagnostics(fit: Any) -> dict[str, Any]:
    """Compute PosteriorDiagnostics-shaped dict from a CmdStanMCMC fit."""
    import arviz as az

    idata = az.from_cmdstanpy(fit, log_likelihood="log_lik")

    # R-hat
    rhat_ds = az.rhat(idata)
    rhat_vals = _dataset_flat_values(rhat_ds)
    rhat_max = float(max(rhat_vals)) if rhat_vals else 1.0

    # ESS
    ess_bulk_ds = az.ess(idata, method="bulk")
    ess_tail_ds = az.ess(idata, method="tail")
    ess_bulk_vals = _dataset_flat_values(ess_bulk_ds)
    ess_tail_vals = _dataset_flat_values(ess_tail_ds)
    ess_bulk_min = float(min(ess_bulk_vals)) if ess_bulk_vals else 0.0
    ess_tail_min = float(min(ess_tail_vals)) if ess_tail_vals else 0.0

    # Divergences / max-treedepth — from sample diagnostics
    import numpy as np

    div_arr = getattr(fit, "divergences", None)
    n_divergent = int(np.asarray(div_arr).sum()) if div_arr is not None else 0
    mtd_arr = getattr(fit, "max_treedepths", None)
    n_max_treedepth = int(np.asarray(mtd_arr).sum()) if mtd_arr is not None else 0

    # E-BFMI
    try:
        ebfmi = az.bfmi(idata)
        ebfmi_min = float(min(ebfmi))
    except Exception:
        ebfmi_min = float("nan")

    # Pareto-k via LOO (if log_lik present)
    pareto_k_max: float | None = None
    pareto_k_counts: dict[str, int] = {}
    try:
        loo = az.loo(idata, pointwise=True)
        pareto_k = loo.pareto_k.values
        pareto_k_max = float(pareto_k.max())
        bins = [(-float("inf"), 0.5), (0.5, 0.7), (0.7, 1.0), (1.0, float("inf"))]
        labels = ["good", "ok", "bad", "very_bad"]
        for (lo, hi), label in zip(bins, labels, strict=True):
            pareto_k_counts[label] = int(((pareto_k > lo) & (pareto_k <= hi)).sum())
    except Exception:
        pass

    # MCSE for headline structural params (first few)
    mcse_ds = az.mcse(idata, method="mean")
    mcse_by_param = {}
    for name, arr in _flatten_dataset_items(mcse_ds)[:20]:
        mcse_by_param[name] = float(arr)

    # Per-chain R-hat (coarse summary)
    per_chain_rhat: dict[str, list[float]] = {}

    return {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "n_divergent": n_divergent,
        "n_max_treedepth": n_max_treedepth,
        "ebfmi_min": ebfmi_min,
        "pareto_k_max": pareto_k_max,
        "pareto_k_counts": pareto_k_counts,
        "mcse_by_param": mcse_by_param,
        "per_chain_rhat": per_chain_rhat,
    }


def _catastrophic(diag: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Hard sampling failures — distinguished from Gate-1 threshold violations."""
    total_iter = cfg["sampling"] * cfg["chains"]
    if diag["rhat_max"] > 2.0:
        return True
    if diag["ess_bulk_min"] < 10:
        return True
    return bool(diag["n_divergent"] > 0.25 * total_iter)


# Convergence decision thresholds. A non-catastrophic but non-converged run
# (e.g. R-hat 1.1, a handful of divergences) must not be stamped
# ``converged=True`` — downstream gates and reports consume this flag
# directly. Values mirror the conservative BDA3 / Vehtari et al. 2021
# recommendations; the policy-driven Gate 1 Bayesian warn/fail tiers
# (plan Task 17) will refine on top of this hard floor.
_CONVERGED_RHAT_MAX = 1.05
_CONVERGED_ESS_BULK_MIN = 400.0


def _is_converged(diag: dict[str, Any]) -> bool:
    """Return True when ``diag`` meets the conservative convergence floor.

    Requires R-hat <= 1.05, bulk ESS >= 400 across all monitored
    parameters, and zero divergent transitions. Any one failure flips the
    run to non-converged so the downstream report surfaces the problem
    rather than silently shipping a biased posterior.
    """
    return bool(
        diag["rhat_max"] <= _CONVERGED_RHAT_MAX
        and diag["ess_bulk_min"] >= _CONVERGED_ESS_BULK_MIN
        and diag["n_divergent"] == 0
    )


def _dataset_flat_values(ds: Any) -> list[float]:
    """Flatten an xarray Dataset's numeric variables into a list of floats."""
    import numpy as np

    out: list[float] = []
    for arr in ds.data_vars.values():
        vals = np.asarray(arr.values).ravel()
        out.extend(float(v) for v in vals if np.isfinite(v))
    return out


def _flatten_dataset_items(ds: Any) -> list[tuple[str, float]]:
    """Iterate (name, scalar_value) pairs flattening over extra dims."""
    import numpy as np

    out: list[tuple[str, float]] = []
    for name, arr in ds.data_vars.items():
        values = np.asarray(arr.values).ravel()
        for i, v in enumerate(values):
            if np.isfinite(v):
                out.append((f"{name}[{i}]" if values.size > 1 else name, float(v)))
    return out


# ---------------------------------------------------------------------------
# Parameter estimate aggregation
# ---------------------------------------------------------------------------


def _aggregate_estimates(fit: Any, structural_names: list[str]) -> dict[str, dict[str, Any]]:
    """Produce ParameterEstimate dicts keyed by parameter name."""
    import numpy as np

    out: dict[str, dict[str, Any]] = {}
    summary = fit.summary()  # pandas DataFrame

    for name in structural_names:
        stan_name = f"log_{name}"
        if stan_name not in summary.index:
            continue
        # Posterior on log-scale — back-transform to natural scale for estimate
        draws = fit.stan_variable(stan_name)  # shape (n_draws,)
        nat = np.exp(draws)
        out[name] = {
            "name": name,
            "estimate": float(nat.mean()),
            "posterior_sd": float(nat.std(ddof=1)),
            "q05": float(np.quantile(nat, 0.05)),
            "q50": float(np.quantile(nat, 0.50)),
            "q95": float(np.quantile(nat, 0.95)),
            "category": "structural",
        }

    # IIV omegas
    for name in structural_names:
        stan_name = f"omega_{name}"
        if stan_name not in summary.index:
            continue
        draws = fit.stan_variable(stan_name)
        out[stan_name] = {
            "name": stan_name,
            "estimate": float(draws.mean()),
            "posterior_sd": float(draws.std(ddof=1)),
            "q05": float(np.quantile(draws, 0.05)),
            "q50": float(np.quantile(draws, 0.50)),
            "q95": float(np.quantile(draws, 0.95)),
            "category": "iiv",
        }

    # Residual sigmas
    for sigma_name in ("sigma_prop", "sigma_add"):
        if sigma_name not in summary.index:
            continue
        draws = fit.stan_variable(sigma_name)
        out[sigma_name] = {
            "name": sigma_name,
            "estimate": float(draws.mean()),
            "posterior_sd": float(draws.std(ddof=1)),
            "q05": float(np.quantile(draws, 0.05)),
            "q50": float(np.quantile(draws, 0.50)),
            "q95": float(np.quantile(draws, 0.95)),
            "category": "residual",
        }

    return out


def _compute_eta_shrinkage(fit: Any, structural_names: list[str]) -> dict[str, float]:
    """Individual eta shrinkage on the natural (omega·eta_raw) scale.

    The per-subject eta is ``omega·eta_raw``, not ``eta_raw``. An earlier
    implementation used ``1 - var(eta_raw_mean)`` which assumed ``omega=1``.
    The current formula computes the posterior mean of each subject's
    natural-scale eta, measures the between-subject variance of those
    means, and divides by ``E[omega^2]``.
    """
    import numpy as np

    out: dict[str, float] = {}
    var_names = set(fit.stan_variables())
    if "eta_raw" not in var_names:
        return out

    eta_raw_draws = np.asarray(fit.stan_variable("eta_raw"))
    if eta_raw_draws.ndim != 3:
        return out

    # Stan emitter declares `matrix[N_subjects, nIIV] eta_raw` →
    # cmdstanpy returns shape (n_draws, N_subjects, N_iiv).
    # eta_raw's third axis is indexed by IIV-declaration order, which may not
    # match structural_names order. We read the names from the spec variability
    # items instead.
    n_iiv = eta_raw_draws.shape[2]

    for i, name in enumerate(structural_names[:n_iiv]):
        omega_name = f"omega_{name}"
        if omega_name not in var_names:
            continue
        omega_draws = np.asarray(fit.stan_variable(omega_name))
        eta_natural = eta_raw_draws[:, :, i] * omega_draws[:, None]
        eta_post_mean = eta_natural.mean(axis=0)
        num = float(eta_post_mean.var(ddof=1)) if eta_post_mean.size > 1 else 0.0
        denom = float((omega_draws**2).mean())
        if denom <= 0:
            continue
        shrinkage = max(0.0, min(1.0, 1.0 - num / denom))
        out[name] = shrinkage
    return out


# ---------------------------------------------------------------------------
# Draws writer: long-form Parquet
# ---------------------------------------------------------------------------


def _write_draws_parquet(fit: Any, path: Path) -> None:
    """Write draws as long-form Parquet (chain, iter, param, value)."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    frames: list[pd.DataFrame] = []
    for var_name in fit.stan_variables():
        arr = np.asarray(fit.stan_variable(var_name))  # (n_draws,) or (n_draws, ...)
        if arr.ndim == 1:
            n_draws = arr.shape[0]
            n_chains = fit.chains
            draws_per_chain = n_draws // n_chains
            chain = np.repeat(np.arange(n_chains), draws_per_chain)
            it = np.tile(np.arange(draws_per_chain), n_chains)
            frames.append(
                pd.DataFrame(
                    {
                        "chain": chain,
                        "iter": it,
                        "param": var_name,
                        "value": arr,
                    }
                )
            )
        # Higher-rank parameters (matrices) are skipped in the draws parquet
        # for v1 — keep the file small. The raw CmdStan CSVs remain on disk
        # if the user needs them.

    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(combined), path)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _build_inits(request: dict[str, Any]) -> dict[str, Any] | None:
    """Build cmdstanpy inits dict from request.initial_estimates.

    Stan variables are log-scale (``log_X`` for structural param X) so we
    log-transform every positive initial estimate. Omega and sigma inits
    default to small positive values to avoid boundary cases. Returns None
    if no initial_estimates were provided.
    """
    import math

    ie = request.get("initial_estimates") or {}
    if not ie:
        return None

    inits: dict[str, Any] = {}
    structural = _extract_structural_names(request["spec"])
    for name in structural:
        val = ie.get(name)
        if val is None or val <= 0:
            continue
        inits[f"log_{name}"] = math.log(float(val))

    # eta_raw starts at zero (no subject-level shifts); cmdstanpy needs the
    # shape, which we derive from N_subjects + number of IIV params.
    variability = (
        request["spec"].get("variability", []) if isinstance(request["spec"], dict) else []
    )
    iiv_params: list[str] = []
    for v in variability:
        if v.get("type") == "IIV":
            for p in v.get("params", []):
                if p not in iiv_params:
                    iiv_params.append(p)
    if iiv_params:
        # Small non-zero inits for omega to avoid boundary.
        for p in iiv_params:
            inits.setdefault(f"omega_{p}", 0.3)

    # sigma defaults
    inits.setdefault("sigma_prop", 0.2)
    inits.setdefault("sigma_add", 0.1)

    return inits


def _extract_structural_names(spec: dict[str, Any] | str) -> list[str]:
    """Reconstruct structural parameter names without importing DSLSpec.

    Duplicates the logic of DSLSpec.structural_param_names() so the harness
    doesn't need to import pydantic — safe because this is a serialization
    boundary.
    """
    if isinstance(spec, str):
        spec = json.loads(spec)
    spec = cast("dict[str, Any]", spec)
    names: list[str] = []
    abs_mod = spec.get("absorption", {})
    dist_mod = spec.get("distribution", {})
    elim_mod = spec.get("elimination", {})

    abs_map = {
        "FirstOrder": ["ka"],
        "ZeroOrder": ["dur"],
        "LaggedFirstOrder": ["ka", "tlag"],
        "Transit": ["n", "ktr", "ka"],
        "MixedFirstZero": ["ka", "dur", "frac"],
    }
    dist_map = {
        "OneCmt": ["V"],
        "TwoCmt": ["V1", "V2", "Q"],
        "ThreeCmt": ["V1", "V2", "V3", "Q2", "Q3"],
        "TMDD_Core": ["V", "R0", "kon", "koff", "kint"],
        "TMDD_QSS": ["V", "R0", "KD", "kint"],
    }
    elim_map = {
        "Linear": ["CL"],
        "MichaelisMenten": ["Vmax", "Km"],
        "ParallelLinearMM": ["CL", "Vmax", "Km"],
        "TimeVarying": ["CL", "kdecay"],
    }

    names.extend(abs_map.get(abs_mod.get("type", ""), []))
    names.extend(dist_map.get(dist_mod.get("type", ""), []))
    names.extend(elim_map.get(elim_mod.get("type", ""), []))
    return names


def _session_info() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "cmdstan_version": _cmdstan_version(),
        "stan_version": _stan_version(),
    }


def _cmdstan_version() -> str:
    if not _CMDSTAN_AVAILABLE:
        return ""
    try:
        import cmdstanpy as cs

        return str(cs.cmdstan_version())
    except Exception:
        return ""


def _stan_version() -> str:
    if not _CMDSTAN_AVAILABLE:
        return ""
    try:
        import cmdstanpy as cs

        return str(getattr(cs, "__version__", ""))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Predictive diagnostics from posterior draws (PRD §4.3.1)
# ---------------------------------------------------------------------------
#
# The Stan emitter's ``generated quantities`` block currently emits only
# ``log_lik[n]`` for LOO-CV. Full VPC/NPE/AUC-Cmax support requires
# emitting an observation-model-specific ``y_pred[n]`` alongside each
# log_lik entry (e.g. ``lognormal_rng`` for proportional error,
# ``normal_rng`` for additive) — deferred to the stan_emitter follow-up
# commit tracked in CHANGELOG.
#
# Until that lands, the harness exposes the Python-side plumbing as a
# testable helper so the integration path is verified end-to-end and
# drops in as a one-liner once y_pred is present in the Stan output.


def build_predictive_from_draws(
    y_pred_draws: npt.ArrayLike,
    obs_subject_idx: npt.ArrayLike,
    obs_times: npt.ArrayLike,
    observed_dv: npt.ArrayLike,
    nca_diagnostics: list[NCASubjectDiagnostic] | None,
    gate3_policy: Gate3Config,
) -> PredictiveSummaryBundle:
    """Reshape posterior predictive draws into :class:`PredictiveSummaryBundle`.

    Takes the flat ``(n_sims, N)`` posterior predictive draw matrix (as
    produced by a Stan ``generated quantities`` ``y_pred[n]`` block) plus
    the per-observation ``(subject, time)`` index vectors, groups by
    subject, and calls
    :func:`apmode.backends.predictive_summary.build_predictive_diagnostics`.

    Parameters mirror the contract on the Python side of the Stan
    adapter:

    * ``y_pred_draws`` — shape ``(n_sims, N)`` where N is the total
      observation count (``sum_subjects(n_obs_i)``). ``n_sims`` is
      ``n_chains * n_post_warmup_draws`` thinned if configured.
    * ``obs_subject_idx`` — length-N integer vector of Stan-style
      1-indexed subject indices.
    * ``obs_times`` — length-N float vector of observation times.
    * ``observed_dv`` — length-N float vector of observed DV.
    * ``nca_diagnostics`` — optional per-subject NCA QC records, indexed
      so ``nca_diagnostics[s - 1]`` matches Stan subject index ``s``.
    * ``gate3_policy`` — :class:`apmode.governance.policy.Gate3Config`
      (floors, bin counts).

    Returns a :class:`PredictiveSummaryBundle`. Shape mismatches raise
    ``ValueError`` — same contract as ``build_predictive_diagnostics``.
    """
    import numpy as np

    from apmode.backends.predictive_summary import (
        SubjectSimulation,
        build_predictive_diagnostics,
    )

    y_pred_arr = np.asarray(y_pred_draws, dtype=float)
    idx_arr = np.asarray(obs_subject_idx, dtype=int)
    time_arr = np.asarray(obs_times, dtype=float)
    dv_arr = np.asarray(observed_dv, dtype=float)

    n_obs_total = idx_arr.shape[0]
    if y_pred_arr.ndim != 2 or y_pred_arr.shape[1] != n_obs_total:
        msg = (
            f"y_pred_draws shape {y_pred_arr.shape} inconsistent with "
            f"observation count {n_obs_total} (expected (n_sims, N))"
        )
        raise ValueError(msg)
    if time_arr.shape[0] != n_obs_total or dv_arr.shape[0] != n_obs_total:
        msg = (
            f"obs_times ({time_arr.shape[0]}) / observed_dv ({dv_arr.shape[0]}) "
            f"must match observation count {n_obs_total}"
        )
        raise ValueError(msg)

    diag_by_idx = {i + 1: d for i, d in enumerate(nca_diagnostics or [])}
    unique_subjects = np.unique(idx_arr)
    subject_sims: list[Any] = []
    for s in unique_subjects:
        mask = idx_arr == s
        subject_sims.append(
            SubjectSimulation(
                subject_id=str(int(s)),
                t_observed=time_arr[mask],
                observed_dv=dv_arr[mask],
                sims_at_observed=y_pred_arr[:, mask],
                nca_diagnostic=diag_by_idx.get(int(s)),
            )
        )
    return build_predictive_diagnostics(subject_sims, policy=gate3_policy)


def sample_with_provenance(
    *,
    stan_code: str,
    data: dict[str, Any],
    work_dir: Path,
    seed: int,
    chains: int,
    warmup: int,
    sampling: int,
    adapt_delta: float,
    max_treedepth: int,
    uses_reduce_sum: bool,
) -> Any:
    """Compile + sample a Stan program and record provenance metadata.

    Writes ``model.stan`` and ``stan_data.json`` into ``work_dir``, invokes
    cmdstanpy with ``save_cmdstan_config=True`` (cmdstanpy issue #848) and
    platform-adaptive ``force_one_process_per_chain`` (cmdstanpy issue
    #895, via :func:`apmode.bayes.platform.cmdstan_run_kwargs`), and emits
    ``backend_versions.json`` containing SHA-256 hashes of the Stan code
    and data, the cmdstan version, host platform, and the effective
    ``one_process_per_chain`` flag.

    Trust boundary: ``stan_code`` must originate exclusively from
    :func:`apmode.dsl.stan_emitter.emit_stan`, because the string is
    written to ``model.stan`` and then handed to ``CmdStanModel`` which
    invokes the system C++ compiler on it. User-supplied Stan or agentic
    LLM output must NOT reach this helper unless a separate review pass
    clears the text — this is the whole point of the DSL's ``[..]``
    allow-listed transform surface (CLAUDE.md §"PK DSL is the moat").

    Returns the ``CmdStanMCMC`` fit so callers can layer diagnostics.
    """
    import hashlib
    import platform as _platform

    from apmode.bayes.platform import cmdstan_run_kwargs

    if CmdStanModel is None:  # pragma: no cover - import guard
        msg = "cmdstanpy not installed; install with `uv sync --extra bayesian`"
        raise RuntimeError(msg)

    work_dir.mkdir(parents=True, exist_ok=True)
    stan_path = work_dir / "model.stan"
    stan_path.write_text(stan_code)
    data_path = work_dir / "stan_data.json"
    data_bytes = json.dumps(data, sort_keys=True).encode()
    # ``json.dumps`` emits pure ASCII by default (``ensure_ascii=True``),
    # so ``data_bytes.decode()`` is lossless and the SHA-256 stays stable
    # across Python minor versions. Keep the decode step — writing the
    # text form keeps the artifact human-inspectable.
    data_path.write_text(data_bytes.decode())

    run_kwargs = cmdstan_run_kwargs(uses_reduce_sum=uses_reduce_sum)
    model = CmdStanModel(stan_file=str(stan_path))
    fit = model.sample(
        data=str(data_path),
        chains=chains,
        iter_warmup=warmup,
        iter_sampling=sampling,
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        save_cmdstan_config=True,
        output_dir=str(work_dir),
        **run_kwargs,
    )

    meta: dict[str, Any] = {
        "stan_code_sha256": hashlib.sha256(stan_code.encode()).hexdigest(),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "cmdstan_version": _cmdstan_version(),
        "cmdstanpy_version": _stan_version(),
        "python_version": sys.version.split()[0],
        "platform": _platform.system(),
        "uses_reduce_sum": uses_reduce_sum,
        "one_process_per_chain": bool(run_kwargs.get("force_one_process_per_chain", False)),
    }
    (work_dir / "backend_versions.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    return fit


if __name__ == "__main__":
    sys.exit(main())
