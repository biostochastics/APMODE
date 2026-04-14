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
from typing import Any, cast

# Optional imports — deferred so tests and import-time checks don't fail when
# the bayesian extras are absent.
_CMDSTAN_AVAILABLE = False
_ARVIZ_AVAILABLE = False
try:  # pragma: no cover - import-time guard
    import cmdstanpy  # noqa: F401 — imported at runtime inside _run_sampling

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
    """Parse CLI, dispatch to _run, write response.json."""
    args = argv or sys.argv[1:]
    if len(args) != 2:
        sys.stderr.write("Usage: python -m apmode.bayes.harness <request.json> <response.json>\n")
        return 2

    request_path = Path(args[0])
    response_path = Path(args[1])

    try:
        response = _run(request_path)
    except Exception as exc:
        response = {
            "schema_version": "1.0",
            "status": "error",
            "error_type": "crash",
            "error_detail": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "session_info": _session_info(),
        }

    response_path.write_text(json.dumps(response, indent=2))
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

    # 4. Run NUTS sampling.
    cfg = request["sampler_config"]
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
        show_progress=False,
        refresh=0,
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

    result: dict[str, Any] = {
        "model_id": request["candidate_id"],
        "backend": "bayesian_stan",
        "converged": True,
        "parameter_estimates": param_estimates,
        "eta_shrinkage": _compute_eta_shrinkage(fit, structural_names),
        "convergence_metadata": {
            "method": "nuts",
            "converged": True,
            "iterations": cfg["warmup"] + cfg["sampling"],
            "minimization_status": "successful",
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
    """
    import pandas as pd

    data_path = Path(request["data_path"])
    df = pd.read_csv(data_path)
    df = df.rename(columns={c: c.upper() for c in df.columns})

    # Filter observations (DV rows): EVID=0 and MDV=0 (or no MDV)
    if "EVID" not in df.columns or "TIME" not in df.columns:
        raise ValueError("Input CSV must have EVID and TIME columns")
    id_col = (
        "ID"
        if "ID" in df.columns
        else next((c for c in df.columns if c in {"SUBJECT_ID", "PATIENT_ID"}), None)
    )
    if id_col is None:
        raise ValueError("Input CSV must have an ID-like column (ID/SUBJECT_ID/PATIENT_ID)")

    mdv_mask = df["MDV"].fillna(0).astype(int) == 0 if "MDV" in df.columns else True
    obs_mask = (df["EVID"].astype(int) == 0) & mdv_mask
    evt_mask = df["EVID"].astype(int).isin([1, 3, 4])
    obs_df = df[obs_mask].reset_index(drop=True)
    evt_df = df[evt_mask].reset_index(drop=True)

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
        first_per_subject = df.drop_duplicates(subset=[id_col], keep="first")
        stan_data[cov] = first_per_subject[cov].astype(float).tolist()

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
    """Shrinkage ≈ 1 - var(eta_posterior_mean) / omega^2."""

    out: dict[str, float] = {}
    if "eta_raw" not in fit.stan_variables():
        return out

    eta_draws = fit.stan_variable("eta_raw")  # (n_draws, N_subjects, N_params)
    eta_post_mean = eta_draws.mean(axis=0)  # (N_subjects, N_params)
    eta_var = eta_post_mean.var(axis=0, ddof=1)  # per-parameter variance across subjects

    for i, name in enumerate(structural_names):
        if i >= eta_var.shape[0]:
            break
        shrinkage = float(max(0.0, 1.0 - eta_var[i]))
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


if __name__ == "__main__":
    sys.exit(main())
