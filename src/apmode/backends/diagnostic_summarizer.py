# SPDX-License-Identifier: GPL-2.0-or-later
"""Diagnostic summarizer for agentic LLM context (PRD §4.2.6).

Converts BackendResult into a concise structured summary suitable for the LLM
system/user prompt. The LLM receives aggregated diagnostics — not raw data —
to reason about model misfit and propose Formular transforms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apmode.bundle.models import (
        BackendResult,
        ImputationStabilityEntry,
        ImputationStabilityManifest,
    )


# Allow-list of fields permitted in LLM context. Any key produced by
# summarize_diagnostics that is not in this set is dropped by redact_for_llm.
# This is the single enforcement gate for PRD §10 "LLM inputs aggregate-only":
# update here (and the matching test) when new aggregate signals are added.
_LLM_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "converged",
        "ofv",
        "aic",
        "bic",
        "parameters",
        "cwres_mean",
        "cwres_sd",
        "outlier_fraction",
        "eta_shrinkage",
        "method",
        "wall_time_seconds",
        "vpc_coverage",
        "split_gof",
        "identifiability_warning",
        # Pooled / stability summary fields (MI runs, directive.llm_pooled_only=True).
        # These replace per-imputation diagnostics so the agentic backend cannot
        # cherry-pick a "lucky" imputation (PRD §4.2.1).
        "pooled_ofv",
        "pooled_aic",
        "pooled_bic",
        "convergence_rate",
        "rank_stability",
        "within_between_var_ratio",
        "covariate_sign_consistency",
        "imputation_method",
        "m_imputations",
    }
)

_LLM_ALLOWED_PARAM_KEYS: frozenset[str] = frozenset({"estimate", "rse_pct"})


def summarize_diagnostics(result: BackendResult) -> dict[str, Any]:
    """Convert BackendResult into a structured summary dict."""
    gof = result.diagnostics.gof
    params: dict[str, dict[str, float | str | None]] = {}
    for name, pe in result.parameter_estimates.items():
        if pe.category == "structural":
            params[name] = {
                "estimate": pe.estimate,
                "rse_pct": pe.rse,
            }

    summary: dict[str, Any] = {
        "converged": result.converged,
        "ofv": result.ofv,
        "aic": result.aic,
        "bic": result.bic,
        "parameters": params,
        "cwres_mean": gof.cwres_mean,
        "cwres_sd": gof.cwres_sd,
        "outlier_fraction": gof.outlier_fraction,
        "eta_shrinkage": dict(result.eta_shrinkage),
        "method": result.convergence_metadata.method,
        "wall_time_seconds": result.wall_time_seconds,
    }

    if result.diagnostics.vpc is not None:
        summary["vpc_coverage"] = dict(result.diagnostics.vpc.coverage)

    if result.diagnostics.split_gof is not None:
        sg = result.diagnostics.split_gof
        summary["split_gof"] = {
            "train_cwres_mean": sg.train_cwres_mean,
            "test_cwres_mean": sg.test_cwres_mean,
            "train_outlier_fraction": sg.train_outlier_fraction,
            "test_outlier_fraction": sg.test_outlier_fraction,
        }

    ident = result.diagnostics.identifiability
    if ident.ill_conditioned:
        summary["identifiability_warning"] = f"Ill-conditioned (CN={ident.condition_number})"

    return summary


def redact_for_llm(summary: dict[str, Any]) -> dict[str, Any]:
    """Drop any key outside the LLM allow-list.

    Every code path that ships a diagnostic summary to a third-party LLM must
    route the dict through this function first. Unknown keys are silently
    dropped rather than raising, so upstream summarizer evolution does not
    leak new fields by default (fail-closed).
    """
    redacted: dict[str, Any] = {}
    for key, value in summary.items():
        if key not in _LLM_ALLOWED_TOP_LEVEL_KEYS:
            continue
        if key == "parameters" and isinstance(value, dict):
            redacted[key] = {
                pname: {pk: pv for pk, pv in pinfo.items() if pk in _LLM_ALLOWED_PARAM_KEYS}
                for pname, pinfo in value.items()
                if isinstance(pinfo, dict)
            }
        else:
            redacted[key] = value
    return redacted


def summarize_for_llm(
    result: BackendResult,
    iteration: int,
    max_iterations: int,
    search_history: list[dict[str, Any]] | None = None,
) -> str:
    """Format a human-readable diagnostic summary for the LLM prompt.

    Highlights actionable signals: high CWRES bias, non-convergence,
    high shrinkage, VPC deficiencies.

    The summary is routed through ``redact_for_llm`` before formatting to
    enforce the PRD §10 aggregate-only allow-list on every field that
    reaches a third-party LLM provider.
    """
    s = redact_for_llm(summarize_diagnostics(result))
    lines: list[str] = []

    lines.append(f"## Iteration {iteration}/{max_iterations}")
    lines.append("")

    # Convergence
    if not s["converged"]:
        lines.append("**WARNING: Model did not converge.**")
        lines.append(f"  Method: {s['method']}")
    else:
        lines.append(f"Converged ({s['method']}). OFV={s['ofv']}, BIC={s['bic']}, AIC={s['aic']}")

    # Parameters
    lines.append("")
    lines.append("### Structural Parameters")
    for name, pinfo in s["parameters"].items():
        rse_str = f"RSE={pinfo['rse_pct']:.1f}%" if pinfo["rse_pct"] else "RSE=N/A"
        lines.append(f"  {name} = {pinfo['estimate']:.4g} ({rse_str})")

    # CWRES
    lines.append("")
    lines.append("### Residual Diagnostics")
    cwres_mean = s["cwres_mean"]
    lines.append(f"  CWRES mean = {cwres_mean:.4f}, SD = {s['cwres_sd']:.4f}")
    if abs(cwres_mean) > 0.3:
        lines.append(f"  **High CWRES bias ({cwres_mean:.4f})** — suggests systematic misfit.")
    lines.append(f"  Outlier fraction = {s['outlier_fraction']:.4f}")

    # Shrinkage
    shrinkage = s["eta_shrinkage"]
    if shrinkage:
        high_shrink = {k: v for k, v in shrinkage.items() if v > 30}
        if high_shrink:
            lines.append(f"  **High shrinkage:** {high_shrink}")

    # VPC
    if "vpc_coverage" in s:
        lines.append(f"  VPC coverage: {s['vpc_coverage']}")

    # Identifiability
    if "identifiability_warning" in s:
        lines.append(f"  **{s['identifiability_warning']}**")

    # History
    if search_history:
        lines.append("")
        lines.append("### Search History (recent)")
        for entry in search_history[-5:]:
            lines.append(
                f"  - {entry.get('model_id', '?')}: "
                f"BIC={entry.get('bic', '?')}, "
                f"converged={entry.get('converged', '?')}"
            )

    return "\n".join(lines)


def summarize_stability_diagnostics(
    stability: ImputationStabilityEntry,
    manifest: ImputationStabilityManifest,
) -> dict[str, Any]:
    """Produce a pooled-only summary dict for the LLM from stability data.

    Used when ``MissingDataDirective.llm_pooled_only`` is True. The output
    only includes pooled scores and cross-imputation stability metrics —
    never per-imputation diagnostics. Raw per-imputation OFV/BIC/parameter
    tables are deliberately excluded to prevent cherry-picking (PRD §4.2.1).
    """
    summary: dict[str, Any] = {
        "imputation_method": manifest.method,
        "m_imputations": manifest.m,
        "pooled_ofv": stability.pooled_ofv,
        "pooled_aic": stability.pooled_aic,
        "pooled_bic": stability.pooled_bic,
        "convergence_rate": stability.convergence_rate,
        "rank_stability": stability.rank_stability,
    }
    if stability.within_between_var_ratio is not None:
        summary["within_between_var_ratio"] = stability.within_between_var_ratio
    if stability.covariate_sign_consistency:
        summary["covariate_sign_consistency"] = dict(stability.covariate_sign_consistency)
    return summary


def summarize_stability_for_llm(
    stability: ImputationStabilityEntry,
    manifest: ImputationStabilityManifest,
    iteration: int,
    max_iterations: int,
    search_history: list[dict[str, Any]] | None = None,
) -> str:
    """Format the pooled-only stability summary for the LLM prompt.

    Returned text is routed through ``redact_for_llm`` before formatting
    so the allow-list enforcement gate still applies — new fields need to
    be added to ``_LLM_ALLOWED_TOP_LEVEL_KEYS`` to reach the LLM.
    """
    s = redact_for_llm(summarize_stability_diagnostics(stability, manifest))
    lines: list[str] = []

    lines.append(f"## Iteration {iteration}/{max_iterations}  (pooled across m imputations)")
    lines.append("")
    lines.append(
        f"Imputation method: {s.get('imputation_method', '?')}  "
        f"(m = {s.get('m_imputations', '?')})"
    )
    lines.append("")

    lines.append("### Pooled Fit Criteria (Rubin arithmetic mean)")
    lines.append(
        f"  Pooled OFV = {s.get('pooled_ofv')}, "
        f"AIC = {s.get('pooled_aic')}, "
        f"BIC = {s.get('pooled_bic')}"
    )

    lines.append("")
    lines.append("### Cross-Imputation Stability")
    conv_rate = s.get("convergence_rate")
    rank_stab = s.get("rank_stability")
    lines.append(f"  Convergence rate: {conv_rate}")
    lines.append(f"  Rank stability (top-K): {rank_stab}")
    if "within_between_var_ratio" in s:
        lines.append(f"  Within/between variance proxy: {s['within_between_var_ratio']:.3f}")
    # Flag clear instability signals for the LLM's reasoning.
    if isinstance(conv_rate, (int, float)) and conv_rate < 0.5:
        lines.append(
            "  **WARNING: Candidate converged on <50% of imputations — "
            "structural instability likely.**"
        )
    if isinstance(rank_stab, (int, float)) and rank_stab < 0.5:
        lines.append(
            "  **WARNING: Rank-unstable across imputations — "
            "covariate effect sensitive to imputation draw.**"
        )

    if "covariate_sign_consistency" in s:
        lines.append("")
        lines.append("### Covariate-Effect Sign Consistency")
        for name, frac in s["covariate_sign_consistency"].items():
            lines.append(f"  {name}: {frac:.2f}")

    if search_history:
        lines.append("")
        lines.append("### Search History (recent, pooled)")
        for entry in search_history[-5:]:
            lines.append(
                f"  - {entry.get('model_id', '?')}: "
                f"pooled_BIC={entry.get('bic', '?')}, "
                f"converged={entry.get('converged', '?')}"
            )

    return "\n".join(lines)
