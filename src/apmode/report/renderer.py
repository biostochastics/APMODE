# SPDX-License-Identifier: GPL-2.0-or-later
"""Run report renderer (PRD §4.3.3, Phase 3).

Renders a human-readable Markdown report from an APMODE run bundle.
Designed for terminal display, PDF export, or documentation embedding.

The report presents all information available from a completed run:
data summary, evidence manifest, dispatch decisions, gate funnel
(with per-check pass/fail detail), ranked survivors, per-candidate
parameter tables, diagnostics, convergence, and credibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apmode.bundle.models import (
        BackendResult,
        CredibilityReport,
        DataManifest,
        EvidenceManifest,
        FailedCandidate,
        GateResult,
        Ranking,
    )


def render_markdown_to_html(markdown_text: str, *, title: str = "APMODE Run Report") -> str:
    """Render a Markdown report to a standalone HTML document.

    Uses ``rich.markdown.Markdown`` (already a project dependency) to
    produce a styled HTML page with the same visual layout as the
    terminal report. Produces inline-styled, self-contained HTML —
    no external CSS or JS needed, safe to email or host statically.

    rc1 scope: terminal-style HTML rendering. For full web-native HTML
    (anchors, responsive tables, collapsible sections), pipe ``report.md``
    through ``pandoc`` or a static-site generator.
    """
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console(record=True, width=120, file=None)
    console.print(Markdown(markdown_text))
    html_doc = console.export_html(inline_styles=True)
    # Inject a <title> so the browser tab and archived file are labeled.
    return html_doc.replace(
        '<head>\n<meta charset="UTF-8">',
        f'<head>\n<meta charset="UTF-8">\n<title>{title}</title>',
        1,
    )


def render_run_report(
    *,
    run_id: str,
    lane: str,
    manifest: DataManifest,
    evidence: EvidenceManifest,
    ranked: list[BackendResult],
    ranking: Ranking | None = None,
    credibility_reports: list[CredibilityReport] | None = None,
    gate_results: list[GateResult] | None = None,
    failed_candidates: list[FailedCandidate] | None = None,
    failed_count: int = 0,
    total_candidates: int = 0,
    backends_dispatched: list[str] | None = None,
    seed: int | None = None,
    policy_version: str | None = None,
) -> str:
    """Render a complete run report as Markdown."""
    cred_map: dict[str, CredibilityReport] = {}
    if credibility_reports:
        cred_map = {r.candidate_id: r for r in credibility_reports}

    gate_map: dict[str, list[GateResult]] = {}
    if gate_results:
        for gr in gate_results:
            gate_map.setdefault(gr.candidate_id, []).append(gr)

    sections = [
        _header(run_id, lane, seed, policy_version),
        _data_summary(manifest, evidence),
        _dispatch_summary(lane, backends_dispatched or []),
        _gate_funnel(total_candidates, len(ranked), failed_count),
    ]

    if ranking and ranking.ranked_candidates:
        sections.append(_ranking_table(ranking))

    # Detailed candidate sections for survivors
    for i, result in enumerate(ranked):
        cred = cred_map.get(result.model_id)
        gates = gate_map.get(result.model_id, [])
        sections.append(_candidate_detail(result, rank=i + 1, credibility=cred, gates=gates))

    # Failed candidates section
    if failed_candidates:
        sections.append(_failed_candidates_section(failed_candidates, gate_map))

    sections.append(_footer())
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _header(
    run_id: str,
    lane: str,
    seed: int | None,
    policy_version: str | None,
) -> str:
    lane_label = {
        "submission": "Submission",
        "discovery": "Discovery",
        "optimization": "Translational Optimization",
    }.get(lane, lane.title())

    lines = [
        "# APMODE Run Report\n",
        "| | |",
        "|---|---|",
        f"| **Run** | `{run_id}` |",
        f"| **Lane** | {lane_label} |",
        f"| **Date** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |",
    ]
    if seed is not None:
        lines.append(f"| **Seed** | {seed} |")
    if policy_version:
        lines.append(f"| **Policy** | {policy_version} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data Summary & Evidence
# ---------------------------------------------------------------------------


def _data_summary(manifest: DataManifest, evidence: EvidenceManifest) -> str:
    nl_cl = evidence.nonlinear_clearance_evidence_strength
    conf_str = ""
    if evidence.nonlinear_clearance_confidence is not None:
        conf_str = f" ({evidence.nonlinear_clearance_confidence:.0%} confidence)"

    lines = [
        "\n---\n",
        "## Data Profile\n",
        "| Property | Value |",
        "|----------|-------|",
        f"| Subjects | {manifest.n_subjects} |",
        f"| Observations | {manifest.n_observations} |",
        f"| Doses | {manifest.n_doses} |",
        f"| Obs/subject ratio | {manifest.n_observations / max(manifest.n_subjects, 1):.1f} |",
        f"| Richness | **{evidence.richness_category}** |",
        f"| Route certainty | {evidence.route_certainty} |",
        f"| Absorption complexity | {evidence.absorption_complexity} |",
        f"| Absorption phase coverage | {evidence.absorption_phase_coverage} |",
        f"| Elimination phase coverage | {evidence.elimination_phase_coverage} |",
        f"| Nonlinear CL signature | {nl_cl}{conf_str} |",
        f"| Identifiability ceiling | {evidence.identifiability_ceiling} |",
        f"| BLQ burden | {evidence.blq_burden:.1%} |",
        f"| Covariate burden | {evidence.covariate_burden} |",
        f"| Correlated covariates | {'Yes' if evidence.covariate_correlated else 'No'} |",
        f"| Protocol heterogeneity | {evidence.protocol_heterogeneity} |",
    ]
    return "\n".join(lines)


def _dispatch_summary(lane: str, backends: list[str]) -> str:
    if not backends:
        return ""
    backend_str = ", ".join(f"`{b}`" for b in backends)
    return f"\n### Dispatch\n\nBackends dispatched for **{lane}** lane: {backend_str}\n"


# ---------------------------------------------------------------------------
# Gate Funnel
# ---------------------------------------------------------------------------


def _gate_funnel(total: int, survivors: int, failed: int) -> str:
    if total == 0:
        return "\n---\n\n## Governance Funnel\n\nNo candidates evaluated.\n"

    survival_rate = survivors / total * 100 if total > 0 else 0

    lines = [
        "\n---\n",
        "## Governance Funnel\n",
        "```",
        f"  {total:>3} candidates entered",
        "   |",
        "   |  Gate 1: Technical Validity",
        "   |  Gate 2: Lane Admissibility",
        "   |  Gate 2.5: Credibility Qualification",
        "   v",
        f"  {survivors:>3} survived  ({survival_rate:.0f}%)",
        f"  {failed:>3} eliminated",
        "```",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranking Table
# ---------------------------------------------------------------------------


def _ranking_table(ranking: Ranking) -> str:
    lines = [
        "\n---\n",
        "## Final Ranking (Gate 3)\n",
        f"Metric: **{ranking.ranking_metric.upper()}**"
        + (f"  Best: **`{ranking.best_candidate_id}`**\n" if ranking.best_candidate_id else "\n"),
        "| Rank | Candidate | Backend | BIC | AIC | k |",
        "|-----:|-----------|---------|----:|----:|--:|",
    ]
    for rc in ranking.ranked_candidates:
        aic_str = f"{rc.aic:.1f}" if rc.aic is not None else "--"
        lines.append(
            f"| {rc.rank} | `{rc.candidate_id}` | {rc.backend} "
            f"| {rc.bic:.1f} | {aic_str} | {rc.n_params} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Candidate Detail
# ---------------------------------------------------------------------------


def _candidate_detail(
    result: BackendResult,
    *,
    rank: int,
    credibility: CredibilityReport | None,
    gates: list[GateResult],
) -> str:
    lines = [
        "\n---\n",
        f"## #{rank}  `{result.model_id}`\n",
    ]

    # Summary line
    parts = [f"**Backend** {result.backend}"]
    if result.ofv is not None:
        parts.append(f"**OFV** {result.ofv:.1f}")
    if result.aic is not None:
        parts.append(f"**AIC** {result.aic:.1f}")
    if result.bic is not None:
        parts.append(f"**BIC** {result.bic:.1f}")
    parts.append(f"**Wall** {result.wall_time_seconds:.1f}s")
    parts.append(f"**Init** {result.initial_estimate_source}")
    lines.append("  ".join(parts) + "\n")

    # Parameters — structural
    structural = {
        k: v for k, v in result.parameter_estimates.items() if v.category == "structural"
    }
    if structural:
        lines.append("### Structural Parameters\n")
        lines.append("| Parameter | Estimate | SE | RSE (%) | CI 95% |")
        lines.append("|-----------|--------:|---:|--------:|--------|")
        for name, pe in structural.items():
            se = f"{pe.se:.4g}" if pe.se is not None else "--"
            rse = f"{pe.rse:.1f}" if pe.rse is not None else "--"
            if pe.ci95_lower is not None and pe.ci95_upper is not None:
                ci = f"[{pe.ci95_lower:.3g}, {pe.ci95_upper:.3g}]"
            else:
                ci = "--"
            lines.append(f"| {name} | {pe.estimate:.4g} | {se} | {rse} | {ci} |")

    # Parameters — IIV/IOV
    iiv = {k: v for k, v in result.parameter_estimates.items() if v.category in ("iiv", "iov")}
    if iiv:
        lines.append("\n### Random Effects\n")
        lines.append("| Parameter | Type | Estimate | Shrinkage (%) |")
        lines.append("|-----------|------|--------:|--------------:|")
        for name, pe in iiv.items():
            shrink = result.eta_shrinkage.get(name)
            s_str = f"{shrink:.1f}" if shrink is not None else "--"
            lines.append(f"| {name} | {pe.category.upper()} | {pe.estimate:.4g} | {s_str} |")

    # Parameters — residual
    residual = {k: v for k, v in result.parameter_estimates.items() if v.category == "residual"}
    if residual:
        lines.append("\n### Residual Error\n")
        for name, pe in residual.items():
            rse = f" (RSE {pe.rse:.1f}%)" if pe.rse is not None else ""
            lines.append(f"- **{name}** = {pe.estimate:.4g}{rse}")
        lines.append("")

    # Diagnostics
    lines.append("### Diagnostics\n")
    gof = result.diagnostics.gof
    lines.append("| Metric | Value | Threshold |")
    lines.append("|--------|------:|-----------|")
    lines.append(f"| CWRES mean | {gof.cwres_mean:.4f} | |")
    lines.append(f"| CWRES SD | {gof.cwres_sd:.4f} | ~1.0 |")
    lines.append(f"| Outlier fraction | {gof.outlier_fraction:.1%} | <5% |")
    if gof.obs_vs_pred_r2 is not None:
        lines.append(f"| Obs vs Pred R^2 | {gof.obs_vs_pred_r2:.4f} | |")

    ident = result.diagnostics.identifiability
    if ident.condition_number is not None:
        status = "ill-conditioned" if ident.ill_conditioned else "OK"
        lines.append(f"| Condition number | {ident.condition_number:.1f} | {status} |")
    if ident.profile_likelihood_ci:
        bounded = [k for k, v in ident.profile_likelihood_ci.items() if v]
        unbounded = [k for k, v in ident.profile_likelihood_ci.items() if not v]
        if bounded:
            lines.append(f"| PL CI bounded | {', '.join(bounded)} | |")
        if unbounded:
            lines.append(f"| PL CI **unbounded** | {', '.join(unbounded)} | |")

    # BLQ
    blq = result.diagnostics.blq
    if blq.n_blq > 0:
        lines.append(
            f"\n**BLQ handling**: {blq.method.upper()} ({blq.n_blq} obs, {blq.blq_fraction:.1%})"
        )
        if blq.lloq is not None:
            lines.append(f"  LLOQ = {blq.lloq}")

    # VPC
    vpc = result.diagnostics.vpc
    if vpc:
        lines.append("\n### VPC Summary\n")
        cov_parts = [
            f"p{int(p)}: {vpc.coverage.get(f'p{int(p)}', 0):.1%}" for p in vpc.percentiles
        ]
        lines.append(f"Coverage: {', '.join(cov_parts)}")
        pc = "prediction-corrected" if vpc.prediction_corrected else "standard"
        lines.append(f"  {vpc.n_bins} bins, {pc}")

    # Split GOF
    sg = result.diagnostics.split_gof
    if sg:
        lines.append("\n### Split-Sample GOF\n")
        lines.append("| Set | CWRES mean | Outlier % | n |")
        lines.append("|-----|----------:|---------:|--:|")
        lines.append(
            f"| Train | {sg.train_cwres_mean:.4f} "
            f"| {sg.train_outlier_fraction:.1%} | {sg.n_train} |"
        )
        lines.append(
            f"| Test | {sg.test_cwres_mean:.4f} | {sg.test_outlier_fraction:.1%} | {sg.n_test} |"
        )

    # Convergence
    cm = result.convergence_metadata
    lines.append("\n### Convergence\n")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Method | {cm.method} |")
    lines.append(f"| Status | **{cm.minimization_status}** |")
    lines.append(f"| Converged | {'Yes' if cm.converged else 'No'} |")
    lines.append(f"| Iterations | {cm.iterations} |")
    if cm.gradient_norm is not None:
        lines.append(f"| Gradient norm | {cm.gradient_norm:.2e} |")
    lines.append(f"| Wall time | {cm.wall_time_seconds:.1f}s |")

    # Backend versions
    if result.backend_versions:
        lines.append("\n### Software Versions\n")
        for k, v in result.backend_versions.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    # Gate details for this candidate
    if gates:
        lines.append("### Gate Results\n")
        for gr in gates:
            status = "PASS" if gr.passed else "FAIL"
            lines.append(f"**{gr.gate_name}** {status}\n")
            lines.append("| Check | Result | Observed | Threshold |")
            lines.append("|-------|--------|----------|-----------|")
            for chk in gr.checks:
                icon = "Pass" if chk.passed else "**FAIL**"
                obs = _format_observed(chk.observed)
                thr = str(chk.threshold) if chk.threshold is not None else "--"
                if chk.units:
                    thr += f" {chk.units}"
                lines.append(f"| {chk.check_id} | {icon} | {obs} | {thr} |")
            lines.append("")

    # Credibility
    if credibility:
        lines.append("### Credibility Assessment\n")
        lines.append(f"**Context of use**: {credibility.context_of_use}\n")
        lines.append(f"**Data adequacy**: {credibility.data_adequacy}\n")

        if credibility.model_credibility:
            lines.append("| Criterion | Value |")
            lines.append("|-----------|-------|")
            for ck, cv in credibility.model_credibility.items():
                lines.append(f"| {ck.replace('_', ' ').title()} | {cv} |")

        if credibility.limitations:
            lines.append("\n**Limitations**:\n")
            for lim in credibility.limitations:
                lines.append(f"- {lim}")

        if credibility.ml_transparency:
            lines.append(f"\n**ML Transparency**: {credibility.ml_transparency}")

        if credibility.sensitivity_results:
            lines.append("\n**Sensitivity**:\n")
            for sk, sv in credibility.sensitivity_results.items():
                lines.append(f"- {sk}: {sv}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Failed Candidates
# ---------------------------------------------------------------------------


def _failed_candidates_section(
    failed: list[FailedCandidate],
    gate_map: dict[str, list[GateResult]],
) -> str:
    lines = [
        "\n---\n",
        "## Eliminated Candidates\n",
        "| Candidate | Backend | Gate | Failed Checks | Reason |",
        "|-----------|---------|------|---------------|--------|",
    ]
    for fc in failed:
        checks_str = ", ".join(fc.failed_checks) if fc.failed_checks else "--"
        reason = fc.summary_reason
        if len(reason) > 80:
            reason = reason[:80] + "..."
        lines.append(
            f"| `{fc.candidate_id}` | {fc.backend} | {fc.gate_failed} | {checks_str} | {reason} |"
        )

    # Detailed gate check breakdown for failed candidates
    detailed = [fc for fc in failed if fc.candidate_id in gate_map]
    if detailed:
        lines.append("\n### Failure Details\n")
        for fc in detailed:
            lines.append(f"#### `{fc.candidate_id}`\n")
            for gr in gate_map[fc.candidate_id]:
                lines.append(f"**{gr.gate_name}**\n")
                for chk in gr.checks:
                    if not chk.passed:
                        obs = _format_observed(chk.observed)
                        thr = str(chk.threshold) if chk.threshold is not None else ""
                        lines.append(f"- **{chk.check_id}**: observed {obs}, threshold {thr}")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_observed(value: float | bool | str) -> str:
    """Format an observed gate check value for display."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) > 1000:
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def _footer() -> str:
    return (
        "\n---\n\n"
        "*Generated by APMODE. "
        "All candidates evaluated under identical evidence standards. "
        "Gate thresholds are versioned policy artifacts.*\n"
    )
