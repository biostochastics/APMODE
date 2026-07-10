# SPDX-License-Identifier: GPL-2.0-or-later
"""V&V40-style Risk Grading Report generator (Gate 2.5 companion to credibility.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmode.bundle.models import CredibilityActivity, RiskGradingReport
from apmode.governance.gates import _RIGOR_ORDER, _obtained_rigor
from apmode.report.credibility import _compute_result_sha256

if TYPE_CHECKING:
    from pathlib import Path

    from apmode.bundle.models import BackendResult, CredibilityContext
    from apmode.governance.policy import Gate25Config


def generate_risk_grading_report(
    result: BackendResult,
    lane: str,
    credibility_context: CredibilityContext,
    gate25_config: Gate25Config,
    *,
    source_result_path: Path | str | None = None,
) -> RiskGradingReport:
    """Generate a V&V40 risk-grading report for a candidate.

    Mirrors generate_credibility_report's provenance-pair pattern
    (source_result_path + sha256) and reuses gates.py's tier/rigor
    computation via RiskGradingConfig.tier_for + _obtained_rigor so the
    report can never disagree with the Gate 2.5 decision that gated it.
    """
    rg = gate25_config.risk_grading
    influence = credibility_context.model_influence or "high"
    consequence = credibility_context.decision_consequence or "high"
    tier = rg.tier_for(influence, consequence) if rg is not None else "high"
    factors = rg.credibility_factors.get(tier, {}) if rg is not None else {}

    activities: list[CredibilityActivity] = []
    gaps: list[str] = []
    for factor, target in factors.items():
        obtained = _obtained_rigor(result, factor)
        is_gap = obtained is None or _RIGOR_ORDER[obtained] < _RIGOR_ORDER[target]
        activities.append(
            CredibilityActivity(
                factor=factor,
                target_rigor=target,
                obtained_rigor=obtained,
                gap=is_gap,
                evidence_ref=f"diagnostics.{factor}",
            )
        )
        if is_gap:
            gaps.append(factor)

    source_path_str = str(source_result_path) if source_result_path is not None else None
    return RiskGradingReport(
        candidate_id=result.model_id,
        context_of_use=(credibility_context.context_of_use or f"{lane} lane: population PK model"),
        model_influence=influence,
        decision_consequence=consequence,
        risk_tier=tier,
        credibility_activities=activities,
        gaps=gaps,
        source_result_path=source_path_str,
        source_result_sha256=_compute_result_sha256(result),
    )
