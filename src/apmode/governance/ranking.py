# SPDX-License-Identifier: GPL-2.0-or-later
"""Cross-paradigm ranking via simulation-based metrics (PRD SS4.3.1).

When candidates from different backends survive Gates 1+2+2.5, ranking
cannot use NLPD (observation models differ). Instead, ranking uses:
  1. VPC coverage concordance (how well each model's VPC covers the data)
  2. AUC/Cmax bioequivalence (80-125% GMR between model predictions)
  3. Normalized Prediction Error (NPE) on held-out data

NLPD is retained within-paradigm only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult


@dataclass(frozen=True)
class CrossParadigmMetrics:
    """Metrics for a single candidate in cross-paradigm comparison."""

    candidate_id: str
    backend: str
    vpc_concordance: float  # 0-1, how well VPC matches data
    npe: float  # normalized prediction error (lower is better)
    composite_score: float  # weighted combination


@dataclass
class CrossParadigmRankingResult:
    """Full simulation-based ranking outcome.

    ``qualification_reason`` captures *why* simulation metrics were used
    — cross-paradigm vs. mixed BLQ within a single paradigm (PRD §10 Q2).
    Downstream report generation quotes this string verbatim so reviewers
    see the explicit comparability justification.
    """

    is_cross_paradigm: bool
    ranked_candidates: list[CrossParadigmMetrics] = field(default_factory=list)
    backends_compared: list[str] = field(default_factory=list)
    ranking_method: str = "simulation_based"
    qualified_comparison: bool = True
    qualification_reason: str = ""


def is_cross_paradigm(survivors: list[BackendResult]) -> bool:
    """Check if survivors span multiple backend paradigms."""
    backends = {r.backend for r in survivors}
    return len(backends) > 1


def ranking_requires_simulation_metrics(
    survivors: list[BackendResult],
) -> tuple[bool, str]:
    """Decide whether Gate 3 ranking must use simulation-based metrics.

    Implements PRD §10 Q2 — NLPD/BIC are likelihood-scale metrics and
    are only comparable when every candidate shares the **same observation
    model**. Two conditions force a switch to simulation-based ranking:

    1. **Cross-paradigm**: survivors come from ≥ 2 backends (e.g.,
       nlmixr2 + jax_node). Each paradigm has its own observation model
       parameterization, so likelihood values aren't on the same scale.
    2. **Within-paradigm but mixed BLQ handling**: survivors share a
       backend but differ in BLQ method (M1 / M3 / M4 / M6+ / M7+). M3
       contributes a censored-likelihood term; M7+ replaces BLQ with an
       imputed zero + inflated additive error. The two likelihoods are
       on different scales even when the structural model is identical.

    Returns:
        ``(requires_simulation, reason)``. ``requires_simulation=False``
        only when all survivors share a backend and a single BLQ method.
        The ``reason`` string is human-readable and gets surfaced into
        the Gate 3 qualified-comparison annotation.
    """
    if not survivors:
        return False, "no survivors"

    backends = {r.backend for r in survivors}
    if len(backends) > 1:
        return True, f"cross-paradigm: backends {sorted(backends)} differ"

    blq_methods = {r.diagnostics.blq.method for r in survivors}
    if len(blq_methods) > 1:
        return True, (
            f"within-paradigm but BLQ handling differs across survivors "
            f"({sorted(blq_methods)}); likelihood scales are not "
            "comparable — PRD §10 Q2 routes to simulation-based metrics"
        )

    return False, "within-paradigm, uniform observation model"


def compute_vpc_concordance(result: BackendResult, *, target: float = 0.90) -> float:
    """Compute VPC coverage concordance score for a candidate.

    Score is based on how close VPC coverage is to the ideal (percentile-matching).
    Ideal: p5 coverage ~ target, p50 coverage ~ target, p95 coverage ~ target.

    Args:
        result: Backend result with VPC diagnostics.
        target: VPC concordance target (default 0.90, configurable via GatePolicy).
    """
    vpc = result.diagnostics.vpc
    if vpc is None:
        return 0.0
    deviations = []
    for _band, coverage in vpc.coverage.items():
        dev = abs(coverage - target)
        deviations.append(dev)

    if not deviations:
        return 0.0

    # Concordance: 1 - mean absolute deviation from target
    mean_dev = float(np.mean(deviations))
    return max(0.0, 1.0 - mean_dev)


def compute_npe(result: BackendResult) -> float:
    """Compute Normalized Prediction Error from GOF diagnostics.

    Uses CWRES as a proxy for NPE: ideal CWRES has mean=0, sd=1.
    NPE score = sqrt(cwres_mean^2 + (cwres_sd - 1)^2).
    Lower is better.
    """
    gof = result.diagnostics.gof
    mean_penalty = gof.cwres_mean**2
    sd_penalty = (gof.cwres_sd - 1.0) ** 2
    return float(np.sqrt(mean_penalty + sd_penalty))


def compute_composite_score(
    vpc_concordance: float,
    npe: float,
    bic: float | None,
    *,
    vpc_weight: float = 0.4,
    npe_weight: float = 0.4,
    bic_weight: float = 0.2,
) -> float:
    """Weighted composite score (lower is better).

    Components:
      - VPC concordance: inverted (1 - concordance) so lower = better
      - NPE: already lower-is-better
      - BIC: normalized to [0, 1] range within the candidate set
    """
    vpc_component = (1.0 - vpc_concordance) * vpc_weight
    npe_component = min(npe, 5.0) / 5.0 * npe_weight  # cap NPE at 5.0
    bic_component = 0.0
    if bic is not None and np.isfinite(bic):
        bic_component = min(max(bic / 1000.0, 0.0), 1.0) * bic_weight
    return vpc_component + npe_component + bic_component


def rank_cross_paradigm(
    survivors: list[BackendResult],
    *,
    vpc_concordance_target: float = 0.90,
    qualification_reason: str | None = None,
) -> CrossParadigmRankingResult:
    """Rank candidates using simulation-based metrics.

    Called when ``ranking_requires_simulation_metrics`` returns True —
    either cross-paradigm or within-paradigm with mixed BLQ handling.

    Args:
        survivors: BackendResults that passed Gates 1+2+2.5.
        vpc_concordance_target: VPC concordance target (from GatePolicy).
        qualification_reason: Optional override for the qualification
            reason; when ``None`` it is computed from the survivor set.

    Returns:
        CrossParadigmRankingResult with ranked candidates and the
        qualification reason wired through from
        ``ranking_requires_simulation_metrics``.
    """
    if not survivors:
        return CrossParadigmRankingResult(is_cross_paradigm=False)

    backends: list[str] = sorted({r.backend for r in survivors})
    cross = len(backends) > 1

    metrics: list[CrossParadigmMetrics] = []
    for result in survivors:
        vpc_conc = compute_vpc_concordance(result, target=vpc_concordance_target)
        npe = compute_npe(result)
        composite = compute_composite_score(vpc_conc, npe, result.bic)
        metrics.append(
            CrossParadigmMetrics(
                candidate_id=result.model_id,
                backend=result.backend,
                vpc_concordance=vpc_conc,
                npe=npe,
                composite_score=composite,
            )
        )

    # Sort by composite score (lower is better)
    metrics.sort(key=lambda m: m.composite_score)

    if qualification_reason is None:
        _, qualification_reason = ranking_requires_simulation_metrics(survivors)

    return CrossParadigmRankingResult(
        is_cross_paradigm=cross,
        ranked_candidates=metrics,
        backends_compared=backends,
        ranking_method="simulation_based",
        qualified_comparison=True,
        qualification_reason=qualification_reason,
    )
