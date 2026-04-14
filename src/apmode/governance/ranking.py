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
    """Full cross-paradigm ranking outcome."""

    is_cross_paradigm: bool
    ranked_candidates: list[CrossParadigmMetrics] = field(default_factory=list)
    backends_compared: list[str] = field(default_factory=list)
    ranking_method: str = "simulation_based"
    qualified_comparison: bool = True


def is_cross_paradigm(survivors: list[BackendResult]) -> bool:
    """Check if survivors span multiple backend paradigms."""
    backends = {r.backend for r in survivors}
    return len(backends) > 1


def compute_vpc_concordance(result: BackendResult) -> float:
    """Compute VPC coverage concordance score for a candidate.

    Score is based on how close VPC coverage is to the ideal (percentile-matching).
    Ideal: p5 coverage ~ 0.90, p50 coverage ~ 0.90-0.95, p95 coverage ~ 0.90.
    """
    vpc = result.diagnostics.vpc
    if vpc is None:
        return 0.0

    # Target: each percentile band should have ~90% coverage
    target = 0.90
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
) -> CrossParadigmRankingResult:
    """Rank candidates across paradigms using simulation-based metrics.

    Args:
        survivors: BackendResults that passed Gates 1+2+2.5.

    Returns:
        CrossParadigmRankingResult with ranked candidates.
    """
    if not survivors:
        return CrossParadigmRankingResult(is_cross_paradigm=False)

    backends: list[str] = sorted({r.backend for r in survivors})
    cross = len(backends) > 1

    metrics: list[CrossParadigmMetrics] = []
    for result in survivors:
        vpc_conc = compute_vpc_concordance(result)
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

    return CrossParadigmRankingResult(
        is_cross_paradigm=cross,
        ranked_candidates=metrics,
        backends_compared=backends,
        ranking_method="simulation_based" if cross else "within_paradigm_bic",
        qualified_comparison=cross,
    )
