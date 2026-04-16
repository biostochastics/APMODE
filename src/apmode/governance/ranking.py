# SPDX-License-Identifier: GPL-2.0-or-later
"""Cross-paradigm ranking via simulation-based metrics (PRD §4.3.1).

When candidates from different backends survive Gates 1+2+2.5, ranking
cannot use NLPD or BIC directly — likelihoods are on incomparable
scales across observation models (PRD §10 Q2). Gate 3 cross-paradigm
composites instead three simulation-based components:

  * VPC coverage concordance — how well each model's VPC percentile
    bands cover the observed data relative to a target concordance.
  * CWRES-based NPE proxy — ``sqrt(mean² + (sd - 1)²)`` on population
    weighted residuals. Proper simulation-based NPE lives in
    :mod:`apmode.benchmarks.scoring` and is not re-exported here to
    keep the naming honest.
  * BIC — retained as an option but *disabled by default for
    cross-paradigm* via ``Gate3Config.bic_weight = 0``. Deployments
    that want cross-paradigm likelihood ranking must opt in.

The aggregation method is policy-driven. ``Gate3Config.composite_method``
chooses between:

  * ``"weighted_sum"`` — legacy, sensitive to metric scaling; bounded by
    ``npe_cap`` and ``bic_norm_scale`` to keep components commensurable.
  * ``"borda"`` — rank each candidate on each *enabled* metric (weight
    > 0), sum the ranks (average-rank for ties), lower wins. Scale-
    invariant; preferred by multi-model consensus (gpt-5.2-pro, glm-5.1).

Within-paradigm BIC ranking (``_gate3_within_paradigm`` in
``gates.py``) is NOT governed by this module and is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult
    from apmode.governance.policy import Gate3Config

NPESource = Literal["simulation", "cwres_proxy"]


@dataclass(frozen=True)
class CrossParadigmMetrics:
    """Per-candidate metrics in a cross-paradigm comparison.

    ``composite_score`` is always interpreted as "lower is better". Under
    the weighted-sum method it is a normalized linear combination; under
    Borda it is the sum of per-metric ranks. ``component_scores``
    preserves the raw per-metric contributions (pre-aggregation) so
    reports can explain *why* a candidate ranked where it did.

    ``npe_source`` records whether the ``npe`` value came from the
    canonical simulation-based NPE emitted by the backend
    (``"simulation"``) or the CWRES distance fallback
    (``"cwres_proxy"``) — downstream reports should surface this to
    avoid overclaiming.
    """

    candidate_id: str
    backend: str
    vpc_concordance: float
    npe: float
    npe_source: NPESource
    composite_score: float
    component_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class CrossParadigmRankingResult:
    """Full simulation-based ranking outcome.

    ``qualification_reason`` captures *why* simulation metrics were used
    — cross-paradigm vs. mixed BLQ within a single paradigm (PRD §10 Q2).
    Downstream report generation quotes this string verbatim so
    reviewers see the explicit comparability justification.
    """

    is_cross_paradigm: bool
    ranked_candidates: list[CrossParadigmMetrics] = field(default_factory=list)
    backends_compared: list[str] = field(default_factory=list)
    ranking_method: str = "simulation_based"
    composite_method: str = "weighted_sum"
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


def compute_vpc_concordance(result: BackendResult, *, target: float) -> float:
    """VPC coverage concordance score for a candidate (higher = better).

    Ideal coverage: every percentile band covers ``target`` fraction of
    observations. Concordance is ``1 - mean(|coverage - target|)``
    across bands; returns 0 when no VPC is available.
    """
    vpc = result.diagnostics.vpc
    if vpc is None:
        return 0.0
    deviations = [abs(coverage - target) for coverage in vpc.coverage.values()]
    if not deviations:
        return 0.0
    return max(0.0, 1.0 - float(np.mean(deviations)))


def compute_cwres_npe_proxy(result: BackendResult) -> float:
    """Distance of CWRES distribution from the ideal ``N(0, 1)`` (lower = better).

    Fallback used ONLY when a backend does not populate
    ``result.diagnostics.npe_score`` with the canonical simulation-based
    NPE from :func:`apmode.benchmarks.scoring.compute_npe`. Computes
    ``sqrt(cwres_mean² + (cwres_sd - 1)²)``. Named with ``_proxy`` to
    prevent any consumer from mistaking it for true NPE.
    """
    gof = result.diagnostics.gof
    return float(np.sqrt(gof.cwres_mean**2 + (gof.cwres_sd - 1.0) ** 2))


def _resolve_npe(result: BackendResult) -> tuple[float, NPESource]:
    """Return ``(npe, source)`` preferring canonical simulation-based NPE.

    When the backend has populated ``result.diagnostics.npe_score`` —
    computed via :func:`apmode.benchmarks.scoring.compute_npe` on the
    same posterior-predictive simulations used for the VPC — that value
    wins. Non-negative is guaranteed by the ``DiagnosticBundle`` field
    validator, so the only failure modes here are ``None`` or NaN /
    Inf from buggy backends. Either falls back to the CWRES proxy so
    ranking is never poisoned by a non-finite score.
    """
    sim_npe = result.diagnostics.npe_score
    if sim_npe is not None and np.isfinite(sim_npe):
        return float(sim_npe), "simulation"
    return compute_cwres_npe_proxy(result), "cwres_proxy"


# ---------------------------------------------------------------------------
# Aggregation strategies (policy-driven)
# ---------------------------------------------------------------------------


def _weighted_sum_composite(
    vpc_concordance: float,
    npe: float,
    bic: float | None,
    config: Gate3Config,
) -> tuple[float, dict[str, float]]:
    """Legacy weighted-sum aggregation (lower = better).

    VPC concordance is inverted so all three components share the
    "lower is better" convention. NPE is divided by ``config.npe_cap``
    (clipped to ``[0, 1]``); BIC by ``config.bic_norm_scale`` (also
    clipped). Components with weight 0 contribute 0 regardless of raw
    value, which lets a lane disable BIC entirely. Non-finite inputs
    (NaN/Inf) are coerced to the worst possible component value so
    they cannot silently win a ranking.
    """
    components: dict[str, float] = {}
    total = 0.0

    # VPC: non-finite → treat as 0 concordance (worst).
    vpc_effective = vpc_concordance if np.isfinite(vpc_concordance) else 0.0
    vpc_component = (1.0 - vpc_effective) * config.vpc_weight
    components["vpc"] = vpc_component
    total += vpc_component

    # NPE: non-finite → treat as worst (= npe_cap, which normalizes to 1.0).
    npe_effective = npe if np.isfinite(npe) else config.npe_cap
    npe_normalized = min(max(npe_effective, 0.0), config.npe_cap) / config.npe_cap
    npe_component = npe_normalized * config.npe_weight
    components["npe"] = npe_component
    total += npe_component

    if config.bic_weight > 0.0 and bic is not None and np.isfinite(bic):
        bic_normalized = min(max(bic / config.bic_norm_scale, 0.0), 1.0)
        bic_component = bic_normalized * config.bic_weight
    else:
        bic_component = 0.0
    components["bic"] = bic_component
    total += bic_component

    return total, components


def _average_rank(values: list[float], *, lower_is_better: bool) -> list[float]:
    """Average-rank a list (ties share the mean of their ordinal positions).

    ``lower_is_better=True`` means rank 1 is the minimum value; the NPE
    and BIC components follow this convention. VPC concordance uses
    ``lower_is_better=False`` (highest concordance gets rank 1).
    """
    n = len(values)
    if n == 0:
        return []
    indexed = list(enumerate(values))
    indexed.sort(key=lambda iv: iv[1], reverse=not lower_is_better)
    ranks = [0.0] * n
    i = 0
    while i < n:
        # Find run of tied values.
        j = i + 1
        while j < n and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2.0  # mean of ordinal positions i+1..j
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def _borda_composite(
    vpc_list: list[float],
    npe_list: list[float],
    bic_list: list[float | None],
    config: Gate3Config,
) -> tuple[list[float], list[dict[str, float]]]:
    """Borda-count aggregation across all candidates (lower sum = better).

    A metric is *enabled* when its weight is strictly positive. Each
    enabled metric ranks the candidates (average-rank on ties); the
    composite is the sum of enabled ranks. Ranks are emitted as
    component_scores for audit traceability. Candidates with missing
    BIC (None / non-finite) receive the worst rank on that metric when
    BIC is enabled; if BIC is disabled the missingness is irrelevant.
    """
    n = len(vpc_list)
    if n == 0:
        return [], []

    per_metric_ranks: dict[str, list[float]] = {}

    # Coerce non-finite per-metric inputs to the worst value for their
    # ordering direction so Python's unstable NaN sort cannot land them
    # at rank 1. VPC: lower=worse → map NaN/Inf to -inf; NPE/BIC:
    # lower=better → map missing/NaN/Inf to +inf.
    if config.vpc_weight > 0.0:
        vpc_numeric = [v if np.isfinite(v) else float("-inf") for v in vpc_list]
        per_metric_ranks["vpc"] = _average_rank(vpc_numeric, lower_is_better=False)
    if config.npe_weight > 0.0:
        npe_numeric = [n if np.isfinite(n) else float("inf") for n in npe_list]
        per_metric_ranks["npe"] = _average_rank(npe_numeric, lower_is_better=True)
    if config.bic_weight > 0.0:
        # Missing/non-finite BIC ties at the worst rank; average-rank
        # semantics mean if 2 of 3 candidates are missing they share
        # rank (2+3)/2 = 2.5. Documented tie handling; strict-worst is
        # not used because it would double-penalize missingness.
        bic_numeric = [
            (b if (b is not None and np.isfinite(b)) else float("inf")) for b in bic_list
        ]
        per_metric_ranks["bic"] = _average_rank(bic_numeric, lower_is_better=True)

    if not per_metric_ranks:
        # Guarded by Gate3Config.at_least_one_metric_active, so this is
        # defensive only.
        return [0.0] * n, [{} for _ in range(n)]

    totals: list[float] = []
    components: list[dict[str, float]] = []
    for idx in range(n):
        row = {name: ranks[idx] for name, ranks in per_metric_ranks.items()}
        totals.append(sum(row.values()))
        components.append(row)
    return totals, components


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def rank_cross_paradigm(
    survivors: list[BackendResult],
    *,
    gate3: Gate3Config,
    vpc_concordance_target: float,
    qualification_reason: str | None = None,
) -> CrossParadigmRankingResult:
    """Rank candidates using simulation-based metrics (Gate 3).

    Called when :func:`ranking_requires_simulation_metrics` returns True.
    Aggregation method, metric weights, and normalization caps are
    sourced from ``gate3``; ``vpc_concordance_target`` comes from the
    lane's ``GatePolicy.vpc_concordance_target``. No defaults here —
    callers must provide an explicit policy.

    Returns a :class:`CrossParadigmRankingResult` with candidates sorted
    ascending by ``composite_score`` (lower = better for both
    aggregation methods) and ``composite_method`` reflecting the
    aggregation used.
    """
    if not survivors:
        return CrossParadigmRankingResult(
            is_cross_paradigm=False,
            composite_method=gate3.composite_method,
        )

    backends: list[str] = sorted({r.backend for r in survivors})
    cross = len(backends) > 1

    vpc_list = [compute_vpc_concordance(r, target=vpc_concordance_target) for r in survivors]
    npe_pairs = [_resolve_npe(r) for r in survivors]
    npe_list = [pair[0] for pair in npe_pairs]
    npe_sources = [pair[1] for pair in npe_pairs]
    bic_list: list[float | None] = [r.bic for r in survivors]

    scores: list[float]
    components_list: list[dict[str, float]]
    if gate3.composite_method == "borda":
        scores, components_list = _borda_composite(vpc_list, npe_list, bic_list, gate3)
    else:
        scores = []
        components_list = []
        for vpc, npe, bic in zip(vpc_list, npe_list, bic_list, strict=True):
            score, comp = _weighted_sum_composite(vpc, npe, bic, gate3)
            scores.append(score)
            components_list.append(comp)

    metrics: list[CrossParadigmMetrics] = [
        CrossParadigmMetrics(
            candidate_id=result.model_id,
            backend=result.backend,
            vpc_concordance=vpc,
            npe=npe,
            npe_source=source,
            composite_score=score,
            component_scores=components,
        )
        for result, vpc, npe, source, score, components in zip(
            survivors, vpc_list, npe_list, npe_sources, scores, components_list, strict=True
        )
    ]
    metrics.sort(key=lambda m: m.composite_score)

    if qualification_reason is None:
        _, qualification_reason = ranking_requires_simulation_metrics(survivors)

    return CrossParadigmRankingResult(
        is_cross_paradigm=cross,
        ranked_candidates=metrics,
        backends_compared=backends,
        ranking_method="simulation_based",
        composite_method=gate3.composite_method,
        qualified_comparison=True,
        qualification_reason=qualification_reason,
    )
