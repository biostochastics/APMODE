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
    invariant across paradigms — the preferred cross-paradigm aggregator
    because likelihood-scale metrics can't be compared directly (§10 Q2).

Within-paradigm BIC ranking (``_gate3_within_paradigm`` in
``gates.py``) is NOT governed by this module and is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult
    from apmode.governance.policy import Gate3Config

NPESource = Literal["simulation", "cwres_proxy"]
AUCCmaxSource = Literal["observed_trapezoid"]


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

    ``auc_cmax_be`` is the per-subject AUC/Cmax bioequivalence rate (see
    :func:`apmode.benchmarks.scoring.compute_auc_cmax_be_score`); it is
    ``None`` when the NCA reference is ineligible for one or more
    candidates and the uniform-drop rule suppressed the metric for the
    ranking. ``auc_cmax_source`` is ``"observed_trapezoid"`` when populated.
    """

    candidate_id: str
    backend: str
    vpc_concordance: float
    npe: float
    npe_source: NPESource
    composite_score: float
    component_scores: dict[str, float] = field(default_factory=dict)
    auc_cmax_be: float | None = None
    auc_cmax_source: AUCCmaxSource | None = None


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
    *,
    gate3: Gate3Config | None = None,
) -> tuple[bool, str]:
    """Decide whether Gate 3 ranking must use simulation-based metrics.

    Implements PRD §10 Q2 — NLPD/BIC are likelihood-scale metrics and
    are only comparable when every candidate shares the **same observation
    model**. Three conditions force a switch to simulation-based ranking:

    1. **Cross-paradigm**: survivors come from ≥ 2 backends (e.g.,
       nlmixr2 + jax_node). Each paradigm has its own observation model
       parameterization, so likelihood values aren't on the same scale.
    2. **Within-paradigm but mixed BLQ handling**: survivors share a
       backend but differ in BLQ method (M1 / M3 / M4 / M6+ / M7+). M3
       contributes a censored-likelihood term; M7+ replaces BLQ with an
       imputed zero + inflated additive error. The two likelihoods are
       on different scales even when the structural model is identical.
    3. **Policy opts into a simulation-only metric**: when ``gate3`` is
       provided and ``gate3.auc_cmax_weight > 0`` the AUC/Cmax BE score
       (fraction-within-BE-goalposts) must be computed from posterior
       predictive sims, so the simulation-based path is required even
       when backends and BLQ handling are uniform.

    Returns:
        ``(requires_simulation, reason)``. ``requires_simulation=False``
        only when survivors share a backend, a single BLQ method, and
        the policy does not enable a simulation-only metric. The
        ``reason`` string is human-readable and gets surfaced into the
        Gate 3 qualified-comparison annotation.
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

    if gate3 is not None and gate3.auc_cmax_weight > 0.0:
        return True, (
            "policy enables AUC/Cmax BE component "
            f"(auc_cmax_weight={gate3.auc_cmax_weight:.3f}); "
            "AUC/Cmax BE is a simulation-based metric per PRD §4.3.1"
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


def _resolve_auc_cmax(result: BackendResult) -> tuple[float | None, AUCCmaxSource | None]:
    """Return ``(auc_cmax_be, source)`` for a candidate.

    Unlike :func:`_resolve_npe` there is no proxy fallback: using a
    candidate-derived reference (e.g. median candidate GMR) would be
    circular and bias Gate 3 ranking toward cohort center. When the
    backend omits the score the ranking layer's uniform-drop rule
    removes the component from the composite for *all* candidates
    rather than synthesizing a value here.
    """
    score = result.diagnostics.auc_cmax_be_score
    if score is None or not np.isfinite(score):
        return None, None
    return float(score), result.diagnostics.auc_cmax_source


# ---------------------------------------------------------------------------
# Aggregation strategies (policy-driven)
# ---------------------------------------------------------------------------


def _weighted_sum_composite(
    vpc_concordance: float,
    npe: float,
    bic: float | None,
    auc_cmax_be: float | None,
    config: Gate3Config,
) -> tuple[float, dict[str, float]]:
    """Legacy weighted-sum aggregation (lower = better).

    VPC concordance and AUC/Cmax BE score are both inverted so every
    component shares the "lower is better" convention. NPE is divided
    by ``config.npe_cap`` (clipped to ``[0, 1]``); BIC by
    ``config.bic_norm_scale`` (also clipped). Components with weight 0
    contribute 0 regardless of raw value, which lets a lane disable BIC
    or AUC/Cmax entirely. Non-finite inputs (NaN/Inf) are coerced to
    the worst possible component value so they cannot silently win a
    ranking. AUC/Cmax missingness for individual candidates is handled
    upstream by the uniform-drop rule in :func:`rank_cross_paradigm` —
    by the time this function sees ``auc_cmax_weight > 0``, every
    candidate is guaranteed to have a finite score.
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

    if config.auc_cmax_weight > 0.0 and auc_cmax_be is not None and np.isfinite(auc_cmax_be):
        # Clip to [0, 1] defensively; the Pydantic field validator already
        # enforces the bound but an upstream bug could slip through.
        be_clipped = min(max(float(auc_cmax_be), 0.0), 1.0)
        auc_cmax_component = (1.0 - be_clipped) * config.auc_cmax_weight
    else:
        auc_cmax_component = 0.0
    components["auc_cmax"] = auc_cmax_component
    total += auc_cmax_component

    return total, components


def _average_rank(values: list[float], *, lower_is_better: bool) -> list[float]:
    """Average-rank a list (ties share the mean of their ordinal positions).

    ``lower_is_better=True`` means rank 1 is the minimum value; the NPE
    and BIC components follow this convention. VPC concordance uses
    ``lower_is_better=False`` (highest concordance gets rank 1).

    Tie detection uses :func:`math.isclose` (``rel_tol=1e-9, abs_tol=1e-12``)
    rather than exact float equality: metric values here flow through
    arithmetic (e.g. `1 - mean(deviations)` for VPC concordance), so
    candidates that are tied in exact real arithmetic can differ by one
    ULP after rounding. Exact ``==`` would silently break ties in favor
    of whichever order Python's sort happened to produce.
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
        while j < n and math.isclose(indexed[j][1], indexed[i][1], rel_tol=1e-9, abs_tol=1e-12):
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
    auc_cmax_list: list[float | None],
    config: Gate3Config,
) -> tuple[list[float], list[dict[str, float]]]:
    """Borda-count aggregation across all candidates (lower sum = better).

    A metric is *enabled* when its weight is strictly positive. Each
    enabled metric ranks the candidates (average-rank on ties); the
    composite is the sum of enabled ranks. Ranks are emitted as
    component_scores for audit traceability. Candidates with missing
    BIC (None / non-finite) receive the worst rank on that metric when
    BIC is enabled; if BIC is disabled the missingness is irrelevant.
    AUC/Cmax missingness is handled upstream by the uniform-drop rule
    in :func:`rank_cross_paradigm` — by the time this function sees
    ``auc_cmax_weight > 0`` every candidate has a finite score.
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
    if config.auc_cmax_weight > 0.0:
        # AUC/Cmax BE higher is better; None / non-finite → -inf (worst).
        # Uniform-drop upstream guarantees all entries are not-None here,
        # so the non-finite branch is defensive-only.
        auc_numeric = [
            (a if (a is not None and np.isfinite(a)) else float("-inf")) for a in auc_cmax_list
        ]
        per_metric_ranks["auc_cmax"] = _average_rank(auc_numeric, lower_is_better=False)

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


def _apply_uniform_auc_cmax_drop(
    gate3: Gate3Config,
    auc_cmax_list: list[float | None],
) -> tuple[Gate3Config, list[float | None], str | None]:
    """Return (effective_gate3, effective_auc_cmax_list, drop_reason).

    Gate 3 ranking requires every candidate to be scored on the *same*
    component set — letting a data-poor candidate "dodge" a metric its
    siblings must satisfy defeats the whole point of a cross-paradigm
    comparability gate (PRD §4.3.1). So when any candidate lacks an
    AUC/Cmax BE score (NCA ineligible, or backend omitted it), the
    component is dropped for *all* candidates and the remaining weights
    are proportionally renormalized. Per-candidate renormalization is
    rejected because it produces apples-to-oranges composite scores.

    A no-op path is taken when (a) ``auc_cmax_weight == 0`` (nothing to
    drop) or (b) every candidate has a finite score.
    """
    if gate3.auc_cmax_weight <= 0.0:
        return gate3, auc_cmax_list, None
    if all(s is not None and np.isfinite(s) for s in auc_cmax_list):
        return gate3, auc_cmax_list, None

    remaining = gate3.vpc_weight + gate3.npe_weight + gate3.bic_weight
    if remaining <= 0.0:
        # Pathological policy: auc_cmax_weight was the only positive
        # component and it's now dropped. Fall back to equal VPC+NPE so
        # ranking still produces a defensible ordering instead of raising.
        effective = gate3.model_copy(
            update={
                "vpc_weight": 0.5,
                "npe_weight": 0.5,
                "bic_weight": 0.0,
                "auc_cmax_weight": 0.0,
            }
        )
        reason = (
            "auc_cmax_be unavailable for at least one survivor and "
            "auc_cmax was the only enabled component; policy is "
            "under-specified, falling back to vpc=0.5/npe=0.5"
        )
    else:
        scale = 1.0 / remaining
        effective = gate3.model_copy(
            update={
                "vpc_weight": gate3.vpc_weight * scale,
                "npe_weight": gate3.npe_weight * scale,
                "bic_weight": gate3.bic_weight * scale,
                "auc_cmax_weight": 0.0,
            }
        )
        reason = (
            "auc_cmax_be unavailable for at least one survivor; "
            f"component dropped uniformly, remaining weights renormalized "
            f"(vpc={effective.vpc_weight:.3f}, npe={effective.npe_weight:.3f}, "
            f"bic={effective.bic_weight:.3f})"
        )
    dropped_list: list[float | None] = [None] * len(auc_cmax_list)
    return effective, dropped_list, reason


def _apply_uniform_bic_drop(
    gate3: Gate3Config,
    bic_list: list[float | None],
) -> tuple[Gate3Config, list[float | None], str | None]:
    """Mirror of ``_apply_uniform_auc_cmax_drop`` for the BIC component.

    When ``bic_weight > 0`` but some candidates lack a finite BIC, the
    composite produces apples-to-oranges totals (candidates without BIC
    contribute 0.0 while others contribute ``bic_normalized * weight``).
    Drop BIC uniformly and renormalize remaining weights, same pattern
    as the AUC/Cmax drop. Same pathological fallback when BIC was the
    only positive component.
    """
    if gate3.bic_weight <= 0.0:
        return gate3, bic_list, None
    if all(b is not None and np.isfinite(b) for b in bic_list):
        return gate3, bic_list, None

    remaining = gate3.vpc_weight + gate3.npe_weight + gate3.auc_cmax_weight
    if remaining <= 0.0:
        effective = gate3.model_copy(
            update={
                "vpc_weight": 0.5,
                "npe_weight": 0.5,
                "bic_weight": 0.0,
                "auc_cmax_weight": 0.0,
            }
        )
        reason = (
            "bic unavailable for at least one survivor and bic was the "
            "only enabled component; policy is under-specified, falling "
            "back to vpc=0.5/npe=0.5"
        )
    else:
        scale = 1.0 / remaining
        effective = gate3.model_copy(
            update={
                "vpc_weight": gate3.vpc_weight * scale,
                "npe_weight": gate3.npe_weight * scale,
                "bic_weight": 0.0,
                "auc_cmax_weight": gate3.auc_cmax_weight * scale,
            }
        )
        reason = (
            "bic unavailable for at least one survivor; component dropped "
            f"uniformly, remaining weights renormalized "
            f"(vpc={effective.vpc_weight:.3f}, npe={effective.npe_weight:.3f}, "
            f"auc_cmax={effective.auc_cmax_weight:.3f})"
        )
    dropped_list: list[float | None] = [None] * len(bic_list)
    return effective, dropped_list, reason


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
    auc_cmax_pairs = [_resolve_auc_cmax(r) for r in survivors]
    auc_cmax_values: list[float | None] = [pair[0] for pair in auc_cmax_pairs]
    auc_cmax_sources: list[AUCCmaxSource | None] = [pair[1] for pair in auc_cmax_pairs]

    effective_gate3, effective_auc_cmax, drop_reason = _apply_uniform_auc_cmax_drop(
        gate3, auc_cmax_values
    )
    # Apply the uniform BIC drop on the already-adjusted gate3 so the
    # two renormalizations compose correctly when both components have
    # at least one missing candidate.
    effective_gate3, effective_bic, bic_drop_reason = _apply_uniform_bic_drop(
        effective_gate3, bic_list
    )
    if bic_drop_reason:
        drop_reason = f"{drop_reason}; {bic_drop_reason}" if drop_reason else bic_drop_reason

    scores: list[float]
    components_list: list[dict[str, float]]
    if effective_gate3.composite_method == "borda":
        scores, components_list = _borda_composite(
            vpc_list, npe_list, effective_bic, effective_auc_cmax, effective_gate3
        )
    else:
        scores = []
        components_list = []
        for vpc, npe, bic, auc_cmax in zip(
            vpc_list, npe_list, effective_bic, effective_auc_cmax, strict=True
        ):
            score, comp = _weighted_sum_composite(vpc, npe, bic, auc_cmax, effective_gate3)
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
            # Report the *original* per-candidate auc_cmax value + source
            # for audit traceability, even when the component was uniformly
            # dropped from the composite. Callers inspecting component_scores
            # will see ``auc_cmax`` absent under Borda or zero under weighted
            # sum — that's the authoritative signal for whether it factored
            # into the rank.
            auc_cmax_be=auc_cmax_value,
            auc_cmax_source=auc_cmax_source,
        )
        for (
            result,
            vpc,
            npe,
            source,
            score,
            components,
            auc_cmax_value,
            auc_cmax_source,
        ) in zip(
            survivors,
            vpc_list,
            npe_list,
            npe_sources,
            scores,
            components_list,
            auc_cmax_values,
            auc_cmax_sources,
            strict=True,
        )
    ]
    metrics.sort(key=lambda m: m.composite_score)

    if qualification_reason is None:
        _, qualification_reason = ranking_requires_simulation_metrics(survivors, gate3=gate3)
    if drop_reason is not None:
        qualification_reason = (
            f"{qualification_reason}; {drop_reason}" if qualification_reason else drop_reason
        )

    return CrossParadigmRankingResult(
        is_cross_paradigm=cross,
        ranked_candidates=metrics,
        backends_compared=backends,
        ranking_method="simulation_based",
        composite_method=gate3.composite_method,
        qualified_comparison=True,
        qualification_reason=qualification_reason,
    )


# --- v0.5.0 M0: ScoringContract-grouped ranking (plan §3) ---


@dataclass
class ContractGroupedRanking:
    """Gate-3 ranking split by :class:`~apmode.bundle.models.ScoringContract`.

    Plan §3 mandates hard-separate leaderboards whenever survivors carry
    different contracts (conditional vs marginal NLPD, integrated vs
    pooled RE, HMC-NUTS vs FOCEI integrator, float32 vs float64 accumulation).
    This type is what Gate 3 produces in the contract-split world.

    ``groups`` lists one ranking per distinct contract, preserving the
    insertion order of ``group_by_scoring_contract`` (which in turn is
    the order survivors arrive from upstream — typically a stable sort
    over backend, then model_id).

    ``recommended_candidate_id`` is populated only in the Submission
    lane when the dominance rule (``re_treatment='integrated'`` AND
    ``nlpd_kind='marginal'``) selects a unique top candidate from the
    dominating group. Discovery and Optimization lanes leave it None
    by design — separate leaderboards are the intended output.

    ``recommended_warning`` is a non-empty string when the lane expected
    a recommended candidate but none was eligible. Reports surface this
    verbatim so reviewers see the disclosure.
    """

    groups: list[CrossParadigmRankingResult]
    contracts: list[object]  # typed ScoringContract; avoids forward-ref gymnastics
    recommended_candidate_id: str | None = None
    recommended_contract_index: int | None = None
    recommended_warning: str | None = None


def group_by_scoring_contract(
    survivors: list[BackendResult],
) -> list[tuple[object, list[BackendResult]]]:
    """Partition survivors into same-contract buckets, preserving order.

    Exact equality on :class:`ScoringContract` — the model is frozen so
    Python's value-equality is well-defined. Returns a list of
    ``(contract, [results])`` tuples in the order the contracts first
    appear in ``survivors``.
    """
    buckets: list[tuple[object, list[BackendResult]]] = []
    for result in survivors:
        contract = result.diagnostics.scoring_contract
        for bucket_contract, bucket_results in buckets:
            if bucket_contract == contract:
                bucket_results.append(result)
                break
        else:
            buckets.append((contract, [result]))
    return buckets


def _apply_submission_dominance_rule(
    grouped: list[tuple[object, CrossParadigmRankingResult]],
) -> tuple[str | None, int | None, str | None]:
    """Return (recommended_candidate_id, contract_index, warning).

    Per plan §3 (Submission-lane dominance rule), only candidates whose
    contract has ``re_treatment='integrated'`` AND ``nlpd_kind='marginal'``
    are eligible for ``recommended``. When no group qualifies, emit a
    warning so reports disclose the gap explicitly — never silently fall
    back to a non-eligible candidate.

    Tiebreaker: when *multiple* eligible groups exist (e.g. nlmixr2-FOCEI
    *and* Stan-HMC-NUTS both qualify), the winner is the group whose top
    candidate has the **lowest per-group composite_score** — this is the
    within-contract ranking metric and is already computed. Insertion-order
    is not a principled choice and would tie the regulatory recommendation
    to scheduling.
    A warning still fires when multiple groups are eligible so reviewers
    see that a cross-contract judgement call was made.
    """
    eligible: list[tuple[int, object, CrossParadigmRankingResult]] = []
    for idx, (contract, ranking) in enumerate(grouped):
        if (
            getattr(contract, "re_treatment", None) == "integrated"
            and getattr(contract, "nlpd_kind", None) == "marginal"
            and ranking.ranked_candidates
        ):
            eligible.append((idx, contract, ranking))

    if not eligible:
        return (
            None,
            None,
            "No integrated+marginal contract group; no candidate is eligible as "
            "'recommended' in the Submission lane. See separate leaderboards.",
        )

    # Deterministic tiebreak: composite_score ascending (lower is better
    # for both weighted-sum and Borda aggregations — see plan §3).
    # Secondary key is the top candidate_id (lexicographic) — this is
    # stable across runs *regardless* of survivor insertion order, which
    # the previous insertion-index approach was not. The nlpd_integrator
    # is the tertiary key so that identical candidate ids across
    # contracts still resolve deterministically.
    eligible.sort(
        key=lambda e: (
            e[2].ranked_candidates[0].composite_score,
            e[2].ranked_candidates[0].candidate_id,
            str(getattr(e[1], "nlpd_integrator", "")),
        )
    )
    best_idx, best_contract, best_ranking = eligible[0]
    warning: str | None = None
    if len(eligible) > 1:
        others = [
            f"{getattr(c, 'nlpd_integrator', 'unknown')}"
            f"(top_composite={r.ranked_candidates[0].composite_score:.4f})"
            for _, c, r in eligible[1:]
        ]
        warning = (
            f"Multiple integrated+marginal contract groups were eligible for "
            f"the Submission 'recommended' slot — chose "
            f"{getattr(best_contract, 'nlpd_integrator', 'unknown')} by lowest "
            f"per-group composite_score; other eligible groups: {', '.join(others)}. "
            f"Review the separate leaderboards before accepting the recommendation."
        )
    return (best_ranking.ranked_candidates[0].candidate_id, best_idx, warning)


def rank_by_scoring_contract(
    survivors: list[BackendResult],
    *,
    gate3: Gate3Config,
    vpc_concordance_target: float,
    lane: str = "discovery",
    qualification_reason: str | None = None,
) -> ContractGroupedRanking:
    """Group survivors by ScoringContract and rank within each group.

    Cross-contract composites are never produced (plan §3). The Submission
    lane additionally receives a ``recommended_candidate_id`` from the
    integrated+marginal group; other lanes get None by design.

    Per-group ranking reuses :func:`rank_cross_paradigm` — within a single
    contract class the survivors are commensurable, so the existing
    simulation-based composite (or within-paradigm BIC if upstream routes
    there) applies.
    """
    buckets = group_by_scoring_contract(survivors)
    rankings: list[CrossParadigmRankingResult] = [
        rank_cross_paradigm(
            results,
            gate3=gate3,
            vpc_concordance_target=vpc_concordance_target,
            qualification_reason=qualification_reason,
        )
        for _, results in buckets
    ]
    recommended_id: str | None = None
    recommended_idx: int | None = None
    warning: str | None = None
    if lane == "submission":
        grouped_for_rule: list[tuple[object, CrossParadigmRankingResult]] = list(
            zip([c for c, _ in buckets], rankings, strict=True)
        )
        recommended_id, recommended_idx, warning = _apply_submission_dominance_rule(
            grouped_for_rule
        )
    return ContractGroupedRanking(
        groups=rankings,
        contracts=[c for c, _ in buckets],
        recommended_candidate_id=recommended_id,
        recommended_contract_index=recommended_idx,
        recommended_warning=warning,
    )
