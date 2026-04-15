# SPDX-License-Identifier: GPL-2.0-or-later
"""Multiple-imputation orchestration + stability aggregation (PRD §4.2.1).

Wraps the classical search engine with an m-imputation loop. Each imputed
dataset is fit independently; the per-imputation results are aggregated into
an ``ImputationStabilityManifest`` (Rubin-pooled fit criteria plus cross-
imputation stability scores).

Responsibilities:
  - Drive a pluggable ``ImputationProvider`` to produce m imputed CSVs.
  - For each imputation, delegate fitting to an injected search callable
    (typically ``SearchEngine.run``) and collect ``PerImputationFit`` rows.
  - Aggregate via ``aggregate_stability`` into ``ImputationStabilityEntry``
    rows, then build the manifest.

Non-goals:
  - FREM fits do not use this module. FREM is a single-model path in the
    nlmixr2 backend (no imputation loop).
  - This module does not itself construct the LLM prompt; it only produces
    the stability artifact that ``agentic_runner`` then passes to the
    diagnostic summarizer when ``directive.llm_pooled_only`` is True.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from apmode.bundle.models import ImputationStabilityEntry
from apmode.data.missing_data import build_stability_manifest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from apmode.bundle.models import ImputationStabilityManifest, MissingDataDirective


@dataclass(frozen=True)
class PerImputationFit:
    """One candidate's result on one imputed dataset.

    The aggregation layer only needs lightweight scalars; upstream callers
    may retain the full ``BackendResult`` separately for bundle emission.
    """

    imputation_idx: int
    candidate_id: str
    converged: bool
    ofv: float | None = None
    aic: float | None = None
    bic: float | None = None
    # Optional sign-of-effect per covariate name for consistency scoring.
    # Example: {"WT_on_CL": +1.0, "AGE_on_V": -1.0}. Empty = not evaluated.
    covariate_effect_signs: dict[str, float] | None = None


class ImputationProvider(Protocol):
    """Produces m imputed CSVs from a source dataset.

    Implementations: ``R_MiceImputer`` (mice package, PMM default) and
    ``R_MissRangerImputer`` (ranger-backed RF + PMM). See
    ``apmode.data.imputers``.
    """

    async def impute(
        self,
        source_csv: Path,
        m: int,
        seed: int,
    ) -> list[Path]:
        """Return paths to m imputed CSV copies of ``source_csv``.

        Caller owns lifecycle of the returned files. Implementations
        should write under a caller-controlled work directory; paths must
        be absolute to satisfy the R subprocess request schema.
        """
        ...


def _rank_candidates(
    fits_for_imputation: list[PerImputationFit],
    *,
    top_k: int,
) -> set[str]:
    """Return the set of candidate_ids in the top_k by BIC for one imputation.

    Non-converged candidates and those with missing BIC are excluded from
    ranking (they cannot be in the top set). Ties are broken by AIC, then
    by candidate_id for determinism.
    """
    scored = [
        f
        for f in fits_for_imputation
        if f.converged and f.bic is not None and f.bic != float("inf")
    ]
    scored.sort(
        key=lambda f: (
            f.bic if f.bic is not None else float("inf"),
            f.aic if f.aic is not None else float("inf"),
            f.candidate_id,
        )
    )
    return {f.candidate_id for f in scored[:top_k]}


def aggregate_stability(
    fits: Sequence[PerImputationFit],
    *,
    m: int,
    top_k: int = 3,
) -> list[ImputationStabilityEntry]:
    """Aggregate per-imputation fits into stability entries.

    Pooling rules:
      - ``pooled_ofv/aic/bic``: arithmetic mean across imputations where
        the candidate converged (Rubin pooling of scalar criteria).
      - ``convergence_rate``: fraction of the m imputations where the
        candidate converged (denominator is m, not the number of fits).
      - ``within_between_var_ratio``: var(OFV across imputations)
        normalized by the pooled OFV magnitude. Low values (<1) indicate
        between-imputation variance dominates — structural instability.
      - ``rank_stability``: fraction of imputations where the candidate
        was in the top-``top_k`` by BIC.
      - ``covariate_sign_consistency``: per-covariate fraction of
        imputations agreeing on the sign of that covariate's effect,
        restricted to imputations that reported a sign for the covariate.

    Args:
        fits: Per-imputation fit records for all candidates.
        m: Total number of imputations (denominator for convergence_rate
            and rank_stability).
        top_k: Rank cutoff for rank_stability.

    Returns:
        One ``ImputationStabilityEntry`` per unique candidate_id.
    """
    # Group fits by candidate
    by_cand: dict[str, list[PerImputationFit]] = defaultdict(list)
    for f in fits:
        by_cand[f.candidate_id].append(f)

    # Compute top-K sets per imputation once
    by_imp: dict[int, list[PerImputationFit]] = defaultdict(list)
    for f in fits:
        by_imp[f.imputation_idx].append(f)
    top_sets_by_imp = {
        imp_idx: _rank_candidates(imp_fits, top_k=top_k) for imp_idx, imp_fits in by_imp.items()
    }

    entries: list[ImputationStabilityEntry] = []
    for candidate_id, candidate_fits in by_cand.items():
        converged = [f for f in candidate_fits if f.converged]
        n_converged = len(converged)
        convergence_rate = min(1.0, n_converged / max(m, 1))

        pooled_ofv = _pool_scalar([f.ofv for f in converged])
        pooled_aic = _pool_scalar([f.aic for f in converged])
        pooled_bic = _pool_scalar([f.bic for f in converged])

        within_between = _within_between_ratio(
            [f.ofv for f in converged if f.ofv is not None],
            pooled_ofv,
        )

        # Rank stability: fraction of imputations (denominator = m) where
        # this candidate was in the top_k. Missing imputations count as
        # "not in top_k" — consistent with the "insufficient evidence is
        # not a pass" convention used elsewhere in governance.
        in_top_count = sum(
            1 for imp_idx in top_sets_by_imp if candidate_id in top_sets_by_imp[imp_idx]
        )
        rank_stability = min(1.0, in_top_count / max(m, 1))

        entries.append(
            ImputationStabilityEntry(
                candidate_id=candidate_id,
                pooled_ofv=pooled_ofv,
                pooled_aic=pooled_aic,
                pooled_bic=pooled_bic,
                convergence_rate=convergence_rate,
                within_between_var_ratio=within_between,
                rank_stability=rank_stability,
                covariate_sign_consistency=_sign_consistency(candidate_fits),
            )
        )

    return entries


def _pool_scalar(values: list[float | None]) -> float | None:
    """Arithmetic mean over non-None values; None if nothing to pool."""
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _within_between_ratio(
    values: list[float],
    pooled: float | None,
) -> float | None:
    """Crude within/between variance proxy for scalar criteria.

    Proper Rubin decomposition requires within-imputation standard errors,
    which nlmixr2 does not emit for OFV itself. We approximate with the
    across-imputation sample variance normalized by |pooled|; downstream
    consumers should treat this as an *indicator*, not a formal Rubin
    statistic. Returns None when n<2 or pooled magnitude is too small to
    normalize meaningfully.
    """
    if len(values) < 2 or pooled is None or abs(pooled) < 1e-10:
        return None
    mean = sum(values) / len(values)
    between_var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return float(between_var / abs(pooled))


def _sign_consistency(
    fits: list[PerImputationFit],
) -> dict[str, float]:
    """Per-covariate fraction of fits agreeing on the sign of the effect.

    For each covariate name seen in any fit, collect the signs across fits
    that reported it and compute (max(pos_count, neg_count) / total_count).
    Covariates with no reported signs are omitted.
    """
    by_cov: dict[str, list[float]] = defaultdict(list)
    for f in fits:
        if not f.covariate_effect_signs:
            continue
        for name, sign in f.covariate_effect_signs.items():
            by_cov[name].append(sign)

    consistency: dict[str, float] = {}
    for name, signs in by_cov.items():
        if not signs:
            continue
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        # Zero signs count as neither; denominator stays at reported count.
        consistency[name] = max(pos, neg) / len(signs)
    return consistency


async def run_with_imputations(
    directive: MissingDataDirective,
    provider: ImputationProvider,
    source_csv: Path,
    search: Callable[[Path, int], Awaitable[list[PerImputationFit]]],
    seed: int,
    *,
    top_k: int = 3,
) -> ImputationStabilityManifest:
    """Drive m imputations, fit each, and return the stability manifest.

    Precondition: ``directive.covariate_method`` is an MI-* method and
    ``directive.m_imputations`` is set. FREM/FFEM/exclude paths do not
    use this function.

    Args:
        directive: Resolved missing-data directive.
        provider: Imputation provider (mice, missForest, or other).
        source_csv: Source CSV with missingness. Absolute path required.
        search: Callable that fits all candidates on one imputed CSV and
            returns ``PerImputationFit`` rows for that imputation.
        seed: Base seed; per-imputation seeds are derived deterministically
            as ``seed + imputation_idx``.
        top_k: Rank cutoff for rank_stability.

    Returns:
        ImputationStabilityManifest ready for bundle emission and LLM
        consumption.

    Raises:
        ValueError: If the directive is not an MI-* method or m is unset.
    """
    if not directive.covariate_method.startswith("MI-"):
        msg = (
            f"run_with_imputations requires an MI-* directive, got {directive.covariate_method!r}"
        )
        raise ValueError(msg)
    if directive.m_imputations is None:
        msg = "directive.m_imputations must be set for MI runs"
        raise ValueError(msg)

    m = directive.m_imputations
    imputed_paths = await provider.impute(source_csv, m=m, seed=seed)
    if len(imputed_paths) != m:
        msg = f"Imputation provider returned {len(imputed_paths)} datasets, expected {m}"
        raise RuntimeError(msg)

    all_fits: list[PerImputationFit] = []
    for idx, path in enumerate(imputed_paths):
        per_imp_fits = await search(path, seed + idx)
        # Tag fits with their imputation index; the search callable is not
        # required to know it.
        all_fits.extend(
            PerImputationFit(
                imputation_idx=idx,
                candidate_id=f.candidate_id,
                converged=f.converged,
                ofv=f.ofv,
                aic=f.aic,
                bic=f.bic,
                covariate_effect_signs=f.covariate_effect_signs,
            )
            for f in per_imp_fits
        )

    entries = aggregate_stability(all_fits, m=m, top_k=top_k)
    return build_stability_manifest(directive, entries, top_k=top_k)
