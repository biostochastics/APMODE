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

    ``parameter_estimates`` is optional per-parameter ``(estimate, se)``
    carried from the backend result so ``aggregate_stability`` can apply
    Rubin's rules (pooled point estimate, pooled total variance with
    between/within decomposition, degrees of freedom). ``se`` is the
    Wald/FIM-based standard error of ``estimate`` on whichever scale the
    runner emitted (typically log-scale for structural params); it may
    be None when the backend did not compute SEs (Rubin pooling then
    degrades to arithmetic mean with no within variance).
    """

    imputation_idx: int
    candidate_id: str
    converged: bool
    ofv: float | None = None
    aic: float | None = None
    bic: float | None = None
    # Per-parameter (estimate, se | None) for Rubin pooling. Dict-valued so
    # callers can include only the parameters they care about pooling;
    # unknown parameters at aggregation time are simply skipped.
    parameter_estimates: dict[str, tuple[float, float | None]] | None = None
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

        # Rubin pooling for per-parameter estimates (when the backend
        # supplied (estimate, se) tuples on each converged fit). Emits
        # a dict keyed by parameter name with the 5-tuple
        # (pooled_est, within_var, between_var, total_var, dof) — the
        # canonical Rubin (1987) decomposition.
        pooled_params = _rubin_pool_candidate(converged, m_total=m)

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
                pooled_parameters={
                    name: {
                        "pooled_estimate": t[0],
                        "within_var": t[1],
                        "between_var": t[2],
                        "total_var": t[3],
                        "dof": t[4],
                    }
                    for name, t in pooled_params.items()
                },
            )
        )

    return entries


def _pool_scalar(values: list[float | None]) -> float | None:
    """Arithmetic mean over non-None values; None if nothing to pool."""
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def rubin_pool(
    estimates: Sequence[float],
    ses: Sequence[float | None],
    *,
    m_total: int | None = None,
) -> tuple[float, float, float, float, float]:
    """Pool one parameter across imputations via Rubin's rules.

    Implements the classical Rubin (1987) pooling for a scalar parameter:

      - Qbar (pooled estimate)  = mean of per-imputation estimates
      - Ubar (within-imp var)    = mean of per-imputation variances (SE²)
      - B   (between-imp var)    = sample variance of per-imp estimates
      - T   (total var)          = Ubar + (1 + 1/m) * B
      - nu  (Barnard-Rubin dof)  = (m - 1) * (1 + Ubar / ((1 + 1/m) * B))**2

    When fewer than 2 imputations contribute, returns ``(Qbar, 0, 0, Ubar or 0,
    inf)``. When no SE was reported, ``Ubar`` is zero and the interval
    degenerates to the between-imputation-only spread — correct behavior
    for the "SE unavailable from backend" case.

    Args:
        estimates: per-imputation point estimates (skip non-converged at
            caller side; this function expects already-filtered input).
        ses: per-imputation standard errors aligned 1:1 with
            ``estimates``. ``None`` entries are treated as 0 variance
            (i.e., SE unknown → no within contribution for that draw).
        m_total: optional total number of imputations (for the standard
            ``(1 + 1/m)`` Rubin factor). When omitted, defaults to the
            number of converged estimates in ``estimates``.

    Returns:
        ``(pooled_estimate, within_var, between_var, total_var, dof)``.
    """
    if len(estimates) != len(ses):
        msg = f"estimates ({len(estimates)}) and ses ({len(ses)}) must align"
        raise ValueError(msg)
    if not estimates:
        return (0.0, 0.0, 0.0, 0.0, float("inf"))

    m = m_total if m_total is not None else len(estimates)
    qbar = float(sum(estimates) / len(estimates))

    # Within-imputation variance: mean of SE². SEs reported as None count
    # as zero so the between-imputation term can still be reported.
    variances = [(se * se) if se is not None else 0.0 for se in ses]
    ubar = float(sum(variances) / len(variances))

    if len(estimates) < 2:
        return (qbar, ubar, 0.0, ubar, float("inf"))

    between = float(sum((q - qbar) ** 2 for q in estimates) / (len(estimates) - 1))
    total = ubar + (1.0 + 1.0 / m) * between

    # Barnard-Rubin degrees of freedom; infinite when between variance is
    # negligible relative to within variance.
    if between <= 0:
        dof = float("inf")
    else:
        ratio = ubar / ((1.0 + 1.0 / m) * between)
        dof = float((len(estimates) - 1) * (1.0 + ratio) ** 2)
    return (qbar, ubar, between, total, dof)


def _rubin_pool_candidate(
    fits: list[PerImputationFit],
    *,
    m_total: int,
) -> dict[str, tuple[float, float, float, float, float]]:
    """Apply ``rubin_pool`` to every parameter reported across ``fits``.

    Input fits should be pre-filtered to converged draws for this
    candidate; the function then unions the parameter names reported in
    each fit (some imputations may omit a parameter if the model failed
    to compute its SE).

    Returns a dict mapping each parameter name to its pooled 5-tuple
    ``(pooled_estimate, within_var, between_var, total_var, dof)``. Empty
    dict when no parameter-level estimates were supplied.
    """
    by_name: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    for f in fits:
        if not f.parameter_estimates:
            continue
        for name, (est, se) in f.parameter_estimates.items():
            by_name[name].append((est, se))

    pooled: dict[str, tuple[float, float, float, float, float]] = {}
    for name, pairs in by_name.items():
        ests = [p[0] for p in pairs]
        ses = [p[1] for p in pairs]
        pooled[name] = rubin_pool(ests, ses, m_total=m_total)
    return pooled


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
