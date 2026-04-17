# SPDX-License-Identifier: GPL-2.0-or-later
"""Policy-driven resolver for missing-data handling (PRD §4.2.1).

Converts a pair of ``(MissingDataPolicy, EvidenceManifest)`` into a binding
``MissingDataDirective`` that downstream backends must honor. Also provides
a builder for ``ImputationStabilityManifest`` from per-imputation results.

Decision logic:

  1. BLQ method selection by BLQ% threshold (policy.blq_m3_threshold)
     - Below threshold → M7+ (impute 0 + inflated additive error; Wijk 2025)
     - Above threshold or policy.blq_force_m3 → M3 (likelihood-based; Beal 2001)

  2. Covariate method selection
     - No missingness → "exclude" (no directive needed)
     - Time-varying + frem_for_time_varying → FREM (Nyberg 2024)
     - %-missing > frem_preferred_above → FREM
     - %-missing in (mi_pmm_max_missingness, frem_preferred_above] → FREM
     - Else → MI-PMM (default), escalate to MI-missRanger if covariate_correlated
       and missforest_fallback is set (nonlinear relations; Bräm CPT:PSP 2022
       concept, implemented with ranger-backed missRanger v2.6+ for speed)

  3. m budget: policy.m_imputations; adaptive_m carries through to the
     stability manifest builder which decides escalation up to m_max.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmode.bundle.models import (
    ImputationStabilityEntry,
    ImputationStabilityManifest,
    MissingDataDirective,
)

if TYPE_CHECKING:
    from typing import Literal

    from apmode.bundle.models import CovariateStrategy, EvidenceManifest
    from apmode.governance.policy import MissingDataPolicy

# Canonical mapping from the policy-side uppercase BLQ nomenclature
# (``MissingDataDirective.blq_method`` — "M1" / "M3" / "M4" / "M6+" / "M7+")
# to the bundle-side lowercase enum (``BLQHandling.method``). The bundle form
# uses underscores so it stays a valid Python ``Literal`` member and is
# directly comparable via equality; the policy form preserves the Beal (2001)
# notation for audit readability. Extending one side without the other
# silently breaks Gate 3 simulation-metric dispatch (see
# :func:`apmode.governance.ranking.ranking_requires_simulation_metrics`), so
# both are defined in a single map here.
_BLQ_POLICY_TO_BUNDLE: dict[str, Literal["none", "m1", "m3", "m4", "m6_plus", "m7_plus"]] = {
    "M1": "m1",
    "M3": "m3",
    "M4": "m4",
    "M6+": "m6_plus",
    "M7+": "m7_plus",
}


def normalize_blq_method_for_bundle(
    policy_method: str,
) -> Literal["none", "m1", "m3", "m4", "m6_plus", "m7_plus"]:
    """Map a policy-side BLQ method string to its bundle enum value.

    ``policy_method`` is the string in ``MissingDataDirective.blq_method``
    (``"M1"``/``"M3"``/``"M4"``/``"M6+"``/``"M7+"``). Returns the matching
    ``BLQHandling.method`` literal. Raises ``ValueError`` on unknown inputs
    rather than coercing to a default — coercion is what let M7+ be silently
    dropped before this helper existed.
    """
    try:
        return _BLQ_POLICY_TO_BUNDLE[policy_method]
    except KeyError as exc:
        allowed = sorted(_BLQ_POLICY_TO_BUNDLE.keys())
        msg = f"Unknown BLQ method {policy_method!r}; expected one of {allowed}"
        raise ValueError(msg) from exc


def resolve_directive(
    policy: MissingDataPolicy,
    manifest: EvidenceManifest,
) -> MissingDataDirective:
    """Resolve a policy + evidence manifest into a binding directive.

    Args:
        policy: Lane-specific missing-data policy.
        manifest: Evidence manifest from data profiling.

    Returns:
        MissingDataDirective with rationale for audit trail.
    """
    rationale: list[str] = []

    covariate_method: CovariateStrategy
    frac = (
        manifest.covariate_missingness.fraction_incomplete
        if manifest.covariate_missingness is not None
        else 0.0
    )

    if manifest.covariate_missingness is None or frac == 0.0:
        covariate_method = "exclude"
        rationale.append("No covariate missingness detected")
        m_imputations: int | None = None
    elif manifest.time_varying_covariates and policy.frem_for_time_varying:
        covariate_method = "FREM"
        rationale.append(
            "Time-varying covariates: FREM preferred (Nyberg 2024) — "
            "avoids per-occasion imputation and Ω pooling."
        )
        m_imputations = None
    elif frac > policy.frem_preferred_above:
        covariate_method = "FREM"
        rationale.append(
            f"Covariate missingness {frac:.2%} > frem_preferred_above "
            f"{policy.frem_preferred_above:.2%}: FREM avoids MI pooling of Ω."
        )
        m_imputations = None
    elif frac > policy.mi_pmm_max_missingness:
        # In the gap between thresholds, prefer FREM over large-m MI.
        covariate_method = "FREM"
        rationale.append(
            f"Covariate missingness {frac:.2%} exceeds MI-PMM ceiling "
            f"{policy.mi_pmm_max_missingness:.2%}: FREM preferred."
        )
        m_imputations = None
    else:
        # MI path: PMM default, missRanger (ranger-backed RF) fallback when
        # covariates are correlated — nonlinear relations benefit from RF
        # imputation (Bräm 2022 concept; missRanger 2.6.x is the modern
        # fast implementation, Mayer CRAN).
        if policy.missforest_fallback and manifest.covariate_correlated:
            covariate_method = "MI-missRanger"
            rationale.append(
                "Correlated covariates + missforest_fallback=True: "
                "missRanger captures nonlinear relationships (ranger-backed RF + PMM)."
            )
        else:
            covariate_method = "MI-PMM"
            rationale.append(
                f"Covariate missingness {frac:.2%} within MI-PMM ceiling "
                f"{policy.mi_pmm_max_missingness:.2%}."
            )
        m_imputations = policy.m_imputations

    # BLQ method selection
    blq_method: str
    if policy.blq_force_m3:
        blq_method = "M3"
        rationale.append("BLQ: M3 forced by policy (blq_force_m3=True).")
    elif manifest.blq_burden > policy.blq_m3_threshold:
        blq_method = "M3"
        rationale.append(
            f"BLQ burden {manifest.blq_burden:.2%} > blq_m3_threshold "
            f"{policy.blq_m3_threshold:.2%}: M3 likelihood (Beal 2001)."
        )
    else:
        blq_method = "M7+"
        rationale.append(
            f"BLQ burden {manifest.blq_burden:.2%} ≤ blq_m3_threshold "
            f"{policy.blq_m3_threshold:.2%}: M7+ (impute 0 + inflated "
            f"additive error; Wijk 2025)."
        )

    # Adaptive-m escalation is declared on MissingDataPolicy but is NOT
    # yet implemented downstream — search/stability.run_with_imputations
    # loops exactly m_imputations times. Refuse policies that set
    # adaptive_m=True so the phantom feature does not silently degrade
    # to fixed m without the caller realising. See findings list:
    # "adaptive_m / m_max dead config chain".
    if policy.adaptive_m and m_imputations is not None:
        msg = (
            "MissingDataPolicy.adaptive_m=True is not yet implemented; "
            "search/stability.run_with_imputations loops exactly "
            "m_imputations times with no variance-threshold escalation. "
            "Set adaptive_m=False (the default) or implement adaptive "
            "escalation before relying on it."
        )
        raise NotImplementedError(msg)

    return MissingDataDirective(
        covariate_method=covariate_method,
        m_imputations=m_imputations,
        adaptive_m=policy.adaptive_m if m_imputations is not None else False,
        m_max=policy.m_max if m_imputations is not None else None,
        blq_method=blq_method,
        llm_pooled_only=policy.llm_pooled_only,
        imputation_stability_penalty=policy.imputation_stability_penalty,
        rationale=rationale,
    )


# Canonical Ω-pooling caveats documented for Gate 2.5 credibility ingestion.
# Surfaced by ``build_stability_manifest`` whenever MI produces the bundle.
OMEGA_POOLING_CAVEATS: tuple[str, ...] = (
    "Rubin's rules apply cleanly to fixed-effect estimates; pooling of random-"
    "effect covariance matrices (Ω) is an approximation that ignores "
    "correlation-structure uncertainty and may break positive-definiteness "
    "under arithmetic averaging.",
    "When pooling is required, use log-Cholesky averaging of Ω or — preferred — "
    "pool derived predictions/exposures via simulation across imputations "
    "rather than pooling Ω directly.",
    "EBEs/ETAs are not directly poolable across imputations; recompute per "
    "imputation and then pool derived quantities.",
)


def build_stability_manifest(
    directive: MissingDataDirective,
    entries: list[ImputationStabilityEntry],
    *,
    top_k: int = 3,
) -> ImputationStabilityManifest:
    """Build the stability manifest from per-imputation candidate results.

    The agentic backend sees this artifact INSTEAD OF raw per-imputation
    diagnostics: the covariate representation is frozen before the LLM
    proposes transforms, so cherry-picking across imputations is
    structurally prevented.

    Args:
        directive: Resolved directive (provides m, method).
        entries: One entry per candidate summarizing cross-imputation stability.
        top_k: Rank cutoff for ``rank_stability`` in the entries.

    Returns:
        ImputationStabilityManifest ready to serialize.
    """
    m = directive.m_imputations if directive.m_imputations is not None else 1
    caveats = list(OMEGA_POOLING_CAVEATS) if directive.covariate_method.startswith("MI-") else []
    return ImputationStabilityManifest(
        m=m,
        method=directive.covariate_method,
        top_k=top_k,
        entries=entries,
        omega_pooling_caveats=caveats,
    )
