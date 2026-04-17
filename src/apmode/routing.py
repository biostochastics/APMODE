# SPDX-License-Identifier: GPL-2.0-or-later
"""Lane Router: routes runs to operating lanes based on intent and evidence
manifest constraints (PRD §3, ARCHITECTURE.md §3).

Three operating lanes with different admissible backends and stopping rules:
  - Submission: classical NLME only. NODE/agentic not eligible.
  - Discovery: all backends including NODE. Broader tolerances.
  - Optimization: all backends. LORO-CV required.

The router enforces dispatch constraints from the EvidenceManifest:
  - richness=sparse + absorption_coverage=inadequate → NODE not dispatched
  - data_insufficient flag for NODE when data quality is inadequate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from apmode.data.missing_data import resolve_directive

if TYPE_CHECKING:
    from apmode.bundle.models import EvidenceManifest, MissingDataDirective
    from apmode.governance.policy import MissingDataPolicy

# M1: Lane was previously duplicated (protocol.py StrEnum, cli.py StrEnum,
# routing.py Literal). The Literal form is kept here to preserve the
# string-level contract with policy JSONs, but the canonical runtime
# enum lives in ``apmode.backends.protocol.Lane``.
Lane = Literal["submission", "discovery", "optimization"]

# Backends available per lane (Phase 1: only nlmixr2 implemented)
_LANE_BACKENDS: dict[str, list[str]] = {
    "submission": ["nlmixr2"],
    "discovery": ["nlmixr2", "jax_node", "agentic_llm"],
    "optimization": ["nlmixr2", "jax_node", "agentic_llm"],
}


@dataclass(frozen=True)
class DispatchDecision:
    """Result of the lane router's dispatch decision."""

    lane: str
    backends: list[str]
    node_eligible: bool
    data_sufficient_for_node: bool
    constraints: list[str] = field(default_factory=list)
    # Policy-resolved missing-data directive. None when route is called
    # without a policy (legacy call sites); backends that see None fall back
    # to their historical behavior.
    missing_data_directive: MissingDataDirective | None = None


def route(
    lane: Lane,
    manifest: EvidenceManifest,
    policy: MissingDataPolicy | None = None,
) -> DispatchDecision:
    """Route a run to the appropriate backends based on lane and manifest.

    Args:
        lane: Operating lane selected by user intent.
        manifest: Evidence manifest from data profiling.
        policy: Optional lane-specific missing-data policy. When provided,
            the returned ``DispatchDecision`` carries a
            ``MissingDataDirective`` resolved from ``(policy, manifest)``.

    Returns:
        DispatchDecision with admissible backends and constraint notes.
    """
    if lane not in _LANE_BACKENDS:
        msg = f"Invalid lane '{lane}'. Must be one of {sorted(_LANE_BACKENDS)}"
        raise ValueError(msg)

    backends = list(_LANE_BACKENDS[lane])
    constraints: list[str] = []
    node_eligible = lane != "submission"
    data_sufficient = True

    # Submission lane: NODE is never eligible (PRD §3 hard rule)
    if lane == "submission":
        if "jax_node" in backends:
            backends.remove("jax_node")
        if "agentic_llm" in backends:
            backends.remove("agentic_llm")
        constraints.append("NODE excluded (submission lane)")

    # NODE data sufficiency check. Follow-up: (
    # ): use the new `node_dim_budget` manifest field as the
    # primary gate. Legacy richness-only check retained as a fallback for
    # v1 manifests (schema_version < 2).
    if node_eligible and "jax_node" in backends:
        budget = getattr(manifest, "node_dim_budget", None)
        if budget is not None and budget == 0:
            backends.remove("jax_node")
            data_sufficient = False
            constraints.append("NODE removed: node_dim_budget=0 (insufficient design feasibility)")
        else:
            # Fallback to v1 richness + coverage gate.
            sparse_and_inadequate = (
                manifest.richness_category == "sparse"
                and manifest.absorption_phase_coverage == "inadequate"
            )
            if sparse_and_inadequate:
                backends.remove("jax_node")
                data_sufficient = False
                constraints.append("NODE removed: sparse data + inadequate absorption coverage")

        # Low identifiability ceiling also constrains NODE (defensive).
        if manifest.identifiability_ceiling == "low" and "jax_node" in backends:
            backends.remove("jax_node")
            data_sufficient = False
            constraints.append("NODE removed: low identifiability ceiling")

        # TAD contamination: shape-heuristic-driven dispatch signals are
        # down-weighted (NODE pulled, MM hint softened). Only applies
        # when the manifest carries the v2 field.
        tad = getattr(manifest, "tad_consistency_flag", None)
        if tad == "contaminated":
            constraints.append(
                "TAD contamination: shape heuristics down-weighted, recheck ingest alignment"
            )
            if "jax_node" in backends:
                backends.remove("jax_node")
                data_sufficient = False
                constraints.append("NODE removed: tad_consistency_flag=contaminated")

    # Flip-flop hint for downstream search seeding. Does not remove
    # backends — just records the directive for initial-estimate generation.
    ff = getattr(manifest, "flip_flop_risk", None)
    if ff in {"possible", "likely"}:
        constraints.append(f"flip_flop_risk={ff}: seed both ka>ke and ka<ke branches")

    # Resolve the missing-data directive (policy-driven).
    directive = resolve_directive(policy, manifest) if policy is not None else None

    # BLQ constraint note. A directive is always present on the live
    # orchestrator path; legacy callers that pass ``policy=None`` get no
    # BLQ advisory. The pre-v0.3 ``0.20`` fallback was removed because it
    # was an unversioned literal that drifted from every lane policy
    # (submission 0.05, optimization 0.10, discovery 0.15).
    if directive is not None:
        constraints.append(
            f"BLQ method {directive.blq_method} selected (burden={manifest.blq_burden:.2%})"
        )

    # Protocol heterogeneity note
    if manifest.protocol_heterogeneity == "pooled-heterogeneous":
        constraints.append("Pooled-heterogeneous: IOV must be tested")

    # Covariate missingness note. When a directive is present use the resolved
    # method; otherwise emit the legacy "full-information recommended" hint.
    if directive is not None and directive.covariate_method != "exclude":
        m_part = f", m={directive.m_imputations}" if directive.m_imputations is not None else ""
        constraints.append(f"Covariate method: {directive.covariate_method}{m_part}")
    elif (
        manifest.covariate_missingness is not None
        and manifest.covariate_missingness.fraction_incomplete > 0.15
    ):
        constraints.append(
            f"Covariate missingness {manifest.covariate_missingness.fraction_incomplete:.2f} "
            f"> 0.15: full-information likelihood recommended"
        )

    return DispatchDecision(
        lane=lane,
        backends=backends,
        node_eligible=node_eligible,
        data_sufficient_for_node=data_sufficient,
        constraints=constraints,
        missing_data_directive=directive,
    )
