# SPDX-License-Identifier: GPL-2.0-or-later
"""Candidate model generation for automated search (PRD §4.2.3).

Search dimensions (all expressed as DSL module combinations):
  - Structural: 1-cmt / 2-cmt / 3-cmt x absorption variants x elimination variants
  - Covariate: stepwise (SCM forward/backward) or LASSO-on-ETAs
  - Random effects: diagonal vs. block omega; IIV and IOV candidates
  - Residual error: additive, proportional, combined

Scoring: AIC/BIC for nested; cross-validated predictive metrics for non-nested.
Search is bounded by EvidenceManifest constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from apmode.dsl.ast_models import (
    IIV,
    Additive,
    Combined,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    Transit,
    TwoCmt,
)
from apmode.ids import generate_candidate_id

if TYPE_CHECKING:
    from apmode.bundle.models import EvidenceManifest


# ---------------------------------------------------------------------------
# Search Space Definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchDimension:
    """A single axis of the search space."""

    name: str
    values: list[str]
    enabled: bool = True


@dataclass
class SearchSpace:
    """Defines the bounded search space for automated model selection.

    Constrained by the EvidenceManifest per PRD §4.2.1 dispatch rules.
    """

    structural_cmt: list[int] = field(default_factory=lambda: [1, 2])
    absorption_types: list[str] = field(
        default_factory=lambda: ["none", "first_order", "lagged_first_order", "transit"]
    )
    elimination_types: list[str] = field(
        default_factory=lambda: ["linear", "michaelis_menten", "parallel"]
    )
    error_types: list[str] = field(
        default_factory=lambda: ["proportional", "additive", "combined"]
    )
    iiv_structures: list[str] = field(default_factory=lambda: ["diagonal", "block"])
    covariates: list[tuple[str, str, str]] = field(default_factory=list)

    @classmethod
    def from_manifest(
        cls,
        manifest: EvidenceManifest,
        covariate_names: list[str] | None = None,
    ) -> SearchSpace:
        """Create a search space bounded by the evidence manifest.

        Applies PRD §4.2.1 dispatch constraints:
          - simple absorption → deprioritize transit
          - nonlinear clearance → include MM candidates
          - sparse data → reduce structural complexity
        """
        space = cls()

        # Structural complexity bounded by richness
        if manifest.richness_category == "sparse":
            space.structural_cmt = [1]
            space.absorption_types = ["first_order"]
        elif manifest.richness_category == "moderate":
            space.structural_cmt = [1, 2]

        # Absorption complexity
        if manifest.absorption_complexity == "simple":
            # Remove transit from search space (simple absorption unlikely to need it)
            if "transit" in space.absorption_types:
                space.absorption_types.remove("transit")
        elif manifest.absorption_complexity == "multi-phase":
            # Multi-phase absorption often requires transit compartment models
            if "transit" not in space.absorption_types:
                space.absorption_types.append("transit")
            if "lagged_first_order" not in space.absorption_types:
                space.absorption_types.append("lagged_first_order")
        elif (
            manifest.absorption_complexity == "lag-signature"
            and "lagged_first_order" not in space.absorption_types
        ):
            space.absorption_types.append("lagged_first_order")
        # "unknown" keeps defaults — explicit no-op

        # Route-based absorption filtering
        if manifest.route_certainty == "confirmed" and "none" not in space.absorption_types:
            pass  # Keep all absorption types for confirmed routes

        # Nonlinear clearance
        if not manifest.nonlinear_clearance_signature:
            space.elimination_types = ["linear"]
        else:
            # Ensure MM candidates present
            if "michaelis_menten" not in space.elimination_types:
                space.elimination_types.append("michaelis_menten")

        # Covariates
        if covariate_names:
            params = ["CL", "V"]
            for cov in covariate_names:
                for param in params:
                    space.covariates.append((param, cov, "power"))

        return space


# ---------------------------------------------------------------------------
# Candidate Generation
# ---------------------------------------------------------------------------


def generate_root_candidates(
    search_space: SearchSpace,
    base_params: dict[str, float] | None = None,
) -> list[DSLSpec]:
    """Generate root candidate DSLSpecs from the search space.

    Root candidates use NCA-derived initial estimates (or defaults).
    Each unique (structural x absorption x elimination x error) combination
    produces one candidate.
    """
    defaults = base_params or {"ka": 1.0, "CL": 5.0, "V": 70.0}
    candidates: list[DSLSpec] = []

    for n_cmt in search_space.structural_cmt:
        for abs_type in search_space.absorption_types:
            for elim_type in search_space.elimination_types:
                for err_type in search_space.error_types:
                    spec = _build_spec(
                        n_cmt=n_cmt,
                        abs_type=abs_type,
                        elim_type=elim_type,
                        err_type=err_type,
                        params=defaults,
                    )
                    if spec is not None:
                        candidates.append(spec)

    return candidates


def _build_spec(
    *,
    n_cmt: int,
    abs_type: str,
    elim_type: str,
    err_type: str,
    params: dict[str, float],
) -> DSLSpec | None:
    """Build a single DSLSpec from search dimensions."""
    # Absorption
    ka = params.get("ka", 1.0)
    absorption: FirstOrder | LaggedFirstOrder | Transit
    has_absorption = True
    if abs_type == "none":
        # IV bolus: approximate with large ka (instantaneous absorption)
        absorption = FirstOrder(ka=100.0)
        has_absorption = False
    elif abs_type == "first_order":
        absorption = FirstOrder(ka=ka)
    elif abs_type == "lagged_first_order":
        absorption = LaggedFirstOrder(ka=ka, tlag=0.5)
    elif abs_type == "transit":
        absorption = Transit(n=3, ktr=2.0, ka=ka)
    else:
        return None

    # Distribution
    v = params.get("V", 70.0)
    distribution: OneCmt | TwoCmt | ThreeCmt
    if n_cmt == 1:
        distribution = OneCmt(V=v)
        iiv_params = ["CL", "V"]
    elif n_cmt == 2:
        distribution = TwoCmt(V1=v, V2=v * 0.5, Q=v * 0.1)
        iiv_params = ["CL", "V1"]
    elif n_cmt == 3:
        distribution = ThreeCmt(V1=v, V2=v * 0.5, V3=v * 0.3, Q2=v * 0.1, Q3=v * 0.05)
        iiv_params = ["CL", "V1"]
    else:
        return None

    # Elimination
    cl = params.get("CL", 5.0)
    elimination: LinearElim | MichaelisMenten | ParallelLinearMM
    if elim_type == "linear":
        elimination = LinearElim(CL=cl)
    elif elim_type == "michaelis_menten":
        elimination = MichaelisMenten(Vmax=cl * 20, Km=10.0)
    elif elim_type == "parallel":
        elimination = ParallelLinearMM(CL=cl, Vmax=cl * 20, Km=10.0)
    else:
        return None

    if has_absorption:
        iiv_params.append("ka")  # include ka IIV for oral models only

    # Observation model
    observation: Proportional | Additive | Combined
    if err_type == "proportional":
        observation = Proportional(sigma_prop=0.15)
    elif err_type == "additive":
        observation = Additive(sigma_add=0.5)
    elif err_type == "combined":
        observation = Combined(sigma_prop=0.1, sigma_add=0.3)
    else:
        return None

    return DSLSpec(
        model_id=generate_candidate_id(),
        absorption=absorption,
        distribution=distribution,
        elimination=elimination,
        variability=[IIV(params=iiv_params, structure="diagonal")],
        observation=observation,
    )


# ---------------------------------------------------------------------------
# Search DAG Tracking
# ---------------------------------------------------------------------------


@dataclass
class SearchNode:
    """A node in the search DAG (candidate lineage)."""

    candidate_id: str
    parent_id: str | None
    spec: DSLSpec
    transform: str | None = None
    score: float | None = None
    converged: bool | None = None


class SearchDAG:
    """Tracks the search DAG for candidate lineage (PRD §4.3.2).

    Each candidate is a node. Edges represent parent→child derivation
    via transforms (structural change, covariate addition, etc.).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, SearchNode] = {}

    def add_root(self, spec: DSLSpec) -> SearchNode:
        """Add a root candidate (NCA-derived, no parent)."""
        node = SearchNode(
            candidate_id=spec.model_id,
            parent_id=None,
            spec=spec,
        )
        self._nodes[spec.model_id] = node
        return node

    def add_child(
        self,
        parent_id: str,
        spec: DSLSpec,
        transform: str,
    ) -> SearchNode:
        """Add a child candidate derived from a parent via transform."""
        node = SearchNode(
            candidate_id=spec.model_id,
            parent_id=parent_id,
            spec=spec,
            transform=transform,
        )
        self._nodes[spec.model_id] = node
        return node

    def update_score(self, candidate_id: str, score: float, converged: bool) -> None:
        """Update a node with estimation results."""
        if candidate_id in self._nodes:
            self._nodes[candidate_id].score = score
            self._nodes[candidate_id].converged = converged

    def get_node(self, candidate_id: str) -> SearchNode | None:
        """Get a search node by ID."""
        return self._nodes.get(candidate_id)

    def get_roots(self) -> list[SearchNode]:
        """Get all root nodes (no parent)."""
        return [n for n in self._nodes.values() if n.parent_id is None]

    def get_children(self, parent_id: str) -> list[SearchNode]:
        """Get all children of a node."""
        return [n for n in self._nodes.values() if n.parent_id == parent_id]

    @property
    def size(self) -> int:
        """Number of nodes in the DAG."""
        return len(self._nodes)

    def to_lineage_entries(self) -> list[dict[str, str | None]]:
        """Export as CandidateLineageEntry-compatible dicts."""
        return [
            {
                "candidate_id": n.candidate_id,
                "parent_id": n.parent_id,
                "transform": n.transform,
            }
            for n in self._nodes.values()
        ]
