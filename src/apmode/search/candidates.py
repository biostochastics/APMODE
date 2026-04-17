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

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    Additive,
    Combined,
    DSLSpec,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    OccasionByStudy,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    Transit,
    TwoCmt,
)
from apmode.ids import generate_candidate_id

if TYPE_CHECKING:
    from apmode.bundle.models import EvidenceManifest, MissingDataDirective


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
    force_blq_method: str | None = None  # "m3" or "m4" when BLQ burden > 0.20
    force_iov: bool = False  # True when protocol_heterogeneity = pooled-heterogeneous
    lloq_value: float = 1.0  # LLOQ for BLQ M3/M4 observation model
    # Full Beal method tag. When set by ``apply_directive``, takes precedence
    # over the heuristic defaults above. Values M1/M3/M4/M6+/M7+ come from
    # ``MissingDataDirective.blq_method``; "none" means no BLQ handling.
    # M1 means drop BLQ rows pre-fit (data preprocessing).
    # M7+/M6+ mean impute zeros pre-fit with inflated additive residual error;
    # the DSL does not emit a dedicated BLQ observation model for them — the
    # R harness is expected to inflate the additive error coefficient.
    blq_strategy: str = "none"

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

        # Structural complexity bounded by richness. Filter absorption to the
        # simplest admissible set; preserve "none" so IV sparse datasets still
        # generate an IV candidate.
        if manifest.richness_category == "sparse":
            space.structural_cmt = [1]
            space.absorption_types = [
                t for t in space.absorption_types if t in ("none", "first_order")
            ]
            if not space.absorption_types:
                space.absorption_types = ["first_order"]
        elif manifest.richness_category == "moderate":
            space.structural_cmt = [1, 2]

        # Absorption complexity. Note: do NOT remove "transit" when the
        # detector reports "simple" — the prominence-based peak detector
        # correctly classifies transit-chain absorption datasets (which
        # produce a smooth single peak with a delayed apex) as "simple"
        # rather than "multi-phase". Transit candidates must remain
        # discoverable so the structural search can recover them via BIC.
        if manifest.absorption_complexity == "multi-phase":
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

        # Nonlinear clearance — graded routing per PRD §10 Q2 follow-up.
        # ``strong`` (all 3 signals: curvature ratio + terminal R^2 failure
        # + dose nonproportionality) → full MM cross-product.
        # ``moderate`` (2 signals) → keep linear + add MM as sentinel; do
        # not blow out the search space on a 2-cmt-linear false positive
        # like warfarin.
        # ``weak`` / ``none`` → linear only.
        if manifest.nonlinear_clearance_evidence_strength in ("moderate", "strong"):
            if "michaelis_menten" not in space.elimination_types:
                space.elimination_types.append("michaelis_menten")
        else:
            space.elimination_types = ["linear"]

        # Error-model preference from profiler heuristic (Beal 2001, Ahn 2008).
        # Supersedes the legacy ``blq_burden > 0.20`` override when present;
        # the heuristic triggers BLQ_M3 at 10% BLQ and prunes additive-only
        # candidates that otherwise let add.sd absorb censored variance.
        pref = manifest.error_model_preference
        if pref is not None:
            if pref.primary in ("blq_m3", "blq_m4"):
                space.force_blq_method = "m3" if pref.primary == "blq_m3" else "m4"
                # ``allowed`` is guaranteed non-empty by ErrorModelPreference; for
                # BLQ primaries it is always ⊆ {proportional, combined} (never
                # additive-only).
                space.error_types = list(pref.allowed)
                if manifest.lloq_value is not None and manifest.lloq_value > 0:
                    space.lloq_value = manifest.lloq_value
                else:
                    warnings.warn(
                        f"BLQ {pref.primary.upper()} forced but "
                        f"manifest.lloq_value is {manifest.lloq_value!r}; "
                        f"falling back to default {space.lloq_value}. "
                        "Populate lloq_value in the profiler for reliable "
                        "censored likelihoods.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                space.error_types = list(pref.allowed)
        elif manifest.blq_burden > 0.20:
            # Legacy fallback for manifests emitted before the heuristic was
            # introduced (missing error_model_preference).
            space.force_blq_method = "m3"
            space.error_types = ["proportional", "combined"]
            if manifest.lloq_value is not None and manifest.lloq_value > 0:
                space.lloq_value = manifest.lloq_value

        # Protocol heterogeneity → IOV must be tested
        if manifest.protocol_heterogeneity == "pooled-heterogeneous":
            space.force_iov = True

        # Covariates
        if covariate_names:
            params = ["CL", "V"]
            for cov in covariate_names:
                for param in params:
                    space.covariates.append((param, cov, "power"))

        return space

    def apply_directive(
        self,
        directive: MissingDataDirective,
        manifest: EvidenceManifest,
    ) -> SearchSpace:
        """Overlay a policy-resolved missing-data directive onto this space.

        The directive's BLQ method supersedes the manifest-derived heuristic
        (``blq_burden > 0.20 → M3``) because it reflects the lane policy's
        explicit threshold and any ``blq_force_m3`` override.

        Mapping (see ``apmode.data.missing_data.resolve_directive`` for the
        policy-side logic):

          - ``M3``/``M4``: DSL BLQ observation model (BLQM3/BLQM4).
          - ``M7+``/``M6+``: no DSL BLQ observation model; R harness
            imputes zeros pre-fit and inflates the additive residual
            error. Standard error models stay in the search space so the
            additive component is estimable.
          - ``M1``: drop BLQ rows pre-fit (data preprocessing). No DSL
            change; additive/proportional error types remain available.
          - Anything else is passed through unchanged.

        Returns a new :class:`SearchSpace` with the directive applied —
        the receiver is not mutated. Previously this method mutated
        ``self`` in place, which made multiple directive applications
        accumulate state silently. Callers should use the returned
        value.
        """
        blq = directive.blq_method

        # Defaults: carry forward current state, override only where the
        # directive speaks. Keep list fields distinct (no aliasing).
        new_blq_strategy = blq
        new_force_blq_method = self.force_blq_method
        new_error_types = list(self.error_types)
        new_lloq_value = self.lloq_value

        if blq in ("M3", "M4"):
            new_force_blq_method = blq.lower()  # "m3" or "m4"
            new_error_types = ["proportional", "combined"]
            if manifest.lloq_value is not None and manifest.lloq_value > 0:
                new_lloq_value = manifest.lloq_value
        elif blq in ("M7+", "M6+", "M1"):
            # No DSL BLQ observation model — preprocessing handles these.
            # Clear any force set by the from_manifest heuristic so the
            # emitter does not inadvertently include a BLQ observation model.
            new_force_blq_method = None

        return replace(
            self,
            blq_strategy=new_blq_strategy,
            force_blq_method=new_force_blq_method,
            error_types=new_error_types,
            lloq_value=new_lloq_value,
        )


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

    Dispatch constraints applied:
      - force_blq_method → BLQ_M3/M4 observation model
      - force_iov → IOV variability item added
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
                        force_blq_method=search_space.force_blq_method,
                        lloq_value=search_space.lloq_value,
                        force_iov=search_space.force_iov,
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
    force_blq_method: str | None = None,
    lloq_value: float = 1.0,
    force_iov: bool = False,
) -> DSLSpec | None:
    """Build a single DSLSpec from search dimensions.

    When force_blq_method is set, observation model uses BLQ_M3 or BLQ_M4
    instead of standard error models (PRD §4.2.1: blq_burden > 0.20).

    When force_iov is set, IOV on CL is added to variability
    (PRD §4.2.1: protocol_heterogeneity = pooled-heterogeneous).
    """
    # Absorption
    ka = params.get("ka", 1.0)
    absorption: IVBolus | FirstOrder | LaggedFirstOrder | Transit
    has_absorption = abs_type != "none"
    if abs_type == "none":
        absorption = IVBolus()
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

    # Observation model — BLQ-aware when forced by dispatch constraints.
    # Additive is excluded from the M3/M4 residual-error slot because the
    # additive sigma absorbs censored mass and biases estimates (Ahn 2008).
    observation: Proportional | Additive | Combined | BLQM3 | BLQM4
    if force_blq_method == "m3":
        valid_errs = ("proportional", "combined")
        blq_err = err_type if err_type in valid_errs else "proportional"
        observation = BLQM3(loq_value=lloq_value, error_model=blq_err)
    elif force_blq_method == "m4":
        valid_errs = ("proportional", "combined")
        blq_err = err_type if err_type in valid_errs else "proportional"
        observation = BLQM4(loq_value=lloq_value, error_model=blq_err)
    elif err_type == "proportional":
        observation = Proportional(sigma_prop=0.15)
    elif err_type == "additive":
        observation = Additive(sigma_add=0.5)
    elif err_type == "combined":
        observation = Combined(sigma_prop=0.1, sigma_add=0.3)
    else:
        return None

    # Variability — add IOV when forced by protocol heterogeneity
    variability_items: list[IIV | IOV] = [IIV(params=iiv_params, structure="diagonal")]
    if force_iov:
        variability_items.append(IOV(params=["CL"], occasions=OccasionByStudy()))

    return DSLSpec(
        model_id=generate_candidate_id(),
        absorption=absorption,
        distribution=distribution,
        elimination=elimination,
        variability=variability_items,
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


class SearchDAGSealedError(RuntimeError):
    """Raised when attempting to mutate a sealed SearchDAG.

    #20: the DAG is the canonical lineage artifact written into the
    reproducibility bundle (candidate_lineage.json). Post-seal mutation
    would silently diverge the in-memory state from the file on disk and
    break the _COMPLETE digest invariant.
    """


class SearchDAG:
    """Tracks the search DAG for candidate lineage (PRD §4.3.2).

    Each candidate is a node. Edges represent parent→child derivation
    via transforms (structural change, covariate addition, etc.).

    A DAG moves through exactly two states: mutable (accepts add_root /
    add_child / update_score) and sealed (all mutators raise
    :class:`SearchDAGSealedError`). Call :meth:`seal` at the end of the
    search run, immediately before writing
    ``candidate_lineage.json``. Re-sealing is idempotent.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, SearchNode] = {}
        self._sealed: bool = False

    def _require_mutable(self, op: str) -> None:
        if self._sealed:
            msg = (
                f"SearchDAG is sealed; cannot {op}. Post-seal mutation "
                "would desync the in-memory DAG from candidate_lineage.json "
                "and break bundle digest integrity."
            )
            raise SearchDAGSealedError(msg)

    def seal(self) -> None:
        """Seal the DAG against further mutation. Idempotent."""
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def add_root(self, spec: DSLSpec) -> SearchNode:
        """Add a root candidate (NCA-derived, no parent)."""
        self._require_mutable("add_root")
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
        self._require_mutable("add_child")
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
        self._require_mutable("update_score")
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

    def iter_nodes(self) -> list[SearchNode]:
        """Return all nodes (public access for graph building)."""
        return list(self._nodes.values())

    def to_edges(self) -> list[tuple[str, str, str]]:
        """Return (parent_id, child_id, transform) for all non-root nodes."""
        return [
            (n.parent_id, n.candidate_id, n.transform or "")
            for n in self._nodes.values()
            if n.parent_id is not None
        ]
