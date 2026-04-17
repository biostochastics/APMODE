# SPDX-License-Identifier: GPL-2.0-or-later
"""Pydantic AST models for the PK DSL (PRD §4.2.5, ARCHITECTURE.md §2.2).

Each DSL module is a discriminated union of typed variants.
The top-level DSLSpec is the compiled model specification that flows
through BackendRunner.run() and into the reproducibility bundle.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from apmode.dsl.priors import PriorSpec  # noqa: TC001 — Pydantic resolves type at runtime

# ---------------------------------------------------------------------------
# Identifier type alias
# ---------------------------------------------------------------------------

# StanIdentifier enforces the Stan language's identifier grammar at AST
# construction time (must start with a letter; subsequent characters are
# letters, digits, or underscores). This prevents injection via
# LLM-proposed covariate/parameter names leaking into emitted Stan code
# (PRD §4.2.5). The nlmixr2 emitter separately accepts this character
# set; R's broader identifier grammar (e.g. ``WT.baseline``) is
# disallowed here to keep the AST Stan-safe. If dotted R-style names
# ever become necessary, they must be translated in the data adapter
# rather than relaxing this contract.
StanIdentifier = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]


# ---------------------------------------------------------------------------
# Absorption Module variants
# ---------------------------------------------------------------------------


class IVBolus(BaseModel):
    """IV bolus dosing — no absorption phase.

    Distinguishes "dose enters the central compartment directly" from
    first-order oral absorption. Emitters should skip the depot compartment
    and route doses straight to the central cmt.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["IVBolus"] = "IVBolus"


class FirstOrder(BaseModel):
    """First-order absorption: ka."""

    model_config = ConfigDict(frozen=True)
    type: Literal["FirstOrder"] = "FirstOrder"
    ka: float


class ZeroOrder(BaseModel):
    """Zero-order (constant-rate) absorption: dur."""

    model_config = ConfigDict(frozen=True)
    type: Literal["ZeroOrder"] = "ZeroOrder"
    dur: float


class LaggedFirstOrder(BaseModel):
    """First-order absorption with lag time: ka, tlag."""

    model_config = ConfigDict(frozen=True)
    type: Literal["LaggedFirstOrder"] = "LaggedFirstOrder"
    ka: float
    tlag: float


class Transit(BaseModel):
    """Transit compartment absorption: n transit compartments, ktr, ka.

    The transit chain (Savic et al. 2007) feeds into a depot compartment
    with first-order transfer rate ka to the central compartment.
    rxode2's transit(n, mtt) handles the chain; ka controls depot→central.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["Transit"] = "Transit"
    n: int
    ktr: float
    ka: float


class MixedFirstZero(BaseModel):
    """Mixed first-order + zero-order absorption: ka, dur, frac."""

    model_config = ConfigDict(frozen=True)
    type: Literal["MixedFirstZero"] = "MixedFirstZero"
    ka: float
    dur: float
    frac: float


class NODEAbsorption(BaseModel):
    """Neural ODE absorption (Discovery/Optimization lanes only)."""

    model_config = ConfigDict(frozen=True)
    type: Literal["NODE_Absorption"] = "NODE_Absorption"
    dim: int
    constraint_template: Literal[
        "monotone_increasing",
        "monotone_decreasing",
        "bounded_positive",
        "saturable",
        "unconstrained_smooth",
    ]


AbsorptionModule = Annotated[
    IVBolus
    | FirstOrder
    | ZeroOrder
    | LaggedFirstOrder
    | Transit
    | MixedFirstZero
    | NODEAbsorption,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Distribution Module variants
# ---------------------------------------------------------------------------


class OneCmt(BaseModel):
    """One-compartment distribution: V."""

    model_config = ConfigDict(frozen=True)
    type: Literal["OneCmt"] = "OneCmt"
    V: float


class TwoCmt(BaseModel):
    """Two-compartment distribution: V1, V2, Q."""

    model_config = ConfigDict(frozen=True)
    type: Literal["TwoCmt"] = "TwoCmt"
    V1: float
    V2: float
    Q: float


class ThreeCmt(BaseModel):
    """Three-compartment distribution: V1, V2, V3, Q2, Q3."""

    model_config = ConfigDict(frozen=True)
    type: Literal["ThreeCmt"] = "ThreeCmt"
    V1: float
    V2: float
    V3: float
    Q2: float
    Q3: float


class TMDDCore(BaseModel):
    """Target-mediated drug disposition (full model): V, R0, kon, koff, kint.

    Ref: Mager & Jusko (2001), J Pharmacokinet Pharmacodyn 28:507-532.
    V is the central volume of distribution, required for dose→concentration
    conversion and dimensional consistency of binding/elimination terms.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["TMDD_Core"] = "TMDD_Core"
    V: float
    R0: float
    kon: float
    koff: float
    kint: float


class TMDDQSS(BaseModel):
    """TMDD quasi-steady-state approximation: V, R0, KD, kint.

    Ref: Gibiansky et al. (2008), J Pharmacokinet Pharmacodyn 35:573-591.
    V is the central volume. KD ≈ koff/kon is the equilibrium dissociation
    constant; note that KSS = (koff + kint)/kon differs from KD when kint > 0.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["TMDD_QSS"] = "TMDD_QSS"
    V: float
    R0: float
    KD: float
    kint: float


DistributionModule = Annotated[
    OneCmt | TwoCmt | ThreeCmt | TMDDCore | TMDDQSS,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Elimination Module variants
# ---------------------------------------------------------------------------


class LinearElim(BaseModel):
    """Linear (first-order) elimination: CL."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Linear"] = "Linear"
    CL: float


class MichaelisMenten(BaseModel):
    """Michaelis-Menten (saturable) elimination: Vmax, Km."""

    model_config = ConfigDict(frozen=True)
    type: Literal["MichaelisMenten"] = "MichaelisMenten"
    Vmax: float
    Km: float


class ParallelLinearMM(BaseModel):
    """Parallel linear + Michaelis-Menten elimination: CL, Vmax, Km."""

    model_config = ConfigDict(frozen=True)
    type: Literal["ParallelLinearMM"] = "ParallelLinearMM"
    CL: float
    Vmax: float
    Km: float


class TimeVaryingElim(BaseModel):
    """Time-varying elimination: CL with decay function.

    kdecay controls the rate of clearance change over time.
    For exponential decay: CL(t) = CL * exp(-kdecay * t).
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["TimeVarying"] = "TimeVarying"
    CL: float
    kdecay: float = 0.1
    decay_fn: Literal["exponential", "half_life", "linear"]


class NODEElimination(BaseModel):
    """Neural ODE elimination (Discovery/Optimization lanes only)."""

    model_config = ConfigDict(frozen=True)
    type: Literal["NODE_Elimination"] = "NODE_Elimination"
    dim: int
    constraint_template: Literal[
        "monotone_increasing",
        "monotone_decreasing",
        "bounded_positive",
        "saturable",
        "unconstrained_smooth",
    ]


EliminationModule = Annotated[
    LinearElim | MichaelisMenten | ParallelLinearMM | TimeVaryingElim | NODEElimination,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Variability Module variants
# ---------------------------------------------------------------------------


class OccasionByStudy(BaseModel):
    """One occasion per study."""

    model_config = ConfigDict(frozen=True)
    type: Literal["ByStudy"] = "ByStudy"


class OccasionByVisit(BaseModel):
    """One occasion per visit."""

    model_config = ConfigDict(frozen=True)
    type: Literal["ByVisit"] = "ByVisit"
    column: str


class OccasionByDoseEpoch(BaseModel):
    """One occasion per dosing epoch."""

    model_config = ConfigDict(frozen=True)
    type: Literal["ByDoseEpoch"] = "ByDoseEpoch"
    column: str


class OccasionCustom(BaseModel):
    """User-defined occasion column."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Custom"] = "Custom"
    column: str


OccasionSpec = Annotated[
    OccasionByStudy | OccasionByVisit | OccasionByDoseEpoch | OccasionCustom,
    Field(discriminator="type"),
]


class IIV(BaseModel):
    """Inter-individual variability: params with diagonal or block structure."""

    model_config = ConfigDict(frozen=True)
    type: Literal["IIV"] = "IIV"
    params: list[StanIdentifier]
    structure: Literal["diagonal", "block"]


class IOV(BaseModel):
    """Inter-occasion variability: params with occasion specification."""

    model_config = ConfigDict(frozen=True)
    type: Literal["IOV"] = "IOV"
    params: list[StanIdentifier]
    occasions: OccasionSpec


class CovariateLink(BaseModel):
    """Covariate effect on a parameter."""

    model_config = ConfigDict(frozen=True)
    type: Literal["CovariateLink"] = "CovariateLink"
    param: StanIdentifier
    covariate: StanIdentifier
    form: Literal["power", "exponential", "linear", "categorical", "maturation"]


VariabilityItem = Annotated[
    IIV | IOV | CovariateLink,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Observation Module variants
# ---------------------------------------------------------------------------


class Proportional(BaseModel):
    """Proportional residual error: sigma_prop."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Proportional"] = "Proportional"
    sigma_prop: float


class Additive(BaseModel):
    """Additive residual error: sigma_add."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Additive"] = "Additive"
    sigma_add: float


class Combined(BaseModel):
    """Combined proportional + additive residual error."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Combined"] = "Combined"
    sigma_prop: float
    sigma_add: float


class BLQM3(BaseModel):
    """BLQ handling via M3 method (left-censoring).

    Composes with an underlying residual error model via error_model.
    Defaults to proportional (prop.sd=0.1) for backward compatibility.
    nlmixr2 censoring uses CENS/LIMIT data columns, not model-block syntax.

    #30: ``sigma_prop`` and ``sigma_add`` are always present on the model
    regardless of ``error_model`` — that keeps ``==`` comparisons stable
    and avoids plumbing ``Optional[float]`` through every downstream
    consumer. Use :meth:`active_sigmas` when counting fitted parameters
    so vestigial defaults are not double-counted.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["BLQ_M3"] = "BLQ_M3"
    loq_value: float
    error_model: Literal["proportional", "additive", "combined"] = "proportional"
    sigma_prop: float = 0.1
    sigma_add: float = 0.5

    def active_sigmas(self) -> list[str]:
        """Return the subset of sigma fields that enter the likelihood.

        ``proportional`` → ``["sigma_prop"]``; ``additive`` →
        ``["sigma_add"]``; ``combined`` → both. Parameter-count and
        prior-coverage helpers should prefer this over inspecting every
        field so that vestigial defaults do not silently inflate the
        count (Gate 1 scoring-contract consistency).
        """
        if self.error_model == "proportional":
            return ["sigma_prop"]
        if self.error_model == "additive":
            return ["sigma_add"]
        return ["sigma_prop", "sigma_add"]


class BLQM4(BaseModel):
    """BLQ handling via M4 method (censoring with positive constraint).

    Composes with an underlying residual error model via error_model.
    Defaults to proportional (prop.sd=0.1) for backward compatibility.
    nlmixr2 censoring uses CENS/LIMIT data columns, not model-block syntax.

    See :class:`BLQM3` for the rationale behind always-present sigma
    fields; use :meth:`active_sigmas` when counting parameters.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["BLQ_M4"] = "BLQ_M4"
    loq_value: float
    error_model: Literal["proportional", "additive", "combined"] = "proportional"
    sigma_prop: float = 0.1
    sigma_add: float = 0.5

    def active_sigmas(self) -> list[str]:
        """Sigma fields that enter the likelihood. See :meth:`BLQM3.active_sigmas`."""
        if self.error_model == "proportional":
            return ["sigma_prop"]
        if self.error_model == "additive":
            return ["sigma_add"]
        return ["sigma_prop", "sigma_add"]


ObservationModule = Annotated[
    Proportional | Additive | Combined | BLQM3 | BLQM4,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Top-level DSL Spec
# ---------------------------------------------------------------------------


class DSLSpec(BaseModel):
    """Compiled DSL specification — the typed AST that flows through the system.

    This is the contract between DSL compiler, backends, and bundle emitter.
    Replaces the ``Any`` placeholder in BackendRunner.run() and RSubprocessRequest.spec.
    """

    model_config = ConfigDict(frozen=True)

    model_id: str
    absorption: AbsorptionModule
    distribution: DistributionModule
    elimination: EliminationModule
    variability: list[VariabilityItem]
    observation: ObservationModule
    priors: list[PriorSpec] = Field(default_factory=list)
    # #17: source_meta is populated by ``parse_dsl_with_source`` as a
    # sidecar map from AST node kind (``"absorption"`` / ``"distribution"``
    # / ``"elimination"`` / ``"observation"`` / ``"variability[i]"``) to
    # a ``(line, column)`` tuple pulled off the Lark parse tree. Empty
    # when the spec was built programmatically (no parse tree). The
    # validator uses it to annotate error messages with ``file.pk:L:C``.
    source_meta: dict[str, tuple[int, int]] = Field(default_factory=dict)

    def has_node_modules(self) -> bool:
        """Check if this spec uses any NODE modules."""
        return isinstance(self.absorption, NODEAbsorption) or isinstance(
            self.elimination, NODEElimination
        )

    def node_max_dim(self) -> int:
        """Return the maximum NODE dimension used, or 0 if no NODE modules."""
        dims: list[int] = []
        if isinstance(self.absorption, NODEAbsorption):
            dims.append(self.absorption.dim)
        if isinstance(self.elimination, NODEElimination):
            dims.append(self.elimination.dim)
        return max(dims) if dims else 0

    def structural_param_names(self) -> list[str]:
        """Return the names of all structural parameters in the spec.

        #11: NODE modules contribute ``node_abs_w[...]`` /
        ``node_elim_w[...]`` entries (one per input-layer weight under the
        Bräm hybrid PRD §4.2.4 layout) so downstream Variability items
        that target NODE weights pass ``_validate_variability`` instead
        of being rejected on a ``valid_params`` miss. IVBolus contributes
        nothing (no absorption parameters — dose enters central directly).
        """
        names: list[str] = []
        # Absorption params
        abs_mod = self.absorption
        if isinstance(abs_mod, FirstOrder):
            names.append("ka")
        elif isinstance(abs_mod, ZeroOrder):
            names.append("dur")
        elif isinstance(abs_mod, LaggedFirstOrder):
            names.extend(["ka", "tlag"])
        elif isinstance(abs_mod, Transit):
            # n is estimated as continuous via log/exp (rxode2 gamma interpolation)
            names.extend(["n", "ktr", "ka"])
        elif isinstance(abs_mod, MixedFirstZero):
            names.extend(["ka", "dur", "frac"])
        elif isinstance(abs_mod, IVBolus):
            # IV bolus has no absorption parameters.
            pass
        elif isinstance(abs_mod, NODEAbsorption):
            # Bräm-style hybrid: IIV lives on input-layer weights. Expose
            # one name per dim so Variability validation accepts them.
            names.extend(f"node_abs_w{i}" for i in range(abs_mod.dim))

        # Distribution params
        dist_mod = self.distribution
        if isinstance(dist_mod, OneCmt):
            names.append("V")
        elif isinstance(dist_mod, TwoCmt):
            names.extend(["V1", "V2", "Q"])
        elif isinstance(dist_mod, ThreeCmt):
            names.extend(["V1", "V2", "V3", "Q2", "Q3"])
        elif isinstance(dist_mod, TMDDCore):
            names.extend(["V", "R0", "kon", "koff", "kint"])
        elif isinstance(dist_mod, TMDDQSS):
            names.extend(["V", "R0", "KD", "kint"])

        # Elimination params
        elim_mod = self.elimination
        if isinstance(elim_mod, LinearElim):
            names.append("CL")
        elif isinstance(elim_mod, MichaelisMenten):
            names.extend(["Vmax", "Km"])
        elif isinstance(elim_mod, ParallelLinearMM):
            names.extend(["CL", "Vmax", "Km"])
        elif isinstance(elim_mod, TimeVaryingElim):
            names.extend(["CL", "kdecay"])
        elif isinstance(elim_mod, NODEElimination):
            names.extend(f"node_elim_w{i}" for i in range(elim_mod.dim))

        return names
