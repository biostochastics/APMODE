# SPDX-License-Identifier: GPL-2.0-or-later
"""Pydantic AST models for the PK DSL (PRD §4.2.5, ARCHITECTURE.md §2.2).

Each DSL module is a discriminated union of typed variants.
The top-level DSLSpec is the compiled model specification that flows
through BackendRunner.run() and into the reproducibility bundle.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    """First-order absorption. Calibration value ``ka`` lives in ``DSLSpec.initial``."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["FirstOrder"] = "FirstOrder"


class ZeroOrder(BaseModel):
    """Zero-order (constant-rate) absorption.

    Calibration value ``dur`` lives in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["ZeroOrder"] = "ZeroOrder"


class LaggedFirstOrder(BaseModel):
    """First-order absorption with lag time.

    Calibration values ``ka``, ``tlag`` live in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["LaggedFirstOrder"] = "LaggedFirstOrder"


class Transit(BaseModel):
    """Transit compartment absorption: ``n`` transit compartments (structural).

    The transit chain (Savic et al. 2007) feeds into a depot compartment
    with first-order transfer rate ``ka`` to the central compartment.
    rxode2's transit(n, mtt) handles the chain; ka controls depot→central.
    Calibration values ``ktr``, ``ka`` live in ``DSLSpec.initial``; ``n`` is
    the structural chain length and stays inline, not an estimated parameter.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["Transit"] = "Transit"
    n: int


class MixedFirstZero(BaseModel):
    """Mixed first-order + zero-order absorption.

    Calibration values ``ka``, ``dur``, ``frac`` live in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["MixedFirstZero"] = "MixedFirstZero"


class Erlang(BaseModel):
    """Erlang absorption: integer ``n`` transit compartments, shared ktr, no terminal ka.

    ``n`` is structural (chain length); lowers to an explicit
    n-compartment ODE chain — *not* rxode2's
    ``transit(n, mtt)`` (which uses gamma interpolation and a terminal ka).
    See ADR-0003 D2. Validator caps ``n ≤ 7`` because longer chains add
    little resolution and inflate state count quadratically. Calibration
    value ``ktr`` lives in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["Erlang"] = "Erlang"
    n: int


class ParallelFirstOrder(BaseModel):
    """Two parallel first-order depots: fast + slow, fraction to depot 1.

    Distinct from :class:`MixedFirstZero` (which is first+zero-order); this
    is two simultaneous first-order routes (Pumas PK43; Soufsaf 2021 PMX).
    Calibration values ``ka1``, ``ka2``, ``frac`` live in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["ParallelFirstOrder"] = "ParallelFirstOrder"


class SumIG(BaseModel):
    """Sum of Inverse Gaussians absorption (Csajka 2005; Weiss & Wegner 2022).

    The validator restricts ``k`` to {1, 2}. ``k`` is structural and stays
    inline; per-component calibration values live in ``DSLSpec.initial``.

    Per-component ``MT_1``, ``MT_2``, ``RD2_1``, ``RD2_2``, ``weight_1`` are
    calibration values that live in ``DSLSpec.initial`` so existing IIV /
    CovariateLink / Prior machinery resolves them as plain ``StanIdentifier``
    strings.

    Label switching is prevented by the validator constraint
    ``MT_1 < MT_2`` (positive-difference parameterisation). The implicit
    second weight is ``w_2 = 1 - weight_1``; not stored.

    Identifiability: when ``k >= 2`` the disposition parameters (CL/V/Q)
    must be fixed externally — enforced as a cross-module validator check
    against fixed-prior entries in ``DSLSpec.priors``. A planned
    ``EvidenceManifest.disposition_fixed`` flag (ADR-0003 D7) is not yet
    implemented. See ADR-0003 D5.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["SumIG"] = "SumIG"
    k: int


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
    | Erlang
    | ParallelFirstOrder
    | SumIG
    | NODEAbsorption,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Distribution Module variants
# ---------------------------------------------------------------------------


class OneCmt(BaseModel):
    """One-compartment distribution.

    Calibration value ``V`` lives in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["OneCmt"] = "OneCmt"


class TwoCmt(BaseModel):
    """Two-compartment distribution.

    Calibration values ``V1``, ``V2``, ``Q`` live in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["TwoCmt"] = "TwoCmt"


class ThreeCmt(BaseModel):
    """Three-compartment distribution.

    Calibration values ``V1``, ``V2``, ``V3``, ``Q2``, ``Q3`` live in
    ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["ThreeCmt"] = "ThreeCmt"


class TMDDCore(BaseModel):
    """Target-mediated drug disposition (full model).

    Ref: Mager & Jusko (2001), J Pharmacokinet Pharmacodyn 28:507-532.
    Calibration values ``V``, ``R0``, ``kon``, ``koff``, ``kint`` live in
    ``DSLSpec.initial``. ``V`` is the central volume of distribution,
    required for dose→concentration conversion and dimensional consistency
    of binding/elimination terms.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["TMDD_Core"] = "TMDD_Core"


class TMDDQSS(BaseModel):
    """TMDD quasi-steady-state approximation.

    Ref: Gibiansky et al. (2008), J Pharmacokinet Pharmacodyn 35:573-591.
    Calibration values ``V``, ``R0``, ``KD``, ``kint`` live in
    ``DSLSpec.initial``. ``V`` is the central volume. ``KD`` ≈ koff/kon is
    the equilibrium dissociation constant; note that ``KSS = (koff +
    kint)/kon`` differs from ``KD`` when ``kint > 0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["TMDD_QSS"] = "TMDD_QSS"


DistributionModule = Annotated[
    OneCmt | TwoCmt | ThreeCmt | TMDDCore | TMDDQSS,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Elimination Module variants
# ---------------------------------------------------------------------------


class LinearElim(BaseModel):
    """Linear (first-order) elimination.

    Calibration value ``CL`` lives in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["Linear"] = "Linear"


class MichaelisMenten(BaseModel):
    """Michaelis-Menten (saturable) elimination.

    Calibration values ``Vmax``, ``Km`` live in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["MichaelisMenten"] = "MichaelisMenten"


class ParallelLinearMM(BaseModel):
    """Parallel linear + Michaelis-Menten elimination.

    Calibration values ``CL``, ``Vmax``, ``Km`` live in ``DSLSpec.initial``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["ParallelLinearMM"] = "ParallelLinearMM"


class TimeVaryingElim(BaseModel):
    """Time-varying elimination: decay function shape (structural).

    ``kdecay`` controls the rate of clearance change over time; for
    exponential decay ``CL(t) = CL * exp(-kdecay * t)``. Calibration values
    ``CL``, ``kdecay`` live in ``DSLSpec.initial`` — ``kdecay`` defaults to
    0.1 there when omitted (see ``DSLSpec.get_initial``). ``decay_fn``
    selects WHICH decay shape and is structural, so it stays inline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["TimeVarying"] = "TimeVarying"
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


VariabilityItem = Annotated[
    IIV | IOV,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------
#
# Covariate effects moved out of the variability-item union entirely (they
# are no longer IIV/IOV siblings) into their own top-level ``covariates:``
# block/``DSLSpec.covariates`` list -- see ``CovariateLink`` docstring for
# the rationale.

_COVARIATE_FORM_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "power": ("theta", "ref"),
    "exponential": ("theta",),
    "linear": ("theta",),
    "categorical": ("reference",),
    "maturation": ("tm50", "hill"),
}


def _require_covariate_fields(
    form: str,
    *,
    theta: float | None,
    ref: float | None,
    reference: str | None,
    tm50: float | None,
    hill: float | None,
) -> None:
    """Enforce the exact per-``form`` field shape for a covariate declaration.

    Each ``form`` requires exactly the fields listed in
    ``_COVARIATE_FORM_REQUIRED_FIELDS`` and no others (every field not
    required by ``form`` must be ``None``) -- this keeps a compiled spec
    from ever carrying a stray reference-value field a given form ignores,
    and gives a fast, specific error at construction time rather than a
    silent no-op in the emitter.
    """
    all_fields: dict[str, float | str | None] = {
        "theta": theta,
        "ref": ref,
        "reference": reference,
        "tm50": tm50,
        "hill": hill,
    }
    required = _COVARIATE_FORM_REQUIRED_FIELDS[form]
    missing = [name for name in required if all_fields[name] is None]
    if missing:
        msg = f"covariate form {form!r} requires field(s) {sorted(missing)} to be set"
        raise ValueError(msg)
    extraneous = sorted(
        name for name, value in all_fields.items() if name not in required and value is not None
    )
    if extraneous:
        msg = f"covariate form {form!r} does not accept field(s) {extraneous}"
        raise ValueError(msg)


class CovariateLink(BaseModel):
    """A covariate effect on a structural parameter, with first-class reference values.

    Each ``form`` carries its own explicit, named field set. All numeric/string
    fields below are calibration-like (excluded from ``structure_fingerprint``,
    included in ``spec_fingerprint`` -- see ``apmode.dsl.canonical``) with one
    exception: ``reference`` (the categorical baseline *level name*) is treated
    as structural, since it identifies which level the model defines as
    baseline rather than a re-estimable numeric value.

    - ``power``: ``theta`` (coefficient's initial/starting estimate) and
      ``ref`` (fixed reference covariate value the formula centers on,
      e.g. ``ref=70`` for a 70 kg reference weight).
    - ``exponential`` / ``linear``: ``theta`` only (no reference-centering
      in either formula).
    - ``categorical``: ``reference`` only (the baseline level's name; the
      numeric 0/1 encoding of non-reference levels is a data-adapter
      concern, not a DSL one -- see ``apmode.data.adapters``). The
      coefficient itself is not yet configurable here.
    - ``maturation``: ``tm50`` and ``hill`` (initial/starting estimates for
      the TM50 and Hill-exponent parameters respectively).

    Exactly the fields required by ``form`` may be set; every other field
    must be ``None`` (enforced by :meth:`_check_field_shape`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["CovariateLink"] = "CovariateLink"
    param: StanIdentifier
    covariate: StanIdentifier
    form: Literal["power", "exponential", "linear", "categorical", "maturation"]
    theta: float | None = None
    ref: float | None = None
    reference: str | None = None
    tm50: float | None = None
    hill: float | None = None

    @model_validator(mode="after")
    def _check_field_shape(self) -> CovariateLink:
        _require_covariate_fields(
            self.form,
            theta=self.theta,
            ref=self.ref,
            reference=self.reference,
            tm50=self.tm50,
            hill=self.hill,
        )
        return self


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


class ObservationEndpoint(BaseModel):
    """One named analyte/endpoint of a multi-analyte ``observations:`` block.

    Formular sharpening plan §4 Phase 1 (P1.7): introduced alongside the new
    plural ``observations:`` grammar block so a spec can declare more than
    one DVID-routed prediction/error-model pair -- e.g. free drug plus total
    target for a TMDD assay design (Gibiansky et al. 2008). ``name`` is the
    block's map key, kept on the model itself (not just as the
    ``DSLSpec.observations`` dict key) so a flattened list view --
    :meth:`DSLSpec.observation_endpoints` -- still carries it.

    Two cross-entry invariants the Pydantic model itself cannot see are
    checked by :func:`apmode.dsl.validator.validate_dsl` instead:
    ``dvid`` must be unique across every entry in the same block
    (``FrmCode.AST_OBSERVATIONS_DVID_COLLISION``), and ``prediction`` must
    name one of :meth:`DSLSpec.known_prediction_variables`
    (``FrmCode.AST_OBSERVATIONS_PREDICTION_UNKNOWN``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    dvid: int
    prediction: StanIdentifier
    error: ObservationModule


class ExperimentalFlags(BaseModel):
    """Explicit opt-in flags for experimental AST variants.

    ``NODEAbsorption`` / ``NODEElimination`` are accepted in the AST, but the
    registered DSL emitters (nlmixr2, Stan, FREM) do not lower them. The
    separate NODE runner/trainer stack owns neural execution. Without an
    explicit gate, a spec author could write a NODE variant and get a generic
    backend-capability failure with no signal that this is an intentionally
    experimental route. ``node=True`` is the author's acknowledgement of that;
    ``validate_dsl`` fails closed with
    ``FrmCode.LANE_NODE_EXPERIMENTAL_GATE`` when a NODE variant is present and
    this flag is unset, independent of lane.
    """

    model_config = ConfigDict(frozen=True)

    node: bool = False


# ---------------------------------------------------------------------------
# Metadata block (Formular sharpening plan §4 Phase 1, P1.2)
# ---------------------------------------------------------------------------


class Metadata(BaseModel):
    """Optional top-level ``metadata: { ... }`` block — free-text spec provenance.

    Every field is an optional string; none affects compilation, validation,
    or emission. Purely descriptive context carried through to the
    reproducibility bundle manifest (``BundleEmitter``) for human/audit
    consumption.
    """

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    intent: str | None = None
    context_of_use: str | None = None
    analyte: str | None = None
    version: str | None = None


# ---------------------------------------------------------------------------
# Units block (Formular sharpening plan §4 Phase 1, P1.3)
# ---------------------------------------------------------------------------


class UnitsDeclaration(BaseModel):
    """Optional top-level ``units: { ... }`` block — GLOBAL measurement units.

    This is *not* per-parameter unit annotation: Formular has no syntax to
    attach a unit to an individual ``CL``/``V``/``ka`` value. Instead, this
    block declares the four base/derived units the spec's data and
    ``initial:`` values are conventionally expressed in, and
    ``apmode.dsl.units`` uses it to (a) infer each calibration parameter's
    *expected* dimension from its structural role and (b) check that
    ``volume`` is dimensionally reachable from ``amount``/``concentration``
    (``Volume = Amount / Concentration``). See that module's docstring for
    the exact consistency algorithm and its documented limitations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    time: str
    amount: str
    concentration: str
    volume: str


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
    # Optional multi-analyte form, additive to the ``observation`` field
    # above. The singular ``observation:`` form remains the common case and
    # stays supported. When an ``observations:`` block is compiled,
    # ``observation`` above is synthesized from the first entry in insertion
    # order so consumers that only understand one endpoint still have a
    # representative value. Code that must see every endpoint should call
    # :meth:`observation_endpoints` instead of reading either field directly.
    observations: dict[str, ObservationEndpoint] | None = None
    # Covariate effects live in their own top-level list, distinct from
    # ``variability`` (whose item kinds are IIV/IOV). See ``CovariateLink``
    # for the per-form field contract.
    covariates: list[CovariateLink] = Field(default_factory=list)
    priors: list[PriorSpec] = Field(default_factory=list)
    # Flat parameter-name -> value dict for calibration values used by the
    # structural modules (absorption/distribution/elimination). See
    # ``calibration_param_names()`` for the exact names. Structural/topology
    # fields (Transit.n, Erlang.n, SumIG.k, NODE dim/constraint_template,
    # TimeVaryingElim.decay_fn) stay inline on their module and are never
    # present here. Observation sigmas and covariate theta/ref values also
    # stay inline.
    initial: dict[str, float] = Field(default_factory=dict)
    # Optional free-text provenance block. None when the spec was built
    # programmatically without a metadata: block.
    metadata: Metadata | None = None
    # Optional global units declaration. ``None`` when the spec was built
    # without a ``units:`` block; ``apmode.dsl.units.unit_coverage_report``
    # returns a ``status="not_declared"`` report in that case rather than
    # raising.
    units: UnitsDeclaration | None = None
    # #17: source_meta is populated by ``parse_dsl_with_source`` as a
    # sidecar map from AST node kind (``"absorption"`` / ``"distribution"``
    # / ``"elimination"`` / ``"observation"`` / ``"variability[i]"``) to
    # a ``(line, column)`` tuple pulled off the Lark parse tree. Empty
    # when the spec was built programmatically (no parse tree). The
    # validator uses it to annotate error messages with ``file.pk:L:C``.
    source_meta: dict[str, tuple[int, int]] = Field(default_factory=dict)
    # Experimental-feature opt-in gate. Defaults to all-False; specs that use
    # a NODE variant need to set ``experimental.node=True``.
    experimental: ExperimentalFlags = Field(default_factory=ExperimentalFlags)
    # Provenance trail for ``use <macro>`` statement expansion. Each entry is
    # ``"{MacroDef.name}@{MacroDef.version}"`` in source order. Deliberately
    # excluded from structure/spec fingerprints because macro expansion is
    # syntax sugar; byte-identical post-expansion specs must fingerprint
    # identically whether or not either used a ``use`` shortcut.
    macros_used: list[str] = Field(default_factory=list)

    def get_initial(self, name: str, default: float | None = None) -> float | None:
        """Look up a calibration value by name, falling back to ``default``.

        ``kdecay`` on :class:`TimeVaryingElim` is the one calibration
        parameter with a conventional non-error default (0.1) when omitted
        from ``initial:`` — callers needing that behaviour pass
        ``default=0.1`` explicitly; this method itself has no opinion on
        per-name defaults.
        """
        return self.initial.get(name, default)

    def observation_endpoints(self) -> list[ObservationEndpoint]:
        """Return every observation endpoint, normalizing legacy and multi-analyte syntax.

        The single unified accessor downstream code (emitters, scoring,
        canonicalization) should prefer over branching on whether the spec
        used the singular ``observation:`` form or the plural multi-analyte
        ``observations:`` block. When ``observations`` is unset (the common case),
        ``observation`` is wrapped in a single synthetic endpoint named
        ``"default"`` with ``dvid=1`` (matching
        ``apmode.data.adapters.PK_DVID_ALLOWLIST``'s numeric convention for
        the sole PK endpoint) and ``prediction="C_central"`` (the canonical
        name every distribution module's primary concentration prediction
        is addressable by -- see :meth:`known_prediction_variables`).
        """
        if self.observations:
            return list(self.observations.values())
        return [
            ObservationEndpoint(
                name="default", dvid=1, prediction="C_central", error=self.observation
            )
        ]

    def known_prediction_variables(self) -> frozenset[str]:
        """Return the prediction-variable names an ``observations:`` entry may reference.

        ``"C_central"`` is the canonical name for the primary disposition
        compartment's concentration -- always available (it is the sole
        observable when only the legacy singular ``observation:`` block is
        used, and every emitter's central-concentration output, ``cp`` in
        nlmixr2, is addressable by this name). ``TMDDQSS`` distribution
        additionally exposes ``"C_target_total"``: the ``Rtot`` ODE state
        (total target/receptor concentration) the nlmixr2 emitter already
        integrates, giving a genuine second analyte for TMDD assay designs
        that measure free drug plus total target (Gibiansky et al. 2008).
        ``TMDDCore`` does not expose an equivalent single named state (total
        target there is the sum of two separate states, ``R`` + ``RC``,
        which no emitter currently synthesizes as one named output) and no
        other structural module exposes a second named prediction state --
        e.g. metabolite/parent-child compartment topology does not exist in
        the DSL yet. An ``observations:`` entry naming
        anything else is rejected by the validator with
        ``FrmCode.AST_OBSERVATIONS_PREDICTION_UNKNOWN``.
        """
        names = {"C_central"}
        if isinstance(self.distribution, TMDDQSS):
            names.add("C_target_total")
        return frozenset(names)

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
            # n is structural topology (set inline on Transit), not an
            # estimated parameter. Only ktr and ka are calibratable.
            names.extend(["ktr", "ka"])
        elif isinstance(abs_mod, MixedFirstZero):
            names.extend(["ka", "dur", "frac"])
        elif isinstance(abs_mod, Erlang):
            # n is structural-integer (set by transform, not estimated); ktr is the
            # only parameter exposed for IIV/priors/covariates.
            names.append("ktr")
        elif isinstance(abs_mod, ParallelFirstOrder):
            names.extend(["ka1", "ka2", "frac"])
        elif isinstance(abs_mod, SumIG):
            # Flattened per-component names so the validator/IIV/Prior
            # machinery sees plain StanIdentifier strings. k itself is
            # structural (validator restricts to {1, 2}); not exposed for
            # variability.
            names.extend(["MT_1", "MT_2", "RD2_1", "RD2_2", "weight_1"])
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

    def calibration_param_names(self) -> list[str]:
        """Return the subset of :meth:`structural_param_names` requiring an ``initial:`` value.

        Excludes names that are structural-but-not-calibrated: Transit/Erlang
        ``n`` (chain length — an integer topology choice, set by the
        transform, not estimated) and NODE ``node_abs_w*``/``node_elim_w*``
        weight names (no DSL primitive exists to give them an initial
        value; that is a NODE-backend concern, not the ``initial:`` block).
        """
        return [
            name
            for name in self.structural_param_names()
            if name != "n" and not name.startswith(("node_abs_w", "node_elim_w"))
        ]
