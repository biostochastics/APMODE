# SPDX-License-Identifier: GPL-2.0-or-later
"""Structured FRM-{TAXON}-NNN error code registry for the Formular DSL.

This module is the single source of truth for the code taxonomy;
:mod:`apmode.dsl.validator` attaches one :class:`FrmCode` value to every
:class:`ValidationError` it constructs so callers can identify checks
without parsing prose.

Taxonomy — ``FRM-{TAXON}-NNN``:

- ``SYN``   — grammar/parse-level errors. Reserved: today, malformed DSL
  text fails inside :func:`apmode.dsl.grammar.compile_dsl` (a Lark
  ``UnexpectedInput`` subclass) *before* a :class:`~apmode.dsl.ast_models.DSLSpec`
  exists, so no ``FRM-SYN-*`` code is emitted by
  :func:`apmode.dsl.validator.validate_dsl` today. The slot is reserved so a
  future pass that re-surfaces parse diagnostics through the same coded
  channel has somewhere to land them.
- ``AST``   — structural integrity of the parsed spec: duplicate
  declarations, empty required lists, references to structural parameters
  that do not exist, and module/module compatibility (e.g. TMDD requires
  Linear elimination). These checks do not compare a value against a
  numeric bound; they check the *shape* of the tree.
- ``SEM``   — semantic/numeric constraint violations: positivity,
  non-negativity, unit-interval membership, integer floors, capped ranges,
  and cross-field numeric relations (e.g. SumIG ``MT_1 < MT_2``).
- ``LANE``  — lane-admissibility rejections (PRD §3): a construct that is
  well-formed and numerically valid but is not admissible in the requested
  lane (NODE modules in Submission, SumIG in Submission, NODE dim above a
  lane's ceiling).
- ``BE``    — backend-capability errors: a capability tag the spec
  requires (:func:`apmode.dsl.capabilities.tags_for_spec`) that a named
  emitter has no working code path for (``supported`` status other than
  ``"supported"`` per :func:`apmode.dsl.capabilities.report`), or a
  request naming an unregistered emitter. Emitted by
  :func:`apmode.dsl.validator.validate_backend_bound` (Formular
  sharpening plan §4 Phase 1, P1.8 — seven-level validator API).
- ``DATA``  — data-bound checks: checks that require the bound dataset
  (a pandas ``DataFrame``), not just the spec — e.g. a multi-analyte
  ``observations:`` block with no ``DVID`` column in the data, or a
  ``covariates:`` entry naming a column absent from the data. Emitted by
  :func:`apmode.dsl.validator.validate_data_bound`.
- ``POLICY`` — policy-bound checks: the spec against a loaded
  :class:`apmode.governance.policy.GatePolicy` — lane/policy mismatch,
  or a NODE module present while the policy's ``gate2.node_eligible`` is
  false. Emitted by :func:`apmode.dsl.validator.validate_policy_bound`.
  Candidate-metric gate threshold checks run later in the governance layer,
  once a fitted candidate exists.
- ``PRIOR`` — prior-related validation. See "Why FRM-PRIOR now has one
  emitted code" below.

Prior validation boundary
-------------------------
:mod:`apmode.dsl.priors` still enforces evidence-quality / justification
checks via exceptions and ``list[str]`` prose — ``PriorSpec``'s
``model_validator`` raises ``pydantic.ValidationError`` for missing
justification/``historical_refs`` on informative sources, and
``validate_prior_justification`` / ``validate_priors`` return ``list[str]``
errors rather than :class:`~apmode.dsl.validator.ValidationError` instances.
That module's return contracts are depended on directly by six-plus call
sites (:mod:`apmode.bundle.emitter`, :mod:`apmode.backends.transform_parser`,
:mod:`apmode.benchmarks.models`, :mod:`apmode.dsl.prior_transforms`,
:mod:`apmode.dsl.stan_emitter`) and by test assertions that pattern-match on
message substrings (e.g. ``assert any("50" in e ...)`` in
``tests/unit/test_prior_justification.py``) — rewiring *those* functions to
emit coded :class:`~apmode.dsl.validator.ValidationError` objects remains
out of scope (an API-breaking change across that call graph). ``priors.py``
itself is unchanged by this section and stays exception/string-based.

The ``priors:`` grammar block lets a human author Bayesian priors directly
in Formular text, and
:func:`apmode.dsl.grammar.compile_dsl` lowers each entry through
:func:`apmode.dsl.priors.build_prior_spec` — the same canonical factory the
agentic ``SetPrior`` transform uses. A ``build_prior_spec`` failure at this
grammar-compile boundary (unresolvable target, family/target-kind mismatch,
or an informative source missing justification/``historical_refs``) is
caught and re-raised as a :class:`FormularCompileError` carrying
``PRIOR_INVALID_DECLARATION`` — the same coded-exception shape already used
for block-cardinality violations (``AST_MISSING_REQUIRED_BLOCK``,
``AST_DUPLICATE_BLOCK``), rather than a bare ``ValueError``. This is safe to
activate now specifically *because* the grammar path is new: nothing
upstream depends on its error shape being a prose ``list[str]``, unlike
``priors.py``'s existing entry points.

Every other prior-validation gap the Phase 0 docstring listed (duplicate
prior target, justification below minimum length, malformed DOI when
called via ``priors.py`` directly) remains prose/exception-based and
un-coded — only the grammar-compile boundary is migrated here.
"""

from __future__ import annotations

from enum import StrEnum


class FrmCode(StrEnum):
    """Stable Formular DSL validation error codes.

    Members are grouped by taxon; the string value is the wire code
    persisted onto ``ValidationError.code`` and surfaced to CLI output,
    bundle artifacts, and any agentic-transform error-recovery loop.
    Every member's value must appear in ``docs/FORMULAR_ERROR_CODES.md``
    (enforced by ``tests/unit/test_dsl_error_codes.py``).
    """

    # -- FRM-SEM: semantic / numeric constraint violations -----------------
    SEM_POSITIVE = "FRM-SEM-001"
    """A structural parameter (volume, rate, sigma, ...) must be > 0."""

    SEM_NON_NEGATIVE = "FRM-SEM-002"
    """A structural parameter (e.g. ``tlag``) must be >= 0."""

    SEM_UNIT_INTERVAL = "FRM-SEM-003"
    """A fraction/weight parameter must lie strictly in (0, 1)."""

    SEM_POSITIVE_INT = "FRM-SEM-004"
    """An integer count (Transit/Erlang ``n``, NODE ``dim``) must be >= 1."""

    SEM_ERLANG_MAX_N = "FRM-SEM-005"
    """Erlang chain length ``n`` exceeds the supported cap of 7."""

    SEM_SUMIG_K_RANGE = "FRM-SEM-006"
    """SumIG ``k`` is outside the supported range [1, 2]."""

    SEM_SUMIG_MT_ORDERING = "FRM-SEM-007"
    """SumIG requires ``MT_1 < MT_2`` (label-switching guard, ADR-0003 D1)."""

    SEM_SUMIG_DISPOSITION_FIXED = "FRM-SEM-008"
    """SumIG k>=2 requires CL/V/Q fixed externally (ADR-0003 D5)."""

    SEM_NODE_TEMPLATE_MAX_DIM = "FRM-SEM-009"
    """NODE ``dim`` exceeds the max dim for its ``constraint_template``."""

    SEM_UNITS_INCONSISTENT = "FRM-SEM-010"
    """The ``units:`` block's ``volume`` is not dimensionally reachable from
    ``amount``/``concentration`` (``Volume = Amount / Concentration``), or
    ``concentration`` is not a mass/volume compound unit. Spec-internal
    (does not need the bound dataset), hence ``SEM`` rather than ``DATA`` --
    see ``apmode.dsl.units`` for the consistency algorithm."""

    # -- FRM-AST: AST-shape / structural-integrity errors -------------------
    AST_IIV_NO_DUPLICATE_PARAMS = "FRM-AST-001"
    """A structural parameter appears in more than one IIV block."""

    AST_COVARIATE_LINK_NO_DUPLICATE = "FRM-AST-002"
    """The same (param, covariate) CovariateLink pair is declared twice."""

    AST_NON_EMPTY_PARAMS = "FRM-AST-003"
    """An IIV/IOV block declares an empty ``params`` list."""

    AST_BLOCK_MIN_PARAMS = "FRM-AST-004"
    """``structure="block"`` IIV requires at least 2 params."""

    AST_IIV_PARAM_EXISTS = "FRM-AST-005"
    """An IIV block references a name with no matching structural parameter."""

    AST_IOV_PARAM_EXISTS = "FRM-AST-006"
    """An IOV block references a name with no matching structural parameter."""

    AST_COVARIATE_PARAM_EXISTS = "FRM-AST-007"
    """A CovariateLink references a name with no matching structural parameter."""

    AST_NO_VARIABILITY_ON_PARAM = "FRM-AST-008"
    """IIV/IOV declared on a parameter the emitters do not apply eta to (e.g. Transit ``n``)."""

    AST_TMDD_REQUIRES_LINEAR_ELIM = "FRM-AST-009"
    """TMDD distribution requires Linear elimination (provides CL for kel = CL/V)."""

    AST_MISSING_REQUIRED_BLOCK = "FRM-AST-010"
    """A required top-level block (absorption/distribution/elimination/observation) is absent."""

    AST_DUPLICATE_BLOCK = "FRM-AST-011"
    """A singleton top-level block or map-like declaration entry appears more than once."""

    AST_INITIAL_VALUE_MISSING = "FRM-AST-012"
    """A calibration parameter used by a structural module has no value in the `initial:` block."""

    AST_INITIAL_VALUE_UNUSED = "FRM-AST-013"
    """The `initial:` block declares a value for a parameter no structural module references."""

    AST_OBSERVATIONS_DVID_COLLISION = "FRM-AST-014"
    """Two entries in an ``observations:`` block declare the same ``dvid``."""

    AST_OBSERVATIONS_PREDICTION_UNKNOWN = "FRM-AST-015"
    """An ``observations:`` entry's ``prediction`` does not name a known state
    variable of the compiled model (see ``DSLSpec.known_prediction_variables``)."""

    AST_MACRO_UNKNOWN = "FRM-AST-016"
    """A top-level ``use <name>`` statement names a macro not present in
    ``apmode.dsl.macros.MACRO_REGISTRY``. Only the vetted standard-library
    registry is supported; user-defined macros are not loaded."""

    AST_MACRO_DUPLICATE_USE = "FRM-AST-017"
    """The same macro name appears in more than one ``use`` statement within
    a single spec. Re-running a macro's expansion twice on one spec is a
    correctness hazard (e.g. ``pkstd.standard_iiv`` would double-declare
    IIV on the same parameters), not merely redundant, so this is rejected
    rather than silently deduplicated."""

    # -- FRM-LANE: lane-admissibility rejections -----------------------------
    LANE_NODE_ADMISSIBILITY = "FRM-LANE-001"
    """NODE absorption/elimination modules are not admissible in Submission lane."""

    LANE_NODE_DIM_CEILING = "FRM-LANE-002"
    """NODE ``dim`` exceeds the requested lane's dimension ceiling."""

    LANE_ABSORPTION_ADMISSIBILITY = "FRM-LANE-003"
    """Absorption form (e.g. SumIG) is not admissible in the requested lane (ADR-0003 D6)."""

    LANE_NODE_EXPERIMENTAL_GATE = "FRM-LANE-004"
    """NODE variant used without the required ``experimental.node`` opt-in."""

    # -- FRM-BE: backend-capability errors -----------------------------------
    BE_UNKNOWN_BACKEND = "FRM-BE-001"
    """The requested backend name is not a registered DSL emitter
    (see ``apmode.dsl.capabilities.registered_emitters``)."""

    BE_CAPABILITY_UNSUPPORTED = "FRM-BE-002"
    """The spec exercises a :class:`~apmode.dsl.capabilities.CapabilityTag`
    the named backend does not report ``"supported"`` for; see
    :func:`apmode.dsl.capabilities.report`."""

    # -- FRM-DATA: data-bound errors ------------------------------------------
    DATA_REQUIRED_COLUMN_MISSING = "FRM-DATA-001"
    """The spec declares a multi-analyte ``observations:`` block but the
    bound dataset has no ``DVID`` column to route rows to each endpoint."""

    DATA_COVARIATE_COLUMN_MISSING = "FRM-DATA-002"
    """A ``covariates:`` entry references a covariate with no matching
    column in the bound dataset."""

    # -- FRM-POLICY: policy-bound errors --------------------------------------
    POLICY_LANE_MISMATCH = "FRM-POLICY-001"
    """The loaded :class:`~apmode.governance.policy.GatePolicy`'s ``lane``
    does not match the lane validation was requested for."""

    POLICY_NODE_INELIGIBLE = "FRM-POLICY-002"
    """The spec uses a NODE absorption/elimination module but the loaded
    policy's ``gate2.node_eligible`` is false."""

    # -- FRM-PRIOR: prior declaration/lowering errors (grammar-compile only) -
    PRIOR_INVALID_DECLARATION = "FRM-PRIOR-001"
    """A ``priors:`` block entry failed :func:`apmode.dsl.priors.build_prior_spec`
    construction at grammar-compile time: unresolvable target, family/target-kind
    mismatch, or an informative source missing justification/``historical_refs``.
    Raised by :func:`apmode.dsl.grammar.compile_dsl`, not :func:`apmode.dsl.validator.validate_dsl`
    — see the module docstring's "Why FRM-PRIOR now has one emitted code" for why
    only this new call site is coded."""


class FormularCompileError(ValueError):
    """A structural (block-cardinality) defect detected before a DSLSpec can exist.

    Raised by :func:`apmode.dsl.grammar.compile_dsl` when the raw parse tree
    violates the "exactly one absorption/distribution/elimination/observation
    block, at most one metadata/initial block, zero-or-more variability
    blocks" cardinality rule (Formular sharpening plan §4 Phase 1, P1.1).
    This has to be a distinct exception from :class:`~apmode.dsl.validator.ValidationError`
    because a missing required block means a :class:`~apmode.dsl.ast_models.DSLSpec`
    cannot be constructed at all (its fields are non-optional) — there is no
    spec instance yet for ``validate_dsl`` to inspect.
    """

    def __init__(self, code: FrmCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


__all__ = ["FormularCompileError", "FrmCode"]
