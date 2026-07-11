# SPDX-License-Identifier: GPL-2.0-or-later
"""Lark Transformer: parse tree → Pydantic AST (ARCHITECTURE.md §2.2).

Converts a Lark parse tree (from pk_grammar.lark) into typed Pydantic AST
models. Each grammar rule maps to a transformer method that returns the
corresponding AST node.

Top-level blocks may appear in any order (Formular sharpening plan §4
Phase 1, P1.1): ``model_body`` receives the already-transformed child of
every ``block`` in source order and dispatches by type, since the
cardinality of each block kind is already enforced pre-transform by
``apmode.dsl.grammar._validate_block_cardinality`` on the raw parse tree.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from lark import Transformer, v_args

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    AbsorptionModule,
    Additive,
    Combined,
    CovariateLink,
    DistributionModule,
    DSLSpec,
    EliminationModule,
    Erlang,
    ExperimentalFlags,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    Metadata,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    ObservationEndpoint,
    ObservationModule,
    OccasionByDoseEpoch,
    OccasionByStudy,
    OccasionByVisit,
    OccasionCustom,
    OneCmt,
    ParallelFirstOrder,
    ParallelLinearMM,
    Proportional,
    SumIG,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    UnitsDeclaration,
    VariabilityItem,
    ZeroOrder,
)
from apmode.dsl.priors import (
    BetaPrior,
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    HistoricalBorrowingPrior,
    InvGammaPrior,
    LKJPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
    PriorFamily,
    PriorSource,
)
from apmode.ids import generate_candidate_id


@dataclass
class RawPriorEntry:
    """An uncommitted ``priors:`` block entry, collected during parse.

    Deliberately *not* lowered to a real ``PriorSpec`` here — that happens in
    ``apmode.dsl.grammar.compile_dsl`` via ``build_prior_spec``, once the
    full spec (and therefore ``structural_param_names()``) is available, and
    outside any ``Transformer`` callback so a ``build_prior_spec`` failure
    surfaces as a plain ``FormularCompileError`` rather than being wrapped in
    ``lark.exceptions.VisitError`` (see ``DSLTransformer.priors_block``).
    """

    target: str
    family: PriorFamily
    source: PriorSource = "weakly_informative"
    justification: str = ""
    doi: str | None = None
    historical_refs: list[str] = field(default_factory=list)


_ABSORPTION_TYPES = (
    IVBolus,
    FirstOrder,
    ZeroOrder,
    LaggedFirstOrder,
    Transit,
    MixedFirstZero,
    Erlang,
    ParallelFirstOrder,
    SumIG,
    NODEAbsorption,
)
_DISTRIBUTION_TYPES = (OneCmt, TwoCmt, ThreeCmt, TMDDCore, TMDDQSS)
_ELIMINATION_TYPES = (
    LinearElim,
    MichaelisMenten,
    ParallelLinearMM,
    TimeVaryingElim,
    NODEElimination,
)
_OBSERVATION_TYPES = (Proportional, Additive, Combined, BLQM3, BLQM4)


@v_args(inline=True)
class DSLTransformer(Transformer):  # type: ignore[type-arg]
    """Transform Lark parse tree into Pydantic AST models.

    Each method name matches a grammar rule. Terminal values arrive as
    ``lark.Token`` (subclass of str). Numeric terminals are converted
    to Python types by the terminal methods below.
    """

    def __init__(self) -> None:
        super().__init__()
        # Side channel for ``priors:`` block entries (P1.5). Populated by
        # ``priors_block``; read by ``apmode.dsl.grammar.compile_dsl`` after
        # ``transform()`` returns. A fresh ``DSLTransformer`` is constructed
        # per ``compile_dsl`` call (never reused/cached), so this instance
        # state cannot leak across compiles.
        self.raw_priors: list[RawPriorEntry] = []
        # Side channel for ``covariates:`` block entries (P1.6), same
        # rationale as ``raw_priors`` above.
        self.raw_covariates: list[CovariateLink] = []
        # Side channel for ``observations:`` block entries (P1.7), same
        # rationale as ``raw_priors``/``raw_covariates`` above. Populated by
        # ``observations_block`` *before* ``model_body`` runs (Lark
        # transforms children bottom-up), so ``model_body`` can read it
        # directly to synthesize the mandatory ``observation`` field when
        # only the plural block was used -- see ``model_body``.
        self.raw_observations: dict[str, ObservationEndpoint] = {}
        # Side channel for `use <macro>` statements (P2.1), same rationale
        # as the other raw_* channels above -- collected here in source
        # order, expanded by `apmode.dsl.macros.expand_macros` from within
        # `apmode.dsl.grammar.compile_dsl` (outside any Transformer
        # callback) so an unknown-macro/duplicate-use FormularCompileError
        # is not wrapped in `lark.exceptions.VisitError`.
        self.raw_macro_uses: list[str] = []

    # --- Terminals ---

    def NUMBER(self, token: str) -> float:
        return float(token)

    def INT(self, token: str) -> int:
        return int(token)

    def NAME(self, token: str) -> str:
        return str(token)

    def STRING(self, token: str) -> str:
        """Decode an ``ESCAPED_STRING`` token (quotes + escapes) to a plain str."""
        decoded = json.loads(str(token))
        return str(decoded)

    def STRUCTURE(self, token: str) -> str:
        return str(token)

    def CONSTRAINT_TEMPLATE(self, token: str) -> str:
        return str(token)

    def DECAY_FN(self, token: str) -> str:
        return str(token)

    def ERROR_MODEL(self, token: str) -> str:
        return str(token)

    def PRIOR_SOURCE(self, token: str) -> str:
        return str(token)

    def BOOL(self, token: str) -> bool:
        return str(token) == "true"

    def DOTTED_NAME(self, token: str) -> str:
        return str(token)

    # --- Absorption ---

    def iv_bolus(self) -> IVBolus:
        return IVBolus()

    def first_order(self) -> FirstOrder:
        return FirstOrder()

    def zero_order(self) -> ZeroOrder:
        return ZeroOrder()

    def lagged_first_order(self) -> LaggedFirstOrder:
        return LaggedFirstOrder()

    def transit(self, n: int) -> Transit:
        return Transit(n=n)

    def mixed_first_zero(self) -> MixedFirstZero:
        return MixedFirstZero()

    def erlang(self, n: int) -> Erlang:
        return Erlang(n=n)

    def parallel_first_order(self) -> ParallelFirstOrder:
        return ParallelFirstOrder()

    def sum_ig(self, k: int) -> SumIG:
        return SumIG(k=k)

    def node_absorption(self, dim: int, ct: str) -> NODEAbsorption:
        return NODEAbsorption(dim=dim, constraint_template=ct)

    def absorption_type(self, variant: object) -> object:
        return variant

    def absorption(self, variant: object) -> object:
        return variant

    # --- Distribution ---

    def one_cmt(self) -> OneCmt:
        return OneCmt()

    def two_cmt(self) -> TwoCmt:
        return TwoCmt()

    def three_cmt(self) -> ThreeCmt:
        return ThreeCmt()

    def tmdd_core(self) -> TMDDCore:
        return TMDDCore()

    def tmdd_qss(self) -> TMDDQSS:
        return TMDDQSS()

    def distribution_type(self, variant: object) -> object:
        return variant

    def distribution(self, variant: object) -> object:
        return variant

    # --- Elimination ---

    def linear_elim(self) -> LinearElim:
        return LinearElim()

    def michaelis_menten(self) -> MichaelisMenten:
        return MichaelisMenten()

    def parallel_linear_mm(self) -> ParallelLinearMM:
        return ParallelLinearMM()

    def time_varying_elim(self, decay_fn: str) -> TimeVaryingElim:
        return TimeVaryingElim(decay_fn=decay_fn)

    def node_elimination(self, dim: int, ct: str) -> NODEElimination:
        return NODEElimination(dim=dim, constraint_template=ct)

    def elimination_type(self, variant: object) -> object:
        return variant

    def elimination(self, variant: object) -> object:
        return variant

    # --- Variability ---

    def param_list(self, *names: str) -> list[str]:
        return list(names)

    def iiv(self, params: list[str], structure: str) -> IIV:
        return IIV(params=params, structure=structure)

    def iov(self, params: list[str], occasions: object) -> IOV:
        return IOV(params=params, occasions=occasions)

    def occasion_spec(self, variant: object) -> object:
        return variant

    def occasion_by_study(self) -> OccasionByStudy:
        return OccasionByStudy()

    def occasion_by_visit(self, column: str) -> OccasionByVisit:
        return OccasionByVisit(column=column)

    def occasion_by_dose_epoch(self, column: str) -> OccasionByDoseEpoch:
        return OccasionByDoseEpoch(column=column)

    def occasion_custom(self, column: str) -> OccasionCustom:
        return OccasionCustom(column=column)

    def variability_item(self, item: object) -> object:
        return item

    def variability_block(self, *items: object) -> list[object]:
        return list(items)

    # --- Observation ---

    def proportional_obs(self, sigma_prop: float) -> Proportional:
        return Proportional(sigma_prop=sigma_prop)

    def additive_obs(self, sigma_add: float) -> Additive:
        return Additive(sigma_add=sigma_add)

    def combined_obs(self, sigma_prop: float, sigma_add: float) -> Combined:
        return Combined(sigma_prop=sigma_prop, sigma_add=sigma_add)

    def blq_m3(self, *args: object) -> BLQM3:
        if len(args) == 4:
            loq, err_model, sigma_prop, sigma_add = args
            return BLQM3(
                loq_value=loq,
                error_model=err_model,
                sigma_prop=sigma_prop,
                sigma_add=sigma_add,
            )
        return BLQM3(loq_value=args[0])

    def blq_m4(self, *args: object) -> BLQM4:
        if len(args) == 4:
            loq, err_model, sigma_prop, sigma_add = args
            return BLQM4(
                loq_value=loq,
                error_model=err_model,
                sigma_prop=sigma_prop,
                sigma_add=sigma_add,
            )
        return BLQM4(loq_value=args[0])

    def observation_type(self, variant: object) -> object:
        return variant

    def observation(self, variant: object) -> object:
        return variant

    # --- Multi-analyte Observations (P1.7) ---

    def observation_entry(
        self, name: str, dvid: int, prediction: str, error: ObservationModule
    ) -> ObservationEndpoint:
        return ObservationEndpoint(name=name, dvid=dvid, prediction=prediction, error=error)

    def observations_block(self, *entries: ObservationEndpoint) -> None:
        """Stash entries on the transformer instance; contribute nothing to ``model_body``.

        Keyed by ``name`` (last entry wins on a duplicate name -- the
        semantic validator does not currently flag duplicate names
        separately from duplicate ``dvid``s, since a name collision is
        already a plain Python dict-key collision here and is rare enough
        not to warrant its own FRM code). See :meth:`priors_block` for the
        identical stash-not-thread rationale.
        """
        self.raw_observations = {entry.name: entry for entry in entries}
        return None

    # --- Metadata (P1.2) ---

    def metadata_title(self, value: str) -> tuple[str, str]:
        return ("title", value)

    def metadata_intent(self, value: str) -> tuple[str, str]:
        return ("intent", value)

    def metadata_context_of_use(self, value: str) -> tuple[str, str]:
        return ("context_of_use", value)

    def metadata_analyte(self, value: str) -> tuple[str, str]:
        return ("analyte", value)

    def metadata_version(self, value: str) -> tuple[str, str]:
        return ("version", value)

    def metadata_block(self, *items: tuple[str, str]) -> Metadata:
        return Metadata(**dict(items))

    # --- Initial estimates (P1.4) ---

    def initial_item(self, name: str, value: float) -> tuple[str, float]:
        return (name, value)

    def initial_block(self, *items: tuple[str, float]) -> dict[str, float]:
        return dict(items)

    # --- Units (P1.3) ---

    def unit_expr(self, *names: str) -> str:
        """Join one or two NAME tokens into a bare ("h") or compound ("ng/mL") unit string."""
        return "/".join(names)

    def units_block(
        self, time: str, amount: str, concentration: str, volume: str
    ) -> UnitsDeclaration:
        return UnitsDeclaration(
            time=time, amount=amount, concentration=concentration, volume=volume
        )

    def experimental_block(self, node: bool) -> ExperimentalFlags:
        return ExperimentalFlags(node=node)

    # --- Priors (P1.5) ---
    #
    # Numeric-expression and prior-family constructors below build the exact
    # same Pydantic classes ``apmode.dsl.priors`` exposes, so a parsed family
    # is field-for-field identical to one a Python caller builds directly.
    # ``prior_entry``/``priors_block`` stop short of constructing a
    # ``PriorSpec`` — that step (via ``build_prior_spec``) happens in
    # ``apmode.dsl.grammar.compile_dsl``, see ``RawPriorEntry``'s docstring.

    def num_literal(self, value: float) -> float:
        return value

    def num_log(self, value: float) -> float:
        return math.log(value)

    def normal_prior(self, mu: float, sigma: float) -> NormalPrior:
        return NormalPrior(mu=mu, sigma=sigma)

    def lognormal_prior(self, mu: float, sigma: float) -> LogNormalPrior:
        return LogNormalPrior(mu=mu, sigma=sigma)

    def halfnormal_prior(self, sigma: float) -> HalfNormalPrior:
        return HalfNormalPrior(sigma=sigma)

    def halfcauchy_prior(self, scale: float) -> HalfCauchyPrior:
        return HalfCauchyPrior(scale=scale)

    def gamma_prior(self, alpha: float, beta: float) -> GammaPrior:
        return GammaPrior(alpha=alpha, beta=beta)

    def invgamma_prior(self, alpha: float, beta: float) -> InvGammaPrior:
        return InvGammaPrior(alpha=alpha, beta=beta)

    def beta_prior(self, alpha: float, beta: float) -> BetaPrior:
        return BetaPrior(alpha=alpha, beta=beta)

    def lkj_prior(self, eta: float) -> LKJPrior:
        return LKJPrior(eta=eta)

    def mixture_component(self, component: object) -> object:
        return component

    def prior_component_list(self, *components: object) -> list[object]:
        return list(components)

    def numexpr_list(self, *values: float) -> list[float]:
        return list(values)

    def mixture_prior(self, components: list[object], weights: list[float]) -> MixturePrior:
        return MixturePrior(components=components, weights=weights)

    def historical_borrowing_prior(self, *args: object) -> HistoricalBorrowingPrior:
        if len(args) == 4:
            map_mean, map_sd, robust_weight, historical_refs = args
            return HistoricalBorrowingPrior(
                map_mean=map_mean,
                map_sd=map_sd,
                robust_weight=robust_weight,
                historical_refs=historical_refs,
            )
        map_mean, map_sd, historical_refs = args
        return HistoricalBorrowingPrior(
            map_mean=map_mean,
            map_sd=map_sd,
            historical_refs=historical_refs,
        )

    def prior_family(self, variant: object) -> object:
        return variant

    def string_list(self, *items: str) -> list[str]:
        return list(items)

    def prior_attr_source(self, value: str) -> tuple[str, str]:
        return ("source", value)

    def prior_attr_doi(self, value: str) -> tuple[str, str]:
        return ("doi", value)

    def prior_attr_justification(self, value: str) -> tuple[str, str]:
        return ("justification", value)

    def prior_attr_historical_refs(self, value: list[str]) -> tuple[str, list[str]]:
        return ("historical_refs", value)

    def prior_entry(
        self, target: str, family: object, *attrs: tuple[str, object]
    ) -> RawPriorEntry:
        attr_map = dict(attrs)
        return RawPriorEntry(
            target=target,
            family=family,  # type: ignore[arg-type]
            source=attr_map.get("source", "weakly_informative"),  # type: ignore[arg-type]
            justification=attr_map.get("justification", ""),  # type: ignore[arg-type]
            doi=attr_map.get("doi"),  # type: ignore[arg-type]
            historical_refs=attr_map.get("historical_refs", []),  # type: ignore[arg-type]
        )

    def priors_block(self, *entries: RawPriorEntry) -> None:
        """Stash entries on the transformer instance; contribute nothing to ``model_body``.

        Returning ``None`` (rather than the entries) keeps ``DSLSpec``
        construction in ``model_body`` free of any ``priors:``-specific
        branching — the one ``None`` produced by an optional block is simply
        skipped by the dispatch loop.
        """
        self.raw_priors = list(entries)
        return None

    # --- Covariates (P1.6) ---
    #
    # Each ``*_form`` method returns ``(form_name, field_kwargs)``; the
    # transform-independent ``covariate_entry`` method assembles the real
    # ``CovariateLink`` from that tuple. ``covariates_block`` stashes parsed
    # entries on the transformer instance and contributes nothing to
    # ``model_body`` — same pattern as ``priors_block`` (P1.5) and for the
    # same reason: a ``CovariateLink`` construction failure (mismatched
    # field shape) should surface as a plain ``pydantic.ValidationError``
    # rather than being wrapped in ``lark.exceptions.VisitError``.

    def power_form(self, theta: float, ref: float) -> tuple[str, dict[str, float | str]]:
        return ("power", {"theta": theta, "ref": ref})

    def exponential_form(self, theta: float) -> tuple[str, dict[str, float | str]]:
        return ("exponential", {"theta": theta})

    def linear_form(self, theta: float) -> tuple[str, dict[str, float | str]]:
        return ("linear", {"theta": theta})

    def categorical_form(self, reference: str) -> tuple[str, dict[str, float | str]]:
        return ("categorical", {"reference": reference})

    def maturation_form(self, tm50: float, hill: float) -> tuple[str, dict[str, float | str]]:
        return ("maturation", {"tm50": tm50, "hill": hill})

    def covariate_form_call(
        self, variant: tuple[str, dict[str, float | str]]
    ) -> tuple[str, dict[str, float | str]]:
        return variant

    def covariate_entry(
        self, param: str, covariate: str, form_call: tuple[str, dict[str, float | str]]
    ) -> CovariateLink:
        form, fields = form_call
        return CovariateLink(param=param, covariate=covariate, form=form, **fields)

    def covariates_block(self, *entries: CovariateLink) -> None:
        """Stash entries on the transformer instance; contribute nothing to ``model_body``.

        See :meth:`priors_block` for the identical rationale.
        """
        self.raw_covariates = list(entries)
        return None

    # --- Macro use (P2.1) ---

    def use_block(self, name: str) -> None:
        """Stash a dotted macro name; contributes nothing to ``model_body``.

        Same stash-not-thread pattern as :meth:`priors_block` /
        :meth:`covariates_block` / :meth:`observations_block`: expansion
        (:func:`apmode.dsl.macros.expand_macros`) needs the fully-assembled
        ``DSLSpec`` (to check what a macro like ``pkstd.standard_iiv``
        already covers) and must raise a plain
        :class:`~apmode.dsl.errors.FormularCompileError` on an unknown or
        duplicate macro name rather than a Lark-wrapped
        ``lark.exceptions.VisitError``, so it runs from
        ``apmode.dsl.grammar.compile_dsl`` after ``transform()`` returns.
        """
        self.raw_macro_uses.append(name)
        return None

    # --- Top-level ---

    def block(self, item: object) -> object:
        return item

    def model_body(self, *blocks: object) -> DSLSpec:
        """Assemble a :class:`DSLSpec` from an order-insensitive block sequence.

        Block-kind cardinality (exactly one absorption/distribution/
        elimination/observation, at most one metadata/initial, zero-or-more
        variability) is enforced pre-transform on the raw parse tree by
        ``apmode.dsl.grammar._validate_block_cardinality`` — by the time
        this method runs, dispatch-by-type below is guaranteed to see
        exactly the right shape, and the ``assert`` calls are defence in
        depth, not user-facing validation.
        """
        absorption: AbsorptionModule | None = None
        distribution: DistributionModule | None = None
        elimination: EliminationModule | None = None
        observation: ObservationModule | None = None
        metadata: Metadata | None = None
        units: UnitsDeclaration | None = None
        experimental = ExperimentalFlags()
        initial: dict[str, float] = {}
        variability: list[VariabilityItem] = []

        for item in blocks:
            if item is None:
                # ``priors_block`` (P1.5), ``covariates_block`` (P1.6),
                # ``observations_block`` (P1.7), and ``use_block`` (P2.1)
                # all produce this: their entries are collected on the
                # transformer instance, not threaded through DSLSpec
                # construction here — see ``DSLTransformer.priors_block`` /
                # ``.covariates_block`` / ``.observations_block`` /
                # ``.use_block``.
                continue
            if isinstance(item, list):
                variability.extend(item)
            elif isinstance(item, dict):
                initial.update(item)
            elif isinstance(item, Metadata):
                metadata = item
            elif isinstance(item, UnitsDeclaration):
                units = item
            elif isinstance(item, ExperimentalFlags):
                experimental = item
            elif isinstance(item, _ABSORPTION_TYPES):
                absorption = item
            elif isinstance(item, _DISTRIBUTION_TYPES):
                distribution = item
            elif isinstance(item, _ELIMINATION_TYPES):
                elimination = item
            elif isinstance(item, _OBSERVATION_TYPES):
                observation = item
            else:  # pragma: no cover — unreachable given the grammar's block alternation
                msg = f"unrecognized model block: {item!r}"
                raise TypeError(msg)

        assert absorption is not None, "missing absorption block slipped past cardinality check"
        assert distribution is not None, (
            "missing distribution block slipped past cardinality check"
        )
        assert elimination is not None, "missing elimination block slipped past cardinality check"

        if observation is None:
            # Cardinality is validated pre-transform (exactly one of
            # observation:/observations: is present -- see
            # apmode.dsl.grammar._validate_block_cardinality). When
            # observations: was used instead of the singular sugar,
            # self.raw_observations is already populated (Lark transforms
            # children bottom-up, so observations_block ran before this
            # method). Synthesize the mandatory `observation` field from
            # the first entry (insertion order) -- see DSLSpec.observations
            # docstring for why this is a safe, documented proxy for
            # pre-P1.7 consumers of spec.observation.
            assert self.raw_observations, (
                "missing observation/observations block slipped past cardinality check"
            )
            observation = next(iter(self.raw_observations.values())).error

        return DSLSpec(
            model_id=generate_candidate_id(),
            absorption=absorption,
            distribution=distribution,
            elimination=elimination,
            variability=variability,
            observation=observation,
            initial=initial,
            metadata=metadata,
            units=units,
            experimental=experimental,
        )

    def model(self, spec: DSLSpec) -> DSLSpec:
        return spec

    def start(self, spec: DSLSpec) -> DSLSpec:
        return spec
