# SPDX-License-Identifier: GPL-2.0-or-later
"""Lark Transformer: parse tree → Pydantic AST (ARCHITECTURE.md §2.2).

Converts a Lark parse tree (from pk_grammar.lark) into typed Pydantic AST models.
Each grammar rule maps to a transformer method that returns the corresponding
AST node.
"""

from __future__ import annotations

from lark import Transformer, v_args

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    Additive,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    OccasionByDoseEpoch,
    OccasionByStudy,
    OccasionByVisit,
    OccasionCustom,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.ids import generate_candidate_id


@v_args(inline=True)
class DSLTransformer(Transformer):  # type: ignore[type-arg]
    """Transform Lark parse tree into Pydantic AST models.

    Each method name matches a grammar rule. Terminal values arrive as
    ``lark.Token`` (subclass of str). Numeric terminals are converted
    to Python types by the terminal methods below.
    """

    # --- Terminals ---

    def NUMBER(self, token: str) -> float:
        return float(token)

    def INT(self, token: str) -> int:
        return int(token)

    def NAME(self, token: str) -> str:
        return str(token)

    def STRUCTURE(self, token: str) -> str:
        return str(token)

    def CONSTRAINT_TEMPLATE(self, token: str) -> str:
        return str(token)

    def COVARIATE_FORM(self, token: str) -> str:
        return str(token)

    def DECAY_FN(self, token: str) -> str:
        return str(token)

    def ERROR_MODEL(self, token: str) -> str:
        return str(token)

    # --- Absorption ---

    def first_order(self, ka: float) -> FirstOrder:
        return FirstOrder(ka=ka)

    def zero_order(self, dur: float) -> ZeroOrder:
        return ZeroOrder(dur=dur)

    def lagged_first_order(self, ka: float, tlag: float) -> LaggedFirstOrder:
        return LaggedFirstOrder(ka=ka, tlag=tlag)

    def transit(self, n: int, ktr: float, ka: float) -> Transit:
        return Transit(n=n, ktr=ktr, ka=ka)

    def mixed_first_zero(self, ka: float, dur: float, frac: float) -> MixedFirstZero:
        return MixedFirstZero(ka=ka, dur=dur, frac=frac)

    def node_absorption(self, dim: int, ct: str) -> NODEAbsorption:
        return NODEAbsorption(dim=dim, constraint_template=ct)

    def absorption_type(self, variant: object) -> object:
        return variant

    def absorption(self, variant: object) -> object:
        return variant

    # --- Distribution ---

    def one_cmt(self, v: float) -> OneCmt:
        return OneCmt(V=v)

    def two_cmt(self, v1: float, v2: float, q: float) -> TwoCmt:
        return TwoCmt(V1=v1, V2=v2, Q=q)

    def three_cmt(self, v1: float, v2: float, v3: float, q2: float, q3: float) -> ThreeCmt:
        return ThreeCmt(V1=v1, V2=v2, V3=v3, Q2=q2, Q3=q3)

    def tmdd_core(self, v: float, r0: float, kon: float, koff: float, kint: float) -> TMDDCore:
        return TMDDCore(V=v, R0=r0, kon=kon, koff=koff, kint=kint)

    def tmdd_qss(self, v: float, r0: float, kd: float, kint: float) -> TMDDQSS:
        return TMDDQSS(V=v, R0=r0, KD=kd, kint=kint)

    def distribution_type(self, variant: object) -> object:
        return variant

    def distribution(self, variant: object) -> object:
        return variant

    # --- Elimination ---

    def linear_elim(self, cl: float) -> LinearElim:
        return LinearElim(CL=cl)

    def michaelis_menten(self, vmax: float, km: float) -> MichaelisMenten:
        return MichaelisMenten(Vmax=vmax, Km=km)

    def parallel_linear_mm(self, cl: float, vmax: float, km: float) -> ParallelLinearMM:
        return ParallelLinearMM(CL=cl, Vmax=vmax, Km=km)

    def time_varying_elim(self, *args: object) -> TimeVaryingElim:
        if len(args) == 3:
            cl, kdecay, decay_fn = args
            return TimeVaryingElim(CL=cl, kdecay=kdecay, decay_fn=decay_fn)
        cl, decay_fn = args
        return TimeVaryingElim(CL=cl, decay_fn=decay_fn)

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

    def covariate_link(self, param: str, covariate: str, form: str) -> CovariateLink:
        return CovariateLink(param=param, covariate=covariate, form=form)

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

    # --- Top-level ---

    def model(
        self,
        absorption: object,
        distribution: object,
        elimination: object,
        variability: object,
        observation: object,
    ) -> DSLSpec:
        # variability can be a single item or a list from variability_block
        var_list = variability if isinstance(variability, list) else [variability]

        return DSLSpec(
            model_id=generate_candidate_id(),
            absorption=absorption,
            distribution=distribution,
            elimination=elimination,
            variability=var_list,
            observation=observation,
        )

    def start(self, spec: DSLSpec) -> DSLSpec:
        return spec
