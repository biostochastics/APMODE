# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for DSL grammar (PRD §4.2.5) — parse-only, no lowering.

Formular sharpening plan §4 Phase 1 (P1.1): top-level blocks may appear in
any order and a spec is separated into structural (module) declarations,
initial-estimate values, and optional metadata.
"""

import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput

from apmode.dsl.ast_models import Combined, Proportional
from apmode.dsl.errors import FormularCompileError, FrmCode
from apmode.dsl.grammar import compile_dsl, load_grammar, parse_dsl


@pytest.fixture
def parser() -> Lark:
    return load_grammar()


class TestParseValidModels:
    """All module combinations from PRD §4.2.5 should parse."""

    def test_simplest_1cmt_oral(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_2cmt_iv_combined_error(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: ZeroOrder(dur)
            distribution: TwoCmt(V1, V2, Q)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V1], structure=block)
            observation: Combined(sigma_prop=0.1, sigma_add=0.5)
            initial: { dur = 0.5, V1 = 10.0, V2 = 20.0, Q = 3.0, CL = 2.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_3cmt_mm_elimination(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: LaggedFirstOrder(ka, tlag)
            distribution: ThreeCmt(V1, V2, V3, Q2, Q3)
            elimination: MichaelisMenten(Vmax, Km)
            variability: IIV(params=[Vmax, Km], structure=diagonal)
            observation: Additive(sigma_add=1.0)
            initial: {
                ka = 1.5, tlag = 0.3,
                V1 = 10.0, V2 = 20.0, V3 = 5.0, Q2 = 3.0, Q3 = 1.0,
                Vmax = 100.0, Km = 10.0
            }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_transit_absorption(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: Transit(n=4, ktr, ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.05)
            initial: { ktr = 2.0, ka = 1.0, V = 50.0, CL = 3.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_mixed_first_zero_absorption(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: MixedFirstZero(ka, dur, frac)
            distribution: TwoCmt(V1, V2, Q)
            elimination: ParallelLinearMM(CL, Vmax, Km)
            variability: IIV(params=[CL, V1, Vmax], structure=block)
            observation: Combined(sigma_prop=0.1, sigma_add=0.3)
            initial: {
                ka = 1.0, dur = 0.5, frac = 0.6,
                V1 = 30.0, V2 = 40.0, Q = 5.0,
                CL = 2.0, Vmax = 50.0, Km = 5.0
            }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_blq_m3(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V, ka], structure=diagonal)
            observation: BLQ_M3(loq_value=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_iov(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IOV(params=[CL], occasions=ByStudy)
            observation: Proportional(sigma_prop=0.08)
            initial: { ka = 1.2, V = 60.0, CL = 4.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_covariate_link(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            covariates: { CL <- WT.power(theta=0.75, ref=70) }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_node_absorption(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: NODE_Absorption(dim=4, constraint_template=monotone_increasing)
            distribution: TwoCmt(V1, V2, Q)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V1], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { V1 = 30.0, V2 = 40.0, Q = 5.0, CL = 3.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_node_elimination(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: NODE_Elimination(dim=6, constraint_template=bounded_positive)
            variability: IIV(params=[V], structure=diagonal)
            observation: Combined(sigma_prop=0.1, sigma_add=0.5)
            initial: { ka = 1.0, V = 70.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_tmdd_core(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: TMDD_Core(V, R0, kon, koff, kint)
            elimination: Linear(CL)
            variability: IIV(params=[CL, R0], structure=diagonal)
            observation: Proportional(sigma_prop=0.15)
            initial: {
                ka = 0.5, V = 50.0, R0 = 10.0, kon = 0.1, koff = 0.01,
                kint = 0.05, CL = 1.0
            }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_tmdd_qss(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: TMDD_QSS(V, R0, KD, kint)
            elimination: Linear(CL)
            variability: IIV(params=[CL, R0], structure=diagonal)
            observation: Proportional(sigma_prop=0.15)
            initial: { ka = 0.5, V = 50.0, R0 = 10.0, KD = 0.5, kint = 0.05, CL = 1.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_occasion_by_visit(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IOV(params=[CL, ka], occasions=ByVisit(VISIT))
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_multi_variability_with_braces(self, parser: Lark) -> None:
        """Real models need IIV + IOV + covariate links simultaneously."""
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: TwoCmt(V1, V2, Q)
            elimination: Linear(CL)
            variability: {
                IIV(params=[CL, V1, ka], structure=block)
                IOV(params=[CL], occasions=ByStudy)
            }
            observation: Combined(sigma_prop=0.1, sigma_add=0.5)
            initial: { ka = 1.0, V1 = 30.0, V2 = 40.0, Q = 5.0, CL = 5.0 }
            covariates: {
                CL <- WT.power(theta=0.75, ref=70),
                V1 <- WT.power(theta=0.75, ref=70)
            }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_time_varying_elimination(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: TimeVarying(CL, decay_fn=exponential)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_time_varying_with_kdecay_in_initial(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: TimeVarying(CL, decay_fn=exponential)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0, kdecay = 0.05 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_blq_m3_with_error_model(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: BLQ_M3(loq_value=0.1, error_model=combined, sigma_prop=0.2, sigma_add=0.5)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_blq_m4_with_error_model(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: BLQ_M4(loq_value=0.5, error_model=additive, sigma_prop=0.1, sigma_add=1.0)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_metadata_block(self, parser: Lark) -> None:
        spec = """
        model {
            metadata: { title = "Simple oral model", analyte = "drugX" }
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_metadata_fields_populate_correctly(self) -> None:
        spec = compile_dsl(
            """
            model {
                metadata: {
                    title = "Simple oral model",
                    intent = "Recover CL/V",
                    context_of_use = "exploratory",
                    analyte = "drugX",
                    version = "v1"
                }
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                variability: IIV(params=[CL, V], structure=diagonal)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            }
            """
        )
        assert spec.metadata is not None
        assert spec.metadata.title == "Simple oral model"
        assert spec.metadata.intent == "Recover CL/V"
        assert spec.metadata.context_of_use == "exploratory"
        assert spec.metadata.analyte == "drugX"
        assert spec.metadata.version == "v1"

    def test_metadata_defaults_to_none_when_absent(self) -> None:
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            }
            """
        )
        assert spec.metadata is None

    def test_metadata_partial_fields_default_to_none(self) -> None:
        spec = compile_dsl(
            """
            model {
                metadata: { title = "Only a title" }
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            }
            """
        )
        assert spec.metadata is not None
        assert spec.metadata.title == "Only a title"
        assert spec.metadata.intent is None
        assert spec.metadata.analyte is None


class TestBlockOrderInsensitivity:
    """Top-level blocks may appear in any order (Formular sharpening plan §4 Phase 1, P1.1)."""

    _CANONICAL = """
    model {
        absorption: FirstOrder(ka)
        distribution: OneCmt(V)
        elimination: Linear(CL)
        variability: IIV(params=[CL, V, ka], structure=diagonal)
        observation: Proportional(sigma_prop=0.1)
        initial: { ka = 1.5, V = 70.0, CL = 5.0 }
    }
    """

    _REVERSED = """
    model {
        initial: { ka = 1.5, V = 70.0, CL = 5.0 }
        observation: Proportional(sigma_prop=0.1)
        variability: IIV(params=[CL, V, ka], structure=diagonal)
        elimination: Linear(CL)
        distribution: OneCmt(V)
        absorption: FirstOrder(ka)
    }
    """

    _SHUFFLED = """
    model {
        elimination: Linear(CL)
        initial: { ka = 1.5, V = 70.0, CL = 5.0 }
        absorption: FirstOrder(ka)
        observation: Proportional(sigma_prop=0.1)
        distribution: OneCmt(V)
        variability: IIV(params=[CL, V, ka], structure=diagonal)
    }
    """

    def test_reversed_order_parses(self, parser: Lark) -> None:
        tree = parser.parse(self._REVERSED)
        assert tree is not None

    def test_reversed_order_compiles_to_equal_ast(self) -> None:
        canonical = compile_dsl(self._CANONICAL)
        reversed_spec = compile_dsl(self._REVERSED)

        assert canonical.absorption == reversed_spec.absorption
        assert canonical.distribution == reversed_spec.distribution
        assert canonical.elimination == reversed_spec.elimination
        assert canonical.observation == reversed_spec.observation
        assert canonical.variability == reversed_spec.variability
        assert canonical.initial == reversed_spec.initial

    def test_shuffled_order_compiles_to_equal_ast(self) -> None:
        canonical = compile_dsl(self._CANONICAL)
        shuffled_spec = compile_dsl(self._SHUFFLED)

        assert canonical.absorption == shuffled_spec.absorption
        assert canonical.distribution == shuffled_spec.distribution
        assert canonical.elimination == shuffled_spec.elimination
        assert canonical.observation == shuffled_spec.observation
        assert canonical.variability == shuffled_spec.variability
        assert canonical.initial == shuffled_spec.initial

    def test_source_meta_maps_to_true_text_position_regardless_of_order(self) -> None:
        """source_meta must track each block's true line, not a canonicalized order."""
        spec = compile_dsl(self._SHUFFLED)
        # In _SHUFFLED: elimination is line 3, absorption is line 5,
        # distribution is line 7, observation is line 6 (1-indexed within
        # the triple-quoted string, blank first line counts as line 1).
        assert spec.source_meta["elimination"][0] < spec.source_meta["absorption"][0]
        assert spec.source_meta["absorption"][0] < spec.source_meta["distribution"][0]


class TestBlockCardinality:
    """compile_dsl enforces block cardinality on the raw parse tree (P1.1)."""

    def test_missing_required_block_raises_formular_compile_error(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_MISSING_REQUIRED_BLOCK
        assert "elimination" in exc_info.value.message

    def test_duplicate_block_raises_formular_compile_error(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            absorption: IVBolus()
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK
        assert "absorption" in exc_info.value.message

    def test_duplicate_metadata_block_raises_formular_compile_error(self) -> None:
        spec = """
        model {
            metadata: { title = "A" }
            metadata: { title = "B" }
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK

    def test_duplicate_initial_block_raises_formular_compile_error(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            initial: { ka = 2.0, V = 71.0, CL = 6.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK

    def test_empty_model_raises_formular_compile_error(self, parser: Lark) -> None:
        """``model {}`` parses at the grammar level (block* admits zero blocks)
        but fails cardinality: every required block is absent."""
        tree = parser.parse("model {}")
        assert tree is not None
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl("model {}")
        assert exc_info.value.code == FrmCode.AST_MISSING_REQUIRED_BLOCK

    def test_zero_or_more_variability_blocks_permitted(self) -> None:
        """No variability: block at all is legal (zero-or-more cardinality)."""
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        compiled = compile_dsl(spec)
        assert compiled.variability == []


class TestOldInlineValueSyntaxRejected:
    """The pre-Phase-1 inline-value syntax (e.g. FirstOrder(ka=1.0)) is fully removed."""

    def test_inline_value_on_absorption_rejected(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)

    def test_inline_value_on_distribution_rejected(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, CL = 5.0 }
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)

    def test_inline_value_on_elimination_rejected(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL=5.0)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0 }
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)


class TestCovariatesBlock:
    """Top-level covariates: block with arrow syntax (Formular sharpening plan §4 P1.6)."""

    def _wrap(self, covariates_text: str) -> str:
        return f"""
        model {{
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: {{ ka = 1.0, V = 70.0, CL = 5.0 }}
            covariates: {{ {covariates_text} }}
        }}
        """

    def test_power_form_parses_with_theta_and_ref(self) -> None:
        spec = compile_dsl(self._wrap("CL <- WT.power(theta=0.75, ref=70)"))
        assert len(spec.covariates) == 1
        link = spec.covariates[0]
        assert (link.param, link.covariate, link.form) == ("CL", "WT", "power")
        assert link.theta == 0.75
        assert link.ref == 70.0
        assert link.reference is None
        assert link.tm50 is None
        assert link.hill is None

    def test_categorical_form_parses_with_reference(self) -> None:
        spec = compile_dsl(self._wrap('CL <- SEX.categorical(reference="M")'))
        assert len(spec.covariates) == 1
        link = spec.covariates[0]
        assert (link.param, link.covariate, link.form) == ("CL", "SEX", "categorical")
        assert link.reference == "M"
        assert link.theta is None
        assert link.ref is None

    def test_maturation_form_parses_with_tm50_and_hill(self) -> None:
        spec = compile_dsl(self._wrap("CL <- PMA.maturation(tm50=45, hill=3)"))
        assert len(spec.covariates) == 1
        link = spec.covariates[0]
        assert (link.param, link.covariate, link.form) == ("CL", "PMA", "maturation")
        assert link.tm50 == 45.0
        assert link.hill == 3.0
        assert link.theta is None

    def test_exponential_form_parses_with_theta(self) -> None:
        spec = compile_dsl(self._wrap("CL <- CRCL.exponential(theta=0.02)"))
        link = spec.covariates[0]
        assert link.form == "exponential"
        assert link.theta == 0.02

    def test_linear_form_parses_with_theta(self) -> None:
        spec = compile_dsl(self._wrap("CL <- AGE.linear(theta=0.01)"))
        link = spec.covariates[0]
        assert link.form == "linear"
        assert link.theta == 0.01

    def test_multiple_covariate_entries(self) -> None:
        spec = compile_dsl(
            self._wrap('CL <- WT.power(theta=0.75, ref=70), CL <- SEX.categorical(reference="M")')
        )
        assert len(spec.covariates) == 2

    def test_empty_covariates_block_permitted(self) -> None:
        spec = compile_dsl(self._wrap(""))
        assert spec.covariates == []

    def test_no_covariates_block_defaults_empty(self) -> None:
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                variability: IIV(params=[CL], structure=diagonal)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            }
            """
        )
        assert spec.covariates == []

    def test_duplicate_covariates_block_raises_formular_compile_error(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            covariates: { CL <- WT.power(theta=0.75, ref=70) }
            covariates: { CL <- WT.exponential(theta=0.5) }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK

    def test_old_compact_covariate_link_syntax_rejected(self, parser: Lark) -> None:
        """The pre-P1.6 ``CovariateLink(param=..., covariate=..., form=...)``
        function-call syntax embedded in variability: is fully removed —
        it must fail to parse, not silently misparse into something else."""
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: CovariateLink(param=CL, covariate=WT, form=power)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)

    def test_old_compact_covariate_link_syntax_rejected_in_braces(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: {
                IIV(params=[CL], structure=diagonal)
                CovariateLink(param=CL, covariate=WT, form=power)
            }
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)


class TestObservationsBlock:
    """Multi-analyte top-level observations: block (Formular sharpening plan §4 P1.7).

    Additive to the singular ``observation:`` sugar -- not a replacement.
    """

    def test_legacy_single_endpoint_accessor_unchanged(self) -> None:
        """A spec using only `observation:` normalizes to one 'default' endpoint."""
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            }
            """
        )
        assert spec.observations is None
        endpoints = spec.observation_endpoints()
        assert len(endpoints) == 1
        endpoint = endpoints[0]
        assert endpoint.name == "default"
        assert endpoint.dvid == 1
        assert endpoint.prediction == "C_central"
        assert endpoint.error == spec.observation

    def test_multi_analyte_block_parses_two_endpoints(self) -> None:
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: TMDD_QSS(V, R0, KD, kint)
                elimination: Linear(CL)
                observations: {
                    plasma: {
                        dvid=1, prediction=C_central,
                        error=Combined(sigma_prop=0.12, sigma_add=0.2)
                    },
                    target: {
                        dvid=2, prediction=C_target_total,
                        error=Proportional(sigma_prop=0.2)
                    }
                }
                initial: { ka = 0.5, V = 50.0, R0 = 10.0, KD = 0.5, kint = 0.05, CL = 1.0 }
            }
            """
        )
        assert spec.observations is not None
        assert set(spec.observations) == {"plasma", "target"}

        endpoints = spec.observation_endpoints()
        assert len(endpoints) == 2
        by_name = {ep.name: ep for ep in endpoints}
        assert by_name["plasma"].dvid == 1
        assert by_name["plasma"].prediction == "C_central"
        assert by_name["plasma"].error == Combined(sigma_prop=0.12, sigma_add=0.2)
        assert by_name["target"].dvid == 2
        assert by_name["target"].prediction == "C_target_total"
        assert by_name["target"].error == Proportional(sigma_prop=0.2)

        # Back-compat proxy: `observation` (singular) mirrors the first
        # entry in declaration order, for pre-P1.7 consumers.
        assert spec.observation == by_name["plasma"].error

    def test_missing_observation_group_raises(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_MISSING_REQUIRED_BLOCK
        assert "observation" in exc_info.value.message

    def test_both_observation_and_observations_raises(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            observations: {
                plasma: { dvid=1, prediction=C_central, error=Additive(sigma_add=0.2) }
            }
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK

    def test_duplicate_observations_block_raises(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observations: {
                plasma: { dvid=1, prediction=C_central, error=Proportional(sigma_prop=0.1) }
            }
            observations: {
                plasma: { dvid=1, prediction=C_central, error=Additive(sigma_add=0.2) }
            }
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK

    def test_duplicate_observation_endpoint_name_raises(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observations: {
                plasma: { dvid=1, prediction=C_central, error=Proportional(sigma_prop=0.1) },
                plasma: { dvid=2, prediction=C_central, error=Additive(sigma_add=0.2) }
            }
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK
        assert "duplicate entry 'plasma'" in exc_info.value.message

    def test_duplicate_initial_entry_raises(self) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, ka = 2.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK
        assert "duplicate entry 'ka'" in exc_info.value.message

    def test_duplicate_metadata_field_raises(self) -> None:
        spec = """
        model {
            metadata: { title = "first", title = "second" }
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.0, V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(FormularCompileError) as exc_info:
            compile_dsl(spec)
        assert exc_info.value.code == FrmCode.AST_DUPLICATE_BLOCK
        assert "duplicate field 'title'" in exc_info.value.message


class TestParseInvalidModels:
    """Syntactically invalid specs should fail to parse."""

    def test_unknown_absorption_type(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: MagicAbsorption(x=1.0)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
            initial: { V = 70.0, CL = 5.0 }
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)

    def test_garbage(self, parser: Lark) -> None:
        with pytest.raises(UnexpectedInput):
            parser.parse("not a model at all")


class TestParseDSLSizeGuard:
    def test_oversized_input_rejected(self) -> None:
        huge = "model {" + " " * 20_000 + "}"
        with pytest.raises(ValueError, match="exceeds"):
            parse_dsl(huge)
