# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for DSL grammar (PRD §4.2.5) — parse-only, no lowering."""

import pytest
from lark import Lark
from lark.exceptions import UnexpectedInput

from apmode.dsl.grammar import load_grammar, parse_dsl


@pytest.fixture
def parser() -> Lark:
    return load_grammar()


class TestParseValidModels:
    """All module combinations from PRD §4.2.5 should parse."""

    def test_simplest_1cmt_oral(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_2cmt_iv_combined_error(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: ZeroOrder(dur=0.5)
            distribution: TwoCmt(V1=10.0, V2=20.0, Q=3.0)
            elimination: Linear(CL=2.0)
            variability: IIV(params=[CL, V1], structure=block)
            observation: Combined(sigma_prop=0.1, sigma_add=0.5)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_3cmt_mm_elimination(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: LaggedFirstOrder(ka=1.5, tlag=0.3)
            distribution: ThreeCmt(V1=10.0, V2=20.0, V3=5.0, Q2=3.0, Q3=1.0)
            elimination: MichaelisMenten(Vmax=100.0, Km=10.0)
            variability: IIV(params=[Vmax, Km], structure=diagonal)
            observation: Additive(sigma_add=1.0)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_transit_absorption(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: Transit(n=4, ktr=2.0, ka=1.0)
            distribution: OneCmt(V=50.0)
            elimination: Linear(CL=3.0)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.05)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_mixed_first_zero_absorption(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: MixedFirstZero(ka=1.0, dur=0.5, frac=0.6)
            distribution: TwoCmt(V1=30.0, V2=40.0, Q=5.0)
            elimination: ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0)
            variability: IIV(params=[CL, V1, Vmax], structure=block)
            observation: Combined(sigma_prop=0.1, sigma_add=0.3)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_blq_m3(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL, V, ka], structure=diagonal)
            observation: BLQ_M3(loq_value=0.1)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_iov(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.2)
            distribution: OneCmt(V=60.0)
            elimination: Linear(CL=4.0)
            variability: IOV(params=[CL], occasions=ByStudy)
            observation: Proportional(sigma_prop=0.08)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_covariate_link(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: CovariateLink(param=CL, covariate=WT, form=power)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_node_absorption(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: NODE_Absorption(dim=4, constraint_template=monotone_increasing)
            distribution: TwoCmt(V1=30.0, V2=40.0, Q=5.0)
            elimination: Linear(CL=3.0)
            variability: IIV(params=[CL, V1], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_node_elimination(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V=70.0)
            elimination: NODE_Elimination(dim=6, constraint_template=bounded_positive)
            variability: IIV(params=[V], structure=diagonal)
            observation: Combined(sigma_prop=0.1, sigma_add=0.5)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_tmdd_core(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=0.5)
            distribution: TMDD_Core(V=50.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05)
            elimination: Linear(CL=1.0)
            variability: IIV(params=[CL, R0], structure=diagonal)
            observation: Proportional(sigma_prop=0.15)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_tmdd_qss(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=0.5)
            distribution: TMDD_QSS(V=50.0, R0=10.0, KD=0.5, kint=0.05)
            elimination: Linear(CL=1.0)
            variability: IIV(params=[CL, R0], structure=diagonal)
            observation: Proportional(sigma_prop=0.15)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_occasion_by_visit(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IOV(params=[CL, ka], occasions=ByVisit(VISIT))
            observation: Proportional(sigma_prop=0.1)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_multi_variability_with_braces(self, parser: Lark) -> None:
        """Real models need IIV + IOV + covariate links simultaneously."""
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: TwoCmt(V1=30.0, V2=40.0, Q=5.0)
            elimination: Linear(CL=5.0)
            variability: {
                IIV(params=[CL, V1, ka], structure=block)
                IOV(params=[CL], occasions=ByStudy)
                CovariateLink(param=CL, covariate=WT, form=power)
                CovariateLink(param=V1, covariate=WT, form=power)
            }
            observation: Combined(sigma_prop=0.1, sigma_add=0.5)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None

    def test_time_varying_elimination(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V=70.0)
            elimination: TimeVarying(CL=5.0, decay_fn=exponential)
            variability: IIV(params=[CL, V], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        tree = parser.parse(spec)
        assert tree is not None


class TestParseInvalidModels:
    """Syntactically invalid specs should fail to parse."""

    def test_missing_module(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: FirstOrder(ka=1.0)
            distribution: OneCmt(V=70.0)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)

    def test_unknown_absorption_type(self, parser: Lark) -> None:
        spec = """
        model {
            absorption: MagicAbsorption(x=1.0)
            distribution: OneCmt(V=70.0)
            elimination: Linear(CL=5.0)
            variability: IIV(params=[CL], structure=diagonal)
            observation: Proportional(sigma_prop=0.1)
        }
        """
        with pytest.raises(UnexpectedInput):
            parser.parse(spec)

    def test_empty_model(self, parser: Lark) -> None:
        with pytest.raises(UnexpectedInput):
            parser.parse("model {}")

    def test_garbage(self, parser: Lark) -> None:
        with pytest.raises(UnexpectedInput):
            parser.parse("not a model at all")


class TestParseDSLSizeGuard:
    def test_oversized_input_rejected(self) -> None:
        huge = "model {" + " " * 20_000 + "}"
        with pytest.raises(ValueError, match="exceeds"):
            parse_dsl(huge)
