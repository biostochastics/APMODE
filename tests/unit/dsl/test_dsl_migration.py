# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the pre-Phase-1 -> current-grammar text migrator (P1.11).

Formular sharpening plan §4 Phase 1 (P1.11). The compiler
(:mod:`apmode.dsl.grammar`) no longer parses the old syntax at all, so these
fixtures are hand-written pre-Phase-1 text reconstructed from the grammar
that shipped before this session (see git history of ``pk_grammar.lark``)
rather than round-tripped through a live old-grammar parser.
"""

from __future__ import annotations

import pytest
from lark.exceptions import UnexpectedInput

from apmode.dsl.ast_models import CovariateLink, FirstOrder, LinearElim, OneCmt
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.migration import migrate_v06_to_v07

_LEGACY_1CMT_ORAL = """
model {
    absorption: FirstOrder(ka=1.2)
    distribution: OneCmt(V=70.0)
    elimination: Linear(CL=5.0)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
}
"""

_LEGACY_WITH_COVARIATE = """
model {
    absorption: FirstOrder(ka=1.2)
    distribution: OneCmt(V=70.0)
    elimination: Linear(CL=5.0)
    variability: {
        IIV(params=[CL, V], structure=diagonal)
        CovariateLink(param=CL, covariate=WT, form=power)
    }
    observation: Proportional(sigma_prop=0.1)
}
"""

_LEGACY_BARE_COVARIATE_LINE = """
model {
    absorption: FirstOrder(ka=1.2)
    distribution: OneCmt(V=70.0)
    elimination: Linear(CL=5.0)
    variability: CovariateLink(param=CL, covariate=WT, form=exponential)
    observation: Proportional(sigma_prop=0.1)
}
"""

_LEGACY_CATEGORICAL_COVARIATE = """
model {
    absorption: FirstOrder(ka=1.2)
    distribution: OneCmt(V=70.0)
    elimination: Linear(CL=5.0)
    variability: {
        IIV(params=[CL, V], structure=diagonal)
        CovariateLink(param=CL, covariate=SEX, form=categorical)
    }
    observation: Proportional(sigma_prop=0.1)
}
"""

_LEGACY_TIME_VARYING_TWO_ARG = """
model {
    absorption: FirstOrder(ka=1.2)
    distribution: OneCmt(V=70.0)
    elimination: TimeVarying(CL=5.0, decay_fn=exponential)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
}
"""

_LEGACY_TIME_VARYING_THREE_ARG = """
model {
    absorption: FirstOrder(ka=1.2)
    distribution: OneCmt(V=70.0)
    elimination: TimeVarying(CL=5.0, kdecay=0.2, decay_fn=exponential)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
}
"""


class TestStructuralCalibrationMigration:
    def test_calibration_values_move_to_initial_block(self) -> None:
        result = migrate_v06_to_v07(_LEGACY_1CMT_ORAL)
        assert result.warnings == []
        assert "FirstOrder(ka)" in result.text
        assert "OneCmt(V)" in result.text
        assert "Linear(CL)" in result.text
        assert "initial:" in result.text

        spec = compile_dsl(result.text)
        assert spec.absorption == FirstOrder()
        assert spec.distribution == OneCmt()
        assert spec.elimination == LinearElim()
        assert spec.initial == {"ka": 1.2, "V": 70.0, "CL": 5.0}

    def test_time_varying_two_arg_form(self) -> None:
        result = migrate_v06_to_v07(_LEGACY_TIME_VARYING_TWO_ARG)
        assert result.warnings == []
        assert "TimeVarying(CL, decay_fn=exponential)" in result.text
        spec = compile_dsl(result.text)
        assert spec.initial == {"ka": 1.2, "V": 70.0, "CL": 5.0}
        assert spec.get_initial("kdecay", 0.1) == 0.1

    def test_time_varying_three_arg_form(self) -> None:
        result = migrate_v06_to_v07(_LEGACY_TIME_VARYING_THREE_ARG)
        assert result.warnings == []
        assert "TimeVarying(CL, decay_fn=exponential)" in result.text
        spec = compile_dsl(result.text)
        assert spec.initial["kdecay"] == 0.2
        assert spec.initial["CL"] == 5.0

    def test_idempotent_on_already_migrated_text(self) -> None:
        once = migrate_v06_to_v07(_LEGACY_1CMT_ORAL)
        twice = migrate_v06_to_v07(once.text)
        assert twice.warnings == []
        spec = compile_dsl(twice.text)
        assert spec.initial == {"ka": 1.2, "V": 70.0, "CL": 5.0}


class TestCovariateLinkMigration:
    def test_covariate_link_inside_braces_migrates_to_arrow_syntax(self) -> None:
        result = migrate_v06_to_v07(_LEGACY_WITH_COVARIATE)
        assert result.warnings == []
        assert "CovariateLink" not in result.text
        assert "CL <- WT.power(theta=0.75, ref=70)" in result.text
        # The surviving IIV item keeps the variability: block non-empty.
        assert "IIV(params=[CL, V], structure=diagonal)" in result.text

        spec = compile_dsl(result.text)
        assert spec.covariates == [
            CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
        ]

    def test_bare_covariate_link_line_is_dropped_not_left_dangling(self) -> None:
        result = migrate_v06_to_v07(_LEGACY_BARE_COVARIATE_LINE)
        assert result.warnings == []
        assert "CovariateLink" not in result.text
        # The whole `variability: CovariateLink(...)` line had no other
        # content, so it must be dropped rather than left as a dangling
        # `variability:` with nothing after it.
        assert "variability:" not in result.text
        assert "CL <- WT.exponential(theta=0.0)" in result.text

        spec = compile_dsl(result.text)
        assert spec.variability == []
        assert spec.covariates == [
            CovariateLink(param="CL", covariate="WT", form="exponential", theta=0.0)
        ]

    def test_categorical_covariate_link_flagged_not_corrupted(self) -> None:
        result = migrate_v06_to_v07(_LEGACY_CATEGORICAL_COVARIATE)

        assert len(result.warnings) == 1
        warning = result.warnings[0]
        assert "could not auto-migrate this construct near line" in warning.message
        assert "please review manually" in warning.message
        assert "categorical" in warning.message

        # The unmigrated construct is left verbatim in the output text
        # (not silently dropped, not guessed at) -- so the surrounding
        # calibration migration for the *rest* of the file still lands.
        assert "CovariateLink(param=CL, covariate=SEX, form=categorical)" in result.text
        assert "FirstOrder(ka)" in result.text
        assert "initial:" in result.text
        assert "ka = 1.2" in result.text

        # The old CovariateLink syntax is no longer parseable at all, so the
        # migrated-but-incomplete text correctly fails to compile rather
        # than silently producing a spec with the covariate link dropped.
        with pytest.raises(UnexpectedInput):
            compile_dsl(result.text)


class TestMigrationIsNoOpOnCurrentSyntax:
    def test_current_syntax_passes_through_unchanged_modulo_whitespace(self) -> None:
        current = """
        model {
            absorption: FirstOrder(ka)
            distribution: OneCmt(V)
            elimination: Linear(CL)
            variability: IIV(params=[CL, V], structure=diagonal)
            covariates: { CL <- WT.power(theta=0.75, ref=70) }
            observation: Proportional(sigma_prop=0.1)
            initial: { ka = 1.2, V = 70.0, CL = 5.0 }
        }
        """
        result = migrate_v06_to_v07(current)
        assert result.warnings == []
        spec = compile_dsl(result.text)
        assert spec.initial == {"ka": 1.2, "V": 70.0, "CL": 5.0}
        assert spec.covariates == [
            CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
        ]
