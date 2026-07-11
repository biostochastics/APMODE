# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for SourceSpan retrofit on DSL validation errors (P0.1).

Verifies that ``validate_dsl`` errors carry a plausible ``source_span``
when the spec was produced by ``compile_dsl`` (parse-tree provenance via
``DSLSpec.source_meta``), and that ``source_span`` stays ``None`` for
specs built programmatically (no parse tree, hence an empty
``source_meta`` sidecar) — preserving backward compatibility for callers
that construct ``DSLSpec`` directly.
"""

from apmode.backends.protocol import Lane
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.spans import SourceSpan
from apmode.dsl.validator import ValidationError, validate_dsl


class TestSourceSpanShape:
    """SourceSpan is a minimal frozen value object."""

    def test_fields_and_frozen(self) -> None:
        span = SourceSpan(line_start=3, col_start=13, line_end=3, col_end=13)
        assert span.line_start == 3
        assert span.col_start == 13
        assert span.line_end == 3
        assert span.col_end == 13
        assert span.text_excerpt is None

    def test_from_point_collapses_to_zero_width(self) -> None:
        span = SourceSpan.from_point(7, 21)
        assert span.line_start == span.line_end == 7
        assert span.col_start == span.col_end == 21

    def test_frozen_rejects_mutation(self) -> None:
        span = SourceSpan.from_point(1, 1)
        try:
            span.line_start = 99  # type: ignore[misc]
        except Exception:
            pass
        else:
            raise AssertionError("SourceSpan should be frozen")


class TestValidationErrorCarriesSourceSpan:
    """Errors raised against a compile_dsl()-produced spec carry a span."""

    def test_negative_ka_has_source_span_on_absorption_line(self) -> None:
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                variability: IIV(params=[CL, V], structure=diagonal)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = -1.0, V = 70.0, CL = 5.0 }
            }
            """
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        err = errors[0]
        assert err.constraint == "positive"
        assert err.source_span is not None
        assert isinstance(err.source_span, SourceSpan)
        # Absorption block is on line 3 of the DSL text above.
        assert err.source_span.line_start == 3
        assert err.source_span.line_start == err.source_span.line_end
        assert err.source_span.col_start == err.source_span.col_end

    def test_multiple_module_errors_each_carry_distinct_spans(self) -> None:
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                variability: IIV(params=[CL, V], structure=diagonal)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = -1.0, V = -70.0, CL = 5.0 }
            }
            """
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        by_module = {e.module: e for e in errors}
        assert "absorption" in by_module
        assert "distribution" in by_module
        assert by_module["absorption"].source_span is not None
        assert by_module["distribution"].source_span is not None
        assert (
            by_module["absorption"].source_span.line_start
            != by_module["distribution"].source_span.line_start
        )

    def test_variability_item_error_carries_indexed_span(self) -> None:
        spec = compile_dsl(
            """
            model {
                absorption: FirstOrder(ka)
                distribution: OneCmt(V)
                elimination: Linear(CL)
                variability: IIV(params=[nonexistent], structure=diagonal)
                observation: Proportional(sigma_prop=0.1)
                initial: { ka = 1.0, V = 70.0, CL = 5.0 }
            }
            """
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        matches = [e for e in errors if e.constraint == "iiv_param_exists"]
        assert len(matches) == 1
        assert matches[0].source_span is not None
        assert matches[0].source_span.line_start == 6


class TestValidationErrorSourceSpanBackwardCompatible:
    """Programmatically-built specs have no parse-tree provenance."""

    def _make_spec(self) -> DSLSpec:
        return DSLSpec(
            model_id="test_id_000000000000",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": -1.0, "V": 70.0, "CL": 5.0},
        )

    def test_source_span_defaults_to_none(self) -> None:
        spec = self._make_spec()
        assert spec.source_meta == {}
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert len(errors) == 1
        assert errors[0].source_span is None

    def test_validation_error_constructs_without_source_span_kwarg(self) -> None:
        err = ValidationError(
            module="absorption",
            param="absorption.ka",
            constraint="positive",
            message="ka must be > 0, got -1.0",
            code="FRM-SEM-001",
        )
        assert err.source_span is None
