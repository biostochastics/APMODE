# SPDX-License-Identifier: GPL-2.0-or-later
"""DSL grammar loader and compiler for PK model specifications.

Provides parse-only (parse tree) and full compilation (parse → AST) modes.
Semantic validation (dim ceilings, constraint enforcement) is a separate phase.
"""

from __future__ import annotations

from pathlib import Path

from lark import Lark, Tree

from apmode.dsl.ast_models import DSLSpec
from apmode.dsl.transformer import DSLTransformer

_GRAMMAR_PATH = Path(__file__).parent / "pk_grammar.lark"
_MAX_DSL_INPUT_CHARS = 10_000


def load_grammar() -> Lark:
    """Load and return the PK DSL Lark parser."""
    return Lark(
        _GRAMMAR_PATH.read_text(),
        parser="earley",
        start="start",
    )


def parse_dsl(text: str) -> Tree:  # type: ignore[type-arg]
    """Parse a DSL spec into a Lark tree with input size guard against DoS."""
    if len(text) > _MAX_DSL_INPUT_CHARS:
        msg = f"DSL input exceeds {_MAX_DSL_INPUT_CHARS} characters"
        raise ValueError(msg)
    parser = load_grammar()
    return parser.parse(text)


def compile_dsl(text: str) -> DSLSpec:
    """Parse and transform a DSL spec into a typed Pydantic AST.

    This is the primary entry point for the DSL compiler. Returns a fully
    typed DSLSpec with a generated model_id.

    Raises ValueError for oversized input, lark.exceptions.UnexpectedInput
    for syntax errors.
    """
    tree = parse_dsl(text)
    transformer = DSLTransformer()
    result = transformer.transform(tree)
    assert isinstance(result, DSLSpec)  # guaranteed by grammar's start rule
    return result
