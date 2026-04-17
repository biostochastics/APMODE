# SPDX-License-Identifier: GPL-2.0-or-later
"""DSL grammar loader and compiler for PK model specifications.

Provides parse-only (parse tree) and full compilation (parse → AST) modes.
Semantic validation (dim ceilings, constraint enforcement) is a separate phase.
"""

from __future__ import annotations

import functools
from pathlib import Path

from lark import Lark, Tree

from apmode.dsl.ast_models import DSLSpec
from apmode.dsl.transformer import DSLTransformer

_GRAMMAR_PATH = Path(__file__).parent / "pk_grammar.lark"
_MAX_DSL_INPUT_CHARS = 10_000


@functools.lru_cache(maxsize=1)
def load_grammar() -> Lark:
    """Load and return the Formular Lark parser (cached after first call)."""
    return Lark(
        _GRAMMAR_PATH.read_text(),
        parser="earley",
        start="start",
        propagate_positions=True,
    )


def parse_dsl(text: str) -> Tree:  # type: ignore[type-arg]
    """Parse a DSL spec into a Lark tree with input size guard against DoS."""
    if len(text) > _MAX_DSL_INPUT_CHARS:
        msg = f"DSL input exceeds {_MAX_DSL_INPUT_CHARS} characters"
        raise ValueError(msg)
    parser = load_grammar()
    return parser.parse(text)


# #17: AST nodes themselves are frozen Pydantic models, so we cannot
# stash per-node line/column on them. Instead, post-transform we walk
# the tree once and build a sidecar map keyed by AST role (absorption /
# distribution / elimination / observation / variability[i]). The
# validator uses this to decorate error messages with source positions
# and the agentic trace carries it for audit playback.
_ROLE_TO_RULE = {
    "absorption": "absorption",
    "distribution": "distribution",
    "elimination": "elimination",
    "observation": "observation",
}


def _collect_source_meta(tree: Tree) -> dict[str, tuple[int, int]]:  # type: ignore[type-arg]
    """Walk a parse tree and collect (line, column) for known top-level roles.

    Variability items are indexed in source order as
    ``variability[0]``, ``variability[1]``…
    """
    out: dict[str, tuple[int, int]] = {}
    var_idx = 0
    for sub in tree.iter_subtrees_topdown():  # type: ignore[no-untyped-call]
        rule = sub.data
        meta = getattr(sub, "meta", None)
        if meta is None or getattr(meta, "empty", True):
            continue
        line = int(meta.line)
        column = int(meta.column)
        if rule in _ROLE_TO_RULE and _ROLE_TO_RULE[rule] not in out:
            out[_ROLE_TO_RULE[rule]] = (line, column)
        elif rule in ("iiv", "iov", "covariate_link"):
            out[f"variability[{var_idx}]"] = (line, column)
            var_idx += 1
    return out


def compile_dsl(text: str) -> DSLSpec:
    """Parse and transform a DSL spec into a typed Pydantic AST.

    This is the primary entry point for the DSL compiler. Returns a fully
    typed DSLSpec with a generated model_id. When the grammar emits
    positional metadata (``propagate_positions=True``) this function
    attaches a ``source_meta`` sidecar so the validator can annotate
    errors with line/column information.

    Raises ValueError for oversized input, lark.exceptions.UnexpectedInput
    for syntax errors.
    """
    tree = parse_dsl(text)
    transformer = DSLTransformer()
    result = transformer.transform(tree)
    assert isinstance(result, DSLSpec)  # guaranteed by grammar's start rule
    meta = _collect_source_meta(tree)
    if meta:
        # DSLSpec is frozen — rebuild with the sidecar populated. Fields
        # that already exist on the result carry through via model_copy.
        result = result.model_copy(update={"source_meta": meta})
    return result
