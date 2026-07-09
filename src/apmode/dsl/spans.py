# SPDX-License-Identifier: GPL-2.0-or-later
"""Source-position value object for DSL diagnostics.

``SourceSpan`` carries the location of a parsed DSL construct so validation
errors and other diagnostics can be rendered with actionable positions
(``file.pk:L:C``). The Lark grammar's ``source_meta`` sidecar
(``apmode.dsl.grammar._collect_source_meta``) is block-level and
single-point today — it records where a block *starts*, not a true
start/end range — so :meth:`SourceSpan.from_point` collapses ``line_end``/
``col_end`` onto the start position. Extending ``source_meta`` to carry
real end positions (Lark exposes ``end_line``/``end_column`` on ``meta``
when ``propagate_positions=True``) is future work, not part of this
retrofit.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SourceSpan(BaseModel):
    """A 1-indexed (line, column) source location span, Lark convention."""

    model_config = ConfigDict(frozen=True)

    line_start: int
    col_start: int
    line_end: int
    col_end: int
    text_excerpt: str | None = None

    @classmethod
    def from_point(cls, line: int, column: int, *, text_excerpt: str | None = None) -> SourceSpan:
        """Build a zero-width span from a single (line, column) anchor point.

        Used when only a block-level anchor is available (e.g. the
        ``DSLSpec.source_meta`` sidecar), not a true start/end range.
        """
        return cls(
            line_start=line,
            col_start=column,
            line_end=line,
            col_end=column,
            text_excerpt=text_excerpt,
        )
