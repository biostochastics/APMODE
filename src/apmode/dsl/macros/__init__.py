# SPDX-License-Identifier: GPL-2.0-or-later
"""Vetted standard-library macro registry for Formular `use <name>` statements.

Formular sharpening plan §4 Phase 2 (P2.1 — "safe macros"). A ``use
pkstd.standard_iiv`` top-level statement expands, at compile time, into
plain AST nodes — never via token-string substitution. Expansion is a pure
function operating on the already-built :class:`~apmode.dsl.ast_models.DSLSpec`
(:func:`expand_macros`), invoked from
:func:`apmode.dsl.grammar.compile_dsl` *after* the main Lark transform pass
and after every other ``raw_*``-block folding step, so a macro sees the
fully-assembled spec (including anything the author declared by hand).

No user-defined macros exist in this phase — :data:`MACRO_REGISTRY` is a
closed, small, vetted set (see :mod:`apmode.dsl.macros.stdlib`), not an
extension point a spec author can add to. This keeps the trust boundary
identical to every other DSL construct: a macro's expansion is fully
determined by code that ships with APMODE, never by spec text itself.

Why expansion lives outside any Lark ``Transformer`` callback
----------------------------------------------------------------
Exactly the same rationale as ``priors_block`` (see
:mod:`apmode.dsl.transformer`): Lark wraps any exception a ``Transformer``
callback raises in ``lark.exceptions.VisitError``, which would bury an
unknown-macro or duplicate-use :class:`~apmode.dsl.errors.FormularCompileError`
behind an extra unwrap every caller would need to perform. ``use_block``
merely stashes the dotted macro name in source order
(``DSLTransformer.raw_macro_uses``); :func:`expand_macros` is called
directly from ``compile_dsl``.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from apmode.dsl.ast_models import DSLSpec
from apmode.dsl.errors import FormularCompileError, FrmCode


class MacroDef(BaseModel):
    """Metadata describing one registered standard-library macro.

    ``name`` is the dotted identifier a ``use <name>`` statement references
    (e.g. ``"pkstd.standard_iiv"``); ``version`` is a plain opaque string
    (this phase uses ``"v1"`` for every macro) appended to ``name`` in
    :attr:`~apmode.dsl.ast_models.DSLSpec.macros_used` entries
    (``"{name}@{version}"``) so the audit trail records exactly which
    version of a macro's expansion function ran, in case a future release
    changes what ``pkstd.standard_iiv`` does.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str


MacroExpander = Callable[[DSLSpec], DSLSpec]
"""A pure ``DSLSpec -> DSLSpec`` macro expansion function.

Deliberately not a Pydantic field anywhere (kept out of
:data:`MACRO_REGISTRY`'s serialized shape were it ever dumped) — the
registry itself is a plain ``dict``, never a model.
"""

MACRO_REGISTRY: dict[str, tuple[MacroDef, MacroExpander]] = {}
"""Dotted macro name -> (metadata, expansion function). Populated at import
time by :mod:`apmode.dsl.macros.stdlib` (imported at the bottom of this
module for its registration side effect) — this is the closed,
vetted set; there is no public API to add to it from spec text."""


def register_macro(macro_def: MacroDef, expander: MacroExpander) -> None:
    """Register one macro under ``macro_def.name``. Internal to this package.

    Not part of the public DSL surface — called only by
    :mod:`apmode.dsl.macros.stdlib` at import time to populate
    :data:`MACRO_REGISTRY`.
    """
    MACRO_REGISTRY[macro_def.name] = (macro_def, expander)


def expand_macros(spec: DSLSpec, uses: list[str]) -> DSLSpec:
    """Expand every ``use <name>`` statement in ``uses`` (source order).

    For each dotted name, in the order the ``use`` statements appeared in
    the source text:

    1. Reject an unknown macro name (:attr:`~apmode.dsl.errors.FrmCode.AST_MACRO_UNKNOWN`)
       — only names present in :data:`MACRO_REGISTRY` are valid.
    2. Reject a duplicate ``use`` of the *same* macro name within one spec
       (:attr:`~apmode.dsl.errors.FrmCode.AST_MACRO_DUPLICATE_USE`) — this is
       a correctness hazard, not a style nit: re-running e.g.
       ``pkstd.standard_iiv``'s expansion twice would double-declare IIV on
       the same parameters (which the AST itself does not otherwise forbid,
       since two separately-authored ``IIV(...)`` blocks covering disjoint
       params is legal).
    3. Apply the registered expansion function to get a new ``DSLSpec``.
    4. Append ``"{MacroDef.name}@{MacroDef.version}"`` to the spec's
       ``macros_used`` list.

    Raises:
        FormularCompileError: on an unknown or duplicate macro name.
    """
    applied: set[str] = set()
    macros_used: list[str] = list(spec.macros_used)
    for name in uses:
        entry = MACRO_REGISTRY.get(name)
        if entry is None:
            msg = f"unknown macro {name!r}; registered macros: {sorted(MACRO_REGISTRY)}"
            raise FormularCompileError(FrmCode.AST_MACRO_UNKNOWN, msg)
        if name in applied:
            msg = f"macro {name!r} is used more than once in this spec (use it at most once)"
            raise FormularCompileError(FrmCode.AST_MACRO_DUPLICATE_USE, msg)
        applied.add(name)

        macro_def, expander = entry
        spec = expander(spec)
        macros_used.append(f"{macro_def.name}@{macro_def.version}")

    return spec.model_copy(update={"macros_used": macros_used})


__all__ = [
    "MACRO_REGISTRY",
    "MacroDef",
    "MacroExpander",
    "expand_macros",
    "register_macro",
]

# Import side effect: registers every `pkstd.*` macro into MACRO_REGISTRY.
# Must be the last statement in this module — stdlib.py imports MacroDef/
# register_macro from this module at its own top, so importing it any
# earlier here would be a circular-import failure.
from apmode.dsl.macros import stdlib as _stdlib  # noqa: E402, F401
