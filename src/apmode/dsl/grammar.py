# SPDX-License-Identifier: GPL-2.0-or-later
"""DSL grammar loader and compiler for PK model specifications.

Provides parse-only (parse tree) and full compilation (parse → AST) modes.
Semantic validation (dim ceilings, constraint enforcement) is a separate phase.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

from lark import Lark, Tree

from apmode.dsl.ast_models import DSLSpec
from apmode.dsl.errors import FormularCompileError, FrmCode
from apmode.dsl.macros import expand_macros
from apmode.dsl.priors import PriorSpec, build_prior_spec
from apmode.dsl.transformer import DSLTransformer, RawPriorEntry

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


def _grammar_version_for_path(path: Path) -> str:
    """sha256 hex digest of a grammar file's raw bytes.

    Lower-level helper so tests (and any future multi-grammar tooling) can
    compute the digest for an arbitrary path without going through the
    cached, module-pinned :func:`grammar_version`.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def grammar_version() -> str:
    """Stable sha256 hex digest of ``pk_grammar.lark``'s bytes.

    Grammar identity is derived from file content — not a hand-maintained
    version string — so an unbumped grammar edit cannot silently drift the
    reproducibility bundle's compiler-provenance record (see
    docs/FINGERPRINT_MIGRATION.md for the analogous spec-content treatment).
    Cached after first call, mirroring :func:`load_grammar` immediately
    above.
    """
    return _grammar_version_for_path(_GRAMMAR_PATH)


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
    "initial_block": "initial",
    "metadata_block": "metadata",
    "units_block": "units",
    "priors_block": "priors",
    "covariates_block": "covariates",
}

# Formular sharpening plan §4 Phase 1 (P1.1): top-level blocks may appear
# in any order (``model_body: block*``); these are the exactly-one-required
# block kinds and the at-most-one-optional block kinds, checked on the raw
# parse tree before ``DSLTransformer`` runs (a missing required block means
# ``DSLSpec`` — whose fields are non-optional — cannot even be constructed).
_REQUIRED_SINGLE_BLOCKS = ("absorption", "distribution", "elimination")
# P1.7: ``observation:`` (legacy singular sugar) and ``observations:`` (new
# multi-analyte block) are mutually exclusive alternatives for the *same*
# required concept — exactly one of the two must appear, not "at most one
# of each independently". See ``_validate_block_cardinality`` below.
_OBSERVATION_GROUP_BLOCKS = ("observation", "observations_block")
_OPTIONAL_SINGLE_BLOCKS = (
    "metadata_block",
    "initial_block",
    "units_block",
    "priors_block",
    "covariates_block",
)


def _validate_block_cardinality(tree: Tree) -> None:  # type: ignore[type-arg]
    """Enforce top-level block cardinality on the raw parse tree.

    Exactly one absorption/distribution/elimination block; exactly one of
    observation:/observations: (P1.7 — the two are mutually exclusive
    alternatives, not independently-optional); at most one metadata/initial/
    units/priors/covariates block; zero-or-more variability blocks. Raises
    :class:`~apmode.dsl.errors.FormularCompileError` (not a bare Lark parse
    error) on violation so callers get a stable ``FRM-AST-0NN`` code instead
    of an unhandled exception.
    """
    counts: dict[str, int] = {}
    for sub in tree.iter_subtrees_topdown():  # type: ignore[no-untyped-call]
        if (
            sub.data in _REQUIRED_SINGLE_BLOCKS
            or sub.data in _OPTIONAL_SINGLE_BLOCKS
            or sub.data in _OBSERVATION_GROUP_BLOCKS
        ):
            counts[sub.data] = counts.get(sub.data, 0) + 1

    for name in _REQUIRED_SINGLE_BLOCKS:
        if counts.get(name, 0) == 0:
            msg = f"model {{ }} is missing a required '{name}:' block"
            raise FormularCompileError(FrmCode.AST_MISSING_REQUIRED_BLOCK, msg)

    obs_total = sum(counts.get(name, 0) for name in _OBSERVATION_GROUP_BLOCKS)
    if obs_total == 0:
        msg = "model { } is missing a required 'observation:' or 'observations:' block"
        raise FormularCompileError(FrmCode.AST_MISSING_REQUIRED_BLOCK, msg)
    if obs_total > 1:
        msg = (
            "model { } declares more than one of 'observation:'/'observations:' "
            "(each may appear at most once, and they are mutually exclusive "
            "alternatives — use 'observation:' for a single endpoint or "
            "'observations:' for multiple analytes, not both)"
        )
        raise FormularCompileError(FrmCode.AST_DUPLICATE_BLOCK, msg)

    block_labels = {
        "metadata_block": "metadata",
        "initial_block": "initial",
        "units_block": "units",
        "priors_block": "priors",
        "covariates_block": "covariates",
    }
    for name in (*_REQUIRED_SINGLE_BLOCKS, *_OPTIONAL_SINGLE_BLOCKS):
        count = counts.get(name, 0)
        if count > 1:
            label = block_labels.get(name, name)
            msg = f"model {{ }} declares '{label}:' {count} times; at most one is allowed"
            raise FormularCompileError(FrmCode.AST_DUPLICATE_BLOCK, msg)


def _child_name(child: object) -> str | None:
    """Return the first token-like child from a parse-tree entry, if present."""
    if not isinstance(child, Tree) or not child.children:
        return None
    return str(child.children[0])


def _validate_unique_named_entries(
    tree: Tree,  # type: ignore[type-arg]
    *,
    block_rule: str,
    entry_rule: str,
    label: str,
) -> None:
    """Reject duplicate names inside one map-like block before transform-time dict collapse."""
    for block in tree.find_data(block_rule):
        seen: set[str] = set()
        for child in block.children:
            if not isinstance(child, Tree) or child.data != entry_rule:
                continue
            name = _child_name(child)
            if name is None:
                continue
            if name in seen:
                msg = f"{label}: duplicate entry '{name}' would overwrite an earlier entry"
                raise FormularCompileError(FrmCode.AST_DUPLICATE_BLOCK, msg)
            seen.add(name)


def _validate_metadata_unique_fields(tree: Tree) -> None:  # type: ignore[type-arg]
    """Reject duplicate metadata fields before ``Metadata(**dict(items))`` overwrites them."""
    for block in tree.find_data("metadata_block"):
        seen: set[str] = set()
        for child in block.children:
            if not isinstance(child, Tree) or not str(child.data).startswith("metadata_"):
                continue
            name = str(child.data).removeprefix("metadata_")
            if name in seen:
                msg = f"metadata: duplicate field '{name}' would overwrite an earlier value"
                raise FormularCompileError(FrmCode.AST_DUPLICATE_BLOCK, msg)
            seen.add(name)


def _validate_entry_uniqueness(tree: Tree) -> None:  # type: ignore[type-arg]
    """Reject duplicate keys in blocks that lower through dict construction.

    ``DSLTransformer`` intentionally stashes several block entries on side
    channels and later builds Python dicts from them. Without this raw-tree
    pass, duplicate source keys are silently collapsed by "last value wins"
    semantics before the validator, formatter, diff, or fingerprint layers
    can see the lost declaration.
    """
    _validate_unique_named_entries(
        tree,
        block_rule="observations_block",
        entry_rule="observation_entry",
        label="observations",
    )
    _validate_unique_named_entries(
        tree,
        block_rule="initial_block",
        entry_rule="initial_item",
        label="initial",
    )
    _validate_metadata_unique_fields(tree)


def _collect_source_meta(tree: Tree) -> dict[str, tuple[int, int]]:  # type: ignore[type-arg]
    """Walk a parse tree and collect (line, column) for known top-level roles.

    Variability items are indexed in source order as
    ``variability[0]``, ``variability[1]``…
    """
    out: dict[str, tuple[int, int]] = {}
    var_idx = 0
    cov_idx = 0
    obs_idx = 0
    for sub in tree.iter_subtrees_topdown():  # type: ignore[no-untyped-call]
        rule = sub.data
        meta = getattr(sub, "meta", None)
        if meta is None or getattr(meta, "empty", True):
            continue
        line = int(meta.line)
        column = int(meta.column)
        if rule in _ROLE_TO_RULE and _ROLE_TO_RULE[rule] not in out:
            out[_ROLE_TO_RULE[rule]] = (line, column)
        elif rule in ("iiv", "iov"):
            out[f"variability[{var_idx}]"] = (line, column)
            var_idx += 1
        elif rule == "covariate_entry":
            out[f"covariates[{cov_idx}]"] = (line, column)
            cov_idx += 1
        elif rule == "observation_entry":
            out[f"observations[{obs_idx}]"] = (line, column)
            obs_idx += 1
    return out


def _lower_priors(spec: DSLSpec, raw_priors: list[RawPriorEntry]) -> DSLSpec:
    """Lower parsed ``priors:`` entries to ``PriorSpec`` via the canonical factory.

    Every entry MUST route through :func:`apmode.dsl.priors.build_prior_spec`
    — the same factory :func:`apmode.dsl.prior_transforms.apply_set_prior`
    uses — so a human-authored Formular prior and an agentic ``SetPrior``
    transform are governed by identical ``classify_target`` /
    ``validate_prior_family`` invariants (Formular sharpening plan §4 P1.5
    parity guarantee). ``structural_params`` is drawn from ``spec`` itself
    (already fully assembled at this point — absorption/distribution/
    elimination are transformed before this runs), exactly as
    ``validate_set_prior`` draws it from the spec a ``SetPrior`` is applied
    to.

    Raises :class:`~apmode.dsl.errors.FormularCompileError` with
    :attr:`FrmCode.PRIOR_INVALID_DECLARATION` when ``build_prior_spec``
    rejects an entry (unresolvable target, family/target-kind mismatch, or
    an informative source missing justification/``historical_refs``).
    """
    structural = set(spec.structural_param_names())
    priors: list[PriorSpec] = []
    for entry in raw_priors:
        try:
            priors.append(
                build_prior_spec(
                    target=entry.target,
                    family=entry.family,
                    source=entry.source,
                    justification=entry.justification,
                    doi=entry.doi,
                    historical_refs=entry.historical_refs,
                    structural_params=structural,
                )
            )
        except ValueError as exc:
            msg = f"priors: entry '{entry.target}' is invalid: {exc}"
            raise FormularCompileError(FrmCode.PRIOR_INVALID_DECLARATION, msg) from exc
    return spec.model_copy(update={"priors": priors})


def compile_dsl(text: str) -> DSLSpec:
    """Parse and transform a DSL spec into a typed Pydantic AST.

    This is the primary entry point for the DSL compiler. Returns a fully
    typed DSLSpec with a generated model_id. When the grammar emits
    positional metadata (``propagate_positions=True``) this function
    attaches a ``source_meta`` sidecar so the validator can annotate
    errors with line/column information.

    Raises ValueError for oversized input, lark.exceptions.UnexpectedInput
    for syntax errors, and FormularCompileError for block-cardinality
    violations (missing/duplicate top-level blocks — P1.1), an invalid
    ``priors:`` entry (P1.5, ``FrmCode.PRIOR_INVALID_DECLARATION``), or an
    invalid ``use:`` macro reference (P2.1, ``FrmCode.AST_MACRO_UNKNOWN`` /
    ``FrmCode.AST_MACRO_DUPLICATE_USE``).
    """
    tree = parse_dsl(text)
    _validate_block_cardinality(tree)
    _validate_entry_uniqueness(tree)
    transformer = DSLTransformer()
    result = transformer.transform(tree)
    assert isinstance(result, DSLSpec)  # guaranteed by grammar's start rule
    if transformer.raw_priors:
        result = _lower_priors(result, transformer.raw_priors)
    if transformer.raw_covariates:
        result = result.model_copy(update={"covariates": transformer.raw_covariates})
    if transformer.raw_observations:
        result = result.model_copy(update={"observations": transformer.raw_observations})
    if transformer.raw_macro_uses:
        # P2.1: expand `use <macro>` statements into plain AST nodes. Must
        # run after every other raw_*-folding step above so macros like
        # pkstd.standard_iiv see the fully-assembled spec (structural
        # modules, any hand-authored variability/priors/covariates) when
        # deciding what is already covered.
        result = expand_macros(result, transformer.raw_macro_uses)
    meta = _collect_source_meta(tree)
    if meta:
        # DSLSpec is frozen — rebuild with the sidecar populated. Fields
        # that already exist on the result carry through via model_copy.
        result = result.model_copy(update={"source_meta": meta})
    return result
