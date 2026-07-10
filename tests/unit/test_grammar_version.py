# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for DSL grammar version pinning (reproducibility bundle provenance).

Grammar identity is a content hash of ``pk_grammar.lark`` rather than a
hand-maintained version string, so a grammar edit cannot silently drift
without changing the pinned value (see docs/plans/2026-07-09-qaqc-remediation.md
§ "Pin DSL grammar version in the reproducibility bundle").
"""

from pathlib import Path

from apmode.dsl.grammar import (
    _GRAMMAR_PATH,
    _grammar_version_for_path,
    compile_dsl,
    grammar_version,
)

_TRIVIAL_SPEC = """
model {
    absorption: FirstOrder(ka)
    distribution: OneCmt(V)
    elimination: Linear(CL)
    variability: IIV(params=[CL, V], structure=diagonal)
    observation: Proportional(sigma_prop=0.1)
    initial: { ka = 1.0, V = 70.0, CL = 5.0 }
}
"""


class TestGrammarVersion:
    def test_returns_sha256_hex_digest(self) -> None:
        version = grammar_version()
        assert isinstance(version, str)
        assert len(version) == 64
        assert all(c in "0123456789abcdef" for c in version)

    def test_deterministic_across_calls(self) -> None:
        assert grammar_version() == grammar_version()

    def test_matches_direct_hash_of_grammar_file(self) -> None:
        assert grammar_version() == _grammar_version_for_path(_GRAMMAR_PATH)

    def test_digest_changes_when_grammar_bytes_change(self, tmp_path: Path) -> None:
        original = _GRAMMAR_PATH.read_bytes()
        modified_path = tmp_path / "pk_grammar_modified.lark"
        modified_path.write_bytes(original + b"\n// trailing byte\n")

        original_digest = _grammar_version_for_path(_GRAMMAR_PATH)
        modified_digest = _grammar_version_for_path(modified_path)

        assert original_digest != modified_digest
        assert len(modified_digest) == 64


class TestCompileDslGrammarVersionIsolation:
    """compile_dsl() stays grammar-identity-agnostic (CLAUDE.md: DSLSpec /
    structure_fingerprint / spec_fingerprint must not vary with compiler
    provenance — only with model content, matching the ``macros_used``
    exclusion-by-omission precedent in canonical.py).

    Grammar identity is threaded into the bundle at the
    ``BundleEmitter.write_compiled_spec`` boundary (see
    tests/unit/test_bundle_emitter.py), not through compile_dsl() itself —
    callers that want the pin call ``grammar_version()`` directly alongside
    ``compile_dsl()``'s result.
    """

    def test_compile_dsl_result_has_no_grammar_version_key(self) -> None:
        spec = compile_dsl(_TRIVIAL_SPEC)
        dumped = spec.model_dump()
        assert "dsl_grammar_version" not in dumped
        assert "grammar_version" not in dumped

    def test_grammar_version_usable_alongside_compile_dsl(self) -> None:
        spec = compile_dsl(_TRIVIAL_SPEC)
        version = grammar_version()
        assert spec.model_id  # compile succeeded
        assert version == grammar_version()  # deterministic, callable independently
