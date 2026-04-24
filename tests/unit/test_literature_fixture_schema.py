# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the LiteratureFixture schema (plan Task 39).

Suite C Phase 1 anchors APMODE's evaluation against published reference
parameterizations. Each fixture must carry verifiable provenance (a
Crossref-canonical DOI on the citation), the DSL spec that the
literature model was fit to, the published parameter values, and a
mapping from published parameter names to the DSL-standard names
(``TVCL`` → ``CL``, ``Theta1`` → ``CL``, etc.).

Tests cover:

* DOI shape — the same Crossref pattern used by the prior-manifest
  validator gates the literature reference.
* citation / population_description length floors prevent stub fixtures.
* ``parameterization_mapping`` keys must be drawn from
  ``reference_params``; otherwise the fixture would map nothing and
  silently mis-align cross-tool comparisons.
* ``reference_params`` must be non-empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apmode.benchmarks.models import LiteratureFixture, LiteratureReference

_VALID_DOI = "10.1007/s10928-018-9588-7"
_LONG_CITATION = "Germovsek et al. (2017) AAC 61(8):e00481-17"
_LONG_POPULATION = "205 neonates, gestational age 24-42 weeks, post-natal age 0-30 days"


def _ref(**overrides: object) -> LiteratureReference:
    base: dict[str, object] = {
        "citation": _LONG_CITATION,
        "doi": _VALID_DOI,
        "population_description": _LONG_POPULATION,
    }
    base.update(overrides)
    return LiteratureReference(**base)  # type: ignore[arg-type]


def _fixture(**overrides: object) -> LiteratureFixture:
    base: dict[str, object] = {
        "dataset_id": "ddmore_gentamicin",
        "reference": _ref(),
        "dsl_spec_path": Path("benchmarks/suite_c/gentamicin_germovsek_2017.dsl.json"),
        "reference_params": {"CL": 0.065, "V": 0.45},
        "parameterization_mapping": {"TVCL": "CL", "TVV": "V"},
    }
    base.update(overrides)
    # Mapping keys must align with reference_params for the validator to pass.
    # The fixture builder picks a parameterization_mapping that matches the
    # default reference_params; tests override both as needed.
    return LiteratureFixture(**base)  # type: ignore[arg-type]


# --- LiteratureReference -------------------------------------------------


def test_valid_reference_constructs() -> None:
    ref = _ref()
    assert ref.doi == _VALID_DOI
    assert ref.citation.startswith("Germovsek")


def test_doi_must_match_crossref_pattern() -> None:
    with pytest.raises(ValueError, match=r"Crossref-canonical pattern"):
        _ref(doi="not-a-doi")


def test_short_citation_rejected() -> None:
    with pytest.raises(ValueError, match=r"at least 10 characters"):
        _ref(citation="short")


def test_short_population_rejected() -> None:
    with pytest.raises(ValueError, match=r"at least 10 characters"):
        _ref(population_description="adults")


def test_doi_with_brackets_accepted() -> None:
    """Wiley SICI-style DOIs use angle/square brackets."""
    _ref(doi="10.1002/(SICI)1097-0258(199608)15:15<1573::AID-SIM263>3.0.CO;2-3")


# --- LiteratureFixture ---------------------------------------------------


def test_valid_fixture_round_trips() -> None:
    fix = _fixture()
    dump = fix.model_dump()
    assert dump["dataset_id"] == "ddmore_gentamicin"
    assert dump["reference_params"] == {"CL": 0.065, "V": 0.45}


def test_empty_dataset_id_rejected() -> None:
    with pytest.raises(ValueError, match=r"at least 1 character"):
        _fixture(dataset_id="")


def test_empty_reference_params_rejected() -> None:
    with pytest.raises(ValueError, match=r"at least 1 item"):
        _fixture(reference_params={}, parameterization_mapping={})


def test_mapping_with_unknown_values_rejected() -> None:
    """Mapping values (DSL-canonical names) must appear in reference_params."""
    with pytest.raises(ValueError, match=r"do not appear in reference_params"):
        _fixture(
            reference_params={"CL": 0.065, "V": 0.45},
            parameterization_mapping={"TVCL": "CL", "TVUNKNOWN": "BOGUS"},
        )


def test_empty_mapping_is_valid() -> None:
    """Phase-1 fixtures may omit the mapping when names already align."""
    fix = _fixture(parameterization_mapping={})
    assert fix.parameterization_mapping == {}


def test_dsl_spec_path_round_trips_as_path() -> None:
    """The path is stored as Path so loaders can call ``.exists()`` / ``.read_text()``."""
    fix = _fixture(dsl_spec_path=Path("/abs/path/to/spec.json"))
    assert isinstance(fix.dsl_spec_path, Path)
