# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration test — Phase-1 Suite C MLE literature fixtures (plan Task 40).

For every fixture id in :data:`PHASE1_MLE_FIXTURE_IDS` the test:

1. Loads the YAML via :func:`load_fixture_by_id` (validates the
   :class:`LiteratureReference` DOI shape, citation length, mapping
   alignment).
2. Materialises the referenced :class:`DSLSpec` JSON and asserts
   :func:`validate_dsl` returns no errors against the submission lane.
3. Calls :func:`emit_nlmixr2` and asserts the rendered R code mentions
   every parameter from ``reference_params``. This is a structural
   smoke test — it does *not* fit the model, only that the DSL → R
   pipeline survives the fixture.
4. Cross-checks that the fixture's ``dataset_id`` exists in
   :file:`benchmarks/datasets/registry.yaml` so a stale fixture
   referencing a deleted dataset card is caught at the test layer.

The full fit + scoring loop lives in plan Task 41 (weekly CI dashboard);
this test only proves the fixtures are loadable and DSL-compatible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apmode.backends.protocol import Lane
from apmode.benchmarks.literature_loader import (
    PHASE1_MLE_FIXTURE_IDS,
    load_dsl_spec,
    load_fixture_by_id,
)
from apmode.benchmarks.models import LiteratureFixture
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2
from apmode.dsl.validator import validate_dsl

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO_ROOT / "benchmarks" / "datasets" / "registry.yaml"


def _registry_dataset_ids() -> set[str]:
    """Pull the dataset_id roster from the YAML registry on disk."""
    with _REGISTRY.open() as fp:
        data = yaml.safe_load(fp)
    return set(data.get("datasets", {}).keys())


# ---------------------------------------------------------------------------
# Roster shape
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_phase1_roster_has_five_fixtures() -> None:
    """The plan calls for exactly five Phase-1 MLE fixtures."""
    assert len(PHASE1_MLE_FIXTURE_IDS) == 5
    # No duplicates — duplicates would silently double-count in the
    # fraction-beats-literature-median scoring (Task 41).
    assert len(set(PHASE1_MLE_FIXTURE_IDS)) == 5


# ---------------------------------------------------------------------------
# Per-fixture loading + DSL validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_MLE_FIXTURE_IDS)
def test_fixture_loads(fixture_id: str) -> None:
    """Each fixture's YAML parses and validates against LiteratureFixture."""
    fix = load_fixture_by_id(fixture_id)
    assert isinstance(fix, LiteratureFixture)
    assert fix.dataset_id  # validator already enforces min_length=1
    assert fix.reference_params  # validator already enforces min_length=1


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_MLE_FIXTURE_IDS)
def test_fixture_dataset_id_in_registry(fixture_id: str) -> None:
    """Each fixture's dataset_id resolves to a card in the dataset registry.

    A stale fixture that references a deleted dataset card would silently
    skip in CI — guard against that here.
    """
    fix = load_fixture_by_id(fixture_id)
    known = _registry_dataset_ids()
    assert fix.dataset_id in known, (
        f"fixture {fixture_id} references unknown dataset_id {fix.dataset_id!r}; "
        f"known ids = {sorted(known)}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_MLE_FIXTURE_IDS)
def test_dsl_spec_validates_under_submission_lane(fixture_id: str) -> None:
    """Each fixture's DSL spec passes the submission-lane validator.

    Submission is the strictest lane (NODE backends not admissible); if
    the spec validates here it will validate in Discovery / Optimization
    too. A failing spec is almost always a literature-fixture-author bug.
    """
    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    errors = validate_dsl(spec, lane=Lane.SUBMISSION)
    assert errors == [], f"fixture {fixture_id} DSL spec failed validation: " + "; ".join(
        e.message for e in errors
    )


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_MLE_FIXTURE_IDS)
def test_emit_nlmixr2_mentions_every_reference_param(fixture_id: str) -> None:
    """The emitted R code names every parameter the fixture says it has.

    Smoke-tests the DSL → R pipeline and catches the common mismatch
    where a fixture's ``reference_params`` lists ``ka`` but the DSL
    spec was written without an absorption parameter (e.g. IVBolus
    accidentally wrote ka into the fixture).
    """
    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    r_code = emit_nlmixr2(spec)
    # Strip the published-name → DSL-name mapping out of the search
    # (we only care that the DSL-canonical names appear in the rendered
    # R code; published names like ``TVCL`` would not).
    dsl_targets = set(fix.parameterization_mapping.values()) | set(fix.reference_params)
    # The DSL-canonical names should appear verbatim in the emitted
    # code — emit_nlmixr2 declares each one in the ini() block. Allow
    # ``tlag`` which the lagged-FO emitter renders as ``tlag``.
    missing = sorted(p for p in dsl_targets if p not in r_code)
    assert missing == [], (
        f"fixture {fixture_id}: emitted R code missing parameters {missing}\n\n"
        f"--- emitted R code ---\n{r_code}\n--- end ---"
    )


# ---------------------------------------------------------------------------
# Mapping invariants — already enforced by LiteratureFixture validator,
# but pin them at the fixture level so a hand-edit can't slip past.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_MLE_FIXTURE_IDS)
def test_mapping_values_are_dsl_names(fixture_id: str) -> None:
    """Every parameterization_mapping value resolves to a reference_param key.

    The LiteratureFixture validator already enforces this — re-asserting
    it at the fixture level catches a class of YAML hand-edit bug where
    someone adds a published name without the DSL-side counterpart.
    """
    fix = load_fixture_by_id(fixture_id)
    unknown = set(fix.parameterization_mapping.values()) - set(fix.reference_params)
    assert unknown == set(), (
        f"fixture {fixture_id}: mapping values {unknown} not in reference_params"
    )


# ---------------------------------------------------------------------------
# DOI provenance — every reference must carry a Crossref-canonical DOI.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_MLE_FIXTURE_IDS)
def test_reference_has_canonical_doi(fixture_id: str) -> None:
    """LiteratureReference enforces the DOI shape; assert it actually is set."""
    fix = load_fixture_by_id(fixture_id)
    assert fix.reference.doi.startswith("10.")
    assert "/" in fix.reference.doi
