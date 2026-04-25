# SPDX-License-Identifier: GPL-2.0-or-later
"""Suite C literature-fixture loader (plan Task 40).

A Phase-1 fixture is a YAML descriptor (see
``benchmarks/suite_c/<dataset_id>.yaml``) plus a sibling DSL-spec JSON
file referenced by ``dsl_spec_path``. This module owns the
``YAML → :class:`LiteratureFixture` → :class:`DSLSpec``` traversal so
the fixture roster, the dataset registry, and the DSL stay in lockstep.

The fixture YAML is the single source of truth for:

* :attr:`LiteratureFixture.reference` — DOI + citation + population so
  reviewers can verify the comparison is in scope (paediatric vs adult,
  healthy vs critically-ill).
* :attr:`LiteratureFixture.reference_params` — the published parameter
  values APMODE is benchmarked against.
* :attr:`LiteratureFixture.parameterization_mapping` — how to translate
  the published parameter names (``TVCL``, ``TVV``, etc.) into the
  DSL-canonical names (``CL``, ``V``).

Keeping the loader here (instead of in :mod:`apmode.benchmarks.suite_c`)
means the case definitions stay declarative — no I/O at import time —
and tests can stub the loader cleanly.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from apmode.benchmarks.models import LiteratureFixture, LiteratureReference
from apmode.dsl.ast_models import DSLSpec

# Resolve fixture paths relative to the repo root by default. The repo
# layout pins ``benchmarks/suite_c/`` next to the package; tests can
# override by passing an explicit ``base_dir``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FIXTURE_DIR = _REPO_ROOT / "benchmarks" / "suite_c"


def load_fixture(fixture_path: Path) -> LiteratureFixture:
    """Parse a Suite C YAML fixture into the validated Pydantic model.

    The path may be absolute or relative; relative paths are resolved
    against the caller's CWD (Pythonic default) — callers that want
    repo-relative resolution should pass an absolute path or use
    :func:`load_fixture_by_id`.

    The ``dsl_spec_path`` field inside the fixture is rewritten to be
    absolute (resolved against the fixture file's parent directory) so
    downstream consumers don't need to track the YAML's location.
    """
    fixture_path = fixture_path.resolve()
    with fixture_path.open() as fp:
        raw = yaml.safe_load(fp)
    if not isinstance(raw, dict):
        msg = f"fixture {fixture_path} did not parse as a mapping"
        raise ValueError(msg)
    # Resolve dsl_spec_path against the fixture's own directory so the
    # YAML can reference siblings with bare filenames.
    spec_field = raw.get("dsl_spec_path")
    if spec_field is None:
        msg = f"fixture {fixture_path} missing required field ``dsl_spec_path``"
        raise ValueError(msg)
    spec_path = Path(str(spec_field))
    if not spec_path.is_absolute():
        spec_path = (fixture_path.parent / spec_path).resolve()
    raw["dsl_spec_path"] = spec_path
    raw["reference"] = LiteratureReference(**raw["reference"])
    return LiteratureFixture(**raw)


def load_fixture_by_id(
    dataset_id: str,
    *,
    base_dir: Path | None = None,
) -> LiteratureFixture:
    """Load ``benchmarks/suite_c/<dataset_id>.yaml`` from the repo.

    Supplying ``base_dir`` overrides the default repo location and is
    used by tests that need to point at a tmp_path.
    """
    fixture_dir = base_dir or _DEFAULT_FIXTURE_DIR
    return load_fixture(fixture_dir / f"{dataset_id}.yaml")


def load_dsl_spec(fixture: LiteratureFixture) -> DSLSpec:
    """Materialise the DSL spec the fixture references.

    The spec lives in a sibling JSON file (one per fixture) so the YAML
    stays human-editable and the strongly-typed AST stays the
    machine-validated source of truth.
    """
    return DSLSpec.model_validate_json(Path(fixture.dsl_spec_path).read_text())


# Phase-1 fixture roster (plan Task 40). The ordering controls the CI
# dashboard's left-to-right column order — keep it stable so the
# fraction-beats-literature-median plots are reviewable across releases.
PHASE1_MLE_FIXTURE_IDS: tuple[str, ...] = (
    # Real, public, no-credentialed-access fixtures from nlmixr2data:
    "theophylline_boeckmann_1992",
    "warfarin_funaki_2018",
    "mavoglurant_wendling_2015",
    "phenobarbital_grasela_1985",
    # Simulated ACOP-2016 ground-truth-recovery fixture:
    "oral_1cpt_acop_2016",
    # Credentialed/manual-download fixtures (kept for operator runs that
    # set --dataset-csv overrides; CI can't reach them):
    "gentamicin_germovsek_2017",
    "schoemaker_nlmixr2_tutorial",
)


__all__ = [
    "PHASE1_MLE_FIXTURE_IDS",
    "load_dsl_spec",
    "load_fixture",
    "load_fixture_by_id",
]
