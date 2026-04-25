# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for plan Task 44 — Phase-1 Suite C live-fit runner.

The R subprocess is mocked at the ``Nlmixr2Runner.run`` boundary so
these tests cover the orchestration (split fan-out, NPE aggregation,
inputs-JSON writer) without needing R / cmdstan on the test runner.
The integration test that actually invokes Rscript lives in
``tests/integration/test_suite_c_phase1_mle.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from apmode.benchmarks.suite_c_phase1_runner import (
    FixturePhase1Inputs,
    _parse_dataset_csv,
    main,
    resolve_dataset_csv,
    run_fixture,
    write_inputs_atomic,
)
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
)


def _make_backend_result(npe: float | None, *, model_id: str = "fake_candidate") -> BackendResult:
    """Minimal BackendResult carrying an NPE score, mirroring suite_b.make_b3_result.

    The runner only reads ``diagnostics.npe_score``; the surrounding
    fields exist purely so the Pydantic model validates. Stub each
    required field to its smallest valid form so the test stays focused
    on the NPE-aggregation contract.
    """
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=True,
        ofv=-200.0,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=5.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=50.0, category="structural"),
            "ka": ParameterEstimate(name="ka", estimate=1.0, category="structural"),
        },
        eta_shrinkage={"CL": 0.0, "V": 0.0},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=1,
            gradient_norm=0.0,
            minimization_status="successful",
            wall_time_seconds=0.1,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.0),
            identifiability=IdentifiabilityFlags(profile_likelihood_ci={}, ill_conditioned=False),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            npe_score=npe,
        ),
        wall_time_seconds=0.1,
        backend_versions={"nlmixr2": "test"},
        initial_estimate_source="nca",
    )


if TYPE_CHECKING:
    from collections.abc import Iterator

    from apmode.benchmarks.models import LiteratureFixture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pk_csv(tmp_path: Path) -> Path:
    """Tiny well-formed NONMEM CSV: 10 subjects x 6 rows each."""
    rng = np.random.default_rng(20260424)
    rows: list[dict[str, object]] = []
    for sid in range(1, 11):
        rows.append(
            {
                "NMID": sid,
                "TIME": 0.0,
                "DV": 0.0,
                "AMT": 320.0,
                "EVID": 1,
                "MDV": 1,
                "CMT": 1,
            }
        )
        for t in (0.5, 1.0, 2.0, 4.0, 8.0):
            conc = max(0.01, 5.0 * np.exp(-0.1 * t) + rng.normal(0, 0.2))
            rows.append(
                {
                    "NMID": sid,
                    "TIME": t,
                    "DV": float(conc),
                    "AMT": 0.0,
                    "EVID": 0,
                    "MDV": 0,
                    "CMT": 1,
                }
            )
    csv_path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def fake_backend_result_factory() -> Iterator[object]:
    """Build a minimal :class:`BackendResult` carrying an NPE score."""
    return cast("Iterator[object]", _make_backend_result)


# ---------------------------------------------------------------------------
# resolve_dataset_csv
# ---------------------------------------------------------------------------


class _StubFixture:
    """Stand-in for LiteratureFixture exposing only ``dataset_id``."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id


def test_resolve_dataset_csv_uses_override(tmp_path: Path, synthetic_pk_csv: Path) -> None:
    fix = cast("LiteratureFixture", _StubFixture("ddmore_gentamicin"))
    out = resolve_dataset_csv(
        fix,
        cache_dir=tmp_path,
        overrides={"ddmore_gentamicin": synthetic_pk_csv},
    )
    assert out == synthetic_pk_csv


def test_resolve_dataset_csv_rejects_missing_override(tmp_path: Path) -> None:
    fix = cast("LiteratureFixture", _StubFixture("ddmore_gentamicin"))
    with pytest.raises(FileNotFoundError, match="not a regular file"):
        resolve_dataset_csv(
            fix,
            cache_dir=tmp_path,
            overrides={"ddmore_gentamicin": tmp_path / "does_not_exist.csv"},
        )


def test_resolve_dataset_csv_actionable_error_for_unknown_id(tmp_path: Path) -> None:
    fix = cast("LiteratureFixture", _StubFixture("mimic_vancomycin"))
    with pytest.raises(KeyError, match="--dataset-csv mimic_vancomycin="):
        resolve_dataset_csv(fix, cache_dir=tmp_path, overrides={})


# ---------------------------------------------------------------------------
# write_inputs_atomic
# ---------------------------------------------------------------------------


def test_write_inputs_atomic_emits_scorer_compatible_shape(tmp_path: Path) -> None:
    inputs = {
        "theophylline_boeckmann_1992": FixturePhase1Inputs(
            fixture_id="theophylline_boeckmann_1992",
            npe_apmode=0.95,
            npe_literature=1.00,
            npe_apmode_per_fold=(0.93, 0.96, 0.94, 0.97, 0.95),
            npe_literature_per_fold=(1.0, 1.01, 0.99, 1.0, 1.02),
            n_subjects=12,
            n_folds=5,
        )
    }
    out = tmp_path / "phase1_npe_inputs.json"
    write_inputs_atomic(out, inputs)

    payload = json.loads(out.read_text())
    entry = payload["theophylline_boeckmann_1992"]
    assert entry["npe_apmode"] == pytest.approx(0.95)
    assert entry["npe_literature"] == pytest.approx(1.00)
    assert entry["npe_apmode_per_fold"] == [0.93, 0.96, 0.94, 0.97, 0.95]
    assert entry["n_subjects"] == 12
    # Per-fold arrays must match n_folds — schema invariant the
    # downstream FixtureScore validator enforces.
    assert len(entry["npe_apmode_per_fold"]) == entry["n_folds"]
    assert len(entry["npe_literature_per_fold"]) == entry["n_folds"]
    # No leftover .tmp file on success.
    assert not list(tmp_path.glob("*.tmp"))


def test_write_inputs_atomic_round_trips_through_scorer_cli(tmp_path: Path) -> None:
    """Writer output round-trips through the Task 41 CLI's ``_load_inputs``."""
    from apmode.benchmarks.suite_c_phase1_cli import _load_inputs

    inputs = {
        "warfarin_funaki_2018": FixturePhase1Inputs(
            fixture_id="warfarin_funaki_2018",
            npe_apmode=0.90,
            npe_literature=1.0,
            npe_apmode_per_fold=(0.88, 0.91, 0.90, 0.92, 0.89),
            npe_literature_per_fold=(1.0,) * 5,
            n_subjects=32,
            n_folds=5,
        ),
        "mavoglurant_wendling_2015": FixturePhase1Inputs(
            fixture_id="mavoglurant_wendling_2015",
            npe_apmode=0.94,
            npe_literature=1.0,
            npe_apmode_per_fold=(0.94,) * 5,
            npe_literature_per_fold=(1.0,) * 5,
            n_subjects=14,
            n_folds=5,
        ),
    }
    out = tmp_path / "phase1_npe_inputs.json"
    write_inputs_atomic(out, inputs)

    loaded = _load_inputs(out)
    assert set(loaded) == {"warfarin_funaki_2018", "mavoglurant_wendling_2015"}
    warf = loaded["warfarin_funaki_2018"]
    assert warf["npe_apmode"] == pytest.approx(0.90)
    assert warf["npe_apmode_per_fold"] == (0.88, 0.91, 0.90, 0.92, 0.89)


# ---------------------------------------------------------------------------
# _parse_dataset_csv
# ---------------------------------------------------------------------------


def test_parse_dataset_csv_handles_repeated_flags(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    a.write_text("ignored")
    b = tmp_path / "b.csv"
    b.write_text("ignored")
    parsed = _parse_dataset_csv([f"id_a={a}", f"id_b={b}"])
    assert parsed["id_a"] == a.resolve()
    assert parsed["id_b"] == b.resolve()


def test_parse_dataset_csv_rejects_malformed() -> None:
    import argparse

    with pytest.raises(argparse.ArgumentTypeError, match="id=path"):
        _parse_dataset_csv(["no_equals"])
    with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
        _parse_dataset_csv(["=/tmp/foo.csv"])


# ---------------------------------------------------------------------------
# run_fixture — mocks out Nlmixr2Runner so no R is required
# ---------------------------------------------------------------------------


def test_run_fixture_aggregates_per_fold_npe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_pk_csv: Path,
    fake_backend_result_factory: object,
) -> None:
    """run_fixture median-aggregates per-fold NPEs from both fits."""
    make_result = fake_backend_result_factory  # callable factory

    # Synthetic per-fold NPE sequences (10 calls = 5 folds x 2 fits).
    apmode_npes = [0.93, 0.96, 0.94, 0.97, 0.95]
    literature_npes = [1.00, 1.01, 0.99, 1.00, 1.02]
    interleaved = [v for pair in zip(apmode_npes, literature_npes, strict=True) for v in pair]

    fake_runner = AsyncMock()
    fake_runner.run = AsyncMock(side_effect=[make_result(npe) for npe in interleaved])

    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner.load_fixture_by_id",
        lambda _fid: cast("LiteratureFixture", _StubFixture("nlmixr2data_theophylline")),
    )
    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner.load_dsl_spec",
        lambda _fix: object(),
    )
    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner.resolve_dataset_csv",
        lambda _fix, *, cache_dir, overrides: synthetic_pk_csv,
    )

    # fixture object also needs reference_params for the literature warm-start;
    # patch the translator directly so we avoid building a real LiteratureFixture.
    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner._translate_reference_params",
        lambda _fix: {"CL": 2.83, "V": 32.0, "ka": 1.5},
    )

    result = asyncio.run(
        run_fixture(
            "theophylline_boeckmann_1992",
            runner=fake_runner,  # type: ignore[arg-type]
            cache_dir=tmp_path / "cache",
            work_dir=tmp_path / "work",
            n_folds=5,
            n_sims=100,
        )
    )

    assert result.npe_apmode == pytest.approx(0.95)  # median of apmode_npes
    assert result.npe_literature == pytest.approx(1.00)  # median of literature_npes
    assert result.npe_apmode_per_fold == tuple(apmode_npes)
    assert result.npe_literature_per_fold == tuple(literature_npes)
    assert result.n_folds == 5
    assert result.n_subjects == 10
    # Each fold runs two fits (APMODE then literature) -> 10 calls total.
    assert fake_runner.run.await_count == 10

    # The literature run must be invoked with the published reference
    # params (not the NCA estimates) — verifying call args directly
    # catches the class of bugs where the warm-start values are
    # misrouted. ``Nlmixr2Runner.run(spec, manifest, initial_estimates,
    # seed, ...)`` takes the first four args positionally.
    literature_init = {"CL": 2.83, "V": 32.0, "ka": 1.5}
    literature_calls = [
        call for call in fake_runner.run.call_args_list if call.args[2] == literature_init
    ]
    assert len(literature_calls) == 5, (
        "literature run must receive reference_params, not NCA estimates"
    )

    # Same-seed-within-fold invariant: the APMODE and literature fits
    # in a fold must share an RNG seed so the per-fold NPE difference
    # is driven by THETA differences, not posterior-predictive noise.
    # The seed is the 4th positional arg.
    seeds = [call.args[3] for call in fake_runner.run.call_args_list]
    assert len(set(seeds)) == 5, (
        f"expected 5 distinct fold seeds (one shared per fold), got {sorted(set(seeds))}"
    )
    for seed_value in set(seeds):
        assert seeds.count(seed_value) == 2, (
            f"seed {seed_value} should appear in exactly 2 calls "
            f"(apmode + literature within one fold), got {seeds.count(seed_value)}"
        )


def test_run_fixture_surfaces_missing_npe_loudly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_pk_csv: Path,
    fake_backend_result_factory: object,
) -> None:
    """A fold that returns ``npe_score=None`` aborts the fixture."""
    make_result = fake_backend_result_factory  # callable factory

    # Provide 10 mock entries so a hypothetical bug that swallows the
    # None silently would *not* trip ``StopAsyncIteration`` at fold 1
    # — the test must catch the RuntimeError from the inner
    # _extract_npe call, not from the mock running dry.
    fake_runner = AsyncMock()
    fake_runner.run = AsyncMock(
        side_effect=[
            make_result(0.95),  # fold 0 apmode OK
            _make_backend_result(None),  # fold 0 literature -> None (must raise here)
            *[make_result(0.95) for _ in range(8)],  # filler so StopAsyncIteration is unreachable
        ]
    )

    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner.load_fixture_by_id",
        lambda _fid: cast("LiteratureFixture", _StubFixture("nlmixr2data_theophylline")),
    )
    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner.load_dsl_spec",
        lambda _fix: object(),
    )
    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner.resolve_dataset_csv",
        lambda _fix, *, cache_dir, overrides: synthetic_pk_csv,
    )
    monkeypatch.setattr(
        "apmode.benchmarks.suite_c_phase1_runner._translate_reference_params",
        lambda _fix: {"CL": 2.83, "V": 32.0, "ka": 1.5},
    )

    with pytest.raises(RuntimeError, match=r"npe_score.*is None"):
        asyncio.run(
            run_fixture(
                "theophylline_boeckmann_1992",
                runner=fake_runner,  # type: ignore[arg-type]
                cache_dir=tmp_path / "cache",
                work_dir=tmp_path / "work",
                n_folds=5,
                n_sims=100,
            )
        )


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_main_returns_usage_error_on_unknown_fixture() -> None:
    rc = main(["--fixtures", "not_a_real_fixture", "--out", "/tmp/x.json"])
    assert rc == 2  # _EXIT_USAGE


def test_main_returns_usage_error_on_malformed_dataset_csv(tmp_path: Path) -> None:
    rc = main(
        [
            "--fixtures",
            "theophylline_boeckmann_1992",
            "--dataset-csv",
            "no_equals_in_this",
            "--out",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 2  # _EXIT_USAGE
