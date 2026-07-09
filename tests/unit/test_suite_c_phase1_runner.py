# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for the Suite C live-fit runner.

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
    PITCalibrationSummary,
)


def _default_pit_calibration() -> PITCalibrationSummary:
    """A well-calibrated PIT summary — good enough for tests that don't target PIT itself."""
    return PITCalibrationSummary(
        probability_levels=[0.05, 0.50, 0.95],
        calibration={"p5": 0.05, "p50": 0.50, "p95": 0.95},
        n_observations=10,
        n_subjects=2,
        aggregation="subject_robust",
    )


def _make_backend_result(
    npe: float | None,
    *,
    model_id: str = "fake_candidate",
    pit_calibration: PITCalibrationSummary | None = None,
) -> BackendResult:
    """Minimal BackendResult carrying an NPE score, mirroring suite_b.make_b3_result.

    The runner only reads ``diagnostics.npe_score`` and (as of the PIT/
    NPDE-lite reporting wire) ``diagnostics.pit_calibration``; the
    surrounding fields exist purely so the Pydantic model validates.
    Stub each required field to its smallest valid form so the test
    stays focused on the NPE-aggregation contract.
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
            pit_calibration=pit_calibration or _default_pit_calibration(),
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
            pit_calibration_apmode={"p5": 0.06, "p50": 0.49, "p95": 0.94},
            pit_calibration_literature={"p5": 0.04, "p50": 0.52, "p95": 0.97},
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
    # PIT/NPDE-lite calibration is reported alongside NPE (previously
    # computed by build_predictive_diagnostics and silently discarded).
    assert entry["pit_calibration_apmode"] == {"p5": 0.06, "p50": 0.49, "p95": 0.94}
    assert entry["pit_calibration_literature"] == {"p5": 0.04, "p50": 0.52, "p95": 0.97}
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
            pit_calibration_apmode={"p5": 0.05, "p50": 0.50, "p95": 0.95},
            pit_calibration_literature={"p5": 0.05, "p50": 0.50, "p95": 0.95},
        ),
        "mavoglurant_wendling_2015": FixturePhase1Inputs(
            fixture_id="mavoglurant_wendling_2015",
            npe_apmode=0.94,
            npe_literature=1.0,
            npe_apmode_per_fold=(0.94,) * 5,
            npe_literature_per_fold=(1.0,) * 5,
            n_subjects=14,
            n_folds=5,
            pit_calibration_apmode={"p5": 0.05, "p50": 0.50, "p95": 0.95},
            pit_calibration_literature={"p5": 0.05, "p50": 0.50, "p95": 0.95},
        ),
    }
    out = tmp_path / "phase1_npe_inputs.json"
    write_inputs_atomic(out, inputs)

    loaded = _load_inputs(out)
    assert set(loaded) == {"warfarin_funaki_2018", "mavoglurant_wendling_2015"}
    warf = loaded["warfarin_funaki_2018"]
    assert warf["npe_apmode"] == pytest.approx(0.90)
    assert warf["npe_apmode_per_fold"] == (0.88, 0.91, 0.90, 0.92, 0.89)
    assert warf["pit_calibration_apmode"] == {"p5": 0.05, "p50": 0.50, "p95": 0.95}
    assert warf["pit_calibration_literature"] == {"p5": 0.05, "p50": 0.50, "p95": 0.95}


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


def test_run_fixture_reports_median_pit_calibration_across_folds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_pk_csv: Path,
) -> None:
    """PIT/NPDE-lite calibration (previously computed and discarded) is
    now median-aggregated across folds and returned on FixturePhase1Inputs,
    independently for the APMODE and literature sides.
    """
    apmode_pit_by_fold = [
        {"p5": 0.03, "p50": 0.40, "p95": 0.90},
        {"p5": 0.05, "p50": 0.50, "p95": 0.95},
        {"p5": 0.07, "p50": 0.60, "p95": 0.99},
    ]
    literature_pit_by_fold = [
        {"p5": 0.10, "p50": 0.55, "p95": 0.85},
        {"p5": 0.12, "p50": 0.58, "p95": 0.88},
        {"p5": 0.14, "p50": 0.61, "p95": 0.91},
    ]

    def _pit(d: dict[str, float]) -> PITCalibrationSummary:
        return PITCalibrationSummary(
            probability_levels=[0.05, 0.50, 0.95],
            calibration=d,
            n_observations=10,
            n_subjects=2,
            aggregation="subject_robust",
        )

    interleaved = [
        _make_backend_result(0.95, pit_calibration=_pit(apmode_pit_by_fold[0])),
        _make_backend_result(1.00, pit_calibration=_pit(literature_pit_by_fold[0])),
        _make_backend_result(0.95, pit_calibration=_pit(apmode_pit_by_fold[1])),
        _make_backend_result(1.00, pit_calibration=_pit(literature_pit_by_fold[1])),
        _make_backend_result(0.95, pit_calibration=_pit(apmode_pit_by_fold[2])),
        _make_backend_result(1.00, pit_calibration=_pit(literature_pit_by_fold[2])),
    ]
    fake_runner = AsyncMock()
    fake_runner.run = AsyncMock(side_effect=interleaved)

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

    result = asyncio.run(
        run_fixture(
            "theophylline_boeckmann_1992",
            runner=fake_runner,  # type: ignore[arg-type]
            cache_dir=tmp_path / "cache",
            work_dir=tmp_path / "work",
            n_folds=3,
            n_sims=100,
        )
    )

    assert result.pit_calibration_apmode == {"p5": 0.05, "p50": 0.50, "p95": 0.95}
    assert result.pit_calibration_literature == {"p5": 0.12, "p50": 0.58, "p95": 0.88}


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


def test_run_fixture_drives_honest_mode_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_pk_csv: Path,
    fake_backend_result_factory: object,
) -> None:
    """Honest-mode contract: per fold the runner emits a disjoint train+test CSV pair,
    APMODE side passes ``test_data_path`` only, literature side passes both
    ``test_data_path`` and ``fixed_parameter=True``.

    These four kwargs are the load-bearing wire that turns the gate from
    a catastrophic-drift detector into a true methodology-drift detector.
    """
    make_result = fake_backend_result_factory  # callable factory
    fake_runner = AsyncMock()
    fake_runner.run = AsyncMock(side_effect=[make_result(0.95) for _ in range(10)])

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

    # 10 calls = 5 folds * (apmode + literature).
    assert fake_runner.run.await_count == 10
    apmode_calls = fake_runner.run.call_args_list[0::2]
    literature_calls = fake_runner.run.call_args_list[1::2]

    # APMODE side: held-out NPE (test_data_path set), free fit
    # (fixed_parameter omitted/False).
    for call in apmode_calls:
        assert call.kwargs.get("test_data_path") is not None
        assert call.kwargs.get("fixed_parameter", False) is False

    # Literature side: held-out NPE + fixed-THETA evaluation.
    for call in literature_calls:
        assert call.kwargs.get("test_data_path") is not None
        assert call.kwargs.get("fixed_parameter") is True

    # Per-fold train/test CSVs were emitted with disjoint subject IDs —
    # collisions would silently recycle posthoc ETAs in rxode2.
    fold_dirs = sorted((tmp_path / "work" / "theophylline_boeckmann_1992").iterdir())
    assert len(fold_dirs) == 5
    for fold_dir in fold_dirs:
        train_csv = next(fold_dir.glob("*_train.csv"))
        test_csv = next(fold_dir.glob("*_test.csv"))
        train_ids = set(pd.read_csv(train_csv)["NMID"].astype(str))
        test_ids = set(pd.read_csv(test_csv)["NMID"].astype(str))
        assert train_ids and test_ids
        assert train_ids.isdisjoint(test_ids), (
            f"fold {fold_dir.name}: train/test ID overlap = {train_ids & test_ids}"
        )

    # Train/test CSV paths in the kwargs match the on-disk fold layout.
    for call, fold_dir in zip(apmode_calls, fold_dirs, strict=True):
        assert call.kwargs["test_data_path"] == next(fold_dir.glob("*_test.csv"))


def test_run_fixture_apmode_side_never_seeded_from_literature_reference_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_pk_csv: Path,
    fake_backend_result_factory: object,
) -> None:
    """Anti-circularity regression test.

    Suite C's "methodology drift" comparison (APMODE's free fit vs. the
    literature's fixed-THETA fit) is only meaningful if the APMODE side's
    initial estimates are genuinely blind to the literature anchor for
    that fixture — otherwise the pipeline could silently hand its own
    "independent" NCA estimate the fixture's answer key, and every
    fraction-beats-literature-median number downstream would be inflated
    by construction rather than by real methodology improvement.

    Pins two invariants:
      1. ``NCAEstimator`` (``run_fixture``'s per-fold APMODE-side estimator,
         imported lazily inside the loop) is never constructed with
         ``fallback_estimates`` equal to the literature's translated
         ``reference_params`` — that would route the paper's own values
         into the estimator through its "dataset_card" fallback channel.
      2. The ``initial_estimates`` actually passed to the APMODE-side
         ``runner.run`` call differ from the literature's reference
         params for every fold.
    """
    make_result = fake_backend_result_factory  # callable factory
    fake_runner = AsyncMock()
    fake_runner.run = AsyncMock(side_effect=[make_result(0.95) for _ in range(10)])

    literature_reference = {"CL": 2.83, "V": 32.0, "ka": 1.5}

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
        lambda _fix: dict(literature_reference),
    )

    from apmode.data.initial_estimates import NCAEstimator as _RealNCAEstimator

    captured_fallback_estimates: list[dict[str, float] | None] = []

    class _SpyNCAEstimator(_RealNCAEstimator):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured_fallback_estimates.append(
                cast("dict[str, float] | None", kwargs.get("fallback_estimates"))
            )
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("apmode.data.initial_estimates.NCAEstimator", _SpyNCAEstimator)

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

    # Invariant 1: no dataset_card-style fallback seeded from the anchor.
    assert len(captured_fallback_estimates) == 5, (
        "expected NCAEstimator to be constructed once per fold (5 folds)"
    )
    for fallback in captured_fallback_estimates:
        assert fallback is None or fallback != literature_reference, (
            "NCAEstimator must not be seeded with the literature fixture's "
            "own reference_params as a fallback prior — that would let the "
            "answer key leak into the 'independent' APMODE-side estimate"
        )

    # Invariant 2: the APMODE-side fit's initial_estimates differ from the
    # literature anchor for every fold (comparing against itself would be
    # a trivial, not a genuine, methodology-drift check).
    apmode_calls = fake_runner.run.call_args_list[0::2]
    assert len(apmode_calls) == 5
    for call in apmode_calls:
        assert call.args[2] != literature_reference


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
