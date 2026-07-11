# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NCA-based initial estimates (PRD §4.2.0.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apmode.bundle.models import (
    ColumnMapping,
    DataManifest,
    InitialEstimateEntry,
    InitialEstimates,
)
from apmode.data.ingest import ingest_nonmem_csv
from apmode.data.initial_estimates import (
    NCAEstimator,
    _compute_nca_single_subject,
    _default_estimates,
    build_initial_estimates_bundle,
    warm_start_estimates,
)
from apmode.data.types import is_steady_state
from tests._helpers.fixtures_data import FIXTURES_DIR

FIXTURE_CSV = FIXTURES_DIR / "pk_data" / "simple_1cmt.csv"


def _manifest_for(df: pd.DataFrame) -> DataManifest:
    return DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", evid="EVID", amt="AMT", mdv="MDV"
        ),
        n_subjects=int(df["NMID"].nunique()),
        n_observations=int(((df["EVID"] == 0) & (df["MDV"] == 0)).sum()),
        n_doses=int(df["EVID"].isin([1, 4]).sum()),
    )


class TestNCAEstimator:
    """NCA-based initial estimate derivation."""

    def test_estimate_per_subject_returns_dict(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_per_subject()
        assert isinstance(result, dict)
        assert "CL" in result
        assert "V" in result
        assert "ka" in result

    def test_estimates_are_positive(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_per_subject()
        for key, val in result.items():
            if key.startswith("_"):
                continue  # skip metadata keys like _auc_extrap_high_fraction
            assert val > 0, f"{key} should be positive, got {val}"

    def test_estimates_in_reasonable_range(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_per_subject()
        # For our test data: 1-cmt oral, ka~1.5, V~70, CL~5
        assert 0.1 < result["CL"] < 100
        assert 1.0 < result["V"] < 1000
        assert 0.1 < result["ka"] < 50

    def test_population_level_fallback(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        result = estimator.estimate_population_level()
        assert isinstance(result, dict)
        assert all(v > 0 for v in result.values())

    def test_build_entry(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        estimator = NCAEstimator(df, manifest)
        entry = estimator.build_entry("candidate_001")
        assert isinstance(entry, InitialEstimateEntry)
        assert entry.candidate_id == "candidate_001"
        assert entry.source == "nca"
        assert len(entry.estimates) > 0

    def test_constructor_excludes_mdv_rows_and_includes_evid4_doses(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [0.0, 1.0, 2.0],
                "DV": [0.0, 5.0, 9999.0],
                "MDV": [1, 0, 1],
                "EVID": [4, 0, 0],
                "AMT": [100.0, 0.0, 0.0],
                "CMT": [1, 1, 1],
            }
        )
        estimator = NCAEstimator(df, _manifest_for(df))
        assert estimator._doses["EVID"].tolist() == [4]
        assert estimator._obs["DV"].tolist() == [5.0]

    def test_multiple_doses_without_ss_are_not_treated_as_steady_state(self) -> None:
        rows = [
            {
                "NMID": 1,
                "TIME": 0.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 1,
                "AMT": 100.0,
                "CMT": 1,
            },
            {
                "NMID": 1,
                "TIME": 12.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 1,
                "AMT": 100.0,
                "CMT": 1,
            },
        ]
        for time, dv in [(12.5, 2.0), (13.0, 5.0), (14.0, 3.0), (16.0, 1.0), (20.0, 0.3)]:
            rows.append(
                {
                    "NMID": 1,
                    "TIME": time,
                    "DV": dv,
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                }
            )
        df = pd.DataFrame(rows)
        result = NCAEstimator(df, _manifest_for(df))._nca_for_subject(1)
        assert result.excluded is True
        assert result.excluded_reason is not None
        assert "without an explicit steady-state SS flag" in result.excluded_reason

    def test_tmax_is_measured_from_nonzero_dose_time(self) -> None:
        rows = [
            {
                "NMID": 1,
                "TIME": 8.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 4,
                "AMT": 100.0,
                "CMT": 1,
            }
        ]
        for time, dv in [(8.5, 2.0), (9.0, 5.0), (10.0, 3.0), (12.0, 1.0), (16.0, 0.3)]:
            rows.append(
                {
                    "NMID": 1,
                    "TIME": time,
                    "DV": dv,
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                }
            )
        df = pd.DataFrame(rows)
        result = NCAEstimator(df, _manifest_for(df))._nca_for_subject(1)
        assert result.tmax == pytest.approx(1.0)


def test_dose_count_alone_does_not_override_half_life_criterion() -> None:
    steady, rationale = is_steady_state(
        np.array([0.0, 24.0, 48.0, 72.0, 96.0]),
        np.repeat(100.0, 5),
        half_life=72.0,
    )
    assert steady is False
    assert "ss_by_t12=False" in rationale


class TestComputeNCASingleSubject:
    """Unit tests for the core NCA computation."""

    def test_simple_monoexponential(self) -> None:
        # C(t) = 10 * exp(-0.2 * t), Dose=100, V=10, CL=2, kel=0.2
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        concs = 10.0 * np.exp(-0.2 * times)
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        assert result is not None
        assert result.cl > 0
        assert result.v > 0
        assert result.ka > 0
        assert result.kel > 0
        # kel should be approximately 0.2
        assert abs(result.kel - 0.2) < 0.1
        # AUC extrapolation fraction should be reasonable
        assert 0.0 <= result.auc_extrap_fraction <= 1.0

    def test_returns_none_for_insufficient_data(self) -> None:
        times = np.array([1.0, 2.0])
        concs = np.array([5.0, 3.0])
        # Only 2 points — not enough for a terminal-phase regression (min 3
        # post-Tmax points), so lambda_z selection fails and computation cannot
        # proceed: the function must return None, not a fabricated estimate.
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        assert result is None

    def test_returns_none_for_zero_auc(self) -> None:
        times = np.array([1.0, 2.0, 3.0])
        concs = np.array([0.0, 0.0, 0.0])
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        assert result is None

    def test_sanity_bounds(self) -> None:
        """Results should be bounded within physiological ranges."""
        times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        concs = np.array([2.0, 5.0, 3.0, 1.0, 0.3])
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        if result is not None:
            assert 0.01 <= result.cl <= 10000
            assert 0.1 <= result.v <= 100000
            assert 0.01 <= result.ka <= 100

    def test_multi_dose_steady_state(self) -> None:
        """Multi-dose AUC_tau estimation at steady state."""
        # Simulate steady-state: concentrations within one dosing interval (tau=12h)
        times = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0])
        concs = np.array([2.0, 8.0, 10.0, 7.0, 4.0, 1.5, 0.8])
        result = _compute_nca_single_subject(
            times, concs, dose=100.0, is_steady_state=True, tau=12.0
        )
        assert result is not None
        assert result.cl > 0
        assert result.auc_extrap_fraction == 0.0  # no extrapolation for AUC_tau

    def test_high_auc_extrapolation_flagged(self) -> None:
        """A slowly-declining tail → large AUC extrapolation → QC exclusion."""
        # Long profile whose terminal phase barely declines: C_last is large
        # relative to AUC_last, so AUC_inf is dominated by the C_last/lambda_z
        # extrapolation term. This must produce a computable NCAResult that the
        # QC gate flags (extrap > 20%), not a silently-accepted estimate.
        times = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        concs = np.array([1.0, 10.0, 9.5, 9.0, 8.5, 8.0])
        result = _compute_nca_single_subject(times, concs, dose=100.0)
        assert result is not None
        # Extrapolation must be a large, well-defined fraction (>20% gate).
        assert result.auc_extrap_fraction > 0.20
        # ...and the QC gate must flag it rather than silently accept it.
        assert result.excluded
        assert result.excluded_reason is not None
        assert "AUC_extrap" in result.excluded_reason


class TestWarmStartEstimates:
    """Warm-start from parent model parameters."""

    def test_inherits_parent_params(self) -> None:
        parent = {"CL": 5.0, "V": 70.0, "ka": 1.5}
        entry = warm_start_estimates(parent, "child_001")
        assert entry.source == "warm_start"
        assert entry.estimates == parent
        assert entry.candidate_id == "child_001"


class TestBuildBundle:
    """Building the initial_estimates.json artifact."""

    def test_builds_from_entries(self) -> None:
        entries = [
            InitialEstimateEntry(
                candidate_id="c1",
                source="nca",
                estimates={"CL": 5.0, "V": 70.0},
            ),
            InitialEstimateEntry(
                candidate_id="c2",
                source="warm_start",
                estimates={"CL": 6.0, "V": 65.0},
            ),
        ]
        bundle = build_initial_estimates_bundle(entries)
        assert isinstance(bundle, InitialEstimates)
        assert "c1" in bundle.entries
        assert "c2" in bundle.entries
        assert bundle.entries["c1"].source == "nca"

    def test_roundtrip_json(self) -> None:
        entries = [
            InitialEstimateEntry(
                candidate_id="c1",
                source="nca",
                estimates={"CL": 5.0},
            ),
        ]
        bundle = build_initial_estimates_bundle(entries)
        json_str = bundle.model_dump_json()
        roundtrip = InitialEstimates.model_validate_json(json_str)
        assert roundtrip.entries["c1"].estimates["CL"] == pytest.approx(5.0)


class TestDefaultEstimates:
    """Fallback estimates."""

    def test_returns_positive_values(self) -> None:
        defaults = _default_estimates()
        assert all(v > 0 for v in defaults.values())

    def test_contains_required_keys(self) -> None:
        defaults = _default_estimates()
        assert "CL" in defaults
        assert "V" in defaults
        assert "ka" in defaults


class TestNCADVIDFilter:
    """Mixed-endpoint datasets must filter to PK rows before NCA.

    Without the filter, datasets like warfarin (DVID="cp" PK rows
    interleaved with DVID="pca" PD rows) feed PD concentrations into
    the per-subject lambda_z regression. The terminal-slope fits then
    collapse for the majority of subjects, the >50% exclusion gate
    fires, and the estimator falls all the way to the conservative
    defaults (CL=5/V=70/ka=1) — values that are 47x off for warfarin
    and force SAEM+FOCEI to traverse 4 OoMs before convergence.
    """

    def _build_pk_pd_frame(self) -> pd.DataFrame:
        """Build a synthetic 6-subject PK+PD dataframe.

        Each subject gets one dose row + 6 PK observations (oral 1-cmt
        kinetics with CL=2 L/h, V=10 L) + 6 PD observations on a
        completely different scale (10-100 IU/mL) so the per-subject
        lambda_z regression on the combined PK+PD set is guaranteed to
        fail QC. The PK rows alone produce a clean monoexponential
        terminal slope.
        """
        import pandas as pd  # local import to avoid module-load cost

        rows: list[dict[str, object]] = []
        for sid in range(1, 7):
            rows.append(
                {
                    "NMID": sid,
                    "TIME": 0.0,
                    "DV": 0.0,
                    "AMT": 100.0,
                    "EVID": 1,
                    "MDV": 1,
                    "CMT": 1,
                    "DVID": "cp",
                }
            )
            for t, c in [(0.5, 8.0), (1.0, 7.0), (2.0, 5.0), (4.0, 3.0), (8.0, 1.5), (12.0, 0.7)]:
                rows.append(
                    {
                        "NMID": sid,
                        "TIME": t,
                        "DV": float(c),
                        "AMT": 0.0,
                        "EVID": 0,
                        "MDV": 0,
                        "CMT": 2,
                        "DVID": "cp",
                    }
                )
            # PD rows on a different scale — unfiltered NCA blows up here.
            for t, c in [
                (0.5, 60.0),
                (1.0, 80.0),
                (2.0, 100.0),
                (4.0, 90.0),
                (8.0, 70.0),
                (12.0, 50.0),
            ]:
                rows.append(
                    {
                        "NMID": sid,
                        "TIME": t,
                        "DV": float(c),
                        "AMT": 0.0,
                        "EVID": 0,
                        "MDV": 0,
                        "CMT": 3,
                        "DVID": "pca",
                    }
                )
        return pd.DataFrame(rows)

    def test_pk_only_filter_recovers_per_subject_nca(self) -> None:
        from apmode.bundle.models import ColumnMapping, DataManifest

        df = self._build_pk_pd_frame()
        manifest = DataManifest(
            data_sha256="d" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
                mdv="MDV",
            ),
            n_subjects=int(df["NMID"].nunique()),
            n_observations=int((df["EVID"] == 0).sum()),
            n_doses=int((df["EVID"] == 1).sum()),
        )
        est = NCAEstimator(df, manifest)
        result = est.estimate_per_subject()
        assert est.fallback_source == "nca", (
            f"DVID-filtered NCA must succeed on a clean PK+PD synthetic "
            f"frame; got fallback_source={est.fallback_source}, result={result}"
        )
        # CL ~= Dose/AUC; for this profile AUC_inf ~= 50 mg.h/L so CL ~= 2 L/h.
        # Just check order-of-magnitude — the unit test pins the filter, not
        # the NCA arithmetic (which has its own dedicated tests).
        assert 0.1 < result["CL"] < 50.0
        assert 0.1 < result["V"] < 1000.0

    def test_dvid_absent_is_no_op(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        assert "DVID" not in df.columns, "fixture must not have DVID for this test"
        est = NCAEstimator(df, manifest)
        result = est.estimate_per_subject()
        # Behaviour identical to the pre-filter baseline — succeeds on the
        # canonical 1-cmt fixture.
        assert est.fallback_source == "nca"
        assert "CL" in result and result["CL"] > 0


class TestDataDrivenFallback:
    """When per-subject NCA fails QC and no dataset_card prior exists,
    the estimator now derives CL/V/ka from observed Cmax/AUC instead of
    the rigid CL=5/V=70/ka=1 defaults. Pin the cascade order so a
    future regression cannot silently revert to the old hard-coded
    fallback (which was 47x off on warfarin and triggered the 600s
    per-fit timeout that motivated this work).
    """

    def _force_qc_failure_frame(self) -> pd.DataFrame:
        """Two-subject sparse frame: 2 obs each, no terminal slope.

        Per-subject NCA needs >=3 lambda_z points; this frame has 2 per
        subject so every subject is excluded → ``_apply_fallback``
        fires, and with no dataset_card prior the estimator should
        choose ``data_driven`` over ``defaults``.
        """
        import pandas as pd

        rows: list[dict[str, object]] = []
        for sid in (1, 2):
            rows.append(
                {"NMID": sid, "TIME": 0.0, "DV": 0.0, "AMT": 100.0, "EVID": 1, "MDV": 1, "CMT": 1}
            )
            rows.append(
                {"NMID": sid, "TIME": 1.0, "DV": 8.0, "AMT": 0.0, "EVID": 0, "MDV": 0, "CMT": 2}
            )
            rows.append(
                {"NMID": sid, "TIME": 2.0, "DV": 6.0, "AMT": 0.0, "EVID": 0, "MDV": 0, "CMT": 2}
            )
        return pd.DataFrame(rows)

    def test_data_driven_beats_hardcoded_defaults(self) -> None:
        from apmode.bundle.models import ColumnMapping, DataManifest

        df = self._force_qc_failure_frame()
        manifest = DataManifest(
            data_sha256="e" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
                mdv="MDV",
            ),
            n_subjects=int(df["NMID"].nunique()),
            n_observations=int((df["EVID"] == 0).sum()),
            n_doses=int((df["EVID"] == 1).sum()),
        )
        est = NCAEstimator(df, manifest)
        result = est.estimate_per_subject()
        assert est.fallback_source == "data_driven", (
            f"sparse data with no NCA + no dataset_card prior must use "
            f"data-driven fallback (not hard-coded defaults); got "
            f"fallback_source={est.fallback_source}"
        )
        # Dose=100, Cmax_geomean=8 → V ~= 12.5 (not the 70 default).
        # Dose=100, AUC_obs (trapezoid 1->2 over 8,6) = 7.0 → CL ~= 14.
        assert result["V"] != 70.0
        assert result["CL"] != 5.0
        # Within the data-driven floors/caps.
        assert 1.0 <= result["V"] <= 1000.0
        assert 0.01 <= result["CL"] <= 500.0
        assert 0.05 <= result["ka"] <= 12.0

    def test_dataset_card_still_wins_over_data_driven(self) -> None:
        from apmode.bundle.models import ColumnMapping, DataManifest

        df = self._force_qc_failure_frame()
        manifest = DataManifest(
            data_sha256="f" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
                mdv="MDV",
            ),
            n_subjects=int(df["NMID"].nunique()),
            n_observations=int((df["EVID"] == 0).sum()),
            n_doses=int((df["EVID"] == 1).sum()),
        )
        est = NCAEstimator(df, manifest, fallback_estimates={"CL": 0.5, "V": 25.0, "ka": 0.7})
        result = est.estimate_per_subject()
        assert est.fallback_source == "dataset_card"
        # Literature priors carried verbatim (cascade short-circuits before
        # the data-driven path).
        assert result["CL"] == 0.5
        assert result["V"] == 25.0
        assert result["ka"] == 0.7
