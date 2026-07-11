# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Data Profiler → Evidence Manifest (PRD §4.2.1).

Validates that the profiler correctly analyzes PK data and produces
typed EvidenceManifest fields that constrain dispatch.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from apmode.bundle.models import ColumnMapping, CovariateMetadata, DataManifest, EvidenceManifest
from apmode.data.ingest import ingest_nonmem_csv
from apmode.data.profiler import (
    _assess_absorption_coverage,
    _assess_elimination_coverage,
    _assess_route_certainty,
    _classify_richness,
    _compute_blq_burden,
    _detect_nonlinear_clearance,
    _spearman_r,
    _subject_covariate_first_values,
    profile_data,
)
from tests._helpers.fixtures_data import FIXTURES_DIR

FIXTURE_CSV = FIXTURES_DIR / "pk_data" / "simple_1cmt.csv"


class TestProfileData:
    """Integration tests for the full profiler pipeline."""

    def test_returns_evidence_manifest(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        assert isinstance(em, EvidenceManifest)

    def test_sha256_matches_data_manifest(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        assert em.data_sha256 == manifest.data_sha256

    def test_route_certainty_for_simple_oral(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        # CMT=1 without RATE/DUR is inferred (could be oral or IV bolus)
        assert em.route_certainty == "inferred"

    def test_richness_category(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        # 2 subjects, 6 obs each = 6 obs/subject = moderate
        assert em.richness_category == "moderate"

    def test_covariate_burden(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        assert em.covariate_burden == 2  # WT and SEX

    def test_blq_burden_zero_when_no_blq(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        assert em.blq_burden == 0.0

    def test_protocol_heterogeneity_single_study(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        assert em.protocol_heterogeneity == "single-study"

    def test_absorption_phase_coverage(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        assert em.absorption_phase_coverage == "adequate"

    def test_elimination_phase_coverage(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        assert em.elimination_phase_coverage == "adequate"

    def test_manifest_is_frozen(self) -> None:
        manifest, df = ingest_nonmem_csv(FIXTURE_CSV)
        em = profile_data(df, manifest)
        with pytest.raises(ValidationError):
            em.richness_category = "rich"  # type: ignore[misc]

    def test_subject_covariate_first_values_skip_leading_missing(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 2, 2],
                "TIME": [0.0, 1.0, 0.0, 1.0],
                "WT": [None, 70.0, None, None],
            }
        )
        manifest = DataManifest(
            data_sha256="0" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
            ),
            n_subjects=2,
            n_observations=2,
            n_doses=1,
            covariates=[CovariateMetadata(name="WT", type="continuous")],
        )

        subj_covs = _subject_covariate_first_values(df, manifest)

        assert subj_covs.loc[1, "WT"] == 70.0
        assert pd.isna(subj_covs.loc[2, "WT"])


class TestRichnessClassification:
    """Per PRD §4.2.1: sparse < 4, moderate 4-8, rich > 8."""

    def test_sparse(self) -> None:
        obs = pd.DataFrame(
            {
                "NMID": [1, 1, 2, 2],
                "TIME": [1.0, 2.0, 1.0, 2.0],
                "DV": [1.0, 2.0, 1.0, 2.0],
                "EVID": [0, 0, 0, 0],
            }
        )
        assert _classify_richness(obs, 2) == "sparse"

    def test_moderate(self) -> None:
        obs = pd.DataFrame(
            {
                "NMID": [1] * 6 + [2] * 6,
                "TIME": list(range(6)) * 2,
                "DV": [1.0] * 12,
                "EVID": [0] * 12,
            }
        )
        assert _classify_richness(obs, 2) == "moderate"

    def test_rich(self) -> None:
        obs = pd.DataFrame(
            {
                "NMID": [1] * 10 + [2] * 10,
                "TIME": list(range(10)) * 2,
                "DV": [1.0] * 20,
                "EVID": [0] * 20,
            }
        )
        assert _classify_richness(obs, 2) == "rich"


class TestRouteCertainty:
    """Route certainty assessment from dosing records."""

    def test_oral_depot_inferred(self) -> None:
        doses = pd.DataFrame(
            {
                "CMT": [1, 1],
                "EVID": [1, 1],
                "AMT": [100, 200],
            }
        )
        # CMT=1 alone is not sufficient for confirmed — could be central
        assert _assess_route_certainty(doses) == "inferred"

    def test_iv_with_rate_confirmed(self) -> None:
        doses = pd.DataFrame(
            {
                "CMT": [2, 2],
                "EVID": [1, 1],
                "AMT": [100, 200],
                "RATE": [50.0, 100.0],
            }
        )
        assert _assess_route_certainty(doses) == "confirmed"

    def test_mixed_cmt_inferred(self) -> None:
        doses = pd.DataFrame(
            {
                "CMT": [1, 2],
                "EVID": [1, 1],
                "AMT": [100, 200],
            }
        )
        assert _assess_route_certainty(doses) == "inferred"

    def test_empty_doses_ambiguous(self) -> None:
        doses = pd.DataFrame(columns=["CMT", "EVID", "AMT"])
        assert _assess_route_certainty(doses) == "ambiguous"


class TestBLQBurden:
    """BLQ burden computation."""

    def test_no_blq_column(self) -> None:
        df = pd.DataFrame(
            {
                "EVID": [0, 0, 0, 1],
                "DV": [1.0, 2.0, 3.0, 0.0],
            }
        )
        assert _compute_blq_burden(df) == 0.0

    def test_some_blq(self) -> None:
        df = pd.DataFrame(
            {
                "EVID": [0, 0, 0, 0, 1],
                "DV": [1.0, 0.1, 2.0, 0.1, 100.0],
                "BLQ_FLAG": [0, 1, 0, 1, 0],
            }
        )
        # 2 BLQ out of 4 observations = 0.5
        assert _compute_blq_burden(df) == 0.5


class TestAbsorptionCoverage:
    """Absorption phase coverage assessment."""

    def test_adequate_coverage(self) -> None:
        obs = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1, 1],
                "TIME": [0.5, 1.0, 2.0, 4.0, 8.0],
                "DV": [1.0, 3.0, 5.0, 3.0, 1.0],
                "EVID": [0, 0, 0, 0, 0],
            }
        )
        # Tmax at TIME=2.0, 2 pre-Tmax obs
        assert _assess_absorption_coverage(obs) == "adequate"

    def test_inadequate_coverage(self) -> None:
        obs = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [4.0, 8.0, 12.0],
                "DV": [5.0, 3.0, 1.0],
                "EVID": [0, 0, 0],
            }
        )
        # Tmax at TIME=4.0, 0 pre-Tmax obs
        assert _assess_absorption_coverage(obs) == "inadequate"


class TestEliminationCoverage:
    """Elimination phase coverage assessment."""

    def test_adequate_coverage(self) -> None:
        obs = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1, 1],
                "TIME": [0.5, 1.0, 2.0, 4.0, 8.0],
                "DV": [1.0, 5.0, 3.0, 2.0, 1.0],
                "EVID": [0, 0, 0, 0, 0],
            }
        )
        # Tmax at TIME=1.0, 3 post-Tmax obs
        assert _assess_elimination_coverage(obs) == "adequate"


class TestNonlinearClearance:
    """Nonlinear clearance detection."""

    def test_linear_clearance_same_dose(self) -> None:
        # All subjects get same dose → dose-normalized AUC constant → no nonlinear signal
        import numpy as np

        rng = np.random.default_rng(42)
        n_subj = 10
        times = [0.5, 1, 2, 4, 8]
        rows: list[dict[str, object]] = []
        dose_rows: list[dict[str, object]] = []
        for subj in range(1, n_subj + 1):
            scale = rng.lognormal(0, 0.3)  # IIV
            dose_rows.append({"NMID": subj, "TIME": 0.0, "DV": 0.0, "EVID": 1, "AMT": 100.0})
            for t in times:
                cp = scale * 5.0 * np.exp(-0.2 * t)  # linear clearance
                rows.append({"NMID": subj, "TIME": t, "DV": float(cp), "EVID": 0, "AMT": 0.0})
        obs = pd.DataFrame(rows)
        doses = pd.DataFrame(dose_rows)
        result = _detect_nonlinear_clearance(obs, doses)
        # Linear clearance + identical dose across subjects → dose-normalized
        # AUC is constant, so there is no nonlinear signal to detect.
        assert result is False

    def test_too_few_subjects(self) -> None:
        obs = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [1, 2, 4],
                "DV": [2, 1, 0.5],
                "EVID": [0] * 3,
            }
        )
        doses = pd.DataFrame(
            {"NMID": [1], "TIME": [0.0], "DV": [0.0], "EVID": [1], "AMT": [100.0]}
        )
        assert _detect_nonlinear_clearance(obs, doses) is False


class TestTMDDScreen:
    """Saturable / TMDD-elimination screening (curvature-ratio threshold).

    TMDD (target-mediated drug disposition) and other saturable-clearance
    profiles are concave-up in log-space: the early post-Cmax decline is
    shallow (target/enzyme saturated at high concentration → slow linear
    elimination) while the terminal decline is steep (high-affinity
    target-mediated clearance dominates once free drug drops). This inverts
    the linear/2-cmt signature (steep alpha, shallow beta). The screen flags
    ``possible`` when the population-median early/late log-slope ratio falls
    BELOW ``_NONLINEAR_TMDD_CURVATURE_RATIO`` (0.3).
    """

    @staticmethod
    def _manifest(n_subjects: int) -> DataManifest:
        return DataManifest(
            data_sha256="0" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
            ),
            n_subjects=n_subjects,
            n_observations=n_subjects,
            n_doses=n_subjects,
            covariates=[],
        )

    @staticmethod
    def _build(
        conc_of: object,
        n_subj: int = 6,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        import numpy as np

        rng = np.random.default_rng(7)
        times = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        obs_rows: list[dict[str, object]] = []
        dose_rows: list[dict[str, object]] = []
        for subj in range(1, n_subj + 1):
            scale = float(rng.lognormal(0.0, 0.15))  # mild IIV
            dose_rows.append(
                {"NMID": subj, "TIME": 0.0, "DV": 0.0, "EVID": 1, "AMT": 100.0, "CMT": 1}
            )
            for t in times:
                cp = scale * conc_of(t)  # type: ignore[operator]
                obs_rows.append(
                    {"NMID": subj, "TIME": t, "DV": float(cp), "EVID": 0, "AMT": 0.0, "CMT": 1}
                )
        return pd.DataFrame(obs_rows + dose_rows), pd.DataFrame([])

    def test_saturable_profile_trips_screen(self) -> None:
        import numpy as np

        # Concave-up: shallow early decline, steep terminal decline.
        def conc(t: float) -> float:
            if t <= 4.0:
                return float(10.0 * np.exp(-0.05 * t))  # shallow early
            return float(10.0 * np.exp(-0.05 * 4.0) * np.exp(-0.6 * (t - 4.0)))  # steep late

        df, _ = self._build(conc)
        manifest = self._manifest(n_subjects=6)
        em = profile_data(df, manifest)
        assert em.tmdd_screen == "possible"

    def test_linear_profile_does_not_trip_screen(self) -> None:
        import numpy as np

        # Monoexponential: early slope ≈ late slope → ratio ≈ 1 (> 0.3).
        def conc(t: float) -> float:
            return float(10.0 * np.exp(-0.2 * t))

        df, _ = self._build(conc)
        manifest = self._manifest(n_subjects=6)
        em = profile_data(df, manifest)
        assert em.tmdd_screen == "none"

    def test_too_few_subjects_is_unknown(self) -> None:
        import numpy as np

        from apmode.data.profiler import _screen_tmdd_saturable

        def conc(t: float) -> float:
            if t <= 4.0:
                return float(10.0 * np.exp(-0.05 * t))
            return float(10.0 * np.exp(-0.05 * 4.0) * np.exp(-0.6 * (t - 4.0)))

        df, _ = self._build(conc, n_subj=2)
        obs = df[df["EVID"] == 0].copy()
        doses = df[df["EVID"] == 1].copy()
        assert _screen_tmdd_saturable(obs, doses, multi_dose=False) == "unknown"


class TestSpearmanR:
    """Spearman rank correlation utility."""

    def test_perfect_correlation(self) -> None:
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert _spearman_r(x, y) == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative_correlation(self) -> None:
        import numpy as np

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        assert _spearman_r(x, y) == pytest.approx(-1.0, abs=1e-10)

    def test_too_few_values(self) -> None:
        import numpy as np

        assert _spearman_r(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
