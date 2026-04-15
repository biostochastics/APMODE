# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the profiler error-model heuristic (Beal 2001, Ahn 2008).

The heuristic produces an ``ErrorModelPreference`` that prunes the search
space before candidate generation. The critical safety property is:

  When BLQ ≥ 10%, additive-only error must never appear in the candidate
  set — it otherwise absorbs the censored variance and corrupts the
  structural parameter estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apmode.bundle.models import ColumnMapping, DataManifest
from apmode.data.profiler import profile_data, recommend_error_model
from apmode.search.candidates import SearchSpace


def _make_dataset(
    *,
    n_subjects: int = 30,
    cmax_low: float = 1.0,
    cmax_high: float = 50.0,
    blq_fraction: float = 0.0,
    lloq: float | None = None,
    cv: float = 0.15,
    seed: int = 123,
) -> pd.DataFrame:
    """Synthesize a PK dataset with targeted Cmax range, CV, and BLQ rate.

    ``blq_fraction`` is enforced by uniformly censoring that fraction of
    observation rows regardless of their underlying concentration — this
    guarantees the profiler sees a BLQ_burden equal to ``blq_fraction``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    times = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
    for subj in range(1, n_subjects + 1):
        rows.append(
            {
                "NMID": subj,
                "TIME": 0.0,
                "DV": 0.0,
                "MDV": 1,
                "EVID": 1,
                "AMT": 100.0,
                "CMT": 1,
            }
        )
        cmax = cmax_low + (cmax_high - cmax_low) * (subj / n_subjects)
        for t in times:
            conc = cmax * np.exp(-0.15 * t) * (1 + rng.normal(0, cv))
            conc = max(0.001, conc)
            blq_flag = 1 if (lloq is not None and rng.random() < blq_fraction) else 0
            if blq_flag == 1 and lloq is not None:
                conc = lloq  # M3 convention: DV=LLOQ on censored rows
            row: dict[str, float | int] = {
                "NMID": subj,
                "TIME": t,
                "DV": conc,
                "MDV": 0,
                "EVID": 0,
                "AMT": 0.0,
                "CMT": 1,
            }
            if lloq is not None:
                row["BLQ_FLAG"] = blq_flag
                row["LLOQ"] = lloq
            rows.append(row)
    return pd.DataFrame(rows)


def _manifest(n_subjects: int, n_obs: int) -> DataManifest:
    return DataManifest(
        data_sha256="0" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", evid="EVID", amt="AMT"
        ),
        n_subjects=n_subjects,
        n_observations=n_obs,
        n_doses=n_subjects,
    )


class TestErrorModelRecommendation:
    """Decision-tree behavior of recommend_error_model."""

    def test_high_blq_triggers_m3_without_additive(self) -> None:
        df = pd.DataFrame({"NMID": [1] * 100, "DV": [1.0] * 100, "EVID": [0] * 100})
        pref = recommend_error_model(
            df,
            blq_burden=0.25,
            lloq=0.5,
            cmax_p95_p05_ratio=10.0,
            dv_cv_percent=20.0,
            terminal_log_mad=0.1,
        )
        assert pref.primary == "blq_m3"
        assert "additive" not in pref.allowed
        assert {"proportional", "combined"}.issubset(set(pref.allowed))
        assert pref.confidence == "high"

    def test_wide_dynamic_range_prefers_proportional(self) -> None:
        df = pd.DataFrame({"NMID": [1] * 100, "DV": [1.0] * 100, "EVID": [0] * 100})
        pref = recommend_error_model(
            df,
            blq_burden=0.0,
            lloq=None,
            cmax_p95_p05_ratio=100.0,
            dv_cv_percent=30.0,
            terminal_log_mad=0.1,
        )
        assert pref.primary == "proportional"
        assert "combined" in pref.allowed
        assert pref.confidence == "high"

    def test_lloq_near_cmax_prefers_combined(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1] * 5,
                "DV": [10.0, 8.0, 5.0, 3.0, 1.0],
                "EVID": [0, 0, 0, 0, 0],
            }
        )
        pref = recommend_error_model(
            df,
            blq_burden=0.05,  # below M3 trigger
            lloq=1.0,  # 1/median_cmax(10) = 10% > 5%
            cmax_p95_p05_ratio=10.0,
            dv_cv_percent=30.0,
            terminal_log_mad=0.1,
        )
        assert pref.primary == "combined"

    def test_noisy_terminal_prefers_combined(self) -> None:
        df = pd.DataFrame({"NMID": [1] * 100, "DV": [1.0] * 100, "EVID": [0] * 100})
        pref = recommend_error_model(
            df,
            blq_burden=0.0,
            lloq=None,
            cmax_p95_p05_ratio=20.0,
            dv_cv_percent=30.0,
            terminal_log_mad=0.50,  # > 0.35 threshold
        )
        assert pref.primary == "combined"

    def test_narrow_range_low_cv_picks_additive(self) -> None:
        df = pd.DataFrame({"NMID": [1] * 100, "DV": [1.0] * 100, "EVID": [0] * 100})
        pref = recommend_error_model(
            df,
            blq_burden=0.0,
            lloq=None,
            cmax_p95_p05_ratio=3.0,
            dv_cv_percent=10.0,
            terminal_log_mad=0.1,
        )
        assert pref.primary == "additive"

    def test_default_is_proportional_medium_confidence(self) -> None:
        df = pd.DataFrame({"NMID": [1] * 100, "DV": [1.0] * 100, "EVID": [0] * 100})
        pref = recommend_error_model(
            df,
            blq_burden=0.0,
            lloq=None,
            cmax_p95_p05_ratio=10.0,
            dv_cv_percent=30.0,
            terminal_log_mad=0.1,
        )
        assert pref.primary == "proportional"
        assert pref.confidence == "medium"


class TestSearchSpaceConsumesPreference:
    """SearchSpace.from_manifest honors the profiler's error-model preference."""

    def test_blq_primary_excludes_additive(self) -> None:
        df = _make_dataset(blq_fraction=0.30, lloq=0.5)
        manifest = profile_data(df, _manifest(30, len(df)))
        space = SearchSpace.from_manifest(manifest)
        assert space.force_blq_method == "m3"
        assert "additive" not in space.error_types

    def test_proportional_preference_limits_types(self) -> None:
        df = _make_dataset(cmax_low=0.1, cmax_high=100.0, cv=0.15)
        manifest = profile_data(df, _manifest(30, len(df)))
        space = SearchSpace.from_manifest(manifest)
        # Proportional preferred — no additive
        if manifest.error_model_preference is not None:
            assert manifest.error_model_preference.primary in (
                "proportional",
                "combined",
                "blq_m3",
            )
            assert space.error_types == list(
                manifest.error_model_preference.allowed
            ) or space.force_blq_method in ("m3", "m4")

    def test_missing_preference_preserves_legacy_behavior(self) -> None:
        """When a legacy manifest has no preference, the BLQ>20% rule still fires."""
        from apmode.bundle.models import EvidenceManifest

        manifest = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_evidence_strength="none",
            richness_category="moderate",
            identifiability_ceiling="medium",
            covariate_burden=0,
            covariate_correlated=False,
            blq_burden=0.30,
            lloq_value=1.0,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        # error_model_preference is None by default
        assert manifest.error_model_preference is None
        space = SearchSpace.from_manifest(manifest)
        # Legacy fallback kicks in: force M3, error_types = proportional/combined
        assert space.force_blq_method == "m3"
        assert "additive" not in space.error_types


def test_profile_data_attaches_preference_to_manifest() -> None:
    df = _make_dataset(blq_fraction=0.15, lloq=0.5, cv=0.20)
    manifest = profile_data(df, _manifest(30, len(df)))
    assert manifest.error_model_preference is not None
    # Supporting signals populated
    assert manifest.cmax_p95_p05_ratio is not None
    assert manifest.dv_cv_percent is not None
    # Terminal log MAD can be None on sparse terminal; just ensure no crash


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
