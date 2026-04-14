# SPDX-License-Identifier: GPL-2.0-or-later
"""Regression test: BLQ perturbation must propagate LLOQ through pipeline.

Verifies the integration chain:
    Perturbation (CENS/BLQ_FLAG/LLOQ columns)
      -> Ingestion (preserves BLQ_FLAG in DataFrame)
      -> Profiler (blq_burden + lloq_value in EvidenceManifest)
      -> SearchSpace (force_blq_method + lloq_value)
      -> Candidates (BLQ_M3(loq_value=actual_lloq))

Prior bug: SearchSpace.from_manifest set force_blq_method="m3" but left
lloq_value at default 1.0, causing Gate 1 failures on real BLQ data when
actual LLOQ was much higher (e.g., 32.8 on mavoglurant).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apmode.benchmarks.models import PerturbationRecipe, PerturbationType
from apmode.benchmarks.perturbations import apply_perturbation
from apmode.data.ingest import ingest_nonmem_csv
from apmode.data.profiler import profile_data
from apmode.search.candidates import SearchSpace, generate_root_candidates


def _make_synthetic_pk_data(n_subjects: int = 50) -> pd.DataFrame:
    """Build a synthetic NONMEM-format DataFrame for BLQ testing."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int]] = []
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
        for t in [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]:
            conc = 50.0 * np.exp(-0.1 * t) * (1 - np.exp(-1.5 * t))
            conc *= 1 + rng.normal(0, 0.15)
            rows.append(
                {
                    "NMID": subj,
                    "TIME": t,
                    "DV": max(0.01, conc),
                    "MDV": 0,
                    "EVID": 0,
                    "AMT": 0.0,
                    "CMT": 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.integration
class TestBLQPipelineIntegration:
    """End-to-end BLQ propagation test."""

    def test_high_blq_burden_forces_m3_with_correct_lloq(self, tmp_path: Path) -> None:
        """30% BLQ (auto-computed LLOQ) → all candidates use BLQ_M3 with that LLOQ."""
        df = _make_synthetic_pk_data(n_subjects=60)

        # Use blq_fraction only — let the perturbation compute LLOQ from 30th percentile
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.30,
            seed=42,
        )
        perturbed, manifest = apply_perturbation(df, recipe)
        computed_lloq = manifest["lloq"]

        # Write to CSV and go through full ingestion pipeline
        csv_path = tmp_path / "blq_test.csv"
        perturbed.to_csv(csv_path, index=False)

        data_manifest, df_ingested = ingest_nonmem_csv(csv_path)
        evidence = profile_data(df_ingested, data_manifest)

        # BLQ burden should be detected (> 20% threshold)
        assert evidence.blq_burden > 0.20
        assert evidence.lloq_value == pytest.approx(computed_lloq, rel=1e-6)

        # SearchSpace must propagate BOTH force_blq_method AND lloq_value
        space = SearchSpace.from_manifest(evidence)
        assert space.force_blq_method == "m3"
        assert space.lloq_value == pytest.approx(computed_lloq, rel=1e-6)

        # All generated candidates must use BLQ_M3 with the correct LLOQ
        candidates = generate_root_candidates(space, {"ka": 1.0, "CL": 5.0, "V": 70.0})
        assert len(candidates) > 0

        for cand in candidates:
            assert cand.observation.type == "BLQ_M3", (
                f"Candidate {cand.model_id} should use BLQ_M3, got {cand.observation.type}"
            )
            assert cand.observation.loq_value == pytest.approx(computed_lloq, rel=1e-6), (
                f"Candidate {cand.model_id} has wrong LLOQ: "
                f"{cand.observation.loq_value} != {computed_lloq}"
            )

    def test_low_blq_burden_does_not_force_m3(self, tmp_path: Path) -> None:
        """10% BLQ (below 20% threshold) → no BLQ forcing."""
        df = _make_synthetic_pk_data(n_subjects=60)

        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.10,
            seed=42,
        )
        perturbed, _ = apply_perturbation(df, recipe)

        csv_path = tmp_path / "blq_low.csv"
        perturbed.to_csv(csv_path, index=False)

        data_manifest, df_ingested = ingest_nonmem_csv(csv_path)
        evidence = profile_data(df_ingested, data_manifest)

        assert evidence.blq_burden < 0.20  # Below threshold

        space = SearchSpace.from_manifest(evidence)
        assert space.force_blq_method is None
        # lloq_value stays at default when force_blq_method is not triggered
        assert space.lloq_value == 1.0

    def test_profiler_fallback_to_dv_when_lloq_column_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """When BLQ_FLAG=1 but no LLOQ column, use DV of censored rows."""
        df = _make_synthetic_pk_data(n_subjects=60)

        # Apply BLQ but then drop the LLOQ column to simulate legacy datasets
        recipe = PerturbationRecipe(
            perturbation_type=PerturbationType.INJECT_BLQ,
            blq_fraction=0.30,
            lloq=7.5,
            seed=42,
        )
        perturbed, _ = apply_perturbation(df, recipe)
        perturbed_no_lloq = perturbed.drop(columns=["LLOQ"])

        csv_path = tmp_path / "blq_no_lloq_col.csv"
        perturbed_no_lloq.to_csv(csv_path, index=False)

        data_manifest, df_ingested = ingest_nonmem_csv(csv_path)
        evidence = profile_data(df_ingested, data_manifest)

        # Profiler falls back to DV of BLQ rows (which was set to LLOQ by M3 convention)
        assert evidence.lloq_value == 7.5
