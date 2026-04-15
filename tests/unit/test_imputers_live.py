# SPDX-License-Identifier: GPL-2.0-or-later
"""Live integration tests for R imputers (PRD §4.2.1).

Exercises ``R_MiceImputer`` and ``R_MissRangerImputer`` against real
Rscript + mice / missRanger installs. Marked ``live`` so they are
skipped in the default fast CI path, and individually gated on package
availability.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from apmode.data.imputers import R_MiceImputer, R_MissRangerImputer


def _r_package_installed(pkg: str) -> bool:
    if not shutil.which("Rscript"):
        return False
    out = subprocess.run(
        ["Rscript", "-e", f'cat(requireNamespace("{pkg}", quietly=TRUE))'],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return out.stdout.strip() == "TRUE"


_MICE_AVAILABLE = _r_package_installed("mice")
_MISSRANGER_AVAILABLE = _r_package_installed("missRanger")


def _write_sample_csv(path: Path) -> None:
    """Tiny dataset with ~25% WT missingness at the subject level."""
    df = pd.DataFrame(
        {
            "NMID": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            "TIME": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "EVID": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "AMT": [100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0],
            "DV": [0, 5.2, 0, 4.8, 0, 6.1, 0, 4.3, 0, 5.7, 0, 5.0, 0, 4.9, 0, 5.5],
            "WT": [70, 70, 80, 80, None, None, 65, 65, 90, 90, None, None, 75, 75, 82, 82],
            "AGE": [30, 30, 45, 45, 52, 52, 28, 28, 60, 60, 41, 41, 35, 35, 48, 48],
        }
    )
    df.to_csv(path, index=False)


@pytest.mark.live
@pytest.mark.skipif(not _MICE_AVAILABLE, reason="R 'mice' package not installed")
def test_mice_imputer_produces_m_csvs(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    _write_sample_csv(src)

    imputer = R_MiceImputer(
        work_dir=tmp_path / "work",
        covariates=["WT", "AGE"],
        id_column="NMID",
    )
    paths = asyncio.run(imputer.impute(src.resolve(), m=3, seed=42))

    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        imputed = pd.read_csv(p)
        assert "WT" in imputed.columns
        # After imputation there should be no missing WT in the subjects
        # that had missing values originally.
        assert imputed["WT"].isna().sum() == 0


@pytest.mark.live
@pytest.mark.skipif(
    not _MISSRANGER_AVAILABLE,
    reason="R 'missRanger' package not installed",
)
def test_missranger_imputer_produces_m_csvs(tmp_path: Path) -> None:
    src = tmp_path / "src.csv"
    _write_sample_csv(src)

    imputer = R_MissRangerImputer(
        work_dir=tmp_path / "work",
        covariates=["WT", "AGE"],
        id_column="NMID",
    )
    paths = asyncio.run(imputer.impute(src.resolve(), m=3, seed=42))

    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        imputed = pd.read_csv(p)
        assert "WT" in imputed.columns
        assert imputed["WT"].isna().sum() == 0


@pytest.mark.live
@pytest.mark.skipif(not _MICE_AVAILABLE, reason="R 'mice' package not installed")
def test_mice_imputer_between_imputation_variance(tmp_path: Path) -> None:
    """Multiple imputation should produce different values across draws for missing subjects."""
    src = tmp_path / "src.csv"
    _write_sample_csv(src)

    imputer = R_MiceImputer(
        work_dir=tmp_path / "work",
        covariates=["WT", "AGE"],
        id_column="NMID",
    )
    paths = asyncio.run(imputer.impute(src.resolve(), m=5, seed=7))

    # Subjects 3 and 6 had missing WT — their imputed values should vary
    # across the 5 draws (not a constant). ``first()`` picks baseline per
    # subject; we compare subject 3's baseline WT across imputations.
    subj3_values = [pd.read_csv(p).query("NMID == 3").iloc[0]["WT"] for p in paths]
    assert len(set(subj3_values)) > 1, (
        "Multiple imputation produced identical values across draws — "
        f"{subj3_values}; MI is not functioning."
    )
