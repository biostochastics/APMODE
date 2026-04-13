# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NONMEM CSV data ingestion (PRD S4.2.0)."""

from pathlib import Path

import pytest

from apmode.data.ingest import ingest_nonmem_csv

FIXTURES = Path(__file__).parent.parent / "fixtures" / "pk_data"


class TestIngestNonmemCsv:
    def test_valid_csv(self) -> None:
        manifest, _df = ingest_nonmem_csv(FIXTURES / "simple_1cmt.csv")
        assert manifest.n_subjects == 2
        assert manifest.n_observations == 12  # rows with EVID=0
        assert manifest.n_doses == 2  # rows with EVID=1
        assert manifest.ingestion_format == "nonmem_csv"
        assert len(manifest.data_sha256) == 64

    def test_column_mapping(self) -> None:
        manifest, _ = ingest_nonmem_csv(FIXTURES / "simple_1cmt.csv")
        assert manifest.column_mapping.subject_id == "NMID"
        assert manifest.column_mapping.time == "TIME"
        assert manifest.column_mapping.dv == "DV"
        assert manifest.column_mapping.evid == "EVID"
        assert manifest.column_mapping.amt == "AMT"

    def test_covariate_detection(self) -> None:
        manifest, _ = ingest_nonmem_csv(FIXTURES / "simple_1cmt.csv")
        cov_names = [c.name for c in manifest.covariates]
        assert "WT" in cov_names
        assert "SEX" in cov_names
        # WT is continuous, SEX is categorical
        cov_map = {c.name: c.type for c in manifest.covariates}
        assert cov_map["WT"] == "continuous"
        assert cov_map["SEX"] == "categorical"

    def test_sha256_deterministic(self) -> None:
        m1, _ = ingest_nonmem_csv(FIXTURES / "simple_1cmt.csv")
        m2, _ = ingest_nonmem_csv(FIXTURES / "simple_1cmt.csv")
        assert m1.data_sha256 == m2.data_sha256

    def test_missing_required_column(self, tmp_path: Path) -> None:
        csv = tmp_path / "bad.csv"
        csv.write_text("ID,TIME,DV\n1,0,5.0\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            ingest_nonmem_csv(csv)

    def test_custom_column_mapping(self, tmp_path: Path) -> None:
        csv = tmp_path / "custom.csv"
        csv.write_text(
            "ID,TIME,CONC,MISS,EVENT,DOSE,COMP\n1,0.0,0.0,1,1,100.0,1\n1,1.0,5.0,0,0,0.0,1\n"
        )
        mapping = {
            "ID": "NMID",
            "CONC": "DV",
            "MISS": "MDV",
            "EVENT": "EVID",
            "DOSE": "AMT",
            "COMP": "CMT",
        }
        manifest, _df = ingest_nonmem_csv(csv, column_mapping=mapping)
        assert manifest.n_subjects == 1
        assert manifest.n_observations == 1
        assert manifest.n_doses == 1

    def test_optional_columns_detected(self, tmp_path: Path) -> None:
        csv = tmp_path / "with_optional.csv"
        csv.write_text(
            "NMID,TIME,DV,MDV,EVID,AMT,CMT,RATE,OCCASION\n"
            "1,0.0,0.0,1,1,100.0,1,0.0,1\n"
            "1,1.0,5.0,0,0,0.0,1,0.0,1\n"
        )
        manifest, _ = ingest_nonmem_csv(csv)
        assert manifest.column_mapping.rate == "RATE"
        assert manifest.column_mapping.occasion == "OCCASION"

    def test_dataframe_returned_has_correct_shape(self) -> None:
        _, df = ingest_nonmem_csv(FIXTURES / "simple_1cmt.csv")
        assert len(df) == 14  # 2 subjects * 7 rows each
        assert "NMID" in df.columns
        assert "WT" in df.columns  # covariates preserved
