# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for data format adapters (PRD S4.2.0)."""

import pandas as pd

from apmode.data.adapters import to_nlmixr2_format


class TestToNlmixr2Format:
    def test_nmid_renamed_to_id(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1],
                "TIME": [0.0, 1.0],
                "DV": [0.0, 5.0],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
            }
        )
        result = to_nlmixr2_format(df)
        assert "ID" in result.columns
        assert "NMID" not in result.columns

    def test_other_columns_preserved(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1],
                "TIME": [0.0],
                "DV": [0.0],
                "MDV": [1],
                "EVID": [1],
                "AMT": [100.0],
                "CMT": [1],
                "WT": [70.0],
            }
        )
        result = to_nlmixr2_format(df)
        assert "TIME" in result.columns
        assert "DV" in result.columns
        assert "WT" in result.columns

    def test_values_preserved(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2],
                "TIME": [0.0, 1.0],
                "DV": [0.0, 5.0],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
            }
        )
        result = to_nlmixr2_format(df)
        assert list(result["ID"]) == [1, 2]
        assert list(result["TIME"]) == [0.0, 1.0]

    def test_duplicate_id_column_raises(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1],
                "ID": [1],
                "TIME": [0.0],
                "DV": [0.0],
                "MDV": [1],
                "EVID": [1],
                "AMT": [100.0],
                "CMT": [1],
            }
        )
        import pytest

        with pytest.raises(ValueError, match="duplicates"):
            to_nlmixr2_format(df)

    def test_does_not_modify_original(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1],
                "TIME": [0.0],
                "DV": [0.0],
                "MDV": [1],
                "EVID": [1],
                "AMT": [100.0],
                "CMT": [1],
            }
        )
        _ = to_nlmixr2_format(df)
        assert "NMID" in df.columns  # original unchanged

    def test_pk_param_collision_columns_are_stripped(self) -> None:
        """ACOP-2016 simulated datasets ship the simulator's true per-
        subject parameter values in columns named ``V`` / ``CL`` /
        ``KA``. When rxode2 sees a data column with the same name as
        a model parameter it silently uses the data column instead of
        the compiled value, which breaks the posterior-predictive
        ``rxSolve`` call inside ``r/harness.R::.simulate_posterior_
        predictive`` and yields ``npe_score: None``. The adapter
        strips these names so the harness sees a parameter-free
        event table.
        """
        df = pd.DataFrame(
            {
                "NMID": [1, 1],
                "TIME": [0.0, 1.0],
                "DV": [0.0, 5.0],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
                # ACOP simulator metadata + per-subject "true" params
                "V": [70.0, 70.0],
                "CL": [4.0, 4.0],
                "KA": [1.0, 1.0],
                "DOSE": [100.0, 100.0],
                "SD": [1, 1],
            }
        )
        result = to_nlmixr2_format(df)
        for collision in ("V", "CL", "KA", "DOSE", "SD"):
            assert collision not in result.columns, (
                f"adapter must strip the {collision!r} parameter-collision column"
            )
        # And the event-table columns survive intact.
        for required in ("ID", "TIME", "DV", "AMT", "EVID", "MDV", "CMT"):
            assert required in result.columns

    def test_collision_strip_does_not_touch_unrelated_covariates(self) -> None:
        """``WT``, ``AGE``, ``SEX`` and other covariates that aren't
        in the PK-parameter-name collision set must pass through.
        """
        df = pd.DataFrame(
            {
                "NMID": [1],
                "TIME": [0.0],
                "DV": [0.0],
                "MDV": [1],
                "EVID": [1],
                "AMT": [100.0],
                "CMT": [1],
                "WT": [70.0],
                "AGE": [42],
            }
        )
        result = to_nlmixr2_format(df)
        assert "WT" in result.columns
        assert "AGE" in result.columns

    def test_custom_dvid_allowlist_preserves_multi_analyte_rows(self) -> None:
        """Multi-endpoint nlmixr2 specs pass their DSL-declared DVIDs.

        The default adapter allowlist is intentionally single-endpoint PK
        focused; passing an explicit allowlist must preserve DVID 2 while
        still dropping unmodeled observation endpoints.
        """
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1],
                "TIME": [0.0, 1.0, 1.0, 1.0],
                "DV": [0.0, 5.0, 10.0, 99.0],
                "MDV": [1, 0, 0, 0],
                "EVID": [1, 0, 0, 0],
                "AMT": [100.0, 0.0, 0.0, 0.0],
                "CMT": [1, 1, 1, 1],
                "DVID": [1, 1, 2, "pca"],
            }
        )
        result = to_nlmixr2_format(df, dvid_allowlist={"1", "2"})
        assert "DVID" in result.columns
        assert list(result["DVID"]) == [1, 1, 2]
        assert list(result["DV"]) == [0.0, 5.0, 10.0]

    def test_single_dose_sumig_helper_column_added(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 2, 2],
                "TIME": [0.0, 1.0, 0.0, 1.0],
                "DV": [0.0, 5.0, 0.0, 7.0],
                "MDV": [1, 0, 1, 0],
                "EVID": [1, 0, 1, 0],
                "AMT": [100.0, 0.0, 250.0, 0.0],
                "CMT": [1, 1, 1, 1],
            }
        )
        result = to_nlmixr2_format(df)
        assert list(result["SUMIG_DOSE"]) == [100.0, 100.0, 250.0, 250.0]
        assert list(result["SUMIG_T0"]) == [0.0, 0.0, 0.0, 0.0]

    def test_sumig_helper_records_nonzero_dose_time(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1],
                "TIME": [8.0, 9.0],
                "DV": [0.0, 5.0],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
            }
        )
        result = to_nlmixr2_format(df)
        assert result["SUMIG_T0"].tolist() == [8.0, 8.0]

    def test_dvid_count_is_recomputed_after_filtering_rows(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 1],
                "TIME": [0.0, 0.0, 1.0, 1.0],
                "DV": [999.0, 0.0, 5.0, 10.0],
                "MDV": [0, 1, 0, 0],
                "EVID": [0, 1, 0, 0],
                "AMT": [0.0, 100.0, 0.0, 0.0],
                "CMT": [1, 1, 1, 1],
                "DVID": ["junk", "cp", "cp", "pd"],
            }
        )
        result = to_nlmixr2_format(df, dvid_allowlist={"cp", "pd"})
        assert "DVID" in result.columns
        assert result.loc[result["EVID"] == 0, "DVID"].tolist() == ["cp", "pd"]

    def test_declared_categorical_reference_controls_zero_level(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2],
                "TIME": [0.0, 0.0],
                "DV": [0.0, 0.0],
                "MDV": [1, 1],
                "EVID": [1, 1],
                "AMT": [100.0, 100.0],
                "CMT": [1, 1],
                "SEX": ["male", "female"],
            }
        )
        result = to_nlmixr2_format(df, categorical_references={"SEX": "male"})
        assert result["SEX"].tolist() == [0, 1]

    def test_sumig_helper_not_added_for_multi_dose_subject(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [0.0, 12.0, 24.0],
                "DV": [0.0, 0.0, 5.0],
                "MDV": [1, 1, 0],
                "EVID": [1, 1, 0],
                "AMT": [100.0, 100.0, 0.0],
                "CMT": [1, 1, 1],
            }
        )
        result = to_nlmixr2_format(df)
        assert "SUMIG_DOSE" not in result.columns
