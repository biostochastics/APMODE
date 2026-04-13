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
