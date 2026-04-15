# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for apmode.data.categorical_encoding."""

from __future__ import annotations

import pandas as pd
import pytest

from apmode.data.categorical_encoding import (
    EXPECTED_BINARY_FORMAT,
    auto_remap_binary_columns,
    detect_encoding,
)


class TestDetectEncoding:
    def test_canonical_zero_one_no_remap(self) -> None:
        s = pd.Series([0, 1, 0, 1, 1, 0], name="SEX")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "binary_zero_one"
        assert hint.suggested_remap is None
        assert hint.applied is False

    def test_one_two_indexed_integers(self) -> None:
        s = pd.Series([1, 2, 1, 2, 2, 1], name="SEX")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "binary_one_two"
        assert hint.suggested_remap == {1: 0, 2: 1}

    def test_male_female_strings(self) -> None:
        s = pd.Series(["male", "female", "male", "female"], name="sex")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "binary_string_pair"
        assert hint.suggested_remap == {"female": 0, "male": 1}

    def test_m_f_uppercase(self) -> None:
        s = pd.Series(["M", "F", "M"], name="sex")
        hint = detect_encoding(s)
        assert hint.suggested_remap == {"F": 0, "M": 1}

    def test_yes_no_strings(self) -> None:
        s = pd.Series(["yes", "no", "yes"], name="DIABETIC")
        hint = detect_encoding(s)
        assert hint.suggested_remap == {"no": 0, "yes": 1}

    def test_boolean_pair(self) -> None:
        s = pd.Series([True, False, True, False], name="ACTIVE")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "binary_boolean"
        assert hint.suggested_remap == {False: 0, True: 1}

    def test_bool_numeric_object_mix_rejected(self) -> None:
        # Gemini review: object-dtype mixing bool and non-bool numeric where
        # the values are actually distinct (True, 2) must be rejected as
        # multi_level — the bool/int hash-equivalence would otherwise
        # produce a meaningless remap.
        # Note: True/1 and False/0 collapse under pandas equality before
        # this detector sees them, so this test covers the detectable path.
        s = pd.Series([True, 2], dtype=object, name="FLAG")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "multi_level"
        assert hint.suggested_remap is None
        assert "bool" in hint.rationale.lower()

    def test_unknown_two_level_string_alphabetic_default(self) -> None:
        s = pd.Series(["AlphaArm", "BetaArm", "AlphaArm"], name="ARM")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "binary_string_pair"
        # Alphabetic order: AlphaArm → 0, BetaArm → 1
        assert hint.suggested_remap == {"AlphaArm": 0, "BetaArm": 1}
        assert "alphabetic order" in hint.rationale.lower()

    def test_multi_level_categorical_no_remap(self) -> None:
        s = pd.Series(["red", "green", "blue", "red"], name="COLOR")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "multi_level"
        assert hint.suggested_remap is None
        assert "one-hot" in hint.rationale.lower()

    def test_continuous_two_distinct_numbers_not_remapped(self) -> None:
        # 50 and 100 are two numbers but not 0/1 or 1/2 — must not be remapped.
        s = pd.Series([50.0, 100.0, 50.0, 100.0], name="DOSE")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "continuous"
        assert hint.suggested_remap is None

    def test_constant_column(self) -> None:
        s = pd.Series([5, 5, 5], name="STUDY")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "constant"
        assert hint.suggested_remap is None

    def test_all_missing(self) -> None:
        s = pd.Series([float("nan"), float("nan")], name="LAB")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "all_missing"
        assert hint.unique_values == []

    def test_continuous_many_values(self) -> None:
        s = pd.Series(list(range(50)), name="WT")
        hint = detect_encoding(s)
        assert hint.detected_encoding == "continuous"


class TestAutoRemapBinaryColumns:
    def test_warfarin_style_string_pair_round_trip(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2, 3, 4, 5, 6],
                "sex": ["male", "female", "male", "female", "male", "female"],
            }
        )
        out, hints = auto_remap_binary_columns(df, ["sex"])
        assert set(out["sex"].unique().tolist()) == {0, 1}
        assert hints[0].applied is True
        assert hints[0].detected_encoding == "binary_string_pair"

    def test_mavoglurant_style_one_two_round_trip(self) -> None:
        df = pd.DataFrame({"NMID": [1, 2, 3], "SEX": [1, 2, 1]})
        out, hints = auto_remap_binary_columns(df, ["SEX"])
        assert set(out["SEX"].unique().tolist()) == {0, 1}
        assert hints[0].applied is True

    def test_override_takes_precedence(self) -> None:
        # Force the polarity opposite to auto-detection.
        df = pd.DataFrame({"NMID": [1, 2, 3], "sex": ["male", "female", "male"]})
        overrides = {"sex": {"male": 0, "female": 1}}  # flipped
        out, hints = auto_remap_binary_columns(df, ["sex"], overrides=overrides)
        # male rows should now be 0 (caller's polarity wins)
        assert out.loc[out["NMID"] == 1, "sex"].iloc[0] == 0
        assert hints[0].applied is True
        assert "override" in hints[0].rationale.lower()

    def test_apply_false_does_not_modify(self) -> None:
        df = pd.DataFrame({"NMID": [1, 2], "sex": ["male", "female"]})
        out, hints = auto_remap_binary_columns(df, ["sex"], apply=False)
        # Inspection-only: original strings preserved.
        assert set(out["sex"].unique().tolist()) == {"male", "female"}
        assert hints[0].applied is False
        # Suggested remap is still surfaced for the inspect/validate report.
        assert hints[0].suggested_remap == {"female": 0, "male": 1}

    def test_canonical_zero_one_passthrough(self) -> None:
        df = pd.DataFrame({"NMID": [1, 2, 3], "SEX": [0, 1, 0]})
        out, hints = auto_remap_binary_columns(df, ["SEX"])
        # Already canonical — no remap applied, hint says so.
        assert hints[0].detected_encoding == "binary_zero_one"
        assert hints[0].applied is False
        assert (out["SEX"] == [0, 1, 0]).all()

    def test_missing_column_returns_diagnostic(self) -> None:
        df = pd.DataFrame({"NMID": [1, 2]})
        _, hints = auto_remap_binary_columns(df, ["NOT_THERE"])
        assert hints[0].detected_encoding == "all_missing"
        assert "not present" in hints[0].rationale.lower()

    def test_multi_level_not_remapped(self) -> None:
        df = pd.DataFrame({"NMID": [1, 2, 3, 4], "ARM": ["A", "B", "C", "A"]})
        out, hints = auto_remap_binary_columns(df, ["ARM"])
        assert hints[0].detected_encoding == "multi_level"
        # No remap applied — the values are unchanged.
        assert (out["ARM"] == ["A", "B", "C", "A"]).all()


class TestExpectedBinaryFormat:
    """Documentation surface — guards against accidental drift."""

    def test_documents_recognised_pairs(self) -> None:
        for pair in ("M/F", "Male/Female", "Yes/No"):
            assert pair in EXPECTED_BINARY_FORMAT
        assert "0, 1" in EXPECTED_BINARY_FORMAT
        assert "binary_encode_overrides" in EXPECTED_BINARY_FORMAT


class TestSummarizeCovariatesIntegration:
    """End-to-end: summarize_covariates auto-remaps before binary validation."""

    def test_warfarin_style_sex_auto_remapped(self) -> None:
        # Without the auto-remap this would raise the "binary transform
        # requires {0, 1}" ValueError that motivated the whole feature.
        from apmode.dsl.frem_emitter import summarize_covariates

        df = pd.DataFrame(
            {
                "NMID": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                "TIME": [0.0, 1.0] * 6,
                "sex": [
                    "male",
                    "male",
                    "female",
                    "female",
                    "male",
                    "male",
                    "female",
                    "female",
                    "male",
                    "male",
                    "female",
                    "female",
                ],
            }
        )
        covs = summarize_covariates(df, ["sex"], transforms={"sex": "binary"})
        assert covs[0].transform == "binary"
        # Binary column should have mu near 0.5 (3 male=1, 3 female=0).
        assert 0.4 < covs[0].mu_init < 0.6

    def test_override_via_summarize_covariates(self) -> None:
        from apmode.dsl.frem_emitter import summarize_covariates

        df = pd.DataFrame(
            {
                "NMID": [1, 2, 3, 4],
                "TIME": [0.0, 0.0, 0.0, 0.0],
                "GROUP": ["A", "B", "A", "B"],
            }
        )
        # Force the polarity: A → 1, B → 0 (opposite of the alphabetic default).
        covs = summarize_covariates(
            df,
            ["GROUP"],
            transforms={"GROUP": "binary"},
            binary_encode_overrides={"GROUP": {"A": 1, "B": 0}},
        )
        # Mean is still 0.5; the value is in the override-applied data
        # but the override polarity is captured in the encoding hint.
        assert covs[0].mu_init == pytest.approx(0.5)
