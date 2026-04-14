# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for variable/schema normalization across the APMODE pipeline.

Covers:
- Parameter name case-insensitive normalization
- Column name case-folding and alias resolution
- Validator accepts case-variant param references
- Transform parser normalizes LLM output
- ColumnMapping bidirectional lookup
- Data ingestion auto-normalization
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from apmode.backends.protocol import Lane
from apmode.backends.transform_parser import parse_llm_response
from apmode.bundle.models import ColumnMapping
from apmode.dsl.ast_models import (
    IIV,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
    TwoCmt,
)
from apmode.dsl.normalize import (
    normalize_column_name,
    normalize_columns,
    normalize_param_list,
    normalize_param_name,
)
from apmode.dsl.transforms import (
    AddCovariateLink,
    AdjustVariability,
    apply_transform,
    validate_transform,
)
from apmode.dsl.validator import validate_dsl

# ---------------------------------------------------------------------------
# normalize_param_name
# ---------------------------------------------------------------------------


class TestNormalizeParamName:
    """Canonical parameter name resolution."""

    def test_canonical_form_unchanged(self) -> None:
        assert normalize_param_name("CL") == "CL"
        assert normalize_param_name("ka") == "ka"
        assert normalize_param_name("V1") == "V1"
        assert normalize_param_name("KD") == "KD"

    def test_lowercase_resolves(self) -> None:
        assert normalize_param_name("cl") == "CL"
        assert normalize_param_name("v") == "V"
        assert normalize_param_name("vmax") == "Vmax"
        assert normalize_param_name("km") == "Km"

    def test_mixed_case_resolves(self) -> None:
        assert normalize_param_name("Ka") == "ka"
        assert normalize_param_name("Cl") == "CL"
        assert normalize_param_name("kD") == "KD"

    def test_unknown_param_passthrough(self) -> None:
        assert normalize_param_name("nonexistent") == "nonexistent"
        assert normalize_param_name("BOGUS") == "BOGUS"

    def test_all_absorption_params(self) -> None:
        assert normalize_param_name("ka") == "ka"
        assert normalize_param_name("dur") == "dur"
        assert normalize_param_name("tlag") == "tlag"
        assert normalize_param_name("n") == "n"
        assert normalize_param_name("ktr") == "ktr"
        assert normalize_param_name("frac") == "frac"

    def test_all_distribution_params(self) -> None:
        assert normalize_param_name("V") == "V"
        assert normalize_param_name("v1") == "V1"
        assert normalize_param_name("v2") == "V2"
        assert normalize_param_name("v3") == "V3"
        assert normalize_param_name("q") == "Q"
        assert normalize_param_name("q2") == "Q2"
        assert normalize_param_name("q3") == "Q3"

    def test_all_tmdd_params(self) -> None:
        assert normalize_param_name("r0") == "R0"
        assert normalize_param_name("kon") == "kon"
        assert normalize_param_name("koff") == "koff"
        assert normalize_param_name("kint") == "kint"
        assert normalize_param_name("kd") == "KD"

    def test_all_elimination_params(self) -> None:
        assert normalize_param_name("cl") == "CL"
        assert normalize_param_name("vmax") == "Vmax"
        assert normalize_param_name("km") == "Km"
        assert normalize_param_name("kdecay") == "kdecay"


class TestNormalizeParamList:
    def test_preserves_order(self) -> None:
        result = normalize_param_list(["cl", "v", "ka"])
        assert result == ["CL", "V", "ka"]

    def test_empty_list(self) -> None:
        assert normalize_param_list([]) == []


# ---------------------------------------------------------------------------
# normalize_column_name / normalize_columns
# ---------------------------------------------------------------------------


class TestNormalizeColumnName:
    def test_uppercase(self) -> None:
        assert normalize_column_name("time") == "TIME"
        assert normalize_column_name("dv") == "DV"
        assert normalize_column_name("evid") == "EVID"

    def test_alias_id_to_nmid(self) -> None:
        assert normalize_column_name("id") == "NMID"
        assert normalize_column_name("ID") == "NMID"

    def test_alias_subject(self) -> None:
        assert normalize_column_name("SUBJ") == "NMID"
        assert normalize_column_name("subject_id") == "NMID"

    def test_covariate_uppercased(self) -> None:
        assert normalize_column_name("wt") == "WT"
        assert normalize_column_name("age") == "AGE"

    def test_already_canonical(self) -> None:
        assert normalize_column_name("NMID") == "NMID"
        assert normalize_column_name("TIME") == "TIME"


class TestNormalizeColumns:
    def test_builds_rename_mapping(self) -> None:
        mapping = normalize_columns(["id", "time", "dv", "evid", "amt"])
        assert mapping == {
            "id": "NMID",
            "time": "TIME",
            "dv": "DV",
            "evid": "EVID",
            "amt": "AMT",
        }

    def test_no_change_returns_empty(self) -> None:
        mapping = normalize_columns(["NMID", "TIME", "DV"])
        assert mapping == {}

    def test_mixed_case_columns(self) -> None:
        mapping = normalize_columns(["Id", "Time", "wt"])
        assert mapping["Id"] == "NMID"
        assert mapping["Time"] == "TIME"
        assert mapping["wt"] == "WT"


# ---------------------------------------------------------------------------
# Validator accepts case-variant params
# ---------------------------------------------------------------------------


def _make_spec(**overrides: object) -> DSLSpec:
    defaults: dict[str, object] = {
        "model_id": "test_norm_000000000000",
        "absorption": FirstOrder(ka=1.0),
        "distribution": OneCmt(V=70.0),
        "elimination": LinearElim(CL=5.0),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)  # type: ignore[arg-type]


class TestValidatorNormalization:
    """Validator should accept case-variant parameter references."""

    def test_iiv_lowercase_cl_accepted(self) -> None:
        """IIV with 'cl' should match structural param 'CL'."""
        spec = _make_spec(variability=[IIV(params=["cl", "V"], structure="diagonal")])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "iiv_param_exists" for e in errors)

    def test_iiv_mixed_case_v_accepted(self) -> None:
        """IIV with 'v' should match structural param 'V'."""
        spec = _make_spec(variability=[IIV(params=["CL", "v"], structure="diagonal")])
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "iiv_param_exists" for e in errors)

    def test_covariate_link_lowercase_param(self) -> None:
        """CovariateLink with 'cl' should match structural param 'CL'."""
        spec = _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="cl", covariate="WT", form="power"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "covariate_param_exists" for e in errors)

    def test_duplicate_iiv_case_insensitive(self) -> None:
        """'CL' and 'cl' in separate IIV blocks should be detected as duplicate."""
        spec = _make_spec(
            variability=[
                IIV(params=["CL"], structure="diagonal"),
                IIV(params=["cl"], structure="diagonal"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "iiv_no_duplicate_params" for e in errors)

    def test_duplicate_covariate_case_insensitive(self) -> None:
        """CovariateLinks on 'CL'+'WT' and 'cl'+'wt' should be duplicate."""
        spec = _make_spec(
            variability=[
                IIV(params=["CL", "V"], structure="diagonal"),
                CovariateLink(param="CL", covariate="WT", form="power"),
                CovariateLink(param="cl", covariate="wt", form="exponential"),
            ],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert any(e.constraint == "covariate_link_no_duplicate" for e in errors)

    def test_2cmt_lowercase_params(self) -> None:
        """IIV with 'v1' should match TwoCmt's structural param 'V1'."""
        spec = _make_spec(
            distribution=TwoCmt(V1=30.0, V2=40.0, Q=5.0),
            variability=[IIV(params=["CL", "v1"], structure="diagonal")],
        )
        errors = validate_dsl(spec, lane=Lane.SUBMISSION)
        assert not any(e.constraint == "iiv_param_exists" for e in errors)


# ---------------------------------------------------------------------------
# Transform validation normalization
# ---------------------------------------------------------------------------


class TestTransformNormalization:
    """Transform validation should accept case-variant param names."""

    def test_add_covariate_lowercase_param(self) -> None:
        spec = _make_spec()
        t = AddCovariateLink(param="cl", covariate="WT", form="power")
        errors = validate_transform(spec, t)
        assert len(errors) == 0

    def test_adjust_variability_lowercase_param(self) -> None:
        spec = _make_spec()
        t = AdjustVariability(param="ka", action="add")
        errors = validate_transform(spec, t)
        assert len(errors) == 0

    def test_apply_covariate_normalizes_param(self) -> None:
        """apply_transform should store canonical param name in CovariateLink."""
        spec = _make_spec()
        t = AddCovariateLink(param="cl", covariate="WT", form="power")
        new_spec = apply_transform(spec, t)
        cov_links = [v for v in new_spec.variability if isinstance(v, CovariateLink)]
        assert len(cov_links) == 1
        assert cov_links[0].param == "CL"  # normalized

    def test_duplicate_covariate_case_insensitive(self) -> None:
        """Duplicate check should be case-insensitive."""
        spec = _make_spec()
        t1 = AddCovariateLink(param="CL", covariate="WT", form="power")
        spec2 = apply_transform(spec, t1)
        t2 = AddCovariateLink(param="cl", covariate="wt", form="exponential")
        errors = validate_transform(spec2, t2)
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Transform parser normalization
# ---------------------------------------------------------------------------


class TestTransformParserNormalization:
    """LLM parser should normalize param names from JSON output."""

    def test_add_covariate_lowercase_cl(self) -> None:
        raw = json.dumps(
            {
                "transforms": [
                    {
                        "type": "add_covariate_link",
                        "param": "cl",
                        "covariate": "WT",
                        "form": "power",
                    },
                ],
                "reasoning": "Weight effect on clearance.",
            }
        )
        result = parse_llm_response(raw)
        assert result.success
        t = result.transforms[0]
        assert isinstance(t, AddCovariateLink)
        assert t.param == "CL"  # normalized from "cl"

    def test_adjust_variability_mixed_case(self) -> None:
        raw = json.dumps(
            {
                "transforms": [
                    {"type": "adjust_variability", "param": "Cl", "action": "add"},
                ],
                "reasoning": "Add IIV on clearance.",
            }
        )
        result = parse_llm_response(raw)
        assert result.success
        t = result.transforms[0]
        assert isinstance(t, AdjustVariability)
        assert t.param == "CL"  # normalized from "Cl"


# ---------------------------------------------------------------------------
# ColumnMapping bidirectional
# ---------------------------------------------------------------------------


class TestColumnMappingBidirectional:
    def test_to_canonical(self) -> None:
        m = ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            mdv="MDV",
            cmt="CMT",
        )
        c = m.to_canonical()
        assert c["subject_id"] == "NMID"
        assert c["time"] == "TIME"
        assert c["dv"] == "DV"

    def test_to_semantic(self) -> None:
        m = ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            mdv="MDV",
            cmt="CMT",
        )
        s = m.to_semantic()
        assert s["NMID"] == "subject_id"
        assert s["TIME"] == "time"
        assert s["DV"] == "dv"

    def test_optional_fields_excluded(self) -> None:
        m = ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        )
        c = m.to_canonical()
        assert "rate" not in c
        s = m.to_semantic()
        assert "RATE" not in s

    def test_roundtrip(self) -> None:
        m = ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            mdv="MDV",
            cmt="CMT",
            rate="RATE",
        )
        canonical = m.to_canonical()
        semantic = m.to_semantic()
        for sem_key, can_val in canonical.items():
            assert semantic[can_val] == sem_key


# ---------------------------------------------------------------------------
# Data ingestion auto-normalization
# ---------------------------------------------------------------------------


class TestIngestNormalization:
    def test_lowercase_columns_auto_normalized(self) -> None:
        """CSV with lowercase column names should be auto-uppercased."""
        from apmode.data.ingest import ingest_nonmem_csv

        df = pd.DataFrame(
            {
                "nmid": [1, 1],
                "time": [0.0, 1.0],
                "dv": [0.0, 5.5],
                "mdv": [1, 0],
                "evid": [1, 0],
                "amt": [100.0, 0.0],
                "cmt": [1, 1],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        try:
            manifest, validated = ingest_nonmem_csv(path)
            assert "NMID" in validated.columns
            assert "TIME" in validated.columns
            assert manifest.n_subjects == 1
        finally:
            path.unlink()

    def test_id_alias_auto_resolved(self) -> None:
        """CSV with 'ID' column should auto-map to 'NMID'."""
        from apmode.data.ingest import ingest_nonmem_csv

        df = pd.DataFrame(
            {
                "ID": [1, 1],
                "TIME": [0.0, 1.0],
                "DV": [0.0, 5.5],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        try:
            _manifest, validated = ingest_nonmem_csv(path)
            assert "NMID" in validated.columns
            assert "ID" not in validated.columns
        finally:
            path.unlink()

    def test_explicit_mapping_takes_precedence(self) -> None:
        """Explicit column_mapping should override auto-normalization."""
        from apmode.data.ingest import ingest_nonmem_csv

        df = pd.DataFrame(
            {
                "subj": [1, 1],
                "TIME": [0.0, 1.0],
                "DV": [0.0, 5.5],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        try:
            _manifest, validated = ingest_nonmem_csv(path, column_mapping={"subj": "NMID"})
            assert "NMID" in validated.columns
        finally:
            path.unlink()

    def test_covariate_columns_uppercased(self) -> None:
        """Non-canonical columns (covariates) should be uppercased too."""
        from apmode.data.ingest import ingest_nonmem_csv

        df = pd.DataFrame(
            {
                "NMID": [1, 1],
                "TIME": [0.0, 1.0],
                "DV": [0.0, 5.5],
                "MDV": [1, 0],
                "EVID": [1, 0],
                "AMT": [100.0, 0.0],
                "CMT": [1, 1],
                "wt": [70.0, 70.0],
            }
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            df.to_csv(f, index=False)
            path = Path(f.name)
        try:
            manifest, validated = ingest_nonmem_csv(path)
            assert "WT" in validated.columns
            assert len(manifest.covariates) == 1
            assert manifest.covariates[0].name == "WT"
        finally:
            path.unlink()
