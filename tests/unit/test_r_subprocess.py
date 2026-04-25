# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for R subprocess request/response schemas (ARCHITECTURE.md §4.2)."""

import json

import pytest
from pydantic import ValidationError

from apmode.backends.r_schemas import (
    RSessionInfo,
    RSubprocessRequest,
    RSubprocessResponse,
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
    Proportional,
)
from apmode.ids import generate_candidate_id, generate_run_id


def _session_info() -> RSessionInfo:
    return RSessionInfo(
        r_version="4.4.1", nlmixr2_version="3.0.0", platform="aarch64-apple-darwin"
    )


def _test_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_model_id_0000000",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=70.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


class TestRSubprocessRequest:
    def test_valid(self) -> None:
        req = RSubprocessRequest(
            schema_version="1.0",
            request_id=generate_run_id(),
            run_id=generate_run_id(),
            candidate_id=generate_candidate_id(),
            spec=_test_spec(),
            data_path="/mnt/data/pk.csv",
            seed=42,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates={"CL": 5.0, "V": 70.0, "ka": 1.0},
            estimation=["saem", "focei"],
        )
        assert req.schema_version == "1.0"
        assert req.seed == 42

    def test_json_roundtrip(self) -> None:
        req = RSubprocessRequest(
            schema_version="1.0",
            request_id="test_req_id_12345678",
            run_id="test_run_id_12345678",
            candidate_id="test_cand_id_1234567",
            spec=_test_spec(),
            data_path="/data/pk.csv",
            seed=123,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates={"CL": 3.0},
            estimation=["saem"],
        )
        json_str = req.model_dump_json()
        parsed = json.loads(json_str)
        roundtripped = RSubprocessRequest.model_validate(parsed)
        assert roundtripped.request_id == req.request_id

    def test_empty_estimation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/data.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=[],
            )

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="traversal"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/tmp/../../etc/passwd",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
            )

    def test_relative_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="data/pk.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
            )

    def test_test_data_path_traversal_rejected(self) -> None:
        """Held-out CSV path is subject to the same no-traversal rule as data_path."""
        with pytest.raises(ValidationError, match="traversal"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/data/train.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
                test_data_path="/tmp/../../etc/shadow",
            )

    def test_test_data_path_relative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            RSubprocessRequest(
                schema_version="1.0",
                request_id="id1",
                run_id="id2",
                candidate_id="id3",
                spec=_test_spec(),
                data_path="/data/train.csv",
                seed=1,
                rng_kind="L'Ecuyer-CMRG",
                initial_estimates={},
                estimation=["saem"],
                test_data_path="relative/test.csv",
            )

    def test_test_data_path_none_is_default(self) -> None:
        """``test_data_path`` defaults to None and ``fixed_parameter`` to False
        so the wire shape is bit-identical to the v0.6 baseline when callers
        do not opt into honest mode.
        """
        req = RSubprocessRequest(
            schema_version="1.0",
            request_id="id1",
            run_id="id2",
            candidate_id="id3",
            spec=_test_spec(),
            data_path="/data/pk.csv",
            seed=1,
            rng_kind="L'Ecuyer-CMRG",
            initial_estimates={},
            estimation=["saem"],
        )
        assert req.test_data_path is None
        assert req.fixed_parameter is False


class TestRSubprocessResponse:
    def test_success(self) -> None:
        resp = RSubprocessResponse(
            schema_version="1.0",
            status="success",
            error_type=None,
            result={"converged": True, "ofv": -1234.5},
            r_session_info=_session_info(),
            random_seed_state=[1, 2, 3, 4, 5],
        )
        assert resp.status == "success"

    def test_error(self) -> None:
        resp = RSubprocessResponse(
            schema_version="1.0",
            status="error",
            error_type="convergence",
            result=None,
            r_session_info=_session_info(),
            random_seed_state=None,
        )
        assert resp.error_type == "convergence"

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            RSubprocessResponse(
                schema_version="1.0",
                status="invalid",
                error_type=None,
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )

    def test_invalid_error_type(self) -> None:
        with pytest.raises(ValidationError):
            RSubprocessResponse(
                schema_version="1.0",
                status="error",
                error_type="invalid_type",
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )

    def test_success_with_error_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be None"):
            RSubprocessResponse(
                schema_version="1.0",
                status="success",
                error_type="convergence",
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )

    def test_error_without_error_type_rejected(self) -> None:
        with pytest.raises(ValidationError, match="required"):
            RSubprocessResponse(
                schema_version="1.0",
                status="error",
                error_type=None,
                result=None,
                r_session_info=_session_info(),
                random_seed_state=None,
            )
