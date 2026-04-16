# SPDX-License-Identifier: GPL-2.0-or-later
"""R subprocess request/response JSON schemas (ARCHITECTURE.md §4.2).

Wire contract between Python orchestrator and R backend containers.
Communication is via files, not stdout, to avoid R stdout contamination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from apmode.dsl.ast_models import DSLSpec  # noqa: TC001 — Pydantic needs runtime access


class RSubprocessRequest(BaseModel):
    """Request written to /tmp/{request_id}/request.json."""

    schema_version: str
    request_id: str
    run_id: str
    candidate_id: str
    spec: DSLSpec
    data_path: str
    seed: int
    rng_kind: Literal["L'Ecuyer-CMRG", "Mersenne-Twister"]
    initial_estimates: dict[str, float]
    estimation: list[str] = Field(min_length=1)
    compiled_r_code: str = ""
    initial_estimate_source: Literal["nca", "warm_start", "fallback"] = "fallback"
    # Optional split manifest for split-aware CWRES diagnostics
    split_manifest: dict[str, object] | None = None
    # Number of posterior-predictive simulation draws for VPC/NPE/AUC-Cmax
    # (see apmode.backends.predictive_summary). 0 disables simulation;
    # the R harness leaves ``predicted_simulations`` NULL and Gate 3
    # ranking falls back to the CWRES NPE proxy. Wired to
    # Gate3Config.n_posterior_predictive_sims by the runner.
    n_posterior_predictive_sims: int = Field(default=0, ge=0)

    @field_validator("data_path")
    @classmethod
    def data_path_no_traversal(cls, v: str) -> str:
        """Reject path traversal sequences. Full directory bounding is deployment-specific."""
        p = Path(v)
        if ".." in p.parts:
            raise ValueError("data_path must not contain '..' traversal")
        if not p.is_absolute():
            raise ValueError("data_path must be an absolute path")
        return v


class PredictedSimulationsSubject(BaseModel):
    """One subject's posterior-predictive simulation matrix.

    Carries the subject_id + observed time vector + observed DV vector
    + nested-list simulation matrix (n_sims rows, n_obs columns). The
    runner converts this into :class:`apmode.backends.predictive_summary.
    SubjectSimulation` and hands it to
    :func:`build_predictive_diagnostics` for atomic VPC/NPE/AUC-Cmax
    computation.
    """

    model_config = ConfigDict(frozen=True)

    subject_id: str
    t_observed: list[float] = Field(min_length=1)
    observed_dv: list[float] = Field(min_length=1)
    # JSON round-trips n_sims * n_obs as list[list[float]]; the runner
    # reshapes into a numpy (n_sims, n_obs) array before validation.
    sims_at_observed: list[list[float]] = Field(min_length=1)


class RSessionInfo(BaseModel):
    """R session info captured from the backend process."""

    r_version: str
    nlmixr2_version: str
    platform: str
    packages: dict[str, str] = Field(default_factory=dict)


class RSubprocessResponse(BaseModel):
    """Response read from /tmp/{request_id}/response.json.

    Exit codes: 0=success, 1=R error, 137=killed, 139=segfault.
    On no response.json: classify as CrashError.
    On timeout: kill process group, classify as TimeoutError.
    """

    schema_version: str
    status: Literal["success", "error"]
    error_type: Literal["convergence", "crash", "invalid_spec"] | None = None
    # Result is a dict matching BackendResult fields. Validated into BackendResult
    # by the runner, not here, because the R side emits raw JSON dicts.
    result: dict[str, Any] | None = None
    r_session_info: RSessionInfo
    random_seed_state: list[int] | None = None

    @model_validator(mode="after")
    def status_error_type_consistency(self) -> RSubprocessResponse:
        if self.status == "success" and self.error_type is not None:
            raise ValueError("error_type must be None when status is 'success'")
        if self.status == "error" and self.error_type is None:
            raise ValueError("error_type is required when status is 'error'")
        return self
