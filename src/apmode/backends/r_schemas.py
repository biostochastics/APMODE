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
    # When set, the R harness fits on ``data_path`` (train) but routes
    # ``rxode2::rxSolve(events=test_data_path_df)`` so the posterior-
    # predictive sim matrix — and therefore NPE / VPC / AUC-Cmax — is
    # held-out (true cross-validation NPE) instead of in-sample. The
    # held-out subjects MUST be disjoint from the training subjects;
    # rxode2 partitions sims by ID and a colliding ID would silently
    # recycle the train subject's posthoc ETA instead of drawing a
    # fresh ETA from the fitted Omega.
    test_data_path: str | None = None
    # When True the harness skips the AIC-best estimation loop and
    # runs ``est='posthoc'`` exactly once: nlmixr2 freezes
    # THETA/OMEGA/SIGMA at the compiled ini() values (which the DSL
    # emitter writes from ``initial_estimates``) and only estimates
    # ETAs. Combined with ``test_data_path`` this turns the literature-
    # side fit in Suite-C Phase-1 into a real methodology-drift
    # detector instead of a warm-start tautology.
    fixed_parameter: bool = False

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

    @field_validator("test_data_path")
    @classmethod
    def test_data_path_no_traversal(cls, v: str | None) -> str | None:
        """Same path-safety rules as ``data_path``; ``None`` is allowed."""
        if v is None:
            return v
        p = Path(v)
        if ".." in p.parts:
            raise ValueError("test_data_path must not contain '..' traversal")
        if not p.is_absolute():
            raise ValueError("test_data_path must be an absolute path")
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

    @field_validator("t_observed", "observed_dv", mode="before")
    @classmethod
    def _coerce_scalar_to_list(cls, v: object) -> object:
        """Tolerate a bare scalar for the n_obs == 1 case.

        The R harness wraps single-observation vectors with ``I(...)``
        so ``jsonlite::toJSON(..., auto_unbox = TRUE)`` keeps length-1
        numeric arrays as JSON arrays. Without the wrap (older bundles
        or a future regression), a subject with n_obs == 1 emits
        ``t_observed: 1.5`` instead of ``t_observed: [1.5]`` and the
        ``list[float]`` field rejects the float with
        ``Input should be a valid list``. Coerce a bare scalar to a
        length-1 list defensively; well-formed inputs (list-of-floats)
        are unaffected.
        """
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return [float(v)]
        return v

    @field_validator("sims_at_observed", mode="before")
    @classmethod
    def _coerce_1d_to_2d(cls, v: object) -> object:
        """Tolerate a flat ``list[float]`` for the n_obs == 1 case.

        The R harness wraps each per-sim row with ``I(...)`` so
        ``jsonlite::toJSON(..., auto_unbox = TRUE)`` keeps length-1
        arrays as JSON arrays rather than unboxing them to scalars.
        Older bundles + any future caller that forgets the ``I()`` wrap
        would emit ``[1.5, 2.0, ...]`` instead of ``[[1.5], [2.0], ...]``
        for sparse single-observation subjects, and the n_sims-long
        list of scalars then fails the ``list[list[float]]`` check
        with one ValidationError per simulated row (200 errors at
        ``--n-sims 200``). Coerce that shape to ``[[x] for x in v]``
        defensively so the runner does not crash on sparse PK fixtures
        even if the upstream R fix regresses; the explicit check on
        the inner type means well-formed inputs are unaffected.
        """
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [[float(x)] for x in v]
        return v


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
