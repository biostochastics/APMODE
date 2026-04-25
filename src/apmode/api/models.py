# SPDX-License-Identifier: GPL-2.0-or-later
"""Request and response models for the APMODE HTTP API (plan Task 32).

These wrap :class:`apmode.api.store.RunRecord` and
:class:`apmode.orchestrator.RunConfig` for the JSON-over-HTTP boundary.
The split is intentional: ``RunRecord`` is a *persistence* row,
``RunStatusResponse`` is the *API* projection — keeping them separate
means a future schema change to the database row (e.g. adding an
internal flag) does not silently leak into the public API.

The Pydantic models live in ``apmode.api`` (not ``apmode.api.routes``)
so :func:`apmode.api.app.build_app` and the unit tests can import them
without pulling in the FastAPI router stack.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from apmode.api.store import RunStatus  # noqa: TC001 — runtime use by Pydantic field validation

# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class CreateRunRequest(BaseModel):
    """Body for ``POST /runs``.

    The dataset is referenced by *server-local path* — the API is bound
    to localhost (plan Task 35: ``apmode serve --host 127.0.0.1``) and
    is not intended to host arbitrary uploads. Multipart upload is a
    follow-up; pinning the contract to a path now keeps the runner
    plumbing identical to the CLI's ``apmode run <csv>`` and avoids a
    half-built upload surface.

    ``backend`` is constrained at the route layer to whatever
    ``build_app(allow_backends=...)`` permitted — accepting an
    unrecognised backend in the body is a 400 (plan Task 36 will add
    ``bayesian_stan`` to the default allowlist).
    """

    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(
        min_length=1,
        description="Absolute path to a NONMEM-style CSV on the server filesystem.",
    )
    lane: Literal["submission", "discovery", "optimization"] = "submission"
    backend: str = Field(
        default="nlmixr2",
        description=(
            "Primary backend. Must be in the app's allow_backends; "
            "see /openapi.json for the live list."
        ),
    )
    seed: int = 753849
    timeout_seconds: int = Field(default=900, gt=0, le=24 * 3600)
    max_concurrency: int = Field(default=1, ge=1, le=64)
    covariate_names: list[str] = Field(default_factory=list)
    column_mapping: dict[str, str] | None = None
    context_of_use: str | None = Field(
        default=None,
        description=(
            "Verbatim Gate 2.5 context_of_use. When omitted the orchestrator "
            "falls back to '<lane> lane analysis' which is a placeholder, "
            "not a regulatory-defensible COU statement."
        ),
    )
    # Plan Task 34: opt-in flag honoured by the lifespan startup sweep.
    # Stored on the RunRecord so a re-queue worker (out of scope for
    # v0.6-rc1) can replay INTERRUPTED rows automatically.
    requeue_on_interrupt: bool = False


# ---------------------------------------------------------------------------
# Response payloads
# ---------------------------------------------------------------------------


class RunCreatedResponse(BaseModel):
    """202 body returned by ``POST /runs``.

    ``status_url`` is a relative path the client can follow without
    needing to know the server's external base URL.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: RunStatus
    status_url: str


class RunStatusResponse(BaseModel):
    """Public projection of one :class:`apmode.api.store.RunRecord`."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: RunStatus
    bundle_dir: str
    lane: str | None = None
    backend: str | None = None
    seed: int | None = None
    created_at: str
    updated_at: str
    error: str | None = None
    requeue_on_interrupt: bool = False


class RunListResponse(BaseModel):
    """Body returned by ``GET /runs``."""

    model_config = ConfigDict(frozen=True)

    runs: list[RunStatusResponse]


class HealthResponse(BaseModel):
    """Body returned by ``GET /healthz``."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    apmode_version: str


__all__ = [
    "CreateRunRequest",
    "HealthResponse",
    "RunCreatedResponse",
    "RunListResponse",
    "RunStatusResponse",
]
