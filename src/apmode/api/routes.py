# SPDX-License-Identifier: GPL-2.0-or-later
# pyright: reportUnusedFunction=false
# Route handlers below are referenced via @router.get / @router.post
# decorators, which Pyright can't see — every "unused function" warning
# in this file is a false positive.
"""FastAPI routes for the APMODE HTTP API (plan Task 32).

The router is built by :func:`build_router` and mounted by
:func:`apmode.api.app.build_app`. Three concerns live here:

1. **Schema validation** — the Pydantic request models reject malformed
   bodies before any orchestrator code runs.
2. **Status orchestration** — every transition goes through the
   :class:`apmode.api.store.RunStore`. The route handlers never mutate
   ``app.state.active_tasks`` directly without also updating the store.
3. **Streaming downloads** — bundle and RO-Crate exports are zipped on
   disk in a per-request temp dir, streamed back via
   :class:`fastapi.responses.FileResponse`, and cleaned up by a
   :class:`starlette.background.BackgroundTask`. We intentionally do
   not stream the bundle in-place: the bundle directory is mutable
   (sidecars like ``bom.cdx.json`` may be added post-seal) and reading
   it concurrently with another writer risks zip-stream corruption.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from starlette.background import BackgroundTask

from apmode import __version__ as _apmode_version
from apmode.api.models import (
    CreateRunRequest,
    HealthResponse,
    RunCreatedResponse,
    RunListResponse,
    RunStatusResponse,
)
from apmode.api.runs import execute_run
from apmode.api.store import RunRecord, RunStatus
from apmode.bundle.rocrate import RoCrateEmitter, RoCrateExportOptions
from apmode.bundle.rocrate.projector import BundleNotSealedError
from apmode.ids import generate_run_id

if TYPE_CHECKING:
    from apmode.api.runs import RunnerFactory
    from apmode.api.store import RunStore

# ``Callable`` is imported at runtime because the factory's return
# annotation references it (and ``from __future__ import annotations``
# does not help Pyright's static resolution of quoted forward refs).
from collections.abc import Callable  # noqa: TC003

logger = logging.getLogger(__name__)


# Status code 425 ("Too Early") is the closest semantic match for "the
# bundle exists but is not yet sealed" — the Submit lane uses 202 +
# Retry-After to signal "still working", and 425 mirrors that on
# polling endpoints. RFC 8470.
_HTTP_425_TOO_EARLY = 425

# Statuses past which a run is no longer cancellable. INTERRUPTED is
# included even though it represents an *unclean* end state: the
# orchestrator never saw the failure, so there is no live process to
# cancel; a re-queue worker is the right tool, not DELETE.
_TERMINAL_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.INTERRUPTED,
    }
)

# Header name used by the static-API-key auth dependency. Loopback
# deployments may run without ``APMODE_API_KEY`` set (auth disabled);
# any non-loopback bind requires the key to be set or the CLI refuses
# to start (see ``apmode.cli:_serve_main``). HMAC compare keeps the
# verification timing-side-channel-safe.
_API_KEY_HEADER = "X-API-Key"
_API_KEY_ENV = "APMODE_API_KEY"
_api_key_scheme = APIKeyHeader(name=_API_KEY_HEADER, auto_error=False)

# Concurrency cap for ``POST /runs`` — prevents an authenticated caller
# from exhausting CPU / memory / LLM-API budget by submitting an
# unbounded number of runs in parallel. The default of 8 leaves headroom
# for typical 4-core lab boxes; operators can override via the
# ``APMODE_MAX_CONCURRENT_RUNS`` env var. Each excess request gets a
# 429 with a 30 s ``Retry-After`` so polite clients back off cleanly.
_MAX_CONCURRENT_RUNS_ENV = "APMODE_MAX_CONCURRENT_RUNS"
_DEFAULT_MAX_CONCURRENT_RUNS = 8


def _resolve_max_concurrent_runs() -> int:
    raw = os.environ.get(_MAX_CONCURRENT_RUNS_ENV)
    if raw is None:
        return _DEFAULT_MAX_CONCURRENT_RUNS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_CONCURRENT_RUNS
    return max(1, value)


def _build_require_api_key() -> Callable[[str | None], None]:
    """Return a dependency that enforces ``APMODE_API_KEY`` when set.

    Reads the env var on every call (rather than at import time) so a
    rotated key takes effect without reloading the process. The early
    ``None``-return path keeps loopback dev deployments unauthenticated
    and avoids breaking the existing ``apmode serve`` UX where the
    operator does not set a key.
    """

    def _dep(api_key: str | None = Depends(_api_key_scheme)) -> None:
        expected = os.environ.get(_API_KEY_ENV)
        if not expected:
            # Auth disabled — caller is on the trusted loopback bind
            # (the CLI guard prevents this state on non-loopback hosts).
            return
        if not api_key or not hmac.compare_digest(api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing or invalid X-API-Key",
                headers={"WWW-Authenticate": _API_KEY_HEADER},
            )

    return _dep


def build_router(
    *,
    store: RunStore,
    runs_dir: Path,
    runner_factory: RunnerFactory,
    allow_backends: tuple[str, ...],
    dataset_root: Path | None = None,
) -> APIRouter:
    """Construct an :class:`fastapi.APIRouter` bound to a store + factory.

    Each handler captures ``store``, ``runs_dir``, ``runner_factory``,
    and ``allow_backends`` via closure so a single FastAPI app can host
    multiple isolated APMODE instances (different stores, different
    backends) without fighting for module-level globals.

    Two sub-routers are wired:

    * an unprotected one for ``/healthz`` so liveness probes work
      without an API key, and
    * a protected one carrying every run-management endpoint, gated by
      the ``require_api_key`` dependency. When ``APMODE_API_KEY`` is
      unset the dependency is a no-op (loopback dev mode); the CLI
      refuses to bind to non-loopback hosts in that state so the
      "auth disabled" state is never reachable from the network.
    """
    router = APIRouter()
    require_api_key = _build_require_api_key()
    protected = APIRouter(dependencies=[Depends(require_api_key)])
    create_run_lock = asyncio.Lock()

    # -----------------------------------------------------------------
    # Health (unauthenticated)
    # -----------------------------------------------------------------

    @router.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:  # pyright: ignore[reportUnusedFunction]
        return HealthResponse(apmode_version=_apmode_version)

    # -----------------------------------------------------------------
    # Runs — create + list + status
    # -----------------------------------------------------------------

    @protected.post(
        "/runs",
        response_model=RunCreatedResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_run(  # pyright: ignore[reportUnusedFunction]
        body: CreateRunRequest,
        request: Request,
        response: Response,
    ) -> RunCreatedResponse:
        if body.backend not in allow_backends:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"backend {body.backend!r} is not enabled for this server; "
                    f"allow_backends={list(allow_backends)}"
                ),
            )

        # Cap concurrency BEFORE allocating a run_id / inserting a row,
        # so a flooded server does not pollute the SQLite store with
        # rejected runs. The lock keeps the capacity check, row insert,
        # and active-task registration atomic across concurrent POSTs.
        async with create_run_lock:
            active_tasks = cast(
                "dict[str, asyncio.Task[None]]",
                request.app.state.active_tasks,
            )
            max_concurrent = _resolve_max_concurrent_runs()
            if len(active_tasks) >= max_concurrent:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"server at capacity: {len(active_tasks)} runs active "
                        f"(cap={max_concurrent}); retry after 30 s"
                    ),
                    headers={"Retry-After": "30"},
                )

            run_id = generate_run_id()
            bundle_dir = runs_dir / run_id

            record = RunRecord(
                run_id=run_id,
                status=RunStatus.PENDING,
                bundle_dir=str(bundle_dir),
                lane=body.lane,
                backend=body.backend,
                seed=body.seed,
                requeue_on_interrupt=body.requeue_on_interrupt,
            )
            await store.create(record)

            async def _on_complete(rid: str) -> None:
                active_tasks.pop(rid, None)

            task = asyncio.create_task(
                execute_run(
                    run_id=run_id,
                    bundle_dir=bundle_dir,
                    request=body,
                    runs_dir=runs_dir,
                    runner_factory=runner_factory,
                    store=store,
                    on_complete=_on_complete,
                    dataset_root=dataset_root,
                ),
                name=f"apmode-run-{run_id}",
            )
            active_tasks[run_id] = task

        # Setting Retry-After lets a polite client back off rather than
        # busy-poll. 5 s is short enough that a fast NLME fit is visible
        # within seconds and long enough that a 30-min SAEM fit doesn't
        # generate 1800 polls/min in the worst case.
        response.headers["Retry-After"] = "5"

        return RunCreatedResponse(
            run_id=run_id,
            status=RunStatus.PENDING,
            status_url=f"/runs/{run_id}/status",
        )

    @protected.get("/runs", response_model=RunListResponse)
    async def list_runs() -> RunListResponse:  # pyright: ignore[reportUnusedFunction]
        rows = await store.list()
        return RunListResponse(runs=[_to_response(r) for r in rows])

    @protected.get("/runs/{run_id}/status", response_model=RunStatusResponse)
    async def get_run_status(run_id: str) -> RunStatusResponse:  # pyright: ignore[reportUnusedFunction]
        row = await store.get(run_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"run_id {run_id!r} not found",
            )
        return _to_response(row)

    # -----------------------------------------------------------------
    # Runs — cancellation (plan Task 33)
    # -----------------------------------------------------------------

    @protected.delete(
        "/runs/{run_id}",
        response_model=RunStatusResponse,
        responses={
            404: {"description": "Run not found."},
            409: {"description": "Run is already in a terminal status."},
        },
    )
    async def cancel_run(  # pyright: ignore[reportUnusedFunction]
        run_id: str,
        request: Request,
    ) -> RunStatusResponse:
        """Cancel a PENDING / RUNNING run.

        Two-phase cancellation:

        1. ``task.cancel()`` raises :class:`asyncio.CancelledError`
           inside the coroutine. The orchestrator's runner catches it,
           SIGTERMs the child process group (5 s grace, then SIGKILL —
           see :func:`apmode.backends.process_lifecycle.terminate_process_group`)
           and re-raises so the cancellation propagates back up to
           :func:`apmode.api.runs.execute_run`.
        2. ``execute_run`` catches the CancelledError and writes
           ``RunStatus.CANCELLED`` to the store. The DELETE handler
           then waits briefly for that transition so the response
           reflects the post-cancellation row.

        Returns the (now-CANCELLED) row.
        """
        row = await store.get(run_id)
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"run_id {run_id!r} not found",
            )
        if row.status in _TERMINAL_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"run {run_id!r} is in terminal status {row.status.value!r}; nothing to cancel"
                ),
            )

        active_tasks = cast(
            "dict[str, asyncio.Task[None]]",
            request.app.state.active_tasks,
        )
        task = active_tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            # Wait for the task to actually finish cancelling so the
            # store row reflects CANCELLED before we return. We do not
            # ``await task`` (which would re-raise CancelledError);
            # instead we poll the store with a short bounded loop. The
            # 5 s budget matches the SIGTERM-to-SIGKILL grace window in
            # ``terminate_process_group``.
            await _wait_for_terminal(store, run_id, timeout_seconds=6.0)
        else:
            # No active task — the row is in PENDING/RUNNING but no
            # asyncio.Task is tracking it (e.g. a stale row left over
            # from a past process restart that the lifespan sweep
            # somehow missed). Mark CANCELLED directly so a subsequent
            # GET reflects the state.
            await store.update_status(
                run_id,
                RunStatus.CANCELLED,
                error="DELETE /runs/{id} on a row with no active task",
            )

        updated = await store.get(run_id)
        if updated is None:
            # Should be unreachable — the row existed at the start of
            # the handler and the store has no delete operation. Treat
            # as a 500 if it does happen rather than serving a stale
            # response from before the cancellation.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"run_id {run_id!r} disappeared after cancellation",
            )
        return _to_response(updated)

    # -----------------------------------------------------------------
    # Runs — bundle + RO-Crate downloads
    # -----------------------------------------------------------------

    @protected.get(
        "/runs/{run_id}/bundle",
        response_class=FileResponse,
        responses={
            404: {"description": "Run not found."},
            _HTTP_425_TOO_EARLY: {
                "description": "Run is not COMPLETED yet — bundle is unsealed.",
            },
        },
    )
    async def get_run_bundle(run_id: str) -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        row = await _require_completed(store, run_id)
        bundle_dir = Path(row.bundle_dir)
        if not bundle_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"bundle row exists but {bundle_dir} is not a directory; "
                    "the run may have been moved or pruned out-of-band"
                ),
            )

        zip_path = await asyncio.to_thread(_build_bundle_zip, bundle_dir)
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=f"{run_id}_bundle.zip",
            background=BackgroundTask(_safe_unlink, zip_path),
        )

    @protected.get(
        "/runs/{run_id}/rocrate",
        response_class=FileResponse,
        responses={
            404: {"description": "Run not found."},
            _HTTP_425_TOO_EARLY: {
                "description": "Run is not COMPLETED yet — bundle is unsealed.",
            },
        },
    )
    async def get_run_rocrate(run_id: str) -> FileResponse:  # pyright: ignore[reportUnusedFunction]
        row = await _require_completed(store, run_id)
        bundle_dir = Path(row.bundle_dir)
        if not bundle_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"bundle row exists but {bundle_dir} is not a directory; "
                    "the run may have been moved or pruned out-of-band"
                ),
            )

        try:
            crate_zip_path = await asyncio.to_thread(_build_rocrate_zip, bundle_dir)
        except BundleNotSealedError as exc:
            # Race: the row went COMPLETED but the bundle was tampered
            # with between the status check and projector inspection.
            raise HTTPException(
                status_code=_HTTP_425_TOO_EARLY,
                detail=str(exc),
            ) from exc
        return FileResponse(
            path=crate_zip_path,
            media_type="application/zip",
            filename=f"{run_id}_rocrate.zip",
            background=BackgroundTask(_safe_unlink, crate_zip_path),
        )

    # Mount the protected sub-router onto the public router so the
    # ``require_api_key`` dependency applies to every run-management
    # endpoint while ``/healthz`` remains open for liveness probes.
    router.include_router(protected)
    return router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ABS_PATH_RE = re.compile(r"(?:/|[A-Za-z]:[\\/])[\w./\-\\]+")


def _redact_error_for_api(error: str | None) -> str | None:
    """Strip Python tracebacks from the ``error`` field before serving over HTTP.

    The store row carries the full ``traceback.format_exc()`` for forensic
    use, but exposing it on ``GET /runs/{id}/status`` leaks filesystem
    paths, package versions, and environment shape — useful reconnaissance
    for a remote attacker. The redaction:

    1. Keeps the last non-empty line (typically ``ExceptionType: message``).
    2. Replaces any absolute filesystem path with ``<path>`` so a remote
       caller cannot map the server's directory structure by submitting
       deliberately bad ``dataset_path`` values.
    3. Caps total length to 240 characters as a side-channel guard.

    The full traceback remains in the bundle / server logs.
    """
    if not error:
        return error
    # Cancellation marker that ``execute_run`` writes is not a traceback;
    # surface it (path-redacted) so clients can distinguish cancellations.
    if not error.startswith("Traceback") and "\n" not in error:
        return _ABS_PATH_RE.sub("<path>", error)[:240]
    last = next(
        (line for line in reversed(error.splitlines()) if line.strip()),
        "",
    )
    if not last:
        return "internal error"
    return _ABS_PATH_RE.sub("<path>", last)[:240]


def _to_response(row: RunRecord) -> RunStatusResponse:
    """Project a persistence row onto the public response shape."""
    return RunStatusResponse(
        run_id=row.run_id,
        status=row.status,
        bundle_dir=row.bundle_dir,
        lane=row.lane,
        backend=row.backend,
        seed=row.seed,
        created_at=row.created_at,
        updated_at=row.updated_at,
        error=_redact_error_for_api(row.error),
        requeue_on_interrupt=row.requeue_on_interrupt,
    )


async def _wait_for_terminal(
    store: RunStore,
    run_id: str,
    *,
    timeout_seconds: float,
    poll_interval: float = 0.05,
) -> None:
    """Poll the store until the row reaches a terminal status or timeout.

    The DELETE handler issues ``task.cancel()`` and then waits for the
    background coroutine to actually surface CancelledError, run its
    finally clause, and persist ``CANCELLED``. Polling — rather than
    awaiting the task directly — keeps us free of CancelledError in
    the request handler and lets us bound the wait without leaking
    background-task complexity into the API contract.
    """
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while asyncio.get_event_loop().time() < deadline:
        row = await store.get(run_id)
        if row is not None and row.status in _TERMINAL_STATUSES:
            return
        await asyncio.sleep(poll_interval)
    # Timed out — fall through; the caller will read the (still
    # non-terminal) row and return it. A subsequent GET will eventually
    # see the terminal state once the runner finishes its CancelledError
    # handling.
    logger.warning(
        "api_cancel_wait_timed_out",
        extra={"run_id": run_id, "timeout_seconds": timeout_seconds},
    )


async def _require_completed(store: RunStore, run_id: str) -> RunRecord:
    """Fetch a row and raise 404/425 if it cannot serve a download."""
    row = await store.get(run_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"run_id {run_id!r} not found",
        )
    if row.status != RunStatus.COMPLETED:
        raise HTTPException(
            status_code=_HTTP_425_TOO_EARLY,
            detail=(
                f"run {run_id!r} is in status {row.status.value!r}; "
                "bundle is only downloadable once the run reaches COMPLETED"
            ),
        )
    return row


def _build_bundle_zip(bundle_dir: Path) -> Path:
    """Materialise a zip of ``bundle_dir`` under a temp file path.

    Runs in a thread (``asyncio.to_thread``) because :func:`shutil.make_archive`
    is synchronous and CPU-bound. A typical bundle is < 50 MB so the
    archive is created in a few hundred milliseconds; for very large
    runs (1+ GB with all parquet posteriors) the request will block one
    worker thread but the rest of the event loop stays responsive.
    """
    # tempfile.mkstemp returns the *file*, but make_archive wants a base
    # path with no extension. Generate a unique base via mkdtemp + a
    # known filename so the cleanup task can delete a single file.
    tmp_dir = Path(tempfile.mkdtemp(prefix="apmode_bundle_"))
    base = tmp_dir / bundle_dir.name
    archive_str = shutil.make_archive(
        base_name=str(base),
        format="zip",
        root_dir=str(bundle_dir.parent),
        base_dir=bundle_dir.name,
    )
    return Path(archive_str)


def _build_rocrate_zip(bundle_dir: Path) -> Path:
    """Project the bundle into an RO-Crate ZIP under a temp path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="apmode_rocrate_"))
    out = tmp_dir / f"{bundle_dir.name}_rocrate.zip"
    emitter = RoCrateEmitter()
    emitter.export_from_sealed_bundle(bundle_dir, out, RoCrateExportOptions())
    return out


def _safe_unlink(path: Path) -> None:
    """Cleanup background task — delete the temp zip and its parent dir.

    Wraps the unlink in a try/except so a missing file (e.g. caller
    deleted it manually) does not surface as a background-task error in
    the server logs.
    """
    try:
        if path.exists():
            path.unlink()
        parent = path.parent
        if parent.exists() and parent.name.startswith(("apmode_bundle_", "apmode_rocrate_")):
            shutil.rmtree(parent, ignore_errors=True)
    except OSError:
        logger.warning("api_temp_zip_cleanup_failed", extra={"path": str(path)})


__all__ = ["build_router"]
