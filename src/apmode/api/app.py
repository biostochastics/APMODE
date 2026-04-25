# SPDX-License-Identifier: GPL-2.0-or-later
"""FastAPI application factory for APMODE (plan Tasks 32 + 34).

:func:`build_app` returns a fully wired :class:`fastapi.FastAPI` whose
routes are produced by :mod:`apmode.api.routes` and whose lifespan
manages the :class:`apmode.api.store.SQLiteRunStore` and the live
``app.state.active_tasks`` dict.

This module is the *only* place the API package constructs a default
runner, so a follow-up that swaps Nlmixr2Runner for an injected
runner_factory in tests does not have to chase global state.

Lifespan contract (plan Task 34)
--------------------------------

Startup
    1. ``store.initialize()`` opens the connection, sets the WAL
       pragmas, and runs ``sweep_interrupted_on_startup`` *before* the
       app is allowed to accept requests. Any row left in ``RUNNING``
       by a crashed prior process is immediately reconciled to
       ``INTERRUPTED`` with a sweep note in ``error``.
    2. ``app.state.active_tasks: dict[str, asyncio.Task[None]]`` is
       seeded as an empty dict. Every ``POST /runs`` mutates this dict
       and every ``DELETE /runs/{id}`` reads from it.
    3. ``app.state.store`` and ``app.state.runs_dir`` are exposed for
       downstream middleware / dependency injection.

Shutdown
    1. Every still-running task in ``app.state.active_tasks`` is
       ``task.cancel()``ed. We do *not* await each task — uvicorn's
       ``timeout_graceful_shutdown`` (recommend ``30`` s; covers the
       ``terminate_process_group`` SIGTERM-then-SIGKILL window plus a
       little headroom for the runner's CancelledError handler) owns
       the wait budget.
    2. ``store.close()`` closes the aiosqlite connection.

The recommended uvicorn invocation (see plan Task 35 for the
``apmode serve`` CLI):

    uvicorn.run(
        app,
        host="127.0.0.1",
        timeout_graceful_shutdown=30,
    )

A 30 s budget pairs with the 5 s SIGTERM-to-SIGKILL grace in
:func:`apmode.backends.process_lifecycle.terminate_process_group` and
leaves headroom for the runner CancelledError path to flush partial
artifacts and update the row to ``CANCELLED`` / ``FAILED`` before
uvicorn force-closes the worker.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI

from apmode.api.routes import build_router
from apmode.api.store import SQLiteRunStore

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator

    from apmode.api.runs import RunnerFactory
    from apmode.api.store import RunStore

logger = logging.getLogger(__name__)

# Default backend allowlist for v0.6-rc1. Task 36 will add
# ``bayesian_stan`` as an experimental opt-in once the BayesianRunner is
# wired through the request body resolver.
DEFAULT_ALLOW_BACKENDS: tuple[str, ...] = ("nlmixr2",)


def _default_runner_factory(work_dir: Path) -> object:
    """Construct a default :class:`Nlmixr2Runner` rooted at ``work_dir``.

    Imported lazily so that tests can build the app with a fake factory
    on a machine that has no R installed (Nlmixr2Runner's ``__init__``
    raises ``FileNotFoundError`` when ``Rscript`` is absent).
    """
    from apmode.backends.nlmixr2_runner import Nlmixr2Runner

    return Nlmixr2Runner(work_dir=work_dir)


def build_app(
    *,
    runs_dir: Path,
    db_path: Path,
    allow_backends: tuple[str, ...] = DEFAULT_ALLOW_BACKENDS,
    runner_factory: RunnerFactory | None = None,
    store: RunStore | None = None,
) -> FastAPI:
    """Wire the APMODE HTTP API together.

    Args:
        runs_dir: Base directory for per-run bundle output. Each run's
            bundle lives at ``runs_dir/<run_id>/`` (the orchestrator's
            ``BundleEmitter`` chooses the leaf name from the API's
            pre-allocated run_id, plan Task 32 contract).
        db_path: SQLite file path for the :class:`SQLiteRunStore`. The
            parent directory is created if missing.
        allow_backends: Backends accepted in the ``POST /runs`` body.
            Default is ``("nlmixr2",)``; Task 36 expands this to
            include ``bayesian_stan``.
        runner_factory: Optional override for the per-run backend
            constructor. Defaults to :func:`_default_runner_factory`
            which builds a vanilla ``Nlmixr2Runner``. Tests should
            inject a fake here to avoid depending on R.
        store: Optional pre-built :class:`RunStore`. When provided the
            lifespan hook still calls ``initialize()`` / ``close()`` so
            the same store object can be reused across test invocations
            without leaking connections.

    Returns:
        A configured :class:`FastAPI` instance ready for
        :func:`uvicorn.run` or test-client mounting.
    """
    runs_dir = Path(runs_dir).resolve()
    db_path = Path(db_path).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    factory: RunnerFactory = runner_factory or _default_runner_factory  # type: ignore[assignment]
    backing_store: RunStore = store or SQLiteRunStore(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # ``store.initialize()`` runs the INTERRUPTED sweep internally
        # (see SQLiteRunStore docstring), so any RUNNING row left by a
        # crashed prior process is reconciled before the first request.
        await backing_store.initialize()
        app.state.active_tasks = {}
        app.state.store = backing_store
        app.state.runs_dir = runs_dir
        try:
            yield
        finally:
            # Cancel every still-running task. We do *not* wait for
            # completion here — uvicorn's ``timeout_graceful_shutdown``
            # owns the wait budget, and the task wrapper updates the
            # store row on cancellation regardless.
            tasks: dict[str, asyncio.Task[None]] = app.state.active_tasks
            for task in tasks.values():
                if not task.done():
                    task.cancel()
            await backing_store.close()

    app = FastAPI(
        title="APMODE HTTP API",
        version="0.6.0-rc1",
        description=(
            "Local HTTP surface for the APMODE governed PK modeling pipeline. "
            "Bind to localhost; this API is not hardened for public exposure "
            "(plan Task 35: ``apmode serve`` defaults to 127.0.0.1)."
        ),
        lifespan=lifespan,
    )
    router = build_router(
        store=backing_store,
        runs_dir=runs_dir,
        runner_factory=factory,
        allow_backends=allow_backends,
    )
    app.include_router(router)
    return app


__all__ = ["DEFAULT_ALLOW_BACKENDS", "build_app"]
