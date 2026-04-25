# SPDX-License-Identifier: GPL-2.0-or-later
"""Background-task glue between the API routes and the Orchestrator.

Plan Task 32 routes ``POST /runs`` to :func:`execute_run_in_background`,
which is fire-and-forget on an ``asyncio.create_task``. This module owns
the *only* place where the orchestrator is invoked from the API layer
so:

* Status transitions (PENDING → RUNNING → COMPLETED / FAILED /
  CANCELLED) live in one location and stay consistent across endpoints.
* The exception envelope is uniform — every uncaught error becomes a
  ``FAILED`` row whose ``error`` column carries the traceback. Without
  this wrapper an unhandled exception inside ``asyncio.create_task``
  would be swallowed silently by the event loop default handler and the
  API would serve a perpetual ``RUNNING`` row.
* :class:`asyncio.CancelledError` is caught explicitly and recorded as
  ``CANCELLED`` with the cancellation re-raised, which is the contract
  Task 33 (``DELETE /runs/{id}``) relies on.

Backend construction is delegated to a caller-supplied
``runner_factory`` so the unit tests can substitute a fake runner that
does not require an R installation.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from apmode.api.store import RunStatus
from apmode.data.ingest import ingest_nonmem_csv
from apmode.orchestrator import Orchestrator, RunConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from apmode.api.models import CreateRunRequest
    from apmode.api.store import RunStore
    from apmode.backends.protocol import BackendRunner

logger = logging.getLogger(__name__)


class RunnerFactory(Protocol):
    """Callable that materialises a :class:`BackendRunner` for one run.

    The API holds one factory per app instance so a per-run runner can
    be configured (e.g. work_dir under the run's bundle dir). Defining
    it as a :class:`typing.Protocol` keeps the test fakes free of
    ``Nlmixr2Runner``-specific imports.
    """

    def __call__(self, work_dir: Path) -> BackendRunner: ...


async def execute_run(
    *,
    run_id: str,
    bundle_dir: Path,
    request: CreateRunRequest,
    runs_dir: Path,
    runner_factory: RunnerFactory,
    store: RunStore,
    on_complete: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Run the orchestrator and update the :class:`RunStore` along the way.

    Designed to be the coroutine wrapped by ``asyncio.create_task`` from
    the route handler. Three terminal transitions:

    * Normal completion → ``RunStatus.COMPLETED`` (only after the
      orchestrator's ``emitter.seal()`` succeeds, which is the bundle's
      integrity anchor).
    * :class:`asyncio.CancelledError` → ``RunStatus.CANCELLED`` with a
      short ``error`` note, then re-raised so the parent
      ``asyncio.Task`` surfaces ``CancelledError`` to anyone awaiting
      it.
    * Any other exception → ``RunStatus.FAILED`` with the traceback
      stored on the row. We intentionally swallow non-cancellation
      exceptions here so a buggy orchestrator path cannot crash the API
      process or leak a stale ``RUNNING`` row.

    ``on_complete`` is an optional callback invoked once the row reaches
    a terminal state, regardless of outcome. Task 33 uses this to clear
    the run from ``app.state.active_tasks``.
    """
    try:
        await store.update_status(run_id, RunStatus.RUNNING)

        dataset_path = Path(request.dataset_path).expanduser().resolve()
        if not dataset_path.is_file():
            msg = f"dataset_path does not exist or is not a file: {dataset_path}"
            raise FileNotFoundError(msg)

        # Build orchestrator. The runner's work dir is scoped to the run
        # bundle dir so a crashed run leaves diagnostic intermediates
        # next to the partial bundle (rather than leaking into a shared
        # work dir where the next run would clobber them).
        runner_work_dir = bundle_dir / "_runner_work"
        runner_work_dir.mkdir(parents=True, exist_ok=True)
        runner = runner_factory(runner_work_dir)

        config = RunConfig(
            lane=request.lane,
            seed=request.seed,
            timeout_seconds=request.timeout_seconds,
            max_concurrency=request.max_concurrency,
            covariate_names=list(request.covariate_names),
            context_of_use=request.context_of_use,
        )
        orchestrator = Orchestrator(runner, runs_dir, config)

        manifest, df = ingest_nonmem_csv(
            dataset_path,
            column_mapping=request.column_mapping,
        )

        await orchestrator.run(manifest, df, dataset_path, run_id=run_id)
        await store.update_status(run_id, RunStatus.COMPLETED)
        logger.info("api_run_completed", extra={"run_id": run_id})

    except asyncio.CancelledError:
        # The DELETE handler triggered ``task.cancel()``. Record the
        # cancellation but re-raise so the orchestrator's structured
        # cancellation propagates up through any awaiting code (and so
        # the asyncio default handler does not log a spurious "task
        # exception was never retrieved" warning).
        await store.update_status(
            run_id,
            RunStatus.CANCELLED,
            error="run cancelled via DELETE /runs/{id}",
        )
        logger.info("api_run_cancelled", extra={"run_id": run_id})
        raise

    except BaseException as exc:
        # Catch BaseException so SystemExit / KeyboardInterrupt also
        # update the row before propagating (a shutdown mid-run should
        # not leave a stale RUNNING). Re-raise non-Exception subclasses.
        tb = traceback.format_exc()
        try:
            await store.update_status(run_id, RunStatus.FAILED, error=tb)
        except Exception:
            logger.exception(
                "api_run_failed_status_update_failed",
                extra={"run_id": run_id},
            )
        logger.error(
            "api_run_failed",
            extra={"run_id": run_id, "exception_type": type(exc).__name__},
        )
        if not isinstance(exc, Exception):
            raise
        # Suppress regular exceptions: the row already reflects the
        # failure, and asyncio.Task would otherwise log
        # "Task exception was never retrieved" when the caller never
        # awaits the task (which is the entire point of fire-and-forget).
        return

    finally:
        if on_complete is not None:
            try:
                await on_complete(run_id)
            except Exception:
                logger.exception(
                    "api_run_on_complete_callback_failed",
                    extra={"run_id": run_id},
                )


__all__ = ["RunnerFactory", "execute_run"]
