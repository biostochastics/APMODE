# SPDX-License-Identifier: GPL-2.0-or-later
"""Cross-runner subprocess termination helpers (plan Task 33).

Both :class:`apmode.backends.nlmixr2_runner.Nlmixr2Runner` and
:class:`apmode.backends.bayesian_runner.BayesianRunner` spawn child
processes inside their own process group (``preexec_fn=os.setsid`` /
``start_new_session=True``). When the API receives ``DELETE /runs/{id}``
the asyncio task running the orchestrator is cancelled, which raises
:class:`asyncio.CancelledError` at the runner's
``await proc.communicate()`` call site — but the *child* R / cmdstan
process is not signalled by asyncio cancellation and would otherwise
keep running until it completed (or the operator killed it manually).

:func:`terminate_process_group` is the single point of truth for the
termination sequence both runners use:

1. ``SIGTERM`` to the process group so the child can exit cleanly. nlmixr2
   leaves intermediate `~/.cache/R/...` artefacts behind on SIGKILL but
   cleans up on SIGTERM, which is the user-visible reason for not
   skipping straight to SIGKILL.
2. Wait up to ``grace_seconds`` (default 5 s) for the process to exit.
3. ``SIGKILL`` to the process group if the child is still alive.

POSIX-only — Windows uses :meth:`asyncio.subprocess.Process.terminate`
(which sends ``CTRL_BREAK_EVENT``). The runner-side caller dispatches
on :mod:`sys.platform` so both branches stay testable without
conditional imports here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

logger = logging.getLogger(__name__)

DEFAULT_GRACE_SECONDS = 5.0
"""SIGTERM-to-SIGKILL grace window. 5 s matches the SQLite ``busy_timeout``
default the API uses elsewhere; long enough that an nlmixr2 fit doing
its final BIC computation can flush, short enough that an unresponsive
child does not hold up uvicorn graceful shutdown."""


async def terminate_process_group(
    proc: asyncio.subprocess.Process,
    *,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
) -> None:
    """SIGTERM the process group, wait for graceful exit, escalate to SIGKILL.

    Idempotent — if the child has already exited (``proc.returncode is
    not None``) the helper returns immediately. ``ProcessLookupError``
    on either signal is suppressed: the process group could have raced
    to exit between the ``returncode`` check and the ``killpg``
    syscall.

    Cancellation semantics: the *entire* SIGTERM-grace-SIGKILL sequence
    runs inside an :func:`asyncio.shield`, so a second
    :class:`asyncio.CancelledError` (e.g. lifespan shutdown firing
    while a DELETE handler is still awaiting this helper) cannot skip
    the SIGKILL escalation, orphan the inner ``proc.wait()``, or
    bypass the grace window. After the shielded sequence has either
    cleanly exited the child or escalated to SIGKILL, any pending
    cancellation is re-raised so the caller's
    :class:`asyncio.CancelledError` propagation is preserved.

    Args:
        proc: The subprocess to terminate. Must have been spawned with
            a fresh process group (``preexec_fn=os.setsid`` or
            ``start_new_session=True``).
        grace_seconds: Maximum wait between SIGTERM and SIGKILL.
    """
    if proc.returncode is not None:
        return

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        # Child already gone — nothing to signal.
        return

    async def _run_grace_then_kill() -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=grace_seconds)
        except TimeoutError:
            logger.warning(
                "subprocess_terminate_grace_exceeded",
                extra={
                    "pid": proc.pid,
                    "grace_seconds": grace_seconds,
                    "trigger": "TimeoutError",
                },
            )
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()

    # Run the SIGTERM/grace/SIGKILL sequence inside a shield so a
    # second cancellation cannot bypass the escalation. The shielded
    # task continues even if the awaiting frame is cancelled — we then
    # re-raise the captured cancellation after the child is reaped, so
    # the four-link cancellation contract documented in CLAUDE.md is
    # preserved end-to-end.
    inner_task = asyncio.ensure_future(_run_grace_then_kill())
    cancelled_during_wait: asyncio.CancelledError | None = None
    while True:
        try:
            await asyncio.shield(inner_task)
            break
        except asyncio.CancelledError as exc:
            # Capture the cancellation but keep awaiting the shielded
            # inner_task so the SIGKILL escalation actually runs to
            # completion. Loop until inner_task is done.
            cancelled_during_wait = exc

    if cancelled_during_wait is not None:
        raise cancelled_during_wait


__all__ = ["DEFAULT_GRACE_SECONDS", "terminate_process_group"]
