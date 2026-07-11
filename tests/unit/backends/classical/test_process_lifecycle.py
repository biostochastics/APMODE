# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for :func:`apmode.backends.process_lifecycle.terminate_process_group`.

Plan Task 33 — verify the SIGTERM-then-SIGKILL escalation against
real, short-lived child processes spawned in their own session group.
We use ``sleep`` as a stand-in for an unresponsive R / cmdstan child.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys

import pytest

from apmode.backends.process_lifecycle import terminate_process_group

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="process-group semantics are POSIX-only; Windows uses proc.terminate()",
)


async def _spawn_sleep(duration_seconds: int = 60) -> asyncio.subprocess.Process:
    """Spawn a long-lived ``sleep`` child in a fresh process group."""
    return await asyncio.create_subprocess_exec(
        "sleep",
        str(duration_seconds),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )


@pytest.mark.asyncio
async def test_terminate_process_group_sigterm_path() -> None:
    """``sleep`` exits cleanly on SIGTERM — no SIGKILL escalation needed."""
    proc = await _spawn_sleep(60)
    await terminate_process_group(proc, grace_seconds=2.0)
    # SIGTERM exits with code -15 on POSIX (negated signal number).
    assert proc.returncode == -15


@pytest.mark.asyncio
async def test_terminate_process_group_idempotent_after_exit() -> None:
    """Calling on an already-exited process is a no-op (no exception)."""
    proc = await _spawn_sleep(60)
    await terminate_process_group(proc, grace_seconds=1.0)
    assert proc.returncode is not None
    # Second call must not raise even though the child is gone.
    await terminate_process_group(proc, grace_seconds=1.0)


@pytest.mark.asyncio
async def test_terminate_process_group_handles_self_completed() -> None:
    """A child that exits on its own still leaves a clean post-call state."""
    proc = await _spawn_sleep(0)
    await proc.wait()
    assert proc.returncode == 0
    # Calling terminate now must be a no-op.
    await terminate_process_group(proc, grace_seconds=1.0)
    assert proc.returncode == 0


@pytest.mark.asyncio
async def test_terminate_group_after_leader_exits_kills_grandchild() -> None:
    """A reaped session leader must not make its surviving child invisible."""
    script = (
        "import subprocess,sys; "
        "p=subprocess.Popen(['sleep','60'], stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL); print(p.pid, flush=True)"
    )
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    assert proc.stdout is not None
    grandchild_pid = int((await proc.stdout.readline()).decode().strip())
    await proc.wait()
    assert proc.returncode == 0
    os.kill(grandchild_pid, 0)  # regression precondition: child survived leader

    try:
        await terminate_process_group(proc, grace_seconds=0.05)
        for _ in range(50):
            try:
                os.kill(grandchild_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("grandchild survived process-group termination")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild_pid, signal.SIGKILL)
