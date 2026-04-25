# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the PR4 streaming addition to ``Nlmixr2Runner._spawn_r``.

The new path teases two helpers apart:

* :func:`apmode.backends.nlmixr2_runner._drain_pipe` — line-by-line
  reader that tees raw bytes to a file and invokes a per-line
  callback. The deadlock-prevention contract (drain must keep up with
  a chatty writer) is verified end-to-end with a Python subprocess
  emitting ~6 MB of stderr.
* The ``audit_log_path`` + ``on_stderr_line`` keyword arguments on
  ``_spawn_r`` are wired into the drain helpers so an existing
  ``Nlmixr2Runner`` can opt in without changing its public API.

Tests live in ``tests/unit/`` because they only spawn lightweight
Python subprocesses; they do **not** exercise R or nlmixr2.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from apmode.backends.nlmixr2_runner import _drain_pipe


async def _spawn_python(*code_args: str) -> asyncio.subprocess.Process:
    """Convenience: launch the current Python with ``-c`` and capture both pipes."""
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        *code_args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# _drain_pipe — basic contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_pipe_collects_all_lines_in_order(tmp_path: Path) -> None:
    """Three-line stderr stream: every line tee'd + delivered in order."""
    proc = await _spawn_python(
        textwrap.dedent(
            """
            import sys
            sys.stderr.write('alpha\\n')
            sys.stderr.write('beta\\n')
            sys.stderr.write('gamma\\n')
            """
        )
    )
    captured: list[bytes] = []
    log_path = tmp_path / "audit.log"
    log_handle = log_path.open("wb", buffering=0)
    try:
        assert proc.stderr is not None
        body = await _drain_pipe(proc.stderr, log_handle, captured.append)
    finally:
        log_handle.close()
        await proc.wait()

    assert captured == [b"alpha\n", b"beta\n", b"gamma\n"]
    assert body == b"alpha\nbeta\ngamma\n"
    assert log_path.read_bytes() == body


@pytest.mark.asyncio
async def test_drain_pipe_returns_full_body_when_callback_is_none() -> None:
    """``on_line=None`` is supported — the body is still returned."""
    proc = await _spawn_python("import sys; sys.stderr.write('one\\ntwo\\n')")
    assert proc.stderr is not None
    body = await _drain_pipe(proc.stderr, None, None)
    await proc.wait()
    assert body == b"one\ntwo\n"


@pytest.mark.asyncio
async def test_drain_pipe_returns_empty_for_empty_stream() -> None:
    """Process that closes stderr immediately → empty body, no crash."""
    proc = await _spawn_python("import sys; sys.stderr.close()")
    assert proc.stderr is not None
    body = await _drain_pipe(proc.stderr, None, None)
    await proc.wait()
    assert body == b""


# ---------------------------------------------------------------------------
# Deadlock prevention — chatty stderr (~6 MB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_pipe_does_not_deadlock_on_chatty_stderr(
    tmp_path: Path,
) -> None:
    """Without concurrent draining, a writer that fills its 64 KB pipe
    buffer would block. The drain helper plus the surrounding
    ``asyncio.gather`` in ``_spawn_r`` must keep up with a high-volume
    writer; we verify with 100k lines (~6 MB total) of stderr.
    """
    proc = await _spawn_python(
        textwrap.dedent(
            """
            import sys
            for i in range(100_000):
                sys.stderr.write(f'iter {i:06d} OBJF=1234.5\\n')
            """
        )
    )
    lines: list[bytes] = []
    log_path = tmp_path / "chatty.log"
    log_handle = log_path.open("wb", buffering=0)
    try:
        assert proc.stderr is not None
        assert proc.stdout is not None
        await asyncio.wait_for(
            asyncio.gather(
                _drain_pipe(proc.stderr, log_handle, lines.append),
                _drain_pipe(proc.stdout, None, None),
            ),
            timeout=30.0,
        )
        await proc.wait()
    finally:
        log_handle.close()

    assert len(lines) == 100_000
    assert lines[0] == b"iter 000000 OBJF=1234.5\n"
    assert lines[-1] == b"iter 099999 OBJF=1234.5\n"
    # Audit log captures every byte verbatim (no buffering loss).
    assert log_path.stat().st_size == sum(len(b) for b in lines)


# ---------------------------------------------------------------------------
# Callback robustness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_pipe_swallows_callback_exceptions() -> None:
    """A callback that raises must not interrupt the drain — otherwise
    the OS pipe could fill and the subprocess would deadlock."""
    proc = await _spawn_python(
        textwrap.dedent(
            """
            import sys
            for i in range(5):
                sys.stderr.write(f'line {i}\\n')
            """
        )
    )

    seen: list[bytes] = []

    def boom(line: bytes) -> None:
        seen.append(line)
        if line == b"line 2\n":
            raise RuntimeError("simulated parser failure")

    assert proc.stderr is not None
    body = await _drain_pipe(proc.stderr, None, boom)
    await proc.wait()
    # The callback was called for EVERY line — including those after
    # the exception — and the body matches what was on the pipe.
    assert len(seen) == 5
    assert b"line 4\n" in body


@pytest.mark.asyncio
async def test_drain_pipe_continues_when_log_write_fails(
    tmp_path: Path,
) -> None:
    """If the audit log handle dies mid-stream, the drain keeps going."""

    class FailingHandle:
        """File-like that raises on the second write."""

        def __init__(self) -> None:
            self.calls = 0

        def write(self, data: bytes) -> int:
            self.calls += 1
            if self.calls == 2:
                raise OSError("disk full simulator")
            return len(data)

        def close(self) -> None:
            pass

    proc = await _spawn_python("import sys; sys.stderr.write('a\\nb\\nc\\nd\\n')")
    handle = FailingHandle()
    seen: list[bytes] = []
    assert proc.stderr is not None
    body = await _drain_pipe(proc.stderr, handle, seen.append)  # type: ignore[arg-type]
    await proc.wait()
    # Drain still consumed every line via the on_line callback.
    assert seen == [b"a\n", b"b\n", b"c\n", b"d\n"]
    assert body == b"a\nb\nc\nd\n"


# ---------------------------------------------------------------------------
# Long-line handling (LimitOverrunError fallback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_pipe_handles_overrun_without_raising() -> None:
    """A line beyond the default 64 KiB StreamReader limit triggers
    ``ValueError`` from ``readline()`` (Python 3.12+; older Pythons
    raise ``LimitOverrunError`` directly). The helper must catch both,
    fall back to a chunk read, and not crash the drain.

    Note: when overrun fires the StreamReader clears its buffer, so a
    portion of the long line is lost — that is an unavoidable cost of
    the default StreamReader. ``Nlmixr2Runner._spawn_r`` bumps the
    limit to 1 MiB at spawn time so this scenario is exceedingly rare
    in practice; this test pins only the no-crash contract.
    """
    proc = await _spawn_python(
        textwrap.dedent(
            """
            import sys
            sys.stderr.write('X' * 200_000)
            """
        )
    )
    seen: list[bytes] = []
    assert proc.stderr is not None
    body = await _drain_pipe(proc.stderr, None, seen.append)
    await proc.wait()
    # No exception leaked; received at least some bytes.
    assert isinstance(body, bytes)
    assert b"X" in body


# ---------------------------------------------------------------------------
# End-to-end: drain → callback → SAEM parser
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_to_parser_pipeline_replays_real_fixture() -> None:
    """Pump the captured nlmixr2 5.0 fixture through ``_drain_pipe`` and
    feed each line into the parser via ``on_stderr_line``.

    Verifies the load-bearing PR4 chain: a real R subprocess could
    replace the Python subprocess below and the parser would still see
    60 iteration states with correct phase classification — because
    the only thing the runner contributes is the stderr stream that
    the parser already understands.
    """
    from apmode.backends.saem_progress import SAEMLineParser, SAEMState

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "saem" / "theo_sd_30b_30e.log"
    fixture_bytes = fixture.read_bytes()

    # Spawn a Python subprocess that replays the fixture verbatim on stderr.
    # The drain helper sees the same byte stream nlmixr2 would emit.
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        textwrap.dedent(
            f"""
            import sys
            sys.stderr.buffer.write({fixture_bytes!r})
            sys.stderr.buffer.flush()
            """
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    parser = SAEMLineParser(nburn=30)
    states: list[SAEMState] = []

    def on_line(raw: bytes) -> None:
        s = parser.parse(raw)
        if s is not None:
            states.append(s)

    assert proc.stderr is not None
    assert proc.stdout is not None
    await asyncio.gather(
        _drain_pipe(proc.stderr, None, on_line),
        _drain_pipe(proc.stdout, None, None),
    )
    await proc.wait()

    # Same assertions as the parser unit tests — pinned end-to-end.
    assert len(states) == 60
    assert [s.iteration for s in states] == list(range(1, 61))
    burnin = sum(1 for s in states if s.phase == "burnin")
    main = sum(1 for s in states if s.phase == "main")
    assert burnin == 30
    assert main == 30
