# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the SQLite run registry (plan Task 31).

These exercise the :class:`SQLiteRunStore` contract:

* Round-trip: a created record can be fetched back unchanged.
* PRAGMAs: WAL mode + busy_timeout are actually applied (the docs are
  emphatic that a missed ``PRAGMA journal_mode=WAL`` silently leaves
  the database in default ``delete`` mode).
* Status updates set ``updated_at`` and propagate ``error`` on failure
  paths.
* The startup sweep flips every ``RUNNING`` row to ``INTERRUPTED`` so
  the API never returns stale status after a restart (the Task 34
  lifespan hook depends on this invariant).
* Idempotent ``initialize()`` and ``close()`` so the FastAPI lifespan
  context manager can call them safely.
* Unknown ``run_id`` raises :class:`KeyError` rather than silently
  no-op'ing — silent updates would mask orchestrator bugs.
* The :class:`RunStore` Protocol is structurally satisfied by the
  concrete implementation (catches Pydantic-vs-Protocol drift early).
"""

from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from apmode.api.store import RunRecord, RunStatus, RunStore, SQLiteRunStore

# ---------------------------------------------------------------------------
# Round-trip + PRAGMA invariants (the test the plan calls out verbatim)
# ---------------------------------------------------------------------------


async def test_sqlite_store_roundtrip(tmp_path: Path) -> None:
    """Create → get → list returns the same record (the plan's smoke test)."""
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    try:
        rec = RunRecord(
            run_id="r1",
            status=RunStatus.PENDING,
            bundle_dir=str(tmp_path / "r1"),
        )
        await store.create(rec)
        got = await store.get("r1")
        assert got is not None
        assert got.run_id == "r1"
        assert got.status == RunStatus.PENDING
        assert got.bundle_dir == str(tmp_path / "r1")
        listed = await store.list()
        assert len(listed) == 1
        assert listed[0].run_id == "r1"
    finally:
        await store.close()


async def test_pragma_journal_mode_wal(tmp_path: Path) -> None:
    """PRAGMA journal_mode=WAL must persist after initialize()."""
    db_path = tmp_path / "runs.db"
    store = SQLiteRunStore(db_path)
    await store.initialize()
    try:
        async with (
            aiosqlite.connect(db_path) as db,
            db.execute("PRAGMA journal_mode;") as cur,
        ):
            row = await cur.fetchone()
        assert row is not None
        assert str(row[0]).lower() == "wal"
    finally:
        await store.close()


async def test_pragma_busy_timeout_set(tmp_path: Path) -> None:
    """PRAGMA busy_timeout must be 5 s on the store's own connection.

    The default of 0 makes every concurrent operation a coin flip; we
    rely on the 5 s retry window to absorb transient writer contention
    when the API event loop holds the write lock briefly.
    """
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    try:
        # Dropping into the private attr is intentional — we want to
        # assert the *store's* connection has the pragma, not a fresh
        # one which would defeat the test.
        conn = store._conn
        assert conn is not None
        async with conn.execute("PRAGMA busy_timeout;") as cur:
            row = await cur.fetchone()
        assert row is not None
        timeout = row[0]
        assert isinstance(timeout, int)
        assert timeout == 5000
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Initialize / close idempotency
# ---------------------------------------------------------------------------


async def test_initialize_is_idempotent(tmp_path: Path) -> None:
    """Repeated initialize() calls must not double-open the DB."""
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    await store.initialize()
    try:
        # Sanity: round-trip still works after the second initialize.
        rec = RunRecord(run_id="x", status=RunStatus.PENDING, bundle_dir=str(tmp_path / "x"))
        await store.create(rec)
        assert await store.get("x") is not None
    finally:
        await store.close()


async def test_close_is_idempotent(tmp_path: Path) -> None:
    """Repeated close() calls must not raise."""
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    await store.close()
    await store.close()


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


async def test_update_status_promotes_to_running(tmp_path: Path) -> None:
    """update_status mutates status and bumps updated_at."""
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    try:
        rec = RunRecord(run_id="r1", status=RunStatus.PENDING, bundle_dir=str(tmp_path / "r1"))
        await store.create(rec)
        before = await store.get("r1")
        assert before is not None

        await store.update_status("r1", RunStatus.RUNNING)
        after = await store.get("r1")
        assert after is not None
        assert after.status == RunStatus.RUNNING
        # updated_at is a UTC ISO-8601 timestamp; lexicographic
        # comparison preserves chronological order so we don't need to
        # parse it back into datetime.
        assert after.updated_at >= before.updated_at
        assert after.error is None
    finally:
        await store.close()


async def test_update_status_records_error_on_failure(tmp_path: Path) -> None:
    """error column is populated when status moves to FAILED."""
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    try:
        rec = RunRecord(run_id="r1", status=RunStatus.RUNNING, bundle_dir=str(tmp_path / "r1"))
        await store.create(rec)
        await store.update_status("r1", RunStatus.FAILED, error="cmdstanpy crashed")
        after = await store.get("r1")
        assert after is not None
        assert after.status == RunStatus.FAILED
        assert after.error == "cmdstanpy crashed"
    finally:
        await store.close()


async def test_update_status_unknown_run_raises(tmp_path: Path) -> None:
    """Updating an unknown run_id raises KeyError instead of silent no-op.

    A silent no-op would mask orchestrator bugs where the record was
    never actually created; better to fail loudly so the bug surfaces
    in the orchestrator log.
    """
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    try:
        with pytest.raises(KeyError, match="not found"):
            await store.update_status("nope", RunStatus.RUNNING)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# create — duplicate keys raise
# ---------------------------------------------------------------------------


async def test_create_duplicate_run_id_raises(tmp_path: Path) -> None:
    """A second create() with the same run_id raises (PRIMARY KEY).

    Catches the case where the API accidentally re-dispatches a run
    rather than picking up an existing one.
    """
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    try:
        rec = RunRecord(run_id="r1", status=RunStatus.PENDING, bundle_dir=str(tmp_path / "r1"))
        await store.create(rec)
        with pytest.raises(aiosqlite.IntegrityError):
            await store.create(rec)
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# sweep_interrupted_on_startup
# ---------------------------------------------------------------------------


async def test_sweep_runs_during_initialize(tmp_path: Path) -> None:
    """initialize() runs the sweep before returning so RUNNING never leaks.

    Simulates a process that crashed mid-run (status RUNNING, no error)
    by writing a row with the underlying aiosqlite connection, closing
    the store, then re-opening it — initialize() should mark the row
    INTERRUPTED before any caller can fetch it.
    """
    db_path = tmp_path / "runs.db"
    store = SQLiteRunStore(db_path)
    await store.initialize()
    try:
        rec = RunRecord(run_id="r1", status=RunStatus.RUNNING, bundle_dir=str(tmp_path / "r1"))
        await store.create(rec)
    finally:
        await store.close()

    # Re-open: a fresh store should sweep the orphan RUNNING row.
    store2 = SQLiteRunStore(db_path)
    await store2.initialize()
    try:
        got = await store2.get("r1")
        assert got is not None
        assert got.status == RunStatus.INTERRUPTED
        # Sweep records a hint in error so operators can distinguish a
        # deliberate FAILED from a swept INTERRUPTED.
        assert got.error is not None
        assert "process exited" in got.error
    finally:
        await store2.close()


async def test_sweep_returns_count(tmp_path: Path) -> None:
    """sweep_interrupted_on_startup returns the number of rows swept."""
    db_path = tmp_path / "runs.db"
    store = SQLiteRunStore(db_path)
    await store.initialize()
    try:
        # Create 3 RUNNING + 1 COMPLETED + 1 FAILED. Only the 3 should
        # be swept.
        for i, status in enumerate(
            (
                RunStatus.RUNNING,
                RunStatus.RUNNING,
                RunStatus.RUNNING,
                RunStatus.COMPLETED,
                RunStatus.FAILED,
            ),
            start=1,
        ):
            await store.create(
                RunRecord(
                    run_id=f"r{i}",
                    status=status,
                    bundle_dir=str(tmp_path / f"r{i}"),
                )
            )
        # Re-running the sweep on the already-initialized store should
        # report 3, then 0 on the next call.
        first = await store.sweep_interrupted_on_startup()
        second = await store.sweep_interrupted_on_startup()
        assert first == 3
        assert second == 0
    finally:
        await store.close()


async def test_sweep_does_not_touch_non_running(tmp_path: Path) -> None:
    """COMPLETED / FAILED / CANCELLED rows are left intact by the sweep."""
    db_path = tmp_path / "runs.db"
    store = SQLiteRunStore(db_path)
    await store.initialize()
    try:
        for i, status in enumerate(
            (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED),
            start=1,
        ):
            await store.create(
                RunRecord(
                    run_id=f"r{i}",
                    status=status,
                    bundle_dir=str(tmp_path / f"r{i}"),
                )
            )
        await store.sweep_interrupted_on_startup()
        for i, expected in enumerate(
            (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED),
            start=1,
        ):
            got = await store.get(f"r{i}")
            assert got is not None
            assert got.status == expected
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# list ordering
# ---------------------------------------------------------------------------


async def test_list_orders_by_created_at_ascending(tmp_path: Path) -> None:
    """list() returns rows oldest-first.

    The API surfaces list() through GET /runs which the dashboard sorts
    chronologically; pinning the order here means the API doesn't have
    to re-sort on every request.
    """
    store = SQLiteRunStore(tmp_path / "runs.db")
    await store.initialize()
    try:
        # Manually craft created_at so the test is deterministic — the
        # default factory uses datetime.now() which is racy at sub-ms
        # resolution.
        for i, ts in enumerate(
            (
                "2026-04-01T00:00:00+00:00",
                "2026-04-02T00:00:00+00:00",
                "2026-04-03T00:00:00+00:00",
            ),
            start=1,
        ):
            await store.create(
                RunRecord(
                    run_id=f"r{i}",
                    status=RunStatus.PENDING,
                    bundle_dir=str(tmp_path / f"r{i}"),
                    created_at=ts,
                    updated_at=ts,
                )
            )
        listed = await store.list()
        assert [r.run_id for r in listed] == ["r1", "r2", "r3"]
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Protocol structural conformance
# ---------------------------------------------------------------------------


def test_sqlite_store_satisfies_runstore_protocol(tmp_path: Path) -> None:
    """SQLiteRunStore must satisfy the RunStore Protocol structurally.

    Catches drift between the Protocol's declared methods and the
    concrete implementation (e.g. someone adds a method to the Protocol
    but forgets to wire SQLiteRunStore).
    """
    store = SQLiteRunStore(tmp_path / "runs.db")
    assert isinstance(store, RunStore)


# ---------------------------------------------------------------------------
# Pre-initialize calls fail loudly
# ---------------------------------------------------------------------------


async def test_create_before_initialize_raises(tmp_path: Path) -> None:
    """Calling store methods before initialize() must raise, not deadlock."""
    store = SQLiteRunStore(tmp_path / "runs.db")
    rec = RunRecord(run_id="r1", status=RunStatus.PENDING, bundle_dir=str(tmp_path / "r1"))
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.create(rec)


# ---------------------------------------------------------------------------
# RunRecord schema invariants
# ---------------------------------------------------------------------------


def test_run_record_rejects_empty_run_id() -> None:
    """run_id min_length=1 must be enforced by Pydantic."""
    with pytest.raises(ValueError, match="at least 1 character"):
        RunRecord(run_id="", status=RunStatus.PENDING, bundle_dir="/tmp/x")


def test_run_record_rejects_empty_bundle_dir() -> None:
    with pytest.raises(ValueError, match="at least 1 character"):
        RunRecord(run_id="r1", status=RunStatus.PENDING, bundle_dir="")


def test_run_status_values_match_strings() -> None:
    """RunStatus is a StrEnum so the underlying value is the SQL column."""
    assert RunStatus.PENDING.value == "pending"
    assert RunStatus.INTERRUPTED.value == "interrupted"
