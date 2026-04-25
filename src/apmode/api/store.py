# SPDX-License-Identifier: GPL-2.0-or-later
"""Async run registry for the APMODE HTTP API (plan Task 31).

The :class:`SQLiteRunStore` persists :class:`RunRecord` rows for every
``POST /runs`` request so the FastAPI app can look up status/bundle for
a long-running PK fit. SQLite was chosen over Postgres because:

* APMODE is a single-process service (CLI + optional ``apmode serve``).
  The run table is small (``~1000`` rows / month at one job per hour)
  and operations are infrequent (``~10`` jobs / day in typical use).
* The whole bundle directory is the source of truth for results — the
  store only persists *status* + ``bundle_dir``. We deliberately avoid
  duplicating bundle artifact data into the SQL row.
* Zero-ops: the database file lives next to the bundle directories;
  no separate service needs to be administered.

Concurrency model (informed by sqlite.org/wal.html and the busy-timeout
guidance the SQLite docs and aiosqlite issues converge on):

* :pep:`249`-style ``aiosqlite.Connection`` is held for the lifetime of
  the store. The connection is opened in :meth:`SQLiteRunStore.initialize`
  and closed in :meth:`SQLiteRunStore.close`. One persistent connection
  is enough for our throughput and keeps the WAL page cache hot.
* Writes are serialised through an :class:`asyncio.Lock` that wraps
  every ``BEGIN IMMEDIATE`` transaction. ``BEGIN IMMEDIATE`` acquires
  the SQLite ``RESERVED`` lock at transaction start, eliminating the
  deferred-read-then-escalate deadlock vector that bites concurrent
  writers in default-mode WAL.
* ``PRAGMA journal_mode=WAL;`` is set on initialize. ``PRAGMA
  synchronous=NORMAL;`` is the recommended companion for WAL — it keeps
  durability across application crashes (the WAL is fsync'd at every
  commit) and only relaxes durability across power loss / hard reboot
  (where the most recent transactions may roll back). This trade-off
  matches a research-grade modeling service: a hard-reset post-commit
  loss of the latest run-status update is far less expensive than
  fsyncing the database file on every status write.
* ``PRAGMA busy_timeout=5000;`` is set so any contended write retries
  for up to 5 seconds before raising ``sqlite3.OperationalError``.

Status semantics:

* :attr:`RunStatus.PENDING` — created by ``POST /runs``, not yet
  scheduled.
* :attr:`RunStatus.RUNNING` — orchestrator has started executing.
* :attr:`RunStatus.COMPLETED` — bundle is sealed (``_COMPLETE`` written)
  and digest verified.
* :attr:`RunStatus.FAILED` — runner raised; ``error`` carries the
  traceback (TEXT, not a separate table — tracebacks are always read
  with the row and never queried independently at this volume).
* :attr:`RunStatus.CANCELLED` — ``DELETE /runs/{id}`` (Task 33) caused
  the orchestrator to terminate the run.
* :attr:`RunStatus.INTERRUPTED` — set by
  :meth:`SQLiteRunStore.sweep_interrupted_on_startup` for any rows still
  marked ``RUNNING`` when the API process restarts. The Task 34 lifespan
  hook calls this in :meth:`SQLiteRunStore.initialize`'s shadow so the
  app never serves stale ``RUNNING`` for a job whose process is gone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Status enum + record schema
# ---------------------------------------------------------------------------


class RunStatus(StrEnum):
    """Lifecycle states for a run row."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # Set by the startup sweep when a row was still RUNNING at process
    # restart. Distinct from FAILED because the orchestrator never
    # observed the failure — a downstream operator may want to requeue.
    INTERRUPTED = "interrupted"


class RunRecord(BaseModel):
    """One row in the run registry.

    The ``run_id`` is the primary key (string so the API is free to use
    UUIDs, sparkids, or content hashes). ``bundle_dir`` is the absolute
    path to the run's reproducibility bundle on disk; the bundle is the
    source of truth for results, this row is metadata only.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    status: RunStatus
    bundle_dir: str = Field(min_length=1)
    # Optional — populated by the orchestrator after dispatch.
    lane: str | None = None
    backend: str | None = None
    seed: int | None = None
    # Wall-clock timestamps (UTC ISO 8601 strings to make the row
    # round-trip through aiosqlite/JSON without timezone surprises).
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    # Failure diagnostics — only populated when status in {FAILED,
    # CANCELLED, INTERRUPTED}. TEXT not a separate table because at
    # ~10 jobs/day the JOIN cost is pure overhead and tracebacks are
    # always read alongside the row.
    error: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RunStore(Protocol):
    """Persistence contract for the API's run registry.

    Implementations may use any backend (SQLite, Postgres, in-memory)
    as long as the methods preserve the documented semantics. Tests can
    sub in a dict-backed fake.
    """

    async def initialize(self) -> None:
        """Open the underlying connection and run any startup invariants.

        Implementations MUST be idempotent — repeated calls do not
        double-initialise. Implementations MUST run
        :meth:`sweep_interrupted_on_startup` before returning so the
        first request the API serves cannot see a stale ``RUNNING`` row.
        """
        ...

    async def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        ...

    async def create(self, record: RunRecord) -> None:
        """Insert a new run row. Raises if ``run_id`` already exists."""
        ...

    async def get(self, run_id: str) -> RunRecord | None:
        """Return the row for ``run_id`` or ``None`` if not present."""
        ...

    async def list(self) -> list[RunRecord]:
        """Return every row, ordered by ``created_at`` ascending."""
        ...

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
    ) -> None:
        """Mutate the row's status (and optional error). Idempotent.

        Sets ``updated_at`` to the current UTC time. Raises
        :class:`KeyError` if the row does not exist — callers must
        ``create`` before they ``update``.
        """
        ...

    async def sweep_interrupted_on_startup(self) -> int:
        """Mark every ``RUNNING`` row as ``INTERRUPTED``. Returns count.

        Called from :meth:`initialize` so the API never returns
        ``RUNNING`` for a job whose process is gone.
        """
        ...


# ---------------------------------------------------------------------------
# SQLite implementation
# ---------------------------------------------------------------------------


# Schema lives next to the implementation rather than a migration file —
# the table is small, single-process, and we have no reason to support
# rolling schema upgrades for the v0.6-rc1 API surface.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    bundle_dir TEXT NOT NULL,
    lane       TEXT,
    backend    TEXT,
    seed       INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error      TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


class SQLiteRunStore:
    """aiosqlite-backed :class:`RunStore` (plan Task 31).

    One connection per store instance — opened in :meth:`initialize`,
    closed in :meth:`close`. Writers are serialised through
    ``self._write_lock`` so that ``BEGIN IMMEDIATE`` cannot race even
    when the FastAPI event loop juggles concurrent ``POST /runs``
    requests on the same store.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        # Serialises *write* transactions (BEGIN IMMEDIATE). Reads are
        # not gated — WAL allows concurrent reads while a writer holds
        # the RESERVED lock.
        import asyncio

        self._write_lock: asyncio.Lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Open the connection, set pragmas, create schema, sweep RUNNING."""
        if self._initialized:
            return
        # Ensure the parent directory exists — aiosqlite refuses to open
        # a database in a non-existent directory.
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        # Order matters: WAL must be set before any other write
        # transaction touches the file, otherwise the journal mode stays
        # at the default ``delete``. ``synchronous=NORMAL`` is the
        # recommended WAL companion (durable across crashes, accepts
        # rollback of the most recent commit on power loss).
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        # 5 s busy_timeout so any contended write retries instead of
        # raising OperationalError immediately. The SQLite docs note the
        # default of 0 makes every concurrent operation a coin-flip.
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        # Foreign keys aren't used today but enabling them now means a
        # later ``runs_artifacts`` join table can rely on cascade
        # deletes without a migration.
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()
        # Reconcile any RUNNING rows from a prior process *before* the
        # API starts serving. Done inside initialize() so callers cannot
        # forget — the Task 34 lifespan hook just calls initialize().
        await self.sweep_interrupted_on_startup()
        self._initialized = True

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        self._initialized = False

    async def create(self, record: RunRecord) -> None:
        conn = self._require_conn()
        async with self._write_lock:
            await conn.execute("BEGIN IMMEDIATE;")
            try:
                await conn.execute(
                    "INSERT INTO runs ("
                    "run_id, status, bundle_dir, lane, backend, seed, "
                    "created_at, updated_at, error"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record.run_id,
                        record.status.value,
                        record.bundle_dir,
                        record.lane,
                        record.backend,
                        record.seed,
                        record.created_at,
                        record.updated_at,
                        record.error,
                    ),
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def get(self, run_id: str) -> RunRecord | None:
        conn = self._require_conn()
        async with conn.execute(
            "SELECT run_id, status, bundle_dir, lane, backend, seed, "
            "created_at, updated_at, error FROM runs WHERE run_id = ?",
            (run_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    async def list(self) -> list[RunRecord]:
        conn = self._require_conn()
        async with conn.execute(
            "SELECT run_id, status, bundle_dir, lane, backend, seed, "
            "created_at, updated_at, error FROM runs ORDER BY created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_record(row) for row in rows]

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error: str | None = None,
    ) -> None:
        conn = self._require_conn()
        now = datetime.now(UTC).isoformat()
        async with self._write_lock:
            await conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = await conn.execute(
                    "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE run_id = ?",
                    (status.value, error, now, run_id),
                )
                if cur.rowcount == 0:
                    await conn.rollback()
                    msg = f"run_id {run_id!r} not found in run store"
                    raise KeyError(msg)
                await conn.commit()
            except Exception:
                # rollback() is a no-op if commit already happened; safe
                # to call unconditionally on the exception path.
                await conn.rollback()
                raise

    async def sweep_interrupted_on_startup(self) -> int:
        """Set status = INTERRUPTED for every row currently RUNNING.

        Returns the number of rows mutated. Called from
        :meth:`initialize`; safe to call again at any time, though the
        idempotent return is 0 once the table is reconciled.
        """
        conn = self._require_conn()
        now = datetime.now(UTC).isoformat()
        sweep_note = "process exited while RUNNING; status reset on app restart"
        async with self._write_lock:
            await conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = await conn.execute(
                    "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE status = ?",
                    (
                        RunStatus.INTERRUPTED.value,
                        sweep_note,
                        now,
                        RunStatus.RUNNING.value,
                    ),
                )
                count = cur.rowcount
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return count

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            msg = (
                "SQLiteRunStore is not initialized; call ``await store."
                "initialize()`` (typically inside the FastAPI lifespan)"
            )
            raise RuntimeError(msg)
        return self._conn


def _row_to_record(row: Sequence[object]) -> RunRecord:
    """Materialise a SQL row tuple back into a Pydantic :class:`RunRecord`.

    The column order is fixed by the SELECT in :meth:`SQLiteRunStore.get`
    / :meth:`SQLiteRunStore.list`. Keeping the projection in one place
    means a column rename touches one helper, not every reader.
    """
    raw_seed = row[5]
    seed = int(raw_seed) if isinstance(raw_seed, int) else None
    return RunRecord(
        run_id=str(row[0]),
        status=RunStatus(str(row[1])),
        bundle_dir=str(row[2]),
        lane=str(row[3]) if row[3] is not None else None,
        backend=str(row[4]) if row[4] is not None else None,
        seed=seed,
        created_at=str(row[6]),
        updated_at=str(row[7]),
        error=str(row[8]) if row[8] is not None else None,
    )


__all__ = [
    "RunRecord",
    "RunStatus",
    "RunStore",
    "SQLiteRunStore",
]
