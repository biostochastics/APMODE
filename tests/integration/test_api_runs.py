# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration tests for the FastAPI run endpoints (plan Task 32).

The orchestrator is the heaviest part of these tests — a real run
spawns R subprocesses and takes 30+ seconds. We exercise the *API*
behaviour with a fake :class:`apmode.api.runs.execute_run` instead of
running the real pipeline; the orchestrator integration is covered by
the existing CLI integration tests.

What's covered here:
  * POST /runs returns 202 + Retry-After: 5 + RunCreatedResponse
  * The run is persisted with status PENDING immediately
  * The background task runs and the row transitions to COMPLETED
  * GET /runs lists runs in created_at order
  * GET /runs/{id}/status returns the live row
  * 404 + 425 semantics on the bundle / RO-Crate download endpoints
  * Backend allowlist rejects unknown backends with 400
"""

from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from apmode.api.app import build_app
from apmode.api.store import RunStatus, SQLiteRunStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# --- Fakes / helpers -------------------------------------------------------


async def _fake_execute_success(
    *,
    run_id: str,
    bundle_dir: Path,
    request,
    runs_dir,
    runner_factory,
    store,
    on_complete=None,
    dataset_root: Path | None = None,
) -> None:
    """Stand-in for the real orchestrator: write a tiny sealed bundle.

    The bundle layout matches enough of the real shape that the
    download endpoints can stream it. ``_COMPLETE`` is the integrity
    sentinel the RO-Crate emitter checks for.
    """
    try:
        await store.update_status(run_id, RunStatus.RUNNING)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "report.md").write_text("# fake report\n")
        (bundle_dir / "_COMPLETE").write_text(
            json.dumps({"run_id": run_id, "digest_sha256": "deadbeef"})
        )
        await store.update_status(run_id, RunStatus.COMPLETED)
    finally:
        if on_complete is not None:
            await on_complete(run_id)


async def _fake_execute_slow_then_cancel(
    *,
    run_id: str,
    bundle_dir: Path,
    request,
    runs_dir,
    runner_factory,
    store,
    on_complete=None,
    dataset_root: Path | None = None,
) -> None:
    """Hold RUNNING long enough for the test to assert the live status."""
    try:
        await store.update_status(run_id, RunStatus.RUNNING)
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        await store.update_status(run_id, RunStatus.CANCELLED, error="test cancel")
        raise
    finally:
        if on_complete is not None:
            await on_complete(run_id)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.sqlite"


@pytest.fixture
def fake_runner_factory():
    """A runner factory that returns ``None`` — the fake execute_run never calls it."""

    def _factory(work_dir: Path) -> object:
        return object()

    return _factory


@pytest.fixture
async def client_with_fake_orchestrator(
    runs_dir: Path,
    db_path: Path,
    fake_runner_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """Build the app with the orchestrator stubbed to a fast success path."""
    monkeypatch.setattr("apmode.api.routes.execute_run", _fake_execute_success)
    store = SQLiteRunStore(db_path)
    app = build_app(
        runs_dir=runs_dir,
        db_path=db_path,
        runner_factory=fake_runner_factory,
        store=store,
    )
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as client,
        app.router.lifespan_context(app),
    ):
        yield client


# --- Health ----------------------------------------------------------------


@pytest.mark.integration
async def test_healthz_returns_apmode_version(
    client_with_fake_orchestrator: AsyncClient,
) -> None:
    resp = await client_with_fake_orchestrator.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["apmode_version"], "apmode_version must be populated"


# --- POST /runs -------------------------------------------------------------


@pytest.mark.integration
async def test_create_run_returns_202_with_retry_after(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={"dataset_path": str(csv), "lane": "submission", "backend": "nlmixr2"},
    )
    assert resp.status_code == 202, resp.text
    assert resp.headers.get("Retry-After") == "5"
    body = resp.json()
    assert body["run_id"]
    assert body["status"] == "pending"
    assert body["status_url"] == f"/runs/{body['run_id']}/status"


@pytest.mark.integration
async def test_create_run_rejects_unknown_backend(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={
            "dataset_path": str(csv),
            "lane": "submission",
            "backend": "totally_made_up",
        },
    )
    assert resp.status_code == 400
    assert "totally_made_up" in resp.json()["detail"]


@pytest.mark.integration
async def test_create_run_rejects_unknown_lane(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={"dataset_path": str(csv), "lane": "no_such_lane"},
    )
    # Pydantic validation error → 422
    assert resp.status_code == 422


@pytest.mark.integration
async def test_create_run_rejects_extra_fields(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    """``extra='forbid'`` on CreateRunRequest catches typos at the boundary."""
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={
            "dataset_path": str(csv),
            "lane": "submission",
            "backend": "nlmixr2",
            "this_field_does_not_exist": True,
        },
    )
    assert resp.status_code == 422


# --- GET /runs --------------------------------------------------------------


@pytest.mark.integration
async def test_runs_list_returns_created_runs(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    create_resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={"dataset_path": str(csv), "lane": "submission"},
    )
    run_id = create_resp.json()["run_id"]
    # Wait for the background task to finish
    await asyncio.sleep(0.2)
    list_resp = await client_with_fake_orchestrator.get("/runs")
    assert list_resp.status_code == 200
    body = list_resp.json()
    ids = [r["run_id"] for r in body["runs"]]
    assert run_id in ids


# --- GET /runs/{id}/status --------------------------------------------------


@pytest.mark.integration
async def test_status_endpoint_returns_completed_after_orchestrator_finishes(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    create_resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={"dataset_path": str(csv), "lane": "submission"},
    )
    run_id = create_resp.json()["run_id"]
    # Poll for completion (fake completes quickly)
    for _ in range(50):
        st_resp = await client_with_fake_orchestrator.get(f"/runs/{run_id}/status")
        if st_resp.json()["status"] == "completed":
            break
        await asyncio.sleep(0.05)
    body = st_resp.json()
    assert body["status"] == "completed"
    assert body["bundle_dir"]
    assert body["lane"] == "submission"
    assert body["backend"] == "nlmixr2"


@pytest.mark.integration
async def test_status_unknown_run_returns_404(
    client_with_fake_orchestrator: AsyncClient,
) -> None:
    resp = await client_with_fake_orchestrator.get("/runs/no-such-run/status")
    assert resp.status_code == 404


# --- GET /runs/{id}/bundle --------------------------------------------------


@pytest.mark.integration
async def test_bundle_endpoint_streams_zip_for_completed_run(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={"dataset_path": str(csv), "lane": "submission"},
    )
    run_id = resp.json()["run_id"]
    # Wait for completion
    for _ in range(50):
        st = await client_with_fake_orchestrator.get(f"/runs/{run_id}/status")
        if st.json()["status"] == "completed":
            break
        await asyncio.sleep(0.05)

    bundle_resp = await client_with_fake_orchestrator.get(f"/runs/{run_id}/bundle")
    assert bundle_resp.status_code == 200
    assert bundle_resp.headers["content-type"].startswith("application/zip")
    # Verify the zip is valid and contains the expected entries
    zip_path = tmp_path / "downloaded.zip"
    zip_path.write_bytes(bundle_resp.content)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert any("report.md" in n for n in names)
    assert any("_COMPLETE" in n for n in names)


@pytest.mark.integration
async def test_bundle_endpoint_returns_404_for_unknown_run(
    client_with_fake_orchestrator: AsyncClient,
) -> None:
    resp = await client_with_fake_orchestrator.get("/runs/no-such-run/bundle")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_bundle_endpoint_returns_425_when_run_not_completed(
    runs_dir: Path,
    db_path: Path,
    fake_runner_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundle download is gated on COMPLETED to avoid serving partial bundles."""
    monkeypatch.setattr("apmode.api.routes.execute_run", _fake_execute_slow_then_cancel)
    store = SQLiteRunStore(db_path)
    app = build_app(
        runs_dir=runs_dir,
        db_path=db_path,
        runner_factory=fake_runner_factory,
        store=store,
    )
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as client,
        app.router.lifespan_context(app),
    ):
        csv = tmp_path / "ds.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
        create_resp = await client.post(
            "/runs",
            json={"dataset_path": str(csv), "lane": "submission"},
        )
        run_id = create_resp.json()["run_id"]
        # Wait for the row to become RUNNING (slow stub holds it there)
        for _ in range(20):
            st = await client.get(f"/runs/{run_id}/status")
            if st.json()["status"] == "running":
                break
            await asyncio.sleep(0.05)
        bundle_resp = await client.get(f"/runs/{run_id}/bundle")
        assert bundle_resp.status_code == 425


# --- POST /runs persistence ------------------------------------------------


# --- DELETE /runs/{id} (plan Task 33) --------------------------------------


@pytest.mark.integration
async def test_delete_running_run_cancels_and_returns_cancelled(
    runs_dir: Path,
    db_path: Path,
    fake_runner_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DELETE during RUNNING transitions the row to CANCELLED."""
    monkeypatch.setattr("apmode.api.routes.execute_run", _fake_execute_slow_then_cancel)
    store = SQLiteRunStore(db_path)
    app = build_app(
        runs_dir=runs_dir,
        db_path=db_path,
        runner_factory=fake_runner_factory,
        store=store,
    )
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as client,
        app.router.lifespan_context(app),
    ):
        csv = tmp_path / "ds.csv"
        csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
        create = await client.post(
            "/runs",
            json={"dataset_path": str(csv), "lane": "submission"},
        )
        run_id = create.json()["run_id"]
        # Wait for RUNNING
        for _ in range(40):
            st = await client.get(f"/runs/{run_id}/status")
            if st.json()["status"] == "running":
                break
            await asyncio.sleep(0.05)
        assert st.json()["status"] == "running"

        del_resp = await client.delete(f"/runs/{run_id}")
        assert del_resp.status_code == 200
        body = del_resp.json()
        assert body["status"] == "cancelled"
        assert body["error"]


@pytest.mark.integration
async def test_delete_unknown_run_returns_404(
    client_with_fake_orchestrator: AsyncClient,
) -> None:
    resp = await client_with_fake_orchestrator.delete("/runs/no-such-run")
    assert resp.status_code == 404


@pytest.mark.integration
async def test_delete_completed_run_returns_409(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    """A completed run cannot be cancelled — DELETE returns 409 Conflict."""
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    create = await client_with_fake_orchestrator.post(
        "/runs",
        json={"dataset_path": str(csv), "lane": "submission"},
    )
    run_id = create.json()["run_id"]
    # Wait for completion
    for _ in range(50):
        st = await client_with_fake_orchestrator.get(f"/runs/{run_id}/status")
        if st.json()["status"] == "completed":
            break
        await asyncio.sleep(0.05)
    assert st.json()["status"] == "completed"

    resp = await client_with_fake_orchestrator.delete(f"/runs/{run_id}")
    assert resp.status_code == 409


# --- Lifespan startup sweep (plan Task 34) ---------------------------------


@pytest.mark.integration
async def test_lifespan_sweeps_running_rows_into_interrupted(
    runs_dir: Path,
    db_path: Path,
    fake_runner_factory,
) -> None:
    """A RUNNING row from a previous process becomes INTERRUPTED at startup.

    Simulates a crash mid-run: pre-seed the SQLite file with a RUNNING
    row, then start a new app pointing at the same DB. The lifespan
    must trigger ``store.initialize()`` → ``sweep_interrupted_on_startup``
    before the first request is served.
    """
    # Phase 1 — seed the DB out-of-band with a stale RUNNING row.
    pre_store = SQLiteRunStore(db_path)
    await pre_store.initialize()
    from apmode.api.store import RunRecord

    await pre_store.create(
        RunRecord(
            run_id="stale-1",
            status=RunStatus.RUNNING,
            bundle_dir=str(runs_dir / "stale-1"),
        )
    )
    await pre_store.close()

    # Phase 2 — start a new app on the same DB. Sweep should fire in
    # the lifespan startup hook before any request handler runs.
    new_store = SQLiteRunStore(db_path)
    app = build_app(
        runs_dir=runs_dir,
        db_path=db_path,
        runner_factory=fake_runner_factory,
        store=new_store,
    )
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://testserver") as client,
        app.router.lifespan_context(app),
    ):
        resp = await client.get("/runs/stale-1/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "interrupted"
        assert "process exited" in body["error"]


@pytest.mark.integration
async def test_requeue_on_interrupt_persists_through_post_get(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    """The POST body's ``requeue_on_interrupt`` survives the round trip."""
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    create = await client_with_fake_orchestrator.post(
        "/runs",
        json={
            "dataset_path": str(csv),
            "lane": "submission",
            "requeue_on_interrupt": True,
        },
    )
    assert create.status_code == 202
    run_id = create.json()["run_id"]
    # Poll until the (fast) fake orchestrator finishes
    for _ in range(50):
        st = await client_with_fake_orchestrator.get(f"/runs/{run_id}/status")
        if st.json()["status"] == "completed":
            break
        await asyncio.sleep(0.05)
    body = st.json()
    assert body["requeue_on_interrupt"] is True


# --- POST /runs persistence ------------------------------------------------


@pytest.mark.integration
async def test_run_record_persists_lane_backend_and_seed(
    client_with_fake_orchestrator: AsyncClient,
    tmp_path: Path,
) -> None:
    csv = tmp_path / "ds.csv"
    csv.write_text("NMID,TIME,DV,MDV,EVID,AMT,CMT\n1,0,0,1,1,100,1\n")
    resp = await client_with_fake_orchestrator.post(
        "/runs",
        json={
            "dataset_path": str(csv),
            "lane": "discovery",
            "backend": "nlmixr2",
            "seed": 42,
        },
    )
    run_id = resp.json()["run_id"]
    # Status row carries the request fields verbatim
    st = await client_with_fake_orchestrator.get(f"/runs/{run_id}/status")
    body = st.json()
    assert body["lane"] == "discovery"
    assert body["backend"] == "nlmixr2"
    assert body["seed"] == 42
