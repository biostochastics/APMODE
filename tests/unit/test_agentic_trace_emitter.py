# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for agentic trace bundle emission (PRD §4.2.6)."""

import json
from pathlib import Path

from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import (
    AgenticTraceInput,
    AgenticTraceMeta,
    AgenticTraceOutput,
    RunLineage,
)


def test_write_agentic_trace_input(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    inp = AgenticTraceInput(
        iteration_id="iter_001",
        run_id="run_abc",
        candidate_id="cand_001",
        prompt_hash="abc123def456" * 5 + "abcd",
        prompt_template="system_v1",
        dsl_spec_json='{"model_id":"test"}',
        diagnostics_summary={"cwres_mean": 0.1},
    )
    path = emitter.write_agentic_trace_input(inp)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["iteration_id"] == "iter_001"


def test_write_agentic_trace_output(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    out = AgenticTraceOutput(
        iteration_id="iter_001",
        raw_output='{"transforms": []}',
        parsed_transforms=["swap_module(elimination, MichaelisMenten)"],
        validation_passed=True,
    )
    path = emitter.write_agentic_trace_output(out)
    assert path.exists()


def test_write_agentic_trace_meta(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    meta = AgenticTraceMeta(
        iteration_id="iter_001",
        model_id="claude-sonnet-4-20250514",
        model_version="claude-sonnet-4-20250514",
        prompt_hash="abc123def456" * 5 + "abcd",
        input_tokens=500,
        output_tokens=200,
        cost_usd=0.005,
        temperature=0.0,
        wall_time_seconds=2.3,
    )
    path = emitter.write_agentic_trace_meta(meta)
    assert path.exists()


def test_write_run_lineage(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    lineage = RunLineage(
        current_run_id="run_002",
        parent_run_ids=["run_001"],
        lineage_type="continuation",
    )
    path = emitter.write_run_lineage(lineage)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["parent_run_ids"] == ["run_001"]
