# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for deep inspection models, emitter, and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from apmode.bundle.models import (
    AgenticIterationEntry,
    SearchGraph,
    SearchGraphEdge,
    SearchGraphNode,
)
from apmode.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Model tests (Task 0 + Task 1)
# ---------------------------------------------------------------------------


class TestSearchGraphModels:
    def test_node_roundtrip(self) -> None:
        node = SearchGraphNode(
            candidate_id="cand_001",
            parent_id=None,
            backend="nlmixr2",
            converged=True,
            bic=342.1,
            aic=330.5,
            n_params=5,
            gate1_passed=True,
            gate2_passed=True,
            rank=1,
        )
        data = node.model_dump()
        assert data["candidate_id"] == "cand_001"
        assert data["converged"] is True
        rebuilt = SearchGraphNode.model_validate(data)
        assert rebuilt == node

    def test_edge_roundtrip(self) -> None:
        edge = SearchGraphEdge(
            parent_id="cand_001",
            child_id="cand_002",
            transform="swap_module(elimination, MichaelisMenten)",
        )
        data = edge.model_dump()
        rebuilt = SearchGraphEdge.model_validate(data)
        assert rebuilt == edge

    def test_graph_serialization(self) -> None:
        graph = SearchGraph(
            nodes=[
                SearchGraphNode(
                    candidate_id="cand_001",
                    backend="nlmixr2",
                    converged=True,
                    bic=342.1,
                    n_params=5,
                ),
                SearchGraphNode(
                    candidate_id="cand_002",
                    parent_id="cand_001",
                    backend="nlmixr2",
                    converged=True,
                    bic=318.7,
                    n_params=6,
                ),
            ],
            edges=[
                SearchGraphEdge(
                    parent_id="cand_001",
                    child_id="cand_002",
                    transform="swap_module(elimination, MichaelisMenten)",
                ),
            ],
        )
        json_str = graph.model_dump_json()
        rebuilt = SearchGraph.model_validate_json(json_str)
        assert len(rebuilt.nodes) == 2
        assert len(rebuilt.edges) == 1
        assert rebuilt.nodes[1].bic == 318.7

    def test_node_backend_literal(self) -> None:
        """Backend must be one of the allowed literals."""
        with pytest.raises(Exception):  # noqa: B017 — Pydantic validation
            SearchGraphNode(
                candidate_id="c",
                backend="nonexistent",  # type: ignore[arg-type]
            )

    def test_node_defaults(self) -> None:
        """Minimal node with just candidate_id."""
        node = SearchGraphNode(candidate_id="c001")
        assert node.backend == "nlmixr2"
        assert node.converged is False
        assert node.bic is None
        assert node.gate1_passed is None


class TestAgenticIterationEntry:
    def test_roundtrip(self) -> None:
        entry = AgenticIterationEntry(
            iteration=3,
            spec_before="cand_001",
            spec_after="cand_002",
            transforms_proposed=["swap_module(elimination, MichaelisMenten)"],
            transforms_rejected=[],
            reasoning="CWRES show systematic bias in elimination phase",
            converged=True,
            bic=318.7,
            error=None,
            validation_feedback=[],
        )
        data = entry.model_dump()
        rebuilt = AgenticIterationEntry.model_validate(data)
        assert rebuilt.iteration == 3
        assert rebuilt.spec_after == "cand_002"

    def test_iteration_must_be_positive(self) -> None:
        with pytest.raises(Exception):  # noqa: B017
            AgenticIterationEntry(iteration=0, spec_before="c")

    def test_defaults(self) -> None:
        entry = AgenticIterationEntry(iteration=1, spec_before="c001")
        assert entry.spec_after is None
        assert entry.transforms_proposed == []
        assert entry.transforms_rejected == []
        assert entry.validation_feedback == []
        assert entry.converged is False


# ---------------------------------------------------------------------------
# Emitter tests (Task 2)
# ---------------------------------------------------------------------------


class TestBundleEmitterSearchGraph:
    def test_write_search_graph(self, tmp_path: Path) -> None:
        from apmode.bundle.emitter import BundleEmitter

        emitter = BundleEmitter(tmp_path, run_id="test_graph")
        emitter.initialize()

        graph = SearchGraph(
            nodes=[
                SearchGraphNode(candidate_id="c001", backend="nlmixr2", converged=True, bic=100.0),
                SearchGraphNode(
                    candidate_id="c002",
                    parent_id="c001",
                    backend="nlmixr2",
                    converged=True,
                    bic=90.0,
                ),
            ],
            edges=[
                SearchGraphEdge(
                    parent_id="c001",
                    child_id="c002",
                    transform="swap_module(elimination, MichaelisMenten)",
                ),
            ],
        )
        path = emitter.write_search_graph(graph)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_search_graph_roundtrip_via_emitter(self, tmp_path: Path) -> None:
        from apmode.bundle.emitter import BundleEmitter

        emitter = BundleEmitter(tmp_path, run_id="test_rt")
        emitter.initialize()

        original = SearchGraph(
            nodes=[SearchGraphNode(candidate_id="c001", backend="jax_node", converged=False)],
        )
        path = emitter.write_search_graph(original)
        rebuilt = SearchGraph.model_validate_json(path.read_text())
        assert rebuilt.nodes[0].backend == "jax_node"
        assert rebuilt.nodes[0].converged is False


# ---------------------------------------------------------------------------
# SearchDAG public API tests
# ---------------------------------------------------------------------------


class TestSearchDAGPublicAPI:
    def test_iter_nodes(self) -> None:
        from apmode.dsl.ast_models import (
            IIV,
            DSLSpec,
            FirstOrder,
            LinearElim,
            OneCmt,
            Proportional,
        )
        from apmode.search.candidates import SearchDAG

        dag = SearchDAG()
        spec = DSLSpec(
            model_id="test_001",
            absorption=FirstOrder(ka=1.0),
            distribution=OneCmt(V=70.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        dag.add_root(spec)
        nodes = dag.iter_nodes()
        assert len(nodes) == 1
        assert nodes[0].candidate_id == "test_001"

    def test_to_edges_empty_for_roots(self) -> None:
        from apmode.dsl.ast_models import (
            IIV,
            DSLSpec,
            FirstOrder,
            LinearElim,
            OneCmt,
            Proportional,
        )
        from apmode.search.candidates import SearchDAG

        dag = SearchDAG()
        spec = DSLSpec(
            model_id="root_001",
            absorption=FirstOrder(ka=1.0),
            distribution=OneCmt(V=70.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
        )
        dag.add_root(spec)
        edges = dag.to_edges()
        assert edges == []


# ---------------------------------------------------------------------------
# CLI: apmode trace (Task 3)
# ---------------------------------------------------------------------------


def _make_agentic_bundle(tmp_path: Path) -> Path:
    """Create a minimal bundle with agentic trace artifacts."""
    bundle = tmp_path / "run_test"
    bundle.mkdir()
    trace_dir = bundle / "agentic_trace"
    trace_dir.mkdir()

    for i in range(1, 4):
        iter_id = f"iter_{i:03d}"
        (trace_dir / f"{iter_id}_input.json").write_text(
            json.dumps(
                {
                    "iteration_id": iter_id,
                    "run_id": "r1",
                    "candidate_id": f"cand_{i:03d}",
                    "prompt_hash": "abc",
                    "prompt_template": "v1",
                    "dsl_spec_json": "{}",
                    "diagnostics_summary": {},
                }
            )
        )
        (trace_dir / f"{iter_id}_output.json").write_text(
            json.dumps(
                {
                    "iteration_id": iter_id,
                    "raw_output": "swap_module(elimination, MM)",
                    "parsed_transforms": ["swap_module(elimination, MichaelisMenten)"],
                    "transforms_rejected": [],
                    "validation_passed": True,
                    "validation_errors": [],
                }
            )
        )
        (trace_dir / f"{iter_id}_meta.json").write_text(
            json.dumps(
                {
                    "iteration_id": iter_id,
                    "model_id": "claude-opus-4-6",
                    "model_version": "claude-opus-4-6-20260401",
                    "prompt_hash": "abc",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cost_usd": 0.05,
                    "temperature": 0.0,
                    "wall_time_seconds": 2.3,
                }
            )
        )

    with (trace_dir / "agentic_iterations.jsonl").open("w") as f:
        for i in range(1, 4):
            f.write(
                json.dumps(
                    {
                        "iteration": i,
                        "spec_before": f"cand_{i:03d}",
                        "spec_after": f"cand_{i + 1:03d}" if i < 3 else None,
                        "transforms_proposed": ["swap_module(elimination, MichaelisMenten)"],
                        "transforms_rejected": [],
                        "reasoning": "CWRES bias",
                        "converged": i == 3,
                        "bic": 400 - i * 30,
                        "error": None,
                        "validation_feedback": [],
                    }
                )
                + "\n"
            )

    return bundle


class TestTraceCommand:
    def test_trace_summary(self, tmp_path: Path) -> None:
        bundle = _make_agentic_bundle(tmp_path)
        result = runner.invoke(app, ["trace", str(bundle)])
        assert result.exit_code == 0
        assert "1" in result.output  # iteration number

    def test_trace_iteration_detail(self, tmp_path: Path) -> None:
        bundle = _make_agentic_bundle(tmp_path)
        result = runner.invoke(app, ["trace", str(bundle), "--iteration", "2"])
        assert result.exit_code == 0
        assert "swap_module" in result.output

    def test_trace_cost(self, tmp_path: Path) -> None:
        bundle = _make_agentic_bundle(tmp_path)
        result = runner.invoke(app, ["trace", str(bundle), "--cost"])
        assert result.exit_code == 0
        assert "0.15" in result.output  # 3 x $0.05

    def test_trace_no_agentic(self, tmp_path: Path) -> None:
        bundle = tmp_path / "run_empty"
        bundle.mkdir()
        result = runner.invoke(app, ["trace", str(bundle)])
        assert result.exit_code == 0
        assert "No agentic trace" in result.output


# ---------------------------------------------------------------------------
# CLI: apmode lineage (Task 4)
# ---------------------------------------------------------------------------


def _make_lineage_bundle(tmp_path: Path) -> Path:
    """Create a minimal bundle with lineage artifacts."""
    bundle = tmp_path / "run_lineage"
    bundle.mkdir()
    (bundle / "compiled_specs").mkdir()
    (bundle / "gate_decisions").mkdir()

    lineage = {
        "entries": [
            {"candidate_id": "root_001", "parent_id": None, "transform": None},
            {
                "candidate_id": "child_002",
                "parent_id": "root_001",
                "transform": "swap_module(elimination, MichaelisMenten)",
            },
            {
                "candidate_id": "child_003",
                "parent_id": "child_002",
                "transform": "add_covariate_link(CL, WT, power)",
            },
        ]
    }
    (bundle / "candidate_lineage.json").write_text(json.dumps(lineage))

    # Gate decisions — use structure matching what cli.py actually reads
    # (simplified dict format used by existing _show_gate_details parser)
    for cid in ["root_001", "child_002", "child_003"]:
        (bundle / "gate_decisions" / f"gate1_{cid}.json").write_text(
            json.dumps({"passed": True, "candidate_id": cid, "checks": {}})
        )
        (bundle / "gate_decisions" / f"gate2_{cid}.json").write_text(
            json.dumps({"passed": cid != "root_001", "candidate_id": cid, "checks": {}})
        )

    return bundle


class TestLineageCommand:
    def test_lineage_chain(self, tmp_path: Path) -> None:
        bundle = _make_lineage_bundle(tmp_path)
        result = runner.invoke(app, ["lineage", str(bundle), "child_003"])
        assert result.exit_code == 0
        assert "root_001" in result.output
        assert "child_002" in result.output
        assert "child_003" in result.output
        assert "swap_module" in result.output
        assert "add_covariate_link" in result.output

    def test_lineage_not_found(self, tmp_path: Path) -> None:
        bundle = _make_lineage_bundle(tmp_path)
        result = runner.invoke(app, ["lineage", str(bundle), "nonexistent"])
        assert result.exit_code == 1

    def test_lineage_root_candidate(self, tmp_path: Path) -> None:
        """Root candidate has no parent — should show just itself."""
        bundle = _make_lineage_bundle(tmp_path)
        result = runner.invoke(app, ["lineage", str(bundle), "root_001"])
        assert result.exit_code == 0
        assert "root_001" in result.output


# ---------------------------------------------------------------------------
# CLI: apmode graph (Task 5)
# ---------------------------------------------------------------------------


def _make_graph_bundle(tmp_path: Path) -> Path:
    """Create a minimal bundle with search graph."""
    bundle = tmp_path / "run_graph"
    bundle.mkdir()

    graph = {
        "nodes": [
            {
                "candidate_id": "root_001",
                "parent_id": None,
                "backend": "nlmixr2",
                "converged": True,
                "bic": 400.0,
                "n_params": 4,
                "gate1_passed": True,
                "gate2_passed": True,
            },
            {
                "candidate_id": "child_002",
                "parent_id": "root_001",
                "backend": "nlmixr2",
                "converged": True,
                "bic": 350.0,
                "n_params": 5,
                "gate1_passed": True,
                "gate2_passed": True,
                "rank": 1,
            },
            {
                "candidate_id": "child_003",
                "parent_id": "root_001",
                "backend": "nlmixr2",
                "converged": False,
                "bic": None,
                "n_params": 5,
            },
        ],
        "edges": [
            {
                "parent_id": "root_001",
                "child_id": "child_002",
                "transform": "swap_module(elimination, MichaelisMenten)",
            },
            {
                "parent_id": "root_001",
                "child_id": "child_003",
                "transform": "warm_start_combined",
            },
        ],
    }
    (bundle / "search_graph.json").write_text(json.dumps(graph))
    return bundle


class TestGraphCommand:
    def test_graph_tree(self, tmp_path: Path) -> None:
        bundle = _make_graph_bundle(tmp_path)
        result = runner.invoke(app, ["graph", str(bundle)])
        assert result.exit_code == 0
        assert "root_001" in result.output
        assert "child_002" in result.output

    def test_graph_dot_export(self, tmp_path: Path) -> None:
        bundle = _make_graph_bundle(tmp_path)
        dot_path = tmp_path / "out.dot"
        result = runner.invoke(
            app, ["graph", str(bundle), "--format", "dot", "--output", str(dot_path)]
        )
        assert result.exit_code == 0
        assert dot_path.exists()
        content = dot_path.read_text()
        assert "digraph" in content
        assert "root_001" in content

    def test_graph_json_export(self, tmp_path: Path) -> None:
        bundle = _make_graph_bundle(tmp_path)
        result = runner.invoke(app, ["graph", str(bundle), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data["nodes"]) == 3

    def test_graph_converged_filter(self, tmp_path: Path) -> None:
        bundle = _make_graph_bundle(tmp_path)
        result = runner.invoke(app, ["graph", str(bundle), "--converged"])
        assert result.exit_code == 0
        assert "child_003" not in result.output  # not converged

    def test_graph_no_artifact(self, tmp_path: Path) -> None:
        bundle = tmp_path / "run_empty"
        bundle.mkdir()
        result = runner.invoke(app, ["graph", str(bundle)])
        assert result.exit_code == 0
        assert "No search graph" in result.output

    def test_graph_mermaid_export(self, tmp_path: Path) -> None:
        bundle = _make_graph_bundle(tmp_path)
        result = runner.invoke(app, ["graph", str(bundle), "--format", "mermaid"])
        assert result.exit_code == 0
        assert "flowchart" in result.output or "graph" in result.output
