# Deep Inspection CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three new CLI commands (`apmode trace`, `apmode lineage`, `apmode graph`) for deep inspection of agentic iteration traces, per-candidate DSL transform history, and the full search DAG.

**Architecture:** Three new top-level Typer commands reading existing bundle artifacts (agentic_trace/, candidate_lineage.json, compiled_specs/, gate_decisions/) plus one new artifact (search_graph.json). No changes to existing `inspect` or `log` behavior — only hints added. All output via Rich tables/trees with optional DOT/Mermaid/JSON export.

**Tech Stack:** Typer, Rich (Table, Tree, Panel), Pydantic v2, structlog. No new dependencies.

---

### Task 0: Add SearchGraph Pydantic Models

**Files:**
- Modify: `src/apmode/bundle/models.py:319-333` (after CandidateLineage)
- Test: `tests/unit/test_deep_inspect.py` (create)

**Step 1: Write the failing test**

```python
# tests/unit/test_deep_inspect.py
"""Tests for deep inspection models and CLI commands."""

from apmode.bundle.models import SearchGraphEdge, SearchGraphNode, SearchGraph


def test_search_graph_node_roundtrip() -> None:
    node = SearchGraphNode(
        candidate_id="cand_001",
        parent_id=None,
        transform=None,
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


def test_search_graph_edge_roundtrip() -> None:
    edge = SearchGraphEdge(
        parent_id="cand_001",
        child_id="cand_002",
        transform="swap_module(elimination, MichaelisMenten)",
    )
    data = edge.model_dump()
    rebuilt = SearchGraphEdge.model_validate(data)
    assert rebuilt == edge


def test_search_graph_serialization() -> None:
    graph = SearchGraph(
        nodes=[
            SearchGraphNode(
                candidate_id="cand_001",
                parent_id=None,
                transform=None,
                backend="nlmixr2",
                converged=True,
                bic=342.1,
                n_params=5,
            ),
            SearchGraphNode(
                candidate_id="cand_002",
                parent_id="cand_001",
                transform="swap_module(elimination, MichaelisMenten)",
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_deep_inspect.py -v`
Expected: FAIL with ImportError (SearchGraphNode not defined)

**Step 3: Write minimal implementation**

Add to `src/apmode/bundle/models.py` after `CandidateLineage` (line ~333):

```python
# --- Search Graph (Deep Inspection) ---


class SearchGraphNode(BaseModel):
    """A node in the search graph (enriched candidate with gate status)."""

    candidate_id: str
    parent_id: str | None = None
    transform: str | None = None
    backend: str = "nlmixr2"
    converged: bool = False
    bic: float | None = None
    aic: float | None = None
    n_params: int = 0
    gate1_passed: bool | None = None
    gate2_passed: bool | None = None
    gate2_5_passed: bool | None = None
    rank: int | None = None


class SearchGraphEdge(BaseModel):
    """An edge in the search graph (parent -> child via transform)."""

    parent_id: str
    child_id: str
    transform: str


class SearchGraph(BaseModel):
    """search_graph.json — enriched DAG of the full search space."""

    nodes: list[SearchGraphNode]
    edges: list[SearchGraphEdge] = Field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_deep_inspect.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_deep_inspect.py src/apmode/bundle/models.py
git commit -m "feat: add SearchGraph Pydantic models for deep inspection"
```

---

### Task 1: Add AgenticIterationEntry Model

**Files:**
- Modify: `src/apmode/bundle/models.py` (after AgenticTraceMeta)
- Test: `tests/unit/test_deep_inspect.py` (append)

**Step 1: Write the failing test**

```python
def test_agentic_iteration_entry_roundtrip() -> None:
    from apmode.bundle.models import AgenticIterationEntry

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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_deep_inspect.py::test_agentic_iteration_entry_roundtrip -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `src/apmode/bundle/models.py` after `AgenticTraceMeta` (line ~508):

```python
class AgenticIterationEntry(BaseModel):
    """One line in agentic_iterations.jsonl — typed audit trail."""

    iteration: int = Field(ge=1)
    spec_before: str
    spec_after: str | None = None
    transforms_proposed: list[str] = Field(default_factory=list)
    transforms_rejected: list[str] = Field(default_factory=list)
    reasoning: str = ""
    converged: bool = False
    bic: float | None = None
    error: str | None = None
    validation_feedback: list[str] = Field(default_factory=list)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_deep_inspect.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/apmode/bundle/models.py tests/unit/test_deep_inspect.py
git commit -m "feat: add AgenticIterationEntry typed model"
```

---

### Task 2: Emit search_graph.json from BundleEmitter

**Files:**
- Modify: `src/apmode/bundle/emitter.py:196-200` (after write_candidate_lineage)
- Modify: `src/apmode/orchestrator/__init__.py` (after gates, before ranking)
- Test: `tests/unit/test_bundle_emitter.py` (append)

**Step 1: Write the failing test**

```python
def test_write_search_graph(tmp_path: Path) -> None:
    from apmode.bundle.emitter import BundleEmitter
    from apmode.bundle.models import SearchGraph, SearchGraphEdge, SearchGraphNode

    emitter = BundleEmitter(tmp_path, run_id="test_graph")
    emitter.initialize()

    graph = SearchGraph(
        nodes=[
            SearchGraphNode(candidate_id="c001", backend="nlmixr2", converged=True, bic=100.0),
            SearchGraphNode(
                candidate_id="c002",
                parent_id="c001",
                transform="swap_module(elimination, MichaelisMenten)",
                backend="nlmixr2",
                converged=True,
                bic=90.0,
            ),
        ],
        edges=[
            SearchGraphEdge(parent_id="c001", child_id="c002", transform="swap_module(elimination, MichaelisMenten)"),
        ],
    )
    path = emitter.write_search_graph(graph)
    assert path.exists()
    data = json.loads(path.read_text())
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_bundle_emitter.py::test_write_search_graph -v`
Expected: FAIL (AttributeError: write_search_graph)

**Step 3: Write minimal implementation**

Add to `src/apmode/bundle/emitter.py`:

1. Add `SearchGraph` to imports from `apmode.bundle.models`
2. Add method after `write_candidate_lineage()`:

```python
def write_search_graph(self, graph: SearchGraph) -> Path:
    """Write search_graph.json (enriched DAG for deep inspection)."""
    path = self.run_dir / "search_graph.json"
    path.write_text(graph.model_dump_json(indent=2))
    return path
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_bundle_emitter.py::test_write_search_graph -v`
Expected: PASS

**Step 5: Wire into orchestrator**

In `src/apmode/orchestrator/__init__.py`, after all gates run and before ranking:
- Build `SearchGraph` from `SearchOutcome.dag` nodes + gate decisions
- Call `emitter.write_search_graph(graph)`

This requires reading the orchestrator to find the exact insertion point. The orchestrator iterates `search_outcome.results`, runs gates, then writes ranking. Between gates and ranking, build the graph by:
1. Converting `search_outcome.dag._nodes` to `SearchGraphNode` list
2. Enriching each node with gate1_passed/gate2_passed from gate decisions already computed
3. Building edges from parent→child relationships
4. Writing via emitter

**Step 6: Commit**

```bash
git add src/apmode/bundle/emitter.py src/apmode/orchestrator/__init__.py tests/unit/test_bundle_emitter.py
git commit -m "feat: emit search_graph.json from orchestrator after gates"
```

---

### Task 3: Implement `apmode trace` CLI Command

**Files:**
- Modify: `src/apmode/cli.py` (add new command after `log_cmd`)
- Test: `tests/unit/test_deep_inspect.py` (append CLI tests)

**Step 1: Write failing test**

```python
import json
from pathlib import Path
from typer.testing import CliRunner
from apmode.cli import app

runner = CliRunner()


def _make_agentic_bundle(tmp_path: Path) -> Path:
    """Create a minimal bundle with agentic trace artifacts."""
    bundle = tmp_path / "run_test"
    bundle.mkdir()
    trace_dir = bundle / "agentic_trace"
    trace_dir.mkdir()

    # Write 3 iterations
    for i in range(1, 4):
        iter_id = f"iter_{i:03d}"
        (trace_dir / f"{iter_id}_input.json").write_text(json.dumps({
            "iteration_id": iter_id, "run_id": "r1", "candidate_id": f"cand_{i:03d}",
            "prompt_hash": "abc", "prompt_template": "v1",
            "dsl_spec_json": "{}", "diagnostics_summary": {},
        }))
        (trace_dir / f"{iter_id}_output.json").write_text(json.dumps({
            "iteration_id": iter_id, "raw_output": "swap_module(elimination, MM)",
            "parsed_transforms": ["swap_module(elimination, MichaelisMenten)"],
            "validation_passed": True, "validation_errors": [],
        }))
        (trace_dir / f"{iter_id}_meta.json").write_text(json.dumps({
            "iteration_id": iter_id, "model_id": "claude-opus-4-6",
            "model_version": "claude-opus-4-6-20260401",
            "prompt_hash": "abc", "input_tokens": 1000, "output_tokens": 200,
            "cost_usd": 0.05, "temperature": 0.0, "wall_time_seconds": 2.3,
        }))

    # Write iterations JSONL
    with (trace_dir / "agentic_iterations.jsonl").open("w") as f:
        for i in range(1, 4):
            f.write(json.dumps({
                "iteration": i, "spec_before": f"cand_{i:03d}",
                "spec_after": f"cand_{i+1:03d}" if i < 3 else None,
                "transforms_proposed": ["swap_module(elimination, MichaelisMenten)"],
                "reasoning": "CWRES bias", "converged": i == 3, "bic": 400 - i * 30,
                "error": None,
            }) + "\n")

    return bundle


def test_trace_summary(tmp_path: Path) -> None:
    bundle = _make_agentic_bundle(tmp_path)
    result = runner.invoke(app, ["trace", str(bundle)])
    assert result.exit_code == 0
    assert "iter_001" in result.output or "1" in result.output


def test_trace_iteration_detail(tmp_path: Path) -> None:
    bundle = _make_agentic_bundle(tmp_path)
    result = runner.invoke(app, ["trace", str(bundle), "--iteration", "2"])
    assert result.exit_code == 0
    assert "swap_module" in result.output


def test_trace_cost(tmp_path: Path) -> None:
    bundle = _make_agentic_bundle(tmp_path)
    result = runner.invoke(app, ["trace", str(bundle), "--cost"])
    assert result.exit_code == 0
    # 3 iterations x $0.05 = $0.15
    assert "0.15" in result.output


def test_trace_no_agentic(tmp_path: Path) -> None:
    bundle = tmp_path / "run_empty"
    bundle.mkdir()
    result = runner.invoke(app, ["trace", str(bundle)])
    assert result.exit_code == 0
    assert "No agentic trace" in result.output
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_deep_inspect.py::test_trace_summary -v`
Expected: FAIL (no "trace" command)

**Step 3: Write implementation**

Add `trace` command to `src/apmode/cli.py`:

```python
@app.command()
def trace(
    bundle_dir: Annotated[Path, typer.Argument(help="Path to a run bundle directory.")],
    iteration: Annotated[int | None, typer.Option("--iteration", "-i", help="Show detail for specific iteration.")] = None,
    cost: Annotated[bool, typer.Option("--cost", help="Show token/cost aggregation.")] = False,
    output_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False,
) -> None:
    """Inspect agentic LLM iteration traces.

    Shows the propose-validate-compile-fit loop history from the agentic
    backend (Phase 3, PRD \u00a74.2.6).

    \b
    Examples:
      apmode trace ./runs/run_abc123                  # iteration summary table
      apmode trace ./runs/run_abc123 --iteration 5    # detail for iteration 5
      apmode trace ./runs/run_abc123 --cost           # token/cost rollup
    """
    # ... implementation reads agentic_trace/ directory
    # Summary: Rich Table with columns Iter, Candidate, Transforms, Converged, BIC, Error
    # Detail: Rich Panel with input/output/meta for specific iteration
    # Cost: Aggregate meta files for tokens/cost
```

Key implementation details:
- Read `agentic_trace/agentic_iterations.jsonl` for summary
- Read `{iter_id}_input.json`, `_output.json`, `_meta.json` for detail
- Aggregate `_meta.json` files for cost
- Graceful "No agentic trace found" when directory missing
- BIC convergence mini-chart using `_mini_bar()` pattern from existing inspect

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_deep_inspect.py -k trace -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/apmode/cli.py tests/unit/test_deep_inspect.py
git commit -m "feat: add apmode trace command for agentic iteration inspection"
```

---

### Task 4: Implement `apmode lineage` CLI Command

**Files:**
- Modify: `src/apmode/cli.py` (add new command)
- Test: `tests/unit/test_deep_inspect.py` (append)

**Step 1: Write failing test**

```python
def _make_lineage_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "run_lineage"
    bundle.mkdir()
    (bundle / "compiled_specs").mkdir()
    (bundle / "gate_decisions").mkdir()

    # candidate_lineage.json with 3-step chain
    lineage = {
        "entries": [
            {"candidate_id": "root_001", "parent_id": None, "transform": None},
            {"candidate_id": "child_002", "parent_id": "root_001", "transform": "swap_module(elimination, MichaelisMenten)"},
            {"candidate_id": "child_003", "parent_id": "child_002", "transform": "add_covariate_link(CL, WT, power)"},
        ]
    }
    (bundle / "candidate_lineage.json").write_text(json.dumps(lineage))

    # Compiled specs (minimal)
    for cid in ["root_001", "child_002", "child_003"]:
        (bundle / "compiled_specs" / f"{cid}.json").write_text(json.dumps({
            "model_id": cid,
            "absorption": {"type": "FirstOrder", "ka": 1.0},
            "distribution": {"type": "OneCmt", "V": 70.0},
            "elimination": {"type": "LinearElim", "CL": 5.0},
        }))

    # Gate decisions
    for cid in ["root_001", "child_002", "child_003"]:
        (bundle / "gate_decisions" / f"gate1_{cid}.json").write_text(
            json.dumps({"passed": True, "candidate_id": cid})
        )
        (bundle / "gate_decisions" / f"gate2_{cid}.json").write_text(
            json.dumps({"passed": cid != "root_001", "candidate_id": cid})
        )

    return bundle


def test_lineage_chain(tmp_path: Path) -> None:
    bundle = _make_lineage_bundle(tmp_path)
    result = runner.invoke(app, ["lineage", str(bundle), "child_003"])
    assert result.exit_code == 0
    assert "root_001" in result.output
    assert "child_002" in result.output
    assert "child_003" in result.output
    assert "swap_module" in result.output
    assert "add_covariate_link" in result.output


def test_lineage_not_found(tmp_path: Path) -> None:
    bundle = _make_lineage_bundle(tmp_path)
    result = runner.invoke(app, ["lineage", str(bundle), "nonexistent"])
    assert result.exit_code == 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_deep_inspect.py::test_lineage_chain -v`
Expected: FAIL (no "lineage" command)

**Step 3: Write implementation**

```python
@app.command()
def lineage(
    bundle_dir: Annotated[Path, typer.Argument(help="Path to a run bundle directory.")],
    candidate_id: Annotated[str, typer.Argument(help="Target candidate ID.")],
    spec: Annotated[bool, typer.Option("--spec", help="Show DSL spec at each step.")] = False,
    diff: Annotated[bool, typer.Option("--diff", help="Show spec diffs between steps.")] = False,
    gate: Annotated[bool, typer.Option("--gate", help="Show gate outcomes per step.")] = True,
    output_json: Annotated[bool, typer.Option("--json", help="Machine-readable JSON output.")] = False,
) -> None:
    """Trace the transform lineage of a specific candidate.

    Shows the chain of DSL transforms from root to target candidate,
    with gate status at each step.

    \b
    Examples:
      apmode lineage ./runs/run_abc123 cand_a3f8              # transform chain
      apmode lineage ./runs/run_abc123 cand_a3f8 --spec       # with DSL snapshots
      apmode lineage ./runs/run_abc123 cand_a3f8 --diff       # spec diffs
    """
```

Key implementation:
- Read `candidate_lineage.json`, build parent map
- Back-trace from target to root
- Reverse to get root→target chain
- For each step: load compiled spec (optional), load gate decisions
- Render as Rich Tree with vertical timeline
- Gate status colored: green PASS, red FAIL

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_deep_inspect.py -k lineage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/apmode/cli.py tests/unit/test_deep_inspect.py
git commit -m "feat: add apmode lineage command for candidate transform history"
```

---

### Task 5: Implement `apmode graph` CLI Command

**Files:**
- Modify: `src/apmode/cli.py` (add new command)
- Test: `tests/unit/test_deep_inspect.py` (append)

**Step 1: Write failing test**

```python
def _make_graph_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "run_graph"
    bundle.mkdir()

    graph = {
        "nodes": [
            {"candidate_id": "root_001", "parent_id": None, "backend": "nlmixr2", "converged": True, "bic": 400.0, "n_params": 4, "gate1_passed": True, "gate2_passed": True},
            {"candidate_id": "child_002", "parent_id": "root_001", "transform": "swap_module(elimination, MichaelisMenten)", "backend": "nlmixr2", "converged": True, "bic": 350.0, "n_params": 5, "gate1_passed": True, "gate2_passed": True, "rank": 1},
            {"candidate_id": "child_003", "parent_id": "root_001", "transform": "warm_start_combined", "backend": "nlmixr2", "converged": False, "bic": None, "n_params": 5},
        ],
        "edges": [
            {"parent_id": "root_001", "child_id": "child_002", "transform": "swap_module(elimination, MichaelisMenten)"},
            {"parent_id": "root_001", "child_id": "child_003", "transform": "warm_start_combined"},
        ],
    }
    (bundle / "search_graph.json").write_text(json.dumps(graph))
    return bundle


def test_graph_tree(tmp_path: Path) -> None:
    bundle = _make_graph_bundle(tmp_path)
    result = runner.invoke(app, ["graph", str(bundle)])
    assert result.exit_code == 0
    assert "root_001" in result.output
    assert "child_002" in result.output


def test_graph_dot_export(tmp_path: Path) -> None:
    bundle = _make_graph_bundle(tmp_path)
    dot_path = tmp_path / "out.dot"
    result = runner.invoke(app, ["graph", str(bundle), "--format", "dot", "--output", str(dot_path)])
    assert result.exit_code == 0
    assert dot_path.exists()
    content = dot_path.read_text()
    assert "digraph" in content
    assert "root_001" in content


def test_graph_json_export(tmp_path: Path) -> None:
    bundle = _make_graph_bundle(tmp_path)
    result = runner.invoke(app, ["graph", str(bundle), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["nodes"]) == 3


def test_graph_converged_filter(tmp_path: Path) -> None:
    bundle = _make_graph_bundle(tmp_path)
    result = runner.invoke(app, ["graph", str(bundle), "--converged"])
    assert result.exit_code == 0
    assert "child_003" not in result.output  # not converged


def test_graph_no_artifact(tmp_path: Path) -> None:
    bundle = tmp_path / "run_empty"
    bundle.mkdir()
    result = runner.invoke(app, ["graph", str(bundle)])
    assert result.exit_code == 0
    assert "No search graph" in result.output
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_deep_inspect.py::test_graph_tree -v`
Expected: FAIL (no "graph" command)

**Step 3: Write implementation**

```python
@app.command()
def graph(
    bundle_dir: Annotated[Path, typer.Argument(help="Path to a run bundle directory.")],
    format: Annotated[str, typer.Option("--format", "-f", help="Output format: tree, dot, mermaid, json.")] = "tree",
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write output to file.")] = None,
    backend: Annotated[str | None, typer.Option("--backend", help="Filter by backend.")] = None,
    converged: Annotated[bool, typer.Option("--converged", help="Show only converged candidates.")] = False,
    gate: Annotated[str | None, typer.Option("--gate", help="Filter: gate1-passed, gate2-passed.")] = None,
    depth: Annotated[int, typer.Option("--depth", help="Max tree depth.")] = 10,
    ancestor_of: Annotated[str | None, typer.Option("--ancestor-of", help="Show only ancestors of candidate.")] = None,
    descendant_of: Annotated[str | None, typer.Option("--descendant-of", help="Show only descendants of candidate.")] = None,
) -> None:
    """Visualize the full search DAG.

    Shows the tree/graph of all candidate models explored during the run,
    with convergence, gate status, and BIC on each node.

    \b
    Examples:
      apmode graph ./runs/run_abc123                           # tree view
      apmode graph ./runs/run_abc123 --format dot -o dag.dot   # Graphviz
      apmode graph ./runs/run_abc123 --converged --backend nlmixr2
      apmode graph ./runs/run_abc123 --ancestor-of cand_a3f8
    """
```

Key implementation:
- Read `search_graph.json` (fallback: reconstruct from `candidate_lineage.json` + `search_trajectory.jsonl`)
- Filter nodes by backend/converged/gate/ancestor/descendant
- **tree format**: Build Rich Tree, nodes colored by gate status (green/yellow/red/dim), ranked nodes marked with star
- **dot format**: Emit Graphviz DOT (`digraph { ... }`) with node shapes/colors
- **mermaid format**: Emit Mermaid flowchart syntax
- **json format**: Dump filtered graph to stdout
- `--output` writes to file instead of stdout (for dot/mermaid)

Node label format: `candidate_id (BIC=xxx) [PASS]` or `[FAIL]` or `[NC]`

**Step 4: Run tests**

Run: `uv run pytest tests/unit/test_deep_inspect.py -k graph -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/apmode/cli.py tests/unit/test_deep_inspect.py
git commit -m "feat: add apmode graph command for search DAG visualization"
```

---

### Task 6: Update inspect/log with Hints + Final Tests

**Files:**
- Modify: `src/apmode/cli.py:599-783` (inspect command)
- Test: `tests/unit/test_deep_inspect.py` (append)

**Step 1: Add hints to `inspect`**

After `sections_shown` counter in the inspect command, add:

```python
# --- Deep inspection hints ---
trace_dir = bundle_dir / "agentic_trace"
if trace_dir.is_dir() and list(trace_dir.glob("iter_*_input.json")):
    console.print("  [dim]Agentic trace available:[/] use [bold]apmode trace[/] for iteration details")
    sections_shown += 1

graph_path = bundle_dir / "search_graph.json"
if graph_path.exists():
    console.print("  [dim]Search graph available:[/] use [bold]apmode graph[/] for DAG visualization")
    sections_shown += 1

lineage_path = bundle_dir / "candidate_lineage.json"
if lineage_path.exists():
    console.print("  [dim]Candidate lineage available:[/] use [bold]apmode lineage[/] <candidate_id>")
    sections_shown += 1
```

**Step 2: Update CLI docstring**

Update the module docstring at the top of `cli.py` to include the three new commands.

**Step 3: Run full test suite**

```bash
uv run pytest tests/unit/test_deep_inspect.py -v
uv run mypy src/apmode/ --strict
uv run ruff check src/apmode/ tests/
uv run ruff format src/apmode/ tests/
```

**Step 4: Commit**

```bash
git add src/apmode/cli.py tests/unit/test_deep_inspect.py
git commit -m "feat: add deep inspection hints to apmode inspect + full test coverage"
```

---

## Dependency Graph

```
Task 0 (SearchGraph models) ──┬──> Task 2 (emit search_graph.json) ──┬──> Task 5 (graph cmd) ──┐
Task 1 (AgenticIterationEntry) ┘                                     │                         ├──> Task 6 (hints + tests)
                                                                     └──> Task 3 (trace cmd) ──┘
                                Task 1 ──────────────────────────────────> Task 4 (lineage cmd) ┘
```

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Large agentic traces (25 iter x raw LLM output) | Medium | Default truncation to 500 chars; `--iteration N` for full detail |
| DAG > 100 nodes unreadable in terminal | Medium | `--depth` default=10, `--converged` filter, `--format dot` for external tools |
| Missing search_graph.json in pre-Phase-3 bundles | Low | Graceful fallback: reconstruct from candidate_lineage.json + search_trajectory.jsonl |
| agentic_trace may not exist (non-agentic runs) | Low | `trace` command says "No agentic trace found" and exits 0 |
| Agentic traces currently written outside bundle | High | Already fixed: BundleEmitter._agentic_trace_dir() writes inside run_dir |
