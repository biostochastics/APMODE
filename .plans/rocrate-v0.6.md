# RO-Crate v0.6 Implementation Plan

**Scope** — net-new module `src/apmode/bundle/rocrate/` that projects a sealed
APMODE bundle into a Workflow Run RO-Crate — **Provenance Run Crate v0.5**
(`https://w3id.org/ro/wfrun/provenance/0.5`). Read-only. Supports
**directory form and zip form**. Authoritative design:
`_research/ROCRATE_INTEGRATION_PLAN.md` §§A–H (already accepted).

This plan focuses on the **v0.6 Submission-lane read-only export** slice with
the additional user-requested scope: *implement all entity projectors
available from current bundle schema* (data, policy, backend, gate, lineage,
credibility, bayesian, agentic, pccp — the PCCP projectors degrade
gracefully when the regulatory files are absent, so they are safe to ship
now).

---

## Acceptance gate (blocking)

1. `apmode bundle rocrate export <bundle_dir> --out <path>` produces a
   directory-form crate that **passes `roc-validator` at REQUIRED** against
   profile `provenance-run-crate-0.5` (which inherits
   `workflow-run-crate-0.5` → `process-run-crate-0.5` → `ro-crate-1.1`).
2. `apmode bundle rocrate export <bundle_dir> --out <path>.zip` produces a
   ZIP with identical metadata + embedded bundle files; validator passes on
   the unzipped copy.
3. Sealed bundle directory is **unchanged** after export.
4. `_COMPLETE` is included as a `File` entity with
   `additionalType="apmode:completeSentinel"` and appears in
   `./` (root Dataset) `hasPart`.
5. One syrupy golden snapshot for a canonical Submission run lives in
   `tests/golden/rocrate/`.
6. `uv run mypy src/apmode/ --strict` and `uv run ruff check/format` clean.
7. Pre-existing 1746 tests continue to pass.

---

## Build order (TDD, one commit per block)

### Commit 1 — Scaffolding: `vocab`, `context`, public surface, deps

- `src/apmode/bundle/rocrate/__init__.py` — re-export public API
- `src/apmode/bundle/rocrate/vocab.py` — `APMODE_TERMS_BASE`, term constants
- `src/apmode/bundle/rocrate/context.py` — `build_rocrate_context(include_provagent=False)`, profile URI constants
- `pyproject.toml` — `rocrate>=0.13` (main), `roc-validator>=0.8.1` (dev group) — **already added**
- Tests: `tests/unit/rocrate/test_vocab.py`, `tests/unit/rocrate/test_context.py`

Write tests first per TDD: assert constants have the correct URIs, that
`build_rocrate_context` includes expected keys and excludes `provagent:`
when `include_provagent=False`.

### Commit 2 — Entity projectors (leaf modules)

- `src/apmode/bundle/rocrate/entities/__init__.py`
- `src/apmode/bundle/rocrate/entities/data.py` — `DataManifest` → root Dataset + File entity
- `src/apmode/bundle/rocrate/entities/policy.py` — `policy_file.json` → File + `apmode:lanePolicy`
- `src/apmode/bundle/rocrate/entities/backend.py` — `BackendResult` + `DSLSpec` → CreateAction + SoftwareApplication + File
- `src/apmode/bundle/rocrate/entities/gate.py` — `GateResult` → ControlAction + HowToStep + File
- `src/apmode/bundle/rocrate/entities/lineage.py` — `CandidateLineage` → `prov:wasDerivedFrom` triples
- `src/apmode/bundle/rocrate/entities/credibility.py` — `CredibilityReport`, `ReportProvenance` → File entities
- `src/apmode/bundle/rocrate/entities/bayesian.py` — `PriorManifest`, `SimulationProtocol`, `PosteriorDiagnostics` → File entities
- `src/apmode/bundle/rocrate/entities/agentic.py` — `AgenticTraceInput/Output/Meta` → CreateAction per iteration + `prov:wasInformedBy`
- `src/apmode/bundle/rocrate/entities/pccp.py` — `regulatory/{md,mp,ia,traceability}.json` file projection (skipped gracefully if absent)

Each projector is a pure function
`add_X(graph: list[dict], bundle_dir: Path, ...) -> str | None` that:
- returns the `@id` of the primary entity it added (or `None` if the
  source artifact is absent);
- appends entities to `graph` idempotently (dedupe by `@id`);
- never reads the live filesystem outside `bundle_dir`;
- never writes to disk.

Corresponding unit tests in `tests/unit/rocrate/test_entities_*.py`.

### Commit 3 — `projector.py` orchestrator + ZIP support

- `RoCrateProfile` (enum) — `PROVENANCE_RUN_CRATE`, `WORKFLOW_RUN_CRATE`, `PROCESS_RUN_CRATE`
- `RoCrateExportOptions` — pydantic dataclass with defaults per plan §F
- `RoCrateEmitter.export_from_sealed_bundle(bundle_dir, out, options=...) -> Path`
  - Refuse unsealed bundles (must contain `_COMPLETE`)
  - Load lane from `policy_file.json` (first check), fall back to
    walking `gate_decisions/` — `Submission` is the v0.6 supported lane;
    other lanes produce a minimal crate with a warning metadata field
  - Read each JSON/JSONL artifact into Pydantic where possible,
    gracefully skipping artifacts with schema drift
  - Call each `add_X` projector in deterministic order
  - Build `@graph` array; sort entities so root Dataset and metadata
    descriptor come first, then SoftwareApplication entries, then
    Actions, then Files — keeps golden snapshots stable
  - Compute SHA-256 for each File entity from the *source bundle*
  - If `out` ends in `.zip` → write ZIP containing all bundle files +
    `ro-crate-metadata.json` at crate root
  - Else → copy bundle contents into `out/` and drop `ro-crate-metadata.json`
  - Return the output `Path`

Tests:
- `tests/unit/rocrate/test_projector.py` — golden with syrupy
  (`tests/golden/rocrate/__snapshots__/test_projector.ambr`)
- `tests/unit/rocrate/test_projector_zip.py` — zip form validation

### Commit 4 — CLI wiring + integration test

- `src/apmode/bundle/rocrate/cli_hooks.py` — `register_rocrate_commands(app)`
  adds a `bundle` subcommand group with `rocrate export` and `publish` (stub)
- `src/apmode/bundle/rocrate/publish.py` — stub that raises
  `NotImplementedError("publishing lands in v0.8")` but has the Typer
  signature per plan §F.
- `src/apmode/cli.py` — import & call `register_rocrate_commands(app)`
- Extend `apmode validate` with `--rocrate / --no-rocrate`,
  `--profile`, `--severity` flags that invoke roc-validator when a
  crate is present at the default path `<bundle_dir>/../rocrate/` or
  when `--crate-dir` is explicit.
- Extend `apmode inspect` with `--rocrate-view` flag that prints key
  `@id` / `mainEntity` / top-level action triad.
- `tests/integration/test_rocrate_export_validate.py` — end-to-end:
  build a Submission-lane bundle fixture, seal it, export, validate
  with roc-validator at REQUIRED, assert `result.passed()`.

---

## Key implementation decisions

**Bundle reading**: Each projector reads the relevant JSON file with
`json.load`, then tries Pydantic validation; if validation fails due to
schema drift, the projector logs a structured warning and falls back to
dict-based minimal projection. This avoids Pydantic-version brittleness
(plan §I risk 3).

**`@id` scheme**:
- Files: bundle-relative POSIX path (e.g., `data_manifest.json`,
  `compiled_specs/c001.json`). ro-crate-py convention.
- Workflow: `workflows/<lane>-lane.apmode` (virtual File entity;
  no file on disk — ro-crate spec permits `File` referencing external
  resources, but for the v0.6 zip form we materialize a tiny text file
  with the lane definition to keep validators happy).
- Action entities: hash-prefix (`#`) — e.g., `#backend-create-c001`,
  `#gate1-control-c001`, `#run-organize-action`.
- DSL language: `#apmode-dsl`.
- Orchestrator: `#apmode-orchestrator`.

**SHA-256**: computed during the copy pass; stored on `File` entities
via schema.org `sha256` property (`https://schema.org/sha256`).

**Conformance**: Root `Dataset.conformsTo` is a list of `{"@id": URI}`
entries — Provenance Run Crate v0.5, Workflow Run Crate v0.5, Process
Run Crate v0.5, Workflow RO-Crate, RO-Crate 1.1. The
metadata-descriptor `conformsTo` points at the RO-Crate 1.1 context.

**License on root Dataset**: GPL-2.0-or-later — required for ro-crate-1.1
REQUIRED validation per roc-validator defaults.

**Determinism**: Projectors build a `dict[str, dict]` keyed by `@id`,
then the orchestrator serializes in a fixed section order (metadata
descriptor → root Dataset → SoftwareApplication → HowTo/HowToStep →
Action entities → File entities, each section sorted by `@id`).

**Lane detection**: `policy_file.json.lane` when present; else search
`gate_decisions/*.json` for `policy_version` to infer; default
`"Submission"` for v0.6. Store in `apmode:lane` on root Dataset.

**PCCP / regulatory**: `entities/pccp.py` looks for
`regulatory/md.json`, `regulatory/mp.json`, `regulatory/ia.json`,
`regulatory/traceability.csv`. If present → project as Files + set
corresponding `apmode:*` properties on root Dataset and
`apmode:regulatoryContext="pccp-ai-dsf"`. If absent → skip silently and
set `apmode:regulatoryContext="research-only"`.

**Agentic trace**: `entities/agentic.py` walks `agentic_trace/`
matching `{iteration_id}_{input,output,meta}.json` triples, emits one
`CreateAction` per iteration linked to the next via
`prov:wasInformedBy`, and attaches `apmode:dslTransform` from
`parsed_transforms`. Degrades gracefully when directory is absent.

---

## Out of scope (per user's line "folder and zip support" + plan §H)

- Publishing to WorkflowHub / Zenodo (publish.py is a CLI-visible stub)
- `apmode bundle import` (reverse direction)
- PROV-AGENT `provagent:ModelInvocation` typing — `include_provagent=False`
  is the default; the context builder already exposes the flag for v0.9
- Discovery-lane tiering — v0.6 emits per-candidate actions for all
  candidates; the tiering cap will land in v0.7
- w3id.org/apmode/terms# URI redirect — the URI constant is wired in
  the code, but the redirect itself is a separate ops task tracked by a
  GitHub issue (`docs/adr/0002-apmode-w3id-uri.md` stub)

---

## Verification commands

```bash
uv run pytest tests/unit/rocrate/ -q
uv run pytest tests/integration/test_rocrate_export_validate.py -q
uv run pytest tests/golden/rocrate/ -q
uv run pytest tests/ -q              # full suite still passes
uv run mypy src/apmode/ --strict
uv run ruff check src/apmode/ tests/
uv run ruff format src/apmode/ tests/
uv run apmode bundle rocrate export tests/fixtures/<bundle> --out /tmp/crate
uv run apmode bundle rocrate export tests/fixtures/<bundle> --out /tmp/crate.zip
```
