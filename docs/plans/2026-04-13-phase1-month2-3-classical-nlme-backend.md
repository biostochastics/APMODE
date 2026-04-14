# Phase 1 Month 2-3: Classical NLME Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the nlmixr2 classical NLME backend via R subprocess integration, data ingestion pipeline, and Benchmark Suite A scaffolding — bringing the system from "DSL compiler complete" to "can run a real PK model estimation end-to-end."

**Architecture:** Python orchestrator spawns R subprocess per estimation request. Communication is file-based (request.json to Rscript harness.R to response.json) to avoid R stdout contamination. Process isolation ensures R segfaults don't crash the orchestrator. Data flows through Pandera-validated canonical schema with adapters for nlmixr2-specific column naming.

**Tech Stack:** Python 3.12+ (asyncio subprocess), R 4.4+ (nlmixr2, rxode2), Pydantic v2, Pandera, pytest, Hypothesis.

**PRD References:** S4.2.0 (Data Ingestion), S4.2.0.1 (Initial Estimates), S4.2.2 (Classical NLME Backend), S4.3.2 (Reproducibility Bundle), S5 Suite A (Benchmark), S8 (Phasing).

**Architecture References:** ARCHITECTURE.md S4.1 (BackendRunner), S4.2 (R Subprocess Contract), S5 (Bundle Structure).

---

## Dependency Graph

```
Task 1: Nlmixr2Runner ──────────┬──> Task 2: R Harness
Task 3: Wire initial_estimates  │
Task 4: Data Ingestion ─────────┼──> Task 5: nlmixr2 Adapter
                                └──> Task 7: Suite A Benchmarks
Task 6: Fix Known Limitations (independent)
```

---

## Task 1: R Subprocess Runner (Nlmixr2Runner)

**Goal:** Concrete BackendRunner implementation that spawns Rscript as a subprocess, communicates via JSON files, and maps exit codes to typed errors.

**Files:**
- Create: `src/apmode/backends/nlmixr2_runner.py`
- Test: `tests/unit/test_nlmixr2_runner.py`
- Test: `tests/integration/test_nlmixr2_integration.py`

### Step 1: Write failing tests for Nlmixr2Runner

Tests should cover:
- Default and custom r_executable
- Default harness_path resolution
- Protocol compliance (isinstance check against BackendRunner)
- Request JSON file creation
- Exit code mapping: 0=success, 1=R error -> CrashError, 137=killed -> CrashError, 139=segfault -> CrashError
- Missing response.json -> CrashError
- Timeout handling -> BackendTimeoutError with process group kill
- Convergence error in response -> ConvergenceError
- Successful response -> BackendResult

### Step 2: Implement Nlmixr2Runner

Key implementation details:
- Uses asyncio.create_subprocess_exec (not shell=True) to spawn Rscript
- Creates a new process group via preexec_fn=os.setsid for clean timeout kill
- On timeout: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
- Writes request.json to unique work directory per run
- Reads response.json after subprocess completes
- Maps RSubprocessResponse to BackendResult or raises typed error

### Step 3: Run tests and commit

---

## Task 2: R-Side Estimation Harness (harness.R)

**Goal:** The R script that Nlmixr2Runner spawns. Reads request.json, constructs nlmixr2 model from compiled R code, runs estimation, extracts results, writes response.json.

**Blocked by:** Task 1

**Files:**
- Create: `src/apmode/r/harness.R`
- Create: `src/apmode/r/install_deps.R` (one-time nlmixr2 install helper)
- Test: `tests/integration/test_r_harness_roundtrip.py`

### Step 1: Create the R harness

The harness must:
1. Read request.json with jsonlite
2. Set RNG seed with specified rng_kind (L'Ecuyer-CMRG or Mersenne-Twister)
3. Load the data CSV from data_path
4. Eval the compiled R model code from compiled_r_code field
5. Call nlmixr2(model, data, est=method) for each estimation method
6. Extract: parameter estimates, OFV, AIC, BIC, convergence info, ETA shrinkage, GOF metrics, condition number
7. Capture sessionInfo() into RSessionInfo structure
8. Write response.json with proper NULL handling (auto_unbox=TRUE, null="null")
9. tryCatch wraps everything: on error, write crash response and exit(1)

### Step 2: Add compiled_r_code field to RSubprocessRequest

Add `compiled_r_code: str = ""` to the Pydantic model so the R harness receives the lowered R code inline.

### Step 3: Wire compiled_r_code in Nlmixr2Runner.run()

Call emit_nlmixr2(spec) and pass the resulting R code string as compiled_r_code in the request.

### Step 4: Integration test (skip if R not available)

Test the full roundtrip: Python writes request.json, Rscript harness.R processes it, Python reads response.json.

### Step 5: Commit

---

## Task 3: Wire initial_estimates Into Emitter

**Goal:** Allow emit_nlmixr2() to accept an optional initial_estimates dict that overrides the DSLSpec parameter values in the ini block.

**Files:**
- Modify: `src/apmode/dsl/nlmixr2_emitter.py` (emit_nlmixr2 signature)
- Modify: `src/apmode/bundle/emitter.py` (write_compiled_spec signature)
- Test: `tests/unit/test_nlmixr2_emitter.py`

### Step 1: Write failing tests

- test_initial_estimates_override: full override of CL, V, ka values
- test_initial_estimates_partial_override: only CL overridden, others keep DSLSpec values
- test_initial_estimates_none: None means use DSLSpec values (existing behavior)

### Step 2: Modify emit_nlmixr2 signature

```python
def emit_nlmixr2(
    spec: DSLSpec,
    initial_estimates: dict[str, float] | None = None,
) -> str:
```

In the ini block emission, check initial_estimates dict before using spec values.

### Step 3: Update BundleEmitter.write_compiled_spec

Pass through initial_estimates to emit_nlmixr2.

### Step 4: Run all tests and commit

Existing golden master snapshots should pass unchanged (None override = same behavior).

---

## Task 4: Data Ingestion Pipeline (NONMEM CSV)

**Goal:** Read NONMEM-style CSV files, validate against CanonicalPKSchema, compute SHA-256, detect covariates, produce DataManifest.

**Files:**
- Create: `src/apmode/data/ingest.py`
- Create: `tests/unit/test_data_ingest.py`
- Create: `tests/fixtures/pk_data/simple_1cmt.csv`

### Step 1: Create test fixture CSV

A minimal NONMEM CSV with 2 subjects, 7 timepoints each, WT and SEX covariates.

### Step 2: Write failing tests

- test_valid_csv: correct n_subjects, n_observations, n_doses, ingestion_format, SHA-256
- test_column_mapping: correct canonical column names
- test_covariate_detection: WT (continuous) and SEX (categorical) detected
- test_sha256_deterministic: same file -> same hash
- test_missing_required_column: raises ValueError

### Step 3: Implement ingest_nonmem_csv()

Key implementation:
- pd.read_csv -> check required columns -> CanonicalPKSchema.validate(lazy=True)
- hashlib.sha256 on raw file bytes
- Detect covariates: columns not in canonical set; dtype=object -> categorical, else continuous
- Build ColumnMapping with optional columns present/absent
- Return (DataManifest, validated DataFrame)

### Step 4: Run tests and commit

---

## Task 5: Data Format Adapter (Canonical to nlmixr2)

**Goal:** Convert from CanonicalPKSchema column names to nlmixr2-expected column names.

**Blocked by:** Task 4

**Files:**
- Create: `src/apmode/data/adapters.py`
- Test: `tests/unit/test_data_adapters.py`

### Step 1: Write failing test

nlmixr2 uses "ID" instead of "NMID". All other columns (TIME, DV, etc.) are the same.

### Step 2: Implement to_nlmixr2_format()

Simple column rename: NMID -> ID.

### Step 3: Commit

---

## Task 6: Fix Known Limitations

**Goal:** Address 3 high-priority issues from code review.

**Files:**
- Modify: `src/apmode/bundle/emitter.py`
- Modify: `tests/unit/test_bundle_emitter.py`

### 6a: Bundle emitter NODE guard

In write_compiled_spec, check spec.has_node_modules() and skip R emission (return None for r_path instead of raising NotImplementedError).

Return type changes to: tuple[Path, Path | None]

### 6b: Document BLQ M3/M4 composition limitation

Add comment in emitter and Known Limitations entry.

### 6c: RSubprocessResponse.result docstring

Clarify the dict[str, Any] contract in the docstring (BackendResult validation happens in the runner, not the response model).

### Step: Run tests and commit

---

## Task 7: Benchmark Suite A Scaffolding

**Goal:** Define 4 benchmark scenarios as DSLSpec objects with reference parameter values.

**Blocked by:** Tasks 1, 3, 4

**Files:**
- Create: `src/apmode/benchmarks/__init__.py`
- Create: `src/apmode/benchmarks/suite_a.py`
- Create: `tests/unit/test_benchmark_suite_a.py`
- Create: `benchmarks/suite_a/` (directory for generated data)

### Step 1: Define scenarios as DSLSpec

Per PRD S5 Suite A:
- A1: 1-cmt oral, first-order absorption, linear elimination
- A2: 2-cmt IV, parallel linear + MM elimination
- A3: Transit absorption (n=3), 1-cmt, linear elimination
- A4: 1-cmt oral, Michaelis-Menten elimination

Each scenario function returns a DSLSpec with realistic PK parameter values.
Include REFERENCE_PARAMS dict mapping scenario names to ground-truth parameter values.

### Step 2: Test scenarios

- Each spec passes validate_spec() with no errors
- Each spec compiles to R via emit_nlmixr2() (has ini and model blocks)
- Reference params match structural_param_names()

### Step 3: Commit

---

## Post-Implementation Checklist

After all 7 tasks are complete:

1. Run full test suite: `uv run pytest tests/ -q`
2. Type check: `uv run mypy src/apmode/ --strict`
3. Lint: `uv run ruff check src/apmode/ tests/`
4. Format: `uv run ruff format src/apmode/ tests/`
5. Update golden snapshots if needed: `uv run pytest tests/ --snapshot-update`
6. Update README.md with Month 2-3 status
7. Update CLAUDE.md if any architectural decisions change

## End-to-End Integration Test (stretch goal)

Once Tasks 1-5 are done, write an integration test that:
1. Ingests simple_1cmt.csv -> DataManifest
2. Creates a DSLSpec (scenario A1)
3. Emits R code with initial_estimates
4. Creates RSubprocessRequest
5. Spawns mock R script -> reads response -> BackendResult
6. Writes reproducibility bundle

This validates the full pipeline without requiring actual nlmixr2 installation.
