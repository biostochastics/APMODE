---
name: apmode-cli
description: Use when invoking the APMODE CLI (`uv run apmode ...`), interpreting reproducibility bundles, choosing a lane (submission / discovery / optimization), picking a backend (`nlmixr2` / `bayesian_stan` / NODE / agentic LLM), debugging gate rejections, or tracing candidate lineage in APMODE pharmacokinetic model-discovery runs.
---

# APMODE CLI

## Overview

APMODE (Adaptive Pharmacokinetic Model Discovery Engine) composes **five** popPK modeling paradigms — classical NLME (nlmixr2), automated structural search, Bayesian (Stan/Torsten via cmdstanpy), hybrid mechanistic-NODE (JAX/Diffrax), and agentic LLM — behind a single CLI. Every `apmode run` emits a **reproducibility bundle** (JSON/JSONL directory); every other command reads from that bundle.

**Invocation pattern:** always `uv run apmode ...` in this repo (Python 3.12–3.14; `uv sync --all-extras` once).

## Core mental model

1. **`apmode run`** produces a bundle directory under `./runs/` (or `--output-dir`).
2. Every other command consumes one or two bundle directories. You never re-run to "see" results — query the bundle.
3. **Lane is load-bearing**, not a dial. Pick deliberately:
   - `submission` — regulatory-grade and the **CLI default** (because it is the most conservative lane). Classical NLME (and Bayesian with prior justification) is eligible for "recommended". NODE/agentic cannot win a submission run — this is a hard PRD rule, not a weight. Choosing `discovery` or `optimization` relaxes that and should be an explicit, asked-for decision.
   - `discovery` — exploratory. All backends eligible for ranking.
   - `optimization` — translational; requires LORO-CV in Gate 2; NODE dim ≤ 4.
4. **Agentic backend is opt-in** (`--agentic`, `discovery` / `optimization` only). Ships aggregated diagnostics to a third-party LLM — full data-exposure caveat in Gotchas; surface it the first time a user enables the flag.
5. **Bayesian backend is opt-in** via `--backend bayesian_stan` (requires `uv sync --extra bayesian` + `cmdstanpy.install_cmdstan()`). Adds R̂/ESS/divergences/E-BFMI/Pareto-k to Gate 1.

**First-response reflexes.** On environment trouble → `apmode doctor`. On an unfamiliar bundle → `apmode inspect BUNDLE`. On a mysterious rejection → `apmode log BUNDLE --failed` then `apmode lineage BUNDLE <id>`.

**Machine-readable outputs.** Every read command (`validate`, `inspect`, `log`, `diff`, `datasets`, `doctor`, `policies`, `report`, `trace`, `lineage`) accepts `--json` and emits a stable `{"ok": bool, ...}` envelope on stdout (Rich output suppressed); errors travel through the envelope (e.g. `{"ok": false, "error": "not_a_directory"}`) rather than stderr. `apmode ls` and `apmode graph` use `--format` instead because they offer multiple text formats (`table|path|json` and `tree|dot|mermaid|json`).

## Commands (16 total)

| Command | Purpose |
|---|---|
| `apmode run DATASET` | Full pipeline: ingest → profile → NCA → search → gates → bundle → report |
| `apmode validate BUNDLE` | Structural completeness + JSONL integrity check; `--rocrate --crate <crate>` runs roc-validator at REQUIRED |
| `apmode inspect BUNDLE` | Summary: data manifest, evidence profile, search trajectory, gate decisions. **First thing to run on an unfamiliar bundle.** `--rocrate-view --crate <crate>` summarises an exported crate. |
| `apmode datasets [NAME]` | Registry of 14 public PK datasets (5 real + 9 simulated) |
| `apmode explore DATASET` | Interactive wizard (profile + NCA + search-space preview); `DATASET` can be a registry name or a local CSV |
| `apmode diff BUNDLE_A BUNDLE_B` | Side-by-side comparison (evidence, rankings, gate pass rates) |
| `apmode log BUNDLE` | Query logs / gate decisions / parameter estimates from a bundle |
| `apmode trace BUNDLE` | Agentic iteration traces (summary, `--iteration N`, `--cost`, `--json`); only meaningful if `--agentic` ran |
| `apmode lineage BUNDLE CANDIDATE_ID` | Transform chain root → candidate with per-step gate status |
| `apmode report BUNDLE` | Regulatory report — HTML in browser (default) or `--format md` |
| `apmode graph BUNDLE` | Search DAG visualization — `tree` / `dot` / `mermaid` / `json` formats |
| `apmode policies [LANE]` | List/inspect gate policies; `--validate` runs the CI schema hook |
| `apmode doctor` | Check R/nlmixr2/CmdStan/Python deps + LLM provider keys |
| `apmode ls` | List bundles under `./runs` with a summary table; `--sort bic/time`, `--limit N`, `--format table|path|json` |
| `apmode serve` | FastAPI HTTP API behind uvicorn (loopback default; refuses non-loopback without `--allow-public`). `POST/GET/DELETE /runs` + `GET /runs/{id}/{status,bundle,rocrate}`. Requires `uv sync --extra api`. |
| `apmode bundle` | RO-Crate + SBOM operations on a sealed bundle: `bundle rocrate export\|import\|publish`, `bundle import` (round-trip with digest verify), `bundle sbom` (CycloneDX sidecar; digest-exempt). |

### `apmode run DATASET` — the pipeline

`DATASET` is a NONMEM-style CSV with `ID, TIME, DV, AMT, EVID, MDV`. Key flags:

| Flag | Default | Use |
|---|---|---|
| `--lane {submission,discovery,optimization}` | `submission` | see mental model |
| `--backend {nlmixr2,bayesian_stan}` | `nlmixr2` | `bayesian_stan` requires `uv sync --extra bayesian` + CmdStan |
| `--seed INT` | `753849` | root seed; propagates to all backends |
| `--policy PATH` | `policies/<lane>.json` | versioned gate thresholds, not a tunable |
| `--timeout SEC` | `900` | per-candidate timeout (SAEM on 50 subj ≈ 10s; 120 subj ≈ 60–120s; 1000+ subj ≈ 300–600s) |
| `--output-dir PATH` / `-o` | `runs` | bundle parent directory |
| `--agentic / --no-agentic` | off | Phase 3 LLM loop (discovery/optimization only) |
| `--resume-agentic` | off | skip Stage 5 (classical SAEM) and reload `classical_checkpoint.json` — use after agentic API failure to avoid re-running multi-hour SAEM |
| `--provider {anthropic,openai,gemini,ollama,openrouter}` | `anthropic` | agentic provider |
| `--model TEXT` | provider default (`anthropic`=`claude-sonnet-4-20250514`, `openai`=`gpt-4o`, `gemini`=`gemini-2.5-flash`, `ollama`=`qwen3:4b`) | override model |
| `--max-iterations INT [1..25]` | `10` | agentic iteration count; PRD §4.2.6 hard cap is 25 |
| `--parallel-models INT` / `-j` | `1` | concurrent R subprocesses |
| `--bayes-chains / --bayes-warmup / --bayes-sampling / --bayes-adapt-delta / --bayes-max-treedepth` | 4 / 1000 / 1000 / 0.8 / 12 | Stan-only knobs |
| `--binary-encode "COL=V1:0,V2:1"` (repeatable) | auto-detect | override binary-categorical remap, e.g. `--binary-encode SEX=M:0,F:1` |
| `--dry-run` | off | preview pipeline without dispatching backends |

**When a user asks "run APMODE on X": ask the lane first if not specified.** Do not silently default to `discovery` — `submission` is the CLI default for a reason.

### `apmode log BUNDLE`

```
apmode log runs/run_abc123                   # summary
apmode log runs/run_abc123 --gate gate1      # details (gate1|gate2|gate2_5|gate3)
apmode log runs/run_abc123 --failed          # only failed candidates
apmode log runs/run_abc123 --top 3           # top-N ranked with parameter estimates
```

`--gate` and `--top` are separate views — `--top` is ignored when `--gate` is set.

### `apmode trace BUNDLE`

Agentic iteration traces. The agentic stage writes two modes — `refine` and `independent` — each in its own subdir.

```
apmode trace runs/run_abc123                  # summary across modes
apmode trace runs/run_abc123 --iteration 5    # one iteration in detail
apmode trace runs/run_abc123 --cost           # token + $ rollup
apmode trace runs/run_abc123 --json           # machine-readable
```

### `apmode graph BUNDLE`

```
apmode graph runs/run_abc123                                 # tree (default)
apmode graph runs/run_abc123 --format dot -o dag.dot         # Graphviz
apmode graph runs/run_abc123 --format mermaid                # for docs
apmode graph runs/run_abc123 --converged --backend nlmixr2   # filter
```

### `apmode lineage BUNDLE CANDIDATE_ID`

Transform chain root → candidate with per-step gate outcomes. Use when explaining *why* a specific model was produced or rejected.

```
apmode lineage runs/run_abc123 cand_a3f8            # chain + gates
apmode lineage runs/run_abc123 cand_a3f8 --spec     # include DSL spec per step
apmode lineage runs/run_abc123 cand_a3f8 --json
```

## Typical workflows

### New dataset, no prior knowledge
```
apmode explore ./data/mydata.csv                         # wizard; NCA + search space preview
apmode run ./data/mydata.csv --lane discovery --seed 42
apmode inspect runs/run_<hash>
apmode log runs/run_<hash> --top 5
apmode report runs/run_<hash>                            # open HTML report
```

### Submission-track model
```
apmode run ./data/trial.csv --lane submission --timeout 1800
apmode validate runs/run_<hash>
apmode log runs/run_<hash> --gate gate3
apmode log runs/run_<hash> --top 3
apmode report runs/run_<hash> --format md | less
```

### Bayesian backend (prior-informed, FDA-aligned)
```
uv sync --extra bayesian
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
apmode run ./data/trial.csv --lane discovery \
    --backend bayesian_stan --bayes-chains 4 --bayes-warmup 1000 --bayes-sampling 1000
apmode log runs/run_<hash> --gate gate1            # inspect R̂/ESS/divergences/E-BFMI/Pareto-k
```

### Enable agentic LLM exploration (opt-in)
```
apmode run ./data/trial.csv \
    --lane discovery --agentic --provider anthropic \
    --max-iterations 15 -j 4
apmode trace runs/run_<hash> --cost
```

### Resume after an agentic API failure
```
apmode run ./data/trial.csv --lane discovery --agentic --resume-agentic \
    -o ./runs/run_<previous_timestamp>
```
Skips Stage 5 (classical SAEM) and reloads `classical_checkpoint.json` from the existing bundle dir.

### Debug a rejected candidate
```
apmode log runs/run_<hash> --failed
apmode lineage runs/run_<hash> <candidate_id> --spec
apmode graph runs/run_<hash> --format mermaid -o dag.md
```

### Compare two policy configurations
```
apmode run data.csv --lane submission --policy policies/submission.json -o runs/a
apmode run data.csv --lane discovery  --policy policies/discovery.json  -o runs/b
apmode diff runs/a/run_<hash> runs/b/run_<hash>
```

### Serve the HTTP API (single-user / lab-network)
```
uv sync --extra api
uv run apmode serve                              # 127.0.0.1:8765, runs in ./runs
uv run apmode serve --port 9000 --runs-dir /scratch/apmode-runs
uv run apmode serve --host 0.0.0.0 --allow-public   # only behind an authenticating reverse proxy
uv run apmode serve --allow-bayesian             # extends POST /runs allowlist
```
Endpoints: `POST /runs` → 202 + `Retry-After: 5`; `GET /runs/{id}/status`; `GET /runs/{id}/bundle` (425 *Too Early* until sealed); `GET /runs/{id}/rocrate`; `DELETE /runs/{id}` (cancels asyncio task → SIGTERM child process group → 5 s grace → SIGKILL → writes `RunStatus.CANCELLED`). The `apmode serve` CLI ships **no auth** — loopback is the security gate.

### Project a sealed bundle to RO-Crate / attach SBOM
```
apmode bundle rocrate export runs/run_<hash> --out runs/run_<hash>.crate.zip
apmode bundle import runs/run_<hash>.crate.zip --out runs/run_<hash>-imported   # SHA-256 round-trip
apmode validate runs/run_<hash> --rocrate --crate runs/run_<hash>.crate.zip      # roc-validator REQUIRED
apmode bundle sbom runs/run_<hash>                                                # bom.cdx.json sidecar
```

## Environment variables (`apmode run`)

The following env-vars bind to `apmode run` flags when the matching CLI flag is omitted (each shows up in `apmode run --help` under `[env var: …]`):

| Variable | Flag bound |
|---|---|
| `APMODE_LANE` | `--lane` |
| `APMODE_BACKEND` | `--backend` |
| `APMODE_SEED` | `--seed` |
| `APMODE_TIMEOUT` | `--timeout` |
| `APMODE_OUTPUT_DIR` | `-o` / `--output-dir` |
| `APMODE_PROVIDER` | `--provider` |
| `APMODE_MODEL` | `--model` |
| `APMODE_AGENTIC` | `--agentic` / `--no-agentic` |
| `APMODE_AGENTIC_MAX_ITER` | `--max-iterations` |
| `APMODE_PARALLEL_MODELS` | `-j` / `--parallel-models` |
| `APMODE_POLICY` | `--policy` |
| `APMODE_POLICIES_DIR` | (resolved separately in `paths.py` — directory override for policy lookup) |

LLM provider auth: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `OPENROUTER_API_KEY`; non-default Ollama via `OLLAMA_HOST`.

## Gotchas

- **Don't conflate lane with provider.** `--lane` picks the pipeline and gate policy; `--provider` only matters if `--agentic` is passed.
- **NODE / agentic cannot be "recommended" in `submission`** — hard PRD rule. If a user expects a NODE model to top a submission run, they're misreading the architecture.
- **`--agentic` on `--lane submission` silently no-ops.** The agentic runner is only constructed for `discovery` / `optimization` — no warning is printed. Check `agentic_trace/` presence to confirm it ran.
- **`--agentic` ships aggregated diagnostics to a third-party LLM** (never per-subject data). Surface this when first enabling. Use `--provider ollama` for local-only runs.
- **`apmode trace --iteration` needs `--mode`** when the run produced multiple agentic modes (default: both `refine` + `independent`). Omitting `--mode` exits code 1.
- **Bayesian backend requires extras + CmdStan**: `uv sync --extra bayesian` and `cmdstanpy.install_cmdstan()` — `apmode doctor` checks both.
- **Bayesian Gate 1 adds five MCMC thresholds** on top of the classical Gate 1: R̂ ≤ 1.01, bulk/tail ESS ≥ 400, divergences = 0, E-BFMI ≥ 0.3, Pareto-k ≤ 0.7 (all in `policies/<lane>.json`).
- **Gate thresholds live in `policies/<lane>.json`, not code.** To change admission criteria, edit JSON or pass `--policy` — don't patch sources. `apmode policies` lists versions; `apmode policies --validate` runs the CI schema hook.
- **`--timeout` too low ≠ "model is bad".** SAEM on 1000+ subjects needs 300–600s; the 900s default can still be tight.
- **Seed is reproducibility, not tuning.** Never sweep `--seed` to "find a better fit" — that's RNG cherry-picking and it shows in the bundle audit trail.
- **Run `apmode run` with `-o` to an ephemeral dir in CI** to avoid polluting `./runs/`.
- **Bundle paths are the API.** Every post-run command takes `BUNDLE_DIR`, not the CSV. If someone passes the dataset to `inspect` / `log` / `trace`, they've confused the layers.
- **`--binary-encode` overrides auto-detection.** Only needed when the profiler's categorical remap has the wrong polarity (e.g. you want `SEX=F:0,M:1` instead of the detected order).
- **`APMODE_POLICIES_DIR` env var** overrides policy file resolution (see `src/apmode/paths.py`). Useful when testing alternate policy sets without mutating `./policies/`. The full `APMODE_*` env-var family for `apmode run` is documented above.
- **`apmode serve` ships no auth.** The loopback default *is* the security gate. Non-loopback binds (`0.0.0.0`, RFC 1918, public IPs, hostnames) exit code 2 unless `--allow-public` is passed; DNS-resolved hostnames are rejected on principle. To expose the API beyond loopback, front it with an authenticating reverse proxy (Caddy `basicauth`, nginx + OIDC sidecar, Tailscale).
- **`DELETE /runs/{id}` is a four-link cancellation.** Cancel the asyncio task → runner's `asyncio.CancelledError` handler invokes `terminate_process_group` (SIGTERM → 5 s grace → SIGKILL on the child process group) → `execute_run` writes `RunStatus.CANCELLED` → uvicorn `--timeout-graceful-shutdown 30` covers the SIGTERM-to-SIGKILL window. If you ever change the runner cancellation path, also update the helper in `src/apmode/backends/process_lifecycle.py` and the lifespan tests in `tests/integration/test_api_runs.py`.
- **HTTP API extras gate.** `apmode serve` raises a typed friendly error (exit 1) when `[api]` extras aren't installed: `Error: the HTTP API extras are not installed. Run uv sync --extra api …`. `--allow-bayesian` extends the `POST /runs` backend allowlist beyond the default `("nlmixr2",)`.
- **GET /runs/{id}/bundle returns 425 Too Early** while the bundle is unsealed (RFC 8470). Don't treat 425 as a failure — back off and re-poll the status endpoint.

## Quick reference: bundle contents (from `apmode run`)

- `_COMPLETE` — atomic seal written last; JSON carrying `{schema_version, run_id, file_digests: {path → sha256}}`. `apmode validate` refuses bundles without it.
- `data_manifest.json`, `split_manifest.json`, `seed_registry.json`, `backend_versions.json` — inputs / provenance
- `evidence_manifest.json` — profiler output (drives backend dispatch); includes structured `nonlinear_clearance_signals` + embedded `policy_sha256`
- `initial_estimates.json`, `nca_diagnostics.jsonl` — NCA output + per-subject QC (unit-aware CL scale factor recorded in `_unit_scale_applied`)
- `search_trajectory.jsonl`, `search_graph.json` — candidates explored + DAG edges
- `failed_candidates.jsonl` — rejections with reasons
- `candidate_lineage.json` — transform lineage (source for `apmode lineage`)
- `ranking.json` — Gate 3 ranking (source for `apmode log --top`)
- `imputation_stability.json` — MI / Rubin-pooled diagnostics when missingness triggered MI
- `missing_data_directive.json` — FREM vs MI-PMM vs MI-missRanger selection
- `categorical_encoding_provenance.json` — per-column binary-remap audit trail
- `gate_decisions/` — per-gate JSON (`gate1_*`, `gate2_*`, `gate25_*`, `gate3_*`)
- `compiled_specs/` — DSL → backend-specific model code (`.json` AST + `.R` lowering)
- `bayesian/` — only if `--backend bayesian_stan` ran: `prior_manifest.json`, `simulation_protocol.json`, `mcmc_diagnostics.json` (R̂ / ESS / E-BFMI / Pareto-k), `sampler_config.json`, `posterior_summary.json`, `posterior_draws/*.parquet`, plus per-candidate `{cid}_loo_summary.json` (Pareto-k bands), `{cid}_prior_data_conflict.json` (Box 1980 / Evans–Moshonov 2006 conflict fraction), `{cid}_prior_sensitivity.json` (Roos 2015 / Kallioinen 2024 power-scaling), and `{cid}_reparameterization_recommendation.json` (Betancourt–Girolami 2015) — all consumed by Gate 1 Bayesian and Gate 2 prior-justification
- `loro_cv/` — only on `--lane optimization`
- `credibility/` — Gate 2.5 per-candidate ICH M15 credibility reports
- `agentic_trace/` + `classical_checkpoint.json` — only if `--agentic` ran; `classical_checkpoint.json` is read by `--resume-agentic`
- `report.html` + `report.md` — regulatory report at run root (source for `apmode report`)
- `bom.cdx.json` — CycloneDX SBOM sidecar (only if `apmode bundle sbom <bundle>` ran or CI/release workflows attached one). **Digest-exempt**: in `_DIGEST_EXCLUDED_NAMES` alongside `_COMPLETE` and `sbc_manifest.json`, so adding/regenerating it never invalidates the seal.
- `sbc_manifest.json` — Talts 2018 Simulation-Based Calibration roll-up. Producer emits a stub (`priors=[]`) so its presence signals the Bayesian path executed end-to-end; nightly runner repopulates. Also digest-exempt.

Prefer `apmode inspect` / `log` / `lineage` / `graph` / `report` over reading these directly — use direct access only for a field the CLI doesn't surface.

## When APMODE is the wrong tool

- **Single-subject PK fitting** → use nlmixr2/Pumas/NONMEM directly; APMODE's governance overhead buys nothing.
- **Pure NCA reporting** → `apmode explore` shows NCA, but a dedicated NCA workflow (PKNCA, Phoenix) is lighter.
- **Non-PK longitudinal modeling** → out of scope; the DSL is PK-specific — a 5-block grammar (`Absorption × Distribution × Elimination × Variability × Observation`) plus a sixth semantic axis (`priors`) populated via `SetPrior` rather than grammar text.
- **Structural-identifiability-only tasks** → use COMBOS/DAISY/GenSSI directly; APMODE folds identifiability into Gate 2 but doesn't expose it as a standalone workflow.
