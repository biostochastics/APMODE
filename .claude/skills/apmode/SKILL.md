---
name: apmode-cli
description: Use when invoking the APMODE CLI (`apmode run/validate/inspect/datasets/explore/diff/log/trace/lineage/graph`), interpreting reproducibility bundles, choosing lanes (submission/discovery/optimization), enabling the agentic LLM backend, or debugging gate decisions and candidate lineage in APMODE pharmacokinetic model-discovery runs.
---

# APMODE CLI

## Overview

APMODE (Adaptive Pharmacokinetic Model Discovery Engine) composes four PK modeling paradigms — classical NLME, automated structural search, agentic LLM, hybrid NODE — behind a single CLI. Every run emits a **reproducibility bundle** (JSON/JSONL directory) that all other commands read from.

**Invocation:** always `uv run apmode ...` in this repo (Python 3.12, `uv sync --all-extras`).

## Core mental model

1. **`apmode run`** produces a bundle directory under `runs/` (or `--output`).
2. Everything else (`inspect`, `log`, `trace`, `lineage`, `graph`, `diff`, `validate`) consumes one or two bundle directories. You never re-run to "see" results — query the bundle.
3. **Lane choice is load-bearing**, not a dial. Pick deliberately:
   - `submission` — regulatory. Classical NLME only is eligible for "recommended." NODE/agentic cannot win here.
   - `discovery` — exploratory. All backends eligible for ranking, including agentic/NODE.
   - `optimization` — translational, LORO-CV, tight NODE constraints (dim ≤ 4).
4. **Agentic backend is opt-in** (`--agentic`) and only meaningful on `discovery`/`optimization`. It ships aggregated diagnostics to a third-party LLM — mention this when enabling for a new user.

## Command reference

### `apmode run DATASET` — the pipeline

Executes: `ingest → profile → NCA → split → search → gates (1, 2, 2.5, 3) → bundle`.

`DATASET` is a NONMEM-style CSV with `ID, TIME, DV, AMT, EVID, MDV` columns.

Key flags:

| Flag | Default | Use |
|---|---|---|
| `--lane {submission,discovery,optimization}` | `submission` | see mental model above |
| `--seed INT` | `753849` | root seed; propagates to all backends |
| `--policy PATH` | `policies/<lane>.json` | gate thresholds — **versioned policy artifact**, not a tunable |
| `--timeout SEC` | `900` | per-candidate backend timeout. 50 subj ≈ 10s, 120 ≈ 60–120s, 1000+ ≈ 300–600s |
| `--output PATH` | `runs` | bundle parent directory |
| `--agentic / --no-agentic` | off | enable Phase 3 LLM loop (discovery/optimization only) |
| `--provider {anthropic,openai,gemini,ollama,openrouter}` | `anthropic` | agentic provider |
| `--model TEXT` | provider default | override model. Defaults: `anthropic=claude-sonnet-4-20250514`, `openai=gpt-4o`, `gemini=gemini-2.5-flash`, `ollama=qwen3:4b`, `openrouter=anthropic/claude-sonnet-4-20250514` |
| `--max-iterations INT [1..25]` | `10` | agentic cap; PRD §4.2.6 hard cap is 25 |
| `--parallel-models / -j INT` | `1` | concurrent R subprocess evaluations |
| `--backend {nlmixr2,bayesian_stan}` | `nlmixr2` | Stan requires `uv sync --extra bayesian` + CmdStan |
| `--bayes-chains / --bayes-warmup / --bayes-sampling / --bayes-adapt-delta / --bayes-max-treedepth` | 4 / 1000 / 1000 / 0.95 / 12 | Stan-only knobs |
| `--binary-encode "COL=V1:0,V2:1"` (repeatable) | auto-detect | override binary categorical remap, e.g. `--binary-encode SEX=M:0,F:1` |
| `-v / -q` | | verbose / quiet logs |

**When the user asks "run APMODE on X":** ask the lane first if not specified. Do not default to `discovery` silently — `submission` is the CLI default for a reason.

### `apmode validate BUNDLE_DIR`
Structural completeness check on a bundle (required/optional JSON files, subdirs). Run after any manual bundle editing or before publishing.

### `apmode inspect BUNDLE_DIR`
Human-readable summary: data manifest, evidence profile, search trajectory, gate decisions. **First thing to run on an unfamiliar bundle.**

### `apmode datasets [NAME]`
Public PK dataset registry.

```
apmode datasets                          # list
apmode datasets theo_sd                  # download theophylline single-dose
apmode datasets --route oral -o ./data   # filter + custom dir
apmode datasets --elimination michaelis_menten
```

### `apmode explore DATASET [--lane] [--output] [--seed] [-y]`
Interactive wizard: fetch → ingest → profile → NCA → search space preview → optional run. `DATASET` is a registry name *or* a local CSV path.

- `--lane` **defaults to `discovery`** here (unlike `run`, which defaults to `submission`). Silent difference — flag it if the user expects `run`-equivalent behavior.
- `-y / --non-interactive` skips prompts and invokes `run` internally with a **hard-coded `timeout=600`s** (not the 900s `run` default). For large datasets (500+ subjects) prefer `run` directly.
- `--output PATH` (default `runs`), `--seed INT` (default `753849`).

### `apmode diff BUNDLE_A BUNDLE_B`
Side-by-side comparison: evidence manifest, search outcomes, gate decisions, rankings. Use after policy edits, seed changes, or backend swaps.

### `apmode log BUNDLE_DIR`
Query logs / gates / parameters from a bundle.

```
apmode log runs/run_abc123                   # summary
apmode log runs/run_abc123 --gate gate1      # details for one gate (gate1|gate2|gate2_5|gate3)
apmode log runs/run_abc123 --failed          # only failed candidates
apmode log runs/run_abc123 --top 3           # top 3 ranked with parameter estimates
```

### `apmode trace BUNDLE_DIR`
Agentic iteration traces (propose → validate → compile → fit). The agentic stage writes two modes by default — `refine` and `independent` — each in its own subdir.

```
apmode trace runs/run_abc123                              # all modes
apmode trace runs/run_abc123 --mode refine                # one mode
apmode trace runs/run_abc123 -i 5 --mode refine           # one iteration
apmode trace runs/run_abc123 --cost                       # token/$ aggregation
apmode trace runs/run_abc123 --json                       # machine-readable
```

Only meaningful if the run had `--agentic`.

### `apmode lineage BUNDLE_DIR CANDIDATE_ID`
Chain of DSL transforms from root to a target candidate, with per-step gate outcomes.

```
apmode lineage runs/run_abc123 cand_a3f8            # transform chain + gates
apmode lineage runs/run_abc123 cand_a3f8 --spec     # include DSL spec at each step
apmode lineage runs/run_abc123 cand_a3f8 --no-gate  # transforms only
apmode lineage runs/run_abc123 cand_a3f8 --json
```

Use when explaining *why* a specific model was produced or rejected.

### `apmode graph BUNDLE_DIR`
Full search DAG across all candidates.

```
apmode graph runs/run_abc123                                     # tree
apmode graph runs/run_abc123 -f dot -o dag.dot                   # Graphviz
apmode graph runs/run_abc123 -f mermaid                          # for docs
apmode graph runs/run_abc123 --converged --backend nlmixr2       # filter
apmode graph runs/run_abc123 --depth 5
```

Formats: `tree` (default), `dot`, `mermaid`, `json`.

## Typical workflows

### New dataset, no prior knowledge
```
apmode explore ./data/mydata.csv            # wizard; inspect NCA + search space
apmode run ./data/mydata.csv --lane discovery --seed 42
apmode inspect runs/run_<hash>
apmode log runs/run_<hash> --top 5
```

### Submission-track model
```
apmode run ./data/trial.csv --lane submission --timeout 1800
apmode validate runs/run_<hash>
apmode log runs/run_<hash> --gate gate3        # gate 3 decisions
apmode log runs/run_<hash> --top 3             # top-ranked with parameter tables
```

Note: `--gate` and `--top` are separate views — `--top` is ignored when `--gate` is set.

### Enable agentic LLM exploration (opt-in, non-default)
```
apmode run ./data/trial.csv \
  --lane discovery --agentic \
  --provider anthropic \
  --max-iterations 15 -j 4
apmode trace runs/run_<hash> --cost
```

### Debug a rejected candidate
```
apmode log runs/run_<hash> --failed
apmode lineage runs/run_<hash> <candidate_id> --spec
apmode graph runs/run_<hash> -f mermaid -o dag.md
```

### Compare two policy configurations
```
apmode run data.csv --lane submission --policy policies/submission.json --output runs/a
apmode run data.csv --lane discovery  --policy policies/discovery.json  --output runs/b
apmode diff runs/a/run_<hash> runs/b/run_<hash>
```

## Gotchas

- **Don't conflate lane with provider.** `--lane` picks the pipeline and gate policy; `--provider` only matters if `--agentic` is passed.
- **NODE / agentic cannot be "recommended" in `submission`.** A hard PRD rule. If a user expects a NODE model to top a submission run, they're misreading the architecture.
- **`--agentic` on `--lane submission` silently no-ops.** The agentic runner is only constructed for `discovery` / `optimization`. No warning is printed — check `agentic_trace/` presence to confirm it ran.
- **`--agentic` sends diagnostics to a third-party LLM.** Always surface this when first enabling. Use `ollama` provider for local-only runs.
- **`apmode trace --iteration` needs `--mode`** when the run produced multiple agentic modes (the default: `refine` + `independent`). Omitting `--mode` exits with code 1.
- **Gate thresholds live in `policies/<lane>.json`, not code.** To change admission criteria, edit or pass `--policy`, don't patch sources.
- **Timeout too low ≠ "model is bad".** SAEM on 1000+ subjects needs 300–600s; the 900s default can still be tight.
- **Seed is reproducibility, not tuning.** Never sweep `--seed` to "find a better fit" — that's RNG cherry-picking and it will show in the bundle audit trail.
- **`run` writes to `runs/` by default** — in CI or tests, always pass `--output` to an ephemeral dir to avoid polluting the workspace.
- **Bundle paths are the API.** Every post-run command takes `BUNDLE_DIR`, not the CSV. If someone passes the dataset to `inspect`/`log`/`trace`, they've confused the layers.
- **`--binary-encode` overrides auto-detection.** Only needed when the profiler's categorical remap is wrong (e.g. you want `SEX=F:0,M:1` instead of the detected order).

## Quick reference: bundle contents (from `run`)

A bundle directory contains (non-exhaustive, names verified against `cli.py`):

- `data_manifest.json`, `split_manifest.json`, `seed_registry.json`, `backend_versions.json` — inputs / provenance
- `evidence_manifest.json` — data profiler output (drives backend dispatch)
- `search_trajectory.jsonl`, `search_graph.json` — candidates explored + DAG
- `failed_candidates.jsonl` — rejections with reasons
- `candidate_lineage.json` — transform lineage (source for `apmode lineage`)
- `ranking.json` — Gate 3 ranking (source for `apmode log --top`)
- `imputation_stability.json` — missingness / imputation diagnostics
- `gate_decisions/` — per-gate JSON (gate1, gate2, gate2_5, gate3)
- `compiled_specs/` — DSL → backend-specific model code
- `agentic_trace/<mode>/agentic_iterations.jsonl` plus `iter_*_{input,output,meta}.json` — present only if `--agentic` ran on a discovery/optimization lane

Use `apmode inspect` / `log` / `lineage` / `graph` rather than reading these directly unless you need a field the CLI doesn't surface.

## When APMODE is the wrong tool

- Single-subject PK fitting → use nlmixr2/Pumas/NONMEM directly; APMODE's governance overhead buys nothing.
- Pure NCA → `apmode explore` shows NCA but a dedicated NCA workflow is lighter.
- Non-PK longitudinal modeling → out of scope; the DSL is PK-specific (`Absorption × Distribution × Elimination × Variability × Observation`).
