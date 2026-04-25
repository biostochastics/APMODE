# Benchmark Suite B — Real-data Anchors with Controlled Perturbations

> PRD §5 / §8 Phase 2. Tests robustness to BLQ, outliers, sparse absorption,
> protocol heterogeneity, null covariates, IOV, covariate missingness, and
> the four PRD §10 stress surfaces (BSV scaling, saturable clearance, TMDD
> approximation, flip-flop).

## Cases

The case definitions live in `src/apmode/benchmarks/suite_b_extended.py`
(B4–B9 real-data anchors) and `src/apmode/benchmarks/suite_b.py` (B1–B3
NODE specs, currently mock-only — live wiring is out of v0.6 scope and
the runner skips them with a clear `skipped=True` flag).

| case | dataset | perturbation |
|---|---|---|
| `b4_theophylline_node` | `nlmixr2data_theophylline` | none (NODE anchor) |
| `b5_mavoglurant_blq25` | `nlmixr2data_mavoglurant` | inject_blq @ 25% |
| `b5_mavoglurant_blq40` | `nlmixr2data_mavoglurant` | inject_blq @ 40% |
| `b6_mavoglurant_outliers5` | `nlmixr2data_mavoglurant` | inject_outliers @ 5% |
| `b7_mavoglurant_sparse_absorption` | `nlmixr2data_mavoglurant` | remove_absorption_samples |
| `b8_mavoglurant_null_covariates` | `nlmixr2data_mavoglurant` | add_null_covariates × 5 |
| `b9_gentamicin_iov` | `ddmore_gentamicin` | none (IOV challenge) |

## Running the live runner

The Suite B runner needs R 4.4+ with `nlmixr2`, `rxode2`, `lotri`,
`xpose` installed. Off-CI box only.

```bash
# Built-in registry datasets resolve automatically; gentamicin needs an override.
uv run python -m apmode.benchmarks.suite_b_runner \
  --out benchmarks/suite_b/suite_b_results.json \
  --dataset-csv ddmore_gentamicin=/abs/path/to/gentamicin.csv \
  --n-seeds 3 \
  --base-seed 20260425 \
  --timeout-seconds 1800
```

For a fast smoke run on a single case:

```bash
uv run python -m apmode.benchmarks.suite_b_runner \
  --cases b4_theophylline_node \
  --n-seeds 2 \
  --estimation focei \
  --timeout-seconds 600
```

## Output schema (`suite_b_results.json`)

```jsonc
{
  "<case_id>": {
    "case_id": "...",
    "suite": "B",
    "dataset_id": "...",
    "skipped": false,                    // true for B1-B3 NODE cases
    "skip_reason": null,
    "n_seeds": 3,
    "convergence_rate": 1.0,             // (0..1) over n_seeds
    "cross_seed_cv_max": 0.07,           // PRD §5 R8 monitor
    "cross_seed_cv_per_param": {"CL": 0.05, "V": 0.07},
    "perturbation_manifests": [...],     // one per applied recipe
    "seed_results": [
      {
        "seed": 20260425,
        "converged": true,
        "minimization_status": "successful",
        "parameter_estimates": {"CL": 5.0, "V": 70.0, "ka": 1.4},
        "bic": 1234.5,
        "wall_time_seconds": 47.3
      },
      // ...
    ]
  }
}
```

## Scoring

```bash
uv run python -m apmode.benchmarks.suite_b_cli \
  --inputs benchmarks/suite_b/suite_b_results.json \
  --out scorecard.json \
  --markdown-summary scorecard.md
```

**Gate semantics** (default thresholds; override via CLI flags):

* `convergence_rate >= 0.80` per case.
* `cross_seed_cv_max <= 0.50` per case (50% — wide but not absurd; tighten
  with more seeds).
* Skipped cases (NODE B1–B3) excluded from gate maths.

A miss is a methodology drift signal, **not** a release block. The
nightly regeneration is operator-driven; this directory holds the
checked-in `suite_b_results.json` snapshot the per-PR scoring tests
read.

## Cross-seed stability monitor (PRD §5 / R8)

For each case the runner performs **N independent fits** at seeds
`base_seed + 7919*i` and reports the across-seed coefficient of
variation per parameter plus the headline `cross_seed_cv_max`. PRD R8
calls this out as a mitigation against diagnostic-side information
leakage in agentic iterations: an agentic LLM that overfits to a
specific RNG signature will exhibit large `cross_seed_cv_max`.

The current threshold (0.50) is intentionally loose — a v0.7 follow-up
should narrow it once enough seeds have been observed across the
B-case roster to set a data-driven ceiling.
