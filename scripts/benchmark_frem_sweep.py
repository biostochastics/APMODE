#!/usr/bin/env python3
# mypy: ignore-errors
# SPDX-License-Identifier: GPL-2.0-or-later
"""FREM benchmark sweep across the nlmixr2data benchmark corpus.

Exercises the full APMODE FREM pipeline (``summarize_covariates`` →
``prepare_frem_data`` → ``emit_nlmixr2_frem`` → nlmixr2 compile) on
every benchmark dataset registered in ``benchmarks/datasets/registry.yaml``
that ships with an nlmixr2data-backed prepare script.

Purpose:
  - Prove the emitter works on real pharmacometric data at every
    registered size tier (12 subjects → 222 subjects).
  - Cover the continuous / log-transformed / binary categorical
    transform branches on real covariates.
  - Surface data-specific issues (DVID collisions, multi-analyte
    layouts, unit weirdness) that synthetic fixtures can miss.

Scope is **compile-only**: we invoke ``nlmixr2(fn)`` and inspect the
emitted ``predDf`` / ``iniDf`` rather than running FOCE-I. Full fits
on the larger datasets (mavoglurant: 222 subjects, gentamicin: 205
subjects) take an hour+ each and belong to a separate nightly/weekly
benchmark, not a developer-time sweep. The compile path still
validates:

  1. DSL emit produces syntactically valid nlmixr2 R code,
  2. ``prepare_frem_data`` produces a dataset that rxode2 accepts
     (DVID routing, compartment inheritance, no contaminating cols),
  3. The joint Ω matrix is constructed correctly from per-dataset
     covariate summaries.

Usage:
  uv run python scripts/benchmark_frem_sweep.py
  uv run python scripts/benchmark_frem_sweep.py --output docs/FREM_BENCHMARK_RESULTS.md

Output:
  A markdown table summarizing per-dataset status, row counts,
  covariate coverage, and emit/compile wall times, written to stdout
  and optionally to a file.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Add src/ to path so we can import apmode without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from apmode.dsl.ast_models import (  # noqa: E402
    IIV,
    Combined,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
)
from apmode.dsl.frem_emitter import (  # noqa: E402
    emit_nlmixr2_frem,
    prepare_frem_data,
    summarize_covariates,
)


@dataclass
class BenchmarkResult:
    """One row in the benchmark sweep summary."""

    dataset: str
    loaded_rows: int = 0
    loaded_subjects: int = 0
    covariates_tested: list[str] = field(default_factory=list)
    induced_missing_subjects: int = 0
    tv_covariates: list[str] = field(default_factory=list)
    binary_covariates: list[str] = field(default_factory=list)
    emit_seconds: float | None = None
    compile_seconds: float | None = None
    emitted_model_bytes: int = 0
    status: str = "pending"  # "pass" | "fail" | "skip"
    note: str = ""


# Each entry: (R expression loading the dataset, covariate configuration).
# ``transforms`` maps covariate name to the desired scale; omitted
# covariates default to the auto-detect in summarize_covariates.
# ``drop_fraction`` sets how many subjects get induced WT missingness.
DATASETS: list[dict[str, object]] = [
    {
        "id": "nlmixr2data_theophylline",
        "r_load": "nlmixr2data::theo_sd",
        "id_col_rename": {"ID": "NMID"},
        "canonicalize_evid": True,  # EVID=101 → EVID=1
        "covariates": ["WT"],
        "transforms": {"WT": "log"},
        "drop_fraction": 0.25,
    },
    {
        "id": "nlmixr2data_warfarin",
        "r_load": "nlmixr2data::warfarin",
        "id_col_rename": {"id": "NMID"},
        "canonicalize_lowercase": True,  # id/time/amt/dv/evid → upper
        "filter_cmt": None,  # keep PK only (dvid/cmt filtering done below)
        "covariates": ["wt", "age", "sex"],
        "transforms": {"wt": "log", "sex": "binary"},
        # warfarin encodes sex as the strings "1"/"2" / "male"/"female"
        # depending on the source release; remap to canonical 0/1.
        "binary_encode": {"sex": {"male": 1, "female": 0, "1": 1, "2": 0, 1: 1, 2: 0}},
        "drop_fraction": 0.20,
    },
    {
        "id": "nlmixr2data_mavoglurant",
        "r_load": "nlmixr2data::mavoglurant",
        "id_col_rename": {"ID": "NMID"},
        "canonicalize_evid": False,
        "covariates": ["WT", "AGE", "SEX"],
        "transforms": {"WT": "log", "SEX": "binary"},
        # mavoglurant encodes SEX with 1-indexed integers (1/2). Remap to
        # 0/1 so the binary FREM endpoint validator accepts them.
        "binary_encode": {"SEX": {1: 0, 2: 1}},
        "drop_fraction": 0.15,
    },
]


def _check_prereqs() -> tuple[bool, str]:
    if not shutil.which("Rscript"):
        return False, "Rscript not in PATH"
    for pkg in ("nlmixr2", "nlmixr2data"):
        out = subprocess.run(
            ["Rscript", "-e", f'cat(requireNamespace("{pkg}", quietly=TRUE))'],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.stdout.strip() != "TRUE":
            return False, f"R package {pkg!r} not installed"
    return True, ""


def _load_dataset(cfg: dict[str, object], work_dir: Path) -> pd.DataFrame:
    """Generate the dataset CSV via Rscript and load as DataFrame."""
    csv_path = work_dir / f"{cfg['id']}.csv"
    gen_script = work_dir / f"{cfg['id']}_gen.R"
    r_load = cfg["r_load"]
    gen_script.write_text(
        f"""
suppressPackageStartupMessages({{ library(nlmixr2data) }})
df <- as.data.frame({r_load})
write.csv(df, '{csv_path}', row.names = FALSE)
"""
    )
    r = subprocess.run(
        ["Rscript", str(gen_script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if r.returncode != 0:
        msg = f"R dataset load failed: {r.stderr[-1500:]}"
        raise RuntimeError(msg)

    df = pd.read_csv(csv_path)
    # Column canonicalization.
    rename = cfg.get("id_col_rename", {})
    df = df.rename(columns=rename)  # type: ignore[arg-type]
    if cfg.get("canonicalize_lowercase", False):
        lower_map = {c: c.upper() for c in ("time", "amt", "dv", "evid", "mdv") if c in df.columns}
        df = df.rename(columns=lower_map)
    if cfg.get("canonicalize_evid", False) and "EVID" in df.columns:
        df.loc[df["EVID"] == 101, "EVID"] = 1

    # Apply per-dataset binary encoding for categorical covariates whose
    # native representation is not 0/1 (warfarin: "male"/"female";
    # mavoglurant: 1/2 1-indexed). The FREM emitter's ``binary``
    # transform requires {0, 1} so the additive-normal endpoint
    # interpretation as a linear group association is well-defined.
    binary_encode = cfg.get("binary_encode", {})
    for col, mapping in binary_encode.items():
        if col not in df.columns:
            continue
        # ``map`` returns NaN for unmapped values; coerce explicitly so
        # downstream summarize_covariates ignores those rows via dropna.
        df[col] = df[col].map(mapping)
    return df


def _compile_check(
    work_dir: Path,
    model_code: str,
    data_df: pd.DataFrame,
) -> tuple[bool, float, str]:
    """Invoke nlmixr2(fn) on the emitted model. Returns (ok, seconds, log)."""
    model_path = work_dir / "frem_model.R"
    model_path.write_text(model_code)
    data_path = work_dir / "frem_data.csv"
    data_df.to_csv(data_path, index=False)
    drive_path = work_dir / "drive.R"
    drive_path.write_text(
        f"""
suppressPackageStartupMessages({{ library(nlmixr2) }})
fn <- base::eval(parse(text = readLines('{model_path}')))
t0 <- Sys.time()
ui <- tryCatch(nlmixr2(fn),
  error = function(e) {{ cat('COMPILE_FAIL:', conditionMessage(e), '\\n'); NULL }})
t1 <- Sys.time()
if (!is.null(ui)) {{
  cat(sprintf('COMPILE_OK seconds=%.2f endpoints=%d\\n',
              as.numeric(t1 - t0, units='secs'), nrow(ui$predDf)))
}}
"""
    )
    t_start = time.monotonic()
    r = subprocess.run(
        ["Rscript", str(drive_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    elapsed = time.monotonic() - t_start
    ok = r.returncode == 0 and "COMPILE_OK" in r.stdout
    log = r.stdout + ("\n--- stderr ---\n" + r.stderr[-1000:] if r.stderr else "")
    return ok, elapsed, log


def _run_dataset(cfg: dict[str, object], work_dir: Path) -> BenchmarkResult:
    import numpy as np

    result = BenchmarkResult(dataset=str(cfg["id"]))
    print(f"[sweep] {result.dataset}: loading ...", flush=True)
    try:
        df = _load_dataset(cfg, work_dir)
    except Exception as e:
        result.status = "fail"
        result.note = f"load failed: {e}"
        return result

    result.loaded_rows = len(df)
    result.loaded_subjects = int(df["NMID"].nunique())
    cov_names = [c for c in cfg["covariates"] if c in df.columns]  # type: ignore[union-attr]
    if not cov_names:
        result.status = "skip"
        result.note = "no configured covariates present in data"
        return result
    result.covariates_tested = cov_names

    # Induce missingness on the first listed covariate.
    primary = cov_names[0]
    rng = np.random.default_rng(31)
    subjects = sorted(df["NMID"].unique().tolist())
    n_drop = max(1, int(len(subjects) * float(cfg.get("drop_fraction", 0.2))))
    drop_ids = rng.choice(subjects, size=n_drop, replace=False)
    df.loc[df["NMID"].isin(drop_ids), primary] = float("nan")
    result.induced_missing_subjects = n_drop

    try:
        t_emit = time.monotonic()
        covs = summarize_covariates(
            df,
            cov_names,
            transforms=cfg.get("transforms", {}),  # type: ignore[arg-type]
        )
        # Filter out zero-variance covariates that slip past summarize
        # (happens for constants like SEX=1 when sampled to a single group).
        covs = [c for c in covs if c.sigma_init >= 1e-4]
        if not covs:
            result.status = "skip"
            result.note = "all covariates degenerate after summarize"
            return result
        result.tv_covariates = [c.name for c in covs if c.time_varying]
        result.binary_covariates = [c.name for c in covs if c.transform == "binary"]
        augmented = prepare_frem_data(df, covs)
        spec = DSLSpec(
            model_id=f"{result.dataset}_frem",
            absorption=FirstOrder(ka=1.0),
            distribution=OneCmt(V=30.0),
            elimination=LinearElim(CL=5.0),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Combined(sigma_prop=0.1, sigma_add=0.1),
        )
        model_code = emit_nlmixr2_frem(spec, covs)
        result.emit_seconds = time.monotonic() - t_emit
        result.emitted_model_bytes = len(model_code)
    except Exception as e:
        result.status = "fail"
        result.note = f"emitter pipeline raised: {type(e).__name__}: {e}"
        return result

    print(
        f"[sweep] {result.dataset}: emit {result.emit_seconds:.2f}s, "
        f"{len(covs)} covs ({len(result.binary_covariates)} binary, "
        f"{len(result.tv_covariates)} TV); running nlmixr2 compile ...",
        flush=True,
    )
    ok, elapsed, log = _compile_check(work_dir, model_code, augmented)
    result.compile_seconds = elapsed
    if ok:
        result.status = "pass"
        result.note = log.strip().splitlines()[-1] if log else ""
    else:
        result.status = "fail"
        result.note = f"compile failed: {log[-1500:]}"
    return result


def _format_markdown(results: list[BenchmarkResult]) -> str:
    lines: list[str] = []
    lines.append("# FREM benchmark sweep results")
    lines.append("")
    lines.append(
        "Exercises the FREM emitter pipeline (`summarize_covariates` → "
        "`prepare_frem_data` → `emit_nlmixr2_frem` → `nlmixr2(fn)` compile) "
        "on every nlmixr2data-backed benchmark dataset with covariate "
        "coverage. Generated by `scripts/benchmark_frem_sweep.py`."
    )
    lines.append("")
    lines.append(
        "| Dataset | Subjects | Rows | Covariates (binary / TV) | "
        "Missing induced | Emit (s) | Compile (s) | Status |"
    )
    lines.append(
        "|---------|---------:|-----:|---------------------------|"
        "----------------:|---------:|------------:|:------:|"
    )
    for r in results:
        covs_desc = ", ".join(r.covariates_tested) or "—"
        bintv = []
        if r.binary_covariates:
            bintv.append(f"bin={','.join(r.binary_covariates)}")
        if r.tv_covariates:
            bintv.append(f"tv={','.join(r.tv_covariates)}")
        covs_label = f"{covs_desc}" + (f" ({'; '.join(bintv)})" if bintv else "")
        emit_s = f"{r.emit_seconds:.2f}" if r.emit_seconds is not None else "—"
        compile_s = f"{r.compile_seconds:.2f}" if r.compile_seconds is not None else "—"
        status_sym = {"pass": "✓", "fail": "✗", "skip": "·", "pending": "?"}[r.status]
        lines.append(
            f"| `{r.dataset}` | {r.loaded_subjects} | {r.loaded_rows} | "
            f"{covs_label} | {r.induced_missing_subjects} | {emit_s} | "
            f"{compile_s} | {status_sym} |"
        )
    lines.append("")
    # Surface failures / skips with notes.
    for r in results:
        if r.status in ("fail", "skip") and r.note:
            lines.append(f"### {r.dataset} ({r.status})")
            lines.append("")
            lines.append(f"```\n{r.note[:1500]}\n```")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the markdown summary to this file (in addition to stdout).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for generated R scripts, CSVs, and compile logs. "
        "Defaults to a temporary directory that is cleaned up on exit.",
    )
    args = parser.parse_args()

    ok, reason = _check_prereqs()
    if not ok:
        print(f"[sweep] prerequisite check failed: {reason}", file=sys.stderr)
        return 2

    import tempfile

    work_dir_ctx: object
    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        work_dir = args.work_dir
        work_dir_ctx = None
    else:
        work_dir_ctx = tempfile.TemporaryDirectory(prefix="frem_sweep_")
        work_dir = Path(work_dir_ctx.name)  # type: ignore[attr-defined]

    try:
        results: list[BenchmarkResult] = []
        for cfg in DATASETS:
            subdir = work_dir / str(cfg["id"])
            subdir.mkdir(parents=True, exist_ok=True)
            results.append(_run_dataset(cfg, subdir))
    finally:
        if work_dir_ctx is not None:
            work_dir_ctx.cleanup()  # type: ignore[attr-defined]

    summary = _format_markdown(results)
    print()
    print(summary)
    if args.output is not None:
        args.output.write_text(summary)
        print(f"\n[sweep] wrote summary to {args.output}", file=sys.stderr)

    n_fail = sum(1 for r in results if r.status == "fail")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
