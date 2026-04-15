#!/usr/bin/env python
"""Smoke test the refactored profiler against real PK datasets.

Run: uv run python scripts/profile_smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apmode.bundle.models import (  # noqa: E402
    ColumnMapping,
    CovariateMetadata,
    DataManifest,
    SignalId,
)
from apmode.data.profiler import profile_data  # noqa: E402


def _stub_manifest(df: pd.DataFrame) -> DataManifest:
    """Build a minimal DataManifest for smoke testing."""
    reserved = {
        "NMID",
        "TIME",
        "DV",
        "AMT",
        "EVID",
        "CMT",
        "MDV",
        "BLQ_FLAG",
        "DVID",
        "RATE",
        "DUR",
        "LLOQ",
        "STUDY_ID",
    }
    cov_specs = [
        CovariateMetadata(
            name=c,
            type="continuous" if pd.api.types.is_numeric_dtype(df[c]) else "categorical",
        )
        for c in df.columns
        if c not in reserved
    ]
    return DataManifest(
        data_sha256="0" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", amt="AMT", evid="EVID", cmt="CMT", mdv="MDV"
        ),
        n_subjects=int(df["NMID"].nunique()),
        n_observations=int((df["EVID"] == 0).sum()),
        n_doses=int((df["EVID"] == 1).sum()),
        has_multidose=bool(df.groupby("NMID")["EVID"].apply(lambda s: (s == 1).sum() >= 2).any()),
        covariates=cov_specs,
    )


def main() -> None:
    datasets = {
        "warfarin": ROOT / "benchmarks/data/warfarin.csv",
        "theo_sd": ROOT / "benchmarks/data/theo_sd.csv",
        "mavoglurant": ROOT / "benchmarks/data/mavoglurant.csv",
        "a1_1cmt_oral_linear": ROOT / "tests/fixtures/suite_a/a1_1cmt_oral_linear.csv",
        "a4_1cmt_oral_mm": ROOT / "tests/fixtures/suite_a/a4_1cmt_oral_mm.csv",
        "a2_2cmt_iv_parallel_mm": ROOT / "tests/fixtures/suite_a/a2_2cmt_iv_parallel_mm.csv",
    }
    headline = (
        "dataset",
        "richness",
        "absorption",
        "compartmentality",
        "nlc_strength",
        "multi_dose",
        "term_R2",
        "curv_ratio",
        "lambdaz_ana",
        "auc_extrap",
        "elim_cov",
        "abs_cov",
    )
    rows: list[tuple] = [headline]
    for name, path in datasets.items():
        if not path.exists():
            print(f"SKIP {name}: file missing at {path}")
            continue
        df = pd.read_csv(path)
        manifest = _stub_manifest(df)
        ev = profile_data(df, manifest)
        term = ev.nonlinear_clearance_signals.get(SignalId.TERMINAL_MONOEXP)
        curv = ev.nonlinear_clearance_signals.get(SignalId.CURVATURE_RATIO)
        term_r2 = (
            f"{term.observed_value:.3f}"
            if term is not None and term.observed_value is not None
            else "-"
        )
        curv_ratio = (
            f"{curv.observed_value:.2f}"
            if curv is not None and curv.observed_value is not None
            else "-"
        )
        rows.append(
            (
                name,
                ev.richness_category,
                ev.absorption_complexity,
                ev.compartmentality,
                ev.nonlinear_clearance_evidence_strength,
                ev.multi_dose_detected,
                term_r2,
                curv_ratio,
                f"{ev.lambda_z_analyzable_fraction:.2f}"
                if ev.lambda_z_analyzable_fraction is not None
                else "-",
                f"{ev.auc_extrap_fraction_median:.3f}"
                if ev.auc_extrap_fraction_median is not None
                else "-",
                ev.elimination_phase_coverage,
                ev.absorption_phase_coverage,
            )
        )
        # Also dump full manifest for debugging.
        out_path = ROOT / f"benchmarks/runs/profile_smoke_{name}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(ev.model_dump(mode="json"), indent=2, default=str))

    # Pretty-print as a table.
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(headline))]
    sep = "  "
    for r in rows:
        print(sep.join(str(r[i]).ljust(widths[i]) for i in range(len(headline))))


if __name__ == "__main__":
    main()
