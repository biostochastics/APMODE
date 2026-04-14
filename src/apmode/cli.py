# SPDX-License-Identifier: GPL-2.0-or-later
"""APMODE CLI entry point (Typer, ARCHITECTURE.md §2.3).

Commands:
  apmode run <dataset> --lane <lane> --seed <seed> --policy <path>
  apmode validate <bundle-dir>
  apmode inspect <bundle-dir>
  apmode version
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(name="apmode", help="Adaptive Pharmacokinetic Model Discovery Engine")


@app.command()
def version() -> None:
    """Print APMODE version."""
    from apmode import __version__

    typer.echo(f"apmode {__version__}")


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="Path to NONMEM-style CSV file")],
    lane: Annotated[str, typer.Option(help="Operating lane")] = "submission",
    seed: Annotated[int, typer.Option(help="Root random seed")] = 42,
    policy: Annotated[Path | None, typer.Option(help="Gate policy JSON")] = None,
    timeout: Annotated[int, typer.Option(help="Backend timeout (seconds)")] = 600,
    output: Annotated[Path, typer.Option(help="Bundle output directory")] = Path("runs"),
) -> None:
    """Run the full APMODE pipeline on a PK dataset."""
    from apmode.backends.nlmixr2_runner import Nlmixr2Runner
    from apmode.data.adapters import to_nlmixr2_format
    from apmode.data.ingest import ingest_nonmem_csv
    from apmode.orchestrator import Orchestrator, RunConfig

    if not dataset.exists():
        typer.echo(f"Error: dataset not found: {dataset}", err=True)
        raise typer.Exit(code=1)

    if lane not in {"submission", "discovery", "optimization"}:
        typer.echo(f"Error: invalid lane '{lane}'", err=True)
        raise typer.Exit(code=1)

    if policy is not None and not policy.is_file():
        typer.echo(f"Error: policy file not found: {policy}", err=True)
        raise typer.Exit(code=1)

    from apmode.logging import configure_logging

    configure_logging(json_output=False)

    typer.echo(f"Ingesting {dataset}...")
    manifest, df = ingest_nonmem_csv(dataset)
    typer.echo(
        f"  {manifest.n_subjects} subjects, "
        f"{manifest.n_observations} observations, "
        f"{manifest.n_doses} doses"
    )

    # Write nlmixr2-ready CSV
    nlmixr2_df = to_nlmixr2_format(df)
    data_csv = output / "_tmp_data.csv"
    data_csv.parent.mkdir(parents=True, exist_ok=True)
    nlmixr2_df.to_csv(data_csv, index=False)

    config = RunConfig(
        lane=lane,  # type: ignore[arg-type]  # validated above (line 50)
        seed=seed,
        timeout_seconds=timeout,
        policy_path=policy,
    )

    runner = Nlmixr2Runner(work_dir=output / "_work", estimation=["saem"])
    orchestrator = Orchestrator(runner, output, config)

    typer.echo(f"Running pipeline (lane={lane}, seed={seed})...")
    result = asyncio.run(orchestrator.run(manifest, df, data_csv))

    typer.echo(f"\nRun ID: {result.run_id}")
    typer.echo(f"Bundle: {result.bundle_dir}")
    if result.search_outcome:
        n_total = len(result.search_outcome.results)
        n_conv = sum(1 for r in result.search_outcome.results if r.converged)
        typer.echo(f"Candidates: {n_total} total, {n_conv} converged")
    typer.echo(f"Gate 1 passed: {sum(1 for _, p in result.gate1_results if p)}")
    typer.echo(f"Gate 2 passed: {sum(1 for _, p in result.gate2_results if p)}")
    typer.echo(f"Recommended: {len(result.recommended)}")
    if result.ranked:
        typer.echo(f"Ranked (best→worst): {', '.join(result.ranked[:5])}")


@app.command()
def validate(
    bundle_dir: Annotated[Path, typer.Argument(help="Run bundle directory")],
) -> None:
    """Validate a reproducibility bundle for completeness."""
    if not bundle_dir.is_dir():
        typer.echo(f"Error: not a directory: {bundle_dir}", err=True)
        raise typer.Exit(code=1)

    required_files = [
        "data_manifest.json",
        "seed_registry.json",
        "backend_versions.json",
    ]
    optional_files = [
        "evidence_manifest.json",
        "initial_estimates.json",
        "split_manifest.json",
        "policy_file.json",
        "search_trajectory.jsonl",
        "failed_candidates.jsonl",
        "candidate_lineage.json",
    ]

    errors: list[str] = []
    for f in required_files:
        p = bundle_dir / f
        if not p.exists():
            errors.append(f"MISSING (required): {f}")
        else:
            # Validate JSON
            try:
                json.loads(p.read_text())
                typer.echo(f"  OK: {f}")
            except json.JSONDecodeError as e:
                errors.append(f"INVALID JSON: {f}: {e}")

    for f in optional_files:
        p = bundle_dir / f
        if p.exists():
            typer.echo(f"  OK: {f}")
        else:
            typer.echo(f"  --: {f} (not present)")

    # Check subdirectories
    for d in ["compiled_specs", "gate_decisions", "results"]:
        dp = bundle_dir / d
        if dp.is_dir():
            n = len(list(dp.glob("*.json"))) + len(list(dp.glob("*.R")))
            typer.echo(f"  OK: {d}/ ({n} files)")
        else:
            typer.echo(f"  --: {d}/ (not present)")

    if errors:
        typer.echo("\nValidation FAILED:")
        for err_msg in errors:
            typer.echo(f"  {err_msg}", err=True)
        raise typer.Exit(code=1)

    typer.echo("\nBundle valid.")


@app.command()
def inspect(
    bundle_dir: Annotated[Path, typer.Argument(help="Run bundle directory")],
) -> None:
    """Inspect a reproducibility bundle and print summary."""
    if not bundle_dir.is_dir():
        typer.echo(f"Error: not a directory: {bundle_dir}", err=True)
        raise typer.Exit(code=1)

    # Data manifest
    dm_path = bundle_dir / "data_manifest.json"
    if dm_path.exists():
        dm = json.loads(dm_path.read_text())
        typer.echo(f"Data: {dm.get('n_subjects')} subjects, {dm.get('n_observations')} obs")

    # Evidence manifest
    em_path = bundle_dir / "evidence_manifest.json"
    if em_path.exists():
        em = json.loads(em_path.read_text())
        typer.echo(
            f"Evidence: richness={em.get('richness_category')}, "
            f"nonlinear_CL={em.get('nonlinear_clearance_signature')}, "
            f"BLQ={em.get('blq_burden')}"
        )

    # Search trajectory
    traj_path = bundle_dir / "search_trajectory.jsonl"
    if traj_path.exists():
        lines = traj_path.read_text().strip().split("\n")
        n_conv = sum(1 for line in lines if json.loads(line).get("converged"))
        typer.echo(f"Search: {len(lines)} candidates evaluated, {n_conv} converged")

    # Gate decisions
    gd_dir = bundle_dir / "gate_decisions"
    if gd_dir.is_dir():
        g1 = list(gd_dir.glob("gate1_*.json"))
        g2 = list(gd_dir.glob("gate2_*.json"))
        g3 = list(gd_dir.glob("gate3_*.json"))
        g1_pass = sum(1 for f in g1 if json.loads(f.read_text()).get("passed"))
        g2_pass = sum(1 for f in g2 if json.loads(f.read_text()).get("passed"))
        typer.echo(f"Gate 1: {g1_pass}/{len(g1)} passed")
        typer.echo(f"Gate 2: {g2_pass}/{len(g2)} passed")
        if g3:
            g3_data = json.loads(g3[0].read_text())
            typer.echo(f"Gate 3: {g3_data.get('summary_reason', 'N/A')}")

    # Failed candidates
    fc_path = bundle_dir / "failed_candidates.jsonl"
    if fc_path.exists():
        lines = fc_path.read_text().strip().split("\n")
        if lines and lines[0]:
            typer.echo(f"Failed: {len(lines)} candidates")
