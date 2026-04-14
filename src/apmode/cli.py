# SPDX-License-Identifier: GPL-2.0-or-later
"""APMODE CLI entry point (Typer + Rich, ARCHITECTURE.md §2.3).

Commands:
  apmode run <dataset> --lane <lane> --seed <seed> --policy <path>
  apmode validate <bundle-dir>
  apmode inspect <bundle-dir>
  apmode version
"""

from __future__ import annotations

import asyncio
import enum
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


class Lane(enum.StrEnum):
    """Operating lanes per PRD §3."""

    submission = "submission"
    discovery = "discovery"
    optimization = "optimization"


app = typer.Typer(
    name="apmode",
    help="Adaptive Pharmacokinetic Model Discovery Engine",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        from apmode import __version__

        console.print(f"[bold]apmode[/bold] {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Print version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """[bold]APMODE[/bold] — Adaptive Pharmacokinetic Model Discovery Engine.

    Composes classical NLME, automated structural search, hybrid NODE,
    and agentic LLM backends into a governed PK model discovery workflow.
    """


@app.command()
def version() -> None:
    """Print APMODE version."""
    from apmode import __version__

    console.print(f"[bold]apmode[/bold] {__version__}")


@app.command()
def run(
    dataset: Annotated[
        Path,
        typer.Argument(
            help="Path to NONMEM-style CSV (ID, TIME, DV, AMT, EVID, MDV columns).",
            show_default=False,
        ),
    ],
    lane: Annotated[
        Lane,
        typer.Option(
            help=(
                "Operating lane. "
                "[dim]submission[/dim]=regulatory, "
                "[dim]discovery[/dim]=exploratory, "
                "[dim]optimization[/dim]=LORO-CV."
            ),
        ),
    ] = Lane.submission,
    seed: Annotated[
        int,
        typer.Option(help="Root random seed for reproducibility."),
    ] = 42,
    policy: Annotated[
        Path | None,
        typer.Option(
            help="Gate policy JSON file. Falls back to policies/<lane>.json.",
            show_default="policies/<lane>.json",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(help="Backend timeout in seconds.", min=1),
    ] = 600,
    output: Annotated[
        Path,
        typer.Option(help="Bundle output directory."),
    ] = Path("runs"),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed pipeline logs."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-essential output."),
    ] = False,
) -> None:
    """Run the full APMODE pipeline on a PK dataset.

    Executes: ingest -> profile -> NCA -> split -> search -> governance gates -> bundle.
    """
    import logging

    from apmode.errors import BackendError, CrashError

    # --- Flag validation ---
    if verbose and quiet:
        err_console.print("[red bold]Error:[/] --verbose and --quiet are mutually exclusive.")
        raise typer.Exit(code=1)

    # --- Input validation ---
    if not dataset.is_file():
        err_console.print(f"[red bold]Error:[/] dataset not found: {escape(str(dataset))}")
        raise typer.Exit(code=1)

    if policy is not None and not policy.is_file():
        err_console.print(f"[red bold]Error:[/] policy file not found: {escape(str(policy))}")
        raise typer.Exit(code=1)

    # --- Logging setup ---
    from apmode.logging import configure_logging

    log_level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    configure_logging(json_output=False, level=log_level)

    # --- Imports (outside try so ImportError is not masked) ---
    from apmode.backends.nlmixr2_runner import Nlmixr2Runner
    from apmode.data.adapters import to_nlmixr2_format
    from apmode.data.ingest import ingest_nonmem_csv
    from apmode.orchestrator import Orchestrator, RunConfig

    # --- Ingestion ---
    if not quiet:
        console.print()
        console.rule("[bold]APMODE Pipeline[/]")

    try:
        with console.status("[bold cyan]Ingesting dataset...[/]", spinner="dots"):
            manifest, df = ingest_nonmem_csv(dataset)
    except Exception as e:
        err_console.print(f"[red bold]Ingestion failed:[/] {escape(str(e))}")
        err_console.print(
            "[dim]Check that the CSV has required columns: ID, TIME, DV, AMT, EVID, MDV[/]"
        )
        raise typer.Exit(code=1) from None

    if not quiet:
        ingest_table = Table(show_header=False, box=None, padding=(0, 2))
        ingest_table.add_column(style="dim")
        ingest_table.add_column(style="bold")
        ingest_table.add_row("Dataset", escape(str(dataset.name)))
        ingest_table.add_row("Subjects", str(manifest.n_subjects))
        ingest_table.add_row("Observations", str(manifest.n_observations))
        ingest_table.add_row("Doses", str(manifest.n_doses))
        ingest_table.add_row("Lane", lane.value)
        ingest_table.add_row("Seed", str(seed))
        console.print(Panel(ingest_table, title="[bold]Data Summary[/]", border_style="blue"))

    # --- Write nlmixr2-ready CSV ---
    nlmixr2_df = to_nlmixr2_format(df)
    data_csv = output / "_tmp_data.csv"
    data_csv.parent.mkdir(parents=True, exist_ok=True)
    nlmixr2_df.to_csv(data_csv, index=False)

    config = RunConfig(
        lane=lane.value,
        seed=seed,
        timeout_seconds=timeout,
        policy_path=policy,
    )

    runner = Nlmixr2Runner(work_dir=output / "_work", estimation=["saem"])
    orchestrator = Orchestrator(runner, output, config)

    # --- Pipeline execution ---
    try:
        with console.status(
            "[bold cyan]Running pipeline...[/]",
            spinner="dots",
        ):
            result = asyncio.run(orchestrator.run(manifest, df, data_csv))
    except BackendError as e:
        err_console.print(f"[red bold]Backend error:[/] {escape(str(e))}")
        if isinstance(e, CrashError) and e.stderr_tail:
            err_console.print(Panel(escape(e.stderr_tail), title="stderr", border_style="red"))
        raise typer.Exit(code=2) from None
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Pipeline interrupted by user.[/]")
        raise typer.Exit(code=130) from None
    except Exception as e:
        err_console.print(f"[red bold]Pipeline failed:[/] {escape(str(e))}")
        if verbose:
            console.print_exception()
        else:
            err_console.print("[dim]Re-run with --verbose for full traceback.[/]")
        raise typer.Exit(code=1) from None
    finally:
        # Clean up temporary data file
        if data_csv.exists():
            data_csv.unlink()

    # --- Results summary ---
    if quiet:
        # Machine-friendly one-liner
        console.print(result.bundle_dir)
        return

    console.print()
    results_table = Table(show_header=False, box=None, padding=(0, 2))
    results_table.add_column(style="dim")
    results_table.add_column()
    results_table.add_row("Run ID", f"[bold]{escape(result.run_id)}[/]")
    results_table.add_row("Bundle", escape(str(result.bundle_dir)))

    if result.search_outcome:
        n_total = len(result.search_outcome.results)
        n_conv = sum(1 for r in result.search_outcome.results if r.converged)
        results_table.add_row("Candidates", f"{n_total} total, {n_conv} converged")

    g1_pass = sum(1 for _, p in result.gate1_results if p)
    g1_total = len(result.gate1_results)
    g2_pass = sum(1 for _, p in result.gate2_results if p)
    g2_total = len(result.gate2_results)
    results_table.add_row("Gate 1", _pass_fraction(g1_pass, g1_total))
    results_table.add_row("Gate 2", _pass_fraction(g2_pass, g2_total))
    results_table.add_row(
        "Recommended",
        f"[bold green]{len(result.recommended)}[/]" if result.recommended else "[dim]0[/]",
    )

    if result.ranked:
        ranked_str = ", ".join(escape(r) for r in result.ranked[:5])
        if len(result.ranked) > 5:
            ranked_str += f" [dim](+{len(result.ranked) - 5} more)[/]"
        results_table.add_row("Ranked", ranked_str)

    console.print(Panel(results_table, title="[bold]Results[/]", border_style="green"))


def _pass_fraction(passed: int, total: int) -> str:
    """Format a pass/total fraction with color."""
    if total == 0:
        return "[dim]--[/]"
    color = "green" if passed == total else "yellow" if passed > 0 else "red"
    return f"[{color}]{passed}[/]/{total} passed"


# ---------------------------------------------------------------------------
# Shared helpers for bundle parsing
# ---------------------------------------------------------------------------


def _load_json(path: Path, label: str) -> dict[str, object] | None:
    """Load a JSON file, printing a warning on decode error."""
    try:
        data: dict[str, object] = json.loads(path.read_text())
        return data
    except json.JSONDecodeError as e:
        console.print(f"  [yellow]Warning:[/] corrupt {escape(label)}: {escape(str(e))}")
        return None


def _validate_jsonl(path: Path) -> str | None:
    """Validate that every non-blank line is valid JSON. Returns error or None."""
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                return f"line {i}: {e}"
    return None


def _validate_file(path: Path, filename: str) -> tuple[str, str]:
    """Validate a single bundle file. Returns (status_markup, note)."""
    if filename.endswith(".jsonl"):
        err = _validate_jsonl(path)
        if err:
            return "[red bold]BAD[/]", err
        return "[green]OK[/]", ""
    try:
        json.loads(path.read_text())
        return "[green]OK[/]", ""
    except json.JSONDecodeError as e:
        return "[red bold]BAD[/]", str(e)


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


@app.command()
def validate(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run bundle directory."),
    ],
) -> None:
    """Validate a reproducibility bundle for completeness.

    Checks required/optional JSON files and subdirectory contents.
    """
    if not bundle_dir.is_dir():
        kind = "not found" if not bundle_dir.exists() else "not a directory"
        err_console.print(f"[red bold]Error:[/] {kind}: {escape(str(bundle_dir))}")
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

    table = Table(title="Bundle Validation", box=None, padding=(0, 1))
    table.add_column("Status", width=8)
    table.add_column("File")
    table.add_column("Notes", style="dim")

    errors: list[str] = []

    for f in required_files:
        p = bundle_dir / f
        if not p.exists():
            errors.append(f)
            table.add_row("[red bold]MISS[/]", f, "required")
        else:
            status, note = _validate_file(p, f)
            if "BAD" in status:
                errors.append(f)
            table.add_row(status, f, note or "required")

    for f in optional_files:
        p = bundle_dir / f
        if not p.exists():
            table.add_row("[dim]--[/]", f, "[dim]optional[/]")
        else:
            status, note = _validate_file(p, f)
            if "BAD" in status:
                errors.append(f)
            table.add_row(status, f, note)

    for d in ["compiled_specs", "gate_decisions", "results"]:
        dp = bundle_dir / d
        if dp.is_dir():
            n = len(list(dp.glob("*.json"))) + len(list(dp.glob("*.R")))
            table.add_row("[green]OK[/]", f"{d}/", f"{n} files")
        else:
            table.add_row("[dim]--[/]", f"{d}/", "[dim]optional[/]")

    console.print()
    console.print(table)
    console.print()

    if errors:
        err_console.print(
            f"[red bold]Validation FAILED[/] — {len(errors)} issue(s): {', '.join(errors)}"
        )
        raise typer.Exit(code=1)

    console.print("[green bold]Bundle valid.[/]")


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


@app.command()
def inspect(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run bundle directory."),
    ],
) -> None:
    """Inspect a reproducibility bundle and print summary.

    Shows data manifest, evidence profile, search trajectory, and gate decisions.
    """
    if not bundle_dir.is_dir():
        kind = "not found" if not bundle_dir.exists() else "not a directory"
        err_console.print(f"[red bold]Error:[/] {kind}: {escape(str(bundle_dir))}")
        raise typer.Exit(code=1)

    console.print()
    console.rule(f"[bold]Bundle: {escape(bundle_dir.name)}[/]")

    sections_shown = 0

    # --- Data manifest ---
    dm_path = bundle_dir / "data_manifest.json"
    if dm_path.exists():
        dm = _load_json(dm_path, "data_manifest.json")
        if dm:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="dim")
            table.add_column(style="bold")
            table.add_row("Subjects", str(dm.get("n_subjects", "?")))
            table.add_row("Observations", str(dm.get("n_observations", "?")))
            table.add_row("Doses", str(dm.get("n_doses", "?")))
            console.print(Panel(table, title="[bold]Data[/]", border_style="blue"))
            sections_shown += 1

    # --- Evidence manifest ---
    em_path = bundle_dir / "evidence_manifest.json"
    if em_path.exists():
        em = _load_json(em_path, "evidence_manifest.json")
        if em:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="dim")
            table.add_column()
            table.add_row("Richness", escape(str(em.get("richness_category", "?"))))
            table.add_row("Nonlinear CL", _bool_badge(em.get("nonlinear_clearance_signature")))
            table.add_row("BLQ burden", escape(str(em.get("blq_burden", "?"))))
            if "absorption_coverage" in em:
                table.add_row("Absorption", escape(str(em["absorption_coverage"])))
            if "protocol_heterogeneity" in em:
                table.add_row("Protocol", escape(str(em["protocol_heterogeneity"])))
            console.print(Panel(table, title="[bold]Evidence Profile[/]", border_style="cyan"))
            sections_shown += 1

    # --- Search trajectory ---
    traj_path = bundle_dir / "search_trajectory.jsonl"
    if traj_path.exists():
        text = traj_path.read_text().strip()
        if text:
            lines = text.split("\n")
            n_total = len(lines)
            n_conv = 0
            parse_ok = True
            for i, line in enumerate(lines, 1):
                try:
                    if json.loads(line).get("converged"):
                        n_conv += 1
                except json.JSONDecodeError:
                    console.print(
                        f"  [yellow]Warning:[/] search_trajectory.jsonl corrupt at line {i}"
                    )
                    parse_ok = False
                    break
            if parse_ok:
                bar = _mini_bar(n_conv, n_total)
                console.print(
                    Panel(
                        f"Evaluated [bold]{n_total}[/] candidates, "
                        f"[bold green]{n_conv}[/] converged  {bar}",
                        title="[bold]Search[/]",
                        border_style="magenta",
                    )
                )
                sections_shown += 1

    # --- Gate decisions ---
    gd_dir = bundle_dir / "gate_decisions"
    if gd_dir.is_dir():
        table = Table(box=None, padding=(0, 2))
        table.add_column("Gate", style="bold")
        table.add_column("Passed")
        table.add_column("Total")
        table.add_column("Rate")

        for gate_name, pattern in [
            ("Gate 1", "gate1_*.json"),
            ("Gate 2", "gate2_*.json"),
        ]:
            files = list(gd_dir.glob(pattern))
            # Exclude gate2_5 files from gate2 count
            if pattern == "gate2_*.json":
                files = [f for f in files if not f.name.startswith("gate2_5")]
            if files:
                n_pass = 0
                for f in files:
                    gd = _load_json(f, f.name)
                    if gd and gd.get("passed"):
                        n_pass += 1
                table.add_row(
                    gate_name,
                    str(n_pass),
                    str(len(files)),
                    _pass_fraction(n_pass, len(files)),
                )

        # Gate 2.5 (credibility)
        g25_files = list(gd_dir.glob("gate2_5_*.json"))
        if g25_files:
            n_pass = 0
            for f in g25_files:
                gd = _load_json(f, f.name)
                if gd and gd.get("passed"):
                    n_pass += 1
            table.add_row(
                "Gate 2.5",
                str(n_pass),
                str(len(g25_files)),
                _pass_fraction(n_pass, len(g25_files)),
            )

        g3 = list(gd_dir.glob("gate3_*.json"))
        if g3:
            g3_data = _load_json(g3[0], "gate3")
            reason = escape(str(g3_data.get("summary_reason", "N/A"))) if g3_data else "?"
            table.add_row("Gate 3", "[dim]--[/]", "[dim]--[/]", reason)

        if table.row_count > 0:
            console.print(Panel(table, title="[bold]Governance[/]", border_style="yellow"))
            sections_shown += 1

    # --- Failed candidates ---
    fc_path = bundle_dir / "failed_candidates.jsonl"
    if fc_path.exists():
        text = fc_path.read_text().strip()
        if text:
            lines = text.split("\n")
            console.print(f"  [dim]Failed candidates:[/] {len(lines)}")
            sections_shown += 1

    # --- Ranking ---
    rank_path = bundle_dir / "ranking.json"
    if rank_path.exists():
        ranking = _load_json(rank_path, "ranking.json")
        if ranking:
            candidates: list[object] = ranking.get("ranked_candidates", [])  # type: ignore[assignment]
            if candidates:
                table = Table(box=None, padding=(0, 1))
                table.add_column("#", style="dim", width=3)
                table.add_column("Candidate", style="bold")
                table.add_column("BIC", justify="right")
                table.add_column("AIC", justify="right")
                table.add_column("Params", justify="right")
                table.add_column("Backend", style="dim")
                for c_raw in candidates[:10]:
                    c: dict[str, object] = c_raw  # type: ignore[assignment]
                    rank_str = str(c.get("rank", "?"))
                    bic_val = c.get("bic")
                    aic_val = c.get("aic")
                    bic = f"{bic_val:.1f}" if isinstance(bic_val, float | int) else "--"
                    aic = f"{aic_val:.1f}" if isinstance(aic_val, float | int) else "--"
                    table.add_row(
                        rank_str,
                        escape(str(c.get("candidate_id", "?"))),
                        bic,
                        aic,
                        str(c.get("n_params", "?")),
                        escape(str(c.get("backend", "?"))),
                    )
                if len(candidates) > 10:
                    table.add_row("", f"[dim]... +{len(candidates) - 10} more[/]", "", "", "", "")
                console.print(Panel(table, title="[bold]Ranking[/]", border_style="green"))
                sections_shown += 1

    if sections_shown == 0:
        console.print("[dim]Bundle is empty or contains no recognized artifacts.[/]")

    console.print()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _bool_badge(value: object) -> str:
    """Format a boolean as a colored badge."""
    if value is True:
        return "[bold yellow]yes[/]"
    if value is False:
        return "[green]no[/]"
    return "[dim]?[/]"


def _mini_bar(passed: int, total: int, width: int = 20) -> str:
    """Render a tiny filled/empty bar."""
    if total == 0:
        return ""
    filled = min(width, round(passed / total * width))
    return f"[green]{'|' * filled}[/][dim]{'|' * (width - filled)}[/]"


# ---------------------------------------------------------------------------
# Dataset discovery and download
# ---------------------------------------------------------------------------


@app.command()
def datasets(
    fetch: Annotated[
        str | None,
        typer.Argument(help="Dataset name to download (omit to list all)."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for downloaded CSVs."),
    ] = Path("datasets"),
    route: Annotated[
        str | None,
        typer.Option(help="Filter by route: oral, iv_bolus, iv_infusion."),
    ] = None,
    elimination: Annotated[
        str | None,
        typer.Option(help="Filter by elimination: linear, michaelis_menten."),
    ] = None,
) -> None:
    """Discover and download public PK datasets.

    List available datasets:   apmode datasets
    Download one:              apmode datasets theo_sd
    Download with filter:      apmode datasets --route oral -o ./data
    """
    from apmode.data.datasets import (
        DATASET_REGISTRY,
        fetch_dataset,
        list_datasets,
    )

    if fetch is not None:
        # Download a specific dataset
        if fetch not in DATASET_REGISTRY:
            available = ", ".join(sorted(DATASET_REGISTRY.keys()))
            err_console.print(
                f"[red bold]Error:[/] Unknown dataset '{escape(fetch)}'. Available: {available}"
            )
            raise typer.Exit(code=1)

        with console.status(f"[bold cyan]Fetching {fetch}...[/]", spinner="dots"):
            try:
                path = fetch_dataset(fetch, output)
            except RuntimeError as e:
                err_console.print(f"[red bold]Error:[/] {escape(str(e))}")
                raise typer.Exit(code=1) from e

        info = DATASET_REGISTRY[fetch]
        console.print(f"\n[green bold]Downloaded:[/] {path}")
        console.print(f"  {info.n_subjects} subjects, {info.n_rows} rows")
        console.print(f"  Route: {info.route}, Elimination: {info.elimination}")
        console.print(f"\n  Run: [bold]apmode run {path} --lane discovery[/]")
        return

    # List available datasets
    results = list_datasets(route=route, elimination=elimination)
    if not results:
        console.print("[yellow]No datasets match the given filters.[/]")
        return

    table = Table(title="Public PK Datasets", show_lines=False)
    table.add_column("Name", style="bold cyan", no_wrap=True)
    table.add_column("Subj", justify="right")
    table.add_column("Route", style="dim")
    table.add_column("Elimination")
    table.add_column("Cmt", justify="center")
    table.add_column("Covariates", style="dim")
    table.add_column("Description", max_width=45)

    for info in results:
        covs = ", ".join(info.covariates) if info.covariates else "-"
        desc = info.description[:45] + "..." if len(info.description) > 45 else info.description
        table.add_row(
            info.name,
            str(info.n_subjects),
            info.route,
            info.elimination,
            str(info.compartments),
            covs,
            desc,
        )

    console.print()
    console.print(table)
    console.print("\n[dim]Download:[/] [bold]apmode datasets <name> -o ./data[/]")
    console.print("[dim]Filter:[/]   [bold]apmode datasets --route oral --elimination linear[/]")
