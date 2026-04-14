# SPDX-License-Identifier: GPL-2.0-or-later
"""APMODE CLI entry point (Typer + Rich, ARCHITECTURE.md §2.3).

Commands:
  apmode run <dataset> --lane <lane> --seed <seed> --policy <path>
  apmode validate <bundle-dir>
  apmode inspect <bundle-dir>
  apmode datasets [name] [-o dir] [--route] [--elimination]
  apmode explore <dataset-or-name> [--lane] [--non-interactive]
  apmode diff <bundle-a> <bundle-b>
  apmode log <bundle-dir> [--gate] [--failed]
  apmode version

Exit codes:
  0  Success
  1  Input/validation error (bad CSV, missing columns, invalid config)
  2  Backend error (R crash, estimation failure)
  130  User interrupt (Ctrl+C)
"""

from __future__ import annotations

import asyncio
import enum
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

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


_COPYRIGHT = "(C) 2026 Biostochastics. For Research Use Only."
_GITHUB = "https://github.com/biostochastics/APMODE"


def _version_callback(value: bool) -> None:
    if value:
        from apmode import __version__

        console.print(f"[bold]apmode[/bold] {__version__}")
        console.print(f"[dim]{_COPYRIGHT}[/]")
        console.print(f"[dim]{_GITHUB}[/]")
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

    \b
    (C) 2026 Biostochastics. For Research Use Only.
    https://github.com/biostochastics/APMODE
    """


@app.command()
def version() -> None:
    """Print APMODE version."""
    from apmode import __version__

    console.print(f"[bold]apmode[/bold] {__version__}")
    console.print(f"[dim]{_COPYRIGHT}[/]")
    console.print(f"[dim]{_GITHUB}[/]")


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


def _load_json(path: Path, label: str) -> dict[str, Any] | None:
    """Load a JSON file, printing a warning on decode error."""
    try:
        data: dict[str, Any] = json.loads(path.read_text())
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
            candidates: list[Any] = ranking.get("ranked_candidates", [])
            if candidates:
                table = Table(box=None, padding=(0, 1))
                table.add_column("#", style="dim", width=3)
                table.add_column("Candidate", style="bold")
                table.add_column("BIC", justify="right")
                table.add_column("AIC", justify="right")
                table.add_column("Params", justify="right")
                table.add_column("Backend", style="dim")
                for c_raw in candidates[:10]:
                    c: dict[str, Any] = c_raw
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


# ---------------------------------------------------------------------------
# explore — interactive dataset workflow
# ---------------------------------------------------------------------------


@app.command()
def explore(
    dataset: Annotated[
        str,
        typer.Argument(
            help=("Dataset name from registry (e.g. 'theo_sd') or path to a local CSV file."),
        ),
    ],
    lane: Annotated[
        Lane,
        typer.Option(help="Operating lane for search space preview."),
    ] = Lane.discovery,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", "-y", help="Skip prompts, run full pipeline."),
    ] = False,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory."),
    ] = Path("runs"),
    seed: Annotated[
        int,
        typer.Option(help="Random seed."),
    ] = 42,
) -> None:
    """Interactive exploration of a PK dataset.

    Walks through the APMODE pipeline step-by-step:
    fetch -> ingest -> profile -> NCA -> search space -> (optional) run.

    \b
    Examples:
      apmode explore theo_sd              # interactive wizard
      apmode explore theo_sd -y           # non-interactive, full run
      apmode explore ./mydata.csv         # local CSV file
    """
    from apmode.data.datasets import DATASET_REGISTRY, fetch_dataset
    from apmode.data.ingest import ingest_nonmem_csv
    from apmode.data.initial_estimates import NCAEstimator
    from apmode.data.profiler import profile_data
    from apmode.search.candidates import SearchSpace, generate_root_candidates

    console.print()
    console.rule("[bold]APMODE Explorer[/]")

    # --- Step 1: Resolve dataset ---
    csv_path = Path(dataset)
    if csv_path.is_file():
        console.print(f"\n  Using local file: [bold]{escape(str(csv_path))}[/]")
    elif dataset in DATASET_REGISTRY:
        info = DATASET_REGISTRY[dataset]
        cache_dir = output / ".dataset_cache"
        cached = cache_dir / f"{dataset}.csv"
        if cached.exists():
            console.print(f"\n  Using cached: [bold]{dataset}[/] ({info.n_subjects} subjects)")
        else:
            with console.status(f"[cyan]Fetching {dataset} from nlmixr2data...[/]"):
                try:
                    fetch_dataset(dataset, cache_dir)
                except RuntimeError as e:
                    err_console.print(f"[red bold]Fetch failed:[/] {escape(str(e))}")
                    raise typer.Exit(code=1) from None
            console.print(f"\n  Fetched: [bold]{dataset}[/] — {info.description[:60]}")
        csv_path = cached
    else:
        err_console.print(
            f"[red bold]Error:[/] '{escape(dataset)}' is not a file or known dataset."
        )
        err_console.print("[dim]Run 'apmode datasets' to see available datasets.[/]")
        raise typer.Exit(code=1)

    # --- Step 2: Ingest ---
    console.print()
    with console.status("[cyan]Ingesting...[/]"):
        try:
            manifest, df = ingest_nonmem_csv(csv_path)
        except Exception as e:
            err_console.print(f"[red bold]Ingestion failed:[/] {escape(str(e))}")
            raise typer.Exit(code=1) from None

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold")
    t.add_row("Subjects", str(manifest.n_subjects))
    t.add_row("Observations", str(manifest.n_observations))
    t.add_row("Doses", str(manifest.n_doses))
    console.print(Panel(t, title="[bold]Data Summary[/]", border_style="blue"))

    # --- Step 3: Profile ---
    with console.status("[cyan]Profiling data...[/]"):
        evidence = profile_data(df, manifest)

    _print_evidence_panel(evidence)

    # --- Step 4: NCA estimates ---
    with console.status("[cyan]Computing NCA estimates...[/]"):
        nca = NCAEstimator(df, manifest)
        estimates = nca.estimate_per_subject()

    t = Table(show_header=True, box=None, padding=(0, 2))
    t.add_column("Parameter", style="bold")
    t.add_column("Estimate", justify="right")
    for name, val in sorted(estimates.items()):
        t.add_row(name, f"{val:.4g}")
    console.print(Panel(t, title="[bold]NCA Initial Estimates[/]", border_style="cyan"))

    # --- Step 5: Search space preview ---
    space = SearchSpace.from_manifest(evidence)
    candidates = generate_root_candidates(space, base_params=estimates)

    _print_search_space_panel(space, candidates, lane.value)

    # --- Step 6: Prompt for full run ---
    if non_interactive:
        console.print("\n[bold yellow]--non-interactive:[/] launching full pipeline...")
        _launch_run(csv_path, lane, seed, output)
        return

    console.print()
    proceed = typer.confirm("Launch full pipeline?", default=False)
    if proceed:
        _launch_run(csv_path, lane, seed, output)
    else:
        console.print("\n[dim]Done. Run 'apmode run' manually when ready.[/]")


def _print_evidence_panel(evidence: object) -> None:
    """Print the evidence manifest as a Rich panel."""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", min_width=28)
    t.add_column(style="bold")
    for field_name in [
        "richness_category",
        "route_certainty",
        "absorption_complexity",
        "nonlinear_clearance_signature",
        "absorption_phase_coverage",
        "elimination_phase_coverage",
        "covariate_burden",
        "blq_burden",
        "protocol_heterogeneity",
    ]:
        val = getattr(evidence, field_name, None)
        if val is not None:
            display = str(val)
            if isinstance(val, bool):
                display = "[green]yes[/]" if val else "[dim]no[/]"
            elif isinstance(val, float):
                display = f"{val:.2f}"
            t.add_row(field_name.replace("_", " ").title(), display)
    console.print(Panel(t, title="[bold]Evidence Manifest[/]", border_style="magenta"))


def _print_search_space_panel(space: object, candidates: Sequence[object], lane: str) -> None:
    """Print search space and dispatch preview."""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold")
    t.add_row("Lane", lane)
    t.add_row("Root candidates", str(len(candidates)))

    # Dispatch preview
    n_classical = sum(1 for c in candidates if not getattr(c, "has_node_modules", lambda: False)())
    n_node = len(candidates) - n_classical
    dispatch_parts = []
    if n_classical:
        dispatch_parts.append(f"[cyan]{n_classical}[/] -> nlmixr2")
    if n_node:
        dispatch_parts.append(f"[yellow]{n_node}[/] -> jax_node")
    t.add_row("Dispatch", " + ".join(dispatch_parts) if dispatch_parts else "[dim]none[/]")

    # Show structural dimensions
    cmt_set: set[int] = getattr(space, "structural_cmt", set())
    abs_set: set[str] = getattr(space, "absorption_types", set())
    elim_set: set[str] = getattr(space, "elimination_types", set())
    if cmt_set:
        t.add_row("Compartments", ", ".join(str(c) for c in sorted(cmt_set)))
    if abs_set:
        t.add_row("Absorption types", ", ".join(sorted(abs_set)))
    if elim_set:
        t.add_row("Elimination types", ", ".join(sorted(elim_set)))

    console.print(Panel(t, title="[bold]Search Space[/]", border_style="yellow"))


def _launch_run(csv_path: Path, lane: Lane, seed: int, output: Path) -> None:
    """Delegate to the run command by invoking it directly."""
    import contextlib

    with contextlib.suppress(SystemExit):
        run(
            dataset=csv_path,
            lane=lane,
            seed=seed,
            output=output,
            timeout=600,
            verbose=False,
            quiet=False,
        )


# ---------------------------------------------------------------------------
# diff — compare two reproducibility bundles
# ---------------------------------------------------------------------------


@app.command()
def diff(
    bundle_a: Annotated[
        Path,
        typer.Argument(help="First bundle directory."),
    ],
    bundle_b: Annotated[
        Path,
        typer.Argument(help="Second bundle directory."),
    ],
) -> None:
    """Compare two reproducibility bundles side-by-side.

    Shows differences in evidence manifest, search outcomes, gate
    decisions, and rankings between two APMODE runs.
    """
    for p, label in [(bundle_a, "A"), (bundle_b, "B")]:
        if not p.is_dir():
            err_console.print(f"[red bold]Error:[/] Bundle {label} not found: {escape(str(p))}")
            raise typer.Exit(code=1)

    console.print()
    console.rule("[bold]Bundle Diff[/]")

    # Compare evidence manifests
    em_a = _load_json(bundle_a / "evidence_manifest.json", "evidence_manifest (A)")
    em_b = _load_json(bundle_b / "evidence_manifest.json", "evidence_manifest (B)")

    if em_a and em_b:
        t = Table(title="Evidence Manifest", show_lines=False)
        t.add_column("Field", style="bold")
        t.add_column(escape(str(bundle_a.name)), style="cyan")
        t.add_column(escape(str(bundle_b.name)), style="yellow")
        t.add_column("Match", justify="center")

        for key in sorted(set(em_a.keys()) | set(em_b.keys())):
            if key.startswith("data_sha"):
                continue
            va, vb = em_a.get(key, "-"), em_b.get(key, "-")
            match = "[green]=[/]" if va == vb else "[red]x[/]"
            t.add_row(key, str(va), str(vb), match)
        console.print(t)

    # Compare rankings
    rank_a = _load_json(bundle_a / "ranking.json", "ranking (A)")
    rank_b = _load_json(bundle_b / "ranking.json", "ranking (B)")

    if rank_a and rank_b:
        console.print()
        t = Table(title="Ranking Comparison", show_lines=False)
        t.add_column("#", style="dim", width=3)
        t.add_column(escape(str(bundle_a.name)), style="cyan")
        t.add_column(escape(str(bundle_b.name)), style="yellow")

        cands_a = rank_a.get("ranked_candidates", [])
        cands_b = rank_b.get("ranked_candidates", [])
        max_len = max(len(cands_a), len(cands_b), 1)

        for i in range(min(max_len, 10)):
            ca = cands_a[i].get("model_id", "?") if i < len(cands_a) else "-"
            cb = cands_b[i].get("model_id", "?") if i < len(cands_b) else "-"
            t.add_row(str(i + 1), ca, cb)
        console.print(t)

    # Compare gate pass rates
    for gd_name in ["gate_decisions"]:
        gd_a = bundle_a / gd_name
        gd_b = bundle_b / gd_name
        if gd_a.is_dir() and gd_b.is_dir():
            console.print()
            t = Table(title="Gate Pass Rates", show_lines=False)
            t.add_column("Gate", style="bold")
            t.add_column(escape(str(bundle_a.name)), justify="right", style="cyan")
            t.add_column(escape(str(bundle_b.name)), justify="right", style="yellow")

            for gate, pattern in [("Gate 1", "gate1_*.json"), ("Gate 2", "gate2_*.json")]:
                fa = [f for f in gd_a.glob(pattern) if "gate2_5" not in f.name]
                fb = [f for f in gd_b.glob(pattern) if "gate2_5" not in f.name]
                pa = sum(1 for f in fa if (_load_json(f, "") or {}).get("passed"))
                pb = sum(1 for f in fb if (_load_json(f, "") or {}).get("passed"))
                t.add_row(gate, f"{pa}/{len(fa)}", f"{pb}/{len(fb)}")
            console.print(t)


# ---------------------------------------------------------------------------
# log — query bundle JSONL logs
# ---------------------------------------------------------------------------


@app.command(name="log")
def log_cmd(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Bundle directory to query."),
    ],
    gate: Annotated[
        str | None,
        typer.Option(help="Filter to a specific gate (gate1, gate2, gate2_5, gate3)."),
    ] = None,
    failed: Annotated[
        bool,
        typer.Option("--failed", help="Show only failed candidates."),
    ] = False,
    top: Annotated[
        int,
        typer.Option("--top", "-n", help="Show top N ranked candidates with parameters.", min=1),
    ] = 0,
) -> None:
    """Query logs, gate decisions, and parameters from a bundle.

    \b
    Examples:
      apmode log ./runs/run_abc123                  # summary
      apmode log ./runs/run_abc123 --failed         # failed candidates
      apmode log ./runs/run_abc123 --gate gate1     # gate 1 details
      apmode log ./runs/run_abc123 --top 3          # top 3 with parameters
    """
    if not bundle_dir.is_dir():
        err_console.print(f"[red bold]Error:[/] not a directory: {escape(str(bundle_dir))}")
        raise typer.Exit(code=1)

    console.print()
    console.rule(f"[bold]Bundle Log: {escape(bundle_dir.name)}[/]")

    # --- Gate decision details ---
    if gate:
        _show_gate_details(bundle_dir, gate)
        return

    # --- Failed candidates ---
    if failed:
        fc_path = bundle_dir / "failed_candidates.jsonl"
        if not fc_path.exists():
            console.print("[dim]No failed candidates file found.[/]")
            return
        text = fc_path.read_text().strip()
        if not text:
            console.print("[green]No failed candidates.[/]")
            return

        t = Table(title="Failed Candidates", show_lines=False)
        t.add_column("Candidate", style="bold")
        t.add_column("Gate", style="red")
        t.add_column("Reason")

        for line in text.split("\n"):
            try:
                rec = json.loads(line)
                t.add_row(
                    rec.get("model_id", "?"),
                    rec.get("failed_gate", "?"),
                    rec.get("reason", "-")[:60],
                )
            except json.JSONDecodeError:
                continue
        console.print(t)
        return

    # --- Top N with parameters ---
    if top > 0:
        _show_top_candidates(bundle_dir, top)
        return

    # --- Default: overview ---
    _show_bundle_overview(bundle_dir)


def _show_gate_details(bundle_dir: Path, gate_name: str) -> None:
    """Show per-check details for a specific gate."""
    gd_dir = bundle_dir / "gate_decisions"
    if not gd_dir.is_dir():
        console.print("[dim]No gate_decisions directory found.[/]")
        return

    pattern = f"{gate_name}_*.json"
    files = sorted(gd_dir.glob(pattern))
    if not files:
        console.print(f"[dim]No {gate_name} decisions found.[/]")
        return

    for f in files:
        data = _load_json(f, f.name)
        if not data:
            continue

        passed = data.get("passed", False)
        status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        model_id = data.get("model_id", f.stem)

        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="dim", min_width=24)
        t.add_column()
        t.add_row("Status", status)

        checks = data.get("checks", data.get("check_results", {}))
        if isinstance(checks, dict):
            for check_name, result in checks.items():
                if isinstance(result, dict):
                    check_pass = result.get("passed", result.get("pass", False))
                    icon = "[green]pass[/]" if check_pass else "[red]FAIL[/]"
                    reason = result.get("reason", result.get("message", ""))
                    t.add_row(check_name, f"{icon}  {str(reason)[:50]}")
                else:
                    icon = "[green]pass[/]" if result else "[red]FAIL[/]"
                    t.add_row(check_name, icon)

        console.print(Panel(t, title=f"[bold]{escape(model_id)}[/]", border_style="yellow"))


def _show_top_candidates(bundle_dir: Path, n: int) -> None:
    """Show top-N ranked candidates with parameter estimates."""
    rank_path = bundle_dir / "ranking.json"
    if not rank_path.exists():
        console.print("[dim]No ranking.json found.[/]")
        return

    ranking = _load_json(rank_path, "ranking.json")
    if not ranking:
        return

    cands = ranking.get("ranked_candidates", [])
    if not cands:
        console.print("[dim]No ranked candidates.[/]")
        return

    for i, cand in enumerate(cands[:n], 1):
        model_id = cand.get("model_id", "?")
        bic = cand.get("bic")
        aic = cand.get("aic")

        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("Parameter", style="bold")
        t.add_column("Estimate", justify="right")
        t.add_column("Category", style="dim")

        params = cand.get("parameter_estimates", cand.get("parameters", {}))
        if isinstance(params, dict):
            for pname, pdata in sorted(params.items()):
                if isinstance(pdata, dict):
                    est = pdata.get("estimate", "?")
                    cat = pdata.get("category", "")
                    t.add_row(pname, f"{est:.4g}" if isinstance(est, float) else str(est), cat)
                else:
                    t.add_row(
                        pname, f"{pdata:.4g}" if isinstance(pdata, float) else str(pdata), ""
                    )

        subtitle = []
        if bic is not None:
            subtitle.append(f"BIC={bic:.1f}")
        if aic is not None:
            subtitle.append(f"AIC={aic:.1f}")
        sub = f" ({', '.join(subtitle)})" if subtitle else ""

        console.print(
            Panel(t, title=f"[bold]#{i} {escape(model_id)}{sub}[/]", border_style="green")
        )


def _show_bundle_overview(bundle_dir: Path) -> None:
    """Quick overview of a bundle's contents."""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold")

    # Evidence manifest
    em = _load_json(bundle_dir / "evidence_manifest.json", "")
    if em:
        t.add_row("Richness", em.get("richness_category", "?"))
        t.add_row("Route", em.get("route_certainty", "?"))
        t.add_row("Nonlinear CL", str(em.get("nonlinear_clearance_signature", "?")))

    # Search trajectory
    st_path = bundle_dir / "search_trajectory.jsonl"
    if st_path.exists():
        lines = st_path.read_text().strip().split("\n")
        n_conv = sum(1 for ln in lines if json.loads(ln).get("converged", False))
        t.add_row("Candidates", f"{len(lines)} total, {n_conv} converged")

    # Gate decisions
    gd_dir = bundle_dir / "gate_decisions"
    if gd_dir.is_dir():
        for gate, pattern in [("Gate 1", "gate1_*.json"), ("Gate 2", "gate2_*.json")]:
            files = [f for f in gd_dir.glob(pattern) if "gate2_5" not in f.name]
            if files:
                passed = sum(1 for f in files if (_load_json(f, "") or {}).get("passed"))
                t.add_row(gate, _pass_fraction(passed, len(files)))

    # Ranking
    rank = _load_json(bundle_dir / "ranking.json", "")
    if rank:
        cands = rank.get("ranked_candidates", [])
        if cands:
            top3 = ", ".join(c.get("model_id", "?") for c in cands[:3])
            t.add_row("Top ranked", top3)

    console.print(Panel(t, title="[bold]Bundle Overview[/]", border_style="blue"))
    console.print("[dim]Use --failed, --gate, or --top for details.[/]")
