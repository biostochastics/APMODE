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
  apmode trace <bundle-dir> [--iteration N] [--cost] [--json]
  apmode lineage <bundle-dir> <candidate-id> [--spec] [--gate]
  apmode graph <bundle-dir> [--format tree|dot|mermaid|json] [--converged]

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
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apmode.backends.agentic_runner import AgenticRunner
    from apmode.backends.protocol import BackendRunner

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def _is_real_number(value: object) -> bool:
    """True iff value is numeric and not a bool (isinstance(True, int) == True)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Lane(enum.StrEnum):
    """Operating lanes per PRD §3."""

    submission = "submission"
    discovery = "discovery"
    optimization = "optimization"


_COPYRIGHT = "(C) 2026 Biostochastics. For Research Use Only."
_CITATION = "Cite: Kornilov, S.A. (2026). APMODE: Adaptive Pharmacokinetic Model Discovery Engine. https://github.com/biostochastics/apmode"


def _get_version() -> str:
    """Return the package version string (lazy import)."""
    from apmode import __version__

    return __version__


app = typer.Typer(
    name="apmode",
    help="Adaptive Pharmacokinetic Model Discovery Engine",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=(f"[dim]Version: {_get_version()}[/]\n[dim]{_COPYRIGHT}[/]\n[dim]{_CITATION}[/]"),
)

# Default model per provider for the agentic backend
_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen3:4b",
    "openrouter": "anthropic/claude-sonnet-4-20250514",
}

# Env var names to check per provider for auto-detection
_PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "ollama": [],  # no key needed
    "openrouter": ["OPENROUTER_API_KEY"],
}


def _try_build_agentic_runner(
    inner_runner: BackendRunner,
    provider: str,
    model_name: str | None,
    max_iterations: int,
    lane: str,
    trace_dir: Path,
    quiet: bool,
) -> AgenticRunner | None:
    """Try to build an AgenticRunner. Returns None if provider unavailable."""
    import os

    from apmode.backends.agentic_runner import AgenticConfig, AgenticRunner
    from apmode.backends.llm_client import LLMConfig
    from apmode.backends.llm_providers import available_providers, create_llm_client

    # Check if provider is valid
    known = available_providers()
    if provider not in known:
        if not quiet:
            err_console.print(
                f"[yellow]Warning:[/] unknown LLM provider '{provider}'. "
                f"Available: {', '.join(known)}. Agentic backend disabled."
            )
        return None

    # Check for API key (ollama needs none)
    env_keys = _PROVIDER_ENV_KEYS.get(provider, [])
    if env_keys and not any(os.environ.get(k) for k in env_keys):
        if not quiet:
            err_console.print(
                f"[yellow]Warning:[/] no API key found for provider '{provider}' "
                f"(checked: {', '.join(env_keys)}). Agentic backend disabled."
            )
        return None

    # Resolve model name
    resolved_model = model_name or _DEFAULT_MODELS.get(provider, "")
    if not resolved_model:
        if not quiet:
            err_console.print(
                f"[yellow]Warning:[/] no default model for provider '{provider}'. "
                "Use --model to specify one. Agentic backend disabled."
            )
        return None

    try:
        llm_config = LLMConfig(model=resolved_model, provider=provider)
        llm_client = create_llm_client(llm_config)
    except Exception as e:
        if not quiet:
            err_console.print(
                f"[yellow]Warning:[/] failed to create LLM client: {e}. Agentic backend disabled."
            )
        return None

    agentic_config = AgenticConfig(
        max_iterations=max_iterations,
        lane=lane,
    )

    if not quiet:
        console.print(
            f"  [bold green]Agentic LLM[/] enabled: "
            f"provider={provider}, model={resolved_model}, "
            f"max_iterations={max_iterations}"
        )

    return AgenticRunner(
        inner_runner=inner_runner,
        llm_client=llm_client,
        config=agentic_config,
        trace_dir=trace_dir,
    )


def _version_callback(value: bool) -> None:
    if value:
        from apmode import __version__

        console.print(f"[bold]apmode[/bold] {__version__}")
        console.print(f"[dim]{_COPYRIGHT}[/]")
        console.print(f"[dim]{_CITATION}[/]")
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
    ] = 753849,
    policy: Annotated[
        Path | None,
        typer.Option(
            help="Gate policy JSON file. Falls back to policies/<lane>.json.",
            show_default="policies/<lane>.json",
        ),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(
            help=(
                "Per-candidate backend timeout in seconds. "
                "SAEM on 50 subjects ~10s, 120 subjects ~60-120s, "
                "1000+ subjects ~300-600s. Default 900s is safe for most datasets."
            ),
            min=1,
        ),
    ] = 900,
    output: Annotated[
        Path,
        typer.Option(help="Bundle output directory."),
    ] = Path("runs"),
    agentic: Annotated[
        bool,
        typer.Option(
            "--agentic/--no-agentic",
            help=(
                "Enable the agentic LLM backend (Phase 3). "
                "OFF by default — the agentic loop ships aggregated diagnostics "
                "to a third-party LLM provider. Pass --agentic to opt in on the "
                "discovery/optimization lanes."
            ),
        ),
    ] = False,
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "LLM provider for the agentic backend. "
                "[dim]anthropic[/dim], [dim]openai[/dim], [dim]gemini[/dim], "
                "[dim]ollama[/dim], [dim]openrouter[/dim]."
            ),
        ),
    ] = "anthropic",
    model: Annotated[
        str | None,
        typer.Option(
            help=(
                "LLM model name. Defaults per provider: "
                "anthropic=claude-sonnet-4-20250514, openai=gpt-4o, "
                "gemini=gemini-2.5-flash, ollama=qwen3:4b."
            ),
        ),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            help="Max agentic LLM iterations per run (PRD §4.2.6: cap=25).",
            min=1,
            max=25,
        ),
    ] = 10,
    parallel_models: Annotated[
        int,
        typer.Option(
            "--parallel-models",
            "-j",
            help="Max concurrent model evaluations (R subprocesses). Default 1 = sequential.",
            min=1,
        ),
    ] = 1,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help=(
                "Estimation backend. [dim]nlmixr2[/dim] (classical SAEM/FOCEi, default) "
                "or [dim]bayesian_stan[/dim] (Stan+Torsten via cmdstanpy, Phase 2+ — "
                "requires `uv sync --extra bayesian` and a CmdStan installation)."
            ),
        ),
    ] = "nlmixr2",
    bayes_chains: Annotated[
        int,
        typer.Option(
            "--bayes-chains",
            help="Number of NUTS chains (Bayesian backend only). Default 4.",
            min=1,
        ),
    ] = 4,
    bayes_warmup: Annotated[
        int,
        typer.Option(
            "--bayes-warmup",
            help="Warmup iterations per chain (Bayesian backend only). Default 1000.",
            min=100,
        ),
    ] = 1000,
    bayes_sampling: Annotated[
        int,
        typer.Option(
            "--bayes-sampling",
            help="Sampling iterations per chain (Bayesian backend only). Default 1000.",
            min=100,
        ),
    ] = 1000,
    bayes_adapt_delta: Annotated[
        float,
        typer.Option(
            "--bayes-adapt-delta",
            help=(
                "NUTS target acceptance (Bayesian backend only). "
                "Default 0.95; raise to 0.99 for funnels."
            ),
            min=0.5,
            max=0.999,
        ),
    ] = 0.95,
    bayes_max_treedepth: Annotated[
        int,
        typer.Option(
            "--bayes-max-treedepth",
            help="NUTS max treedepth (Bayesian backend only). Default 12.",
            min=4,
            max=20,
        ),
    ] = 12,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed pipeline logs."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-essential output."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help=(
                "Skip interactive confirmations "
                "(e.g. agentic data-sharing prompt for non-local providers)."
            ),
        ),
    ] = False,
    resume_agentic: Annotated[
        bool,
        typer.Option(
            "--resume-agentic",
            help=(
                "Skip classical search (Stage 5) and resume from an existing "
                "``classical_checkpoint.json`` in the output bundle directory. "
                "Use after an agentic API failure to restart the LLM loop without "
                "re-running the full SAEM search."
            ),
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Preview the pipeline without running any R backends. "
                "Runs ingest → profile → NCA → search-space enumeration and "
                "prints the dispatch plan (candidate count, backends, gate policy). "
                "Useful for validating data and estimating compute before committing."
            ),
        ),
    ] = False,
    binary_encode: Annotated[
        list[str] | None,
        typer.Option(
            "--binary-encode",
            help=(
                "Override the auto-detected remap for a binary categorical "
                "covariate. Format: ``COL=VAL1:0,VAL2:1`` (repeatable). "
                "Example: ``--binary-encode SEX=M:0,F:1``. The raw values "
                "are parsed as strings if non-numeric, else as ints/floats."
            ),
        ),
    ] = None,
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

    # --- Parse --binary-encode flag (must happen before ingestion) ---
    binary_encode_overrides: dict[str, dict[object, int]] | None = None
    if binary_encode:
        binary_encode_overrides = {}
        for entry in binary_encode:
            if "=" not in entry:
                err_console.print(
                    f"[red bold]Invalid --binary-encode:[/] {escape(entry)} "
                    "(expected COL=VAL1:0,VAL2:1)"
                )
                raise typer.Exit(code=1)
            col, pairs = entry.split("=", 1)
            remap: dict[object, int] = {}
            for pair in pairs.split(","):
                if ":" not in pair:
                    err_console.print(
                        f"[red bold]Invalid --binary-encode pair:[/] {escape(pair)} "
                        "(expected VAL:0 or VAL:1)"
                    )
                    raise typer.Exit(code=1)
                raw, target_str = pair.rsplit(":", 1)
                try:
                    target = int(target_str)
                except ValueError:
                    err_console.print(
                        f"[red bold]Invalid --binary-encode target:[/] {escape(target_str)} "
                        "(must be 0 or 1)"
                    )
                    raise typer.Exit(code=1) from None
                if target not in (0, 1):
                    err_console.print(
                        f"[red bold]Invalid --binary-encode target:[/] {target} (must be 0 or 1)"
                    )
                    raise typer.Exit(code=1)
                # Try numeric first, fall back to stripped string.
                key: object
                try:
                    key = int(raw)
                except ValueError:
                    try:
                        key = float(raw)
                    except ValueError:
                        key = raw.strip()
                remap[key] = target
            binary_encode_overrides[col.strip()] = remap

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
        # Resolve and display policy path + version for traceability.
        # Delegate the fallback lookup to apmode.paths so CLI and
        # orchestrator share one source of truth.
        from apmode.paths import policy_path_for_lane as _policy_path_for_lane

        _policy_path: Path | None = policy or _policy_path_for_lane(lane.value)
        if _policy_path is not None:
            _pol_version = "?"
            try:
                _pol_data = json.loads(_policy_path.read_text())
                _pol_version = str(_pol_data.get("policy_version", "?"))
            except Exception:
                pass
            ingest_table.add_row("Policy", escape(str(_policy_path)))
            ingest_table.add_row("Policy version", escape(_pol_version))
        console.print(Panel(ingest_table, title="[bold]Data Summary[/]", border_style="blue"))

    # --- Dry-run mode: preview dispatch plan, do not execute backends ---
    if dry_run:
        from apmode.data.initial_estimates import NCAEstimator
        from apmode.data.profiler import profile_data
        from apmode.search.candidates import SearchSpace, generate_root_candidates

        try:
            with console.status("[cyan]Profiling data (dry-run)...[/]", spinner="dots"):
                evidence = profile_data(df, manifest)
            with console.status("[cyan]Computing NCA estimates (dry-run)...[/]", spinner="dots"):
                nca = NCAEstimator(df, manifest)
                estimates = nca.estimate_per_subject()
            space = SearchSpace.from_manifest(evidence)
            root_cands = generate_root_candidates(space, base_params=estimates)
        except Exception as e:
            err_console.print(f"[red bold]Dry-run failed:[/] {escape(str(e))}")
            raise typer.Exit(code=1) from None

        dry_table = Table(show_header=False, box=None, padding=(0, 2))
        dry_table.add_column(style="dim")
        dry_table.add_column(style="bold")
        n_classical = sum(
            1 for c in root_cands if not getattr(c, "has_node_modules", lambda: False)()
        )
        n_node = len(root_cands) - n_classical
        dispatch_parts: list[str] = []
        if n_classical:
            dispatch_parts.append(f"{n_classical} → nlmixr2 (SAEM)")
        if n_node:
            dispatch_parts.append(f"{n_node} → jax_node")
        dry_table.add_row("Root candidates", str(len(root_cands)))
        dry_table.add_row("Dispatch plan", " + ".join(dispatch_parts) or "[dim]none[/]")
        cmt_set: set[int] = getattr(space, "structural_cmt", set())
        abs_set: set[str] = getattr(space, "absorption_types", set())
        elim_set: set[str] = getattr(space, "elimination_types", set())
        if cmt_set:
            dry_table.add_row("Compartments", ", ".join(str(c) for c in sorted(cmt_set)))
        if abs_set:
            dry_table.add_row("Absorption", ", ".join(sorted(abs_set)))
        if elim_set:
            dry_table.add_row("Elimination", ", ".join(sorted(elim_set)))
        dry_table.add_row("Lane", lane.value)
        console.print(
            Panel(
                dry_table,
                title="[bold]Dry-Run Preview[/]  [dim](no R backends executed)[/]",
                border_style="yellow",
            )
        )
        console.print("  [dim]To execute:[/] [bold]apmode run[/] [dim](remove --dry-run)[/]")
        return

    # --- Write nlmixr2-ready CSV ---
    nlmixr2_df = to_nlmixr2_format(df)
    # Unique name prevents concurrent runs from clobbering each other's temp file.
    data_csv = (output / f"_tmp_data_{int(time.time_ns())}.csv").resolve()
    data_csv.parent.mkdir(parents=True, exist_ok=True)
    nlmixr2_df.to_csv(data_csv, index=False)

    config = RunConfig(
        lane=lane.value,
        seed=seed,
        timeout_seconds=timeout,
        policy_path=policy,
        max_concurrency=parallel_models,
        binary_encode_overrides=binary_encode_overrides,
    )

    # --- Backend selection ---
    runner: BackendRunner
    if backend == "bayesian_stan":
        try:
            from apmode.backends.bayesian_runner import BayesianRunner
            from apmode.bundle.models import SamplerConfig
        except ImportError as imp_exc:
            err_console.print(
                f"[red bold]Bayesian backend requires extras:[/] "
                f"uv sync --extra bayesian ({imp_exc})"
            )
            raise typer.Exit(code=1) from None
        sampler_cfg = SamplerConfig(
            chains=bayes_chains,
            warmup=bayes_warmup,
            sampling=bayes_sampling,
            adapt_delta=bayes_adapt_delta,
            max_treedepth=bayes_max_treedepth,
            seed=seed,
        )
        runner = BayesianRunner(
            work_dir=output / "_work",
            default_sampler_config=sampler_cfg,
        )
        if not quiet:
            console.print(
                f"[dim]Bayesian backend: chains={bayes_chains} warmup={bayes_warmup} "
                f"sampling={bayes_sampling} adapt_delta={bayes_adapt_delta}[/]"
            )
    elif backend == "nlmixr2":
        runner = Nlmixr2Runner(work_dir=output / "_work", estimation=["saem"])
    else:
        err_console.print(
            f"[red bold]Unknown backend:[/] {backend!r} (expected nlmixr2 or bayesian_stan)"
        )
        raise typer.Exit(code=1)

    # --- Agentic LLM backend (Phase 3) ---
    agentic_runner_instance = None
    _agentic_enabled = agentic and lane.value in ("discovery", "optimization")
    if agentic and lane == Lane.submission and not quiet:
        console.print(
            "[dim]Note: --agentic is not permitted in the submission lane "
            "(PRD §3) — agentic backend disabled.[/]"
        )
    if _agentic_enabled:
        # For non-local providers, confirm that aggregated diagnostics may be
        # transmitted to a third-party API — required for data governance compliance.
        if provider != "ollama" and not yes and not quiet:
            console.print()
            console.print(
                Panel(
                    f"[yellow]Agentic mode will transmit aggregated PK diagnostics "
                    f"to provider [bold]{escape(provider)}[/].[/]\n\n"
                    "Ensure your data governance policy permits sending aggregated "
                    "summary statistics (no individual-level data) to external APIs.\n"
                    "Pass [bold]--yes[/] / [bold]-y[/] to suppress this prompt.",
                    title="[bold yellow]Data Sharing Notice[/]",
                    border_style="yellow",
                )
            )
            _agentic_enabled = typer.confirm("Continue with agentic backend?", default=False)
            if not _agentic_enabled:
                console.print("[dim]Agentic backend disabled.[/]")
        if _agentic_enabled:
            agentic_runner_instance = _try_build_agentic_runner(
                inner_runner=runner,
                provider=provider,
                model_name=model,
                max_iterations=max_iterations,
                lane=lane.value,
                trace_dir=output / "agentic_trace",
                quiet=quiet,
            )

    # Orchestrator currently types the primary runner as Nlmixr2Runner; the
    # BayesianRunner shares the BackendRunner protocol so the cast is safe.
    from typing import cast as _cast

    from apmode.backends.nlmixr2_runner import Nlmixr2Runner as _Nlmixr2Runner

    orchestrator = Orchestrator(
        _cast("_Nlmixr2Runner", runner),
        output,
        config,
        agentic_runner=agentic_runner_instance,
    )

    # --- Pipeline execution ---
    # No global spinner: the orchestrator's structlog output reports each
    # stage (profile → NCA → search → gates → bundle) and a long-running
    # spinner would hide those messages. Progress is visible via the logger.
    if not quiet:
        console.print("  [dim]Running pipeline (stage progress via log)...[/]")

    try:
        result = asyncio.run(
            orchestrator.run(manifest, df, data_csv, skip_classical=resume_agentic)
        )
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

    # --- Submission-lane NODE/agentic exclusion hint ---
    if lane == Lane.submission:
        _g2_pass_count = sum(1 for _, p in result.gate2_results if p)
        _n_recommended = len(result.recommended)
        _n_excluded = _g2_pass_count - _n_recommended
        if _n_excluded > 0:
            console.print(
                f"  [dim]Note: {_n_excluded} candidate(s) excluded from Recommended"
                " — NODE/agentic models are ineligible in the submission lane (PRD §3).[/]"
            )

    # --- Top model detail (loaded from bundle artifacts) ---
    _ranking = _load_json(result.bundle_dir / "ranking.json", "")
    if _ranking and _ranking.get("ranked_candidates"):
        _best = _ranking["ranked_candidates"][0]
        _best_id = _best.get("candidate_id", "?")
        _best_res = _load_result_json(result.bundle_dir, _best_id)
        _top_table = Table(show_header=False, box=None, padding=(0, 2))
        _top_table.add_column(style="dim")
        _top_table.add_column(style="bold")
        _top_table.add_row("Best model", escape(_best_id))
        _ofv = _best_res.get("ofv") if _best_res else None
        if _is_real_number(_ofv):
            _top_table.add_row("OFV", f"{_ofv:.2f}")
        _bic = _best.get("bic")
        _aic = _best.get("aic")
        if _is_real_number(_bic):
            _top_table.add_row("BIC", f"{_bic:.1f}")
        if _is_real_number(_aic):
            _top_table.add_row("AIC", f"{_aic:.1f}")
        _eta_shk = _best_res.get("eta_shrinkage", {}) if _best_res else {}
        if isinstance(_eta_shk, dict) and _eta_shk:
            _shk_vals = [v for v in _eta_shk.values() if _is_real_number(v)]
            if _shk_vals:
                _shk_max = max(_shk_vals)
                if _shk_max > 0.3:
                    _shk_display = f"[bold red]⚠ {_shk_max:.0%}[/]"
                elif _shk_max > 0.2:
                    _shk_display = f"[yellow]~ {_shk_max:.0%}[/]"
                else:
                    _shk_display = f"[cyan]{_shk_max:.0%}[/]"
                _top_table.add_row("η-shrinkage (max)", _shk_display)
        _top_table.add_row("Backend", escape(str(_best.get("backend", "?"))))
        console.print(Panel(_top_table, title="[bold]Top Model[/]", border_style="cyan"))

    console.print(
        f"  [dim]Full parameters:[/] [bold]apmode log {escape(str(result.bundle_dir))} --top 3[/]"
    )


def _pass_fraction(passed: int, total: int) -> str:
    """Format a pass/total fraction with symbol + color (colorblind-safe)."""
    if total == 0:
        return "[dim]--[/]"
    if passed == total:
        return f"[green]✓ {passed}[/]/{total} passed"
    if passed > 0:
        return f"[yellow]~ {passed}[/]/{total} passed"
    return f"[red]✗ {passed}[/]/{total} passed"


# ---------------------------------------------------------------------------
# Shared helpers for bundle parsing
# ---------------------------------------------------------------------------


def _load_json(path: Path, label: str) -> dict[str, Any] | None:
    """Load a JSON file, returning None if missing, unreadable, or not a JSON object."""
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (PermissionError, IsADirectoryError, OSError) as e:
        if label:
            console.print(f"  [yellow]Warning:[/] cannot read {escape(label)}: {escape(str(e))}")
        return None
    except json.JSONDecodeError as e:
        if label:
            console.print(f"  [yellow]Warning:[/] corrupt {escape(label)}: {escape(str(e))}")
        return None
    if not isinstance(raw, dict):
        if label:
            console.print(f"  [yellow]Warning:[/] {escape(label)} is not a JSON object")
        return None
    return raw


def _load_result_json(bundle_dir: Path, candidate_id: str) -> dict[str, Any] | None:
    """Load results/{candidate_id}_result.json from a bundle. Returns None if missing."""
    return _load_json(bundle_dir / "results" / f"{candidate_id}_result.json", "")


def _discover_agentic_mode_dirs(bundle_dir: Path) -> dict[str, Path]:
    """Return map of mode_name -> trace_dir for all agentic modes found.

    Current layout writes per-mode subdirectories:
      bundle/agentic_trace/refine/
      bundle/agentic_trace/independent/

    Legacy (pre-fix) layout used a flat directory:
      bundle/agentic_trace/iter_*.json
    This helper transparently handles both — legacy bundles are keyed as
    ``"default"`` so existing inspection commands still work.
    """
    base = bundle_dir / "agentic_trace"
    if not base.is_dir():
        return {}

    mode_dirs: dict[str, Path] = {}
    for name in ("refine", "independent"):
        candidate = base / name
        if candidate.is_dir() and any(candidate.glob("iter_*_input.json")):
            mode_dirs[name] = candidate

    # Legacy flat layout fallback
    if not mode_dirs and any(base.glob("iter_*_input.json")):
        mode_dirs["default"] = base

    return mode_dirs


def _parse_json_dict_row(line: str) -> dict[str, Any] | None:
    """Parse a single JSONL row; return the dict or None if missing/malformed."""
    if not line.strip():
        return None
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    return raw if isinstance(raw, dict) else None


def _validate_jsonl(path: Path) -> str | None:
    """Validate that every non-blank line is a valid JSON object. Returns error or None."""
    try:
        text = path.read_text()
    except OSError as e:
        return str(e)
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                return f"line {i}: {e}"
            if not isinstance(raw, dict):
                return f"line {i}: expected JSON object, got {type(raw).__name__}"
    return None


def _validate_file(path: Path, filename: str) -> tuple[str, str]:
    """Validate a single bundle file. Returns (status_markup, note)."""
    if filename.endswith(".jsonl"):
        err = _validate_jsonl(path)
        if err:
            return "[bold red]✗ BAD[/]", err
        return "[green]✓ OK[/]", ""
    try:
        raw = json.loads(path.read_text())
    except OSError as e:
        return "[bold red]✗ BAD[/]", str(e)
    except json.JSONDecodeError as e:
        return "[bold red]✗ BAD[/]", str(e)
    if not isinstance(raw, dict):
        return "[bold red]✗ BAD[/]", f"expected JSON object, got {type(raw).__name__}"
    return "[green]✓ OK[/]", ""


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
            table.add_row("[bold red]✗ MISS[/]", f, "required")
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
            table.add_row("[green]✓ OK[/]", f"{d}/", f"{n} files")
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
            table.add_row(
                "Nonlinear CL",
                escape(str(em.get("nonlinear_clearance_evidence_strength", "?"))),
            )
            table.add_row("BLQ burden", escape(str(em.get("blq_burden", "?"))))
            if "time_varying_covariates" in em:
                table.add_row(
                    "Time-varying covariates", _neutral_yes_no(em["time_varying_covariates"])
                )
            cov_miss = em.get("covariate_missingness")
            if cov_miss:
                pattern = cov_miss.get("pattern", "?")
                frac = cov_miss.get("fraction_incomplete", 0.0)
                table.add_row("Covariate missingness", f"{pattern} ({frac:.1%})")
            if "absorption_coverage" in em:
                table.add_row("Absorption", escape(str(em["absorption_coverage"])))
            if "protocol_heterogeneity" in em:
                table.add_row("Protocol", escape(str(em["protocol_heterogeneity"])))
            console.print(Panel(table, title="[bold]Evidence Profile[/]", border_style="cyan"))
            sections_shown += 1

            # ---- Per-signal provenance (manifest schema v3) ----
            signals = em.get("nonlinear_clearance_signals") or {}
            if signals:
                sig_table = Table(show_header=True, box=None, padding=(0, 2))
                sig_table.add_column("Signal", style="dim")
                sig_table.add_column("Eligible", justify="center")
                sig_table.add_column("Voted", justify="center")
                sig_table.add_column("Value", justify="right")
                sig_table.add_column("Threshold", justify="right")
                sig_table.add_column("Citation", style="dim")
                for sid, sig in signals.items():
                    obs = sig.get("observed_value")
                    thr = sig.get("threshold_value")
                    sig_table.add_row(
                        escape(str(sid)),
                        _bool_badge(bool(sig.get("eligible"))),
                        _bool_badge(bool(sig.get("voted"))),
                        f"{obs:.3f}" if isinstance(obs, (int, float)) else "—",
                        f"{thr:.3f}" if isinstance(thr, (int, float)) else "—",
                        escape(str(sig.get("citation", "")))[:32],
                    )
                console.print(
                    Panel(
                        sig_table,
                        title="[bold]Nonlinear-Clearance Signals[/]",
                        border_style="cyan",
                    )
                )
                sections_shown += 1

    # --- Imputation stability (optional, MI runs only) ---
    stab_path = bundle_dir / "imputation_stability.json"
    if stab_path.exists():
        stab = _load_json(stab_path, "imputation_stability.json")
        if stab:
            table = Table(show_header=True, box=None, padding=(0, 2))
            table.add_column("Candidate", style="dim")
            table.add_column("Pooled BIC", justify="right")
            table.add_column("Convergence", justify="right")
            table.add_column("Rank stability", justify="right")
            for entry in stab.get("entries", [])[:10]:
                pb = entry.get("pooled_bic")
                pb_str = f"{pb:.1f}" if isinstance(pb, (int, float)) else "—"
                cr = entry.get("convergence_rate", 0.0)
                rs = entry.get("rank_stability", 0.0)
                table.add_row(
                    escape(str(entry.get("candidate_id", "?"))[:20]),
                    pb_str,
                    f"{cr:.0%}",
                    f"{rs:.0%}",
                )
            method = stab.get("method", "?")
            m = stab.get("m", "?")
            console.print(
                Panel(
                    table,
                    title=f"[bold]Imputation Stability[/]  ({method}, m={m})",
                    border_style="yellow",
                )
            )
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
                row = _parse_json_dict_row(line)
                if row is None:
                    if line.strip():
                        console.print(
                            f"  [yellow]Warning:[/] search_trajectory.jsonl corrupt at line {i}"
                        )
                        parse_ok = False
                        break
                    continue
                if row.get("converged"):
                    n_conv += 1
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
            _g3_gate_data = _load_json(g3[0], "gate3")
            # Load ranking.json for winner/metric/ΔBIC detail
            _rank_data = _load_json(bundle_dir / "ranking.json", "")
            _g3_metric = "BIC"
            _g3_winner = "—"
            _g3_delta = ""
            if _rank_data:
                _g3_metric = str(_rank_data.get("ranking_metric", "bic")).upper()
                _g3_cands = _rank_data.get("ranked_candidates", [])
                if _g3_cands:
                    _g3_winner = escape(str(_g3_cands[0].get("candidate_id", "?")))
                    if len(_g3_cands) >= 2:
                        _b1 = _g3_cands[0].get("bic")
                        _b2 = _g3_cands[1].get("bic")
                        if _is_real_number(_b1) and _is_real_number(_b2):
                            _g3_delta = f"  Δ{_g3_metric}={_b2 - _b1:+.1f} vs runner-up"
            g3_detail = f"winner={_g3_winner}  metric={_g3_metric}{_g3_delta}"
            if _g3_gate_data and _g3_gate_data.get("summary_reason"):
                g3_detail += f"  [{escape(str(_g3_gate_data['summary_reason'])[:60])}]"
            table.add_row("Gate 3", "[dim]--[/]", "[dim]--[/]", g3_detail)

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
                table.add_column("OFV", justify="right")
                table.add_column("BIC", justify="right")
                table.add_column("AIC", justify="right")
                table.add_column("η-shk max", justify="right")
                table.add_column("Params", justify="right")
                table.add_column("Backend", style="dim")
                for c_raw in candidates[:10]:
                    c: dict[str, Any] = c_raw
                    cid = str(c.get("candidate_id", "?"))
                    rank_str = str(c.get("rank", "?"))
                    bic_val = c.get("bic")
                    aic_val = c.get("aic")
                    bic = f"{bic_val:.1f}" if _is_real_number(bic_val) else "--"
                    aic = f"{aic_val:.1f}" if _is_real_number(aic_val) else "--"
                    # Load OFV and η-shrinkage from result artifact
                    _res = _load_result_json(bundle_dir, cid)
                    ofv_val = _res.get("ofv") if _res else None
                    ofv = f"{ofv_val:.2f}" if _is_real_number(ofv_val) else "--"
                    eta_shk: dict[str, Any] = (_res or {}).get("eta_shrinkage", {}) or {}
                    shk_vals = [v for v in eta_shk.values() if _is_real_number(v)]
                    if shk_vals:
                        shk_max = max(shk_vals)
                        # ⚠/~ symbols ensure legibility without color alone
                        if shk_max > 0.3:
                            shk_str = f"[bold red]⚠ {shk_max:.0%}[/]"
                        elif shk_max > 0.2:
                            shk_str = f"[yellow]~ {shk_max:.0%}[/]"
                        else:
                            shk_str = f"[cyan]{shk_max:.0%}[/]"
                    else:
                        shk_str = "--"
                    table.add_row(
                        rank_str,
                        escape(cid),
                        ofv,
                        bic,
                        aic,
                        shk_str,
                        str(c.get("n_params", "?")),
                        escape(str(c.get("backend", "?"))),
                    )
                if len(candidates) > 10:
                    more = f"[dim]... +{len(candidates) - 10} more[/]"
                    table.add_row("", more, "", "", "", "", "", "")
                console.print(Panel(table, title="[bold]Ranking[/]", border_style="green"))
                sections_shown += 1

    # --- Backend versions and seed registry ---
    bv_path = bundle_dir / "backend_versions.json"
    if bv_path.exists():
        bv = _load_json(bv_path, "backend_versions.json")
        if bv:
            bv_table = Table(show_header=False, box=None, padding=(0, 2))
            bv_table.add_column(style="dim")
            bv_table.add_column()
            for bv_key, bv_val in bv.items():
                if isinstance(bv_val, str) and bv_val:
                    bv_table.add_row(bv_key.replace("_", " ").title(), escape(bv_val))
            sr_path = bundle_dir / "seed_registry.json"
            if sr_path.exists():
                sr = _load_json(sr_path, "seed_registry.json")
                if sr:
                    bv_table.add_row("Root seed", str(sr.get("root_seed", "?")))
                    bv_table.add_row("R seed", str(sr.get("r_seed", "?")))
                    bv_table.add_row("NumPy seed", str(sr.get("np_seed", "?")))
            console.print(Panel(bv_table, title="[bold]Versions & Seeds[/]", border_style="dim"))
            sections_shown += 1

    # --- Deep inspection hints ---
    hints: list[str] = []
    if _discover_agentic_mode_dirs(bundle_dir):
        hints.append("[bold]apmode trace[/] for agentic iteration details")
    graph_path = bundle_dir / "search_graph.json"
    if graph_path.exists():
        hints.append("[bold]apmode graph[/] for search DAG visualization")
    lineage_path = bundle_dir / "candidate_lineage.json"
    if lineage_path.exists():
        hints.append("[bold]apmode lineage[/] <candidate_id> for transform history")
    if hints:
        console.print("  [dim]Deep inspection:[/] " + " | ".join(hints))
        sections_shown += 1

    if sections_shown == 0:
        console.print("[dim]Bundle is empty or contains no recognized artifacts.[/]")

    console.print()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _bool_badge(value: object) -> str:
    """Format a boolean as a pass/fail badge (symbol + color — colorblind-safe).

    Uses ✓/✗ symbols so status is legible regardless of color perception.
    Use for eligibility, pass/fail, and opt-in flags.
    For neutral descriptive flags use _neutral_yes_no().
    """
    if value is True:
        return "[green]✓ yes[/]"
    if value is False:
        return "[red]✗ no[/]"
    return "[dim]?[/]"


def _neutral_yes_no(value: object) -> str:
    """Format a boolean as a dim yes/no — for neutral descriptive data characteristics."""
    if value is True:
        return "[dim]yes[/]"
    if value is False:
        return "[dim]no[/]"
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
    ] = 753849,
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
                    cached = fetch_dataset(dataset, cache_dir)
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
    try:
        with console.status("[cyan]Profiling data...[/]"):
            evidence = profile_data(df, manifest)
    except Exception as e:
        err_console.print(f"[red bold]Profiling failed:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    _print_evidence_panel(evidence)

    # --- Step 4: NCA estimates ---
    try:
        with console.status("[cyan]Computing NCA estimates...[/]"):
            nca = NCAEstimator(df, manifest)
            estimates = nca.estimate_per_subject()
    except Exception as e:
        err_console.print(f"[red bold]NCA failed:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from None

    t = Table(show_header=True, box=None, padding=(0, 2))
    t.add_column("Parameter", style="bold")
    t.add_column("Estimate", justify="right")
    for name, val in sorted(estimates.items()):
        t.add_row(name, f"{val:.4g}")
    console.print(Panel(t, title="[bold]NCA Initial Estimates[/]", border_style="cyan"))

    # --- Step 5: Search space preview ---
    try:
        space = SearchSpace.from_manifest(evidence)
        candidates = generate_root_candidates(space, base_params=estimates)
    except Exception as e:
        err_console.print(f"[red bold]Search space generation failed:[/] {escape(str(e))}")
        raise typer.Exit(code=1) from None

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
        "nonlinear_clearance_evidence_strength",
        "compartmentality",
        "multi_dose_detected",
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


def _launch_run(
    csv_path: Path,
    lane: Lane,
    seed: int,
    output: Path,
    parallel_models: int = 1,
    timeout: int = 900,
) -> None:
    """Delegate to the run command by invoking it directly.

    Propagates any non-zero exit code from the underlying pipeline so
    ``explore -y`` does not lie about success. ``timeout`` matches the
    default used by the standalone ``run`` command (900s). Callers can
    override when they know the dataset size warrants it.
    """
    try:
        run(
            dataset=csv_path,
            lane=lane,
            seed=seed,
            output=output,
            timeout=timeout,
            parallel_models=parallel_models,
            verbose=False,
            quiet=False,
        )
    except typer.Exit as e:
        # Typer's Exit is a RuntimeError subclass, not SystemExit.
        if e.exit_code != 0:
            raise
    except SystemExit as e:
        # Defense in depth: inner code might raise SystemExit directly.
        raw = e.code
        code = (
            0
            if raw is None
            else (raw if isinstance(raw, int) and not isinstance(raw, bool) else 1)
        )
        if code != 0:
            raise typer.Exit(code=code) from None


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
        t.add_column("BIC_A", justify="right", style="dim")
        t.add_column(escape(str(bundle_b.name)), style="yellow")
        t.add_column("BIC_B", justify="right", style="dim")

        cands_a = rank_a.get("ranked_candidates", [])
        cands_b = rank_b.get("ranked_candidates", [])
        max_len = max(len(cands_a), len(cands_b), 1)

        for i in range(min(max_len, 10)):
            # Use candidate_id (new schema) with model_id fallback for legacy bundles
            if i < len(cands_a):
                ca = cands_a[i].get("candidate_id", cands_a[i].get("model_id", "?"))
                bic_a_val = cands_a[i].get("bic")
                bic_a = f"{bic_a_val:.1f}" if _is_real_number(bic_a_val) else "—"
                # Append spec one-liner if compiled spec exists
                spec_a = _load_json(bundle_a / "compiled_specs" / f"{ca}.json", "")
                if spec_a:
                    ca = f"{ca} [dim]({escape(_spec_one_liner(spec_a))})[/]"
            else:
                ca, bic_a = "—", "—"
            if i < len(cands_b):
                cb = cands_b[i].get("candidate_id", cands_b[i].get("model_id", "?"))
                bic_b_val = cands_b[i].get("bic")
                bic_b = f"{bic_b_val:.1f}" if _is_real_number(bic_b_val) else "—"
                spec_b = _load_json(bundle_b / "compiled_specs" / f"{cb}.json", "")
                if spec_b:
                    cb = f"{cb} [dim]({escape(_spec_one_liner(spec_b))})[/]"
            else:
                cb, bic_b = "—", "—"
            t.add_row(str(i + 1), ca, bic_a, cb, bic_b)
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
        typer.Option(
            "--top",
            "-n",
            help="Show top N ranked candidates with parameters (0 = disabled).",
            min=0,
        ),
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
        t.add_column("Gate Failed", style="red")
        t.add_column("Failed Checks", style="dim")
        t.add_column("Reason")

        for line in text.split("\n"):
            rec = _parse_json_dict_row(line)
            if rec is None:
                continue
            # Support both new schema (candidate_id/gate_failed/summary_reason)
            # and legacy fallback field names.
            cand_id = rec.get("candidate_id", rec.get("model_id", "?"))
            gate_failed = rec.get("gate_failed", rec.get("failed_gate", "?"))
            checks_list = rec.get("failed_checks", [])
            checks_str = (
                ", ".join(checks_list) if isinstance(checks_list, list) else str(checks_list)
            )
            reason = rec.get("summary_reason", rec.get("reason", "-"))
            t.add_row(
                escape(str(cand_id)),
                escape(str(gate_failed)),
                escape(checks_str[:40]),
                escape(str(reason)[:80]),
            )
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
        # Support both candidate_id (new) and model_id (legacy) field
        candidate_id = data.get("candidate_id", data.get("model_id", f.stem))

        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="dim", min_width=28)
        t.add_column()
        t.add_row("Status", status)
        if data.get("policy_version"):
            t.add_row("Policy version", escape(str(data["policy_version"])))

        checks_raw = data.get("checks", data.get("check_results", []))
        if isinstance(checks_raw, list):
            # New schema: list of GateCheckResult dicts
            for check in checks_raw:
                if not isinstance(check, dict):
                    continue
                check_id = str(check.get("check_id", "?"))
                check_pass = bool(check.get("passed", False))
                # ✓/✗ symbols make status legible without relying on color alone
                icon = "[green]✓[/]" if check_pass else "[bold red]✗[/]"
                observed = check.get("observed")
                threshold = check.get("threshold")
                units = str(check.get("units") or "")
                detail_parts: list[str] = []
                if observed is not None:
                    obs_str = f"{observed:.4g}" if isinstance(observed, float) else str(observed)
                    detail_parts.append(f"obs={obs_str}{' ' + units if units else ''}")
                if threshold is not None:
                    thr_str = (
                        f"{threshold:.4g}" if isinstance(threshold, float) else str(threshold)
                    )
                    detail_parts.append(f"thr={thr_str}{' ' + units if units else ''}")
                detail = "  " + ", ".join(detail_parts) if detail_parts else ""
                t.add_row(escape(check_id), f"{icon}{escape(detail)}")
        elif isinstance(checks_raw, dict):
            # Legacy dict format
            for check_name, result in checks_raw.items():
                if isinstance(result, dict):
                    check_pass = bool(result.get("passed", result.get("pass", False)))
                    icon = "[green]✓[/]" if check_pass else "[bold red]✗[/]"
                    reason = result.get("reason", result.get("message", ""))
                    t.add_row(escape(check_name), f"{icon}  {escape(str(reason)[:100])}")
                else:
                    icon = "[green]✓[/]" if result else "[bold red]✗[/]"
                    t.add_row(escape(check_name), icon)

        summary = data.get("summary_reason", "")
        if summary:
            t.add_row("[dim]Summary[/]", escape(str(summary)[:120]))

        console.print(Panel(t, title=f"[bold]{escape(candidate_id)}[/]", border_style="yellow"))


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
        # Support both candidate_id (new) and model_id (legacy)
        cand_id = cand.get("candidate_id", cand.get("model_id", "?"))
        bic = cand.get("bic")
        aic = cand.get("aic")

        # Load full result artifact for OFV and η-shrinkage
        _res = _load_result_json(rank_path.parent, cand_id)
        ofv = _res.get("ofv") if _res else None
        eta_shk: dict[str, Any] = (_res or {}).get("eta_shrinkage", {}) or {}

        t = Table(show_header=True, box=None, padding=(0, 2))
        t.add_column("Parameter", style="bold", no_wrap=True)
        t.add_column("Estimate", justify="right")
        t.add_column("SE", justify="right", style="dim")
        t.add_column("RSE%", justify="right")
        t.add_column("95% CI", justify="right", style="dim")
        t.add_column("Category", style="dim")

        params = cand.get("parameter_estimates", cand.get("parameters", {}))
        if isinstance(params, dict):
            for pname, pdata in sorted(params.items()):
                if isinstance(pdata, dict):
                    est = pdata.get("estimate", "?")
                    se = pdata.get("se")
                    rse = pdata.get("rse")
                    ci_lo = pdata.get("ci95_lower")
                    ci_hi = pdata.get("ci95_upper")
                    cat = str(pdata.get("category", ""))
                    est_str = f"{est:.4g}" if isinstance(est, float) else str(est)
                    se_str = f"{se:.4g}" if _is_real_number(se) else "—"
                    # RSE may be fractional (0-1) or already in percent; normalise to percent.
                    # ⚠/~ symbols ensure legibility without depending on color alone.
                    if _is_real_number(rse):
                        assert isinstance(rse, (int, float))
                        rse_pct = rse * 100 if abs(rse) <= 2 else float(rse)
                        if rse_pct > 50:
                            rse_str = f"[bold red]⚠ {rse_pct:.0f}%[/]"
                        elif rse_pct > 30:
                            rse_str = f"[yellow]~ {rse_pct:.0f}%[/]"
                        else:
                            rse_str = f"[cyan]{rse_pct:.0f}%[/]"
                    else:
                        rse_str = "—"
                    ci_str = (
                        f"[{ci_lo:.3g}, {ci_hi:.3g}]"
                        if _is_real_number(ci_lo) and _is_real_number(ci_hi)
                        else "—"
                    )
                    t.add_row(escape(pname), est_str, se_str, rse_str, escape(ci_str), escape(cat))
                else:
                    t.add_row(
                        escape(pname),
                        f"{pdata:.4g}" if isinstance(pdata, float) else str(pdata),
                        "—",
                        "—",
                        "—",
                        "",
                    )

        # η-shrinkage mini-table (appended as separate rows below params)
        if isinstance(eta_shk, dict) and eta_shk:
            t.add_row("[dim]—[/]", "[dim]—[/]", "[dim]—[/]", "[dim]—[/]", "[dim]—[/]", "[dim]—[/]")
            for eta_name, eta_val in sorted(eta_shk.items()):
                if _is_real_number(eta_val):
                    assert isinstance(eta_val, (int, float))
                    _ev = float(eta_val)
                    shk_pct = _ev * 100 if _ev <= 1.5 else _ev
                    # ⚠/~ symbols preserve meaning without relying on color alone
                    if shk_pct > 30:
                        shk_str = f"[bold red]⚠ {shk_pct:.0f}%[/]"
                    elif shk_pct > 20:
                        shk_str = f"[yellow]~ {shk_pct:.0f}%[/]"
                    else:
                        shk_str = f"[cyan]{shk_pct:.0f}%[/]"
                    t.add_row(
                        f"[dim]{escape(eta_name)} (η-shk)[/]",
                        shk_str,
                        "—",
                        "—",
                        "—",
                        "iiv",
                    )

        subtitle: list[str] = []
        if _is_real_number(ofv):
            subtitle.append(f"OFV={ofv:.2f}")
        if _is_real_number(bic):
            subtitle.append(f"BIC={bic:.1f}")
        if _is_real_number(aic):
            subtitle.append(f"AIC={aic:.1f}")
        # Simulation-based diagnostics (populated by backends that emit
        # posterior-predictive draws via build_predictive_diagnostics).
        # Absent → quietly omitted so legacy bundles still render.
        _diag = (_res or {}).get("diagnostics", {}) or {}
        _npe = _diag.get("npe_score") if isinstance(_diag, dict) else None
        _auc_cmax = _diag.get("auc_cmax_be_score") if isinstance(_diag, dict) else None
        _auc_src = _diag.get("auc_cmax_source") if isinstance(_diag, dict) else None
        _vpc_obj = _diag.get("vpc") if isinstance(_diag, dict) else None
        if _is_real_number(_npe):
            subtitle.append(f"NPE={_npe:.3g}")
        if _is_real_number(_auc_cmax):
            assert isinstance(_auc_cmax, (int, float))
            src_tag = f" [{_auc_src}]" if _auc_src else ""
            subtitle.append(f"AUC/Cmax BE={float(_auc_cmax):.0%}{src_tag}")
        # Per-percentile VPC coverage when populated. The ranker's scalar
        # concordance hides which percentile failed; surfacing the raw
        # per-percentile dict here lets a reviewer distinguish median
        # miscalibration from tail underprediction.
        if isinstance(_vpc_obj, dict):
            _cov_raw = _vpc_obj.get("coverage")
            if isinstance(_cov_raw, dict) and _cov_raw:
                vpc_parts: list[str] = []
                for _pkey in sorted(_cov_raw.keys()):
                    _pval = _cov_raw[_pkey]
                    if _is_real_number(_pval):
                        assert isinstance(_pval, (int, float))
                        vpc_parts.append(f"{_pkey}={float(_pval):.2f}")
                if vpc_parts:
                    subtitle.append("VPC[" + "/".join(vpc_parts) + "]")
        sub = f" ({', '.join(subtitle)})" if subtitle else ""

        console.print(
            Panel(t, title=f"[bold]#{i} {escape(cand_id)}{sub}[/]", border_style="green")
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
        t.add_row("Nonlinear CL", str(em.get("nonlinear_clearance_evidence_strength", "?")))

    # Search trajectory
    st_path = bundle_dir / "search_trajectory.jsonl"
    if st_path.exists():
        text = st_path.read_text().strip()
        lines = text.split("\n") if text else []
        n_conv = 0
        for ln in lines:
            row = _parse_json_dict_row(ln)
            if row and row.get("converged", False):
                n_conv += 1
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
            # Prefer candidate_id (current schema) with model_id fallback
            # for legacy bundles — matches the pattern used elsewhere in
            # the CLI.
            top3 = ", ".join(c.get("candidate_id", c.get("model_id", "?")) for c in cands[:3])
            t.add_row("Top ranked", top3)

    console.print(Panel(t, title="[bold]Bundle Overview[/]", border_style="blue"))
    console.print("[dim]Use --failed, --gate, or --top for details.[/]")


# ---------------------------------------------------------------------------
# trace command — agentic iteration traces (Phase 3 deep inspection)
# ---------------------------------------------------------------------------


@app.command()
def trace(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run bundle directory."),
    ],
    iteration: Annotated[
        int | None,
        typer.Option("--iteration", "-i", help="Show detail for a specific iteration."),
    ] = None,
    cost: Annotated[
        bool,
        typer.Option("--cost", help="Show token/cost aggregation."),
    ] = False,
    mode: Annotated[
        str | None,
        typer.Option(
            "--mode",
            help=(
                "Filter to a single agentic mode: [dim]refine[/] or [dim]independent[/]. "
                "Defaults to showing all modes."
            ),
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Machine-readable JSON output."),
    ] = False,
) -> None:
    """Inspect agentic LLM iteration traces.

    Shows the propose-validate-compile-fit loop history from the agentic
    backend (Phase 3, PRD §4.2.6). The agentic stage runs two modes by default
    (refine + independent) each written to its own subdirectory.

    \b
    Examples:
      apmode trace ./runs/run_abc123                       # all modes
      apmode trace ./runs/run_abc123 --mode refine         # refine only
      apmode trace ./runs/run_abc123 --iteration 5 --mode refine
      apmode trace ./runs/run_abc123 --cost                # cost across modes
    """
    if not bundle_dir.is_dir():
        err_console.print(f"[red bold]Error:[/] not a directory: {escape(str(bundle_dir))}")
        raise typer.Exit(code=1)

    mode_dirs = _discover_agentic_mode_dirs(bundle_dir)
    if not mode_dirs:
        console.print("[dim]No agentic trace found in this bundle.[/]")
        return

    if mode is not None:
        if mode not in mode_dirs:
            err_console.print(
                f"[red]Mode '{escape(mode)}' not found. "
                f"Available: {', '.join(sorted(mode_dirs))}[/]"
            )
            raise typer.Exit(code=1)
        mode_dirs = {mode: mode_dirs[mode]}

    # --- Cost aggregation (sums across modes) ---
    if cost:
        _show_trace_cost_multi(mode_dirs, output_json)
        return

    # --- Single iteration detail (requires --mode when multiple) ---
    if iteration is not None:
        if len(mode_dirs) > 1:
            err_console.print(
                f"[red]--iteration requires --mode when multiple modes present "
                f"({', '.join(sorted(mode_dirs))}).[/]"
            )
            raise typer.Exit(code=1)
        only_dir = next(iter(mode_dirs.values()))
        _show_trace_iteration(only_dir, iteration, output_json)
        return

    # --- Summary table across modes ---
    _show_trace_summary_multi(mode_dirs, output_json)


def _show_trace_summary_multi(mode_dirs: dict[str, Path], as_json: bool) -> None:
    """Show summary table(s) across one or more agentic modes."""
    if as_json:
        payload = {mode: _read_iterations_jsonl(mode_dir) for mode, mode_dir in mode_dirs.items()}
        console.print(json.dumps(payload, indent=2))
        return

    for mode, mode_dir in mode_dirs.items():
        console.print()
        console.rule(f"[bold]Agentic Trace — mode: {escape(mode)}[/]")
        _show_trace_summary(mode_dir, as_json=False)


def _read_iterations_jsonl(trace_dir: Path) -> list[dict[str, Any]]:
    """Parse agentic_iterations.jsonl, skipping corrupt or non-dict lines."""
    iters_path = trace_dir / "agentic_iterations.jsonl"
    if not iters_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    text = iters_path.read_text().strip()
    if not text:
        return entries
    for i, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        row = _parse_json_dict_row(line)
        if row is None:
            console.print(
                f"  [yellow]Warning:[/] corrupt or non-object line {i} in "
                f"{escape(str(iters_path))}"
            )
            continue
        entries.append(row)
    return entries


def _show_trace_summary(trace_dir: Path, as_json: bool) -> None:
    """Show summary table of all agentic iterations."""
    iters_path = trace_dir / "agentic_iterations.jsonl"
    if not iters_path.exists():
        console.print("[dim]No agentic_iterations.jsonl found.[/]")
        return

    entries = _read_iterations_jsonl(trace_dir)

    if not entries:
        console.print("[dim]No iterations recorded.[/]")
        return

    if as_json:
        console.print(json.dumps(entries, indent=2))
        return

    console.print()
    console.rule("[bold]Agentic Iteration Trace[/]")

    t = Table(show_lines=False)
    t.add_column("#", style="dim", width=3, justify="right")
    t.add_column("Before", style="bold", max_width=20)
    t.add_column("After", max_width=20)
    t.add_column("Transforms", max_width=40)
    t.add_column("Conv", justify="center", width=4)
    t.add_column("BIC", justify="right", width=8)
    t.add_column("Error", style="red", max_width=30)

    for e in entries:
        it = str(e.get("iteration", "?"))
        before = e.get("spec_before", "?")
        after = e.get("spec_after") or "[dim]--[/]"
        transforms = ", ".join(e.get("transforms_proposed", [])) or "[dim]none[/]"
        conv = "[green]✓[/]" if e.get("converged") else "[dim]✗[/]"
        bic_val = e.get("bic")
        bic = f"{bic_val:.1f}" if _is_real_number(bic_val) else "--"
        error = (e.get("error") or "")[:30]
        t.add_row(it, before, after, transforms, conv, bic, error)

    console.print(t)

    # Mini convergence chart
    bics = [e.get("bic") for e in entries if e.get("bic") is not None]
    if len(bics) >= 2:
        n_conv = sum(1 for e in entries if e.get("converged"))
        bar = _mini_bar(n_conv, len(entries))
        console.print(f"\n  [dim]Convergence:[/] {n_conv}/{len(entries)} iterations  {bar}")

    console.print()


def _show_trace_iteration(trace_dir: Path, iteration: int, as_json: bool) -> None:
    """Show detail for a specific agentic iteration."""
    iter_id = f"iter_{iteration:03d}"

    input_path = trace_dir / f"{iter_id}_input.json"
    output_path = trace_dir / f"{iter_id}_output.json"
    meta_path = trace_dir / f"{iter_id}_meta.json"

    if not input_path.exists():
        err_console.print(f"[red]Iteration {iteration} not found.[/]")
        raise typer.Exit(code=1)

    inp: dict[str, Any] = _load_json(input_path, f"{iter_id}_input.json") or {}
    out: dict[str, Any] = (
        _load_json(output_path, f"{iter_id}_output.json") or {} if output_path.exists() else {}
    )
    meta: dict[str, Any] = (
        _load_json(meta_path, f"{iter_id}_meta.json") or {} if meta_path.exists() else {}
    )

    if as_json:
        console.print(json.dumps({"input": inp, "output": out, "meta": meta}, indent=2))
        return

    console.print()
    console.rule(f"[bold]Iteration {iteration}[/]")

    # Input panel
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim", min_width=20)
    t.add_column()
    t.add_row("Candidate", inp.get("candidate_id", "?"))
    t.add_row("Prompt Template", inp.get("prompt_template", "?"))
    diag = inp.get("diagnostics_summary", {})
    if diag:
        for k, v in diag.items():
            t.add_row(f"  {k}", str(v))
    console.print(Panel(t, title="[bold]Input[/]", border_style="blue"))

    # Output panel
    transforms = out.get("parsed_transforms", [])
    rejected = out.get("transforms_rejected", [])
    valid = "[green]PASS[/]" if out.get("validation_passed") else "[red]FAIL[/]"
    errors = out.get("validation_errors", [])

    t2 = Table(show_header=False, box=None, padding=(0, 2))
    t2.add_column(style="dim", min_width=20)
    t2.add_column()
    t2.add_row("Validation", valid)
    if transforms:
        t2.add_row("Transforms", "\n".join(f"[green]✓[/] {tr}" for tr in transforms))
    if rejected:
        t2.add_row("Rejected", "\n".join(f"[red]✗[/] {tr}" for tr in rejected))
    if errors:
        t2.add_row("Errors", "\n".join(errors))

    # Show reasoning (truncated)
    raw = out.get("raw_output", "")
    if raw:
        display = raw[:500]
        if len(raw) > 500:
            display += f"\n[dim]... ({len(raw) - 500} chars truncated, use --json for full)[/]"
        t2.add_row("LLM Output", display)

    console.print(Panel(t2, title="[bold]Output[/]", border_style="cyan"))

    # Meta panel
    if meta:
        t3 = Table(show_header=False, box=None, padding=(0, 2))
        t3.add_column(style="dim", min_width=20)
        t3.add_column()
        t3.add_row("Model", meta.get("model_id", "?"))
        t3.add_row("Version", meta.get("model_version", "?"))
        in_tok = meta.get("input_tokens", 0)
        out_tok = meta.get("output_tokens", 0)
        t3.add_row("Tokens", f"{in_tok} in / {out_tok} out")
        cost_val = meta.get("cost_usd", 0)
        t3.add_row("Cost", f"${cost_val:.4f}")
        t3.add_row("Wall Time", f"{meta.get('wall_time_seconds', 0):.1f}s")
        console.print(Panel(t3, title="[bold]Meta[/]", border_style="magenta"))

    console.print()


def _show_trace_cost_multi(mode_dirs: dict[str, Path], as_json: bool) -> None:
    """Aggregate token/cost across all modes and show per-mode + total."""
    per_mode: dict[str, dict[str, float | int]] = {}
    for mode, mode_dir in mode_dirs.items():
        meta_files = sorted(mode_dir.glob("iter_*_meta.json"))
        tot_in = 0
        tot_out = 0
        tot_cost = 0.0
        tot_time = 0.0
        for f in meta_files:
            meta = _load_json(f, f.name)
            if meta is None:
                continue
            tot_in += int(meta.get("input_tokens", 0) or 0)
            tot_out += int(meta.get("output_tokens", 0) or 0)
            tot_cost += float(meta.get("cost_usd", 0.0) or 0.0)
            tot_time += float(meta.get("wall_time_seconds", 0.0) or 0.0)
        per_mode[mode] = {
            "iterations": len(meta_files),
            "input_tokens": tot_in,
            "output_tokens": tot_out,
            "total_tokens": tot_in + tot_out,
            "cost_usd": round(tot_cost, 4),
            "wall_time_seconds": round(tot_time, 1),
        }

    grand_total = {
        "iterations": sum(int(m["iterations"]) for m in per_mode.values()),
        "input_tokens": sum(int(m["input_tokens"]) for m in per_mode.values()),
        "output_tokens": sum(int(m["output_tokens"]) for m in per_mode.values()),
        "cost_usd": round(sum(float(m["cost_usd"]) for m in per_mode.values()), 4),
        "wall_time_seconds": round(
            sum(float(m["wall_time_seconds"]) for m in per_mode.values()), 1
        ),
    }
    grand_total["total_tokens"] = grand_total["input_tokens"] + grand_total["output_tokens"]

    if as_json:
        console.print(json.dumps({"per_mode": per_mode, "total": grand_total}, indent=2))
        return

    console.print()
    console.rule("[bold]Agentic Cost Summary[/]")
    t = Table(show_header=True, box=None, padding=(0, 2))
    t.add_column("Mode", style="bold")
    t.add_column("Iterations", justify="right")
    t.add_column("Input tok", justify="right")
    t.add_column("Output tok", justify="right")
    t.add_column("Cost (USD)", justify="right")
    t.add_column("Wall (s)", justify="right")
    for mode, m in per_mode.items():
        t.add_row(
            mode,
            str(m["iterations"]),
            str(m["input_tokens"]),
            str(m["output_tokens"]),
            f"${float(m['cost_usd']):.4f}",
            f"{float(m['wall_time_seconds']):.1f}",
        )
    if len(per_mode) > 1:
        t.add_row(
            "[bold]TOTAL[/]",
            str(grand_total["iterations"]),
            str(grand_total["input_tokens"]),
            str(grand_total["output_tokens"]),
            f"${float(grand_total['cost_usd']):.4f}",
            f"{float(grand_total['wall_time_seconds']):.1f}",
        )
    console.print(t)
    console.print()


# ---------------------------------------------------------------------------
# lineage command — per-candidate DSL transform history (deep inspection)
# ---------------------------------------------------------------------------


@app.command()
def lineage(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run bundle directory."),
    ],
    candidate_id: Annotated[
        str,
        typer.Argument(help="Target candidate ID to trace."),
    ],
    spec: Annotated[
        bool,
        typer.Option("--spec", help="Show DSL spec at each step."),
    ] = False,
    show_gate: Annotated[
        bool,
        typer.Option("--gate/--no-gate", help="Show gate outcomes per step."),
    ] = True,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Machine-readable JSON output."),
    ] = False,
) -> None:
    """Trace the transform lineage of a specific candidate.

    Shows the chain of DSL transforms from root to the target candidate,
    with gate status at each step.

    \b
    Examples:
      apmode lineage ./runs/run_abc123 cand_a3f8              # transform chain
      apmode lineage ./runs/run_abc123 cand_a3f8 --spec       # with DSL snapshots
      apmode lineage ./runs/run_abc123 cand_a3f8 --no-gate    # skip gate details
    """
    if not bundle_dir.is_dir():
        err_console.print(f"[red bold]Error:[/] not a directory: {escape(str(bundle_dir))}")
        raise typer.Exit(code=1)

    lineage_path = bundle_dir / "candidate_lineage.json"
    if not lineage_path.exists():
        err_console.print("[red]No candidate_lineage.json found.[/]")
        raise typer.Exit(code=1)

    lineage_data = _load_json(lineage_path, "candidate_lineage.json")
    if not lineage_data:
        raise typer.Exit(code=1)

    entries: list[dict[str, Any]] = lineage_data.get("entries", [])

    # Merge agentic lineage from each mode subdir (refine/, independent/) or
    # legacy flat agentic_trace/ layout
    existing_ids = {e.get("candidate_id") for e in entries}
    for mode, mode_dir in _discover_agentic_mode_dirs(bundle_dir).items():
        al_path = mode_dir / "agentic_lineage.json"
        if not al_path.exists():
            continue
        al_data = _load_json(al_path, f"{mode}/agentic_lineage.json")
        if not al_data:
            continue
        for ae in al_data.get("entries", []):
            if ae.get("candidate_id") and ae["candidate_id"] not in existing_ids:
                entries.append(ae)
                existing_ids.add(ae["candidate_id"])

    # Build parent map: candidate_id -> entry
    by_id: dict[str, dict[str, Any]] = {}
    for e in entries:
        cid = e.get("candidate_id")
        if cid:
            by_id[cid] = e

    if candidate_id not in by_id:
        err_console.print(f"[red]Candidate '{escape(candidate_id)}' not found in lineage.[/]")
        raise typer.Exit(code=1)

    # Back-trace to root
    chain: list[dict[str, Any]] = []
    current: str | None = candidate_id
    visited: set[str] = set()
    while current is not None and current not in visited:
        visited.add(current)
        entry = by_id.get(current)
        if entry is None:
            break
        chain.append(entry)
        parent: str | None = entry.get("parent_id")
        current = parent

    chain.reverse()  # root → target

    # Enrich with gate decisions
    gd_dir = bundle_dir / "gate_decisions"
    specs_dir = bundle_dir / "compiled_specs"

    if output_json:
        result_entries: list[dict[str, Any]] = []
        for step in chain:
            step_data: dict[str, Any] = dict(step)
            if show_gate and gd_dir.is_dir():
                step_data["gates"] = _get_gate_status(gd_dir, step["candidate_id"])
            if spec and specs_dir.is_dir():
                spec_path = specs_dir / f"{step['candidate_id']}.json"
                if spec_path.exists():
                    spec_data = _load_json(spec_path, f"spec {step['candidate_id']}")
                    if spec_data is not None:
                        step_data["spec"] = spec_data
            result_entries.append(step_data)
        console.print(json.dumps(result_entries, indent=2))
        return

    console.print()
    console.rule(f"[bold]Lineage: {escape(candidate_id)}[/]")

    from rich.tree import Tree

    tree = Tree(f"[bold]{escape(chain[0]['candidate_id'])}[/] [dim](root)[/]")

    current_branch = tree
    for i, step in enumerate(chain):
        cid = step["candidate_id"]
        transform = step.get("transform")

        # Gate status
        gate_str = ""
        if show_gate and gd_dir.is_dir():
            gates = _get_gate_status(gd_dir, cid)
            parts: list[str] = []
            for gname, gpassed in gates.items():
                if gpassed:
                    parts.append(f"[green]✓ {gname}[/]")
                else:
                    parts.append(f"[red]✗ {gname}[/]")
            if parts:
                gate_str = "  " + "  ".join(parts)

        if i == 0:
            # Root already shown as tree label
            if gate_str:
                current_branch.add(f"[dim]Gates:[/]{gate_str}")
        else:
            label = f"[dim]→[/] [bold cyan]{escape(transform or '?')}[/]"
            transform_node = current_branch.add(label)
            node_label = f"[bold]{escape(cid)}[/]{gate_str}"
            current_branch = transform_node.add(node_label)

        # Optional spec display
        if spec and specs_dir.is_dir():
            spec_path = specs_dir / f"{cid}.json"
            if spec_path.exists():
                spec_data = _load_json(spec_path, f"spec {cid}")
                if spec_data is not None:
                    spec_summary = _spec_one_liner(spec_data)
                    current_branch.add(f"[dim]{spec_summary}[/]")

    console.print(tree)
    console.print()


def _get_gate_status(gd_dir: Path, candidate_id: str) -> dict[str, bool]:
    """Read gate pass/fail status for a candidate."""
    gates: dict[str, bool] = {}
    for gate_name, pattern in [
        ("G1", f"gate1_{candidate_id}.json"),
        ("G2", f"gate2_{candidate_id}.json"),
        ("G2.5", f"gate2_5_{candidate_id}.json"),
    ]:
        path = gd_dir / pattern
        if path.exists():
            data = _load_json(path, pattern)
            if data:
                gates[gate_name] = bool(data.get("passed", False))
    return gates


def _spec_one_liner(spec_data: dict[str, Any]) -> str:
    """Generate a one-line summary of a DSL spec."""
    parts: list[str] = []
    for key in ["absorption", "distribution", "elimination", "observation"]:
        if key in spec_data:
            val = spec_data[key]
            if isinstance(val, dict) and "type" in val:
                parts.append(val["type"])
            elif isinstance(val, str):
                parts.append(val)
    return " x ".join(parts) if parts else "?"


# ---------------------------------------------------------------------------
# report command — structured regulatory report (Phase 3)
# ---------------------------------------------------------------------------


@app.command()
def report(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run bundle directory."),
    ],
    fmt: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format when artifacts exist: html (opens browser) or md (pager).",
        ),
    ] = "html",
    no_browser: Annotated[
        bool,
        typer.Option("--no-browser", help="Print path instead of opening browser (HTML)."),
    ] = False,
) -> None:
    """View or generate a structured regulatory report from a bundle.

    If report.html / report.md artifacts exist in the bundle (written by the
    orchestrator at run completion), displays them immediately.  Full
    on-demand report generation (PDF, DOCX) is a Phase 3 feature.

    \b
    Examples:
      apmode report ./runs/run_abc123
      apmode report ./runs/run_abc123 --format md
      apmode report ./runs/run_abc123 --no-browser
    """
    if not bundle_dir.is_dir():
        err_console.print(f"[red bold]Error:[/] not a directory: {escape(str(bundle_dir))}")
        raise typer.Exit(code=1)

    html_path = bundle_dir / "report.html"
    md_path = bundle_dir / "report.md"

    # If the user asked for markdown or no html, prefer md.
    prefer_md = fmt.lower() == "md" or (not html_path.exists() and md_path.exists())

    if prefer_md and md_path.exists():
        console.print(f"[dim]Viewing [bold]{escape(str(md_path))}[/] — press q to exit.[/]")
        with open(md_path) as fh:
            content = fh.read()
        with console.pager():
            console.print(content)
        return

    if html_path.exists():
        console.print()
        if no_browser:
            console.print(
                Panel(
                    f"[bold]{escape(str(html_path))}[/]",
                    title="[bold]Report Path[/]",
                    border_style="cyan",
                )
            )
        else:
            import webbrowser

            webbrowser.open(html_path.as_uri())
            console.print(f"[dim]Opened [bold]{escape(str(html_path))}[/] in browser.[/]")
        return

    # No artifacts yet — show stub with context.
    console.print()
    console.print(
        Panel(
            "[bold yellow]No report artifacts found in this bundle.[/]\n\n"
            "The orchestrator writes [bold]report.md[/] and [bold]report.html[/] at "
            "the end of a completed run.  If the run is still in progress, wait for "
            "it to finish and re-run this command.\n\n"
            "Full on-demand PDF/DOCX generation is a Phase 3 feature.\n\n"
            "Alternatives:\n"
            f"  [bold]apmode inspect {escape(str(bundle_dir))}[/]  — structured bundle summary\n"
            f"  [bold]apmode log {escape(str(bundle_dir))} --top 3[/]  — parameter estimates\n"
            f"  [bold]apmode log {escape(str(bundle_dir))} --gate gate1[/]  — Gate 1 details",
            title="[bold]Report[/]",
            border_style="yellow",
        )
    )


# ---------------------------------------------------------------------------
# doctor command — R / nlmixr2 / LLM provider health check
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Check that all runtime dependencies are installed and reachable.

    Verifies: R, nlmixr2, rxode2, cmdstan (optional), and LLM provider
    connectivity (requires API keys to be set).

    \b
    Examples:
      apmode doctor
    """
    import shutil
    import subprocess

    console.print()
    console.print(Panel("[bold]APMODE Environment Health Check[/]", border_style="cyan"))
    console.print()

    table = Table(box=None, padding=(0, 1))
    table.add_column("Component", style="bold", width=28)
    table.add_column("Status", width=12)
    table.add_column("Detail", style="dim")

    all_ok = True

    # ---- R ----
    r_exe = shutil.which("Rscript")
    if r_exe:
        try:
            r_ver = subprocess.check_output(
                ["Rscript", "--version"], stderr=subprocess.STDOUT, text=True, timeout=10
            ).strip()
        except Exception:
            r_ver = "?"
        table.add_row("R (Rscript)", "[green]✓ found[/]", r_ver)
    else:
        table.add_row(
            "R (Rscript)", "[red]✗ missing[/]", "Install R ≥ 4.4 from https://cran.r-project.org"
        )
        all_ok = False

    # ---- nlmixr2 ----
    if r_exe:
        try:
            nlmixr2_ver: str = subprocess.check_output(
                [
                    "Rscript",
                    "-e",
                    "cat(as.character(packageVersion('nlmixr2')))",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=20,
            ).strip()
            if nlmixr2_ver:
                table.add_row("nlmixr2 (R pkg)", "[green]✓ found[/]", f"v{nlmixr2_ver}")
            else:
                table.add_row(
                    "nlmixr2 (R pkg)", "[red]✗ missing[/]", "install.packages('nlmixr2')"
                )
                all_ok = False
        except Exception:
            table.add_row("nlmixr2 (R pkg)", "[red]✗ error[/]", "Could not query R packages")
            all_ok = False
    else:
        table.add_row("nlmixr2 (R pkg)", "[dim]-- skipped[/]", "R not found")

    # ---- rxode2 ----
    if r_exe:
        try:
            rxode2_ver: str = subprocess.check_output(
                ["Rscript", "-e", "cat(as.character(packageVersion('rxode2')))"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=15,
            ).strip()
            if rxode2_ver:
                table.add_row("rxode2 (R pkg)", "[green]✓ found[/]", f"v{rxode2_ver}")
            else:
                table.add_row("rxode2 (R pkg)", "[red]✗ missing[/]", "install.packages('rxode2')")
                all_ok = False
        except Exception:
            table.add_row("rxode2 (R pkg)", "[red]✗ error[/]", "Could not query R packages")
            all_ok = False
    else:
        table.add_row("rxode2 (R pkg)", "[dim]-- skipped[/]", "R not found")

    # ---- CmdStan (optional) ----
    cmdstan_home = shutil.which("cmdstan") or ""
    try:
        import cmdstanpy

        cs_path = cmdstanpy.cmdstan_path()
        table.add_row("CmdStan (optional)", "[green]✓ found[/]", cs_path)
    except Exception:
        table.add_row(
            "CmdStan (optional)",
            "[dim]-- not found[/]",
            "Optional; needed for bayesian_stan backend",
        )
        _ = cmdstan_home  # suppress unused-var warning

    # ---- Python packages ----
    for pkg in ("apmode", "pydantic", "typer", "rich", "lark", "pandera"):
        try:
            import importlib.metadata as _meta

            ver = _meta.version(pkg)
            table.add_row(f"{pkg} (Python)", "[green]✓ found[/]", f"v{ver}")
        except Exception:
            table.add_row(f"{pkg} (Python)", "[red]✗ missing[/]", f"pip install {pkg}")
            all_ok = False

    # ---- LLM providers (API key presence only) ----
    import os

    provider_checks: list[tuple[str, str]] = [
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("OpenAI", "OPENAI_API_KEY"),
        ("Google Gemini", "GOOGLE_API_KEY"),
        ("OpenRouter", "OPENROUTER_API_KEY"),
    ]
    for pname, env_key in provider_checks:
        val = os.environ.get(env_key, "")
        if val:
            table.add_row(
                f"LLM: {pname}",
                "[green]✓ key set[/]",
                f"{env_key}=****{val[-4:]}",
            )
        else:
            table.add_row(
                f"LLM: {pname}",
                "[dim]-- not set[/]",
                f"{env_key} not in environment",
            )

    # Ollama (local)
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434", timeout=2)
        table.add_row("LLM: Ollama (local)", "[green]✓ running[/]", "http://localhost:11434")
    except Exception:
        table.add_row("LLM: Ollama (local)", "[dim]-- not reachable[/]", "http://localhost:11434")

    console.print(table)
    console.print()

    if all_ok:
        console.print("[green bold]✓ All required components found.[/]")
    else:
        err_console.print(
            "[yellow bold]⚠ Some required components are missing. See table above for details.[/]"
        )
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# ls command — list run bundles
# ---------------------------------------------------------------------------


@app.command(name="ls")
def ls_command(
    runs_dir: Annotated[
        Path,
        typer.Argument(help="Directory to search for run bundles. Defaults to ./runs."),
    ] = Path("runs"),
    sort_by: Annotated[
        str,
        typer.Option("--sort", help="Sort column: time, lane, status, bic."),
    ] = "time",
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum runs to show (0 = all)."),
    ] = 20,
) -> None:
    """List APMODE run bundles with a summary table.

    Scans RUNS_DIR for subdirectories that look like run bundles
    (contain data_manifest.json) and shows a sortable summary.

    \b
    Examples:
      apmode ls
      apmode ls ./my_runs
      apmode ls --sort bic --limit 10
    """
    if not runs_dir.is_dir():
        err_console.print(f"[red bold]Error:[/] directory not found: {escape(str(runs_dir))}")
        raise typer.Exit(code=1)

    import os

    bundles: list[dict[str, Any]] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        dm_path = entry / "data_manifest.json"
        if not dm_path.exists():
            continue

        dm = _load_json(dm_path, "") or {}
        # Lane lives in policy_file.json (not data_manifest)
        pf = _load_json(entry / "policy_file.json", "") or {}
        lane: str = pf.get("lane") or dm.get("lane") or "?"

        ranking = _load_json(entry / "ranking.json", "") or {}
        ranked: list[dict[str, Any]] = ranking.get("ranked_candidates", [])
        best_bic: str = "—"
        best_id: str = "—"
        if ranked:
            top = ranked[0]
            cid = top.get("candidate_id", top.get("model_id", "?"))
            best_bic = str(round(float(top.get("bic", float("nan"))), 1))
            best_id = cid[:16] + ("…" if len(cid) > 16 else "")
        else:
            # Fall back to search_trajectory.jsonl — pick best converged BIC
            traj_path = entry / "search_trajectory.jsonl"
            if traj_path.exists():
                try:
                    traj_lines = traj_path.read_text().splitlines()
                    converged = [
                        json.loads(ln)
                        for ln in traj_lines
                        if ln.strip() and json.loads(ln).get("converged")
                    ]
                    if converged:
                        best_row = min(converged, key=lambda r: float(r.get("bic", float("inf"))))
                        cid2 = best_row.get("candidate_id", "?")
                        best_bic = str(round(float(best_row["bic"]), 1))
                        best_id = cid2[:16] + ("…" if len(cid2) > 16 else "")
                except Exception:
                    pass

        # Count candidates from trajectory (more reliable than ranking)
        traj_p = entry / "search_trajectory.jsonl"
        if traj_p.exists():
            try:
                n_candidates = str(sum(1 for ln in traj_p.read_text().splitlines() if ln.strip()))
            except Exception:
                n_candidates = str(len(ranked)) if ranked else "?"
        else:
            n_candidates = str(len(ranked)) if ranked else "?"
        mtime = os.path.getmtime(dm_path)
        bundles.append(
            {
                "name": entry.name,
                "lane": lane,
                "n_candidates": n_candidates,
                "best_bic": best_bic,
                "best_id": best_id,
                "mtime": mtime,
                "path": str(entry),
            }
        )

    if not bundles:
        console.print(f"[dim]No run bundles found in [bold]{escape(str(runs_dir))}[/].[/]")
        return

    # Sort
    if sort_by == "bic":
        bundles.sort(
            key=lambda b: float(b["best_bic"]) if b["best_bic"] not in ("?", "—") else float("inf")
        )
    elif sort_by == "lane":
        bundles.sort(key=lambda b: b["lane"])
    elif sort_by == "status":
        bundles.sort(key=lambda b: b["n_candidates"])
    else:
        bundles.sort(key=lambda b: b["mtime"], reverse=True)

    if limit > 0:
        bundles = bundles[:limit]

    table = Table(title=f"Runs in {escape(str(runs_dir))}", box=None, padding=(0, 1))
    table.add_column("Bundle", style="bold", no_wrap=True)
    table.add_column("Lane", width=12)
    table.add_column("Candidates", width=12)
    table.add_column("Best BIC", width=10)
    table.add_column("Best ID", width=20, style="dim")

    for b in bundles:
        lane_color = (
            "cyan"
            if b["lane"] == "submission"
            else "yellow"
            if b["lane"] == "discovery"
            else "magenta"
            if b["lane"] == "optimization"
            else "dim"
        )
        table.add_row(
            b["name"],
            f"[{lane_color}]{escape(b['lane'])}[/]",
            b["n_candidates"],
            b["best_bic"],
            b["best_id"],
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[dim]{len(bundles)} run(s) shown. Use [bold]apmode inspect <bundle>[/] for details.[/]"
    )


# ---------------------------------------------------------------------------
# policies command — list and validate gate policy files
# ---------------------------------------------------------------------------


@app.command()
def policies(
    lane: Annotated[
        str | None,
        typer.Argument(help="Filter to a specific lane (submission, discovery, optimization)."),
    ] = None,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Validate JSON schema of each policy file."),
    ] = False,
) -> None:
    """List and inspect gate policy files.

    Gate policy files live in the repository's policies/ directory and are
    versioned artifacts (PRD §4.3.1).  Each file controls Gate 1/2/2.5/3
    thresholds for its lane.

    \b
    Examples:
      apmode policies
      apmode policies submission
      apmode policies --validate
    """
    # Locate policies/ via shared helper (H3: single source of truth).
    from apmode.paths import policies_dir as _policies_dir

    policies_dir = _policies_dir()
    if policies_dir is None or not policies_dir.is_dir():
        err_console.print(
            "[red bold]Error:[/] policies directory not found (set APMODE_POLICIES_DIR "
            "to override)."
        )
        raise typer.Exit(code=1)

    lanes_to_show = [lane] if lane else ["submission", "discovery", "optimization"]

    table = Table(title="Gate Policies", box=None, padding=(0, 1))
    table.add_column("Lane", style="bold", width=16)
    table.add_column("File", width=30)
    table.add_column("Version", width=12)
    table.add_column("Status", width=12)
    table.add_column("Gates", style="dim")

    all_ok = True

    for ln in lanes_to_show:
        pfile = policies_dir / f"{ln}.json"
        if not pfile.exists():
            table.add_row(ln, f"{ln}.json", "—", "[dim]-- missing[/]", "—")
            all_ok = False
            continue

        pdata = _load_json(pfile, f"{ln}.json")
        if pdata is None:
            table.add_row(ln, f"{ln}.json", "—", "[red]✗ corrupt[/]", "—")
            all_ok = False
            continue

        version = str(pdata.get("policy_version", "?"))
        gates_present: list[str] = []
        for g in ("gate1", "gate2", "gate2_5", "gate3"):
            if g in pdata:
                gates_present.append(g.replace("_", "."))

        status = "[green]✓ OK[/]"
        if validate:
            # Minimal schema check: required top-level keys
            required_keys = {"policy_version", "lane"}
            missing = required_keys - set(pdata.keys())
            if missing:
                status = f"[red]✗ missing {', '.join(sorted(missing))}[/]"
                all_ok = False

        table.add_row(
            ln,
            f"{ln}.json",
            version,
            status,
            ", ".join(gates_present) or "—",
        )

    console.print()
    console.print(table)
    console.print()

    # Show full policy detail if a single lane was requested
    if lane:
        pfile = policies_dir / f"{lane}.json"
        if pfile.exists():
            pdata = _load_json(pfile, f"{lane}.json") or {}
            dtable = Table(
                title=f"Policy: {lane}",
                show_header=True,
                box=None,
                padding=(0, 1),
            )
            dtable.add_column("Gate", style="bold", width=10)
            dtable.add_column("Check", width=32)
            dtable.add_column("Threshold", width=14)

            for g in ("gate1", "gate2", "gate2_5", "gate3"):
                gdata = pdata.get(g)
                if not isinstance(gdata, dict):
                    continue
                checks = gdata.get("checks", {})
                if isinstance(checks, dict) and checks:
                    for cname, cval in checks.items():
                        dtable.add_row(g.replace("_", "."), cname, str(cval))
                elif isinstance(checks, list) and checks:
                    for ch in checks:
                        if isinstance(ch, dict):
                            dtable.add_row(
                                g.replace("_", "."),
                                str(ch.get("check_id", "?")),
                                str(ch.get("threshold", "?")),
                            )
                else:
                    # Flat-style block (Gate 3 shape: top-level keys, no
                    # nested "checks"). Render each leaf so predictive-
                    # diagnostics knobs (vpc_n_bins, npe_weight,
                    # auc_cmax_nca_min_eligible, …) are visible.
                    for cname, cval in gdata.items():
                        dtable.add_row(g.replace("_", "."), cname, str(cval))

            console.print(dtable)
            console.print()

    if not all_ok:
        err_console.print("[yellow]⚠ Some policy files are missing or invalid.[/]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# graph command — full search DAG visualization (deep inspection)
# ---------------------------------------------------------------------------


@app.command()
def graph(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a run bundle directory."),
    ],
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help="Output format: tree, dot, mermaid, json."),
    ] = "tree",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write output to file."),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Filter by backend (nlmixr2, jax_node, agentic_llm)."),
    ] = None,
    converged: Annotated[
        bool,
        typer.Option("--converged", help="Show only converged candidates."),
    ] = False,
    depth: Annotated[
        int,
        typer.Option("--depth", help="Max tree depth."),
    ] = 10,
) -> None:
    """Visualize the full search DAG.

    Shows the tree/graph of all candidate models explored during the run,
    with convergence, gate status, and BIC on each node.

    \b
    Examples:
      apmode graph ./runs/run_abc123                           # tree view
      apmode graph ./runs/run_abc123 --format dot -o dag.dot   # Graphviz export
      apmode graph ./runs/run_abc123 --converged --backend nlmixr2
    """
    if not bundle_dir.is_dir():
        err_console.print(f"[red bold]Error:[/] not a directory: {escape(str(bundle_dir))}")
        raise typer.Exit(code=1)

    graph_path = bundle_dir / "search_graph.json"
    if not graph_path.exists():
        console.print("[dim]No search graph found in this bundle.[/]")
        return

    graph_data = _load_json(graph_path, "search_graph.json")
    if not graph_data:
        # File exists but is unreadable / corrupt / not a JSON object:
        err_console.print("[red bold]Error:[/] search_graph.json is empty or malformed.")
        raise typer.Exit(code=1)

    nodes: list[dict[str, Any]] = graph_data.get("nodes", [])
    edges: list[dict[str, Any]] = graph_data.get("edges", [])

    # Apply filters
    if backend:
        nodes = [n for n in nodes if n.get("backend") == backend]
        node_ids = {n["candidate_id"] for n in nodes}
        edges = [e for e in edges if e["parent_id"] in node_ids and e["child_id"] in node_ids]

    if converged:
        nodes = [n for n in nodes if n.get("converged")]
        node_ids = {n["candidate_id"] for n in nodes}
        edges = [e for e in edges if e["parent_id"] in node_ids and e["child_id"] in node_ids]

    valid_formats = {"tree", "dot", "mermaid", "json"}
    if fmt not in valid_formats:
        err_console.print(
            f"[red bold]Error:[/] unknown format '{escape(fmt)}'. "
            f"Choose from: {', '.join(sorted(valid_formats))}"
        )
        raise typer.Exit(code=1)

    def _write(text: str, label: str) -> None:
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        console.print(f"[green]{label} written to {escape(str(output))}[/]")

    if fmt == "json":
        text = json.dumps({"nodes": nodes, "edges": edges}, indent=2)
        if output:
            _write(text, "JSON")
        else:
            console.print(text)
        return

    if fmt == "dot":
        text = _graph_to_dot(nodes, edges)
        if output:
            _write(text, "DOT")
        else:
            console.print(text)
        return

    if fmt == "mermaid":
        text = _graph_to_mermaid(nodes, edges)
        if output:
            _write(text, "Mermaid")
        else:
            console.print(text)
        return

    # Default: tree view
    _graph_tree_view(nodes, edges, depth)


def _node_label(node: dict[str, Any]) -> str:
    """Build a display label for a graph node."""
    cid = node.get("candidate_id", "?")
    bic = node.get("bic")
    bic_str = f" BIC={bic:.1f}" if _is_real_number(bic) else ""
    rank = node.get("rank")
    rank_str = f" #{rank}" if rank else ""

    conv = node.get("converged", False)
    g1 = node.get("gate1_passed")
    g2 = node.get("gate2_passed")

    if not conv:
        status = "[dim]○ NC[/]"
    elif g2 is True:
        status = "[green]✓ PASS[/]"
    elif g1 is True:
        status = "[yellow]~ G1[/]"
    elif g1 is False:
        status = "[red]✗ FAIL[/]"
    else:
        status = "[dim]?[/]"

    star = " ★" if rank == 1 else ""
    return f"{escape(cid)}{bic_str}{rank_str} {status}{star}"


def _graph_tree_view(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    max_depth: int,
) -> None:
    """Render the search DAG as a Rich tree."""
    from rich.tree import Tree

    # Build children map from edges
    children_map: dict[str, list[tuple[str, str]]] = {}  # parent -> [(child, transform)]
    for e in edges:
        pid = e["parent_id"]
        cid = e["child_id"]
        transform = e.get("transform", "?")
        children_map.setdefault(pid, []).append((cid, transform))

    # Index nodes by id
    node_by_id: dict[str, dict[str, Any]] = {n["candidate_id"]: n for n in nodes}

    # Find roots (nodes with no parent or parent not in node set)
    child_ids = {e["child_id"] for e in edges}
    roots = [n for n in nodes if n["candidate_id"] not in child_ids]

    if not roots:
        console.print("[dim]No root nodes found in graph.[/]")
        return

    console.print()
    console.rule("[bold]Search DAG[/]")

    tree = Tree("[bold]Search Space[/]")

    visited: set[str] = set()

    def _add_children(parent_tree: Tree, parent_id: str, current_depth: int) -> None:
        if current_depth >= max_depth:
            remaining = len(children_map.get(parent_id, []))
            if remaining:
                parent_tree.add(f"[dim]... {remaining} more (--depth to expand)[/]")
            return
        for child_id, transform in children_map.get(parent_id, []):
            if child_id in visited:
                parent_tree.add(f"[dim]→ {escape(child_id)} (cycle)[/]")
                continue
            visited.add(child_id)
            child_node = node_by_id.get(child_id)
            if child_node is None:
                continue
            label = f"[dim]→[/] [cyan]{escape(transform)}[/] → {_node_label(child_node)}"
            child_tree = parent_tree.add(label)
            _add_children(child_tree, child_id, current_depth + 1)

    for root in roots:
        root_tree = tree.add(_node_label(root))
        _add_children(root_tree, root["candidate_id"], 1)

    console.print(tree)
    console.print(f"\n  [dim]{len(nodes)} nodes, {len(edges)} edges[/]")
    console.print()


def _dot_escape(s: str) -> str:
    """Escape a string for use in DOT labels."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _graph_to_dot(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    """Convert search graph to Graphviz DOT format."""
    lines = ["digraph search_dag {", "  rankdir=TB;", "  node [shape=box, fontsize=10];", ""]

    for n in nodes:
        cid = n["candidate_id"]
        bic = n.get("bic")
        conv = n.get("converged", False)
        rank = n.get("rank")

        label = _dot_escape(cid)
        if bic is not None:
            label += f"\\nBIC={bic:.1f}"
        if rank:
            label += f"\\n#{rank}"

        color = "gray" if not conv else ("green" if n.get("gate2_passed") else "yellow")
        style = "bold" if rank == 1 else "solid"
        lines.append(f'  "{_dot_escape(cid)}" [label="{label}", color={color}, style={style}];')

    lines.append("")
    for e in edges:
        transform = _dot_escape(e.get("transform", ""))
        pid = _dot_escape(e["parent_id"])
        cid = _dot_escape(e["child_id"])
        lines.append(f'  "{pid}" -> "{cid}" [label="{transform}"];')

    lines.append("}")
    return "\n".join(lines)


def _graph_to_mermaid(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> str:
    """Convert search graph to Mermaid flowchart format."""
    lines = ["flowchart TD"]

    def _mermaid_id(s: str) -> str:
        """Sanitize a string for use as a Mermaid node ID."""
        return re.sub(r"[^a-zA-Z0-9_]", "_", s)

    def _mermaid_label(s: str) -> str:
        """Escape a string for Mermaid labels."""
        return s.replace('"', "'").replace("|", "/")

    for n in nodes:
        cid = n["candidate_id"]
        mid = _mermaid_id(cid)
        bic = n.get("bic")
        label = cid
        if bic is not None:
            label += f" BIC={bic:.1f}"
        rank = n.get("rank")
        if rank:
            label += f" #{rank}"
        lines.append(f'  {mid}["{_mermaid_label(label)}"]')

    for e in edges:
        transform = _mermaid_label(e.get("transform", ""))
        pid = _mermaid_id(e["parent_id"])
        cid = _mermaid_id(e["child_id"])
        lines.append(f'  {pid} -->|"{transform}"| {cid}')

    return "\n".join(lines)
