# SPDX-License-Identifier: GPL-2.0-or-later
"""``apmode formular {fmt,lint,validate,explain,diff,lower,compat}`` Typer sub-app.

Formular sharpening plan §4 Phase 1 (P1.9): a thin renderer over the
existing Formular DSL machinery — the compiler (:mod:`apmode.dsl.grammar`),
the seven-level validator (:mod:`apmode.dsl.validation_levels`), the
canonical serializer (:mod:`apmode.dsl.serializer`), and the code-derived
capability matrix (:mod:`apmode.dsl.capabilities`). No command here
duplicates validation/emission logic; every one is a CLI-shaped call into
one of those modules.

Commands:
  apmode formular fmt <spec-file> [--in-place] [--migrate]
  apmode formular lint <spec-file> [--lane]
  apmode formular validate <spec-file> [--level ...] [--data] [--backend] [--policy] [--lane]
  apmode formular explain <spec-file> [--equations]
  apmode formular signature <spec-file>
  apmode formular diff <spec-file-a> <spec-file-b>
  apmode formular lower <spec-file> --backend {nlmixr2,stan,frem} [--out]
  apmode formular compat [<spec-file>] [--backend]

Registered into :data:`apmode.cli.app` via
``app.add_typer(formular_app, name="formular")`` — the same
``add_typer``-on-a-module-level-sub-app pattern as
:mod:`apmode.cli_completion`.
"""

from __future__ import annotations

import enum
import json
from pathlib import Path  # noqa: TC003 — used at runtime in Typer annotations
from typing import TYPE_CHECKING, Annotated

import typer
from lark.exceptions import UnexpectedInput
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from apmode.dsl.capabilities import CapabilityTag, registered_emitters
from apmode.dsl.capabilities import report as capability_report
from apmode.dsl.errors import FormularCompileError
from apmode.dsl.grammar import compile_dsl
from apmode.dsl.lane import Lane
from apmode.dsl.migration import migrate_v06_to_v07
from apmode.dsl.serializer import build_signature, diff_specs, serialize_spec
from apmode.dsl.units import unit_coverage_report
from apmode.dsl.validation_levels import ValidationLevel, ValidationReport, validate
from apmode.governance.policy import GatePolicy

if TYPE_CHECKING:
    from apmode.dsl.ast_models import DSLSpec
    from apmode.dsl.validator import ValidationError

# Sub-app registered into the main typer app at ``apmode.cli`` via
# ``app.add_typer(formular_app, name="formular")``.
formular_app = typer.Typer(
    name="formular",
    help="Inspect, validate, lower, and diff Formular DSL specs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

console = Console()
err_console = Console(stderr=True)


class LowerBackend(enum.StrEnum):
    """Registered emitter names, for ``apmode formular lower --backend``."""

    nlmixr2 = "nlmixr2"
    stan = "stan"
    frem = "frem"


def _compile_or_exit(spec_file: Path) -> DSLSpec:
    """Read + compile ``spec_file``, or exit(1) with a diagnostic on stderr."""
    if not spec_file.exists():
        err_console.print(f"[red bold]Error:[/] spec file not found: {escape(str(spec_file))}")
        raise typer.Exit(code=1)
    text = spec_file.read_text()
    try:
        return compile_dsl(text)
    except UnexpectedInput as exc:
        err_console.print(f"[red bold]Syntax error in {escape(str(spec_file))}:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except FormularCompileError as exc:
        err_console.print(
            f"[red bold]Compile error in {escape(str(spec_file))} "
            f"({exc.code}):[/] {escape(exc.message)}"
        )
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        err_console.print(f"[red bold]Error compiling {escape(str(spec_file))}:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _migrate_command(spec_file: Path, *, in_place: bool) -> None:
    """``fmt --migrate`` body: rewrite legacy text, never compile it first.

    Legacy text is (by definition) not parseable by the current grammar, so
    unlike every other command in this module this one does not route
    through :func:`_compile_or_exit` — it reads the raw file and hands it
    to :func:`apmode.dsl.migration.migrate_v06_to_v07` unconditionally.
    """
    if not spec_file.exists():
        err_console.print(f"[red bold]Error:[/] spec file not found: {escape(str(spec_file))}")
        raise typer.Exit(code=1)

    result = migrate_v06_to_v07(spec_file.read_text())

    for warning in result.warnings:
        err_console.print(f"[yellow bold]Warning:[/] {escape(warning.message)}")

    if in_place:
        spec_file.write_text(result.text)
        console.print(f"[green bold]Migrated in place:[/] {escape(str(spec_file))}")
    else:
        typer.echo(result.text, nl=False)

    if result.warnings:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# fmt
# ---------------------------------------------------------------------------


@formular_app.command("fmt")
def fmt_command(
    spec_file: Annotated[Path, typer.Argument(help="Path to a Formular DSL spec file.")],
    in_place: Annotated[
        bool,
        typer.Option("--in-place", help="Write the canonical text back to spec_file."),
    ] = False,
    migrate: Annotated[
        bool,
        typer.Option(
            "--migrate",
            help=(
                "Best-effort migrate pre-Phase-1 syntax (inline calibration "
                "values, compact CovariateLink) to the current grammar. See "
                "docs/FORMULAR_MIGRATION_v0.6_to_v0.7.md."
            ),
        ),
    ] = False,
) -> None:
    """Re-serialize a spec in canonical block order, or migrate legacy syntax.

    Canonical order: metadata, units, absorption, distribution,
    elimination, variability, covariates, priors,
    observation-or-observations, initial (see
    ``apmode.dsl.serializer.CANONICAL_BLOCK_ORDER``). Prints to stdout by
    default; pass ``--in-place`` to overwrite ``spec_file``.

    With ``--migrate``, ``spec_file`` is treated as pre-Phase-1 text (which
    the compiler can no longer parse) and rewritten with
    ``apmode.dsl.migration.migrate_v06_to_v07`` instead of being compiled —
    see that module and docs/FORMULAR_MIGRATION_v0.6_to_v0.7.md for exactly
    what it does and does not handle. Constructs it cannot safely migrate
    (e.g. a categorical ``CovariateLink``) are left in place with a
    manual-review warning on stderr and a non-zero exit code.
    """
    if migrate:
        _migrate_command(spec_file, in_place=in_place)
        return

    spec = _compile_or_exit(spec_file)
    formatted = serialize_spec(spec)

    err_console.print(
        "[yellow bold]Warning:[/] `formular fmt` reorders top-level blocks into "
        "canonical order. Any external line-number references into the "
        "original file (bookmarks, review comments, source_meta captured "
        "before this run) are invalidated by this reordering."
    )

    if in_place:
        spec_file.write_text(formatted)
        console.print(f"[green bold]Formatted in place:[/] {escape(str(spec_file))}")
    else:
        typer.echo(formatted, nl=False)


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------

_LINT_LEVELS: tuple[ValidationLevel, ...] = (
    ValidationLevel.AST,
    ValidationLevel.SEMANTIC,
    ValidationLevel.LANE_BOUND,
)


def _severity_style(severity: str) -> str:
    return {"error": "red bold", "warning": "yellow", "info": "cyan"}.get(severity, "white")


def _render_errors(errors: list[ValidationError], spec_file: Path) -> None:
    for err in errors:
        style = _severity_style(err.severity)
        span = err.source_span
        loc = f"{span.line_start}:{span.col_start}" if span is not None else "?:?"
        console.print(
            f"  [{style}]{escape(str(spec_file))}:{loc}[/] "
            f"[{style}]{err.code}[/] [{style}]{err.severity}[/]: {escape(err.message)}"
        )
        if err.remediation:
            console.print(f"      [dim]remediation:[/] {escape(err.remediation)}")


def _render_report(report: ValidationReport, spec_file: Path) -> None:
    total = 0
    for level in ValidationLevel:
        if level is ValidationLevel.ALL or level not in report.levels_run:
            continue
        errors = report.by_level.get(level, [])
        total += len(errors)
        skipped_reason = report.skipped_levels.get(level)
        if skipped_reason is not None:
            console.print(f"[bold]{level.value}[/] ([yellow]skipped[/])")
            console.print(f"  [yellow]skipped[/]: {escape(skipped_reason)}")
            continue
        noun = "finding" if len(errors) == 1 else "findings"
        console.print(f"[bold]{level.value}[/] ({len(errors)} {noun})")
        _render_errors(errors, spec_file)
    if total == 0 and not report.skipped_levels:
        console.print("[green bold]No findings.[/]")


@formular_app.command("lint")
def lint_command(
    spec_file: Annotated[Path, typer.Argument(help="Path to a Formular DSL spec file.")],
    lane: Annotated[
        Lane,
        typer.Option("--lane", help="Operating lane for lane-bound checks."),
    ] = Lane.SUBMISSION,
) -> None:
    """Run the ast/semantic/lane-bound validation levels and print span-anchored diagnostics.

    These are the levels that need no extra input beyond the spec and a
    lane — ``data_bound``/``backend_bound``/``policy_bound`` are covered by
    ``apmode formular validate``, which accepts the extra ``--data``/
    ``--backend``/``--policy`` flags those levels need.
    """
    spec = _compile_or_exit(spec_file)
    report = validate(spec, level=list(_LINT_LEVELS), lane=lane)
    _render_report(report, spec_file)
    if not report.ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@formular_app.command("validate")
def validate_command(
    spec_file: Annotated[Path, typer.Argument(help="Path to a Formular DSL spec file.")],
    level: Annotated[
        list[ValidationLevel] | None,
        typer.Option(
            "--level",
            help="Validation level(s) to run. Repeatable. Defaults to `all`.",
        ),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option("--data", help="Bound dataset CSV, for `data_bound` checks."),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Emitter name, for `backend_bound` checks."),
    ] = None,
    policy: Annotated[
        Path | None,
        typer.Option("--policy", help="Gate policy JSON file, for `policy_bound` checks."),
    ] = None,
    lane: Annotated[
        Lane,
        typer.Option("--lane", help="Operating lane."),
    ] = Lane.SUBMISSION,
) -> None:
    """Run `validate()` at the requested levels and print the report level-by-level."""
    spec = _compile_or_exit(spec_file)
    levels = level if level else [ValidationLevel.ALL]

    df = None
    if data is not None:
        import pandas as pd

        try:
            df = pd.read_csv(data)
        except OSError as exc:
            err_console.print(f"[red bold]Error reading --data:[/] {exc}")
            raise typer.Exit(code=1) from exc

    loaded_policy: GatePolicy | None = None
    if policy is not None:
        if not policy.exists():
            err_console.print(f"[red bold]Error:[/] --policy file not found: {policy}")
            raise typer.Exit(code=1)
        try:
            loaded_policy = GatePolicy.model_validate(json.loads(policy.read_text()))
        except (ValueError, TypeError) as exc:
            err_console.print(f"[red bold]Error loading --policy:[/] {exc}")
            raise typer.Exit(code=1) from exc

    report = validate(
        spec, level=levels, lane=lane, data=df, backend=backend, policy=loaded_policy
    )
    _render_report(report, spec_file)
    if not report.ok:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


@formular_app.command("explain")
def explain_command(
    spec_file: Annotated[Path, typer.Argument(help="Path to a Formular DSL spec file.")],
    equations: Annotated[
        bool,
        typer.Option("--equations", help="Print the derived ODE system (Phase 2, P2.3)."),
    ] = False,
) -> None:
    """Print a human-readable summary of a compiled spec's module choices."""
    spec = _compile_or_exit(spec_file)

    if equations:
        from apmode.dsl.equations import build_equations, render_equations

        try:
            system = build_equations(spec)
        except NotImplementedError as exc:
            err_console.print(f"[red bold]Cannot render equations:[/] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(render_equations(system), markup=False, highlight=False)
        return

    from apmode.dsl.serializer import (
        serialize_absorption_module,
        serialize_distribution_module,
        serialize_elimination_module,
        serialize_observation_module,
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    if spec.metadata is not None:
        meta_bits = [
            f"{name}={value!r}"
            for name in ("title", "intent", "context_of_use", "analyte", "version")
            if (value := getattr(spec.metadata, name)) is not None
        ]
        table.add_row("Metadata", ", ".join(meta_bits) or "(empty)")
    else:
        table.add_row("Metadata", "[dim](none)[/]")

    if spec.units is not None:
        coverage = unit_coverage_report(spec)
        units_text = (
            f"time={spec.units.time}, amount={spec.units.amount}, "
            f"concentration={spec.units.concentration}, volume={spec.units.volume} "
            f"[dim]({coverage.status}, {len(coverage.mismatched)} mismatched)[/]"
        )
        table.add_row("Units", units_text)
    else:
        table.add_row("Units", "[dim](not declared)[/]")

    table.add_row("Absorption", serialize_absorption_module(spec.absorption))
    table.add_row("Distribution", serialize_distribution_module(spec.distribution))
    table.add_row("Elimination", serialize_elimination_module(spec.elimination))

    if spec.variability:
        from apmode.dsl.serializer import serialize_variability_item

        table.add_row(
            "Variability",
            "; ".join(serialize_variability_item(item) for item in spec.variability),
        )
    else:
        table.add_row("Variability", "[dim](none)[/]")

    if spec.covariates:
        from apmode.dsl.serializer import serialize_covariate_link

        table.add_row(
            "Covariates", "; ".join(serialize_covariate_link(c) for c in spec.covariates)
        )
    else:
        table.add_row("Covariates", "[dim](none)[/]")

    if spec.priors:
        from apmode.dsl.serializer import serialize_prior

        table.add_row("Priors", "; ".join(serialize_prior(p) for p in spec.priors))
    else:
        table.add_row("Priors", "[dim](none)[/]")

    if spec.observations:
        endpoints = ", ".join(
            f"{name}(dvid={ep.dvid}, prediction={ep.prediction}, "
            f"error={serialize_observation_module(ep.error)})"
            for name, ep in spec.observations.items()
        )
        table.add_row("Observations", endpoints)
    else:
        table.add_row("Observation", serialize_observation_module(spec.observation))

    initial_text = ", ".join(f"{k}={v}" for k, v in sorted(spec.initial.items()))
    table.add_row("Initial", initial_text or "[dim](none)[/]")

    if spec.experimental.node:
        table.add_row("Experimental", "node=True")

    console.print(table)


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------


@formular_app.command("signature")
def signature_command(
    spec_file: Annotated[Path, typer.Argument(help="Path to a Formular DSL spec file.")],
) -> None:
    """Print a compact one-line summary of a spec's module choices (P2.4).

    e.g. ``FO absorption | 1CMT | Linear CL | IIV(CL,V,ka) diag | Prop error``.
    Plain ``typer.echo`` (no rich markup/table) — the line is meant to be
    grep-able and pipeable, unlike ``apmode formular explain``'s table.
    """
    spec = _compile_or_exit(spec_file)
    typer.echo(build_signature(spec))


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


@formular_app.command("diff")
def diff_command(
    spec_file_a: Annotated[Path, typer.Argument(help="First Formular DSL spec file.")],
    spec_file_b: Annotated[Path, typer.Argument(help="Second Formular DSL spec file.")],
) -> None:
    """Compare two specs block-by-block, after canonicalizing block/entry order.

    Reordering top-level blocks (or the entries within
    ``variability:``/``covariates:``/``priors:``) never shows up as a
    diff — only genuine content differences do (see
    ``apmode.dsl.serializer.diff_specs``).
    """
    spec_a = _compile_or_exit(spec_file_a)
    spec_b = _compile_or_exit(spec_file_b)
    diffs = diff_specs(spec_a, spec_b)

    if not diffs:
        console.print("[green bold]No differences[/] (after canonicalizing block order).")
        return

    for block in sorted(diffs):
        a_val, b_val = diffs[block]
        console.print(f"[bold]{block}[/] differs:")
        console.print(f"  [red]- {escape(str(spec_file_a))}:[/] {a_val}")
        console.print(f"  [green]+ {escape(str(spec_file_b))}:[/] {b_val}")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# lower
# ---------------------------------------------------------------------------


_LOWER_LEVELS: tuple[ValidationLevel, ...] = (
    ValidationLevel.AST,
    ValidationLevel.SEMANTIC,
    ValidationLevel.LANE_BOUND,
    ValidationLevel.BACKEND_BOUND,
)


@formular_app.command("lower")
def lower_command(
    spec_file: Annotated[Path, typer.Argument(help="Path to a Formular DSL spec file.")],
    backend: Annotated[
        LowerBackend,
        typer.Option("--backend", help="Target emitter."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write generated code here instead of stdout."),
    ] = None,
    lane: Annotated[
        Lane,
        typer.Option("--lane", help="Operating lane, for lane-bound checks."),
    ] = Lane.SUBMISSION,
    data: Annotated[
        Path | None,
        typer.Option(
            "--data",
            help=(
                "Dataset CSV for FREM covariate summarization "
                "(required with --backend frem, ignored otherwise)."
            ),
        ),
    ] = None,
    frem_covariates: Annotated[
        str | None,
        typer.Option(
            "--frem-covariates",
            help=(
                "Comma-separated covariate column names to summarize into the "
                "joint Omega block (required with --backend frem)."
            ),
        ),
    ] = None,
) -> None:
    """Emit backend code for a spec, after ast/semantic/lane/backend validation.

    Fails fast (exit 1) and prints the full validation report — the same
    ``ast``/``semantic``/``lane_bound`` findings ``apmode formular lint``
    would report (positivity, Erlang max-n, TMDD/elimination compatibility,
    etc.), plus ``backend_bound`` capability-matrix findings — rather than
    silently lowering a semantically invalid spec to broken backend code.
    ``data_bound``/``policy_bound`` are intentionally not run here: they
    need extra input (``--data``/``--policy``) this command does not
    accept for that purpose; use ``apmode formular validate`` for those.
    """
    spec = _compile_or_exit(spec_file)

    report = validate(spec, level=list(_LOWER_LEVELS), lane=lane, backend=backend.value)
    if not report.ok:
        err_console.print(f"[red bold]Cannot lower to {backend.value!r}:[/] validation failed.")
        _render_report(report, spec_file)
        raise typer.Exit(code=1)

    initial_estimates = dict(spec.initial) or None
    code: str
    if backend is LowerBackend.nlmixr2:
        from apmode.dsl.nlmixr2_emitter import emit_nlmixr2

        code = emit_nlmixr2(spec, initial_estimates=initial_estimates)
    elif backend is LowerBackend.stan:
        from apmode.dsl.stan_emitter import emit_stan

        code = emit_stan(spec, initial_estimates=initial_estimates)
    elif backend is LowerBackend.frem:
        code = _lower_frem(spec, initial_estimates, data, frem_covariates)
    else:  # pragma: no cover - exhaustive over LowerBackend
        msg = f"unhandled LowerBackend member: {backend!r}"
        raise AssertionError(msg)

    if out is not None:
        out.write_text(code)
        console.print(f"[green bold]Wrote {backend.value} code to[/] {escape(str(out))}")
    else:
        typer.echo(code, nl=False)


def _lower_frem(
    spec: DSLSpec,
    initial_estimates: dict[str, float] | None,
    data: Path | None,
    frem_covariates: str | None,
) -> str:
    """Resolve ``--data``/``--frem-covariates`` into ``FREMCovariate``s and emit.

    The capability pre-flight in :func:`lower_command` already rejects any
    spec whose ``covariates:`` block is non-empty (``VARIABILITY_COVARIATE_LINK``
    is ``EXPLICITLY_UNSUPPORTED`` for frem — it replaces explicit covariate
    effects with the joint random-effect structure). But
    ``emit_nlmixr2_frem`` unconditionally requires a *non-empty*
    ``FREMCovariate`` list (``ValueError`` otherwise), and those can only be
    computed from real observed data via
    ``apmode.dsl.frem_emitter.summarize_covariates`` — there is no AST-level
    source for "which covariates go into the joint Omega block" once
    ``covariates:`` is stripped. So frem lowering is inherently data-bound;
    fail fast with a clear message rather than let a bare
    ``emit_nlmixr2_frem(spec, [])`` crash with an unrelated-looking
    ``ValueError``.
    """
    if data is None or frem_covariates is None:
        err_console.print(
            "[red bold]Error:[/] --backend frem requires both --data <csv> and "
            "--frem-covariates <col1,col2,...> — FREM's joint covariate "
            "structure is computed from observed baseline data "
            "(mu_init/sigma_init per covariate), not from the DSL spec."
        )
        raise typer.Exit(code=1)

    import pandas as pd

    from apmode.dsl.frem_emitter import emit_nlmixr2_frem, summarize_covariates

    try:
        df = pd.read_csv(data)
    except OSError as exc:
        err_console.print(f"[red bold]Error reading --data:[/] {exc}")
        raise typer.Exit(code=1) from exc

    names = [n.strip() for n in frem_covariates.split(",") if n.strip()]
    if not names:
        err_console.print("[red bold]Error:[/] --frem-covariates is empty.")
        raise typer.Exit(code=1)

    try:
        covariates = summarize_covariates(df, names)
        return emit_nlmixr2_frem(spec, covariates, initial_estimates=initial_estimates)
    except (ValueError, NotImplementedError) as exc:
        err_console.print(f"[red bold]Cannot lower to frem:[/] {exc}")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# compat
# ---------------------------------------------------------------------------


def _print_full_matrix() -> None:
    """Print the raw tag x emitter support-status matrix (no spec bound)."""
    emitters = registered_emitters()
    table = Table(title="Formular capability matrix", box=None, padding=(0, 1))
    table.add_column("tag", style="bold")
    for emitter in emitters:
        table.add_column(emitter.name)

    for tag in sorted(CapabilityTag, key=lambda t: t.value):
        row = [tag.value]
        for emitter in emitters:
            if tag in emitter.supports:
                row.append("[green]supported[/]")
            elif tag in emitter.explicitly_unsupported:
                row.append("[yellow]unsupported[/]")
            else:
                row.append("[red bold]** GAP **[/]")
        table.add_row(*row)
    console.print(table)


def _print_spec_report(report: dict[str, dict[str, str]]) -> None:
    table = Table(title="Formular capability report", box=None, padding=(0, 1))
    table.add_column("tag", style="bold")
    emitter_names = sorted(report)
    for name in emitter_names:
        table.add_column(name)

    tags = sorted({tag for statuses in report.values() for tag in statuses})
    _STYLE = {
        "supported": "green",
        "explicitly_unsupported": "yellow",
        "experimental_no_stable_backend": "cyan",
        "unknown_gap": "red bold",
    }
    for tag in tags:
        row = [tag]
        for name in emitter_names:
            status = report[name].get(tag, "-")
            style = _STYLE.get(status, "")
            row.append(f"[{style}]{status}[/]" if style else status)
        table.add_row(*row)
    console.print(table)


@formular_app.command("compat")
def compat_command(
    spec_file: Annotated[
        Path | None,
        typer.Argument(
            help="Optional spec file. Omit to print the raw full tag x emitter matrix."
        ),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Restrict output to one emitter."),
    ] = None,
) -> None:
    """Print the code-derived capability matrix, optionally scoped to one spec/backend."""
    if spec_file is None:
        _print_full_matrix()
        return

    spec = _compile_or_exit(spec_file)
    report = capability_report(spec)
    if backend is not None:
        if backend not in report:
            err_console.print(
                f"[red bold]Error:[/] unknown --backend {backend!r}; "
                f"known emitters: {sorted(report)}"
            )
            raise typer.Exit(code=1)
        report = {backend: report[backend]}
    _print_spec_report(report)


__all__ = ["formular_app"]
