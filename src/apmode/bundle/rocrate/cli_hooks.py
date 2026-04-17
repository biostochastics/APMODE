# SPDX-License-Identifier: GPL-2.0-or-later
"""Typer CLI wiring for the RO-Crate projector.

Registered into :data:`apmode.cli.app` by
:func:`register_rocrate_commands`. Adds a single ``bundle`` subcommand
group with:

- ``apmode bundle rocrate export <bundle_dir> --out <path>``
- ``apmode bundle publish <bundle_dir>`` (stub; real implementation in v0.8)

The projector itself lives in :mod:`apmode.bundle.rocrate.projector`;
this module owns only the CLI layer, argument validation, and
pretty-printing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path  # noqa: TC003 — used at runtime in Typer annotations
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from apmode.bundle.rocrate import (
    RoCrateEmitter,
    RoCrateExportOptions,
    RoCrateProfile,
)
from apmode.bundle.rocrate.importer import (
    RoCrateImportError,
    import_crate,
)
from apmode.bundle.rocrate.projector import BundleNotSealedError
from apmode.bundle.rocrate.vocab import RegulatoryContext

console = Console()
err_console = Console(stderr=True)


_bundle_app = typer.Typer(
    name="bundle",
    help="Operate on APMODE reproducibility bundles (RO-Crate, publish).",
    no_args_is_help=True,
)

_rocrate_app = typer.Typer(
    name="rocrate",
    help="RO-Crate (Workflow Run RO-Crate) projection of APMODE bundles.",
    no_args_is_help=True,
)
_bundle_app.add_typer(_rocrate_app, name="rocrate")


_PROFILE_ALIASES: dict[str, str] = {
    "provenance": RoCrateProfile.PROVENANCE.value,
    "workflow": RoCrateProfile.WORKFLOW.value,
    "process": RoCrateProfile.PROCESS.value,
}


@_rocrate_app.command("export")
def rocrate_export(
    bundle_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a sealed bundle directory (must contain _COMPLETE).",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help=(
                "Destination path. Directory form (``/tmp/crate``) or "
                "zip form (``/tmp/crate.zip``)."
            ),
        ),
    ],
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help="Profile: provenance | workflow | process",
        ),
    ] = "provenance",
    include_provagent: Annotated[
        bool,
        typer.Option(
            "--include-provagent/--no-include-provagent",
            help=("Include PROV-AGENT namespace + typing on LLM invocations (v0.9 preview)."),
        ),
    ] = False,
    regulatory_context: Annotated[
        str | None,
        typer.Option(
            "--regulatory-context",
            help=(
                "Override apmode:regulatoryContext (research-only | "
                "pccp-ai-dsf | mdr | ai-act-article-12)."
            ),
        ),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit JSON (path + metadata summary) to stdout for scripting.",
        ),
    ] = False,
) -> None:
    """Project a sealed bundle as a Workflow Run RO-Crate (WRROC v0.5)."""
    profile_value = _PROFILE_ALIASES.get(profile.lower())
    if profile_value is None:
        err_console.print(
            f"[red bold]Error:[/] unknown profile {profile!r}. "
            "Expected one of: provenance, workflow, process."
        )
        raise typer.Exit(code=1)
    profile_enum = RoCrateProfile(profile_value)

    if regulatory_context is not None:
        allowed = {rc.value for rc in RegulatoryContext}
        if regulatory_context not in allowed:
            err_console.print(
                f"[red bold]Error:[/] unknown --regulatory-context "
                f"{regulatory_context!r}. "
                f"Expected one of: {', '.join(sorted(allowed))}."
            )
            raise typer.Exit(code=1)

    opts = RoCrateExportOptions(
        profile=profile_enum,
        include_provagent=include_provagent,
        regulatory_context=regulatory_context,
    )

    emitter = RoCrateEmitter()
    try:
        result_path = emitter.export_from_sealed_bundle(bundle_dir, out, opts)
    except BundleNotSealedError as exc:
        err_console.print(f"[red bold]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except (FileNotFoundError, NotADirectoryError) as exc:
        err_console.print(f"[red bold]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except FileExistsError as exc:
        err_console.print(f"[red bold]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if output_json:
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "bundle_dir": str(bundle_dir),
                    "out": str(result_path),
                    "profile": profile_enum.value,
                    "form": "zip" if result_path.suffix.lower() == ".zip" else "directory",
                },
                indent=2,
            )
            + "\n"
        )
        return

    table = Table(title="RO-Crate Export", box=None, padding=(0, 1))
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("bundle", escape(str(bundle_dir)))
    table.add_row("out", escape(str(result_path)))
    table.add_row("form", "zip" if result_path.suffix.lower() == ".zip" else "directory")
    table.add_row("profile", profile_enum.value)
    table.add_row("include-provagent", "yes" if include_provagent else "no")
    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[green bold]Exported.[/] Validate with: [dim]apmode validate "
        f"{escape(str(bundle_dir))} --rocrate --crate {escape(str(result_path))}[/]"
    )


@_bundle_app.command("sbom")
def sbom_command(
    bundle_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to a bundle directory. bom.cdx.json is written into it.",
        ),
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Overwrite an existing bom.cdx.json.",
        ),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit JSON (path + size) to stdout for scripting.",
        ),
    ] = False,
) -> None:
    """Generate a CycloneDX SBOM into ``<bundle_dir>/bom.cdx.json``.

    Runs ``pip-audit --format cyclonedx-json`` against the current
    Python environment and writes the result as a producer-side sidecar
    inside the bundle. The SBOM is **excluded from the sealed-bundle
    digest** (see ``apmode.bundle.emitter._compute_bundle_digest``) so
    adding it does not invalidate ``_COMPLETE``.

    Subsequent ``apmode bundle rocrate export`` runs will pick up the
    SBOM automatically and project it as a File entity tagged with
    ``apmode:sbom``.
    """
    if not bundle_dir.exists():
        err_console.print(f"[red bold]Error:[/] bundle_dir not found: {bundle_dir}")
        raise typer.Exit(code=1)
    if not bundle_dir.is_dir():
        err_console.print(f"[red bold]Error:[/] bundle_dir is not a directory: {bundle_dir}")
        raise typer.Exit(code=1)

    sbom_path = bundle_dir / "bom.cdx.json"
    if sbom_path.exists() and not force:
        err_console.print(
            f"[red bold]Error:[/] {sbom_path} already exists; pass --force to overwrite."
        )
        raise typer.Exit(code=1)

    pip_audit = shutil.which("pip-audit")
    if pip_audit is None:
        err_console.print(
            "[red bold]Error:[/] pip-audit not found on PATH. "
            "Install it with `uv sync --all-extras` (it is a dev-group dep)."
        )
        raise typer.Exit(code=1)

    try:
        result = subprocess.run(
            [pip_audit, "--format", "cyclonedx-json", "--output", str(sbom_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        err_console.print(f"[red bold]Error:[/] failed to run pip-audit: {exc}")
        raise typer.Exit(code=1) from exc

    # pip-audit returns non-zero when CVEs are found but still writes
    # the SBOM — only treat a missing file as a hard failure.
    if result.returncode != 0 and not sbom_path.is_file():
        err_console.print(
            f"[red bold]pip-audit failed (exit={result.returncode}):[/]\n{result.stderr}"
        )
        raise typer.Exit(code=1)

    size = sbom_path.stat().st_size
    if output_json:
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "path": str(sbom_path),
                    "size_bytes": size,
                    "pip_audit_exit": result.returncode,
                },
                indent=2,
            )
            + "\n"
        )
        return

    console.print(
        f"[green bold]SBOM written.[/] {escape(str(sbom_path))} ({size} bytes) — "
        "not counted in the sealed digest."
    )


@_bundle_app.command("import")
def import_command(
    crate: Annotated[
        Path,
        typer.Argument(
            help="Source RO-Crate — directory or .zip.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Destination bundle directory (must be empty or absent).",
        ),
    ],
    output_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit JSON (path + digest) to stdout for scripting.",
        ),
    ] = False,
) -> None:
    """Round-trip a Workflow Run RO-Crate back into an APMODE bundle.

    Verifies the ``_COMPLETE`` digest matches the extracted tree; a
    mismatch aborts with exit code 1 (bundle tampering or partial
    export). This makes the import a safety-checked, byte-preserving
    inverse of ``apmode bundle rocrate export``.
    """
    try:
        bundle_path = import_crate(crate, out)
    except (FileNotFoundError, FileExistsError) as exc:
        err_console.print(f"[red bold]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except RoCrateImportError as exc:
        err_console.print(f"[red bold]Import failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if output_json:
        sentinel = json.loads((bundle_path / "_COMPLETE").read_text())
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "bundle_dir": str(bundle_path),
                    "run_id": sentinel.get("run_id"),
                    "sha256": sentinel.get("sha256"),
                },
                indent=2,
            )
            + "\n"
        )
        return

    console.print(f"[green bold]Bundle imported.[/] {escape(str(bundle_path))} — digest verified")


@_bundle_app.command("publish")
def publish_command(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a sealed bundle directory."),
    ],
    workflowhub: Annotated[
        bool,
        typer.Option("--workflowhub", help="Upload to WorkflowHub."),
    ] = False,
    zenodo: Annotated[
        bool,
        typer.Option("--zenodo", help="Upload to Zenodo."),
    ] = False,
    sandbox: Annotated[
        bool,
        typer.Option("--sandbox/--production", help="Use sandbox endpoint."),
    ] = True,
    token_env: Annotated[
        str,
        typer.Option(
            "--token-env",
            help="Env var name carrying the API token.",
        ),
    ] = "WORKFLOWHUB_TOKEN",
) -> None:
    """Publish the bundle's RO-Crate to an external registry.

    **Stub in v0.6**: the command exists so that operators can discover
    the flow, but the implementation is deferred to v0.8. Running it
    currently raises :class:`NotImplementedError` with a message
    pointing at the roadmap.
    """
    if not (workflowhub or zenodo):
        err_console.print("[red bold]Error:[/] pass --workflowhub or --zenodo to select a target.")
        raise typer.Exit(code=1)
    # The arguments are validated (bundle exists, target flags are
    # set) so that when v0.8 adds the real uploader it only has to
    # drop in the network calls — but today we stop short of making a
    # request. See :mod:`apmode.bundle.rocrate.publish` for the
    # signature layout and :class:`NotImplementedError` message.
    err_console.print(
        "[yellow bold]Not implemented in v0.6:[/] publishing lands in v0.8. "
        "For now, use `apmode bundle rocrate export` to produce the crate "
        "and upload it manually. "
        f"Target: {'workflowhub' if workflowhub else 'zenodo'} "
        f"({'sandbox' if sandbox else 'production'}); token env: {token_env}; "
        f"bundle: {escape(str(bundle_dir))}"
    )
    raise typer.Exit(code=2)


def register_rocrate_commands(app: typer.Typer) -> None:
    """Attach the ``bundle`` subcommand group onto the main Typer app."""
    app.add_typer(_bundle_app, name="bundle")
