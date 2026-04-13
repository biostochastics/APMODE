# SPDX-License-Identifier: GPL-2.0-or-later
"""APMODE CLI entry point (Typer)."""

import typer

app = typer.Typer(name="apmode", help="Adaptive Pharmacokinetic Model Discovery Engine")


@app.command()
def version() -> None:
    """Print APMODE version."""
    from apmode import __version__

    typer.echo(f"apmode {__version__}")
