# SPDX-License-Identifier: GPL-2.0-or-later
"""``apmode attest <bundle_dir>`` — human-in-the-loop reviewer attestation.

Writes ``attestation.json`` into a *sealed* reproducibility bundle,
recording that a named human reviewer examined the bundle's gate
decisions and diagnostics and reached a decision (approve / approve
with conditions / reject), plus any explicit overrides of automated
gate verdicts.

Kept out of :mod:`apmode.cli` following the ``cli_formular.py`` /
``bundle/rocrate/cli_hooks.py`` precedent of keeping large or
self-contained CLI surfaces in their own module, registered onto the
main Typer app via ``app.command()``.

This command makes attestation *possible and auditable*; it does not
make it *mandatory* — enforcing attestation as a blocking Gate check
before export is an explicitly deferred follow-up (would require a new
``policies/*.json`` knob), not part of this change.

Commands:
  apmode attest <bundle-dir> --reviewer-id ID --reviewer-role ROLE
      --decision {approved,approved_with_conditions,rejected}
      --rationale TEXT
      [--gate-override GATE_ID:CHECK_ID:ORIGINAL_PASSED:JUSTIFICATION[:AUTHORIZED_BY]]...
      [--force] [--json]
"""

from __future__ import annotations

import enum
import json
import re
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 — used at runtime in Typer annotations
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from apmode.bundle.emitter import BundleEmitter, BundleNotSealedError
from apmode.bundle.models import GateOverride, ReviewerAttestation

console = Console()
err_console = Console(stderr=True)


class AttestDecision(enum.StrEnum):
    """Mirrors ``ReviewerAttestation.decision``'s ``Literal`` values."""

    approved = "approved"
    approved_with_conditions = "approved_with_conditions"
    rejected = "rejected"


_TRUE_TOKENS = frozenset({"true", "1", "yes", "pass", "passed"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "fail", "failed"})
_AUTHORIZED_BY_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,128}$")


class GateOverrideParseError(ValueError):
    """Raised when a ``--gate-override`` value does not parse."""


def _parse_bool_token(token: str, *, raw: str) -> bool:
    lowered = token.strip().lower()
    if lowered in _TRUE_TOKENS:
        return True
    if lowered in _FALSE_TOKENS:
        return False
    msg = (
        f"invalid --gate-override {raw!r}: ORIGINAL_PASSED must be one of "
        f"{sorted(_TRUE_TOKENS | _FALSE_TOKENS)}, got {token!r}"
    )
    raise GateOverrideParseError(msg)


def _parse_gate_override(raw: str, *, default_authorized_by: str) -> GateOverride:
    """Parse ``GATE_ID:CHECK_ID:ORIGINAL_PASSED:JUSTIFICATION[:AUTHORIZED_BY]``.

    ``AUTHORIZED_BY`` is optional and defaults to the attesting
    reviewer (``--reviewer-id``) when omitted — the common case is the
    same person both reviews and authorizes the override. The first
    three colons delimit the required fields. A final colon-delimited
    segment is treated as ``AUTHORIZED_BY`` only when it looks like an
    identifier; otherwise it remains part of the free-text justification.
    """
    parts = raw.split(":", 3)
    if len(parts) != 4:
        msg = (
            f"invalid --gate-override {raw!r}: expected "
            "GATE_ID:CHECK_ID:ORIGINAL_PASSED:JUSTIFICATION[:AUTHORIZED_BY]"
        )
        raise GateOverrideParseError(msg)
    gate_id, check_id, original_passed_raw, remainder = parts
    justification = remainder
    authorized_by = default_authorized_by
    if ":" in remainder:
        maybe_justification, maybe_authorized_by = remainder.rsplit(":", 1)
        if maybe_justification and _AUTHORIZED_BY_RE.fullmatch(maybe_authorized_by):
            justification = maybe_justification
            authorized_by = maybe_authorized_by
    if not gate_id or not check_id:
        msg = f"invalid --gate-override {raw!r}: GATE_ID and CHECK_ID must be non-empty"
        raise GateOverrideParseError(msg)
    if not justification:
        msg = f"invalid --gate-override {raw!r}: JUSTIFICATION must be non-empty"
        raise GateOverrideParseError(msg)
    original_passed = _parse_bool_token(original_passed_raw, raw=raw)
    return GateOverride(
        gate_id=gate_id,
        check_id=check_id,
        original_passed=original_passed,
        override_justification=justification,
        authorized_by=authorized_by,
    )


def attest(
    bundle_dir: Annotated[
        Path,
        typer.Argument(help="Path to a sealed run bundle directory."),
    ],
    reviewer_id: Annotated[
        str,
        typer.Option("--reviewer-id", help="Identifier of the attesting reviewer."),
    ],
    reviewer_role: Annotated[
        str,
        typer.Option("--reviewer-role", help="Reviewer's role (e.g. 'PK reviewer')."),
    ],
    decision: Annotated[
        AttestDecision,
        typer.Option("--decision", help="Attestation decision."),
    ],
    rationale: Annotated[
        str,
        typer.Option("--rationale", help="Free-text rationale for the decision."),
    ],
    gate_override: Annotated[
        list[str],
        typer.Option(
            "--gate-override",
            help=(
                "Override an automated gate check's verdict. Repeatable. Format: "
                "GATE_ID:CHECK_ID:ORIGINAL_PASSED:JUSTIFICATION[:AUTHORIZED_BY] "
                "(ORIGINAL_PASSED is true/false; AUTHORIZED_BY defaults to "
                "--reviewer-id when omitted)."
            ),
        ),
    ] = [],  # noqa: B006 — Typer requires a literal default for CLI introspection
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Overwrite an existing attestation.json."),
    ] = False,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit {'ok': ..., 'path': ...} as JSON to stdout."),
    ] = False,
) -> None:
    """Attest to a sealed reproducibility bundle (human-in-the-loop sign-off).

    Writes ``<bundle_dir>/attestation.json``. Requires the bundle to
    already be sealed (contain ``_COMPLETE``) — attestation reviews a
    completed run, it does not participate in producing one. The
    sidecar is excluded from the sealed-bundle digest, so writing (or,
    with ``--force``, re-writing) it never invalidates ``_COMPLETE``.
    """
    if not bundle_dir.exists():
        _fail(
            f"bundle_dir not found: {bundle_dir}",
            output_json=output_json,
            bundle_dir=bundle_dir,
        )
    if not bundle_dir.is_dir():
        _fail(
            f"bundle_dir is not a directory: {bundle_dir}",
            output_json=output_json,
            bundle_dir=bundle_dir,
        )

    try:
        overrides = [
            _parse_gate_override(raw, default_authorized_by=reviewer_id) for raw in gate_override
        ]
    except GateOverrideParseError as exc:
        _fail(str(exc), output_json=output_json, bundle_dir=bundle_dir)
        return  # pragma: no cover - _fail always raises typer.Exit

    attestation = ReviewerAttestation(
        reviewer_id=reviewer_id,
        reviewer_role=reviewer_role,
        timestamp=datetime.now(tz=UTC).isoformat(),
        decision=decision.value,
        rationale=rationale,
        gate_overrides=overrides,
    )

    # ``bundle_dir`` (as given on the CLI) *is* the run directory, so
    # split it into (parent, run_id) to reuse the public constructor
    # rather than poking at ``BundleEmitter``'s private attributes.
    # ``initialize()`` is deliberately not called: a sealed bundle
    # already exists on disk and ``initialize()`` would refuse to
    # touch it anyway (``BundleAlreadySealedError``), plus the CLI
    # only ever calls ``write_attestation`` here.
    emitter = BundleEmitter(bundle_dir.parent, run_id=bundle_dir.name)

    try:
        path = emitter.write_attestation(attestation, force=force)
    except BundleNotSealedError as exc:
        _fail(str(exc), output_json=output_json, bundle_dir=bundle_dir)
        return  # pragma: no cover - _fail always raises typer.Exit
    except FileExistsError as exc:
        _fail(str(exc), output_json=output_json, bundle_dir=bundle_dir)
        return  # pragma: no cover - _fail always raises typer.Exit

    if output_json:
        print(json.dumps({"ok": True, "path": str(path)}, indent=2))
        return

    console.print(f"[green bold]Attestation written.[/] {escape(str(path))}")
    console.print(f"  decision: [bold]{decision.value}[/]")
    if overrides:
        console.print(f"  gate overrides: {len(overrides)}")


def _fail(message: str, *, output_json: bool, bundle_dir: Path) -> None:
    if output_json:
        print(json.dumps({"ok": False, "error": message, "bundle_dir": str(bundle_dir)}, indent=2))
    else:
        err_console.print(f"[red bold]Error:[/] {escape(message)}")
    raise typer.Exit(code=1)


def register_attest_command(app: typer.Typer) -> None:
    """Attach ``apmode attest`` onto the main Typer app."""
    app.command("attest")(attest)


__all__ = ["AttestDecision", "GateOverrideParseError", "register_attest_command"]
