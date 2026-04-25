# SPDX-License-Identifier: GPL-2.0-or-later
"""``apmode`` console-script entrypoint with typed-error catcher.

This module replaces the ``apmode = "apmode.cli:app"`` direct-Typer
binding with a thin ``main()`` wrapper so exceptions can propagate to a
single, centralised handler. The handler:

1. **Pre-scans ``argv`` for ``--json``** before Click parses anything.
   Click usage errors (unknown command, bad argument) are raised during
   parsing — earlier than any ``@app.callback()`` could fire — so the
   only way to know "the operator wants JSON output" at error time is to
   sniff ``sys.argv`` first.
2. **Runs the Typer app with ``standalone_mode=False``.** That tells
   Click to *propagate* exceptions instead of catching them and calling
   ``sys.exit`` itself. We need that propagation to translate domain
   errors into the typed-envelope format.
3. **Catches in priority order**:

   * ``APModeCLIError`` — render via Rich panel or JSON envelope, exit
     with the class's ``code``.
   * ``click.exceptions.Exit`` — preserve the integer code from existing
     ``raise typer.Exit(N)`` call sites; PR2 ships zero migrations of
     these on purpose, so this branch keeps the legacy CLI working
     identically.
   * ``click.ClickException`` (``UsageError``, ``BadParameter``, …) —
     render as ``kind="usage"`` and exit ``2`` (GNU + Click convention).
   * ``KeyboardInterrupt`` — render as ``user_abort`` and exit ``130``
     (POSIX SIGINT convention).

Anything else escapes — Python's default traceback printing kicks in
and the process exits ``1``. That is intentional: an uncaught exception
class is a bug; hiding it behind a generic envelope would make it
harder to diagnose.
"""

from __future__ import annotations

import json
import sys
from typing import NoReturn

import click
from rich.console import Console

from apmode._json_ctx import is_json_output, set_json_output
from apmode.cli_errors import APModeCLIError, UserAbortError

# Stderr console so error output never collides with ``--json`` payloads
# on stdout. Rich's stderr console honours ``NO_COLOR`` and detects TTY
# state automatically.
_err_console = Console(stderr=True)


def _argv_has_json_flag(argv: list[str]) -> bool:
    """Coarse scan: ``--json`` before any ``--`` sentinel flips JSON mode.

    The scan is intentionally permissive — placing ``--json`` before *or*
    after the failing argument both work. Click's own parser would not
    have reached the flag if it raised a usage error mid-parse, so this
    sniff is the only reliable signal. ``--json=true`` / ``--json=false``
    are ignored: the project's commands all model ``--json`` as a flag.

    The POSIX ``--`` sentinel terminates option parsing; tokens after it
    are positional. Honouring this here means a future command like
    ``apmode foo -- --json`` (where ``--json`` is a literal positional
    value, not a flag) does not silently flip the renderer.
    """
    try:
        end = argv.index("--")
    except ValueError:
        return "--json" in argv
    return "--json" in argv[:end]


def _emit_typed_error(exc: APModeCLIError, *, json_mode: bool) -> None:
    """Print ``exc`` either as a JSON envelope (stdout) or Rich panel."""
    if json_mode:
        # Stdout per the envelope contract; ``ensure_ascii=False`` keeps
        # non-ASCII messages readable without ``\uXXXX`` escapes.
        sys.stdout.write(json.dumps(exc.to_envelope(), ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    _err_console.print(f"[red bold]Error[/] [dim]({exc.kind}, exit {exc.code})[/]: {exc.message}")
    if exc.details:
        # ``json.dumps`` keeps the diagnostic dict compact and round-trip
        # safe; rendering through Rich keeps the leading "details:" label
        # consistent with the rest of the project's error styling.
        _err_console.print(f"[dim]details: {json.dumps(exc.details, ensure_ascii=False)}[/]")


def _emit_click_error(exc: click.ClickException, *, json_mode: bool) -> None:
    """Translate a Click ``UsageError`` / ``BadParameter`` / ``FileError`` etc.

    Click already has its own ``format_message()`` and ``show()`` paths;
    we use ``format_message`` (raw text) and re-render so the human and
    JSON variants share a wording. The exit code is read from the
    exception (``UsageError`` → 2, ``FileError`` → 1, custom subclasses
    may override) instead of being hard-coded — that keeps wrapper
    scripts that branch on the actual ``ClickException.exit_code``
    contract working as Click intended.
    """
    if json_mode:
        envelope = {
            "ok": False,
            "error": {
                "kind": "usage",
                "code": exc.exit_code,
                "message": exc.format_message(),
                "details": {},
            },
        }
        sys.stdout.write(json.dumps(envelope, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        return
    # Defer to Click's own renderer so the wording matches what an
    # operator would have seen pre-PR2 — keeps muscle memory intact.
    exc.show(file=sys.stderr)


def main(argv: list[str] | None = None) -> NoReturn:
    """Console-script entrypoint. Sets ``--json`` mode, runs Typer app.

    The function is annotated ``NoReturn`` because every exit branch
    calls ``sys.exit``; the type-checker can therefore prove that any
    code below an ``app(...)``-style invocation is unreachable.
    """
    if argv is None:
        argv = sys.argv[1:]
    json_mode = _argv_has_json_flag(argv)
    set_json_output(json_mode)

    # Local import: keeps ``apmode.cli`` (a heavyweight module that pulls
    # in pandas, pydantic, lark, the entire backend stack) out of the
    # cold-start path until the entrypoint has actually been invoked.
    from apmode.cli import app

    try:
        # standalone_mode=False → Click propagates UsageError / unknown
        # exceptions instead of catching them and calling sys.exit
        # itself. Without this, our try/except below would never see
        # APModeCLIError.
        #
        # CAVEAT (Click 8.x): ``raise typer.Exit(N)`` from a command
        # body is *not* re-raised in non-standalone mode — Click
        # converts it to a non-zero *return value* of ``N``. We capture
        # that below the try/except and forward it through ``sys.exit``,
        # which is the bridge that keeps every existing
        # ``raise typer.Exit(N)`` call site working identically while we
        # ship PR2 without migrating any of them.
        return_code = app(args=argv, standalone_mode=False)
    except APModeCLIError as exc:
        _emit_typed_error(exc, json_mode=json_mode)
        sys.exit(exc.code)
    except click.exceptions.Exit as exc:  # pragma: no cover - defensive
        # Click 8.x converts Exit to a return value in non-standalone
        # mode (handled below), but older / patched versions may still
        # re-raise. Keep the catch so a future Click change cannot
        # silently break exit-code propagation.
        sys.exit(exc.exit_code)
    except click.ClickException as exc:
        _emit_click_error(exc, json_mode=json_mode)
        sys.exit(exc.exit_code)
    except click.Abort:
        # Raised when the user hits Ctrl-C inside ``click.confirm`` /
        # ``click.prompt`` (which Click translates from KeyboardInterrupt
        # internally before re-raising as ``Abort``). Without this branch
        # the exception escapes every catch and the operator sees a
        # raw traceback — same UX bug as an uncaught KeyboardInterrupt.
        _emit_typed_error(UserAbortError("aborted at prompt"), json_mode=json_mode)
        sys.exit(130)
    except KeyboardInterrupt:
        _emit_typed_error(UserAbortError("interrupted by user"), json_mode=json_mode)
        sys.exit(130)

    # Honour the typer.Exit-as-return-value semantics described above.
    # ``return_code`` is ``int`` for ``raise typer.Exit(N)`` and ``None``
    # (or a non-int return value from the command function) for normal
    # completion.
    if isinstance(return_code, int):
        sys.exit(return_code)
    sys.exit(0)


# Re-export ``is_json_output`` so call-site migrations (PR6+) can do
# ``from apmode.__main__ import is_json_output`` without taking a
# dependency on the underlying module — the entrypoint is the
# user-facing shape.
__all__ = ["is_json_output", "main"]


if __name__ == "__main__":
    main()
