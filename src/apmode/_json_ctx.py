# SPDX-License-Identifier: GPL-2.0-or-later
"""``--json`` output mode propagated via :class:`contextvars.ContextVar`.

The Typer entrypoint (``apmode.__main__.main``) sets this flag *before*
Click parses ``argv`` — by sniffing ``"--json"`` in ``sys.argv`` directly.
That is necessary because Click usage errors (unknown command, missing
argument, bad value) are raised during parsing, *before* any
``@app.callback()`` would have a chance to run. Without the pre-scan, the
error renderer could not know whether to emit a Rich panel on stderr or a
JSON envelope on stdout.

A ``ContextVar`` (rather than a module-global) keeps the value
asyncio-safe — useful in case the CLI ever drives concurrent subcommands
through a shared event loop. The default is ``False`` so importing this
module from a library context (e.g. from a unit test that doesn't go
through ``main``) does not silently flip every command into JSON mode.
"""

from __future__ import annotations

from contextvars import ContextVar

# Name uses the ``apmode-`` prefix so it shows up correctly in any future
# ``contextvars.copy_context()`` introspection — ContextVar's ``name`` is
# advisory but the prefix avoids collisions with library code.
_JSON_OUTPUT: ContextVar[bool] = ContextVar("apmode-json-output", default=False)


def set_json_output(value: bool) -> None:
    """Mark the current context as ``--json`` (or back to human-readable)."""
    _JSON_OUTPUT.set(value)


def is_json_output() -> bool:
    """True iff the current context is rendering machine-readable JSON.

    Error renderers consult this to choose between a Rich panel on stderr
    and a ``{"ok": false, "error": {...}}`` envelope on stdout.
    """
    return _JSON_OUTPUT.get()
