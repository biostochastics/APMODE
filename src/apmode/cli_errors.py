# SPDX-License-Identifier: GPL-2.0-or-later
"""Typed CLI error hierarchy with stable exit codes and JSON envelope.

Each subclass carries a class-level ``kind`` (machine-readable identifier)
and ``code`` (process exit code). The entrypoint catcher in
``apmode.__main__`` translates an exception into either a Rich panel on
stderr (default) or a JSON envelope on stdout (when ``--json`` is set):

    {"ok": false, "error": {"kind": "bundle_not_found", "code": 10,
                            "message": "...", "details": {...}}}

Exit-code policy (locked):

    0    success
    1    unhandled exception (generic)
    2    Click usage error — bad argument, unknown command, missing value
    10   BundleNotFoundError
    11   BundleInvalidError
    12   PolicyValidationError
    13   BackendUnavailableError
    14   ConfigError
    130  UserAbortError (SIGINT / Ctrl-C)

Why ``2`` is reserved for Click. ``2`` is the GNU coreutils + Click + most
shells convention for "you typed the command wrong". Reusing it for a
domain error like ``BundleNotFoundError`` would surprise anyone running

    apmode validate ./bundle || handle_invalid_args $?

so the domain codes start at ``10``. The numeric layout has 8 reserved
slots before ``20`` for any future Bayesian / RO-Crate error classes.

Migration. Existing ``raise typer.Exit(N)`` call sites in ``cli.py`` are
*preserved* by this PR — they continue to work because the entrypoint
catches ``click.exceptions.Exit`` and forwards the integer through
``sys.exit``. Subsequent PRs swap individual call sites over to the typed
classes and the CI ratchet at ``scripts/check_typer_exit_count.py``
prevents the count from regressing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Mapping

# JSON-able value type — the ``details`` payload on every error must be
# JSON-serialisable so the envelope round-trips through ``json.dumps``.
# The recursive-shaped form would be more accurate but mypy + pydantic
# both choke on truly recursive ``Union`` aliases here, and the practical
# need (error details are flat dictionaries 99% of the time) is met by
# allowing nested dicts / lists with leaf scalars.
JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list[Any] | dict[str, Any]


class APModeCLIError(Exception):
    """Base class for CLI-surfaced typed errors.

    Subclasses set two class-level attributes:

    * ``kind`` — machine-readable identifier (snake_case). The value is
      stable: scripts that branch on ``error.kind`` must keep working
      across versions.
    * ``code`` — process exit code (see module docstring for policy).
    """

    kind: ClassVar[str] = "apmode_cli_error"
    code: ClassVar[int] = 1

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, JSONValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        # Normalise to a plain dict so callers cannot mutate a frozen
        # ``Mapping`` later and surprise the JSON renderer.
        self.details: dict[str, JSONValue] = dict(details) if details else {}

    def to_envelope(self) -> dict[str, Any]:
        """Render the canonical ``{"ok": false, "error": {...}}`` payload."""
        return {
            "ok": False,
            "error": {
                "kind": self.kind,
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class BundleNotFoundError(APModeCLIError):
    """Bundle directory or file does not exist on disk."""

    kind: ClassVar[str] = "bundle_not_found"
    code: ClassVar[int] = 10


class BundleInvalidError(APModeCLIError):
    """Bundle exists but is structurally invalid — missing ``_COMPLETE``,
    JSONL integrity break, schema mismatch, digest verification failure."""

    kind: ClassVar[str] = "bundle_invalid"
    code: ClassVar[int] = 11


class PolicyValidationError(APModeCLIError):
    """Gate-policy JSON failed schema validation or version mismatch."""

    kind: ClassVar[str] = "policy_validation"
    code: ClassVar[int] = 12


class BackendUnavailableError(APModeCLIError):
    """Required backend toolchain is not installed or not reachable.

    Examples: ``apmode run --backend bayesian_stan`` without cmdstanpy,
    or ``apmode serve`` without the ``[api]`` extras."""

    kind: ClassVar[str] = "backend_unavailable"
    code: ClassVar[int] = 13


class ConfigError(APModeCLIError):
    """Invalid runtime configuration that is not policy-shaped — bad env
    var value, missing API key, contradictory flags."""

    kind: ClassVar[str] = "config_error"
    code: ClassVar[int] = 14


class UserAbortError(APModeCLIError):
    """Translated from ``KeyboardInterrupt`` — user hit Ctrl-C."""

    kind: ClassVar[str] = "user_abort"
    code: ClassVar[int] = 130


# Tuple-of-classes export so the entrypoint catcher can do ``except
# APMODE_ERRORS`` if a future change wants to discriminate by base. Today
# we just catch ``APModeCLIError`` directly.
APMODE_ERRORS: tuple[type[APModeCLIError], ...] = (
    BundleNotFoundError,
    BundleInvalidError,
    PolicyValidationError,
    BackendUnavailableError,
    ConfigError,
    UserAbortError,
)
