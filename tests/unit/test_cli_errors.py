# SPDX-License-Identifier: GPL-2.0-or-later
"""Unit tests for ``apmode.cli_errors`` typed-error hierarchy.

The hierarchy is small and pure (no I/O), so the tests pin the contract
that downstream consumers — wrapper scripts running ``apmode validate ||
handle_invalid``, the JSON-envelope renderer, and PR6+ migrations of
existing ``raise typer.Exit`` call sites — depend on:

* ``kind`` is stable and snake_case (script branching key).
* ``code`` follows the locked exit-code policy (10/11/12/13/14/130).
* ``to_envelope()`` round-trips through ``json.dumps`` cleanly.
* ``details`` is a defensively-copied dict (cannot be mutated through
  the original ``Mapping`` after raising).
"""

from __future__ import annotations

import json
import re
from types import MappingProxyType

import pytest

from apmode.cli_errors import (
    APMODE_ERRORS,
    APModeCLIError,
    BackendUnavailableError,
    BundleInvalidError,
    BundleNotFoundError,
    ConfigError,
    PolicyValidationError,
    UserAbortError,
)

# ---------------------------------------------------------------------------
# Class-level metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "expected_kind", "expected_code"),
    [
        (BundleNotFoundError, "bundle_not_found", 10),
        (BundleInvalidError, "bundle_invalid", 11),
        (PolicyValidationError, "policy_validation", 12),
        (BackendUnavailableError, "backend_unavailable", 13),
        (ConfigError, "config_error", 14),
        (UserAbortError, "user_abort", 130),
    ],
)
def test_class_metadata_matches_locked_policy(
    cls: type[APModeCLIError], expected_kind: str, expected_code: int
) -> None:
    """Exit-code policy is part of the public CLI contract — pin it."""
    assert cls.kind == expected_kind
    assert cls.code == expected_code


def test_kinds_are_snake_case() -> None:
    """Scripts branch on ``error.kind``; mixed case would be a foot-gun."""
    snake_re = re.compile(r"^[a-z][a-z0-9_]*$")
    for cls in APMODE_ERRORS:
        assert snake_re.match(cls.kind), f"{cls.__name__} kind '{cls.kind}' is not snake_case"


def test_kinds_are_unique() -> None:
    kinds = [cls.kind for cls in APMODE_ERRORS]
    assert len(set(kinds)) == len(kinds), f"duplicate kind in {kinds}"


def test_codes_are_unique_and_outside_reserved_range() -> None:
    """Exit codes must avoid 0/1/2 (success / generic / Click usage)."""
    codes = [cls.code for cls in APMODE_ERRORS]
    assert len(set(codes)) == len(codes), f"duplicate code in {codes}"
    for code in codes:
        assert code >= 10, f"code {code} clashes with reserved 0/1/2"


# ---------------------------------------------------------------------------
# Construction + envelope rendering
# ---------------------------------------------------------------------------


def test_message_is_attached_and_str_able() -> None:
    exc = BundleNotFoundError("./missing")
    assert exc.message == "./missing"
    assert str(exc) == "./missing"  # superclass behaviour


def test_details_default_to_empty_dict() -> None:
    exc = BundleInvalidError("bad seal")
    assert exc.details == {}


def test_details_round_trip_through_envelope() -> None:
    exc = PolicyValidationError(
        "schema mismatch",
        details={"field": "policy_version", "expected": "0.6.0", "got": "0.5.0"},
    )
    envelope = exc.to_envelope()
    assert envelope == {
        "ok": False,
        "error": {
            "kind": "policy_validation",
            "code": 12,
            "message": "schema mismatch",
            "details": {"field": "policy_version", "expected": "0.6.0", "got": "0.5.0"},
        },
    }
    # Round-trip through JSON to confirm serialisability.
    loaded = json.loads(json.dumps(envelope, ensure_ascii=False))
    assert loaded == envelope


def test_details_are_defensively_copied() -> None:
    """A caller passing a mutable mapping cannot mutate the error post-hoc."""
    src: dict[str, str] = {"path": "./bundle"}
    exc = BundleNotFoundError("missing", details=src)
    src["path"] = "./other"
    src["sneaky"] = "added"
    assert exc.details == {"path": "./bundle"}


def test_details_accept_immutable_mapping() -> None:
    """The signature says ``Mapping``; pass a frozen one and confirm it works."""
    frozen = MappingProxyType({"reason": "missing _COMPLETE"})
    exc = BundleInvalidError("seal failed", details=frozen)
    assert exc.details == {"reason": "missing _COMPLETE"}


def test_envelope_marks_ok_false() -> None:
    """Wrapper scripts test ``error.ok`` first — must always be ``False``."""
    for cls in APMODE_ERRORS:
        envelope = cls("any").to_envelope()
        assert envelope["ok"] is False


def test_envelope_supports_nested_details() -> None:
    """JSONValue allows nested dict / list with leaf scalars."""
    exc = ConfigError(
        "contradictory flags",
        details={
            "flags": ["--agentic", "--lane", "submission"],
            "context": {"lane": "submission", "expected_lanes": ["discovery", "optimization"]},
        },
    )
    envelope = exc.to_envelope()
    json.dumps(envelope, ensure_ascii=False)  # must not raise
    assert envelope["error"]["details"]["flags"][0] == "--agentic"
    assert envelope["error"]["details"]["context"]["lane"] == "submission"


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", APMODE_ERRORS)
def test_subclasses_inherit_from_apmode_cli_error(cls: type[APModeCLIError]) -> None:
    assert issubclass(cls, APModeCLIError)
    assert issubclass(cls, Exception)


def test_apmode_cli_error_base_is_catchable() -> None:
    """``except APModeCLIError`` must catch every typed subclass."""
    for cls in APMODE_ERRORS:
        try:
            raise cls("boom")
        except APModeCLIError as caught:
            assert caught.message == "boom"
        else:  # pragma: no cover - unreachable; raise above always fires
            pytest.fail(f"{cls.__name__} did not propagate as APModeCLIError")
