# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the typed ``Lane`` enum (Formular sharpening plan §4 Phase 0, P0.3).

Covers:
- Canonical member values match the PRD §3 lane taxonomy and the plain
  strings already persisted in ``policies/*.json``.
- ``Lane`` round-trips from a plain string at a system boundary (CLI arg
  parsing, API request bodies) via ``Lane(the_string)``.
- ``apmode.backends.protocol.Lane`` is a re-export of the same class
  object, not a parallel/incompatible definition.
- No bare string-literal lane comparisons (``lane == "submission"``,
  ``lane != "optimization"``, etc.) remain in ``src/apmode/dsl/`` or
  ``src/apmode/governance/`` -- those modules must compare against
  ``Lane`` members so a typo or a lane rename is caught by mypy/IDE
  tooling instead of silently comparing against a stale string.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from apmode.dsl.lane import LANE_TAXONOMY_VERSION, Lane

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCANNED_DIRS = (
    _REPO_ROOT / "src" / "apmode" / "dsl",
    _REPO_ROOT / "src" / "apmode" / "governance",
)

# Matches `lane == "submission"`, `lane != 'discovery'`, `self.lane == "optimization"`,
# etc. -- an identifier named/ending in `lane` compared against a string literal.
_BARE_LANE_STRING_COMPARISON = re.compile(
    r"\blane\s*(?:==|!=)\s*[\"'](submission|discovery|optimization)[\"']"
)


class TestLaneMembers:
    """Canonical taxonomy: values must match PRD §3 and existing policy JSON."""

    def test_members_match_prd_taxonomy(self) -> None:
        assert Lane.SUBMISSION.value == "submission"
        assert Lane.DISCOVERY.value == "discovery"
        assert Lane.OPTIMIZATION.value == "optimization"

    def test_exactly_three_lanes(self) -> None:
        assert {lane.value for lane in Lane} == {"submission", "discovery", "optimization"}

    def test_taxonomy_version_is_declared(self) -> None:
        assert isinstance(LANE_TAXONOMY_VERSION, str)
        assert re.fullmatch(r"\d+\.\d+\.\d+", LANE_TAXONOMY_VERSION)


class TestLaneStringInterop:
    """StrEnum round-trip: string in, ``Lane`` out; ``Lane`` behaves as ``str``."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("submission", Lane.SUBMISSION),
            ("discovery", Lane.DISCOVERY),
            ("optimization", Lane.OPTIMIZATION),
        ],
    )
    def test_construct_from_string_at_boundary(self, raw: str, expected: Lane) -> None:
        assert Lane(raw) is expected

    def test_invalid_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="not_a_real_lane"):
            Lane("not_a_real_lane")

    def test_str_enum_compares_equal_to_plain_string(self) -> None:
        # Existing policy JSON / dataclass fields typed `str`/`Literal[...]`
        # keep working unchanged when compared against a `Lane` member.
        assert Lane.SUBMISSION == "submission"
        assert str(Lane.SUBMISSION) == "submission"

    def test_json_round_trip_via_pydantic(self) -> None:
        from pydantic import BaseModel

        class _Holder(BaseModel):
            lane: Lane

        holder = _Holder.model_validate({"lane": "discovery"})
        assert holder.lane is Lane.DISCOVERY
        assert holder.model_dump()["lane"] == Lane.DISCOVERY
        assert holder.model_dump_json() == '{"lane":"discovery"}'


class TestLaneSingleSourceOfTruth:
    """`backends.protocol.Lane` must be a re-export, not a parallel enum."""

    def test_backends_protocol_reexports_same_class(self) -> None:
        from apmode.backends.protocol import Lane as BackendLane

        assert BackendLane is Lane

    def test_validator_uses_canonical_lane(self) -> None:
        import typing

        from apmode.dsl.validator import validate_dsl

        hints = typing.get_type_hints(validate_dsl)
        assert hints["lane"] is Lane


class TestNoBareLaneStringComparisons:
    """Grep-based guard: dsl/ and governance/ compare against `Lane`, not strings."""

    def test_no_bare_string_lane_comparisons_in_dsl_and_governance(self) -> None:
        offenders: list[str] = []
        for directory in _SCANNED_DIRS:
            for path in sorted(directory.rglob("*.py")):
                text = path.read_text(encoding="utf-8")
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if _BARE_LANE_STRING_COMPARISON.search(line):
                        rel = path.relative_to(_REPO_ROOT)
                        offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert offenders == [], (
            "Bare string-literal lane comparisons found; compare against "
            "apmode.dsl.lane.Lane members instead:\n" + "\n".join(offenders)
        )
