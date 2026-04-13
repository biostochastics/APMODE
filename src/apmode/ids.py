# SPDX-License-Identifier: GPL-2.0-or-later
"""sparkid integration for time-sortable, monotonic ID generation."""

from sparkid import generate_id


def generate_run_id() -> str:
    """Generate a unique run ID (21-char, time-sortable, Base58)."""
    return generate_id()


def generate_candidate_id() -> str:
    """Generate a unique candidate model ID."""
    return generate_id()


def generate_gate_id() -> str:
    """Generate a unique gate decision ID."""
    return generate_id()
