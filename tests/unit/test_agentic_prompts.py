# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for agentic system prompt template."""

from apmode.backends.prompts.system_v1 import SYSTEM_PROMPT_VERSION, build_system_prompt


def test_system_prompt_contains_constraints() -> None:
    prompt = build_system_prompt(
        lane="discovery",
        available_transforms=[
            "swap_module",
            "add_covariate_link",
            "adjust_variability",
            "set_transit_n",
            "toggle_lag",
            "set_prior",
            "convert_transit_to_erlang",
            "add_parallel_route",
            "set_sumig_components",
            "replace_with_node",
        ],
    )
    assert "temperature=0" in prompt or "temperature 0" in prompt
    assert "Formular" in prompt
    assert "swap_module" in prompt
    assert "JSON" in prompt


def test_system_prompt_excludes_node_for_submission() -> None:
    prompt = build_system_prompt(
        lane="submission",
        available_transforms=[
            "swap_module",
            "add_covariate_link",
            "adjust_variability",
            "set_transit_n",
            "toggle_lag",
        ],
    )
    assert "NOT eligible" in prompt


def test_system_prompt_has_version() -> None:
    assert SYSTEM_PROMPT_VERSION.startswith("v1")


def test_system_prompt_json_schema_example() -> None:
    prompt = build_system_prompt(lane="discovery", available_transforms=["swap_module"])
    assert '"transforms"' in prompt
    assert '"reasoning"' in prompt


def test_system_prompt_documents_all_parser_transforms() -> None:
    from apmode.backends.transform_parser import _TRANSFORM_PARSERS

    prompt = build_system_prompt(
        lane="discovery",
        available_transforms=list(_TRANSFORM_PARSERS),
    )

    for transform_name in _TRANSFORM_PARSERS:
        assert transform_name in prompt
