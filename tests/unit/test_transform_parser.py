# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for LLM response → Formular transform parser."""

import json

from apmode.backends.transform_parser import parse_llm_response
from apmode.dsl.transforms import AddCovariateLink, AdjustVariability, SwapModule


def test_parse_json_single_transform() -> None:
    raw = json.dumps(
        {
            "transforms": [
                {
                    "type": "swap_module",
                    "position": "elimination",
                    "new_module": {"type": "MichaelisMenten", "Vmax": 50.0, "Km": 5.0},
                },
            ],
            "reasoning": "CWRES show time-dependent bias in elimination phase.",
        }
    )
    result = parse_llm_response(raw)
    assert result.success
    assert len(result.transforms) == 1
    assert isinstance(result.transforms[0], SwapModule)


def test_parse_json_compound_transforms() -> None:
    raw = json.dumps(
        {
            "transforms": [
                {
                    "type": "swap_module",
                    "position": "elimination",
                    "new_module": {"type": "MichaelisMenten", "Vmax": 50.0, "Km": 5.0},
                },
                {"type": "add_covariate_link", "param": "CL", "covariate": "WT", "form": "power"},
            ],
            "reasoning": "MM elimination + weight effect on CL.",
        }
    )
    result = parse_llm_response(raw)
    assert result.success
    assert len(result.transforms) == 2
    assert isinstance(result.transforms[0], SwapModule)
    assert isinstance(result.transforms[1], AddCovariateLink)


def test_parse_stop_signal() -> None:
    raw = json.dumps({"transforms": [], "reasoning": "Model is adequate.", "stop": True})
    result = parse_llm_response(raw)
    assert result.success
    assert result.stop is True
    assert len(result.transforms) == 0


def test_parse_invalid_json() -> None:
    result = parse_llm_response("this is not json at all")
    assert not result.success
    assert len(result.errors) > 0


def test_parse_unknown_transform_type() -> None:
    raw = json.dumps(
        {
            "transforms": [{"type": "unknown_transform", "foo": "bar"}],
            "reasoning": "test",
        }
    )
    result = parse_llm_response(raw)
    assert not result.success
    assert any("unknown" in e.lower() for e in result.errors)


def test_parse_extracts_reasoning() -> None:
    raw = json.dumps(
        {
            "transforms": [
                {"type": "adjust_variability", "param": "CL", "action": "upgrade_to_block"},
            ],
            "reasoning": "High correlation between CL and V etas.",
        }
    )
    result = parse_llm_response(raw)
    assert result.success
    assert result.reasoning == "High correlation between CL and V etas."
    assert isinstance(result.transforms[0], AdjustVariability)


def test_parse_json_in_code_fence() -> None:
    raw = """Here's my proposal:
```json
{
  "transforms": [{"type": "toggle_lag", "on": true}],
  "reasoning": "Delayed absorption visible."
}
```
"""
    result = parse_llm_response(raw)
    assert result.success
    assert len(result.transforms) == 1
