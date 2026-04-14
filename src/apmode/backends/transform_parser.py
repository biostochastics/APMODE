# SPDX-License-Identifier: GPL-2.0-or-later
"""LLM response parser: raw text → typed Formular transforms (PRD §4.2.6).

Parses the LLM's JSON output into a list of FormularTransform objects.
Handles stop signals, compound transforms, and malformed responses gracefully.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict

from apmode.dsl.ast_models import (
    Additive,
    Combined,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    TimeVaryingElim,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.normalize import normalize_param_name
from apmode.dsl.transforms import (
    AddCovariateLink,
    AdjustVariability,
    FormularTransform,
    ReplaceWithNODE,
    SetTransitN,
    SwapModule,
    ToggleLag,
)

logger = structlog.get_logger(__name__)

# Maps type names to AST model classes for module deserialization
_MODULE_REGISTRY: dict[str, type[BaseModel]] = {
    "FirstOrder": FirstOrder,
    "ZeroOrder": ZeroOrder,
    "LaggedFirstOrder": LaggedFirstOrder,
    "Transit": Transit,
    "MixedFirstZero": MixedFirstZero,
    "OneCmt": OneCmt,
    "TwoCmt": TwoCmt,
    "ThreeCmt": ThreeCmt,
    "Linear": LinearElim,
    "MichaelisMenten": MichaelisMenten,
    "ParallelLinearMM": ParallelLinearMM,
    "TimeVarying": TimeVaryingElim,
    "Proportional": Proportional,
    "Additive": Additive,
    "Combined": Combined,
}


class ParseResult(BaseModel):
    """Result of parsing an LLM response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    transforms: list[Any] = []  # FormularTransform instances
    reasoning: str = ""
    stop: bool = False
    errors: list[str] = []


def parse_llm_response(raw_text: str) -> ParseResult:
    """Parse an LLM response into Formular transforms.

    Extracts JSON from the response (handles markdown code fences),
    validates the schema, and converts to typed transform objects.
    """
    # Try to extract JSON from the response
    json_str = _extract_json(raw_text)
    if json_str is None:
        return ParseResult(
            success=False,
            errors=[f"Could not parse JSON from response: {raw_text[:200]}"],
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return ParseResult(
            success=False,
            errors=[f"Invalid JSON: {e}"],
        )

    if not isinstance(data, dict):
        return ParseResult(
            success=False,
            errors=["Response must be a JSON object with 'transforms' key"],
        )

    reasoning = data.get("reasoning", "")
    stop = bool(data.get("stop", False))
    raw_transforms = data.get("transforms", [])

    if not isinstance(raw_transforms, list):
        return ParseResult(
            success=False,
            errors=["'transforms' must be a list"],
            reasoning=reasoning,
        )

    # Stop signal with empty transforms
    if stop and len(raw_transforms) == 0:
        return ParseResult(success=True, stop=True, reasoning=reasoning)

    transforms: list[Any] = []
    errors: list[str] = []

    for i, raw_t in enumerate(raw_transforms):
        if not isinstance(raw_t, dict):
            errors.append(f"Transform {i}: expected dict, got {type(raw_t).__name__}")
            continue

        t_type = raw_t.get("type")
        try:
            transform = _parse_single_transform(raw_t, t_type)
            transforms.append(transform)
        except Exception as e:
            errors.append(f"Transform {i} (type={t_type}): {e}")

    if errors:
        return ParseResult(success=False, errors=errors, reasoning=reasoning)

    return ParseResult(
        success=True,
        transforms=transforms,
        reasoning=reasoning,
        stop=stop,
    )


def _extract_json(text: str) -> str | None:
    """Extract JSON from raw text, handling markdown code fences."""
    # Try direct JSON parse first
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped

    # Try extracting from code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try finding first { ... } block
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


def _parse_single_transform(raw: dict[str, Any], t_type: str | None) -> FormularTransform:
    """Parse a single transform dict into a typed FormularTransform."""
    if t_type == "swap_module":
        new_module_raw = raw.get("new_module", {})
        module_type = new_module_raw.get("type")
        if module_type not in _MODULE_REGISTRY:
            msg = f"Unknown module type '{module_type}' in swap_module"
            raise ValueError(msg)
        module_cls = _MODULE_REGISTRY[module_type]
        new_module = module_cls(**{k: v for k, v in new_module_raw.items() if k != "type"})
        return SwapModule(position=raw["position"], new_module=new_module)

    if t_type == "add_covariate_link":
        return AddCovariateLink(
            param=normalize_param_name(raw["param"]),
            covariate=raw["covariate"],
            form=raw["form"],
        )

    if t_type == "adjust_variability":
        return AdjustVariability(param=normalize_param_name(raw["param"]), action=raw["action"])

    if t_type == "set_transit_n":
        return SetTransitN(n=raw["n"])

    if t_type == "toggle_lag":
        return ToggleLag(on=raw["on"])

    if t_type == "replace_with_node":
        return ReplaceWithNODE(
            position=raw["position"],
            constraint_template=raw["constraint_template"],
            dim=raw["dim"],
        )

    msg = f"Unknown transform type: '{t_type}'"
    raise ValueError(msg)
