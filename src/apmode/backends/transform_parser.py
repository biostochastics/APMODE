# SPDX-License-Identifier: GPL-2.0-or-later
"""LLM response parser: raw text → typed Formular transforms (PRD §4.2.6).

Parses the LLM's JSON output into a list of FormularTransform objects.
Handles stop signals, compound transforms, and malformed responses gracefully.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable
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
from apmode.dsl.prior_transforms import SetPrior
from apmode.dsl.priors import PriorFamily
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

# Fallback initial-estimate kwargs for the short-form string syntax
# ("new_module": "MichaelisMenten").  Values are in typical population-PK
# units; the R harness will override with NCA-derived estimates if available.
_MODULE_DEFAULTS: dict[str, dict[str, float | int | str]] = {
    "FirstOrder": {"ka": 1.0},
    "ZeroOrder": {"dur": 1.0},
    "LaggedFirstOrder": {"ka": 1.0, "tlag": 0.5},
    "Transit": {"n": 3, "ktr": 2.0, "ka": 1.0},
    "MixedFirstZero": {"ka": 1.0, "dur": 1.0, "frac": 0.5},
    "OneCmt": {"V": 90.0},
    "TwoCmt": {"V1": 50.0, "V2": 20.0, "Q": 2.0},
    "ThreeCmt": {"V1": 50.0, "V2": 20.0, "V3": 10.0, "Q2": 5.0, "Q3": 2.0},
    "Linear": {"CL": 5.0},
    "MichaelisMenten": {"Vmax": 50.0, "Km": 10.0},
    "ParallelLinearMM": {"CL": 3.0, "Vmax": 30.0, "Km": 10.0},
    "TimeVarying": {"CL": 5.0, "kdecay": 0.1, "decay_fn": "exponential"},
    "Proportional": {"sigma_prop": 0.2},
    "Additive": {"sigma_add": 1.0},
    "Combined": {"sigma_prop": 0.15, "sigma_add": 0.5},
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


def _parse_swap_module(raw: dict[str, Any]) -> FormularTransform:
    new_module_raw = raw.get("new_module", {})
    # Accept both short form ("new_module": "Linear") and long form
    # ("new_module": {"type": "Linear", ...}) for LLM robustness.
    if isinstance(new_module_raw, str):
        new_module_raw = {"type": new_module_raw}
    module_type = new_module_raw.get("type")
    if module_type not in _MODULE_REGISTRY:
        msg = f"Unknown module type '{module_type}' in swap_module"
        raise ValueError(msg)
    module_cls = _MODULE_REGISTRY[module_type]
    # Merge: LLM-provided fields override defaults; drop the "type" key.
    kwargs = dict(_MODULE_DEFAULTS.get(module_type, {}))
    kwargs.update({k: v for k, v in new_module_raw.items() if k != "type"})
    new_module = module_cls(**kwargs)
    return SwapModule(position=raw["position"], new_module=new_module)


def _parse_add_covariate_link(raw: dict[str, Any]) -> FormularTransform:
    return AddCovariateLink(
        param=normalize_param_name(raw["param"]),
        covariate=raw["covariate"],
        form=raw["form"],
    )


def _parse_adjust_variability(raw: dict[str, Any]) -> FormularTransform:
    return AdjustVariability(param=normalize_param_name(raw["param"]), action=raw["action"])


def _parse_set_transit_n(raw: dict[str, Any]) -> FormularTransform:
    return SetTransitN(n=raw["n"])


def _parse_toggle_lag(raw: dict[str, Any]) -> FormularTransform:
    return ToggleLag(on=raw["on"])


def _parse_replace_with_node(raw: dict[str, Any]) -> FormularTransform:
    return ReplaceWithNODE(
        position=raw["position"],
        constraint_template=raw["constraint_template"],
        dim=raw["dim"],
    )


def _parse_set_prior(raw: dict[str, Any]) -> FormularTransform:
    """Parse a ``set_prior`` transform — previously missing (H7).

    The ``family`` field is the discriminated-union ``PriorFamily``;
    we use Pydantic's TypeAdapter to deserialize without hard-coding
    every family variant here.
    """
    from pydantic import TypeAdapter

    family_raw = raw.get("family")
    if not isinstance(family_raw, dict):
        msg = "set_prior: 'family' must be an object with a discriminator field"
        raise ValueError(msg)
    family: PriorFamily = TypeAdapter(PriorFamily).validate_python(family_raw)
    return SetPrior(
        target=raw["target"],
        family=family,
        source=raw.get("source", "weakly_informative"),
        justification=raw.get("justification", ""),
        historical_refs=list(raw.get("historical_refs", [])),
    )


# Registration-map approach (H7): adding a new transform type requires a
# new entry here and a new test in ``tests/unit/test_transform_parser.py``
# (``test_parser_covers_all_transforms``) catches drift at test-collection
# time rather than at runtime.
_TRANSFORM_PARSERS: dict[str, Callable[[dict[str, Any]], FormularTransform]] = {
    "swap_module": _parse_swap_module,
    "add_covariate_link": _parse_add_covariate_link,
    "adjust_variability": _parse_adjust_variability,
    "set_transit_n": _parse_set_transit_n,
    "toggle_lag": _parse_toggle_lag,
    "replace_with_node": _parse_replace_with_node,
    "set_prior": _parse_set_prior,
}


def _parse_single_transform(raw: dict[str, Any], t_type: str | None) -> FormularTransform:
    """Parse a single transform dict into a typed FormularTransform."""
    if t_type is None:
        msg = "Transform missing 'type' field"
        raise ValueError(msg)
    parser = _TRANSFORM_PARSERS.get(t_type)
    if parser is None:
        msg = f"Unknown transform type: {t_type!r}"
        raise ValueError(msg)
    return parser(raw)
