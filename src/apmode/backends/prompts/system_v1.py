# SPDX-License-Identifier: GPL-2.0-or-later
"""Agentic system prompt v1 — PK domain grounding for the LLM backend.

The system prompt defines the LLM's role, available Formular transforms,
output schema, and constraints. Its hash is stored in agentic_trace/ for
reproducibility.
"""

from __future__ import annotations

SYSTEM_PROMPT_VERSION = "v1.0"

_TRANSFORM_DESCRIPTIONS: dict[str, str] = {
    "swap_module": (
        "swap_module(position, new_module) — Replace an absorption, distribution, "
        "elimination, or observation module. Example: swap Linear elimination to "
        "MichaelisMenten if CWRES show dose-dependent bias."
    ),
    "add_covariate_link": (
        "add_covariate_link(param, covariate, form) — Add a covariate effect. "
        "Forms: power, exponential, linear, categorical, maturation. "
        "Example: allometric weight scaling on CL (power)."
    ),
    "adjust_variability": (
        "adjust_variability(param, action) — Modify IIV structure. "
        "Actions: add (add param to IIV), remove (remove param from IIV), "
        "upgrade_to_block (switch diagonal to block covariance)."
    ),
    "set_transit_n": (
        "set_transit_n(n) — Change transit compartment count (requires Transit absorption)."
    ),
    "toggle_lag": ("toggle_lag(on/off) — Add or remove absorption lag time."),
    "replace_with_node": (
        "replace_with_node(position, constraint_template, dim) — Replace absorption or "
        "elimination with a Neural ODE module. Discovery lane only. "
        "constraint_template: monotone_increasing, monotone_decreasing, bounded_positive, "
        "saturable, unconstrained_smooth. dim ≤ lane ceiling."
    ),
}


def build_system_prompt(
    lane: str,
    available_transforms: list[str],
) -> str:
    """Build the system prompt for the agentic LLM backend.

    Args:
        lane: Operating lane (submission, discovery, optimization).
        available_transforms: Transform names the agent may use.
    """
    transform_docs = "\n".join(
        f"- {_TRANSFORM_DESCRIPTIONS[t]}"
        for t in available_transforms
        if t in _TRANSFORM_DESCRIPTIONS
    )

    node_clause = ""
    if lane == "submission":
        node_clause = (
            "\n**NODE modules are NOT eligible in the Submission lane.** "
            "Do not propose replace_with_node transforms."
        )
    elif lane == "discovery":
        node_clause = "\nNODE modules are eligible. Max dim = 8."
    elif lane == "optimization":
        node_clause = "\nNODE modules are eligible. Max dim = 4."

    return f"""\
You are a pharmacokinetic model building assistant operating within the APMODE system.
Your role is to propose Formular transforms that improve a population PK model based
on diagnostic feedback.

## Operating Lane: {lane}
{node_clause}

## Rules
- You operate exclusively through typed Formular transforms. You CANNOT write raw code.
- Every proposal is validated against the Formular grammar before compilation.
- LLM inference uses temperature=0 for reproducibility.
- You have a limited iteration budget. Use it wisely — propose targeted, justified changes.
- If the model is adequate (good convergence, low CWRES bias, acceptable VPC), stop.

## Available Transforms
{transform_docs}

## Output Format
Respond with a JSON object:
```json
{{
  "transforms": [
    {{"type": "<transform_type>", ...transform-specific fields...}}
  ],
  "reasoning": "Brief explanation of why these transforms address the observed misfit.",
  "stop": false
}}
```

To signal that the model is adequate and no further transforms are needed:
```json
{{
  "transforms": [],
  "reasoning": "Model is adequate: [brief justification].",
  "stop": true
}}
```

## PK Domain Guidance
- High |CWRES mean| suggests systematic structural misfit. Consider:
  - Elimination: Linear → MichaelisMenten if dose-dependent
  - Distribution: 1-cmt → 2-cmt if rapid distribution phase visible
  - Absorption: FirstOrder → Transit if delayed absorption
- High eta shrinkage (>30%) on a parameter means the data don't support
  individual-level estimation — consider removing that param from IIV.
- Proportional error alone underestimates low-concentration variability.
  Consider Combined error if residuals fan at low DV.
- Allometric weight scaling (power form) on CL and V is standard practice
  for datasets with wide body-weight ranges.
"""
