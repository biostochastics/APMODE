# SPDX-License-Identifier: GPL-2.0-or-later
"""SetPrior transform — the 7th FormularTransform (plan 2026-04-14 §3.2).

Extends the agentic LLM's admissible transform set so it can propose and
justify priors within the DSL ceiling, preserving the audit trail.

Integration (post-merge):
  - Add `SetPrior` to the `FormularTransform` union in transforms.py.
  - Add `priors: list[PriorSpec]` field to `DSLSpec` (default empty).
  - `apply_set_prior` produces a new DSLSpec with the updated priors list.

This file is intentionally standalone so it can be reviewed independently of
the DSLSpec structural change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from apmode.dsl.priors import (
    PriorFamily,
    PriorSource,
    PriorSpec,
    classify_target,
    validate_prior_family,
)
from apmode.ids import generate_candidate_id

if TYPE_CHECKING:
    from apmode.dsl.ast_models import DSLSpec


class SetPrior(BaseModel):
    """set_prior(target, family, source, justification) — declare or replace a prior.

    Semantics:
      - If no prior on `target` exists, append a new one.
      - If a prior on `target` already exists, replace it (idempotent re-declaration).

    Validation (at plan time, before apply):
      - target resolves to a known parameter in the current spec.
      - family matches the parameterization schema for that target kind.
      - source ∈ {historical_data, expert_elicitation, meta_analysis} requires
        non-empty justification (enforced by PriorSpec).
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["set_prior"] = "set_prior"
    target: str
    family: PriorFamily
    source: PriorSource = "weakly_informative"
    justification: str = ""
    historical_refs: list[str] = Field(default_factory=list)


def validate_set_prior(spec: DSLSpec, transform: SetPrior) -> list[str]:
    """Validate a SetPrior transform against a spec. Returns errors."""
    errors: list[str] = []
    structural = set(spec.structural_param_names())
    kind = classify_target(transform.target, structural)
    if kind is None:
        errors.append(
            f"SetPrior target {transform.target!r} does not resolve to any parameter "
            f"in spec (structural: {sorted(structural)})"
        )
        return errors

    family_err = validate_prior_family(kind, transform.family)
    if family_err:
        errors.append(family_err)

    # Construct-and-validate a PriorSpec to trigger justification rules
    try:
        PriorSpec(
            target=transform.target,
            family=transform.family,
            source=transform.source,
            justification=transform.justification,
            historical_refs=transform.historical_refs,
        )
    except ValueError as exc:
        errors.append(str(exc))

    return errors


def apply_set_prior(spec: DSLSpec, transform: SetPrior) -> DSLSpec:
    """Apply a SetPrior transform. Returns a new DSLSpec with a fresh candidate_id.

    Assumes DSLSpec has been extended with `priors: list[PriorSpec]` — otherwise
    raises AttributeError to surface the integration gap loudly.
    """
    errors = validate_set_prior(spec, transform)
    if errors:
        raise ValueError(f"SetPrior validation failed: {'; '.join(errors)}")

    if not hasattr(spec, "priors"):
        raise AttributeError(
            "DSLSpec does not have a `priors` field. "
            "Complete the DSLSpec extension before applying SetPrior transforms."
        )

    new_prior = PriorSpec(
        target=transform.target,
        family=transform.family,
        source=transform.source,
        justification=transform.justification,
        historical_refs=transform.historical_refs,
    )

    # Idempotent replace-or-append semantics
    existing_priors: list[PriorSpec] = list(spec.priors)
    new_priors = [p for p in existing_priors if p.target != transform.target]
    new_priors.append(new_prior)

    # Rebuild spec with fresh model_id; all other fields preserved.
    # We use model_copy rather than direct DSLSpec() construction to keep this
    # file decoupled from the full AST import graph.
    return spec.model_copy(update={"model_id": generate_candidate_id(), "priors": new_priors})
