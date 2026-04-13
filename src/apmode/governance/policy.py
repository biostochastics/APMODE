# SPDX-License-Identifier: GPL-2.0-or-later
"""Gate policy file schema (PRD §4.3.1, ARCHITECTURE.md §4.3).

Gate thresholds are versioned policy artifacts, not hard-coded constants.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Gate1Config(BaseModel):
    """Gate 1: Technical Validity thresholds (PRD §4.3.1)."""

    convergence_required: bool = True
    parameter_plausibility_required: bool = True
    state_trajectory_validity_required: bool = True
    split_integrity_required: bool = True
    cwres_mean_max: float
    outlier_fraction_max: float = Field(ge=0.0, le=1.0)
    vpc_coverage_lower: float
    vpc_coverage_upper: float
    seed_stability_n: int = Field(ge=1)

    @model_validator(mode="after")
    def vpc_bounds_ordered(self) -> Gate1Config:
        if self.vpc_coverage_lower >= self.vpc_coverage_upper:
            msg = (
                f"vpc_coverage_lower ({self.vpc_coverage_lower}) must be < "
                f"vpc_coverage_upper ({self.vpc_coverage_upper})"
            )
            raise ValueError(msg)
        return self


class Gate2Config(BaseModel):
    """Gate 2: Lane-Specific Admissibility thresholds (PRD §4.3.1)."""

    interpretable_parameterization: Literal["required", "preferred", "not_required"]
    reproducible_estimation: Literal["required", "not_required"]
    shrinkage_max: float | None = None
    identifiability_required: bool
    node_eligible: bool
    loro_required: bool


class Gate25Config(BaseModel):
    """Gate 2.5: Credibility Qualification (Phase 2, PRD §4.3.1 / ICH M15).

    Stubbed for forward compatibility. Fields will be refined in Phase 2.
    """

    context_of_use_required: bool = True
    limitation_to_risk_mapping_required: bool = False
    data_adequacy_required: bool = True
    sensitivity_analysis_required: bool = False
    ai_ml_transparency_required: bool = False


class GatePolicy(BaseModel):
    """Top-level policy file: versioned gate configuration per lane."""

    policy_version: str
    lane: Literal["submission", "discovery", "optimization"]
    gate1: Gate1Config
    gate2: Gate2Config
    gate2_5: Gate25Config | None = None
