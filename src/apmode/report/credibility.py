# SPDX-License-Identifier: GPL-2.0-or-later
"""Credibility Assessment Report generator (ARCHITECTURE.md SS4.4, PRD SS4.3.3).

Generates per-candidate CredibilityReport JSON for recommended models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmode.bundle.models import CredibilityReport

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult


def generate_credibility_report(
    result: BackendResult,
    lane: str,
    n_observations: int,
) -> CredibilityReport:
    """Generate a credibility report for a recommended candidate.

    Args:
        result: BackendResult for the candidate.
        lane: Operating lane.
        n_observations: Total observations in the dataset.

    Returns:
        CredibilityReport with ICH M15-aligned fields.
    """
    is_ml = result.backend in ("jax_node", "agentic_llm")

    limitations: list[str] = []
    if is_ml:
        limitations.append(
            "NODE random effects are on latent computational weights, not physiological parameters"
        )
    if result.diagnostics.blq.blq_fraction > 0.10:
        limitations.append(
            f"BLQ fraction {result.diagnostics.blq.blq_fraction:.2f} "
            f"may affect parameter precision"
        )

    return CredibilityReport(
        candidate_id=result.model_id,
        context_of_use=f"{lane} lane: population PK model for dose optimization",
        model_credibility={
            "estimation_method": result.convergence_metadata.method,
            "converged": result.convergence_metadata.converged,
            "minimization_status": result.convergence_metadata.minimization_status,
            "n_parameters": len(result.parameter_estimates),
            "data_adequacy_ratio": round(
                n_observations / max(len(result.parameter_estimates), 1), 1
            ),
        },
        data_adequacy=(
            "adequate"
            if n_observations / max(len(result.parameter_estimates), 1) >= 5.0
            else "marginal"
        ),
        limitations=limitations,
        ml_transparency=(
            f"Backend: {result.backend}. "
            f"NODE sub-model with constrained output layer. "
            f"Random effects on input-layer weights (Bram et al. 2024)."
            if is_ml
            else None
        ),
    )
