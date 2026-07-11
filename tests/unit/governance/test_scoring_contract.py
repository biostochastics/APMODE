# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for :class:`apmode.bundle.models.ScoringContract` (plan §3)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    ScoringContract,
)


def _classical_contract() -> ScoringContract:
    return ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="nlmixr2_focei",
        blq_method="none",
        observation_model="combined",
        float_precision="float64",
    )


def _node_pooled_contract() -> ScoringContract:
    return ScoringContract(
        nlpd_kind="conditional",
        re_treatment="pooled",
        nlpd_integrator="none",
        blq_method="none",
        observation_model="combined",
        float_precision="float32",
    )


class TestScoringContract:
    def test_is_frozen(self) -> None:
        contract = _classical_contract()
        with pytest.raises(ValidationError):
            contract.nlpd_kind = "conditional"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = _classical_contract()
        b = _classical_contract()
        assert a == b
        assert hash(a) == hash(b)

    def test_inequality_when_field_differs(self) -> None:
        nlmixr2 = _classical_contract()
        stan = _classical_contract().model_copy(update={"nlpd_integrator": "hmc_nuts"})
        assert nlmixr2 != stan

    def test_precision_matters_for_equality(self) -> None:
        a = _classical_contract()
        b = a.model_copy(update={"float_precision": "float32"})
        assert a != b

    def test_rejects_unknown_literal(self) -> None:
        with pytest.raises(ValidationError):
            ScoringContract(
                nlpd_kind="conditional",
                re_treatment="integrated",
                nlpd_integrator="invalid_integrator",  # type: ignore[arg-type]
                blq_method="none",
                observation_model="combined",
                float_precision="float64",
            )

    def test_roundtrip_json(self) -> None:
        contract = _node_pooled_contract()
        roundtripped = ScoringContract.model_validate_json(contract.model_dump_json())
        assert roundtripped == contract

    def test_default_diagnostic_bundle_contract_is_classical(self) -> None:
        """New DiagnosticBundles should default to the nlmixr2 contract."""
        db = DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.0),
            identifiability=IdentifiabilityFlags(
                condition_number=None,
                profile_likelihood_ci={},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        )
        assert db.scoring_contract == _classical_contract()

    def test_contract_survives_model_copy_update(self) -> None:
        db = DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.0),
            identifiability=IdentifiabilityFlags(
                condition_number=None,
                profile_likelihood_ci={},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            scoring_contract=_node_pooled_contract(),
        )
        updated = db.model_copy(update={"vpc": None})
        assert updated.scoring_contract == _node_pooled_contract()

    def test_serializes_contract_version(self) -> None:
        payload = json.loads(_classical_contract().model_dump_json())
        assert payload["contract_version"] == 1

    def test_backend_result_carries_contract(self) -> None:
        db = DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.0),
            identifiability=IdentifiabilityFlags(
                condition_number=None,
                profile_likelihood_ci={},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            scoring_contract=_node_pooled_contract(),
        )
        result = BackendResult(
            model_id="m1",
            backend="jax_node",
            converged=True,
            parameter_estimates={
                "CL": ParameterEstimate(name="CL", estimate=1.0, category="structural")
            },
            eta_shrinkage={"CL": 0.0},
            convergence_metadata=ConvergenceMetadata(
                method="adam",
                converged=True,
                iterations=1,
                minimization_status="successful",
                wall_time_seconds=0.1,
            ),
            diagnostics=db,
            wall_time_seconds=0.1,
            backend_versions={"jax": "0.0.0", "python": "3.12"},
            initial_estimate_source="fallback",
        )
        assert result.diagnostics.scoring_contract.float_precision == "float32"
