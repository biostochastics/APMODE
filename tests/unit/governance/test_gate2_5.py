# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for Gate 2.5 (Credibility Qualification — ICH M15).

Verifies the governance funnel: gates are sequential disqualifiers,
failures logged with per-check reasons, thresholds from policy files.
"""

from __future__ import annotations

import json

from apmode.governance.gates import (
    evaluate_gate2_5,
)
from apmode.governance.policy import GatePolicy
from tests._helpers.builders import make_backend_result as _make_backend_result
from tests._helpers.policies import POLICY_DIR

# ---------------------------------------------------------------------------
# Gate 2.5 Tests (Credibility Qualification — ICH M15)
# ---------------------------------------------------------------------------


def _make_policy_with_gate25(
    lane: str = "submission",
    *,
    context_required: bool = True,
    limitation_required: bool = False,
    data_adequacy_required: bool = True,
    data_adequacy_ratio_min: float = 5.0,
    sensitivity_required: bool = False,
    ml_transparency_required: bool = False,
) -> GatePolicy:
    """Build a policy with Gate 2.5 config for testing."""
    base = json.loads(POLICY_DIR.joinpath(f"{lane}.json").read_text())
    base["gate2_5"] = {
        "context_of_use_required": context_required,
        "limitation_to_risk_mapping_required": limitation_required,
        "data_adequacy_required": data_adequacy_required,
        "data_adequacy_ratio_min": data_adequacy_ratio_min,
        "sensitivity_analysis_required": sensitivity_required,
        "ai_ml_transparency_required": ml_transparency_required,
    }
    return GatePolicy.model_validate(base)


class TestGate25:
    """Gate 2.5: Credibility Qualification (ICH M15)."""

    def test_no_gate25_config_passes(self) -> None:
        """When policy has no gate2_5, all candidates pass."""
        result = _make_backend_result()
        # submission.json now ships with a gate2_5 block (policy_version 0.5.1);
        # build an explicit no-gate2_5 policy to exercise the ``g25 is None``
        # branch of ``evaluate_gate2_5``.
        base = json.loads(POLICY_DIR.joinpath("submission.json").read_text())
        base.pop("gate2_5", None)
        policy = GatePolicy.model_validate(base)
        g25 = evaluate_gate2_5(result, policy)
        assert g25.passed is True
        assert g25.gate_name == "credibility_qualification"

    def test_passes_with_adequate_context(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25()
        ctx = CredibilityContext(
            context_of_use="Dose adjustment for renal impairment",
            risk_level="medium",
            n_observations=100,
            n_parameters=8,
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is True

    def test_fails_missing_context_of_use(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(context_required=True)
        ctx = CredibilityContext(n_observations=100, n_parameters=8)  # no COU
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "context_of_use" in failed_ids

    def test_fails_insufficient_data_adequacy(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(data_adequacy_ratio_min=10.0)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=20,
            n_parameters=8,  # ratio = 2.5 < 10.0
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "data_adequacy" in failed_ids

    def test_data_adequacy_passes_when_ratio_sufficient(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(data_adequacy_ratio_min=5.0)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=100,
            n_parameters=8,  # ratio = 12.5 >= 5.0
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        da = next(c for c in g25.checks if c.check_id == "data_adequacy")
        assert da.passed is True

    def test_node_requires_ml_transparency(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result(backend="jax_node")
        policy = _make_policy_with_gate25(
            lane="discovery",
            ml_transparency_required=True,
        )
        ctx = CredibilityContext(
            context_of_use="Discovery analysis",
            n_observations=200,
            n_parameters=10,
            # No ml_transparency_statement
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "ml_transparency" in failed_ids

    def test_node_with_transparency_passes(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result(backend="jax_node")
        policy = _make_policy_with_gate25(
            lane="discovery",
            ml_transparency_required=True,
        )
        ctx = CredibilityContext(
            context_of_use="Discovery analysis",
            n_observations=200,
            n_parameters=10,
            ml_transparency_statement=(
                "NODE used for elimination; bounded_positive constraint; 3-dim"
            ),
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        ml = next(c for c in g25.checks if c.check_id == "ml_transparency")
        assert ml.passed is True

    def test_classical_skips_ml_transparency(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result(backend="nlmixr2")
        policy = _make_policy_with_gate25(ml_transparency_required=True)
        ctx = CredibilityContext(
            context_of_use="Submission analysis",
            n_observations=200,
            n_parameters=8,
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        ml = next(c for c in g25.checks if c.check_id == "ml_transparency")
        assert ml.passed is True  # not applicable for classical

    def test_sensitivity_required_but_missing(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(sensitivity_required=True)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=100,
            n_parameters=8,
            sensitivity_available=False,
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "sensitivity_analysis" in failed_ids

    def test_limitation_to_risk_required_but_missing(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25(limitation_required=True)
        ctx = CredibilityContext(
            context_of_use="Test",
            n_observations=100,
            n_parameters=8,
            # No limitations or risk_level
        )
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        assert g25.passed is False
        failed_ids = {c.check_id for c in g25.checks if not c.passed}
        assert "limitation_to_risk" in failed_ids

    def test_all_checks_present(self) -> None:
        from apmode.bundle.models import CredibilityContext

        result = _make_backend_result()
        policy = _make_policy_with_gate25()
        ctx = CredibilityContext(context_of_use="Test", n_observations=100, n_parameters=8)
        g25 = evaluate_gate2_5(result, policy, credibility_context=ctx)
        check_ids = {c.check_id for c in g25.checks}
        expected = {
            "context_of_use",
            "limitation_to_risk",
            "data_adequacy",
            "sensitivity_analysis",
            "ml_transparency",
            "risk_grading",
        }
        assert expected == check_ids

    def test_risk_grading_not_required_passes(self) -> None:
        from apmode.bundle.models import CredibilityContext

        policy = _make_policy_with_gate25()  # risk_grading omitted -> None
        result = _make_backend_result()
        gate_result = evaluate_gate2_5(result, policy, CredibilityContext())
        check = next(c for c in gate_result.checks if c.check_id == "risk_grading")
        assert check.passed is True
        assert check.observed == "not_required"

    def test_risk_grading_gap_fails(self) -> None:
        from apmode.bundle.models import CredibilityContext

        base = json.loads(POLICY_DIR.joinpath("submission.json").read_text())
        policy = GatePolicy.model_validate(base)  # submission.json now ships risk_grading enabled
        result = _make_backend_result()  # no NPE/AUC-Cmax diagnostics by default
        ctx = CredibilityContext(model_influence="high", decision_consequence="high")
        gate_result = evaluate_gate2_5(result, policy, ctx)
        check = next(c for c in gate_result.checks if c.check_id == "risk_grading")
        assert check.passed is False
        assert "tier=high" in str(check.observed)
        assert gate_result.passed is False

    def test_risk_grading_tier_low_influence_low_consequence_passes(self) -> None:
        from apmode.bundle.models import CredibilityContext

        base = json.loads(POLICY_DIR.joinpath("submission.json").read_text())
        policy = GatePolicy.model_validate(base)
        result = _make_backend_result()
        ctx = CredibilityContext(
            context_of_use="Test",
            model_influence="low",
            decision_consequence="low",
            n_observations=100,
            n_parameters=5,
        )
        gate_result = evaluate_gate2_5(result, policy, ctx)
        check = next(c for c in gate_result.checks if c.check_id == "risk_grading")
        # low/low tier has empty credibility_factors in submission.json fixture
        assert check.passed is True
