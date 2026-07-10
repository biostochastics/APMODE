# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for V&V40-style risk grading report models, generator, and emitter."""

from __future__ import annotations

import json
from pathlib import Path

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    CredibilityActivity,
    CredibilityContext,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    RiskGradingReport,
    ScoringContract,
    VPCSummary,
)
from apmode.governance.policy import Gate25Config, GatePolicy
from apmode.report.risk_grading import generate_risk_grading_report

_POLICIES = Path(__file__).resolve().parents[2] / "policies"


def _contract() -> ScoringContract:
    return ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="nlmixr2_focei",
        blq_method="none",
        observation_model="combined",
        float_precision="float64",
    )


def _make_result(backend: str = "nlmixr2") -> BackendResult:
    return BackendResult(
        model_id="test_model",
        backend=backend,  # type: ignore[arg-type]
        converged=True,
        ofv=150.0,
        aic=160.0,
        bic=170.0,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=5.0, category="structural"),
            "V": ParameterEstimate(name="V", estimate=70.0, category="structural"),
        },
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=200,
            minimization_status="successful",
            wall_time_seconds=45.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.01, cwres_sd=1.0, outlier_fraction=0.02),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage={"p5": 0.60, "p50": 0.62, "p95": 0.58},
                n_bins=10,
                prediction_corrected=False,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={"CL": True, "V": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            scoring_contract=_contract(),
        ),
        wall_time_seconds=45.0,
        backend_versions={"nlmixr2": "2.1.2"},
        initial_estimate_source="nca",
    )


def _submission_policy_with_risk_grading() -> GatePolicy:
    data = json.loads((_POLICIES / "submission.json").read_text())
    return GatePolicy.model_validate(data)


class TestRiskGradingModels:
    def test_credibility_context_has_risk_axes(self) -> None:
        ctx = CredibilityContext(model_influence="high", decision_consequence="medium")
        assert ctx.model_influence == "high"
        assert ctx.decision_consequence == "medium"
        assert ctx.risk_level is None  # unchanged back-compat default

    def test_risk_grading_report_shape(self) -> None:
        report = RiskGradingReport(
            candidate_id="cand-1",
            context_of_use="submission lane: dose selection",
            model_influence="high",
            decision_consequence="high",
            risk_tier="high",
            credibility_activities=[
                CredibilityActivity(
                    factor="vpc_coverage",
                    target_rigor="c",
                    obtained_rigor="b",
                    gap=True,
                    evidence_ref="diagnostics.vpc.coverage",
                )
            ],
            gaps=["vpc_coverage"],
        )
        assert report.risk_tier == "high"
        assert report.gaps == ["vpc_coverage"]


class TestRiskGradingReportGenerator:
    def test_generates_report_with_gaps(self) -> None:
        result = _make_result()
        policy = _submission_policy_with_risk_grading()
        assert policy.gate2_5 is not None
        ctx = CredibilityContext(model_influence="high", decision_consequence="high")
        report = generate_risk_grading_report(
            result,
            lane="submission",
            credibility_context=ctx,
            gate25_config=policy.gate2_5,
        )
        assert report.risk_tier == "high"
        assert report.candidate_id == result.model_id
        assert isinstance(report.gaps, list)
        assert report.source_result_sha256 is not None

    def test_no_risk_grading_config_defaults_to_high_no_factors(self) -> None:
        result = _make_result()
        ctx = CredibilityContext()
        report = generate_risk_grading_report(
            result,
            lane="submission",
            credibility_context=ctx,
            gate25_config=Gate25Config(),
        )
        assert report.risk_tier == "high"
        assert report.gaps == []
        assert report.credibility_activities == []


class TestRiskGradingReportEmitter:
    def test_writes_risk_grading_json(self, tmp_path: Path) -> None:
        from apmode.bundle.emitter import BundleEmitter

        emitter = BundleEmitter(tmp_path)
        emitter.initialize()

        report = RiskGradingReport(
            candidate_id="cand-1",
            context_of_use="cou",
            model_influence="high",
            decision_consequence="high",
            risk_tier="high",
        )
        path = emitter.write_risk_grading_report(report)
        assert path.exists()
        assert path.parent.name == "risk_grading"
        assert path.name == "cand-1.json"
        assert json.loads(path.read_text())["risk_tier"] == "high"
