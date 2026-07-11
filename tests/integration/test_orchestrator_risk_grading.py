# SPDX-License-Identifier: GPL-2.0-or-later
"""Orchestrator wiring for V&V40-style risk grading (Gate 2.5 companion).

Covers:
  - RunConfig accepts model_influence / decision_consequence.
  - Orchestrator.run computes CredibilityContext.risk_level from the
    submission-lane policy's RiskGradingConfig matrix instead of the
    former hard-coded risk_level="medium" literal.
  - risk_grading/{id}.json is emitted alongside credibility/{id}.json.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ColumnMapping,
    ConvergenceMetadata,
    DataManifest,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    PITCalibrationSummary,
    ScoringContract,
    VPCSummary,
)
from apmode.dsl.ast_models import DSLSpec

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


def _make_mock_result(model_id: str, bic: float = 540.0) -> BackendResult:
    """Build a realistic mock BackendResult that passes submission Gate 1/2."""
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=True,
        ofv=bic - 40.0,
        aic=bic - 20.0,
        bic=bic,
        parameter_estimates={
            "ka": ParameterEstimate(
                name="ka", estimate=1.0, se=0.1, rse=10.0, category="structural"
            ),
            "V": ParameterEstimate(
                name="V", estimate=30.0, se=3.0, rse=10.0, category="structural"
            ),
            "CL": ParameterEstimate(
                name="CL", estimate=2.0, se=0.2, rse=10.0, category="structural"
            ),
        },
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=300,
            gradient_norm=0.0005,
            minimization_status="successful",
            wall_time_seconds=60.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.02, cwres_sd=1.01, outlier_fraction=0.01),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage={"p5": 0.92, "p50": 0.96, "p95": 0.93},
                n_bins=10,
                prediction_corrected=False,
            ),
            pit_calibration=PITCalibrationSummary(
                probability_levels=[0.05, 0.50, 0.95],
                calibration={"p5": 0.05, "p50": 0.50, "p95": 0.95},
                n_observations=96,
                n_subjects=12,
                aggregation="subject_robust",
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={"ka": True, "V": True, "CL": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            scoring_contract=ScoringContract(
                nlpd_kind="marginal",
                re_treatment="integrated",
                nlpd_integrator="nlmixr2_focei",
                blq_method="none",
                observation_model="combined",
                float_precision="float64",
            ),
            # Populated so the "high" V&V40 risk tier's vpc_coverage /
            # npe_agreement / nca_eligibility credibility factors (see
            # policies/submission.json) all obtain rigor "d" and never
            # gap this fixture regardless of which tier is selected.
            npe_score=0.05,
            auc_cmax_be_score=0.95,
        ),
        wall_time_seconds=60.0,
        backend_versions={"nlmixr2": "3.0.0", "R": "4.4.1"},
        initial_estimate_source="nca",
    )


class MockNlmixr2Runner:
    """Mock nlmixr2 runner that always returns a passing classical result."""

    def __init__(self, bic: float = 540.0) -> None:
        self._bic = bic
        self.call_count = 0

    async def run(
        self,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,
        *,
        data_path: Path | None = None,
        split_manifest: dict[str, object] | None = None,
        gate3_policy: object | None = None,
        nca_diagnostics: object | None = None,
        test_data_path: Path | None = None,
        fixed_parameter: bool = False,
    ) -> BackendResult:
        self.call_count += 1
        return _make_mock_result(spec.model_id, self._bic)


def _write_test_policy(tmp_path: Path) -> Path:
    """submission.json with the *other* Gate 2.5 checks relaxed.

    Isolates this module's risk-grading wiring assertions from the
    pre-existing (unrelated) fact that the orchestrator never populates
    ``CredibilityContext.limitations``/``sensitivity_available`` — so
    ``limitation_to_risk_mapping_required`` / ``sensitivity_analysis_required``
    would otherwise always fail Gate 2.5 for every e2e run regardless of
    risk-grading, independent of anything this task changed.
    """
    data = json.loads((POLICY_DIR / "submission.json").read_text())
    data["gate2_5"]["limitation_to_risk_mapping_required"] = False
    data["gate2_5"]["sensitivity_analysis_required"] = False
    policy_path = tmp_path / "submission_risk_grading_test.json"
    policy_path.write_text(json.dumps(data))
    return policy_path


def _write_disabled_risk_grading_policy(tmp_path: Path) -> Path:
    data = json.loads((POLICY_DIR / "discovery.json").read_text())
    data["gate2_5"]["context_of_use_required"] = False
    data["gate2_5"]["data_adequacy_required"] = False
    policy_path = tmp_path / "discovery_risk_grading_disabled_test.json"
    policy_path.write_text(json.dumps(data))
    return policy_path


def _build_manifest_and_data(tmp_path: Path) -> tuple[DataManifest, Path]:
    import pandas as pd

    data_path = tmp_path / "test_data.csv"
    df = pd.DataFrame(
        {
            "NMID": [1] * 10 + [2] * 10,
            "TIME": list(range(10)) * 2,
            "DV": [0.0, 5.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5] * 2,
            "EVID": [1] + [0] * 9 + [1] + [0] * 9,
            "AMT": [100.0] + [0.0] * 9 + [100.0] + [0.0] * 9,
            "MDV": [1] + [0] * 9 + [1] + [0] * 9,
            "CMT": [1] + [2] * 9 + [1] + [2] * 9,
        }
    )
    df.to_csv(data_path, index=False)

    manifest = DataManifest(
        data_sha256="d" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID", time="TIME", dv="DV", evid="EVID", amt="AMT", mdv="MDV"
        ),
        n_subjects=2,
        n_observations=18,
        n_doses=2,
    )
    return manifest, data_path


def test_run_config_accepts_risk_axes() -> None:
    from apmode.orchestrator import RunConfig

    cfg = RunConfig(
        lane="submission",
        model_influence="high",
        decision_consequence="high",
    )
    assert cfg.model_influence == "high"
    assert cfg.decision_consequence == "high"


def test_run_config_risk_axes_default_none() -> None:
    from apmode.orchestrator import RunConfig

    cfg = RunConfig(lane="submission")
    assert cfg.model_influence is None
    assert cfg.decision_consequence is None


def test_orchestrator_computes_risk_level_from_matrix(tmp_path: Path) -> None:
    """submission.json ships risk_grading enabled; matrix[high][high] == 'high'."""
    from apmode.orchestrator import Orchestrator, RunConfig

    nlmixr2 = MockNlmixr2Runner(bic=540.0)
    config = RunConfig(
        lane="submission",
        seed=42,
        timeout_seconds=60,
        policy_path=_write_test_policy(tmp_path),
        model_influence="high",
        decision_consequence="high",
    )
    orch = Orchestrator(runner=nlmixr2, bundle_base_dir=tmp_path, config=config)
    manifest, data_path = _build_manifest_and_data(tmp_path)

    import pandas as pd

    df = pd.read_csv(data_path)
    outcome = asyncio.run(orch.run(manifest, df, data_path))

    assert len(outcome.recommended) > 0, "expected at least one Gate 1/2/2.5 survivor"

    cred_dir = outcome.bundle_dir / "credibility"
    cred_files = sorted(cred_dir.glob("*.json"))
    assert cred_files, "expected credibility reports to be written"

    rg_dir = outcome.bundle_dir / "risk_grading"
    rg_files = sorted(rg_dir.glob("*.json"))
    assert rg_files, "expected risk_grading reports to be written"

    rg_payload = json.loads(rg_files[0].read_text())
    # matrix[high][high] == "high" per policies/submission.json
    assert rg_payload["risk_tier"] == "high"
    assert rg_payload["model_influence"] == "high"
    assert rg_payload["decision_consequence"] == "high"


def test_orchestrator_risk_grading_absent_when_axes_unset(tmp_path: Path) -> None:
    """No model_influence/decision_consequence -> conservative 'high' tier
    still computed (submission.json's risk_grading.enabled=True), matching
    _check_risk_grading's own conservative default."""
    from apmode.orchestrator import Orchestrator, RunConfig

    nlmixr2 = MockNlmixr2Runner(bic=540.0)
    config = RunConfig(
        lane="submission",
        seed=42,
        timeout_seconds=60,
        policy_path=_write_test_policy(tmp_path),
    )
    orch = Orchestrator(runner=nlmixr2, bundle_base_dir=tmp_path, config=config)
    manifest, data_path = _build_manifest_and_data(tmp_path)

    import pandas as pd

    df = pd.read_csv(data_path)
    outcome = asyncio.run(orch.run(manifest, df, data_path))

    rg_dir = outcome.bundle_dir / "risk_grading"
    rg_files = sorted(rg_dir.glob("*.json"))
    assert rg_files
    rg_payload = json.loads(rg_files[0].read_text())
    assert rg_payload["risk_tier"] == "high"
    assert rg_payload["model_influence"] == "high"
    assert rg_payload["decision_consequence"] == "high"


def test_orchestrator_does_not_emit_risk_report_when_policy_disabled(tmp_path: Path) -> None:
    """A present but disabled risk_grading block is not an active report."""
    from apmode.orchestrator import Orchestrator, RunConfig

    nlmixr2 = MockNlmixr2Runner(bic=540.0)
    config = RunConfig(
        lane="discovery",
        seed=42,
        timeout_seconds=60,
        policy_path=_write_disabled_risk_grading_policy(tmp_path),
    )
    orch = Orchestrator(runner=nlmixr2, bundle_base_dir=tmp_path, config=config)
    manifest, data_path = _build_manifest_and_data(tmp_path)

    import pandas as pd

    df = pd.read_csv(data_path)
    outcome = asyncio.run(orch.run(manifest, df, data_path))

    assert outcome.recommended
    assert not (outcome.bundle_dir / "risk_grading").exists()
