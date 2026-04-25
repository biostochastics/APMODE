# SPDX-License-Identifier: GPL-2.0-or-later
"""LORO-CV end-to-end orchestrator integration tests (PRD §3.3, plan P3.B-6).

Verifies the lane gating for LORO dispatch:
  - Optimization lane: LORO fires on Gate 1 survivors and writes
    ``loro_cv/{candidate_id}.json`` artifacts.
  - Submission lane: LORO is never dispatched; no ``loro_cv`` directory is
    emitted even when the underlying data is LORO-eligible.

Uses a local mock runner that honours the full ``BackendRunner`` protocol
(including ``fixed_parameter`` for fold-level prediction runs).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

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
from apmode.orchestrator import Orchestrator, RunConfig

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


def _make_mock_result(model_id: str, bic: float = 540.0) -> BackendResult:
    """Return a classical nlmixr2 BackendResult that clears Gate 1."""
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=True,
        ofv=bic - 40.0,
        aic=bic - 20.0,
        bic=bic,
        parameter_estimates={
            "ka": ParameterEstimate(
                name="ka",
                estimate=1.0,
                se=0.1,
                rse=10.0,
                ci95_lower=0.8,
                ci95_upper=1.2,
                category="structural",
            ),
            "V": ParameterEstimate(
                name="V",
                estimate=30.0,
                se=3.0,
                rse=10.0,
                ci95_lower=24.0,
                ci95_upper=36.0,
                category="structural",
            ),
            "CL": ParameterEstimate(
                name="CL",
                estimate=2.0,
                se=0.2,
                rse=10.0,
                ci95_lower=1.6,
                ci95_upper=2.4,
                category="structural",
            ),
        },
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=300,
            gradient_norm=0.0005,
            minimization_status="successful",
            wall_time_seconds=1.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(
                cwres_mean=0.02,
                cwres_sd=1.01,
                outlier_fraction=0.01,
                obs_vs_pred_r2=0.95,
            ),
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
        ),
        wall_time_seconds=1.0,
        backend_versions={"nlmixr2": "3.0.0", "R": "4.4.1"},
        initial_estimate_source="nca",
    )


class _FullProtocolMockRunner:
    """Mock nlmixr2 runner honouring the full BackendRunner protocol.

    Unlike ``MockNlmixr2Runner`` in ``test_discovery_lane``, this mock
    accepts ``fixed_parameter`` so LORO fold dispatch does not fail with a
    TypeError (see ``BackendRunner.run`` — the kwarg is mandated by the
    protocol).
    """

    def __init__(self, bic: float = 540.0) -> None:
        self._bic = bic
        self.call_count = 0
        self.fixed_parameter_calls = 0

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
        fixed_parameter: bool = False,
        test_data_path: Path | None = None,
    ) -> BackendResult:
        self.call_count += 1
        if fixed_parameter:
            self.fixed_parameter_calls += 1
        _ = test_data_path  # mock honours the protocol but ignores the kwarg
        return _make_mock_result(spec.model_id, self._bic)


def _loro_eligible_df(n_per_group: int = 3) -> pd.DataFrame:
    """Build a canonical PK dataframe with 3 regimen groups (LORO-eligible).

    ``min_folds`` defaults to 3 in the policy, so 3 distinct modal dose
    amounts are required. Each subject carries one dose row (EVID==1) and
    six observation rows to satisfy basic profiling + NCA requirements.
    """
    amts: tuple[Literal[100, 200, 400], ...] = (100, 200, 400)
    rows: list[dict[str, object]] = []
    sid = 1
    for amt in amts:
        for _ in range(n_per_group):
            rows.append(
                {
                    "NMID": sid,
                    "TIME": 0.0,
                    "DV": 0.0,
                    "EVID": 1,
                    "AMT": float(amt),
                    "MDV": 1,
                    "CMT": 1,
                }
            )
            for t in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0):
                rows.append(
                    {
                        "NMID": sid,
                        "TIME": t,
                        "DV": float(amt) / 10.0,
                        "EVID": 0,
                        "AMT": 0.0,
                        "MDV": 0,
                        "CMT": 2,
                    }
                )
            sid += 1
    return pd.DataFrame(rows)


def _manifest_from(df: pd.DataFrame) -> DataManifest:
    return DataManifest(
        data_sha256="d" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
            mdv="MDV",
        ),
        n_subjects=int(df["NMID"].nunique()),
        n_observations=int((df["EVID"] == 0).sum()),
        n_doses=int((df["EVID"] == 1).sum()),
    )


def test_optimization_lane_runs_loro_cv_on_gate1_survivors(tmp_path: Path) -> None:
    """Optimization lane + eligible data → LORO fires and writes artifacts."""
    runner = _FullProtocolMockRunner(bic=540.0)
    config = RunConfig(
        lane="optimization",
        seed=42,
        timeout_seconds=60,
        policy_path=POLICY_DIR / "optimization.json",
        covariate_names=[],
    )
    orch = Orchestrator(runner=runner, bundle_base_dir=tmp_path, config=config)

    df = _loro_eligible_df(n_per_group=3)
    data_path = tmp_path / "loro_data.csv"
    df.to_csv(data_path, index=False)

    outcome = asyncio.run(orch.run(_manifest_from(df), df, data_path))

    loro_dir = outcome.bundle_dir / "loro_cv"
    assert loro_dir.exists(), (
        f"loro_cv/ must exist in optimization lane; bundle={outcome.bundle_dir}"
    )
    json_files = list(loro_dir.glob("*.json"))
    assert json_files, f"loro_cv/ must contain at least one candidate JSON; files={json_files}"
    sample = json.loads(json_files[0].read_text())
    assert "metrics" in sample
    assert "fold_results" in sample
    assert runner.fixed_parameter_calls > 0


def test_loro_skipped_in_submission_lane(tmp_path: Path) -> None:
    """Submission lane must not dispatch LORO, even with eligible data."""
    runner = _FullProtocolMockRunner(bic=540.0)
    config = RunConfig(
        lane="submission",
        seed=42,
        timeout_seconds=60,
        policy_path=POLICY_DIR / "submission.json",
        covariate_names=[],
    )
    orch = Orchestrator(runner=runner, bundle_base_dir=tmp_path, config=config)

    df = _loro_eligible_df(n_per_group=3)
    data_path = tmp_path / "loro_data.csv"
    df.to_csv(data_path, index=False)

    outcome = asyncio.run(orch.run(_manifest_from(df), df, data_path))

    loro_dir = outcome.bundle_dir / "loro_cv"
    assert not loro_dir.exists(), (
        f"loro_cv/ must NOT exist in submission lane; bundle={outcome.bundle_dir}"
    )
    assert runner.fixed_parameter_calls == 0


@pytest.mark.integration
def test_loro_lane_gating_is_lane_driven_not_data_driven(tmp_path: Path) -> None:
    """The data is identical in both lanes; only lane flips LORO dispatch."""
    df = _loro_eligible_df(n_per_group=3)
    manifest = _manifest_from(df)

    # Optimization run
    opt_runner = _FullProtocolMockRunner()
    opt_config = RunConfig(
        lane="optimization",
        seed=42,
        timeout_seconds=60,
        policy_path=POLICY_DIR / "optimization.json",
        covariate_names=[],
    )
    opt_orch = Orchestrator(runner=opt_runner, bundle_base_dir=tmp_path / "opt", config=opt_config)
    opt_path = tmp_path / "opt_data.csv"
    df.to_csv(opt_path, index=False)
    opt_outcome = asyncio.run(opt_orch.run(manifest, df, opt_path))

    # Submission run (same data, same manifest)
    sub_runner = _FullProtocolMockRunner()
    sub_config = RunConfig(
        lane="submission",
        seed=42,
        timeout_seconds=60,
        policy_path=POLICY_DIR / "submission.json",
        covariate_names=[],
    )
    sub_orch = Orchestrator(runner=sub_runner, bundle_base_dir=tmp_path / "sub", config=sub_config)
    sub_path = tmp_path / "sub_data.csv"
    df.to_csv(sub_path, index=False)
    sub_outcome = asyncio.run(sub_orch.run(manifest, df, sub_path))

    assert (opt_outcome.bundle_dir / "loro_cv").exists()
    assert not (sub_outcome.bundle_dir / "loro_cv").exists()
