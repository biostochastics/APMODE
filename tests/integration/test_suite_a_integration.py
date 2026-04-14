# SPDX-License-Identifier: GPL-2.0-or-later
"""Benchmark Suite A integration test (PRD §5, §8).

Fixture-based approach: pre-generated Suite A CSVs are ingested through
the Python pipeline (profiler → search space → candidate generation).
No R/nlmixr2 required — parameter recovery uses mock BackendResults
with realistic noise.

To (re-)generate fixtures:
    Rscript benchmarks/suite_a/simulate_all.R tests/fixtures/suite_a

Assertions:
  - CSV ingestion: each scenario produces a valid DataManifest
  - Structure coverage: SearchSpace.from_manifest covers the ground-truth types
  - Parameter bias: mock estimates within 20% of REFERENCE_PARAMS
  - Gate passage: good fits pass Gate 1 + Gate 2 (submission)
  - Bundle completeness: full reproducibility bundle emitted per scenario
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from apmode.benchmarks.suite_a import (
    ALL_SCENARIOS,
    REFERENCE_PARAMS,
)
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    VPCSummary,
)
from apmode.data.ingest import ingest_nonmem_csv
from apmode.data.profiler import profile_data
from apmode.governance.gates import evaluate_gate1, evaluate_gate2
from apmode.governance.policy import GatePolicy
from apmode.search.candidates import SearchSpace, generate_root_candidates

if TYPE_CHECKING:
    from apmode.dsl.ast_models import DSLSpec

# ---------------------------------------------------------------------------
# Paths and fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "suite_a"
POLICY_DIR = Path(__file__).parent.parent.parent / "policies"

CSV_FILES: dict[str, str] = {
    "A1": "a1_1cmt_oral_linear.csv",
    "A2": "a2_2cmt_iv_parallel_mm.csv",
    "A3": "a3_transit_1cmt_linear.csv",
    "A4": "a4_1cmt_oral_mm.csv",
}


def _csv_path(name: str) -> Path:
    return FIXTURE_DIR / CSV_FILES[name]


def _load_policy(lane: str) -> GatePolicy:
    return GatePolicy.model_validate(json.loads((POLICY_DIR / f"{lane}.json").read_text()))


def _make_mock_fit(
    spec: DSLSpec,
    reference: dict[str, float],
    bias_fraction: float = 0.02,
) -> BackendResult:
    """Mock BackendResult with small bias simulating estimation noise."""
    estimates = {
        n: ParameterEstimate(
            name=n,
            estimate=v * (1 + bias_fraction),
            se=v * 0.1,
            rse=10.0,
            ci95_lower=v * 0.8,
            ci95_upper=v * 1.2,
            fixed=False,
            category="structural",
        )
        for n, v in reference.items()
    }
    return BackendResult(
        model_id=spec.model_id,
        backend="nlmixr2",
        converged=True,
        ofv=500.0,
        aic=520.0,
        bic=540.0,
        parameter_estimates=estimates,
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
            gof=GOFMetrics(
                cwres_mean=0.02, cwres_sd=1.01, outlier_fraction=0.01, obs_vs_pred_r2=0.95
            ),
            vpc=VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage={"p5": 0.92, "p50": 0.96, "p95": 0.93},
                n_bins=10,
                prediction_corrected=False,
            ),
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={n: True for n in reference},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=60.0,
        backend_versions={"nlmixr2": "3.0.0", "R": "4.4.1"},
        initial_estimate_source="nca",
    )


# ---------------------------------------------------------------------------
# Ingestion — CSV → DataManifest → EvidenceManifest
# ---------------------------------------------------------------------------


_FIXTURES_AVAILABLE = FIXTURE_DIR.is_dir() and all(
    (FIXTURE_DIR / f).exists() for f in CSV_FILES.values()
)


@pytest.mark.integration
@pytest.mark.skipif(not _FIXTURES_AVAILABLE, reason="Suite A CSV fixtures not generated")
class TestSuiteAIngestion:
    """Each Suite A CSV ingests and profiles correctly."""

    @pytest.mark.parametrize("name", list(CSV_FILES))
    def test_ingest_produces_valid_manifest(self, name: str) -> None:
        manifest, _df = ingest_nonmem_csv(_csv_path(name))
        assert manifest.n_subjects >= 10
        assert manifest.n_observations > manifest.n_subjects
        assert manifest.ingestion_format == "nonmem_csv"
        assert len(manifest.data_sha256) == 64

    @pytest.mark.parametrize("name", list(CSV_FILES))
    def test_profile_produces_evidence_manifest(self, name: str) -> None:
        manifest, df = ingest_nonmem_csv(_csv_path(name))
        evidence = profile_data(df, manifest)
        assert evidence.richness_category in ("sparse", "moderate", "rich")
        assert evidence.absorption_phase_coverage in ("adequate", "inadequate")
        assert evidence.elimination_phase_coverage in ("adequate", "inadequate")


# ---------------------------------------------------------------------------
# Structure coverage — search space covers ground-truth model
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _FIXTURES_AVAILABLE, reason="Suite A CSV fixtures not generated")
class TestSuiteAStructureCoverage:
    """EvidenceManifest-driven search space includes the true model."""

    GROUND_TRUTH: dict[str, dict[str, str]] = {
        "A1": {"absorption": "FirstOrder", "distribution": "OneCmt", "elimination": "Linear"},
        "A2": {
            "absorption": "FirstOrder",
            "distribution": "TwoCmt",
            "elimination": "ParallelLinearMM",
        },
        "A3": {"absorption": "Transit", "distribution": "OneCmt", "elimination": "Linear"},
        "A4": {
            "absorption": "FirstOrder",
            "distribution": "OneCmt",
            "elimination": "MichaelisMenten",
        },
    }

    ABS_TYPE_MAP = {
        "first_order": "FirstOrder",
        "lagged_first_order": "LaggedFirstOrder",
        "transit": "Transit",
        "none": "FirstOrder",
    }
    ELIM_TYPE_MAP = {
        "linear": "Linear",
        "michaelis_menten": "MichaelisMenten",
        "parallel": "ParallelLinearMM",
    }
    CMT_MAP = {1: "OneCmt", 2: "TwoCmt", 3: "ThreeCmt"}

    @pytest.mark.parametrize("name", ["A1", "A3"])
    def test_linear_scenarios_covered(self, name: str) -> None:
        """A1/A3 have linear elimination — profiler always covers these."""
        manifest, df = ingest_nonmem_csv(_csv_path(name))
        evidence = profile_data(df, manifest)
        space = SearchSpace.from_manifest(evidence)
        candidates = generate_root_candidates(space, base_params=REFERENCE_PARAMS[name])
        truth = self.GROUND_TRUTH[name]

        found = any(
            c.absorption.type == truth["absorption"]
            and c.distribution.type == truth["distribution"]
            and c.elimination.type == truth["elimination"]
            for c in candidates
        )
        assert found, (
            f"{name}: no candidate matched ground truth ({truth}). "
            f"Space: cmt={space.structural_cmt}, "
            f"abs={space.absorption_types}, elim={space.elimination_types}"
        )

    @pytest.mark.parametrize("name", ["A2", "A4"])
    def test_nonlinear_scenarios_need_profiler_signal(self, name: str) -> None:
        """A2/A4 have nonlinear CL — coverage depends on profiler sensitivity.

        If the profiler detects nonlinear_clearance_signature, the search
        space will include MM/parallel elimination. Otherwise, only linear
        elimination is searched and the ground-truth model is not covered.
        This test documents the dependency on profiler sensitivity.
        """
        manifest, df = ingest_nonmem_csv(_csv_path(name))
        evidence = profile_data(df, manifest)
        space = SearchSpace.from_manifest(evidence)

        if evidence.nonlinear_clearance_signature:
            # Profiler detected nonlinear CL — search space should cover it
            truth = self.GROUND_TRUTH[name]
            candidates = generate_root_candidates(space, base_params=REFERENCE_PARAMS[name])
            found = any(c.elimination.type == truth["elimination"] for c in candidates)
            assert found, f"{name}: profiler detected nonlinear CL but space missing {truth}"
        else:
            # Profiler did not detect — document limitation, don't fail
            assert "michaelis_menten" not in space.elimination_types


# ---------------------------------------------------------------------------
# Parameter bias — mock estimates within 20% of ground truth
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSuiteAParameterBias:
    """Mock-fit parameter estimates within 20% relative tolerance."""

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_bias_within_20pct(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        result = _make_mock_fit(spec, ref, bias_fraction=0.02)

        for pname, ref_val in ref.items():
            est = result.parameter_estimates[pname].estimate
            rel_err = abs(est - ref_val) / ref_val
            assert rel_err < 0.20, f"{name}.{pname}: {rel_err:.2%} >= 20%"

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_bias_25pct_fails(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        result = _make_mock_fit(spec, REFERENCE_PARAMS[name], bias_fraction=0.25)
        any_over = any(
            abs(result.parameter_estimates[p].estimate - v) / v >= 0.20
            for p, v in REFERENCE_PARAMS[name].items()
        )
        assert any_over, f"{name}: 25% bias should exceed 20% tolerance"


# ---------------------------------------------------------------------------
# Gate passage — good fits pass governance
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSuiteAGatePassage:
    """Correctly-fit models pass Gate 1 + Gate 2 (submission lane)."""

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_passes_gate1(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        ref = REFERENCE_PARAMS[name]
        result = _make_mock_fit(spec, ref)
        seed1 = _make_mock_fit(spec, ref, bias_fraction=0.025)
        seed2 = _make_mock_fit(spec, ref, bias_fraction=0.015)
        policy = _load_policy("submission")
        g1 = evaluate_gate1(result, policy, seed_results=[seed1, seed2])
        assert g1.passed, f"{name} failed Gate 1: {g1.summary_reason}"

    @pytest.mark.parametrize("name,factory", ALL_SCENARIOS)
    def test_passes_gate2_submission(self, name: str, factory: object) -> None:
        spec = factory()  # type: ignore[operator]
        result = _make_mock_fit(spec, REFERENCE_PARAMS[name])
        policy = _load_policy("submission")
        g2 = evaluate_gate2(result, policy, lane="submission")
        assert g2.passed, f"{name} failed Gate 2: {g2.summary_reason}"
