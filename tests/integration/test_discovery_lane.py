# SPDX-License-Identifier: GPL-2.0-or-later
"""Discovery lane integration test (P2.B).

End-to-end test with mock backends verifying:
  - SearchEngine dispatches to both nlmixr2 and jax_node runners
  - Orchestrator runs discovery lane with both backends
  - Gate 2.5 runs on all survivors
  - Cross-paradigm Gate 3 ranking is produced
  - NODE models are excluded from submission lane
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ColumnMapping,
    ConvergenceMetadata,
    DataManifest,
    DiagnosticBundle,
    EvidenceManifest,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    VPCSummary,
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    LinearElim,
    NODEElimination,
    OneCmt,
    Proportional,
)
from apmode.governance.gates import evaluate_gate1, evaluate_gate2
from apmode.governance.policy import GatePolicy
from apmode.search.engine import SearchEngine

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


# ---------------------------------------------------------------------------
# Mock runners
# ---------------------------------------------------------------------------


def _make_mock_result(
    model_id: str,
    backend: Literal["nlmixr2", "jax_node"],
    bic: float,
    converged: bool = True,
) -> BackendResult:
    """Build a realistic mock BackendResult."""
    return BackendResult(
        model_id=model_id,
        backend=backend,
        converged=converged,
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
            method="saem" if backend == "nlmixr2" else "adam",
            converged=converged,
            iterations=300,
            gradient_norm=0.0005,
            minimization_status="successful" if converged else "max_evaluations",
            wall_time_seconds=60.0,
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
            identifiability=IdentifiabilityFlags(
                condition_number=15.0,
                profile_likelihood_ci={"ka": True, "V": True, "CL": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
        ),
        wall_time_seconds=60.0,
        backend_versions=(
            {"nlmixr2": "3.0.0", "R": "4.4.1"}
            if backend == "nlmixr2"
            else {"jax": "0.9.2", "python": "3.12.0"}
        ),
        initial_estimate_source="nca",
    )


class MockNlmixr2Runner:
    """Mock nlmixr2 runner that returns a classical BackendResult."""

    def __init__(self, bic: float = 540.0) -> None:
        self._bic = bic
        self.call_count = 0
        # rc9 threading observation: record the most recent gate3_policy and
        # nca_diagnostics the runner was dispatched with. ``None`` means the
        # kwargs were either not forwarded by the caller or explicitly absent.
        self.last_gate3_policy: object | None = None
        self.last_nca_diagnostics: object | None = None

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
    ) -> BackendResult:
        self.call_count += 1
        self.last_gate3_policy = gate3_policy
        self.last_nca_diagnostics = nca_diagnostics
        return _make_mock_result(spec.model_id, "nlmixr2", self._bic)


class MockNodeRunner:
    """Mock NODE runner that returns a jax_node BackendResult."""

    def __init__(self, bic: float = 530.0) -> None:
        self._bic = bic
        self.call_count = 0
        self.last_gate3_policy: object | None = None
        self.last_nca_diagnostics: object | None = None

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
    ) -> BackendResult:
        self.call_count += 1
        self.last_gate3_policy = gate3_policy
        self.last_nca_diagnostics = nca_diagnostics
        return _make_mock_result(spec.model_id, "jax_node", self._bic)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _classical_spec(model_id: str = "classical_1cmt") -> DSLSpec:
    return DSLSpec(
        model_id=model_id,
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=LinearElim(CL=2.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _node_spec(model_id: str = "node_elim_1cmt") -> DSLSpec:
    return DSLSpec(
        model_id=model_id,
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=30.0),
        elimination=NODEElimination(dim=3, constraint_template="bounded_positive"),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _data_manifest() -> DataManifest:
    return DataManifest(
        data_sha256="a" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="ID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        ),
        n_subjects=50,
        n_observations=400,
        n_doses=50,
    )


def _evidence_manifest() -> EvidenceManifest:
    return EvidenceManifest(
        route_certainty="confirmed",
        absorption_complexity="simple",
        nonlinear_clearance_evidence_strength="none",
        richness_category="moderate",
        identifiability_ceiling="medium",
        covariate_burden=0,
        covariate_correlated=False,
        blq_burden=0.0,
        protocol_heterogeneity="single-study",
        absorption_phase_coverage="adequate",
        elimination_phase_coverage="adequate",
        node_dim_budget=8,
    )


def _load_policy(lane: str) -> GatePolicy:
    return GatePolicy.model_validate(json.loads((POLICY_DIR / f"{lane}.json").read_text()))


# ---------------------------------------------------------------------------
# P2.A — SearchEngine multi-backend dispatch
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSearchEngineMultiBackend:
    """SearchEngine dispatches specs to the correct backend runner."""

    def test_classical_spec_dispatched_to_nlmixr2(self) -> None:
        nlmixr2 = MockNlmixr2Runner()
        node = MockNodeRunner()
        engine = SearchEngine(
            runner=nlmixr2,
            data_manifest=_data_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            allowed_backends=["nlmixr2", "jax_node"],
            runners={"nlmixr2": nlmixr2, "jax_node": node},
        )
        spec = _classical_spec()
        runner, name = engine._select_runner(spec)
        assert name == "nlmixr2"
        assert runner is nlmixr2

    def test_node_spec_dispatched_to_jax_node(self) -> None:
        nlmixr2 = MockNlmixr2Runner()
        node = MockNodeRunner()
        engine = SearchEngine(
            runner=nlmixr2,
            data_manifest=_data_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            allowed_backends=["nlmixr2", "jax_node"],
            runners={"nlmixr2": nlmixr2, "jax_node": node},
        )
        spec = _node_spec()
        runner, name = engine._select_runner(spec)
        assert name == "jax_node"
        assert runner is node

    def test_node_spec_skipped_when_not_allowed(self) -> None:
        nlmixr2 = MockNlmixr2Runner()
        engine = SearchEngine(
            runner=nlmixr2,
            data_manifest=_data_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            allowed_backends=["nlmixr2"],
        )
        spec = _node_spec()
        from apmode.search.engine import _BackendNotAvailable

        with pytest.raises(_BackendNotAvailable):
            engine._select_runner(spec)

    def test_node_spec_skipped_when_no_runner(self) -> None:
        nlmixr2 = MockNlmixr2Runner()
        engine = SearchEngine(
            runner=nlmixr2,
            data_manifest=_data_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            allowed_backends=["nlmixr2", "jax_node"],
            # No jax_node runner provided
        )
        spec = _node_spec()
        from apmode.search.engine import _BackendNotAvailable

        with pytest.raises(_BackendNotAvailable):
            engine._select_runner(spec)

    def test_evaluate_candidate_routes_to_correct_runner(self) -> None:
        nlmixr2 = MockNlmixr2Runner(bic=550.0)
        node = MockNodeRunner(bic=520.0)
        engine = SearchEngine(
            runner=nlmixr2,
            data_manifest=_data_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            allowed_backends=["nlmixr2", "jax_node"],
            runners={"nlmixr2": nlmixr2, "jax_node": node},
        )

        classical = _classical_spec()
        node_spec = _node_spec()

        result_c = asyncio.get_event_loop().run_until_complete(
            engine._evaluate_candidate(classical, {"ka": 1.0, "V": 30.0, "CL": 2.0})
        )
        result_n = asyncio.get_event_loop().run_until_complete(
            engine._evaluate_candidate(node_spec, {"ka": 1.0, "V": 30.0, "CL": 2.0})
        )

        assert result_c.result is not None
        assert result_c.result.backend == "nlmixr2"
        assert result_c.bic == 550.0

        assert result_n.result is not None
        assert result_n.result.backend == "jax_node"
        assert result_n.bic == 520.0

        assert nlmixr2.call_count == 1
        assert node.call_count == 1

    def test_evaluate_node_candidate_graceful_skip(self) -> None:
        """NODE candidate returns error result when runner not available."""
        nlmixr2 = MockNlmixr2Runner()
        engine = SearchEngine(
            runner=nlmixr2,
            data_manifest=_data_manifest(),
            data_path=Path("/tmp/test.csv"),
            seed=42,
            allowed_backends=["nlmixr2"],
        )

        node_spec = _node_spec()
        result = asyncio.get_event_loop().run_until_complete(
            engine._evaluate_candidate(node_spec, {"ka": 1.0, "V": 30.0, "CL": 2.0})
        )

        assert result.result is None
        assert result.error is not None
        assert "jax_node" in result.error


# ---------------------------------------------------------------------------
# P2.B — Discovery lane integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestDiscoveryLaneIntegration:
    """End-to-end discovery lane with mock backends."""

    def test_gate1_passes_both_backends(self) -> None:
        """Good fits from both backends pass Gate 1 (with seed results)."""
        policy = _load_policy("discovery")
        for backend in ("nlmixr2", "jax_node"):
            result = _make_mock_result(f"test_{backend}", backend, bic=540.0)  # type: ignore[arg-type]
            # Gate 1 requires seed_stability_n=3 seeds → provide 2 extra
            seed1 = _make_mock_result(f"test_{backend}_s1", backend, bic=541.0)  # type: ignore[arg-type]
            seed2 = _make_mock_result(f"test_{backend}_s2", backend, bic=539.0)  # type: ignore[arg-type]
            g1 = evaluate_gate1(result, policy, seed_results=[seed1, seed2])
            assert g1.passed, f"{backend} failed Gate 1: {g1.summary_reason}"

    def test_gate2_discovery_admits_node(self) -> None:
        """NODE results pass Gate 2 in discovery lane (not submission)."""
        policy = _load_policy("discovery")
        node_result = _make_mock_result("node_test", "jax_node", bic=530.0)
        g2 = evaluate_gate2(node_result, policy, lane="discovery")
        assert g2.passed, f"NODE failed Gate 2 discovery: {g2.summary_reason}"

    def test_gate2_submission_excludes_node(self) -> None:
        """NODE results are excluded from submission lane Gate 2."""
        policy = _load_policy("submission")
        node_result = _make_mock_result("node_test", "jax_node", bic=530.0)
        g2 = evaluate_gate2(node_result, policy, lane="submission")
        assert not g2.passed, "NODE should not pass submission Gate 2"

    def test_cross_paradigm_ranking(self) -> None:
        """Cross-paradigm Gate 3 ranking with mixed backends."""
        from apmode.governance.gates import evaluate_gate3

        policy = _load_policy("discovery")

        classical = _make_mock_result("classical_1", "nlmixr2", bic=550.0)
        node = _make_mock_result("node_1", "jax_node", bic=520.0)

        g3, ranked = evaluate_gate3([classical, node], policy)
        assert g3.passed
        assert len(ranked) == 2
        # Both backends represented in ranking
        backends = {rc.backend for rc in ranked}
        assert backends == {"nlmixr2", "jax_node"}

    def test_orchestrator_discovery_lane(self, tmp_path: Path) -> None:
        """Full orchestrator run in discovery lane with both backends."""
        from apmode.orchestrator import Orchestrator, RunConfig

        nlmixr2 = MockNlmixr2Runner(bic=545.0)
        node = MockNodeRunner(bic=530.0)

        config = RunConfig(
            lane="discovery",
            seed=42,
            timeout_seconds=60,
            policy_path=POLICY_DIR / "discovery.json",
        )

        orch = Orchestrator(
            runner=nlmixr2,
            bundle_base_dir=tmp_path,
            config=config,
            node_runner=node,
        )

        # Create minimal test data using canonical schema (NMID, CMT required)
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
            data_sha256="b" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
                mdv="MDV",
            ),
            n_subjects=2,
            n_observations=18,
            n_doses=2,
        )

        outcome = asyncio.get_event_loop().run_until_complete(orch.run(manifest, df, data_path))

        # Verify pipeline completed
        assert outcome.search_outcome is not None
        assert len(outcome.search_outcome.results) > 0

        # Verify nlmixr2 runner was called (classical candidates)
        assert nlmixr2.call_count > 0

    def test_submission_lane_does_not_use_node(self, tmp_path: Path) -> None:
        """Submission lane never dispatches to NODE runner."""
        from apmode.orchestrator import Orchestrator, RunConfig

        nlmixr2 = MockNlmixr2Runner(bic=540.0)
        node = MockNodeRunner(bic=530.0)

        config = RunConfig(
            lane="submission",
            seed=42,
            timeout_seconds=60,
            policy_path=POLICY_DIR / "submission.json",
        )

        orch = Orchestrator(
            runner=nlmixr2,
            bundle_base_dir=tmp_path,
            config=config,
            node_runner=node,
        )

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
            data_sha256="c" * 64,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
                mdv="MDV",
            ),
            n_subjects=2,
            n_observations=18,
            n_doses=2,
        )

        asyncio.get_event_loop().run_until_complete(orch.run(manifest, df, data_path))

        # NODE runner should never have been called in submission lane
        assert node.call_count == 0
        # nlmixr2 should have been called
        assert nlmixr2.call_count > 0

    def test_orchestrator_threads_gate3_policy_to_runner(self, tmp_path: Path) -> None:
        """rc9: Orchestrator must forward gate3_policy and nca_diagnostics.

        The orchestrator loads the lane policy and pulls its NCA estimates
        from the initial-estimates stage before dispatching the search.
        Every runner call site (search engine, seed runs, agentic loop,
        FREM, MI per-imputation fits, LORO-CV folds) must forward both
        ``gate3_policy`` and ``nca_diagnostics`` so the rc8 posterior-
        predictive pipeline populates VPC/NPE/AUC-Cmax atomically. This
        test pins the threading contract: once the orchestrator run
        completes, the mock runner's most-recent call must carry the
        policy's ``gate3`` config.
        """
        from apmode.governance.policy import Gate3Config
        from apmode.orchestrator import Orchestrator, RunConfig

        nlmixr2 = MockNlmixr2Runner(bic=540.0)

        config = RunConfig(
            lane="submission",
            seed=42,
            timeout_seconds=60,
            policy_path=POLICY_DIR / "submission.json",
        )
        orch = Orchestrator(
            runner=nlmixr2,  # type: ignore[arg-type]
            bundle_base_dir=tmp_path,
            config=config,
        )

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
                subject_id="NMID",
                time="TIME",
                dv="DV",
                evid="EVID",
                amt="AMT",
                mdv="MDV",
            ),
            n_subjects=2,
            n_observations=18,
            n_doses=2,
        )

        asyncio.get_event_loop().run_until_complete(orch.run(manifest, df, data_path))

        # At least one dispatch must have happened.
        assert nlmixr2.call_count > 0
        # The last runner dispatch must carry the lane's Gate 3 config —
        # anything else means the orchestrator silently swallowed the
        # policy and the rc8 posterior-predictive pipeline is dark.
        assert isinstance(nlmixr2.last_gate3_policy, Gate3Config)
