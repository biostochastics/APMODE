# SPDX-License-Identifier: GPL-2.0-or-later
"""B1b-refit integration tests — fidelity-gated promotion of a distilled NODE
surrogate into Gate 3 (plan ``docs/plans/2026-07-10-node-remaining-roadmap.md``
§4.1).

Exercises ``Orchestrator._run_node_distillation_promotion`` end-to-end through
``orch.run()``:

* a ``jax_node`` candidate whose distilled parametric surrogate clears fidelity
  is re-fit through the classical ``Nlmixr2Runner`` and admitted into the SAME
  Gate 1/2/3 pipeline (a ``<id>_distilled`` classical candidate with its own
  Gate-1 decision + a lineage edge back to the NODE run);
* a failing-fidelity report is NOT promoted;
* a re-fit that raises is recorded fail-soft as
  ``gate_failed="distillation_refit"`` without aborting the run;
* the promotion threshold is read from the NODE runner's
  ``fidelity_min_r_squared`` (``getattr`` fallback ``0.8``).

The automated search never *generates* NODE candidates (they arrive from
explicit NODE dispatch / agentic transforms), so these tests stub the
``SearchEngine`` to inject a controlled ``jax_node`` result carrying a
``DistillationReport`` — the realistic upstream that the promotion stage
consumes. The injected NODE result is marked non-converged so it fails Gate 1
and never reaches Gate 3, isolating each assertion to the promoted, within-
classical-paradigm nlmixr2 candidate (promotion is fidelity-gated, not source-
convergence-gated — plan §4.1).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ColumnMapping,
    ConvergenceMetadata,
    DataManifest,
    DiagnosticBundle,
    DistillationReport,
    FidelityResult,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    PITCalibrationSummary,
    ScoringContract,
    SurrogateResult,
    VPCSummary,
)
from apmode.dsl.ast_models import (
    DSLSpec,
    FirstOrder,
    NODEElimination,
    OneCmt,
    Proportional,
)
from apmode.errors import BackendError
from apmode.orchestrator import Orchestrator, RunConfig
from apmode.search.engine import SearchOutcome, SearchResult

if TYPE_CHECKING:
    import pytest

POLICY_DIR = Path(__file__).parent.parent.parent / "policies"


# --------------------------------------------------------------------------- #
# Result / distillation fixtures                                              #
# --------------------------------------------------------------------------- #
def _param_estimates() -> dict[str, ParameterEstimate]:
    return {
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
    }


def _classical_scoring_contract() -> ScoringContract:
    return ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="nlmixr2_focei",
        blq_method="none",
        observation_model="combined",
        float_precision="float64",
    )


def _node_scoring_contract() -> ScoringContract:
    # jax_node pooled-likelihood contract (no random effects yet): the model
    # validator restricts ``nlpd_integrator`` to laplace_*/none for jax_node.
    return ScoringContract(
        nlpd_kind="conditional",
        re_treatment="pooled",
        nlpd_integrator="none",
        blq_method="none",
        observation_model="combined",
        float_precision="float32",
    )


def _diagnostics(scoring_contract: ScoringContract | None = None) -> DiagnosticBundle:
    return DiagnosticBundle(
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
        scoring_contract=scoring_contract or _classical_scoring_contract(),
    )


def _make_classical_result(model_id: str, bic: float = 540.0) -> BackendResult:
    """A converged classical nlmixr2 result that clears Gate 1."""
    return BackendResult(
        model_id=model_id,
        backend="nlmixr2",
        converged=True,
        ofv=bic - 40.0,
        aic=bic - 20.0,
        bic=bic,
        parameter_estimates=_param_estimates(),
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=300,
            gradient_norm=0.0005,
            minimization_status="successful",
            wall_time_seconds=1.0,
        ),
        diagnostics=_diagnostics(),
        wall_time_seconds=1.0,
        backend_versions={"nlmixr2": "3.0.0", "R": "4.4.1"},
        initial_estimate_source="nca",
    )


def _make_node_result(
    model_id: str,
    report: DistillationReport,
    *,
    converged: bool = False,
    bic: float = 560.0,
) -> BackendResult:
    """A ``jax_node`` result carrying a distillation report.

    Marked non-converged by default so it fails Gate 1 and never reaches Gate 3
    — promotion of its distilled surrogate is fidelity-gated, independent of the
    source NODE fit's own convergence (plan §4.1).
    """
    return BackendResult(
        model_id=model_id,
        backend="jax_node",
        converged=converged,
        ofv=bic - 40.0,
        aic=bic - 20.0,
        bic=bic,
        parameter_estimates=_param_estimates(),
        eta_shrinkage={"CL": 0.05, "V": 0.08},
        convergence_metadata=ConvergenceMetadata(
            method="adam",
            converged=converged,
            iterations=500,
            gradient_norm=0.01,
            minimization_status="successful" if converged else "terminated",
            wall_time_seconds=2.0,
        ),
        diagnostics=_diagnostics(_node_scoring_contract()),
        wall_time_seconds=2.0,
        backend_versions={"jax": "0.4.30"},
        initial_estimate_source="nca",
        distillation=report,
    )


def _distillation_report(
    candidate_id: str,
    *,
    r_squared: float = 0.99,
    overall_pass: bool = True,
) -> DistillationReport:
    """A linear-elimination distillation report.

    ``distillation_passes_fidelity`` requires BOTH ``fidelity.overall_pass`` and
    ``surrogate.r_squared >= min_r_squared``; toggle either input to build a
    non-promotable report.
    """
    return DistillationReport(
        candidate_id=candidate_id,
        node_position="elimination",
        sub_function_x=[0.1, 1.0, 10.0],
        sub_function_y=[0.05, 0.5, 5.0],
        surrogate=SurrogateResult(
            surrogate_type="linear",
            params={"slope": 0.05, "intercept": 0.0},
            residual_ss=0.001,
            r_squared=r_squared,
        ),
        fidelity=FidelityResult(
            auc_gmr=1.0,
            cmax_gmr=1.0,
            auc_pass=overall_pass,
            cmax_pass=overall_pass,
            overall_pass=overall_pass,
        ),
    )


def _node_spec(model_id: str) -> DSLSpec:
    return DSLSpec(
        model_id=model_id,
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=NODEElimination(dim=2, constraint_template="saturable"),
        variability=[],
        observation=Proportional(sigma_prop=0.1),
    )


# --------------------------------------------------------------------------- #
# Data + runner + stubbed-search fixtures                                     #
# --------------------------------------------------------------------------- #
def _pk_df() -> pd.DataFrame:
    """A small single-dose oral PK dataframe (6 subjects, 6 obs each)."""
    rows: list[dict[str, object]] = []
    for sid in range(1, 7):
        rows.append(
            {"NMID": sid, "TIME": 0.0, "DV": 0.0, "EVID": 1, "AMT": 100.0, "MDV": 1, "CMT": 1}
        )
        for t in (0.5, 1.0, 2.0, 4.0, 8.0, 12.0):
            rows.append(
                {
                    "NMID": sid,
                    "TIME": t,
                    "DV": round(10.0 - t * 0.3, 3),
                    "EVID": 0,
                    "AMT": 0.0,
                    "MDV": 0,
                    "CMT": 2,
                }
            )
    return pd.DataFrame(rows)


def _manifest_from(df: pd.DataFrame) -> DataManifest:
    return DataManifest(
        data_sha256="e" * 64,
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


class _SuccessClassicalRunner:
    """Classical nlmixr2 mock: always returns a converged result. Used for the
    distillation re-fit and for seed-stability re-runs of the promoted
    candidate."""

    def __init__(self, bic: float = 540.0) -> None:
        self._bic = bic
        self.call_count = 0
        self.refit_model_ids: list[str] = []

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
        self.refit_model_ids.append(spec.model_id)
        _ = (fixed_parameter, test_data_path)
        return _make_classical_result(spec.model_id, self._bic)


class _RaisingClassicalRunner:
    """Classical nlmixr2 mock: always raises ``BackendError`` — stands in for a
    distilled surrogate that is faithful yet unidentifiable / non-convergent
    under nlmixr2 (fail-soft path, plan §3 risk #2)."""

    def __init__(self) -> None:
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
        fixed_parameter: bool = False,
        test_data_path: Path | None = None,
    ) -> BackendResult:
        self.call_count += 1
        msg = "nlmixr2 re-fit of distilled surrogate diverged (singular Omega)"
        raise BackendError(msg)


class _ThresholdNodeRunner:
    """Minimal NODE runner stand-in exposing ``fidelity_min_r_squared`` so the
    promotion stage reads the threshold via ``getattr(node_runner, ...)``. Its
    ``run`` is never invoked (search is stubbed; the injected NODE result is
    non-converged so no seed run dispatches to it)."""

    def __init__(self, fidelity_min_r_squared: float) -> None:
        self.fidelity_min_r_squared = fidelity_min_r_squared

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
    ) -> BackendResult:  # pragma: no cover - never called in these tests
        raise NotImplementedError


def _install_stub_search(monkeypatch: pytest.MonkeyPatch, results: list[SearchResult]) -> None:
    """Replace ``apmode.search.engine.SearchEngine`` with a stub whose ``run``
    returns a fixed ``SearchOutcome`` — the orchestrator imports the name lazily
    inside ``run()``, so patching the module attribute is sufficient."""

    class _StubSearchEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, *args: object, **kwargs: object) -> SearchOutcome:
            return SearchOutcome(results=list(results))

    monkeypatch.setattr("apmode.search.engine.SearchEngine", _StubSearchEngine)


def _discovery_config() -> RunConfig:
    return RunConfig(
        lane="discovery",
        seed=42,
        timeout_seconds=60,
        policy_path=POLICY_DIR / "discovery.json",
        covariate_names=[],
    )


def _read_failed_candidates(bundle_dir: Path) -> list[dict[str, object]]:
    path = bundle_dir / "failed_candidates.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #
def test_passing_distillation_is_promoted_and_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fidelity-clearing NODE distillation → a ``<id>_distilled`` classical
    candidate that is re-fit and flows through the SAME Gate 1 pipeline, with a
    lineage edge back to the NODE run and a re-sealed promoted report."""
    df = _pk_df()
    manifest = _manifest_from(df)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    report = _distillation_report("node_cand", r_squared=0.99, overall_pass=True)
    node_sr = SearchResult(
        candidate_id="node_cand",
        spec=_node_spec("node_cand"),
        result=_make_node_result("node_cand", report, converged=False),
        converged=False,
        bic=560.0,
        aic=540.0,
        n_params=3,
    )
    _install_stub_search(monkeypatch, [node_sr])

    runner = _SuccessClassicalRunner(bic=540.0)
    orch = Orchestrator(
        runner=runner,
        bundle_base_dir=tmp_path / "bundle",
        config=_discovery_config(),
    )

    outcome = asyncio.run(orch.run(manifest, df, data_path))

    distilled_id = "node_cand_distilled"

    # 1. The classical Nlmixr2Runner was invoked to re-fit the promoted spec.
    assert distilled_id in runner.refit_model_ids

    # 2. The promoted candidate has its own Gate-1 decision (flowed into the
    #    SAME governance pipeline) and passed.
    g1_path = outcome.bundle_dir / "gate_decisions" / f"gate1_{distilled_id}.json"
    assert g1_path.exists(), f"missing Gate-1 decision for promoted candidate: {g1_path}"
    assert (distilled_id, True) in outcome.gate1_results

    # 3. The distillation report was re-sealed with promotion provenance.
    dist_path = outcome.bundle_dir / "distillation" / "node_cand.json"
    sealed = json.loads(dist_path.read_text())
    assert sealed["promoted"] is True
    assert sealed["promoted_model_id"] == distilled_id

    # 4. A lineage edge (source NODE run -> distilled candidate) was recorded.
    lineage = json.loads((outcome.bundle_dir / "candidate_lineage.json").read_text())
    edges = {(e["parent_id"], e["candidate_id"]) for e in lineage["entries"]}
    assert ("node_cand", distilled_id) in edges

    # 5. The run sealed successfully.
    assert (outcome.bundle_dir / "_COMPLETE").exists()


def test_failing_fidelity_is_not_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A distillation whose fidelity does not clear → no promotion, no re-fit,
    no ``distillation_refit`` failed-candidate entry, report stays unpromoted."""
    df = _pk_df()
    manifest = _manifest_from(df)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    # overall_pass=False → distillation_passes_fidelity is False.
    report = _distillation_report("node_cand", r_squared=0.99, overall_pass=False)
    node_sr = SearchResult(
        candidate_id="node_cand",
        spec=_node_spec("node_cand"),
        result=_make_node_result("node_cand", report, converged=False),
        converged=False,
        bic=560.0,
        aic=540.0,
        n_params=3,
    )
    _install_stub_search(monkeypatch, [node_sr])

    runner = _SuccessClassicalRunner(bic=540.0)
    orch = Orchestrator(
        runner=runner,
        bundle_base_dir=tmp_path / "bundle",
        config=_discovery_config(),
    )

    outcome = asyncio.run(orch.run(manifest, df, data_path))

    distilled_id = "node_cand_distilled"

    # No re-fit, no promoted candidate, no gate decision for it.
    assert distilled_id not in runner.refit_model_ids
    assert not (outcome.bundle_dir / "gate_decisions" / f"gate1_{distilled_id}.json").exists()

    # No distillation_refit failed-candidate entry.
    failed = _read_failed_candidates(outcome.bundle_dir)
    assert not any(fc["gate_failed"] == "distillation_refit" for fc in failed)

    # Report sealed but unpromoted.
    sealed = json.loads((outcome.bundle_dir / "distillation" / "node_cand.json").read_text())
    assert sealed["promoted"] is False
    assert sealed["promoted_model_id"] is None
    assert (outcome.bundle_dir / "_COMPLETE").exists()


def test_refit_failure_is_recorded_fail_soft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A faithful surrogate whose nlmixr2 re-fit raises → a
    ``gate_failed="distillation_refit"`` failed-candidate, and the run still
    completes (never crashes)."""
    df = _pk_df()
    manifest = _manifest_from(df)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    report = _distillation_report("node_cand", r_squared=0.99, overall_pass=True)
    node_sr = SearchResult(
        candidate_id="node_cand",
        spec=_node_spec("node_cand"),
        result=_make_node_result("node_cand", report, converged=False),
        converged=False,
        bic=560.0,
        aic=540.0,
        n_params=3,
    )
    _install_stub_search(monkeypatch, [node_sr])

    runner = _RaisingClassicalRunner()
    orch = Orchestrator(
        runner=runner,
        bundle_base_dir=tmp_path / "bundle",
        config=_discovery_config(),
    )

    outcome = asyncio.run(orch.run(manifest, df, data_path))

    distilled_id = "node_cand_distilled"

    # The re-fit was attempted (and raised).
    assert runner.call_count >= 1

    # Fail-soft: recorded as a distillation_refit failed candidate.
    failed = _read_failed_candidates(outcome.bundle_dir)
    matching = [
        fc
        for fc in failed
        if fc["candidate_id"] == distilled_id and fc["gate_failed"] == "distillation_refit"
    ]
    assert matching, f"expected a distillation_refit failed candidate; got {failed}"
    assert matching[0]["backend"] == "nlmixr2"

    # The promoted candidate was NOT admitted to governance.
    assert not (outcome.bundle_dir / "gate_decisions" / f"gate1_{distilled_id}.json").exists()

    # The run completed and sealed despite the fail-soft.
    assert (outcome.bundle_dir / "_COMPLETE").exists()


def test_node_runner_fidelity_threshold_gates_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promotion R² floor is read from the NODE runner's
    ``fidelity_min_r_squared``: a report at R²=0.90 is promotable under the 0.8
    default but NOT under a runner configured at 0.95."""
    df = _pk_df()
    manifest = _manifest_from(df)
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    report = _distillation_report("node_cand", r_squared=0.90, overall_pass=True)
    node_sr = SearchResult(
        candidate_id="node_cand",
        spec=_node_spec("node_cand"),
        result=_make_node_result("node_cand", report, converged=False),
        converged=False,
        bic=560.0,
        aic=540.0,
        n_params=3,
    )
    _install_stub_search(monkeypatch, [node_sr])

    runner = _SuccessClassicalRunner(bic=540.0)
    orch = Orchestrator(
        runner=runner,
        bundle_base_dir=tmp_path / "bundle",
        config=_discovery_config(),
        node_runner=_ThresholdNodeRunner(fidelity_min_r_squared=0.95),
    )

    outcome = asyncio.run(orch.run(manifest, df, data_path))

    distilled_id = "node_cand_distilled"

    # 0.90 < 0.95 → below the runner's floor → not promoted.
    assert distilled_id not in runner.refit_model_ids
    assert not (outcome.bundle_dir / "gate_decisions" / f"gate1_{distilled_id}.json").exists()
    sealed = json.loads((outcome.bundle_dir / "distillation" / "node_cand.json").read_text())
    assert sealed["promoted"] is False
    assert (outcome.bundle_dir / "_COMPLETE").exists()
