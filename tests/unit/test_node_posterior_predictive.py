# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE pooled posterior-predictive diagnostics (plan §4.2, consensus subset).

Ships only NPE + AUC/Cmax-BE + real pooled CWRES. VPC / NPDE / PIT require
between-subject variability (mixed-effects item A) and are deliberately left
unset — these tests assert both the positive population (npe/auc-cmax) and the
consensus gate (vpc/npde/pit stay None).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apmode.backends.node_runner import NodeBackendRunner
from apmode.backends.node_trainer import TrainingConfig
from apmode.bundle.models import (
    ColumnMapping,
    DataManifest,
    NCASubjectDiagnostic,
)
from apmode.dsl.ast_models import (
    IIV,
    DSLSpec,
    FirstOrder,
    NODEElimination,
    OneCmt,
    Proportional,
)
from apmode.governance.policy import Gate3Config

_N_MOCK_SUBJECTS = 10


def _manifest() -> DataManifest:
    return DataManifest(
        data_sha256="b" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="ID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        ),
        n_subjects=_N_MOCK_SUBJECTS,
        n_observations=80,
        n_doses=_N_MOCK_SUBJECTS,
    )


def _node_spec() -> DSLSpec:
    return DSLSpec(
        model_id="test_node_ppc",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=NODEElimination(dim=3, constraint_template="bounded_positive"),
        variability=[IIV(params=["ka", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
    )


def _eligible_nca_diagnostics() -> list[NCASubjectDiagnostic]:
    # Mock subjects are keyed ``MOCK0``..``MOCK{n-1}`` by _make_mock_subjects.
    return [
        NCASubjectDiagnostic(subject_id=f"MOCK{i}", excluded=False)
        for i in range(_N_MOCK_SUBJECTS)
    ]


def _fast_runner(tmp_path: Path) -> NodeBackendRunner:
    return NodeBackendRunner(
        work_dir=tmp_path,
        training_config=TrainingConfig(epochs=3, early_stop_patience=5),
        distill=False,
    )


def _small_policy() -> Gate3Config:
    return Gate3Config(n_posterior_predictive_sims=100)


@pytest.mark.asyncio
async def test_gate3_policy_populates_npe_and_auc_cmax(tmp_path: Path) -> None:
    runner = _fast_runner(tmp_path)
    result = await runner.run(
        spec=_node_spec(),
        data_manifest=_manifest(),
        initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
        seed=7,
        gate3_policy=_small_policy(),
        nca_diagnostics=_eligible_nca_diagnostics(),
    )

    diag = result.diagnostics
    assert diag.npe_score is not None
    assert isinstance(diag.npe_score, float)
    import math

    assert math.isfinite(diag.npe_score)
    assert diag.npe_score >= 0.0
    # 10 eligible subjects clears the default min_eligible floor (8).
    assert diag.auc_cmax_be_score is not None
    assert 0.0 <= diag.auc_cmax_be_score <= 1.0
    assert diag.auc_cmax_source == "observed_trapezoid"


@pytest.mark.asyncio
async def test_vpc_npde_pit_stay_none_consensus_gate(tmp_path: Path) -> None:
    """Consensus Decision 3: no pooled VPC/NPDE/PIT without BSV (item A)."""
    runner = _fast_runner(tmp_path)
    result = await runner.run(
        spec=_node_spec(),
        data_manifest=_manifest(),
        initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
        seed=7,
        gate3_policy=_small_policy(),
        nca_diagnostics=_eligible_nca_diagnostics(),
    )

    diag = result.diagnostics
    # Population diagnostics that require between-subject variability must
    # remain unset even though NPE/AUC-Cmax were computed.
    assert diag.vpc is None
    assert diag.npde is None
    assert diag.pit_calibration is None


@pytest.mark.asyncio
async def test_no_gate3_policy_leaves_predictive_fields_none(tmp_path: Path) -> None:
    runner = _fast_runner(tmp_path)
    result = await runner.run(
        spec=_node_spec(),
        data_manifest=_manifest(),
        initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
        seed=7,
        gate3_policy=None,
    )

    diag = result.diagnostics
    assert diag.npe_score is None
    assert diag.auc_cmax_be_score is None
    assert diag.auc_cmax_source is None
    assert diag.vpc is None
    assert diag.npde is None
    assert diag.pit_calibration is None


@pytest.mark.asyncio
async def test_pooled_cwres_not_placeholder(tmp_path: Path) -> None:
    """Real pooled/population residual moments replace the 0.0/1.0 stub."""
    runner = _fast_runner(tmp_path)
    result = await runner.run(
        spec=_node_spec(),
        data_manifest=_manifest(),
        initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
        seed=7,
        gate3_policy=_small_policy(),
        nca_diagnostics=_eligible_nca_diagnostics(),
    )

    gof = result.diagnostics.gof
    assert gof.cwres_mean is not None
    assert gof.cwres_sd is not None
    # The hard-coded placeholder was exactly (0.0, 1.0). Real pooled
    # standardized residuals will not land on that exact pair.
    assert (gof.cwres_mean, gof.cwres_sd) != (0.0, 1.0)


@pytest.mark.asyncio
async def test_sim_path_failure_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in the simulation/scoring path must not crash the run.

    Mirrors nlmixr2's contract: npe_score stays None and Gate 3 falls back
    to the CWRES NPE proxy.
    """

    def _boom(*_args: object, **_kwargs: object) -> object:
        msg = "forced posterior-predictive failure"
        raise RuntimeError(msg)

    monkeypatch.setattr("apmode.backends.predictive_summary.build_predictive_diagnostics", _boom)

    runner = _fast_runner(tmp_path)
    result = await runner.run(
        spec=_node_spec(),
        data_manifest=_manifest(),
        initial_estimates={"ka": 1.0, "V": 30.0, "CL": 2.0},
        seed=7,
        gate3_policy=_small_policy(),
        nca_diagnostics=_eligible_nca_diagnostics(),
    )

    diag = result.diagnostics
    assert diag.npe_score is None
    assert diag.auc_cmax_be_score is None
    # CWRES still ships — it does not depend on the posterior-predictive path.
    assert diag.gof.cwres_mean is not None
