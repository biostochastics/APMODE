# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for run_frem_fit orchestration.

Verifies the composition of ``summarize_covariates`` +
``prepare_frem_data`` + ``emit_nlmixr2_frem`` + ``Nlmixr2Runner`` using
a stub runner that records the arguments it received. Live-R coverage
for the full fit path lives in
``tests/unit/test_frem_emitter.py::TestFREMLiveIntegration`` (marked
``live`` + ``slow``) — that suite proves the emitted code and data
routing actually compile and converge under nlmixr2. This file covers
the Python plumbing in isolation so it can run on every PR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apmode.backends.frem_runner import run_frem_fit
from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ColumnMapping,
    ConvergenceMetadata,
    DataManifest,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
)
from apmode.dsl.ast_models import (
    IIV,
    Combined,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
)


@dataclass
class _StubRunner:
    """Records the last ``run`` invocation for assertion."""

    last_spec: DSLSpec | None = None
    last_data_path: Path | None = None
    last_code: str | None = None
    last_seed: int | None = None
    received: list[dict[str, object]] = field(default_factory=list)

    async def run(
        self,
        *,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,
        data_path: Path | None = None,
        split_manifest: dict[str, object] | None = None,
        compiled_code_override: str | None = None,
    ) -> BackendResult:
        del data_manifest, split_manifest, timeout_seconds
        self.last_spec = spec
        self.last_data_path = data_path
        self.last_code = compiled_code_override
        self.last_seed = seed
        self.received.append(
            {
                "initial_estimates": dict(initial_estimates),
                "data_path": str(data_path),
            }
        )
        # Return a minimally-valid BackendResult so downstream code keeps moving.
        return BackendResult(
            model_id=spec.model_id,
            backend="nlmixr2",
            converged=True,
            ofv=0.0,
            aic=10.0,
            bic=15.0,
            parameter_estimates={},
            eta_shrinkage={},
            convergence_metadata=ConvergenceMetadata(
                method="focei",
                converged=True,
                iterations=1,
                minimization_status="successful",
                wall_time_seconds=0.1,
            ),
            diagnostics=DiagnosticBundle(
                gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.0),
                identifiability=IdentifiabilityFlags(
                    condition_number=10.0,
                    profile_likelihood_ci={},
                    ill_conditioned=False,
                ),
                blq=BLQHandling(method="none", lloq=None, n_blq=0, blq_fraction=0.0),
            ),
            wall_time_seconds=0.1,
            backend_versions={},
            initial_estimate_source="fallback",
        )


def _spec() -> DSLSpec:
    return DSLSpec(
        model_id="frem_compose_test",
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=50.0),
        elimination=LinearElim(CL=5.0),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Combined(sigma_prop=0.1, sigma_add=0.01),
    )


def _df(n_subj: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    wt = rng.normal(70, 10, size=n_subj)
    rows: list[dict[str, object]] = []
    for i in range(n_subj):
        rows.append(
            {
                "NMID": i + 1,
                "TIME": 0.0,
                "EVID": 1,
                "AMT": 100.0,
                "DV": 0.0,
                "WT": float(wt[i]),
            }
        )
        for t in (1.0, 4.0, 8.0):
            rows.append(
                {
                    "NMID": i + 1,
                    "TIME": float(t),
                    "EVID": 0,
                    "AMT": 0.0,
                    "DV": 2.0,
                    "WT": float(wt[i]),
                }
            )
    df = pd.DataFrame(rows)
    df.loc[df["NMID"].isin([3, 5]), "WT"] = np.nan  # induce missingness
    return df


def _manifest() -> DataManifest:
    return DataManifest(
        data_sha256="0" * 64,
        ingestion_format="nonmem_csv",
        column_mapping=ColumnMapping(
            subject_id="NMID",
            time="TIME",
            dv="DV",
            evid="EVID",
            amt="AMT",
        ),
        n_subjects=10,
        n_observations=30,
        n_doses=10,
    )


@pytest.mark.asyncio
async def test_run_frem_fit_composes_augmented_data_and_code(tmp_path: Path) -> None:
    df = _df()
    source = tmp_path / "src.csv"
    df.to_csv(source, index=False)
    runner = _StubRunner()

    result = await run_frem_fit(
        spec_template=_spec(),
        df=df,
        data_path=source.resolve(),
        data_manifest=_manifest(),
        covariate_names=["WT"],
        runner=runner,  # type: ignore[arg-type]
        work_dir=tmp_path,
        seed=42,
    )

    # Runner received the FREM-emitted code (not the classical emitter
    # output). Canonical fingerprints of a FREM model.
    assert runner.last_code is not None
    assert "FREM" in runner.last_code  # header comment
    assert "eta.cov.WT" in runner.last_code  # joint Omega entry
    assert "mu_WT" in runner.last_code
    # Augmented CSV written and passed to the runner.
    assert runner.last_data_path is not None
    assert runner.last_data_path.name == "frem_augmented.csv"
    assert runner.last_data_path.exists()
    aug = pd.read_csv(runner.last_data_path)
    assert "DVID" in aug.columns
    # Two subjects had missing WT → 8 augmentation rows (one per observed
    # subject). Original rows preserved.
    wt_rows = aug[aug["DVID"] == 2]
    assert len(wt_rows) == 8
    assert result.converged


@pytest.mark.asyncio
async def test_run_frem_fit_honors_log_transform(tmp_path: Path) -> None:
    df = _df()
    source = tmp_path / "src.csv"
    df.to_csv(source, index=False)
    runner = _StubRunner()

    await run_frem_fit(
        spec_template=_spec(),
        df=df,
        data_path=source.resolve(),
        data_manifest=_manifest(),
        covariate_names=["WT"],
        runner=runner,  # type: ignore[arg-type]
        work_dir=tmp_path,
        seed=42,
        transforms={"WT": "log"},
    )

    aug = pd.read_csv(runner.last_data_path)  # type: ignore[arg-type]
    wt_rows = aug[aug["DVID"] == 2]
    # Log-transformed WT values land in roughly log(50..100) = 3.9..4.6
    assert (wt_rows["DV"].between(3.5, 5.0)).all()
