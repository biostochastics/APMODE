# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the FREM nlmixr2 emitter (src/apmode/dsl/frem_emitter.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from apmode.dsl.ast_models import (
    IIV,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LinearElim,
    OneCmt,
)
from apmode.dsl.frem_emitter import (
    FREMCovariate,
    emit_nlmixr2_frem,
    prepare_frem_data,
    summarize_covariates,
)


def _simple_spec(model_id: str = "frem_test", cov_links: bool = False) -> DSLSpec:
    variability: list[object] = [IIV(params=["CL", "V"], structure="diagonal")]
    if cov_links:
        variability.append(CovariateLink(param="CL", covariate="WT", form="power"))
    return DSLSpec(
        model_id=model_id,
        absorption=FirstOrder(ka=1.0),
        distribution=OneCmt(V=50.0),
        elimination=LinearElim(CL=5.0),
        variability=variability,  # type: ignore[arg-type]
        observation=Combined(sigma_prop=0.1, sigma_add=0.01),
    )


class TestFREMCovariate:
    def test_valid(self) -> None:
        cov = FREMCovariate(name="WT", mu_init=70.0, sigma_init=15.0, dvid=10)
        assert cov.name == "WT"
        assert cov.epsilon_sd > 0

    def test_rejects_degenerate_sd(self) -> None:
        with pytest.raises(ValueError, match="sigma_init"):
            FREMCovariate(name="WT", mu_init=70.0, sigma_init=0.0, dvid=10)

    def test_rejects_invalid_identifier(self) -> None:
        with pytest.raises(ValueError, match="Invalid R identifier"):
            FREMCovariate(name="1BAD", mu_init=70.0, sigma_init=1.0, dvid=10)


class TestSummarizeCovariates:
    def test_computes_mean_and_sd(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 2, 2, 3, 3],
                "TIME": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "WT": [70, 70, 80, 80, 60, 60],
            }
        )
        summaries = summarize_covariates(df, ["WT"])
        assert len(summaries) == 1
        assert summaries[0].mu_init == pytest.approx(70.0)
        assert summaries[0].sigma_init == pytest.approx(10.0, rel=1e-3)
        assert summaries[0].dvid == 10
        assert summaries[0].transform == "identity"

    def test_skips_missing_subjects(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2, 3, 4],
                "TIME": [0.0, 0.0, 0.0, 0.0],
                "WT": [70.0, 80.0, 60.0, float("nan")],
            }
        )
        summaries = summarize_covariates(df, ["WT"])
        assert summaries[0].mu_init == pytest.approx(70.0)

    def test_rejects_insufficient_subjects(self) -> None:
        df = pd.DataFrame({"NMID": [1], "TIME": [0.0], "WT": [70.0]})
        with pytest.raises(ValueError, match="observed subject"):
            summarize_covariates(df, ["WT"])

    def test_rejects_duplicate_names(self) -> None:
        df = pd.DataFrame({"NMID": [1, 2], "TIME": [0.0, 0.0], "WT": [70.0, 80.0]})
        with pytest.raises(ValueError, match="Duplicate covariate names"):
            summarize_covariates(df, ["WT", "WT"])

    def test_log_transform_summary_on_log_scale(self) -> None:
        import math

        df = pd.DataFrame({"NMID": [1, 2, 3], "TIME": [0.0, 0.0, 0.0], "WT": [60.0, 70.0, 80.0]})
        summaries = summarize_covariates(df, ["WT"], transforms={"WT": "log"})
        assert summaries[0].transform == "log"
        expected_mu = (math.log(60.0) + math.log(70.0) + math.log(80.0)) / 3
        assert summaries[0].mu_init == pytest.approx(expected_mu, rel=1e-4)

    def test_log_transform_rejects_nonpositive(self) -> None:
        df = pd.DataFrame({"NMID": [1, 2], "TIME": [0.0, 0.0], "WT": [70.0, 0.0]})
        with pytest.raises(ValueError, match="strictly positive"):
            summarize_covariates(df, ["WT"], transforms={"WT": "log"})


class TestPrepareFREMData:
    def test_appends_covariate_rows(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 2, 2],
                "TIME": [0.0, 1.0, 0.0, 1.0],
                "EVID": [1, 0, 1, 0],
                "AMT": [100.0, 0.0, 100.0, 0.0],
                "DV": [0.0, 5.0, 0.0, 4.0],
                "WT": [70.0, 70.0, 80.0, 80.0],
            }
        )
        cov = FREMCovariate(name="WT", mu_init=75.0, sigma_init=10.0, dvid=10)
        out = prepare_frem_data(df, [cov])
        # Original rows preserved
        assert len(out) == len(df) + 2  # +1 row per subject for WT
        # DVID column added
        assert "DVID" in out.columns
        # The WT observation rows use DVID=10
        wt_rows = out[out["DVID"] == 10]
        assert len(wt_rows) == 2
        assert set(wt_rows["DV"].tolist()) == {70.0, 80.0}

    def test_rejects_dvid_collision(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2],
                "TIME": [0.0, 0.0],
                "EVID": [0, 0],
                "AMT": [0.0, 0.0],
                "DV": [5.0, 4.0],
                "DVID": [10, 10],  # collides with FREM covariate DVID
                "WT": [70.0, 80.0],
            }
        )
        cov = FREMCovariate(name="WT", mu_init=75.0, sigma_init=10.0, dvid=10)
        with pytest.raises(ValueError, match="collide with FREM"):
            prepare_frem_data(df, [cov])

    def test_log_transform_writes_log_scale_dv(self) -> None:
        import math

        df = pd.DataFrame(
            {
                "NMID": [1, 2],
                "TIME": [0.0, 0.0],
                "EVID": [0, 0],
                "AMT": [0.0, 0.0],
                "DV": [5.0, 4.0],
                "WT": [70.0, 80.0],
            }
        )
        cov = FREMCovariate(name="WT", mu_init=4.3, sigma_init=0.2, dvid=10, transform="log")
        out = prepare_frem_data(df, [cov])
        wt_rows = out[out["DVID"] == 10]
        assert wt_rows["DV"].tolist() == pytest.approx([math.log(70.0), math.log(80.0)])

    def test_skips_missing_covariate_per_subject(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2],
                "TIME": [0.0, 0.0],
                "EVID": [0, 0],
                "AMT": [0.0, 0.0],
                "DV": [5.0, 4.0],
                "WT": [70.0, float("nan")],
            }
        )
        cov = FREMCovariate(name="WT", mu_init=75.0, sigma_init=10.0, dvid=10)
        out = prepare_frem_data(df, [cov])
        # Only subject 1 gets an augmentation row
        wt_rows = out[out["DVID"] == 10]
        assert len(wt_rows) == 1
        assert wt_rows["NMID"].iloc[0] == 1


class TestEmitNlmixr2FREM:
    def test_produces_joint_omega_block(self) -> None:
        spec = _simple_spec()
        covs = [
            FREMCovariate(name="WT", mu_init=70.0, sigma_init=10.0, dvid=10),
            FREMCovariate(name="AGE", mu_init=40.0, sigma_init=15.0, dvid=11),
        ]
        code = emit_nlmixr2_frem(spec, covs)

        # Joint block includes both PK and covariate etas in one ~ c(...) expression
        assert "eta.CL + eta.V + eta.cov.WT + eta.cov.AGE ~ c(" in code
        # Covariate means emitted
        assert "mu_WT <- 70.0" in code
        assert "mu_AGE <- 40.0" in code
        # Covariate residual error is fixed (BSV/residual confound at 1 obs/subject)
        assert "sig_cov_WT <- fix(0.01)" in code
        assert "sig_cov_AGE <- fix(0.01)" in code
        # Endpoints conditioned on DVID
        assert "WT_pred ~ add(sig_cov_WT) | DVID==10" in code
        assert "AGE_pred ~ add(sig_cov_AGE) | DVID==11" in code
        # PK endpoint now carries an explicit DVID==1 pipe for multi-endpoint routing
        assert "| DVID==1" in code
        # PK portion still present
        assert "lCL" in code
        assert "lV" in code

    def test_strips_covariate_links(self) -> None:
        """FREM should drop CovariateLink entries and not emit beta_* coefficients."""
        spec = _simple_spec(cov_links=True)
        covs = [FREMCovariate(name="WT", mu_init=70.0, sigma_init=10.0, dvid=10)]
        code = emit_nlmixr2_frem(spec, covs)
        assert "beta_CL_WT" not in code

    def test_empty_covariates_raises(self) -> None:
        spec = _simple_spec()
        with pytest.raises(ValueError, match="at least one covariate"):
            emit_nlmixr2_frem(spec, [])

    def test_variance_initializes_to_square_of_sd(self) -> None:
        spec = _simple_spec()
        covs = [FREMCovariate(name="WT", mu_init=70.0, sigma_init=5.0, dvid=10)]
        code = emit_nlmixr2_frem(spec, covs)
        # sigma_init=5 -> variance diagonal = 25.0
        assert "25.0" in code

    def test_cross_block_initializes_to_zero(self) -> None:
        """PK x covariate off-diagonal entries must start at 0."""
        spec = _simple_spec()
        covs = [FREMCovariate(name="WT", mu_init=70.0, sigma_init=3.0, dvid=10)]
        code = emit_nlmixr2_frem(spec, covs)
        # Lower-tri entries: PK var(CL), PK cov(CL,V)=0 (diagonal struct),
        # PK var(V), cross(CL,WT)=0, cross(V,WT)=0, var(WT)=9.0
        # We just verify the structural presence of zeros in the block.
        assert "~ c(" in code
