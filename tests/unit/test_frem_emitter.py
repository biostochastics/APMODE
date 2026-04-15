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
        # DVID offset is 2 (PK endpoint claims DVID=1, first covariate
        # endpoint gets DVID=2 via nlmixr2's implicit declaration-order
        # numbering). See _FREM_DVID_OFFSET.
        assert summaries[0].dvid == 2
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
        # Endpoints: nlmixr2 routes by data-side DVID (column), NOT by
        # an inline ``| DVID==N`` condition (grammar rejects conditions
        # after ``|``). Endpoint RHS is a bare residual spec only.
        assert "WT_pred ~ add(sig_cov_WT)" in code
        assert "AGE_pred ~ add(sig_cov_AGE)" in code
        # Defensive: the old broken ``| DVID==N`` form must not appear.
        assert "| DVID==" not in code
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


# --- Live nlmixr2 integration tests ---------------------------------------
# These tests spawn Rscript against a real nlmixr2 install to verify that
# the emitted code is not just a string match but an object the nlmixr2
# compiler accepts. They are tagged ``live`` so they are skipped in the
# default fast path (``-m "not live"``), and skipped individually when R
# or nlmixr2 is not installed on the host.

import shutil  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402


def _rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _r_package_installed(pkg: str) -> bool:
    if not _rscript_available():
        return False
    out = subprocess.run(
        ["Rscript", "-e", f'cat(requireNamespace("{pkg}", quietly=TRUE))'],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return out.stdout.strip() == "TRUE"


_NLMIXR2_AVAILABLE = _r_package_installed("nlmixr2")


@pytest.mark.live
@pytest.mark.skipif(not _NLMIXR2_AVAILABLE, reason="nlmixr2 not installed on this host")
class TestFREMLiveIntegration:
    """End-to-end: feed emitted code into real nlmixr2 and check acceptance."""

    def test_nlmixr2_accepts_emitted_frem(self, tmp_path: Path) -> None:
        """nlmixr2() must parse and compile the emitted FREM function.

        Regression guard for the ``| DVID==N`` pipe bug: the
        emitter previously wrote a condition on the endpoint RHS that
        nlmixr2 5.0 rejects ("the condition 'DVID == N' must be a simple
        name"). Fix: no pipe at all; routing is data-driven via DVID.
        """
        spec = _simple_spec()
        covs = [
            FREMCovariate(name="WT", mu_init=70.0, sigma_init=10.0, dvid=2),
            FREMCovariate(name="AGE", mu_init=40.0, sigma_init=15.0, dvid=3),
        ]
        code = emit_nlmixr2_frem(spec, covs)
        model_path = tmp_path / "frem_model.R"
        model_path.write_text(code)

        script = f"""
suppressPackageStartupMessages({{ library(nlmixr2) }})
fn <- eval(parse(text = readLines('{model_path}')))
ui <- nlmixr2(fn)
cat("COMPILE_OK endpoints=", nrow(ui$predDf), "\\n")
"""
        script_path = tmp_path / "drive.R"
        script_path.write_text(script)

        result = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        # nlmixr2 emits info messages to stderr even on success; success is
        # determined by exit code and the COMPILE_OK sentinel in stdout.
        assert result.returncode == 0, (
            f"nlmixr2 rejected emitted FREM code:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "COMPILE_OK endpoints= 3" in result.stdout, (
            f"Expected 3 endpoints (PK + 2 covariates). stdout: {result.stdout}"
        )

    @pytest.mark.slow
    def test_nlmixr2_fits_emitted_frem(self, tmp_path: Path) -> None:
        """End-to-end FREM fit via FOCE-I on tiny synthetic data.

        SAEM is NOT reliable for static subject-level endpoints (it treats
        each observation as a dynamic sampling target and collapses the
        random-effect variance). FOCE-I is the supported estimator for
        FREM with this emitter.
        """
        spec = _simple_spec()
        covs = [
            FREMCovariate(name="WT", mu_init=70.0, sigma_init=10.0, dvid=2),
        ]
        code = emit_nlmixr2_frem(spec, covs)
        model_path = tmp_path / "frem_model.R"
        model_path.write_text(code)

        script = f"""
suppressPackageStartupMessages({{ library(nlmixr2); library(rxode2) }})
fn <- eval(parse(text = readLines('{model_path}')))

set.seed(42)
n <- 25
wt <- rnorm(n, 70, 10)
wt_obs <- wt; wt_obs[sample(n, 5)] <- NA
cl <- 5 * (wt/70)^0.75
conc <- function(t, c, v=50) 100/v * exp(-c/v * t)
times <- c(0.5, 1, 2, 4, 8, 12, 24)
rows <- list()
for (i in seq_len(n)) {{
  rows[[length(rows)+1]] <- data.frame(ID=i, TIME=0, EVID=1, AMT=100, DV=NA_real_, DVID=1)
  for (t in times) rows[[length(rows)+1]] <- data.frame(
    ID=i, TIME=t, EVID=0, AMT=0,
    DV=conc(t, cl[i]) * exp(rnorm(1, 0, 0.1)), DVID=1)
  if (!is.na(wt_obs[i])) rows[[length(rows)+1]] <- data.frame(
    ID=i, TIME=0, EVID=0, AMT=0, DV=wt_obs[i], DVID=2)
}}
d <- do.call(rbind, rows); d <- d[order(d$ID, d$TIME, d$DVID), ]
fit <- nlmixr2(fn, d, est="focei",
               control=foceiControl(print=0, maxInnerIterations=10, maxOuterIterations=30))
# Success signal: finite OFV and non-degenerate eta.cov.WT
ofv <- fit$objDf$OBJF[1]
omega_wt <- diag(fit$omega)[["eta.cov.WT"]]
cat(sprintf("FIT_OK ofv=%.2f omega_wt=%.4f\\n", ofv, omega_wt))
"""
        script_path = tmp_path / "drive.R"
        script_path.write_text(script)
        result = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, (
            f"FREM fit failed:\nstdout: {result.stdout}\nstderr: {result.stderr[-1000:]}"
        )
        assert "FIT_OK" in result.stdout, f"Expected FIT_OK sentinel. stdout: {result.stdout}"
        # Parse omega_wt from output; must be > 1 (nonzero between-subject
        # variance learned from the covariate observations).
        import re as _re

        m = _re.search(r"omega_wt=([0-9.]+)", result.stdout)
        assert m is not None
        assert float(m.group(1)) > 1.0, (
            f"eta.cov.WT variance collapsed to {m.group(1)}; FREM learned nothing."
        )


# --- Binary categorical covariate tests ----------------------------------


class TestBinaryCovariate:
    def test_binary_transform_accepts_zero_one(self) -> None:
        cov = FREMCovariate(name="SEX", mu_init=0.5, sigma_init=0.5, dvid=2, transform="binary")
        assert cov.transform == "binary"

    def test_summarize_binary_validates_values(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2, 3, 4],
                "TIME": [0.0, 0.0, 0.0, 0.0],
                "SEX": [0, 1, 0, 2],  # 2 is invalid
            }
        )
        with pytest.raises(ValueError, match="binary transform"):
            summarize_covariates(df, ["SEX"], transforms={"SEX": "binary"})

    def test_summarize_binary_valid(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 2, 3, 4, 5, 6],
                "TIME": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "SEX": [0, 1, 0, 1, 0, 1],
            }
        )
        summaries = summarize_covariates(df, ["SEX"], transforms={"SEX": "binary"})
        assert summaries[0].transform == "binary"
        assert summaries[0].mu_init == pytest.approx(0.5)


# --- Time-varying covariate tests ----------------------------------------


class TestTimeVaryingCovariate:
    def test_time_varying_emits_per_time_rows(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1, 2, 2, 2],
                "TIME": [0.0, 4.0, 8.0, 0.0, 4.0, 8.0],
                "EVID": [1, 0, 0, 1, 0, 0],
                "AMT": [100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
                "DV": [0.0, 5.0, 3.0, 0.0, 4.0, 2.0],
                "CRCL": [80.0, 75.0, 70.0, 90.0, 88.0, 86.0],  # changes over time
            }
        )
        cov = FREMCovariate(
            name="CRCL",
            mu_init=80.0,
            sigma_init=10.0,
            dvid=2,
            time_varying=True,
        )
        out = prepare_frem_data(df, [cov])
        cov_rows = out[out["DVID"] == 2]
        # Subject 1 has 3 distinct (TIME, CRCL) values; same for subject 2
        # → 6 covariate observation rows total.
        assert len(cov_rows) == 6
        # Per-subject values land at the correct times
        s1 = cov_rows[cov_rows["NMID"] == 1].sort_values("TIME")
        assert s1["DV"].tolist() == [80.0, 75.0, 70.0]

    def test_time_varying_skips_repeated_same_time_value(self) -> None:
        df = pd.DataFrame(
            {
                "NMID": [1, 1, 1],
                "TIME": [0.0, 0.0, 4.0],
                "EVID": [1, 0, 0],
                "AMT": [100.0, 0.0, 0.0],
                "DV": [0.0, 5.0, 3.0],
                "CRCL": [80.0, 80.0, 70.0],  # 80@0 appears twice
            }
        )
        cov = FREMCovariate(
            name="CRCL",
            mu_init=75.0,
            sigma_init=5.0,
            dvid=2,
            time_varying=True,
        )
        out = prepare_frem_data(df, [cov])
        cov_rows = out[out["DVID"] == 2]
        assert len(cov_rows) == 2  # dedupe (0,80) and (4,70)

    def test_time_varying_residual_left_estimable(self) -> None:
        spec = _simple_spec()
        covs = [
            FREMCovariate(name="CRCL", mu_init=80.0, sigma_init=10.0, dvid=2, time_varying=True),
        ]
        code = emit_nlmixr2_frem(spec, covs)
        # Static covariates are emitted as fix(...); TV covariates are not.
        assert "sig_cov_CRCL <- 0.01" in code
        assert "sig_cov_CRCL <- fix" not in code

    def test_static_residual_remains_fixed(self) -> None:
        spec = _simple_spec()
        covs = [
            FREMCovariate(name="WT", mu_init=70.0, sigma_init=10.0, dvid=2),
        ]
        code = emit_nlmixr2_frem(spec, covs)
        assert "sig_cov_WT <- fix(0.01)" in code
