# SPDX-License-Identifier: GPL-2.0-or-later
"""MCMC diagnostics + sampler config emitters (plan Task 12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bundle.emitter import BundleEmitter
from apmode.bundle.models import PosteriorDiagnostics, SamplerConfig


def _diag() -> PosteriorDiagnostics:
    return PosteriorDiagnostics(
        rhat_max=1.01,
        ess_bulk_min=4200.0,
        ess_tail_min=3900.0,
        n_divergent=0,
        n_max_treedepth=0,
        ebfmi_min=0.9,
        pareto_k_max=0.35,
        pareto_k_counts={"good": 120, "ok": 5, "bad": 0, "very_bad": 0},
        mcse_by_param={"log_CL": 0.002, "log_V": 0.003},
        per_chain_rhat={"log_CL": [1.00, 1.01, 1.00, 1.01]},
    )


def _sampler_cfg() -> SamplerConfig:
    return SamplerConfig(
        chains=4,
        warmup=1000,
        sampling=1000,
        adapt_delta=0.85,
        max_treedepth=10,
        seed=42,
    )


def test_write_mcmc_diagnostics_round_trip(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    diag = _diag()
    path = emitter.write_mcmc_diagnostics(diag, candidate_id="cand_001")
    assert path.name == "cand_001_mcmc_diagnostics.json"
    loaded = json.loads(path.read_text())
    assert loaded["rhat_max"] == pytest.approx(1.01)
    assert loaded["n_divergent"] == 0


def test_write_sampler_config_round_trip(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    cfg = _sampler_cfg()
    path = emitter.write_sampler_config(cfg, candidate_id="cand_001")
    assert path.name == "cand_001_sampler_config.json"
    loaded = json.loads(path.read_text())
    assert loaded["chains"] == 4
    assert loaded["warmup"] == 1000
    assert loaded["seed"] == 42


def test_emitters_reject_invalid_candidate_id(tmp_path: Path) -> None:
    emitter = BundleEmitter(tmp_path)
    emitter.initialize()
    with pytest.raises(ValueError, match="candidate_id"):
        emitter.write_mcmc_diagnostics(_diag(), candidate_id="../escape")
    with pytest.raises(ValueError, match="candidate_id"):
        emitter.write_sampler_config(_sampler_cfg(), candidate_id="../escape")
