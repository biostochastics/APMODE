# SPDX-License-Identifier: GPL-2.0-or-later
"""Provenance wiring for ``apmode.bayes.harness.sample_with_provenance``.

Covers cmdstanpy issues #848 (``save_cmdstan_config``) and #895 (Windows
thread_local) by ensuring:

* ``sample_with_provenance`` passes ``save_cmdstan_config=True`` and the
  platform-adaptive ``force_one_process_per_chain`` kwarg to
  ``CmdStanModel.sample``.
* ``backend_versions.json`` is written in the working directory with SHA-256
  hashes of the Stan code + data, the cmdstan version, host platform, and
  the ``one_process_per_chain`` flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from apmode.bayes.harness import sample_with_provenance


def _fake_model_factory() -> MagicMock:
    fake_model = MagicMock()
    fake_fit = MagicMock()
    fake_model.sample.return_value = fake_fit
    return fake_model


def test_sample_persists_stan_code_hash_and_version(tmp_path: Path) -> None:
    stan_code = "data { int N; } parameters { real mu; } model { mu ~ normal(0,1); }"
    data_json = {"N": 10}
    with (
        patch("platform.system", return_value="Windows"),
        patch("apmode.bayes.harness.CmdStanModel", return_value=_fake_model_factory()),
    ):
        sample_with_provenance(
            stan_code=stan_code,
            data=data_json,
            work_dir=tmp_path,
            seed=42,
            chains=2,
            warmup=200,
            sampling=200,
            adapt_delta=0.85,
            max_treedepth=10,
            uses_reduce_sum=False,
        )
    meta = json.loads((tmp_path / "backend_versions.json").read_text())
    assert "stan_code_sha256" in meta
    assert len(meta["stan_code_sha256"]) == 64  # SHA-256 hex digest
    assert "data_sha256" in meta
    assert "cmdstan_version" in meta
    assert "platform" in meta
    assert meta["one_process_per_chain"] is True
    assert meta["uses_reduce_sum"] is False


def test_sample_passes_save_cmdstan_config_and_platform_kwargs(tmp_path: Path) -> None:
    stan_code = "data { int N; } parameters { real mu; } model { mu ~ normal(0,1); }"
    fake_model = _fake_model_factory()
    with (
        patch("platform.system", return_value="Linux"),
        patch("apmode.bayes.harness.CmdStanModel", return_value=fake_model),
    ):
        sample_with_provenance(
            stan_code=stan_code,
            data={"N": 3},
            work_dir=tmp_path,
            seed=7,
            chains=2,
            warmup=100,
            sampling=100,
            adapt_delta=0.9,
            max_treedepth=10,
            uses_reduce_sum=True,
        )
    kwargs = fake_model.sample.call_args.kwargs
    assert kwargs["save_cmdstan_config"] is True
    assert kwargs["force_one_process_per_chain"] is False
    assert kwargs["cpp_options"]["STAN_THREADS"] is True
    assert kwargs["seed"] == 7


def test_hashes_are_deterministic_across_invocations(tmp_path: Path) -> None:
    stan_code = "data { int N; }"
    data_json = {"N": 5}
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    for work in (dir_a, dir_b):
        with patch("apmode.bayes.harness.CmdStanModel", return_value=_fake_model_factory()):
            sample_with_provenance(
                stan_code=stan_code,
                data=data_json,
                work_dir=work,
                seed=1,
                chains=1,
                warmup=10,
                sampling=10,
                adapt_delta=0.8,
                max_treedepth=8,
                uses_reduce_sum=False,
            )
    meta_a = json.loads((dir_a / "backend_versions.json").read_text())
    meta_b = json.loads((dir_b / "backend_versions.json").read_text())
    assert meta_a["stan_code_sha256"] == meta_b["stan_code_sha256"]
    assert meta_a["data_sha256"] == meta_b["data_sha256"]
