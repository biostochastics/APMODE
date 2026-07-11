# SPDX-License-Identifier: GPL-2.0-or-later
"""Cmdstanpy platform-adaptive defaults (cmdstanpy issues #780, #895)."""

from __future__ import annotations

from unittest.mock import patch

from apmode.bayes.platform import cmdstan_run_kwargs


def test_windows_forces_one_process_per_chain() -> None:
    with patch("platform.system", return_value="Windows"):
        kw = cmdstan_run_kwargs(uses_reduce_sum=False)
    assert kw["force_one_process_per_chain"] is True
    assert "cpp_options" not in kw or "STAN_THREADS" not in kw.get("cpp_options", {})


def test_windows_never_enables_stan_threads_even_with_reduce_sum() -> None:
    with patch("platform.system", return_value="Windows"):
        kw = cmdstan_run_kwargs(uses_reduce_sum=True)
    assert kw["force_one_process_per_chain"] is True
    assert "cpp_options" not in kw or "STAN_THREADS" not in kw.get("cpp_options", {})


def test_linux_multichain_no_reduce_sum() -> None:
    with patch("platform.system", return_value="Linux"):
        kw = cmdstan_run_kwargs(uses_reduce_sum=False)
    assert kw["force_one_process_per_chain"] is False
    assert "cpp_options" not in kw


def test_linux_enables_stan_threads_with_reduce_sum() -> None:
    with patch("platform.system", return_value="Linux"):
        kw = cmdstan_run_kwargs(uses_reduce_sum=True)
    assert kw["force_one_process_per_chain"] is False
    assert kw["cpp_options"]["STAN_THREADS"] is True


def test_macos_behaves_like_linux_with_reduce_sum() -> None:
    with patch("platform.system", return_value="Darwin"):
        kw = cmdstan_run_kwargs(uses_reduce_sum=True)
    assert kw["force_one_process_per_chain"] is False
    assert kw["cpp_options"]["STAN_THREADS"] is True


def test_macos_behaves_like_linux_no_reduce_sum() -> None:
    with patch("platform.system", return_value="Darwin"):
        kw = cmdstan_run_kwargs(uses_reduce_sum=False)
    assert kw["force_one_process_per_chain"] is False
    assert "cpp_options" not in kw
