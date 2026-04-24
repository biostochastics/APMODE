# SPDX-License-Identifier: GPL-2.0-or-later
"""ParameterEstimate posterior summary fields (plan Task 13).

The existing ParameterEstimate exposes posterior_sd and the 5/50/95 credible
interval quantiles. ``posterior_mean`` is added as a dedicated field for
reports that need to distinguish posterior mean from ``estimate`` (which is
the primary point estimate — posterior mean for Bayesian, MLE for classical).
"""

from __future__ import annotations

import json

from apmode.bundle.models import ParameterEstimate


def test_parameter_estimate_accepts_all_posterior_fields() -> None:
    pe = ParameterEstimate(
        name="CL",
        estimate=2.0,
        category="structural",
        posterior_mean=2.01,
        posterior_sd=0.1,
        q05=1.85,
        q50=2.00,
        q95=2.18,
    )
    assert pe.posterior_mean == 2.01
    assert pe.posterior_sd == 0.1
    assert pe.q05 == 1.85
    assert pe.q50 == 2.00
    assert pe.q95 == 2.18


def test_parameter_estimate_json_roundtrip() -> None:
    pe = ParameterEstimate(
        name="CL",
        estimate=2.0,
        category="structural",
        posterior_mean=2.01,
        posterior_sd=0.1,
        q05=1.85,
        q50=2.00,
        q95=2.18,
    )
    raw = pe.model_dump_json()
    data = json.loads(raw)
    assert data["posterior_mean"] == 2.01
    restored = ParameterEstimate.model_validate_json(raw)
    assert restored == pe


def test_parameter_estimate_posterior_fields_default_to_none() -> None:
    pe = ParameterEstimate(name="CL", estimate=2.0, category="structural")
    assert pe.posterior_mean is None
    assert pe.posterior_sd is None
    assert pe.q05 is None
    assert pe.q50 is None
    assert pe.q95 is None
