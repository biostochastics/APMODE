# SPDX-License-Identifier: GPL-2.0-or-later
"""Default-freeze contract for the shared ``tests/_helpers`` builders.

These builders feed load-bearing gate/ranking/RO-Crate invariant pins across
the suite. A silent drift in a default field value would break those pins in
ways that are hard to trace back to the builder. This test pins the frozen
default field values verbatim so any future edit to ``tests/_helpers`` that
changes a default fails here first, loudly and locally.
"""

from __future__ import annotations

from apmode.dsl.ast_models import FirstOrder, LinearElim, OneCmt, Proportional
from tests._helpers.builders import make_backend_result, make_spec


def test_make_backend_result_frozen_defaults() -> None:
    result = make_backend_result()

    # Backend + information criteria.
    assert result.backend == "nlmixr2"
    assert result.ofv == 150.0
    assert result.aic == 160.0
    assert result.bic == 170.0

    # GOF diagnostics.
    assert result.diagnostics.gof.cwres_mean == 0.01
    assert result.diagnostics.gof.cwres_sd == 1.0
    assert result.diagnostics.gof.outlier_fraction == 0.02
    assert result.diagnostics.gof.obs_vs_pred_r2 == 0.95

    # VPC coverage dict.
    assert result.diagnostics.vpc is not None
    assert result.diagnostics.vpc.coverage == {"p5": 0.92, "p50": 0.97, "p95": 0.93}

    # PIT calibration dict (well-calibrated default: c_p == p).
    assert result.diagnostics.pit_calibration is not None
    assert result.diagnostics.pit_calibration.calibration == {
        "p5": 0.05,
        "p50": 0.50,
        "p95": 0.95,
    }

    # Eta shrinkage.
    assert result.eta_shrinkage == {"CL": 0.05, "V": 0.08, "ka": 0.12}

    # Identifiability.
    assert result.diagnostics.identifiability.condition_number == 15.0
    assert result.diagnostics.identifiability.ill_conditioned is False


def test_make_spec_frozen_defaults() -> None:
    spec = make_spec()

    assert isinstance(spec.absorption, FirstOrder)
    assert isinstance(spec.distribution, OneCmt)
    assert isinstance(spec.elimination, LinearElim)
    assert isinstance(spec.observation, Proportional)
    assert spec.observation.sigma_prop == 0.1
    assert spec.initial == {"ka": 1.0, "V": 70.0, "CL": 5.0}
