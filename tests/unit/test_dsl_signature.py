# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for apmode.dsl.serializer.build_signature (Formular sharpening plan §4 Phase 2, P2.4).

Covers:
- One test per absorption/distribution/elimination short-code mapping
  (parametrized)
- Multi-param-IIV test and no-IIV test
- Full end-to-end test compiling a real literature-anchor DSLSpec fixture
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    TMDDQSS,
    Additive,
    Combined,
    DSLSpec,
    Erlang,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    NODEAbsorption,
    NODEElimination,
    ObservationEndpoint,
    OneCmt,
    ParallelFirstOrder,
    ParallelLinearMM,
    Proportional,
    SumIG,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.serializer import build_signature

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THEOPHYLLINE_FIXTURE = _REPO_ROOT / "benchmarks/suite_c/theophylline_boeckmann_1992.dsl.json"


def _minimal_spec(**overrides: object) -> DSLSpec:
    fields: dict[str, object] = {
        "model_id": "sig_test",
        "absorption": FirstOrder(),
        "distribution": OneCmt(),
        "elimination": LinearElim(),
        "variability": [],
        "observation": Proportional(sigma_prop=0.1),
        "initial": {"ka": 1.0, "V": 10.0, "CL": 5.0},
    }
    fields.update(overrides)
    return DSLSpec.model_validate(fields)


# ---------------------------------------------------------------------------
# Absorption short codes
# ---------------------------------------------------------------------------

_ABSORPTION_CASES = [
    (IVBolus(), "IV bolus"),
    (FirstOrder(), "FO"),
    (ZeroOrder(), "ZO"),
    (LaggedFirstOrder(), "FO+lag"),
    (Transit(n=3), "Transit(3)"),
    (MixedFirstZero(), "FO+ZO mixed"),
    (Erlang(n=5), "Erlang(5)"),
    (ParallelFirstOrder(), "Parallel-FO"),
    (SumIG(k=2), "SumIG(2)"),
]


@pytest.mark.parametrize(("module", "code"), _ABSORPTION_CASES)
def test_absorption_short_code(module: object, code: str) -> None:
    spec = _minimal_spec(absorption=module)
    assert build_signature(spec).startswith(f"{code} absorption |")


def test_node_absorption_short_code() -> None:
    spec = _minimal_spec(absorption=NODEAbsorption(dim=2, constraint_template="bounded_positive"))
    assert build_signature(spec).startswith("NODE-abs absorption |")


# ---------------------------------------------------------------------------
# Distribution short codes
# ---------------------------------------------------------------------------

_DISTRIBUTION_CASES = [
    (OneCmt(), "1CMT"),
    (TwoCmt(), "2CMT"),
    (ThreeCmt(), "3CMT"),
    (TMDDCore(), "TMDD-full"),
    (TMDDQSS(), "TMDD-QSS"),
]


@pytest.mark.parametrize(("module", "code"), _DISTRIBUTION_CASES)
def test_distribution_short_code(module: object, code: str) -> None:
    spec = _minimal_spec(distribution=module)
    signature = build_signature(spec)
    segments = signature.split(" | ")
    assert segments[1] == code


# ---------------------------------------------------------------------------
# Elimination short codes
# ---------------------------------------------------------------------------

_ELIMINATION_CASES = [
    (LinearElim(), "Linear CL"),
    (MichaelisMenten(), "MM"),
    (ParallelLinearMM(), "Linear+MM"),
    (TimeVaryingElim(decay_fn="exponential"), "TimeVarying CL"),
    (NODEElimination(dim=2, constraint_template="saturable"), "NODE-elim"),
]


@pytest.mark.parametrize(("module", "code"), _ELIMINATION_CASES)
def test_elimination_short_code(module: object, code: str) -> None:
    spec = _minimal_spec(elimination=module)
    signature = build_signature(spec)
    segments = signature.split(" | ")
    assert segments[2] == code


@pytest.mark.parametrize("tmdd_distribution", [TMDDCore(), TMDDQSS()])
def test_tmdd_distribution_omits_elimination_segment(tmdd_distribution: object) -> None:
    """TMDD dynamics ignore spec.elimination entirely (always kel = CL/V).

    Rendering the declared (but inert) elimination module's short code
    next to a TMDD distribution segment would misleadingly imply e.g.
    Michaelis-Menten kinetics are active — see nlmixr2_emitter's
    ``_emit_tmdd_core_odes``/``_emit_tmdd_qss_odes``.
    """
    spec = _minimal_spec(distribution=tmdd_distribution, elimination=MichaelisMenten())
    signature = build_signature(spec)
    assert "MM" not in signature.split(" | ")
    assert "Linear CL" not in signature
    assert "Linear+MM" not in signature


# ---------------------------------------------------------------------------
# Observation short codes (single-endpoint form)
# ---------------------------------------------------------------------------

_OBSERVATION_CASES = [
    (Proportional(sigma_prop=0.1), "Prop error"),
    (Additive(sigma_add=0.1), "Add error"),
    (Combined(sigma_prop=0.1, sigma_add=0.1), "Combined error"),
    (BLQM3(loq_value=0.1), "BLQ-M3"),
    (BLQM4(loq_value=0.1), "BLQ-M4"),
]


@pytest.mark.parametrize(("module", "code"), _OBSERVATION_CASES)
def test_observation_short_code(module: object, code: str) -> None:
    spec = _minimal_spec(observation=module)
    signature = build_signature(spec)
    assert signature.endswith(code)


# ---------------------------------------------------------------------------
# IIV segment: multi-param, multi-item, and no-IIV cases
# ---------------------------------------------------------------------------


def test_no_iiv_omits_segment() -> None:
    spec = _minimal_spec(variability=[])
    signature = build_signature(spec)
    assert "IIV" not in signature
    # 3 fixed segments (absorption, distribution, elimination) + observation.
    assert len(signature.split(" | ")) == 4


def test_multi_param_iiv_diagonal() -> None:
    spec = _minimal_spec(variability=[IIV(params=["ka", "V", "CL"], structure="diagonal")])
    signature = build_signature(spec)
    assert "IIV(CL,V,ka) diag" in signature


def test_multi_iiv_items_block_and_diagonal() -> None:
    spec = _minimal_spec(
        variability=[
            IIV(params=["CL", "V"], structure="diagonal"),
            IIV(params=["ka"], structure="block"),
        ]
    )
    signature = build_signature(spec)
    assert "IIV(CL,V) diag" in signature
    assert "IIV(ka) block" in signature


# ---------------------------------------------------------------------------
# Multi-analyte observations segment
# ---------------------------------------------------------------------------


def test_multi_analyte_observations_segment() -> None:
    spec = _minimal_spec(
        distribution=TMDDQSS(),
        observation=Proportional(sigma_prop=0.1),
        observations={
            "free_drug": ObservationEndpoint(
                name="free_drug",
                dvid=1,
                prediction="C_central",
                error=Proportional(sigma_prop=0.1),
            ),
            "total_target": ObservationEndpoint(
                name="total_target",
                dvid=2,
                prediction="C_target_total",
                error=Additive(sigma_add=0.1),
            ),
        },
        initial={
            "ka": 1.0,
            "V": 10.0,
            "R0": 1.0,
            "KD": 1.0,
            "kint": 0.1,
        },
    )
    signature = build_signature(spec)
    assert signature.endswith("2 endpoints (Prop error, Add error)")


# ---------------------------------------------------------------------------
# End-to-end: real literature-anchor fixture
# ---------------------------------------------------------------------------


def test_theophylline_fixture_exact_signature() -> None:
    spec = DSLSpec.model_validate_json(_THEOPHYLLINE_FIXTURE.read_text())
    assert (
        build_signature(spec)
        == "FO absorption | 1CMT | Linear CL | IIV(CL,V,ka) diag | Combined error"
    )
