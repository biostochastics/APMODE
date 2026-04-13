# SPDX-License-Identifier: GPL-2.0-or-later
"""Golden master (snapshot) tests for nlmixr2 R code lowering.

Uses pytest-syrupy to snapshot emitted R code. These snapshots represent
pharmacometrician-validated output and catch unintended changes to the
R code emitter.
"""

from syrupy.assertion import SnapshotAssertion

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    Additive,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    OccasionByStudy,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.nlmixr2_emitter import emit_nlmixr2

# Fixed model_id for deterministic snapshots
_MODEL_ID = "golden_test_model_id_0"


def _make_spec(**overrides: object) -> DSLSpec:
    defaults: dict[str, object] = {
        "model_id": _MODEL_ID,
        "absorption": FirstOrder(ka=1.0),
        "distribution": OneCmt(V=70.0),
        "elimination": LinearElim(CL=5.0),
        "variability": [IIV(params=["CL", "V"], structure="diagonal")],
        "observation": Proportional(sigma_prop=0.1),
    }
    defaults.update(overrides)
    return DSLSpec(**defaults)  # type: ignore[arg-type]


class TestGoldenMaster1CmtOral:
    """Golden master: simplest 1-compartment oral model."""

    def test_1cmt_fo_linear_prop(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec())
        assert r_code == snapshot

    def test_1cmt_fo_linear_additive(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=Additive(sigma_add=1.0)))
        assert r_code == snapshot

    def test_1cmt_fo_linear_combined(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=Combined(sigma_prop=0.1, sigma_add=0.5)))
        assert r_code == snapshot


class TestGoldenMasterAbsorptionVariants:
    """Golden master: different absorption mechanisms."""

    def test_zero_order(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=ZeroOrder(dur=0.5)))
        assert r_code == snapshot

    def test_lagged_first_order(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=LaggedFirstOrder(ka=1.5, tlag=0.3)))
        assert r_code == snapshot

    def test_transit(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=Transit(n=4, ktr=2.0, ka=1.0)))
        assert r_code == snapshot

    def test_mixed_first_zero(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(absorption=MixedFirstZero(ka=1.0, dur=0.5, frac=0.6)))
        assert r_code == snapshot


class TestGoldenMasterDistributionVariants:
    """Golden master: multi-compartment and TMDD."""

    def test_2cmt(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(distribution=TwoCmt(V1=10.0, V2=20.0, Q=3.0)))
        assert r_code == snapshot

    def test_3cmt(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(
            _make_spec(distribution=ThreeCmt(V1=10.0, V2=20.0, V3=5.0, Q2=3.0, Q3=1.0))
        )
        assert r_code == snapshot

    def test_tmdd_core(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                distribution=TMDDCore(V=50.0, R0=10.0, kon=0.1, koff=0.01, kint=0.05),
            )
        )
        assert r_code == snapshot

    def test_tmdd_qss(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                distribution=TMDDQSS(V=50.0, R0=10.0, KD=0.5, kint=0.05),
            )
        )
        assert r_code == snapshot


class TestGoldenMasterEliminationVariants:
    """Golden master: non-linear elimination."""

    def test_michaelis_menten(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(elimination=MichaelisMenten(Vmax=100.0, Km=10.0)))
        assert r_code == snapshot

    def test_parallel_linear_mm(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0)))
        assert r_code == snapshot

    def test_time_varying(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(
            _make_spec(elimination=TimeVaryingElim(CL=5.0, decay_fn="exponential"))
        )
        assert r_code == snapshot


class TestGoldenMasterBLQ:
    """Golden master: BLQ handling."""

    def test_blq_m3(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=BLQM3(loq_value=0.1)))
        assert r_code == snapshot

    def test_blq_m4(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(observation=BLQM4(loq_value=0.5)))
        assert r_code == snapshot


class TestGoldenMasterVariability:
    """Golden master: variability patterns."""

    def test_block_iiv(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(_make_spec(variability=[IIV(params=["CL", "V"], structure="block")]))
        assert r_code == snapshot

    def test_iov(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="diagonal"),
                    IOV(params=["CL"], occasions=OccasionByStudy()),
                ]
            )
        )
        assert r_code == snapshot

    def test_covariate_power(self, snapshot: SnapshotAssertion) -> None:
        r_code = emit_nlmixr2(
            _make_spec(
                variability=[
                    IIV(params=["CL", "V"], structure="diagonal"),
                    CovariateLink(param="CL", covariate="WT", form="power"),
                    CovariateLink(param="V", covariate="WT", form="power"),
                ]
            )
        )
        assert r_code == snapshot


class TestGoldenMasterComplex:
    """Golden master: realistic complex models."""

    def test_2cmt_mm_block_covariate(self, snapshot: SnapshotAssertion) -> None:
        """2-compartment, parallel MM elimination, block IIV, covariates."""
        r_code = emit_nlmixr2(
            _make_spec(
                absorption=LaggedFirstOrder(ka=1.5, tlag=0.3),
                distribution=TwoCmt(V1=30.0, V2=40.0, Q=5.0),
                elimination=ParallelLinearMM(CL=2.0, Vmax=50.0, Km=5.0),
                variability=[
                    IIV(params=["CL", "V1", "ka"], structure="block"),
                    CovariateLink(param="CL", covariate="WT", form="power"),
                    CovariateLink(param="V1", covariate="WT", form="power"),
                ],
                observation=Combined(sigma_prop=0.1, sigma_add=0.5),
            )
        )
        assert r_code == snapshot

    def test_3cmt_transit_blq(self, snapshot: SnapshotAssertion) -> None:
        """3-compartment, transit absorption, BLQ M3."""
        r_code = emit_nlmixr2(
            _make_spec(
                absorption=Transit(n=5, ktr=3.0, ka=1.5),
                distribution=ThreeCmt(V1=15.0, V2=25.0, V3=8.0, Q2=4.0, Q3=1.5),
                elimination=LinearElim(CL=3.5),
                variability=[
                    IIV(params=["CL", "V1", "ktr"], structure="diagonal"),
                ],
                observation=BLQM3(loq_value=0.05),
            )
        )
        assert r_code == snapshot
