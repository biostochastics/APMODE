# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for reproducibility bundle Pydantic models (ARCHITECTURE.md §5)."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from apmode.bundle.models import (
    BackendResult,
    BackendVersions,
    BLQHandling,
    CandidateLineage,
    CandidateLineageEntry,
    ColumnMapping,
    ConvergenceMetadata,
    CovariateMetadata,
    CovariateSpec,
    DataManifest,
    DiagnosticBundle,
    EvidenceManifest,
    FailedCandidate,
    GOFMetrics,
    IdentifiabilityFlags,
    InitialEstimateEntry,
    InitialEstimates,
    ParameterEstimate,
    PolicyFile,
    ReportProvenance,
    SearchTrajectoryEntry,
    SeedRegistry,
    SplitManifest,
    SubjectAssignment,
    VPCSummary,
)
from apmode.ids import generate_candidate_id

VALID_SHA256 = "a" * 64


class TestParameterEstimate:
    def test_valid(self) -> None:
        pe = ParameterEstimate(
            name="CL",
            estimate=5.0,
            se=0.3,
            rse=6.0,
            ci95_lower=4.4,
            ci95_upper=5.6,
            fixed=False,
            category="structural",
        )
        assert pe.name == "CL"

    def test_fixed_parameter_no_se(self) -> None:
        pe = ParameterEstimate(name="V1", estimate=70.0, fixed=True, category="structural")
        assert pe.fixed is True

    def test_invalid_category(self) -> None:
        with pytest.raises(ValidationError):
            ParameterEstimate(name="CL", estimate=5.0, fixed=False, category="invalid_category")


class TestConvergenceMetadata:
    def test_valid(self) -> None:
        cm = ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=300,
            gradient_norm=1e-6,
            minimization_status="successful",
            wall_time_seconds=45.2,
        )
        assert cm.converged is True

    def test_extended_statuses(self) -> None:
        for status in ("rounding_errors", "max_evaluations"):
            cm = ConvergenceMetadata(
                method="focei",
                converged=False,
                iterations=500,
                minimization_status=status,
                wall_time_seconds=60.0,
            )
            assert cm.minimization_status == status


class TestGOFMetrics:
    def test_valid(self) -> None:
        gof = GOFMetrics(cwres_mean=0.01, cwres_sd=1.02, outlier_fraction=0.02)
        assert gof.outlier_fraction == pytest.approx(0.02)

    def test_outlier_fraction_bounds(self) -> None:
        with pytest.raises(ValidationError):
            GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=1.5)


class TestVPCSummary:
    def test_valid(self) -> None:
        vpc = VPCSummary(
            percentiles=[5.0, 50.0, 95.0],
            coverage={"p5": 0.93, "p50": 0.97, "p95": 0.94},
            n_bins=10,
            prediction_corrected=True,
        )
        assert vpc.prediction_corrected is True

    def test_coverage_keys_must_match_percentiles(self) -> None:
        with pytest.raises(ValidationError, match="coverage keys"):
            VPCSummary(
                percentiles=[5.0, 50.0, 95.0],
                coverage={"wrong": 0.9},
                n_bins=10,
                prediction_corrected=True,
            )


class TestIdentifiabilityFlags:
    def test_valid(self) -> None:
        idf = IdentifiabilityFlags(
            condition_number=45.3,
            profile_likelihood_ci={"CL": True, "V": True, "ka": False},
            ill_conditioned=False,
        )
        assert idf.ill_conditioned is False


class TestDiagnosticBundle:
    def test_valid(self) -> None:
        db = DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.01),
            vpc=None,
            identifiability=IdentifiabilityFlags(
                condition_number=30.0,
                profile_likelihood_ci={"CL": True},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", lloq=None, n_blq=0, blq_fraction=0.0),
            diagnostic_plots={},
        )
        assert db.vpc is None


class TestBackendResult:
    def _make_result(self, **overrides: object) -> BackendResult:
        defaults = dict(
            model_id=generate_candidate_id(),
            backend="nlmixr2",
            converged=True,
            ofv=-1234.5,
            aic=2479.0,
            bic=2495.0,
            parameter_estimates={
                "CL": ParameterEstimate(
                    name="CL", estimate=5.0, fixed=False, category="structural"
                )
            },
            eta_shrinkage={"CL": 0.12},
            convergence_metadata=ConvergenceMetadata(
                method="saem",
                converged=True,
                iterations=300,
                minimization_status="successful",
                wall_time_seconds=42.0,
            ),
            diagnostics=DiagnosticBundle(
                gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.01),
                vpc=None,
                identifiability=IdentifiabilityFlags(
                    condition_number=30.0,
                    profile_likelihood_ci={"CL": True},
                    ill_conditioned=False,
                ),
                blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
                diagnostic_plots={},
            ),
            wall_time_seconds=42.0,
            backend_versions={"nlmixr2": "3.0.0", "R": "4.4.1"},
            initial_estimate_source="nca",
        )
        defaults.update(overrides)
        return BackendResult(**defaults)

    def test_valid_full(self) -> None:
        result = self._make_result()
        assert result.converged is True
        assert result.backend == "nlmixr2"

    def test_json_roundtrip(self) -> None:
        result = self._make_result(converged=False, ofv=None, aic=None, bic=None)
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        roundtripped = BackendResult.model_validate(parsed)
        assert roundtripped.model_id == result.model_id


class TestDataManifest:
    def test_valid(self) -> None:
        dm = DataManifest(
            data_sha256=VALID_SHA256,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT"
            ),
            n_subjects=50,
            n_observations=500,
            n_doses=150,
            blq_coding="flag",
            covariates=[CovariateMetadata(name="WT", type="continuous", time_varying=False)],
        )
        assert dm.n_subjects == 50

    def test_invalid_ingestion_format(self) -> None:
        with pytest.raises(ValidationError):
            DataManifest(
                data_sha256=VALID_SHA256,
                ingestion_format="invalid_format",
                column_mapping=ColumnMapping(
                    subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT"
                ),
                n_subjects=1,
                n_observations=1,
                n_doses=1,
            )

    def test_invalid_sha256(self) -> None:
        with pytest.raises(ValidationError):
            DataManifest(
                data_sha256="not_hex_at_all_" * 5,
                ingestion_format="nonmem_csv",
                column_mapping=ColumnMapping(
                    subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT"
                ),
                n_subjects=1,
                n_observations=1,
                n_doses=1,
            )

    def test_frozen(self) -> None:
        dm = DataManifest(
            data_sha256=VALID_SHA256,
            ingestion_format="nonmem_csv",
            column_mapping=ColumnMapping(
                subject_id="ID", time="TIME", dv="DV", evid="EVID", amt="AMT"
            ),
            n_subjects=1,
            n_observations=1,
            n_doses=1,
        )
        with pytest.raises(ValidationError):
            dm.n_subjects = 99  # type: ignore[misc]


class TestSplitManifest:
    def test_valid(self) -> None:
        sm = SplitManifest(
            split_seed=42,
            split_strategy="subject_level",
            assignments=[
                SubjectAssignment(subject_id="1", fold="train"),
                SubjectAssignment(subject_id="2", fold="test"),
            ],
        )
        assert len(sm.assignments) == 2


class TestSeedRegistry:
    def test_valid(self) -> None:
        sr = SeedRegistry(
            root_seed=42,
            r_seed=42,
            r_rng_kind="L'Ecuyer-CMRG",
            np_seed=42,
            jax_key=None,
            backend_seeds={"nlmixr2_run1": 42},
        )
        assert sr.root_seed == 42

    def test_frozen(self) -> None:
        sr = SeedRegistry(
            root_seed=42,
            r_seed=42,
            r_rng_kind="L'Ecuyer-CMRG",
            np_seed=42,
        )
        with pytest.raises(ValidationError):
            sr.root_seed = 99  # type: ignore[misc]


class TestEvidenceManifest:
    def test_valid(self) -> None:
        em = EvidenceManifest(
            route_certainty="confirmed",
            absorption_complexity="simple",
            nonlinear_clearance_signature=False,
            richness_category="moderate",
            identifiability_ceiling="medium",
            covariate_burden=5,
            covariate_correlated=False,
            covariate_missingness=CovariateSpec(
                pattern="MCAR", fraction_incomplete=0.02, strategy="impute-median"
            ),
            blq_burden=0.05,
            protocol_heterogeneity="single-study",
            absorption_phase_coverage="adequate",
            elimination_phase_coverage="adequate",
        )
        assert em.richness_category == "moderate"

    def test_blq_burden_bounds(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceManifest(
                route_certainty="confirmed",
                absorption_complexity="simple",
                nonlinear_clearance_signature=False,
                richness_category="moderate",
                identifiability_ceiling="medium",
                covariate_burden=5,
                covariate_correlated=False,
                blq_burden=1.5,
                protocol_heterogeneity="single-study",
                absorption_phase_coverage="adequate",
                elimination_phase_coverage="adequate",
            )


class TestInitialEstimates:
    def test_valid(self) -> None:
        ie = InitialEstimates(
            entries={
                "candidate1": InitialEstimateEntry(
                    candidate_id="candidate1",
                    source="nca",
                    estimates={"CL": 5.0, "V": 70.0},
                    inputs_used=["per_subject_nca"],
                ),
            }
        )
        assert ie.entries["candidate1"].source == "nca"


class TestSearchTrajectoryEntry:
    def test_valid(self) -> None:
        ste = SearchTrajectoryEntry(
            candidate_id=generate_candidate_id(),
            parent_id=None,
            backend="nlmixr2",
            converged=True,
            ofv=-1234.5,
            aic=2479.0,
            bic=2495.0,
            gate1_passed=True,
            gate2_passed=True,
            wall_time_seconds=42.0,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        assert ste.gate1_passed is True


class TestFailedCandidate:
    def test_valid(self) -> None:
        fc = FailedCandidate(
            candidate_id=generate_candidate_id(),
            backend="nlmixr2",
            gate_failed="gate1",
            failed_checks=["convergence", "cwres_mean"],
            summary_reason="SAEM did not converge",
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        assert fc.gate_failed == "gate1"


class TestCandidateLineage:
    def test_valid(self) -> None:
        cl = CandidateLineage(
            entries=[
                CandidateLineageEntry(candidate_id="child1"),
                CandidateLineageEntry(
                    candidate_id="child2",
                    parent_id="child1",
                    transform="swap_module(elimination, MichaelisMenten)",
                ),
            ]
        )
        assert len(cl.entries) == 2


class TestBackendVersions:
    def test_valid(self) -> None:
        bv = BackendVersions(
            apmode_version="0.1.0",
            python_version="3.12.5",
            r_version="4.4.1",
            nlmixr2_version="3.0.0",
            git_sha="abc123def456",
        )
        assert bv.r_version == "4.4.1"


class TestPolicyFile:
    def test_valid(self) -> None:
        pf = PolicyFile(
            policy_version="1.0.0",
            lane="submission",
            gate1_thresholds={"cwres_mean_max": 0.1, "outlier_fraction_max": 0.05},
            gate2_thresholds={"shrinkage_max": 0.30},
        )
        assert pf.lane == "submission"

    def test_invalid_lane(self) -> None:
        with pytest.raises(ValidationError):
            PolicyFile(
                policy_version="1.0.0",
                lane="invalid_lane",
                gate1_thresholds={},
                gate2_thresholds={},
            )


class TestReportProvenance:
    def test_valid(self) -> None:
        rp = ReportProvenance(
            generated_at=datetime.now(tz=UTC).isoformat(),
            apmode_version="0.1.0",
            generator="apmode.bundle.emitter",
            component_versions={"nlmixr2": "3.0.0"},
        )
        assert rp.generator == "apmode.bundle.emitter"
