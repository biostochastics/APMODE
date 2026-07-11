# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for apmode.dsl.canonical (P0.5 — canonical schema + fingerprints).

Covers:
- fingerprint stability: same structure, different calibrated values
  -> same structure_fingerprint, different spec_fingerprint
- fingerprint sensitivity: changing a module variant type changes
  structure_fingerprint (and spec_fingerprint)
- canonicalization determinism: dict/list construction order does not
  perturb the digest
- justification_hash isolation from spec_fingerprint
- schema version travels with every result
- initial_fingerprint hashes DSLSpec.initial directly (Phase 1, P1.4)
"""

from __future__ import annotations

from apmode.dsl.ast_models import (
    IIV,
    IOV,
    Additive,
    CovariateLink,
    DSLSpec,
    Erlang,
    FirstOrder,
    LinearElim,
    ObservationEndpoint,
    OccasionByVisit,
    OneCmt,
    Proportional,
    Transit,
    TwoCmt,
)
from apmode.dsl.canonical import (
    CANONICAL_SCHEMA_VERSION,
    initial_fingerprint,
    justification_hash,
    spec_fingerprint,
    structure_fingerprint,
)
from apmode.dsl.priors import NormalPrior, PriorSpec


def _base_spec(*, ka: float = 1.0, cl: float = 5.0, v: float = 70.0) -> DSLSpec:
    return DSLSpec(
        model_id="fp_test_model_0000001",
        absorption=FirstOrder(),
        distribution=OneCmt(),
        elimination=LinearElim(),
        variability=[IIV(params=["CL", "V"], structure="diagonal")],
        observation=Proportional(sigma_prop=0.1),
        initial={"ka": ka, "CL": cl, "V": v},
    )


class TestMacrosUsedExcludedFromFingerprints:
    """P2.1: macro-expansion provenance is sugar, not semantics (no schema bump)."""

    def test_macros_used_does_not_perturb_structure_fingerprint(self) -> None:
        spec_a = _base_spec()
        spec_b = spec_a.model_copy(update={"macros_used": ["pkstd.standard_iiv@v1"]})
        assert spec_a.macros_used != spec_b.macros_used
        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)

    def test_macros_used_does_not_perturb_spec_fingerprint(self) -> None:
        spec_a = _base_spec()
        spec_b = spec_a.model_copy(
            update={"macros_used": ["pkstd.standard_iiv@v1", "pkstd.standard_priors@v1"]}
        )
        assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)

    def test_schema_version_unchanged_by_macros_used_field(self) -> None:
        """Adding `macros_used` to DSLSpec required no CANONICAL_SCHEMA_VERSION bump."""
        assert CANONICAL_SCHEMA_VERSION == "2.3.0"


class TestFingerprintResultShape:
    def test_every_fingerprint_carries_schema_version(self) -> None:
        spec = _base_spec()
        for fn in (
            structure_fingerprint,
            spec_fingerprint,
            initial_fingerprint,
            justification_hash,
        ):
            result = fn(spec)
            assert result["schema"] == CANONICAL_SCHEMA_VERSION
            assert isinstance(result["digest"], str)
            assert len(result["digest"]) == 64  # sha256 hex


class TestFingerprintStability:
    """Same structural spec, different calibrated (initial-estimate) values."""

    def test_structure_fingerprint_ignores_calibrated_values(self) -> None:
        spec_a = _base_spec(ka=1.0, cl=5.0, v=70.0)
        spec_b = _base_spec(ka=2.5, cl=12.3, v=45.0)

        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)

    def test_spec_fingerprint_distinguishes_calibrated_values(self) -> None:
        spec_a = _base_spec(ka=1.0, cl=5.0, v=70.0)
        spec_b = _base_spec(ka=2.5, cl=12.3, v=45.0)

        assert spec_fingerprint(spec_a) != spec_fingerprint(spec_b)

    def test_initial_fingerprint_distinguishes_calibrated_values(self) -> None:
        spec_a = _base_spec(ka=1.0, cl=5.0, v=70.0)
        spec_b = _base_spec(ka=2.5, cl=12.3, v=45.0)

        assert initial_fingerprint(spec_a) != initial_fingerprint(spec_b)

    def test_initial_fingerprint_ignores_structure(self) -> None:
        """Two specs with different topology but the same initial: dict collide here."""
        spec_a = _base_spec(ka=1.0, cl=5.0, v=70.0)
        transit_spec = DSLSpec(
            model_id="transit_same_initial_0001",
            absorption=Transit(n=3),
            distribution=spec_a.distribution,
            elimination=spec_a.elimination,
            variability=spec_a.variability,
            observation=spec_a.observation,
            initial={**spec_a.initial, "ktr": 2.0},
        )
        # Different topology (Transit adds ktr) -> different structure fingerprint...
        assert structure_fingerprint(spec_a) != structure_fingerprint(transit_spec)
        # ...but initial_fingerprint compares dicts directly, so adding a key changes it too.
        assert initial_fingerprint(spec_a) != initial_fingerprint(transit_spec)
        # Same initial dict, different structure -> same initial_fingerprint.
        transit_spec_2 = transit_spec.model_copy(update={"absorption": Transit(n=5)})
        assert initial_fingerprint(transit_spec) == initial_fingerprint(transit_spec_2)
        assert structure_fingerprint(transit_spec) != structure_fingerprint(transit_spec_2)

    def test_model_id_does_not_affect_any_fingerprint(self) -> None:
        spec_a = _base_spec()
        spec_b = spec_a.model_copy(update={"model_id": "a_completely_different_id_00"})

        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)
        assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)
        assert initial_fingerprint(spec_a) == initial_fingerprint(spec_b)
        assert justification_hash(spec_a) == justification_hash(spec_b)

    def test_structure_fingerprint_ignores_covariate_theta_and_ref(self) -> None:
        """P1.6: theta/ref are calibration-like, excluded from structure_fingerprint."""
        spec_a = _base_spec().model_copy(
            update={
                "covariates": [
                    CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
                ]
            }
        )
        spec_b = _base_spec().model_copy(
            update={
                "covariates": [
                    CovariateLink(param="CL", covariate="WT", form="power", theta=0.5, ref=80.0)
                ]
            }
        )

        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)
        assert spec_fingerprint(spec_a) != spec_fingerprint(spec_b)


class TestFingerprintSensitivity:
    """Structural (topology) changes must change structure_fingerprint."""

    def test_absorption_variant_type_change_changes_structure_fingerprint(self) -> None:
        first_order_spec = _base_spec()
        transit_spec = DSLSpec(
            model_id=first_order_spec.model_id,
            absorption=Transit(n=3),
            distribution=first_order_spec.distribution,
            elimination=first_order_spec.elimination,
            variability=first_order_spec.variability,
            observation=first_order_spec.observation,
            initial={"ktr": 1.0, "ka": 1.0, "CL": 5.0, "V": 70.0},
        )

        assert structure_fingerprint(first_order_spec) != structure_fingerprint(transit_spec)
        assert spec_fingerprint(first_order_spec) != spec_fingerprint(transit_spec)

    def test_distribution_variant_type_change_changes_structure_fingerprint(self) -> None:
        one_cmt_spec = _base_spec()
        two_cmt_spec = DSLSpec(
            model_id=one_cmt_spec.model_id,
            absorption=one_cmt_spec.absorption,
            distribution=TwoCmt(),
            elimination=one_cmt_spec.elimination,
            variability=one_cmt_spec.variability,
            observation=one_cmt_spec.observation,
            initial={"ka": 1.0, "CL": 5.0, "V1": 40.0, "V2": 60.0, "Q": 3.0},
        )

        assert structure_fingerprint(one_cmt_spec) != structure_fingerprint(two_cmt_spec)

    def test_transit_compartment_count_changes_structure_fingerprint(self) -> None:
        transit_3 = DSLSpec(
            model_id="transit_test_00000001",
            absorption=Transit(n=3),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ktr": 1.0, "ka": 1.0, "CL": 5.0, "V": 70.0},
        )
        transit_5 = DSLSpec(
            model_id=transit_3.model_id,
            absorption=Transit(n=5),
            distribution=transit_3.distribution,
            elimination=transit_3.elimination,
            variability=transit_3.variability,
            observation=transit_3.observation,
            initial=transit_3.initial,
        )

        # n is structural (compartment count) -> must perturb structure_fingerprint,
        # even though ktr/ka (calibrated values) are identical.
        assert structure_fingerprint(transit_3) != structure_fingerprint(transit_5)

    def test_erlang_n_is_structural(self) -> None:
        erlang_3 = DSLSpec(
            model_id="erlang_test_000000001",
            absorption=Erlang(n=3),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ktr": 1.0, "CL": 5.0, "V": 70.0},
        )
        erlang_4 = DSLSpec(
            model_id=erlang_3.model_id,
            absorption=Erlang(n=4),
            distribution=erlang_3.distribution,
            elimination=erlang_3.elimination,
            variability=erlang_3.variability,
            observation=erlang_3.observation,
            initial=erlang_3.initial,
        )

        assert structure_fingerprint(erlang_3) != structure_fingerprint(erlang_4)

    def test_iiv_structure_diagonal_vs_block_is_structural(self) -> None:
        diagonal_spec = _base_spec()
        block_spec = DSLSpec(
            model_id=diagonal_spec.model_id,
            absorption=diagonal_spec.absorption,
            distribution=diagonal_spec.distribution,
            elimination=diagonal_spec.elimination,
            variability=[IIV(params=["CL", "V"], structure="block")],
            observation=diagonal_spec.observation,
            initial=diagonal_spec.initial,
        )

        assert structure_fingerprint(diagonal_spec) != structure_fingerprint(block_spec)

    def test_covariate_link_addition_is_structural(self) -> None:
        spec_no_cov = _base_spec()
        spec_with_cov = spec_no_cov.model_copy(
            update={
                "covariates": [
                    CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
                ]
            }
        )

        assert structure_fingerprint(spec_no_cov) != structure_fingerprint(spec_with_cov)

    def test_covariate_form_change_is_structural(self) -> None:
        spec_power = _base_spec().model_copy(
            update={
                "covariates": [
                    CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
                ]
            }
        )
        spec_exponential = _base_spec().model_copy(
            update={
                "covariates": [
                    CovariateLink(param="CL", covariate="WT", form="exponential", theta=0.75)
                ]
            }
        )

        assert structure_fingerprint(spec_power) != structure_fingerprint(spec_exponential)

    def test_iov_occasion_column_is_structural(self) -> None:
        spec_visit_a = DSLSpec(
            model_id="iov_test_0000000000001",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IOV(params=["CL"], occasions=OccasionByVisit(column="VISIT"))],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "CL": 5.0, "V": 70.0},
        )
        spec_visit_b = DSLSpec(
            model_id=spec_visit_a.model_id,
            absorption=spec_visit_a.absorption,
            distribution=spec_visit_a.distribution,
            elimination=spec_visit_a.elimination,
            variability=[IOV(params=["CL"], occasions=OccasionByVisit(column="EPOCH"))],
            observation=spec_visit_a.observation,
            initial=spec_visit_a.initial,
        )

        assert structure_fingerprint(spec_visit_a) != structure_fingerprint(spec_visit_b)

    def test_prior_target_presence_is_structural(self) -> None:
        spec_no_prior = _base_spec()
        spec_with_prior = spec_no_prior.model_copy(
            update={
                "priors": [
                    PriorSpec(
                        target="CL",
                        family=NormalPrior(mu=0.0, sigma=2.0),
                        source="weakly_informative",
                    )
                ]
            }
        )

        assert structure_fingerprint(spec_no_prior) != structure_fingerprint(spec_with_prior)

    def test_prior_hyperparameters_not_structural_but_are_in_spec_fingerprint(self) -> None:
        prior_a = PriorSpec(
            target="CL",
            family=NormalPrior(mu=0.0, sigma=2.0),
            source="weakly_informative",
        )
        prior_b = PriorSpec(
            target="CL",
            family=NormalPrior(mu=0.0, sigma=5.0),
            source="weakly_informative",
        )
        spec_a = _base_spec().model_copy(update={"priors": [prior_a]})
        spec_b = _base_spec().model_copy(update={"priors": [prior_b]})

        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)
        assert spec_fingerprint(spec_a) != spec_fingerprint(spec_b)


class TestCanonicalizationDeterminism:
    def test_repeated_serialization_is_identical(self) -> None:
        spec = _base_spec()

        assert structure_fingerprint(spec) == structure_fingerprint(spec)
        assert spec_fingerprint(spec) == spec_fingerprint(spec)
        assert initial_fingerprint(spec) == initial_fingerprint(spec)
        assert justification_hash(spec) == justification_hash(spec)

    def test_variability_list_order_does_not_affect_digest(self) -> None:
        iiv_cl = IIV(params=["CL"], structure="diagonal")
        iov_cl = IOV(params=["CL"], occasions=OccasionByVisit(column="VISIT"))

        spec_order_a = DSLSpec(
            model_id="order_test_00000000001",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[iiv_cl, iov_cl],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "CL": 5.0, "V": 70.0},
        )
        spec_order_b = DSLSpec(
            model_id=spec_order_a.model_id,
            absorption=spec_order_a.absorption,
            distribution=spec_order_a.distribution,
            elimination=spec_order_a.elimination,
            variability=[iov_cl, iiv_cl],
            observation=spec_order_a.observation,
            initial=spec_order_a.initial,
        )

        assert structure_fingerprint(spec_order_a) == structure_fingerprint(spec_order_b)
        assert spec_fingerprint(spec_order_a) == spec_fingerprint(spec_order_b)

    def test_covariates_list_order_does_not_affect_digest(self) -> None:
        link_wt = CovariateLink(param="CL", covariate="WT", form="power", theta=0.75, ref=70.0)
        link_sex = CovariateLink(param="CL", covariate="SEX", form="categorical", reference="M")

        spec_order_a = _base_spec().model_copy(update={"covariates": [link_wt, link_sex]})
        spec_order_b = _base_spec().model_copy(update={"covariates": [link_sex, link_wt]})

        assert structure_fingerprint(spec_order_a) == structure_fingerprint(spec_order_b)
        assert spec_fingerprint(spec_order_a) == spec_fingerprint(spec_order_b)

    def test_priors_list_order_does_not_affect_digest(self) -> None:
        prior_cl = PriorSpec(target="CL", family=NormalPrior(mu=0.0, sigma=2.0))
        prior_v = PriorSpec(target="V", family=NormalPrior(mu=1.0, sigma=2.0))

        spec_order_a = _base_spec().model_copy(update={"priors": [prior_cl, prior_v]})
        spec_order_b = _base_spec().model_copy(update={"priors": [prior_v, prior_cl]})

        assert structure_fingerprint(spec_order_a) == structure_fingerprint(spec_order_b)
        assert spec_fingerprint(spec_order_a) == spec_fingerprint(spec_order_b)

    def test_iiv_params_order_within_item_does_not_affect_digest(self) -> None:
        spec_a = DSLSpec(
            model_id="params_order_test_0001",
            absorption=FirstOrder(),
            distribution=OneCmt(),
            elimination=LinearElim(),
            variability=[IIV(params=["CL", "V"], structure="diagonal")],
            observation=Proportional(sigma_prop=0.1),
            initial={"ka": 1.0, "CL": 5.0, "V": 70.0},
        )
        spec_b = DSLSpec(
            model_id=spec_a.model_id,
            absorption=spec_a.absorption,
            distribution=spec_a.distribution,
            elimination=spec_a.elimination,
            variability=[IIV(params=["V", "CL"], structure="diagonal")],
            observation=spec_a.observation,
            initial=spec_a.initial,
        )

        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)
        assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)

    def test_source_meta_does_not_affect_any_fingerprint(self) -> None:
        spec_no_meta = _base_spec()
        spec_with_meta = spec_no_meta.model_copy(update={"source_meta": {"absorption": (3, 7)}})

        assert structure_fingerprint(spec_no_meta) == structure_fingerprint(spec_with_meta)
        assert spec_fingerprint(spec_no_meta) == spec_fingerprint(spec_with_meta)


class TestJustificationHashIsolation:
    def test_justification_text_change_does_not_affect_spec_fingerprint(self) -> None:
        base_prior = PriorSpec(
            target="CL",
            family=NormalPrior(mu=0.0, sigma=2.0),
            source="historical_data",
            justification=(
                "Derived from a prior IV reference study in healthy volunteers "
                "with matched dosing; see internal report 2024-001."
            ),
            doi="10.1000/xyz123",
            historical_refs=["study-2024-001"],
        )
        reworded_prior = base_prior.model_copy(
            update={
                "justification": (
                    "This value comes from an earlier IV study in healthy "
                    "subjects with equivalent dosing; report 2024-001 has details."
                )
            }
        )

        spec_a = _base_spec().model_copy(update={"priors": [base_prior]})
        spec_b = _base_spec().model_copy(update={"priors": [reworded_prior]})

        assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)
        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)
        assert justification_hash(spec_a) != justification_hash(spec_b)

    def test_doi_change_does_not_affect_spec_fingerprint_but_changes_justification_hash(
        self,
    ) -> None:
        base_prior = PriorSpec(
            target="CL",
            family=NormalPrior(mu=0.0, sigma=2.0),
            source="historical_data",
            justification="A sufficiently long justification string for testing purposes here.",
            doi="10.1000/aaa111",
            historical_refs=["study-a"],
        )
        other_doi_prior = base_prior.model_copy(update={"doi": "10.1000/bbb222"})

        spec_a = _base_spec().model_copy(update={"priors": [base_prior]})
        spec_b = _base_spec().model_copy(update={"priors": [other_doi_prior]})

        assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)
        assert justification_hash(spec_a) != justification_hash(spec_b)

    def test_no_priors_produces_stable_empty_hash(self) -> None:
        spec = _base_spec()
        result = justification_hash(spec)
        assert result["schema"] == CANONICAL_SCHEMA_VERSION
        # Empty list canonicalizes deterministically regardless of caller.
        assert justification_hash(spec) == result


class TestMultiAnalyteObservationsFingerprint:
    """P1.7: DSLSpec.observations participates in both fingerprints (schema 2.2.0)."""

    def test_schema_version_bumped(self) -> None:
        assert CANONICAL_SCHEMA_VERSION == "2.3.0"

    def test_differing_observations_content_changes_both_fingerprints(self) -> None:
        base = _base_spec()
        spec_a = base.model_copy(
            update={
                "observations": {
                    "plasma": ObservationEndpoint(
                        name="plasma",
                        dvid=1,
                        prediction="C_central",
                        error=Proportional(sigma_prop=0.1),
                    ),
                    "metabolite": ObservationEndpoint(
                        name="metabolite",
                        dvid=2,
                        prediction="C_central",
                        error=Additive(sigma_add=0.2),
                    ),
                }
            }
        )
        spec_b = base.model_copy(
            update={
                "observations": {
                    "plasma": ObservationEndpoint(
                        name="plasma",
                        dvid=1,
                        prediction="C_central",
                        error=Proportional(sigma_prop=0.1),
                    ),
                    "metabolite": ObservationEndpoint(
                        name="metabolite",
                        dvid=3,  # different dvid than spec_a
                        prediction="C_central",
                        error=Additive(sigma_add=0.2),
                    ),
                }
            }
        )

        # Without the P1.7 fingerprint addition, both specs would only
        # differ in the `observations` field the pre-P1.7 hash never saw
        # -- and would incorrectly collide on both fingerprints.
        assert structure_fingerprint(spec_a) != structure_fingerprint(spec_b)
        assert spec_fingerprint(spec_a) != spec_fingerprint(spec_b)

    def test_no_observations_matches_pre_p17_spec(self) -> None:
        """A spec with observations=None fingerprints identically to a bare legacy spec."""
        spec_legacy = _base_spec()
        spec_explicit_none = _base_spec().model_copy(update={"observations": None})
        assert structure_fingerprint(spec_legacy) == structure_fingerprint(spec_explicit_none)
        assert spec_fingerprint(spec_legacy) == spec_fingerprint(spec_explicit_none)

    def test_endpoint_error_module_calibration_value_only_affects_spec_fingerprint(self) -> None:
        """Same endpoint shape, different sigma -- structure collides, spec does not."""
        base = _base_spec()
        spec_a = base.model_copy(
            update={
                "observations": {
                    "plasma": ObservationEndpoint(
                        name="plasma",
                        dvid=1,
                        prediction="C_central",
                        error=Proportional(sigma_prop=0.1),
                    ),
                }
            }
        )
        spec_b = base.model_copy(
            update={
                "observations": {
                    "plasma": ObservationEndpoint(
                        name="plasma",
                        dvid=1,
                        prediction="C_central",
                        error=Proportional(sigma_prop=0.9),
                    ),
                }
            }
        )
        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)
        assert spec_fingerprint(spec_a) != spec_fingerprint(spec_b)

    def test_combined_observations_ordering_does_not_affect_fingerprint(self) -> None:
        """observations is a semantically-unordered set, like variability/priors."""
        endpoints_forward = {
            "plasma": ObservationEndpoint(
                name="plasma", dvid=1, prediction="C_central", error=Proportional(sigma_prop=0.1)
            ),
            "metabolite": ObservationEndpoint(
                name="metabolite", dvid=2, prediction="C_central", error=Additive(sigma_add=0.2)
            ),
        }
        endpoints_reversed = {
            "metabolite": endpoints_forward["metabolite"],
            "plasma": endpoints_forward["plasma"],
        }
        spec_a = _base_spec().model_copy(update={"observations": endpoints_forward})
        spec_b = _base_spec().model_copy(update={"observations": endpoints_reversed})
        assert structure_fingerprint(spec_a) == structure_fingerprint(spec_b)
        assert spec_fingerprint(spec_a) == spec_fingerprint(spec_b)
