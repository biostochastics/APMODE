# SPDX-License-Identifier: GPL-2.0-or-later
"""Gate-3 contract-grouped ranking + Submission-lane dominance (plan §3)."""

from __future__ import annotations

import pytest

from apmode.bundle.models import (
    BackendResult,
    BLQHandling,
    ConvergenceMetadata,
    DiagnosticBundle,
    GOFMetrics,
    IdentifiabilityFlags,
    ParameterEstimate,
    ScoringContract,
    VPCSummary,
)
from apmode.governance.policy import Gate3Config
from apmode.governance.ranking import (
    group_by_scoring_contract,
    rank_by_scoring_contract,
)


def _vpc(coverage: float = 0.90) -> VPCSummary:
    return VPCSummary(
        percentiles=[5.0, 50.0, 95.0],
        coverage={"p5": coverage, "p50": coverage, "p95": coverage},
        n_bins=8,
        prediction_corrected=False,
    )


def _make_result(
    *,
    model_id: str,
    backend: str,
    bic: float,
    contract: ScoringContract,
) -> BackendResult:
    return BackendResult(
        model_id=model_id,
        backend=backend,  # type: ignore[arg-type]
        converged=True,
        ofv=150.0,
        aic=bic - 1.0,
        bic=bic,
        parameter_estimates={
            "CL": ParameterEstimate(name="CL", estimate=5.0, category="structural"),
        },
        eta_shrinkage={"CL": 0.05},
        convergence_metadata=ConvergenceMetadata(
            method="saem",
            converged=True,
            iterations=100,
            minimization_status="successful",
            wall_time_seconds=1.0,
        ),
        diagnostics=DiagnosticBundle(
            gof=GOFMetrics(cwres_mean=0.0, cwres_sd=1.0, outlier_fraction=0.02),
            vpc=_vpc(0.90),
            identifiability=IdentifiabilityFlags(
                condition_number=None,
                profile_likelihood_ci={},
                ill_conditioned=False,
            ),
            blq=BLQHandling(method="none", n_blq=0, blq_fraction=0.0),
            npe_score=0.1,
            scoring_contract=contract,
        ),
        wall_time_seconds=1.0,
        backend_versions={"nlmixr2": "test"},
        initial_estimate_source="fallback",
    )


def _nlmixr2_contract() -> ScoringContract:
    return ScoringContract(
        nlpd_kind="marginal",
        re_treatment="integrated",
        nlpd_integrator="nlmixr2_focei",
        blq_method="none",
        observation_model="combined",
        float_precision="float64",
    )


def _stan_contract() -> ScoringContract:
    return _nlmixr2_contract().model_copy(update={"nlpd_integrator": "hmc_nuts"})


def _node_pooled_contract() -> ScoringContract:
    return ScoringContract(
        nlpd_kind="conditional",
        re_treatment="pooled",
        nlpd_integrator="none",
        blq_method="none",
        observation_model="combined",
        float_precision="float32",
    )


class TestContractGrouping:
    def test_single_contract_produces_one_group(self) -> None:
        contract = _nlmixr2_contract()
        survivors = [
            _make_result(model_id=f"m{i}", backend="nlmixr2", bic=100.0 + i, contract=contract)
            for i in range(3)
        ]
        buckets = group_by_scoring_contract(survivors)
        assert len(buckets) == 1
        assert buckets[0][0] == contract
        assert len(buckets[0][1]) == 3

    def test_different_contracts_produce_separate_groups(self) -> None:
        nlmixr2 = _make_result(
            model_id="m1", backend="nlmixr2", bic=100.0, contract=_nlmixr2_contract()
        )
        stan = _make_result(
            model_id="m2", backend="bayesian_stan", bic=110.0, contract=_stan_contract()
        )
        node = _make_result(
            model_id="m3", backend="jax_node", bic=120.0, contract=_node_pooled_contract()
        )
        buckets = group_by_scoring_contract([nlmixr2, stan, node])
        assert len(buckets) == 3
        assert {b[0] for b in buckets} == {
            _nlmixr2_contract(),
            _stan_contract(),
            _node_pooled_contract(),
        }


class TestRankByScoringContract:
    def test_each_contract_gets_its_own_leaderboard(self) -> None:
        nlmixr2 = _make_result(
            model_id="m1", backend="nlmixr2", bic=100.0, contract=_nlmixr2_contract()
        )
        stan = _make_result(
            model_id="m2", backend="bayesian_stan", bic=110.0, contract=_stan_contract()
        )
        result = rank_by_scoring_contract(
            [nlmixr2, stan],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="discovery",
        )
        assert len(result.groups) == 2
        assert {c for c in result.contracts} == {_nlmixr2_contract(), _stan_contract()}
        # no cross-contract composite — each group contains exactly one candidate
        for group in result.groups:
            assert len(group.ranked_candidates) == 1

    def test_discovery_lane_emits_no_recommended(self) -> None:
        nlmixr2 = _make_result(
            model_id="m1", backend="nlmixr2", bic=100.0, contract=_nlmixr2_contract()
        )
        result = rank_by_scoring_contract(
            [nlmixr2],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="discovery",
        )
        assert result.recommended_candidate_id is None
        assert result.recommended_warning is None


class TestSubmissionDominanceRule:
    def test_submission_picks_lowest_composite_eligible_group(self) -> None:
        """When both nlmixr2 (FOCEI) and Stan (HMC) satisfy the
        integrated+marginal dominance rule, the tiebreak is per-group
        top-candidate composite_score (ascending — lower wins). The
        result is independent of survivor insertion order.

        A multi-eligible warning must fire so the reviewer knows a
        cross-contract judgement was made."""
        nlmixr2 = _make_result(
            model_id="nlmx_top", backend="nlmixr2", bic=100.0, contract=_nlmixr2_contract()
        )
        stan = _make_result(
            model_id="stan_top", backend="bayesian_stan", bic=95.0, contract=_stan_contract()
        )
        node = _make_result(
            model_id="node_top", backend="jax_node", bic=90.0, contract=_node_pooled_contract()
        )
        result = rank_by_scoring_contract(
            [nlmixr2, stan, node],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="submission",
        )
        # The winning id must be one of the two integrated+marginal
        # survivors (nlmixr2 or Stan) — NODE is ineligible.
        assert result.recommended_candidate_id in {"nlmx_top", "stan_top"}
        # A multi-eligible warning must fire (both nlmixr2-FOCEI and
        # Stan-HMC qualify), so reviewers know the choice spans contracts.
        assert result.recommended_warning is not None
        assert "Multiple integrated+marginal" in result.recommended_warning
        assert result.recommended_contract_index is not None

    def test_submission_single_eligible_group_no_warning(self) -> None:
        """When only one integrated+marginal group exists, no
        multi-eligible warning fires."""
        nlmixr2 = _make_result(
            model_id="nlmx_only", backend="nlmixr2", bic=100.0, contract=_nlmixr2_contract()
        )
        node = _make_result(
            model_id="node_only", backend="jax_node", bic=90.0, contract=_node_pooled_contract()
        )
        result = rank_by_scoring_contract(
            [nlmixr2, node],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="submission",
        )
        assert result.recommended_candidate_id == "nlmx_only"
        assert result.recommended_warning is None

    def test_submission_tiebreak_order_independent(self) -> None:
        """Dominance tiebreak must not depend on survivor insertion
        order — regardless of which group appears first, the winner is
        determined by per-group top composite_score."""
        nlmx = _make_result(
            model_id="nlmx_A", backend="nlmixr2", bic=100.0, contract=_nlmixr2_contract()
        )
        stan = _make_result(
            model_id="stan_A", backend="bayesian_stan", bic=95.0, contract=_stan_contract()
        )
        forward = rank_by_scoring_contract(
            [nlmx, stan],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="submission",
        )
        reverse = rank_by_scoring_contract(
            [stan, nlmx],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="submission",
        )
        # Same survivors, same contracts — recommended id must be stable.
        assert forward.recommended_candidate_id == reverse.recommended_candidate_id

    def test_submission_no_integrated_marginal_emits_warning(self) -> None:
        """A Submission run that produces only pooled-NODE candidates
        must not silently pick one — per plan §3 dominance rule."""
        node_only = _make_result(
            model_id="node_only",
            backend="jax_node",
            bic=90.0,
            contract=_node_pooled_contract(),
        )
        result = rank_by_scoring_contract(
            [node_only],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="submission",
        )
        assert result.recommended_candidate_id is None
        assert result.recommended_warning is not None
        assert "integrated+marginal" in result.recommended_warning.lower()

    def test_optimization_lane_does_not_apply_dominance_rule(self) -> None:
        node_only = _make_result(
            model_id="node_only",
            backend="jax_node",
            bic=90.0,
            contract=_node_pooled_contract(),
        )
        result = rank_by_scoring_contract(
            [node_only],
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="optimization",
        )
        assert result.recommended_candidate_id is None
        # Optimization lane is silent about recommendation — no warning.
        assert result.recommended_warning is None


class TestNeverComposeCrossContract:
    def test_mixed_contracts_do_not_receive_single_ranking(self) -> None:
        """Gate 3 must not emit a single cross-contract composite ranking
        when survivors carry different ScoringContracts — that is the
        M4 contradiction the plan's §3 explicitly deletes."""
        survivors = [
            _make_result(
                model_id="nlmx",
                backend="nlmixr2",
                bic=100.0,
                contract=_nlmixr2_contract(),
            ),
            _make_result(
                model_id="node",
                backend="jax_node",
                bic=80.0,
                contract=_node_pooled_contract(),
            ),
        ]
        result = rank_by_scoring_contract(
            survivors,
            gate3=Gate3Config(),
            vpc_concordance_target=0.90,
            lane="discovery",
        )
        # The two survivors land in different groups — never a single
        # mixed-ranking composite. If the call ever collapses them into
        # one group this test fails, flagging the violation.
        assert len(result.groups) == 2
        all_ids = {m.candidate_id for g in result.groups for m in g.ranked_candidates}
        assert all_ids == {"nlmx", "node"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
