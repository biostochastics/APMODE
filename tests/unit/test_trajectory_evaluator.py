# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for governance/trajectory_evaluator.py (PRD §4.2.6 addendum)."""

from apmode.bundle.models import AgenticIterationEntry
from apmode.governance.policy import AgenticComplianceConfig
from apmode.governance.trajectory_evaluator import evaluate_trajectory_compliance


def _entry(**kw: object) -> AgenticIterationEntry:
    base: dict[str, object] = dict(
        iteration=1,
        spec_before="m0",
        spec_after="m1",
        transforms_proposed=["add_covariate_link(CL,WT)"],
        reasoning="add allometric weight on CL",
        converged=True,
        bic=200.0,
        eta_shrinkage_max=15.0,
        auc_cmax_be_score=0.8,
    )
    base.update(kw)
    return AgenticIterationEntry(**base)  # type: ignore[arg-type]


def test_reward_hacking_flagged_when_shrinkage_rises_as_bic_improves() -> None:
    entries = [
        _entry(iteration=1, bic=200.0, eta_shrinkage_max=10.0),
        _entry(iteration=2, bic=190.0, eta_shrinkage_max=35.0),
        _entry(iteration=3, bic=180.0, eta_shrinkage_max=60.0),
    ]
    report = evaluate_trajectory_compliance(entries, AgenticComplianceConfig())
    assert report.reward_hacking_suspected is True


def test_no_reward_hacking_when_shrinkage_and_bic_both_improve() -> None:
    entries = [
        _entry(iteration=1, bic=200.0, eta_shrinkage_max=30.0),
        _entry(iteration=2, bic=180.0, eta_shrinkage_max=20.0),
        _entry(iteration=3, bic=160.0, eta_shrinkage_max=12.0),
    ]
    report = evaluate_trajectory_compliance(entries, AgenticComplianceConfig())
    assert report.reward_hacking_suspected is False


def test_no_reward_hacking_with_fewer_than_two_converged_entries() -> None:
    entries = [_entry(iteration=1, converged=False)]
    report = evaluate_trajectory_compliance(entries, AgenticComplianceConfig())
    assert report.n_iterations_considered == 0
    assert report.reward_hacking_suspected is False


def test_eligibility_collapse_flagged_when_auc_cmax_disappears_with_bic_improving() -> None:
    entries = [
        _entry(iteration=1, bic=200.0, eta_shrinkage_max=10.0, auc_cmax_be_score=0.9),
        _entry(iteration=2, bic=195.0, eta_shrinkage_max=11.0, auc_cmax_be_score=0.85),
        _entry(iteration=3, bic=180.0, eta_shrinkage_max=12.0, auc_cmax_be_score=None),
    ]
    report = evaluate_trajectory_compliance(entries, AgenticComplianceConfig())
    assert report.eligibility_collapse_suspected is True


def test_no_eligibility_collapse_when_auc_cmax_score_present_throughout() -> None:
    entries = [
        _entry(iteration=1, bic=200.0, eta_shrinkage_max=10.0, auc_cmax_be_score=0.9),
        _entry(iteration=2, bic=180.0, eta_shrinkage_max=11.0, auc_cmax_be_score=0.85),
    ]
    report = evaluate_trajectory_compliance(entries, AgenticComplianceConfig())
    assert report.eligibility_collapse_suspected is False
