# SPDX-License-Identifier: GPL-2.0-or-later
"""Prior-justification structural validator (plan Task 14, FDA Gate 2).

Adds structural checks on informative priors: minimum justification length
and DOI format for provenance. Complements the existing PriorSpec
``model_validator`` (which blocks empty justification and missing
historical refs) by enforcing evidence quality, not just field presence.
"""

from __future__ import annotations

from apmode.dsl.priors import NormalPrior, PriorSpec, validate_prior_justification


def test_informative_prior_with_valid_doi_and_long_justification_passes() -> None:
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=2.0, sigma=0.3),
        source="historical_data",
        justification=(
            "Historical borrowing from Schmidli 2014 robust MAP mixture; "
            "populations deemed exchangeable on adult-only studies."
        ),
        doi="10.1111/biom.12242",
        historical_refs=["Schmidli2014"],
    )
    assert validate_prior_justification(spec) == []


def test_informative_prior_with_short_justification_fails() -> None:
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=2.0, sigma=0.3),
        source="expert_elicitation",
        justification="short note",
        doi="10.1111/biom.12242",
    )
    errors = validate_prior_justification(spec)
    assert any("50" in e or "length" in e.lower() for e in errors)


def test_informative_prior_without_doi_fails() -> None:
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=2.0, sigma=0.3),
        source="meta_analysis",
        justification=(
            "Pooled meta-analysis across five published oral PK studies with "
            "consistent dosing and population descriptors matched."
        ),
        doi=None,
    )
    errors = validate_prior_justification(spec)
    assert any("doi" in e.lower() for e in errors)


def test_informative_prior_with_malformed_doi_fails() -> None:
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=2.0, sigma=0.3),
        source="historical_data",
        justification=(
            "Borrowed structural priors from internal legacy dataset with a "
            "matched covariate distribution in adult healthy volunteers."
        ),
        doi="not-a-doi",
        historical_refs=["internal-2021"],
    )
    errors = validate_prior_justification(spec)
    assert any("doi" in e.lower() for e in errors)


def test_weakly_informative_source_is_exempt() -> None:
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=0.0, sigma=2.0),
        source="weakly_informative",
        justification="",
        doi=None,
    )
    assert validate_prior_justification(spec) == []


def test_uninformative_source_is_exempt() -> None:
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=0.0, sigma=10.0),
        source="uninformative",
        justification="",
        doi=None,
    )
    assert validate_prior_justification(spec) == []


def test_sici_bracketed_doi_accepted() -> None:
    """Wiley SICI-style DOIs carry angle/square brackets; accept them."""
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=2.0, sigma=0.3),
        source="meta_analysis",
        justification=(
            "Historical SICI-identified pooled analysis across adult oral PK "
            "studies aligned on weight-corrected clearance as the endpoint."
        ),
        doi="10.1002/(SICI)1099-081X(199601)17:1<1::AID-BDD931>3.0.CO;2-G",
    )
    assert validate_prior_justification(spec) == []


def test_min_length_override_relaxes_threshold() -> None:
    """Callers can relax the default 50-char floor through ``min_length``."""
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=2.0, sigma=0.3),
        source="meta_analysis",
        justification="Vancomycin pop-PK meta-analysis.",
        doi="10.1111/biom.12242",
    )
    # Default rejects the short justification.
    assert any("50" in e for e in validate_prior_justification(spec))
    # Relaxing to 20 accepts it.
    assert validate_prior_justification(spec, min_length=20) == []


def test_min_length_override_can_tighten_threshold() -> None:
    """Lane policies may tighten the floor above the module default."""
    spec = PriorSpec(
        target="CL",
        family=NormalPrior(mu=2.0, sigma=0.3),
        source="expert_elicitation",
        justification=(
            "Anchored elicitation across three NCS panelists with calibrated "
            "uncertainty bounds recorded in the panel workbook."
        ),
        doi="10.1111/biom.12242",
    )
    # Default (50) accepts.
    assert validate_prior_justification(spec) == []
    # Tightened (500) rejects and mentions the bespoke threshold.
    errs = validate_prior_justification(spec, min_length=500)
    assert any("500" in e for e in errs)
