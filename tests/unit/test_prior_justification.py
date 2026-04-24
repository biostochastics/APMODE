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
