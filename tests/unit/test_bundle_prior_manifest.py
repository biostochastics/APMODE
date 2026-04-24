# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the prior-manifest emitter (plan Task 15).

``BundleEmitter.write_prior_manifest_from_specs`` is the atomic path from
``DSLSpec.priors`` to ``bayesian/{candidate_id}_prior_manifest.json``. It
must:

* Run :func:`apmode.dsl.priors.validate_prior_justification` on every
  informative spec (length + DOI) and raise ``ValueError`` with an
  aggregated message when any spec fails — no partial writes.
* Accept weakly-informative / uninformative priors without justification
  or DOI.
* Round-trip ``PriorSpec`` → ``PriorManifestEntry``, preserving ``doi``
  and ``historical_refs`` and flattening the discriminated-union
  ``PriorFamily`` into the primitive ``hyperparams`` shape.
* Respect a caller-supplied ``justification_min_length`` (so Task 19's
  submission-lane tightening is a parameter bump, not a code edit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apmode.bundle.emitter import (
    BundleEmitter,
    _prior_family_hyperparams,
    _prior_spec_to_manifest_entry,
    build_prior_manifest,
)
from apmode.bundle.models import PriorManifest
from apmode.dsl.priors import (
    HalfCauchyPrior,
    HistoricalBorrowingPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
    PriorSpec,
)

# A plausible Crossref DOI (Beal/Sheiner NONMEM extended; arbitrary choice)
_VALID_DOI = "10.1007/BF01064740"
# >= 50 characters so the default justification_min_length passes
_VALID_JUSTIFICATION = (
    "Allometric exponent anchored to Holford 1996 meta-analysis; posterior "
    "mean 0.75, half-width 0.1 on the log scale."
)


def _emitter(tmp_path: Path) -> BundleEmitter:
    em = BundleEmitter(base_dir=tmp_path, run_id="run_priors")
    em.initialize()
    return em


# --- Happy path ----------------------------------------------------------


def test_round_trip_weakly_informative_writes_file(tmp_path: Path) -> None:
    em = _emitter(tmp_path)
    specs = [
        PriorSpec(target="CL", family=NormalPrior(mu=0.0, sigma=2.0)),
        PriorSpec(target="omega_CL", family=HalfCauchyPrior(scale=1.0)),
    ]
    path = em.write_prior_manifest_from_specs(
        specs,
        candidate_id="cand001",
        policy_version="0.5.1",
    )
    assert path.name == "cand001_prior_manifest.json"
    body = json.loads(path.read_text())
    assert body["policy_version"] == "0.5.1"
    assert body["default_prior_policy"] == "weakly_informative"
    assert len(body["entries"]) == 2
    families = {e["family"] for e in body["entries"]}
    assert families == {"Normal", "HalfCauchy"}
    # Weakly-informative priors persist without DOI
    assert all(e["doi"] is None for e in body["entries"])


def test_informative_prior_with_valid_doi_and_justification_roundtrips(
    tmp_path: Path,
) -> None:
    em = _emitter(tmp_path)
    spec = PriorSpec(
        target="CL",
        family=LogNormalPrior(mu=2.3, sigma=0.3),
        source="meta_analysis",
        justification=_VALID_JUSTIFICATION,
        doi=_VALID_DOI,
    )
    path = em.write_prior_manifest_from_specs(
        [spec],
        candidate_id="cand042",
        policy_version="0.5.1",
    )
    body = json.loads(path.read_text())
    entry = body["entries"][0]
    assert entry["source"] == "meta_analysis"
    assert entry["doi"] == _VALID_DOI
    assert entry["hyperparams"] == {"mu": 2.3, "sigma": 0.3}
    assert entry["justification"].startswith("Allometric exponent")


def test_historical_borrowing_retains_refs(tmp_path: Path) -> None:
    em = _emitter(tmp_path)
    hb = HistoricalBorrowingPrior(
        map_mean=1.1,
        map_sd=0.2,
        robust_weight=0.2,
        historical_refs=["study_A_n48", "study_B_n72"],
    )
    spec = PriorSpec(
        target="CL",
        family=hb,
        source="historical_data",
        justification=_VALID_JUSTIFICATION,
        doi=_VALID_DOI,
        historical_refs=["study_A_n48", "study_B_n72"],
    )
    path = em.write_prior_manifest_from_specs(
        [spec],
        candidate_id="cand_hb",
        policy_version="0.5.1",
    )
    body = json.loads(path.read_text())
    entry = body["entries"][0]
    assert entry["historical_refs"] == ["study_A_n48", "study_B_n72"]
    assert entry["hyperparams"]["map_mean"] == 1.1
    assert entry["hyperparams"]["map_sd"] == 0.2
    assert entry["hyperparams"]["robust_weight"] == 0.2
    assert entry["hyperparams"]["historical_refs"] == ["study_A_n48", "study_B_n72"]


# --- Validation failures -------------------------------------------------


def test_short_justification_raises_and_writes_nothing(tmp_path: Path) -> None:
    em = _emitter(tmp_path)
    spec = PriorSpec(
        target="CL",
        family=LogNormalPrior(mu=2.3, sigma=0.3),
        source="meta_analysis",
        justification="too short",  # length 9 < 50
        doi=_VALID_DOI,
    )
    with pytest.raises(ValueError, match=r"justification of length >= 50"):
        em.write_prior_manifest_from_specs(
            [spec],
            candidate_id="cand_short",
            policy_version="0.5.1",
        )
    # Crucially: no partial write
    assert not (em.run_dir / "bayesian" / "cand_short_prior_manifest.json").exists()


def test_missing_doi_on_informative_prior_raises(tmp_path: Path) -> None:
    em = _emitter(tmp_path)
    spec = PriorSpec(
        target="CL",
        family=LogNormalPrior(mu=2.3, sigma=0.3),
        source="expert_elicitation",
        justification=_VALID_JUSTIFICATION,
        doi=None,
    )
    with pytest.raises(ValueError, match=r"requires a valid DOI"):
        em.write_prior_manifest_from_specs(
            [spec],
            candidate_id="cand_nodoi",
            policy_version="0.5.1",
        )


def test_multiple_errors_are_aggregated(tmp_path: Path) -> None:
    em = _emitter(tmp_path)
    specs = [
        PriorSpec(
            target="CL",
            family=LogNormalPrior(mu=2.3, sigma=0.3),
            source="meta_analysis",
            justification="too short",
            doi="not-a-doi",
        ),
        PriorSpec(
            target="V",
            family=NormalPrior(mu=0.0, sigma=2.0),
            source="expert_elicitation",
            justification=_VALID_JUSTIFICATION,
            doi=None,
        ),
    ]
    with pytest.raises(ValueError) as exc_info:
        em.write_prior_manifest_from_specs(
            specs,
            candidate_id="cand_multi",
            policy_version="0.5.1",
        )
    msg = str(exc_info.value)
    # Both offending priors must appear in the aggregated message
    assert "priors[0]" in msg
    assert "priors[1]" in msg
    assert "'CL'" in msg
    assert "'V'" in msg


def test_min_length_override_tightens_threshold(tmp_path: Path) -> None:
    """Submission lane will bump the floor to 500 (Task 19) — verify the knob works."""
    em = _emitter(tmp_path)
    spec = PriorSpec(
        target="CL",
        family=LogNormalPrior(mu=2.3, sigma=0.3),
        source="meta_analysis",
        justification=_VALID_JUSTIFICATION,  # length ~130, passes at default 50
        doi=_VALID_DOI,
    )
    # Default floor (50) accepts
    em.write_prior_manifest_from_specs(
        [spec],
        candidate_id="cand_default",
        policy_version="0.5.1",
    )
    # Tightened floor (500) rejects the same spec
    with pytest.raises(ValueError, match=r"length >= 500"):
        em.write_prior_manifest_from_specs(
            [spec],
            candidate_id="cand_strict",
            policy_version="0.5.1",
            justification_min_length=500,
        )


# --- Helper-level behaviour (unit-scoped, no filesystem) -----------------


def test_prior_family_hyperparams_flattens_mixture() -> None:
    mix = MixturePrior(
        components=[
            LogNormalPrior(mu=2.3, sigma=0.3),
            NormalPrior(mu=0.0, sigma=2.0),
        ],
        weights=[0.8, 0.2],
    )
    flat = _prior_family_hyperparams(
        PriorSpec(target="CL", family=mix),
    )
    # Weights survive as list[float]
    assert flat["weights"] == [0.8, 0.2]
    # Components stringified to JSON so the primitive schema holds
    components = flat["components"]
    assert isinstance(components, list)
    for blob in components:
        assert isinstance(blob, str)
        parsed = json.loads(blob)
        assert parsed["type"] in {"LogNormal", "Normal"}


def test_build_prior_manifest_returns_model() -> None:
    """The free builder is the unit callers use outside the emitter (Task 19)."""
    specs = [PriorSpec(target="CL", family=NormalPrior(mu=0.0, sigma=2.0))]
    manifest = build_prior_manifest(specs, policy_version="0.5.1")
    assert isinstance(manifest, PriorManifest)
    assert manifest.policy_version == "0.5.1"
    assert manifest.entries[0].family == "Normal"


def test_spec_to_entry_preserves_doi_and_source() -> None:
    spec = PriorSpec(
        target="CL",
        family=LogNormalPrior(mu=2.3, sigma=0.3),
        source="meta_analysis",
        justification=_VALID_JUSTIFICATION,
        doi=_VALID_DOI,
    )
    entry = _prior_spec_to_manifest_entry(spec)
    assert entry.source == "meta_analysis"
    assert entry.doi == _VALID_DOI
    assert entry.family == "LogNormal"
    assert entry.hyperparams == {"mu": 2.3, "sigma": 0.3}
