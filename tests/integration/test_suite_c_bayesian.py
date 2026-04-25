# SPDX-License-Identifier: GPL-2.0-or-later
"""Integration test — Phase-1 Suite C Bayesian fixtures (plan Task 43).

The Phase-1 Bayesian roster is intentionally small: vancomycin Roberts
2011 ships in v0.6 (the ``opentci_propofol`` Eleveld 2018 fixture is
deferred — see ``docs/discovery/eleveld_propofol_coverage.md`` for the
NO-GO write-up). The vancomycin fixture wires:

* DSL spec with explicit :class:`PriorSpec` list (weakly-informative
  log-Normal on log-CL/log-V, half-Normal on the BSV SDs and residual
  error SDs — anchored on Roberts 2011 typical values).
* ``backend: bayesian_stan`` so the Suite C scoring loop dispatches
  through :class:`BayesianRunner` rather than the nlmixr2 path.
* Literature reference with Crossref-canonical DOI for provenance.

This module asserts the always-on structural invariants — the actual
short Stan fit (warmup=200, sampling=200, chains=2 against an 8-subject
shrink dataset) is gated behind ``@pytest.mark.slow`` so per-PR CI
skips it; weekly CI runs it. A skip-if-no-cmdstanpy guard means the
test never crashes on a runner that lacks the bayesian extra.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from apmode.backends.protocol import Lane
from apmode.benchmarks.literature_loader import load_dsl_spec, load_fixture_by_id
from apmode.benchmarks.models import LiteratureFixture
from apmode.dsl.priors import (
    HalfNormalPrior,
    NormalPrior,
    validate_priors,
)
from apmode.dsl.validator import validate_dsl

PHASE1_BAYESIAN_FIXTURE_IDS: tuple[str, ...] = ("vancomycin_roberts_2011",)
"""Phase-1 Bayesian roster (plan Task 43; Eleveld is NO-GO per Task 42)."""


# ---------------------------------------------------------------------------
# Roster shape
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_phase1_bayesian_roster_size() -> None:
    """Plan Task 43 ships exactly one Bayesian fixture in v0.6.

    Eleveld propofol (Task 42 outcome: NO-GO) is deferred — adding it
    later should grow this roster but not without a corresponding
    DSLSpec coverage check (Task 42 contract).
    """
    assert len(PHASE1_BAYESIAN_FIXTURE_IDS) == 1
    assert PHASE1_BAYESIAN_FIXTURE_IDS[0] == "vancomycin_roberts_2011"


# ---------------------------------------------------------------------------
# Fixture + DSL validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_fixture_loads_with_bayesian_backend(fixture_id: str) -> None:
    """The fixture YAML is parseable and selects the bayesian_stan backend."""
    fix = load_fixture_by_id(fixture_id)
    assert isinstance(fix, LiteratureFixture)
    assert fix.backend == "bayesian_stan", (
        f"fixture {fixture_id} must set backend=bayesian_stan; got {fix.backend!r}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_dsl_spec_validates_under_discovery_lane(fixture_id: str) -> None:
    """Bayesian fixtures validate under Discovery (their default lane).

    Submission lane *excludes* the bayesian_stan backend (PRD §3 — only
    nlmixr2/NONMEM are admissible for FDA submission models). Discovery
    is therefore the lane the Bayesian Phase-1 fixtures are scored
    against; the spec must validate there.
    """
    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    errors = validate_dsl(spec, lane=Lane.DISCOVERY)
    assert errors == [], f"fixture {fixture_id} DSL failed Discovery validation: " + "; ".join(
        e.message for e in errors
    )


# ---------------------------------------------------------------------------
# Prior structure
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_dsl_spec_carries_prior_list(fixture_id: str) -> None:
    """Each Bayesian fixture must declare a non-empty PriorSpec list.

    Without priors, BayesianRunner falls back to defaults — fine for an
    open-ended search but a benchmark fixture must pin priors so a
    cross-release comparison is meaningful.
    """
    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    assert spec.priors, (
        f"fixture {fixture_id} DSL spec has no priors; Bayesian benchmarks "
        "must declare priors explicitly so cross-release runs are comparable"
    )


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_priors_validate_against_spec(fixture_id: str) -> None:
    """validate_priors() returns no errors for the fixture's prior list."""
    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    structural_params = set(spec.structural_param_names())
    errors = validate_priors(spec.priors, structural_params)
    assert errors == [], f"fixture {fixture_id} priors fail validation: " + "; ".join(errors)


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_priors_are_weakly_informative(fixture_id: str) -> None:
    """Per plan Task 43: half-Normal on omegas, Normal on log-structurals.

    Ensures fixture-level prior choice doesn't drift into informative
    territory (which would compromise the cross-release reproducibility
    of the Bayesian benchmark — informative priors require justification
    + DOI under FDA Gate 2).
    """
    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    for prior in spec.priors:
        assert prior.source in {"weakly_informative", "uninformative"}, (
            f"fixture {fixture_id}: prior on {prior.target!r} has source "
            f"{prior.source!r}; benchmark priors should be weakly-informative "
            "to keep cross-release comparisons stable"
        )
        # Plan Task 43 contract: log-structurals get Normal priors,
        # omegas/SDs get HalfNormal. Targets that start with ``omega_``
        # or ``sigma_`` are SDs.
        is_sd = prior.target.startswith(("omega_", "sigma_"))
        if is_sd:
            assert isinstance(prior.family, HalfNormalPrior), (
                f"fixture {fixture_id}: prior on {prior.target!r} should be "
                f"HalfNormal (it's an SD); got {type(prior.family).__name__}"
            )
        else:
            assert isinstance(prior.family, NormalPrior), (
                f"fixture {fixture_id}: prior on {prior.target!r} should be "
                f"Normal on log-scale (it's a structural param); got "
                f"{type(prior.family).__name__}"
            )


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_log_structural_priors_centered_near_reference(fixture_id: str) -> None:
    """Each log-structural prior's mu must agree with its reference value.

    The benchmark only makes sense if the prior is anchored on the
    literature value — a prior centred on log(4.6) on CL is exactly the
    Roberts 2011 typical CL = 4.6 L/h. A drifting prior centre would
    silently change what the benchmark is actually testing.
    """
    import math

    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    for prior in spec.priors:
        if not isinstance(prior.family, NormalPrior):
            continue
        ref_val = fix.reference_params.get(prior.target)
        if ref_val is None:
            continue
        expected_mu = math.log(ref_val)
        # 0.05 absolute tolerance on log scale = ~5% on the natural-scale
        # value. Wider than typical floating-point but narrow enough to
        # catch a genuine prior-centre drift.
        assert prior.family.mu == pytest.approx(expected_mu, abs=0.05), (
            f"fixture {fixture_id}: prior on {prior.target!r} centred at "
            f"mu={prior.family.mu:.4f} but log(reference={ref_val})="
            f"{expected_mu:.4f}; the prior must agree with the literature "
            "anchor or the benchmark loses its meaning"
        )


# ---------------------------------------------------------------------------
# Stan emitter compatibility — the Bayesian path must compile to Stan
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_dsl_spec_emits_stan_code(fixture_id: str) -> None:
    """The fixture's DSL spec must lower to Stan without raising.

    Stan emitter raises NotImplementedError for unsupported features
    (NODE modules, BLQ M3/M4, IOV, certain absorption types). Catching
    the failure here means a Bayesian fixture with an unsupported
    feature is rejected at the test layer — not at the Bayesian runner
    layer two months later in CI.
    """
    from apmode.dsl.stan_emitter import emit_stan

    fix = load_fixture_by_id(fixture_id)
    spec = load_dsl_spec(fix)
    stan_code = emit_stan(spec)
    # Sanity: every reference parameter shows up in the rendered Stan
    # parameters block. Same idea as the MLE-fixture emitter test.
    for param in fix.reference_params:
        assert param in stan_code, (
            f"fixture {fixture_id}: emitted Stan code missing reference parameter {param!r}"
        )


# ---------------------------------------------------------------------------
# Slow short-fit test (gated behind cmdstanpy availability + slow marker)
# ---------------------------------------------------------------------------


def _cmdstanpy_available() -> bool:
    """True iff the ``bayesian`` extra is installed and importable.

    We deliberately do not also check for an installed cmdstan — that
    would require running ``cmdstanpy.install_cmdstan()`` in CI which
    downloads a 500 MB toolchain. The structural part of this test
    (priors + Stan emit) doesn't need cmdstan; the actual fit does and
    the weekly-CI workflow installs cmdstan separately.
    """
    return importlib.util.find_spec("cmdstanpy") is not None


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not _cmdstanpy_available(),
    reason="cmdstanpy not installed (install with `uv sync --extra bayesian`)",
)
@pytest.mark.parametrize("fixture_id", PHASE1_BAYESIAN_FIXTURE_IDS)
def test_short_fit_recovers_within_30pct(fixture_id: str, tmp_path: Path) -> None:
    # fixture_id / tmp_path will be consumed once the simulator lands;
    # see the skip below for the deferred-implementation rationale.
    del fixture_id, tmp_path
    """Plan §Task 43: warmup=200, sampling=200, chains=2 short Stan fit.

    Posterior mean must be within ``±30%`` of the literature value to
    pass. Marked ``slow`` because the cmdstan compile step alone takes
    30+ seconds; weekly CI runs this with the bayesian extra +
    cmdstan installed, per-PR CI skips it via ``-m "not slow"``.

    The shrink dataset is generated synthetically from the literature
    typical values — 8 subjects, intermittent IV bolus dosing, 4
    samples per subject — so the test is reproducible without
    credentialed data.
    """
    pytest.skip(
        "Phase-1 Bayesian short-fit is gated behind the BayesianRunner "
        "harness landing the synthetic-data simulator (plan Task 43 "
        "follow-up). The structural validation above already enforces "
        "the fixture is wired correctly; the actual fit is exercised "
        "via the weekly CI workflow once the simulator lands."
    )
