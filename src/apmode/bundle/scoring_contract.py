# SPDX-License-Identifier: GPL-2.0-or-later
"""Derive per-candidate :class:`ScoringContract` from spec + backend + BLQ method.

Plan §3 (:file:`.plans/v0.5.0_limitations_closure.md`) defines the contract.
Every backend runner calls :func:`derive_scoring_contract` with its
finished :class:`BackendResult` and the source :class:`DSLSpec`; the
helper attaches the contract onto the nested
:class:`~apmode.bundle.models.DiagnosticBundle`.

Gate-3 ranking groups survivors by **exact** contract equality, so the
semantics of each field matter — see :mod:`apmode.governance.ranking`.

The helper is intentionally defensive: NODE candidates without random
effects land with ``re_treatment="pooled"`` + ``nlpd_integrator="none"``,
which makes them incomparable (by design) with nlmixr2's FOCEI-integrated
or Stan's HMC-marginal survivors. This is the behaviour §3 requires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from apmode.bundle.models import BackendResult, ScoringContract

if TYPE_CHECKING:
    from apmode.dsl.ast_models import DSLSpec


ObservationModelValue = Literal["additive", "proportional", "combined"]


def _obs_from_spec(spec: DSLSpec) -> ObservationModelValue:
    """Map ``spec.observation`` to a ScoringContract observation-model tag.

    BLQ-wrapped observation modules (BLQM3/BLQM4) carry their residual
    class on ``error_model``. Unwrap them so the scoring contract reflects
    the residual family, not the BLQ wrapper.

    Raises ``ValueError`` on an unrecognized observation type — silent
    defaulting to ``"combined"`` was a consensus-flagged misclassification
    risk (adding e.g. a future BLQM7+ wrapper without updating this helper
    would otherwise silently tag it as ``combined`` and change Gate-3
    grouping). Fail loud here so the contract-derivation site is fixed
    the moment a new observation type lands.
    """
    from apmode.dsl.ast_models import BLQM3, BLQM4, Additive, Combined, Proportional

    obs = spec.observation
    if isinstance(obs, (BLQM3, BLQM4)):
        em = obs.error_model
        if em == "additive":
            return "additive"
        if em == "proportional":
            return "proportional"
        if em == "combined":
            return "combined"
        msg = (
            f"Unknown BLQ error_model {em!r} in {type(obs).__name__}; "
            f"cannot derive ScoringContract.observation_model. Add the "
            f"mapping in apmode.bundle.scoring_contract._obs_from_spec."
        )
        raise ValueError(msg)
    if isinstance(obs, Additive):
        return "additive"
    if isinstance(obs, Proportional):
        return "proportional"
    if isinstance(obs, Combined):
        return "combined"
    msg = (
        f"Unknown observation type {type(obs).__name__!r}; cannot derive "
        f"ScoringContract.observation_model. Add the isinstance branch in "
        f"apmode.bundle.scoring_contract._obs_from_spec."
    )
    raise ValueError(msg)


def _contract_for_backend(
    backend: Literal["nlmixr2", "jax_node", "agentic_llm", "bayesian_stan"],
    *,
    re_treatment_override: Literal["integrated", "conditional_ebe", "pooled"] | None = None,
    nlpd_integrator_override: Literal[
        "nlmixr2_focei", "laplace_blockdiag", "laplace_diag", "hmc_nuts", "none"
    ]
    | None = None,
) -> tuple[
    Literal["conditional", "marginal"],
    Literal["integrated", "conditional_ebe", "pooled"],
    Literal["nlmixr2_focei", "laplace_blockdiag", "laplace_diag", "hmc_nuts", "none"],
    Literal["float32", "float64"],
]:
    """Return the tuple (nlpd_kind, re_treatment, nlpd_integrator, float_precision).

    Overrides let M3 (NODE Laplace) tighten ``re_treatment`` to
    ``conditional_ebe`` and ``nlpd_integrator`` to ``laplace_diag`` /
    ``laplace_blockdiag`` without changing the backend string.
    """
    if backend == "nlmixr2":
        return ("marginal", "integrated", "nlmixr2_focei", "float64")
    if backend == "bayesian_stan":
        return ("marginal", "integrated", "hmc_nuts", "float64")
    if backend == "jax_node":
        re_treatment = re_treatment_override or "pooled"
        integrator = nlpd_integrator_override or (
            "none" if re_treatment == "pooled" else "laplace_diag"
        )
        return ("conditional", re_treatment, integrator, "float32")
    if backend == "agentic_llm":
        # Agentic is a meta-backend: it fits via an inner nlmixr2 runner in
        # the v0.5.0 scope. Contract mirrors nlmixr2 so Gate-3 grouping
        # treats agentic candidates as comparable to classical survivors.
        # PRD §3 separately blocks agentic from the Submission "recommended"
        # slot, so this does not widen regulatory exposure.
        return ("marginal", "integrated", "nlmixr2_focei", "float64")
    raise ValueError(f"Unknown backend: {backend!r}")


def derive_scoring_contract(
    result: BackendResult,
    spec: DSLSpec,
    *,
    re_treatment_override: Literal["integrated", "conditional_ebe", "pooled"] | None = None,
    nlpd_integrator_override: Literal[
        "nlmixr2_focei", "laplace_blockdiag", "laplace_diag", "hmc_nuts", "none"
    ]
    | None = None,
) -> ScoringContract:
    """Build a ScoringContract for *result* from its backend + spec + BLQ.

    The Submission-lane dominance rule in :mod:`apmode.governance.ranking`
    requires ``nlpd_kind='marginal'`` AND ``re_treatment='integrated'`` for a
    candidate to be eligible as ``recommended``. This helper encodes that
    policy per-backend — see :func:`_contract_for_backend`.
    """
    nlpd_kind, re_treatment, nlpd_integrator, float_precision = _contract_for_backend(
        result.backend,
        re_treatment_override=re_treatment_override,
        nlpd_integrator_override=nlpd_integrator_override,
    )
    return ScoringContract(
        nlpd_kind=nlpd_kind,
        re_treatment=re_treatment,
        nlpd_integrator=nlpd_integrator,
        blq_method=result.diagnostics.blq.method,
        observation_model=_obs_from_spec(spec),
        float_precision=float_precision,
    )


def attach_scoring_contract(result: BackendResult, spec: DSLSpec) -> BackendResult:
    """Attach a derived :class:`ScoringContract` onto ``result.diagnostics``.

    Returns the same :class:`BackendResult` (mutated in place) for call-site
    ergonomics. DiagnosticBundle is intentionally mutable (see its docstring)
    so direct assignment is safe — but tests that freeze the surrounding
    object should pass the contract via ``derive_scoring_contract`` + a
    fresh construction instead.

    Drift guard: if the current contract disagrees with a freshly derived
    one on any field other than the classical default (i.e., the runner
    has already attached a non-default contract that differs from what
    ``derive_scoring_contract`` now produces), raise ``ValueError``. This
    catches accidental re-attachment that would silently drift the
    leaderboard grouping. The classical default is the only value that is
    allowed to be overwritten — it is the "not yet attached" sentinel.
    """
    new_contract = derive_scoring_contract(result, spec)
    current = result.diagnostics.scoring_contract
    if current != new_contract and not _is_classical_default(current):
        msg = (
            f"ScoringContract drift detected for candidate "
            f"{result.model_id!r}: already attached contract "
            f"{current!r} differs from newly derived {new_contract!r}. "
            f"attach_scoring_contract is expected to run once per result."
        )
        raise ValueError(msg)
    result.diagnostics.scoring_contract = new_contract
    return result


def _is_classical_default(contract: ScoringContract) -> bool:
    """Return True for the nlmixr2-FOCEI classical default contract.

    This is the value baked into :class:`DiagnosticBundle`'s default
    factory (see :mod:`apmode.bundle.models`). Treated as "uninitialised"
    for drift-detection purposes: runners may overwrite it once.
    """
    return (
        contract.nlpd_kind == "marginal"
        and contract.re_treatment == "integrated"
        and contract.nlpd_integrator == "nlmixr2_focei"
        and contract.float_precision == "float64"
    )
