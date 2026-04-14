# SPDX-License-Identifier: GPL-2.0-or-later
"""Prior AST models for the PK DSL (Phase 2+, plan 2026-04-14).

Priors are a first-class DSL field on `DSLSpec.priors`. They are consumed by
`stan_emitter` (inject into model block) and ignored by `nlmixr2_emitter`,
`node_runner`, and the agentic LLM backend unless explicitly wired.

A prior targets one of:
- a structural parameter name (e.g. "CL", "V", "ka") → log-scale Normal / LogNormal
- an IIV SD: "omega_<param>" (e.g. "omega_CL") → half-family
- a residual error SD: "sigma_prop" | "sigma_add" → half-family
- an IIV correlation matrix: "corr_iiv" → LKJ only

Validation enforces the parameterization schema: invalid (family, target) pairs
raise at compile time so the agentic LLM cannot propose nonsense priors.

References:
  FDA draft guidance FDA-2025-D-3217 (Jan 2026) — prior justification artifact.
  Schmidli et al. 2014 Biometrics — robust MAP (historical borrowing).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Prior source taxonomy — aligns with FDA 2026 prior-justification framing
# ---------------------------------------------------------------------------

PriorSource = Literal[
    "uninformative",
    "weakly_informative",
    "historical_data",
    "expert_elicitation",
    "meta_analysis",
]

# ---------------------------------------------------------------------------
# Target taxonomy
# ---------------------------------------------------------------------------

TargetKind = Literal["structural", "iiv_sd", "iov_sd", "residual_sd", "corr_iiv", "covariate"]


# ---------------------------------------------------------------------------
# Prior family variants
# ---------------------------------------------------------------------------


class NormalPrior(BaseModel):
    """Normal(mu, sigma) — real-valued; log-scale structural params and covariate coefficients."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Normal"] = "Normal"
    mu: float
    sigma: float = Field(gt=0)


class LogNormalPrior(BaseModel):
    """LogNormal(mu, sigma) — positive alternative for structural params on natural scale."""

    model_config = ConfigDict(frozen=True)
    type: Literal["LogNormal"] = "LogNormal"
    mu: float
    sigma: float = Field(gt=0)


class HalfNormalPrior(BaseModel):
    """Half-Normal(sigma) — positive-valued, for SDs. Weakly informative default for omega_*."""

    model_config = ConfigDict(frozen=True)
    type: Literal["HalfNormal"] = "HalfNormal"
    sigma: float = Field(gt=0)


class HalfCauchyPrior(BaseModel):
    """Half-Cauchy(scale) — positive-valued, heavy tail. Useful when uncertainty in SD is large."""

    model_config = ConfigDict(frozen=True)
    type: Literal["HalfCauchy"] = "HalfCauchy"
    scale: float = Field(gt=0)


class GammaPrior(BaseModel):
    """Gamma(alpha, beta) — positive-valued SDs or rates; conjugate for precision parameters."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Gamma"] = "Gamma"
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)


class InvGammaPrior(BaseModel):
    """InverseGamma(alpha, beta) — positive-valued. Classical choice for variances."""

    model_config = ConfigDict(frozen=True)
    type: Literal["InvGamma"] = "InvGamma"
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)


class BetaPrior(BaseModel):
    """Beta(alpha, beta) — in [0,1], for bioavailability F or mixing fractions."""

    model_config = ConfigDict(frozen=True)
    type: Literal["Beta"] = "Beta"
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)


class LKJPrior(BaseModel):
    """LKJ(eta) on correlation matrix. eta=1 uniform, eta>1 shrinks toward identity."""

    model_config = ConfigDict(frozen=True)
    type: Literal["LKJ"] = "LKJ"
    eta: float = Field(gt=0)


class MixturePrior(BaseModel):
    """Mixture of priors with weights. Core primitive for robust MAP historical borrowing.

    Example robust MAP: 80% LogNormal(historical mean, historical sd) + 20% weakly-informative.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["Mixture"] = "Mixture"
    components: list[
        Annotated[
            NormalPrior
            | LogNormalPrior
            | HalfNormalPrior
            | HalfCauchyPrior
            | GammaPrior
            | InvGammaPrior
            | BetaPrior,
            Field(discriminator="type"),
        ]
    ] = Field(min_length=2)
    weights: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def weights_match_components(self) -> MixturePrior:
        if len(self.weights) != len(self.components):
            raise ValueError("weights and components must have equal length")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError(f"mixture weights must sum to 1.0, got {sum(self.weights)}")
        if any(w < 0 for w in self.weights):
            raise ValueError("mixture weights must be non-negative")
        return self


class HistoricalBorrowingPrior(BaseModel):
    """Robust MAP prior built from historical dataset summaries (Schmidli 2014).

    Compiles into a MixturePrior at emit time: (1-robust_weight) on the MAP component,
    robust_weight on a weakly-informative component.
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["HistoricalBorrowing"] = "HistoricalBorrowing"
    map_mean: float
    map_sd: float = Field(gt=0)
    robust_weight: float = Field(ge=0, le=1, default=0.2)
    historical_refs: list[str] = Field(min_length=1)


PriorFamily = Annotated[
    NormalPrior
    | LogNormalPrior
    | HalfNormalPrior
    | HalfCauchyPrior
    | GammaPrior
    | InvGammaPrior
    | BetaPrior
    | LKJPrior
    | MixturePrior
    | HistoricalBorrowingPrior,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# PriorSpec — a prior with a target and justification
# ---------------------------------------------------------------------------


class PriorSpec(BaseModel):
    """A single prior declaration: target parameter, family, source, justification.

    The `target` resolves against the DSLSpec at validation time:
      - structural_param_names() → "structural" target
      - omega_<name> for name in IIV → "iiv_sd" target
      - omega_iov_<name> → "iov_sd" target
      - "sigma_prop" / "sigma_add" → "residual_sd" target
      - "corr_iiv" → "corr_iiv" target (LKJ required)
      - "beta_<param>_<covariate>" → "covariate" target

    `justification` is required for any `source` other than uninformative/weakly_informative.
    This is an FDA Gate 2 requirement (see plan §3.6).
    """

    model_config = ConfigDict(frozen=True)

    target: str
    family: PriorFamily
    source: PriorSource = "weakly_informative"
    justification: str = ""
    historical_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def justification_required_for_informative_sources(self) -> PriorSpec:
        if self.source in ("historical_data", "expert_elicitation", "meta_analysis"):
            if not self.justification.strip():
                raise ValueError(
                    f"source={self.source} requires a non-empty justification "
                    f"(FDA Gate 2 requirement)"
                )
            if self.source == "historical_data" and not self.historical_refs:
                raise ValueError(
                    "source=historical_data requires historical_refs (dataset identifiers)"
                )
        return self


# ---------------------------------------------------------------------------
# Parameterization schema — valid (target_kind, family) pairs
# ---------------------------------------------------------------------------


_VALID_FAMILIES: dict[TargetKind, frozenset[str]] = {
    "structural": frozenset({"Normal", "LogNormal", "Mixture", "HistoricalBorrowing"}),
    "iiv_sd": frozenset({"HalfNormal", "HalfCauchy", "Gamma", "InvGamma"}),
    "iov_sd": frozenset({"HalfNormal", "HalfCauchy", "Gamma", "InvGamma"}),
    "residual_sd": frozenset({"HalfNormal", "HalfCauchy", "Gamma", "InvGamma"}),
    # LKJ on corr_iiv is accepted by the schema; stan_emitter does not yet
    # declare corr_iiv in the parameters block and will raise NotImplementedError
    # if it sees it. Accepting here so agentic transforms can plan for it.
    "corr_iiv": frozenset({"LKJ"}),
    # Covariate betas accept Normal (default) and meta-analytic robust priors
    # (e.g., allometric exponents with external historical support).
    "covariate": frozenset({"Normal", "Mixture", "HistoricalBorrowing"}),
}


def classify_target(target: str, structural_params: set[str]) -> TargetKind | None:
    """Infer the target kind from the target name and the spec's structural params.

    Returns None if the target doesn't match any known pattern — caller should
    surface this as a validation error.
    """
    if target in structural_params:
        return "structural"
    if target.startswith("omega_iov_"):
        return "iov_sd"
    if target.startswith("omega_"):
        return "iiv_sd"
    if target in ("sigma_prop", "sigma_add"):
        return "residual_sd"
    if target == "corr_iiv":
        return "corr_iiv"
    if target.startswith("beta_"):
        return "covariate"
    return None


def validate_prior_family(target_kind: TargetKind, family: PriorFamily) -> str | None:
    """Return error string if family doesn't match target kind; None otherwise."""
    allowed = _VALID_FAMILIES[target_kind]
    if family.type not in allowed:
        return (
            f"Prior family {family.type!r} is not valid for target kind {target_kind!r}. "
            f"Allowed: {sorted(allowed)}"
        )
    return None


def validate_priors(
    priors: list[PriorSpec],
    structural_params: set[str],
) -> list[str]:
    """Validate a list of priors against a spec's parameter universe. Returns errors."""
    errors: list[str] = []
    seen_targets: set[str] = set()

    for prior in priors:
        if prior.target in seen_targets:
            errors.append(f"Duplicate prior on target {prior.target!r}")
            continue
        seen_targets.add(prior.target)

        kind = classify_target(prior.target, structural_params)
        if kind is None:
            errors.append(
                f"Prior target {prior.target!r} does not match any known pattern "
                f"(structural params: {sorted(structural_params)})"
            )
            continue

        family_err = validate_prior_family(kind, prior.family)
        if family_err:
            errors.append(family_err)

    return errors


# ---------------------------------------------------------------------------
# Convenience constructors — defaults per FDA weakly-informative recommendations
# ---------------------------------------------------------------------------


def default_structural_prior(log_init: float = 0.0) -> NormalPrior:
    """Default weakly-informative prior on log-scale structural param: Normal(log_init, 2)."""
    return NormalPrior(mu=log_init, sigma=2.0)


def default_iiv_prior() -> HalfCauchyPrior:
    """Default weakly-informative prior on omega: HalfCauchy(1)."""
    return HalfCauchyPrior(scale=1.0)


def default_residual_prior(init: float = 0.3) -> HalfCauchyPrior:
    """Default weakly-informative prior on sigma: HalfCauchy(init)."""
    return HalfCauchyPrior(scale=init)


def default_corr_prior() -> LKJPrior:
    """Default LKJ(2) — mild shrinkage toward independence."""
    return LKJPrior(eta=2.0)


def default_covariate_prior(form: str = "power") -> NormalPrior:
    """Default covariate-coefficient prior: Normal(0.75, 0.5) for power, else Normal(0, 1)."""
    if form == "power":
        return NormalPrior(mu=0.75, sigma=0.5)
    return NormalPrior(mu=0.0, sigma=1.0)
