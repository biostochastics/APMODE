# SPDX-License-Identifier: GPL-2.0-or-later
"""Stan codegen emitter: DSL AST -> Stan program (PRD v0.3, Phase 2+).

Generates a complete Stan program from a DSLSpec for probabilistic inference.
Uses ODE-based models via Stan's `ode_rk45` integrator for non-linear dynamics,
and analytical solutions (matrix exponential) for linear compartment models.

Observation model:
  - Proportional: y ~ lognormal(log(f), sigma)
  - Additive:     y ~ normal(f, sigma)
  - Combined:     y ~ normal(f, sqrt((sigma_prop * f)^2 + sigma_add^2))

IIV is modeled via log-normal random effects:
  theta_i = theta * exp(eta_i), eta_i ~ N(0, omega^2)

NODE modules are not supported (Stan has no neural ODE support).
BLQ M3/M4 and IOV are not yet implemented (marked as Phase 3).
"""

from __future__ import annotations

import re

from apmode.dsl.ast_models import (
    BLQM3,
    BLQM4,
    IIV,
    IOV,
    TMDDQSS,
    Additive,
    Combined,
    CovariateLink,
    DSLSpec,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    OneCmt,
    ParallelLinearMM,
    Proportional,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.priors import (
    BetaPrior,
    GammaPrior,
    HalfCauchyPrior,
    HalfNormalPrior,
    HistoricalBorrowingPrior,
    InvGammaPrior,
    LKJPrior,
    LogNormalPrior,
    MixturePrior,
    NormalPrior,
    PriorFamily,
    PriorSpec,
)


def emit_stan(
    spec: DSLSpec,
    initial_estimates: dict[str, float] | None = None,
) -> str:
    """Emit a complete Stan program from a DSLSpec.

    Args:
        spec: The compiled DSL specification.
        initial_estimates: Optional initial estimate overrides (used for
            informative priors centered on NCA/classical estimates).

    Returns:
        A Stan program string.

    Raises:
        NotImplementedError: For NODE modules, BLQ M3/M4, or IOV.
    """
    if spec.has_node_modules():
        raise NotImplementedError(
            "NODE module codegen is not supported for Stan. "
            "NODE backends use the JAX/Diffrax emitter."
        )

    # IOV: etas are declared but not applied in transformed parameters.
    if any(isinstance(v, IOV) for v in spec.variability):
        raise NotImplementedError(
            "IOV is not yet fully implemented in Stan codegen. "
            "IOV etas are declared but not applied to parameter back-transforms."
        )

    # Unsupported absorption types in ODE mode
    if _needs_ode(spec) and isinstance(spec.absorption, (ZeroOrder, MixedFirstZero)):
        raise NotImplementedError(
            f"Stan ODE codegen does not support {spec.absorption.type} absorption. "
            f"Use the nlmixr2 backend."
        )

    blocks: list[str] = []
    blocks.append(f"// APMODE generated Stan model: {spec.model_id}")
    blocks.append("")

    needs_ode = _needs_ode(spec)

    if needs_ode:
        blocks.append(_emit_functions_block(spec))

    blocks.append(_emit_data_block(spec))
    blocks.append(_emit_transformed_data_block())
    blocks.append(_emit_parameters_block(spec))
    blocks.append(_emit_transformed_parameters_block(spec, needs_ode))
    blocks.append(_emit_model_block(spec, initial_estimates))
    blocks.append(_emit_generated_quantities_block(spec))

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Helpers: prior lookup and Stan prior statement emission
# ---------------------------------------------------------------------------


def _find_prior(priors: list[PriorSpec], target: str) -> PriorSpec | None:
    """Return the user-declared prior for `target`, or None."""
    for p in priors:
        if p.target == target:
            return p
    return None


def _emit_user_prior(
    stan_param: str,
    family: PriorFamily,
    *,
    on_log_scale: bool,
) -> list[str]:
    """Emit Stan prior statements for a declared prior family on a parameter.

    Parameters
    ----------
    stan_param
        The Stan variable name to sample (e.g. "log_CL", "omega_CL", "sigma_prop",
        "beta_CL_WT"). Note: on_log_scale means the Stan variable is itself the
        log-transform of the user-space parameter, which matters for emitting
        LogNormal (must be transformed to Normal on the log-scale variable).
    family
        The prior family (discriminated union from priors.py).
    on_log_scale
        True when `stan_param` is the log of the user-space parameter (i.e.
        structural log_{name}). LogNormal on a log-scale variable becomes
        Normal(mu, sigma); on a natural-scale variable it stays LogNormal.
    """
    prefix = f"  {stan_param} ~ "

    if isinstance(family, NormalPrior):
        return [f"{prefix}normal({family.mu:.6f}, {family.sigma:.6f});"]

    if isinstance(family, LogNormalPrior):
        # If stan_param is already log-scale, LogNormal(mu, sigma) becomes
        # Normal(mu, sigma) on the log variable.
        if on_log_scale:
            return [f"{prefix}normal({family.mu:.6f}, {family.sigma:.6f});"]
        return [f"{prefix}lognormal({family.mu:.6f}, {family.sigma:.6f});"]

    if isinstance(family, HalfNormalPrior):
        # Relies on the parameter having <lower=0> in the parameters block.
        return [f"{prefix}normal(0, {family.sigma:.6f});"]

    if isinstance(family, HalfCauchyPrior):
        return [f"{prefix}cauchy(0, {family.scale:.6f});"]

    if isinstance(family, GammaPrior):
        return [f"{prefix}gamma({family.alpha:.6f}, {family.beta:.6f});"]

    if isinstance(family, InvGammaPrior):
        return [f"{prefix}inv_gamma({family.alpha:.6f}, {family.beta:.6f});"]

    if isinstance(family, BetaPrior):
        return [f"{prefix}beta({family.alpha:.6f}, {family.beta:.6f});"]

    if isinstance(family, LKJPrior):
        # LKJ targets a correlation matrix. Emit lkj_corr; caller must ensure
        # the Stan variable is declared as corr_matrix[K] (out of scope for v1
        # since IIV correlation emission is planned for a follow-up).
        return [f"{prefix}lkj_corr({family.eta:.6f});"]

    if isinstance(family, MixturePrior):
        return _emit_mixture_prior(stan_param, family, on_log_scale=on_log_scale)

    if isinstance(family, HistoricalBorrowingPrior):
        return _emit_historical_borrowing_prior(stan_param, family, on_log_scale=on_log_scale)

    raise NotImplementedError(f"Unsupported prior family: {type(family).__name__}")


_LOG_2 = 0.6931471805599453


def _component_lpdf(
    stan_param: str,
    component: NormalPrior
    | LogNormalPrior
    | HalfNormalPrior
    | HalfCauchyPrior
    | GammaPrior
    | InvGammaPrior
    | BetaPrior,
    *,
    on_log_scale: bool,
) -> str:
    """Return a Stan lpdf expression for a single mixture component.

    Used by _emit_mixture_prior and _emit_historical_borrowing_prior to build
    log_sum_exp form: target += log_sum_exp(log(w_k) + lpdf_k(...)).

    For half-family components (HalfNormal, HalfCauchy) on a <lower=0> variable,
    the correct normalized log-density is normal_lpdf(x | 0, sigma) + log(2).
    The +log(2) MUST be retained in mixtures so that half-* components are not
    artificially down-weighted by 50% relative to fully-supported components
    like Gamma/InvGamma.
    """
    if isinstance(component, NormalPrior):
        return f"normal_lpdf({stan_param} | {component.mu:.6f}, {component.sigma:.6f})"
    if isinstance(component, LogNormalPrior):
        if on_log_scale:
            return f"normal_lpdf({stan_param} | {component.mu:.6f}, {component.sigma:.6f})"
        return f"lognormal_lpdf({stan_param} | {component.mu:.6f}, {component.sigma:.6f})"
    if isinstance(component, HalfNormalPrior):
        # Half-Normal lpdf = normal_lpdf(x | 0, sigma) + log(2) on x>=0.
        return f"(normal_lpdf({stan_param} | 0, {component.sigma:.6f}) + {_LOG_2:.6f})"
    if isinstance(component, HalfCauchyPrior):
        # Half-Cauchy lpdf = cauchy_lpdf(x | 0, scale) + log(2) on x>=0.
        return f"(cauchy_lpdf({stan_param} | 0, {component.scale:.6f}) + {_LOG_2:.6f})"
    if isinstance(component, GammaPrior):
        return f"gamma_lpdf({stan_param} | {component.alpha:.6f}, {component.beta:.6f})"
    if isinstance(component, InvGammaPrior):
        return f"inv_gamma_lpdf({stan_param} | {component.alpha:.6f}, {component.beta:.6f})"
    if isinstance(component, BetaPrior):
        return f"beta_lpdf({stan_param} | {component.alpha:.6f}, {component.beta:.6f})"
    raise NotImplementedError(
        f"Mixture component type {type(component).__name__} is not supported"
    )


def _emit_mixture_prior(
    stan_param: str,
    family: MixturePrior,
    *,
    on_log_scale: bool,
) -> list[str]:
    """Emit a k-component mixture prior as target += log_sum_exp(...).

    See Stan User's Guide §13.1 (Mixture modeling). Weights are on the
    simplex (validated in priors.py); we pre-log them for numerical stability.

    Zero-weight components are dropped (no need to emit log(0) placeholders);
    if all components have zero weight the mixture is degenerate and this
    function raises ValueError.
    """
    import math

    active = [
        (w, comp) for w, comp in zip(family.weights, family.components, strict=True) if w > 0.0
    ]
    if not active:
        raise ValueError(
            f"Mixture prior on {stan_param!r} has no active components (all weights zero)"
        )

    terms = [
        f"{math.log(w):.6f} + {_component_lpdf(stan_param, comp, on_log_scale=on_log_scale)}"
        for w, comp in active
    ]
    joined = ",\n    ".join(terms)
    return [
        f"  // Mixture prior on {stan_param} (k={len(active)})",
        f"  target += log_sum_exp([\n    {joined}\n  ]);",
    ]


def _emit_historical_borrowing_prior(
    stan_param: str,
    family: HistoricalBorrowingPrior,
    *,
    on_log_scale: bool,
) -> list[str]:
    """Compile Schmidli 2014 robust MAP to a 2-component mixture.

    Weights: (1 - robust_weight) on the MAP component, robust_weight on a
    weakly-informative component. The MAP component is Normal on log-scale
    structural params (the canonical PK use case).
    """
    if not on_log_scale:
        # Structural params in this emitter are always on the log scale;
        # HistoricalBorrowing on non-log targets (e.g., sigma) is out of scope.
        raise NotImplementedError(
            f"HistoricalBorrowingPrior on non-log-scale target {stan_param!r} is not supported. "
            "Use MixturePrior directly instead."
        )

    map_component = NormalPrior(mu=family.map_mean, sigma=family.map_sd)
    # Weak component must span the plausible range of PK parameters on the log

    # scale. V can easily exceed 100 L and CL can range 0.1-100 L/h, so a
    # Normal(0, 2) log-scale prior (95% CI ≈ [0.02, 55]) is too narrow and would
    # penalize true values. Use Normal(0, 10) instead.
    weak_component = NormalPrior(mu=0.0, sigma=10.0)
    mixture = MixturePrior(
        components=[map_component, weak_component],
        weights=[1.0 - family.robust_weight, family.robust_weight],
    )
    return _emit_mixture_prior(stan_param, mixture, on_log_scale=on_log_scale)


# ---------------------------------------------------------------------------
# Helper: does this spec require ODE integration?
# ---------------------------------------------------------------------------


# Shared helper so both emitters stay in sync on the "does this spec
# need an ODE?" decision.
from apmode.dsl._emitter_utils import needs_ode as _needs_ode  # noqa: E402

# ---------------------------------------------------------------------------
# functions {} block — ODE system definition
# ---------------------------------------------------------------------------


def _emit_functions_block(spec: DSLSpec) -> str:
    """Emit the functions{} block with the ODE system."""
    lines = ["functions {"]
    lines.append("  vector ode_rhs(real t, vector y, array[] real theta,")
    lines.append("                 array[] real x_r, array[] int x_i) {")

    n_states = _n_states(spec)
    lines.append(f"    vector[{n_states}] dydt;")
    lines.append("")

    # Unpack theta
    lines.extend(_emit_theta_unpack(spec, indent=4))
    lines.append("")

    # State aliases
    lines.extend(_emit_state_aliases(spec, indent=4))
    lines.append("")

    # Dynamics
    lines.extend(_emit_ode_dynamics(spec, indent=4))
    lines.append("")

    lines.append("    return dydt;")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# data {} block
# ---------------------------------------------------------------------------


def _emit_data_block(spec: DSLSpec) -> str:
    lines = ["data {"]
    lines.append("  int<lower=1> N;              // total observations")
    lines.append("  int<lower=1> N_subjects;     // number of subjects")
    lines.append("  array[N] int<lower=1,upper=N_subjects> subject;  // subject index")
    lines.append("  vector[N] time;              // observation times")
    lines.append("  vector<lower=0>[N] dv;       // observed concentrations")
    lines.append("")
    lines.append("  // Multi-dose event schedule")
    lines.append("  int<lower=0> N_events;       // total dose/reset events across all subjects")
    lines.append("  array[N_events] int<lower=1,upper=N_subjects> event_subject;")
    lines.append("  vector[N_events] event_time;  // event times")
    lines.append("  vector[N_events] event_amt;   // dose amounts (0 for resets)")
    lines.append("  array[N_events] int event_cmt; // compartment")
    lines.append("  array[N_events] int event_evid; // 1=dose, 3=reset, 4=reset+dose")
    lines.append("  vector[N_events] event_rate;   // infusion rate (0=bolus)")
    lines.append("  // Per-subject event index ranges (1-indexed, inclusive)")
    lines.append("  array[N_subjects] int event_start;  // first event index for subject")
    lines.append("  array[N_subjects] int event_end;    // last event index for subject")

    # IOV: occasion index per observation
    if _has_iov(spec):
        lines.append("  int<lower=1> N_occ;          // number of occasions")
        lines.append("  array[N] int<lower=1,upper=N_occ> occ;  // occasion index")

    # BLQ: censoring indicator (1 = below LOQ)
    if _is_blq(spec):
        lines.append("  array[N] int<lower=0,upper=1> cens;  // censoring indicator")
        obs = spec.observation
        if isinstance(obs, (BLQM3, BLQM4)):
            lines.append(
                f"  real<lower=0> loq;           // limit of quantification ({obs.loq_value})"
            )

    # Covariates
    cov_links = [v for v in spec.variability if isinstance(v, CovariateLink)]
    cov_names = sorted({_sanitize_stan_name(c.covariate, context="covariate") for c in cov_links})
    for cov in cov_names:
        lines.append(f"  vector[N_subjects] {cov};")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# transformed data {} block
# ---------------------------------------------------------------------------


def _emit_transformed_data_block() -> str:
    lines = ["transformed data {"]
    lines.append("  array[0] real x_r;")
    lines.append("  array[0] int x_i;")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parameters {} block
# ---------------------------------------------------------------------------


def _emit_parameters_block(spec: DSLSpec) -> str:
    lines = ["parameters {"]

    # Structural params (log-domain)
    for name in spec.structural_param_names():
        lines.append(f"  real log_{name};")

    # IIV omega
    iiv_params = _iiv_params(spec)
    for p in iiv_params:
        lines.append(f"  real<lower=0> omega_{p};")

    # Individual etas
    if iiv_params:
        lines.append(f"  matrix[N_subjects, {len(iiv_params)}] eta_raw;")

    # IOV omega + etas
    iov_params = _iov_params(spec)
    for p in iov_params:
        lines.append(f"  real<lower=0> omega_iov_{p};")
    if iov_params and _has_iov(spec):
        lines.append(f"  matrix[N_subjects * N_occ, {len(iov_params)}] eta_iov_raw;")

    # Residual error
    lines.extend(_emit_sigma_params(spec, _is_blq(spec)))

    # Covariate coefficients
    cov_links = [v for v in spec.variability if isinstance(v, CovariateLink)]
    for cov in cov_links:
        p = _sanitize_stan_name(cov.param, context="covariate-target parameter")
        c = _sanitize_stan_name(cov.covariate, context="covariate")
        lines.append(f"  real beta_{p}_{c};")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# transformed parameters {} block
# ---------------------------------------------------------------------------


def _emit_transformed_parameters_block(spec: DSLSpec, needs_ode: bool) -> str:
    lines = ["transformed parameters {"]
    lines.append("  vector<lower=0>[N] f;  // predicted concentrations")
    lines.append("")

    iiv_params = _iiv_params(spec)

    lines.append("  for (i in 1:N_subjects) {")

    # Back-transform structural params with IIV
    for name in spec.structural_param_names():
        eta_expr = ""
        if name in iiv_params:
            idx = iiv_params.index(name) + 1
            eta_expr = f" + omega_{name} * eta_raw[i, {idx}]"

        # Covariate effects
        cov_expr = _covariate_expr(spec, name, "i")
        lines.append(f"    real {name}_i = exp(log_{name}{eta_expr}{cov_expr});")

    lines.append("")

    if needs_ode:
        lines.extend(_emit_ode_solve(spec, indent=4))
    else:
        lines.extend(_emit_analytical_solve(spec, indent=4))

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# model {} block
# ---------------------------------------------------------------------------


def _emit_model_block(
    spec: DSLSpec,
    initial_estimates: dict[str, float] | None = None,
) -> str:
    lines = ["model {"]
    ie = initial_estimates or {}

    # Priors on structural params — user overrides via spec.priors; otherwise default.
    lines.append("  // Priors on structural parameters")
    for name in spec.structural_param_names():
        user_prior = _find_prior(spec.priors, name)
        if user_prior is not None:
            lines.extend(_emit_user_prior(f"log_{name}", user_prior.family, on_log_scale=True))
        else:
            center = ie.get(name)
            if center is not None and center > 0:
                lines.append(f"  log_{name} ~ normal({_log(center):.4f}, 1);")
            else:
                lines.append(f"  log_{name} ~ normal(0, 2);")

    lines.append("")

    # Priors on IIV omega — user overrides via spec.priors; otherwise half-cauchy.
    iiv_params = _iiv_params(spec)
    for p in iiv_params:
        user_prior = _find_prior(spec.priors, f"omega_{p}")
        if user_prior is not None:
            lines.extend(_emit_user_prior(f"omega_{p}", user_prior.family, on_log_scale=False))
        else:
            lines.append(f"  omega_{p} ~ cauchy(0, 1);")

    # Standard normal etas (non-centered parameterization)
    if iiv_params:
        lines.append("  to_vector(eta_raw) ~ std_normal();")

    # IOV priors — user overrides via spec.priors; otherwise half-cauchy(0.5).
    iov_params = _iov_params(spec)
    for p in iov_params:
        user_prior = _find_prior(spec.priors, f"omega_iov_{p}")
        if user_prior is not None:
            lines.extend(_emit_user_prior(f"omega_iov_{p}", user_prior.family, on_log_scale=False))
        else:
            lines.append(f"  omega_iov_{p} ~ cauchy(0, 0.5);")
    if iov_params and _has_iov(spec):
        lines.append("  to_vector(eta_iov_raw) ~ std_normal();")

    lines.append("")

    # Priors on covariate coefficients
    cov_links = [v for v in spec.variability if isinstance(v, CovariateLink)]
    for cov in cov_links:
        p = _sanitize_stan_name(cov.param, context="covariate-target parameter")
        c = _sanitize_stan_name(cov.covariate, context="covariate")
        cov_target = f"beta_{p}_{c}"
        user_prior = _find_prior(spec.priors, cov_target)
        if user_prior is not None:
            lines.extend(_emit_user_prior(cov_target, user_prior.family, on_log_scale=False))
        elif cov.form == "power":
            lines.append(f"  {cov_target} ~ normal(0.75, 0.5);")
        else:
            lines.append(f"  {cov_target} ~ normal(0, 1);")

    # Sigma priors — user overrides handled inside _emit_sigma_priors.
    lines.extend(_emit_sigma_priors(spec))

    lines.append("")

    # Likelihood
    lines.append("  // Likelihood")
    lines.extend(_emit_likelihood(spec))

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# generated quantities {} block
# ---------------------------------------------------------------------------


def _emit_generated_quantities_block(spec: DSLSpec) -> str:
    lines = ["generated quantities {"]

    # Log-likelihood for LOO-CV
    lines.append("  vector[N] log_lik;")
    lines.append("  for (n in 1:N) {")
    lines.extend(_emit_log_lik(spec, indent=4))
    lines.append("  }")

    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_STAN_IDENT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# Stan language keywords and reserved names. Using any of these as a
# user-supplied parameter/covariate would produce uncompilable Stan code
# and must be rejected at emission time. Not exhaustive, but covers the
# common pitfalls; Pydantic's StanIdentifier regex catches syntax
# violations separately.
_STAN_RESERVED: frozenset[str] = frozenset(
    {
        "data",
        "parameters",
        "model",
        "functions",
        "transformed",
        "generated",
        "quantities",
        "return",
        "if",
        "else",
        "for",
        "while",
        "in",
        "break",
        "continue",
        "true",
        "false",
        "real",
        "int",
        "vector",
        "row_vector",
        "matrix",
        "array",
        "tuple",
        "complex",
        "target",
        "print",
        "reject",
        "lower",
        "upper",
    }
)


def _sanitize_stan_name(name: str, *, context: str = "identifier") -> str:
    """Validate a name against Stan's identifier rules + reserved-word list.

    Returns the name unchanged when valid; raises ``ValueError`` otherwise.
    Called at every f-string interpolation site in the Stan emitter to
    provide defense in depth against identifier injection — the AST's
    ``StanIdentifier`` type catches most violations at construction, but
    keyword collisions (``data``, ``model``) need this runtime check, and
    specs built without Pydantic validation still flow through here.
    """
    if not _STAN_IDENT_RE.fullmatch(name):
        msg = f"invalid Stan {context}: {name!r}"
        raise ValueError(msg)
    if name in _STAN_RESERVED:
        msg = f"Stan reserved keyword used as {context}: {name!r}"
        raise ValueError(msg)
    if name.endswith("__"):
        msg = f"Stan {context} must not end with double underscore: {name!r}"
        raise ValueError(msg)
    return name


def _needs_depot(spec: DSLSpec) -> bool:
    """Whether the spec emits a depot compartment.

    IV bolus delivers dose directly into the central compartment; no depot
    state exists. All other absorption types (first-order, transit, etc.)
    require a depot state.
    """
    return not isinstance(spec.absorption, IVBolus)


def _centr_idx(spec: DSLSpec) -> int:
    """Stan-array index of the central compartment in ``y[]``.

    Depot is always ``y[1]`` when present, pushing central to ``y[2]``.
    Under IVBolus there is no depot, so central is ``y[1]``.
    """
    return 2 if _needs_depot(spec) else 1


def _n_states(spec: DSLSpec) -> int:
    """Number of ODE states."""
    base = 1 if _needs_depot(spec) else 0  # depot only when absorption requires it
    dist = spec.distribution
    if isinstance(dist, OneCmt):
        return base + 1
    if isinstance(dist, TwoCmt):
        return base + 2
    if isinstance(dist, ThreeCmt):
        return base + 3
    if isinstance(dist, TMDDCore):
        return base + 3  # centr, R, RC
    if isinstance(dist, TMDDQSS):
        return base + 2  # Atot, Rtot
    return base + 1


def _iiv_params(spec: DSLSpec) -> list[str]:
    """Collect IIV parameter names in order, sanitized for Stan emission."""
    params: list[str] = []
    for v in spec.variability:
        if isinstance(v, IIV):
            for p in v.params:
                safe = _sanitize_stan_name(p, context="IIV parameter")
                if safe not in params:
                    params.append(safe)
    return params


def _iov_params(spec: DSLSpec) -> list[str]:
    """Collect IOV parameter names in order, sanitized for Stan emission."""
    params: list[str] = []
    for v in spec.variability:
        if isinstance(v, IOV):
            for p in v.params:
                safe = _sanitize_stan_name(p, context="IOV parameter")
                if safe not in params:
                    params.append(safe)
    return params


def _has_iov(spec: DSLSpec) -> bool:
    return any(isinstance(v, IOV) for v in spec.variability)


def _is_blq(spec: DSLSpec) -> bool:
    return isinstance(spec.observation, (BLQM3, BLQM4))


def _blq_sigma_value(spec: DSLSpec) -> tuple[str, float]:
    """Return (error_model, sigma_init) for BLQ observation models."""
    obs = spec.observation
    if isinstance(obs, (BLQM3, BLQM4)):
        return obs.error_model, obs.sigma_prop
    return "proportional", 0.1


def _log(x: float) -> float:
    import math

    return math.log(max(x, 1e-10))


def _covariate_expr(spec: DSLSpec, param: str, idx_var: str) -> str:
    """Build covariate effect expression for a parameter."""
    parts: list[str] = []
    for v in spec.variability:
        if isinstance(v, CovariateLink) and v.param == param:
            p = _sanitize_stan_name(v.param, context="covariate-target parameter")
            c = _sanitize_stan_name(v.covariate, context="covariate")
            coeff = f"beta_{p}_{c}"
            if v.form == "power":
                parts.append(f" + {coeff} * log({c}[{idx_var}] / 70)")
            elif v.form in ("exponential", "categorical"):
                parts.append(f" + {coeff} * {c}[{idx_var}]")
            elif v.form == "linear":
                parts.append(f" + log(1 + {coeff} * {c}[{idx_var}])")
            elif v.form == "maturation":
                raise NotImplementedError(
                    f"Maturation covariate form not yet supported in Stan codegen "
                    f"(param={p}, covariate={c})."
                )
    return "".join(parts)


def _emit_sigma_params(spec: DSLSpec, is_blq: bool = False) -> list[str]:
    obs = spec.observation
    if isinstance(obs, (BLQM3, BLQM4)):
        if obs.error_model == "additive":
            return ["  real<lower=0> sigma_add;"]
        if obs.error_model == "combined":
            return ["  real<lower=0> sigma_prop;", "  real<lower=0> sigma_add;"]
        return ["  real<lower=0> sigma_prop;"]
    if isinstance(obs, Proportional):
        return ["  real<lower=0> sigma_prop;"]
    if isinstance(obs, Additive):
        return ["  real<lower=0> sigma_add;"]
    if isinstance(obs, Combined):
        return ["  real<lower=0> sigma_prop;", "  real<lower=0> sigma_add;"]
    return ["  real<lower=0> sigma_prop;"]


def _emit_sigma_priors(spec: DSLSpec) -> list[str]:
    """Emit priors on sigma_prop / sigma_add with user-override support."""

    def _for(target: str, default_scale: float) -> list[str]:
        user = _find_prior(spec.priors, target)
        if user is not None:
            return _emit_user_prior(target, user.family, on_log_scale=False)
        return [f"  {target} ~ cauchy(0, {default_scale});"]

    obs = spec.observation
    lines: list[str] = []
    if isinstance(obs, (BLQM3, BLQM4)):
        if obs.error_model == "additive":
            lines.extend(_for("sigma_add", obs.sigma_add))
        elif obs.error_model == "combined":
            lines.extend(_for("sigma_prop", obs.sigma_prop))
            lines.extend(_for("sigma_add", obs.sigma_add))
        else:
            lines.extend(_for("sigma_prop", obs.sigma_prop))
    elif isinstance(obs, Proportional):
        lines.extend(_for("sigma_prop", obs.sigma_prop))
    elif isinstance(obs, Additive):
        lines.extend(_for("sigma_add", obs.sigma_add))
    elif isinstance(obs, Combined):
        lines.extend(_for("sigma_prop", obs.sigma_prop))
        lines.extend(_for("sigma_add", obs.sigma_add))
    else:
        lines.extend(_for("sigma_prop", 0.3))
    return lines


def _emit_likelihood(spec: DSLSpec) -> list[str]:
    obs = spec.observation
    if isinstance(obs, (BLQM3, BLQM4)):
        return _emit_blq_likelihood(spec)
    if isinstance(obs, Proportional):
        return ["  dv ~ lognormal(log(f), sigma_prop);"]
    if isinstance(obs, Additive):
        return ["  dv ~ normal(f, sigma_add);"]
    if isinstance(obs, Combined):
        return [
            "  for (n in 1:N)",
            "    dv[n] ~ normal(f[n], sqrt(square(sigma_prop * f[n]) + square(sigma_add)));",
        ]
    return ["  dv ~ lognormal(log(f), sigma_prop);"]


def _emit_blq_likelihood(spec: DSLSpec) -> list[str]:
    """Emit BLQ M3/M4 censored likelihood.

    M3: Left-censoring at LOQ.
      - Observed (cens=0): normal_lpdf(dv | f, sigma)
      - Censored (cens=1): normal_lcdf(loq | f, sigma)  [P(Y < LOQ)]

    M4: Censoring with positivity constraint (interval censoring 0..LOQ).
      - Censored: log_diff_exp(normal_lcdf(loq | f, sigma), normal_lcdf(0 | f, sigma))
    """
    obs = spec.observation
    is_m4 = isinstance(obs, BLQM4)

    # Determine sigma expression
    assert isinstance(obs, (BLQM3, BLQM4))
    if obs.error_model == "additive":
        sigma_expr = "sigma_add"
    elif obs.error_model == "combined":
        sigma_expr = "sqrt(square(sigma_prop * f[n]) + square(sigma_add))"
    else:
        sigma_expr = "sigma_prop * f[n]"  # proportional on natural scale

    lines: list[str] = []
    lines.append("  for (n in 1:N) {")
    lines.append("    if (cens[n] == 0) {")
    lines.append(f"      target += normal_lpdf(dv[n] | f[n], {sigma_expr});")
    lines.append("    } else {")
    if is_m4:
        lines.append(
            f"      target += log_diff_exp("
            f"normal_lcdf(loq | f[n], {sigma_expr}), "
            f"normal_lcdf(0 | f[n], {sigma_expr}));"
        )
    else:
        lines.append(f"      target += normal_lcdf(loq | f[n], {sigma_expr});")
    lines.append("    }")
    lines.append("  }")
    return lines


def _emit_log_lik(spec: DSLSpec, indent: int = 4) -> list[str]:
    pad = " " * indent
    obs = spec.observation
    if isinstance(obs, (BLQM3, BLQM4)):
        return _emit_blq_log_lik(spec, indent)
    if isinstance(obs, Proportional):
        return [f"{pad}log_lik[n] = lognormal_lpdf(dv[n] | log(f[n]), sigma_prop);"]
    if isinstance(obs, Additive):
        return [f"{pad}log_lik[n] = normal_lpdf(dv[n] | f[n], sigma_add);"]
    if isinstance(obs, Combined):
        return [
            f"{pad}log_lik[n] = normal_lpdf(dv[n] | f[n], "
            f"sqrt(square(sigma_prop * f[n]) + square(sigma_add)));"
        ]
    return [f"{pad}log_lik[n] = lognormal_lpdf(dv[n] | log(f[n]), sigma_prop);"]


def _emit_blq_log_lik(spec: DSLSpec, indent: int = 4) -> list[str]:
    """Emit per-observation log-likelihood for BLQ models."""
    pad = " " * indent
    obs = spec.observation
    assert isinstance(obs, (BLQM3, BLQM4))
    is_m4 = isinstance(obs, BLQM4)

    if obs.error_model == "additive":
        sigma_expr = "sigma_add"
    elif obs.error_model == "combined":
        sigma_expr = "sqrt(square(sigma_prop * f[n]) + square(sigma_add))"
    else:
        sigma_expr = "sigma_prop * f[n]"

    lines: list[str] = []
    lines.append(f"{pad}if (cens[n] == 0) {{")
    lines.append(f"{pad}  log_lik[n] = normal_lpdf(dv[n] | f[n], {sigma_expr});")
    lines.append(f"{pad}}} else {{")
    if is_m4:
        lines.append(
            f"{pad}  log_lik[n] = log_diff_exp("
            f"normal_lcdf(loq | f[n], {sigma_expr}), "
            f"normal_lcdf(0 | f[n], {sigma_expr}));"
        )
    else:
        lines.append(f"{pad}  log_lik[n] = normal_lcdf(loq | f[n], {sigma_expr});")
    lines.append(f"{pad}}}")
    return lines


def _emit_theta_unpack(spec: DSLSpec, indent: int = 4) -> list[str]:
    """Unpack theta array in the ODE function."""
    pad = " " * indent
    lines: list[str] = []
    names = spec.structural_param_names()
    for idx, name in enumerate(names, 1):
        lines.append(f"{pad}real {name} = theta[{idx}];")
    return lines


def _emit_state_aliases(spec: DSLSpec, indent: int = 4) -> list[str]:
    """Emit state variable aliases.

    Indices shift down by 1 when there is no depot compartment (IVBolus).
    """
    pad = " " * indent
    lines: list[str] = []
    if _needs_depot(spec):
        lines.append(f"{pad}real depot = y[1];")

    centr = _centr_idx(spec)
    dist = spec.distribution
    if isinstance(dist, (OneCmt, TMDDCore, TMDDQSS)):
        lines.append(f"{pad}real centr = y[{centr}];")
    elif isinstance(dist, TwoCmt):
        lines.append(f"{pad}real centr = y[{centr}];")
        lines.append(f"{pad}real periph = y[{centr + 1}];")
    elif isinstance(dist, ThreeCmt):
        lines.append(f"{pad}real centr = y[{centr}];")
        lines.append(f"{pad}real periph1 = y[{centr + 1}];")
        lines.append(f"{pad}real periph2 = y[{centr + 2}];")

    if isinstance(dist, TMDDCore):
        lines.append(f"{pad}real R = y[{centr + 1}];")
        lines.append(f"{pad}real RC = y[{centr + 2}];")
    elif isinstance(dist, TMDDQSS):
        lines.append(f"{pad}real Rtot = y[{centr + 1}];")

    return lines


def _emit_ode_dynamics(spec: DSLSpec, indent: int = 4) -> list[str]:
    """Emit ODE RHS in the functions block.

    For IVBolus absorption there is no depot compartment; dose is routed
    directly to central at event time, and no ``ka * depot`` term appears
    in the continuous dynamics. State indices shift accordingly.
    """
    pad = " " * indent
    lines: list[str] = []

    abs_mod = spec.absorption
    dist_mod = spec.distribution
    elim_mod = spec.elimination

    # Volume variable
    vol = "V" if isinstance(dist_mod, (OneCmt, TMDDCore, TMDDQSS)) else "V1"

    # Concentration
    lines.append(f"{pad}real conc = centr / {vol};")

    centr = _centr_idx(spec)

    # Absorption rate — IVBolus has no depot, no absorption phase
    if isinstance(abs_mod, IVBolus):
        abs_influx = "0"
    elif isinstance(abs_mod, FirstOrder):
        lines.append(f"{pad}dydt[1] = -ka * depot;")
        abs_influx = "ka * depot"
    elif isinstance(abs_mod, Transit):
        lines.append(f"{pad}real mtt = (n + 1) / ktr;")
        lines.append(f"{pad}// Transit compartment approximation")
        lines.append(f"{pad}real ktr_eff = (n + 1) / mtt;")
        lines.append(f"{pad}dydt[1] = ktr_eff * depot * exp(-ktr_eff * t) - ka * depot;")
        abs_influx = "ka * depot"
    else:
        # LaggedFirstOrder and any future first-order-like case
        lines.append(f"{pad}dydt[1] = -ka * depot;")
        abs_influx = "ka * depot"

    # Elimination expression
    elim_expr = _stan_elim_expr(elim_mod, "centr", vol)

    # Central compartment — index depends on whether a depot was emitted
    if isinstance(dist_mod, OneCmt):
        lines.append(f"{pad}dydt[{centr}] = {abs_influx} - {elim_expr};")
    elif isinstance(dist_mod, TwoCmt):
        lines.append(
            f"{pad}dydt[{centr}] = {abs_influx} - {elim_expr} - Q / V1 * centr + Q / V2 * periph;"
        )
        lines.append(f"{pad}dydt[{centr + 1}] = Q / V1 * centr - Q / V2 * periph;")
    elif isinstance(dist_mod, ThreeCmt):
        lines.append(
            f"{pad}dydt[{centr}] = {abs_influx} - {elim_expr}"
            f" - Q2 / V1 * centr + Q2 / V2 * periph1"
            f" - Q3 / V1 * centr + Q3 / V3 * periph2;"
        )
        lines.append(f"{pad}dydt[{centr + 1}] = Q2 / V1 * centr - Q2 / V2 * periph1;")
        lines.append(f"{pad}dydt[{centr + 2}] = Q3 / V1 * centr - Q3 / V3 * periph2;")
    elif isinstance(dist_mod, TMDDCore):
        lines.append(f"{pad}real kel = CL / {vol};")
        lines.append(f"{pad}real kdeg = koff;  // receptor degradation ~ koff")
        lines.append(f"{pad}real ksyn = kdeg * R0;  // receptor synthesis at steady state")
        lines.append(
            f"{pad}dydt[{centr}] = {abs_influx} - kel * centr"
            f" - kon * conc * R * {vol} + koff * RC * {vol};"
        )
        lines.append(f"{pad}dydt[{centr + 1}] = ksyn - kdeg * R - kon * conc * R + koff * RC;")
        lines.append(f"{pad}dydt[{centr + 2}] = kon * conc * R - koff * RC - kint * RC;")
    elif isinstance(dist_mod, TMDDQSS):
        lines.append(f"{pad}real kel = CL / {vol};")
        lines.append(f"{pad}real kdeg = kint;  // receptor degradation initial estimate")
        lines.append(f"{pad}real ksyn = kdeg * R0;  // receptor synthesis at steady state")
        lines.append(
            f"{pad}real Cfree = 0.5 * ((conc - Rtot - KD)"
            f" + sqrt(square(conc - Rtot - KD) + 4 * KD * conc));"
        )
        lines.append(f"{pad}real Rfree = Rtot * KD / (KD + Cfree);")
        lines.append(f"{pad}real RC_conc = conc - Cfree;")
        lines.append(
            f"{pad}dydt[{centr}] = {abs_influx} - kel * Cfree * {vol} - kint * RC_conc * {vol};"
        )
        lines.append(f"{pad}dydt[{centr + 1}] = ksyn - kdeg * Rfree - kint * RC_conc;")

    return lines


def _stan_elim_expr(elim_mod: object, cmt: str, vol: str) -> str:
    if isinstance(elim_mod, LinearElim):
        return f"CL / {vol} * {cmt}"
    if isinstance(elim_mod, MichaelisMenten):
        return f"Vmax * ({cmt}/{vol}) / (Km + {cmt}/{vol})"
    if isinstance(elim_mod, ParallelLinearMM):
        return f"(CL / {vol} * {cmt} + Vmax * ({cmt}/{vol}) / (Km + {cmt}/{vol}))"
    if isinstance(elim_mod, TimeVaryingElim):
        # Plan §4 / #9: three decay forms supported in Stan ODE RHS.
        #   exponential: CL(t) = CL * exp(-kdecay * t)
        #   half_life:   CL(t) = CL / (1 + kdecay * t)
        #   linear:      CL(t) = fmax(CL * (1 - kdecay * t), 0)
        if elim_mod.decay_fn == "half_life":
            return f"CL / (1 + kdecay * t) / {vol} * {cmt}"
        if elim_mod.decay_fn == "linear":
            return f"fmax(CL * (1 - kdecay * t), 0.0) / {vol} * {cmt}"
        return f"CL * exp(-kdecay * t) / {vol} * {cmt}"
    return f"CL / {vol} * {cmt}"


def _emit_ode_solve(spec: DSLSpec, indent: int = 4) -> list[str]:
    """Emit piecewise ODE solve with event-driven dose injection.

    For each subject: iterate through their events in time order.
    Between consecutive event times, integrate the ODE forward.
    At dose events (EVID=1), add AMT to the appropriate state.
    At reset events (EVID=3), zero all states.
    At reset+dose (EVID=4), zero then add.
    At observation times, extract predicted concentration.
    """
    pad = " " * indent
    lines: list[str] = []
    n_states = _n_states(spec)
    n_params = len(spec.structural_param_names())

    # Pack theta
    lines.append(f"{pad}array[{n_params}] real theta_i;")
    for idx, name in enumerate(spec.structural_param_names(), 1):
        lines.append(f"{pad}theta_i[{idx}] = {name}_i;")

    # Initial state
    lines.append(f"{pad}vector[{n_states}] y_state = rep_vector(0, {n_states});")
    lines.append(f"{pad}real t_prev = 0.0;")

    # Volume for concentration conversion
    vol = "V" if isinstance(spec.distribution, (OneCmt, TMDDCore, TMDDQSS)) else "V1"

    lines.append("")
    lines.append(f"{pad}// Merged chronological pass: interleave dose events and observations")
    lines.append(f"{pad}// Process dose events before observations at the same time")
    lines.append(f"{pad}int e_idx = event_start[i];  // current dose event cursor")
    lines.append(f"{pad}for (n in 1:N) {{")
    lines.append(f"{pad}  if (subject[n] == i) {{")
    lines.append(f"{pad}    // Apply all pending dose events up to (and including) this obs time")
    lines.append(f"{pad}    while (e_idx >= 1 && e_idx <= event_end[i]")
    lines.append(f"{pad}           && event_time[e_idx] <= time[n]) {{")
    lines.append(f"{pad}      // Integrate to event time")
    lines.append(f"{pad}      if (event_time[e_idx] > t_prev) {{")
    lines.append(f"{pad}        array[1] real ts_e = {{event_time[e_idx]}};")
    lines.append(f"{pad}        array[1] vector[{n_states}] y_e =")
    lines.append(f"{pad}          ode_rk45(ode_rhs, y_state, t_prev, ts_e, theta_i, x_r, x_i);")
    lines.append(f"{pad}        y_state = y_e[1];")
    lines.append(f"{pad}        t_prev = event_time[e_idx];")
    lines.append(f"{pad}      }}")
    lines.append(f"{pad}      // Apply reset (EVID=3 or 4)")
    lines.append(f"{pad}      if (event_evid[e_idx] == 3 || event_evid[e_idx] == 4)")
    lines.append(f"{pad}        y_state = rep_vector(0, {n_states});")
    lines.append(f"{pad}      // Apply dose (EVID=1 or 4)")
    lines.append(f"{pad}      if (event_evid[e_idx] == 1 || event_evid[e_idx] == 4) {{")
    lines.append(f"{pad}        if (event_cmt[e_idx] <= {n_states})")
    lines.append(f"{pad}          y_state[event_cmt[e_idx]] += event_amt[e_idx];")
    lines.append(f"{pad}      }}")
    lines.append(f"{pad}      e_idx += 1;")
    lines.append(f"{pad}    }}")
    lines.append(f"{pad}    // Integrate to observation time")
    lines.append(f"{pad}    if (time[n] > t_prev) {{")
    lines.append(f"{pad}      array[1] real ts_o = {{time[n]}};")
    lines.append(f"{pad}      array[1] vector[{n_states}] y_o =")
    lines.append(f"{pad}        ode_rk45(ode_rhs, y_state, t_prev, ts_o, theta_i, x_r, x_i);")
    lines.append(f"{pad}      y_state = y_o[1];")
    lines.append(f"{pad}      t_prev = time[n];")
    lines.append(f"{pad}    }}")

    centr = _centr_idx(spec)
    if isinstance(spec.distribution, TMDDQSS):
        lines.append(f"{pad}    real Ctot_n = y_state[{centr}] / {vol}_i;")
        lines.append(f"{pad}    real Rtot_n = y_state[{centr + 1}];")
        lines.append(
            f"{pad}    f[n] = fmax(0.5 * ((Ctot_n - Rtot_n - KD_i)"
            f" + sqrt(square(Ctot_n - Rtot_n - KD_i)"
            f" + 4 * KD_i * Ctot_n)), 1e-10);"
        )
    else:
        lines.append(f"{pad}    f[n] = fmax(y_state[{centr}] / {vol}_i, 1e-10);")

    lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")

    return lines


def _emit_analytical_solve(spec: DSLSpec, indent: int = 4) -> list[str]:
    """Emit analytical solution with superposition for multi-dose linear models.

    Uses the superposition principle: for linear time-invariant systems,
    the response to multiple doses is the sum of individual dose responses.
    At each observation time, sum contributions from all prior dose events.
    """
    pad = " " * indent
    lines: list[str] = []

    abs_mod = spec.absorption
    dist_mod = spec.distribution
    elim_mod = spec.elimination

    if not isinstance(elim_mod, LinearElim):
        return _emit_ode_solve(spec, indent)

    lines.append(f"{pad}// Analytical solution with superposition for multi-dose")
    lines.append(f"{pad}// Track last reset time to invalidate prior dose contributions")
    lines.append(f"{pad}real t_last_reset = 0.0;")
    lines.append(f"{pad}if (event_start[i] > 0) {{")
    lines.append(f"{pad}  for (e in event_start[i]:event_end[i]) {{")
    lines.append(f"{pad}    if (event_evid[e] == 3 || event_evid[e] == 4)")
    lines.append(f"{pad}      t_last_reset = event_time[e];")
    lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")
    lines.append(f"{pad}for (n in 1:N) {{")
    lines.append(f"{pad}  if (subject[n] == i) {{")
    lines.append(f"{pad}    real t_obs = time[n];")
    lines.append(f"{pad}    real conc = 0.0;")

    if isinstance(dist_mod, OneCmt) and isinstance(abs_mod, (FirstOrder, LaggedFirstOrder)):
        lines.append(f"{pad}    real ke = CL_i / V_i;")
        lines.append(f"{pad}    // Superposition: sum contributions from doses after last reset")
        lines.append(f"{pad}    if (event_start[i] > 0) {{")
        lines.append(f"{pad}      for (e in event_start[i]:event_end[i]) {{")
        lines.append(
            f"{pad}        if (event_evid[e] == 1 && event_time[e] <= t_obs"
            f" && event_time[e] >= t_last_reset) {{"
        )
        lines.append(f"{pad}          real t_since_dose = t_obs - event_time[e];")
        if isinstance(abs_mod, LaggedFirstOrder):
            lines.append(f"{pad}          real t_eff = fmax(t_since_dose - tlag_i, 0);")
            t_var = "t_eff"
        else:
            t_var = "t_since_dose"
        lines.append(
            f"{pad}          conc += event_amt[e] * ka_i / (V_i * (ka_i - ke))"
            f" * (exp(-ke * {t_var}) - exp(-ka_i * {t_var}));"
        )
        lines.append(f"{pad}        }}")
        lines.append(f"{pad}      }}")
        lines.append(f"{pad}    }}")
        lines.append(f"{pad}    f[n] = fmax(conc, 1e-10);")

    elif isinstance(dist_mod, TwoCmt) and isinstance(abs_mod, FirstOrder):
        lines.append(f"{pad}    real ke = CL_i / V1_i;")
        lines.append(f"{pad}    real k12 = Q_i / V1_i;")
        lines.append(f"{pad}    real k21 = Q_i / V2_i;")
        lines.append(
            f"{pad}    real a1 = 0.5 * ((ke + k12 + k21)"
            f" + sqrt(square(ke + k12 + k21) - 4 * ke * k21));"
        )
        lines.append(
            f"{pad}    real a2 = 0.5 * ((ke + k12 + k21)"
            f" - sqrt(square(ke + k12 + k21) - 4 * ke * k21));"
        )
        lines.append(f"{pad}    // Superposition: sum contributions from doses after last reset")
        lines.append(f"{pad}    if (event_start[i] > 0) {{")
        lines.append(f"{pad}      for (e in event_start[i]:event_end[i]) {{")
        lines.append(
            f"{pad}        if (event_evid[e] == 1 && event_time[e] <= t_obs"
            f" && event_time[e] >= t_last_reset) {{"
        )
        lines.append(f"{pad}          real td = t_obs - event_time[e];")
        lines.append(f"{pad}          real Ae = event_amt[e] * ka_i / V1_i;")
        lines.append(
            f"{pad}          conc += Ae * ((k21 - a1) / ((ka_i - a1) * (a2 - a1)) * exp(-a1 * td)"
        )
        lines.append(f"{pad}            + (k21 - a2) / ((ka_i - a2) * (a1 - a2)) * exp(-a2 * td)")
        lines.append(
            f"{pad}            + (k21 - ka_i) / ((a1 - ka_i) * (a2 - ka_i)) * exp(-ka_i * td));"
        )
        lines.append(f"{pad}        }}")
        lines.append(f"{pad}      }}")
        lines.append(f"{pad}    }}")
        lines.append(f"{pad}    f[n] = fmax(conc, 1e-10);")
    else:
        return _emit_ode_solve(spec, indent)

    lines.append(f"{pad}  }}")
    lines.append(f"{pad}}}")

    return lines
