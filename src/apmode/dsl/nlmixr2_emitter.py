# SPDX-License-Identifier: GPL-2.0-or-later
"""nlmixr2 lowering emitter: DSL AST → R code strings (ARCHITECTURE.md §2.2).

Emits a complete nlmixr2 model function with:
- ini({}) block: parameter initial estimates, eta definitions, sigma definitions
- model({}) block: rxode2 ODE/algebraic model, covariate effects, observation model

NODE modules are Phase 2 and raise NotImplementedError.

References for ODE formulations:
- TMDD full binding: Mager & Jusko (2001), J Pharmacokinet Pharmacodyn 28:507-532
- TMDD QSS: Gibiansky et al. (2008), J Pharmacokinet Pharmacodyn 35:573-591
- Transit compartments: Savic et al. (2007), J Pharmacokinet Pharmacodyn 34:711-726
- BLQ M3/M4: nlmixr2 censoring via CENS/LIMIT data columns
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
    DSLSpec,
    Erlang,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    ObservationModule,
    OccasionByDoseEpoch,
    OccasionByStudy,
    OccasionByVisit,
    OccasionCustom,
    OneCmt,
    ParallelFirstOrder,
    ParallelLinearMM,
    Proportional,
    SumIG,
    ThreeCmt,
    TimeVaryingElim,
    TMDDCore,
    Transit,
    TwoCmt,
    ZeroOrder,
)
from apmode.dsl.capabilities import CapabilityTag

# Valid R identifier pattern for name sanitization
_R_IDENT_RE = re.compile(r"^[a-zA-Z_.][a-zA-Z0-9_.]*$")

# Capability matrix (P0.7, docs/plans/2026-07-08-formular-sharpening-and-adoption-design.md
# §4 Phase 0): every module-axis variant this emitter has a real lowering
# path for, versus what it explicitly rejects. NODE modules are Phase 2
# (JAX/Diffrax emitter) and raise via ``spec.has_node_modules()`` above.
SUPPORTS: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.ABSORPTION_IV_BOLUS,
        CapabilityTag.ABSORPTION_FIRST_ORDER,
        CapabilityTag.ABSORPTION_ZERO_ORDER,
        CapabilityTag.ABSORPTION_LAGGED_FIRST_ORDER,
        CapabilityTag.ABSORPTION_TRANSIT,
        CapabilityTag.ABSORPTION_MIXED_FIRST_ZERO,
        CapabilityTag.ABSORPTION_ERLANG,
        CapabilityTag.ABSORPTION_PARALLEL_FIRST_ORDER,
        CapabilityTag.ABSORPTION_SUM_IG,
        CapabilityTag.DISTRIBUTION_ONE_CMT,
        CapabilityTag.DISTRIBUTION_TWO_CMT,
        CapabilityTag.DISTRIBUTION_THREE_CMT,
        CapabilityTag.DISTRIBUTION_TMDD_CORE,
        CapabilityTag.DISTRIBUTION_TMDD_QSS,
        CapabilityTag.ELIMINATION_LINEAR,
        CapabilityTag.ELIMINATION_MICHAELIS_MENTEN,
        CapabilityTag.ELIMINATION_PARALLEL_LINEAR_MM,
        CapabilityTag.ELIMINATION_TIME_VARYING,
        CapabilityTag.VARIABILITY_IIV,
        CapabilityTag.VARIABILITY_IIV_BLOCK_STRUCTURE,
        CapabilityTag.VARIABILITY_IOV,
        CapabilityTag.VARIABILITY_COVARIATE_LINK,
        CapabilityTag.VARIABILITY_COVARIATE_MATURATION_FORM,
        CapabilityTag.OBSERVATION_PROPORTIONAL,
        CapabilityTag.OBSERVATION_ADDITIVE,
        CapabilityTag.OBSERVATION_COMBINED,
        CapabilityTag.OBSERVATION_BLQ_M3,
        CapabilityTag.OBSERVATION_BLQ_M4,
        CapabilityTag.OBSERVATION_MULTI_ANALYTE,
    }
)

EXPLICITLY_UNSUPPORTED: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.ABSORPTION_NODE,
        CapabilityTag.ELIMINATION_NODE,
    }
)


def _sanitize_r_name(name: str) -> str:
    """Validate that a name is safe for use in R code generation."""
    if not _R_IDENT_RE.match(name):
        msg = f"Invalid R identifier: {name!r}"
        raise ValueError(msg)
    return name


def emit_nlmixr2(
    spec: DSLSpec,
    initial_estimates: dict[str, float] | None = None,
) -> str:
    """Emit a complete nlmixr2 model function from a DSLSpec.

    Args:
        spec: The compiled DSL specification.
        initial_estimates: Optional parameter name -> value overrides for the
            ini() block. When provided, these values replace the DSLSpec defaults.
            Keys are structural parameter names (e.g. "CL", "V", "ka").

    Returns an R code string defining an nlmixr2-compatible model function
    with ini() and model() blocks.

    Raises NotImplementedError for NODE modules (Phase 2).
    """
    if spec.has_node_modules():
        raise NotImplementedError(
            "NODE module lowering to nlmixr2 is not supported. "
            "NODE backends use the JAX/Diffrax emitter (Phase 2)."
        )

    ini_lines = _emit_ini(spec, initial_estimates=initial_estimates)
    model_lines = _emit_model(spec)

    lines = [
        f"# APMODE generated model: {spec.model_id}",
        "function() {",
        "  ini({",
        *[f"    {line}" for line in ini_lines],
        "  })",
        "  model({",
        *[f"    {line}" for line in model_lines],
        "  })",
        "}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ini() block emission
# ---------------------------------------------------------------------------


def _emit_ini(
    spec: DSLSpec,
    initial_estimates: dict[str, float] | None = None,
) -> list[str]:
    """Emit the ini({}) block: structural params, etas, sigmas."""
    lines: list[str] = []

    lines.append("# Structural parameters")
    lines.extend(_emit_structural_ini(spec, initial_estimates=initial_estimates))

    lines.append("")
    lines.append("# Inter-individual variability")
    lines.extend(_emit_variability_ini(spec))

    lines.append("")
    lines.append("# Residual error")
    lines.extend(_emit_sigma_ini(spec))

    return lines


def _emit_structural_ini(
    spec: DSLSpec,
    initial_estimates: dict[str, float] | None = None,
) -> list[str]:
    """Emit structural parameter initial estimates.

    Calibration values come from ``spec.initial`` (Formular sharpening plan
    §4 Phase 1, P1.4); when ``initial_estimates`` is also provided, those
    values take precedence for matching parameter names (e.g. runtime
    NCA-derived overrides beat the DSL-declared ``initial:`` block).
    """
    ov = initial_estimates or {}
    lines: list[str] = []
    abs_mod = spec.absorption
    dist_mod = spec.distribution
    elim_mod = spec.elimination

    def val(name: str, default: float = 1.0) -> float:
        return ov.get(name, spec.initial.get(name, default))

    # --- Absorption ---
    if isinstance(abs_mod, IVBolus):
        # IV bolus: no absorption parameters. Dose is routed directly to
        # the central compartment; depot is omitted by the structural emitter.
        pass
    elif isinstance(abs_mod, FirstOrder):
        lines.append(f"lka <- log({val('ka')})")
    elif isinstance(abs_mod, ZeroOrder):
        lines.append(f"ldur <- log({val('dur')})")
    elif isinstance(abs_mod, LaggedFirstOrder):
        lines.append(f"lka <- log({val('ka')})")
        tlag = val("tlag", 0.0)
        lines.append(f"ltlag <- log({tlag})" if tlag > 0 else "ltlag <- -10")
    elif isinstance(abs_mod, Transit):
        lines.append(f"lka <- log({val('ka')})")
        lines.append(f"lktr <- log({val('ktr')})")
        # n is estimated as continuous via log/exp transform; rxode2's transit()
        # uses gamma-function interpolation for non-integer n values
        # (Savic et al. 2007, J Pharmacokinet Pharmacodyn 34:711-726)
        lines.append(f"ln <- log({ov.get('n', abs_mod.n)})")
    elif isinstance(abs_mod, MixedFirstZero):
        lines.append(f"lka <- log({val('ka')})")
        lines.append(f"ldur <- log({val('dur')})")
        # #14: frac == 1.0 (perfect bioavailability) produced a
        # ZeroDivisionError on log(1 / 0) at emit time. Clamp to the
        # 99.99% ceiling and warn — the user can always drop the
        # ZeroOrder leg or use FirstOrder if they truly want fraction=1.
        frac_raw = val("frac", 0.5)
        _frac_epsilon = 1e-4
        frac_clamped = min(max(frac_raw, _frac_epsilon), 1.0 - _frac_epsilon)
        if frac_clamped != frac_raw:
            lines.append(
                f"# frac clamped from {frac_raw} to {frac_clamped} "
                f"to avoid singular logit (APMODE #14)"
            )
        lines.append(f"logit_frac <- log({frac_clamped} / (1 - {frac_clamped}))")
    elif isinstance(abs_mod, Erlang):
        # n is structural-integer (set by transform, not estimated). Only ktr
        # is exposed for IIV/priors/covariates. See ADR-0003 D2.
        lines.append(f"lktr <- log({val('ktr')})")
    elif isinstance(abs_mod, ParallelFirstOrder):
        lines.append(f"lka1 <- log({val('ka1')})")
        lines.append(f"lka2 <- log({val('ka2')})")
        frac_raw = val("frac", 0.5)
        _frac_epsilon = 1e-4
        frac_clamped = min(max(frac_raw, _frac_epsilon), 1.0 - _frac_epsilon)
        if frac_clamped != frac_raw:
            lines.append(
                f"# frac clamped from {frac_raw} to {frac_clamped} to avoid singular logit"
            )
        lines.append(f"logit_frac <- log({frac_clamped} / (1 - {frac_clamped}))")
    elif isinstance(abs_mod, SumIG):
        # Per-component params on log scale; weight_1 on logit scale.
        # Positive-difference parameterisation for MT_2 (delta = MT_2 - MT_1)
        # prevents label switching during FOCEI.
        mt_1 = val("MT_1")
        mt_2 = val("MT_2")
        lines.append(f"lMT_1 <- log({mt_1})")
        delta = mt_2 - mt_1
        lines.append(f"ldelta_MT_2 <- log({max(delta, 1e-6)})  # MT_2 = MT_1 + exp(ldelta_MT_2)")
        lines.append(f"lRD2_1 <- log({val('RD2_1')})")
        lines.append(f"lRD2_2 <- log({val('RD2_2')})")
        weight_raw = val("weight_1", 0.5)
        _weight_epsilon = 1e-4
        weight_clamped = min(max(weight_raw, _weight_epsilon), 1.0 - _weight_epsilon)
        if weight_clamped != weight_raw:
            lines.append(
                f"# weight_1 clamped from {weight_raw} to {weight_clamped} to avoid singular logit"
            )
        lines.append(f"logit_weight_1 <- log({weight_clamped} / (1 - {weight_clamped}))")

    # --- Distribution ---
    if isinstance(dist_mod, OneCmt):
        lines.append(f"lV <- log({val('V')})")
    elif isinstance(dist_mod, TwoCmt):
        lines.append(f"lV1 <- log({val('V1')})")
        lines.append(f"lV2 <- log({val('V2')})")
        lines.append(f"lQ <- log({val('Q')})")
    elif isinstance(dist_mod, ThreeCmt):
        lines.append(f"lV1 <- log({val('V1')})")
        lines.append(f"lV2 <- log({val('V2')})")
        lines.append(f"lV3 <- log({val('V3')})")
        lines.append(f"lQ2 <- log({val('Q2')})")
        lines.append(f"lQ3 <- log({val('Q3')})")
    elif isinstance(dist_mod, TMDDCore):
        lines.append(f"lV <- log({val('V')})")
        lines.append(f"lR0 <- log({val('R0')})")
        lines.append(f"lkon <- log({val('kon')})")
        lines.append(f"lkoff <- log({val('koff')})")
        lines.append(f"lkint <- log({val('kint')})")
    elif isinstance(dist_mod, TMDDQSS):
        lines.append(f"lV <- log({val('V')})")
        lines.append(f"lR0 <- log({val('R0')})")
        lines.append(f"lKD <- log({val('KD')})")
        lines.append(f"lkint <- log({val('kint')})")

    # --- Elimination ---
    if isinstance(elim_mod, LinearElim):
        lines.append(f"lCL <- log({val('CL')})")
    elif isinstance(elim_mod, MichaelisMenten):
        lines.append(f"lVmax <- log({val('Vmax')})")
        lines.append(f"lKm <- log({val('Km')})")
    elif isinstance(elim_mod, ParallelLinearMM):
        lines.append(f"lCL <- log({val('CL')})")
        lines.append(f"lVmax <- log({val('Vmax')})")
        lines.append(f"lKm <- log({val('Km')})")
    elif isinstance(elim_mod, TimeVaryingElim):
        # All three decay forms (exponential | half_life | linear) are
        # supported as of v0.5.0 — the per-form ODE RHS is emitted by
        # ``_elim_rate_expr`` (see lines ~610-620). This block only
        # writes the log-parameter scaffolding shared by all forms.
        lines.append(f"lCL <- log({val('CL')})")
        lines.append(f"lkdecay <- log({val('kdecay', 0.1)})")

    # Covariate coefficients. Initial/starting values come from the
    # covariate declaration's own ``theta``/``hill``/``tm50`` fields
    # (Formular sharpening plan §4 Phase 1, P1.6) rather than a hardcoded
    # per-form constant — ``power``/``exponential``/``linear`` use
    # ``theta``; ``maturation`` uses ``hill`` (coefficient) and ``tm50``.
    # ``categorical`` has no configurable coefficient yet (Phase 2
    # candidate; see ``CovariateLink`` docstring) and keeps the pre-P1.6
    # hardcoded starting value of 0.
    cov_links = list(spec.covariates)
    if cov_links:
        lines.append("")
        lines.append("# Covariate coefficients")
        for cov in cov_links:
            p = _sanitize_r_name(cov.param)
            c = _sanitize_r_name(cov.covariate)
            coeff_name = f"beta_{p}_{c}"
            if cov.form in ("power", "exponential", "linear"):
                lines.append(f"{coeff_name} <- {cov.theta}")
            elif cov.form == "maturation":
                lines.append(f"{coeff_name} <- {cov.hill}")
                lines.append(f"TM50_{p}_{c} <- {cov.tm50}")
            else:
                lines.append(f"{coeff_name} <- 0")

    return lines


def _emit_variability_ini(spec: DSLSpec) -> list[str]:
    """Emit IIV/IOV eta definitions in the ini block."""
    lines: list[str] = []

    for item in spec.variability:
        if isinstance(item, IIV):
            if item.structure == "diagonal":
                for param in item.params:
                    p = _sanitize_r_name(param)
                    lines.append(f"eta.{p} ~ 0.1")
            elif item.structure == "block":
                n = len(item.params)
                eta_names = " + ".join(f"eta.{_sanitize_r_name(p)}" for p in item.params)
                lines.append(f"{eta_names} ~ c(")
                # Lower-triangular covariance matrix initial values
                entries: list[str] = []
                for i in range(n):
                    for j in range(i + 1):
                        entries.append("0.1" if i == j else "0.01")
                lines.append(f"  {', '.join(entries)}")
                lines.append(")")
        elif isinstance(item, IOV):
            col = _get_occasion_column(item)
            for param in item.params:
                p = _sanitize_r_name(param)
                # nlmixr2 IOV syntax: eta ~ variance | occ(column)
                lines.append(f"eta.iov.{p} ~ 0.05 | occ({col})")

    return lines


def _endpoint_sigma_suffix(endpoint_name: str) -> str:
    """Return the sigma-variable-name suffix for one multi-analyte endpoint.

    Empty string for the synthetic single-endpoint case (``"default"`` —
    see ``DSLSpec.observation_endpoints``) so single-endpoint specs keep
    emitting the pre-P1.7 bare ``prop.sd``/``add.sd`` names unchanged; a
    real ``.<name>`` suffix otherwise, so two endpoints never collide on
    the same sigma variable in the ini block.
    """
    if endpoint_name == "default":
        return ""
    return f".{_sanitize_r_name(endpoint_name)}"


def _emit_endpoint_sigma_ini(obs: ObservationModule, suffix: str) -> list[str]:
    """Emit residual error sigma definitions for one endpoint's error module."""
    if isinstance(obs, Proportional):
        return [f"prop.sd{suffix} <- {obs.sigma_prop}"]
    elif isinstance(obs, Additive):
        return [f"add.sd{suffix} <- {obs.sigma_add}"]
    elif isinstance(obs, Combined):
        return [
            f"prop.sd{suffix} <- {obs.sigma_prop}",
            f"add.sd{suffix} <- {obs.sigma_add}",
        ]
    elif isinstance(obs, (BLQM3, BLQM4)):
        # BLQ composes with underlying error model; censoring is data-driven
        if obs.error_model == "proportional":
            return [f"prop.sd{suffix} <- {obs.sigma_prop}"]
        elif obs.error_model == "additive":
            return [f"add.sd{suffix} <- {obs.sigma_add}"]
        else:  # combined
            return [
                f"prop.sd{suffix} <- {obs.sigma_prop}",
                f"add.sd{suffix} <- {obs.sigma_add}",
            ]
    return []


def _emit_sigma_ini(spec: DSLSpec) -> list[str]:
    """Emit residual error sigma definitions for every observation endpoint.

    Formular sharpening plan §4 Phase 1 (P1.7): iterates
    ``spec.observation_endpoints()`` — the single unified accessor that
    normalizes both the legacy singular ``observation:`` sugar and the
    multi-analyte ``observations:`` block — rather than reading
    ``spec.observation`` directly, so this emits identical R code to the
    pre-P1.7 implementation for every existing single-endpoint spec (one
    synthetic ``"default"`` endpoint, empty sigma suffix).
    """
    lines: list[str] = []
    for endpoint in spec.observation_endpoints():
        lines.extend(
            _emit_endpoint_sigma_ini(endpoint.error, _endpoint_sigma_suffix(endpoint.name))
        )
    return lines


# ---------------------------------------------------------------------------
# model() block emission
# ---------------------------------------------------------------------------


def _get_occasion_column(iov: IOV) -> str:
    """Extract the data column name for IOV occasion indexing.

    nlmixr2 requires an occ(column) statement in the model block to map
    IOV etas to occasion-defining data columns.
    """
    occ = iov.occasions
    if isinstance(occ, OccasionByStudy):
        return "STUDY_ID"  # canonical schema column name (PRD §4.2.0)
    elif isinstance(occ, (OccasionByVisit, OccasionByDoseEpoch, OccasionCustom)):
        return _sanitize_r_name(occ.column)
    return "OCC"  # fallback


def _emit_iov_occasion(spec: DSLSpec) -> list[str]:
    """Emit IOV occasion context for nlmixr2 model block.

    In nlmixr2 >= 2.1, IOV occasion binding is specified in the ini block
    via the pipe syntax: ``eta.iov.CL ~ 0.05 | occ(COLUMN)``.
    No standalone ``occ()`` call is needed in the model block.
    This function emits only a comment for documentation/traceability.
    """
    lines: list[str] = []
    for item in spec.variability:
        if isinstance(item, IOV):
            col = _get_occasion_column(item)
            lines.append(f"# IOV bound to occasion column: {col} (specified in ini block)")
    return lines


def _emit_model(spec: DSLSpec) -> list[str]:
    """Emit the model({}) block: back-transforms, ODEs, observation model."""
    lines: list[str] = []

    # IOV occasion column specification (must precede back-transforms)
    iov_lines = _emit_iov_occasion(spec)
    if iov_lines:
        lines.extend(iov_lines)
        lines.append("")

    lines.append("# Back-transform parameters")
    lines.extend(_emit_backtransform(spec))

    lines.append("")
    lines.append("# Compartment dynamics")
    lines.extend(_emit_dynamics(spec))

    lines.append("")
    lines.append("# Observation model")
    lines.extend(_emit_observation_model(spec))

    return lines


def _emit_backtransform(spec: DSLSpec) -> list[str]:
    """Emit parameter back-transformations from log-domain."""
    lines: list[str] = []

    # Collect IIV/IOV params
    iiv_params: set[str] = set()
    iov_params: set[str] = set()
    for item in spec.variability:
        if isinstance(item, IIV):
            iiv_params.update(item.params)
        elif isinstance(item, IOV):
            iov_params.update(item.params)

    cov_links = list(spec.covariates)

    def _bt(param: str, log_name: str) -> str:
        """Build back-transform expression with eta and covariate effects."""
        expr = log_name
        if param in iiv_params:
            expr += f" + eta.{param}"
        if param in iov_params:
            expr += f" + eta.iov.{param}"
        for cov in cov_links:
            if cov.param == param:
                coeff = f"beta_{cov.param}_{cov.covariate}"
                if cov.form == "power":
                    # ``ref`` is the covariate's fixed reference value
                    # (Formular sharpening plan §4 P1.6, e.g. 70 kg per
                    # Anderson & Holford 2008, Clin Pharmacokinet 47:455-467)
                    expr += f" + {coeff} * log({cov.covariate} / {cov.ref})"
                elif cov.form == "exponential":
                    expr += f" + {coeff} * {cov.covariate}"
                elif cov.form == "linear":
                    expr += f" + log(1 + {coeff} * {cov.covariate})"
                elif cov.form == "categorical":
                    expr += f" + {coeff} * {cov.covariate}"
                elif cov.form == "maturation":
                    tm50 = f"TM50_{cov.param}_{cov.covariate}"
                    expr += (
                        f" + log({cov.covariate}^{coeff} / "
                        f"({cov.covariate}^{coeff} + {tm50}^{coeff}))"
                    )
        return f"{param} <- exp({expr})"

    def _bt_logit(param: str, logit_name: str) -> str:
        """Build logit-domain back-transform with eta and covariate effects.

        For parameters constrained to (0, 1) like bioavailability fraction.

        #12: mirror the cov.form routing from :func:`_bt` so that power /
        exponential / linear / maturation relationships are not silently
        flattened to linear-additive on the logit scale. The functional
        forms below all target the *logit* (unbounded) scale so ``exp(expr)``
        is applied in the back-transform at the end; power uses the same
        ``cov.ref`` reference value as in ``_bt`` (Formular sharpening plan
        §4 P1.6).
        """
        expr = logit_name
        if param in iiv_params:
            expr += f" + eta.{param}"
        if param in iov_params:
            expr += f" + eta.iov.{param}"
        # Covariate effects on logit scale. The logit is unbounded so
        # each cov.form maps naturally from its _bt counterpart:
        #   - power:      β·log(cov / ref)        (log-linear on odds)
        #   - exponential: β·cov                   (linear on odds)
        #   - linear:     log(1 + β·cov)          (matches _bt — non-negative effect)
        #   - categorical: β·cov                   (indicator on odds)
        #   - maturation: log(cov^β / (cov^β + TM50^β))
        for cov in cov_links:
            if cov.param == param:
                coeff = f"beta_{cov.param}_{cov.covariate}"
                if cov.form == "power":
                    expr += f" + {coeff} * log({cov.covariate} / {cov.ref})"
                elif cov.form == "exponential":
                    expr += f" + {coeff} * {cov.covariate}"
                elif cov.form == "linear":
                    expr += f" + log(1 + {coeff} * {cov.covariate})"
                elif cov.form == "categorical":
                    expr += f" + {coeff} * {cov.covariate}"
                elif cov.form == "maturation":
                    tm50 = f"TM50_{cov.param}_{cov.covariate}"
                    expr += (
                        f" + log({cov.covariate}^{coeff} / "
                        f"({cov.covariate}^{coeff} + {tm50}^{coeff}))"
                    )
        return f"{param} <- 1 / (1 + exp(-({expr})))"

    abs_mod = spec.absorption
    dist_mod = spec.distribution
    elim_mod = spec.elimination

    # Absorption
    if isinstance(abs_mod, IVBolus):
        pass  # no absorption parameters to back-transform
    elif isinstance(abs_mod, FirstOrder):
        lines.append(_bt("ka", "lka"))
    elif isinstance(abs_mod, ZeroOrder):
        lines.append(_bt("dur", "ldur"))
    elif isinstance(abs_mod, LaggedFirstOrder):
        lines.append(_bt("ka", "lka"))
        lines.append(_bt("tlag", "ltlag"))
    elif isinstance(abs_mod, Transit):
        lines.append(_bt("ka", "lka"))
        lines.append(_bt("ktr", "lktr"))
        lines.append("n <- exp(ln)")
        lines.append("mtt <- (n + 1) / ktr  # mean transit time for rxode2")
    elif isinstance(abs_mod, MixedFirstZero):
        lines.append(_bt("ka", "lka"))
        lines.append(_bt("dur", "ldur"))
        lines.append(_bt_logit("frac", "logit_frac"))
    elif isinstance(abs_mod, Erlang):
        lines.append(_bt("ktr", "lktr"))
    elif isinstance(abs_mod, ParallelFirstOrder):
        lines.append(_bt("ka1", "lka1"))
        lines.append(_bt("ka2", "lka2"))
        lines.append(_bt_logit("frac", "logit_frac"))
    elif isinstance(abs_mod, SumIG):
        lines.append(_bt("MT_1", "lMT_1"))
        # Positive-difference parameterisation: MT_2 = MT_1 + exp(ldelta_MT_2)
        # ensures MT_2 > MT_1 by construction.
        lines.append("delta_MT_2 <- exp(ldelta_MT_2)")
        lines.append("MT_2 <- MT_1 + delta_MT_2")
        lines.append(_bt("RD2_1", "lRD2_1"))
        lines.append(_bt("RD2_2", "lRD2_2"))
        lines.append(_bt_logit("weight_1", "logit_weight_1"))
        lines.append("weight_2 <- 1 - weight_1  # implicit second weight")

    # Distribution
    if isinstance(dist_mod, OneCmt):
        lines.append(_bt("V", "lV"))
    elif isinstance(dist_mod, TwoCmt):
        lines.append(_bt("V1", "lV1"))
        lines.append(_bt("V2", "lV2"))
        lines.append(_bt("Q", "lQ"))
    elif isinstance(dist_mod, ThreeCmt):
        lines.append(_bt("V1", "lV1"))
        lines.append(_bt("V2", "lV2"))
        lines.append(_bt("V3", "lV3"))
        lines.append(_bt("Q2", "lQ2"))
        lines.append(_bt("Q3", "lQ3"))
    elif isinstance(dist_mod, TMDDCore):
        lines.append(_bt("V", "lV"))
        lines.append(_bt("R0", "lR0"))
        lines.append(_bt("kon", "lkon"))
        lines.append(_bt("koff", "lkoff"))
        lines.append(_bt("kint", "lkint"))
        # kdeg for receptor turnover (Mager & Jusko 2001: ksyn = kdeg * R0)
        lines.append("kdeg <- koff  # receptor degradation ~ koff initial estimate")
        lines.append("ksyn <- kdeg * R0  # receptor synthesis at baseline")
    elif isinstance(dist_mod, TMDDQSS):
        lines.append(_bt("V", "lV"))
        lines.append(_bt("R0", "lR0"))
        lines.append(_bt("KD", "lKD"))
        lines.append(_bt("kint", "lkint"))
        lines.append("kdeg <- kint  # receptor degradation initial estimate")
        lines.append("ksyn <- kdeg * R0")

    # Elimination
    if isinstance(elim_mod, LinearElim):
        lines.append(_bt("CL", "lCL"))
    elif isinstance(elim_mod, MichaelisMenten):
        lines.append(_bt("Vmax", "lVmax"))
        lines.append(_bt("Km", "lKm"))
    elif isinstance(elim_mod, ParallelLinearMM):
        lines.append(_bt("CL", "lCL"))
        lines.append(_bt("Vmax", "lVmax"))
        lines.append(_bt("Km", "lKm"))
    elif isinstance(elim_mod, TimeVaryingElim):
        lines.append(_bt("CL", "lCL"))
        lines.append(_bt("kdecay", "lkdecay"))

    return lines


def _emit_dynamics(spec: DSLSpec) -> list[str]:
    """Emit compartment dynamics (ODEs or linCmt())."""
    if _needs_ode(spec):
        return _emit_ode_dynamics(spec)
    return _emit_lincmt_dynamics(spec)


# Shared helper so both emitters stay in sync on the "does this spec
# need an ODE?" decision.
from apmode.dsl._emitter_utils import needs_ode as _needs_ode  # noqa: E402


def _emit_lincmt_dynamics(spec: DSLSpec) -> list[str]:
    """Emit linCmt() shorthand for linear compartment models."""
    lines: list[str] = []

    if isinstance(spec.absorption, LaggedFirstOrder):
        lines.append("alag(depot) <- tlag")

    lines.append("cp <- linCmt()")
    return lines


def _emit_ode_dynamics(spec: DSLSpec) -> list[str]:
    """Emit explicit ODE dynamics for non-linear models."""
    lines: list[str] = []
    abs_mod = spec.absorption
    dist_mod = spec.distribution
    elim_mod = spec.elimination

    # --- Absorption compartment ---
    if isinstance(abs_mod, IVBolus):
        # No depot compartment. The dose event must route directly to
        # the central compartment via CMT=1 in the NONMEM event table.
        _abs_influx = ""
    elif isinstance(abs_mod, FirstOrder):
        lines.append("d/dt(depot) <- -ka * depot")
        _abs_influx = "ka * depot"
    elif isinstance(abs_mod, ZeroOrder):
        # Zero-order absorption via rxode2 modeled duration.
        # dur(<cmt>) sets the infusion duration: dose AMT enters the
        # central compartment at constant rate AMT/dur over dur hours.
        # #13: under TMDDQSS the central compartment is ``Atot`` (total
        # drug), not ``centr`` — hardcoding ``dur(centr)`` would
        # fail rxode2 compilation. _central_cmt_name resolves the
        # correct name from the distribution module.
        _cmt = _central_cmt_name(dist_mod)
        lines.append(f"dur({_cmt}) <- dur")
        _abs_influx = ""  # handled by rxode2 infusion mechanism
    elif isinstance(abs_mod, LaggedFirstOrder):
        lines.append("alag(depot) <- tlag")
        lines.append("d/dt(depot) <- -ka * depot")
        _abs_influx = "ka * depot"
    elif isinstance(abs_mod, Transit):
        # rxode2 transit() takes (n, mtt, bio); mtt = (n+1)/ktr
        # Ref: Savic et al. (2007); rxode2 transit compartment docs
        lines.append("d/dt(depot) <- transit(n, mtt) - ka * depot")
        _abs_influx = "ka * depot"
    elif isinstance(abs_mod, MixedFirstZero):
        # Mixed first-order + zero-order: two depot compartments
        lines.append("d/dt(depot_fo) <- -ka * depot_fo")
        lines.append("dur(depot_zo) <- dur")
        lines.append("d/dt(depot_zo) <- -depot_zo")
        lines.append("f(depot_fo) <- frac")
        lines.append("f(depot_zo) <- 1 - frac")
        _abs_influx = "ka * depot_fo + depot_zo"
    elif isinstance(abs_mod, Erlang):
        # Explicit n-compartment chain (ADR-0003 D2). Dose enters E1; each
        # transit step drains at rate ktr; the last compartment feeds the
        # central compartment directly. No terminal first-order ka.
        for i in range(1, abs_mod.n + 1):
            if i == 1:
                lines.append(f"d/dt(E{i}) <- -ktr * E{i}")
            else:
                lines.append(f"d/dt(E{i}) <- ktr * E{i - 1} - ktr * E{i}")
        _abs_influx = f"ktr * E{abs_mod.n}"
    elif isinstance(abs_mod, ParallelFirstOrder):
        # Two parallel first-order depots: fast (ka1) at fraction frac,
        # slow (ka2) at fraction 1-frac. Both feed central simultaneously.
        lines.append("d/dt(depot_fast) <- -ka1 * depot_fast")
        lines.append("d/dt(depot_slow) <- -ka2 * depot_slow")
        lines.append("f(depot_fast) <- frac")
        lines.append("f(depot_slow) <- 1 - frac")
        _abs_influx = "ka1 * depot_fast + ka2 * depot_slow"
    elif isinstance(abs_mod, SumIG):
        # Closed-form analytical input rate (Csajka 2005; Weiss 2022).
        # I(t) = D·F · Σᵢ wᵢ · sqrt(RD2ᵢ / (2π·t³)) · exp(-RD2ᵢ·(t-MTᵢ)² / (2·MTᵢ²·t))
        # Single-dose only in v0.7 (multi-dose superposition deferred,
        # ADR-0003 D4). Guard against t=0 with a small floor — rxode2's
        # LSODA evaluates RHS at output-time grid; output at exactly t=0
        # would force a 0^(-3/2) singularity. The `_t_safe` guard keeps
        # the integrator stable; the contribution near t=0 is ~0 anyway
        # because exp(-RD2·(t-MT)²/(2·MT²·t)) → 0 as t → 0⁺.
        lines.append("# SumIG closed-form input rate (Csajka 2005; Weiss 2022)")
        lines.append("_t_safe <- ifelse(t > 1e-6, t, 1e-6)")
        lines.append(
            "ig_1 <- sqrt(RD2_1 / (2 * 3.141592653589793 * _t_safe^3)) * "
            "exp(-RD2_1 * (_t_safe - MT_1)^2 / (2 * MT_1^2 * _t_safe))"
        )
        lines.append(
            "ig_2 <- sqrt(RD2_2 / (2 * 3.141592653589793 * _t_safe^3)) * "
            "exp(-RD2_2 * (_t_safe - MT_2)^2 / (2 * MT_2^2 * _t_safe))"
        )
        lines.append("sumig_input <- weight_1 * ig_1 + weight_2 * ig_2  # ∫sumig_input dt = 1")
        # The input rate above integrates to 1 over (0, ∞). Multiply by the
        # dose amount so the central-compartment influx has units of
        # [mass·time⁻¹]. rxode2 exposes the dose via `amt` in the model
        # block (resolved per-event by the data adapter).
        _abs_influx = "amt * sumig_input"
    else:
        _abs_influx = "0"

    # --- Distribution compartments ---
    # When _abs_influx is empty (zero-order via dur(centr)), rxode2 handles
    # the infusion directly into centr — no explicit influx term needed.
    if isinstance(dist_mod, OneCmt):
        _elim_expr = _elimination_rate_expr(elim_mod, "centr", "V")
        if _abs_influx:
            lines.append(f"d/dt(centr) <- {_abs_influx} - {_elim_expr}")
        else:
            lines.append(f"d/dt(centr) <- -{_elim_expr}")
        lines.append("cp <- centr / V")
    elif isinstance(dist_mod, TwoCmt):
        _elim_expr = _elimination_rate_expr(elim_mod, "centr", "V1")
        if _abs_influx:
            lines.append(
                f"d/dt(centr) <- {_abs_influx} - {_elim_expr} - Q / V1 * centr + Q / V2 * periph"
            )
        else:
            lines.append(f"d/dt(centr) <- -{_elim_expr} - Q / V1 * centr + Q / V2 * periph")
        lines.append("d/dt(periph) <- Q / V1 * centr - Q / V2 * periph")
        lines.append("cp <- centr / V1")
    elif isinstance(dist_mod, ThreeCmt):
        _elim_expr = _elimination_rate_expr(elim_mod, "centr", "V1")
        if _abs_influx:
            lines.append(
                f"d/dt(centr) <- {_abs_influx} - {_elim_expr} "
                f"- Q2 / V1 * centr + Q2 / V2 * periph1 "
                f"- Q3 / V1 * centr + Q3 / V3 * periph2"
            )
        else:
            lines.append(
                f"d/dt(centr) <- -{_elim_expr} "
                f"- Q2 / V1 * centr + Q2 / V2 * periph1 "
                f"- Q3 / V1 * centr + Q3 / V3 * periph2"
            )
        lines.append("d/dt(periph1) <- Q2 / V1 * centr - Q2 / V2 * periph1")
        lines.append("d/dt(periph2) <- Q3 / V1 * centr - Q3 / V3 * periph2")
        lines.append("cp <- centr / V1")
    elif isinstance(dist_mod, TMDDCore):
        _emit_tmdd_core_odes(lines, _abs_influx, elim_mod)
    elif isinstance(dist_mod, TMDDQSS):
        _emit_tmdd_qss_odes(lines, _abs_influx, elim_mod)

    return lines


def _central_cmt_name(dist_mod: object) -> str:
    """Return the central-compartment identifier emitted by this module.

    #13: ZeroOrder absorption uses ``dur(<cmt>)`` to set the modelled
    infusion duration. Under :class:`TMDDCore` / :class:`TMDDQSS` the
    total-drug pool is named ``Atot`` by :func:`_emit_tmdd_core_odes` /
    :func:`_emit_tmdd_qss_odes`; all other distributions use ``centr``.
    """
    if isinstance(dist_mod, (TMDDCore, TMDDQSS)):
        return "Atot"
    return "centr"


def _elimination_rate_expr(elim_mod: object, cmt: str, vol: str) -> str:
    """Build the elimination rate expression for the central compartment.

    All expressions use concentration (cmt/vol) in the MM term to ensure
    dimensional consistency (Km is in concentration units).
    Returns a parenthesized expression when compound (ParallelLinearMM).
    """
    if isinstance(elim_mod, LinearElim):
        return f"CL / {vol} * {cmt}"
    elif isinstance(elim_mod, MichaelisMenten):
        return f"Vmax * ({cmt}/{vol}) / (Km + {cmt}/{vol})"
    elif isinstance(elim_mod, ParallelLinearMM):
        return f"(CL / {vol} * {cmt} + Vmax * ({cmt}/{vol}) / (Km + {cmt}/{vol}))"
    elif isinstance(elim_mod, TimeVaryingElim):
        # Plan §4 / #9: three decay forms supported.
        #   exponential: CL(t) = CL * exp(-kdecay * t)
        #   half_life:   CL(t) = CL / (1 + kdecay * t)
        #   linear:      CL(t) = max(CL * (1 - kdecay * t), 0)  (floor at 0 in R)
        if elim_mod.decay_fn == "half_life":
            return f"CL / (1 + kdecay * t) / {vol} * {cmt}"
        if elim_mod.decay_fn == "linear":
            return f"max(CL * (1 - kdecay * t), 0) / {vol} * {cmt}"
        return f"CL * exp(-kdecay * t) / {vol} * {cmt}"
    return f"CL / {vol} * {cmt}"


def _emit_tmdd_core_odes(lines: list[str], abs_influx: str, elim_mod: object) -> None:
    """Emit TMDD full binding model ODEs (Mager & Jusko 2001).

    Ref: Mager DE, Jusko WJ. J Pharmacokinet Pharmacodyn. 2001;28:507-532.
    States: centr = drug amount, R = free receptor conc, RC = complex conc.
    Drug concentration L = centr/V used in binding terms for dimensional consistency.
    d/dt(centr) = input - elim(centr) - kon*(centr/V)*R*V + koff*RC*V, where elim()
    is :func:`_elimination_rate_expr` so TMDDCore respects whichever elimination
    module (linear/MM/parallel/time-varying) is paired with it instead of a
    hardcoded linear ``kel``.
    d/dt(R) = ksyn - kdeg*R - kon*(centr/V)*R + koff*RC
    d/dt(RC) = kon*(centr/V)*R - (koff + kint)*RC
    where ksyn = kdeg*R0 at steady state.
    """
    _elim_expr = _elimination_rate_expr(elim_mod, "centr", "V")
    lines.append("# TMDD full binding model (Mager & Jusko 2001)")
    lines.append("L <- centr / V  # drug concentration")
    lines.append(f"d/dt(centr) <- {abs_influx} - {_elim_expr} - kon * L * R * V + koff * RC * V")
    lines.append("d/dt(R) <- ksyn - kdeg * R - kon * L * R + koff * RC")
    lines.append("d/dt(RC) <- kon * L * R - koff * RC - kint * RC")
    lines.append("R(0) <- R0")
    lines.append("cp <- centr / V")


def _emit_tmdd_qss_odes(lines: list[str], abs_influx: str, elim_mod: object) -> None:
    """Emit TMDD quasi-steady-state ODEs (Gibiansky et al. 2008).

    Ref: Gibiansky L, et al. J Pharmacokinet Pharmacodyn. 2008;35:573-591.
    Uses total drug amount (Atot) and total receptor conc (Rtot) as states.
    Free drug concentration solved algebraically from QSS condition.
    KSS = (koff + kint) / kon; KD = koff/kon is used as approximation.
    Elimination acts on free drug via :func:`_elimination_rate_expr` (cmt=Cfree*V,
    i.e. free amount, vol=V) so TMDDQSS respects whichever elimination module is
    paired with it — e.g. linear resolves to ``CL/V*(Cfree*V)`` = ``CL*Cfree``,
    MM resolves to ``Vmax*Cfree/(Km+Cfree)`` — instead of a hardcoded linear ``kel``.
    """
    lines.append("# TMDD quasi-steady-state (Gibiansky et al. 2008)")
    lines.append("# KSS = (koff + kint)/kon; KD = koff/kon.")
    lines.append("# When kint << koff, KSS ≈ KD. When kint is significant,")
    lines.append("# KSS > KD; using KD underestimates KSS, which can")
    lines.append("# overestimate complex formation and target-mediated elimination.")
    lines.append("# The TMDDQSS DSL module estimates KD directly; to use the")
    lines.append("# full KSS, convert to TMDDCore (kon, koff, kint) instead.")
    lines.append("KSS <- KD  # QSS approximation: KSS ≈ KD")
    lines.append("# Convert total drug amount to concentration")
    lines.append("Ctot <- Atot / V")
    lines.append("# Algebraic QSS: solve for free concentrations")
    lines.append(
        "Cfree <- 0.5 * ((Ctot - Rtot - KSS) + sqrt((Ctot - Rtot - KSS)^2 + 4 * KSS * Ctot))"
    )
    lines.append("Rfree <- Rtot * KSS / (KSS + Cfree)")
    lines.append("RC <- Ctot - Cfree")
    _elim_expr = _elimination_rate_expr(elim_mod, "Cfree * V", "V")
    lines.append(f"d/dt(Atot) <- {abs_influx} - {_elim_expr} - kint * RC * V")
    lines.append("d/dt(Rtot) <- ksyn - kdeg * Rfree - kint * RC")
    lines.append("Atot(0) <- 0")
    lines.append("Rtot(0) <- R0")
    lines.append("cp <- Cfree")


# Maps ObservationEndpoint.prediction -> the R variable name the nlmixr2
# emitter's dynamics assign that prediction to (see DSLSpec.
# known_prediction_variables for which names are valid and why). Only
# consulted for the multi-endpoint path -- the single-endpoint path always
# targets "cp" directly, matching every pre-P1.7 emission byte-for-byte.
_PREDICTION_STATE_NAMES: dict[str, str] = {
    "C_central": "cp",
    "C_target_total": "Rtot",
}


def _emit_endpoint_residual(obs: ObservationModule, prediction_var: str, suffix: str) -> list[str]:
    """Emit one endpoint's residual-error statement against its prediction variable.

    For BLQ M3/M4: censoring is handled via CENS/LIMIT data columns
    (not a model-block function). The model block uses standard residual
    error. Ref: nlmixr2 censoring documentation.
    """
    if isinstance(obs, Proportional):
        return [f"{prediction_var} ~ prop(prop.sd{suffix})"]
    elif isinstance(obs, Additive):
        return [f"{prediction_var} ~ add(add.sd{suffix})"]
    elif isinstance(obs, Combined):
        return [f"{prediction_var} ~ prop(prop.sd{suffix}) + add(add.sd{suffix})"]
    elif isinstance(obs, (BLQM3, BLQM4)):
        blq_type = "M3" if isinstance(obs, BLQM3) else "M4"
        if blq_type == "M3":
            comment = f"# BLQ M3: set CENS=1 and DV=LLOQ={obs.loq_value} in data for BLQ obs"
        else:
            comment = f"# BLQ M4: set CENS=1, DV=LLOQ={obs.loq_value}, LIMIT=0 in data"
        # Use the composed error model
        if obs.error_model == "proportional":
            return [comment, f"{prediction_var} ~ prop(prop.sd{suffix})"]
        elif obs.error_model == "additive":
            return [comment, f"{prediction_var} ~ add(add.sd{suffix})"]
        else:  # combined
            return [comment, f"{prediction_var} ~ prop(prop.sd{suffix}) + add(add.sd{suffix})"]
    # #28: catching every other ObservationModule with a silent
    # proportional fallback is how unknown AST nodes reach backends
    # unnoticed. Raise so unimplemented obs modules are caught at
    # emit time rather than producing a wrong model.
    msg = (
        f"nlmixr2 emitter: unsupported observation module "
        f"{type(obs).__name__} — implement a new branch instead of "
        "relying on the proportional default."
    )
    raise NotImplementedError(msg)


def _emit_observation_model(spec: DSLSpec) -> list[str]:
    """Emit the observation/residual error model for every observation endpoint.

    Formular sharpening plan §4 Phase 1 (P1.7): iterates
    ``spec.observation_endpoints()``. The single-endpoint case (the
    synthetic ``"default"`` endpoint every legacy ``observation:`` spec
    normalizes to) emits ``cp ~ ...`` exactly as before P1.7. For a genuine
    multi-analyte ``observations:`` block, each endpoint's ``prediction``
    name is resolved to its emitter-internal R variable (see
    ``_PREDICTION_STATE_NAMES`` / ``DSLSpec.known_prediction_variables``)
    and a per-endpoint residual statement is emitted, with sigma names
    suffixed by endpoint name so multiple endpoints never share a sigma
    variable.

    Routing observed data rows to the correct endpoint by ``dvid`` (e.g. via
    a ``cmt``/DVID data column nlmixr2 matches against these prediction
    names) is a data-adapter/runner concern, not this emitter's — see
    ``apmode.data.adapters.PK_DVID_ALLOWLIST`` and the ``Nlmixr2Runner``
    two-layer adapter contract (CLAUDE.md); wiring that end-to-end for
    multi-analyte data is a Phase 2 candidate.
    """
    endpoints = spec.observation_endpoints()
    if len(endpoints) == 1:
        endpoint = endpoints[0]
        return _emit_endpoint_residual(endpoint.error, "cp", "")

    lines: list[str] = []
    for endpoint in endpoints:
        state = _PREDICTION_STATE_NAMES.get(endpoint.prediction)
        if state is None:
            # Unreachable when the spec passed through validate_dsl (which
            # rejects unresolvable predictions via
            # FrmCode.AST_OBSERVATIONS_PREDICTION_UNKNOWN before any emitter
            # runs) -- fail loudly rather than emit broken R code for a
            # caller that skipped validation.
            msg = (
                f"nlmixr2 emitter: endpoint '{endpoint.name}' references "
                f"prediction {endpoint.prediction!r}, which is not in "
                "_PREDICTION_STATE_NAMES — not in "
                "DSLSpec.known_prediction_variables()."
            )
            raise NotImplementedError(msg)
        suffix = _endpoint_sigma_suffix(endpoint.name)
        lines.append(f"# endpoint '{endpoint.name}' (DVID={endpoint.dvid})")
        lines.extend(_emit_endpoint_residual(endpoint.error, state, suffix))
    return lines
