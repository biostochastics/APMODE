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
    CovariateLink,
    DSLSpec,
    FirstOrder,
    IVBolus,
    LaggedFirstOrder,
    LinearElim,
    MichaelisMenten,
    MixedFirstZero,
    OccasionByDoseEpoch,
    OccasionByStudy,
    OccasionByVisit,
    OccasionCustom,
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

# Valid R identifier pattern for name sanitization
_R_IDENT_RE = re.compile(r"^[a-zA-Z_.][a-zA-Z0-9_.]*$")


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

    When initial_estimates is provided, override DSLSpec values for matching
    parameter names (e.g. "CL" -> use override instead of spec.elimination.CL).
    """
    ov = initial_estimates or {}
    lines: list[str] = []
    abs_mod = spec.absorption
    dist_mod = spec.distribution
    elim_mod = spec.elimination

    # --- Absorption ---
    if isinstance(abs_mod, IVBolus):
        # IV bolus: no absorption parameters. Dose is routed directly to
        # the central compartment; depot is omitted by the structural emitter.
        pass
    elif isinstance(abs_mod, FirstOrder):
        lines.append(f"lka <- log({ov.get('ka', abs_mod.ka)})")
    elif isinstance(abs_mod, ZeroOrder):
        lines.append(f"ldur <- log({ov.get('dur', abs_mod.dur)})")
    elif isinstance(abs_mod, LaggedFirstOrder):
        lines.append(f"lka <- log({ov.get('ka', abs_mod.ka)})")
        tlag = ov.get("tlag", abs_mod.tlag)
        lines.append(f"ltlag <- log({tlag})" if tlag > 0 else "ltlag <- -10")
    elif isinstance(abs_mod, Transit):
        lines.append(f"lka <- log({ov.get('ka', abs_mod.ka)})")
        lines.append(f"lktr <- log({ov.get('ktr', abs_mod.ktr)})")
        # n is estimated as continuous via log/exp transform; rxode2's transit()
        # uses gamma-function interpolation for non-integer n values
        # (Savic et al. 2007, J Pharmacokinet Pharmacodyn 34:711-726)
        lines.append(f"ln <- log({ov.get('n', abs_mod.n)})")
    elif isinstance(abs_mod, MixedFirstZero):
        lines.append(f"lka <- log({ov.get('ka', abs_mod.ka)})")
        lines.append(f"ldur <- log({ov.get('dur', abs_mod.dur)})")
        # #14: frac == 1.0 (perfect bioavailability) produced a
        # ZeroDivisionError on log(1 / 0) at emit time. Clamp to the
        # 99.99% ceiling and warn — the user can always drop the
        # ZeroOrder leg or use FirstOrder if they truly want fraction=1.
        frac_raw = float(ov.get("frac", abs_mod.frac))
        _frac_epsilon = 1e-4
        frac_clamped = min(max(frac_raw, _frac_epsilon), 1.0 - _frac_epsilon)
        if frac_clamped != frac_raw:
            lines.append(
                f"# frac clamped from {frac_raw} to {frac_clamped} "
                f"to avoid singular logit (APMODE #14)"
            )
        lines.append(f"logit_frac <- log({frac_clamped} / (1 - {frac_clamped}))")

    # --- Distribution ---
    if isinstance(dist_mod, OneCmt):
        lines.append(f"lV <- log({ov.get('V', dist_mod.V)})")
    elif isinstance(dist_mod, TwoCmt):
        lines.append(f"lV1 <- log({ov.get('V1', dist_mod.V1)})")
        lines.append(f"lV2 <- log({ov.get('V2', dist_mod.V2)})")
        lines.append(f"lQ <- log({ov.get('Q', dist_mod.Q)})")
    elif isinstance(dist_mod, ThreeCmt):
        lines.append(f"lV1 <- log({ov.get('V1', dist_mod.V1)})")
        lines.append(f"lV2 <- log({ov.get('V2', dist_mod.V2)})")
        lines.append(f"lV3 <- log({ov.get('V3', dist_mod.V3)})")
        lines.append(f"lQ2 <- log({ov.get('Q2', dist_mod.Q2)})")
        lines.append(f"lQ3 <- log({ov.get('Q3', dist_mod.Q3)})")
    elif isinstance(dist_mod, TMDDCore):
        lines.append(f"lV <- log({ov.get('V', dist_mod.V)})")
        lines.append(f"lR0 <- log({ov.get('R0', dist_mod.R0)})")
        lines.append(f"lkon <- log({ov.get('kon', dist_mod.kon)})")
        lines.append(f"lkoff <- log({ov.get('koff', dist_mod.koff)})")
        lines.append(f"lkint <- log({ov.get('kint', dist_mod.kint)})")
    elif isinstance(dist_mod, TMDDQSS):
        lines.append(f"lV <- log({ov.get('V', dist_mod.V)})")
        lines.append(f"lR0 <- log({ov.get('R0', dist_mod.R0)})")
        lines.append(f"lKD <- log({ov.get('KD', dist_mod.KD)})")
        lines.append(f"lkint <- log({ov.get('kint', dist_mod.kint)})")

    # --- Elimination ---
    if isinstance(elim_mod, LinearElim):
        lines.append(f"lCL <- log({ov.get('CL', elim_mod.CL)})")
    elif isinstance(elim_mod, MichaelisMenten):
        lines.append(f"lVmax <- log({ov.get('Vmax', elim_mod.Vmax)})")
        lines.append(f"lKm <- log({ov.get('Km', elim_mod.Km)})")
    elif isinstance(elim_mod, ParallelLinearMM):
        lines.append(f"lCL <- log({ov.get('CL', elim_mod.CL)})")
        lines.append(f"lVmax <- log({ov.get('Vmax', elim_mod.Vmax)})")
        lines.append(f"lKm <- log({ov.get('Km', elim_mod.Km)})")
    elif isinstance(elim_mod, TimeVaryingElim):
        # All three decay forms (exponential | half_life | linear) are
        # supported as of v0.5.0 — the per-form ODE RHS is emitted by
        # ``_elim_rate_expr`` (see lines ~610-620). This block only
        # writes the log-parameter scaffolding shared by all forms.
        lines.append(f"lCL <- log({ov.get('CL', elim_mod.CL)})")
        lines.append(f"lkdecay <- log({ov.get('kdecay', elim_mod.kdecay)})")

    # Covariate coefficients
    cov_links = [v for v in spec.variability if isinstance(v, CovariateLink)]
    if cov_links:
        lines.append("")
        lines.append("# Covariate coefficients")
        for cov in cov_links:
            p = _sanitize_r_name(cov.param)
            c = _sanitize_r_name(cov.covariate)
            coeff_name = f"beta_{p}_{c}"
            if cov.form == "power":
                lines.append(f"{coeff_name} <- 0.75")  # allometric default
            elif cov.form == "maturation":
                lines.append(f"{coeff_name} <- 1")
                lines.append(f"TM50_{p}_{c} <- 1")
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


def _emit_sigma_ini(spec: DSLSpec) -> list[str]:
    """Emit residual error sigma definitions."""
    obs = spec.observation

    if isinstance(obs, Proportional):
        return [f"prop.sd <- {obs.sigma_prop}"]
    elif isinstance(obs, Additive):
        return [f"add.sd <- {obs.sigma_add}"]
    elif isinstance(obs, Combined):
        return [
            f"prop.sd <- {obs.sigma_prop}",
            f"add.sd <- {obs.sigma_add}",
        ]
    elif isinstance(obs, (BLQM3, BLQM4)):
        # BLQ composes with underlying error model; censoring is data-driven
        if obs.error_model == "proportional":
            return [f"prop.sd <- {obs.sigma_prop}"]
        elif obs.error_model == "additive":
            return [f"add.sd <- {obs.sigma_add}"]
        else:  # combined
            return [
                f"prop.sd <- {obs.sigma_prop}",
                f"add.sd <- {obs.sigma_add}",
            ]
    return []


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

    cov_links = [v for v in spec.variability if isinstance(v, CovariateLink)]

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
                    # 70 kg is the standard pharmacometric reference weight
                    # (Anderson & Holford 2008, Clin Pharmacokinet 47:455-467)
                    expr += f" + {coeff} * log({cov.covariate} / 70)"
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
        70-kg reference as in ``_bt`` (Anderson & Holford 2008).
        """
        expr = logit_name
        if param in iiv_params:
            expr += f" + eta.{param}"
        if param in iov_params:
            expr += f" + eta.iov.{param}"
        # Covariate effects on logit scale. The logit is unbounded so
        # each cov.form maps naturally from its _bt counterpart:
        #   - power:      β·log(cov / 70)        (log-linear on odds)
        #   - exponential: β·cov                   (linear on odds)
        #   - linear:     log(1 + β·cov)          (matches _bt — non-negative effect)
        #   - categorical: β·cov                   (indicator on odds)
        #   - maturation: log(cov^β / (cov^β + TM50^β))
        for cov in cov_links:
            if cov.param == param:
                coeff = f"beta_{cov.param}_{cov.covariate}"
                if cov.form == "power":
                    expr += f" + {coeff} * log({cov.covariate} / 70)"
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
        lines.append("kel <- CL / V  # elimination rate constant")
    elif isinstance(dist_mod, TMDDQSS):
        lines.append(_bt("V", "lV"))
        lines.append(_bt("R0", "lR0"))
        lines.append(_bt("KD", "lKD"))
        lines.append(_bt("kint", "lkint"))
        lines.append("kdeg <- kint  # receptor degradation initial estimate")
        lines.append("ksyn <- kdeg * R0")
        lines.append("kel <- CL / V  # elimination rate constant")

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
        _emit_tmdd_core_odes(lines, _abs_influx)
    elif isinstance(dist_mod, TMDDQSS):
        _emit_tmdd_qss_odes(lines, _abs_influx)

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


def _emit_tmdd_core_odes(lines: list[str], abs_influx: str) -> None:
    """Emit TMDD full binding model ODEs (Mager & Jusko 2001).

    Ref: Mager DE, Jusko WJ. J Pharmacokinet Pharmacodyn. 2001;28:507-532.
    States: centr = drug amount, R = free receptor conc, RC = complex conc.
    Drug concentration L = centr/V used in binding terms for dimensional consistency.
    d/dt(centr) = input - kel*centr - kon*(centr/V)*R*V + koff*RC*V
    d/dt(R) = ksyn - kdeg*R - kon*(centr/V)*R + koff*RC
    d/dt(RC) = kon*(centr/V)*R - (koff + kint)*RC
    where ksyn = kdeg*R0 at steady state.
    """
    lines.append("# TMDD full binding model (Mager & Jusko 2001)")
    lines.append("L <- centr / V  # drug concentration")
    lines.append(f"d/dt(centr) <- {abs_influx} - kel * centr - kon * L * R * V + koff * RC * V")
    lines.append("d/dt(R) <- ksyn - kdeg * R - kon * L * R + koff * RC")
    lines.append("d/dt(RC) <- kon * L * R - koff * RC - kint * RC")
    lines.append("R(0) <- R0")
    lines.append("cp <- centr / V")


def _emit_tmdd_qss_odes(lines: list[str], abs_influx: str) -> None:
    """Emit TMDD quasi-steady-state ODEs (Gibiansky et al. 2008).

    Ref: Gibiansky L, et al. J Pharmacokinet Pharmacodyn. 2008;35:573-591.
    Uses total drug amount (Atot) and total receptor conc (Rtot) as states.
    Free drug concentration solved algebraically from QSS condition.
    KSS = (koff + kint) / kon; KD = koff/kon is used as approximation.
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
    lines.append(f"d/dt(Atot) <- {abs_influx} - kel * Cfree * V - kint * RC * V")
    lines.append("d/dt(Rtot) <- ksyn - kdeg * Rfree - kint * RC")
    lines.append("Atot(0) <- 0")
    lines.append("Rtot(0) <- R0")
    lines.append("cp <- Cfree")


def _emit_observation_model(spec: DSLSpec) -> list[str]:
    """Emit the observation/residual error model.

    For BLQ M3/M4: censoring is handled via CENS/LIMIT data columns
    (not a model-block function). The model block uses standard residual
    error. Ref: nlmixr2 censoring documentation.
    """
    obs = spec.observation

    if isinstance(obs, Proportional):
        return ["cp ~ prop(prop.sd)"]
    elif isinstance(obs, Additive):
        return ["cp ~ add(add.sd)"]
    elif isinstance(obs, Combined):
        return ["cp ~ prop(prop.sd) + add(add.sd)"]
    elif isinstance(obs, (BLQM3, BLQM4)):
        blq_type = "M3" if isinstance(obs, BLQM3) else "M4"
        if blq_type == "M3":
            comment = f"# BLQ M3: set CENS=1 and DV=LLOQ={obs.loq_value} in data for BLQ obs"
        else:
            comment = f"# BLQ M4: set CENS=1, DV=LLOQ={obs.loq_value}, LIMIT=0 in data"
        # Use the composed error model
        if obs.error_model == "proportional":
            return [comment, "cp ~ prop(prop.sd)"]
        elif obs.error_model == "additive":
            return [comment, "cp ~ add(add.sd)"]
        else:  # combined
            return [comment, "cp ~ prop(prop.sd) + add(add.sd)"]
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
