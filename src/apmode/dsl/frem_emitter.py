# SPDX-License-Identifier: GPL-2.0-or-later
"""FREM (Full Random Effects Model) nlmixr2 emitter.

FREM treats each covariate as an **observation** drawn from a multivariate
normal distribution jointly with the PK random effects. A single joint Ω
matrix carries the PK-covariate covariances; missing covariate values are
handled naturally by omitting those observation rows — no imputation, no
Rubin pooling, no per-occasion MI loops (Nyberg 2024, Karlsson et al.).

References:
- Karlsson 2011 (PAGE): FREM rationale and full covariance derivation.
- Yngman et al. 2022 CPT:PSP — practical FREM implementation patterns.
- Nyberg 2024, Jonsson 2024 — FREM vs FFEM + mean imputation under MAR.

Scope of this emitter (v1):
- Static (baseline-only) subject-level covariates.
- Continuous covariates. Categorical covariates require a separate
  logit-transformed parameterization and are not emitted.
- Joint Ω is structured as a block matrix with the PK IIV block (from
  the underlying spec) in the top-left and covariate variances on the
  diagonal of the bottom-right block; off-diagonal PKxcovariate
  covariances initialize to zero and are estimated by nlmixr2.
- CovariateLink entries in the spec are dropped because FREM replaces
  explicit covariate-on-parameter effects with the joint random-effect
  structure.

Time-varying covariates are currently handled upstream (routing forces
FREM when ``manifest.time_varying_covariates`` is set), but the
emitter treats them as if they were baseline. Extending to per-occasion
etas is future work and is called out in the module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from apmode.dsl.ast_models import (
    IIV,
    IOV,
    CovariateLink,
    DSLSpec,
)
from apmode.dsl.nlmixr2_emitter import (
    _emit_model,
    _emit_sigma_ini,
    _emit_structural_ini,
    _emit_variability_ini,
    _sanitize_r_name,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


# DVID offset for FREM covariate observations. PK observations keep
# their native DVID (typically 1 if present, or absent for single-DV data;
# the emitter treats absence as DVID==1 via the default endpoint). Covariate
# observations get DVID = 10 + covariate_index to avoid collisions with any
# realistic multi-analyte DVID scheme.
_FREM_DVID_OFFSET: int = 10

# Default residual error SD for covariate observations. Must be >0 so the
# likelihood is well-defined; kept small so the random-effect component
# absorbs the between-subject variance rather than the residual component.
_DEFAULT_COV_EPS_SD: float = 0.01

# Floor on covariate initial variance. Covariates with observed SD below this
# are not useful FREM targets (degenerate); the emitter raises when that
# happens so callers can drop the covariate explicitly.
_MIN_COV_SD: float = 1e-6


@dataclass(frozen=True)
class FREMCovariate:
    """Per-covariate metadata for FREM emission.

    Attributes:
        name: Covariate name as it appears in the data CSV. Must be a valid
            R identifier.
        mu_init: Initial estimate for the covariate mean (typically sample
            mean from observed data, on the emitted scale — so after any
            ``transform`` applied by ``summarize_covariates``).
        sigma_init: Initial SD for the covariate random-effect diagonal
            (sample SD on the emitted scale).
        dvid: Integer DVID value that identifies this covariate's
            observation rows in the prepared long-format data.
        epsilon_sd: Residual error SD for the covariate observation. Small
            positive value, emitted with ``fix(...)`` — with only one
            observation per subject per covariate, BSV and residual are
            confounded, so fixing the residual lets the random-effect
            component absorb all between-subject variance (Karlsson 2011).
        transform: Scale on which the covariate is represented in the
            model. ``"identity"`` (default) fits the covariate on its
            raw scale; ``"log"`` fits on the natural-log scale and is
            recommended for strictly positive, right-skewed covariates
            (weight, creatinine, etc.) so the Ω block stays well
            conditioned.
    """

    name: str
    mu_init: float
    sigma_init: float
    dvid: int
    epsilon_sd: float = _DEFAULT_COV_EPS_SD
    transform: str = "identity"

    def __post_init__(self) -> None:
        _sanitize_r_name(self.name)
        if self.transform not in ("identity", "log"):
            msg = (
                f"FREMCovariate {self.name!r}: transform must be 'identity' or "
                f"'log', got {self.transform!r}"
            )
            raise ValueError(msg)
        if not np.isfinite(self.mu_init):
            msg = f"FREMCovariate {self.name!r}: mu_init must be finite, got {self.mu_init}"
            raise ValueError(msg)
        if not np.isfinite(self.sigma_init) or self.sigma_init < _MIN_COV_SD:
            msg = (
                f"FREMCovariate {self.name!r}: sigma_init must be >= {_MIN_COV_SD}, "
                f"got {self.sigma_init}. Degenerate covariates (constant across "
                f"subjects) should be dropped before calling the FREM emitter."
            )
            raise ValueError(msg)
        if self.epsilon_sd <= 0:
            msg = f"FREMCovariate {self.name!r}: epsilon_sd must be > 0"
            raise ValueError(msg)


def summarize_covariates(
    df: pd.DataFrame,
    covariate_names: Sequence[str],
    *,
    id_col: str = "NMID",
    time_col: str = "TIME",
    transforms: dict[str, str] | None = None,
) -> list[FREMCovariate]:
    """Compute per-covariate ``(mu_init, sigma_init)`` from observed data.

    Uses each subject's **baseline** row (minimum ``TIME``) rather than the
    first row as stored; in randomly-ordered data these can differ and
    baseline is the pharmacometric convention for subject-level covariates.
    Subjects with a missing value for a given covariate are excluded from
    that covariate's mean/SD only — this matches the FREM likelihood
    treatment of missingness (those subjects simply have no observation
    row for that covariate).

    Args:
        df: Source DataFrame.
        covariate_names: Ordered list of covariate column names. Duplicates
            raise ``ValueError`` — they would otherwise produce colliding
            DVIDs.
        id_col: Subject identifier column (default ``NMID``).
        time_col: Time column used to pick each subject's baseline row
            (default ``TIME``).
        transforms: Optional per-covariate scale override, e.g.
            ``{"WT": "log"}``. Unspecified covariates default to
            ``"identity"``. Positive/right-skewed covariates (body weight,
            creatinine, etc.) are typically better modeled on the log
            scale so the joint Ω is well conditioned (Yngman 2022).

    Returns:
        One ``FREMCovariate`` per name, in the same order as
        ``covariate_names``.

    Raises:
        ValueError: Duplicate names, covariate missing from ``df``, or
            fewer than 2 observed subjects for any covariate.
    """
    if len(covariate_names) != len(set(covariate_names)):
        dupes = {n for n in covariate_names if covariate_names.count(n) > 1}
        msg = f"Duplicate covariate names not allowed: {sorted(dupes)}"
        raise ValueError(msg)

    tr_map = transforms or {}
    baseline_idx = df.groupby(id_col)[time_col].idxmin()
    per_subj = df.loc[baseline_idx].set_index(id_col)

    summaries: list[FREMCovariate] = []
    for idx, name in enumerate(covariate_names):
        if name not in per_subj.columns:
            msg = f"Covariate {name!r} not in data columns"
            raise ValueError(msg)
        transform = tr_map.get(name, "identity")
        observed = per_subj[name].dropna().astype(float)
        if transform == "log":
            if (observed <= 0).any():
                msg = (
                    f"Covariate {name!r}: log transform requires strictly "
                    f"positive values; found non-positive entries."
                )
                raise ValueError(msg)
            observed = np.log(observed)
        if len(observed) < 2:
            msg = (
                f"Covariate {name!r}: only {len(observed)} observed subject(s); "
                f"FREM requires at least 2 subjects with observed covariate values."
            )
            raise ValueError(msg)
        mu = float(observed.mean())
        sd = float(observed.std(ddof=1))
        summaries.append(
            FREMCovariate(
                name=name,
                mu_init=mu,
                sigma_init=sd,
                dvid=_FREM_DVID_OFFSET + idx,
                transform=transform,
            )
        )
    return summaries


def prepare_frem_data(
    df: pd.DataFrame,
    covariates: Sequence[FREMCovariate],
    *,
    id_col: str = "NMID",
    time_col: str = "TIME",
) -> pd.DataFrame:
    """Augment a DataFrame with per-covariate observation rows for FREM.

    The returned DataFrame has:
      - Every original row preserved, with a ``DVID`` column added
        (value ``1`` for PK observations / doses; anything already
        present in ``DVID`` is preserved).
      - One extra observation row per subject per covariate, placed at
        the subject's baseline time (minimum ``TIME``) with ``EVID=0``,
        ``AMT=0``, ``MDV=0``, ``DV = covariate value on the emitted
        scale`` (log-transformed when ``covariate.transform == "log"``),
        ``DVID = covariate.dvid``. Subjects missing a covariate get no
        row for that covariate — FREM's likelihood handles them.

    Raises ``ValueError`` if the source DataFrame already carries DVIDs
    that would collide with the FREM covariate DVID range; the caller
    must re-map their multi-analyte DVID scheme before augmentation.

    Assumes standard APMODE canonical columns (``NMID``, ``TIME``, ``EVID``,
    ``DV``, ``AMT``). Other columns are copied from the subject's baseline
    row so that rxode2's state machine has consistent covariates on the
    augmentation rows.
    """
    import pandas as pd  # local runtime import (pd is only a TYPE_CHECKING name at module level)

    out = df.copy()
    if "DVID" not in out.columns:
        out["DVID"] = 1
    else:
        out["DVID"] = out["DVID"].fillna(1).astype(int)

    # Collision check: the emitter assumes DVIDs >= _FREM_DVID_OFFSET are
    # reserved for FREM covariates. Refuse to overwrite an existing scheme.
    cov_dvids = {cov.dvid for cov in covariates}
    existing = set(out["DVID"].unique().tolist())
    overlap = existing & cov_dvids
    if overlap:
        msg = (
            f"Source data DVIDs {sorted(overlap)} collide with FREM covariate "
            f"DVIDs. Remap your multi-analyte DVID scheme below "
            f"{_FREM_DVID_OFFSET} before calling prepare_frem_data."
        )
        raise ValueError(msg)

    baseline_idx = out.groupby(id_col, sort=False)[time_col].idxmin()
    first_per_subj = out.loc[baseline_idx].reset_index(drop=True)

    aug_rows: list[dict[str, object]] = []
    for _, subj_row in first_per_subj.iterrows():
        sid = subj_row[id_col]
        baseline_time = float(subj_row[time_col])
        for cov in covariates:
            cov_value = subj_row.get(cov.name)
            # pd.isna covers None, np.nan, pd.NA, and NaT in a single idiom.
            if pd.isna(cov_value):
                continue
            cov_float = float(cov_value)
            if cov.transform == "log":
                if cov_float <= 0:
                    msg = (
                        f"Subject {sid!r}: covariate {cov.name!r} has non-positive "
                        f"value {cov_float} but transform='log' — skip this subject or "
                        f"drop the covariate."
                    )
                    raise ValueError(msg)
                cov_float = float(np.log(cov_float))
            new_row: dict[str, object] = {k: v for k, v in subj_row.items()}
            new_row[id_col] = sid
            new_row[time_col] = baseline_time
            if "EVID" in new_row:
                new_row["EVID"] = 0
            if "AMT" in new_row:
                new_row["AMT"] = 0.0
            if "MDV" in new_row:
                new_row["MDV"] = 0
            new_row["DV"] = cov_float
            new_row["DVID"] = cov.dvid
            aug_rows.append(new_row)

    if not aug_rows:
        return out

    aug_df = pd.DataFrame(aug_rows, columns=out.columns)
    combined = pd.concat([out, aug_df], ignore_index=True)
    combined = combined.sort_values([id_col, time_col, "DVID"], kind="stable").reset_index(
        drop=True
    )
    return combined


# ---------------------------------------------------------------------------
# ini() block — joint Ω block for PK + covariate etas
# ---------------------------------------------------------------------------


def _emit_frem_joint_block(
    pk_iiv_names: Sequence[str],
    covariates: Sequence[FREMCovariate],
    *,
    pk_block_structure: str = "diagonal",
    pk_iiv_var: float = 0.1,
    pk_iiv_cov: float = 0.01,
) -> list[str]:
    """Emit a single ``eta.A + eta.B + ... ~ c(...)`` joint-Ω block.

    Layout (K = len(pk_iiv_names), C = len(covariates)):

        [ Ω_PK (KxK)        Ω_PKxCOV (KxC) = 0 (initial) ]
        [ Ω_COVxPK (CxK) = 0 (initial)    Ω_COV (CxC, diagonal) ]

    The lower-triangular vector is emitted row-by-row; nlmixr2 unpacks it
    into the symmetric Ω at compile time.

    Off-diagonal PK-covariate entries initialize to zero so the optimizer
    can move them away from zero only if data supports. PK intra-block
    covariances default to ``pk_iiv_cov`` when ``pk_block_structure="block"``.
    """
    k = len(pk_iiv_names)
    c = len(covariates)
    n = k + c

    all_etas = [f"eta.{_sanitize_r_name(p)}" for p in pk_iiv_names]
    all_etas.extend(f"eta.cov.{_sanitize_r_name(cov.name)}" for cov in covariates)

    # Build the lower-triangular vector row-by-row (i=0..n-1, j=0..i).
    # Partition-aware initializers:
    #   - Top-left KxK PK block: diagonal = pk_iiv_var, off-diagonal depends on structure.
    #   - Bottom-right CxC covariate block: diagonal = sigma_init**2, off-diagonal = 0.
    #   - KxC cross block: always 0 initially.
    entries: list[str] = []
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                if i < k:
                    entries.append(f"{pk_iiv_var}")
                else:
                    cov = covariates[i - k]
                    entries.append(f"{cov.sigma_init**2}")
            elif i < k and j < k:
                # Within-PK off-diagonal
                entries.append(f"{pk_iiv_cov}" if pk_block_structure == "block" else "0")
            else:
                # PK-cov cross block or between-covariate off-diagonal: 0
                entries.append("0")

    eta_sum = " + ".join(all_etas)
    lines = [f"{eta_sum} ~ c("]
    # Group by row for readability.
    idx = 0
    for i in range(n):
        row = entries[idx : idx + i + 1]
        suffix = "," if i < n - 1 else ""
        lines.append(f"  {', '.join(row)}{suffix}")
        idx += i + 1
    lines.append(")")
    return lines


def _detect_pk_block_structure(spec: DSLSpec) -> tuple[str, list[str]]:
    """Return (structure, param_order) for the spec's PK IIV block."""
    for item in spec.variability:
        if isinstance(item, IIV):
            return item.structure, list(item.params)
    return "diagonal", []


def _emit_frem_ini(
    spec: DSLSpec,
    covariates: Sequence[FREMCovariate],
    initial_estimates: dict[str, float] | None,
) -> list[str]:
    """Emit the full ini block for a FREM-augmented model."""
    lines: list[str] = []
    lines.append(
        "# Structural parameters (FREM: covariate effects carried by joint Ω, not fixed effects)"
    )
    # CovariateLink entries are superseded by FREM — strip them before
    # delegating to the structural emitter so beta coefficients are not
    # emitted redundantly.
    frem_spec = _strip_covariate_links(spec)
    lines.extend(_emit_structural_ini(frem_spec, initial_estimates=initial_estimates))

    # Covariate means (fixed effects in the ini block)
    lines.append("")
    lines.append("# FREM covariate means (initial = observed sample mean)")
    for cov in covariates:
        name = _sanitize_r_name(cov.name)
        lines.append(f"mu_{name} <- {cov.mu_init}")

    # Joint Ω block
    lines.append("")
    lines.append("# FREM joint Omega: PK IIV etas + covariate etas")
    pk_struct, pk_params = _detect_pk_block_structure(frem_spec)
    if not pk_params:
        # Fallback: if the spec has no IIV block at all, FREM still needs
        # at least one PK random effect to anchor the joint Ω. We synthesize
        # a minimal diagonal eta.CL — this is only a guardrail; spec
        # validation should catch empty-IIV specs before they reach FREM.
        pk_params = ["CL"]
        pk_struct = "diagonal"
    lines.extend(_emit_frem_joint_block(pk_params, covariates, pk_block_structure=pk_struct))

    # IOV etas (if any) — emitted as usual (not merged into the joint block)
    iov_lines: list[str] = []
    for item in frem_spec.variability:
        if isinstance(item, IOV):
            iov_lines.extend(_emit_variability_ini_for_iov_only(item))
    if iov_lines:
        lines.append("")
        lines.append("# IOV etas (separate from FREM joint block)")
        lines.extend(iov_lines)

    # PK residual error (unchanged)
    lines.append("")
    lines.append("# PK residual error")
    lines.extend(_emit_sigma_ini(frem_spec))

    # Covariate residual error (one per covariate).
    # Fixed via ``fix(...)`` because with one covariate observation per
    # subject, the subject-level eta and residual sd are perfectly
    # confounded; fixing ε lets eta absorb all between-subject variance
    # (standard FREM practice, Karlsson 2011).
    lines.append("")
    lines.append("# FREM covariate residual error (fixed; eta absorbs BSV)")
    for cov in covariates:
        name = _sanitize_r_name(cov.name)
        lines.append(f"sig_cov_{name} <- fix({cov.epsilon_sd})")

    return lines


def _emit_variability_ini_for_iov_only(item: IOV) -> list[str]:
    """Emit only the IOV portion of the variability block."""
    # Construct a temporary spec-like object holding a single IOV item so
    # we can reuse ``_emit_variability_ini`` without duplicating its logic.
    # The existing helper is shape-agnostic, so a namespace wrapper works.
    from types import SimpleNamespace

    wrapper = SimpleNamespace(variability=[item])
    return _emit_variability_ini(wrapper)  # type: ignore[arg-type]


def _strip_covariate_links(spec: DSLSpec) -> DSLSpec:
    """Return a copy of ``spec`` with CovariateLink entries removed.

    FREM replaces explicit covariate-on-parameter effects with the joint
    random-effect structure, so any CovariateLink in the input spec
    would duplicate information and destabilize estimation. We keep IIV
    and IOV intact — only the beta links are pruned.
    """
    new_variability = [v for v in spec.variability if not isinstance(v, CovariateLink)]
    return spec.model_copy(update={"variability": new_variability})


# ---------------------------------------------------------------------------
# model() block — PK + covariate observation endpoints
# ---------------------------------------------------------------------------


def _append_dvid_pipe(line: str, dvid: int) -> str:
    """Append ``| DVID==<dvid>`` to an endpoint line that does not already
    carry a pipe condition.

    The base emitter's PK endpoint (``cp ~ prop(...)``) is unqualified
    because single-DV data does not need routing; under FREM the same
    model file carries multiple endpoints, so nlmixr2 requires an
    explicit DVID condition to disambiguate the PK endpoint from the
    covariate endpoints (Codex review 2026-04-14).
    """
    if "|" in line:
        return line
    return f"{line} | DVID=={dvid}"


def _emit_frem_model(
    spec: DSLSpec,
    covariates: Sequence[FREMCovariate],
    *,
    pk_dvid: int = 1,
) -> list[str]:
    """Emit the model block with FREM covariate-observation endpoints.

    The PK portion is delegated to the base emitter, then post-processed
    to attach an explicit ``| DVID==<pk_dvid>`` condition on the PK
    observation line. Covariate observations are then appended as
    additional endpoints keyed on each covariate's DVID.
    """
    frem_spec = _strip_covariate_links(spec)
    base_lines = _emit_model(frem_spec)

    # The PK observation line is the tail of the base emitter's output.
    # Look for any ``<name> ~ <residual-spec>`` line (no leading "#") and
    # attach the PK DVID condition. There is exactly one PK endpoint in
    # the base emitter today, so a single match is expected.
    patched: list[str] = []
    for line in base_lines:
        stripped = line.lstrip()
        is_endpoint = "~" in stripped and not stripped.startswith("#") and "d/dt" not in stripped
        patched.append(_append_dvid_pipe(line, pk_dvid) if is_endpoint else line)

    cov_lines: list[str] = [""]
    cov_lines.append("# FREM covariate observation endpoints")
    for cov in covariates:
        name = _sanitize_r_name(cov.name)
        # Predicted covariate value: mean + subject-specific eta. For
        # ``transform="log"`` the DV in the augmented data is already on
        # the log scale (set by ``prepare_frem_data``), so the same
        # algebraic form applies — the scale is carried by the data, not
        # by the emitted model expression.
        cov_lines.append(f"{name}_pred <- mu_{name} + eta.cov.{name}")
        cov_lines.append(f"{name}_pred ~ add(sig_cov_{name}) | DVID=={cov.dvid}")

    return patched + cov_lines


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def emit_nlmixr2_frem(
    spec: DSLSpec,
    covariates: Sequence[FREMCovariate],
    initial_estimates: dict[str, float] | None = None,
) -> str:
    """Emit a FREM-augmented nlmixr2 model function.

    Args:
        spec: The PK DSL specification. CovariateLink entries are stripped
            (FREM supersedes explicit covariate effects).
        covariates: FREM covariate metadata (typically from
            ``summarize_covariates``).
        initial_estimates: Optional overrides for structural parameters.

    Returns:
        R code string defining an nlmixr2-compatible model function with
        the joint Ω block and per-covariate observation endpoints.

    Raises:
        NotImplementedError: If the spec contains NODE modules (handled by
            the JAX/Diffrax emitter, not FREM).
        ValueError: If no covariates are supplied (caller should fall back
            to the standard emitter) or if a covariate has degenerate
            initial SD.
    """
    if spec.has_node_modules():
        msg = (
            "FREM emitter does not support NODE modules. "
            "NODE + FREM is a research-branch topic — use the classical emitter for now."
        )
        raise NotImplementedError(msg)
    if not covariates:
        msg = (
            "emit_nlmixr2_frem requires at least one covariate. "
            "Callers with no covariate missingness should use the standard emitter."
        )
        raise ValueError(msg)

    ini_lines = _emit_frem_ini(spec, covariates, initial_estimates=initial_estimates)
    model_lines = _emit_frem_model(spec, covariates)

    header = f"# APMODE FREM-augmented model: {spec.model_id}"
    lines = [
        header,
        f"# Covariates ({len(covariates)}): "
        + ", ".join(f"{c.name}@DVID={c.dvid}" for c in covariates),
        "function() {",
        "  ini({",
        *[f"    {line}" for line in ini_lines],
        "  })",
        "  model({",
        *[f"    {line}" for line in model_lines],
        "  })",
        "}",
    ]
    # IOV comment lines come from ``_emit_iov_occasion``; they are injected
    # by ``_emit_model`` into the returned block, so no extra wiring is
    # needed here.
    return "\n".join(lines)
