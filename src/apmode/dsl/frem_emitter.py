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

Scope of this emitter:
- Static (baseline-only) and time-varying subject-level covariates.
  ``prepare_frem_data`` writes one observation row per subject per
  covariate at the subject's baseline time by default; pass
  ``time_varying_covariates`` to ``prepare_frem_data`` to emit one
  row per *visit* (per subject x time x covariate) for covariates
  whose value changes over time. nlmixr2 then treats the per-visit
  observations as repeated measurements of the joint Ω structure.
- Continuous, log-transformed continuous, and binary categorical
  covariates. ``transform="binary"`` accepts 0/1-coded categorical
  inputs and uses an additive-normal endpoint — the off-diagonal Ω
  entry between the binary covariate eta and a PK eta then estimates
  the linear association between the PK parameter and the binary
  group (a common categorical-FREM compromise; for proper
  logit-likelihood handling you would need a custom binomial endpoint
  beyond this emitter's scope).
- Multi-level categorical covariates (k > 2 categories): one-hot
  encode upstream into k-1 binary indicators and pass each indicator
  as a ``"binary"`` covariate.
- Joint Ω is structured as a block matrix with the PK IIV block (from
  the underlying spec) in the top-left and covariate variances on the
  diagonal of the bottom-right block; off-diagonal PKxcovariate
  covariances initialize to zero and are estimated by nlmixr2.
- CovariateLink entries in the spec are dropped because FREM replaces
  explicit covariate-on-parameter effects with the joint random-effect
  structure.
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


# DVID assignment for FREM covariate observations.
#
# nlmixr2 assigns each endpoint an integer DVID in declaration order: the PK
# endpoint (declared first) becomes DVID 1, the first covariate endpoint
# becomes DVID 2, and so on. The emitter must therefore hand the imputer a
# ``FREMCovariate`` list whose ``dvid`` values start at 2 and advance by 1
# per covariate — otherwise the data written by ``prepare_frem_data`` would
# not match nlmixr2's implicit endpoint numbering and routing would fail
# (verified against nlmixr2 5.0).
#
# ``prepare_frem_data`` raises a collision error when source data already
# uses any DVID in this range; callers with multi-analyte data must remap
# their PK DVID scheme to stay at 1 (or below the covariate range) before
# augmentation.
_FREM_DVID_OFFSET: int = 2

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
        time_varying: When True, ``prepare_frem_data`` writes one
            observation row per (subject x distinct TIME) where the
            covariate value is observed (non-NaN), instead of a single
            baseline-row per subject. The covariate residual SD
            (``sig_cov_*``) is also left **estimable** (not fixed), so
            it absorbs within-subject variation while the joint Ω
            entry continues to capture between-subject variance and
            the PK-covariate covariance. Default ``False`` (baseline
            FREM, one obs/subject, residual fixed for identifiability).
        transform: Scale on which the covariate is represented in the
            model. ``"identity"`` (default) fits the covariate on its
            raw scale; ``"log"`` fits on the natural-log scale and is
            recommended for strictly positive, right-skewed covariates
            (weight, creatinine, etc.) so the Ω block stays well
            conditioned; ``"binary"`` treats a 0/1-coded categorical
            covariate as continuous with an additive-normal endpoint —
            the off-diagonal Ω entry between the binary covariate eta
            and a PK eta then estimates the linear association between
            PK clearance/volume and the binary group, a common
            categorical-FREM compromise (see module docstring on
            multi-level coding). Inputs are validated to be exactly
            ``{0, 1}`` for ``"binary"``; for multi-level categorical
            covariates, one-hot encode to k-1 binary indicators
            upstream and pass each indicator as a ``"binary"``
            covariate.
    """

    name: str
    mu_init: float
    sigma_init: float
    dvid: int
    epsilon_sd: float = _DEFAULT_COV_EPS_SD
    transform: str = "identity"
    time_varying: bool = False

    def __post_init__(self) -> None:
        _sanitize_r_name(self.name)
        if self.transform not in ("identity", "log", "binary"):
            msg = (
                f"FREMCovariate {self.name!r}: transform must be 'identity', "
                f"'log', or 'binary', got {self.transform!r}"
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
    binary_encode_overrides: dict[str, dict[object, int]] | None = None,
    encoding_hints_out: list[object] | None = None,
) -> list[FREMCovariate]:
    """Compute per-covariate ``(mu_init, sigma_init)`` from observed data.

    Uses each subject's **baseline** row (minimum ``TIME``) rather than the
    first row as stored; in randomly-ordered data these can differ and
    baseline is the pharmacometric convention for subject-level covariates.
    Subjects with a missing value for a given covariate are excluded from
    that covariate's mean/SD only — this matches the FREM likelihood
    treatment of missingness (those subjects simply have no observation
    row for that covariate).

    Binary categorical covariates (``transforms[name] == "binary"``) are
    auto-remapped to ``{0, 1}`` via
    ``apmode.data.categorical_encoding.auto_remap_binary_columns`` before
    summary statistics are computed. Recognised forms include booleans,
    1-indexed integers, and standard string pairs (M/F, Yes/No, etc.);
    unknown two-level string pairs get a deterministic alphabetic-order
    remap with a warning logged via the encoding hint. Override the
    auto-detected polarity with ``binary_encode_overrides``, e.g.
    ``{"SEX": {"male": 0, "female": 1}}``.

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
        binary_encode_overrides: Optional per-column explicit remap dict
            applied before summary statistics. Takes precedence over the
            auto-detection. Use when the auto-detected polarity is
            wrong for the analysis (rare; auto-detection is
            alphabetically deterministic).

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

    # Auto-remap binary categorical columns to canonical {0, 1} *before*
    # baseline summarization. Without this the downstream
    # ``binary``-transform validator would reject native string pairs
    # (warfarin's ``"male"``/``"female"``) or 1-indexed integers
    # (mavoglurant's ``1``/``2``) — both common pharmacometric
    # conventions. Auto-detection logic + override surface live in
    # ``apmode.data.categorical_encoding`` so the same rules can be
    # surfaced via the CLI ``inspect`` / ``validate`` reports.
    binary_targets = [n for n, t in tr_map.items() if t == "binary" and n in df.columns]
    if binary_targets:
        from apmode.data.categorical_encoding import auto_remap_binary_columns

        df, _hints = auto_remap_binary_columns(
            df, binary_targets, overrides=binary_encode_overrides, apply=True
        )
        # Provenance capture: when the caller supplies a list, surface
        # every encoding decision so the orchestrator can write
        # ``categorical_encoding_provenance.json`` in the run bundle.
        # The list is appended to in place for backward-compatible API.
        if encoding_hints_out is not None:
            encoding_hints_out.extend(_hints)

    baseline_idx = df.groupby(id_col)[time_col].idxmin()
    per_subj = df.loc[baseline_idx].set_index(id_col)

    # Detect per-covariate time-varying status by checking whether any
    # subject has more than one distinct non-NaN value across the full
    # dataset. The upstream ``EvidenceManifest.time_varying_covariates``
    # flag is a binary aggregate; per-covariate resolution is needed so
    # we emit the right ``time_varying`` flag on each ``FREMCovariate``.
    tv_flags: dict[str, bool] = {}
    for name in covariate_names:
        if name not in df.columns:
            tv_flags[name] = False
            continue
        per_subj_unique = df.groupby(id_col)[name].apply(lambda s: int(s.dropna().nunique()))
        tv_flags[name] = bool((per_subj_unique > 1).any())

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
        elif transform == "binary":
            unique_vals = set(observed.unique().tolist())
            if not unique_vals.issubset({0.0, 1.0}):
                msg = (
                    f"Covariate {name!r}: binary transform requires values "
                    f"in {{0, 1}}, found {sorted(unique_vals)}. For multi-level "
                    f"categorical covariates, one-hot encode to k-1 binary "
                    f"indicators upstream and pass each as a 'binary' covariate."
                )
                raise ValueError(msg)
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
                time_varying=tv_flags[name],
            )
        )
    return summaries


def _apply_cov_transform(
    cov: FREMCovariate,
    cov_float: float,
    sid: object,
) -> float:
    """Apply the covariate's transform, validating positivity / 0-1 coding."""
    if cov.transform == "log":
        if cov_float <= 0:
            msg = (
                f"Subject {sid!r}: covariate {cov.name!r} has non-positive value "
                f"{cov_float} but transform='log' — skip this subject or drop "
                f"the covariate."
            )
            raise ValueError(msg)
        return float(np.log(cov_float))
    if cov.transform == "binary" and cov_float not in (0.0, 1.0):
        msg = (
            f"Subject {sid!r}: covariate {cov.name!r} has value {cov_float} but "
            f"transform='binary' requires {{0, 1}}. For multi-level categorical "
            f"covariates, one-hot encode upstream."
        )
        raise ValueError(msg)
    return cov_float


def _build_aug_row(
    src_row: object,
    id_col: str,
    time_col: str,
    sid: object,
    obs_time: float,
    dv: float,
    dvid: int,
) -> dict[str, object]:
    """Construct one augmentation row from a source row + emitted DV.

    Per-subject covariate columns are inherited from the source row so
    rxode2's state machine has a complete event record at the
    augmentation time, but event/dose/censoring columns that belong to
    the original PK observation context are explicitly cleared. Leaving
    CENS/LIMIT/BLQ_FLAG/RATE/DUR on the covariate row would route the
    covariate observation through the PK BLQ likelihood or treat it as
    part of an infusion envelope.
    """
    new_row: dict[str, object] = {k: v for k, v in src_row.items()}  # type: ignore[attr-defined]
    new_row[id_col] = sid
    new_row[time_col] = obs_time
    if "EVID" in new_row:
        new_row["EVID"] = 0
    if "AMT" in new_row:
        new_row["AMT"] = 0.0
    if "MDV" in new_row:
        new_row["MDV"] = 0
    # Clear columns that would contaminate the covariate-endpoint likelihood.
    for col, reset in (
        ("CENS", 0),
        ("LIMIT", 0.0),
        ("BLQ_FLAG", 0),
        ("RATE", 0.0),
        ("DUR", 0.0),
        ("SS", 0),  # steady state marker — covariate obs is not in SS
        ("II", 0.0),  # inter-dose interval — N/A for covariate obs
    ):
        if col in new_row:
            new_row[col] = reset
    # CMT is deliberately not cleared here: nlmixr2 routes covariate
    # endpoints via the DVID column in this emitter's design, and the
    # source row's CMT is typically the dose compartment for the subject.
    new_row["DV"] = dv
    new_row["DVID"] = dvid
    return new_row


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

    # Auto-remap any covariate marked as ``transform="binary"`` whose
    # source values are not yet canonical {0, 1}. summarize_covariates
    # already does this for the *summary* statistics it computes, but
    # operates on a local copy; the caller's DataFrame still carries
    # the native encoding when it reaches here. The remap is idempotent
    # for already-canonical data and ensures the per-subject row
    # writer below sees 0/1 floats instead of strings or 1/2 codes.
    binary_targets = [
        c.name for c in covariates if c.transform == "binary" and c.name in df.columns
    ]
    if binary_targets:
        from apmode.data.categorical_encoding import auto_remap_binary_columns

        df, _hints = auto_remap_binary_columns(df, binary_targets, apply=True)

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

    static_covs = [c for c in covariates if not c.time_varying]
    tv_covs = [c for c in covariates if c.time_varying]

    aug_rows: list[dict[str, object]] = []

    # Static covariates: one observation row per subject at baseline.
    for subj_row in first_per_subj.to_dict("records"):
        sid = subj_row[id_col]
        baseline_time = float(subj_row[time_col])
        for cov in static_covs:
            cov_value = subj_row.get(cov.name)
            if pd.isna(cov_value):
                continue
            cov_float = _apply_cov_transform(cov, float(cov_value), sid)
            aug_rows.append(
                _build_aug_row(subj_row, id_col, time_col, sid, baseline_time, cov_float, cov.dvid)
            )

    # Time-varying covariates: one observation row per (subject, time) where
    # the covariate value is non-NaN. Dedup is keyed on ``(subject, TIME)``
    # alone and the stored ``(subject, TIME) → value`` map is consulted on
    # subsequent rows to detect conflicting values at the same timepoint —
    # an input-data ambiguity that would otherwise produce multiple
    # contradictory observation rows and distort the likelihood.
    # Raises ``ValueError`` with a clear pointer so callers can dedupe
    # upstream (e.g., take the first observation per subject/time or
    # average before calling prepare_frem_data).
    _TV_CONFLICT_TOL: float = 1e-9
    for cov in tv_covs:
        per_subj_groups = out.groupby(id_col, sort=False)
        for sid, subj_df in per_subj_groups:
            seen_at_time: dict[float, float] = {}
            for row in subj_df.to_dict("records"):
                cov_value = row.get(cov.name)
                if pd.isna(cov_value):
                    continue
                cov_float = _apply_cov_transform(cov, float(cov_value), sid)
                t = float(row[time_col])
                if t in seen_at_time:
                    prior = seen_at_time[t]
                    if abs(prior - cov_float) > _TV_CONFLICT_TOL:
                        msg = (
                            f"Subject {sid!r}: covariate {cov.name!r} has "
                            f"conflicting values at TIME={t}: {prior} vs "
                            f"{cov_float}. Deduplicate or aggregate upstream "
                            f"before calling prepare_frem_data."
                        )
                        raise ValueError(msg)
                    continue
                seen_at_time[t] = cov_float
                aug_rows.append(_build_aug_row(row, id_col, time_col, sid, t, cov_float, cov.dvid))

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
    # For static (baseline-only) covariates we ``fix(...)`` the residual
    # because one obs/subject perfectly confounds eta and sigma; fixing
    # the residual lets the eta absorb all between-subject variance
    # (standard FREM practice, Karlsson 2011).
    # For time-varying covariates we leave the residual estimable so it
    # can absorb within-subject variation while the eta continues to
    # capture between-subject variance.
    lines.append("")
    lines.append("# FREM covariate residual error")
    lines.append(
        "# (fixed for static covariates so eta absorbs BSV; "
        "estimable for time-varying so it absorbs WSV)"
    )
    for cov in covariates:
        name = _sanitize_r_name(cov.name)
        if cov.time_varying:
            lines.append(f"sig_cov_{name} <- {cov.epsilon_sd}")
        else:
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


def _emit_frem_model(
    spec: DSLSpec,
    covariates: Sequence[FREMCovariate],
) -> list[str]:
    """Emit the model block with FREM covariate-observation endpoints.

    Endpoint-routing design (verified live against nlmixr2 5.0,
    2026-04-14): nlmixr2's grammar requires a **bare symbol**
    (compartment/endpoint name) after the ``|`` in an endpoint
    definition — not a condition. ``| DVID==N`` is rejected with
    ``the condition 'DVID == N' must be a simple name``. Instead,
    each endpoint's DVID is assigned **implicitly by declaration
    order** (PK first = DVID 1, first covariate = DVID 2, etc.),
    and the runtime data must carry a matching ``DVID`` column.
    ``prepare_frem_data`` writes the DVID column so routing is
    data-driven, not model-driven.

    The base PK endpoint line is emitted unchanged.
    """
    frem_spec = _strip_covariate_links(spec)
    base_lines = _emit_model(frem_spec)

    cov_lines: list[str] = [""]
    cov_lines.append("# FREM covariate observation endpoints (routed via DVID column in data)")
    for cov in covariates:
        name = _sanitize_r_name(cov.name)
        # Predicted covariate value: mean + subject-specific eta. For
        # ``transform="log"`` the DV in the augmented data is already on
        # the log scale (set by ``prepare_frem_data``), so the same
        # algebraic form applies — the scale is carried by the data, not
        # by the emitted model expression.
        cov_lines.append(f"{name}_pred <- mu_{name} + eta.cov.{name}")
        cov_lines.append(f"{name}_pred ~ add(sig_cov_{name})")

    return base_lines + cov_lines


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
