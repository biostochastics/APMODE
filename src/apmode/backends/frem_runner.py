# SPDX-License-Identifier: GPL-2.0-or-later
"""FREM execution helper.

Composes the FREM emitter with :class:`Nlmixr2Runner` to run a single
FREM-augmented fit. FREM is an alternative to Multiple Imputation for
covariate-missingness handling: rather than looping the search engine
across m imputed datasets, FREM augments the dataset once with
covariate-observation rows and lets the NLME likelihood marginalize
over missing covariates through the joint Ω matrix.

The helper:

1. Computes per-covariate summary stats (mean, SD, transform) from the
   source data via :func:`summarize_covariates`.
2. Augments the data with DVID-routed covariate observation rows via
   :func:`prepare_frem_data`, writing the result to a new CSV in the
   run's work directory.
3. Emits the FREM-augmented R model via :func:`emit_nlmixr2_frem`.
4. Calls :meth:`Nlmixr2Runner.run` with ``compiled_code_override`` set
   to the FREM code and ``data_path`` pointing at the augmented CSV.

Estimator selection: FREM requires FOCE-I. SAEM treats subject-level
covariate observations as dynamic sampling targets and collapses the
random-effect variance (verified against nlmixr2 5.0). Callers must
pass a runner configured with ``estimation=["focei"]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from apmode.dsl.frem_emitter import (
    emit_nlmixr2_frem,
    prepare_frem_data,
    summarize_covariates,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

    from apmode.backends.nlmixr2_runner import Nlmixr2Runner
    from apmode.bundle.models import BackendResult, DataManifest, NCASubjectDiagnostic
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config


async def run_frem_fit(
    *,
    spec_template: DSLSpec,
    df: pd.DataFrame,
    data_path: Path,
    data_manifest: DataManifest,
    covariate_names: Sequence[str],
    runner: Nlmixr2Runner,
    work_dir: Path,
    seed: int,
    timeout_seconds: int | None = None,
    transforms: dict[str, str] | None = None,
    binary_encode_overrides: dict[str, dict[object, int]] | None = None,
    initial_estimates: dict[str, float] | None = None,
    gate3_policy: Gate3Config | None = None,
    nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
) -> BackendResult:
    """Execute a single FREM fit end-to-end.

    Args:
        spec_template: Base PK DSL spec. ``CovariateLink`` entries are
            stripped inside the emitter because FREM supersedes explicit
            covariate effects with the joint Ω. The spec's ``model_id``
            is reused for bundle bookkeeping.
        df: Full source DataFrame (canonical APMODE schema). Covariate
            columns matching ``covariate_names`` must exist; NaN values
            in those columns are handled by FREM natively.
        data_path: Original data CSV path (informational only — the
            augmented CSV path is passed to the runner).
        data_manifest: Data manifest for the runner.
        covariate_names: Ordered list of covariates to treat as FREM
            observations. Duplicates and degenerate covariates are
            rejected by ``summarize_covariates``.
        runner: Configured ``Nlmixr2Runner``. Should be set up with
            ``estimation=["focei"]``; SAEM is not supported for FREM
            endpoints.
        work_dir: Directory for the augmented CSV. Must already exist.
        seed: RNG seed forwarded to the runner.
        timeout_seconds: Optional timeout forwarded to the runner.
        transforms: Optional per-covariate transform overrides (e.g.,
            ``{"WT": "log"}``). Passed through to
            ``summarize_covariates``.
        initial_estimates: Optional structural-parameter initial-estimate
            overrides for the emitted model.

    Returns:
        ``BackendResult`` from the FREM fit. ``result.backend`` is
        reported as ``"nlmixr2"`` (the actual backend); FREM-specific
        metadata lives in the run directory's ``request.json``.
    """
    del data_path  # kept in signature for symmetry with classical fits

    summaries = summarize_covariates(
        df,
        list(covariate_names),
        transforms=transforms,
        binary_encode_overrides=binary_encode_overrides,
    )
    augmented = prepare_frem_data(df, summaries)

    aug_path = work_dir / "frem_augmented.csv"
    augmented.to_csv(aug_path, index=False)

    frem_code = emit_nlmixr2_frem(spec_template, summaries, initial_estimates=initial_estimates)

    return await runner.run(
        spec=spec_template,
        data_manifest=data_manifest,
        initial_estimates=initial_estimates or {},
        seed=seed,
        timeout_seconds=timeout_seconds,
        data_path=aug_path.resolve(),
        compiled_code_override=frem_code,
        gate3_policy=gate3_policy,
        nca_diagnostics=nca_diagnostics,
    )
