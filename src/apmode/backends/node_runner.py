# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE backend runner implementing BackendRunner protocol (ARCHITECTURE.md SS4.1).

Maps DSLSpec with NODE modules -> HybridPKODE -> training -> BackendResult.
Uses JAX/Diffrax/Equinox for neural ODE integration and training.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from pathlib import Path  # noqa: TC003 — used at runtime in run()
from typing import TYPE_CHECKING, Literal, TypedDict

# JAX/Equinox imports are deferred to prevent thread-pool initialization
# before R subprocess forks (os.fork() + JAX threads = potential deadlock).
# The actual imports happen in __init__ when execution_mode is set.
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.backends.node_trainer import TrainingConfig, predict_subject_conc, train_node
from apmode.bundle.models import ParameterEstimate
from apmode.errors import InvalidSpecError

if TYPE_CHECKING:
    import pandas as pd

    from apmode.backends.predictive_summary import SubjectSimulation
    from apmode.bundle.models import (
        BackendResult,
        ColumnMapping,
        DataManifest,
        NCASubjectDiagnostic,
        ScoringContract,
    )
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config

logger = logging.getLogger(__name__)

_JAX_PLATFORM_LOCKED: str | None = None


class _SubjectRequired(TypedDict):
    subject_id: str
    times: jax.Array
    observations: jax.Array
    y0: jax.Array
    obs_cmt: jax.Array


class SubjectRecord(_SubjectRequired, total=False):
    """Per-subject payload for NODE training.

    The ``dose_events`` field is only present on the event-driven piecewise
    path (multi-dose, delayed dose, or infusion); the legacy single-dose-at-t0
    path encodes the dose in ``y0`` instead and omits this field.

    Each event is a 5-tuple ``(time, amt, cmt, evid, inf_rate)`` where
    ``inf_rate`` is the signed per-event infusion rate (``> 0`` at an
    ``EVID in {1, 4}`` start, ``< 0`` at the paired synthetic ``EVID=9`` stop,
    ``0`` for a bolus). See ``node_trainer._solve_multidose_eager``.
    """

    dose_events: list[tuple[float, float, int, int, float, int]]


def configure_jax_platform(platform: Literal["cpu", "gpu"]) -> None:
    """Configure JAX platform globally (once per process).

    JAX's platform is a process-wide setting that effectively locks after
    first backend use. Calling this twice with different values logs a
    warning instead of silently drifting. Import side effects (e.g. another
    module's ``import jax``) can pin the platform before this is invoked.
    """
    global _JAX_PLATFORM_LOCKED
    if _JAX_PLATFORM_LOCKED is not None and platform != _JAX_PLATFORM_LOCKED:
        warnings.warn(
            f"JAX platform already set to '{_JAX_PLATFORM_LOCKED}'; "
            f"request for '{platform}' ignored. Platform is process-global.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    jax.config.update("jax_platform_name", platform)  # type: ignore[no-untyped-call]
    _JAX_PLATFORM_LOCKED = platform


def _node_pooled_contract(spec: DSLSpec) -> ScoringContract:
    """Build the pooled-NODE scoring contract from a DSLSpec.

    NODE runs use pooled NLL with no per-subject random effects, so the
    contract is fixed to pooled random-effect treatment. A future
    random-effects implementation must update this contract alongside the
    trainer behavior.

    Float precision is locked to ``float32`` — JAX defaults to float32
    on TPU/GPU and APMODE does not enable x64 on NODE runs.
    """
    from apmode.bundle.models import ScoringContract
    from apmode.bundle.scoring_contract import _obs_from_spec

    return ScoringContract(
        nlpd_kind="conditional",
        re_treatment="pooled",
        nlpd_integrator="none",
        blq_method="none",
        observation_model=_obs_from_spec(spec),
        float_precision="float32",
    )


def _pooled_cwres(
    model: HybridPKODE,
    subjects: list[SubjectRecord],
    sigma: float,
) -> tuple[float, float]:
    """Pooled/population standardized-residual mean and SD.

    Concatenates the standardized residuals ``(obs - PRED) / sigma`` across
    every subject (pooled — the NODE run has no per-subject random effects,
    so ``ScoringContract.re_treatment`` stays ``"pooled"``) and returns
    ``(mean, sd)``. This replaces the placeholder ``cwres_mean=0.0 /
    cwres_sd=1.0`` with the real population residual moments. PRED comes from
    :func:`apmode.backends.node_trainer.predict_subject_conc`, the same path
    the training likelihood minimises against.

    Returns the ``(0.0, 1.0)`` placeholder only when there is nothing to
    pool (no subjects / no observations); callers guard non-finite results.
    """
    residuals: list[jax.Array] = []
    for subj in subjects:
        pred = predict_subject_conc(model, subj)
        obs = subj["observations"]
        residuals.append((obs - pred) / sigma)
    if not residuals:
        return 0.0, 1.0
    pooled = jnp.concatenate(residuals)
    if pooled.shape[0] == 0:
        return 0.0, 1.0
    return float(jnp.mean(pooled)), float(jnp.std(pooled))


class NodeBackendRunner:
    """BackendRunner implementation for the in-process JAX/Diffrax NODE backend."""

    def __init__(
        self,
        work_dir: Path,
        execution_mode: Literal["cpu_deterministic", "gpu_fast"] = "cpu_deterministic",
        training_config: TrainingConfig | None = None,
        *,
        distill: bool = True,
        fidelity_min_r_squared: float = 0.8,
    ) -> None:
        self.work_dir = work_dir
        self.execution_mode = execution_mode
        self.training_config = training_config or TrainingConfig()
        # Functional distillation: produce a DistillationReport per NODE fit.
        # The report is observability + the input to orchestrator-side promotion;
        # ``fidelity_min_r_squared`` is the promotion gate (kept a runner param,
        # not a Gate3Config field, to avoid a gate policy_version bump).
        self.distill = distill
        self.fidelity_min_r_squared = fidelity_min_r_squared

        # Force CPU mode for determinism if requested
        if execution_mode == "cpu_deterministic":
            configure_jax_platform("cpu")

    def sample_posterior_predictive(
        self,
        training_result: object,
        subjects: object,
        *,
        n_sims: int,
        seed: int,
    ) -> None:
        """Draw a *random-effects* posterior-predictive sample (item A only).

        Returns ``None`` because the current NODE trainer is pooled and does
        not produce a per-subject random-effects posterior. This method is
        reserved for the between-subject-variability (BSV) path that unlocks
        formal VPC / NPDE (plan §4.2 / §4.4, mixed-effects item A).

        The *pooled* structural diagnostics that ship today — NPE,
        AUC/Cmax-BE and real pooled CWRES — do **not** go through here: they
        are built directly in :meth:`run` from the pooled structural PRED plus
        additive residual draws (see
        :meth:`_attach_predictive_diagnostics`). VPC / NPDE / PIT stay unset
        until BSV lands, because a pooled predictive distribution under-covers
        and would mislead a pharmacometrician (consensus Decision 3).

        A future BSV implementation must:
          1. Sample ``n_sims`` ETA vectors from the approximate posterior
             (Laplace or SVI) on the trained model's input-layer RE weights.
          2. Forward-solve the structural model via
             :func:`apmode.backends.node_trainer._solve_multidose_eager`
             at each subject's observed time vector for every draw.
          3. Return a ``list[SubjectSimulation]`` ready to feed
             :func:`apmode.backends.predictive_summary.
             build_predictive_diagnostics` — this time populating ``vpc`` /
             ``npde`` / ``pit_calibration`` too.

        The runtime stub emits a :class:`UserWarning` so a caller that
        accidentally wires this in sees the unsupported path loudly rather
        than treating the ``None`` return as an expected no-op.
        """
        _ = training_result, subjects, n_sims, seed  # explicit unused
        warnings.warn(
            "NodeBackendRunner.sample_posterior_predictive is not implemented "
            "and returns None. Random-effects infrastructure must land in "
            "node_trainer before this can emit real simulations. Gate 3 "
            "falls back to the CWRES NPE proxy for NODE candidates.",
            UserWarning,
            stacklevel=2,
        )
        return None

    async def run(
        self,
        spec: DSLSpec,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        seed: int,
        timeout_seconds: int | None = None,
        *,
        data_path: Path | None = None,
        split_manifest: dict[str, object] | None = None,
        gate3_policy: Gate3Config | None = None,
        nca_diagnostics: list[NCASubjectDiagnostic] | None = None,
        fixed_parameter: bool = False,
        test_data_path: Path | None = None,
    ) -> BackendResult:
        """Run NODE estimation.

        Args:
            spec: DSLSpec with NODE modules.
            data_manifest: Data manifest for the dataset.
            initial_estimates: NCA-derived initial estimates.
            seed: Random seed.
            timeout_seconds: Not enforced for in-process JAX (JAX is non-interruptible).
            data_path: Path to CSV data file.
            split_manifest: Split assignments (currently unused by NODE).
            gate3_policy: When supplied, drives the pooled posterior-predictive
                diagnostics (NPE + AUC/Cmax-BE) built from the structural PRED
                plus additive residual draws. VPC / NPDE / PIT are *not*
                populated — they require between-subject variability (item A).
                When ``None`` the predictive fields stay unset and Gate 3 falls
                back to the CWRES NPE proxy for NODE candidates.
            nca_diagnostics: Per-subject observed-data NCA QC records used to
                gate AUC/Cmax-BE eligibility. Matched to subjects by
                ``subject_id``; subjects without a record are NCA-ineligible.
            test_data_path: Accepted for protocol conformance; NODE does not
                honour held-out routing (only ``Nlmixr2Runner`` does today).

        Returns:
            BackendResult with backend="jax_node".

        Raises:
            InvalidSpecError: If spec has no NODE modules.
        """
        # NODE ignores held-out routing (protocol conformance only); the
        # gate3_policy / nca_diagnostics are consumed below.
        _ = test_data_path
        if fixed_parameter:
            msg = (
                "fixed_parameter=True not yet honoured by NODE runner "
                "(requires a no-refit evaluate() path in node_trainer — see "
                "loro_cv.py). Refusing to evaluate to "
                "avoid silent train/test leakage."
            )
            raise NotImplementedError(msg)
        # #23: the BackendRunner protocol promises timeout enforcement.
        # JAX training runs in-process and cannot be interrupted from
        # another task, so silently accepting timeout_seconds would be
        # a lie. Surface it explicitly — orchestrators that need a hard
        # wall-clock bound on NODE fits must spawn a subprocess
        # watchdog.
        if timeout_seconds is not None:
            msg = (
                f"NodeBackendRunner cannot enforce timeout_seconds={timeout_seconds}: "
                "JAX training is non-interruptible from the calling asyncio "
                "task. Pass timeout_seconds=None or run the NODE backend in "
                "a watchdog subprocess."
            )
            raise NotImplementedError(msg)
        from apmode.bundle.models import (
            BackendResult,
            BLQHandling,
            ConvergenceMetadata,
            DiagnosticBundle,
            GOFMetrics,
            IdentifiabilityFlags,
        )

        if not spec.has_node_modules():
            raise InvalidSpecError(
                "NodeBackendRunner requires NODE modules in spec",
                spec_id=spec.model_id,
            )

        start_time = time.monotonic()

        # Build ODE config from DSLSpec
        ode_config = self._build_ode_config(spec, initial_estimates)
        key = jax.random.PRNGKey(seed)

        # Build hybrid ODE model with transfer learning
        from apmode.backends.node_init import transfer_from_classical

        transfer_result = transfer_from_classical(
            ode_config,
            classical_estimates=initial_estimates,
            key=key,
            use_pretrained=True,
        )
        hybrid_model = transfer_result.model
        # Map transfer source to BackendResult's Literal type
        init_source: str = (
            "warm_start"
            if transfer_result.source in ("pretrained", "classical_transfer")
            else "fallback"
        )

        # Load and prepare data for training
        subjects = self._prepare_subjects(
            data_path,
            data_manifest,
            initial_estimates,
            n_cmt=ode_config.n_cmt,
        )

        # Train
        result = train_node(hybrid_model, subjects, self.training_config)

        wall_time = time.monotonic() - start_time

        # Build BackendResult
        param_estimates = self._extract_parameters(
            result.trained_model, result.trained_sigma, spec
        )

        # Count actual trainable parameters (MLP weights+biases + mechanistic log-params)
        n_trainable = sum(
            x.size for x in jax.tree.leaves(eqx.filter(result.trained_model, eqx.is_array))
        )
        n_trainable += 1  # log_sigma
        n_obs_total = sum(len(s["observations"]) for s in subjects)

        # Pooled/population standardized residuals replace the hard-coded
        # 0.0/1.0 placeholder. re_treatment stays "pooled" (no random
        # effects) — see ``_pooled_cwres`` and ``_node_pooled_contract``.
        cwres_mean, cwres_sd = _pooled_cwres(result.trained_model, subjects, result.trained_sigma)
        gof_cwres_mean = cwres_mean if math.isfinite(cwres_mean) else None
        gof_cwres_sd = cwres_sd if math.isfinite(cwres_sd) else None

        backend_result = BackendResult(
            model_id=spec.model_id,
            backend="jax_node",
            converged=result.converged,
            ofv=result.final_loss * 2,  # -2LL
            aic=result.final_loss * 2 + 2 * n_trainable,
            bic=result.final_loss * 2 + np.log(n_obs_total) * n_trainable,
            parameter_estimates=param_estimates,
            eta_shrinkage={},
            convergence_metadata=ConvergenceMetadata(
                method="adam",
                converged=result.converged,
                iterations=result.n_epochs,
                gradient_norm=None,
                minimization_status=result.minimization_status,
                wall_time_seconds=result.wall_time_seconds,
            ),
            # Pooled posterior-predictive diagnostics (NPE + AUC/Cmax-BE) are
            # attached below via ``_attach_predictive_diagnostics`` when a
            # ``gate3_policy`` is supplied — they are built from the pooled
            # structural PRED plus additive residual draws and copied through
            # ``apmode.backends.predictive_summary.build_predictive_diagnostics``.
            # ``vpc`` / ``npde`` / ``pit_calibration`` stay unset here and are
            # NOT populated by that path: a pooled predictive distribution
            # (no between-subject variability) under-covers and would mislead
            # (consensus Decision 3). Those await mixed-effects item A (plan
            # §4.2 / §4.4), which retains Ω + per-subject η MAP estimates and
            # feeds BSV-carrying draws into the same helper. When no policy is
            # supplied the predictive fields remain ``None`` and Gate 3 falls
            # back to the CWRES NPE proxy (see
            # ``apmode.governance.ranking._resolve_npe``).
            diagnostics=DiagnosticBundle(
                gof=GOFMetrics(
                    cwres_mean=gof_cwres_mean,
                    cwres_sd=gof_cwres_sd,
                    outlier_fraction=0.0,
                ),
                identifiability=IdentifiabilityFlags(
                    condition_number=None,
                    profile_likelihood_ci={},
                    ill_conditioned=False,
                ),
                blq=BLQHandling(
                    method="none",
                    n_blq=0,
                    blq_fraction=0.0,
                ),
                # v0.5.0 NODE runs pooled (no RE). M3 flips
                # re_treatment→"conditional_ebe" and nlpd_integrator
                # →"laplace_diag"/"laplace_blockdiag". Plan §8.
                scoring_contract=_node_pooled_contract(spec),
            ),
            wall_time_seconds=wall_time,
            backend_versions={
                "jax": str(jax.__version__),
                "python": _python_version(),
            },
            initial_estimate_source=init_source,
        )

        # Pooled posterior-predictive diagnostics (NPE + AUC/Cmax-BE only).
        if gate3_policy is not None:
            backend_result = self._attach_predictive_diagnostics(
                backend_result,
                model=result.trained_model,
                subjects=subjects,
                sigma=result.trained_sigma,
                spec=spec,
                gate3_policy=gate3_policy,
                nca_diagnostics=nca_diagnostics,
                seed=seed,
            )

        # Functional distillation (PRD §4.2.4): approximate the learned NODE
        # sub-function with a classical surrogate and attach the sealed report.
        # Promotion (fidelity-gated re-fit into Gate 3) is an orchestrator
        # concern; the runner only produces the report. distill() is a CPU-only
        # forward evaluation of the trained MLP — no ODE solve, no timeout risk.
        if self.distill:
            from apmode.backends.node_distillation import distill as run_distill

            report = run_distill(result.trained_model, spec.model_id)
            backend_result = backend_result.model_copy(update={"distillation": report})

        return backend_result

    def _build_subject_sims(
        self,
        model: HybridPKODE,
        subjects: list[SubjectRecord],
        sigma: float,
        *,
        n_sims: int,
        seed: int,
        nca_diagnostics: list[NCASubjectDiagnostic] | None,
    ) -> list[SubjectSimulation]:
        """Build per-subject pooled posterior-predictive simulation matrices.

        Pooled structural PRED (:func:`predict_subject_conc`) plus additive
        residual draws ``N(0, sigma)``, ``n_sims`` uniform across subjects.
        There is NO between-subject variability — this is a structural-fit
        predictive band, sufficient for NPE and AUC/Cmax-BE but deliberately
        not for VPC / NPDE (which the caller does not copy; item A).
        """
        from apmode.backends.predictive_summary import SubjectSimulation

        rng = np.random.default_rng(seed)
        diag_by_id = {d.subject_id: d for d in (nca_diagnostics or [])}
        sims: list[SubjectSimulation] = []
        for subj in subjects:
            pred = np.asarray(predict_subject_conc(model, subj), dtype=float)
            obs = np.asarray(subj["observations"], dtype=float)
            t_obs = np.asarray(subj["times"], dtype=float)
            n_obs = int(pred.shape[0])
            noise = rng.normal(0.0, sigma, size=(n_sims, n_obs))
            sims_matrix = pred[np.newaxis, :] + noise
            sid = subj["subject_id"]
            sims.append(
                SubjectSimulation(
                    subject_id=sid,
                    t_observed=t_obs,
                    observed_dv=obs,
                    sims_at_observed=sims_matrix,
                    nca_diagnostic=diag_by_id.get(sid),
                )
            )
        return sims

    def _attach_predictive_diagnostics(
        self,
        backend_result: BackendResult,
        *,
        model: HybridPKODE,
        subjects: list[SubjectRecord],
        sigma: float,
        spec: DSLSpec,
        gate3_policy: Gate3Config,
        nca_diagnostics: list[NCASubjectDiagnostic] | None,
        seed: int,
    ) -> BackendResult:
        """Copy pooled NPE + AUC/Cmax-BE onto the diagnostics (non-fatal).

        Consensus-scoped subset (plan §4.2): only ``npe_score``,
        ``auc_cmax_be_score`` and ``auc_cmax_source`` are copied — ``vpc`` /
        ``npde`` / ``pit_calibration`` are left unset because a pooled
        predictive distribution (no BSV) under-covers and misleads (Decision
        3). ``spec`` is forwarded so ``_observation_error_model`` drives the
        NPE residual scaling identically to nlmixr2.

        Any failure in the simulation / scoring path is swallowed: the
        predictive fields stay ``None`` and Gate 3 falls back to the CWRES NPE
        proxy, mirroring the nlmixr2 contract.
        """
        from apmode.backends.predictive_summary import build_predictive_diagnostics

        try:
            subject_sims = self._build_subject_sims(
                model,
                subjects,
                sigma,
                n_sims=gate3_policy.n_posterior_predictive_sims,
                seed=seed,
                nca_diagnostics=nca_diagnostics,
            )
            if not subject_sims:
                return backend_result
            predictive = build_predictive_diagnostics(subject_sims, policy=gate3_policy, spec=spec)
        except Exception:  # predictive path is best-effort — never fatal
            logger.warning(
                "NODE posterior-predictive path failed for model %s; Gate 3 "
                "falls back to the CWRES NPE proxy.",
                spec.model_id,
                exc_info=True,
            )
            return backend_result

        updated_diagnostics = backend_result.diagnostics.model_copy(
            update={
                "npe_score": predictive.npe_score,
                "auc_cmax_be_score": predictive.auc_cmax_be_score,
                "auc_cmax_source": predictive.auc_cmax_source,
            }
        )
        return backend_result.model_copy(update={"diagnostics": updated_diagnostics})

    def _build_ode_config(
        self,
        spec: DSLSpec,
        initial_estimates: dict[str, float],
    ) -> ODEConfig:
        """Build ODEConfig from DSLSpec."""
        from apmode.dsl.ast_models import NODEAbsorption, NODEElimination, OneCmt, TwoCmt

        # Determine compartment count from distribution
        if isinstance(spec.distribution, OneCmt):
            n_cmt = 1
        elif isinstance(spec.distribution, TwoCmt):
            n_cmt = 2
        else:
            n_cmt = 1  # default fallback

        # Determine NODE position and extract NODE config
        node_position: Literal["absorption", "elimination"] = "elimination"
        node_dim = 3
        constraint_template = "bounded_positive"

        if isinstance(spec.absorption, NODEAbsorption):
            node_position = "absorption"
            node_dim = spec.absorption.dim
            constraint_template = spec.absorption.constraint_template
        elif isinstance(spec.elimination, NODEElimination):
            node_position = "elimination"
            node_dim = spec.elimination.dim
            constraint_template = spec.elimination.constraint_template

        # Collect mechanistic params from initial estimates + spec
        mech_params: dict[str, float] = {}
        for name in ["ka", "V", "V1", "V2", "Q", "CL"]:
            if name in initial_estimates:
                mech_params[name] = initial_estimates[name]
        # Map V1 -> V if using TwoCmt
        if "V1" in mech_params and "V" not in mech_params:
            mech_params["V"] = mech_params["V1"]

        return ODEConfig(
            n_cmt=n_cmt,  # type: ignore[arg-type]
            node_position=node_position,
            constraint_template=constraint_template,
            node_dim=node_dim,
            mechanistic_params=mech_params,
        )

    def _prepare_subjects(
        self,
        data_path: Path | None,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        *,
        n_cmt: int = 1,
    ) -> list[SubjectRecord]:
        """Prepare subject data for training.

        If no data_path, creates synthetic subjects from initial estimates
        (for testing / mock mode).
        """
        if data_path is not None and data_path.exists():
            return self._load_subjects_from_csv(data_path, data_manifest, n_cmt=n_cmt)

        # Mock mode: create synthetic subjects from initial estimates
        return self._make_mock_subjects(data_manifest, initial_estimates, n_cmt=n_cmt)

    def _reject_unsupported_rows(self, event_df: pd.DataFrame, cm: ColumnMapping) -> None:
        """Fail loudly on dosing constructs the NODE runner does not yet support.

        Phase C1 supports bolus multi-dose and zero-order infusions. Steady-state
        (SS), other-event (EVID=2) rows, and observations outside the central
        compartment (CMT != 1) are not implemented; rejecting them explicitly
        prevents a silent, wrong fit.

        CMT convention (see also ``_load_subjects_from_csv`` /
        ``node_trainer._solve_multidose_eager``). The hybrid ODE state vector is
        ``[A_depot, A_central, ...]`` — index 0 is the absorption depot, index 1
        is central. Dose rows route by ``cmt_idx = CMT - 1`` (CMT=1 -> depot,
        CMT=2 -> central), whereas observations are always read from the central
        compartment and are required to carry the data label CMT=1. Because a
        zero-order infusion (RATE>0) into the depot is an absorption-delayed,
        materially wrong solve for the IV case, an infusion whose dose row is
        labelled CMT=1 (depot) is rejected here: IV infusions must target the
        central compartment (CMT=2 for the ``[depot, central]`` layout).
        """
        # Only treat a column as the steady-state control flag when the manifest
        # explicitly maps it. A dataset may carry a benign covariate literally
        # named "SS" (e.g. a simulation-parameter column with values like 99 on
        # observation rows); a hardcoded "SS" fallback would false-reject it.
        if cm.ss and cm.ss in event_df.columns and (event_df[cm.ss].fillna(0) != 0).any():
            raise InvalidSpecError(
                "NODE backend does not yet support steady-state dosing (SS != 0). "
                "Use the nlmixr2 backend for steady-state data.",
                spec_id="node_runner",
            )

        if (event_df[cm.evid] == 2).any():
            raise InvalidSpecError(
                "NODE backend does not yet support other-type events (EVID=2). "
                "Use the nlmixr2 backend for data with EVID=2 rows.",
                spec_id="node_runner",
            )

        cmt_col = cm.cmt or "CMT"

        # Reject infusions that land in the absorption depot (CMT<=1 -> index 0).
        # ``expand_infusion_events`` stores the (signed) rate in ``_INF_RATE``;
        # start rows are EVID in {1, 4} with ``_INF_RATE > 0``.
        if "_INF_RATE" in event_df.columns:
            inf_starts = event_df[
                event_df[cm.evid].isin([1, 4]) & (event_df["_INF_RATE"].fillna(0.0) > 0.0)
            ]
            if not inf_starts.empty:
                # No CMT column => every dose defaults to CMT=1 (depot).
                into_depot = (
                    (inf_starts[cmt_col].fillna(1) <= 1).any()
                    if cmt_col in inf_starts.columns
                    else True
                )
                if into_depot:
                    raise InvalidSpecError(
                        "NODE backend routes an infusion (RATE>0) labelled CMT=1 into "
                        "the absorption depot, not the central compartment, producing "
                        "an absorption-delayed and materially wrong concentration "
                        "curve. IV infusions must target the central compartment "
                        "(CMT=2 for the [depot, central] state layout). Re-label the "
                        "infusion dose rows or use the nlmixr2 backend.",
                        spec_id="node_runner",
                    )

        obs_rows = event_df[event_df[cm.evid] == 0]
        if cmt_col in obs_rows.columns and (obs_rows[cmt_col].fillna(1) != 1).any():
            raise InvalidSpecError(
                "NODE backend only supports observations in the central compartment "
                "(CMT=1); multi-endpoint observation routing is not yet implemented.",
                spec_id="node_runner",
            )

    def _load_subjects_from_csv(
        self,
        data_path: Path,
        data_manifest: DataManifest,
        *,
        n_cmt: int = 1,
    ) -> list[SubjectRecord]:
        """Load subject data from CSV with multi-dose event support."""
        import pandas as pd

        from apmode.data.dosing import build_event_table

        df = pd.read_csv(data_path)
        cm = data_manifest.column_mapping

        # Expand ADDL/II into explicit dose rows
        event_df = build_event_table(
            df,
            col_time=cm.time,
            col_id=cm.subject_id,
            col_evid=cm.evid,
            col_addl=cm.addl or "ADDL",
            col_ii=cm.ii or "II",
            col_amt=cm.amt,
            col_rate=cm.rate or "RATE",
            col_dur=cm.dur or "DUR",
            col_cmt=cm.cmt or "CMT",
            col_dv=cm.dv,
        )

        subjects: list[SubjectRecord] = []

        # Phase C1 MVP: infusions (RATE>0) ARE supported via the piecewise
        # eager solver. The following remain unimplemented and must fail
        # loudly rather than being silently ignored.
        self._reject_unsupported_rows(event_df, cm)

        cmt_col = cm.cmt or "CMT"

        for _sid, sdf in event_df.groupby(cm.subject_id):
            subject_id = str(_sid)
            obs_rows = sdf[sdf[cm.evid] == 0].sort_values(cm.time)
            # Include EVID=3 (resets) and EVID=9 (synthetic infusion stops).
            event_rows = sdf[sdf[cm.evid].isin([1, 3, 4, 9])].sort_values(cm.time)

            if len(obs_rows) == 0:
                continue

            times = jnp.array(obs_rows[cm.time].values, dtype=jnp.float32)
            observations = jnp.array(obs_rows[cm.dv].values, dtype=jnp.float32)
            n_states = 3 if n_cmt == 2 else 2

            if len(event_rows) == 0:
                # No doses/events — zero initial state
                y0 = jnp.zeros(n_states, dtype=jnp.float32)
                subjects.append(
                    {
                        "subject_id": subject_id,
                        "times": times,
                        "observations": observations,
                        "y0": y0,
                        "obs_cmt": jnp.array(1),
                    }
                )
                continue

            # Check if we can use the legacy single-dose JIT path:
            # exactly 1 dose event at TIME=0 with EVID=1, no resets, no infusion
            dose_events_only = event_rows[event_rows[cm.evid].isin([1, 4])]
            has_resets = (event_rows[cm.evid] == 3).any()
            has_infusion = "_INF_RATE" in sdf.columns and bool(
                (sdf["_INF_RATE"].fillna(0.0) != 0.0).any()
            )
            single_dose_at_zero = (
                len(dose_events_only) == 1
                and not has_resets
                and not has_infusion
                and float(dose_events_only[cm.time].iloc[0]) == 0.0
                and int(dose_events_only[cm.evid].iloc[0]) == 1
            )

            if single_dose_at_zero:
                # Legacy path: dose in y0 (JIT-compatible)
                dose_amt = float(dose_events_only[cm.amt].iloc[0])
                dose_cmt = (
                    int(dose_events_only[cmt_col].iloc[0])
                    if cmt_col in dose_events_only.columns
                    else 1
                )
                y0 = jnp.zeros(n_states, dtype=jnp.float32)
                idx = max(0, min(dose_cmt - 1, n_states - 1))
                y0 = y0.at[idx].set(dose_amt)
                subjects.append(
                    {
                        "subject_id": subject_id,
                        "times": times,
                        "observations": observations,
                        "y0": y0,
                        "obs_cmt": jnp.array(1),
                    }
                )
            else:
                # Multi-dose / delayed dose / reset / infusion: event-driven path.
                y0 = jnp.zeros(n_states, dtype=jnp.float32)
                all_events: list[tuple[float, float, int, int, float, int]] = []
                has_cmt = cmt_col in event_rows.columns
                has_inf = "_INF_RATE" in event_rows.columns
                has_inf_id = "_INF_ID" in event_rows.columns
                event_cols = [cm.time, cm.evid, cm.amt]
                cmt_pos = inf_pos = inf_id_pos = -1
                if has_cmt:
                    cmt_pos = len(event_cols)
                    event_cols.append(cmt_col)
                if has_inf:
                    inf_pos = len(event_cols)
                    event_cols.append("_INF_RATE")
                if has_inf_id:
                    inf_id_pos = len(event_cols)
                    event_cols.append("_INF_ID")
                for values in event_rows[event_cols].itertuples(index=False, name=None):
                    time_val = float(values[0])
                    evid = int(values[1])
                    amt = float(values[2]) if evid in (1, 4) else 0.0
                    cmt_val = int(values[cmt_pos]) if has_cmt else 1
                    # _INF_RATE is already signed (+ at start, - at the synthetic
                    # EVID=9 stop); _INF_ID links a stop to the start it ends.
                    inf_rate = float(values[inf_pos]) if has_inf else 0.0
                    inf_id = int(values[inf_id_pos]) if has_inf_id else -1
                    all_events.append((time_val, amt, cmt_val, evid, inf_rate, inf_id))

                subjects.append(
                    {
                        "subject_id": subject_id,
                        "times": times,
                        "observations": observations,
                        "y0": y0,
                        "obs_cmt": jnp.array(1),
                        "dose_events": all_events,
                    }
                )

        return subjects

    def _make_mock_subjects(
        self,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
        *,
        n_cmt: int = 1,
    ) -> list[SubjectRecord]:
        """Create synthetic subjects for mock/test mode."""
        n_subj = min(data_manifest.n_subjects, 10)
        subjects: list[SubjectRecord] = []

        key = jax.random.PRNGKey(0)
        for _i in range(n_subj):
            key, subkey = jax.random.split(key)
            times = jnp.linspace(0.5, 24.0, 8)
            ka = initial_estimates.get("ka", 1.0)
            V = initial_estimates.get("V", initial_estimates.get("V1", 30.0))
            ke = initial_estimates.get("CL", 2.0) / V
            dose = 100.0
            conc = (dose * ka) / (V * (ka - ke)) * (jnp.exp(-ke * times) - jnp.exp(-ka * times))
            conc = jnp.maximum(conc, 0.01)
            noise = 0.1 * conc * jax.random.normal(subkey, shape=times.shape)
            obs = jnp.maximum(conc + noise, 0.001)
            # Legacy single-dose: dose in y0[0] (compatible with JIT training)
            y0 = jnp.array([dose, 0.0, 0.0]) if n_cmt == 2 else jnp.array([dose, 0.0])
            subjects.append(
                {
                    "subject_id": f"MOCK{_i}",
                    "times": times,
                    "observations": obs,
                    "y0": y0,
                    "obs_cmt": jnp.array(1),
                }
            )

        return subjects

    @staticmethod
    def _extract_parameters(
        model: HybridPKODE,
        sigma: float,
        spec: object,
    ) -> dict[str, ParameterEstimate]:
        """Extract parameter estimates from trained model."""
        params: dict[str, ParameterEstimate] = {}

        # Mechanistic params
        params["ka"] = ParameterEstimate(
            name="ka", estimate=float(model.ka), category="structural"
        )
        params["CL"] = ParameterEstimate(
            name="CL", estimate=float(model.CL), category="structural"
        )
        params["V"] = ParameterEstimate(name="V", estimate=float(model.V), category="structural")

        if model.n_cmt == 2:
            params["V2"] = ParameterEstimate(
                name="V2", estimate=float(model.V2), category="structural"
            )
            params["Q"] = ParameterEstimate(
                name="Q", estimate=float(model.Q), category="structural"
            )

        # NODE weights are not individually interpretable, but we record
        # a summary: total weight norm as a structural "parameter"
        node_leaves = jax.tree.leaves(model.node)
        weight_arrays = [w for w in node_leaves if hasattr(w, "shape") and w.ndim >= 1]
        if weight_arrays:
            total_norm = float(sum(jnp.sum(w**2) for w in weight_arrays) ** 0.5)
            params["node_weight_norm"] = ParameterEstimate(
                name="node_weight_norm", estimate=total_norm, category="structural"
            )

        # Residual error
        params["sigma"] = ParameterEstimate(name="sigma", estimate=sigma, category="residual")

        return params


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
