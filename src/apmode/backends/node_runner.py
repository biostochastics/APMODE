# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE backend runner implementing BackendRunner protocol (ARCHITECTURE.md SS4.1).

Maps DSLSpec with NODE modules -> HybridPKODE -> training -> BackendResult.
Uses JAX/Diffrax/Equinox for neural ODE integration and training.
"""

from __future__ import annotations

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
from apmode.backends.node_trainer import TrainingConfig, train_node
from apmode.bundle.models import ParameterEstimate
from apmode.errors import InvalidSpecError

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, DataManifest, NCASubjectDiagnostic
    from apmode.dsl.ast_models import DSLSpec
    from apmode.governance.policy import Gate3Config

_JAX_PLATFORM_LOCKED: str | None = None


class _SubjectRequired(TypedDict):
    times: jax.Array
    observations: jax.Array
    y0: jax.Array
    obs_cmt: jax.Array


class SubjectRecord(_SubjectRequired, total=False):
    """Per-subject payload for NODE training.

    The ``dose_events`` field is only present on the event-driven piecewise
    path (multi-dose or delayed dose); the legacy single-dose-at-t0 path
    encodes the dose in ``y0`` instead and omits this field.
    """

    dose_events: list[tuple[float, float, int, int]]


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


class NodeBackendRunner:
    """BackendRunner implementation for JAX/Diffrax NODE backend.

    Phase 2: in-process JAX execution (no subprocess needed).
    """

    def __init__(
        self,
        work_dir: Path,
        execution_mode: Literal["cpu_deterministic", "gpu_fast"] = "cpu_deterministic",
        training_config: TrainingConfig | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.execution_mode = execution_mode
        self.training_config = training_config or TrainingConfig()

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
        """Draw posterior-predictive simulations for Gate 3 diagnostics.

        Returns ``None`` until NODE Phase 3 random-effects infrastructure
        (per-subject input-layer RE weights with Laplace-approximate
        posterior) lands in
        :mod:`apmode.backends.node_trainer`. The cross-paradigm ranker's
        uniform-drop rule already treats ``None`` as "backend did not
        emit sims" and falls back to the CWRES NPE proxy, so the return
        type here is load-bearing for Gate 3.

        When implemented, this method must:
          1. Sample ``n_sims`` ETA vectors from the Laplace-approximate
             posterior on the trained model's input-layer RE weights.
          2. Forward-solve the structural model via
             :func:`apmode.backends.node_trainer._solve_multidose_eager`
             at each subject's observed time vector for every draw.
          3. Return a ``list[SubjectSimulation]`` ready to feed
             :func:`apmode.backends.predictive_summary.
             build_predictive_diagnostics`.

        The runtime stub emits a :class:`UserWarning` so a caller that
        accidentally wires this in today sees the deferral loudly rather
        than treating the ``None`` return as an expected no-op.
        """
        _ = training_result, subjects, n_sims, seed  # explicit unused
        warnings.warn(
            "NodeBackendRunner.sample_posterior_predictive is a Phase 3 stub "
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
    ) -> BackendResult:
        """Run NODE estimation.

        Args:
            spec: DSLSpec with NODE modules.
            data_manifest: Data manifest for the dataset.
            initial_estimates: NCA-derived initial estimates.
            seed: Random seed.
            timeout_seconds: Not enforced for in-process JAX (JAX is non-interruptible).
            data_path: Path to CSV data file.
            split_manifest: Split assignments (unused in Phase 2 NODE).
            gate3_policy: Accepted for BackendRunner-protocol conformance but
                currently ignored — NODE posterior-predictive sampling is a
                Phase 3 stub (see ``sample_posterior_predictive``). Gate 3
                falls back to the CWRES NPE proxy for NODE candidates.
            nca_diagnostics: Accepted for protocol conformance. Unused until
                ``sample_posterior_predictive`` lands.

        Returns:
            BackendResult with backend="jax_node".

        Raises:
            InvalidSpecError: If spec has no NODE modules.
        """
        _ = gate3_policy, nca_diagnostics  # reserved for Phase 3 posterior-predictive path
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

        return BackendResult(
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
            # ``vpc`` / ``npe_score`` / ``auc_cmax_be_score`` are intentionally
            # left unset: the NODE backend does not yet emit posterior-
            # predictive simulations. The canonical path is the shared helper
            # in ``apmode.backends.predictive_summary.build_predictive_diagnostics``
            # — drop in by (1) extending ``node_trainer.train_node`` to retain
            # the Ω / per-subject eta MAP estimates for input-layer random
            # effects (PRD §4.2.4 R6), (2) adding
            # ``sample_posterior_predictive`` that draws n_sims ETA vectors
            # via Laplace approximation at the MAP, forward-solves via
            # ``_solve_multidose_eager`` at each subject's observed times,
            # (3) calling ``build_predictive_diagnostics(subject_sims,
            # policy=gate3_policy)`` and ``diagnostics.model_copy(update=...)``
            # with the four returned fields.  Random-effects infrastructure
            # is Phase 3 scope per ``node_trainer.py`` module docstring.
            # Until then Gate 3 falls back to the CWRES NPE proxy for NODE
            # candidates (see ``apmode.governance.ranking._resolve_npe``).
            diagnostics=DiagnosticBundle(
                gof=GOFMetrics(
                    cwres_mean=0.0,
                    cwres_sd=1.0,
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
            ),
            wall_time_seconds=wall_time,
            backend_versions={
                "jax": str(jax.__version__),
                "python": _python_version(),
            },
            initial_estimate_source=init_source,
        )

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
        )

        subjects: list[SubjectRecord] = []

        # Reject infusions for NODE (not yet consumed in piecewise solver)
        rate_col = cm.rate or "RATE"
        if rate_col in event_df.columns and (event_df[rate_col].fillna(0) > 0).any():
            raise InvalidSpecError(
                "NODE backend does not yet support infusion dosing (RATE > 0). "
                "Use the nlmixr2 backend for infusion data.",
                spec_id="node_runner",
            )

        for _sid, sdf in event_df.groupby(cm.subject_id):
            obs_rows = sdf[sdf[cm.evid] == 0].sort_values(cm.time)
            # Include EVID=3 (resets) in event extraction
            event_rows = sdf[sdf[cm.evid].isin([1, 3, 4])].sort_values(cm.time)

            if len(obs_rows) == 0:
                continue

            times = jnp.array(obs_rows[cm.time].values, dtype=jnp.float32)
            observations = jnp.array(obs_rows[cm.dv].values, dtype=jnp.float32)
            n_states = 3 if n_cmt == 2 else 2
            cmt_col = cm.cmt or "CMT"

            if len(event_rows) == 0:
                # No doses/events — zero initial state
                y0 = jnp.zeros(n_states, dtype=jnp.float32)
                subjects.append(
                    {
                        "times": times,
                        "observations": observations,
                        "y0": y0,
                        "obs_cmt": jnp.array(1),
                    }
                )
                continue

            # Check if we can use the legacy single-dose JIT path:
            # exactly 1 dose event at TIME=0 with EVID=1 and no resets
            dose_events_only = event_rows[event_rows[cm.evid].isin([1, 4])]
            has_resets = (event_rows[cm.evid] == 3).any()
            single_dose_at_zero = (
                len(dose_events_only) == 1
                and not has_resets
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
                        "times": times,
                        "observations": observations,
                        "y0": y0,
                        "obs_cmt": jnp.array(1),
                    }
                )
            else:
                # Multi-dose / delayed dose / reset: use event-driven piecewise path
                y0 = jnp.zeros(n_states, dtype=jnp.float32)
                all_events: list[tuple[float, float, int, int]] = []
                for _, row in event_rows.iterrows():
                    evid = int(row[cm.evid])
                    amt = float(row[cm.amt]) if evid in (1, 4) else 0.0
                    cmt_val = int(row[cmt_col]) if cmt_col in row.index else 1
                    all_events.append((float(row[cm.time]), amt, cmt_val, evid))

                subjects.append(
                    {
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
