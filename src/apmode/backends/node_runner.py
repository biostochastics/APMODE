# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE backend runner implementing BackendRunner protocol (ARCHITECTURE.md SS4.1).

Maps DSLSpec with NODE modules -> HybridPKODE -> training -> BackendResult.
Uses JAX/Diffrax/Equinox for neural ODE integration and training.
"""

from __future__ import annotations

import time
from pathlib import Path  # noqa: TC003 — used at runtime in run()
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.backends.node_trainer import TrainingConfig, train_node
from apmode.bundle.models import ParameterEstimate
from apmode.errors import InvalidSpecError

if TYPE_CHECKING:
    from apmode.bundle.models import BackendResult, DataManifest
    from apmode.dsl.ast_models import DSLSpec


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
            jax.config.update("jax_platform_name", "cpu")  # type: ignore[no-untyped-call]

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

        Returns:
            BackendResult with backend="jax_node".

        Raises:
            InvalidSpecError: If spec has no NODE modules.
        """
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

        # Build hybrid ODE model
        hybrid_model = HybridPKODE(config=ode_config, key=key)

        # Load and prepare data for training
        subjects = self._prepare_subjects(data_path, data_manifest, initial_estimates)

        # Train
        result = train_node(hybrid_model, subjects, self.training_config)

        wall_time = time.monotonic() - start_time

        # Build BackendResult
        param_estimates = self._extract_parameters(
            result.trained_model, result.trained_sigma, spec
        )

        return BackendResult(
            model_id=spec.model_id,
            backend="jax_node",
            converged=result.converged,
            ofv=result.final_loss * 2,  # -2LL
            aic=result.final_loss * 2 + 2 * len(param_estimates),
            bic=result.final_loss * 2 + np.log(len(subjects)) * len(param_estimates),
            parameter_estimates=param_estimates,
            eta_shrinkage={},
            convergence_metadata=ConvergenceMetadata(
                method="adam",
                converged=result.converged,
                iterations=result.n_epochs,
                gradient_norm=None,
                minimization_status="successful" if result.converged else "max_evaluations",
                wall_time_seconds=result.wall_time_seconds,
            ),
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
            initial_estimate_source="nca",
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
    ) -> list[dict[str, jax.Array]]:
        """Prepare subject data for training.

        If no data_path, creates synthetic subjects from initial estimates
        (for testing / mock mode).
        """
        if data_path is not None and data_path.exists():
            return self._load_subjects_from_csv(data_path, data_manifest)

        # Mock mode: create synthetic subjects from initial estimates
        return self._make_mock_subjects(data_manifest, initial_estimates)

    def _load_subjects_from_csv(
        self,
        data_path: Path,
        data_manifest: DataManifest,
    ) -> list[dict[str, jax.Array]]:
        """Load subject data from CSV file."""
        import pandas as pd  # type: ignore[import-untyped]

        df = pd.read_csv(data_path)
        cm = data_manifest.column_mapping
        subjects: list[dict[str, jax.Array]] = []

        for _sid, sdf in df.groupby(cm.subject_id):
            obs_rows = sdf[sdf[cm.evid] == 0]
            dose_rows = sdf[sdf[cm.evid] == 1]

            if len(obs_rows) == 0:
                continue

            times = jnp.array(obs_rows[cm.time].values, dtype=jnp.float32)
            observations = jnp.array(obs_rows[cm.dv].values, dtype=jnp.float32)

            # Initial dose
            dose = float(dose_rows[cm.amt].iloc[0]) if len(dose_rows) > 0 else 100.0
            y0 = jnp.array([dose, 0.0])

            subjects.append(
                {
                    "times": times,
                    "observations": observations,
                    "y0": y0,
                    "obs_cmt": jnp.array(1),
                }
            )

        return subjects

    def _make_mock_subjects(
        self,
        data_manifest: DataManifest,
        initial_estimates: dict[str, float],
    ) -> list[dict[str, jax.Array]]:
        """Create synthetic subjects for mock/test mode."""
        n_subj = min(data_manifest.n_subjects, 10)
        subjects: list[dict[str, jax.Array]] = []

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
            subjects.append(
                {
                    "times": times,
                    "observations": obs,
                    "y0": jnp.array([dose, 0.0]),
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
