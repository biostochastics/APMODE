# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE training loop for population PK fitting (PRD SS4.2.4).

Fits the hybrid ODE to population PK data by optimizing MLP weights +
mechanistic params. Uses Optax for optimization with early stopping.

Training objective: population negative log-likelihood (pooled).
  theta = MLP weights + mechanistic params (ka, V) + sigma

Phase 2 limitation: performs pooled (naive) population fit. Per-subject
random effects via Laplace approximation are deferred to Phase 3.
The model.apply_random_effects() API is available for future RE integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from apmode.backends.node_ode import HybridPKODE  # noqa: TC001 — used at runtime


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for NODE training."""

    epochs: int = 200
    learning_rate: float = 1e-3
    grad_clip: float = 10.0
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4
    execution_mode: Literal["cpu_deterministic", "gpu_fast"] = "cpu_deterministic"
    sigma_init: float = 0.3


@dataclass
class TrainingResult:
    """Result of NODE training."""

    trained_model: HybridPKODE
    trained_sigma: float
    final_loss: float
    n_epochs: int
    converged: bool
    loss_history: list[float] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    method: str = "adam"


def _population_nll(
    model: HybridPKODE,
    log_sigma: jax.Array,
    subjects: list[dict[str, jax.Array]],
) -> jax.Array:
    """Population negative log-likelihood (normal residual model).

    For each subject: solve ODE at observation times, compute
    -log N(y_obs | y_pred, sigma^2).
    """
    sigma = jnp.exp(log_sigma)
    total_nll = jnp.array(0.0)

    for subj in subjects:
        times = subj["times"]
        obs = subj["observations"]
        y0 = subj["y0"]
        cmt_idx = int(subj.get("obs_cmt", jnp.array(1)))

        # Solve ODE for this subject
        sol = model.solve(y0, times)
        pred = sol[:, cmt_idx] / model.V  # concentration = amount / V

        # Normal log-likelihood
        residuals = obs - pred
        nll = 0.5 * jnp.sum((residuals / sigma) ** 2) + len(obs) * jnp.log(sigma)
        total_nll = total_nll + nll

    return total_nll


def train_node(
    model: HybridPKODE,
    subjects: list[dict[str, jax.Array]],
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train the hybrid NODE model on population data.

    Args:
        model: Initial HybridPKODE model.
        subjects: List of subject data dicts with keys:
            'times' (1D array), 'observations' (1D array), 'y0' (1D array),
            optionally 'obs_cmt' (int, default 1).
        config: Training configuration.

    Returns:
        TrainingResult with trained model and metadata.
    """
    config = config or TrainingConfig()
    start_time = time.monotonic()

    # Trainable parameters: model + log(sigma)
    log_sigma = jnp.log(jnp.array(config.sigma_init))

    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )

    # Bundle model + log_sigma into a single pytree for Optax
    # Use a list so Equinox can filter/update it as one unit
    params = (model, log_sigma)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    @eqx.filter_jit
    def step(
        params: tuple[HybridPKODE, jax.Array],
        opt_state: optax.OptState,
    ) -> tuple[tuple[HybridPKODE, jax.Array], optax.OptState, jax.Array]:
        """One optimization step."""

        def loss_fn(p: tuple[HybridPKODE, jax.Array]) -> jax.Array:
            m, ls = p
            return _population_nll(m, ls, subjects)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(params, eqx.is_array),
        )
        new_params = eqx.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Training loop with early stopping
    loss_history: list[float] = []
    best_loss = float("inf")
    patience_counter = 0
    converged = False

    for _epoch in range(config.epochs):
        params, opt_state, loss_val = step(params, opt_state)
        loss_float = float(loss_val)
        loss_history.append(loss_float)

        # Early stopping
        if loss_float < best_loss - config.early_stop_min_delta:
            best_loss = loss_float
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            converged = True
            break

        # NaN detection
        if not jnp.isfinite(loss_val):
            break

    # If we ran all epochs without triggering patience, that's also convergence
    if not converged and patience_counter < config.early_stop_patience and len(loss_history) > 1:
        converged = loss_history[-1] < loss_history[0]

    wall_time = time.monotonic() - start_time
    final_model, final_log_sigma = params

    return TrainingResult(
        trained_model=final_model,
        trained_sigma=float(jnp.exp(final_log_sigma)),
        final_loss=loss_history[-1] if loss_history else float("inf"),
        n_epochs=len(loss_history),
        converged=converged,
        loss_history=loss_history,
        wall_time_seconds=wall_time,
    )
