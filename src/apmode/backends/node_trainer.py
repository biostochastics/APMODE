# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE training loop for population PK fitting (PRD SS4.2.4).

Fits the hybrid ODE to population PK data by optimizing MLP weights +
mechanistic params. Uses Optax for optimization with early stopping.

Training objective: population negative log-likelihood (pooled).
  theta = MLP weights + mechanistic params (ka, V) + sigma

Current limitation: performs pooled population fitting. Per-subject random
effects via Laplace approximation are not implemented; the
``model.apply_random_effects()`` API is available for that integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from apmode.backends.node_ode import HybridPKODE  # noqa: TC001 — used at runtime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apmode.backends.node_runner import SubjectRecord


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
    minimization_status: str = "max_evaluations"


# Event sort priority within one timestamp (lower = earlier). Mirrors
# apmode.data.dosing._EVID_SORT_PRIORITY so the eager solver applies events
# in the same deterministic order as the on-disk event table:
#   reset -> reset+dose -> dose/infusion-start -> infusion-stop -> observation.
_DOSE_EVID_PRIORITY: dict[int, int] = {3: 0, 4: 1, 1: 2, 9: 3}
_OBS_PRIORITY = 5


def _solve_multidose_eager(
    model: HybridPKODE,
    y0: jax.Array,
    obs_times: jax.Array,
    dose_events: list[tuple[float, float, int, int, float, int]],
) -> jax.Array:
    """Piecewise ODE integration with merged dose+observation timeline.

    Merges dose events and observation times into a single chronological
    timeline. Integrates forward segment-by-segment, applying state jumps at
    bolus events and threading a per-compartment infusion-rate vector through
    each segment. Records predicted state at observation times.

    Each dose event is a 6-tuple ``(time, amt, cmt, evid, inf_rate, inf_id)``:
      * bolus:           ``evid in {1, 4}``, ``amt > 0``, ``inf_rate == 0``.
      * infusion start:  ``evid in {1, 4}``, ``inf_rate > 0`` (no bolus jump).
      * infusion stop:   ``evid == 9``, ``inf_rate < 0`` (synthetic, from
        :func:`apmode.data.dosing.build_event_table`).
      * reset:           ``evid in {3, 4}`` zeros the state.
    ``inf_id`` is a shared identity assigned by ``build_event_table``: an
    infusion start and its paired stop carry the *same* id; every non-infusion
    row carries ``-1``.

    Overlapping infusions sum. Each active infusion is tracked by its ``inf_id``;
    the per-compartment rate vector passed as ``args`` to ``model.solve`` is the
    sum over active infusions, and when nothing is active ``args=None`` is passed
    so the bolus-only path is byte-identical to the pre-infusion behaviour. A
    stop removes the active infusion with its *matching id* — pairing on identity
    (not on ``(cmt, rate)``) is what keeps repeat-same-rate-across-reset regimens
    correct: a reset (EVID 3/4) terminates all ongoing infusions (clears the
    active map), so a later stop whose start the reset already removed simply
    finds no entry and no-ops, and can never clip an unrelated identically-rated
    infusion started after the reset.

    This function uses concrete Python values for control flow (not traced),
    so it works with eager JAX execution but NOT inside JIT.

    Returns predicted states at obs_times (shape: [n_obs, n_states]).
    """
    n_states = int(y0.shape[0])
    n_obs = int(obs_times.shape[0])

    if not dose_events and n_obs > 0:
        return model.solve(y0, obs_times)

    # Build merged chronological timeline of (time, priority, kind, index):
    #   kind is "dose" or "obs"; index points into dose_events or obs_times.
    timeline: list[tuple[float, int, str, int]] = []
    for i, (t, _amt, _cmt, evid, _rate, _id) in enumerate(dose_events):
        timeline.append((t, _DOSE_EVID_PRIORITY.get(int(evid), _OBS_PRIORITY), "dose", i))
    for i in range(n_obs):
        timeline.append((float(obs_times[i]), _OBS_PRIORITY, "obs", i))

    # Deterministic order: by time, then event priority (dose-start before
    # dose-stop before observation). Stable sort keeps input order as tiebreak.
    timeline.sort(key=lambda x: (x[0], x[1]))

    state = y0
    t_current = 0.0
    # Active infusions keyed by their shared ``inf_id`` -> (cmt_idx, rate)
    # (untraced Python floats so control flow stays outside JIT). The solver
    # ``args`` is their per-compartment sum; empty => ``args=None``.
    active_infusions: dict[int, tuple[int, float]] = {}
    predictions = [jnp.zeros(n_states)] * n_obs  # placeholder

    for t_event, _priority, event_type, idx in timeline:
        # Integrate to this event time if needed, threading the active rate.
        if t_event > t_current + 1e-12:
            rate_args: jax.Array | None = None
            if active_infusions:
                rate_vec = [0.0] * n_states
                for ci, r in active_infusions.values():
                    rate_vec[ci] += r
                if any(r != 0.0 for r in rate_vec):
                    rate_args = jnp.asarray(rate_vec, dtype=y0.dtype)
            sol = model.solve(state, jnp.array([t_event]), t0=t_current, args=rate_args)
            state = sol[0]
            t_current = t_event

        if event_type == "dose":
            _t_dose, amt, cmt, evid, inf_rate, inf_id = dose_events[idx]
            cmt_idx = max(0, min(cmt - 1, n_states - 1))
            # Apply reset (EVID=3 or 4): zero the state AND terminate ongoing
            # infusions. A later stop whose start id is gone then no-ops, so the
            # summed rate never goes negative and no live infusion is clipped.
            if evid in (3, 4):
                state = jnp.zeros(n_states)
                active_infusions.clear()
            # Apply bolus (EVID=1 or 4) — only for non-infusion rows; an
            # infusion delivers its AMT gradually via the rate term instead.
            if evid in (1, 4) and amt > 0 and inf_rate == 0.0:
                state = state.at[cmt_idx].add(amt)
            # Start (inf_rate > 0) an infusion under its id, or stop (inf_rate < 0)
            # by removing the active infusion with the matching id. An orphaned
            # stop (id already reset-cleared) simply finds no entry -> no-op.
            if inf_rate > 0.0:
                active_infusions[inf_id] = (cmt_idx, inf_rate)
            elif inf_rate < 0.0:
                active_infusions.pop(inf_id, None)
        else:
            # Record state at observation time
            predictions[idx] = state

    return jnp.stack(predictions)


def _population_nll(
    model: HybridPKODE,
    log_sigma: jax.Array,
    subjects: Sequence[SubjectRecord],
) -> jax.Array:
    """Population negative log-likelihood (normal residual model).

    For each subject: solve ODE at observation times, compute
    -log N(y_obs | y_pred, sigma^2).

    Subjects may use either:
    - Legacy single-dose: dose in y0[0], no dose_events key
    - Multi-dose: dose_events as Python list of (time, amt, cmt, evid) tuples
    """
    sigma = jnp.exp(log_sigma)
    total_nll = jnp.array(0.0)

    for subj in subjects:
        times = subj["times"]
        obs = subj["observations"]
        y0 = subj["y0"]
        _obs_cmt = subj.get("obs_cmt", jnp.array(1))
        cmt_idx = int(_obs_cmt)

        # Multi-dose path (eager, non-JIT): dose_events is a Python list
        dose_events = subj.get("dose_events")
        if dose_events is not None and len(dose_events) > 0:
            sol = _solve_multidose_eager(
                model,
                y0,
                times,
                dose_events,
            )
        else:
            # Legacy single-dose path: dose is in y0[0], JIT-compatible
            sol = model.solve(y0, times)

        # Use appropriate volume for compartment
        v_scale = model.V if cmt_idx <= 1 else model.V2
        pred = sol[:, cmt_idx] / v_scale

        # Normal negative log-likelihood (full, for cross-backend comparability)
        residuals = obs - pred
        n = len(obs)
        nll = (
            0.5 * jnp.sum((residuals / sigma) ** 2)
            + n * jnp.log(sigma)
            + 0.5 * n * jnp.log(2 * jnp.pi)
        )
        total_nll = total_nll + nll

    return total_nll


def train_node(
    model: HybridPKODE,
    subjects: Sequence[SubjectRecord],
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
    minimization_status = "max_evaluations"

    for _epoch in range(config.epochs):
        params, opt_state, loss_val = step(params, opt_state)
        loss_float = float(loss_val)
        loss_history.append(loss_float)

        # NaN detection — abort immediately
        if not jnp.isfinite(loss_val):
            minimization_status = "nan_detected"
            break

        # Early stopping check
        if loss_float < best_loss - config.early_stop_min_delta:
            best_loss = loss_float
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            # Loss plateaued — converged only if we improved significantly
            converged = len(loss_history) > 1 and best_loss < loss_history[0] * 0.99
            minimization_status = "plateau" if not converged else "successful"
            break
    else:
        # Completed all epochs without early stopping
        if len(loss_history) > 1 and best_loss < loss_history[0] * 0.99:
            converged = True
            minimization_status = "successful"

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
        minimization_status=minimization_status,
    )
