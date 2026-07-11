# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE training loop for population PK fitting (PRD SS4.2.4).

Fits the hybrid ODE to population PK data by optimizing MLP weights +
mechanistic params. Uses Optax for optimization with early stopping.

Training objective: population negative log-likelihood (pooled).
  theta = MLP weights + mechanistic params (ka, V) + sigma

Two training entry points share the same eager multidose solver:

* :func:`train_node` — pooled population fit (no per-subject random effects).
* :func:`train_node_vi` — mixed-effects fit via native reparameterized
  variational inference. Random effects act multiplicatively on the NODE
  input-layer weights (Bräm et al. 2024, doi:10.1007/s10928-023-09886-4:
  ``W_i = W_pop * exp(eta_i)``); the ELBO uses the path-derivative /
  reparameterization estimator with an analytic diagonal-Gaussian KL
  (Janssen et al. 2024, doi:10.1007/s10928-024-09931-w).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from apmode.backends.node_ode import HybridPKODE

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
    # Mixed-effects (VI) outputs. Populated only by ``train_node_vi``; the
    # pooled ``train_node`` path leaves them at the no-IIV defaults.
    random_effects: bool = False
    omega: list[float] | None = None
    """Population RE scale — sqrt of the diagonal RE variance (per hidden dim)."""
    subject_re_means: dict[str, list[float]] | None = None
    """Per-subject posterior mean eta (variational ``mu_i``), keyed by subject id."""
    eta_shrinkage: list[float] | None = None
    """Per-dim eta shrinkage ``1 - mean_i(s_ij^2) / omega_j^2``."""


# Event sort priority within one timestamp (lower = earlier). Mirrors
# apmode.data.dosing._EVID_SORT_PRIORITY so the eager solver applies events
# in the same deterministic order as the on-disk event table:
#   reset -> reset+dose -> dose/infusion-start -> infusion-stop -> observation.
_DOSE_EVID_PRIORITY: dict[int, int] = {3: 0, 4: 1, 1: 2, 9: 3}
_OBS_PRIORITY = 5

# --- Variational-inference (mixed-effects) constants ------------------------
# Positive variational scales are parameterized as softplus(raw) + floor so the
# optimizer sees an unconstrained real; the floor keeps s, omega strictly > 0.
_SCALE_FLOOR = 1e-4
# Inverse-softplus initial values so the initial scales land at the targets
# below: s_i ~ 0.07 (weak per-subject spread), omega ~ 0.10 (weak population RE).
_S_INIT_TARGET = 0.07
_OMEGA_INIT_TARGET = 0.10
# Clamp the reparameterized eta before it enters exp() on the NODE weights, so
# an early large draw cannot blow up W_i = W_pop * exp(eta). exp(+/-4) ~= [0.018, 55].
_ETA_CLAMP = 4.0

# One VI parameter pytree: (model, log_sigma, mu, raw_s, raw_omega).
_VIParams = tuple[HybridPKODE, jax.Array, jax.Array, jax.Array, jax.Array]


def _solve_multidose_eager(
    model: HybridPKODE,
    y0: jax.Array,
    obs_times: jax.Array,
    dose_events: list[tuple[float, float, int, int, float, int]],
    observation_times: list[float] | None = None,
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
    concrete_obs_times = observation_times
    if concrete_obs_times is None:
        # Direct/eager callers can safely materialize the array.  Training
        # supplies a pre-materialized Python list so autodiff never attempts
        # float() on a traced JAX value.
        concrete_obs_times = [float(obs_times[i]) for i in range(n_obs)]
    if len(concrete_obs_times) != n_obs:
        raise ValueError("observation_times length must match obs_times")
    for i, obs_time in enumerate(concrete_obs_times):
        timeline.append((obs_time, _OBS_PRIORITY, "obs", i))

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


def predict_subject_conc(model: HybridPKODE, subject: SubjectRecord) -> jax.Array:
    """Structural predicted concentration at a subject's observation times.

    Single source of truth for the PRED trajectory: both the population
    likelihood (:func:`_population_nll`) and the posterior-predictive
    simulation path in ``node_runner`` call this, so the simulated PRED is
    byte-identical to the PRED the fit objective minimised against (no drift
    between the loss and the diagnostics).

    Subjects may use either:
    - Legacy single-dose: dose in ``y0[0]``, no ``dose_events`` key.
    - Multi-dose / infusion: ``dose_events`` as a Python list of
      ``(time, amt, cmt, evid, inf_rate, inf_id)`` tuples (eager, non-JIT).

    Returns the predicted concentration vector, shape ``(n_obs,)``.
    """
    times = subject["times"]
    y0 = subject["y0"]
    _obs_cmt = subject.get("obs_cmt", jnp.array(1))
    cmt_idx = int(_obs_cmt)

    # Multi-dose path (eager, non-JIT): dose_events is a Python list.
    dose_events = subject.get("dose_events")
    if dose_events is not None and len(dose_events) > 0:
        sol = _solve_multidose_eager(
            model,
            y0,
            times,
            dose_events,
            observation_times=subject.get("observation_times"),
        )
    else:
        # Legacy single-dose path: dose is in y0[0], JIT-compatible.
        sol = model.solve(y0, times)

    # Use the appropriate volume for the observed compartment.
    v_scale = model.V if cmt_idx <= 1 else model.V2
    return sol[:, cmt_idx] / v_scale


def _subject_nll(
    model: HybridPKODE,
    sigma: jax.Array,
    subject: SubjectRecord,
) -> jax.Array:
    """Per-subject normal negative log-likelihood at a fixed ``sigma``.

    Solves the (possibly RE-perturbed) ODE at the subject's observation times
    via :func:`predict_subject_conc` and returns the full normal NLL (including
    the ``log sigma`` and ``2*pi`` constants, for cross-backend comparability).
    Shared by :func:`_population_nll` and the VI ELBO so the pooled and
    mixed-effects likelihoods are computed by one code path.
    """
    obs = subject["observations"]
    pred = predict_subject_conc(model, subject)
    residuals = obs - pred
    n = len(obs)
    return (
        0.5 * jnp.sum((residuals / sigma) ** 2)
        + n * jnp.log(sigma)
        + 0.5 * n * jnp.log(2 * jnp.pi)
    )


def _population_nll(
    model: HybridPKODE,
    log_sigma: jax.Array,
    subjects: Sequence[SubjectRecord],
) -> jax.Array:
    """Population negative log-likelihood (normal residual model).

    For each subject: solve ODE at observation times, compute
    -log N(y_obs | y_pred, sigma^2).

    The structural PRED is computed by :func:`predict_subject_conc` so the
    likelihood and the posterior-predictive diagnostics share one code path.
    """
    sigma = jnp.exp(log_sigma)
    total_nll = jnp.array(0.0)

    for subj in subjects:
        total_nll = total_nll + _subject_nll(model, sigma, subj)

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

    def _step(
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

    # The event-driven multi-dose solver intentionally uses concrete Python
    # control flow and is not JIT-compatible.  Single-dose subjects retain the
    # compiled fast path; multi-dose/infusion fits use the differentiable eager
    # step instead of tracing float(obs_time) and raising ConcretizationTypeError.
    has_event_driven_subject = any(bool(s.get("dose_events")) for s in subjects)
    step = _step if has_event_driven_subject else eqx.filter_jit(_step)

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


def train_node_vi(
    model: HybridPKODE,
    subjects: Sequence[SubjectRecord],
    config: TrainingConfig | None = None,
    *,
    init_result: TrainingResult | None = None,
    n_samples: int = 1,
    re_dim: int | None = None,
    seed: int = 0,
) -> TrainingResult:
    """Fit a mixed-effects NODE via native reparameterized variational inference.

    Random effects act multiplicatively on the NODE input-layer weights
    (Bräm et al. 2024, doi:10.1007/s10928-023-09886-4: ``W_i = W_pop*exp(eta_i)``,
    one ``eta`` per hidden unit). The variational family is a per-subject diagonal
    Gaussian ``q_i(eta_i) = N(mu_i, diag(s_i^2))`` under the population prior
    ``eta_i ~ N(0, diag(omega^2))``. The loss is the negative ELBO summed over
    subjects (consistent with :func:`_population_nll`, which *sums* the
    likelihood — averaging the NLL while adding a raw KL would overweight the KL):

        L = sum_i [ (1/K) sum_k NLL(y_i | theta, eta_ik) + KL(q_i || N(0, omega^2)) ]
        eta_ik = mu_i + s_i * eps_ik,   eps_ik ~ N(0, I)     (reparameterization)

    with the analytic diagonal-Gaussian KL (lower variance than a Monte-Carlo
    ``log q - log p`` estimator; Janssen et al. 2024, doi:10.1007/s10928-024-09931-w
    call the reparameterized/path-derivative ELBO simple to implement):

        KL_i = 0.5 * sum_j [ log(omega_j^2/s_ij^2) + (s_ij^2 + mu_ij^2)/omega_j^2 - 1 ]

    Positive scales use ``softplus(raw) + eps`` and the sampled ``eta`` is clamped
    before ``exp`` to prevent early weight blow-up. The epsilon draws are threaded
    from a JAX PRNG key split per epoch, so training is deterministic given
    ``seed``.

    Args:
        model: Initial (or warm-start) HybridPKODE model.
        subjects: Per-subject records (see :func:`train_node`).
        config: Training configuration (shared with :func:`train_node`).
        init_result: Optional two-stage (Bräm) warm start. When given, the fit
            starts from ``init_result.trained_model`` and its sigma; otherwise a
            pooled no-IIV :func:`train_node` fit is run internally for the warm
            start.
        n_samples: Monte-Carlo samples ``K`` per subject per step (default 1).
        re_dim: RE vector length; defaults to and must equal the NODE hidden dim.
        seed: PRNG seed for the reparameterization draws.

    Returns:
        TrainingResult with ``random_effects=True`` plus ``omega``,
        ``subject_re_means``, and ``eta_shrinkage`` populated.
    """
    config = config or TrainingConfig()
    start_time = time.monotonic()

    hidden_dim = model.node.hidden_dim
    if re_dim is None:
        re_dim = hidden_dim
    if re_dim != hidden_dim:
        msg = f"re_dim={re_dim} must equal the NODE hidden_dim={hidden_dim}"
        raise ValueError(msg)

    # Two-stage warm start (Bräm): pooled no-IIV fit unless one was supplied.
    if init_result is None:
        init_result = train_node(model, subjects, config)
    warm_model = init_result.trained_model
    log_sigma = jnp.log(jnp.array(init_result.trained_sigma))

    n_subj = len(subjects)
    # Variational parameters. mu init 0; raw_s / raw_omega inverse-softplus so the
    # initial scales land at the small targets above.
    mu = jnp.zeros((n_subj, re_dim))
    raw_s = jnp.full((n_subj, re_dim), float(jnp.log(jnp.expm1(_S_INIT_TARGET))))
    raw_omega = jnp.full((re_dim,), float(jnp.log(jnp.expm1(_OMEGA_INIT_TARGET))))

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adam(config.learning_rate),
    )
    params: _VIParams = (warm_model, log_sigma, mu, raw_s, raw_omega)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    def _elbo(p: _VIParams, eps: jax.Array) -> jax.Array:
        m, ls, mu_v, raw_s_v, raw_omega_v = p
        sigma = jnp.exp(ls)
        s = jax.nn.softplus(raw_s_v) + _SCALE_FLOOR
        omega = jax.nn.softplus(raw_omega_v) + _SCALE_FLOOR
        omega_sq = omega**2

        total = jnp.array(0.0)
        for i, subj in enumerate(subjects):
            mu_i = mu_v[i]
            s_i = s[i]
            nll_i = jnp.array(0.0)
            for k in range(n_samples):
                eta = jnp.clip(mu_i + s_i * eps[i, k], -_ETA_CLAMP, _ETA_CLAMP)
                model_i = m.apply_subject_re(eta)
                nll_i = nll_i + _subject_nll(model_i, sigma, subj)
            nll_i = nll_i / n_samples

            s_i_sq = s_i**2
            kl_i = 0.5 * jnp.sum(jnp.log(omega_sq / s_i_sq) + (s_i_sq + mu_i**2) / omega_sq - 1.0)
            total = total + nll_i + kl_i
        return total

    def _step(
        p: _VIParams,
        opt_state: optax.OptState,
        eps: jax.Array,
    ) -> tuple[_VIParams, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(_elbo)(p, eps)
        updates, new_opt_state = optimizer.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(p, eqx.is_array),
        )
        new_params: _VIParams = eqx.apply_updates(p, updates)
        return new_params, new_opt_state, loss

    # The eager multidose solver is not JIT-compatible; single-dose subjects
    # keep the compiled fast path (mirrors ``train_node``).
    has_event_driven_subject = any(bool(s.get("dose_events")) for s in subjects)
    step = _step if has_event_driven_subject else eqx.filter_jit(_step)

    key = jax.random.PRNGKey(seed)
    loss_history: list[float] = []
    best_loss = float("inf")
    patience_counter = 0
    converged = False
    minimization_status = "max_evaluations"

    for _epoch in range(config.epochs):
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, (n_subj, n_samples, re_dim))
        params, opt_state, loss_val = step(params, opt_state, eps)
        loss_float = float(loss_val)
        loss_history.append(loss_float)

        if not jnp.isfinite(loss_val):
            minimization_status = "nan_detected"
            break

        if loss_float < best_loss - config.early_stop_min_delta:
            best_loss = loss_float
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            converged = len(loss_history) > 1 and best_loss < loss_history[0] * 0.99
            minimization_status = "plateau" if not converged else "successful"
            break
    else:
        if len(loss_history) > 1 and best_loss < loss_history[0] * 0.99:
            converged = True
            minimization_status = "successful"

    wall_time = time.monotonic() - start_time
    final_model, final_log_sigma, final_mu, final_raw_s, final_raw_omega = params

    final_s = jax.nn.softplus(final_raw_s) + _SCALE_FLOOR
    final_omega = jax.nn.softplus(final_raw_omega) + _SCALE_FLOOR
    final_omega_sq = final_omega**2
    # Per-dim shrinkage: 1 - mean_i(s_ij^2) / omega_j^2.
    shrinkage = 1.0 - jnp.mean(final_s**2, axis=0) / final_omega_sq

    subject_re_means = {
        subj["subject_id"]: [float(v) for v in final_mu[i]] for i, subj in enumerate(subjects)
    }

    return TrainingResult(
        trained_model=final_model,
        trained_sigma=float(jnp.exp(final_log_sigma)),
        final_loss=loss_history[-1] if loss_history else float("inf"),
        n_epochs=len(loss_history),
        converged=converged,
        loss_history=loss_history,
        wall_time_seconds=wall_time,
        minimization_status=minimization_status,
        random_effects=True,
        omega=[float(v) for v in final_omega],
        subject_re_means=subject_re_means,
        eta_shrinkage=[float(v) for v in shrinkage],
    )
