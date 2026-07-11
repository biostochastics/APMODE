# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NODE training loop."""

from __future__ import annotations

import math

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
import pytest

from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.backends.node_trainer import (
    TrainingConfig,
    TrainingResult,
    _solve_multidose_eager,
    train_node,
)


def _make_synthetic_subjects(
    n_subjects: int = 5,
    n_obs: int = 8,
    seed: int = 42,
) -> list[dict[str, jax.Array]]:
    """Create synthetic PK data: 1-cmt oral, first-order absorption + elimination."""
    key = jax.random.PRNGKey(seed)
    subjects = []
    for _i in range(n_subjects):
        key, subkey = jax.random.split(key)
        times = jnp.linspace(0.5, 24.0, n_obs)
        # Analytical 1-cmt oral: C = (D*ka)/(V*(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
        ka = 1.0 + 0.2 * float(jax.random.normal(subkey))
        ke = 0.1
        V = 30.0
        dose = 100.0
        true_conc = (dose * ka) / (V * (ka - ke)) * (jnp.exp(-ke * times) - jnp.exp(-ka * times))
        true_conc = jnp.maximum(true_conc, 0.01)
        # Add noise
        key, subkey = jax.random.split(key)
        noise = 0.1 * true_conc * jax.random.normal(subkey, shape=times.shape)
        obs = true_conc + noise
        subjects.append(
            {
                "times": times,
                "observations": jnp.maximum(obs, 0.001),
                "y0": jnp.array([dose, 0.0]),
                "obs_cmt": jnp.array(1),
            }
        )
    return subjects


def _make_model(seed: int = 0) -> HybridPKODE:
    return HybridPKODE(
        config=ODEConfig(
            n_cmt=1,
            node_position="elimination",
            constraint_template="bounded_positive",
            node_dim=3,
            mechanistic_params={"ka": 1.0, "V": 30.0},
        ),
        key=jax.random.PRNGKey(seed),
    )


class TestTrainNode:
    """Training loop basics."""

    def test_training_reduces_loss(self) -> None:
        model = _make_model()
        subjects = _make_synthetic_subjects(n_subjects=3, n_obs=6)
        config = TrainingConfig(epochs=30, learning_rate=1e-3, early_stop_patience=50)

        result = train_node(model, subjects, config)

        assert isinstance(result, TrainingResult)
        assert len(result.loss_history) > 1
        # Loss should decrease from start
        assert result.loss_history[-1] < result.loss_history[0]

    def test_returns_trained_model(self) -> None:
        model = _make_model()
        subjects = _make_synthetic_subjects(n_subjects=2, n_obs=6)
        config = TrainingConfig(epochs=10)

        result = train_node(model, subjects, config)

        assert result.trained_model is not None
        assert isinstance(result.trained_model, HybridPKODE)

    def test_convergence_metadata(self) -> None:
        model = _make_model()
        subjects = _make_synthetic_subjects(n_subjects=2, n_obs=6)
        config = TrainingConfig(epochs=10)

        result = train_node(model, subjects, config)

        assert result.wall_time_seconds > 0
        assert result.n_epochs > 0
        assert result.method == "adam"
        assert result.trained_sigma > 0

    def test_sigma_is_positive(self) -> None:
        model = _make_model()
        subjects = _make_synthetic_subjects(n_subjects=2, n_obs=6)
        config = TrainingConfig(epochs=15)

        result = train_node(model, subjects, config)
        assert result.trained_sigma > 0


class TestEarlyStopping:
    """Early stopping behavior."""

    def test_stops_before_max_epochs(self) -> None:
        model = _make_model()
        subjects = _make_synthetic_subjects(n_subjects=3, n_obs=6)
        # Very long max epochs but short patience
        config = TrainingConfig(epochs=500, early_stop_patience=5, learning_rate=1e-2)

        result = train_node(model, subjects, config)

        # Should stop well before 500 epochs due to early stopping
        assert result.n_epochs < 500


def _iv_mechanistic_model(cl: float = 2.0, v: float = 20.0) -> HybridPKODE:
    """A mechanistic 1-cmt IV model (absorption-mode NODE with an empty depot).

    In ``node_position='absorption'`` the elimination is the mechanistic
    ``CL/V`` law; with an empty depot the NODE absorption term is
    ``node(x) * A_depot = 0``, so the central compartment obeys the exact
    1-cmt IV ODE ``dA/dt = R - (CL/V) A``. This gives an analytic ground
    truth for infusion/bolus dosing without any training.
    """
    return HybridPKODE(
        config=ODEConfig(
            n_cmt=1,
            node_position="absorption",
            constraint_template="bounded_positive",
            node_dim=3,
            mechanistic_params={"ka": 1.0, "CL": cl, "V": v},
        ),
        key=jax.random.PRNGKey(0),
    )


def _analytic_infusion(t: float, rate: float, cl: float, v: float, dur: float) -> float:
    """C(t) for a constant-rate IV infusion of duration ``dur`` into 1-cmt."""
    k = cl / v
    css = rate / cl
    if t <= dur:
        return css * (1.0 - math.exp(-k * t))
    c_end = css * (1.0 - math.exp(-k * dur))
    return c_end * math.exp(-k * (t - dur))


class TestInfusionSolve:
    """Constant-rate infusion through the piecewise eager solver."""

    def test_iv_infusion_matches_analytic(self) -> None:
        cl, v, rate, dur = 2.0, 20.0, 10.0, 10.0
        amt = rate * dur
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([1.0, 3.0, 5.0, 10.0, 12.0, 16.0, 24.0])
        # Central compartment is state index 1 -> CMT=2. inf_id links start->stop.
        dose_events = [
            (0.0, amt, 2, 1, rate, 0),  # infusion start
            (dur, 0.0, 2, 9, -rate, 0),  # synthetic infusion stop
        ]

        sol = _solve_multidose_eager(model, y0, obs_times, dose_events)
        conc = sol[:, 1] / v

        expected = jnp.array([_analytic_infusion(float(t), rate, cl, v, dur) for t in obs_times])
        assert jnp.allclose(conc, expected, rtol=1e-2, atol=1e-2)

    def test_overlapping_infusions_sum_rates(self) -> None:
        cl, v, dur = 2.0, 20.0, 10.0
        r1, r2 = 6.0, 4.0
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([2.0, 6.0, 10.0, 14.0, 20.0])
        dose_events = [
            (0.0, r1 * dur, 2, 1, r1, 0),
            (0.0, r2 * dur, 2, 1, r2, 1),
            (dur, 0.0, 2, 9, -r1, 0),
            (dur, 0.0, 2, 9, -r2, 1),
        ]

        sol = _solve_multidose_eager(model, y0, obs_times, dose_events)
        conc = sol[:, 1] / v

        # Overlapping infusions sum: effective rate r1 + r2.
        expected = jnp.array(
            [_analytic_infusion(float(t), r1 + r2, cl, v, dur) for t in obs_times]
        )
        assert jnp.allclose(conc, expected, rtol=1e-2, atol=1e-2)

    def test_evid9_stop_ends_infusion(self) -> None:
        cl, v, rate, dur = 2.0, 20.0, 10.0, 5.0
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([5.0, 6.0, 8.0])

        with_stop = [
            (0.0, rate * dur, 2, 1, rate, 0),
            (dur, 0.0, 2, 9, -rate, 0),
        ]
        without_stop = [
            (0.0, rate * dur, 2, 1, rate, 0),
        ]

        sol_stop = _solve_multidose_eager(model, y0, obs_times, with_stop)[:, 1] / v
        sol_run = _solve_multidose_eager(model, y0, obs_times, without_stop)[:, 1] / v

        # After the EVID=9 stop the concentration must decay, not keep rising.
        assert float(sol_stop[2]) < float(sol_stop[0])
        # And it must be strictly below the never-stopped infusion.
        assert float(sol_stop[2]) < float(sol_run[2])
        # The stop matches analytic post-infusion decay.
        expected_decay = _analytic_infusion(8.0, rate, cl, v, dur)
        assert float(sol_stop[2]) == pytest.approx(expected_decay, rel=1e-2)

    def test_bolus_multidose_matches_superposition(self) -> None:
        """Bolus-only multidose is unchanged and matches IV superposition."""
        cl, v, dose, tau = 2.0, 20.0, 50.0, 12.0
        k = cl / v
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([1.0, 6.0, 12.0, 13.0, 18.0, 24.0])
        dose_events = [
            (0.0, dose, 2, 1, 0.0, -1),  # IV bolus (RATE=0)
            (tau, dose, 2, 1, 0.0, -1),
        ]

        sol = _solve_multidose_eager(model, y0, obs_times, dose_events)
        conc = sol[:, 1] / v

        def superpos(t: float) -> float:
            c = (dose / v) * math.exp(-k * t)
            if t >= tau:
                c += (dose / v) * math.exp(-k * (t - tau))
            return c

        expected = jnp.array([superpos(float(t)) for t in obs_times])
        assert jnp.allclose(conc, expected, rtol=1e-2, atol=1e-2)

    def test_reset_terminates_ongoing_infusion(self) -> None:
        """A reset (EVID=3) mid-infusion clears it; the orphaned stop no-ops.

        Regression for the deferred C1 minor: a reset must terminate an ongoing
        infusion (zero the state AND drop the active infusion). The paired
        EVID=9 stop then arrives after the reset, finds its id already cleared,
        and must no-op — otherwise it would drive the summed rate negative and
        pull concentration below zero.
        """
        cl, v, rate = 2.0, 20.0, 10.0
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([2.0, 7.0, 12.0])
        dose_events = [
            (0.0, rate * 10.0, 2, 1, rate, 0),  # infusion start (into central)
            (5.0, 0.0, 2, 3, 0.0, -1),  # reset — terminates the infusion
            (10.0, 0.0, 2, 9, -rate, 0),  # orphaned stop (id 0 already cleared)
        ]

        conc = _solve_multidose_eager(model, y0, obs_times, dose_events)[:, 1] / v

        # Before the reset the infusion is delivering -> concentration is up.
        assert float(conc[0]) > 0.1
        # After the reset the state is zeroed and no infusion remains -> ~0.
        assert float(conc[1]) == pytest.approx(0.0, abs=1e-4)
        # The orphaned stop must not create a negative rate -> still ~0, not < 0.
        assert float(conc[2]) == pytest.approx(0.0, abs=1e-4)
        assert float(conc[2]) >= -1e-6

    def test_reset_then_restart_same_rate_does_not_clip_new_infusion(self) -> None:
        """A reset-orphaned stop must not cancel a same-rate post-reset infusion.

        Regression for the identity-linkage bug: infusion A (rate 10, id 0)
        starts at t=0 with a phantom stop at t=10; a reset at t=5 terminates A;
        infusion B (also rate 10, id 1) starts at t=6 with its own stop at t=16.
        A naive ``(cmt, rate)`` stop match would let A's orphaned t=10 stop pop B
        (same compartment and rate), ending B six hours early. With id pairing
        A's stop (id 0, already cleared) no-ops and B (id 1) runs to t=16.
        """
        cl, v, rate, dur = 2.0, 20.0, 10.0, 10.0
        start_b = 6.0
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([8.0, 12.0, 16.0, 20.0])  # all after the reset
        dose_events = [
            (0.0, rate * dur, 2, 1, rate, 0),  # A start id 0 (phantom stop at t=10)
            (5.0, 0.0, 2, 3, 0.0, -1),  # reset — terminates A
            (start_b, rate * dur, 2, 1, rate, 1),  # B start id 1, same rate
            (10.0, 0.0, 2, 9, -rate, 0),  # A's orphaned stop id 0 (must no-op)
            (start_b + dur, 0.0, 2, 9, -rate, 1),  # B's real stop id 1 at t=16
        ]

        conc = _solve_multidose_eager(model, y0, obs_times, dose_events)[:, 1] / v

        # B alone contributes post-reset; A was zeroed at t=5.
        expected = jnp.array(
            [_analytic_infusion(float(t) - start_b, rate, cl, v, dur) for t in obs_times]
        )
        assert jnp.allclose(conc, expected, rtol=1e-2, atol=1e-2)
        # Anti-regression: B is still rising at t=12 (not clipped by A's stop at t=10).
        assert float(conc[1]) > float(conc[0])

    def test_staggered_overlapping_infusions_superpose(self) -> None:
        """Two infusions with distinct start/stop times superpose correctly.

        The linear IV model superposes, so the response to two staggered
        infusions equals the sum of their individual responses. This exercises
        selective stop-removal: the stop of A must drop A's interval while B's
        rate stays active.
        """
        cl, v = 2.0, 20.0
        r_a, r_b, dur = 6.0, 4.0, 10.0
        start_b = 4.0
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([2.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0])
        dose_events = [
            (0.0, r_a * dur, 2, 1, r_a, 0),  # A start id 0
            (start_b, r_b * dur, 2, 1, r_b, 1),  # B start id 1 (staggered)
            (dur, 0.0, 2, 9, -r_a, 0),  # A stop id 0
            (start_b + dur, 0.0, 2, 9, -r_b, 1),  # B stop id 1
        ]

        conc = _solve_multidose_eager(model, y0, obs_times, dose_events)[:, 1] / v

        def superpos(t: float) -> float:
            c = _analytic_infusion(t, r_a, cl, v, dur)
            if t >= start_b:
                c += _analytic_infusion(t - start_b, r_b, cl, v, dur)
            return c

        expected = jnp.array([superpos(float(t)) for t in obs_times])
        assert jnp.allclose(conc, expected, rtol=1e-2, atol=1e-2)

    def test_infusion_never_stops_in_window(self) -> None:
        """An infusion with no EVID=9 stop in the window keeps rising to Css."""
        cl, v, rate = 2.0, 20.0, 10.0
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([1.0, 4.0, 8.0, 16.0, 30.0])
        dose_events = [(0.0, rate * 100.0, 2, 1, rate, 0)]  # start only, no stop

        conc = _solve_multidose_eager(model, y0, obs_times, dose_events)[:, 1] / v

        # Monotonically increasing and approaching steady state Css = rate/CL.
        assert all(float(conc[i]) < float(conc[i + 1]) for i in range(len(conc) - 1))
        css = rate / cl
        expected = jnp.array(
            [_analytic_infusion(float(t), rate, cl, v, dur=1e9) for t in obs_times]
        )
        assert jnp.allclose(conc, expected, rtol=1e-2, atol=1e-2)
        assert float(conc[-1]) == pytest.approx(css, rel=5e-2)

    def test_bolus_and_infusion_at_same_timestamp(self) -> None:
        """A bolus and an infusion start at the same time both apply and sum."""
        cl, v = 2.0, 20.0
        bolus, rate, dur = 40.0, 8.0, 10.0
        k = cl / v
        model = _iv_mechanistic_model(cl=cl, v=v)
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([1.0, 4.0, 8.0, 10.0, 14.0, 20.0])
        dose_events = [
            (0.0, bolus, 2, 1, 0.0, -1),  # IV bolus at t=0
            (0.0, rate * dur, 2, 1, rate, 0),  # infusion start at t=0
            (dur, 0.0, 2, 9, -rate, 0),  # infusion stop
        ]

        conc = _solve_multidose_eager(model, y0, obs_times, dose_events)[:, 1] / v

        def expected_fn(t: float) -> float:
            return (bolus / v) * math.exp(-k * t) + _analytic_infusion(t, rate, cl, v, dur)

        expected = jnp.array([expected_fn(float(t)) for t in obs_times])
        assert jnp.allclose(conc, expected, rtol=1e-2, atol=1e-2)

    def test_bolus_only_no_drift_vs_pre_infusion_algorithm(self) -> None:
        """Bolus-only multidose is bit-identical to the pre-infusion algorithm.

        Reproduces the segment-by-segment loop as it existed before infusion
        support (plain ``model.solve`` with no ``args``) and asserts the new
        code path — which passes ``args=None`` whenever no infusion is active
        — produces byte-identical predictions.
        """
        model = _iv_mechanistic_model()
        y0 = jnp.array([0.0, 0.0])
        obs_times = jnp.array([1.0, 6.0, 12.0, 13.0, 18.0, 24.0])
        dose_events = [
            (0.0, 50.0, 2, 1, 0.0, -1),
            (12.0, 50.0, 2, 1, 0.0, -1),
        ]

        # --- Reference: pre-infusion eager algorithm (bolus doses, no args).
        n_states = 2
        timeline: list[tuple[float, str, int]] = []
        for i, (t, _amt, _cmt, _evid, _rate, _id) in enumerate(dose_events):
            timeline.append((t, "dose", i))
        for i in range(len(obs_times)):
            timeline.append((float(obs_times[i]), "obs", i))
        timeline.sort(key=lambda x: (x[0], 0 if x[1] == "dose" else 1))
        state = y0
        t_current = 0.0
        ref = [jnp.zeros(n_states)] * len(obs_times)
        for t_event, kind, idx in timeline:
            if t_event > t_current + 1e-12:
                state = model.solve(state, jnp.array([t_event]), t0=t_current)[0]
                t_current = t_event
            if kind == "dose":
                _t, amt, cmt, evid, _rate, _id = dose_events[idx]
                if evid in (1, 4) and amt > 0:
                    state = state.at[cmt - 1].add(amt)
            else:
                ref[idx] = state
        reference = jnp.stack(ref)

        new = _solve_multidose_eager(model, y0, obs_times, dose_events)
        assert jnp.array_equal(reference, new)


class TestDeterminism:
    """CPU deterministic mode should produce identical results."""

    def test_same_seed_same_result(self) -> None:
        subjects = _make_synthetic_subjects(n_subjects=2, n_obs=6, seed=42)
        config = TrainingConfig(epochs=10, execution_mode="cpu_deterministic")

        model1 = _make_model(seed=0)
        result1 = train_node(model1, subjects, config)

        model2 = _make_model(seed=0)
        result2 = train_node(model2, subjects, config)

        assert result1.final_loss == pytest.approx(result2.final_loss, rel=1e-5)
        assert len(result1.loss_history) == len(result2.loss_history)
