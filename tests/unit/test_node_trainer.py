# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NODE training loop."""

from __future__ import annotations

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
import pytest

from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.backends.node_trainer import TrainingConfig, TrainingResult, train_node


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
