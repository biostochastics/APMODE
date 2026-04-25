# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for the Bräm-style NODE sub-model."""

from __future__ import annotations

import equinox as eqx  # type: ignore[import-untyped]
import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
import pytest

from apmode.backends.node_model import NODESubModel


class TestNODESubModelInit:
    """Construction and validation."""

    def test_creates_with_valid_params(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=4,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        assert model.input_dim == 2
        assert model.hidden_dim == 4
        assert model.constraint_name == "bounded_positive"

    def test_rejects_dim_exceeding_template_max(self) -> None:
        with pytest.raises(ValueError, match=r"dim.*exceeds"):
            NODESubModel(
                input_dim=2,
                hidden_dim=5,
                constraint_template="monotone_increasing",
                key=jax.random.PRNGKey(0),
            )

    @pytest.mark.parametrize(
        "template",
        [
            "monotone_increasing",
            "monotone_decreasing",
            "bounded_positive",
            "saturable",
            "unconstrained_smooth",
        ],
    )
    def test_creates_with_each_template(self, template: str) -> None:
        model = NODESubModel(
            input_dim=2, hidden_dim=2, constraint_template=template, key=jax.random.PRNGKey(0)
        )
        assert model.constraint_name == template


class TestNODESubModelForward:
    """Forward pass produces valid output."""

    def test_output_shape_scalar(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=4,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        x = jnp.array([1.0, 0.5])
        y = model(x)
        assert y.shape == (1,)

    def test_output_positive_with_bounded_positive(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=3,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(42),
        )
        xs = jnp.array([[0.1, 0.0], [5.0, 1.0], [100.0, 24.0]])
        ys = jax.vmap(model)(xs)
        assert jnp.all(ys > 0)

    def test_output_finite(self) -> None:
        model = NODESubModel(
            input_dim=3,
            hidden_dim=4,
            constraint_template="unconstrained_smooth",
            key=jax.random.PRNGKey(7),
        )
        xs = jnp.array([[0.0, 0.0, 0.0], [1e3, 1e3, 1e3], [-1e3, -1e3, -1e3]])
        ys = jax.vmap(model)(xs)
        assert jnp.all(jnp.isfinite(ys))

    def test_vmap_batch(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=3,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        batch = jnp.ones((10, 2))
        ys = jax.vmap(model)(batch)
        assert ys.shape == (10, 1)

    def test_differentiable(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=4,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        x = jnp.array([1.0, 0.5])

        @eqx.filter_grad
        def grad_fn(m: NODESubModel) -> jax.Array:
            return jnp.sum(m(x) ** 2)

        grads = grad_fn(model)
        flat_grads = jax.tree.leaves(grads)
        assert any(jnp.any(g != 0) for g in flat_grads if hasattr(g, "shape"))


class TestRandomEffects:
    """Random effects on input-layer weights."""

    def test_re_perturbation_changes_output(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=4,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        x = jnp.array([5.0, 1.0])

        y_pop = model(x)
        re = jnp.array([0.1, -0.1, 0.05, -0.05])
        model_re = model.apply_random_effects(re)
        y_re = model_re(x)

        assert not jnp.allclose(y_pop, y_re), "RE should change output"

    def test_zero_re_preserves_output(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=3,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        x = jnp.array([5.0, 1.0])

        y_pop = model(x)
        re = jnp.zeros(3)
        model_re = model.apply_random_effects(re)
        y_re = model_re(x)

        assert jnp.allclose(y_pop, y_re), "Zero RE should preserve output"

    def test_re_returns_new_model(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=3,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        re = jnp.array([0.1, -0.1, 0.05])
        model_re = model.apply_random_effects(re)

        # Original model weights unchanged
        assert not jnp.allclose(model.linear1.weight, model_re.linear1.weight)

    def test_re_is_differentiable(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=3,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        x = jnp.array([5.0, 1.0])

        def loss_fn(re: jax.Array) -> jax.Array:
            m = model.apply_random_effects(re)
            return jnp.sum(m(x) ** 2)

        re = jnp.array([0.1, -0.1, 0.05])
        grad = jax.grad(loss_fn)(re)
        assert jnp.all(jnp.isfinite(grad))
