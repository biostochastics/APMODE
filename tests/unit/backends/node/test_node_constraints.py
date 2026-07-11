# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NODE constraint template enforcement."""

from __future__ import annotations

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
import pytest

from apmode.backends.node_constraints import (
    CONSTRAINT_REGISTRY,
    ConstraintTemplate,
    build_constraint_layer,
)


class TestConstraintRegistry:
    """All 5 templates must be registered."""

    def test_all_templates_registered(self) -> None:
        expected = {
            "monotone_increasing",
            "monotone_decreasing",
            "bounded_positive",
            "saturable",
            "unconstrained_smooth",
        }
        assert set(CONSTRAINT_REGISTRY.keys()) == expected

    @pytest.mark.parametrize("template_name", list(CONSTRAINT_REGISTRY.keys()))
    def test_each_template_returns_constraint(self, template_name: str) -> None:
        ct = build_constraint_layer(template_name, dim=2)
        assert isinstance(ct, ConstraintTemplate)

    def test_unknown_template_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown constraint template"):
            build_constraint_layer("nonexistent", dim=2)


class TestMonotoneIncreasing:
    """monotone_increasing: output must be non-decreasing."""

    def test_output_non_decreasing(self) -> None:
        ct = build_constraint_layer("monotone_increasing", dim=2)
        # Sorted raw inputs should produce sorted outputs
        raws = jnp.array([[-2.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        ys = jax.vmap(ct.apply)(raws)
        diffs = jnp.diff(ys.squeeze())
        assert jnp.all(diffs >= -1e-6), f"Not monotone: min diff = {float(diffs.min())}"

    def test_output_shape(self) -> None:
        ct = build_constraint_layer("monotone_increasing", dim=3)
        raw = jnp.array([1.0, 2.0, 3.0])
        y = ct.apply(raw)
        assert y.shape == (1,)


class TestMonotoneDecreasing:
    """monotone_decreasing: output must be non-increasing."""

    def test_output_non_increasing(self) -> None:
        ct = build_constraint_layer("monotone_decreasing", dim=2)
        raws = jnp.array([[-2.0, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        ys = jax.vmap(ct.apply)(raws)
        diffs = jnp.diff(ys.squeeze())
        assert jnp.all(diffs <= 1e-6), f"Not monotone decreasing: max diff = {float(diffs.max())}"


class TestBoundedPositive:
    """bounded_positive: output must be > 0."""

    def test_output_positive(self) -> None:
        ct = build_constraint_layer("bounded_positive", dim=3)
        raws = jnp.array([[-5.0, -5.0, -5.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
        ys = jax.vmap(ct.apply)(raws)
        assert jnp.all(ys > 0), f"Not positive: min = {float(ys.min())}"

    def test_output_shape(self) -> None:
        ct = build_constraint_layer("bounded_positive", dim=4)
        raw = jnp.array([1.0, 2.0, 3.0, 4.0])
        y = ct.apply(raw)
        assert y.shape == (1,)


class TestSaturable:
    """saturable: output > 0, approaches plateau for large input."""

    def test_output_positive(self) -> None:
        ct = build_constraint_layer("saturable", dim=2)
        raws = jnp.array([[-10.0, -10.0], [0.0, 0.0], [10.0, 10.0], [100.0, 100.0]])
        ys = jax.vmap(ct.apply)(raws)
        assert jnp.all(ys > 0)

    def test_output_bounded(self) -> None:
        ct = build_constraint_layer("saturable", dim=2)
        raws = jnp.array([[-10.0, -10.0], [0.0, 0.0], [10.0, 10.0], [100.0, 100.0]])
        ys = jax.vmap(ct.apply)(raws)
        # Sigmoid output is bounded by scale (default 1.0)
        assert jnp.all(ys <= 1.0 + 1e-6)

    def test_saturates_at_large_input(self) -> None:
        ct = build_constraint_layer("saturable", dim=2)
        large = jnp.array([[50.0, 50.0], [100.0, 100.0], [200.0, 200.0]])
        ys = jax.vmap(ct.apply)(large)
        # Should all be very close together (near the plateau)
        assert jnp.std(ys) < 0.01


class TestUnconstrainedSmooth:
    """unconstrained_smooth: output in (-1, 1) via tanh."""

    def test_output_bounded_by_tanh(self) -> None:
        ct = build_constraint_layer("unconstrained_smooth", dim=4)
        raws = jnp.array([[-100.0] * 4, [0.0] * 4, [100.0] * 4])
        ys = jax.vmap(ct.apply)(raws)
        assert jnp.all(jnp.abs(ys) <= 1.0 + 1e-6)

    def test_output_shape(self) -> None:
        ct = build_constraint_layer("unconstrained_smooth", dim=5)
        raw = jnp.ones(5)
        y = ct.apply(raw)
        assert y.shape == (1,)


class TestDimLimits:
    """Constraint templates enforce max dimension limits."""

    def test_monotone_increasing_max_dim_4(self) -> None:
        build_constraint_layer("monotone_increasing", dim=4)  # OK
        with pytest.raises(ValueError, match=r"dim.*exceeds.*max"):
            build_constraint_layer("monotone_increasing", dim=5)

    def test_monotone_decreasing_max_dim_4(self) -> None:
        build_constraint_layer("monotone_decreasing", dim=4)  # OK
        with pytest.raises(ValueError, match=r"dim.*exceeds.*max"):
            build_constraint_layer("monotone_decreasing", dim=5)

    def test_saturable_max_dim_4(self) -> None:
        build_constraint_layer("saturable", dim=4)  # OK
        with pytest.raises(ValueError, match=r"dim.*exceeds.*max"):
            build_constraint_layer("saturable", dim=5)

    def test_bounded_positive_max_dim_6(self) -> None:
        build_constraint_layer("bounded_positive", dim=6)  # OK
        with pytest.raises(ValueError, match=r"dim.*exceeds.*max"):
            build_constraint_layer("bounded_positive", dim=7)

    def test_unconstrained_max_dim_8(self) -> None:
        build_constraint_layer("unconstrained_smooth", dim=8)  # OK
        with pytest.raises(ValueError, match=r"dim.*exceeds.*max"):
            build_constraint_layer("unconstrained_smooth", dim=9)

    @pytest.mark.parametrize("template_name", list(CONSTRAINT_REGISTRY.keys()))
    def test_dim_1_always_valid(self, template_name: str) -> None:
        ct = build_constraint_layer(template_name, dim=1)
        raw = jnp.array([1.0])
        y = ct.apply(raw)
        assert y.shape == (1,)


class TestDifferentiability:
    """All templates must be JAX-differentiable."""

    @pytest.mark.parametrize("template_name", list(CONSTRAINT_REGISTRY.keys()))
    def test_gradient_exists(self, template_name: str) -> None:
        ct = build_constraint_layer(template_name, dim=2)
        raw = jnp.array([1.0, 0.5])

        def loss(r: jax.Array) -> jax.Array:
            return jnp.sum(ct.apply(r) ** 2)

        grad = jax.grad(loss)(raw)
        assert jnp.all(jnp.isfinite(grad))
