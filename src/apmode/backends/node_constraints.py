# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE constraint template enforcement (PRD SS4.2.5).

Five enumerated constraint templates that shape NODE sub-model output
to maintain physiological plausibility. Each template is an Equinox
module that transforms raw MLP output into constrained output.

Template dim limits (PRD SS4.2.5 table):
  monotone_increasing:   4
  monotone_decreasing:   4
  bounded_positive:      6
  saturable:             4
  unconstrained_smooth:  8
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

# Max dim per constraint template (matches validator.py _TEMPLATE_MAX_DIM)
TEMPLATE_MAX_DIM: dict[str, int] = {
    "monotone_increasing": 4,
    "monotone_decreasing": 4,
    "bounded_positive": 6,
    "saturable": 4,
    "unconstrained_smooth": 8,
}

# Public registry (for test introspection)
CONSTRAINT_REGISTRY = TEMPLATE_MAX_DIM


class ConstraintTemplate(eqx.Module):
    """Base for constraint templates. Subclasses implement apply()."""

    dim: int

    def apply(self, raw: jax.Array) -> jax.Array:
        """Transform raw MLP output into constrained output."""
        raise NotImplementedError


class MonotoneIncreasing(ConstraintTemplate):
    """Output is non-decreasing: cumulative softplus of raw values."""

    def apply(self, raw: jax.Array) -> jax.Array:
        increments = jnp.log1p(jnp.exp(raw))
        return jnp.sum(increments, keepdims=True)


class MonotoneDecreasing(ConstraintTemplate):
    """Output is non-increasing: negative cumulative softplus."""

    def apply(self, raw: jax.Array) -> jax.Array:
        increments = jnp.log1p(jnp.exp(raw))
        return -jnp.sum(increments, keepdims=True)


class BoundedPositive(ConstraintTemplate):
    """Output is strictly positive: softplus of sum."""

    def apply(self, raw: jax.Array) -> jax.Array:
        s = jnp.sum(raw)
        return jnp.log1p(jnp.exp(s)).reshape(1)


class Saturable(ConstraintTemplate):
    """Output > 0, saturates for large input: sigmoid-scaled.

    Uses a learned scale parameter (JAX array, trainable) to set the
    saturation plateau height.
    """

    scale: jax.Array

    def __init__(self, dim: int, scale: float = 1.0) -> None:
        self.dim = dim
        self.scale = jnp.array(scale)

    def apply(self, raw: jax.Array) -> jax.Array:
        s = jnp.sum(raw)
        return (self.scale * jax.nn.sigmoid(s)).reshape(1)


class UnconstrainedSmooth(ConstraintTemplate):
    """Smooth output, no positivity/monotonicity constraint."""

    def apply(self, raw: jax.Array) -> jax.Array:
        return jnp.tanh(jnp.sum(raw)).reshape(1)


_TEMPLATE_CLASSES: dict[str, type[ConstraintTemplate]] = {
    "monotone_increasing": MonotoneIncreasing,
    "monotone_decreasing": MonotoneDecreasing,
    "bounded_positive": BoundedPositive,
    "saturable": Saturable,
    "unconstrained_smooth": UnconstrainedSmooth,
}


def build_constraint_layer(template_name: str, dim: int) -> ConstraintTemplate:
    """Build a constraint template layer.

    Args:
        template_name: One of the 5 enumerated templates.
        dim: NODE dimension (must be <= template max).

    Raises:
        ValueError: If template unknown or dim exceeds max.
    """
    if template_name not in TEMPLATE_MAX_DIM:
        msg = f"Unknown constraint template '{template_name}'. Valid: {sorted(TEMPLATE_MAX_DIM)}"
        raise ValueError(msg)

    max_dim = TEMPLATE_MAX_DIM[template_name]
    if dim > max_dim:
        msg = f"dim={dim} exceeds max={max_dim} for constraint template '{template_name}'"
        raise ValueError(msg)

    cls = _TEMPLATE_CLASSES[template_name]
    return cls(dim=dim)
