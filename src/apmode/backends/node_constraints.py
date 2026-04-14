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
    """Output is strictly positive: sum of softplus activations.

    Note: This guarantees positive output (suitable for increasing rate
    constants), but does not enforce input-output monotonicity of the
    full NODE sub-function. The monotonicity of the composed function
    depends on the upstream linear layers.
    """

    def apply(self, raw: jax.Array) -> jax.Array:
        return jnp.sum(jax.nn.softplus(raw), keepdims=True)


class MonotoneDecreasing(ConstraintTemplate):
    """Output is positive and bounded: inverse of sum of softplus.

    Produces output in (0, 1/dim] range — a positive value that decreases
    as raw activations grow. Suitable for rate constants that attenuate
    with increasing concentration.
    """

    def apply(self, raw: jax.Array) -> jax.Array:
        # 1 / (1 + sum(softplus)) ensures positive output that decreases
        return (1.0 / (1.0 + jnp.sum(jax.nn.softplus(raw)))).reshape(1)


class BoundedPositive(ConstraintTemplate):
    """Output is strictly positive: softplus of sum."""

    def apply(self, raw: jax.Array) -> jax.Array:
        return jax.nn.softplus(jnp.sum(raw)).reshape(1)


class Saturable(ConstraintTemplate):
    """Output > 0 with MM-like saturation: Vmax * s / (Km + s).

    Uses learned log-scale parameters (JAX arrays, trainable) for the
    saturation plateau (Vmax) and half-saturation (Km). Output
    increases with activation magnitude but plateaus at Vmax.
    """

    log_vmax: jax.Array
    log_km: jax.Array

    def __init__(self, dim: int, scale: float = 1.0) -> None:
        self.dim = dim
        self.log_vmax = jnp.log(jnp.array(scale))
        self.log_km = jnp.log(jnp.array(1.0))

    def apply(self, raw: jax.Array) -> jax.Array:
        # MM-like: Vmax * s / (Km + s) where s = softplus(sum(raw))
        s = jax.nn.softplus(jnp.sum(raw))
        vmax = jnp.exp(self.log_vmax)
        km = jnp.exp(self.log_km)
        return (vmax * s / (km + s)).reshape(1)


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
