# SPDX-License-Identifier: GPL-2.0-or-later
"""Bram-style NODE sub-model (PRD SS4.2.4).

Low-dimensional MLP that learns a PK sub-function (absorption or elimination
rate). Random effects on input-layer weights for per-subject variation.
Constraint template on output for physiological plausibility.

Reference: Bram DS et al. J Pharmacokinet Pharmacodyn 51:123-140, 2024.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from apmode.backends.node_constraints import (
    TEMPLATE_MAX_DIM,
    ConstraintTemplate,
    build_constraint_layer,
)


class NODESubModel(eqx.Module):
    """Low-dimensional NODE sub-model with constraint enforcement.

    Architecture:
      input (input_dim) -> Linear -> tanh -> Linear -> constraint -> scalar output

    Random effects are additive perturbations on the first Linear layer's weights.
    """

    input_dim: int
    hidden_dim: int
    constraint_name: str
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    constraint: ConstraintTemplate

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        constraint_template: str,
        key: jax.Array,
    ) -> None:
        max_dim = TEMPLATE_MAX_DIM.get(constraint_template)
        if max_dim is not None and hidden_dim > max_dim:
            msg = f"dim={hidden_dim} exceeds max={max_dim} for '{constraint_template}'"
            raise ValueError(msg)

        k1, k2 = jax.random.split(key)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.constraint_name = constraint_template
        self.linear1 = eqx.nn.Linear(input_dim, hidden_dim, key=k1)
        self.linear2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=k2)
        self.constraint = build_constraint_layer(constraint_template, hidden_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: input -> hidden -> constraint -> scalar."""
        h = jnp.tanh(self.linear1(x))
        raw = self.linear2(h)
        return self.constraint.apply(raw)

    def apply_random_effects(self, re: jax.Array) -> NODESubModel:
        """Return a new model with RE-perturbed input-layer weights.

        Per Bram et al.: random effects are additive perturbations on the
        first layer's weight matrix (broadcast across input_dim columns).
        """
        old_weight = self.linear1.weight  # shape: (hidden_dim, input_dim)
        new_weight = old_weight + re[:, None]  # broadcast re across input_dim
        new_linear1 = eqx.tree_at(lambda lin: lin.weight, self.linear1, new_weight)
        return eqx.tree_at(lambda m: m.linear1, self, new_linear1)  # type: ignore[no-any-return]
