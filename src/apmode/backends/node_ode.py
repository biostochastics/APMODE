# SPDX-License-Identifier: GPL-2.0-or-later
"""Hybrid mechanistic + NODE ODE system (PRD SS4.2.4).

Composes a mechanistic PK skeleton (1-cmt or 2-cmt) with a NODE sub-model
replacing either absorption or elimination. Diffrax integrates the system.

Example (1-cmt + NODE elimination):
  dA_depot/dt   = -ka * A_depot
  dA_central/dt = ka * A_depot - NODE(C, t) * A_central / V
  C = A_central / V
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from apmode.backends.node_model import NODESubModel


@dataclass(frozen=True)
class ODEConfig:
    """Configuration for the hybrid ODE system.

    This is a construction-time config, not stored on the eqx.Module.
    """

    n_cmt: Literal[1, 2] = 1
    node_position: Literal["absorption", "elimination"] = "elimination"
    constraint_template: str = "bounded_positive"
    node_dim: int = 3
    mechanistic_params: dict[str, float] | None = None


class HybridPKODE(eqx.Module):
    """Hybrid mechanistic + NODE ODE system.

    State vector:
      1-cmt: [A_depot, A_central]
      2-cmt: [A_depot, A_central, A_peripheral]

    The NODE sub-model replaces either the absorption or elimination
    rate law while the rest of the ODE skeleton remains mechanistic.
    """

    # Static (non-traced) config fields
    n_cmt: int = eqx.field(static=True)
    node_position: str = eqx.field(static=True)

    node: NODESubModel
    # Mechanistic params in log-space for positivity during optimization
    log_ka: jax.Array
    log_CL: jax.Array
    log_V: jax.Array
    log_V2: jax.Array
    log_Q: jax.Array

    # Keep config for external access (not traced)
    config: ODEConfig = eqx.field(static=True)

    def __init__(self, config: ODEConfig, key: jax.Array) -> None:
        params = config.mechanistic_params or {}
        self.config = config
        self.n_cmt = config.n_cmt
        self.node_position = config.node_position

        # Input dim: concentration + time
        input_dim = 2
        self.node = NODESubModel(
            input_dim=input_dim,
            hidden_dim=config.node_dim,
            constraint_template=config.constraint_template,
            key=key,
        )

        # Mechanistic parameters stored in log-space to ensure positivity
        # during optimization (CL, V, ka must all be > 0).
        self.log_ka = jnp.log(jnp.array(params.get("ka", 1.0)))
        self.log_CL = jnp.log(jnp.array(params.get("CL", 2.0)))
        self.log_V = jnp.log(jnp.array(params.get("V", 30.0)))
        self.log_V2 = jnp.log(jnp.array(params.get("V2", 40.0)))
        self.log_Q = jnp.log(jnp.array(params.get("Q", 5.0)))

    @property
    def ka(self) -> jax.Array:
        return jnp.exp(self.log_ka)

    @property
    def CL(self) -> jax.Array:
        return jnp.exp(self.log_CL)

    @property
    def V(self) -> jax.Array:
        return jnp.exp(self.log_V)

    @property
    def V2(self) -> jax.Array:
        return jnp.exp(self.log_V2)

    @property
    def Q(self) -> jax.Array:
        return jnp.exp(self.log_Q)

    def vector_field(self, t: jax.Array, y: jax.Array, _args: None) -> jax.Array:
        """ODE right-hand side for Diffrax."""
        if self.n_cmt == 1:
            return self._vf_1cmt(t, y)
        return self._vf_2cmt(t, y)

    def _vf_1cmt(self, t: jax.Array, y: jax.Array) -> jax.Array:
        """1-compartment hybrid ODE."""
        a_depot, a_central = y[0], y[1]
        conc = a_central / self.V
        node_input = jnp.array([conc, t])

        if self.node_position == "absorption":
            # NODE replaces absorption rate; elimination is mechanistic CL/V
            abs_rate = self.node(node_input).squeeze() * a_depot
            elim_rate = self.CL / self.V * a_central
            da_depot = -abs_rate
            da_central = abs_rate - elim_rate
        else:
            # NODE replaces elimination rate
            node_elim = self.node(node_input).squeeze()
            da_depot = -self.ka * a_depot
            da_central = self.ka * a_depot - node_elim * a_central
        return jnp.array([da_depot, da_central])

    def _vf_2cmt(self, t: jax.Array, y: jax.Array) -> jax.Array:
        """2-compartment hybrid ODE."""
        a_depot, a_central, a_periph = y[0], y[1], y[2]
        conc = a_central / self.V
        conc2 = a_periph / self.V2
        node_input = jnp.array([conc, t])

        # Inter-compartmental flow is always mechanistic
        flow = self.Q * (conc - conc2)

        if self.node_position == "absorption":
            abs_rate = self.node(node_input).squeeze() * a_depot
            elim_rate = self.CL / self.V * a_central
            da_depot = -abs_rate
            da_central = abs_rate - elim_rate - flow
        else:
            node_elim = self.node(node_input).squeeze()
            da_depot = -self.ka * a_depot
            da_central = self.ka * a_depot - node_elim * a_central - flow
        da_periph = flow
        return jnp.array([da_depot, da_central, da_periph])

    def solve(
        self,
        y0: jax.Array,
        times: jax.Array,
        *,
        t0: float = 0.0,
        solver: diffrax.AbstractSolver | None = None,  # type: ignore[type-arg]
        max_steps: int = 4096,
    ) -> jax.Array:
        """Integrate the ODE and return state at requested times.

        Args:
            y0: Initial state vector (post-dose).
            times: Sorted time points to evaluate at.
            t0: Integration start time (dose time). Defaults to 0.0.
            solver: Diffrax solver (default: Tsit5).
            max_steps: Maximum solver steps.

        Returns:
            Array of shape (len(times), n_states).
        """
        solver = solver or diffrax.Tsit5()
        term = diffrax.ODETerm(self.vector_field)  # type: ignore[arg-type]
        saveat = diffrax.SaveAt(ts=times)
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

        # Integrate from dose time (t0), not first observation time
        t_start = jnp.minimum(jnp.array(t0), times[0])

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_start,
            t1=times[-1],
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=max_steps,
        )
        return sol.ys  # type: ignore[no-any-return]

    def apply_subject_re(self, re: jax.Array) -> HybridPKODE:
        """Return a copy with RE-perturbed NODE input-layer weights."""
        new_node = self.node.apply_random_effects(re)
        return eqx.tree_at(lambda m: m.node, self, new_node)  # type: ignore[no-any-return]
