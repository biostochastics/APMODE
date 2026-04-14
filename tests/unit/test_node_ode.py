# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for hybrid mechanistic + NODE ODE system."""

from __future__ import annotations

import equinox as eqx  # type: ignore[import-untyped]
import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
import pytest

from apmode.backends.node_ode import HybridPKODE, ODEConfig


class TestHybridODEConstruction:
    """ODE system creation."""

    def test_creates_1cmt_node_elimination(self) -> None:
        ode = HybridPKODE(
            config=ODEConfig(
                n_cmt=1,
                node_position="elimination",
                constraint_template="bounded_positive",
                node_dim=3,
            ),
            key=jax.random.PRNGKey(0),
        )
        assert ode.config.n_cmt == 1
        assert ode.config.node_position == "elimination"

    def test_creates_1cmt_node_absorption(self) -> None:
        ode = HybridPKODE(
            config=ODEConfig(
                n_cmt=1,
                node_position="absorption",
                constraint_template="monotone_decreasing",
                node_dim=3,
            ),
            key=jax.random.PRNGKey(0),
        )
        assert ode.config.node_position == "absorption"

    def test_creates_2cmt_node_elimination(self) -> None:
        ode = HybridPKODE(
            config=ODEConfig(
                n_cmt=2,
                node_position="elimination",
                constraint_template="bounded_positive",
                node_dim=3,
                mechanistic_params={"ka": 1.0, "V": 30.0, "V2": 40.0, "Q": 5.0},
            ),
            key=jax.random.PRNGKey(0),
        )
        assert ode.config.n_cmt == 2

    def test_uses_custom_mechanistic_params(self) -> None:
        ode = HybridPKODE(
            config=ODEConfig(mechanistic_params={"ka": 2.5, "V": 50.0}),
            key=jax.random.PRNGKey(0),
        )
        assert float(ode.ka) == pytest.approx(2.5)
        assert float(ode.V) == pytest.approx(50.0)


class TestHybridODESolve1Cmt:
    """1-compartment ODE integration."""

    def _make_ode(self, node_position: str = "elimination") -> HybridPKODE:
        return HybridPKODE(
            config=ODEConfig(
                n_cmt=1,
                node_position=node_position,  # type: ignore[arg-type]
                constraint_template="bounded_positive",
                node_dim=3,
                mechanistic_params={"ka": 1.0, "V": 30.0},
            ),
            key=jax.random.PRNGKey(0),
        )

    def test_produces_finite_concentrations(self) -> None:
        ode = self._make_ode()
        times = jnp.linspace(0.0, 24.0, 49)
        y0 = jnp.array([100.0, 0.0])
        sol = ode.solve(y0, times)

        assert sol.shape == (49, 2)
        assert jnp.all(jnp.isfinite(sol))

    def test_depot_depletes(self) -> None:
        ode = self._make_ode()
        times = jnp.linspace(0.0, 24.0, 49)
        y0 = jnp.array([100.0, 0.0])
        sol = ode.solve(y0, times)

        depot = sol[:, 0]
        # Depot should decrease from initial dose
        assert float(depot[-1]) < float(depot[0])

    def test_central_rises_then_falls(self) -> None:
        ode = self._make_ode()
        times = jnp.linspace(0.0, 48.0, 97)
        y0 = jnp.array([100.0, 0.0])
        sol = ode.solve(y0, times)

        central = sol[:, 1]
        # Should peak somewhere in the middle
        peak_idx = int(jnp.argmax(central))
        assert 0 < peak_idx < len(times) - 1

    def test_node_absorption_produces_valid_curve(self) -> None:
        ode = self._make_ode("absorption")
        times = jnp.linspace(0.0, 24.0, 49)
        y0 = jnp.array([100.0, 0.0])
        sol = ode.solve(y0, times)

        assert jnp.all(jnp.isfinite(sol))
        assert sol.shape == (49, 2)


class TestHybridODESolve2Cmt:
    """2-compartment ODE integration."""

    def test_produces_3_state_solution(self) -> None:
        ode = HybridPKODE(
            config=ODEConfig(
                n_cmt=2,
                node_position="elimination",
                constraint_template="bounded_positive",
                node_dim=3,
                mechanistic_params={"ka": 1.0, "V": 30.0, "V2": 40.0, "Q": 5.0},
            ),
            key=jax.random.PRNGKey(0),
        )
        times = jnp.linspace(0.0, 24.0, 49)
        y0 = jnp.array([100.0, 0.0, 0.0])
        sol = ode.solve(y0, times)

        assert sol.shape == (49, 3)
        assert jnp.all(jnp.isfinite(sol))


class TestDifferentiability:
    """ODE solution must be differentiable for training."""

    def test_grad_through_solve(self) -> None:
        ode = HybridPKODE(
            config=ODEConfig(
                n_cmt=1,
                node_position="elimination",
                constraint_template="bounded_positive",
                node_dim=3,
                mechanistic_params={"ka": 1.0, "V": 30.0},
            ),
            key=jax.random.PRNGKey(0),
        )
        times = jnp.linspace(0.0, 12.0, 13)
        y0 = jnp.array([100.0, 0.0])

        @eqx.filter_grad
        def grad_fn(model: HybridPKODE) -> jax.Array:
            sol = model.solve(y0, times)
            return jnp.sum(sol[:, 1] ** 2)

        grads = grad_fn(ode)
        flat = jax.tree.leaves(grads)
        has_nonzero = any(jnp.any(g != 0) for g in flat if hasattr(g, "shape"))
        assert has_nonzero, "Gradients should be non-zero"


class TestSubjectRE:
    """Per-subject random effects on the hybrid ODE."""

    def test_re_changes_trajectory(self) -> None:
        ode = HybridPKODE(
            config=ODEConfig(
                n_cmt=1,
                node_position="elimination",
                constraint_template="bounded_positive",
                node_dim=3,
                mechanistic_params={"ka": 1.0, "V": 30.0},
            ),
            key=jax.random.PRNGKey(0),
        )
        times = jnp.linspace(0.0, 12.0, 13)
        y0 = jnp.array([100.0, 0.0])

        sol_pop = ode.solve(y0, times)
        re = jnp.array([0.2, -0.1, 0.05])
        ode_re = ode.apply_subject_re(re)
        sol_re = ode_re.solve(y0, times)

        assert not jnp.allclose(sol_pop, sol_re)
