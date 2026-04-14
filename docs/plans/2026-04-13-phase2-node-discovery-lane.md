# Phase 2: Hybrid NODE Backend + Discovery Lane Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the hybrid mechanistic-NODE backend (Bram-style low-dimensional), Gate 2.5 credibility qualification, cross-paradigm ranking, functional distillation, and activate the Discovery lane with full orchestrator wiring.

**Architecture:** JAX/Diffrax/Equinox stack for neural ODE integration. The NODE sub-model is an Equinox MLP that replaces either the absorption or elimination sub-function in a mechanistic PK ODE skeleton. Random effects are placed on NODE input-layer weights (Bram et al. 2024). Constraint templates (5 enumerated types) enforce physiological plausibility via output-layer activation functions. The `NodeBackendRunner` implements the same `BackendRunner` protocol as `Nlmixr2Runner`, producing identical `BackendResult` artifacts. Governance adds Gate 2.5 (ICH M15 credibility) between Gate 2 and Gate 3, and Gate 3 gains cross-paradigm ranking via simulation-based metrics (VPC coverage concordance, AUC/Cmax bioequivalence, NPE).

**Tech Stack:** Python 3.12+, JAX >= 0.4.30, Diffrax >= 0.6, Equinox >= 0.11, Optax >= 0.2, jaxtyping >= 0.2. Existing: Pydantic v2, structlog, Typer, Pandera, Lark, sparkid.

**Key PRD references:**
- PRD v0.3 SS4.2.4 — Hybrid Mechanistic-NODE Backend
- PRD v0.3 SS4.2.5 — NODE constraint templates and dim ceilings
- PRD v0.3 SS4.3.1 — Gate 2.5 Credibility Qualification, Gate 3 Cross-paradigm ranking
- PRD v0.3 SS4.3.3 — Credibility Assessment Reporting
- PRD v0.3 SS5 — Benchmark Suite A (full) + B
- PRD v0.3 SS8 — Phase 2 scope
- ARCHITECTURE.md SS2.7.1 — GPU Non-Determinism Boundary
- ARCHITECTURE.md SS4.4 — Credibility Assessment Report Schema
- ARCHITECTURE.md SS6 — Phase 2 checklist

---

## Dependency Graph

```
Task 0 (deps)
  |
Task 1 (constraints)
  |
Task 2 (sub-model)  Task 12 (policies) -- independent
  |                  Task 7 (Gate 2.5) -- independent
  +-------+          Task 8 (Gate 3) -- independent
  |       |
Task 3    |
(ODE)     |
  |       |
  +---+---+
      |
Task 4 (trainer)
      |
Task 5 (runner)  Task 9 (distillation) -- needs Task 5
      |                |
      +-------+--------+
              |
Task 6 (Gate2.5 wired)
              |
Task 10 (orchestrator)
              |
Task 11 (credibility report)
              |
Task 12 (benchmarks)
```

---

## Task 0: Add JAX/Diffrax/Equinox/Optax Dependencies

**Files:**
- Modify: `pyproject.toml:24-50`
- Modify: `.github/workflows/ci.yml` (add JAX to CI matrix)

**Step 1: Add `node` optional dependency group to pyproject.toml**

In `pyproject.toml`, add after the `dev` group:

```toml
node = [
    "jax[cpu]>=0.4.30",
    "diffrax>=0.6",
    "equinox>=0.11",
    "optax>=0.2",
    "jaxtyping>=0.2",
]
```

Update the `test` group to include node deps for CI:

```toml
test = [
    "pytest>=8.0",
    "hypothesis>=6.100",
    "syrupy>=4.0",
    "pytest-asyncio>=0.23",
    "pandas>=2.0",
    "jax[cpu]>=0.4.30",
    "diffrax>=0.6",
    "equinox>=0.11",
    "optax>=0.2",
    "jaxtyping>=0.2",
]
```

**Step 2: Sync and verify**

```bash
uv sync --all-extras
uv run python -c "import jax; import diffrax; import equinox; import optax; print('JAX', jax.__version__)"
```

Expected: prints JAX version, no import errors.

**Step 3: Run existing tests to confirm no regressions**

```bash
uv run pytest tests/ -q
```

Expected: 679 tests passed.

**Step 4: Type-check**

```bash
uv run mypy src/apmode/ --strict
```

Expected: 0 errors.

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(phase2): add JAX/Diffrax/Equinox/Optax dependencies for NODE backend"
```

---

## Task 1: NODE Constraint Enforcement Module

**Files:**
- Create: `src/apmode/backends/node_constraints.py`
- Test: `tests/unit/test_node_constraints.py`

**Context:** PRD SS4.2.5 specifies 5 enumerated constraint templates that NODE modules must use. Each template constrains the NODE output to maintain physiological plausibility. The constraint is applied as the final layer of the NODE MLP.

**Constraint template specifications:**

| Template | Output constraint | Activation | Max dim |
|----------|------------------|------------|---------|
| `monotone_increasing` | dy/dx >= 0 | Softplus on weights | 4 |
| `monotone_decreasing` | dy/dx <= 0 | Negative softplus | 4 |
| `bounded_positive` | y > 0 | Softplus on output | 6 |
| `saturable` | y > 0, dy/dx <= 0 for large x | Sigmoid-scaled | 4 |
| `unconstrained_smooth` | smooth only | Tanh hidden layers | 8 |

**Step 1: Write the failing test**

Create `tests/unit/test_node_constraints.py`:

```python
"""Tests for NODE constraint template enforcement."""

from __future__ import annotations

import jax
import jax.numpy as jnp
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

    @pytest.mark.parametrize("template_name", list(CONSTRAINT_REGISTRY.keys()) if CONSTRAINT_REGISTRY else [])
    def test_each_template_returns_constraint(self, template_name: str) -> None:
        ct = build_constraint_layer(template_name, dim=2)
        assert isinstance(ct, ConstraintTemplate)


class TestMonotoneIncreasing:
    """monotone_increasing: output must be non-decreasing."""

    def test_output_non_decreasing(self) -> None:
        ct = build_constraint_layer("monotone_increasing", dim=2)
        key = jax.random.PRNGKey(42)
        x_sorted = jnp.linspace(0.1, 10.0, 50).reshape(-1, 1)
        # The constraint layer transforms raw MLP output to enforce monotonicity
        raw = jnp.linspace(-2.0, 2.0, 50).reshape(-1, 1)
        y = jax.vmap(ct.apply)(raw)
        diffs = jnp.diff(y.squeeze())
        assert jnp.all(diffs >= -1e-6), f"Not monotone: min diff = {diffs.min()}"


class TestBoundedPositive:
    """bounded_positive: output must be > 0."""

    def test_output_positive(self) -> None:
        ct = build_constraint_layer("bounded_positive", dim=3)
        raw = jnp.array([[-5.0], [0.0], [5.0], [-100.0]])
        y = jax.vmap(ct.apply)(raw)
        assert jnp.all(y > 0), f"Not positive: min = {y.min()}"


class TestSaturable:
    """saturable: output > 0, approaches plateau for large input."""

    def test_output_positive_and_bounded(self) -> None:
        ct = build_constraint_layer("saturable", dim=2)
        raw = jnp.linspace(-5.0, 50.0, 100).reshape(-1, 1)
        y = jax.vmap(ct.apply)(raw)
        assert jnp.all(y > 0)
        # Output should plateau (last values should be close together)
        tail = y[-10:]
        assert jnp.std(tail) < 0.5 * jnp.mean(tail), "Not saturating"


class TestDimLimits:
    """Constraint templates enforce max dimension limits."""

    def test_monotone_max_dim_4(self) -> None:
        with pytest.raises(ValueError, match="dim.*exceeds.*max"):
            build_constraint_layer("monotone_increasing", dim=5)

    def test_saturable_max_dim_4(self) -> None:
        with pytest.raises(ValueError, match="dim.*exceeds.*max"):
            build_constraint_layer("saturable", dim=5)

    def test_unconstrained_max_dim_8(self) -> None:
        with pytest.raises(ValueError, match="dim.*exceeds.*max"):
            build_constraint_layer("unconstrained_smooth", dim=9)

    def test_bounded_positive_max_dim_6(self) -> None:
        with pytest.raises(ValueError, match="dim.*exceeds.*max"):
            build_constraint_layer("bounded_positive", dim=7)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_node_constraints.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'apmode.backends.node_constraints'`

**Step 3: Implement `src/apmode/backends/node_constraints.py`**

```python
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

from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

# Max dim per constraint template (matches validator.py _TEMPLATE_MAX_DIM)
TEMPLATE_MAX_DIM: dict[str, int] = {
    "monotone_increasing": 4,
    "monotone_decreasing": 4,
    "bounded_positive": 6,
    "saturable": 4,
    "unconstrained_smooth": 8,
}


class ConstraintTemplate(eqx.Module):
    """Base for constraint templates. Subclasses implement apply()."""

    dim: int

    def apply(self, raw: Float[Array, "dim"]) -> Float[Array, "1"]:
        raise NotImplementedError


class MonotoneIncreasing(ConstraintTemplate):
    """Output is non-decreasing: cumulative softplus of raw values."""

    def apply(self, raw: Float[Array, "dim"]) -> Float[Array, "1"]:
        # Softplus ensures non-negative increments; cumsum gives monotonicity
        increments = jnp.log1p(jnp.exp(raw))
        return jnp.sum(increments, keepdims=True)


class MonotoneDecreasing(ConstraintTemplate):
    """Output is non-increasing: negative cumulative softplus."""

    def apply(self, raw: Float[Array, "dim"]) -> Float[Array, "1"]:
        increments = jnp.log1p(jnp.exp(raw))
        return -jnp.sum(increments, keepdims=True)


class BoundedPositive(ConstraintTemplate):
    """Output is strictly positive: softplus of sum."""

    def apply(self, raw: Float[Array, "dim"]) -> Float[Array, "1"]:
        s = jnp.sum(raw)
        return jnp.log1p(jnp.exp(s)).reshape(1)


class Saturable(ConstraintTemplate):
    """Output > 0, saturates for large input: sigmoid-scaled."""

    scale: float = 1.0

    def apply(self, raw: Float[Array, "dim"]) -> Float[Array, "1"]:
        s = jnp.sum(raw)
        # Sigmoid gives (0, 1) range; scale shifts to (0, scale)
        return (self.scale * jax.nn.sigmoid(s)).reshape(1)


class UnconstrainedSmooth(ConstraintTemplate):
    """Smooth output, no positivity/monotonicity constraint."""

    def apply(self, raw: Float[Array, "dim"]) -> Float[Array, "1"]:
        return jnp.tanh(jnp.sum(raw)).reshape(1)


# Registry mapping template name -> class
_TEMPLATE_CLASSES: dict[str, type[ConstraintTemplate]] = {
    "monotone_increasing": MonotoneIncreasing,
    "monotone_decreasing": MonotoneDecreasing,
    "bounded_positive": BoundedPositive,
    "saturable": Saturable,
    "unconstrained_smooth": UnconstrainedSmooth,
}

# Public registry (for test introspection)
CONSTRAINT_REGISTRY = TEMPLATE_MAX_DIM


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
```

Note: The `Saturable` class needs `import jax` at module level. Add it.

**Step 4: Run tests**

```bash
uv run pytest tests/unit/test_node_constraints.py -v
```

Expected: All tests pass.

**Step 5: Type-check and lint**

```bash
uv run mypy src/apmode/backends/node_constraints.py --strict
uv run ruff check src/apmode/backends/node_constraints.py
```

**Step 6: Commit**

```bash
git add src/apmode/backends/node_constraints.py tests/unit/test_node_constraints.py
git commit -m "feat(node): constraint template enforcement module (5 templates)"
```

---

## Task 2: NODE Sub-Model (Equinox MLP with Constraints)

**Files:**
- Create: `src/apmode/backends/node_model.py`
- Test: `tests/unit/test_node_model.py`

**Context:** The NODE sub-model is a low-dimensional MLP (Bram-style) that learns a single PK sub-function (absorption rate or elimination rate). Inputs are interpretable: (concentration, time, optional covariates). Output is a scalar (rate). Random effects are placed on input-layer weights to create per-subject variation.

**Key design decisions (from PRD SS4.2.4):**
- Population parameters: MLP weights/biases (shared across subjects)
- Random effects: perturbations on input-layer weights (dim = NODE dim)
- Constraint template: applied to output layer
- Input features: concentration (1D), time (1D), covariates (variable)

**Step 1: Write the failing test**

Create `tests/unit/test_node_model.py`:

```python
"""Tests for the Bram-style NODE sub-model."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from apmode.backends.node_model import NODESubModel


class TestNODESubModelInit:
    """Construction and shape checks."""

    def test_creates_with_valid_params(self) -> None:
        model = NODESubModel(
            input_dim=2,  # concentration + time
            hidden_dim=4,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        assert model.input_dim == 2
        assert model.hidden_dim == 4

    def test_rejects_dim_exceeding_template_max(self) -> None:
        with pytest.raises(ValueError, match="dim.*exceeds"):
            NODESubModel(
                input_dim=2,
                hidden_dim=5,  # bounded_positive max is 6, but this is hidden_dim not node dim
                constraint_template="monotone_increasing",
                key=jax.random.PRNGKey(0),
            )


class TestNODESubModelForward:
    """Forward pass produces valid output."""

    def test_output_shape_scalar(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=4,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        x = jnp.array([1.0, 0.5])  # [concentration, time]
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

    def test_differentiable(self) -> None:
        model = NODESubModel(
            input_dim=2,
            hidden_dim=4,
            constraint_template="bounded_positive",
            key=jax.random.PRNGKey(0),
        )
        x = jnp.array([1.0, 0.5])

        def loss_fn(m: NODESubModel) -> jax.Array:
            return jnp.sum(m(x) ** 2)

        grads = jax.grad(loss_fn)(model)
        # Grads should exist (not all zero for a non-trivial model)
        flat_grads = jax.tree.leaves(grads)
        assert any(jnp.any(g != 0) for g in flat_grads)


class TestRandomEffects:
    """Random effects on input-layer weights."""

    def test_re_perturbation_changes_output(self) -> None:
        key = jax.random.PRNGKey(0)
        model = NODESubModel(
            input_dim=2, hidden_dim=4,
            constraint_template="bounded_positive", key=key,
        )
        x = jnp.array([5.0, 1.0])

        y_pop = model(x)
        # Apply random effect: perturb input-layer weights
        re = jnp.array([0.1, -0.1, 0.05, 0.0])  # dim = hidden_dim
        model_re = model.apply_random_effects(re)
        y_re = model_re(x)

        assert not jnp.allclose(y_pop, y_re), "RE should change output"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_node_model.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement `src/apmode/backends/node_model.py`**

```python
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
from jaxtyping import Array, Float, PRNGKeyArray

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
        key: PRNGKeyArray,
    ) -> None:
        # Validate dim against constraint template max
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

    def __call__(self, x: Float[Array, "input_dim"]) -> Float[Array, "1"]:
        """Forward pass: input -> hidden -> constraint -> scalar."""
        h = jnp.tanh(self.linear1(x))
        raw = self.linear2(h)
        return self.constraint.apply(raw)

    def apply_random_effects(
        self, re: Float[Array, "hidden_dim"],
    ) -> NODESubModel:
        """Return a new model with RE-perturbed input-layer weights.

        Per Bram et al.: random effects are additive perturbations on the
        first layer's weight matrix (broadcast across input_dim columns).
        """
        # Perturb: new_weight[i, j] = weight[i, j] + re[i]
        old_weight = self.linear1.weight  # shape: (hidden_dim, input_dim)
        new_weight = old_weight + re[:, None]  # broadcast re across input_dim
        new_linear1 = eqx.tree_at(lambda l: l.weight, self.linear1, new_weight)
        return eqx.tree_at(lambda m: m.linear1, self, new_linear1)
```

**Step 4: Run tests and type-check**

```bash
uv run pytest tests/unit/test_node_model.py -v
uv run mypy src/apmode/backends/node_model.py --strict
```

**Step 5: Commit**

```bash
git add src/apmode/backends/node_model.py tests/unit/test_node_model.py
git commit -m "feat(node): Bram-style NODE sub-model with RE on input weights"
```

---

## Task 3: Hybrid ODE System (Mechanistic + NODE)

**Files:**
- Create: `src/apmode/backends/node_ode.py`
- Test: `tests/unit/test_node_ode.py`

**Context:** The hybrid ODE composes a mechanistic PK skeleton (standard 1-cmt or 2-cmt) with a NODE sub-model replacing either the absorption or elimination sub-function. Diffrax integrates the combined system.

**ODE structure (1-cmt + NODE elimination example):**
```
dA_depot/dt = -ka * A_depot                     # mechanistic absorption
dA_central/dt = ka * A_depot - NODE(C, t) * C   # NODE-learned elimination
C = A_central / V                                # concentration
```

**Step 1: Write the failing test**

Create `tests/unit/test_node_ode.py`:

```python
"""Tests for hybrid mechanistic + NODE ODE system."""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import pytest

from apmode.backends.node_ode import HybridPKODE, ODEConfig


class TestHybridODEConstruction:
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


class TestHybridODESolve:
    """Integration produces valid PK curves."""

    def test_1cmt_node_elim_produces_concentrations(self) -> None:
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
        # Single dose: 100 mg at t=0
        times = jnp.linspace(0.0, 24.0, 49)
        y0 = jnp.array([100.0, 0.0])  # [depot, central]
        sol = ode.solve(y0, times)
        concentrations = sol[:, 1] / 30.0  # central/V

        assert concentrations.shape == (49,)
        assert jnp.all(jnp.isfinite(concentrations))
        # Concentration should rise then fall (typical PK curve)
        assert concentrations[0] < concentrations[5]  # absorption phase
        assert concentrations[-1] < concentrations[12]  # elimination phase

    def test_solution_is_differentiable(self) -> None:
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
        times = jnp.linspace(0.0, 24.0, 25)
        y0 = jnp.array([100.0, 0.0])

        def loss_fn(model: HybridPKODE) -> jax.Array:
            sol = model.solve(y0, times)
            return jnp.sum(sol[:, 1] ** 2)

        grads = jax.grad(loss_fn)(ode)
        flat = jax.tree.leaves(grads)
        assert any(jnp.any(g != 0) for g in flat if isinstance(g, jnp.ndarray))
```

**Step 2: Run to verify failure, then implement**

**Step 3: Implement `src/apmode/backends/node_ode.py`**

The module defines:
- `ODEConfig`: dataclass with n_cmt, node_position, constraint_template, node_dim, mechanistic_params
- `HybridPKODE(eqx.Module)`: contains NODESubModel + mechanistic params, implements `vector_field()` and `solve()` using `diffrax.diffeqsolve` with `Tsit5` solver

Key implementation detail: The `vector_field` method must switch on `node_position` to route either absorption or elimination through the NODE sub-model while keeping the other sub-function mechanistic.

**Step 4: Run tests, type-check, commit**

```bash
git add src/apmode/backends/node_ode.py tests/unit/test_node_ode.py
git commit -m "feat(node): hybrid mechanistic + NODE ODE system with Diffrax integration"
```

---

## Task 4: NODE Training Loop (Population-Level Fitting)

**Files:**
- Create: `src/apmode/backends/node_trainer.py`
- Test: `tests/unit/test_node_trainer.py`

**Context:** The training loop fits the hybrid ODE to population PK data by optimizing MLP weights + RE variance parameters. Uses Optax for optimization with early stopping.

**Training objective:**
- Negative log-likelihood: sum over subjects of -log p(y_i | theta, eta_i)
- theta = population MLP weights + mechanistic params + sigma
- eta_i = per-subject RE on input-layer weights
- Outer optimization: theta via Adam
- Inner optimization (or Laplace approximation): eta_i per subject

**Step 1: Write tests for NodeTrainer**

Key test cases:
- `test_training_reduces_loss`: loss decreases over 10 epochs on synthetic data
- `test_early_stopping`: training stops when loss plateau detected
- `test_convergence_metadata_produced`: returns ConvergenceMetadata-compatible dict
- `test_cpu_deterministic_mode`: same seed produces same result
- `test_gradient_clipping_prevents_nan`: extreme data doesn't produce NaN

**Step 2: Implement `src/apmode/backends/node_trainer.py`**

Key components:
- `TrainingConfig`: epochs, lr, lr_schedule, grad_clip, early_stop_patience, execution_mode
- `NodeTrainer`: takes HybridPKODE + data, runs Optax loop, returns trained model + metadata
- Population NLL objective function
- Laplace approximation for per-subject RE (avoids inner loop cost)

**Step 3: Run tests, type-check, commit**

---

## Task 5: NodeBackendRunner (BackendRunner Protocol Implementation)

**Files:**
- Create: `src/apmode/backends/node_runner.py`
- Modify: `src/apmode/backends/__init__.py` (re-export)
- Test: `tests/unit/test_node_runner.py`

**Context:** Maps DSLSpec with NODE modules -> HybridPKODE -> training -> BackendResult. Must satisfy the `BackendRunner` protocol (`async def run(...) -> BackendResult`).

**Step 1: Write tests**

Key test cases:
- `test_implements_backend_runner_protocol`: `isinstance(NodeBackendRunner(...), BackendRunner)`
- `test_run_with_node_absorption_spec`: dispatches a spec with NODE_Absorption, returns BackendResult
- `test_run_with_node_elimination_spec`: dispatches a spec with NODE_Elimination
- `test_rejects_spec_without_node_modules`: raises InvalidSpecError for classical specs
- `test_backend_field_is_jax_node`: result.backend == "jax_node"
- `test_convergence_metadata_populated`: wall_time > 0, method = "adam"

**Step 2: Implement `src/apmode/backends/node_runner.py`**

The runner:
1. Validates spec has NODE modules
2. Extracts mechanistic params from spec (distribution, non-NODE absorption/elimination)
3. Builds ODEConfig from DSLSpec
4. Constructs HybridPKODE
5. Runs NodeTrainer on the data
6. Extracts trained params into ParameterEstimate dict
7. Computes diagnostics (GOF, VPC via simulation)
8. Returns BackendResult with backend="jax_node"

Process isolation: runs in-process (JAX is already isolated from R). The `execution_mode` config controls CPU vs GPU.

**Step 3: Run tests, type-check, commit**

---

## Task 6: Gate 2.5 -- Credibility Qualification (ICH M15)

**Files:**
- Modify: `src/apmode/governance/gates.py:691-751` (replace scaffold with real checks)
- Modify: `src/apmode/governance/policy.py:55-66` (flesh out Gate25Config)
- Modify: `src/apmode/bundle/models.py` (add ContextOfUse, SensitivityResult models)
- Modify: `src/apmode/bundle/emitter.py` (add write_context_of_use, write_sensitivity)
- Test: `tests/unit/test_gates.py` (add Gate 2.5 test cases)

**Context:** PRD SS4.3.1 Gate 2.5 table. ICH M15 requires context-of-use specification and risk-based model assessment. This is a qualifying gate, not documentation.

**Gate 2.5 checks (from PRD SS4.3.1 table):**

| Check | Submission | Discovery | Optimization |
|-------|-----------|-----------|-------------|
| Context-of-use statement | Required: decision + risk | Required: intent | Required: decision + population |
| Limitation-to-risk mapping | Required | Encouraged | Required |
| Data adequacy vs complexity | n_obs/n_params > threshold | Relaxed | n_obs/n_params > threshold |
| Sensitivity analysis | Required | Not required | Required |
| AI/ML transparency | N/A | Required (if NODE) | Required (if NODE) |

**Step 1: Write failing tests**

Add to `tests/unit/test_gates.py`:

```python
class TestGate25Credibility:
    """Gate 2.5: Credibility Qualification (ICH M15)."""

    def test_passes_with_adequate_context(self) -> None:
        ...  # Backend result + policy with gate2_5 config + context_of_use

    def test_fails_missing_context_of_use(self) -> None:
        ...  # No context statement when required -> fail

    def test_fails_insufficient_data_adequacy(self) -> None:
        ...  # n_obs/n_params below threshold -> fail

    def test_node_requires_ml_transparency(self) -> None:
        ...  # NODE backend + no transparency statement -> fail

    def test_classical_skips_ml_transparency(self) -> None:
        ...  # nlmixr2 backend -> ML transparency check passes (not applicable)
```

**Step 2: Implement real Gate 2.5 checks**

Replace the scaffold in `evaluate_gate2_5()` with:
- `_check_context_of_use()`: validates presence and structure of context statement
- `_check_limitation_to_risk()`: validates limitation-risk mapping exists when required
- `_check_data_adequacy()`: n_obs / n_params > policy threshold
- `_check_sensitivity()`: sensitivity results available when required
- `_check_ml_transparency()`: AI/ML transparency statement present for NODE/agentic

**Step 3: Update Gate25Config with real thresholds**

```python
class Gate25Config(BaseModel):
    context_of_use_required: bool = True
    limitation_to_risk_mapping_required: bool = False
    data_adequacy_required: bool = True
    data_adequacy_ratio_min: float = Field(default=5.0, gt=0)  # n_obs/n_params
    sensitivity_analysis_required: bool = False
    ai_ml_transparency_required: bool = False
```

**Step 4: Run tests, type-check, commit**

---

## Task 7: Gate 3 -- Cross-Paradigm Ranking

**Files:**
- Modify: `src/apmode/governance/gates.py:591-683` (extend evaluate_gate3)
- Create: `src/apmode/governance/ranking.py` (simulation-based metrics)
- Modify: `src/apmode/bundle/models.py` (add CrossParadigmMetrics)
- Test: `tests/unit/test_cross_paradigm_ranking.py`

**Context:** PRD SS4.3.1 specifies cross-paradigm ranking via simulation-based metrics when candidates come from different backends. NLPD is retained within-paradigm only.

**Cross-paradigm metrics (PRD SS4.3.1):**
1. **VPC coverage concordance:** all candidates simulate from the same dosing scenarios; VPC coverage overlap is computed
2. **AUC/Cmax bioequivalence:** geometric mean ratio within 80-125%
3. **NPE (Normalized Prediction Error):** prediction accuracy on held-out subjects

**Step 1: Write failing tests**

```python
class TestCrossParadigmRanking:
    def test_within_paradigm_uses_bic(self) -> None:
        """Single-backend survivors: rank by BIC as before."""
        ...

    def test_cross_paradigm_uses_simulation_metrics(self) -> None:
        """Mixed-backend survivors: rank by VPC concordance + NPE."""
        ...

    def test_nlpd_not_used_cross_paradigm(self) -> None:
        """NLPD must not be the primary metric for cross-paradigm."""
        ...

    def test_qualified_comparison_flag(self) -> None:
        """Cross-paradigm rankings flagged as qualified comparisons."""
        ...
```

**Step 2: Implement `src/apmode/governance/ranking.py`**

Functions:
- `compute_vpc_concordance(results: list[BackendResult]) -> float`
- `compute_auc_cmax_be(results: list[BackendResult]) -> dict[str, float]`
- `compute_npe(results: list[BackendResult], test_data: ...) -> float`
- `rank_cross_paradigm(survivors: list[BackendResult], ...) -> list[RankedCandidate]`

**Step 3: Update `evaluate_gate3` to detect mixed backends and switch to cross-paradigm ranking**

**Step 4: Run tests, type-check, commit**

---

## Task 8: Functional Distillation Module

**Files:**
- Create: `src/apmode/backends/node_distillation.py`
- Modify: `src/apmode/bundle/models.py` (add DistillationReport)
- Test: `tests/unit/test_node_distillation.py`

**Context:** PRD SS4.2.4 specifies functional distillation (NOT SHAP) for NODE interpretability. Three components: visualization, surrogate fitting, fidelity quantification.

**Step 1: Write failing tests**

```python
class TestSubFunctionVisualization:
    def test_produces_clearance_curve_data(self) -> None:
        """NODE elimination: clearance vs concentration curve."""
        ...

    def test_produces_absorption_curve_data(self) -> None:
        """NODE absorption: absorption rate vs time curve."""
        ...

class TestSurrogateFitting:
    def test_fits_parametric_surrogate(self) -> None:
        """Fit classical parametric form to NODE output."""
        ...

    def test_surrogate_has_interpretable_params(self) -> None:
        """Surrogate returns named params (CL, Vmax, Km, etc.)."""
        ...

class TestFidelityQuantification:
    def test_auc_cmax_be_within_bounds(self) -> None:
        """Surrogate AUC/Cmax within 80-125% GMR of NODE."""
        ...

    def test_fidelity_report_structure(self) -> None:
        """DistillationReport has required fields."""
        ...
```

**Step 2: Implement `src/apmode/backends/node_distillation.py`**

Functions:
- `visualize_sub_function(model, conc_range, time_range) -> dict`
- `fit_parametric_surrogate(model, conc_range) -> SurrogateResult`
- `quantify_fidelity(node_model, surrogate, dosing_scenarios) -> FidelityResult`
- `distill(model, config) -> DistillationReport`

**Step 3: Add DistillationReport to bundle models**

```python
class DistillationReport(BaseModel):
    candidate_id: str
    node_position: Literal["absorption", "elimination"]
    sub_function_data: dict[str, list[float]]  # x, y pairs
    surrogate_type: str  # e.g., "MichaelisMenten", "FirstOrder"
    surrogate_params: dict[str, float]
    fidelity_auc_gmr: float
    fidelity_cmax_gmr: float
    fidelity_pass: bool  # 80-125% BE
```

**Step 4: Run tests, type-check, commit**

---

## Task 9: Discovery Lane Activation + Orchestrator Wiring

**Files:**
- Modify: `src/apmode/orchestrator/__init__.py` (multi-backend dispatch, Gate 2.5)
- Modify: `src/apmode/search/engine.py` (multi-backend support)
- Modify: `src/apmode/bundle/emitter.py` (distillation, credibility artifacts)
- Test: `tests/integration/test_discovery_lane.py`

**Context:** The orchestrator must:
1. Accept a `NodeBackendRunner` alongside `Nlmixr2Runner`
2. SearchEngine dispatches candidates to the appropriate backend based on Lane Router
3. Gate 2.5 runs between Gate 2 and Gate 3
4. Gate 3 uses cross-paradigm ranking when mixed backends survive
5. Distillation runs on NODE survivors

**Step 1: Modify Orchestrator to accept multiple runners**

Change `__init__` signature:

```python
def __init__(
    self,
    runner: Nlmixr2Runner,
    bundle_base_dir: Path,
    config: RunConfig | None = None,
    node_runner: NodeBackendRunner | None = None,  # Phase 2
) -> None:
```

**Step 2: Modify SearchEngine for multi-backend dispatch**

Add `runners: dict[str, BackendRunner]` parameter. When a candidate spec has NODE modules, dispatch to the node runner; otherwise dispatch to nlmixr2.

**Step 3: Wire Gate 2.5 into the governance pipeline**

In the orchestrator's gate evaluation loop, after Gate 2 passes, run `evaluate_gate2_5()` with the credibility context. Only Gate 2.5 survivors proceed to Gate 3.

**Step 4: Write integration test**

```python
class TestDiscoveryLanePipeline:
    """End-to-end Discovery lane with classical + NODE candidates."""

    async def test_discovery_produces_mixed_ranking(self) -> None:
        """Classical and NODE candidates both evaluated and ranked."""
        ...

    async def test_node_excluded_from_submission(self) -> None:
        """Submission lane does not dispatch NODE candidates."""
        ...

    async def test_gate25_filters_before_ranking(self) -> None:
        """Gate 2.5 failures don't reach Gate 3."""
        ...
```

**Step 5: Run full test suite, type-check, commit**

---

## Task 10: Credibility Assessment Report Generator

**Files:**
- Create: `src/apmode/report/__init__.py`
- Create: `src/apmode/report/credibility.py`
- Modify: `src/apmode/bundle/emitter.py` (write report artifacts)
- Test: `tests/unit/test_credibility_report.py`

**Context:** ARCHITECTURE.md SS4.4 specifies the credibility report schema. Each recommended model gets a `{candidate_id}_credibility.json` in the `report/` directory.

**Step 1: Write tests for report generation**

```python
class TestCredibilityReportGenerator:
    def test_generates_report_for_classical_model(self) -> None:
        ...  # CredibilityReport with all required fields

    def test_generates_report_with_ml_transparency_for_node(self) -> None:
        ...  # NODE model includes ML transparency section

    def test_report_matches_schema(self) -> None:
        ...  # Validates against CredibilityReport Pydantic model

    def test_context_of_use_populated(self) -> None:
        ...
```

**Step 2: Implement report generator**

**Step 3: Wire into orchestrator (after Gate 3, for recommended candidates)**

**Step 4: Run tests, type-check, commit**

---

## Task 11: Benchmark Suite A Expansion + Suite B

**Files:**
- Create: `benchmarks/suite_b/` directory
- Modify: `benchmarks/suite_a/simulate_all.R` (add A5-A8)
- Create: `src/apmode/benchmarks/suite_b.py`
- Test: `tests/unit/test_benchmark_suite_b.py`

**Context:** PRD SS5 defines:
- A5-A8 scenarios for Suite A (full)
- Suite B: NODE-specific evaluation scenarios

**Suite A new scenarios (from PRD SS5 table):**
- A5: 2-cmt + IOV (multiple occasions)
- A6: 1-cmt + parallel linear+MM elimination + BLQ
- A7: 2-cmt + NODE-generated nonlinear absorption
- A8: 1-cmt + time-varying clearance + covariate

**Suite B scenarios:**
- B1: NODE absorption recovery (known ground truth function)
- B2: NODE elimination under sparse data (should produce `data_insufficient`)
- B3: Cross-paradigm ranking correctness (classical should win when data is simple)

**Step 1: Write R simulation scripts for A5-A8, B1-B3**

**Step 2: Write Python benchmark module and test assertions**

**Step 3: Update CI workflow to run new benchmarks**

**Step 4: Commit**

---

## Task 12: Policy Updates + execution_mode Config

**Files:**
- Modify: `policies/discovery.json` (add gate2_5 thresholds)
- Modify: `policies/optimization.json` (add gate2_5 thresholds)
- Modify: `policies/submission.json` (add gate2_5 thresholds)
- Modify: `src/apmode/orchestrator/__init__.py:52-60` (add execution_mode to RunConfig)
- Modify: `src/apmode/bundle/models.py:205-216` (update SeedRegistry)
- Test: `tests/unit/test_gate_policy.py`

**Step 1: Update policy files**

Add to each policy JSON:

```json
"gate2_5": {
    "context_of_use_required": true,
    "limitation_to_risk_mapping_required": false,
    "data_adequacy_required": true,
    "data_adequacy_ratio_min": 5.0,
    "sensitivity_analysis_required": false,
    "ai_ml_transparency_required": false
}
```

For discovery.json, set `ai_ml_transparency_required: true`.
For optimization.json, set `ai_ml_transparency_required: true, sensitivity_analysis_required: true`.

**Step 2: Add execution_mode to RunConfig**

```python
@dataclass
class RunConfig:
    lane: Literal["submission", "discovery", "optimization"] = "submission"
    seed: int = 42
    timeout_seconds: int = 600
    policy_path: Path | None = None
    covariate_names: list[str] = field(default_factory=list)
    execution_mode: Literal["cpu_deterministic", "gpu_fast"] = "cpu_deterministic"
```

**Step 3: Run policy validation CI check**

```bash
uv run python -m apmode.governance.validate_policies
```

**Step 4: Run full test suite, commit**

---

## Verification Checklist

After all tasks complete:

1. `uv run pytest tests/ -q` -- all tests pass (old + new)
2. `uv run mypy src/apmode/ --strict` -- 0 errors
3. `uv run ruff check src/apmode/ tests/` -- 0 violations
4. `uv run python -m apmode.governance.validate_policies` -- all policies valid
5. Discovery lane integration test passes
6. NODE constraint templates all tested
7. Functional distillation produces valid reports
8. Gate 2.5 rejects models without required credibility context
9. Gate 3 cross-paradigm ranking produces qualified comparisons
10. Benchmark Suite B assertions pass

---

## Open Questions Surfaced (PRD SS10)

These affect Phase 2 implementation but are NOT decided -- surface them, don't invent answers:

1. **Q2: Cross-paradigm NLPD comparability protocol** -- The plan implements simulation-based metrics per PRD v0.3 decision. However, the exact VPC concordance computation (which percentiles, binning strategy) needs pharmacometrician sign-off before Suite B assertions are finalized.

2. **Q3: DSL extensibility process** -- Affects Task 11 (Suite A7 NODE absorption). New NODE constraint templates for Suite B scenarios may need the formal specification process. Stan codegen is deferred to a separate plan.

3. **Q4: Covariate missingness strategy** -- Affects Task 4 (NODE trainer) if covariates are NODE inputs. Current plan uses complete-case for NODE covariate inputs; full-information likelihood is deferred.
