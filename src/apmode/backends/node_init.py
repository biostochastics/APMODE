# SPDX-License-Identifier: GPL-2.0-or-later
"""NODE initial estimate strategy (ARCHITECTURE.md §2.5, PRD §4.2.0.1).

Two strategies for faster NODE convergence:

1. **Pre-trained weight library**: Reference MLP weights that approximate
   classical PK sub-functions (first-order absorption, linear elimination,
   Michaelis-Menten elimination) for 1-cmt and 2-cmt models. These are
   generated offline by fitting the NODE sub-model to known analytical
   rate laws over a standardised concentration/time grid.

2. **Transfer learning from classical fit**: When a classical backend has
   already converged, its best-fit structural parameters (ka, CL, V, etc.)
   warm-start the mechanistic parameters of HybridPKODE and shape the NODE
   input-layer weights via a linear scaling heuristic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from apmode.backends.node_constraints import TEMPLATE_MAX_DIM
from apmode.backends.node_model import NODESubModel
from apmode.backends.node_ode import HybridPKODE, ODEConfig

# ---------------------------------------------------------------------------
# Pre-trained weight library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceProfile:
    """Defines a classical rate-law that the NODE sub-model should approximate.

    The NODE is trained to reproduce `target_fn(concentration, time)` over
    a standardised grid so that the resulting MLP weights embed the shape
    of the known analytical solution.
    """

    name: str
    node_position: Literal["absorption", "elimination"]
    n_cmt: Literal[1, 2]
    target_params: dict[str, float]


# Canonical reference profiles — cover the most common PK sub-functions.
REFERENCE_PROFILES: dict[str, ReferenceProfile] = {
    "1cmt_firstorder_abs": ReferenceProfile(
        name="1cmt_firstorder_abs",
        node_position="absorption",
        n_cmt=1,
        target_params={"ka": 1.0},
    ),
    "1cmt_linear_elim": ReferenceProfile(
        name="1cmt_linear_elim",
        node_position="elimination",
        n_cmt=1,
        target_params={"CL": 2.0, "V": 30.0},
    ),
    "1cmt_mm_elim": ReferenceProfile(
        name="1cmt_mm_elim",
        node_position="elimination",
        n_cmt=1,
        target_params={"Vmax": 10.0, "Km": 5.0},
    ),
    "2cmt_linear_elim": ReferenceProfile(
        name="2cmt_linear_elim",
        node_position="elimination",
        n_cmt=2,
        target_params={"CL": 3.0, "V": 40.0},
    ),
    "2cmt_firstorder_abs": ReferenceProfile(
        name="2cmt_firstorder_abs",
        node_position="absorption",
        n_cmt=2,
        target_params={"ka": 1.5},
    ),
}


def _target_fn_value(
    profile: ReferenceProfile,
    conc: float,
    time: float,
) -> float:
    """Evaluate the classical rate-law that the NODE should reproduce.

    For absorption: rate = ka (constant, independent of concentration).
    For linear elimination: rate = CL / V (concentration-independent clearance rate).
    For MM elimination: rate = Vmax * conc / (Km + conc).
    """
    p = profile.target_params
    if profile.node_position == "absorption":
        return p.get("ka", 1.0)
    # Elimination
    if "Vmax" in p and "Km" in p:
        # Michaelis-Menten: rate varies with concentration
        return p["Vmax"] * conc / (p["Km"] + conc)
    # Linear elimination: constant rate constant ke = CL/V
    cl = p.get("CL", 2.0)
    v = p.get("V", 30.0)
    return cl / v


def _build_training_grid(
    n_conc: int = 30,
    n_time: int = 10,
    conc_range: tuple[float, float] = (0.1, 100.0),
    time_range: tuple[float, float] = (0.1, 24.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Build a standardised (concentration, time) grid for fitting.

    Returns two 1-D arrays of shape (n_conc * n_time,) for concentration
    and time respectively.
    """
    concs = np.linspace(conc_range[0], conc_range[1], n_conc)
    times = np.linspace(time_range[0], time_range[1], n_time)
    cc, tt = np.meshgrid(concs, times, indexing="ij")
    return cc.ravel(), tt.ravel()


def pretrain_node_weights(
    profile: ReferenceProfile,
    constraint_template: str = "bounded_positive",
    hidden_dim: int = 3,
    *,
    epochs: int = 200,
    learning_rate: float = 5e-3,
    seed: int = 753849,
) -> NODESubModel:
    """Pre-train a NODE sub-model to approximate a classical rate law.

    Fits the MLP weights so that `node([conc, time]) ≈ target_fn(conc, time)`
    over a standardised grid using MSE loss + Adam optimiser.

    Args:
        profile: Reference profile defining the target rate law.
        constraint_template: Constraint template for the NODE sub-model.
        hidden_dim: Hidden dimension of the MLP.
        epochs: Number of training epochs.
        learning_rate: Adam learning rate.
        seed: PRNG seed.

    Returns:
        Pre-trained NODESubModel.
    """
    max_dim = TEMPLATE_MAX_DIM.get(constraint_template)
    if max_dim is not None and hidden_dim > max_dim:
        msg = f"hidden_dim={hidden_dim} exceeds max={max_dim} for '{constraint_template}'"
        raise ValueError(msg)

    key = jax.random.PRNGKey(seed)
    model = NODESubModel(
        input_dim=2,
        hidden_dim=hidden_dim,
        constraint_template=constraint_template,
        key=key,
    )

    # Build training data
    concs, times = _build_training_grid()
    targets = np.array(
        [_target_fn_value(profile, float(c), float(t)) for c, t in zip(concs, times, strict=True)]
    )

    x_data = jnp.array(np.column_stack([concs, times]), dtype=jnp.float32)
    y_data = jnp.array(targets, dtype=jnp.float32)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(
        m: NODESubModel, state: optax.OptState
    ) -> tuple[NODESubModel, optax.OptState, jax.Array]:
        def loss_fn(m: NODESubModel) -> jax.Array:
            preds = jax.vmap(m)(x_data)  # shape: (N, 1)
            return jnp.mean((preds.squeeze() - y_data) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
        updates, new_state = optimizer.update(
            eqx.filter(grads, eqx.is_array), state, eqx.filter(m, eqx.is_array)
        )
        new_m = eqx.apply_updates(m, updates)
        return new_m, new_state, loss

    for _ in range(epochs):
        model, opt_state, _loss = step(model, opt_state)

    return model


# ---------------------------------------------------------------------------
# Pre-trained weight cache (lazy singleton)
# ---------------------------------------------------------------------------


@dataclass
class WeightLibrary:
    """Cache of pre-trained NODE sub-models keyed by profile name + config.

    Thread-safe for concurrent async NODE runs.
    Use reset() to clear the cache for test isolation.
    """

    _cache: dict[str, NODESubModel] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def reset(self) -> None:
        """Clear all cached pre-trained models."""
        with self._lock:
            self._cache.clear()

    def get(
        self,
        profile_name: str,
        constraint_template: str = "bounded_positive",
        hidden_dim: int = 3,
        *,
        seed: int = 753849,
    ) -> NODESubModel | None:
        """Retrieve or generate pre-trained weights for a reference profile.

        Returns None if the profile name is unknown.
        """
        if profile_name not in REFERENCE_PROFILES:
            return None

        cache_key = f"{profile_name}:{constraint_template}:{hidden_dim}:{seed}"
        with self._lock:
            if cache_key not in self._cache:
                profile = REFERENCE_PROFILES[profile_name]
                self._cache[cache_key] = pretrain_node_weights(
                    profile,
                    constraint_template=constraint_template,
                    hidden_dim=hidden_dim,
                    seed=seed,
                )
            return self._cache[cache_key]


# Module-level singleton
_weight_library = WeightLibrary()


def get_weight_library() -> WeightLibrary:
    """Return the module-level WeightLibrary singleton."""
    return _weight_library


def reset_weight_library() -> None:
    """Clear the module-level WeightLibrary cache.

    Call between test runs or pipeline runs to ensure reproducibility.
    """
    _weight_library.reset()


# ---------------------------------------------------------------------------
# Transfer learning from classical backend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferResult:
    """Result of transfer learning initialisation."""

    model: HybridPKODE
    source: str  # "pretrained", "classical_transfer", "random"
    profile_name: str | None
    transferred_params: list[str]


def select_reference_profile(
    ode_config: ODEConfig,
) -> str | None:
    """Select the best matching reference profile for the given ODE config.

    Matches on (n_cmt, node_position). Returns the profile name or None.
    """
    candidates = [
        name
        for name, prof in REFERENCE_PROFILES.items()
        if prof.n_cmt == ode_config.n_cmt and prof.node_position == ode_config.node_position
    ]
    if not candidates:
        return None
    # Prefer linear_elim over mm_elim as default (more common in PK)
    for c in candidates:
        if "linear" in c or "firstorder" in c:
            return c
    return candidates[0]


def transfer_from_classical(
    ode_config: ODEConfig,
    classical_estimates: dict[str, float],
    key: jax.Array,
    *,
    use_pretrained: bool = True,
) -> TransferResult:
    """Initialise a HybridPKODE with transfer learning from classical fit.

    Strategy:
    1. Warm-start mechanistic params (ka, V, CL, etc.) from classical estimates.
    2. If a matching reference profile exists and `use_pretrained` is True,
       transplant pre-trained NODE weights into the HybridPKODE.
    3. Otherwise, fall back to random initialisation (standard Xavier).

    The classical estimates are injected into the ODEConfig.mechanistic_params
    so HybridPKODE.__init__ picks them up for log-space initialisation.

    Args:
        ode_config: ODE configuration (determines model structure).
        classical_estimates: Parameter estimates from the classical backend
            (e.g. {"ka": 1.2, "V": 35.0, "CL": 3.5}).
        key: JAX PRNG key.
        use_pretrained: Whether to use the pre-trained weight library.

    Returns:
        TransferResult with initialised model and provenance metadata.
    """
    # Step 1: merge classical estimates into mechanistic params
    mech = dict(ode_config.mechanistic_params or {})
    transferred: list[str] = []
    for param_name in ["ka", "V", "V1", "V2", "Q", "CL", "Vmax", "Km"]:
        if param_name in classical_estimates:
            mech[param_name] = classical_estimates[param_name]
            transferred.append(param_name)

    # Map V1 -> V if needed
    if "V1" in mech and "V" not in mech:
        mech["V"] = mech["V1"]

    warm_config = ODEConfig(
        n_cmt=ode_config.n_cmt,
        node_position=ode_config.node_position,
        constraint_template=ode_config.constraint_template,
        node_dim=ode_config.node_dim,
        mechanistic_params=mech,
    )

    # Step 2: build model with warm-started mechanistic params
    model = HybridPKODE(config=warm_config, key=key)

    # Step 3: transplant pre-trained NODE weights if available
    profile_name = select_reference_profile(ode_config) if use_pretrained else None

    if profile_name is not None:
        library = get_weight_library()
        pretrained_node = library.get(
            profile_name,
            constraint_template=ode_config.constraint_template,
            hidden_dim=ode_config.node_dim,
        )
        if pretrained_node is not None:
            # Replace the randomly-initialised NODE with pre-trained one
            model = eqx.tree_at(lambda m: m.node, model, pretrained_node)
            return TransferResult(
                model=model,
                source="pretrained",
                profile_name=profile_name,
                transferred_params=transferred,
            )

    source = "classical_transfer" if transferred else "random"
    return TransferResult(
        model=model,
        source=source,
        profile_name=None,
        transferred_params=transferred,
    )
