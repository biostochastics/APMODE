# SPDX-License-Identifier: GPL-2.0-or-later
"""Tests for NODE initial estimate strategy (node_init.py)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apmode.backends.node_constraints import TEMPLATE_MAX_DIM
from apmode.backends.node_init import (
    REFERENCE_PROFILES,
    TransferResult,
    WeightLibrary,
    _build_training_grid,
    _target_fn_value,
    get_weight_library,
    pretrain_node_weights,
    reset_weight_library,
    select_reference_profile,
    transfer_from_classical,
)
from apmode.backends.node_model import NODESubModel
from apmode.backends.node_ode import HybridPKODE, ODEConfig

# ---------------------------------------------------------------------------
# Reference profiles
# ---------------------------------------------------------------------------


class TestReferenceProfiles:
    """Validate the canonical reference profile definitions."""

    def test_all_profiles_have_required_fields(self) -> None:
        for name, prof in REFERENCE_PROFILES.items():
            assert prof.name == name
            assert prof.node_position in ("absorption", "elimination")
            assert prof.n_cmt in (1, 2)
            assert isinstance(prof.target_params, dict)

    def test_profile_count(self) -> None:
        assert len(REFERENCE_PROFILES) >= 5

    @pytest.mark.parametrize("profile_name", list(REFERENCE_PROFILES.keys()))
    def test_target_fn_is_positive(self, profile_name: str) -> None:
        prof = REFERENCE_PROFILES[profile_name]
        for conc in [0.1, 1.0, 10.0, 50.0]:
            val = _target_fn_value(prof, conc, 1.0)
            assert val > 0, f"target_fn({conc}, 1.0) = {val} for {profile_name}"


# ---------------------------------------------------------------------------
# Training grid
# ---------------------------------------------------------------------------


class TestTrainingGrid:
    def test_grid_shape(self) -> None:
        concs, times = _build_training_grid(n_conc=10, n_time=5)
        assert concs.shape == (50,)
        assert times.shape == (50,)

    def test_grid_range(self) -> None:
        concs, times = _build_training_grid(conc_range=(1.0, 100.0), time_range=(0.5, 12.0))
        assert concs.min() >= 1.0
        assert concs.max() <= 100.0
        assert times.min() >= 0.5
        assert times.max() <= 12.0


# ---------------------------------------------------------------------------
# Pre-training
# ---------------------------------------------------------------------------


class TestPretrainNodeWeights:
    def test_returns_node_submodel(self) -> None:
        prof = REFERENCE_PROFILES["1cmt_linear_elim"]
        model = pretrain_node_weights(prof, epochs=10, seed=0)
        assert isinstance(model, NODESubModel)
        assert model.input_dim == 2

    def test_pretrained_approximates_linear_elim(self) -> None:
        """Pre-trained weights should approximate CL/V = 2/30 ≈ 0.0667."""
        prof = REFERENCE_PROFILES["1cmt_linear_elim"]
        model = pretrain_node_weights(
            prof, constraint_template="bounded_positive", hidden_dim=3, epochs=300, seed=42
        )
        # Evaluate at several concentrations — should be roughly constant
        outputs = []
        for conc in [1.0, 10.0, 50.0]:
            out = float(model(jnp.array([conc, 1.0])).squeeze())
            outputs.append(out)
        # All outputs should be close to each other (constant rate)
        assert np.std(outputs) < 0.5 * np.mean(outputs), (
            f"Expected roughly constant output for linear elim, got {outputs}"
        )

    def test_pretrained_mm_varies_with_concentration(self) -> None:
        """MM elimination: rate increases then saturates with concentration."""
        prof = REFERENCE_PROFILES["1cmt_mm_elim"]
        model = pretrain_node_weights(
            prof, constraint_template="bounded_positive", hidden_dim=4, epochs=300, seed=42
        )
        out_low = float(model(jnp.array([0.5, 1.0])).squeeze())
        out_high = float(model(jnp.array([50.0, 1.0])).squeeze())
        # MM: output at high conc should be greater than at low conc
        assert out_high > out_low

    def test_dim_validation(self) -> None:
        prof = REFERENCE_PROFILES["1cmt_linear_elim"]
        max_dim = TEMPLATE_MAX_DIM["bounded_positive"]
        with pytest.raises(ValueError, match="exceeds max"):
            pretrain_node_weights(prof, hidden_dim=max_dim + 1)

    def test_deterministic_with_same_seed(self) -> None:
        prof = REFERENCE_PROFILES["1cmt_firstorder_abs"]
        m1 = pretrain_node_weights(prof, epochs=10, seed=123)
        m2 = pretrain_node_weights(prof, epochs=10, seed=123)
        w1 = jax.tree.leaves(m1)
        w2 = jax.tree.leaves(m2)
        for a, b in zip(w1, w2, strict=True):
            if hasattr(a, "shape"):
                np.testing.assert_array_equal(np.array(a), np.array(b))


# ---------------------------------------------------------------------------
# Weight library
# ---------------------------------------------------------------------------


class TestWeightLibrary:
    def test_get_known_profile(self) -> None:
        lib = WeightLibrary()
        model = lib.get("1cmt_linear_elim", hidden_dim=3, seed=0)
        assert model is not None
        assert isinstance(model, NODESubModel)

    def test_get_unknown_profile_returns_none(self) -> None:
        lib = WeightLibrary()
        assert lib.get("nonexistent_profile") is None

    def test_caching(self) -> None:
        lib = WeightLibrary()
        m1 = lib.get("1cmt_linear_elim", hidden_dim=3, seed=0)
        m2 = lib.get("1cmt_linear_elim", hidden_dim=3, seed=0)
        assert m1 is m2  # same object from cache

    def test_different_config_different_cache_entry(self) -> None:
        lib = WeightLibrary()
        m1 = lib.get("1cmt_linear_elim", hidden_dim=3, seed=0)
        m2 = lib.get("1cmt_linear_elim", hidden_dim=4, seed=0)
        assert m1 is not m2

    def test_module_singleton(self) -> None:
        lib = get_weight_library()
        assert isinstance(lib, WeightLibrary)

    def test_reset_clears_cache(self) -> None:
        lib = WeightLibrary()
        lib.get("1cmt_linear_elim", hidden_dim=3, seed=0)
        assert len(lib._cache) > 0
        lib.reset()
        assert len(lib._cache) == 0

    def test_reset_module_library(self) -> None:
        lib = get_weight_library()
        lib.get("1cmt_linear_elim", hidden_dim=3, seed=99)
        assert len(lib._cache) > 0
        reset_weight_library()
        assert len(lib._cache) == 0


# ---------------------------------------------------------------------------
# Reference profile selection
# ---------------------------------------------------------------------------


class TestSelectReferenceProfile:
    def test_1cmt_elimination(self) -> None:
        config = ODEConfig(n_cmt=1, node_position="elimination")
        name = select_reference_profile(config)
        assert name is not None
        assert "1cmt" in name and "elim" in name

    def test_2cmt_absorption(self) -> None:
        config = ODEConfig(n_cmt=2, node_position="absorption")
        name = select_reference_profile(config)
        assert name is not None
        assert "2cmt" in name and "abs" in name

    def test_prefers_linear_over_mm(self) -> None:
        config = ODEConfig(n_cmt=1, node_position="elimination")
        name = select_reference_profile(config)
        assert name is not None
        assert "linear" in name


# ---------------------------------------------------------------------------
# Transfer from classical
# ---------------------------------------------------------------------------


class TestTransferFromClassical:
    def test_transfer_warm_starts_mechanistic_params(self) -> None:
        config = ODEConfig(
            n_cmt=1,
            node_position="elimination",
            node_dim=3,
            mechanistic_params={"ka": 1.0, "V": 30.0},
        )
        key = jax.random.PRNGKey(0)
        result = transfer_from_classical(
            config,
            classical_estimates={"ka": 2.5, "V": 45.0, "CL": 5.0},
            key=key,
            use_pretrained=False,
        )
        assert isinstance(result, TransferResult)
        assert result.source == "classical_transfer"
        assert "ka" in result.transferred_params
        assert "V" in result.transferred_params
        assert "CL" in result.transferred_params

        # Check mechanistic params were warm-started
        assert abs(float(result.model.ka) - 2.5) < 0.01
        assert abs(float(result.model.V) - 45.0) < 0.1

    def test_transfer_with_pretrained(self) -> None:
        config = ODEConfig(
            n_cmt=1,
            node_position="elimination",
            constraint_template="bounded_positive",
            node_dim=3,
        )
        key = jax.random.PRNGKey(0)
        result = transfer_from_classical(
            config,
            classical_estimates={"ka": 1.5, "V": 35.0},
            key=key,
            use_pretrained=True,
        )
        assert result.source == "pretrained"
        assert result.profile_name is not None

    def test_transfer_no_classical_estimates(self) -> None:
        config = ODEConfig(n_cmt=1, node_position="elimination", node_dim=3)
        key = jax.random.PRNGKey(0)
        result = transfer_from_classical(
            config,
            classical_estimates={},
            key=key,
            use_pretrained=False,
        )
        assert result.source == "random"
        assert result.transferred_params == []

    def test_transfer_v1_to_v_mapping(self) -> None:
        config = ODEConfig(n_cmt=2, node_position="elimination", node_dim=3)
        key = jax.random.PRNGKey(0)
        result = transfer_from_classical(
            config,
            classical_estimates={"V1": 40.0, "V2": 50.0, "Q": 6.0},
            key=key,
            use_pretrained=False,
        )
        assert "V1" in result.transferred_params
        # V should have been set from V1
        assert abs(float(result.model.V) - 40.0) < 0.1

    def test_result_model_is_hybrid_pk_ode(self) -> None:
        config = ODEConfig(n_cmt=1, node_position="absorption", node_dim=3)
        key = jax.random.PRNGKey(0)
        result = transfer_from_classical(
            config,
            classical_estimates={"ka": 1.0},
            key=key,
        )
        assert isinstance(result.model, HybridPKODE)
        # Model should be functional (can solve)
        y0 = jnp.array([100.0, 0.0])
        times = jnp.array([0.5, 1.0, 2.0])
        sol = result.model.solve(y0, times)
        assert sol.shape == (3, 2)
