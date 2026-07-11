# SPDX-License-Identifier: GPL-2.0-or-later
"""Acceptance gates for the mixed-effects NODE (native reparameterized VI).

These tests are the merge gate for roadmap item A. They validate that random
effects act multiplicatively on the NODE input-layer weights (Bräm et al. 2024,
doi:10.1007/s10928-023-09886-4: ``W_i = W_pop * exp(eta_i)``) and that the
native reparameterized ELBO (Janssen et al. 2024, doi:10.1007/s10928-024-09931-w)
recovers real between-subject variability and improves on the pooled no-IIV fit.

The five gates:

1. No-IIV parity — :func:`train_node` is structurally unchanged.
2. Identity RE — ``apply_subject_re(0)`` is bit-identical to the base model.
3. Log-multiplicative positivity — RE weights equal ``W_pop * exp(eta)``.
4. Synthetic parameter recovery — VI recovers a non-collapsed ``omega`` and
   beats the pooled fit on a dataset with genuine elimination BSV.
5. JIT/gradient smoke — the ELBO and its gradient are finite through Diffrax.
"""

from __future__ import annotations

import math

import jax  # type: ignore[import-untyped]
import jax.numpy as jnp  # type: ignore[import-untyped]
import pytest

from apmode.backends.node_ode import HybridPKODE, ODEConfig
from apmode.backends.node_trainer import (
    _SCALE_FLOOR,
    TrainingConfig,
    _subject_nll,
    predict_subject_conc,
    train_node,
    train_node_vi,
)

_HIDDEN_DIM = 3


def _make_model(seed: int = 0) -> HybridPKODE:
    """1-cmt oral hybrid with NODE elimination (the item-A reference model)."""
    return HybridPKODE(
        config=ODEConfig(
            n_cmt=1,
            node_position="elimination",
            constraint_template="bounded_positive",
            node_dim=_HIDDEN_DIM,
            mechanistic_params={"ka": 1.0, "V": 30.0},
        ),
        key=jax.random.PRNGKey(seed),
    )


def _analytic_oral_conc(
    times: jax.Array, ka: float, ke: float, v: float, dose: float
) -> jax.Array:
    """1-cmt first-order oral absorption analytic concentration."""
    return (dose * ka) / (v * (ka - ke)) * (jnp.exp(-ke * times) - jnp.exp(-ka * times))


def _make_mixed_effects_subjects(
    n_subjects: int,
    n_obs: int,
    *,
    ke_pop: float = 0.15,
    omega_true: float = 0.35,
    prop_noise: float = 0.03,
    seed: int = 7,
) -> list[dict[str, jax.Array]]:
    """Synthetic 1-cmt oral PK with a log-normal RE on the elimination rate.

    Each subject draws ``ke_i = ke_pop * exp(eta_i)``, ``eta_i ~ N(0, omega^2)``,
    inducing real between-subject spread in the elimination sub-function. Noise
    is kept small (proportional) so the BSV — not residual error — dominates,
    making per-subject random effects genuinely beneficial.
    """
    key = jax.random.PRNGKey(seed)
    ka = 1.0
    v = 30.0
    dose = 100.0
    times = jnp.linspace(0.5, 24.0, n_obs)
    subjects: list[dict[str, jax.Array]] = []
    for i in range(n_subjects):
        key, eta_key, noise_key = jax.random.split(key, 3)
        eta = omega_true * float(jax.random.normal(eta_key))
        ke_i = ke_pop * math.exp(eta)
        true_conc = jnp.maximum(_analytic_oral_conc(times, ka, ke_i, v, dose), 1e-3)
        noise = prop_noise * true_conc * jax.random.normal(noise_key, shape=times.shape)
        obs = jnp.maximum(true_conc + noise, 1e-3)
        subjects.append(
            {
                "subject_id": f"S{i:02d}",
                "times": times,
                "observations": obs,
                "y0": jnp.array([dose, 0.0]),
                "obs_cmt": jnp.array(1),
            }
        )
    return subjects


class TestNoIIVParity:
    """Gate 1: the pooled path is unchanged and reports no random effects."""

    def test_train_node_reports_no_random_effects(self) -> None:
        model = _make_model()
        subjects = _make_mixed_effects_subjects(n_subjects=4, n_obs=6)
        result = train_node(model, subjects, TrainingConfig(epochs=5))

        assert result.random_effects is False
        assert result.omega is None
        assert result.subject_re_means is None
        assert result.eta_shrinkage is None

    def test_population_nll_finite(self) -> None:
        from apmode.backends.node_trainer import _population_nll

        model = _make_model()
        subjects = _make_mixed_effects_subjects(n_subjects=4, n_obs=6)
        log_sigma = jnp.log(jnp.array(0.3))
        nll = _population_nll(model, log_sigma, subjects)
        assert jnp.isfinite(nll)


class TestIdentityRandomEffect:
    """Gate 2: eta=0 collapses to the population model, bit-for-bit."""

    def test_zero_re_predictions_bit_identical(self) -> None:
        model = _make_model(seed=1)
        subject = _make_mixed_effects_subjects(n_subjects=1, n_obs=8)[0]

        base_pred = predict_subject_conc(model, subject)
        re_model = model.apply_subject_re(jnp.zeros(_HIDDEN_DIM))
        re_pred = predict_subject_conc(re_model, subject)

        # atol=0, rtol=0 — exp(0)=1.0 exactly, so weights and PRED are identical.
        assert jnp.array_equal(base_pred, re_pred)
        assert jnp.allclose(base_pred, re_pred, atol=0.0, rtol=0.0)

    def test_zero_re_weights_bit_identical(self) -> None:
        model = _make_model(seed=1)
        re_model = model.apply_subject_re(jnp.zeros(_HIDDEN_DIM))
        assert jnp.array_equal(model.node.linear1.weight, re_model.node.linear1.weight)


class TestLogMultiplicativePositivity:
    """Gate 3: RE is W_pop * exp(eta), sign-preserving and multiplicative."""

    def test_re_weights_are_log_multiplicative(self) -> None:
        model = _make_model(seed=2)
        eta = jnp.array([0.3, -0.4, 0.15])
        re_model = model.apply_subject_re(eta)

        old_w = model.node.linear1.weight  # (hidden_dim, input_dim)
        new_w = re_model.node.linear1.weight
        expected = old_w * jnp.exp(eta)[:, None]

        assert jnp.allclose(new_w, expected, atol=0.0, rtol=0.0)
        # Multiplicative log-scale => every weight keeps its sign.
        assert jnp.all(jnp.sign(new_w) == jnp.sign(old_w))
        # Nonzero eta genuinely moves the weights.
        assert not jnp.allclose(new_w, old_w)


class TestGradientSmoke:
    """Gate 5: ELBO value and gradient are finite through the Diffrax solve."""

    def test_elbo_and_grad_finite(self) -> None:
        model = _make_model(seed=3)
        subjects = _make_mixed_effects_subjects(n_subjects=3, n_obs=5)
        n_subj = len(subjects)
        re_dim = _HIDDEN_DIM

        log_sigma = jnp.log(jnp.array(0.3))
        mu = jnp.zeros((n_subj, re_dim))
        raw_s = jnp.full((n_subj, re_dim), float(jnp.log(jnp.expm1(0.07))))
        raw_omega = jnp.full((re_dim,), float(jnp.log(jnp.expm1(0.10))))
        params = (model, log_sigma, mu, raw_s, raw_omega)

        eps = jax.random.normal(jax.random.PRNGKey(0), (n_subj, re_dim))

        def elbo(
            p: tuple[HybridPKODE, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            m, ls, mu_v, raw_s_v, raw_omega_v = p
            sigma = jnp.exp(ls)
            s = jax.nn.softplus(raw_s_v) + _SCALE_FLOOR
            omega = jax.nn.softplus(raw_omega_v) + _SCALE_FLOOR
            omega_sq = omega**2
            total = jnp.array(0.0)
            for i, subj in enumerate(subjects):
                eta = mu_v[i] + s[i] * eps[i]
                model_i = m.apply_subject_re(eta)
                nll_i = _subject_nll(model_i, sigma, subj)
                s_i_sq = s[i] ** 2
                kl_i = 0.5 * jnp.sum(
                    jnp.log(omega_sq / s_i_sq) + (s_i_sq + mu_v[i] ** 2) / omega_sq - 1.0
                )
                total = total + nll_i + kl_i
            return total

        import equinox as eqx

        loss, grads = eqx.filter_value_and_grad(elbo)(params)
        assert jnp.isfinite(loss)
        # Every array leaf of the gradient pytree is finite.
        leaves = [g for g in jax.tree_util.tree_leaves(grads) if eqx.is_array(g)]
        assert leaves
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


@pytest.mark.slow
class TestSyntheticRecovery:
    """Gate 4: VI recovers non-collapsed omega and beats the pooled fit."""

    def test_vi_recovers_bsv_and_beats_pooled(self) -> None:
        model = _make_model(seed=0)
        subjects = _make_mixed_effects_subjects(
            n_subjects=16, n_obs=8, ke_pop=0.15, omega_true=0.35, seed=11
        )

        # Fair comparison: the pooled fit is the VI warm start, so VI must
        # descend *below* the same pooled endpoint to win.
        pooled_cfg = TrainingConfig(epochs=300, learning_rate=3e-3, early_stop_patience=400)
        pooled = train_node(model, subjects, pooled_cfg)

        vi_cfg = TrainingConfig(epochs=300, learning_rate=3e-3, early_stop_patience=400)
        vi = train_node_vi(model, subjects, vi_cfg, init_result=pooled, n_samples=1, seed=0)

        # (a) Ran and returned finite loss.
        assert vi.random_effects is True
        assert math.isfinite(vi.final_loss)
        assert math.isfinite(pooled.final_loss)

        # (b) omega present, per-dim positive/finite, not all collapsed to floor;
        #     at least one dim carries plausible-magnitude variability.
        assert vi.omega is not None
        assert len(vi.omega) == _HIDDEN_DIM
        assert all(math.isfinite(o) and o > 0.0 for o in vi.omega)
        floor_tol = _SCALE_FLOOR * 10.0
        assert any(o > floor_tol for o in vi.omega), f"all omega collapsed: {vi.omega}"
        assert any(0.01 <= o <= 2.0 for o in vi.omega), f"no plausible omega: {vi.omega}"

        # (c) VI ELBO strictly better (lower) than the pooled NLL on same data.
        assert vi.final_loss < pooled.final_loss, (
            f"VI did not beat pooled: vi={vi.final_loss:.4f} pooled={pooled.final_loss:.4f}"
        )

        # (d) One posterior-mean entry per subject, each of length re_dim.
        assert vi.subject_re_means is not None
        assert set(vi.subject_re_means) == {s["subject_id"] for s in subjects}
        assert all(len(v) == _HIDDEN_DIM for v in vi.subject_re_means.values())
