# SPDX-License-Identifier: GPL-2.0-or-later
"""Laplace/MVN approximate posterior-predictive draws for MLE backends.

Gate 3 commensurability (PRD §4.3.1) requires the Bayesian and MLE paths to
produce comparable uncertainty tuples. The Bayesian path draws from
``posterior_draws.parquet`` directly; the MLE path does not have posterior
samples, so this module synthesises them by sampling from the Laplace
approximation to the posterior — a multivariate Normal centred at the MLE
point estimate with the asymptotic covariance matrix returned by the
backend's Hessian-based standard-error routine.

References:
  MacKay 2003, *Information Theory, Inference, and Learning Algorithms*
    — Chapter 27 on the Laplace method.
  Gelman et al. 2013, *Bayesian Data Analysis*, 3rd ed.
    — §4.1 on asymptotic normality of the posterior.

**Not a substitute for MCMC.** The Laplace approximation can be terrible in
the presence of strong posterior skewness, multimodality, or boundary
constraints. Gate 3 ranking uses it purely to produce a commensurate
interval on the MLE side of the ledger; the Bayesian path's full
posterior should still be preferred when both are available.

The primary path uses ``numpy.random.multivariate_normal`` which falls
back to a pseudoinverse of ``cov`` if the Cholesky factorisation fails.
A caller-visible fallback then triggers when:

* ``numpy.linalg.LinAlgError`` is raised during sampling.
* The condition number of ``cov`` exceeds ``1e12`` (near-singular Hessian
  — common when a parameter was weakly identified or pegged at a bound).

The supported fallback is ``"empirical_bootstrap"``: draws are returned by
re-centring a pool of Normal(0, diag(cov)) samples around ``theta``. This
is not a real bootstrap — there is no resampled dataset — but it gives
downstream consumers a defensible pseudo-sample with the right marginal
scale while signalling that the correlation structure was lost. Callers
that need a more faithful fallback should refit with profile likelihood
or switch to the Bayesian backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LaplaceDrawsResult(NamedTuple):
    """Approximate-posterior draws plus a tag for the path that produced them.

    Callers that need to populate ``MetricTuple.method`` (Gate 3 ranker
    on the MLE side) should consume the named ``method`` field rather
    than inferring the path from the draws themselves — the diagonal
    bootstrap fallback is statistically much weaker than the full MVN
    Laplace approximation, and the distinction must survive round-trip
    serialization.
    """

    draws: NDArray[np.float64]
    method: DrawMethod


_COND_NUMBER_THRESHOLD = 1e12
# Variance floor for the diagonal-bootstrap fallback. Engaged only when
# the Hessian-derived diagonal is itself NaN/inf/zero — small enough
# that downstream credibility checks see "near-zero variance" and flag
# the fit, rather than receiving a plausible-looking sample.
_BOOTSTRAP_VARIANCE_FLOOR = 1e-6

FallbackStrategy = Literal["empirical_bootstrap", "raise"]
DrawMethod = Literal["laplace_mvn", "laplace_bootstrap_diagonal"]


def _cov_is_ill_conditioned(cov: NDArray[np.float64]) -> bool:
    """Flag covariance matrices that will choke ``multivariate_normal``.

    The cheap checks — not positive-semi-definite by symmetry, or any
    non-finite entry — run before the more expensive ``numpy.linalg.cond``
    call, which itself goes through SVD.
    """
    if not np.all(np.isfinite(cov)):
        return True
    if cov.shape[0] != cov.shape[1]:
        return True
    if not np.allclose(cov, cov.T, rtol=1e-8, atol=1e-10):
        return True
    try:
        cond = float(np.linalg.cond(cov))
    except np.linalg.LinAlgError:
        return True
    return cond > _COND_NUMBER_THRESHOLD


def _empirical_bootstrap_draws(
    theta: NDArray[np.float64],
    cov: NDArray[np.float64],
    n_draws: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Degenerate fallback: diagonal Normal around ``theta``.

    Preserves marginal standard errors but drops off-diagonal correlation
    structure. Callers that need correlations must either get a
    well-conditioned Hessian or switch to MCMC.

    The variance floor (``_BOOTSTRAP_VARIANCE_FLOOR``) only activates
    when the Hessian-derived diagonal is itself non-finite or zero —
    pathological inputs that would otherwise produce NaN draws. When
    it activates, the resulting marginal SE is intentionally tiny so a
    downstream credibility check flags the fit, rather than letting a
    silent zero-variance sample masquerade as a real estimate.
    """
    raw_diag = np.abs(np.diag(cov))
    # Use ``np.errstate`` to silence the "invalid value encountered in
    # greater" warning that NaN entries would otherwise raise; we
    # explicitly route NaN/inf entries to the floor below.
    with np.errstate(invalid="ignore"):
        clean = np.isfinite(raw_diag) & (raw_diag > 0)
    diag = np.where(clean, raw_diag, _BOOTSTRAP_VARIANCE_FLOOR)
    std = np.sqrt(diag)
    noise = rng.standard_normal(size=(n_draws, theta.shape[0]))
    # ``std`` is 1-D and broadcasts naturally over the trailing axis of
    # ``noise`` — no extra ``[np.newaxis, :]`` reshape needed.
    return theta + noise * std


def laplace_draws_with_method(
    theta: NDArray[np.float64],
    cov: NDArray[np.float64],
    n_draws: int,
    seed: int,
    *,
    fallback: FallbackStrategy = "empirical_bootstrap",
) -> LaplaceDrawsResult:
    """Method-aware variant of :func:`laplace_draws`.

    Returns a :class:`LaplaceDrawsResult` tagging which path produced
    the samples (``laplace_mvn`` for the full MVN draw, or
    ``laplace_bootstrap_diagonal`` for the degraded fallback). Callers
    that need to record provenance on ``MetricTuple.method`` should
    consume this entry point; the array-only :func:`laplace_draws`
    wrapper is kept for back-compat with sites that don't care.
    """
    if n_draws < 1:
        msg = f"n_draws must be >= 1, got {n_draws}"
        raise ValueError(msg)
    theta = np.asarray(theta, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    if theta.ndim != 1:
        msg = f"theta must be 1-D, got shape {theta.shape}"
        raise ValueError(msg)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        msg = f"cov must be 2-D square, got shape {cov.shape}"
        raise ValueError(msg)
    if cov.shape[0] != theta.shape[0]:
        msg = f"cov shape {cov.shape} does not match theta length {theta.shape[0]}"
        raise ValueError(msg)
    if not np.all(np.isfinite(theta)):
        msg = "theta contains non-finite entries"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    if _cov_is_ill_conditioned(cov):
        if fallback == "raise":
            msg = (
                f"covariance matrix is ill-conditioned "
                f"(cond > {_COND_NUMBER_THRESHOLD:g}); refusing to sample"
            )
            raise np.linalg.LinAlgError(msg)
        return LaplaceDrawsResult(
            draws=_empirical_bootstrap_draws(theta, cov, n_draws, rng),
            method="laplace_bootstrap_diagonal",
        )

    try:
        # ``method="svd"`` tolerates the small negative eigenvalues that
        # numerical Hessian inversion sometimes produces by zero-clipping.
        # Keeping it explicit is preferable to NumPy's default ``"cholesky"``
        # which would raise on those cases and force the fallback path.
        draws = rng.multivariate_normal(mean=theta, cov=cov, size=n_draws, method="svd")
    except np.linalg.LinAlgError:
        if fallback == "raise":
            raise
        return LaplaceDrawsResult(
            draws=_empirical_bootstrap_draws(theta, cov, n_draws, rng),
            method="laplace_bootstrap_diagonal",
        )
    return LaplaceDrawsResult(
        draws=draws.astype(np.float64, copy=False),
        method="laplace_mvn",
    )


def laplace_draws(
    theta: NDArray[np.float64],
    cov: NDArray[np.float64],
    n_draws: int,
    seed: int,
    *,
    fallback: FallbackStrategy = "empirical_bootstrap",
) -> NDArray[np.float64]:
    """Draw approximate posterior samples from the Laplace approximation.

    Thin wrapper around :func:`laplace_draws_with_method` that returns
    only the draws array. Use the method-aware variant when the caller
    needs to populate ``MetricTuple.method`` / similar provenance.

    Args:
        theta: Point estimate (MLE) of length ``p``. Must be finite.
        cov: Asymptotic covariance matrix of shape ``(p, p)``. Should be
            symmetric PSD; ill-conditioned matrices trigger ``fallback``.
        n_draws: Number of samples to return.
        seed: RNG seed for reproducibility. Paired with ``seed_registry``
            so a downstream rerun can reproduce the exact draws.
        fallback: Behaviour when the covariance is ill-conditioned or
            ``multivariate_normal`` raises. ``"empirical_bootstrap"``
            returns diagonal-only draws (correlations lost);
            ``"raise"`` re-raises the underlying ``LinAlgError``.

    Returns:
        Array of shape ``(n_draws, p)``. Rows are independent.

    Raises:
        ValueError: ``n_draws < 1``, ``theta`` is not 1-D, ``cov`` is not
            2-D square, or ``theta``/``cov`` shapes disagree.
        numpy.linalg.LinAlgError: ``fallback == "raise"`` and sampling
            fails (no implicit fallback).
    """
    return laplace_draws_with_method(theta, cov, n_draws, seed, fallback=fallback).draws
