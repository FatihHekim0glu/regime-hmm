"""Baum-Welch EM with multiple seeded restarts.

Fits a Gaussian HMM by the Expectation-Maximization (Baum-Welch) algorithm: the
E-step is the log-space forward-backward pass (:mod:`regimehmm.hmm.forward_backward`),
the M-step re-estimates ``pi``, ``A``, the per-state means, and the (floored)
covariances from the smoothed sufficient statistics. To dodge EM's local optima we
run several restarts from different seeded initializations and keep the highest
log-likelihood fit. Restart seeds come from reproducible PCG64 substreams
(:func:`regimehmm._rng.spawn_substreams`), so a fixed master seed reproduces the
fit exactly.

CORRECTNESS INVARIANT: within a single restart the per-iteration log-likelihood is
non-decreasing (Baum-Welch is monotone). This is asserted at fit time and pinned by
a unit test.

Importing this module has no side effects.
"""

from __future__ import annotations

from regimehmm._typing import FloatArray
from regimehmm.hmm.filter import HMMModel
from regimehmm.hmm.kernel import CovarianceType


def fit_hmm(
    observations: FloatArray,
    n_states: int,
    *,
    covariance_type: CovarianceType = "diag",
    n_restarts: int = 8,
    max_iter: int = 200,
    tol: float = 1e-4,
    covariance_floor: float = 1e-6,
    seed: int = 7,
) -> HMMModel:
    r"""Fit a Gaussian HMM by Baum-Welch EM with seeded restarts.

    Runs ``n_restarts`` independent EM optimizations from distinct seeded
    initializations and returns the :class:`~regimehmm.hmm.filter.HMMModel` with
    the highest training log-likelihood. Each restart alternates:

    * **E-step** — :func:`regimehmm.hmm.forward_backward.forward_backward` to get
      the smoothed posteriors ``gamma`` and pair-marginals ``xi``;
    * **M-step** — re-estimate ``pi = gamma_1``, ``A`` from summed ``xi``, the
      per-state means as ``gamma``-weighted observation averages, and the
      covariances as ``gamma``-weighted second moments, FLOORED via
      :func:`regimehmm.hmm.kernel.floor_covariance`.

    A restart stops when the log-likelihood gain falls below ``tol`` or after
    ``max_iter`` iterations.

    LEAKAGE NOTE: ``fit_hmm`` must be called on the TRAIN fold only. The fitted
    model is then consumed by the ONLINE filter (:mod:`regimehmm.hmm.filter`) to
    label the out-of-sample window — never by the smoothed/Viterbi decoders.

    REPRODUCIBILITY: restart initializations draw from
    :func:`regimehmm._rng.spawn_substreams(seed, n_restarts)`, so a fixed ``seed``
    reproduces the fit byte-for-byte.

    MONOTONICITY INVARIANT: within each restart the per-iteration log-likelihood is
    non-decreasing; a violation beyond floating tolerance raises (unit-tested).

    Parameters
    ----------
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix (scaler fit train-only).
    n_states:
        The number of hidden regimes ``K`` (``>= 1``).
    covariance_type:
        ``"diag"`` or ``"full"`` emission covariance.
    n_restarts:
        Number of seeded EM restarts (best LL wins).
    max_iter:
        Maximum EM iterations per restart.
    tol:
        Convergence tolerance on the log-likelihood increment.
    covariance_floor:
        Strictly positive variance floor passed to the M-step.
    seed:
        Master seed for the restart substreams.

    Returns
    -------
    HMMModel
        The best (highest training log-likelihood) fitted model.

    Raises
    ------
    ValidationError
        If ``n_states < 1``, ``n_restarts < 1``, ``max_iter < 1``, ``tol <= 0``,
        ``covariance_floor <= 0``, or ``observations`` is not 2-D.
    InsufficientDataError
        If there are fewer observations than states.
    """
    raise NotImplementedError


def em_single_run(
    observations: FloatArray,
    startprob: FloatArray,
    transmat: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    *,
    covariance_type: CovarianceType = "diag",
    max_iter: int = 200,
    tol: float = 1e-4,
    covariance_floor: float = 1e-6,
) -> tuple[HMMModel, list[float]]:
    r"""Run a SINGLE EM optimization from a given initialization.

    The deterministic inner loop used by :func:`fit_hmm` (one restart). Returns the
    converged model together with the per-iteration log-likelihood trace, so the
    monotonicity invariant can be asserted by the caller and inspected in tests.

    Parameters
    ----------
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix.
    startprob, transmat, means, covariances:
        The initial parameter bundle for this restart.
    covariance_type:
        ``"diag"`` or ``"full"``.
    max_iter:
        Maximum EM iterations.
    tol:
        Convergence tolerance on the log-likelihood increment.
    covariance_floor:
        Strictly positive variance floor for the M-step.

    Returns
    -------
    tuple[HMMModel, list[float]]
        The fitted model and the (non-decreasing) per-iteration log-likelihood
        trace.

    Raises
    ------
    ValidationError
        If shapes are inconsistent or scalar parameters are out of range.
    """
    raise NotImplementedError
