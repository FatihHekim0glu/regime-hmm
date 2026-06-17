"""Gaussian emission log-density with a covariance floor.

The per-state emission model is a multivariate Gaussian over the (scaled) feature
vector at each time step. This module computes the LOG emission density
``log p(x_t | state=k)`` for every observation and state, supporting both
``"diag"`` (independent features) and ``"full"`` covariance parameterizations. A
strictly positive covariance FLOOR is applied so a (near-)degenerate state — one
that collapses onto a handful of identical points during EM — cannot drive the
log-density to ``+inf`` or produce a singular covariance.

The kernel is pure numpy/scipy: no global RNG, no hmmlearn. ``hmmlearn`` is a
dev-only parity oracle and is NEVER imported here.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Literal

from regimehmm._typing import FloatArray

#: Supported covariance parameterizations for the Gaussian emissions.
CovarianceType = Literal["diag", "full"]


def floor_covariance(
    covariances: FloatArray,
    *,
    covariance_type: CovarianceType = "diag",
    floor: float = 1e-6,
) -> FloatArray:
    r"""Apply a strictly positive variance floor to per-state covariances.

    For ``covariance_type="diag"`` the input is a ``(n_states, n_features)`` array
    of per-feature variances; each entry is raised to at least ``floor``. For
    ``covariance_type="full"`` the input is a ``(n_states, n_features, n_features)``
    stack; the floor is added to the diagonal (equivalently, eigenvalues are
    clipped up to ``floor``) so each matrix is symmetric positive-definite.

    DEGENERACY GUARD: this is the single chokepoint that keeps EM from emitting a
    singular or zero-variance state. A state that collapses onto identical points
    would otherwise send the Gaussian log-density to ``+inf``; the floor makes the
    emission model well-defined for every state on every iteration.

    Parameters
    ----------
    covariances:
        Per-state variances (diag) or covariance matrices (full).
    covariance_type:
        ``"diag"`` or ``"full"``.
    floor:
        The strictly positive minimum variance / minimum eigenvalue.

    Returns
    -------
    FloatArray
        The floored covariances, same shape as the input.

    Raises
    ------
    ValidationError
        If ``floor <= 0`` or ``covariances`` has the wrong rank for
        ``covariance_type``.
    """
    raise NotImplementedError


def gaussian_log_density(
    observations: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    *,
    covariance_type: CovarianceType = "diag",
    floor: float = 1e-6,
) -> FloatArray:
    r"""Per-state Gaussian log emission density for every observation.

    Returns the ``(n_obs, n_states)`` matrix

    .. math::

        \log p(x_t \mid \text{state}=k) =
            -\tfrac{1}{2}\big[d\log(2\pi) + \log\det\Sigma_k
            + (x_t - \mu_k)^\top \Sigma_k^{-1} (x_t - \mu_k)\big],

    where :math:`d` is the feature dimension. The covariance is floored via
    :func:`floor_covariance` before use, so the log-determinant and quadratic form
    are always finite. For ``covariance_type="diag"`` the quadratic form and
    log-determinant reduce to per-feature sums (no matrix inverse needed).

    PARITY REQUIREMENT: validated against ``hmmlearn``'s ``_compute_log_likelihood``
    to ``1e-6`` on seeded 2/3-state data in the parity suite.

    Parameters
    ----------
    observations:
        The ``(n_obs, n_features)`` (already scaled) feature matrix.
    means:
        The ``(n_states, n_features)`` per-state mean vectors.
    covariances:
        Per-state variances (diag) or covariance matrices (full).
    covariance_type:
        ``"diag"`` or ``"full"``.
    floor:
        Variance floor passed through to :func:`floor_covariance`.

    Returns
    -------
    FloatArray
        The ``(n_obs, n_states)`` log-density matrix.

    Raises
    ------
    ValidationError
        If shapes are inconsistent across the three arrays.
    """
    raise NotImplementedError
