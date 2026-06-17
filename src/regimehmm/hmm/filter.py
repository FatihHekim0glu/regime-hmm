r"""Online forward filter — the ONLY tradable, no-lookahead regime posterior.

This module is the heart of the project's leakage discipline. The online filter
posterior at time ``t``,

.. math::

    p(\text{state}_t = k \mid x_1, \dots, x_t),

conditions on data UP TO AND INCLUDING ``t`` only — never the future. It is the
normalized forward message ``alpha_t`` and is the **only** regime signal that may
drive an out-of-sample trade. By contrast the smoothed (forward-backward)
posterior and the Viterbi path both use the WHOLE sample and therefore PEEK AHEAD;
they are in-sample EDA only and must NEVER be turned into an OOS label or signal.

The frozen :class:`HMMModel` produced by EM lives here (it is what the filter
consumes). Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regimehmm._typing import FloatArray
from regimehmm.hmm.kernel import CovarianceType


@dataclass(frozen=True, slots=True)
class HMMModel:
    """An immutable, fitted Gaussian HMM.

    The parameter bundle the online filter (and Viterbi/forward-backward) consume.
    Frozen and slotted so a fitted model is hashable-by-identity, cheap, and safe
    to cache; :meth:`to_dict` makes it JSON-serializable across the API boundary.

    Attributes
    ----------
    startprob:
        ``(n_states,)`` initial-state distribution ``pi`` (sums to 1).
    transmat:
        ``(n_states, n_states)`` row-stochastic transition matrix ``A``.
    means:
        ``(n_states, n_features)`` per-state emission means.
    covariances:
        Per-state variances (diag) or covariance matrices (full).
    covariance_type:
        ``"diag"`` or ``"full"``.
    log_likelihood:
        The training-sequence log-likelihood at convergence.
    n_iter:
        The number of EM iterations the winning restart ran.
    converged:
        Whether the winning restart met the EM tolerance before the iteration cap.
    """

    startprob: FloatArray
    transmat: FloatArray
    means: FloatArray
    covariances: FloatArray
    covariance_type: CovarianceType
    log_likelihood: float
    n_iter: int
    converged: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        """The number of hidden states ``K``."""
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        """The emission feature dimension ``d``."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the fitted model."""
        return {
            "startprob": [float(v) for v in self.startprob],
            "transmat": [[float(v) for v in row] for row in self.transmat],
            "means": [[float(v) for v in row] for row in self.means],
            "covariances": _covariances_to_list(self.covariances),
            "covariance_type": str(self.covariance_type),
            "log_likelihood": float(self.log_likelihood),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "meta": dict(self.meta),
        }


def _covariances_to_list(covariances: FloatArray) -> list[Any]:
    """Recursively coerce a (diag or full) covariance array to nested lists."""
    return [_covariances_to_list(c) if hasattr(c, "__len__") else float(c) for c in covariances]


def online_filter(model: HMMModel, observations: FloatArray) -> FloatArray:
    r"""Causal online forward-filter posteriors (the ONLY tradable signal).

    Returns the ``(n_obs, n_states)`` matrix of filtered posteriors

    .. math::

        f_t(k) = p(\text{state}_t = k \mid x_1, \dots, x_t),

    each row computed by the log-space forward recursion normalized AT EACH STEP.
    Row ``t`` depends on observations ``x_1..x_t`` ONLY — it is invariant to any
    ``x_s`` with ``s > t``.

    NO-LOOKAHEAD GUARANTEE (property-tested): perturbing the returns AFTER time
    ``t`` leaves ``f_t`` byte-identical (future-perturbation invariance /
    prefix-determinism). This is what makes the filtered posterior — and only the
    filtered posterior — a legitimate out-of-sample regime signal. Smoothed and
    Viterbi posteriors fail this test and are non-tradable.

    Parameters
    ----------
    model:
        A fitted :class:`HMMModel` (its parameters were estimated on the TRAIN
        fold only).
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix for the window being
        filtered (the scaler was also fit train-only).

    Returns
    -------
    FloatArray
        The ``(n_obs, n_states)`` causal filtered posterior; each row sums to 1.

    Raises
    ------
    ValidationError
        If ``observations`` is not 2-D or its feature dimension does not match the
        model.
    """
    raise NotImplementedError


def filtered_states(model: HMMModel, observations: FloatArray) -> FloatArray:
    """Causal hard regime labels: the argmax of the online filter posterior.

    A convenience wrapper around :func:`online_filter` returning the
    ``(n_obs,)`` integer MAP state under the FILTERED (no-lookahead) posterior.
    Unlike the Viterbi path this never peeks ahead, so it is tradable.

    Parameters
    ----------
    model:
        A fitted :class:`HMMModel`.
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix.

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` filtered hard-state labels (integer-valued floats).

    Raises
    ------
    ValidationError
        If ``observations`` is malformed.
    """
    raise NotImplementedError
