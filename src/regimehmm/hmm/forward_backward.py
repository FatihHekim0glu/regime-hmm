"""Log-space forward-backward (Baum-Welch E-step).

Computes, in LOG space (via ``scipy.special.logsumexp`` for numerical stability),
the forward messages ``alpha``, backward messages ``beta``, the SMOOTHED state
posteriors ``gamma`` and pair-marginals ``xi``, and the total sequence
log-likelihood. These quantities use the WHOLE sample (data before AND after each
``t``), so the smoothed posteriors PEEK AHEAD and are **in-sample EDA only —
NEVER tradable**. The online, no-lookahead posterior lives in
:mod:`regimehmm.hmm.filter`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regimehmm._typing import FloatArray


@dataclass(frozen=True, slots=True)
class ForwardBackwardResult:
    """Immutable bundle of forward-backward quantities for one sequence.

    NON-TRADABLE: ``gamma`` and ``xi`` are SMOOTHED (they condition on the entire
    sequence, including the future) and must only be used for in-sample EDA and as
    the EM E-step sufficient statistics — never as an out-of-sample trading signal.

    Attributes
    ----------
    log_alpha:
        ``(n_obs, n_states)`` log forward messages
        ``log p(x_1..x_t, state_t=k)``.
    log_beta:
        ``(n_obs, n_states)`` log backward messages
        ``log p(x_{t+1}..x_T | state_t=k)``.
    gamma:
        ``(n_obs, n_states)`` SMOOTHED posteriors ``p(state_t=k | x_1..x_T)``;
        each row sums to 1. EDA-only.
    xi:
        ``(n_obs - 1, n_states, n_states)`` pair-marginals
        ``p(state_t=i, state_{t+1}=j | x_1..x_T)``; each slice sums to 1. EDA-only.
    log_likelihood:
        The total sequence log-likelihood ``log p(x_1..x_T)``.
    """

    log_alpha: FloatArray
    log_beta: FloatArray
    gamma: FloatArray
    xi: FloatArray
    log_likelihood: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "log_alpha": [list(map(float, row)) for row in self.log_alpha],
            "log_beta": [list(map(float, row)) for row in self.log_beta],
            "gamma": [list(map(float, row)) for row in self.gamma],
            "xi": [[list(map(float, r)) for r in slc] for slc in self.xi],
            "log_likelihood": float(self.log_likelihood),
            "meta": dict(self.meta),
        }


def forward_pass(
    log_startprob: FloatArray,
    log_transmat: FloatArray,
    log_emission: FloatArray,
) -> tuple[FloatArray, float]:
    r"""Run the log-space forward recursion.

    Computes ``log_alpha[t, k] = log p(x_1..x_t, state_t=k)`` via

    .. math::

        \log\alpha_1(k) &= \log\pi_k + \log b_k(x_1), \\
        \log\alpha_{t}(j) &= \log b_j(x_t)
            + \operatorname*{logsumexp}_i\big[\log\alpha_{t-1}(i) + \log A_{ij}\big],

    and returns the total log-likelihood ``logsumexp(log_alpha[-1])``. All sums are
    taken with ``scipy.special.logsumexp`` so the recursion never underflows.

    Parameters
    ----------
    log_startprob:
        ``(n_states,)`` log initial-state distribution ``log pi``.
    log_transmat:
        ``(n_states, n_states)`` log transition matrix ``log A`` (rows are
        stochastic in probability space).
    log_emission:
        ``(n_obs, n_states)`` log emission densities from
        :func:`regimehmm.hmm.kernel.gaussian_log_density`.

    Returns
    -------
    tuple[FloatArray, float]
        ``(log_alpha, log_likelihood)``.

    Raises
    ------
    ValidationError
        If the operand shapes are inconsistent.
    """
    raise NotImplementedError


def backward_pass(
    log_transmat: FloatArray,
    log_emission: FloatArray,
) -> FloatArray:
    r"""Run the log-space backward recursion.

    Computes ``log_beta[t, k] = log p(x_{t+1}..x_T | state_t=k)`` via the standard
    backward recursion with ``log_beta[-1] = 0`` (an empty product), again using
    ``logsumexp`` for stability.

    Parameters
    ----------
    log_transmat:
        ``(n_states, n_states)`` log transition matrix.
    log_emission:
        ``(n_obs, n_states)`` log emission densities.

    Returns
    -------
    FloatArray
        The ``(n_obs, n_states)`` log backward messages.

    Raises
    ------
    ValidationError
        If the operand shapes are inconsistent.
    """
    raise NotImplementedError


def forward_backward(
    log_startprob: FloatArray,
    log_transmat: FloatArray,
    log_emission: FloatArray,
) -> ForwardBackwardResult:
    r"""Full forward-backward pass: smoothed posteriors and pair-marginals.

    Runs :func:`forward_pass` and :func:`backward_pass`, then forms the SMOOTHED
    posteriors ``gamma`` and pair-marginals ``xi`` (the EM E-step sufficient
    statistics):

    .. math::

        \gamma_t(k) &\propto \alpha_t(k)\,\beta_t(k), \\
        \xi_t(i, j) &\propto \alpha_t(i)\,A_{ij}\,b_j(x_{t+1})\,\beta_{t+1}(j),

    normalized in log space. Each ``gamma`` row sums to 1 and each ``xi`` slice
    sums to 1 (asserted in the property suite).

    NON-TRADABLE: ``gamma``/``xi`` condition on the full sequence and PEEK AHEAD;
    they are EDA / EM internals only. Use :mod:`regimehmm.hmm.filter` for any
    out-of-sample regime signal.

    PARITY REQUIREMENT: ``gamma`` and ``log_likelihood`` are validated against
    ``hmmlearn.GaussianHMM`` to ``1e-6`` in the parity suite.

    Parameters
    ----------
    log_startprob:
        ``(n_states,)`` log initial-state distribution.
    log_transmat:
        ``(n_states, n_states)`` log transition matrix.
    log_emission:
        ``(n_obs, n_states)`` log emission densities.

    Returns
    -------
    ForwardBackwardResult
        The frozen bundle of forward-backward quantities.

    Raises
    ------
    ValidationError
        If the operand shapes are inconsistent.
    """
    raise NotImplementedError
