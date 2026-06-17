"""Viterbi MAP decoding — EDA ONLY, never tradable.

The Viterbi algorithm returns the single most likely STATE SEQUENCE given the
ENTIRE observation sequence, ``argmax_{s_1..s_T} p(s_1..s_T | x_1..x_T)``. Because
every label conditions on the whole sample (including the future), the Viterbi
path PEEKS AHEAD: a return realized at ``t+5`` can change the decoded label at
``t``. It is therefore an in-sample / exploratory tool ONLY and must NEVER be used
to generate an out-of-sample regime label or trading signal. The only tradable
decoder is the online forward filter in :mod:`regimehmm.hmm.filter`.

Importing this module has no side effects.
"""

from __future__ import annotations

from regimehmm._typing import FloatArray
from regimehmm.hmm.filter import HMMModel


def viterbi_path(model: HMMModel, observations: FloatArray) -> FloatArray:
    r"""Most likely hidden-state path (Viterbi MAP decoding) — EDA only.

    Runs the log-space Viterbi recursion to return the ``(n_obs,)`` integer state
    sequence maximizing the joint posterior over the whole sequence:

    .. math::

        s^\* = \operatorname*{arg\,max}_{s_1, \dots, s_T}
               \; p(s_1, \dots, s_T \mid x_1, \dots, x_T).

    LOOK-AHEAD WARNING: the Viterbi label at ``t`` depends on observations both
    before AND after ``t``. This is acceptable for in-sample regime visualization
    and parity checking against ``hmmlearn.GaussianHMM.decode`` (validated to an
    exact label match in the parity suite), but it is **never** a valid OOS signal.
    Any tradable label must come from :func:`regimehmm.hmm.filter.filtered_states`.

    Parameters
    ----------
    model:
        A fitted :class:`~regimehmm.hmm.filter.HMMModel`.
    observations:
        The ``(n_obs, n_features)`` scaled feature matrix.

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` Viterbi MAP state path (integer-valued floats).

    Raises
    ------
    ValidationError
        If ``observations`` is malformed or its feature dimension mismatches the
        model.
    """
    raise NotImplementedError
