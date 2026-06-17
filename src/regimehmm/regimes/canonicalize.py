"""State canonicalization for stable, cross-fold regime labels.

A raw HMM fit assigns arbitrary integer indices to its hidden states: the "high
vol" regime might be state 2 in one fold and state 0 in the next, purely because
of restart initialization. That label arbitrariness would scramble any cross-fold
characterization or golden regression. This module imposes a CANONICAL ordering —
states sorted by ascending mean conditional return, tie-broken by ascending
volatility — and returns a permutation that relabels the model and any decoded
state series consistently.

After canonicalization, "regime 0" always means the same kind of regime (lowest
mean return) across folds, so the characterization is relabeling-invariant
(property-tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from regimehmm._typing import FloatArray
from regimehmm.hmm.filter import HMMModel


def canonical_order(model: HMMModel) -> FloatArray:
    r"""Return the permutation that sorts states into canonical order.

    States are ordered by ASCENDING mean conditional (per-state) return; ties on
    the mean are broken by ASCENDING per-state volatility. The result is the index
    array ``perm`` such that ``perm[0]`` is the original index of the new "regime
    0" (lowest mean / lowest vol), and so on.

    Parameters
    ----------
    model:
        A fitted :class:`~regimehmm.hmm.filter.HMMModel`. The ordering key is the
        per-state mean of the RETURN feature (feature column 0 by convention) and,
        as a tie-break, the per-state return volatility.

    Returns
    -------
    FloatArray
        The ``(n_states,)`` integer permutation mapping new label -> original
        index.
    """
    raise NotImplementedError


def canonicalize_model(model: HMMModel) -> HMMModel:
    """Return a relabeled copy of ``model`` in canonical state order.

    Applies :func:`canonical_order` to permute ``startprob``, the rows and columns
    of ``transmat``, ``means``, and ``covariances`` so the returned model's state
    ``k`` is the canonical regime ``k``. The fit's likelihood is unchanged (a pure
    relabeling); the model becomes comparable across folds and serializable to a
    stable schema.

    Parameters
    ----------
    model:
        The fitted model to canonicalize.

    Returns
    -------
    HMMModel
        A canonically-ordered copy of ``model``.
    """
    raise NotImplementedError


def relabel_states(states: FloatArray, order: FloatArray) -> FloatArray:
    """Relabel a decoded state series under a canonical-ordering permutation.

    Given a raw decoded series (from the filter or Viterbi) and the ``order``
    permutation from :func:`canonical_order`, return the series with each raw state
    index replaced by its canonical label, so downstream characterization and
    golden regressions are invariant to the fit's arbitrary internal labelling.

    Parameters
    ----------
    states:
        The ``(n_obs,)`` raw decoded state labels.
    order:
        The ``(n_states,)`` permutation (new label -> original index) from
        :func:`canonical_order`.

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` canonically-relabeled state series.

    Raises
    ------
    ValidationError
        If ``order`` is not a valid permutation or ``states`` contains an
        out-of-range label.
    """
    raise NotImplementedError
