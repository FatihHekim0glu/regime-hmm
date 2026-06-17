"""Per-regime characterization — the project's real deliverable.

Given a (canonicalized) decoded regime series and the realized returns, this module
computes the per-regime descriptive statistics that ARE the honest headline of the
project: each regime's mean return, volatility, persistence (self-transition
probability), expected duration, and worst drawdown. Regime CHARACTERIZATION is the
win; the regime-timing overlay (see :mod:`regimehmm.backtest.overlay`) is the
honest null.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from regimehmm._typing import FloatArray
from regimehmm.hmm.filter import HMMModel


@dataclass(frozen=True, slots=True)
class RegimeStats:
    """Immutable descriptive statistics for a single regime.

    Attributes
    ----------
    state:
        The canonical regime label (``0`` = lowest mean return).
    frequency:
        The fraction of observations assigned to this regime.
    mean_return:
        The annualized mean return while in this regime.
    volatility:
        The annualized return volatility while in this regime.
    persistence:
        The self-transition probability ``A[k, k]`` (how "sticky" the regime is).
    expected_duration:
        The expected dwell time ``1 / (1 - persistence)`` in periods.
    max_drawdown:
        The worst peak-to-trough drawdown of the within-regime return stream
        (``<= 0``).
    """

    state: int
    frequency: float
    mean_return: float
    volatility: float
    persistence: float
    expected_duration: float
    max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this regime's stats."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegimeCharacterization:
    """Immutable bundle of per-regime statistics for a fitted model.

    Attributes
    ----------
    n_states:
        The number of regimes.
    stats:
        The per-regime :class:`RegimeStats`, in canonical order.
    """

    n_states: int
    stats: tuple[RegimeStats, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the characterization."""
        return {
            "n_states": int(self.n_states),
            "stats": [s.to_dict() for s in self.stats],
            "meta": dict(self.meta),
        }


def characterize_regimes(
    model: HMMModel,
    states: FloatArray,
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
) -> RegimeCharacterization:
    r"""Compute per-regime descriptive statistics (the headline deliverable).

    For each canonical regime ``k`` this measures, over the observations the
    decoder assigned to ``k``: the occupancy ``frequency``, the annualized
    ``mean_return`` and ``volatility`` of the realized returns, the ``persistence``
    ``A[k, k]`` and the implied ``expected_duration = 1 / (1 - A[k, k])``, and the
    within-regime ``max_drawdown``.

    LABEL DISCIPLINE: ``states`` must already be in CANONICAL order (see
    :mod:`regimehmm.regimes.canonicalize`) so the output is relabeling-invariant
    across folds (property-tested). The decoded ``states`` should come from the
    smoothed/Viterbi decoder for in-sample EDA, or the online filter for an honest
    out-of-sample picture — but characterization itself is descriptive, not a
    tradable signal.

    Parameters
    ----------
    model:
        The fitted (canonicalized) :class:`~regimehmm.hmm.filter.HMMModel`,
        supplying the transition matrix for persistence/duration.
    states:
        The ``(n_obs,)`` canonical decoded regime labels.
    returns:
        The realized per-period return series aligned to ``states``.
    periods_per_year:
        Annualization factor for the mean/volatility (``252`` for daily).

    Returns
    -------
    RegimeCharacterization
        The frozen per-regime statistics bundle in canonical order.

    Raises
    ------
    ValidationError
        If ``states`` and ``returns`` are misaligned or contain out-of-range
        labels.
    """
    raise NotImplementedError
