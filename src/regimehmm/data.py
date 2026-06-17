"""Data layer: synthetic regime-switch generator + real EOD price loader.

Two responsibilities, both side-effect-free at import:

1. A SYNTHETIC regime-switch generator (:func:`generate_regime_switch`) — a 2/3-state
   Gaussian HMM-style return series with PERSISTENT states of sharply different
   volatility, seeded via :func:`regimehmm._rng.make_rng`. Every test in the suite
   runs on this generator; no test touches the network.

2. A real EOD price loader (:func:`get_prices`, :func:`compute_returns`) for the
   deployed backend: it tries the Polygon provider (when ``source_pref="polygon"``)
   and degrades to the synthetic series on any upstream failure, reporting the
   provenance via the returned :data:`DataSource`.

Heavy data dependencies (polygon/httpx, yfinance, diskcache, pyarrow) live behind
the ``data`` extra and are imported LAZILY inside the loader functions. Importing
this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from regimehmm._typing import PricesLike

#: Where a price/return series ultimately came from. Returned alongside data so
#: callers (and the API ``data_source`` field) can report provenance.
DataSource = Literal["polygon", "yfinance", "synthetic", "cache"]

#: Supported synthetic feature sets, mirroring the API ``feature_set`` field.
FeatureSet = Literal["returns", "returns_vol", "returns_vol_macro"]


@dataclass(frozen=True, slots=True)
class RegimeSwitchSeries:
    """A synthetic regime-switch sample with its ground-truth hidden states.

    Attributes
    ----------
    returns:
        The generated per-period return series (a ``pd.Series`` indexed by a
        business-day :class:`pandas.DatetimeIndex`).
    states:
        The ground-truth hidden-state path that generated ``returns`` (canonical
        order: ``0`` = lowest-mean / lowest-vol regime). Used to score the decoder
        in tests; NEVER available to the model at fit time.
    means:
        The ``(n_states,)`` per-state mean returns used by the generator.
    vols:
        The ``(n_states,)`` per-state volatilities (persistent, sharply different).
    transmat:
        The ``(n_states, n_states)`` row-stochastic generating transition matrix.
    """

    returns: pd.Series
    states: np.ndarray
    means: np.ndarray
    vols: np.ndarray
    transmat: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this sample."""
        return {
            "returns": {str(k): float(v) for k, v in self.returns.items()},
            "states": [int(s) for s in self.states],
            "means": [float(m) for m in self.means],
            "vols": [float(v) for v in self.vols],
            "transmat": [[float(v) for v in row] for row in self.transmat],
            "meta": dict(self.meta),
        }


def generate_regime_switch(
    n_obs: int,
    *,
    n_states: int = 2,
    persistence: float = 0.97,
    means: tuple[float, ...] | None = None,
    vols: tuple[float, ...] | None = None,
    seed: int = 7,
) -> RegimeSwitchSeries:
    r"""Generate a seeded synthetic regime-switch return series.

    Builds a ``persistence``-sticky ``n_states``-state Markov chain (a transition
    matrix with ``persistence`` on the diagonal and the remainder spread evenly
    off-diagonal), walks it for ``n_obs`` steps, and emits a Gaussian return from
    the active state's ``(mean, vol)`` at each step. The default per-state ``vols``
    are sharply different (e.g. a calm low-vol regime and a turbulent high-vol
    regime) and the states are PERSISTENT, exactly the structure a Gaussian HMM is
    meant to recover.

    DETERMINISM: all randomness flows from :func:`regimehmm._rng.make_rng(seed)`, so
    a fixed ``(n_obs, n_states, persistence, seed)`` reproduces the series and its
    ground-truth states byte-for-byte. This is the generator the ENTIRE test suite
    runs on; it touches no network.

    GROUND TRUTH: the returned ``states`` are the true generating path (canonical
    order). Tests use them to score the decoder, but the model never sees them at
    fit time.

    Parameters
    ----------
    n_obs:
        The number of observations to generate (``>= 1``).
    n_states:
        The number of regimes (``2`` or ``3`` for the canonical fixtures).
    persistence:
        The self-transition probability on the diagonal (``in (0, 1)``); higher =
        stickier regimes.
    means:
        Optional per-state mean returns; defaults to a spread of small
        positive/negative means in canonical (ascending-mean) order.
    vols:
        Optional per-state volatilities; defaults to sharply different, ascending
        vols (low-vol calm regime ... high-vol crisis regime).
    seed:
        Master seed for the generator.

    Returns
    -------
    RegimeSwitchSeries
        The frozen sample (returns + ground-truth states + generating parameters).

    Raises
    ------
    ValidationError
        If ``n_obs < 1``, ``n_states < 1``, ``persistence`` is outside ``(0, 1)``,
        or a supplied ``means``/``vols`` length does not match ``n_states``.
    """
    raise NotImplementedError


def build_features(
    returns: pd.Series,
    *,
    feature_set: FeatureSet = "returns_vol",
    vol_window: int = 21,
) -> pd.DataFrame:
    r"""Build the HMM emission feature matrix from a return series.

    Assembles the per-observation feature columns selected by ``feature_set``:

    * ``"returns"`` — the return itself (column 0, the canonicalization key);
    * ``"returns_vol"`` — return plus a TRAILING realized-volatility feature
      (rolling std over ``vol_window``, using only past returns — no lookahead);
    * ``"returns_vol_macro"`` — the above plus a slow macro/trend feature.

    NO-LOOKAHEAD: every derived feature at ``t`` uses returns ``<= t`` only
    (trailing windows), so the feature matrix is causal. The leading rows with an
    incomplete window are dropped. Standardization (StandardScaler) is applied
    train-only by the caller, never here.

    Parameters
    ----------
    returns:
        The per-period return series.
    feature_set:
        Which feature columns to build.
    vol_window:
        Trailing window length for the realized-volatility feature.

    Returns
    -------
    pandas.DataFrame
        The causal feature matrix (column 0 is the return, the canonicalization
        key), leading incomplete-window rows dropped.

    Raises
    ------
    ValidationError
        If ``feature_set`` is unsupported or ``vol_window < 1``.
    """
    raise NotImplementedError


def _synthetic_prices(ticker: str, start: date, end: date, *, seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic single-ticker price panel from a regime-switch series.

    Offline/CI fallback used by :func:`get_prices`: generates a regime-switch
    return series via :func:`generate_regime_switch`, compounds it into a strictly
    positive price level, and returns a one-column ``date x ticker`` frame. Seeded
    off the request so the same ``(ticker, start, end, seed)`` is byte-identical.
    """
    raise NotImplementedError


def get_prices(
    ticker: str,
    start: date,
    end: date,
    *,
    source_pref: Literal["polygon", "yfinance", "auto", "synthetic"] = "auto",
    seed: int = 7,
) -> tuple[pd.DataFrame, DataSource]:
    """Load a single-ticker EOD price panel with graceful synthetic fallback.

    With ``source_pref="polygon"`` the real Polygon EOD provider
    (:class:`regimehmm.data_providers.polygon.PolygonProvider`) is tried first and,
    on ANY failure (missing key, network error, empty payload), falls through to a
    deterministic synthetic regime-switch price series so the loader never
    hard-fails. ``source_pref="synthetic"`` forces the synthetic path (used by the
    backend's ``data_source_pref`` and by offline runs).

    LAZY IMPORT: the polygon provider (and ``httpx``) are imported inside this
    function, never at module import time.

    Parameters
    ----------
    ticker:
        The asset symbol (e.g. ``"SPY"``).
    start, end:
        Inclusive date range.
    source_pref:
        Source preference; ``"auto"`` and ``"polygon"`` try Polygon then fall back
        to synthetic, ``"synthetic"`` forces the offline path.
    seed:
        Seed for the synthetic fallback.

    Returns
    -------
    tuple[pandas.DataFrame, DataSource]
        The single-column price panel and the source it came from.

    Raises
    ------
    ValidationError
        If ``ticker`` is empty or ``end <= start``.
    """
    raise NotImplementedError


def compute_returns(prices: PricesLike) -> pd.Series:
    r"""Convert a single-ticker price panel to a simple return series.

    NO-LOOKAHEAD REQUIREMENT: returns are computed with
    ``prices.pct_change(fill_method=None)`` — prices are NEVER forward-filled before
    differencing (ffill-then-diff manufactures spurious zero returns across gaps and
    leaks information). The leading NaN row is dropped and the result is squeezed to
    a 1-D Series.

    Parameters
    ----------
    prices:
        A single-column price panel (rows = date).

    Returns
    -------
    pandas.Series
        The simple-return series with the leading NaN removed.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed or has more than one column.
    """
    raise NotImplementedError
