"""Plotly figure builders (lazy).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}`` —
the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) imported
LAZILY inside each builder; importing this module has no side effects and does not
require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from regimehmm._typing import FloatArray

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    return value


def regime_shaded_figure(
    series: pd.Series,
    states: FloatArray,
    *,
    n_states: int,
    title: str = "Regime-shaded series",
) -> FigureDict:
    r"""Build a price/return line with a FILTERED-regime background ribbon.

    Plots ``series`` (price or cumulative return) as a line, with the plot
    background shaded by the (canonical) decoded regime at each step — one band
    colour per regime — so the persistent high/low-vol regimes are visible at a
    glance. A small legend maps each band colour to its regime label.

    SIGNAL DISCIPLINE: for an HONEST out-of-sample picture, ``states`` should be the
    ONLINE-FILTER labels (:func:`regimehmm.hmm.filter.filtered_states`). Smoothed /
    Viterbi labels may be shaded for in-sample EDA but must be captioned as
    non-tradable.

    Parameters
    ----------
    series:
        The per-period series to draw (price level or cumulative return), indexed
        by date.
    states:
        The ``(n_obs,)`` canonical decoded regime labels aligned to ``series``.
    n_states:
        The number of regimes (drives the colour palette and legend).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping with the line trace and per-regime
        background shapes.

    Raises
    ------
    ValidationError
        If ``series`` and ``states`` are misaligned or a label is out of range.
    """
    raise NotImplementedError


def regime_stats_figure(
    means: FloatArray,
    vols: FloatArray,
    *,
    title: str = "Per-regime risk/return",
) -> FigureDict:
    """Build a per-regime mean-vs-volatility bar/scatter summary.

    Renders each regime's annualized mean return against its annualized volatility
    so the reader sees the characterization headline: low-vol regimes cluster at
    modest positive mean, high-vol regimes at lower/negative mean and much higher
    risk.

    Parameters
    ----------
    means:
        The ``(n_states,)`` per-regime annualized mean returns (canonical order).
    vols:
        The ``(n_states,)`` per-regime annualized volatilities (canonical order).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping.

    Raises
    ------
    ValidationError
        If ``means`` and ``vols`` have different lengths.
    """
    raise NotImplementedError


def oos_equity_figure(
    overlay_returns: pd.Series,
    buyhold_returns: pd.Series,
    *,
    title: str = "OOS equity: regime overlay vs buy-and-hold",
) -> FigureDict:
    r"""Build the out-of-sample equity curve: overlay vs buy-and-hold.

    Compounds each net OOS return series into a wealth curve and overlays the two
    so the reader sees the honest headline directly: the regime-timing overlay does
    NOT reliably separate from buy-and-hold after costs. Both curves start at 1.0.

    Parameters
    ----------
    overlay_returns:
        The net (after-cost) OOS overlay return series.
    buyhold_returns:
        The buy-and-hold benchmark return series over the same OOS window.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping with the two equity traces.

    Raises
    ------
    ValidationError
        If the two series cannot be aligned.
    """
    raise NotImplementedError
