"""Regime-conditioned exposure overlay (the honest null) vs buy-and-hold.

The overlay scales market exposure by the FILTERED regime: it cuts (or zeroes)
exposure in high-volatility / low-mean regimes and holds full exposure otherwise.
Two non-negotiable leakage guards apply:

1. The regime signal is the ONLINE FILTER posterior only (data <= t), NEVER the
   smoothed/Viterbi posterior.
2. The exposure decided at ``t`` is applied via ``signal.shift(1)`` so the return
   realized at ``t`` is earned with an exposure chosen strictly before ``t``.

Per-side basis-point costs are charged on exposure changes, and a sensitivity grid
sweeps the cost level. The headline comparison is overlay-vs-buy-and-hold, and the
honest result is that the overlay does NOT reliably beat buy-and-hold OOS after
costs.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from regimehmm._typing import FloatArray


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Immutable result of a regime-timing overlay backtest vs buy-and-hold.

    Attributes
    ----------
    overlay_returns:
        The net (after-cost) out-of-sample overlay return series.
    buyhold_returns:
        The buy-and-hold benchmark return series over the same OOS window.
    exposure:
        The applied (``shift(1)``-ed) exposure series in ``[0, 1]``.
    overlay_sharpe:
        Annualized OOS Sharpe of the net overlay returns.
    buyhold_sharpe:
        Annualized OOS Sharpe of buy-and-hold.
    cost_bps:
        The per-side cost level used for this run.
    turnover:
        Total one-way exposure turnover charged.
    """

    overlay_returns: pd.Series
    buyhold_returns: pd.Series
    exposure: pd.Series
    overlay_sharpe: float
    buyhold_sharpe: float
    cost_bps: float
    turnover: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "overlay_returns": {str(k): _safe_float(v) for k, v in self.overlay_returns.items()},
            "buyhold_returns": {str(k): _safe_float(v) for k, v in self.buyhold_returns.items()},
            "exposure": {str(k): _safe_float(v) for k, v in self.exposure.items()},
            "overlay_sharpe": _safe_float(self.overlay_sharpe),
            "buyhold_sharpe": _safe_float(self.buyhold_sharpe),
            "cost_bps": float(self.cost_bps),
            "turnover": float(self.turnover),
            "meta": dict(self.meta),
        }


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    import numpy as np

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def regime_exposure(
    filtered_posterior: FloatArray,
    *,
    risk_off_states: tuple[int, ...],
    risk_off_exposure: float = 0.0,
) -> FloatArray:
    r"""Map a FILTERED regime posterior to a target market exposure.

    Computes a per-observation target exposure in ``[0, 1]`` by reducing exposure
    in the designated risk-off (typically high-vol / low-mean) regimes. With a hard
    decoder this is ``risk_off_exposure`` whenever the filtered MAP state is
    risk-off and ``1.0`` otherwise; with the soft posterior it can blend by the
    risk-off probability mass.

    LEAKAGE GUARD: ``filtered_posterior`` MUST be the online-filter posterior
    (:func:`regimehmm.hmm.filter.online_filter`), whose row ``t`` uses data
    ``<= t`` only. Passing a smoothed/Viterbi posterior here is look-ahead leakage
    and is rejected upstream.

    Parameters
    ----------
    filtered_posterior:
        The ``(n_obs, n_states)`` causal filtered posterior (canonical state
        order).
    risk_off_states:
        The canonical state indices treated as risk-off.
    risk_off_exposure:
        Exposure held while risk-off (``0.0`` = flat, ``0.5`` = half, etc.).

    Returns
    -------
    FloatArray
        The ``(n_obs,)`` target exposure series in ``[0, 1]`` (pre-``shift``).

    Raises
    ------
    ValidationError
        If ``risk_off_exposure`` is outside ``[0, 1]`` or a state index is out of
        range.
    """
    raise NotImplementedError


def overlay_backtest(
    returns: pd.Series,
    target_exposure: FloatArray,
    *,
    cost_bps: float = 10.0,
) -> OverlayResult:
    r"""Backtest a regime-timing exposure overlay against buy-and-hold.

    Applies ``target_exposure`` to the market return series with a one-step
    ``shift(1)`` chokepoint (the exposure decided at ``t`` earns the return at
    ``t + 1``), charges ``cost_bps`` per side on each exposure change, and compares
    the net overlay Sharpe to buy-and-hold over the same OOS window.

    NO-LOOKAHEAD: exposure is ``shift(1)``-ed before being multiplied into returns,
    mirroring the walk-forward engine's weight-application chokepoint. Combined with
    the online-filter-only signal, no future information reaches any earned return.

    HONEST-NULL EXPECTATION: on the persistent-vol ``regime_switch`` fixture the net
    overlay Sharpe does NOT reliably exceed buy-and-hold once costs are charged
    (regression-pinned), feeding the ``no_timing_edge`` verdict.

    Parameters
    ----------
    returns:
        The realized per-period market return series (the OOS window).
    target_exposure:
        The ``(n_obs,)`` pre-shift target exposure from :func:`regime_exposure`.
    cost_bps:
        Per-side transaction cost in basis points charged on exposure changes.

    Returns
    -------
    OverlayResult
        The frozen overlay-vs-buy-and-hold result bundle.

    Raises
    ------
    ValidationError
        If ``cost_bps < 0`` or ``returns`` and ``target_exposure`` are misaligned.
    """
    raise NotImplementedError


def overlay_cost_grid(
    returns: pd.Series,
    target_exposure: FloatArray,
    *,
    cost_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0),
) -> tuple[OverlayResult, ...]:
    """Run the overlay across a grid of per-side cost levels (sensitivity sweep).

    Calls :func:`overlay_backtest` once per ``cost_bps`` in ``cost_grid`` and
    returns the results in grid order. The net overlay Sharpe must be
    non-increasing in ``cost_bps`` (cost-monotonicity, regression-pinned), and the
    grid size feeds the Deflated-Sharpe ``n_effective_trials`` count.

    Parameters
    ----------
    returns:
        The realized per-period market return series.
    target_exposure:
        The ``(n_obs,)`` pre-shift target exposure.
    cost_grid:
        The per-side cost levels (bps) to sweep.

    Returns
    -------
    tuple[OverlayResult, ...]
        One :class:`OverlayResult` per cost level, in grid order.

    Raises
    ------
    ValidationError
        If ``cost_grid`` is empty or contains a negative cost.
    """
    raise NotImplementedError
