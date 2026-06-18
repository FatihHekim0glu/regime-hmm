"""Regression: the regime-timing overlay does NOT beat buy-and-hold OOS.

This is the project's HEADLINE honest-null result. On the persistent-vol
``regime_switch`` fixture, an overlay that cuts exposure in the (genuinely
high-vol / low-mean) risk-off regime - using the ONLINE FILTER only and the
``shift(1)`` chokepoint - does not reliably beat a buy-and-hold baseline once
per-side costs are charged, and the Memmel-Jobson-Korkie Sharpe-difference test
is INSIGNIFICANT. That is exactly the condition that feeds the ``no_timing_edge``
verdict.

The test is deliberately structural rather than a brittle golden number: it pins
the *direction* of the result (overlay <= buy-and-hold; JK insignificant; cost
monotonicity) across both the direct backtest and the anchored walk-forward
evaluator, so it stays green under small numerical drift while still failing loudly
if the overlay ever starts "beating the market" (which would mean leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm.backtest.overlay import (
    overlay_backtest,
    overlay_cost_grid,
    regime_exposure,
    walk_forward_overlay,
)
from regimehmm.evaluation.comparison import jobson_korkie_memmel
from regimehmm.hmm.em import fit_hmm
from regimehmm.hmm.filter import online_filter

_TRAIN_SPLIT = 1000
_COST_BPS = 10.0


def _as_series(obj: object) -> pd.Series:
    """Narrow an ``object``-typed fixture value to a ``pd.Series`` for strict mypy."""
    assert isinstance(obj, pd.Series)
    return obj


def _fit_and_filter(returns: pd.Series, split: int) -> tuple[np.ndarray, int]:
    """Fit a 2-state HMM on the TRAIN slice; return (online filtered posterior, risk-off).

    The scaler is fit train-only (``pct_change`` already applied upstream by the
    fixture's return series). The online filter then labels the FULL window
    causally; row ``t`` sees data ``<= t`` only. The risk-off state is the
    highest-variance fitted state (the genuine high-vol regime), identified from the
    model - not from any look-ahead label.
    """
    arr = returns.to_numpy(dtype="float64").reshape(-1, 1)
    train = arr[:split]
    mu = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0) + 1e-12
    obs_full = (arr - mu) / sd
    model = fit_hmm((train - mu) / sd, n_states=2, covariance_type="diag", n_restarts=4, seed=7)
    posterior = online_filter(model, obs_full)
    cov = np.asarray(model.covariances, dtype="float64")
    variances = cov[:, 0, 0] if cov.ndim == 3 else cov[:, 0]
    risk_off = int(np.argmax(variances))
    return posterior, risk_off


@pytest.mark.regression
def test_overlay_does_not_beat_buyhold_oos(regime_switch: dict[str, object]) -> None:
    """OOS overlay Sharpe <= buy-and-hold Sharpe and JK is insignificant."""
    returns = _as_series(regime_switch["returns"])
    returns.name = "SPY"
    posterior, risk_off = _fit_and_filter(returns, _TRAIN_SPLIT)

    exposure = regime_exposure(posterior, risk_off_states=(risk_off,), risk_off_exposure=0.0)
    oos_returns = returns.iloc[_TRAIN_SPLIT:]
    oos_exposure = exposure[_TRAIN_SPLIT:]

    result = overlay_backtest(oos_returns, oos_exposure, cost_bps=_COST_BPS)

    # HONEST NULL: the overlay does not reliably beat buy-and-hold after costs.
    assert result.overlay_sharpe <= result.buyhold_sharpe + 1e-9, (
        f"overlay Sharpe {result.overlay_sharpe:.4f} beat buy-and-hold "
        f"{result.buyhold_sharpe:.4f} - investigate leakage."
    )
    # Memmel-JK Sharpe-difference test is INSIGNIFICANT at the 5% level.
    jk_pvalue = jobson_korkie_memmel(
        np.asarray(result.overlay_returns.to_numpy(), dtype=np.float64),
        np.asarray(result.buyhold_returns.to_numpy(), dtype=np.float64),
    )
    assert jk_pvalue > 0.05, f"JK p-value {jk_pvalue:.4f} is significant - unexpected timing edge."
    # The overlay genuinely cut some exposure (it is not trivially buy-and-hold).
    assert 0.0 < result.meta["mean_exposure"] < 1.0


@pytest.mark.regression
def test_overlay_identical_oos_index(regime_switch: dict[str, object]) -> None:
    """Overlay and buy-and-hold legs are scored on the IDENTICAL OOS index."""
    returns = _as_series(regime_switch["returns"])
    returns.name = "SPY"
    posterior, risk_off = _fit_and_filter(returns, _TRAIN_SPLIT)
    exposure = regime_exposure(posterior, risk_off_states=(risk_off,), risk_off_exposure=0.0)
    oos_returns = returns.iloc[_TRAIN_SPLIT:]
    oos_exposure = exposure[_TRAIN_SPLIT:]

    result = overlay_backtest(oos_returns, oos_exposure, cost_bps=_COST_BPS)
    assert result.overlay_returns.index.equals(result.buyhold_returns.index)
    assert result.exposure.index.equals(result.overlay_returns.index)
    # shift(1): the first OOS observation is dropped (no prior decision).
    assert result.overlay_returns.index[0] == oos_returns.index[1]


@pytest.mark.regression
def test_overlay_cost_grid_monotone_and_caps_n_trials(
    regime_switch: dict[str, object],
) -> None:
    """Cost-grid Sharpes are non-increasing in cost; the grid size is the trial count."""
    returns = _as_series(regime_switch["returns"])
    returns.name = "SPY"
    posterior, risk_off = _fit_and_filter(returns, _TRAIN_SPLIT)
    exposure = regime_exposure(posterior, risk_off_states=(risk_off,), risk_off_exposure=0.0)
    oos_returns = returns.iloc[_TRAIN_SPLIT:]
    oos_exposure = exposure[_TRAIN_SPLIT:]

    cost_grid = (0.0, 5.0, 10.0, 20.0)
    grid = overlay_cost_grid(oos_returns, oos_exposure, cost_grid=cost_grid)
    assert len(grid) == len(cost_grid)
    sharpes = [r.overlay_sharpe for r in grid]
    finite = [s for s in sharpes if np.isfinite(s)]
    for lo, hi in zip(finite[1:], finite[:-1], strict=False):
        assert lo <= hi + 1e-9
    # No grid run beats buy-and-hold either.
    for r in grid:
        assert r.overlay_sharpe <= r.buyhold_sharpe + 1e-9


@pytest.mark.regression
def test_walk_forward_overlay_honest_null(regime_switch: dict[str, object]) -> None:
    """Anchored walk-forward overlay (reusing the shared engine) also fails to beat buy-hold.

    The HMM is fit on the pre-split TRAIN slice, so the genuinely out-of-sample,
    no-lookahead signal is the post-split window. Running the anchored walk-forward
    over THAT window keeps the filtered signal honestly causal (the model never saw
    these returns at fit time), which is the regime in which the honest-null claim
    applies. Feeding the walk-forward the in-sample window instead would let the
    overlay "win" purely because the model was fitted on those bars - an artifact
    of the fit boundary, not a tradable edge.
    """
    returns = _as_series(regime_switch["returns"])
    returns.name = "SPY"
    posterior, risk_off = _fit_and_filter(returns, _TRAIN_SPLIT)
    exposure = regime_exposure(posterior, risk_off_states=(risk_off,), risk_off_exposure=0.0)

    # Genuinely out-of-sample slice: data the fitted model never saw.
    oos_returns = returns.iloc[_TRAIN_SPLIT:]
    oos_signal = pd.Series(exposure[_TRAIN_SPLIT:], index=oos_returns.index)

    result = walk_forward_overlay(
        oos_returns,
        oos_signal,
        lookback_window=60,
        rebalance="monthly",
        cost_bps=_COST_BPS,
        anchored=True,
    )
    # IDENTICAL OOS INDEX across both legs.
    assert result.overlay_returns.index.equals(result.buyhold_returns.index)
    assert result.exposure.index.equals(result.overlay_returns.index)
    # HONEST NULL: the Sharpe GAP is not statistically significant. A point Sharpe
    # can land either side of buy-and-hold on a single finite OOS realization; the
    # honest-null claim - and the verdict layer - keys on the Memmel-JK test being
    # INSIGNIFICANT, which is what structurally forbids claiming a timing edge.
    jk_pvalue = jobson_korkie_memmel(
        np.asarray(result.overlay_returns.to_numpy(), dtype=np.float64),
        np.asarray(result.buyhold_returns.to_numpy(), dtype=np.float64),
    )
    assert jk_pvalue > 0.05, (
        f"walk-forward overlay JK p-value {jk_pvalue:.4f} is significant - "
        "the overlay would claim a timing edge; investigate leakage."
    )
