"""Unit tests for the overlay primitives and the walk-forward engine.

These exercise the cheap, pure parts (no per-fold HMM refit): the exposure map,
the single-window overlay backtest, the cost grid, the risk-off selector, and the
leakage / validation guards of the shared walk-forward engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm._exceptions import InsufficientDataError, ValidationError
from regimehmm.backtest.overlay import (
    overlay_backtest,
    overlay_cost_grid,
    regime_exposure,
    select_risk_off_state,
)
from regimehmm.backtest.walk_forward import BacktestResult, walk_forward_backtest
from regimehmm.regimes.characterize import RegimeCharacterization, RegimeStats


def _stat(state: int, vol: float) -> RegimeStats:
    """A minimal RegimeStats carrying only the volatility used by the selector."""
    return RegimeStats(
        state=state,
        frequency=0.5,
        mean_return=0.0,
        volatility=vol,
        persistence=0.9,
        expected_duration=10.0,
        max_drawdown=-0.1,
    )


# --------------------------------------------------------------------------- #
# regime_exposure                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_regime_exposure_hard_posterior_is_flat_when_risk_off() -> None:
    """A hard 0/1 posterior holds ``risk_off_exposure`` while risk-off, else 1.0."""
    posterior = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    exposure = regime_exposure(posterior, risk_off_states=(1,), risk_off_exposure=0.0)
    np.testing.assert_array_equal(exposure, np.array([1.0, 0.0, 1.0]))


@pytest.mark.unit
def test_regime_exposure_soft_blends_by_mass() -> None:
    """A soft posterior blends exposure by the risk-off probability mass."""
    posterior = np.array([[0.7, 0.3]])
    exposure = regime_exposure(posterior, risk_off_states=(1,), risk_off_exposure=0.0)
    assert exposure[0] == pytest.approx(0.7)


@pytest.mark.unit
def test_regime_exposure_validation_branches() -> None:
    """Out-of-range exposure, wrong rank, empty, and bad state index all raise."""
    posterior = np.array([[0.5, 0.5]])
    with pytest.raises(ValidationError, match=r"risk_off_exposure must be in \[0, 1\]"):
        regime_exposure(posterior, risk_off_states=(0,), risk_off_exposure=2.0)
    with pytest.raises(ValidationError, match="must be 2-D"):
        regime_exposure(np.array([0.5, 0.5]), risk_off_states=(0,))
    with pytest.raises(ValidationError, match="non-empty"):
        regime_exposure(np.empty((0, 2)), risk_off_states=(0,))
    with pytest.raises(ValidationError, match="risk_off_states must be non-empty"):
        regime_exposure(posterior, risk_off_states=())
    with pytest.raises(ValidationError, match="out of range"):
        regime_exposure(posterior, risk_off_states=(5,))


# --------------------------------------------------------------------------- #
# overlay_backtest + cost grid                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_overlay_backtest_drops_first_bar_and_shares_index() -> None:
    """The overlay drops the first bar and scores both legs on one OOS index."""
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.0], index=idx)
    exposure = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    res = overlay_backtest(returns, exposure, cost_bps=10.0)
    assert len(res.overlay_returns) == 5
    assert res.overlay_returns.index[0] == idx[1]
    assert res.overlay_returns.index.equals(res.buyhold_returns.index)
    assert res.exposure.index.equals(res.overlay_returns.index)


@pytest.mark.unit
def test_overlay_backtest_validation_branches() -> None:
    """Negative cost, wrong rank, and length mismatch each raise."""
    returns = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(ValidationError, match="cost_bps must be"):
        overlay_backtest(returns, np.array([1.0, 0.0, 1.0]), cost_bps=-1.0)
    with pytest.raises(ValidationError, match="must be 1-D"):
        overlay_backtest(returns, np.zeros((3, 1)))
    with pytest.raises(ValidationError, match="length"):
        overlay_backtest(returns, np.array([1.0, 0.0]))


@pytest.mark.unit
def test_overlay_cost_grid_is_sharpe_non_increasing() -> None:
    """Net overlay Sharpe is non-increasing across an ascending cost grid."""
    rng = np.random.default_rng(0)
    n = 150
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.Series(rng.normal(0.0, 0.01, size=n), index=idx)
    target = rng.integers(0, 2, size=n).astype("float64")
    grid = overlay_cost_grid(returns, target, cost_grid=(0.0, 10.0, 25.0, 50.0))
    finite = [r.overlay_sharpe for r in grid if np.isfinite(r.overlay_sharpe)]
    for lo, hi in zip(finite[1:], finite[:-1], strict=False):
        assert lo <= hi + 1e-9


@pytest.mark.unit
def test_overlay_result_to_dict_is_json_clean() -> None:
    """``OverlayResult.to_dict`` emits only JSON-native scalars."""
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])
    res = overlay_backtest(returns, np.array([1.0, 0.0, 1.0, 0.0]))
    payload = res.to_dict()
    assert set(payload) >= {"overlay_returns", "buyhold_returns", "exposure", "cost_bps"}
    assert isinstance(payload["cost_bps"], float)


# --------------------------------------------------------------------------- #
# select_risk_off_state                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_select_risk_off_state_picks_highest_vol() -> None:
    """The risk-off state is the highest-volatility regime, not a positional one."""
    char = RegimeCharacterization(n_states=3, stats=(_stat(0, 0.2), _stat(1, 0.05), _stat(2, 0.1)))
    assert select_risk_off_state(char) == 0


@pytest.mark.unit
def test_select_risk_off_state_ignores_nan_vol() -> None:
    """An unvisited (NaN-vol) regime never wins over a populated one."""
    char = RegimeCharacterization(n_states=2, stats=(_stat(0, 0.08), _stat(1, float("nan"))))
    assert select_risk_off_state(char) == 0


@pytest.mark.unit
def test_select_risk_off_state_rejects_empty() -> None:
    """An empty characterization has no risk-off state."""
    char = RegimeCharacterization(n_states=0, stats=())
    with pytest.raises(ValidationError, match="no regime stats"):
        select_risk_off_state(char)


# --------------------------------------------------------------------------- #
# walk_forward_backtest                                                       #
# --------------------------------------------------------------------------- #
def _equal_weight(in_sample: pd.DataFrame) -> pd.Series:
    """A trivial equal-weight allocator for engine smoke tests."""
    cols = list(in_sample.columns)
    return pd.Series(1.0 / len(cols), index=cols, dtype="float64")


def _panel(n: int = 120, n_assets: int = 2, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    data = rng.normal(0.0003, 0.01, size=(n, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


@pytest.mark.unit
def test_walk_forward_runs_and_serializes() -> None:
    """A basic walk-forward produces aligned OOS series and a JSON-clean dict."""
    res = walk_forward_backtest(_panel(), _equal_weight, lookback_window=40, rebalance="monthly")
    assert isinstance(res, BacktestResult)
    assert res.n_rebalances >= 1
    assert len(res.oos_returns) == len(res.gross_returns)
    payload = res.to_dict()
    assert set(payload) >= {"oos_returns", "gross_returns", "weights", "turnover", "costs"}


@pytest.mark.unit
def test_walk_forward_validation_branches() -> None:
    """Negative cost, bad cadence, small lookback, and negative purge each raise."""
    panel = _panel()
    with pytest.raises(ValidationError, match="cost_bps must be"):
        walk_forward_backtest(panel, _equal_weight, lookback_window=40, cost_bps=-1.0)
    with pytest.raises(ValidationError, match="unsupported rebalance"):
        walk_forward_backtest(panel, _equal_weight, lookback_window=40, rebalance="weekly")
    with pytest.raises(ValidationError, match="lookback_window"):
        walk_forward_backtest(panel, _equal_weight, lookback_window=1)
    with pytest.raises(ValidationError, match="purge and embargo"):
        walk_forward_backtest(panel, _equal_weight, lookback_window=40, purge=-1)


@pytest.mark.unit
def test_walk_forward_insufficient_history_raises() -> None:
    """A panel too short for a single split raises InsufficientDataError."""
    short = _panel(n=30)
    with pytest.raises(InsufficientDataError):
        walk_forward_backtest(short, _equal_weight, lookback_window=28, rebalance="monthly")
