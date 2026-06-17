"""Unit tests for the overlay module: validation guards, costs, and serialization.

Fast, fixture-free checks that pin the overlay's input validation, the per-side
cost arithmetic, the ``shift(1)`` chokepoint, the soft/hard exposure map, and the
JSON-serialization of :class:`~regimehmm.backtest.overlay.OverlayResult`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm._exceptions import ValidationError
from regimehmm.backtest.overlay import (
    OverlayResult,
    _safe_float,
    overlay_backtest,
    overlay_cost_grid,
    regime_exposure,
    select_risk_off_state,
    walk_forward_overlay,
)
from regimehmm.regimes.characterize import RegimeCharacterization, RegimeStats


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.5, 1.5),
        (float("nan"), None),
        (float("inf"), None),
        (None, None),
        ("not-a-number", None),
    ],
)
def test_safe_float_scrubs_non_finite_and_uncoercible(
    value: object, expected: float | None
) -> None:
    """``_safe_float`` returns a finite float or ``None`` (NaN/Inf/None/uncoercible)."""
    assert _safe_float(value) == expected if expected is not None else _safe_float(value) is None


def _toy_returns(n: int = 30, *, seed: int = 0, name: str = "SPY") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(scale=0.01, size=n), index=idx, name=name)


# --------------------------------------------------------------------------- #
# regime_exposure                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_regime_exposure_hard_decoder() -> None:
    """A degenerate 0/1 posterior gives a clean on/off exposure series."""
    posterior = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    exposure = regime_exposure(posterior, risk_off_states=(1,), risk_off_exposure=0.0)
    np.testing.assert_array_equal(exposure, np.array([1.0, 0.0, 0.0, 1.0]))


@pytest.mark.unit
def test_regime_exposure_soft_blend_and_partial_risk_off() -> None:
    """The soft posterior blends by risk-off mass, scaled by ``risk_off_exposure``."""
    posterior = np.array([[0.7, 0.3], [0.2, 0.8]])
    # risk_off_exposure=0.0 -> e = 1 - q
    np.testing.assert_allclose(
        regime_exposure(posterior, risk_off_states=(1,), risk_off_exposure=0.0),
        np.array([0.7, 0.2]),
    )
    # risk_off_exposure=0.5 -> e = 1 - 0.5*q
    np.testing.assert_allclose(
        regime_exposure(posterior, risk_off_states=(1,), risk_off_exposure=0.5),
        np.array([1.0 - 0.5 * 0.3, 1.0 - 0.5 * 0.8]),
    )


@pytest.mark.unit
def test_regime_exposure_multiple_risk_off_states() -> None:
    """Risk-off mass sums over all designated states."""
    posterior = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    exposure = regime_exposure(posterior, risk_off_states=(1, 2), risk_off_exposure=0.0)
    np.testing.assert_allclose(exposure, np.array([1.0 - 0.5, 1.0 - 0.9]))


@pytest.mark.unit
@pytest.mark.parametrize("bad_exposure", [-0.1, 1.5, float("nan"), float("inf")])
def test_regime_exposure_rejects_bad_risk_off_exposure(bad_exposure: float) -> None:
    posterior = np.array([[0.5, 0.5]])
    with pytest.raises(ValidationError):
        regime_exposure(posterior, risk_off_states=(1,), risk_off_exposure=bad_exposure)


@pytest.mark.unit
def test_regime_exposure_rejects_non_2d() -> None:
    with pytest.raises(ValidationError):
        regime_exposure(np.array([0.5, 0.5]), risk_off_states=(0,))


@pytest.mark.unit
def test_regime_exposure_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        regime_exposure(np.empty((0, 2)), risk_off_states=(0,))


@pytest.mark.unit
def test_regime_exposure_rejects_non_finite_posterior() -> None:
    with pytest.raises(ValidationError):
        regime_exposure(np.array([[np.nan, 1.0]]), risk_off_states=(1,))


@pytest.mark.unit
def test_regime_exposure_rejects_empty_risk_off() -> None:
    with pytest.raises(ValidationError):
        regime_exposure(np.array([[0.5, 0.5]]), risk_off_states=())


@pytest.mark.unit
@pytest.mark.parametrize("bad_state", [-1, 2, 5])
def test_regime_exposure_rejects_out_of_range_state(bad_state: int) -> None:
    with pytest.raises(ValidationError):
        regime_exposure(np.array([[0.5, 0.5]]), risk_off_states=(bad_state,))


# --------------------------------------------------------------------------- #
# select_risk_off_state (FIX #2: risk-off == argmax(vol), not positional)      #
# --------------------------------------------------------------------------- #
def _stat(state: int, *, mean_return: float, volatility: float) -> RegimeStats:
    """Build a minimal RegimeStats for risk-off selection tests."""
    return RegimeStats(
        state=state,
        frequency=0.5,
        mean_return=mean_return,
        volatility=volatility,
        persistence=0.9,
        expected_duration=10.0,
        max_drawdown=-0.1,
    )


def _characterization(stats: tuple[RegimeStats, ...]) -> RegimeCharacterization:
    return RegimeCharacterization(n_states=len(stats), stats=stats)


@pytest.mark.unit
def test_select_risk_off_is_argmax_vol_not_last_state() -> None:
    """Risk-off is the HIGHEST-VOL regime, even when that is NOT the last (high-mean) state.

    After canonicalization the LAST state is highest-MEAN-return. A high-mean state
    can carry low vol; the risk-off overlay must still target the max-vol regime.
    Here state 2 is the highest mean (canonical last) but state 0 is the highest vol
    — risk-off must select state 0, never the positional last state.
    """
    char = _characterization(
        (
            _stat(0, mean_return=-0.30, volatility=0.40),  # highest vol
            _stat(1, mean_return=0.05, volatility=0.12),
            _stat(2, mean_return=0.25, volatility=0.15),  # canonical last (highest mean)
        )
    )
    assert select_risk_off_state(char) == 0
    # The positional (n_states - 1) choice would have wrongly picked state 2.
    assert select_risk_off_state(char) != char.n_states - 1


@pytest.mark.unit
def test_select_risk_off_matches_numpy_argmax_vol() -> None:
    """``select_risk_off_state`` == ``argmax`` of the per-regime volatilities."""
    vols = [0.10, 0.35, 0.22, 0.08]
    stats = tuple(_stat(i, mean_return=0.01 * i, volatility=v) for i, v in enumerate(vols))
    char = _characterization(stats)
    assert select_risk_off_state(char) == int(np.argmax(np.asarray(vols)))


@pytest.mark.unit
def test_select_risk_off_ignores_nan_vol_unvisited_regime() -> None:
    """An unvisited (NaN-vol) regime never wins risk-off over a populated one."""
    char = _characterization(
        (
            _stat(0, mean_return=-0.1, volatility=0.20),
            _stat(1, mean_return=0.2, volatility=float("nan")),  # unvisited
        )
    )
    assert select_risk_off_state(char) == 0


@pytest.mark.unit
def test_select_risk_off_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        select_risk_off_state(RegimeCharacterization(n_states=0, stats=()))


# --------------------------------------------------------------------------- #
# overlay_backtest                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_overlay_backtest_shift_and_cost_arithmetic() -> None:
    """``shift(1)`` drops the first bar and the per-side cost is exact."""
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    returns = pd.Series([0.01, -0.02, 0.03, 0.04], index=idx)
    target = np.array([1.0, 0.0, 1.0, 1.0])  # decided pre-shift
    result = overlay_backtest(returns, target, cost_bps=10.0)

    # First bar dropped; applied exposure is target shifted by one.
    np.testing.assert_array_equal(result.exposure.to_numpy(), np.array([1.0, 0.0, 1.0]))
    # Gross: applied * return; net subtracts |Δexposure| * 10/1e4 (from flat book).
    # bar1: e=1 (Δ=1), bar2: e=0 (Δ=1), bar3: e=1 (Δ=1)
    expected_cost = np.array([1.0, 1.0, 1.0]) * (10.0 / 10_000.0)
    gross = np.array([1.0, 0.0, 1.0]) * returns.to_numpy()[1:]
    np.testing.assert_allclose(result.overlay_returns.to_numpy(), gross - expected_cost)
    # Total turnover counts every change (including the initial entry from flat).
    assert result.turnover == pytest.approx(3.0)
    # Buy-and-hold leg is just the OOS market.
    np.testing.assert_allclose(result.buyhold_returns.to_numpy(), returns.to_numpy()[1:])


@pytest.mark.unit
def test_overlay_backtest_zero_cost_has_no_charge() -> None:
    returns = _toy_returns(20)
    target = np.ones(20)
    result = overlay_backtest(returns, target, cost_bps=0.0)
    # Full-exposure overlay at zero cost equals buy-and-hold exactly.
    np.testing.assert_allclose(result.overlay_returns.to_numpy(), result.buyhold_returns.to_numpy())
    assert result.overlay_sharpe == pytest.approx(result.buyhold_sharpe, abs=1e-12)


@pytest.mark.unit
def test_overlay_backtest_rejects_negative_cost() -> None:
    with pytest.raises(ValidationError):
        overlay_backtest(_toy_returns(10), np.ones(10), cost_bps=-1.0)


@pytest.mark.unit
def test_overlay_backtest_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        overlay_backtest(_toy_returns(10), np.ones(9))


@pytest.mark.unit
def test_overlay_backtest_rejects_non_1d_exposure() -> None:
    with pytest.raises(ValidationError):
        overlay_backtest(_toy_returns(10), np.ones((10, 1)))


@pytest.mark.unit
def test_overlay_backtest_rejects_non_finite_exposure() -> None:
    target = np.ones(10)
    target[3] = np.nan
    with pytest.raises(ValidationError):
        overlay_backtest(_toy_returns(10), target)


# --------------------------------------------------------------------------- #
# overlay_cost_grid                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_overlay_cost_grid_order_and_length() -> None:
    returns = _toy_returns(40)
    target = (np.arange(40) % 2).astype(float)
    grid = overlay_cost_grid(returns, target, cost_grid=(0.0, 5.0, 10.0))
    assert len(grid) == 3
    assert [r.cost_bps for r in grid] == [0.0, 5.0, 10.0]
    # Sharpe non-increasing in cost.
    sharpes = [r.overlay_sharpe for r in grid]
    assert sharpes[2] <= sharpes[1] + 1e-9 <= sharpes[0] + 2e-9


@pytest.mark.unit
def test_overlay_cost_grid_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        overlay_cost_grid(_toy_returns(10), np.ones(10), cost_grid=())


@pytest.mark.unit
def test_overlay_cost_grid_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        overlay_cost_grid(_toy_returns(10), np.ones(10), cost_grid=(0.0, -1.0))


# --------------------------------------------------------------------------- #
# OverlayResult.to_dict                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_overlay_result_to_dict_is_json_serializable() -> None:
    import json

    returns = _toy_returns(15)
    target = (np.arange(15) % 2).astype(float)
    result = overlay_backtest(returns, target, cost_bps=10.0)
    payload = result.to_dict()
    # Round-trips through JSON (all scalars finite-or-None, keys stringified).
    text = json.dumps(payload)
    restored = json.loads(text)
    assert set(restored) == {
        "overlay_returns",
        "buyhold_returns",
        "exposure",
        "overlay_sharpe",
        "buyhold_sharpe",
        "cost_bps",
        "turnover",
        "meta",
    }
    assert restored["cost_bps"] == 10.0


@pytest.mark.unit
def test_overlay_result_to_dict_scrubs_non_finite() -> None:
    """A NaN/None scalar maps to ``None`` (not a JSON-invalid ``NaN``)."""
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    res = OverlayResult(
        overlay_returns=pd.Series([float("nan")], index=idx[:1]),
        buyhold_returns=pd.Series([0.01], index=idx[:1]),
        exposure=pd.Series([1.0], index=idx[:1]),
        overlay_sharpe=float("nan"),
        buyhold_sharpe=float("inf"),
        cost_bps=10.0,
        turnover=0.0,
    )
    payload = res.to_dict()
    assert payload["overlay_sharpe"] is None
    assert payload["buyhold_sharpe"] is None
    assert payload["overlay_returns"][str(idx[0])] is None


# --------------------------------------------------------------------------- #
# walk_forward_overlay validation                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_walk_forward_overlay_rejects_signal_length_mismatch() -> None:
    returns = _toy_returns(200)
    signal = pd.Series(np.ones(199), index=returns.index[:199])
    with pytest.raises(ValidationError):
        walk_forward_overlay(returns, signal, lookback_window=30)


@pytest.mark.unit
def test_walk_forward_overlay_rejects_out_of_range_signal() -> None:
    returns = _toy_returns(200)
    signal = pd.Series(np.full(200, 1.5), index=returns.index)
    with pytest.raises(ValidationError):
        walk_forward_overlay(returns, signal, lookback_window=30)


@pytest.mark.unit
def test_walk_forward_overlay_identical_index_and_unnamed_returns() -> None:
    """A constant signal scores both legs on the IDENTICAL index (unnamed series ok)."""
    returns = _toy_returns(250, name="")  # falsy name -> falls back to 'asset'
    returns.name = None
    signal = pd.Series(np.full(250, 0.5), index=returns.index)
    result = walk_forward_overlay(
        returns, signal, lookback_window=40, rebalance="monthly", cost_bps=5.0, anchored=True
    )
    assert result.overlay_returns.index.equals(result.buyhold_returns.index)
    assert result.exposure.index.equals(result.overlay_returns.index)
    assert result.meta["engine"] == "walk_forward_backtest"
