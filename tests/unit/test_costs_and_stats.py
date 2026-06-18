"""Unit tests for transaction costs and performance statistics.

Covers :class:`regimehmm.backtest.costs.FixedBpsCost` (validation + linearity) and
the scalar summaries in :mod:`regimehmm.backtest.stats` (Sharpe, annualized vol,
turnover, max drawdown) including the degenerate / error branches.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from regimehmm._exceptions import ValidationError
from regimehmm.backtest.costs import FixedBpsCost
from regimehmm.backtest.stats import annualized_vol, max_drawdown, sharpe_ratio, turnover


@pytest.mark.unit
def test_fixed_bps_cost_is_linear_in_turnover() -> None:
    """Cost is ``turnover * bps / 1e4`` and scales linearly."""
    model = FixedBpsCost(bps=10.0)
    assert model.cost(1.0) == pytest.approx(10.0 / 10_000.0)
    assert model.cost(0.5) == pytest.approx(model.cost(1.0) / 2.0)
    assert model.cost(0.0) == 0.0


@pytest.mark.unit
def test_fixed_bps_cost_rejects_negative_bps() -> None:
    """A negative or non-finite bps is rejected at construction."""
    with pytest.raises(ValidationError, match="non-negative"):
        FixedBpsCost(bps=-1.0)
    with pytest.raises(ValidationError, match="non-negative"):
        FixedBpsCost(bps=float("inf"))


@pytest.mark.unit
def test_fixed_bps_cost_rejects_negative_turnover() -> None:
    """A negative turnover passed to ``cost`` is rejected."""
    model = FixedBpsCost(bps=5.0)
    with pytest.raises(ValidationError, match="non-negative"):
        model.cost(-0.1)


@pytest.mark.unit
def test_sharpe_ratio_positive_drift() -> None:
    """A positive-drift series has a positive annualized Sharpe."""
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=500)
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert sr > 0.0
    assert math.isfinite(sr)


@pytest.mark.unit
def test_sharpe_ratio_zero_vol_is_nan() -> None:
    """A flat (zero-vol) series has an undefined Sharpe reported as NaN."""
    flat = pd.Series([0.001] * 50)
    assert math.isnan(sharpe_ratio(flat))


@pytest.mark.unit
def test_annualized_vol_scales_with_sqrt_ppy() -> None:
    """Annualized vol is the per-period std times sqrt(periods_per_year)."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    per_period = float(returns.std(ddof=1))
    assert annualized_vol(returns, periods_per_year=252) == pytest.approx(
        per_period * math.sqrt(252)
    )


@pytest.mark.unit
def test_turnover_half_l1_with_alignment() -> None:
    """Turnover is half the L1 weight change, aligning on the asset union."""
    prev = pd.Series({"A": 1.0})
    new = pd.Series({"B": 1.0})
    # Full rotation A->B: 0.5 * (|0-1| + |1-0|) = 1.0.
    assert turnover(prev, new) == pytest.approx(1.0)
    assert turnover(prev, prev) == pytest.approx(0.0)


@pytest.mark.unit
def test_max_drawdown_is_nonpositive_and_zero_for_monotone() -> None:
    """Max drawdown is <= 0 and exactly 0 for a never-declining series."""
    rising = pd.Series([0.01, 0.01, 0.01])
    assert max_drawdown(rising) == pytest.approx(0.0)

    drop = pd.Series([0.1, -0.5, 0.0])
    mdd = max_drawdown(drop)
    assert mdd < 0.0
    # Wealth peaks at 1.1 then falls to 0.55: drawdown = 0.55/1.1 - 1 = -0.5.
    assert mdd == pytest.approx(-0.5)
