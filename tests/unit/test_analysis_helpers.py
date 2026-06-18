"""Unit tests for the pure helpers in :mod:`regimehmm.analysis`.

These cover the cheap, side-effect-free pieces of the orchestration module (the
JSON-clean float coercion, the train-only scaler, the OOS lookback sizer, and the
cross-trial Sharpe-variance estimator) without running the full fit pipeline,
which the integration suite covers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm.analysis import (
    _oos_lookback_window,
    _safe_float,
    _scale_train_only,
    _trial_sharpe_variance,
)


@pytest.mark.unit
def test_safe_float_scrubs_non_finite() -> None:
    """NaN, Inf and unparseable values map to None; finite values pass through."""
    assert _safe_float(1.5) == 1.5
    assert _safe_float(np.float64(2.0)) == 2.0
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float("not-a-number") is None
    assert _safe_float(None) is None


@pytest.mark.unit
def test_scale_train_only_standardizes_columns() -> None:
    """Each column is centred to ~0 mean and unit (population) variance."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(5.0, 3.0, size=(200, 2)), columns=["a", "b"])
    scaled = _scale_train_only(frame)
    assert scaled.shape == (200, 2)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(scaled.std(axis=0, ddof=0), 1.0, atol=1e-9)


@pytest.mark.unit
def test_scale_train_only_handles_zero_variance_column() -> None:
    """A constant column is divided by 1.0, never producing NaNs."""
    frame = pd.DataFrame({"const": [2.0] * 50, "vary": np.arange(50.0)})
    scaled = _scale_train_only(frame)
    assert np.isfinite(scaled).all()
    np.testing.assert_allclose(scaled[:, 0], 0.0, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("n_obs", [200, 800, 1500, 3000])
def test_oos_lookback_window_leaves_room_for_oos(n_obs: int) -> None:
    """The lookback is at least 60 bars and never swallows the whole panel."""
    lookback = _oos_lookback_window(n_obs)
    assert lookback >= 60
    assert lookback <= n_obs


@pytest.mark.unit
def test_trial_sharpe_variance_is_non_negative() -> None:
    """The cross-trial Sharpe variance is a finite, non-negative number."""
    rng = np.random.default_rng(1)
    overlay = pd.Series(rng.normal(0.0005, 0.01, size=400))
    var = _trial_sharpe_variance(overlay, (0.0, 5.0, 10.0, 20.0))
    assert var >= 0.0
    assert np.isfinite(var)


@pytest.mark.unit
def test_trial_sharpe_variance_degenerate_returns_zero() -> None:
    """A too-short series falls back to the degenerate 0.0 variance."""
    assert _trial_sharpe_variance(pd.Series([0.01]), (0.0, 10.0)) == 0.0
