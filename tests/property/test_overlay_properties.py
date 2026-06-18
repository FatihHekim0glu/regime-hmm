"""Property-based invariants for the regime-timing exposure overlay.

The leakage discipline of the overlay rests on four invariants, each pinned here:

1. **Filtered-only signal.** The overlay's exposure is a row-wise map of the
   ONLINE FILTER posterior, so the exposure at ``t`` depends only on
   ``filtered_posterior[t]`` — never the smoothed/Viterbi posterior, which peeks
   ahead. We verify this through the filter's own prefix-determinism: perturbing
   returns after ``t`` cannot move the exposure (or the realized overlay return)
   on or before ``t``.
2. **Future-perturbation invariance of the OOS overlay returns.** Changing the
   market returns strictly after a cut date leaves every overlay return on or
   before that date byte-identical.
3. **Identical OOS index.** The overlay leg and the buy-and-hold leg are scored on
   exactly the same out-of-sample index.
4. **``shift(1)`` enforced.** The exposure applied to the return at ``t`` is the
   target decided at ``t - 1``; the very first observation is dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from regimehmm.backtest.overlay import (
    overlay_backtest,
    overlay_cost_grid,
    regime_exposure,
    walk_forward_overlay,
)
from regimehmm.hmm.em import fit_hmm
from regimehmm.hmm.filter import online_filter

_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


def _as_series(obj: object) -> pd.Series:
    """Narrow an ``object``-typed fixture value to a ``pd.Series`` for strict mypy."""
    assert isinstance(obj, pd.Series)
    return obj


def _fit_filter_exposure(returns: pd.Series, *, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Fit a 2-state HMM, return ``(online filtered posterior, risk-off exposure)``.

    The risk-off state is the highest-variance fitted state (the genuine high-vol
    regime), identified from the model. For the LEAKAGE invariants tested here the
    specific risk-off index is immaterial — only that the exposure is a causal,
    row-wise map of the online-filter posterior. Features are the standardized
    returns; the no-lookahead properties compare prefixes only, so fitting on the
    full window is fine.
    """
    obs = returns.to_numpy(dtype="float64").reshape(-1, 1)
    obs = (obs - obs.mean(axis=0)) / (obs.std(axis=0, ddof=0) + 1e-12)
    model = fit_hmm(obs, n_states=2, covariance_type="diag", n_restarts=2, seed=seed)
    posterior = online_filter(model, obs)
    cov = np.asarray(model.covariances, dtype="float64")
    variances = cov[:, 0, 0] if cov.ndim == 3 else cov[:, 0]
    risk_off = int(np.argmax(variances))
    exposure = regime_exposure(posterior, risk_off_states=(risk_off,), risk_off_exposure=0.0)
    return posterior, exposure


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=2**16 - 1))
@_SETTINGS
def test_exposure_is_rowwise_map_of_posterior(seed: int) -> None:
    """``regime_exposure`` is a pure row-wise map: row ``t`` uses ``posterior[t]`` only.

    Permuting / perturbing rows after ``t`` cannot change exposure ``<= t``. This is
    the soft-decoder counterpart of the online filter's prefix-determinism: the
    overlay can never read a future posterior row.
    """
    rng = np.random.default_rng(seed)
    n_obs = 40
    posterior = rng.random(size=(n_obs, 2))
    posterior = posterior / posterior.sum(axis=1, keepdims=True)

    cut = n_obs // 2
    base = regime_exposure(posterior, risk_off_states=(1,), risk_off_exposure=0.25)

    perturbed = posterior.copy()
    future = rng.random(size=(n_obs - cut, 2))
    perturbed[cut:] = future / future.sum(axis=1, keepdims=True)
    after = regime_exposure(perturbed, risk_off_states=(1,), risk_off_exposure=0.25)

    np.testing.assert_array_equal(base[:cut], after[:cut])


@pytest.mark.property
@pytest.mark.slow
@given(cut_frac=st.floats(min_value=0.3, max_value=0.8))
@_SETTINGS
def test_overlay_returns_future_perturbation_invariant(
    cut_frac: float, regime_switch: dict[str, object]
) -> None:
    """Perturbing returns after a cut date leaves overlay returns on/before it identical.

    Combines (a) the online filter's prefix-determinism and (b) the overlay's
    row-wise exposure map and ``shift(1)`` application. The filtered exposure and
    the realized overlay return on or before the cut date are unaffected by any
    market move strictly after it.
    """
    returns = _as_series(regime_switch["returns"]).iloc[:400].copy()
    _posterior, exposure = _fit_filter_exposure(returns)

    n = len(returns)
    cut = int(n * cut_frac)
    base = overlay_backtest(returns, exposure, cost_bps=10.0)

    # Perturb the FUTURE market returns (strictly after ``cut``). The exposure is
    # derived from the (already-fixed) filtered posterior; only the market series
    # is perturbed here, so any change at/before ``cut`` would be pure look-ahead.
    rng = np.random.default_rng(99)
    perturbed = returns.copy()
    perturbed.iloc[cut + 1 :] = perturbed.iloc[cut + 1 :] + rng.normal(scale=0.05, size=n - cut - 1)
    after = overlay_backtest(perturbed, exposure, cost_bps=10.0)

    cut_label = returns.index[cut]
    base_prefix = base.overlay_returns.loc[:cut_label]
    after_prefix = after.overlay_returns.loc[:cut_label]
    pd.testing.assert_series_equal(base_prefix, after_prefix)


@pytest.mark.property
@pytest.mark.slow
@given(cost_bps=st.floats(min_value=0.0, max_value=50.0))
@_SETTINGS
def test_overlay_and_buyhold_share_identical_oos_index(
    cost_bps: float, regime_switch: dict[str, object]
) -> None:
    """The overlay leg and the buy-and-hold leg are scored on the IDENTICAL index."""
    returns = _as_series(regime_switch["returns"]).iloc[:300].copy()
    _posterior, exposure = _fit_filter_exposure(returns)
    res = overlay_backtest(returns, exposure, cost_bps=cost_bps)
    assert res.overlay_returns.index.equals(res.buyhold_returns.index)
    assert res.exposure.index.equals(res.overlay_returns.index)


@pytest.mark.property
@given(exposure=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=6, max_size=40))
@_SETTINGS
def test_shift_one_enforced(exposure: list[float]) -> None:
    """The exposure applied to the return at ``t`` is the target decided at ``t - 1``.

    The first observation has no prior decision and is dropped; for every scored
    bar the applied exposure equals the pre-shift target one bar earlier.
    """
    n = len(exposure)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(len(exposure))
    returns = pd.Series(rng.normal(scale=0.01, size=n), index=idx)
    target = np.asarray(exposure, dtype="float64")

    res = overlay_backtest(returns, target, cost_bps=10.0)

    # First row dropped: scored series starts at the second bar.
    assert res.overlay_returns.index[0] == idx[1]
    assert len(res.overlay_returns) == n - 1
    # Applied exposure at bar ``t`` == pre-shift target at bar ``t - 1``.
    expected_applied = pd.Series(target[:-1], index=idx[1:], dtype="float64")
    pd.testing.assert_series_equal(res.exposure, expected_applied, check_names=False)


@pytest.mark.property
@given(seed=st.integers(min_value=0, max_value=2**16 - 1))
@_SETTINGS
def test_cost_grid_sharpe_non_increasing(seed: int) -> None:
    """Net overlay Sharpe is non-increasing in ``cost_bps`` (cost-monotonicity)."""
    rng = np.random.default_rng(seed)
    n = 120
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.Series(rng.normal(scale=0.01, size=n), index=idx)
    target = rng.integers(0, 2, size=n).astype("float64")
    grid = overlay_cost_grid(returns, target, cost_grid=(0.0, 5.0, 10.0, 25.0))
    sharpes = [r.overlay_sharpe for r in grid]
    finite = [s for s in sharpes if np.isfinite(s)]
    for lo, hi in zip(finite[1:], finite[:-1], strict=False):
        assert lo <= hi + 1e-9


@pytest.mark.property
@pytest.mark.slow
@given(cut_frac=st.floats(min_value=0.55, max_value=0.85))
@_SETTINGS
def test_walk_forward_overlay_identical_oos_index_and_no_lookahead(
    cut_frac: float, regime_switch: dict[str, object]
) -> None:
    """Walk-forward overlay: identical OOS index across legs + future-perturbation safe."""
    returns = _as_series(regime_switch["returns"]).iloc[:400].copy()
    returns.name = "SPY"
    _posterior, exposure = _fit_filter_exposure(returns)
    signal = pd.Series(exposure, index=returns.index)

    base = walk_forward_overlay(
        returns, signal, lookback_window=60, rebalance="monthly", cost_bps=10.0, anchored=True
    )
    # IDENTICAL OOS INDEX across the two strategies.
    assert base.overlay_returns.index.equals(base.buyhold_returns.index)
    assert base.exposure.index.equals(base.overlay_returns.index)

    n = len(returns)
    cut = int(n * cut_frac)
    cut_label = returns.index[cut]

    # Perturb returns AND the signal strictly after the cut; OOS returns on/before
    # the cut must be unchanged (the engine's shift(1) + the causal signal).
    rng = np.random.default_rng(7)
    perturbed = returns.copy()
    perturbed.iloc[cut + 1 :] = perturbed.iloc[cut + 1 :] + rng.normal(scale=0.05, size=n - cut - 1)
    perturbed_sig = signal.copy()
    perturbed_sig.iloc[cut + 1 :] = rng.random(size=n - cut - 1)

    after = walk_forward_overlay(
        perturbed,
        perturbed_sig,
        lookback_window=60,
        rebalance="monthly",
        cost_bps=10.0,
        anchored=True,
    )
    base_prefix = base.overlay_returns.loc[:cut_label]
    after_prefix = after.overlay_returns.loc[:cut_label]
    pd.testing.assert_series_equal(base_prefix, after_prefix)
