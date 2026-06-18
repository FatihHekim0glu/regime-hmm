"""End-to-end no-lookahead regression for the public entrypoint.

The online filter is unit-tested for prefix-determinism in isolation; this module
pins the property end-to-end through :func:`regimehmm.run_regime_analysis`. We
future-perturb the returns strictly AFTER a cut date, re-run the FULL pipeline, and
assert that BOTH the genuinely-OOS overlay returns AND the per-fold OOS regime
labels on the pre-cut portion are byte-identical. Because every walk-forward fold
refits the scaler + HMM on its TRAIN window only and decodes the upcoming window
with the ONLINE FILTER (data <= t), no future bar can move any OOS quantity at or
before the cut.

The ``leaky`` sensitivity arm reproduces the OLD in-sample flow (a single
full-window scaler + HMM fit, then the overlay scored on the full window) and shows
that future-perturbation DOES move the pre-cut overlay returns - proving this test
is a genuine invariant, not a tautology that would pass for any implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm import run_regime_analysis
from regimehmm._rng import make_rng
from regimehmm.backtest.overlay import overlay_backtest, regime_exposure
from regimehmm.data import build_features
from regimehmm.hmm.em import fit_hmm
from regimehmm.hmm.filter import online_filter
from regimehmm.regimes.canonicalize import canonicalize_model

pytestmark = pytest.mark.integration


def _persistent_returns(n_obs: int, seed: int) -> pd.Series:
    """A sticky two-state persistent-vol return series (a compact regime_switch)."""
    gen = make_rng(seed)
    persistence = 0.97
    means = np.array([0.0008, -0.0010], dtype="float64")
    vols = np.array([0.006, 0.020], dtype="float64")
    transmat = np.array(
        [[persistence, 1.0 - persistence], [1.0 - persistence, persistence]],
        dtype="float64",
    )
    rets = np.empty(n_obs, dtype="float64")
    state = 0
    for t in range(n_obs):
        if t > 0:
            state = int(gen.choice(2, p=transmat[state]))
        rets[t] = gen.normal(loc=means[state], scale=vols[state])
    idx = pd.date_range("2012-01-01", periods=n_obs, freq="B")
    return pd.Series(rets, index=idx, name="SPY")


def _future_perturb(returns: pd.Series, cut: int, *, seed: int) -> pd.Series:
    """Return a copy whose values strictly AFTER ``cut`` are perturbed."""
    rng = np.random.default_rng(seed)
    perturbed = returns.copy()
    n = len(returns)
    perturbed.iloc[cut + 1 :] = perturbed.iloc[cut + 1 :] + rng.normal(scale=0.05, size=n - cut - 1)
    return perturbed


def _leaky_overlay_returns(returns: pd.Series, *, n_states: int, seed: int) -> pd.Series:
    """The OLD in-sample flow: ONE full-window scaler + HMM fit, overlay on the full window.

    This is exactly the leakage the fix removed - the scaler and HMM see the WHOLE
    sample, so the filtered exposure at every bar depends (through the fit) on future
    returns. Used only as the sensitivity arm that MUST trip the no-lookahead
    invariant.
    """
    features = build_features(returns, feature_set="returns")
    aligned = returns.reindex(features.index)
    raw = features.to_numpy(dtype="float64")
    mean = raw.mean(axis=0, keepdims=True)
    std = raw.std(axis=0, ddof=0, keepdims=True)
    std = np.where(std > 0.0, std, 1.0)
    scaled = (raw - mean) / std
    model = canonicalize_model(fit_hmm(scaled, n_states, n_restarts=2, max_iter=40, seed=seed))
    posterior = online_filter(model, scaled)
    # Positional risk-off (the old, buggy selection) - immaterial to the leakage arm.
    target = regime_exposure(posterior, risk_off_states=(model.n_states - 1,))
    return overlay_backtest(aligned, target, cost_bps=10.0).overlay_returns


def test_run_regime_analysis_oos_is_future_perturbation_invariant() -> None:
    """Perturbing post-cut returns leaves the OOS overlay returns + OOS labels on/before cut identical.

    Runs the FULL :func:`run_regime_analysis` twice (base vs future-perturbed) and
    asserts the genuinely-OOS overlay return series AND the descriptive/OOS regime
    map agree bar-for-bar up to the cut. This is the end-to-end no-lookahead
    guarantee the leakage fix restores.
    """
    n_obs = 500
    base_returns = _persistent_returns(n_obs, seed=4242)
    cut = int(n_obs * 0.75)
    cut_label = base_returns.index[cut]
    perturbed_returns = _future_perturb(base_returns, cut, seed=7)

    base = run_regime_analysis(
        base_returns, n_states=2, feature_set="returns", cost_bps=10.0, seed=7
    )
    after = run_regime_analysis(
        perturbed_returns, n_states=2, feature_set="returns", cost_bps=10.0, seed=7
    )

    # 1. OOS overlay returns on/before the cut are byte-identical.
    base_overlay = base.overlay_returns.loc[:cut_label]
    after_overlay = after.overlay_returns.loc[:cut_label]
    assert len(base_overlay) > 0, "expected a non-trivial pre-cut OOS overlay window"
    pd.testing.assert_series_equal(base_overlay, after_overlay)

    # 2. OOS buy-and-hold returns on/before the cut are byte-identical too.
    pd.testing.assert_series_equal(
        base.buyhold_returns.loc[:cut_label], after.buyhold_returns.loc[:cut_label]
    )

    # 3. The per-fold OOS regime labels (the no-lookahead regime MAP) on/before the
    #    cut are unchanged - each was decoded with a TRAIN-only fit, so the online
    #    filter never peeked ahead. These are the genuinely-OOS labels, distinct from
    #    the descriptive in-sample ``states`` (which legitimately see the whole
    #    sample for the figure and so are NOT expected to be invariant).
    base_oos_labels = base.oos_states.loc[:cut_label]
    after_oos_labels = after.oos_states.loc[:cut_label]
    assert len(base_oos_labels) > 0, "expected non-trivial pre-cut OOS regime labels"
    pd.testing.assert_series_equal(base_oos_labels, after_oos_labels)


def test_leaky_full_sample_fit_trips_the_no_lookahead_invariant() -> None:
    """SENSITIVITY ARM: the OLD full-sample-fit overlay is NOT future-perturbation invariant.

    Injecting the leaky in-sample flow (a single full-window scaler + HMM fit, then
    overlay on the full window) MUST move the pre-cut overlay returns when the
    post-cut returns are perturbed - proving the invariant above is genuine and not a
    tautology that would pass for any implementation.
    """
    n_obs = 500
    base_returns = _persistent_returns(n_obs, seed=4242)
    cut = int(n_obs * 0.75)
    perturbed_returns = _future_perturb(base_returns, cut, seed=7)

    base_leaky = _leaky_overlay_returns(base_returns, n_states=2, seed=7)
    after_leaky = _leaky_overlay_returns(perturbed_returns, n_states=2, seed=7)
    cut_label = base_leaky.index[base_leaky.index.get_indexer([base_returns.index[cut]])[0]]

    base_prefix = base_leaky.loc[:cut_label]
    after_prefix = after_leaky.loc[:cut_label]

    # The leaky flow's pre-cut overlay returns MUST differ (the full-window fit saw
    # the perturbed future, so the filtered exposure before the cut changed).
    assert not np.allclose(
        base_prefix.to_numpy(dtype="float64"),
        after_prefix.to_numpy(dtype="float64"),
    ), "leaky full-sample-fit overlay was unexpectedly invariant - the test would be a tautology"
