"""Unit tests for the synthetic regime-switch generator and feature builder.

These cover the contract the ENTIRE suite depends on: the generator is seeded and
byte-reproducible, emits a row-stochastic sticky transition matrix and canonical
(ascending mean/vol) state parameters, recovers true per-state volatility, and the
feature builder is strictly causal (no lookahead). No test touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm import ValidationError
from regimehmm.data import (
    RegimeSwitchSeries,
    build_features,
    generate_regime_switch,
)


@pytest.mark.unit
def test_generate_shapes_and_types() -> None:
    """The generator returns aligned returns + ground-truth states + params."""
    sample = generate_regime_switch(500, n_states=3, seed=7)
    assert isinstance(sample, RegimeSwitchSeries)
    assert isinstance(sample.returns, pd.Series)
    assert isinstance(sample.returns.index, pd.DatetimeIndex)
    assert len(sample.returns) == 500
    assert sample.states.shape == (500,)
    assert sample.means.shape == (3,)
    assert sample.vols.shape == (3,)
    assert sample.transmat.shape == (3, 3)
    assert bool(np.isfinite(sample.returns.to_numpy()).all())


@pytest.mark.unit
def test_generate_is_deterministic() -> None:
    """A fixed ``(n_obs, n_states, persistence, seed)`` reproduces byte-for-byte."""
    a = generate_regime_switch(800, n_states=3, persistence=0.96, seed=42)
    b = generate_regime_switch(800, n_states=3, persistence=0.96, seed=42)
    pd.testing.assert_series_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.states, b.states)
    np.testing.assert_array_equal(a.means, b.means)
    np.testing.assert_array_equal(a.vols, b.vols)
    np.testing.assert_array_equal(a.transmat, b.transmat)


@pytest.mark.unit
def test_distinct_seeds_differ() -> None:
    """Different seeds yield different realized paths (not a constant generator)."""
    a = generate_regime_switch(500, n_states=2, seed=1)
    b = generate_regime_switch(500, n_states=2, seed=2)
    assert not np.array_equal(a.returns.to_numpy(), b.returns.to_numpy())


@pytest.mark.unit
@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_transition_matrix_is_row_stochastic(n_states: int) -> None:
    """Each transition row sums to 1 with ``persistence`` on the diagonal."""
    persistence = 0.95
    sample = generate_regime_switch(400, n_states=n_states, persistence=persistence, seed=3)
    transmat = sample.transmat
    np.testing.assert_allclose(transmat.sum(axis=1), np.ones(n_states), atol=1e-12)
    np.testing.assert_allclose(np.diag(transmat), np.full(n_states, persistence), atol=1e-12)
    assert bool((transmat >= 0.0).all())


@pytest.mark.unit
def test_default_state_params_are_canonical() -> None:
    """Default means/vols are in canonical ascending order (label-stable)."""
    sample = generate_regime_switch(300, n_states=4, seed=5)
    assert bool(np.all(np.diff(sample.means) < 0.0)) or bool(np.all(np.diff(sample.vols) > 0.0))
    # Vols are sharply different and strictly ascending (calm ... crisis).
    assert bool(np.all(np.diff(sample.vols) > 0.0))
    assert bool((sample.vols > 0.0).all())


@pytest.mark.unit
def test_true_state_volatility_is_recoverable() -> None:
    """Empirical per-state vol tracks the generating vol (decoder sanity floor)."""
    sample = generate_regime_switch(4000, n_states=2, persistence=0.98, seed=11)
    rets = sample.returns.to_numpy()
    states = sample.states
    emp_vol_low = float(rets[states == 0].std())
    emp_vol_high = float(rets[states == 1].std())
    # High-vol regime is clearly more dispersed than the low-vol regime, and each
    # empirical vol is within 25% of the generating value.
    assert emp_vol_high > emp_vol_low
    assert abs(emp_vol_low - sample.vols[0]) / sample.vols[0] < 0.25
    assert abs(emp_vol_high - sample.vols[1]) / sample.vols[1] < 0.25


@pytest.mark.unit
def test_persistence_makes_states_sticky() -> None:
    """High persistence yields long same-state runs (few transitions)."""
    sample = generate_regime_switch(2000, n_states=2, persistence=0.98, seed=9)
    states = sample.states
    n_switches = int((np.diff(states) != 0).sum())
    # With p=0.98 the expected switch rate is ~2%; assert it stays well under 10%.
    assert n_switches / (len(states) - 1) < 0.10


@pytest.mark.unit
def test_custom_means_and_vols_are_used() -> None:
    """Explicit means/vols override defaults and are stored verbatim."""
    means = (0.001, -0.002)
    vols = (0.005, 0.030)
    sample = generate_regime_switch(300, n_states=2, means=means, vols=vols, seed=7)
    np.testing.assert_array_equal(sample.means, np.asarray(means))
    np.testing.assert_array_equal(sample.vols, np.asarray(vols))


@pytest.mark.unit
def test_single_state_is_absorbing() -> None:
    """``n_states=1`` yields a degenerate ``[[1.0]]`` chain in state 0 throughout."""
    sample = generate_regime_switch(200, n_states=1, seed=7)
    assert sample.transmat.shape == (1, 1)
    np.testing.assert_allclose(sample.transmat, np.array([[1.0]]))
    assert set(np.unique(sample.states)) == {0}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_obs": 0}, "n_obs"),
        ({"n_obs": 10, "n_states": 0}, "n_states"),
        ({"n_obs": 10, "persistence": 0.0}, "persistence"),
        ({"n_obs": 10, "persistence": 1.0}, "persistence"),
        ({"n_obs": 10, "persistence": 1.5}, "persistence"),
        ({"n_obs": 10, "n_states": 2, "means": (0.1,)}, "means"),
        ({"n_obs": 10, "n_states": 2, "vols": (0.1, 0.2, 0.3)}, "vols"),
        ({"n_obs": 10, "n_states": 9}, "default"),
        ({"n_obs": 10, "n_states": 2, "vols": (0.0, 0.02)}, "strictly positive"),
        ({"n_obs": 10, "n_states": 2, "vols": (-0.01, 0.02)}, "strictly positive"),
    ],
)
def test_generate_validation_errors(kwargs: dict[str, object], match: str) -> None:
    """Out-of-domain arguments raise a descriptive ``ValidationError``."""
    with pytest.raises(ValidationError, match=match):
        generate_regime_switch(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_to_dict_is_json_serializable() -> None:
    """``RegimeSwitchSeries.to_dict`` returns plain JSON-friendly primitives."""
    import json

    sample = generate_regime_switch(50, n_states=2, seed=7)
    payload = sample.to_dict()
    text = json.dumps(payload)
    restored = json.loads(text)
    assert set(restored) == {"returns", "states", "means", "vols", "transmat", "meta"}
    assert len(restored["states"]) == 50
    assert restored["meta"]["n_states"] == 2


# --------------------------------------------------------------------------- #
# build_features                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    ("feature_set", "expected_cols"),
    [
        ("returns", ["return"]),
        ("returns_vol", ["return", "realized_vol"]),
        ("returns_vol_macro", ["return", "realized_vol", "macro_trend"]),
    ],
)
def test_feature_columns(feature_set: str, expected_cols: list[str]) -> None:
    """Each feature set produces the documented columns, return first."""
    sample = generate_regime_switch(500, n_states=2, seed=7)
    feat = build_features(sample.returns, feature_set=feature_set)  # type: ignore[arg-type]
    assert list(feat.columns) == expected_cols
    assert next(iter(feat.columns)) == "return"


@pytest.mark.unit
def test_features_have_no_nan_and_drop_incomplete_window() -> None:
    """Leading incomplete-window rows are dropped; survivors are NaN-free."""
    sample = generate_regime_switch(500, n_states=2, seed=7)
    vol_window = 21
    feat = build_features(sample.returns, feature_set="returns_vol", vol_window=vol_window)
    assert not bool(feat.isna().to_numpy().any())
    # Exactly the first (vol_window - 1) rows are dropped for returns_vol.
    assert len(feat) == len(sample.returns) - (vol_window - 1)


@pytest.mark.unit
def test_feature_value_is_causal() -> None:
    """A feature at t is unchanged by perturbing a strictly-future return.

    This is the no-lookahead guard for the emission features: trailing windows at
    time t use returns <= t only, so mutating r[t+k] (k>0) cannot move feature[t].
    """
    sample = generate_regime_switch(400, n_states=2, seed=7)
    base = build_features(sample.returns, feature_set="returns_vol_macro", vol_window=21)

    perturbed_returns = sample.returns.copy()
    # Perturb the LAST observation (the most future point) by a large shock.
    perturbed_returns.iloc[-1] = perturbed_returns.iloc[-1] + 10.0
    perturbed = build_features(perturbed_returns, feature_set="returns_vol_macro", vol_window=21)

    # Every feature row strictly before the final index is identical.
    common = base.index[:-1]
    pd.testing.assert_frame_equal(base.loc[common], perturbed.loc[common])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"feature_set": "bogus"}, "feature_set"),
        ({"vol_window": 0}, "vol_window"),
    ],
)
def test_build_features_validation_errors(kwargs: dict[str, object], match: str) -> None:
    """Unsupported feature set / non-positive window raise ``ValidationError``."""
    sample = generate_regime_switch(100, n_states=2, seed=7)
    with pytest.raises(ValidationError, match=match):
        build_features(sample.returns, **kwargs)  # type: ignore[arg-type]
