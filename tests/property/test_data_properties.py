"""Property-based tests for the synthetic data layer.

Hypothesis-driven invariants that must hold for ANY admissible parameters:

- the generator is a pure function of its seed (determinism / reproducibility);
- its transition matrix is always row-stochastic;
- the feature builder is strictly causal - perturbing a strictly-future return
  never changes an earlier feature row (the no-lookahead guard, mirrored from the
  online-filter discipline that governs the rest of the project).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from regimehmm.data import build_features, generate_regime_switch


@pytest.mark.property
@given(
    n_obs=st.integers(min_value=20, max_value=400),
    n_states=st.integers(min_value=1, max_value=4),
    persistence=st.floats(min_value=0.5, max_value=0.999),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=60, deadline=None)
def test_generator_is_deterministic(
    n_obs: int, n_states: int, persistence: float, seed: int
) -> None:
    """The generator output is fully determined by its arguments (a pure function)."""
    a = generate_regime_switch(n_obs, n_states=n_states, persistence=persistence, seed=seed)
    b = generate_regime_switch(n_obs, n_states=n_states, persistence=persistence, seed=seed)
    np.testing.assert_array_equal(a.returns.to_numpy(), b.returns.to_numpy())
    np.testing.assert_array_equal(a.states, b.states)


@pytest.mark.property
@given(
    n_states=st.integers(min_value=1, max_value=4),
    persistence=st.floats(min_value=0.5, max_value=0.999),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=60, deadline=None)
def test_transition_rows_are_stochastic(n_states: int, persistence: float, seed: int) -> None:
    """Every transition row is non-negative and sums to 1."""
    sample = generate_regime_switch(100, n_states=n_states, persistence=persistence, seed=seed)
    transmat = sample.transmat
    assert bool((transmat >= 0.0).all())
    np.testing.assert_allclose(transmat.sum(axis=1), np.ones(n_states), atol=1e-12)


@pytest.mark.property
@given(
    n_obs=st.integers(min_value=80, max_value=300),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=None)
def test_states_stay_in_range(n_obs: int, seed: int) -> None:
    """Ground-truth states always fall within ``[0, n_states)``."""
    n_states = 3
    sample = generate_regime_switch(n_obs, n_states=n_states, seed=seed)
    assert sample.states.min() >= 0
    assert sample.states.max() < n_states


@pytest.mark.property
@given(
    n_obs=st.integers(min_value=60, max_value=300),
    shock=st.floats(min_value=1.0, max_value=50.0),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=None)
def test_features_are_causal_under_future_perturbation(n_obs: int, shock: float, seed: int) -> None:
    """Perturbing the final (most-future) return leaves every earlier feature row fixed.

    Trailing windows at t use returns <= t only, so feature[t] is invariant to any
    change at t' > t. This is the data-layer counterpart to the online-filter
    no-lookahead guarantee.
    """
    sample = generate_regime_switch(n_obs, n_states=2, seed=seed)
    base = build_features(sample.returns, feature_set="returns_vol", vol_window=10)

    perturbed = sample.returns.copy()
    perturbed.iloc[-1] = perturbed.iloc[-1] + shock
    after = build_features(perturbed, feature_set="returns_vol", vol_window=10)

    common = base.index[:-1]
    pd.testing.assert_frame_equal(base.loc[common], after.loc[common])
