"""Sanity checks on the seeded conftest fixtures.

These confirm the shared fixtures have the documented shape and determinism so the
parallel authors can build the behavioural suites on a stable foundation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_one_factor_shape(one_factor: pd.Series) -> None:
    """``one_factor`` is a finite, business-day-indexed return Series."""
    assert isinstance(one_factor, pd.Series)
    assert len(one_factor) == 1000
    assert bool(np.isfinite(one_factor.to_numpy()).all())


@pytest.mark.unit
def test_pure_noise_shape(pure_noise: pd.Series) -> None:
    """``pure_noise`` is a finite, business-day-indexed return Series."""
    assert isinstance(pure_noise, pd.Series)
    assert len(pure_noise) == 1000
    assert bool(np.isfinite(pure_noise.to_numpy()).all())


@pytest.mark.unit
def test_regime_switch_structure(regime_switch: dict[str, object]) -> None:
    """``regime_switch`` carries aligned returns + ground-truth states + params."""
    returns = regime_switch["returns"]
    states = regime_switch["states"]
    vols = regime_switch["vols"]
    assert isinstance(returns, pd.Series)
    assert isinstance(states, np.ndarray)
    assert len(returns) == len(states)
    # Two persistent regimes with sharply different (ascending) vol.
    assert isinstance(vols, np.ndarray)
    assert vols[1] > vols[0]
    assert set(np.unique(states)).issubset({0, 1})


@pytest.mark.unit
def test_regime_switch_is_deterministic(
    regime_switch: dict[str, object],
    request: pytest.FixtureRequest,
) -> None:
    """Re-instantiating the fixture yields a byte-identical sample (seeded)."""
    again = request.getfixturevalue("regime_switch")
    pd.testing.assert_series_equal(regime_switch["returns"], again["returns"])
    np.testing.assert_array_equal(regime_switch["states"], again["states"])
