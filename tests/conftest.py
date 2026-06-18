"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`regimehmm._rng.make_rng`) and
returns pandas/numpy objects, so tests across the suite share identical synthetic
data with known structure. The whole suite runs OFFLINE on these fixtures - no
test touches the network.

- ``one_factor`` - a single-state, well-behaved Gaussian return series (the "no
  regimes" control: a Gaussian HMM should find one dominant state / weak
  switching).
- ``regime_switch`` - a 2/3-state PERSISTENT-vol return series from the synthetic
  generator, carrying its ground-truth hidden states (the core fixture: the HMM
  must recover the persistent high/low-vol regimes, and the timing overlay must
  NOT beat buy-and-hold OOS after costs).
- ``pure_noise`` - i.i.d. Gaussian returns with no persistence (the null: no
  meaningful regime structure to find).

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm._rng import make_rng

_SEED = 20260617


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def one_factor() -> pd.Series:
    """Single-state, well-behaved Gaussian return series (the no-regime control).

    Length ``1000``. A stationary Gaussian return stream with a single (mean, vol)
    - no persistent vol switching - so a fitted HMM should collapse toward one
    dominant state and the timing overlay has nothing to exploit. Indexed by a
    business-day :class:`pandas.DatetimeIndex`.
    """
    gen = make_rng(_SEED)
    n_obs = 1000
    data = gen.normal(loc=0.0003, scale=0.01, size=n_obs)
    index = pd.date_range("2018-01-01", periods=n_obs, freq="B")
    return pd.Series(data, index=index, name="one_factor")


@pytest.fixture
def regime_switch() -> dict[str, object]:
    """Synthetic 2-state persistent-vol return series with ground-truth states.

    The core fixture. A sticky two-state Markov chain (calm low-vol regime vs.
    turbulent high-vol regime) emits Gaussian returns, mirroring
    :func:`regimehmm.data.generate_regime_switch`. Returns a dict with:

    - ``returns`` (``pd.Series``): the generated return series;
    - ``states`` (``np.ndarray``): the ground-truth hidden-state path (canonical
      order, ``0`` = low-vol); the model never sees this at fit time;
    - ``means``/``vols`` (``np.ndarray``): the per-state generating parameters;
    - ``transmat`` (``np.ndarray``): the generating transition matrix.

    Built inline (not via the still-stubbed generator) so the fixture is usable
    while ``regimehmm.data.generate_regime_switch`` is being implemented; the
    parameters match the generator's canonical defaults.
    """
    gen = make_rng(_SEED + 1)
    n_obs = 1500
    persistence = 0.97
    means = np.array([0.0008, -0.0010], dtype="float64")  # canonical: ascending mean
    vols = np.array([0.006, 0.020], dtype="float64")  # sharply different, ascending
    transmat = np.array(
        [[persistence, 1.0 - persistence], [1.0 - persistence, persistence]],
        dtype="float64",
    )

    states = np.empty(n_obs, dtype=np.intp)
    returns = np.empty(n_obs, dtype="float64")
    state = 0
    for t in range(n_obs):
        if t > 0:
            state = int(gen.choice(2, p=transmat[state]))
        states[t] = state
        returns[t] = gen.normal(loc=means[state], scale=vols[state])

    index = pd.date_range("2010-01-01", periods=n_obs, freq="B")
    return {
        "returns": pd.Series(returns, index=index, name="regime_switch"),
        "states": states,
        "means": means,
        "vols": vols,
        "transmat": transmat,
    }


@pytest.fixture
def pure_noise() -> pd.Series:
    """I.i.d. Gaussian return series with no persistence (the null).

    Length ``1000``. Independent draws with a single (mean, vol) and NO Markov
    persistence, so there is no regime structure to recover. Indexed by a
    business-day :class:`pandas.DatetimeIndex`.
    """
    gen = make_rng(_SEED + 2)
    n_obs = 1000
    data = gen.standard_normal(n_obs) * 0.012
    index = pd.date_range("2018-01-01", periods=n_obs, freq="B")
    return pd.Series(data, index=index, name="pure_noise")
