"""Unit + property tests for per-regime characterization.

These verify the descriptive statistics in :mod:`regimehmm.regimes.characterize`
on the seeded ``regime_switch`` fixture (the HMM must recover the sharply different
low/high-vol regimes) and the relabeling-invariance contract: characterizing a
decoded series and its canonically-relabeled twin yields the same per-regime stats
once both are read in canonical order.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimehmm._constants import PERIODS_PER_YEAR
from regimehmm._exceptions import ValidationError
from regimehmm.backtest.stats import max_drawdown
from regimehmm.hmm.filter import HMMModel
from regimehmm.regimes.canonicalize import (
    canonical_order,
    canonicalize_model,
    relabel_states,
)
from regimehmm.regimes.characterize import (
    RegimeCharacterization,
    RegimeStats,
    characterize_regimes,
)


def _model_from_transmat(transmat: np.ndarray, n_features: int = 1) -> HMMModel:
    """Build a minimal HMMModel carrying just the transition matrix we need."""
    transmat = np.asarray(transmat, dtype="float64")
    n_states = transmat.shape[0]
    return HMMModel(
        startprob=np.full(n_states, 1.0 / n_states, dtype="float64"),
        transmat=transmat,
        means=np.zeros((n_states, n_features), dtype="float64"),
        covariances=np.ones((n_states, n_features), dtype="float64"),
        covariance_type="diag",
        log_likelihood=-1.0,
        n_iter=1,
        converged=True,
    )


@pytest.mark.unit
def test_characterize_recovers_regime_stats_on_fixture(
    regime_switch: dict[str, object],
) -> None:
    """On the ground-truth states, regime 1 is higher-vol than regime 0."""
    returns = regime_switch["returns"]
    states = np.asarray(regime_switch["states"], dtype="float64")
    transmat = np.asarray(regime_switch["transmat"], dtype="float64")
    model = _model_from_transmat(transmat)

    char = characterize_regimes(model, states, returns)
    assert isinstance(char, RegimeCharacterization)
    assert char.n_states == 2
    assert tuple(s.state for s in char.stats) == (0, 1)

    low, high = char.stats
    # high-vol regime has materially larger annualized volatility.
    assert high.volatility > low.volatility
    # frequencies sum to 1 (every observation assigned to exactly one regime).
    assert math.isclose(low.frequency + high.frequency, 1.0, abs_tol=1e-12)
    # persistence matches the generating transition matrix.
    assert math.isclose(low.persistence, transmat[0, 0], abs_tol=1e-12)
    assert math.isclose(high.persistence, transmat[1, 1], abs_tol=1e-12)
    # expected duration is 1 / (1 - persistence).
    assert math.isclose(low.expected_duration, 1.0 / (1.0 - transmat[0, 0]), rel_tol=1e-12)
    # drawdowns are non-positive.
    assert low.max_drawdown <= 0.0
    assert high.max_drawdown <= 0.0


@pytest.mark.unit
def test_characterize_annualizes_mean_and_vol() -> None:
    """Mean scales by ppy and vol by sqrt(ppy) vs the per-period statistics."""
    rng = np.random.default_rng(0)
    n = 400
    ret = rng.normal(0.001, 0.01, size=n)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.Series(ret, index=index)
    states = np.zeros(n, dtype="float64")
    transmat = np.array([[0.9, 0.1], [0.1, 0.9]], dtype="float64")
    model = _model_from_transmat(transmat)

    char = characterize_regimes(model, states, returns)
    s0 = char.stats[0]
    assert math.isclose(s0.mean_return, float(ret.mean()) * PERIODS_PER_YEAR, rel_tol=1e-9)
    assert math.isclose(
        s0.volatility, float(np.std(ret, ddof=1)) * math.sqrt(PERIODS_PER_YEAR), rel_tol=1e-9
    )
    # the unoccupied regime 1 has NaN return stats but a defined persistence.
    s1 = char.stats[1]
    assert math.isnan(s1.mean_return)
    assert math.isnan(s1.volatility)
    assert s1.frequency == 0.0
    assert math.isclose(s1.persistence, 0.9, abs_tol=1e-12)


@pytest.mark.unit
def test_characterize_within_regime_drawdown_matches_stats_helper() -> None:
    """The reported per-regime drawdown equals max_drawdown of the in-regime stream."""
    index = pd.date_range("2021-01-01", periods=6, freq="B")
    returns = pd.Series([0.01, -0.05, 0.02, -0.10, 0.03, 0.04], index=index)
    states = np.array([0, 0, 1, 1, 0, 1], dtype="float64")
    model = _model_from_transmat(np.array([[0.8, 0.2], [0.3, 0.7]]))

    char = characterize_regimes(model, states, returns)
    regime0 = pd.Series([0.01, -0.05, 0.03])
    regime1 = pd.Series([0.02, -0.10, 0.04])
    assert math.isclose(char.stats[0].max_drawdown, max_drawdown(regime0), rel_tol=1e-12)
    assert math.isclose(char.stats[1].max_drawdown, max_drawdown(regime1), rel_tol=1e-12)


@pytest.mark.unit
def test_characterize_to_dict_is_json_friendly(
    regime_switch: dict[str, object],
) -> None:
    """The characterization serializes to plain dicts/lists/scalars."""
    returns = regime_switch["returns"]
    states = np.asarray(regime_switch["states"], dtype="float64")
    model = _model_from_transmat(np.asarray(regime_switch["transmat"], dtype="float64"))
    payload = characterize_regimes(model, states, returns).to_dict()
    assert payload["n_states"] == 2
    assert isinstance(payload["stats"], list)
    assert {
        "state",
        "frequency",
        "mean_return",
        "volatility",
        "persistence",
        "expected_duration",
        "max_drawdown",
    } <= set(payload["stats"][0].keys())
    assert payload["meta"]["n_obs"] == len(states)


@pytest.mark.unit
def test_characterize_rejects_length_mismatch() -> None:
    """Misaligned states and returns raise ValidationError."""
    returns = pd.Series([0.01, 0.02, -0.01])
    model = _model_from_transmat(np.array([[0.9, 0.1], [0.1, 0.9]]))
    with pytest.raises(ValidationError):
        characterize_regimes(model, np.array([0.0, 1.0]), returns)


@pytest.mark.unit
def test_characterize_rejects_out_of_range_label() -> None:
    """A label >= n_states raises ValidationError."""
    returns = pd.Series([0.01, 0.02, -0.01])
    model = _model_from_transmat(np.array([[0.9, 0.1], [0.1, 0.9]]))
    with pytest.raises(ValidationError):
        characterize_regimes(model, np.array([0.0, 1.0, 2.0]), returns)


@pytest.mark.unit
def test_characterize_rejects_non_square_transmat() -> None:
    """A non-square transition matrix raises ValidationError."""
    returns = pd.Series([0.01, 0.02])
    bad = HMMModel(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]),
        means=np.zeros((2, 1)),
        covariances=np.ones((2, 1)),
        covariance_type="diag",
        log_likelihood=-1.0,
        n_iter=1,
        converged=True,
    )
    with pytest.raises(ValidationError):
        characterize_regimes(bad, np.array([0.0, 1.0]), returns)


@pytest.mark.unit
def test_characterize_rejects_non_series_returns() -> None:
    """A non-Series ``returns`` raises ValidationError."""
    model = _model_from_transmat(np.array([[0.9, 0.1], [0.1, 0.9]]))
    with pytest.raises(ValidationError):
        characterize_regimes(model, np.array([0.0, 1.0]), np.array([0.01, 0.02]))  # type: ignore[arg-type]


@pytest.mark.unit
def test_characterize_rejects_non_1d_states() -> None:
    """A 2-D states array raises ValidationError."""
    returns = pd.Series([0.01, 0.02])
    model = _model_from_transmat(np.array([[0.9, 0.1], [0.1, 0.9]]))
    with pytest.raises(ValidationError):
        characterize_regimes(model, np.array([[0.0, 1.0]]), returns)


@pytest.mark.unit
def test_characterize_rejects_non_integer_states() -> None:
    """A fractional decoded label raises ValidationError."""
    returns = pd.Series([0.01, 0.02])
    model = _model_from_transmat(np.array([[0.9, 0.1], [0.1, 0.9]]))
    with pytest.raises(ValidationError):
        characterize_regimes(model, np.array([0.0, 0.5]), returns)


@pytest.mark.unit
def test_regime_stats_is_frozen() -> None:
    """RegimeStats is immutable (frozen dataclass)."""
    s = RegimeStats(
        state=0,
        frequency=0.5,
        mean_return=0.1,
        volatility=0.2,
        persistence=0.9,
        expected_duration=10.0,
        max_drawdown=-0.1,
    )
    with pytest.raises(AttributeError):
        s.state = 1  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Property test: relabeling invariance (core correctness guarantee)           #
# --------------------------------------------------------------------------- #


@pytest.mark.property
@given(perm=st.permutations([0, 1, 2]))
def test_characterization_is_relabeling_invariant(perm: list[int]) -> None:
    """Permuting raw labels then relabeling to canonical recovers identical stats.

    Build a decoded series and a transition matrix in canonical order. Scramble
    BOTH by an arbitrary permutation ``p`` (as a raw fit would index its states),
    derive ``order`` straight from the raw model via :func:`canonical_order`, then
    relabel the raw series back to canonical via :func:`relabel_states`. The
    per-regime characterization must be byte-identical to the canonical original —
    the headline stats do not depend on the fit's arbitrary state indexing.
    """
    rng = np.random.default_rng(123)
    n = 600
    # canonical decoded series + per-state returns with regime-dependent vol.
    canon_states = rng.integers(0, 3, size=n)
    # Ascending mean ensures the "canonical" ordering below really is canonical.
    means = np.array([-0.0008, 0.0002, 0.0010])
    vols = np.array([0.025, 0.012, 0.005])
    ret = rng.normal(means[canon_states], vols[canon_states])
    index = pd.date_range("2015-01-01", periods=n, freq="B")
    returns = pd.Series(ret, index=index)

    transmat_canon = np.array(
        [[0.90, 0.07, 0.03], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]],
        dtype="float64",
    )
    canon_char = characterize_regimes(
        _model_from_transmat(transmat_canon),
        canon_states.astype("float64"),
        returns,
    )

    p = np.asarray(perm)
    inverse = np.empty(3, dtype=np.intp)
    inverse[p] = np.arange(3)
    # Scramble canonical state c into raw label inverse[c]; the raw model carries
    # the same means/vols/transmat under that arbitrary indexing.
    raw_states = inverse[canon_states].astype("float64")
    raw_model = HMMModel(
        startprob=np.full(3, 1 / 3, dtype="float64"),
        transmat=transmat_canon[np.ix_(p, p)],
        means=means[p].reshape(-1, 1),
        covariances=(vols[p] ** 2).reshape(-1, 1),
        covariance_type="diag",
        log_likelihood=-1.0,
        n_iter=1,
        converged=True,
    )

    # order is derived from the raw fit, exactly as production code would: relabel
    # the decoded series AND canonicalize the model, then characterize.
    order = canonical_order(raw_model)
    relabeled = relabel_states(raw_states, order)
    assert np.array_equal(relabeled, canon_states)
    raw_char = characterize_regimes(canonicalize_model(raw_model), relabeled, returns)

    for a, b in zip(canon_char.stats, raw_char.stats, strict=True):
        assert a.state == b.state
        assert math.isclose(a.frequency, b.frequency, abs_tol=1e-12)
        assert math.isclose(a.mean_return, b.mean_return, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(a.volatility, b.volatility, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(a.persistence, b.persistence, abs_tol=1e-12)
        assert math.isclose(a.max_drawdown, b.max_drawdown, rel_tol=1e-12, abs_tol=1e-15)
