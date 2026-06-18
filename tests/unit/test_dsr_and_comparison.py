"""Unit tests for the overfitting guards and Sharpe-difference inference.

Covers the Probabilistic / Deflated Sharpe ratios in
:mod:`regimehmm.evaluation.dsr` (bounds, monotonicity in ``n_trials``, the
single-trial collapse, and the validation branches) and the
Jobson-Korkie-Memmel p-value plus the bootstrap CI in
:mod:`regimehmm.evaluation.comparison`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm._exceptions import ValidationError
from regimehmm.evaluation.comparison import (
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)
from regimehmm.evaluation.dsr import (
    _norm_ppf,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


@pytest.mark.unit
def test_psr_half_at_zero_sharpe() -> None:
    """A zero observed Sharpe against a zero benchmark gives PSR = 0.5."""
    assert probabilistic_sharpe_ratio(0.0, n_obs=100) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.unit
def test_psr_increases_with_sharpe_and_is_bounded() -> None:
    """PSR rises with the observed Sharpe and stays in (0, 1)."""
    low = probabilistic_sharpe_ratio(0.05, n_obs=250)
    high = probabilistic_sharpe_ratio(0.20, n_obs=250)
    assert 0.5 < low < high < 1.0


@pytest.mark.unit
def test_psr_rejects_short_sample_and_bad_variance() -> None:
    """PSR needs n_obs >= 2 and a positive bracket variance."""
    with pytest.raises(ValidationError, match="n_obs >= 2"):
        probabilistic_sharpe_ratio(0.1, n_obs=1)
    # A large skew can drive the bracket variance non-positive.
    with pytest.raises(ValidationError, match="non-positive variance"):
        probabilistic_sharpe_ratio(2.0, n_obs=100, skew=5.0, kurtosis=3.0)


@pytest.mark.unit
def test_dsr_single_trial_equals_psr_against_zero() -> None:
    """With one trial the deflation benchmark collapses to zero (DSR == PSR)."""
    dsr = deflated_sharpe_ratio(0.1, n_obs=250, n_trials=1, variance_of_trial_sharpes=0.04)
    psr = probabilistic_sharpe_ratio(0.1, n_obs=250)
    assert dsr == pytest.approx(psr, abs=1e-12)


@pytest.mark.unit
def test_dsr_non_increasing_in_n_trials() -> None:
    """More trials raises the expected-max benchmark, so DSR cannot increase."""
    kwargs = {"n_obs": 500, "variance_of_trial_sharpes": 0.05}
    dsr_small = deflated_sharpe_ratio(0.15, n_trials=2, **kwargs)
    dsr_large = deflated_sharpe_ratio(0.15, n_trials=100, **kwargs)
    assert dsr_large <= dsr_small + 1e-12


@pytest.mark.unit
def test_dsr_zero_variance_collapses_to_psr() -> None:
    """Zero cross-trial variance leaves the benchmark at zero."""
    dsr = deflated_sharpe_ratio(0.1, n_obs=250, n_trials=50, variance_of_trial_sharpes=0.0)
    assert dsr == pytest.approx(probabilistic_sharpe_ratio(0.1, n_obs=250), abs=1e-12)


@pytest.mark.unit
def test_dsr_validation_branches() -> None:
    """DSR rejects short samples, sub-1 trial counts, and negative variance."""
    with pytest.raises(ValidationError, match="n_obs >= 2"):
        deflated_sharpe_ratio(0.1, n_obs=1, n_trials=5, variance_of_trial_sharpes=0.01)
    with pytest.raises(ValidationError, match="n_trials >= 1"):
        deflated_sharpe_ratio(0.1, n_obs=10, n_trials=0, variance_of_trial_sharpes=0.01)
    with pytest.raises(ValidationError, match="variance_of_trial_sharpes >= 0"):
        deflated_sharpe_ratio(0.1, n_obs=10, n_trials=5, variance_of_trial_sharpes=-0.01)


@pytest.mark.unit
def test_norm_ppf_round_trips_and_validates() -> None:
    """The inverse-CDF inverts the CDF on the tails and rejects p outside (0, 1)."""
    # Far tails exercise both the lower and upper rational branches.
    assert _norm_ppf(0.01) < -2.0
    assert _norm_ppf(0.99) > 2.0
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValidationError, match=r"p in \(0, 1\)"):
        _norm_ppf(0.0)
    with pytest.raises(ValidationError, match=r"p in \(0, 1\)"):
        _norm_ppf(1.0)


@pytest.mark.unit
def test_jkm_pvalue_in_unit_interval_for_identical_series() -> None:
    """Identical return series have a zero Sharpe gap and a large p-value."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, size=300)
    p = jobson_korkie_memmel(r, r)
    assert 0.0 <= p <= 1.0
    assert p == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_jkm_rejects_short_and_unalignable() -> None:
    """JKM needs >= 3 observations and alignable series."""
    with pytest.raises(ValidationError, match="at least 3"):
        jobson_korkie_memmel([0.01, 0.02], [0.0, 0.0])
    # Disjoint indexes AND different lengths -> cannot align positionally either.
    a = pd.Series([0.01, 0.02, 0.03, 0.04], index=[0, 1, 2, 3])
    b = pd.Series([0.0, 0.0, 0.0], index=[10, 11, 12])
    with pytest.raises(ValidationError, match="different lengths"):
        jobson_korkie_memmel(a, b)


@pytest.mark.unit
def test_jkm_zero_variance_series_returns_one() -> None:
    """A degenerate zero-variance series yields no evidence against the null."""
    constant = np.full(50, 0.001)
    moving = np.random.default_rng(2).normal(0.0, 0.01, size=50)
    assert jobson_korkie_memmel(constant, moving) == pytest.approx(1.0)


@pytest.mark.unit
def test_block_bootstrap_ci_brackets_point_gap() -> None:
    """The bootstrap CI brackets the point Sharpe gap and is reproducible."""
    rng = np.random.default_rng(3)
    a = rng.normal(0.001, 0.01, size=400)
    b = rng.normal(0.0, 0.01, size=400)
    res1 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=200, seed=42)
    res2 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=200, seed=42)
    assert res1.to_dict() == res2.to_dict()  # reproducible for a fixed seed
    assert res1.ci_low <= res1.sharpe_gap <= res1.ci_high
    assert res1.n_bootstrap == 200


@pytest.mark.unit
def test_block_bootstrap_validation_branches() -> None:
    """The bootstrap rejects a sub-1 resample count and a bad confidence level."""
    a = np.random.default_rng(4).normal(size=20)
    b = np.random.default_rng(5).normal(size=20)
    with pytest.raises(ValidationError, match="n_bootstrap must be"):
        block_bootstrap_sharpe_gap(a, b, n_bootstrap=0)
    with pytest.raises(ValidationError, match=r"confidence must be in \(0, 1\)"):
        block_bootstrap_sharpe_gap(a, b, confidence=1.5)
    with pytest.raises(ValidationError, match="at least 3"):
        block_bootstrap_sharpe_gap([0.1, 0.2], [0.0, 0.0])
