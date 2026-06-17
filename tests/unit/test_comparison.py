"""Unit tests for the Sharpe-difference comparison helpers.

Covers the public :func:`regimehmm.evaluation.comparison.block_bootstrap_sharpe_gap`
(the stationary block-bootstrap CI on the Sharpe gap) and its sibling p-value test
:func:`~regimehmm.evaluation.comparison.jobson_korkie_memmel`. These run on the
deterministic synthetic ``regime_switch`` fixture / inline seeded arrays — no
network.

The block bootstrap is exercised with a small ``n_bootstrap`` so the test stays
fast while still walking the geometric-block resampling loop and percentile-CI
construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimehmm.evaluation.comparison import (
    ComparisonResult,
    block_bootstrap_sharpe_gap,
    jobson_korkie_memmel,
)

pytestmark = pytest.mark.unit


def _two_series(seed: int = 11, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Two correlated seeded return series for the comparison helpers."""
    from regimehmm._rng import make_rng

    gen = make_rng(seed)
    base = gen.normal(0.0004, 0.01, size=n)
    a = base + gen.normal(0.0, 0.002, size=n)
    b = base + gen.normal(0.0, 0.002, size=n)
    return a, b


def test_block_bootstrap_sharpe_gap_basic() -> None:
    """The bootstrap returns a well-formed result with an ordered CI bracket."""
    a, b = _two_series()
    result = block_bootstrap_sharpe_gap(a, b, n_bootstrap=64, confidence=0.9, seed=3)

    assert isinstance(result, ComparisonResult)
    assert result.n_bootstrap == 64
    # The gap equals the difference of the two annualized Sharpes.
    assert result.sharpe_gap == pytest.approx(result.sharpe_a - result.sharpe_b, abs=1e-9)
    # The CI brackets are ordered and (for near-equal series) straddle ~zero.
    assert result.ci_low <= result.ci_high
    assert 0.0 <= result.jkm_pvalue <= 1.0
    # to_dict is JSON-clean.
    payload = result.to_dict()
    assert payload["n_bootstrap"] == 64


def test_block_bootstrap_is_reproducible() -> None:
    """A fixed seed reproduces the CI byte-for-byte (seeded PCG64)."""
    a, b = _two_series()
    r1 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=48, seed=7)
    r2 = block_bootstrap_sharpe_gap(a, b, n_bootstrap=48, seed=7)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_block_bootstrap_explicit_block_size() -> None:
    """An explicit block size also runs and yields an ordered CI."""
    a, b = _two_series(seed=5)
    result = block_bootstrap_sharpe_gap(a, b, n_bootstrap=32, block_size=10, seed=1)
    assert result.ci_low <= result.ci_high


def test_block_bootstrap_rejects_bad_args() -> None:
    """Invalid ``n_bootstrap`` / ``confidence`` are rejected."""
    from regimehmm._exceptions import ValidationError

    a, b = _two_series()
    with pytest.raises(ValidationError):
        block_bootstrap_sharpe_gap(a, b, n_bootstrap=0)
    with pytest.raises(ValidationError):
        block_bootstrap_sharpe_gap(a, b, confidence=1.5)


def test_jkm_pvalue_in_unit_interval() -> None:
    """The Memmel-JK p-value of two near-identical series is a valid probability."""
    a, b = _two_series()
    p = jobson_korkie_memmel(a, b)
    assert 0.0 <= p <= 1.0


def test_jkm_degenerate_series_returns_one() -> None:
    """A zero-variance series leaves the Sharpe gap ill-defined -> p-value 1.0."""
    flat = np.zeros(50, dtype="float64")
    moving = np.linspace(-0.01, 0.01, 50)
    assert jobson_korkie_memmel(flat, moving) == 1.0
