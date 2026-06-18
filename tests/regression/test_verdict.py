"""Verdict + DSR-wiring tests for the regime-timing overlay (honest-null group).

These cover the three contracts owned by the evaluation/verdict group:

* **DSR n_trials guard** - :func:`effective_n_trials` returns the product of the
  swept axes (``|n_states grid| x |feature variants| x |cost grid|``) and never
  silently collapses to ``1``; the guard rejects any factor ``< 1``.
* **Verdict truth table** - :func:`derive_timing_verdict` is a pure function of
  ``(jk_pvalue, deflated_sharpe, sharpe_diff)`` and is STRUCTURALLY unable to
  return ``timing_edge`` whenever Memmel-JK is insignificant or the Deflated
  Sharpe is ``<= 0``. Includes the honest-null wiring: on the synthetic
  ``regime_switch`` fixture, the regime-timing overlay does NOT beat buy-and-hold
  out-of-sample (Memmel-JK insignificant) so the wired verdict is
  ``no_timing_edge``.
* **DSR parity 1e-10** - the verdict's Deflated-Sharpe wiring reproduces the
  closed-form Bailey-Lopez de Prado DSR (an independent inline reference) to
  ``1e-10``.

The whole suite runs OFFLINE on the seeded fixtures; no test touches the network.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from regimehmm._exceptions import ValidationError
from regimehmm.evaluation.comparison import jobson_korkie_memmel
from regimehmm.evaluation.dsr import deflated_sharpe_ratio
from regimehmm.evaluation.verdict import (
    TimingVerdict,
    derive_timing_verdict,
    effective_n_trials,
)

# ---------------------------------------------------------------------------
# Effective-n_trials / DSR multiplicity guard
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_effective_n_trials_is_product_of_swept_axes() -> None:
    """``n_effective_trials`` is exactly the product of the three swept axes."""
    # Project sweep: n_states in {2, 3, 4}; feature variants in
    # {returns, returns_vol, returns_vol_macro}; a 4-level cost grid.
    assert effective_n_trials(3, 3, 4) == 36
    assert effective_n_trials(2, 1, 1) == 2


@pytest.mark.regression
@pytest.mark.parametrize(
    ("n_states", "n_features", "n_costs"),
    [(2, 3, 5), (4, 2, 3), (3, 3, 3), (1, 1, 1), (5, 4, 7)],
)
def test_effective_n_trials_matches_explicit_product(
    n_states: int, n_features: int, n_costs: int
) -> None:
    """For any valid grid, the count equals ``n_states * n_features * n_costs``."""
    assert effective_n_trials(n_states, n_features, n_costs) == (n_states * n_features * n_costs)


@pytest.mark.regression
def test_effective_n_trials_never_collapses_below_product() -> None:
    """The honest count is >= the product (never silently collapsed to 1).

    This is the multiplicity guard from the brief: feeding the DSR a single trial
    when many were explored would defeat the deflation, so the count must be at
    least the product of the swept axes.
    """
    n_states, n_features, n_costs = 3, 3, 4
    product = n_states * n_features * n_costs
    count = effective_n_trials(n_states, n_features, n_costs)
    assert count >= product
    assert count != 1  # explicitly: not collapsed to a single trial


@pytest.mark.regression
@pytest.mark.parametrize(
    ("n_states", "n_features", "n_costs"),
    [(0, 3, 4), (3, 0, 4), (3, 3, 0), (-1, 2, 2), (2, -5, 2), (2, 2, -1)],
)
def test_effective_n_trials_rejects_subunit_factors(
    n_states: int, n_features: int, n_costs: int
) -> None:
    """Any factor ``< 1`` is rejected (no degenerate / empty grid)."""
    with pytest.raises(ValidationError):
        effective_n_trials(n_states, n_features, n_costs)


# ---------------------------------------------------------------------------
# Verdict truth table (pure function over the OOS evidence)
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize(
    ("jk_pvalue", "deflated_sharpe", "sharpe_diff", "expected"),
    [
        # --- NO_TIMING_EDGE: insignificant Memmel-JK (gap indistinguishable) ---
        (0.40, 0.99, 0.50, TimingVerdict.NO_TIMING_EDGE),
        (0.05, 0.99, 0.50, TimingVerdict.NO_TIMING_EDGE),  # boundary: not < alpha
        # --- NO_TIMING_EDGE: non-positive Sharpe gap (no edge by definition) ---
        (0.01, 0.99, 0.0, TimingVerdict.NO_TIMING_EDGE),
        (0.01, 0.99, -0.30, TimingVerdict.NO_TIMING_EDGE),
        # --- NO_TIMING_EDGE: non-positive Deflated Sharpe (degenerate guard) ---
        (0.01, 0.0, 0.50, TimingVerdict.NO_TIMING_EDGE),
        (0.01, -0.20, 0.50, TimingVerdict.NO_TIMING_EDGE),
        # --- MARGINAL: significant + positive gap, but DSR below threshold ---
        (0.01, 0.80, 0.50, TimingVerdict.MARGINAL),
        (0.001, 0.949, 0.10, TimingVerdict.MARGINAL),
        # --- TIMING_EDGE: significant, positive gap, DSR clears threshold ---
        (0.001, 0.99, 0.50, TimingVerdict.TIMING_EDGE),
        (0.01, 0.95, 0.05, TimingVerdict.TIMING_EDGE),  # boundary: DSR == threshold
    ],
)
def test_verdict_truth_table(
    jk_pvalue: float,
    deflated_sharpe: float,
    sharpe_diff: float,
    expected: TimingVerdict,
) -> None:
    """The verdict follows the documented decision rule for every cell."""
    assert derive_timing_verdict(jk_pvalue, deflated_sharpe, sharpe_diff) is expected


@pytest.mark.regression
def test_timing_edge_is_structurally_impossible_when_jk_insignificant() -> None:
    """No combination with insignificant Memmel-JK can yield ``timing_edge``.

    Sweep a dense grid of (deflated_sharpe, sharpe_diff) with the p-value pinned
    above ``alpha``; the verdict must NEVER be ``timing_edge`` (nor ``marginal``,
    which also requires significance). This is the honest-null structural guard.
    """
    for dsr in np.linspace(-0.5, 1.5, 21):
        for gap in np.linspace(-1.0, 2.0, 31):
            verdict = derive_timing_verdict(0.20, float(dsr), float(gap))
            assert verdict is TimingVerdict.NO_TIMING_EDGE


@pytest.mark.regression
def test_timing_edge_is_structurally_impossible_when_dsr_nonpositive() -> None:
    """No combination with a non-positive Deflated Sharpe yields ``timing_edge``."""
    for pval in np.linspace(0.0, 1.0, 21):
        for gap in np.linspace(-1.0, 2.0, 31):
            for dsr in (-0.5, -0.1, 0.0):
                verdict = derive_timing_verdict(float(pval), dsr, float(gap))
                assert verdict is TimingVerdict.NO_TIMING_EDGE


@pytest.mark.regression
def test_verdict_rejects_out_of_range_pvalue() -> None:
    """A ``jk_pvalue`` outside ``[0, 1]`` (or NaN) is rejected."""
    for bad in (-0.01, 1.01, math.nan):
        with pytest.raises(ValidationError):
            derive_timing_verdict(bad, 0.99, 0.50)


@pytest.mark.regression
def test_verdict_handles_nan_evidence_as_no_edge() -> None:
    """NaN Deflated Sharpe / Sharpe gap fail their positivity checks (no edge)."""
    assert derive_timing_verdict(0.001, math.nan, 0.50) is TimingVerdict.NO_TIMING_EDGE
    assert derive_timing_verdict(0.001, 0.99, math.nan) is TimingVerdict.NO_TIMING_EDGE


# ---------------------------------------------------------------------------
# Honest-null wiring: regime_switch overlay does NOT beat buy-and-hold
# ---------------------------------------------------------------------------


def _filtered_regime_overlay(returns: pd.Series, states: np.ndarray) -> pd.Series:
    """A generous, leakage-free regime-timing overlay return series.

    Scales next-period exposure by the (causally available) regime label with the
    mandatory ``shift(1)`` chokepoint: full exposure in the calm low-vol regime
    (state ``0``), de-risked exposure in the turbulent regime. This is the SAME
    construction the overlay group wires through the online filter; here we feed it
    the ground-truth states (an upper bound on what the filter could know) to make
    the honest-null result conservative - even with a perfect regime label and no
    costs modelled here, the overlay does not reliably beat buy-and-hold OOS.
    """
    exposure = np.where(states == 0, 1.0, 0.0)
    signal = pd.Series(exposure, index=returns.index).shift(1).fillna(0.0)
    return signal * returns


@pytest.mark.regression
def test_honest_null_regime_switch_overlay_does_not_beat_buyhold(
    regime_switch: dict[str, object],
) -> None:
    """On ``regime_switch`` the wired verdict is ``no_timing_edge`` (the headline).

    Wires the real Memmel-JK comparison (``comparison.py``) and Deflated Sharpe
    (``dsr.py``) on the synthetic fixture: the regime-timing overlay's OOS Sharpe
    gap over buy-and-hold is NOT statistically distinguishable from zero
    (Memmel-JK insignificant), so the honest verdict collapses to
    ``no_timing_edge`` - exactly the project's pinned headline.
    """
    returns = regime_switch["returns"]
    states = regime_switch["states"]
    assert isinstance(returns, pd.Series)
    assert isinstance(states, np.ndarray)

    overlay = _filtered_regime_overlay(returns, states)
    buyhold = returns

    # Memmel-JK vs buy-and-hold (reuse comparison.py).
    jk_pvalue = jobson_korkie_memmel(overlay, buyhold)

    # The overlay's point Sharpe gap and its Deflated Sharpe over the FULL grid.
    overlay_excess = overlay.to_numpy(dtype="float64")
    buyhold_excess = buyhold.to_numpy(dtype="float64")
    sharpe_diff = _per_period_sharpe(overlay_excess) - _per_period_sharpe(buyhold_excess)

    n_trials = effective_n_trials(3, 3, 4)  # |n_states| x |features| x |costs|
    deflated = deflated_sharpe_ratio(
        _per_period_sharpe(overlay_excess),
        n_obs=overlay_excess.shape[0],
        n_trials=n_trials,
        variance_of_trial_sharpes=0.01,
    )

    verdict = derive_timing_verdict(jk_pvalue, deflated, sharpe_diff)
    assert verdict is TimingVerdict.NO_TIMING_EDGE
    # The honest-null is driven by an INSIGNIFICANT Memmel-JK test (the gap is
    # indistinguishable from zero), not an accident of the DSR threshold.
    assert jk_pvalue >= 0.05


def _per_period_sharpe(excess: np.ndarray) -> float:
    """Per-period (non-annualized) Sharpe of an excess-return array."""
    std = float(excess.std(ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return float(excess.mean()) / std


# ---------------------------------------------------------------------------
# DSR parity to an independent closed-form reference (1e-10)
# ---------------------------------------------------------------------------

_EULER_MASCHERONI = 0.5772156649015329


def _reference_dsr(
    observed_sharpe: float,
    *,
    n_obs: int,
    n_trials: int,
    variance_of_trial_sharpes: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Independent closed-form Bailey-Lopez de Prado Deflated Sharpe Ratio.

    A self-contained reference (SciPy-based normal CDF/PPF) for the DSR the verdict
    group wires through ``dsr.py``. Pinned to ``deflated_sharpe_ratio`` at 1e-10.
    """
    from scipy import stats  # dev-only; not a runtime import of the kernel

    sr = float(observed_sharpe)
    sqrt_v = math.sqrt(variance_of_trial_sharpes)
    if n_trials == 1 or sqrt_v == 0.0:
        benchmark = 0.0
    else:
        n = float(n_trials)
        z1 = float(stats.norm.ppf(1.0 - 1.0 / n))
        z2 = float(stats.norm.ppf(1.0 - 1.0 / (n * math.e)))
        benchmark = sqrt_v * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)

    variance = 1.0 - skew * sr + 0.25 * (kurtosis - 1.0) * sr * sr
    z = (sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    return float(stats.norm.cdf(z))


@pytest.mark.regression
@pytest.mark.parametrize(
    ("observed_sharpe", "n_obs", "n_trials", "var_trials", "skew", "kurt"),
    [
        (0.10, 1000, 36, 0.01, 0.0, 3.0),
        (0.05, 750, 12, 0.02, -0.3, 5.0),
        (0.20, 1500, 100, 0.005, 0.1, 4.0),
        (0.08, 500, 1, 0.0, 0.0, 3.0),  # single-trial collapse path
        (0.12, 1200, 48, 0.015, -0.1, 6.0),
    ],
)
def test_dsr_wiring_matches_reference_to_1e_10(
    observed_sharpe: float,
    n_obs: int,
    n_trials: int,
    var_trials: float,
    skew: float,
    kurt: float,
) -> None:
    """The verdict's DSR (``dsr.py``) reproduces the closed form to 1e-10."""
    ours = deflated_sharpe_ratio(
        observed_sharpe,
        n_obs=n_obs,
        n_trials=n_trials,
        variance_of_trial_sharpes=var_trials,
        skew=skew,
        kurtosis=kurt,
    )
    ref = _reference_dsr(
        observed_sharpe,
        n_obs=n_obs,
        n_trials=n_trials,
        variance_of_trial_sharpes=var_trials,
        skew=skew,
        kurtosis=kurt,
    )
    assert abs(ours - ref) < 1e-10


@pytest.mark.regression
def test_dsr_is_non_increasing_in_n_trials() -> None:
    """More trials can only DEFLATE the Sharpe (monotone non-increasing in N).

    Confirms the multiplicity penalty actually bites - the honest yardstick that
    makes an in-sample edge decay out-of-sample once the full grid is counted.
    """
    prev = 1.0
    for n_trials in (1, 2, 5, 10, 36, 100, 500):
        dsr = deflated_sharpe_ratio(
            0.10,
            n_obs=1000,
            n_trials=n_trials,
            variance_of_trial_sharpes=0.01,
        )
        assert dsr <= prev + 1e-12
        prev = dsr
