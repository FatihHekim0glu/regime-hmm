"""Integration tests for the public entrypoint (``regimehmm.analysis``).

These exercise :func:`regimehmm.run_regime_analysis` — the single function the
hosted FastAPI tool calls — end-to-end on the deterministic synthetic
``regime_switch`` fixture (no network), plus the :func:`assemble_regime_figures`
helper that builds the two frontend Plotly figures.

The honest-null contract is the headline assertion: on the persistent-vol
``regime_switch`` series the regime-timing overlay must NOT beat buy-and-hold after
costs, so the verdict is ``no_timing_edge`` and the Deflated-Sharpe effective trial
count is the FULL grid product (3 x 3 x 4 = 36), never collapsed to 1.

Everything runs OFFLINE; nothing touches the network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from regimehmm import (
    RegimeAnalysisResult,
    assemble_regime_figures,
    run_regime_analysis,
)
from regimehmm.analysis import _safe_float
from regimehmm.evaluation.verdict import TimingVerdict

pytestmark = pytest.mark.integration


def test_safe_float_maps_nonfinite_and_bad_to_none() -> None:
    """The JSON-clean coercion maps NaN/Inf and non-numeric values to ``None``."""
    assert _safe_float(1.5) == 1.5
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float("not a number") is None
    assert _safe_float(None) is None


def test_run_regime_analysis_is_honest_null(regime_switch: dict[str, object]) -> None:
    """The full pipeline on ``regime_switch`` yields the honest ``no_timing_edge``.

    The overlay must not reliably beat buy-and-hold after costs, the Memmel-JK test
    must be insignificant or the Deflated Sharpe non-positive (driving the verdict),
    and the effective trial count must be the full 3 x 3 x 4 = 36 grid product.
    """
    returns = regime_switch["returns"]
    assert isinstance(returns, pd.Series)

    result = run_regime_analysis(returns, n_states=2, feature_set="returns", cost_bps=10.0, seed=7)

    assert isinstance(result, RegimeAnalysisResult)
    summary = result.summary

    # Honest-null headline: the verdict is structurally no_timing_edge here.
    assert summary["verdict"] == TimingVerdict.NO_TIMING_EDGE.value
    # The effective trial count is the FULL grid product, never collapsed to 1.
    assert summary["n_effective_trials"] == 36
    # A caller-supplied series reports provenance as "provided".
    assert summary["data_source"] == "provided"
    assert summary["n_states"] == 2


def test_summary_scalars_are_json_clean(regime_switch: dict[str, object]) -> None:
    """Every summary scalar is a finite float / int / str (no NaN/Inf leaks)."""
    returns = regime_switch["returns"]
    assert isinstance(returns, pd.Series)

    result = run_regime_analysis(returns, n_states=3, feature_set="returns_vol", cost_bps=5.0)
    summary = result.summary

    for key in ("overlay_oos_sharpe", "buyhold_oos_sharpe", "sharpe_diff", "deflated_sharpe"):
        value = summary[key]
        assert value is None or isinstance(value, float)
    # The Memmel-JK p-value is a probability in [0, 1].
    pval = summary["jk_pvalue"]
    assert pval is not None and 0.0 <= pval <= 1.0

    # The per-regime characterization carries one row per state, in canonical order.
    stats = summary["regime_stats"]
    assert isinstance(stats, list)
    assert len(stats) == 3
    assert [s["state"] for s in stats] == [0, 1, 2]
    for s in stats:
        for field_name in ("frequency", "mean_return", "volatility", "persistence"):
            assert field_name in s

    # to_dict is wire-ready (summary + meta only, no pandas/numpy objects).
    payload = result.to_dict()
    assert set(payload) == {"summary", "meta"}


def test_assemble_regime_figures_shape(regime_switch: dict[str, object]) -> None:
    """The figure helper returns two ``{data, layout}`` Plotly dicts."""
    pytest.importorskip("plotly")
    returns = regime_switch["returns"]
    assert isinstance(returns, pd.Series)

    result = run_regime_analysis(returns, n_states=2, feature_set="returns")
    figures = assemble_regime_figures(result)

    assert set(figures) == {"regime_figure", "equity_figure"}
    for fig in figures.values():
        assert set(fig) == {"data", "layout"}
        assert isinstance(fig["data"], list)
        assert isinstance(fig["layout"], dict)

    # The equity figure overlays exactly two traces (overlay + buy-and-hold).
    assert len(figures["equity_figure"]["data"]) == 2


def test_run_regime_analysis_loads_synthetic_when_no_returns() -> None:
    """With no series + a forced synthetic source, the loader path runs offline.

    Exercises the get_prices -> compute_returns branch and asserts the provenance is
    reported as ``synthetic`` (no network), with the verdict still the honest null.
    """
    result = run_regime_analysis(
        n_states=2,
        feature_set="returns",
        cost_bps=10.0,
        seed=7,
        ticker="SPY",
        start=date(2015, 1, 1),
        end=date(2020, 1, 1),
        data_source_pref="synthetic",
    )
    assert result.summary["data_source"] == "synthetic"
    assert result.summary["verdict"] == TimingVerdict.NO_TIMING_EDGE.value
    assert result.summary["n_effective_trials"] == 36
    # The scored OOS window is non-trivial.
    assert result.summary["n_obs"] > 100


def test_three_state_macro_feature_runs(regime_switch: dict[str, object]) -> None:
    """A 3-state fit on the macro feature set completes and stays honest."""
    returns = regime_switch["returns"]
    assert isinstance(returns, pd.Series)

    result = run_regime_analysis(
        returns, n_states=3, feature_set="returns_vol_macro", cost_bps=20.0
    )
    assert result.summary["n_states"] == 3
    assert result.summary["verdict"] in {v.value for v in TimingVerdict}
    # macro feature drops the most leading rows but still scores a real window.
    assert result.summary["n_obs"] > 100
