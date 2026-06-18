"""Unit tests for the lazy Plotly figure builders.

Covers ``regimehmm.plots`` - the three figure builders:

* :func:`~regimehmm.plots.regime_shaded_figure` (price/return line + filtered-regime
  background ribbon + per-regime legend),
* :func:`~regimehmm.plots.regime_stats_figure` (per-regime risk/return scatter),
* :func:`~regimehmm.plots.oos_equity_figure` (overlay vs buy-and-hold equity curves).

Every builder must return a plain ``{"data", "layout"}`` mapping whose contents are
JSON-serializable (no numpy/pandas/Plotly object leaks across the API boundary) and
all-finite, and we assert real numerical structure rather than merely "it runs".
All inputs are synthetic/seeded; nothing touches the network and Plotly itself is
not required (the builders emit plain dicts).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from regimehmm import plots
from regimehmm._exceptions import ValidationError

pytestmark = pytest.mark.unit


def _assert_figure_dict(fig: object) -> dict[str, Any]:
    """Assert ``fig`` is a finite, JSON-safe ``{"data", "layout"}`` mapping.

    The ``json.dumps`` round-trip is the load-bearing check: it fails loudly if any
    numpy scalar/array, pandas object, or Plotly graph-object leaked through. We
    also walk every numeric leaf and assert it is finite (no NaN/Inf crosses the
    API boundary).
    """
    assert isinstance(fig, dict)
    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)

    encoded = json.dumps(fig)  # raises if a non-JSON object leaked
    assert json.loads(encoded) == fig

    _assert_all_finite(fig)
    return fig


def _assert_all_finite(value: Any) -> None:
    """Recursively assert every numeric leaf is finite (no NaN/Inf)."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        assert math.isfinite(value), f"non-finite numeric leaf: {value!r}"
        return
    if isinstance(value, dict):
        for v in value.values():
            _assert_all_finite(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _assert_all_finite(v)


@pytest.fixture
def shaded_inputs() -> tuple[pd.Series, np.ndarray]:
    """A seeded cumulative-return series with a 2-regime label path."""
    gen = np.random.default_rng(11)
    n = 120
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = gen.normal(0.0003, 0.01, size=n)
    series = pd.Series(np.cumprod(1.0 + rets), index=index, name="price")
    # A blocky regime path: first third calm (0), middle turbulent (1), rest calm.
    states = np.zeros(n, dtype="float64")
    states[n // 3 : 2 * n // 3] = 1.0
    return series, states


# --------------------------------------------------------------------------- #
# _jsonify (recursive numpy/pandas -> native coercion)                         #
# --------------------------------------------------------------------------- #
def test_jsonify_coerces_nested_numpy_and_pandas() -> None:
    """``_jsonify`` recurses through dicts/lists and unwraps numpy/pandas leaves."""
    payload = {
        "arr": np.array([1.0, 2.0]),
        "scalar": np.float64(3.5),
        "nested": [np.int64(7), {"ts": pd.Timestamp("2020-01-02")}],
        "period": pd.Period("2020-01", freq="M"),
        "plain": "kept",
    }
    out = plots._jsonify(payload)
    # The whole structure must be JSON-serializable (no numpy/pandas leaks).
    json.dumps(out)
    assert out["arr"] == [1.0, 2.0]
    assert out["scalar"] == 3.5
    assert out["nested"][0] == 7
    assert out["nested"][1]["ts"] == "2020-01-02T00:00:00"
    assert isinstance(out["period"], str)
    assert out["plain"] == "kept"


# --------------------------------------------------------------------------- #
# regime_shaded_figure                                                         #
# --------------------------------------------------------------------------- #
def test_regime_shaded_figure_structure(shaded_inputs: tuple[pd.Series, np.ndarray]) -> None:
    """The shaded figure carries the line trace, a legend marker per regime, and bands."""
    series, states = shaded_inputs
    fig = _assert_figure_dict(plots.regime_shaded_figure(series, states, n_states=2))

    # One price line + one (invisible) legend marker per regime.
    line_traces = [t for t in fig["data"] if t.get("mode") == "lines"]
    legend_traces = [t for t in fig["data"] if t.get("name", "").startswith("regime ")]
    assert len(line_traces) == 1
    assert len(legend_traces) == 2

    # The line's y-values match the input series exactly.
    np.testing.assert_allclose(line_traces[0]["y"], series.to_numpy())

    # Background bands exist (>= 3 runs: calm / turbulent / calm).
    shapes = fig["layout"]["shapes"]
    assert len(shapes) >= 3
    assert all(s["type"] == "rect" and s["layer"] == "below" for s in shapes)
    assert fig["layout"]["title"]["text"] == "Regime-shaded series"


def test_regime_shaded_figure_single_run_single_band() -> None:
    """A constant regime path collapses to exactly one background band."""
    index = pd.date_range("2021-01-01", periods=30, freq="B")
    series = pd.Series(np.linspace(1.0, 2.0, 30), index=index, name="s")
    states = np.zeros(30, dtype="float64")
    fig = _assert_figure_dict(plots.regime_shaded_figure(series, states, n_states=1))
    assert len(fig["layout"]["shapes"]) == 1


def test_regime_shaded_figure_rejects_length_mismatch() -> None:
    """Misaligned series/states raise a ValidationError."""
    index = pd.date_range("2021-01-01", periods=10, freq="B")
    series = pd.Series(np.arange(10.0), index=index)
    with pytest.raises(ValidationError, match="same length"):
        plots.regime_shaded_figure(series, np.zeros(9), n_states=2)


def test_regime_shaded_figure_rejects_out_of_range_label() -> None:
    """A label >= n_states is rejected."""
    index = pd.date_range("2021-01-01", periods=5, freq="B")
    series = pd.Series(np.arange(5.0), index=index)
    states = np.array([0, 0, 2, 0, 0], dtype="float64")
    with pytest.raises(ValidationError, match="outside"):
        plots.regime_shaded_figure(series, states, n_states=2)


def test_regime_shaded_figure_rejects_non_series() -> None:
    """A non-Series ``series`` argument is rejected."""
    with pytest.raises(ValidationError, match="must be a pandas Series"):
        plots.regime_shaded_figure([1.0, 2.0], np.zeros(2), n_states=1)  # type: ignore[arg-type]


def test_regime_shaded_figure_rejects_bad_n_states() -> None:
    """A non-positive ``n_states`` is rejected."""
    index = pd.date_range("2021-01-01", periods=3, freq="B")
    series = pd.Series(np.arange(3.0), index=index)
    with pytest.raises(ValidationError, match="n_states must be >= 1"):
        plots.regime_shaded_figure(series, np.zeros(3), n_states=0)


def test_regime_shaded_figure_rejects_2d_states() -> None:
    """A 2-D ``states`` array is rejected."""
    index = pd.date_range("2021-01-01", periods=4, freq="B")
    series = pd.Series(np.arange(4.0), index=index)
    with pytest.raises(ValidationError, match="must be 1-D"):
        plots.regime_shaded_figure(series, np.zeros((2, 2)), n_states=1)


def test_regime_shaded_figure_rejects_non_integer_labels() -> None:
    """Fractional state labels are rejected."""
    index = pd.date_range("2021-01-01", periods=3, freq="B")
    series = pd.Series(np.arange(3.0), index=index)
    with pytest.raises(ValidationError, match="integer-valued"):
        plots.regime_shaded_figure(series, np.array([0.0, 0.5, 1.0]), n_states=2)


def test_regime_shaded_figure_non_datetime_index() -> None:
    """A plain integer index serializes to string x-values (the str() branch)."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=[0, 1, 2, 3], name="s")
    fig = _assert_figure_dict(plots.regime_shaded_figure(series, np.zeros(4), n_states=1))
    line = next(t for t in fig["data"] if t.get("mode") == "lines")
    assert line["x"] == ["0", "1", "2", "3"]


# --------------------------------------------------------------------------- #
# regime_stats_figure                                                          #
# --------------------------------------------------------------------------- #
def test_regime_stats_figure_plots_each_regime() -> None:
    """The scatter carries one (vol, mean) point per regime in canonical order."""
    means = np.array([0.08, -0.04, -0.15])
    vols = np.array([0.10, 0.22, 0.45])
    fig = _assert_figure_dict(plots.regime_stats_figure(means, vols))

    trace = fig["data"][0]
    assert trace["type"] == "scatter"
    np.testing.assert_allclose(trace["x"], vols)
    np.testing.assert_allclose(trace["y"], means)
    assert trace["text"] == ["regime 0", "regime 1", "regime 2"]
    # A zero-mean reference line lives in the layout shapes.
    assert any(s["type"] == "line" for s in fig["layout"]["shapes"])


def test_regime_stats_figure_rejects_length_mismatch() -> None:
    """means and vols of different lengths raise a ValidationError."""
    with pytest.raises(ValidationError, match="same length"):
        plots.regime_stats_figure(np.array([0.1, 0.2]), np.array([0.1]))


# --------------------------------------------------------------------------- #
# oos_equity_figure                                                            #
# --------------------------------------------------------------------------- #
def test_oos_equity_figure_compounds_from_one() -> None:
    """Both equity curves start at the first compounded step and are finite."""
    index = pd.date_range("2022-01-01", periods=50, freq="B")
    gen = np.random.default_rng(3)
    overlay = pd.Series(gen.normal(0.0002, 0.008, 50), index=index)
    buyhold = pd.Series(gen.normal(0.0004, 0.010, 50), index=index)

    fig = _assert_figure_dict(plots.oos_equity_figure(overlay, buyhold))

    names = {t["name"] for t in fig["data"]}
    assert names == {"regime overlay", "buy & hold"}

    overlay_trace = next(t for t in fig["data"] if t["name"] == "regime overlay")
    expected = np.cumprod(1.0 + overlay.to_numpy())
    np.testing.assert_allclose(overlay_trace["y"], expected)


def test_oos_equity_figure_aligns_on_common_index() -> None:
    """Non-identical indexes are reduced to their (sorted) intersection."""
    idx_a = pd.date_range("2022-01-01", periods=40, freq="B")
    idx_b = idx_a[5:]  # overlapping but shorter
    overlay = pd.Series(np.full(40, 0.001), index=idx_a)
    buyhold = pd.Series(np.full(35, 0.002), index=idx_b)
    fig = _assert_figure_dict(plots.oos_equity_figure(overlay, buyhold))
    # The common index has 35 points; each trace's x has that length.
    assert all(len(t["x"]) == 35 for t in fig["data"])


def test_oos_equity_figure_rejects_disjoint_index() -> None:
    """Disjoint indexes raise a ValidationError."""
    overlay = pd.Series([0.01, 0.02], index=pd.date_range("2022-01-01", periods=2, freq="B"))
    buyhold = pd.Series([0.01, 0.02], index=pd.date_range("2023-01-01", periods=2, freq="B"))
    with pytest.raises(ValidationError, match="no common index"):
        plots.oos_equity_figure(overlay, buyhold)


def test_oos_equity_figure_rejects_non_series_overlay() -> None:
    """A non-Series overlay argument is rejected."""
    buyhold = pd.Series([0.01], index=pd.date_range("2022-01-01", periods=1, freq="B"))
    with pytest.raises(ValidationError, match="overlay_returns must be a pandas Series"):
        plots.oos_equity_figure([0.01], buyhold)  # type: ignore[arg-type]


def test_oos_equity_figure_rejects_non_series_buyhold() -> None:
    """A non-Series buy-and-hold argument is rejected."""
    overlay = pd.Series([0.01], index=pd.date_range("2022-01-01", periods=1, freq="B"))
    with pytest.raises(ValidationError, match="buyhold_returns must be a pandas Series"):
        plots.oos_equity_figure(overlay, [0.01])  # type: ignore[arg-type]


def test_oos_equity_figure_handles_nan_returns() -> None:
    """A NaN return is treated as a flat step; the curve stays finite and JSON-safe."""
    index = pd.date_range("2022-01-01", periods=6, freq="B")
    overlay = pd.Series([0.01, np.nan, 0.02, 0.0, -0.01, 0.005], index=index)
    buyhold = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01], index=index)
    fig = _assert_figure_dict(plots.oos_equity_figure(overlay, buyhold))
    overlay_trace = next(t for t in fig["data"] if t["name"] == "regime overlay")
    # Step 2 (the NaN) leaves wealth unchanged from step 1.
    assert overlay_trace["y"][1] == pytest.approx(overlay_trace["y"][0])
