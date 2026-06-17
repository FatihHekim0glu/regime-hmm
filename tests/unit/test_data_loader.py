"""Unit tests for the EOD price loader and import purity.

Cover the deployed backend's data path WITHOUT touching the network: the loader
falls back to a deterministic synthetic price panel when Polygon is unavailable
(or forced off), ``compute_returns`` differences without forward-fill, and the
``regimehmm.data`` module has no import-time side effects (no network at import).
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd
import pytest

from regimehmm import ValidationError
from regimehmm.data import (
    _synthetic_prices,
    compute_returns,
    get_prices,
)

_START = dt.date(2020, 1, 1)
_END = dt.date(2020, 6, 1)


@pytest.mark.unit
def test_synthetic_prices_shape_and_positivity() -> None:
    """The synthetic panel is a single positive ``date x ticker`` column."""
    panel = _synthetic_prices("SPY", _START, _END)
    assert isinstance(panel, pd.DataFrame)
    assert list(panel.columns) == ["SPY"]
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert len(panel) > 0
    assert bool((panel.to_numpy() > 0.0).all())
    assert not bool(panel.isna().to_numpy().any())


@pytest.mark.unit
def test_synthetic_prices_is_deterministic() -> None:
    """Same ``(ticker, start, end, seed)`` reproduces a byte-identical panel."""
    a = _synthetic_prices("SPY", _START, _END, seed=7)
    b = _synthetic_prices("SPY", _START, _END, seed=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
def test_synthetic_prices_differ_by_ticker_and_seed() -> None:
    """Distinct ticker or seed yields a distinct (request-keyed) panel."""
    base = _synthetic_prices("SPY", _START, _END, seed=7)
    other_ticker = _synthetic_prices("QQQ", _START, _END, seed=7)
    other_seed = _synthetic_prices("SPY", _START, _END, seed=8)
    assert not np.array_equal(base.to_numpy(), other_ticker.to_numpy())
    assert not np.array_equal(base.to_numpy(), other_seed.to_numpy())


@pytest.mark.unit
def test_synthetic_prices_rejects_bad_range() -> None:
    """``end <= start`` raises ``ValidationError``."""
    with pytest.raises(ValidationError, match="end"):
        _synthetic_prices("SPY", _END, _START)


@pytest.mark.unit
def test_synthetic_prices_empty_span_is_typed_empty() -> None:
    """A weekend-only span (no business days) yields a typed, empty panel."""
    # 2021-01-02 is a Saturday and 2021-01-03 a Sunday: zero business days.
    panel = _synthetic_prices("SPY", dt.date(2021, 1, 2), dt.date(2021, 1, 3))
    assert panel.empty
    assert list(panel.columns) == ["SPY"]
    assert str(panel["SPY"].dtype) == "float64"


@pytest.mark.unit
def test_get_prices_synthetic_pref_is_offline() -> None:
    """``source_pref='synthetic'`` returns the synthetic panel and provenance."""
    panel, source = get_prices("SPY", _START, _END, source_pref="synthetic")
    assert source == "synthetic"
    assert list(panel.columns) == ["SPY"]
    assert bool((panel.to_numpy() > 0.0).all())


@pytest.mark.unit
def test_get_prices_falls_back_to_synthetic_offline() -> None:
    """With no Polygon key/network, ``auto`` degrades to synthetic (never fails)."""
    panel, source = get_prices("SPY", _START, _END, source_pref="auto")
    # In CI/offline there is no Polygon key, so the loader falls through to
    # synthetic rather than raising.
    assert source == "synthetic"
    assert len(panel) > 0


@pytest.mark.unit
def test_get_prices_polygon_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising Polygon fetch is swallowed and the synthetic panel is returned."""

    def _boom(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise RuntimeError("simulated polygon outage")

    monkeypatch.setattr("regimehmm.data._fetch_polygon", _boom)
    panel, source = get_prices("SPY", _START, _END, source_pref="polygon")
    assert source == "synthetic"
    assert bool((panel.to_numpy() > 0.0).all())


@pytest.mark.unit
def test_get_prices_uses_polygon_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful Polygon fetch is returned verbatim with ``polygon`` provenance."""
    fake_index = pd.date_range("2020-01-01", periods=30, freq="B")
    fake_panel = pd.DataFrame(
        {"SPY": np.linspace(300.0, 330.0, num=len(fake_index))},
        index=fake_index,
        dtype="float64",
    )

    def _fake_fetch(ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        assert ticker == "SPY"
        return fake_panel

    monkeypatch.setattr("regimehmm.data._fetch_polygon", _fake_fetch)
    panel, source = get_prices("SPY", _START, _END, source_pref="polygon")
    assert source == "polygon"
    pd.testing.assert_frame_equal(panel, fake_panel)


@pytest.mark.unit
def test_fetch_polygon_returns_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_fetch_polygon`` returns a non-empty provider frame verbatim."""
    from regimehmm.data import _fetch_polygon

    fake_index = pd.date_range("2020-01-01", periods=10, freq="B")
    fake_panel = pd.DataFrame({"SPY": np.arange(10, dtype="float64") + 1.0}, index=fake_index)

    class _FakeProvider:
        def fetch(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
            assert tickers == ["SPY"]
            return fake_panel

    monkeypatch.setattr("regimehmm.data_providers.polygon.PolygonProvider", _FakeProvider)
    out = _fetch_polygon("SPY", _START, _END)
    pd.testing.assert_frame_equal(out, fake_panel)


@pytest.mark.unit
def test_fetch_polygon_empty_payload_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty/all-NaN provider payload raises (so the loader falls back)."""
    from regimehmm.data import _fetch_polygon

    class _EmptyProvider:
        def fetch(self, tickers: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
            return pd.DataFrame({"SPY": [np.nan, np.nan]})

    monkeypatch.setattr("regimehmm.data_providers.polygon.PolygonProvider", _EmptyProvider)
    with pytest.raises(ValueError, match="no usable price data"):
        _fetch_polygon("SPY", _START, _END)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "start", "end"),
    [
        ("", _START, _END),
        ("   ", _START, _END),
        ("SPY", _END, _START),
        ("SPY", _START, _START),
    ],
)
def test_get_prices_validation_errors(ticker: str, start: dt.date, end: dt.date) -> None:
    """Empty ticker or non-positive span raises ``ValidationError``."""
    with pytest.raises(ValidationError):
        get_prices(ticker, start, end)


# --------------------------------------------------------------------------- #
# compute_returns                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_compute_returns_basic() -> None:
    """Returns are simple pct-changes with the leading NaN dropped, as a Series."""
    index = pd.date_range("2021-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 99.0, 99.0]}, index=index)
    ret = compute_returns(prices)
    assert isinstance(ret, pd.Series)
    assert ret.name == "SPY"
    assert len(ret) == 3
    np.testing.assert_allclose(ret.to_numpy(), [0.10, -0.10, 0.0], atol=1e-12)


@pytest.mark.unit
def test_compute_returns_no_forward_fill() -> None:
    """A gap (NaN price) must NOT be forward-filled into a spurious zero return.

    ``pct_change(fill_method=None)`` yields NaN across the gap rather than
    manufacturing a 0%-return; ffill-then-diff would leak information.
    """
    index = pd.date_range("2021-01-01", periods=4, freq="B")
    prices = pd.DataFrame({"SPY": [100.0, np.nan, 121.0, 121.0]}, index=index)
    ret = compute_returns(prices)
    # Row 1 (the NaN price) propagates NaN; it is NOT a clean 0.0 return.
    assert bool(np.isnan(ret.to_numpy()).any())
    # The final, gap-free step is a true 0% return.
    assert ret.iloc[-1] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_returns_rejects_multicolumn() -> None:
    """A multi-column price panel raises ``ValidationError`` (single-ticker only)."""
    index = pd.date_range("2021-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"SPY": [1.0, 2.0, 3.0], "QQQ": [4.0, 5.0, 6.0]}, index=index)
    with pytest.raises(ValidationError, match="single-column"):
        compute_returns(prices)


@pytest.mark.unit
def test_loader_round_trip_returns_are_finite() -> None:
    """Synthetic prices -> compute_returns yields a finite return series."""
    panel, _ = get_prices("SPY", _START, _END, source_pref="synthetic")
    ret = compute_returns(panel)
    assert len(ret) == len(panel) - 1
    assert bool(np.isfinite(ret.to_numpy()).all())


# --------------------------------------------------------------------------- #
# import purity                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_data_module_has_no_network_imports_at_import_time() -> None:
    """Importing ``regimehmm.data`` must not pull in network/data-extra packages.

    The polygon provider, httpx, yfinance, diskcache and pyarrow are all imported
    LAZILY inside the loader functions; merely importing the module must not load
    them (so ``import regimehmm`` stays cheap and side-effect-free).
    """
    heavy = {"httpx", "yfinance", "diskcache", "pyarrow", "curl_cffi"}
    already_loaded = heavy & set(sys.modules)
    # Drop any that a prior test already imported so the assertion is about the
    # act of importing regimehmm.data, not global suite state.
    for name in list(sys.modules):
        if name == "regimehmm.data":
            del sys.modules[name]

    import importlib

    importlib.import_module("regimehmm.data")
    newly_loaded = (heavy & set(sys.modules)) - already_loaded
    assert not newly_loaded, f"regimehmm.data imported heavy deps at import time: {newly_loaded}"
