"""Unit tests for the input-coercion guardrails in :mod:`regimehmm._validation`.

Covers the happy-path coercions and every validation error branch (wrong rank,
empty input, NaN rejection, empty index intersection, insufficient rows) so the
boundary helpers the compute kernels rely on are pinned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regimehmm._exceptions import InsufficientDataError, ValidationError
from regimehmm._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)


@pytest.mark.unit
def test_ensure_series_copies_and_casts_float() -> None:
    """A Series input is copied (not mutated) and cast to float64."""
    src = pd.Series([1, 2, 3], dtype="int64")
    out = ensure_series(src)
    assert out.dtype == np.float64
    out.iloc[0] = 99.0
    assert src.iloc[0] == 1  # original untouched


@pytest.mark.unit
def test_ensure_series_accepts_1d_array_and_sequence() -> None:
    """A 1-D ndarray and a plain list both coerce to a float Series."""
    from_array = ensure_series(np.array([0.1, 0.2]))
    from_list = ensure_series([0.1, 0.2])
    assert from_array.tolist() == pytest.approx([0.1, 0.2])
    assert from_list.tolist() == pytest.approx([0.1, 0.2])


@pytest.mark.unit
def test_ensure_series_rejects_2d_array() -> None:
    """A 2-D ndarray is not a series and is rejected."""
    with pytest.raises(ValidationError, match="1-dimensional"):
        ensure_series(np.zeros((3, 2)))


@pytest.mark.unit
def test_ensure_series_rejects_empty() -> None:
    """An empty input raises rather than producing a degenerate series."""
    with pytest.raises(ValidationError, match="non-empty"):
        ensure_series([])


@pytest.mark.unit
def test_ensure_series_rejects_nan_by_default() -> None:
    """NaN is rejected unless explicitly allowed."""
    with pytest.raises(ValidationError, match="NaN"):
        ensure_series([1.0, np.nan, 3.0])
    out = ensure_series([1.0, np.nan, 3.0], allow_nan=True)
    assert bool(out.isna().any())


@pytest.mark.unit
def test_ensure_dataframe_happy_path_and_columns() -> None:
    """A 2-D ndarray coerces to a float frame with the requested columns."""
    out = ensure_dataframe(np.array([[1.0, 2.0], [3.0, 4.0]]), columns=["a", "b"])
    assert list(out.columns) == ["a", "b"]
    assert out.to_numpy().dtype == np.float64


@pytest.mark.unit
def test_ensure_dataframe_rejects_1d_and_empty_and_nan() -> None:
    """Wrong rank, empty shape, and NaN each raise."""
    with pytest.raises(ValidationError, match="2-dimensional"):
        ensure_dataframe(np.zeros(3))
    with pytest.raises(ValidationError, match="at least one row"):
        ensure_dataframe(pd.DataFrame())
    with pytest.raises(ValidationError, match="NaN"):
        ensure_dataframe(np.array([[1.0, np.nan]]))


@pytest.mark.unit
def test_ensure_dataframe_accepts_mapping() -> None:
    """A mapping coerces through the pandas boundary branch."""
    out = ensure_dataframe({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert out.shape == (2, 2)


@pytest.mark.unit
def test_align_inner_intersects_and_sorts() -> None:
    """Two frames align on the sorted intersection of their indexes."""
    left = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=[3, 1, 2])
    right = pd.DataFrame({"y": [4.0, 5.0]}, index=[2, 1])
    al, ar = align_inner(left, right)
    assert list(al.index) == [1, 2]
    assert list(ar.index) == [1, 2]


@pytest.mark.unit
def test_align_inner_rejects_disjoint_index() -> None:
    """No shared labels is an error, not a silent empty join."""
    left = pd.DataFrame({"x": [1.0]}, index=[0])
    right = pd.DataFrame({"y": [2.0]}, index=[9])
    with pytest.raises(ValidationError, match="no common index"):
        align_inner(left, right)


@pytest.mark.unit
def test_validate_min_obs() -> None:
    """The row-count guard passes at the threshold and raises below it."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    validate_min_obs(frame, 3)  # exactly enough -> no raise
    with pytest.raises(InsufficientDataError, match="at least 4"):
        validate_min_obs(frame, 4)
