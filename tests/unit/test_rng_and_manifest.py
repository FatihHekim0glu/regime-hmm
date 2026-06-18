"""Unit tests for the seeded RNG helpers and the reproducibility manifest.

Pins determinism/independence of :mod:`regimehmm._rng` and the canonical hashing
plus git-introspection behaviour of :mod:`regimehmm._manifest`.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimehmm._manifest import RunManifest, _git_dirty, _git_sha, config_hash
from regimehmm._rng import make_rng, spawn_substreams


@pytest.mark.unit
def test_make_rng_is_deterministic() -> None:
    """Two generators from the same seed produce identical draws."""
    a = make_rng(123).standard_normal(8)
    b = make_rng(123).standard_normal(8)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_make_rng_rejects_negative_seed() -> None:
    """A negative seed is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        make_rng(-1)


@pytest.mark.unit
def test_spawn_substreams_are_independent_and_reproducible() -> None:
    """Substreams are reproducible for a given (seed, n) and mutually distinct."""
    first = spawn_substreams(7, 3)
    second = spawn_substreams(7, 3)
    draws_a = [g.standard_normal(5) for g in first]
    draws_b = [g.standard_normal(5) for g in second]
    for da, db in zip(draws_a, draws_b, strict=True):
        np.testing.assert_array_equal(da, db)
    # Distinct children: stream 0 must not equal stream 1.
    assert not np.array_equal(draws_a[0], draws_a[1])


@pytest.mark.unit
def test_spawn_substreams_zero_is_empty() -> None:
    """Requesting zero substreams yields an empty list, not an error."""
    assert spawn_substreams(0, 0) == []


@pytest.mark.unit
def test_spawn_substreams_rejects_negative() -> None:
    """Negative seed or negative count both raise."""
    with pytest.raises(ValueError, match="seed must be non-negative"):
        spawn_substreams(-1, 2)
    with pytest.raises(ValueError, match="n must be non-negative"):
        spawn_substreams(1, -2)


@pytest.mark.unit
def test_config_hash_is_order_invariant() -> None:
    """Key order does not change the canonical hash; values do."""
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 32
    assert config_hash({"a": 1, "b": 3}) != h1


@pytest.mark.unit
def test_config_hash_handles_non_json_values() -> None:
    """Non-JSON-native values fall back to ``str`` rather than raising."""
    from datetime import date

    digest = config_hash({"when": date(2020, 1, 1)})
    assert len(digest) == 32


@pytest.mark.unit
def test_git_helpers_return_expected_types() -> None:
    """The git helpers return a string SHA and a bool dirty flag (never raise)."""
    assert isinstance(_git_sha(), str)
    assert isinstance(_git_sha(short=False), str)
    assert isinstance(_git_dirty(), bool)


@pytest.mark.unit
def test_run_manifest_capture_and_to_dict() -> None:
    """A captured manifest hashes the config, records the seed, and serializes."""
    manifest = RunManifest.capture({"n_states": 3}, seed=11)
    assert manifest.seed == 11
    assert manifest.config_hash == config_hash({"n_states": 3})
    payload = manifest.to_dict()
    assert payload["seed"] == 11
    assert set(payload) >= {"git_sha", "dirty", "config_hash", "seed"}
