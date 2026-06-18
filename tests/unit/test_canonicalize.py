"""Unit + property tests for regime state canonicalization.

These cover the determinism and relabeling-invariance contract of
:mod:`regimehmm.regimes.canonicalize`: the canonical ordering is a pure function
of the per-state (mean, vol) keys, it is stable under any input permutation of the
states, and relabeling a decoded series composes correctly with it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from regimehmm._exceptions import ValidationError
from regimehmm.hmm.filter import HMMModel
from regimehmm.regimes.canonicalize import (
    canonical_order,
    canonicalize_model,
    relabel_states,
)


def _make_model(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    covariance_type: str = "diag",
    transmat: np.ndarray | None = None,
) -> HMMModel:
    """Build a frozen HMMModel from per-state (return-feature) means/variances."""
    means = np.asarray(means, dtype="float64")
    variances = np.asarray(variances, dtype="float64")
    n_states = means.shape[0]
    if transmat is None:
        transmat = np.full((n_states, n_states), 1.0 / n_states, dtype="float64")
    startprob = np.full(n_states, 1.0 / n_states, dtype="float64")
    if covariance_type == "full":
        covs = np.stack([np.diag(v) for v in variances]).astype("float64")
    else:
        covs = variances.copy()
    return HMMModel(
        startprob=startprob,
        transmat=np.asarray(transmat, dtype="float64"),
        means=means,
        covariances=covs,
        covariance_type=covariance_type,  # type: ignore[arg-type]
        log_likelihood=-1.0,
        n_iter=1,
        converged=True,
    )


@pytest.mark.unit
def test_canonical_order_sorts_by_ascending_mean() -> None:
    """Permutation indexes states from lowest to highest mean return."""
    model = _make_model(
        means=np.array([[0.002], [-0.001], [0.0005]]),
        variances=np.array([[0.01], [0.04], [0.02]]),
    )
    perm = canonical_order(model).astype(int)
    # original state 1 has the lowest mean, then 2, then 0.
    assert perm.tolist() == [1, 2, 0]


@pytest.mark.unit
def test_canonical_order_tie_breaks_on_volatility() -> None:
    """Equal means are ordered by ascending volatility."""
    model = _make_model(
        means=np.array([[0.001], [0.001]]),
        variances=np.array([[0.04], [0.01]]),  # state 1 less volatile
    )
    perm = canonical_order(model).astype(int)
    assert perm.tolist() == [1, 0]


@pytest.mark.unit
def test_canonical_order_handles_full_covariance() -> None:
    """The volatility tie-break reads the leading diagonal for full covariances."""
    model = _make_model(
        means=np.array([[0.001, 0.0], [0.001, 0.0]]),
        variances=np.array([[0.04, 0.5], [0.01, 0.5]]),
        covariance_type="full",
    )
    perm = canonical_order(model).astype(int)
    assert perm.tolist() == [1, 0]


@pytest.mark.unit
def test_canonicalize_model_preserves_likelihood_and_orders_means() -> None:
    """Canonicalization is a pure relabeling: LL unchanged, means ascending."""
    transmat = np.array(
        [[0.90, 0.05, 0.05], [0.10, 0.80, 0.10], [0.02, 0.08, 0.90]],
        dtype="float64",
    )
    model = _make_model(
        means=np.array([[0.002], [-0.001], [0.0005]]),
        variances=np.array([[0.01], [0.04], [0.02]]),
        transmat=transmat,
    )
    canon = canonicalize_model(model)
    assert canon.log_likelihood == model.log_likelihood
    # means are now ascending by canonical label.
    assert np.all(np.diff(canon.means[:, 0]) > 0)
    # startprob/transmat remain valid stochastic objects.
    assert np.isclose(canon.startprob.sum(), 1.0)
    assert np.allclose(canon.transmat.sum(axis=1), 1.0)
    # canonicalizing an already-canonical model is idempotent.
    assert np.allclose(canonicalize_model(canon).means, canon.means)
    assert np.allclose(canonicalize_model(canon).transmat, canon.transmat)


@pytest.mark.unit
def test_canonicalize_model_permutes_transition_entries_consistently() -> None:
    """A transition prob keeps its value after rows AND cols are permuted."""
    transmat = np.array(
        [[0.90, 0.05, 0.05], [0.10, 0.80, 0.10], [0.02, 0.08, 0.90]],
        dtype="float64",
    )
    model = _make_model(
        means=np.array([[0.002], [-0.001], [0.0005]]),
        variances=np.array([[0.01], [0.04], [0.02]]),
        transmat=transmat,
    )
    perm = canonical_order(model).astype(int)
    canon = canonicalize_model(model)
    for i in range(3):
        for j in range(3):
            assert canon.transmat[i, j] == transmat[perm[i], perm[j]]


@pytest.mark.unit
def test_relabel_states_maps_raw_to_canonical() -> None:
    """relabel_states applies the inverse permutation to a decoded series."""
    # order = new label -> original index. order=[1,2,0] means raw state 1 -> 0.
    order = np.array([1.0, 2.0, 0.0])
    raw = np.array([0, 1, 2, 1, 0], dtype="float64")
    relabeled = relabel_states(raw, order).astype(int)
    # inverse: raw 1 -> 0, raw 2 -> 1, raw 0 -> 2.
    assert relabeled.tolist() == [2, 0, 1, 0, 2]


@pytest.mark.unit
def test_relabel_states_identity_for_identity_order() -> None:
    """The identity permutation leaves the series unchanged."""
    order = np.array([0.0, 1.0, 2.0])
    raw = np.array([0, 2, 1, 1], dtype="float64")
    assert relabel_states(raw, order).astype(int).tolist() == raw.astype(int).tolist()


@pytest.mark.unit
@pytest.mark.parametrize(
    "order",
    [
        np.array([0.0, 0.0, 1.0]),  # not a permutation (repeat)
        np.array([0.0, 1.0, 3.0]),  # out of range
        np.array([[0.0, 1.0]]),  # not 1-D
        np.array([]),  # empty
    ],
)
def test_relabel_states_rejects_bad_order(order: np.ndarray) -> None:
    """A non-permutation ``order`` raises ValidationError."""
    with pytest.raises(ValidationError):
        relabel_states(np.array([0.0, 1.0]), order)


@pytest.mark.unit
def test_relabel_states_rejects_out_of_range_label() -> None:
    """A decoded label outside the permutation range raises ValidationError."""
    with pytest.raises(ValidationError):
        relabel_states(np.array([0.0, 5.0]), np.array([0.0, 1.0]))


@pytest.mark.unit
def test_relabel_states_rejects_non_1d_states() -> None:
    """A 2-D states array raises ValidationError."""
    with pytest.raises(ValidationError):
        relabel_states(np.array([[0.0, 1.0]]), np.array([0.0, 1.0]))


@pytest.mark.unit
def test_relabel_states_rejects_non_integer_order() -> None:
    """A fractional ``order`` value raises ValidationError."""
    with pytest.raises(ValidationError):
        relabel_states(np.array([0.0, 1.0]), np.array([0.0, 1.5]))


@pytest.mark.unit
def test_relabel_states_rejects_non_integer_states() -> None:
    """A fractional decoded label raises ValidationError."""
    with pytest.raises(ValidationError):
        relabel_states(np.array([0.0, 0.7]), np.array([0.0, 1.0]))


@pytest.mark.unit
def test_canonical_order_rejects_non_2d_means() -> None:
    """A 1-D means array raises ValidationError."""
    model = _make_model(np.array([[0.001], [0.002]]), np.array([[0.01], [0.02]]))
    bad = replace(model, means=np.array([0.001, 0.002]))
    with pytest.raises(ValidationError):
        canonical_order(bad)


@pytest.mark.unit
def test_canonical_order_rejects_bad_covariance_rank() -> None:
    """A 1-D covariances array raises ValidationError."""
    model = _make_model(np.array([[0.001], [0.002]]), np.array([[0.01], [0.02]]))
    bad = replace(model, covariances=np.array([0.01, 0.02]))
    with pytest.raises(ValidationError):
        canonical_order(bad)


# --------------------------------------------------------------------------- #
# Property tests                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.property
@given(perm=st.permutations([0, 1, 2]))
def test_canonical_ordering_is_permutation_invariant(perm: list[int]) -> None:
    """Canonicalizing always yields the SAME model regardless of input order.

    Start from a reference model with distinct (mean, vol) per state; apply an
    arbitrary permutation of the states; the canonicalized result must be
    byte-identical to canonicalizing the reference - labels are stable across
    folds no matter how the fit happened to index its states.
    """
    base_means = np.array([[0.002], [-0.001], [0.0005]], dtype="float64")
    base_vars = np.array([[0.01], [0.04], [0.02]], dtype="float64")
    transmat = np.array(
        [[0.90, 0.05, 0.05], [0.10, 0.80, 0.10], [0.02, 0.08, 0.90]],
        dtype="float64",
    )
    ref = canonicalize_model(_make_model(base_means, base_vars, transmat=transmat))

    p = np.asarray(perm)
    shuffled = _make_model(
        base_means[p],
        base_vars[p],
        transmat=transmat[np.ix_(p, p)],
    )
    canon = canonicalize_model(shuffled)

    assert np.allclose(canon.means, ref.means)
    assert np.allclose(canon.transmat, ref.transmat)
    assert np.allclose(canon.startprob, ref.startprob)
    assert np.allclose(canon.covariances, ref.covariances)


@pytest.mark.property
@given(perm=st.permutations([0, 1, 2]))
def test_relabel_inverts_canonical_order(perm: list[int]) -> None:
    """relabel_states(states, order) maps each original index to its new label."""
    base_means = np.array([[0.002], [-0.001], [0.0005]], dtype="float64")
    base_vars = np.array([[0.01], [0.04], [0.02]], dtype="float64")
    p = np.asarray(perm)
    model = _make_model(base_means[p], base_vars[p])
    order = canonical_order(model)
    # A series visiting every raw state exactly once...
    raw = np.arange(3, dtype="float64")
    relabeled = relabel_states(raw, order).astype(int)
    # ...must become a permutation of 0..2 (a valid relabeling).
    assert sorted(relabeled.tolist()) == [0, 1, 2]
    # and the canonical label of raw==order[k] is exactly k.
    for new_label, orig_idx in enumerate(order.astype(int)):
        assert relabeled[orig_idx] == new_label
