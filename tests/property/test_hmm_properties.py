"""Property-based invariants for the HMM kernel (Hypothesis).

The headline invariant is ONLINE-FILTER NO-LOOKAHEAD: the filtered posterior at
time ``t`` must be invariant to any perturbation of the returns AFTER ``t``
(future-perturbation invariance / prefix-determinism). This is what makes the
filtered posterior — and ONLY it — a legitimate out-of-sample regime signal;
smoothed/Viterbi posteriors fail it by construction.

Also pinned here: posterior rows sum to 1 (filtered and smoothed), transition rows
are stochastic, and the EM fit returns a valid stochastic model.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from regimehmm.hmm.em import fit_hmm
from regimehmm.hmm.filter import HMMModel, filtered_states, online_filter
from regimehmm.hmm.forward_backward import forward_backward
from regimehmm.hmm.kernel import gaussian_log_density

_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _toy_model(n_states: int, n_features: int, *, seed: int) -> HMMModel:
    """A small, well-conditioned diagonal HMM for filter/decoder property checks."""
    rng = np.random.default_rng(seed)
    persistence = 0.9
    off = (1.0 - persistence) / (n_states - 1) if n_states > 1 else 0.0
    transmat = np.full((n_states, n_states), off)
    np.fill_diagonal(transmat, persistence if n_states > 1 else 1.0)
    means = rng.normal(scale=0.5, size=(n_states, n_features))
    covariances = np.abs(rng.normal(loc=1.0, scale=0.2, size=(n_states, n_features))) + 0.1
    startprob = np.full(n_states, 1.0 / n_states)
    return HMMModel(
        startprob=startprob,
        transmat=transmat,
        means=means,
        covariances=covariances,
        covariance_type="diag",
        log_likelihood=0.0,
        n_iter=1,
        converged=True,
    )


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_states=st.integers(min_value=2, max_value=4),
    n_obs=st.integers(min_value=20, max_value=120),
    cut=st.integers(min_value=5, max_value=15),
)
@_SETTINGS
def test_online_filter_no_lookahead(seed: int, n_states: int, n_obs: int, cut: int) -> None:
    """Perturbing returns AFTER ``t`` leaves the filtered posterior at ``t`` identical.

    This is the project's central leakage guard: ``online_filter`` row ``t`` is a
    function of ``obs[0..t]`` ONLY, so any change to ``obs[t+1..]`` cannot move it
    (byte-for-byte, ``atol=0``).
    """
    rng = np.random.default_rng(seed)
    model = _toy_model(n_states, 1, seed=seed)
    obs = rng.normal(size=(n_obs, 1))
    t = min(cut, n_obs - 2)

    base = online_filter(model, obs)
    perturbed_obs = obs.copy()
    perturbed_obs[t + 1 :] += rng.normal(scale=5.0, size=perturbed_obs[t + 1 :].shape)
    perturbed = online_filter(model, perturbed_obs)

    # Exact prefix determinism: rows 0..t are unchanged with zero tolerance.
    assert np.array_equal(base[: t + 1], perturbed[: t + 1])


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_states=st.integers(min_value=2, max_value=4),
    n_obs=st.integers(min_value=10, max_value=80),
)
@_SETTINGS
def test_filtered_posterior_rows_sum_to_one(seed: int, n_states: int, n_obs: int) -> None:
    """Every filtered-posterior row is a probability distribution (sums to 1)."""
    rng = np.random.default_rng(seed)
    model = _toy_model(n_states, 1, seed=seed)
    obs = rng.normal(size=(n_obs, 1))
    posterior = online_filter(model, obs)
    row_sums = posterior.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)
    assert bool((posterior >= 0.0).all())


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_states=st.integers(min_value=2, max_value=4),
    n_obs=st.integers(min_value=10, max_value=80),
)
@_SETTINGS
def test_smoothed_posterior_rows_and_xi_sum_to_one(seed: int, n_states: int, n_obs: int) -> None:
    """Smoothed ``gamma`` rows sum to 1 and each ``xi`` slice sums to 1."""
    rng = np.random.default_rng(seed)
    model = _toy_model(n_states, 1, seed=seed)
    obs = rng.normal(size=(n_obs, 1))
    log_b = gaussian_log_density(obs, model.means, model.covariances, covariance_type="diag")
    result = forward_backward(np.log(model.startprob), np.log(model.transmat), log_b)
    assert np.allclose(result.gamma.sum(axis=1), 1.0, atol=1e-10)
    if result.xi.shape[0] > 0:
        assert np.allclose(result.xi.sum(axis=(1, 2)), 1.0, atol=1e-10)


@pytest.mark.property
@pytest.mark.slow
@given(
    seed=st.integers(min_value=0, max_value=2_000),
    n_states=st.integers(min_value=2, max_value=3),
)
@_SETTINGS
def test_fitted_transition_rows_are_stochastic(seed: int, n_states: int) -> None:
    """A fitted model has a row-stochastic ``A`` and a simplex ``pi``."""
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(200, 1)) * 0.01
    model = fit_hmm(obs, n_states, n_restarts=2, max_iter=30, seed=seed)
    assert np.allclose(model.transmat.sum(axis=1), 1.0, atol=1e-10)
    assert bool((model.transmat >= 0.0).all())
    assert abs(float(model.startprob.sum()) - 1.0) < 1e-10
    assert bool((model.startprob >= 0.0).all())


@pytest.mark.property
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n_states=st.integers(min_value=2, max_value=4),
    n_obs=st.integers(min_value=15, max_value=80),
)
@_SETTINGS
def test_filtered_states_match_filter_argmax(seed: int, n_states: int, n_obs: int) -> None:
    """``filtered_states`` is exactly the argmax of the online-filter posterior."""
    rng = np.random.default_rng(seed)
    model = _toy_model(n_states, 1, seed=seed)
    obs = rng.normal(size=(n_obs, 1))
    posterior = online_filter(model, obs)
    expected = np.argmax(posterior, axis=1).astype(np.float64)
    assert np.array_equal(filtered_states(model, obs), expected)
