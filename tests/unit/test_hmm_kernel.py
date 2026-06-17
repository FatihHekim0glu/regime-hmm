"""Unit tests for the Gaussian HMM kernel.

Covers the EM monotonic-LL invariant, the covariance-floor degeneracy guard, the
log-space forward/backward recursions, the online filter's no-lookahead structure,
and the validation error paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimehmm._exceptions import InsufficientDataError, ValidationError
from regimehmm.hmm.em import _init_params, em_single_run, fit_hmm
from regimehmm.hmm.filter import HMMModel, filtered_states, online_filter
from regimehmm.hmm.forward_backward import (
    backward_pass,
    forward_backward,
    forward_pass,
)
from regimehmm.hmm.kernel import floor_covariance, gaussian_log_density
from regimehmm.hmm.viterbi import viterbi_path


def _regime_obs(n_obs: int = 500, *, seed: int = 7) -> np.ndarray:
    """A seeded 2-state persistent-vol return series (single feature)."""
    rng = np.random.default_rng(seed)
    means = np.array([0.0005, -0.001])
    vols = np.array([0.006, 0.02])
    transmat = np.array([[0.97, 0.03], [0.05, 0.95]])
    obs = np.empty((n_obs, 1), dtype=np.float64)
    state = 0
    for t in range(n_obs):
        if t > 0:
            state = int(rng.choice(2, p=transmat[state]))
        obs[t, 0] = rng.normal(means[state], vols[state])
    return obs


# --------------------------------------------------------------------------- #
# Covariance floor / degeneracy guard                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_floor_covariance_diag_raises_floor() -> None:
    """Diagonal variances below the floor are raised exactly to the floor."""
    cov = np.array([[0.0, 1e-12], [4.0, 2.0]])
    floored = floor_covariance(cov, covariance_type="diag", floor=1e-6)
    assert floored.min() == pytest.approx(1e-6)
    assert floored[1, 0] == pytest.approx(4.0)


@pytest.mark.unit
def test_floor_covariance_full_is_spd() -> None:
    """A singular full covariance is floored to symmetric positive-definite."""
    cov = np.zeros((2, 2, 2))
    cov[1] = np.array([[1.0, 0.9], [0.9, 1.0]])
    floored = floor_covariance(cov, covariance_type="full", floor=1e-6)
    for k in range(2):
        eigvals = np.linalg.eigvalsh(floored[k])
        assert eigvals.min() >= 1e-6 - 1e-12
        assert np.allclose(floored[k], floored[k].T)


@pytest.mark.unit
def test_floor_covariance_rejects_bad_floor_and_rank() -> None:
    """Non-positive floor and wrong-rank covariances raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        floor_covariance(np.ones((2, 1)), floor=0.0)
    with pytest.raises(ValidationError):
        floor_covariance(np.ones((2, 2, 2)), covariance_type="diag")
    with pytest.raises(ValidationError):
        floor_covariance(np.ones((2, 1)), covariance_type="full")


@pytest.mark.unit
def test_gaussian_log_density_validates_shapes() -> None:
    """Malformed observation/means/covariance shapes raise ``ValidationError``."""
    obs = np.zeros((10, 2))
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValidationError):
        gaussian_log_density(np.zeros(10), means, np.ones((2, 2)))  # 1-D obs
    with pytest.raises(ValidationError):
        gaussian_log_density(obs, np.zeros(2), np.ones((2, 2)))  # 1-D means
    with pytest.raises(ValidationError):
        gaussian_log_density(obs, np.zeros((2, 3)), np.ones((2, 3)))  # feature mismatch
    with pytest.raises(ValidationError):
        # full covariances with wrong trailing shape
        gaussian_log_density(obs, means, np.ones((2, 3, 3)), covariance_type="full")


@pytest.mark.unit
def test_gaussian_log_density_full_matches_independent_features() -> None:
    """A diagonal full covariance matches the diag formula on 2 features."""
    obs = np.random.default_rng(0).normal(size=(50, 2))
    means = np.array([[0.0, 0.0], [0.3, -0.2]])
    diag = np.array([[0.5, 1.5], [2.0, 0.8]])
    full = np.stack([np.diag(diag[0]), np.diag(diag[1])])
    ld_diag = gaussian_log_density(obs, means, diag, covariance_type="diag")
    ld_full = gaussian_log_density(obs, means, full, covariance_type="full")
    assert np.allclose(ld_diag, ld_full, atol=1e-9)


@pytest.mark.unit
def test_gaussian_log_density_degenerate_state_is_finite() -> None:
    """A state collapsing onto identical points yields a finite (floored) density."""
    obs = np.zeros((10, 1))
    means = np.array([[0.0], [1.0]])
    covariances = np.array([[0.0], [1.0]])  # state 0 is degenerate
    log_b = gaussian_log_density(obs, means, covariances, covariance_type="diag", floor=1e-6)
    assert np.isfinite(log_b).all()


@pytest.mark.unit
def test_fit_hmm_survives_degenerate_data() -> None:
    """EM on data with a zero-variance cluster returns a finite, floored model."""
    obs = np.vstack([np.zeros((40, 1)), np.random.default_rng(0).normal(size=(160, 1))])
    model = fit_hmm(obs, 2, n_restarts=3, max_iter=40, seed=1, covariance_floor=1e-6)
    assert np.isfinite(model.log_likelihood)
    assert model.covariances.min() >= 1e-6 - 1e-12


# --------------------------------------------------------------------------- #
# EM monotonicity + reproducibility                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_em_log_likelihood_is_monotonic() -> None:
    """Within a single restart the per-iteration log-likelihood never decreases."""
    obs = _regime_obs()
    sp, tr, mu, cov = _init_params(obs, 2, covariance_type="diag", rng=np.random.default_rng(3))
    _, trace = em_single_run(obs, sp, tr, mu, cov, max_iter=100)
    diffs = np.diff(np.asarray(trace))
    assert bool((diffs >= -1e-6).all())
    assert len(trace) >= 2


@pytest.mark.unit
def test_fit_hmm_is_reproducible() -> None:
    """A fixed seed reproduces the fit byte-for-byte (means + LL + transmat)."""
    obs = _regime_obs()
    a = fit_hmm(obs, 2, n_restarts=4, max_iter=80, seed=7)
    b = fit_hmm(obs, 2, n_restarts=4, max_iter=80, seed=7)
    assert a.log_likelihood == b.log_likelihood
    assert np.array_equal(a.means, b.means)
    assert np.array_equal(a.transmat, b.transmat)


@pytest.mark.unit
def test_fit_hmm_recovers_two_persistent_regimes() -> None:
    """The fit separates the calm/turbulent regimes (sharply different vol)."""
    obs = _regime_obs(n_obs=800)
    model = fit_hmm(obs, 2, n_restarts=6, max_iter=100, seed=7)
    vols = np.sqrt(model.covariances[:, 0])
    # The two states have materially different volatility (regime structure found).
    assert vols.max() / vols.min() > 1.8
    assert model.converged


@pytest.mark.unit
def test_fit_hmm_validates_inputs() -> None:
    """Out-of-range scalars and malformed observations raise the right errors."""
    obs = _regime_obs(n_obs=50)
    with pytest.raises(ValidationError):
        fit_hmm(obs, 0)
    with pytest.raises(ValidationError):
        fit_hmm(obs, 2, n_restarts=0)
    with pytest.raises(ValidationError):
        fit_hmm(obs, 2, max_iter=0)
    with pytest.raises(ValidationError):
        fit_hmm(obs, 2, tol=0.0)
    with pytest.raises(ValidationError):
        fit_hmm(obs, 2, covariance_floor=0.0)
    with pytest.raises(ValidationError):
        fit_hmm(np.zeros(10), 2)  # 1-D
    with pytest.raises(InsufficientDataError):
        fit_hmm(np.zeros((1, 1)), 3)


# --------------------------------------------------------------------------- #
# Forward / backward recursions                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_forward_pass_likelihood_matches_full_pass() -> None:
    """``forward_pass`` LL equals the full forward-backward LL."""
    obs = _regime_obs(n_obs=200)
    model = fit_hmm(obs, 2, n_restarts=2, max_iter=40, seed=2)
    log_b = gaussian_log_density(obs, model.means, model.covariances, covariance_type="diag")
    log_pi = np.log(model.startprob)
    log_a = np.log(model.transmat)
    _, ll_fwd = forward_pass(log_pi, log_a, log_b)
    result = forward_backward(log_pi, log_a, log_b)
    assert ll_fwd == pytest.approx(result.log_likelihood, abs=1e-9)


@pytest.mark.unit
def test_backward_last_row_is_zero() -> None:
    """The backward message at the final step is the empty product (log 1 = 0)."""
    obs = _regime_obs(n_obs=120)
    model = fit_hmm(obs, 2, n_restarts=2, max_iter=30, seed=4)
    log_b = gaussian_log_density(obs, model.means, model.covariances, covariance_type="diag")
    log_beta = backward_pass(np.log(model.transmat), log_b)
    assert np.allclose(log_beta[-1], 0.0)


@pytest.mark.unit
def test_forward_backward_validates_shapes() -> None:
    """Inconsistent operand shapes raise ``ValidationError``."""
    log_b = np.zeros((10, 2))
    with pytest.raises(ValidationError):
        forward_pass(np.zeros(3), np.log(np.full((2, 2), 0.5)), log_b)
    with pytest.raises(ValidationError):
        forward_pass(np.log(np.full(2, 0.5)), np.zeros((3, 3)), log_b)
    with pytest.raises(ValidationError):
        forward_backward(np.log(np.full(2, 0.5)), np.log(np.full((2, 2), 0.5)), np.zeros((1,)))


@pytest.mark.unit
def test_single_observation_sequence_is_handled() -> None:
    """A length-1 sequence gives a valid posterior and an empty ``xi``."""
    model = HMMModel(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.9, 0.1], [0.1, 0.9]]),
        means=np.array([[0.0], [1.0]]),
        covariances=np.array([[1.0], [1.0]]),
        covariance_type="diag",
        log_likelihood=0.0,
        n_iter=1,
        converged=True,
    )
    obs = np.array([[0.3]])
    log_b = gaussian_log_density(obs, model.means, model.covariances, covariance_type="diag")
    result = forward_backward(np.log(model.startprob), np.log(model.transmat), log_b)
    assert result.xi.shape == (0, 2, 2)
    assert np.allclose(result.gamma.sum(axis=1), 1.0)
    posterior = online_filter(model, obs)
    assert np.allclose(posterior.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# Online filter / Viterbi structure                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_online_filter_no_lookahead_exact() -> None:
    """Perturbing returns after ``t`` leaves the filtered prefix byte-identical."""
    obs = _regime_obs(n_obs=300)
    model = fit_hmm(obs, 2, n_restarts=2, max_iter=40, seed=5)
    base = online_filter(model, obs)
    perturbed = obs.copy()
    perturbed[201:] += np.random.default_rng(9).normal(size=perturbed[201:].shape)
    after = online_filter(model, perturbed)
    assert np.array_equal(base[:201], after[:201])


@pytest.mark.unit
def test_filtered_and_viterbi_validate_feature_dim() -> None:
    """A feature-dim mismatch against the model raises ``ValidationError``."""
    obs = _regime_obs(n_obs=80)
    model = fit_hmm(obs, 2, n_restarts=2, max_iter=20, seed=6)
    bad = np.zeros((80, 2))  # model has 1 feature
    with pytest.raises(ValidationError):
        online_filter(model, bad)
    with pytest.raises(ValidationError):
        viterbi_path(model, bad)
    with pytest.raises(ValidationError):
        online_filter(model, np.zeros(80))  # 1-D


@pytest.mark.unit
def test_viterbi_and_filtered_states_are_integer_labels() -> None:
    """Both decoders return integer-valued labels in ``[0, n_states)``."""
    obs = _regime_obs(n_obs=200)
    model = fit_hmm(obs, 2, n_restarts=3, max_iter=40, seed=8)
    vit = viterbi_path(model, obs)
    fil = filtered_states(model, obs)
    for labels in (vit, fil):
        assert labels.shape == (200,)
        assert set(np.unique(labels)).issubset({0.0, 1.0})


@pytest.mark.unit
def test_hmm_model_dimensions_and_to_dict() -> None:
    """``HMMModel`` reports its dims and serializes to a JSON-able dict."""
    obs = _regime_obs(n_obs=120)
    model = fit_hmm(obs, 2, n_restarts=2, max_iter=30, seed=11)
    assert model.n_states == 2
    assert model.n_features == 1
    payload = model.to_dict()
    assert payload["covariance_type"] == "diag"
    assert len(payload["startprob"]) == 2
    assert isinstance(payload["converged"], bool)


@pytest.mark.unit
def test_em_single_run_validates_inputs() -> None:
    """``em_single_run`` rejects bad scalars and inconsistent parameter shapes."""
    obs = _regime_obs(n_obs=60)
    sp = np.array([0.5, 0.5])
    tr = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[0.0], [0.01]])
    cov = np.array([[1e-4], [4e-4]])
    with pytest.raises(ValidationError):
        em_single_run(np.zeros(10), sp, tr, mu, cov)  # 1-D obs
    with pytest.raises(ValidationError):
        em_single_run(obs, sp, tr, mu, cov, max_iter=0)
    with pytest.raises(ValidationError):
        em_single_run(obs, sp, tr, mu, cov, tol=0.0)
    with pytest.raises(ValidationError):
        em_single_run(obs, sp, tr, mu, cov, covariance_floor=0.0)
    with pytest.raises(ValidationError):
        em_single_run(obs, np.array([1.0]), tr, mu, cov)  # bad startprob shape
    with pytest.raises(ValidationError):
        em_single_run(obs, sp, np.zeros((3, 3)), mu, cov)  # bad transmat shape


@pytest.mark.unit
def test_em_monotonicity_violation_raises() -> None:
    """A genuine LL decrease (forced via a patched M-step) raises ``ValidationError``."""
    import regimehmm.hmm.em as em_mod

    obs = _regime_obs(n_obs=120)
    sp = np.array([0.5, 0.5])
    tr = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu = np.array([[0.0], [0.01]])
    cov = np.array([[1e-4], [4e-4]])

    calls = {"n": 0}
    real_m_step = em_mod._m_step

    def _bad_m_step(*args: object, **kwargs: object) -> object:
        # First M-step is honest (so iter 2 has a real LL); second corrupts the
        # parameters so the LL strictly decreases and trips the invariant.
        calls["n"] += 1
        pi2, a2, mu2, cov2 = real_m_step(*args, **kwargs)  # type: ignore[arg-type]
        if calls["n"] >= 2:
            mu2 = mu2 + 100.0  # wildly wrong means -> LL collapses
        return pi2, a2, mu2, cov2

    em_mod._m_step = _bad_m_step  # type: ignore[assignment]
    try:
        with pytest.raises(ValidationError, match="monotonicity"):
            em_single_run(obs, sp, tr, mu, cov, max_iter=10)
    finally:
        em_mod._m_step = real_m_step  # type: ignore[assignment]


@pytest.mark.unit
def test_fit_hmm_full_covariance_multifeature() -> None:
    """The full-covariance EM path fits a 2-feature series and stays SPD."""
    rng = np.random.default_rng(13)
    n_obs = 400
    transmat = np.array([[0.95, 0.05], [0.05, 0.95]])
    mean_a = np.array([0.0, 0.0])
    mean_b = np.array([0.5, -0.5])
    cov_a = np.array([[0.04, 0.01], [0.01, 0.04]])
    cov_b = np.array([[0.25, -0.1], [-0.1, 0.25]])
    obs = np.empty((n_obs, 2), dtype=np.float64)
    state = 0
    for t in range(n_obs):
        if t > 0:
            state = int(rng.choice(2, p=transmat[state]))
        mean = mean_a if state == 0 else mean_b
        cov = cov_a if state == 0 else cov_b
        obs[t] = rng.multivariate_normal(mean, cov)

    model = fit_hmm(obs, 2, covariance_type="full", n_restarts=4, max_iter=80, seed=7)
    assert model.covariances.shape == (2, 2, 2)
    assert np.isfinite(model.log_likelihood)
    for k in range(2):
        eigvals = np.linalg.eigvalsh(model.covariances[k])
        assert eigvals.min() > 0.0  # symmetric positive-definite
    # Filter still works through the full-covariance emission path.
    posterior = online_filter(model, obs)
    assert np.allclose(posterior.sum(axis=1), 1.0)


@pytest.mark.unit
def test_init_params_handles_more_states_than_obs() -> None:
    """``_init_params`` falls back to sampling with replacement when K > n_obs."""
    obs = _regime_obs(n_obs=3)
    sp, _tr, mu, cov = _init_params(obs, 5, covariance_type="diag", rng=np.random.default_rng(0))
    assert mu.shape == (5, 1)
    assert cov.shape == (5, 1)
    assert np.allclose(sp.sum(), 1.0)


@pytest.mark.unit
def test_full_covariance_path_matches_diag_on_one_feature() -> None:
    """With one feature, the full-covariance density equals the diagonal one."""
    obs = _regime_obs(n_obs=150)
    means = np.array([[0.0], [0.5]])
    diag_cov = np.array([[0.02], [0.05]])
    full_cov = diag_cov.reshape(2, 1, 1)
    ld_diag = gaussian_log_density(obs, means, diag_cov, covariance_type="diag")
    ld_full = gaussian_log_density(obs, means, full_cov, covariance_type="full")
    assert np.allclose(ld_diag, ld_full, atol=1e-10)
