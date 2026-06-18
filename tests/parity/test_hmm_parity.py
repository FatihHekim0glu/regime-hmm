"""Parity of the hand-rolled Gaussian HMM kernel against ``hmmlearn``.

``hmmlearn`` is a DEV-ONLY oracle (never a runtime dep). These tests pin the
from-scratch kernel to ``hmmlearn.GaussianHMM`` to ``1e-6`` on seeded 2- and
3-state data for:

* the per-state Gaussian log emission density (``_compute_log_likelihood``);
* the smoothed posteriors ``gamma`` and the total sequence log-likelihood
  (``predict_proba`` / ``score``);
* the Viterbi MAP path (``predict`` / ``decode``) - an exact label match.

To make the comparison apples-to-apples we let ``hmmlearn`` fit, then evaluate our
kernel on ITS converged parameters (so any difference is in the math, not the EM
trajectory).
"""

from __future__ import annotations

import numpy as np
import pytest

from regimehmm.hmm.em import fit_hmm
from regimehmm.hmm.filter import HMMModel
from regimehmm.hmm.forward_backward import forward_backward
from regimehmm.hmm.kernel import gaussian_log_density
from regimehmm.hmm.viterbi import viterbi_path

hmm = pytest.importorskip("hmmlearn.hmm")

_TOL = 1e-6


def _make_obs(n_states: int, *, seed: int, n_obs: int = 700) -> np.ndarray:
    """Seeded persistent-vol return series with ``n_states`` regimes (1 feature)."""
    rng = np.random.default_rng(seed)
    means = np.linspace(-0.001, 0.001, n_states)
    vols = np.linspace(0.006, 0.025, n_states)
    persistence = 0.96
    off = (1.0 - persistence) / (n_states - 1)
    transmat = np.full((n_states, n_states), off)
    np.fill_diagonal(transmat, persistence)
    obs = np.empty((n_obs, 1), dtype=np.float64)
    state = 0
    for t in range(n_obs):
        if t > 0:
            state = int(rng.choice(n_states, p=transmat[state]))
        obs[t, 0] = rng.normal(means[state], vols[state])
    return obs


def _fit_reference(obs: np.ndarray, n_states: int) -> hmm.GaussianHMM:
    """Fit a diagonal ``hmmlearn`` reference model on ``obs``."""
    ref = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        random_state=0,
        min_covar=1e-6,
        tol=1e-6,
    )
    ref.fit(obs)
    return ref


def _model_from_reference(ref: hmm.GaussianHMM) -> HMMModel:
    """Wrap a fitted ``hmmlearn`` reference's params in our frozen ``HMMModel``."""
    n_states = ref.n_components
    n_features = ref.means_.shape[1]
    return HMMModel(
        startprob=np.asarray(ref.startprob_, dtype=np.float64),
        transmat=np.asarray(ref.transmat_, dtype=np.float64),
        means=np.asarray(ref.means_, dtype=np.float64),
        covariances=np.asarray(ref.covars_, dtype=np.float64).reshape(n_states, n_features),
        covariance_type="diag",
        log_likelihood=0.0,
        n_iter=int(ref.monitor_.iter),
        converged=True,
    )


@pytest.mark.parity
@pytest.mark.parametrize("n_states", [2, 3])
def test_log_density_matches_hmmlearn(n_states: int) -> None:
    """Our Gaussian log-density matches ``hmmlearn._compute_log_likelihood`` to 1e-6."""
    obs = _make_obs(n_states, seed=10 + n_states)
    ref = _fit_reference(obs, n_states)
    ref_log_b = ref._compute_log_likelihood(obs)  # oracle internal
    ours_log_b = gaussian_log_density(
        obs,
        np.asarray(ref.means_, dtype=np.float64),
        np.asarray(ref.covars_, dtype=np.float64).reshape(n_states, 1),
        covariance_type="diag",
    )
    assert np.max(np.abs(ref_log_b - ours_log_b)) < _TOL


@pytest.mark.parity
@pytest.mark.parametrize("n_states", [2, 3])
def test_smoothed_posteriors_and_ll_match_hmmlearn(n_states: int) -> None:
    """Smoothed ``gamma`` and total log-likelihood match ``hmmlearn`` to 1e-6."""
    obs = _make_obs(n_states, seed=20 + n_states)
    ref = _fit_reference(obs, n_states)
    ref_ll = float(ref.score(obs))
    ref_gamma = ref.predict_proba(obs)

    log_b = gaussian_log_density(
        obs,
        np.asarray(ref.means_, dtype=np.float64),
        np.asarray(ref.covars_, dtype=np.float64).reshape(n_states, 1),
        covariance_type="diag",
    )
    result = forward_backward(
        np.log(np.asarray(ref.startprob_, dtype=np.float64)),
        np.log(np.asarray(ref.transmat_, dtype=np.float64)),
        log_b,
    )
    assert abs(ref_ll - result.log_likelihood) < _TOL
    assert np.max(np.abs(ref_gamma - result.gamma)) < _TOL


@pytest.mark.parity
@pytest.mark.parametrize("n_states", [2, 3])
def test_viterbi_path_matches_hmmlearn(n_states: int) -> None:
    """The Viterbi MAP path matches ``hmmlearn.predict`` label-for-label."""
    obs = _make_obs(n_states, seed=30 + n_states)
    ref = _fit_reference(obs, n_states)
    ref_path = ref.predict(obs)
    model = _model_from_reference(ref)
    ours_path = viterbi_path(model, obs).astype(int)
    assert np.array_equal(ref_path, ours_path)


@pytest.mark.parity
@pytest.mark.slow
@pytest.mark.parametrize("n_states", [2, 3])
def test_em_reaches_comparable_optimum_to_hmmlearn(n_states: int) -> None:
    """Our Baum-Welch EM converges to a log-likelihood on par with ``hmmlearn``.

    Both fitters maximize the same likelihood; with multiple seeded restarts ours
    should reach an LL at least as high as ``hmmlearn``'s (modulo a small EM-local
    tolerance), confirming the M-step re-estimation matches the oracle.
    """
    obs = _make_obs(n_states, seed=40 + n_states)
    ref = _fit_reference(obs, n_states)
    ref_ll = float(ref.score(obs))
    ours = fit_hmm(obs, n_states, n_restarts=8, max_iter=200, seed=7)
    # Ours must not be materially WORSE than the oracle (it may be better thanks to
    # the extra restarts). Allow a small relative slack for EM local optima.
    assert ours.log_likelihood >= ref_ll - 1e-3 * abs(ref_ll)
