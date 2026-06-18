"""Unit tests for the Gaussian kernel error branches and the full-covariance path.

Complements ``test_hmm_kernel.py`` by pinning the validation messages and the
full-covariance log-density branch that the diagonal-only tests do not reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimehmm._exceptions import ValidationError
from regimehmm.hmm.kernel import floor_covariance, gaussian_log_density


@pytest.mark.unit
def test_floor_covariance_rejects_bad_floor_and_rank() -> None:
    """A non-positive floor and a wrong-rank covariance both raise."""
    with pytest.raises(ValidationError, match="strictly positive"):
        floor_covariance(np.ones((2, 1)), floor=0.0)
    with pytest.raises(ValidationError, match="must be 2-D"):
        floor_covariance(np.ones((2, 1, 1)), covariance_type="diag")
    with pytest.raises(ValidationError, match="must be 3-D"):
        floor_covariance(np.ones((2, 2)), covariance_type="full")


@pytest.mark.unit
def test_floor_covariance_rejects_unknown_type() -> None:
    """An unsupported covariance type is rejected."""
    with pytest.raises(ValidationError, match="unsupported covariance_type"):
        floor_covariance(np.ones((2, 1)), covariance_type="banded")  # type: ignore[arg-type]


@pytest.mark.unit
def test_gaussian_log_density_full_matches_diag_for_diagonal_cov() -> None:
    """The full-covariance path agrees with the diag path on a diagonal covariance."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(40, 2))
    means = np.array([[0.0, 0.0], [1.0, -1.0]])
    diag_var = np.array([[1.0, 2.0], [0.5, 1.5]])
    full_cov = np.stack([np.diag(diag_var[k]) for k in range(2)])

    diag_density = gaussian_log_density(obs, means, diag_var, covariance_type="diag")
    full_density = gaussian_log_density(obs, means, full_cov, covariance_type="full")
    np.testing.assert_allclose(diag_density, full_density, atol=1e-10)


@pytest.mark.unit
def test_gaussian_log_density_validation_branches() -> None:
    """Rank mismatches across observations, means, and covariances all raise."""
    obs = np.zeros((10, 2))
    means = np.zeros((2, 2))
    cov = np.ones((2, 2))
    with pytest.raises(ValidationError, match="observations must be 2-D"):
        gaussian_log_density(np.zeros(10), means, cov)
    with pytest.raises(ValidationError, match="means must be 2-D"):
        gaussian_log_density(obs, np.zeros(2), cov)
    with pytest.raises(ValidationError, match="feature dim"):
        gaussian_log_density(obs, np.zeros((2, 3)), np.ones((2, 3)))
