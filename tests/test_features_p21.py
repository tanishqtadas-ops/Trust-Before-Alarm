"""Unit tests for P2.1: Causal windows and sigma-hat estimator."""

import numpy as np
import pytest
from src.features.windows import CausalWindows, causal_slice
from src.features.sigma_hat import compute_sigma_hat


def test_causal_windows_config():
    """Verify windows load correctly from config/features.yaml."""
    cw = CausalWindows.from_config()
    assert cw.short == 5
    assert cw.medium == 30
    assert cw.long == 120
    assert cw.reference == 480
    assert cw.causal_only is True


def test_causal_slice_bounds():
    """Verify causal_slice strictly respects temporal causality bounds."""
    arr = np.arange(100)
    # At t=0, window=5 should only return [0]
    s0 = causal_slice(arr, 0, 5)
    assert len(s0) == 1
    assert s0[0] == 0

    # At t=10, window=5 should return [6, 7, 8, 9, 10]
    s10 = causal_slice(arr, 10, 5)
    assert len(s10) == 5
    assert np.array_equal(s10, np.array([6, 7, 8, 9, 10]))
    assert s10[-1] == 10


def test_sigma_hat_causality():
    """Verify modifying future readings does not change past sigma_hat."""
    np.random.seed(42)
    x = np.random.normal(20.0, 0.2, 1000)
    sigma_orig = compute_sigma_hat(x)

    # Modify future readings after t=500 drastically (spikes, huge noise, zeros)
    x_mod = x.copy()
    x_mod[501:] += np.random.normal(0, 50.0, len(x_mod) - 501)
    sigma_mod = compute_sigma_hat(x_mod)

    # For all t <= 500, sigma_hat MUST be identical
    np.testing.assert_allclose(sigma_orig[:501], sigma_mod[:501], rtol=1e-12, atol=1e-12)


def test_sigma_hat_estimation_accuracy():
    """Verify sigma_hat estimates stationary Gaussian noise accurately."""
    np.random.seed(123)
    true_sigma = 0.20
    noise = np.random.normal(0, true_sigma, 1920)
    # Slow drift
    t = np.arange(1920)
    trend = 22.0 + 2.0 * np.sin(2 * np.pi * t / 1440)
    x = trend + noise

    est = compute_sigma_hat(x)
    # After warm-up (t >= 480), estimate should be well calibrated to true_sigma
    scored_est = est[480:]
    median_est = np.median(scored_est)
    # Due to 25th percentile of 30-sample MADs, median is slightly below true sigma (~0.85x),
    # strictly bounded within [0.12, 0.25]
    assert 0.12 <= median_est <= 0.25
    assert np.all(est > 0)


def test_sigma_hat_robust_to_spikes():
    """Verify 25th percentile prevents temporary spike bursts from contaminating baseline noise."""
    np.random.seed(99)
    x = np.random.normal(20.0, 0.15, 1000)
    sigma_baseline = compute_sigma_hat(x)

    # Inject 20-step spike burst at t=600..620
    x_spiked = x.copy()
    x_spiked[600:620] += 10.0
    sigma_spiked = compute_sigma_hat(x_spiked)

    # At t=700 (within 480-step reference window after the burst),
    # 25th percentile ensures estimate remains close to baseline
    assert abs(sigma_spiked[700] - sigma_baseline[700]) < 0.05


def test_sigma_hat_zero_noise_and_frozen():
    """Verify constant/frozen signals floor at epsilon without division error."""
    x_frozen = np.full(500, 25.0)
    est = compute_sigma_hat(x_frozen)
    assert len(est) == 500
    assert np.all(est >= 1e-6)
    assert not np.any(np.isnan(est))
    assert not np.any(np.isinf(est))
