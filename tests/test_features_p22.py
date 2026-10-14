"""Unit tests for P2.2: Own-signal temporal, statistical, and data-quality features."""

import numpy as np
import pandas as pd
import pytest

from src.features.sensor_features import compute_own_signal_features
from src.features.sigma_hat import compute_sigma_hat


def test_own_signal_feature_counts_and_names():
    """Verify exact 26 own-signal features are generated with matching names."""
    x = np.random.normal(20.0, 0.2, 500)
    sig = compute_sigma_hat(x)
    df = compute_own_signal_features(x, sig)

    temporal_cols = [
        "diff_1", "absdiff_max_s", "absdiff_max_m", "slope_s", "slope_m",
        "slope_l", "slope_change", "dev_baseline", "dev_duration", "n_spikes_m",
    ]
    statistical_cols = [
        "noise_ratio_s", "noise_ratio_m", "noise_ratio_l", "noise_short_long",
        "kurtosis_m", "range_m",
    ]
    data_quality_cols = [
        "cur_missing", "frac_missing_m", "frac_missing_l", "consec_missing",
        "longest_gap_m", "steps_since_valid", "repeat_run_len", "longest_repeat_m",
        "frac_zero_diff_m", "unique_ratio_m",
    ]

    expected_cols = temporal_cols + statistical_cols + data_quality_cols
    assert len(expected_cols) == 26
    assert list(df.columns) == expected_cols


def test_constant_frozen_signal_properties():
    """Verify data quality and temporal features for an exactly constant frozen signal."""
    n = 200
    x = np.full(n, 22.5)
    sig = compute_sigma_hat(x)
    df = compute_own_signal_features(x, sig)

    # First diffs and slopes must be 0
    np.testing.assert_allclose(df["diff_1"].iloc[1:], 0.0)
    np.testing.assert_allclose(df["slope_s"].iloc[5:], 0.0, atol=1e-10)
    np.testing.assert_allclose(df["slope_m"].iloc[30:], 0.0, atol=1e-10)

    # Repeat run length must monotonically increase to n
    assert df["repeat_run_len"].iloc[-1] == n
    assert df["longest_repeat_m"].iloc[-1] == 30.0
    assert df["frac_zero_diff_m"].iloc[-1] == 1.0
    assert df["unique_ratio_m"].iloc[-1] == 1.0 / 30.0

    # No missing readings
    assert (df["cur_missing"] == 0.0).all()
    assert (df["consec_missing"] == 0.0).all()


def test_linear_ramp_slope_tracking():
    """Verify linear regression slopes accurately track constant ramp slope."""
    n = 300
    slope_true = 0.05  # 0.05 C/min
    t = np.arange(n)
    x = 20.0 + slope_true * t
    sig = compute_sigma_hat(x)
    df = compute_own_signal_features(x, sig)

    # Check slopes after window is full
    np.testing.assert_allclose(df["slope_s"].iloc[10:], slope_true, atol=1e-6)
    np.testing.assert_allclose(df["slope_m"].iloc[40:], slope_true, atol=1e-6)
    np.testing.assert_allclose(df["slope_l"].iloc[130:], slope_true, atol=1e-6)
    np.testing.assert_allclose(df["slope_change"].iloc[40:], 0.0, atol=1e-6)


def test_missing_data_quality_tracking():
    """Verify missing data metrics correctly track bursts and recovery."""
    n = 200
    x = np.random.normal(20.0, 0.2, n)
    # Inject a 7-step missing burst at t=50..56
    x[50:57] = np.nan
    sig = compute_sigma_hat(x)
    df = compute_own_signal_features(x, sig)

    # During missing burst
    assert (df["cur_missing"].iloc[50:57] == 1.0).all()
    assert df["consec_missing"].iloc[56] == 7.0
    assert df["steps_since_valid"].iloc[56] == 7.0

    # Longest gap in window 30 should reflect 7.0
    assert df["longest_gap_m"].iloc[60] == 7.0

    # At t=57 (first valid reading after burst)
    assert df["cur_missing"].iloc[57] == 0.0
    assert df["consec_missing"].iloc[57] == 0.0
    assert df["steps_since_valid"].iloc[57] == 0.0


def test_spike_detection_and_kurtosis():
    """Verify n_spikes_m and kurtosis respond to isolated spikes."""
    np.random.seed(42)
    n = 200
    x = np.random.normal(20.0, 0.1, n)
    # Inject a 15-sigma spike at t=80
    x[80] += 5.0
    sig = compute_sigma_hat(x)
    df = compute_own_signal_features(x, sig)

    # Spike count in window 30 covering t=80
    assert df["n_spikes_m"].iloc[85] >= 1.0
    # Absdiff max should reflect the ~5.0 jump
    assert df["absdiff_max_s"].iloc[80] > 4.5
    # Kurtosis at t=85 should be heavily positive (peaked outlier)
    assert df["kurtosis_m"].iloc[85] > 3.0


def test_dev_baseline_and_duration():
    """Verify dev_baseline and dev_duration count sustained displacement."""
    np.random.seed(101)
    n = 600
    # First 480 steps healthy
    x = np.random.normal(20.0, 0.1, n)
    # Sustained shift of +2.0 C from step 500 onwards (20 sigma shift)
    x[500:] += 2.0
    sig = compute_sigma_hat(x)
    df = compute_own_signal_features(x, sig)

    # At t=550 (51 steps into shift)
    assert df["dev_baseline"].iloc[550] > 1.0
    assert df["dev_duration"].iloc[550] == 51.0
