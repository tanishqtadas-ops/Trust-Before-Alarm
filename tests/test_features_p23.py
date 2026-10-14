"""Unit tests for P2.3: Cross-sensor features, self-exclusion, gap centering, and single-sensor fallback."""

import numpy as np
import pandas as pd
import pytest

from src.features.sensor_features import compute_cross_sensor_features
from src.features.sigma_hat import compute_sigma_hat


def test_cross_sensor_feature_names_and_counts():
    """Verify 14 cross-sensor features and 3 long-term degradation features."""
    n = 200
    t = np.arange(n)
    df_group = pd.DataFrame({
        "s_0": 20.0 + 0.01 * t + np.random.normal(0, 0.1, n),
        "s_1": 20.5 + 0.01 * t + np.random.normal(0, 0.1, n),
        "s_2": 19.8 + 0.01 * t + np.random.normal(0, 0.1, n),
    })
    sig = compute_sigma_hat(df_group["s_0"])
    feat = compute_cross_sensor_features("s_0", df_group, sig)

    expected_cols = [
        "noise_growth", "missing_growth",
        "gap", "gap_mean_m", "gap_slope_m", "gap_change_m", "gap_absmax_m",
        "gap_std_m", "gap_autocorr_m", "others_spread", "others_agreement",
        "diff_corr_m", "net_change_gap_m", "gain_slope_l", "gain_r2_l", "n_valid_others",
        "gap_slope_xl",
    ]
    for col in expected_cols:
        assert col in feat.columns, f"Missing feature {col}"
    assert len(feat.columns) == len(expected_cols)


def test_self_exclusion_guarantee():
    """Verify target sensor is strictly excluded when forming group reference."""
    n = 200
    base = 20.0 + np.random.normal(0, 0.1, n)
    df_clean = pd.DataFrame({
        "target": base.copy(),
        "other_1": base + 0.5,
        "other_2": base - 0.5,
    })
    sig_clean = compute_sigma_hat(df_clean["target"])
    feat_clean = compute_cross_sensor_features("target", df_clean, sig_clean)

    # Now corrupt target with massive spikes and extreme values
    df_corrupt = df_clean.copy()
    df_corrupt["target"] += 500.0  # +500 C spike
    sig_corrupt = compute_sigma_hat(df_corrupt["target"])
    feat_corrupt = compute_cross_sensor_features("target", df_corrupt, sig_corrupt)

    # Reference derived properties of the *others* must be 100% IDENTICAL
    np.testing.assert_allclose(feat_clean["others_spread"], feat_corrupt["others_spread"])
    np.testing.assert_allclose(feat_clean["n_valid_others"], feat_corrupt["n_valid_others"])
    np.testing.assert_allclose(feat_clean["others_agreement"].dropna(), feat_corrupt["others_agreement"].dropna())


def test_gap_centering_removes_static_offset():
    """Verify gap centering eliminates static offset so healthy sensors have zero centered gap."""
    np.random.seed(42)
    n = 600
    t = np.arange(n)
    env = 20.0 + np.sin(2 * np.pi * t / 200)

    # Sensor A has huge static offset +3.0 C, others have 0.0 C
    df_group = pd.DataFrame({
        "s_A": env + 3.0 + np.random.normal(0, 0.05, n),
        "s_B": env + 0.0 + np.random.normal(0, 0.05, n),
        "s_C": env + 0.0 + np.random.normal(0, 0.05, n),
    })
    sig = compute_sigma_hat(df_group["s_A"])
    feat = compute_cross_sensor_features("s_A", df_group, sig)

    # After initial reference history (t >= 480), centered gap must be ~0.0 C despite +3.0 C static offset
    scored_gap = feat["gap"].iloc[480:]
    assert abs(scored_gap.mean()) < 0.05
    assert (abs(scored_gap) < 0.30).all()


def test_single_sensor_fallback():
    """Verify all 14 cross-sensor features and gap_slope_xl are NaN when group size is 1."""
    n = 200
    df_single = pd.DataFrame({
        "sole_sensor": 20.0 + np.random.normal(0, 0.1, n)
    })
    sig = compute_sigma_hat(df_single["sole_sensor"])
    feat = compute_cross_sensor_features("sole_sensor", df_single, sig)

    cross_cols = [
        "gap", "gap_mean_m", "gap_slope_m", "gap_change_m", "gap_absmax_m",
        "gap_std_m", "gap_autocorr_m", "others_spread", "others_agreement",
        "diff_corr_m", "net_change_gap_m", "gain_slope_l", "gain_r2_l", "n_valid_others",
        "gap_slope_xl",
    ]
    for col in cross_cols:
        assert feat[col].isna().all(), f"{col} must be NaN for single-sensor fallback"

    # Degradation features on own signal should still be valid
    assert not feat["noise_growth"].isna().all()
    assert not feat["missing_growth"].isna().all()


def test_gain_slope_and_r2():
    """Verify gain regression slope captures sensor gain multiplier relative to group."""
    n = 300
    t = np.arange(n)
    # Environment with good variance
    env = 20.0 + 5.0 * np.sin(2 * np.pi * t / 100)

    # Target has gain 1.50
    df_group = pd.DataFrame({
        "s_gain": 20.0 + 1.50 * (env - 20.0) + np.random.normal(0, 0.05, n),
        "s_ref1": env + np.random.normal(0, 0.05, n),
        "s_ref2": env + np.random.normal(0, 0.05, n),
        "s_ref3": env + np.random.normal(0, 0.05, n),
    })
    sig = compute_sigma_hat(df_group["s_gain"])
    feat = compute_cross_sensor_features("s_gain", df_group, sig)

    # For t >= 120 (full long window), gain slope should be close to 1.50 and R^2 high
    slopes = feat["gain_slope_l"].iloc[120:]
    r2s = feat["gain_r2_l"].iloc[120:]
    assert np.allclose(slopes.median(), 1.50, atol=0.08)
    assert r2s.median() > 0.95
