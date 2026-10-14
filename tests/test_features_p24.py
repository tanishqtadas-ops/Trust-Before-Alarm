"""Unit tests for P2.4: Group-level feature table and Head C features."""

import numpy as np
import pandas as pd
import pytest

from src.features.group_features import compute_group_features
from src.features.sigma_hat import compute_sigma_hat


def test_group_feature_names_and_counts():
    """Verify exact 6 joined group features and 11 Head C features."""
    n = 200
    df_group = pd.DataFrame({
        "s_0": 20.0 + np.random.normal(0, 0.1, n),
        "s_1": 20.2 + np.random.normal(0, 0.1, n),
        "s_2": 19.9 + np.random.normal(0, 0.1, n),
    })
    sig_hats = {c: compute_sigma_hat(df_group[c]) for c in df_group.columns}
    joined_df, head_c_df = compute_group_features(df_group, sig_hats)

    joined_cols = [
        "event_breadth_s", "event_breadth_m", "group_change_s",
        "group_change_m", "group_dispersion", "same_direction_frac",
    ]
    head_c_cols = joined_cols + [
        "group_slope_s", "group_dev_baseline", "n_valid_sensors",
        "group_peak_m", "group_noise_ratio",
    ]

    assert list(joined_df.columns) == joined_cols
    assert list(head_c_df.columns) == head_c_cols
    assert len(joined_df.columns) == 6
    assert len(head_c_df.columns) == 11


def test_genuine_event_breadth_and_direction():
    """Verify event breadth and direction agreement detect widespread environmental shifts."""
    np.random.seed(42)
    n = 300
    # 5 sensors in group
    sensors = [f"s_{i}" for i in range(5)]
    data = {s: 20.0 + np.random.normal(0, 0.1, n) for s in sensors}

    # Inject global warming event at t=150: all sensors rise by +3.0 C over 5 steps
    for s in sensors:
        data[s][150:] += 3.0

    df_group = pd.DataFrame(data)
    sig_hats = {c: compute_sigma_hat(df_group[c]) for c in sensors}
    joined_df, head_c_df = compute_group_features(df_group, sig_hats)

    # During event rise (t=155)
    # Event breadth should jump to 1.0 (100% of sensors changing)
    assert joined_df["event_breadth_s"].iloc[155] == 1.0
    # Direction agreement should be 1.0
    assert joined_df["same_direction_frac"].iloc[155] == 1.0
    # Group change should reflect the ~3.0 C rise
    assert joined_df["group_change_s"].iloc[155] > 2.5
    # Head C group deviation should be positive
    assert head_c_df["group_dev_baseline"].iloc[160] > 2.0


def test_isolated_fault_does_not_trigger_event_breadth():
    """Verify single-sensor fault produces low event breadth (Head C isolation)."""
    np.random.seed(123)
    n = 300
    sensors = [f"s_{i}" for i in range(6)]
    data = {s: 20.0 + np.random.normal(0, 0.1, n) for s in sensors}

    # Only sensor 0 suffers a massive +10 C drift
    data["s_0"][150:] += 10.0

    df_group = pd.DataFrame(data)
    sig_hats = {c: compute_sigma_hat(df_group[c]) for c in sensors}
    joined_df, head_c_df = compute_group_features(df_group, sig_hats)

    # Event breadth should remain far below majority (< 0.50), never indicating widespread event
    breadth_s = joined_df["event_breadth_s"].iloc[160:200]
    assert (breadth_s < 0.50).all()
    # Once step shift has settled, breadth drops back to 0.0
    assert (joined_df["event_breadth_s"].iloc[180:200] == 0.0).all()
    # Group change median should remain near zero because majority of sensors are healthy
    assert abs(joined_df["group_change_s"].iloc[170]) < 0.2


def test_single_sensor_group_features():
    """Verify group-level table computes successfully for a single-sensor group."""
    n = 200
    df_single = pd.DataFrame({"s_sole": 22.0 + np.random.normal(0, 0.15, n)})
    sig_hats = {"s_sole": compute_sigma_hat(df_single["s_sole"])}
    joined_df, head_c_df = compute_group_features(df_single, sig_hats)

    assert len(joined_df.columns) == 6
    assert len(head_c_df.columns) == 11
    assert (head_c_df["n_valid_sensors"] == 1.0).all()
    assert (joined_df["group_dispersion"] == 0.0).all()
