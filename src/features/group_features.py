"""Group-level feature computation for Head C and group/event features joined to sensor rows.

Appendix B:
- Group/event joined onto sensor rows (6):
  event_breadth_s, event_breadth_m, group_change_s, group_change_m,
  group_dispersion, same_direction_frac.
- Group table for Head C (11):
  the six group/event features plus:
  group_slope_s, group_dev_baseline, n_valid_sensors, group_peak_m,
  group_noise_ratio.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np
import pandas as pd

from src.utils.config import get_config_dir, load_yaml
from src.features.windows import CausalWindows
from src.features.sensor_features import compute_fast_rolling_slope, compute_rolling_mad_noise


def compute_group_features(
    group_pivot_df: pd.DataFrame,
    sensor_sigma_hats: Dict[str, np.ndarray],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute group/event features and Head C group table causally.
    
    Args:
        group_pivot_df: DataFrame indexed by timestamp with columns as sensor_ids.
        sensor_sigma_hats: Dictionary mapping sensor_id -> sigma_hat array.
        config: Optional features configuration dict.
        
    Returns:
        Tuple of (joined_features_df, head_c_table_df):
        - joined_features_df: DataFrame with 6 features (indexed by timestamp).
        - head_c_table_df: DataFrame with 11 features (indexed by timestamp).
    """
    if config is None:
        cfg_path = get_config_dir() / "features.yaml"
        config = load_yaml(cfg_path)

    windows = CausalWindows.from_config(config)
    w_s = windows.short
    w_m = windows.medium
    w_ref = windows.reference

    changing_cfg = config.get("sensor_changing_rule", {})
    lag_s = int(changing_cfg.get("short_lag_steps", 15))
    lag_m = int(changing_cfg.get("medium_lag_steps", 60))
    thresh_mult = float(changing_cfg.get("threshold_sigma_multiplier", 3.0))

    timestamps = group_pivot_df.index
    n = len(group_pivot_df)
    sensor_ids = list(group_pivot_df.columns)
    n_sensors = len(sensor_ids)

    # 1. Per-sensor changing deltas and indicators
    delta_s_list = []
    delta_m_list = []
    is_changing_s_list = []
    is_changing_m_list = []

    for sid in sensor_ids:
        x_s = group_pivot_df[sid].to_numpy(dtype=float)
        sig = sensor_sigma_hats.get(sid, np.full(n, 1e-6))

        s_series = pd.Series(x_s)
        # Median of last 3 valid readings
        s_med3 = s_series.rolling(3, min_periods=1).median()
        s_lag15 = s_med3.shift(lag_s)
        s_lag60 = s_med3.shift(lag_m)

        d_s = (s_med3 - s_lag15).to_numpy(dtype=float)
        d_m = (s_med3 - s_lag60).to_numpy(dtype=float)

        delta_s_list.append(d_s)
        delta_m_list.append(d_m)

        ch_s = np.where(~np.isnan(d_s), np.abs(d_s) > (thresh_mult * sig), False)
        ch_m = np.where(~np.isnan(d_m), np.abs(d_m) > (thresh_mult * sig), False)

        is_changing_s_list.append(ch_s)
        is_changing_m_list.append(ch_m)

    arr_delta_s = np.column_stack(delta_s_list) if n_sensors > 0 else np.empty((n, 0))
    arr_delta_m = np.column_stack(delta_m_list) if n_sensors > 0 else np.empty((n, 0))
    arr_ch_s = np.column_stack(is_changing_s_list) if n_sensors > 0 else np.empty((n, 0))
    arr_ch_m = np.column_stack(is_changing_m_list) if n_sensors > 0 else np.empty((n, 0))

    readings = group_pivot_df.to_numpy(dtype=float)
    is_valid_reading = ~np.isnan(readings)
    n_valid_sensors = np.sum(is_valid_reading, axis=1).astype(float)

    # 2. Group aggregates
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        group_median = np.nanmedian(readings, axis=1)
        group_dispersion = np.nanstd(readings, axis=1)
        group_change_s = np.nanmedian(arr_delta_s, axis=1)
        group_change_m = np.nanmedian(arr_delta_m, axis=1)

    # event_breadth_s, event_breadth_m
    # Fraction of valid sensors changing
    valid_denom = np.maximum(n_valid_sensors, 1.0)
    breadth_s_counts = np.sum(arr_ch_s & is_valid_reading, axis=1).astype(float)
    breadth_m_counts = np.sum(arr_ch_m & is_valid_reading, axis=1).astype(float)

    event_breadth_s = np.where(n_valid_sensors > 0, breadth_s_counts / valid_denom, np.nan)
    event_breadth_m = np.where(n_valid_sensors > 0, breadth_m_counts / valid_denom, np.nan)

    # same_direction_frac: fraction of valid sensors agreeing with sign(group_change_m)
    sign_group_m = np.sign(group_change_m)
    sign_sensor_m = np.sign(arr_delta_m)
    agree_counts = np.sum((sign_sensor_m == sign_group_m[:, None]) & is_valid_reading, axis=1).astype(float)
    same_direction_frac = np.where(n_valid_sensors > 0, agree_counts / valid_denom, np.nan)

    # Dispersion is 0 if 1 or fewer valid sensors
    group_dispersion = np.where(n_valid_sensors <= 1, 0.0, group_dispersion)

    # --- Construct Joined 6 Group/Event Features ---
    joined_df = pd.DataFrame(index=timestamps)
    joined_df["event_breadth_s"] = event_breadth_s
    joined_df["event_breadth_m"] = event_breadth_m
    joined_df["group_change_s"] = group_change_s
    joined_df["group_change_m"] = group_change_m
    joined_df["group_dispersion"] = group_dispersion
    joined_df["same_direction_frac"] = same_direction_frac

    # --- Head C Table Features (11 features) ---
    head_c_df = joined_df.copy()

    # 7. group_slope_s: slope of group median series over short window (5)
    head_c_df["group_slope_s"] = compute_fast_rolling_slope(group_median, w_s)

    # 8. group_dev_baseline: group median - reference baseline median
    group_baseline = pd.Series(group_median).rolling(w_ref, min_periods=1).median().to_numpy()
    head_c_df["group_dev_baseline"] = group_median - group_baseline

    # 9. n_valid_sensors
    head_c_df["n_valid_sensors"] = n_valid_sensors

    # 10. group_peak_m: max - min of group median in medium window (30)
    g_max_m = pd.Series(group_median).rolling(w_m, min_periods=1).max().to_numpy()
    g_min_m = pd.Series(group_median).rolling(w_m, min_periods=1).min().to_numpy()
    head_c_df["group_peak_m"] = g_max_m - g_min_m

    # 11. group_noise_ratio: medium noise of group median / reference noise of group median
    mad_scale = float(config.get("sigma_hat", {}).get("mad_scale_factor", 1.4826))
    sqrt_2 = float(config.get("sigma_hat", {}).get("sqrt_2_divisor", 1.4142135623730951))
    scale = mad_scale / sqrt_2
    diff_g = np.diff(group_median, prepend=np.nan)
    noise_g_m = compute_rolling_mad_noise(diff_g, w_m, min_periods=3, scale=scale)
    noise_g_m_fb = pd.Series(diff_g).rolling(w_m, min_periods=3).std().to_numpy() / np.sqrt(2)
    noise_g_m = np.where(np.isnan(noise_g_m), noise_g_m_fb, noise_g_m)

    noise_g_ref = compute_rolling_mad_noise(diff_g, w_ref, min_periods=10, scale=scale)
    noise_g_ref_fb = pd.Series(diff_g).rolling(w_ref, min_periods=10).std().to_numpy() / np.sqrt(2)
    noise_g_ref = np.where(np.isnan(noise_g_ref), noise_g_ref_fb, noise_g_ref)
    eps_g_ref = np.maximum(np.nan_to_num(noise_g_ref, nan=1e-6), 1e-6)
    head_c_df["group_noise_ratio"] = noise_g_m / eps_g_ref

    return joined_df, head_c_df
