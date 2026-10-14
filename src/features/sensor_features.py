"""Sensor-level feature computation (own-signal, statistical, data quality, cross-sensor).

Appendix B:
- Own-signal temporal (10): diff_1, absdiff_max_s, absdiff_max_m, slope_s, slope_m,
  slope_l, slope_change, dev_baseline, dev_duration, n_spikes_m.
- Statistical (6): noise_ratio_s, noise_ratio_m, noise_ratio_l, noise_short_long,
  kurtosis_m, range_m.
- Data quality (10): cur_missing, frac_missing_m, frac_missing_l, consec_missing,
  longest_gap_m, steps_since_valid, repeat_run_len, longest_repeat_m,
  frac_zero_diff_m, unique_ratio_m.
- Cross-sensor (14): gap, gap_mean_m, gap_slope_m, gap_change_m, gap_absmax_m,
  gap_std_m, gap_autocorr_m, others_spread, others_agreement, diff_corr_m,
  net_change_gap_m, gain_slope_l, gain_r2_l, n_valid_others.
- Long-term degradation (3): noise_growth, missing_growth, gap_slope_xl.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import warnings

from src.utils.config import get_config_dir, load_yaml
from src.features.windows import CausalWindows


def compute_fast_rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute causal rolling linear regression slope over a window with missing-data support.
    
    Slope is dy/dt where dt is 1 step (1 minute).
    Returns NaN if valid points in window < 2.
    """
    n = len(arr)
    slopes = np.full(n, np.nan, dtype=float)
    if n == 0:
        return slopes

    u_full = np.arange(window, dtype=float)
    u_mean = u_full.mean()
    u_var = np.sum((u_full - u_mean) ** 2)
    weights = (u_full - u_mean) / u_var

    # Convolve for fast evaluation on full valid windows
    # For causal window ending at i, values are arr[i - window + 1 : i + 1]
    conv = np.convolve(np.nan_to_num(arr, nan=0.0), weights[::-1], mode="full")[:n]

    # Mask of NaNs
    is_nan = np.isnan(arr)
    nan_counts = pd.Series(is_nan.astype(int)).rolling(window, min_periods=1).sum().to_numpy()

    for i in range(n):
        start = max(0, i - window + 1)
        w_len = i - start + 1
        nans = nan_counts[i]

        if w_len == window and nans == 0:
            slopes[i] = conv[i]
        else:
            w = arr[start : i + 1]
            valid_mask = ~np.isnan(w)
            v_cnt = np.sum(valid_mask)
            if v_cnt < 2:
                slopes[i] = np.nan
            else:
                u = np.arange(w_len, dtype=float)[valid_mask]
                y = w[valid_mask]
                u_bar = np.mean(u)
                denom = np.sum((u - u_bar) ** 2)
                if denom > 1e-12:
                    slopes[i] = np.sum((u - u_bar) * (y - np.mean(y))) / denom
                else:
                    slopes[i] = 0.0
    return slopes


def compute_rolling_mad_noise(diff: np.ndarray, window: int, min_periods: int = 3, scale: float = 1.048356) -> np.ndarray:
    """Compute causal rolling local MAD of first differences scaled by (1.4826 / sqrt(2))."""
    n = len(diff)
    if n == 0:
        return np.array([], dtype=float)

    padded = np.pad(diff, (window - 1, 0), constant_values=np.nan)
    windows = sliding_window_view(padded, window_shape=window)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        meds = np.nanmedian(windows, axis=1, keepdims=True)
        devs = np.abs(windows - meds)
        mads = np.nanmedian(devs, axis=1)

    valid_counts = np.sum(~np.isnan(windows), axis=1)
    mads[valid_counts < min_periods] = np.nan
    return mads * scale


def compute_own_signal_features(
    values: Union[np.ndarray, pd.Series],
    sigma_hat: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute 26 own-signal features (10 temporal, 6 statistical, 10 data quality)."""
    if config is None:
        cfg_path = get_config_dir() / "features.yaml"
        config = load_yaml(cfg_path)

    windows = CausalWindows.from_config(config)
    w_s = windows.short
    w_m = windows.medium
    w_l = windows.long
    w_ref = windows.reference

    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    n = len(arr)
    df = pd.DataFrame(index=np.arange(n))

    # --- 1. Own-Signal Temporal (10 features) ---
    # diff_1: first difference
    diff_1 = np.diff(arr, prepend=np.nan)
    abs_diff_1 = np.abs(diff_1)
    df["diff_1"] = diff_1

    # absdiff_max_s, absdiff_max_m
    df["absdiff_max_s"] = pd.Series(abs_diff_1).rolling(w_s, min_periods=1).max().to_numpy()
    df["absdiff_max_m"] = pd.Series(abs_diff_1).rolling(w_m, min_periods=1).max().to_numpy()

    # slope_s, slope_m, slope_l
    df["slope_s"] = compute_fast_rolling_slope(arr, w_s)
    df["slope_m"] = compute_fast_rolling_slope(arr, w_m)
    df["slope_l"] = compute_fast_rolling_slope(arr, w_l)

    # slope_change
    df["slope_change"] = df["slope_s"] - df["slope_m"]

    # dev_baseline: x_t - median(x_{t-w_ref+1..t})
    baseline_median = pd.Series(arr).rolling(w_ref, min_periods=1).median().to_numpy()
    dev_baseline = arr - baseline_median
    df["dev_baseline"] = dev_baseline

    # dev_duration: consecutive steps where |dev_baseline| > 3 * sigma_hat
    is_dev = ~np.isnan(dev_baseline) & (np.abs(dev_baseline) > 3.0 * sigma_hat)
    dev_duration = np.zeros(n, dtype=float)
    for i in range(n):
        if is_dev[i]:
            dev_duration[i] = dev_duration[i - 1] + 1.0 if i > 0 else 1.0
        else:
            dev_duration[i] = 0.0
    df["dev_duration"] = dev_duration

    # n_spikes_m: count of spikes in medium window (where |diff_1| > 3 * sigma_hat)
    is_spike = ~np.isnan(diff_1) & (abs_diff_1 > 3.0 * sigma_hat)
    df["n_spikes_m"] = pd.Series(is_spike.astype(float)).rolling(w_m, min_periods=1).sum().to_numpy()

    # --- 2. Statistical (6 features) ---
    mad_scale = float(config.get("sigma_hat", {}).get("mad_scale_factor", 1.4826))
    sqrt_2 = float(config.get("sigma_hat", {}).get("sqrt_2_divisor", 1.4142135623730951))
    scale = mad_scale / sqrt_2

    noise_s = compute_rolling_mad_noise(diff_1, w_s, min_periods=2, scale=scale)
    noise_m = compute_rolling_mad_noise(diff_1, w_m, min_periods=3, scale=scale)
    noise_l = compute_rolling_mad_noise(diff_1, w_l, min_periods=5, scale=scale)

    # Fallback to diff std when noise is NaN
    noise_s_fallback = pd.Series(diff_1).rolling(w_s, min_periods=2).std().to_numpy() / np.sqrt(2)
    noise_s = np.where(np.isnan(noise_s), noise_s_fallback, noise_s)

    noise_m_fallback = pd.Series(diff_1).rolling(w_m, min_periods=3).std().to_numpy() / np.sqrt(2)
    noise_m = np.where(np.isnan(noise_m), noise_m_fallback, noise_m)

    noise_l_fallback = pd.Series(diff_1).rolling(w_l, min_periods=5).std().to_numpy() / np.sqrt(2)
    noise_l = np.where(np.isnan(noise_l), noise_l_fallback, noise_l)

    eps_sigma = np.maximum(sigma_hat, 1e-6)
    noise_ratio_s = noise_s / eps_sigma
    noise_ratio_m = noise_m / eps_sigma
    noise_ratio_l = noise_l / eps_sigma

    df["noise_ratio_s"] = noise_ratio_s
    df["noise_ratio_m"] = noise_ratio_m
    df["noise_ratio_l"] = noise_ratio_l
    df["noise_short_long"] = np.divide(
        noise_s,
        noise_l,
        out=np.full(n, np.nan, dtype=float),
        where=(noise_l > 1e-6) & (~np.isnan(noise_s)) & (~np.isnan(noise_l)),
    )

    # kurtosis_m: excess kurtosis in medium window (30)
    kurtosis_m = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - w_m + 1)
        w = arr[start : i + 1]
        v = w[~np.isnan(w)]
        if len(v) >= 4:
            m2 = np.mean((v - np.mean(v)) ** 2)
            if m2 > 1e-12:
                m4 = np.mean((v - np.mean(v)) ** 4)
                kurtosis_m[i] = (m4 / (m2 ** 2)) - 3.0
            else:
                kurtosis_m[i] = 0.0
    df["kurtosis_m"] = kurtosis_m

    # range_m: max - min in window 30
    roll_max_m = pd.Series(arr).rolling(w_m, min_periods=1).max().to_numpy()
    roll_min_m = pd.Series(arr).rolling(w_m, min_periods=1).min().to_numpy()
    df["range_m"] = roll_max_m - roll_min_m

    # --- 3. Data Quality (10 features) ---
    cur_missing = np.isnan(arr).astype(float)
    df["cur_missing"] = cur_missing
    df["frac_missing_m"] = pd.Series(cur_missing).rolling(w_m, min_periods=1).mean().to_numpy()
    df["frac_missing_l"] = pd.Series(cur_missing).rolling(w_l, min_periods=1).mean().to_numpy()

    consec_missing = np.zeros(n, dtype=float)
    steps_since_valid = np.zeros(n, dtype=float)
    repeat_run_len = np.zeros(n, dtype=float)

    for i in range(n):
        if cur_missing[i]:
            consec_missing[i] = consec_missing[i - 1] + 1.0 if i > 0 else 1.0
            steps_since_valid[i] = steps_since_valid[i - 1] + 1.0 if i > 0 else 1.0
            repeat_run_len[i] = 0.0
        else:
            consec_missing[i] = 0.0
            steps_since_valid[i] = 0.0
            if i > 0 and not cur_missing[i - 1] and arr[i] == arr[i - 1]:
                repeat_run_len[i] = repeat_run_len[i - 1] + 1.0
            else:
                repeat_run_len[i] = 1.0

    df["consec_missing"] = consec_missing
    df["steps_since_valid"] = steps_since_valid
    df["repeat_run_len"] = repeat_run_len

    longest_gap_m = np.zeros(n, dtype=float)
    longest_repeat_m = np.zeros(n, dtype=float)
    frac_zero_diff_m = np.full(n, np.nan, dtype=float)
    unique_ratio_m = np.full(n, np.nan, dtype=float)

    for i in range(n):
        start = max(0, i - w_m + 1)
        w_arr = arr[start : i + 1]
        w_miss = cur_missing[start : i + 1]
        w_diff = diff_1[start : i + 1]

        # Longest missing gap in window
        gaps = np.diff(np.concatenate(([0.0], w_miss, [0.0])))
        starts = np.where(gaps == 1.0)[0]
        ends = np.where(gaps == -1.0)[0]
        longest_gap_m[i] = float(np.max(ends - starts)) if len(starts) > 0 else 0.0

        # Valid readings in window
        valid_w = w_arr[~np.isnan(w_arr)]
        if len(valid_w) > 0:
            unique_ratio_m[i] = float(len(np.unique(valid_w))) / float(len(valid_w))
            # Longest repeat run in window
            max_rep = 1
            cur_rep = 1
            for k in range(1, len(valid_w)):
                if valid_w[k] == valid_w[k - 1]:
                    cur_rep += 1
                    if cur_rep > max_rep:
                        max_rep = cur_rep
                else:
                    cur_rep = 1
            longest_repeat_m[i] = float(max_rep)
        else:
            longest_repeat_m[i] = 0.0

        # Frac zero diff in window
        valid_d = w_diff[~np.isnan(w_diff)]
        if len(valid_d) > 0:
            frac_zero_diff_m[i] = float(np.mean(valid_d == 0.0))

    df["longest_gap_m"] = longest_gap_m
    df["steps_since_valid"] = steps_since_valid
    df["repeat_run_len"] = repeat_run_len
    df["longest_repeat_m"] = longest_repeat_m
    df["frac_zero_diff_m"] = frac_zero_diff_m
    df["unique_ratio_m"] = unique_ratio_m

    # Reorder DataFrame columns to strictly match Appendix B
    ordered_cols = [
        # Own-signal temporal (10)
        "diff_1", "absdiff_max_s", "absdiff_max_m", "slope_s", "slope_m",
        "slope_l", "slope_change", "dev_baseline", "dev_duration", "n_spikes_m",
        # Statistical (6)
        "noise_ratio_s", "noise_ratio_m", "noise_ratio_l", "noise_short_long",
        "kurtosis_m", "range_m",
        # Data quality (10)
        "cur_missing", "frac_missing_m", "frac_missing_l", "consec_missing",
        "longest_gap_m", "steps_since_valid", "repeat_run_len", "longest_repeat_m",
        "frac_zero_diff_m", "unique_ratio_m",
    ]
    df = df[ordered_cols]
    return df


def compute_cross_sensor_features(
    target_sensor_id: str,
    group_pivot_df: pd.DataFrame,
    sigma_hat: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compute 14 cross-sensor features and 3 long-term degradation features for a target sensor.
    
    Self-exclusion rule: the group reference is formed strictly from OTHER sensors in the group.
    Single-sensor fallback: all 14 cross-sensor features and gap_slope_xl are NaN if group size == 1.
    """
    if config is None:
        cfg_path = get_config_dir() / "features.yaml"
        config = load_yaml(cfg_path)

    windows = CausalWindows.from_config(config)
    w_m = windows.medium
    w_l = windows.long
    w_ref = windows.reference

    n = len(group_pivot_df)
    df = pd.DataFrame(index=group_pivot_df.index)

    # Sensor's own series
    x_target = group_pivot_df[target_sensor_id].to_numpy(dtype=float)

    # Other sensors in group (strictly excluding target sensor)
    other_cols = [c for c in group_pivot_df.columns if c != target_sensor_id]
    n_others = len(other_cols)

    # --- Long-term degradation features (own signal components) ---
    # noise_growth: noise_l / sigma_hat_ref
    mad_scale = float(config.get("sigma_hat", {}).get("mad_scale_factor", 1.4826))
    sqrt_2 = float(config.get("sigma_hat", {}).get("sqrt_2_divisor", 1.4142135623730951))
    scale = mad_scale / sqrt_2
    diff_target = np.diff(x_target, prepend=np.nan)
    noise_l = compute_rolling_mad_noise(diff_target, w_l, min_periods=5, scale=scale)
    noise_l_fallback = pd.Series(diff_target).rolling(w_l, min_periods=5).std().to_numpy() / np.sqrt(2)
    noise_l = np.where(np.isnan(noise_l), noise_l_fallback, noise_l)
    eps_sigma = np.maximum(sigma_hat, 1e-6)
    df["noise_growth"] = noise_l / eps_sigma

    # missing_growth: frac_missing_l - frac_missing_ref
    cur_missing = np.isnan(x_target).astype(float)
    frac_missing_l = pd.Series(cur_missing).rolling(w_l, min_periods=1).mean().to_numpy()
    frac_missing_ref = pd.Series(cur_missing).rolling(w_ref, min_periods=1).mean().to_numpy()
    df["missing_growth"] = frac_missing_l - frac_missing_ref

    # If single sensor: cross-sensor features are NaN
    if n_others == 0:
        for col in [
            "gap", "gap_mean_m", "gap_slope_m", "gap_change_m", "gap_absmax_m",
            "gap_std_m", "gap_autocorr_m", "others_spread", "others_agreement",
            "diff_corr_m", "net_change_gap_m", "gain_slope_l", "gain_r2_l", "n_valid_others",
            "gap_slope_xl",
        ]:
            df[col] = np.nan
        return df

    # Multi-sensor group: compute reference series r_t across other sensors
    others_arr = group_pivot_df[other_cols].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        r = np.nanmedian(others_arr, axis=1)
        n_valid_others = np.sum(~np.isnan(others_arr), axis=1).astype(float)
        others_spread = np.nanstd(others_arr, axis=1)

    # Raw gap: x_t - r_t
    raw_gap = x_target - r

    # Gap centering (Appendix A.8): relative to sensor's own median gap over 480-step reference
    ref_gap = pd.Series(raw_gap).rolling(w_ref, min_periods=1).median().to_numpy()
    # Centered gap
    centered_gap = raw_gap - ref_gap
    df["gap"] = centered_gap

    # gap_mean_m, gap_absmax_m, gap_std_m
    df["gap_mean_m"] = pd.Series(centered_gap).rolling(w_m, min_periods=1).mean().to_numpy()
    df["gap_absmax_m"] = pd.Series(np.abs(centered_gap)).rolling(w_m, min_periods=1).max().to_numpy()
    df["gap_std_m"] = pd.Series(centered_gap).rolling(w_m, min_periods=2).std().to_numpy()

    # gap_slope_m, gap_slope_xl
    df["gap_slope_m"] = compute_fast_rolling_slope(centered_gap, w_m)
    df["gap_slope_xl"] = compute_fast_rolling_slope(centered_gap, w_ref)

    # gap_change_m: change in centered gap over window 30
    gap_short_mean = pd.Series(centered_gap).rolling(windows.short, min_periods=1).mean().to_numpy()
    df["gap_change_m"] = gap_short_mean - df["gap_mean_m"]

    # gap_autocorr_m: lag-1 autocorrelation of centered gap over window 30
    gap_autocorr_m = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - w_m + 1)
        w = centered_gap[start : i + 1]
        v = w[~np.isnan(w)]
        if len(v) >= 5:
            x_lead = v[1:]
            x_lag = v[:-1]
            s_lead = np.std(x_lead)
            s_lag = np.std(x_lag)
            if s_lead > 1e-12 and s_lag > 1e-12:
                r_corr = np.corrcoef(x_lead, x_lag)[0, 1]
                gap_autocorr_m[i] = r_corr if not np.isnan(r_corr) else 0.0
            else:
                gap_autocorr_m[i] = 0.0
    df["gap_autocorr_m"] = gap_autocorr_m

    # others_spread
    df["others_spread"] = np.where(n_valid_others <= 1.0, 0.0, others_spread)

    # others_agreement: fraction of other sensors whose short-term direction matches group ref
    diff_others = np.diff(others_arr, axis=0, prepend=np.nan)
    diff_r = np.diff(r, prepend=np.nan)
    sign_r = np.sign(diff_r)

    others_agreement = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if n_valid_others[i] <= 1.0:
            others_agreement[i] = 1.0 if n_valid_others[i] == 1.0 else np.nan
        else:
            sr = sign_r[i]
            so = np.sign(diff_others[i, :])
            valid_so = so[~np.isnan(so)]
            if len(valid_so) > 0:
                others_agreement[i] = float(np.mean(valid_so == sr))
            else:
                others_agreement[i] = 1.0
    df["others_agreement"] = others_agreement

    # diff_corr_m: correlation between sensor diff and ref diff over window 30
    diff_corr_m = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - w_m + 1)
        w_s_diff = diff_target[start : i + 1]
        w_r_diff = diff_r[start : i + 1]
        valid_mask = ~np.isnan(w_s_diff) & ~np.isnan(w_r_diff)
        if np.sum(valid_mask) >= 5:
            vs = w_s_diff[valid_mask]
            vr = w_r_diff[valid_mask]
            std_s = np.std(vs)
            std_r = np.std(vr)
            if std_s > 1e-12 and std_r > 1e-12:
                c = np.corrcoef(vs, vr)[0, 1]
                diff_corr_m[i] = c if not np.isnan(c) else 0.0
            else:
                diff_corr_m[i] = 0.0
    df["diff_corr_m"] = diff_corr_m

    # net_change_gap_m: (x_t - x_{t-w_m}) - (r_t - r_{t-w_m})
    x_change_m = arr_lag_diff(x_target, w_m)
    r_change_m = arr_lag_diff(r, w_m)
    df["net_change_gap_m"] = x_change_m - r_change_m

    # gain_slope_l, gain_r2_l: regression of x_target onto r over long window 120
    gain_slope_l = np.full(n, np.nan, dtype=float)
    gain_r2_l = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = max(0, i - w_l + 1)
        w_y = x_target[start : i + 1]
        w_x = r[start : i + 1]
        valid_mask = ~np.isnan(w_y) & ~np.isnan(w_x)
        if np.sum(valid_mask) >= 10:
            vy = w_y[valid_mask]
            vx = w_x[valid_mask]
            var_x = np.var(vx)
            var_y = np.var(vy)
            if var_x > 1e-12:
                cov_xy = np.mean((vx - np.mean(vx)) * (vy - np.mean(vy)))
                slope = cov_xy / var_x
                gain_slope_l[i] = slope
                if var_y > 1e-12:
                    r_val = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))
                    gain_r2_l[i] = float(np.clip(r_val ** 2, 0.0, 1.0))
                else:
                    gain_r2_l[i] = 0.0
            else:
                gain_slope_l[i] = 1.0
                gain_r2_l[i] = 0.0
        elif np.sum(valid_mask) >= 2:
            gain_slope_l[i] = 1.0
            gain_r2_l[i] = 0.0
    df["gain_slope_l"] = gain_slope_l
    df["gain_r2_l"] = gain_r2_l

    # n_valid_others
    df["n_valid_others"] = n_valid_others

    return df


def arr_lag_diff(arr: np.ndarray, lag: int) -> np.ndarray:
    """Compute x_t - x_{t-lag} causally using most recent valid reading if needed."""
    n = len(arr)
    res = np.full(n, np.nan, dtype=float)
    s = pd.Series(arr)
    s_cur = s.ffill()
    s_lag = s_cur.shift(lag)
    res = (s_cur - s_lag).to_numpy()
    return res
