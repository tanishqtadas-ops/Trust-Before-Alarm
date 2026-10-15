"""Sigma-hat noise estimator implementation.

Appendix A.8:
- sigma_hat: 1.4826 x MAD of first differences, divided by sqrt(2), taking the
  25th percentile of rolling estimates over the 480-step reference.
- Causal only: estimate at step t depends only on readings at or before step t.
"""

from typing import Any, Dict, Optional, Union
import warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.utils.config import get_config_dir, load_yaml


def compute_sigma_hat(
    values: Union[np.ndarray, pd.Series],
    config: Optional[Dict[str, Any]] = None,
    local_window: Optional[int] = None,
    min_periods_local: int = 3,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute causal rolling sigma_hat noise floor for a sensor series.
    
    Args:
        values: 1D array or Series of sensor readings (may contain NaNs).
        config: Optional features configuration dict.
        local_window: Window size for local MAD of first differences (defaults to windows.medium = 30).
        min_periods_local: Minimum valid differences required to estimate local MAD.
        epsilon: Positive floor to prevent divide-by-zero on frozen/zero-noise signals.
        
    Returns:
        np.ndarray of length N containing strictly causal sigma_hat estimates.
    """
    if config is None:
        cfg_path = get_config_dir() / "features.yaml"
        config = load_yaml(cfg_path)

    sigma_cfg = config.get("sigma_hat", {})
    mad_scale = float(sigma_cfg.get("mad_scale_factor", 1.4826))
    sqrt_2 = float(sigma_cfg.get("sqrt_2_divisor", 1.4142135623730951))
    rolling_pct = float(sigma_cfg.get("rolling_percentile", 25.0))
    ref_win = int(sigma_cfg.get("reference_window", 480))

    if local_window is None:
        local_window = int(config.get("windows", {}).get("medium", 30))

    if isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    n = len(arr)
    if n == 0:
        return np.array([], dtype=float)

    # Step 1: First differences (causal, diff[0] = NaN)
    diff = np.diff(arr, prepend=np.nan)

    # Step 2: Causal rolling local MAD over local_window
    scale = mad_scale / sqrt_2
    padded = np.pad(diff, (local_window - 1, 0), constant_values=np.nan)
    windows = sliding_window_view(padded, window_shape=local_window)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        meds = np.nanmedian(windows, axis=1, keepdims=True)
        devs = np.abs(windows - meds)
        mads = np.nanmedian(devs, axis=1)

    valid_counts = np.sum(~np.isnan(windows), axis=1)
    mads[valid_counts < min_periods_local] = np.nan

    # When MAD collapses below float precision due to repeated differences from quantisation grid,
    # fallback to rolling standard deviation / sqrt(2) to preserve scale invariance
    zero_mad = (mads < 1e-11) & (valid_counts >= min_periods_local)
    if np.any(zero_mad):
        stds = np.nanstd(windows[zero_mad], axis=1) / np.sqrt(2)
        mads[zero_mad] = np.where(stds > 1e-11, stds, 0.0)

    local_sigma = pd.Series(mads * scale)

    # Step 3: 25th percentile of rolling estimates over the 480-step reference causally
    quantile_q = rolling_pct / 100.0
    rolling_ref = local_sigma.rolling(ref_win, min_periods=1).quantile(quantile_q).to_numpy(copy=True)

    # Step 4: Fallback for early or missing steps and floor at epsilon
    # If early steps don't have enough local windows or collapse due to repeats, fallback causally
    needs_fallback = np.isnan(rolling_ref) | (rolling_ref <= 1e-11)
    if np.any(needs_fallback):
        last_good = epsilon
        for idx in range(n):
            if needs_fallback[idx]:
                sub_diff = diff[: idx + 1]
                valid_sub = sub_diff[~np.isnan(sub_diff)]
                nonzero = valid_sub[np.abs(valid_sub) > 1e-11]
                if len(valid_sub) >= min_periods_local:
                    m = np.median(valid_sub)
                    mad_val = np.median(np.abs(valid_sub - m)) * scale
                    if mad_val > 1e-11:
                        rolling_ref[idx] = mad_val
                    elif len(nonzero) > 0:
                        std_val = np.std(nonzero) / np.sqrt(2)
                        rolling_ref[idx] = std_val if std_val > 1e-11 else np.abs(nonzero[-1]) * scale
                    else:
                        rolling_ref[idx] = last_good
                elif len(nonzero) > 0:
                    rolling_ref[idx] = np.abs(nonzero[-1]) * scale
                else:
                    rolling_ref[idx] = last_good
            if rolling_ref[idx] > 1e-11:
                last_good = rolling_ref[idx]

    # Floor at epsilon
    rolling_ref = np.maximum(np.nan_to_num(rolling_ref, nan=epsilon), epsilon)
    return rolling_ref


def estimate_q_hat(
    values: Union[np.ndarray, pd.Series],
    min_valid: int = 30,
    min_distinct: int = 2,
    epsilon_gap: float = 1e-6,
) -> np.ndarray:
    """Causally estimate quantisation grid step q_hat from sensor values.

    Definition:
        q_hat = smallest positive gap between distinct valid values seen so far.
        Causal: requires at least min_valid (30) valid readings and at least
        min_distinct (2) distinct values; otherwise NaN.
        Maintains causality: estimate at step t depends only on readings up to step t.

    Args:
        values: 1D array or Series of sensor readings (may contain NaNs).
        min_valid: Minimum valid readings required before emitting an estimate.
        min_distinct: Minimum distinct values required before emitting an estimate.
        epsilon_gap: Small threshold to distinguish distinct quantisation levels from float noise.

    Returns:
        np.ndarray of length N containing causal q_hat estimates.
    """
    if hasattr(values, "to_numpy"):
        arr = values.to_numpy(dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    n = len(arr)
    q_hat = np.full(n, np.nan, dtype=float)
    if n == 0:
        return q_hat

    valid_count = 0
    unique_set = set()
    sorted_unique = []
    min_gap = np.inf

    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            if i > 0 and not np.isnan(q_hat[i - 1]):
                q_hat[i] = q_hat[i - 1]
            continue

        valid_count += 1
        # Round to 7 decimal places to absorb IEEE 754 float representation noise
        v_round = round(float(v), 7)
        if v_round not in unique_set:
            unique_set.add(v_round)
            idx = np.searchsorted(sorted_unique, v_round)
            sorted_unique.insert(idx, v_round)
            if idx > 0:
                gap_l = sorted_unique[idx] - sorted_unique[idx - 1]
                if gap_l > epsilon_gap and gap_l < min_gap:
                    min_gap = gap_l
            if idx < len(sorted_unique) - 1:
                gap_r = sorted_unique[idx + 1] - sorted_unique[idx]
                if gap_r > epsilon_gap and gap_r < min_gap:
                    min_gap = gap_r

        if valid_count >= min_valid and len(unique_set) >= min_distinct and not np.isinf(min_gap):
            q_hat[i] = min_gap

    return q_hat


def compute_sigma_eff(
    values: Union[np.ndarray, pd.Series],
    config: Optional[Dict[str, Any]] = None,
    sigma_hat: Optional[np.ndarray] = None,
    q_hat: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute causal effective noise floor sigma_eff = max(sigma_hat, q_hat / sqrt(12)).

    Appendix A.8 & Decision 50:
        sigma_hat: 1.4826 x MAD of first differences / sqrt(2), 25th percentile of rolling 480-step estimates.
        q_hat: smallest positive gap between distinct valid values seen so far.
        sigma_eff: max(sigma_hat, q_hat / sqrt(12)), bounded below by epsilon.

    Args:
        values: 1D array or Series of sensor readings.
        config: Optional features configuration dict.
        sigma_hat: Optional precomputed sigma_hat array.
        q_hat: Optional precomputed q_hat array.
        epsilon: Minimum positive floor.

    Returns:
        np.ndarray of length N containing strictly causal, strictly positive sigma_eff.
    """
    if config is None:
        cfg_path = get_config_dir() / "features.yaml"
        config = load_yaml(cfg_path)

    q_floor_cfg = config.get("quantisation_floor", {})
    min_valid = int(q_floor_cfg.get("min_valid_readings", 30))
    min_distinct = int(q_floor_cfg.get("min_distinct_values", 2))
    sqrt_12 = float(q_floor_cfg.get("uniform_variance_divisor", 3.4641016151377544))

    if sigma_hat is None:
        sigma_hat = compute_sigma_hat(values, config=config, epsilon=epsilon)
    if q_hat is None:
        q_hat = estimate_q_hat(values, min_valid=min_valid, min_distinct=min_distinct)

    # Quantisation noise floor is q_hat / sqrt(12) where available, otherwise epsilon
    q_floor = np.where(~np.isnan(q_hat), q_hat / sqrt_12, epsilon)
    sigma_eff = np.maximum(sigma_hat, q_floor)
    sigma_eff = np.maximum(np.nan_to_num(sigma_eff, nan=epsilon), epsilon)
    return sigma_eff


