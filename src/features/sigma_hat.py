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

    local_sigma = pd.Series(mads * scale)

    # Step 3: 25th percentile of rolling estimates over the 480-step reference causally
    quantile_q = rolling_pct / 100.0
    rolling_ref = local_sigma.rolling(ref_win, min_periods=1).quantile(quantile_q).to_numpy(copy=True)

    # Step 4: Fallback for early or missing steps and floor at epsilon
    # If early steps don't have enough local windows, fallback to available differences up to step t
    is_nan = np.isnan(rolling_ref)
    if np.any(is_nan):
        # Fallback using cumulative causal differences where rolling_ref is NaN
        for idx in np.where(is_nan)[0]:
            sub_diff = diff[: idx + 1]
            valid_sub = sub_diff[~np.isnan(sub_diff)]
            if len(valid_sub) >= min_periods_local:
                m = np.median(valid_sub)
                mad_val = np.median(np.abs(valid_sub - m)) * scale
                rolling_ref[idx] = mad_val
            elif len(valid_sub) >= 1:
                rolling_ref[idx] = np.abs(valid_sub[-1]) * scale
            else:
                rolling_ref[idx] = epsilon

    # Floor at epsilon
    rolling_ref = np.maximum(np.nan_to_num(rolling_ref, nan=epsilon), epsilon)
    return rolling_ref
