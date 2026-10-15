"""Raw alarm detector and cause labeling.

Specifications locked from Appendix A.9 and config/alarm.yaml:
- 8-hour median and robust spread (1.4826 x MAD of readings themselves)
- Baseline band: [median - 3 x robust_spread, median + 3 x robust_spread]
- Starts at 1 reading outside the band (persistence_start_readings = 1)
- Ends after 10 consecutive readings inside (consecutive_inside_to_end = 10)
- Episodes merge when gaps are 5 readings or fewer (episode_merge_gap_max = 5)
- Cause labels (from truth, evaluation-only): EVENT, FAULT, BOTH, NEITHER using 3 x sigma_k rule
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.utils.config import get_config_dir, load_yaml


# Truth columns strictly forbidden in detector inputs
PROHIBITED_TRUTH_COLUMNS = {
    "true_value",
    "sensor_untrusted",
    "f_spike",
    "f_offset_drift",
    "f_gain_error",
    "f_frozen",
    "f_high_noise",
    "f_dropout",
    "f_slow_degradation",
    "fault_severity",
    "true_health",
    "fault_id",
    "genuine_event",
    "event_below_detectability",
    "event_id",
    "event_magnitude_sigma",
    "true_event_breadth",
    "true_group_value",
}


class RawAlarmEpisode(NamedTuple):
    """Represents a merged raw alarm episode."""
    scenario_id: str
    group_id: str
    sensor_id: str
    start_step: int
    end_step: int
    duration: int
    median_base: float
    spread_base: float
    lower_bound: float
    upper_bound: float
    min_value: float
    max_value: float


def compute_robust_spread(values: np.ndarray, mad_factor: float = 1.4826, epsilon: float = 1e-6) -> Tuple[float, float]:
    """Compute median and robust spread = 1.4826 x MAD of readings themselves.
    
    Args:
        values: 1D array of readings (may contain NaNs).
        mad_factor: Scale factor for normal consistency (1.4826).
        epsilon: Positive floor to prevent zero spread on frozen/constant series.
        
    Returns:
        (median, robust_spread)
    """
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan, epsilon
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    spread = max(float(mad_factor * mad), epsilon)
    return med, spread


def compute_causal_rolling_bands(
    values: np.ndarray,
    window_steps: int = 480,
    min_periods: int = 3,
    mad_factor: float = 1.4826,
    epsilon: float = 1e-6,
    threshold_multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute causal 8-hour rolling median, robust spread, and alarm bounds per Appendix A.9.
    
    At step t, the reference window comprises up to window_steps (480) past and current
    readings: tau in [max(0, t - window_steps + 1), t]. No future readings are used.
    
    Args:
        values: 1D array of sensor readings of length N.
        window_steps: Causal window length in steps (480 for 8 hours).
        min_periods: Minimum valid readings to form an estimate (default 3).
        mad_factor: Scale factor for normal consistency (1.4826).
        epsilon: Positive floor to prevent zero spread on constant series (1e-6).
        threshold_multiplier: Multiplier for robust spread (3.0).
        
    Returns:
        (meds, spreads, lower_bounds, upper_bounds) as length N float arrays.
    """
    n = len(values)
    meds = np.full(n, np.nan, dtype=float)
    spreads = np.full(n, np.nan, dtype=float)

    if n >= window_steps:
        win_view = sliding_window_view(values, window_steps)  # shape (n - window_steps + 1, window_steps)
        m = np.full(len(win_view), np.nan, dtype=float)
        mad = np.full(len(win_view), np.nan, dtype=float)

        nan_rows = np.isnan(win_view).any(axis=1)
        if not np.any(nan_rows):
            # Fast vectorized calculation when all readings in windows are finite
            m = np.median(win_view, axis=1)
            d = np.abs(win_view - m[:, None])
            mad = np.median(d, axis=1)
        else:
            clean_idx = np.where(~nan_rows)[0]
            if len(clean_idx) > 0:
                clean_win = win_view[clean_idx]
                clean_m = np.median(clean_win, axis=1)
                clean_d = np.abs(clean_win - clean_m[:, None])
                clean_mad = np.median(clean_d, axis=1)
                m[clean_idx] = clean_m
                mad[clean_idx] = clean_mad

            nan_idx = np.where(nan_rows)[0]
            if len(nan_idx) > 0:
                nan_win = win_view[nan_idx]
                valid_counts = np.sum(~np.isnan(nan_win), axis=1)
                ok_mask = valid_counts >= min_periods
                if np.any(ok_mask):
                    sub_win = nan_win[ok_mask]
                    sub_m = np.nanmedian(sub_win, axis=1)
                    sub_d = np.abs(sub_win - sub_m[:, None])
                    sub_mad = np.nanmedian(sub_d, axis=1)
                    actual_ok_idx = nan_idx[ok_mask]
                    m[actual_ok_idx] = sub_m
                    mad[actual_ok_idx] = sub_mad

        sp = np.where(np.isnan(mad), np.nan, np.maximum(mad_factor * mad, epsilon))
        meds[window_steps - 1 :] = m
        spreads[window_steps - 1 :] = sp

    # Early warm-up steps (t < window_steps - 1)
    for t in range(min(n, window_steps - 1)):
        w = values[: t + 1]
        valid = w[~np.isnan(w)]
        if len(valid) >= min_periods:
            m_t = float(np.median(valid))
            mad_t = float(np.median(np.abs(valid - m_t)))
            meds[t] = m_t
            spreads[t] = max(mad_factor * mad_t, epsilon)

    low = meds - threshold_multiplier * spreads
    high = meds + threshold_multiplier * spreads
    return meds, spreads, low, high


def detect_episodes_from_bounds(
    values: np.ndarray,
    lower_bounds: Union[float, np.ndarray],
    upper_bounds: Union[float, np.ndarray],
    start_step: int = 480,
    persistence_start: int = 1,
    consecutive_inside_to_end: int = 10,
    merge_gap_max: int = 5,
) -> List[Tuple[int, int]]:
    """Detect alarm episodes using state machine with clearing hysteresis and gap merging.
    
    Args:
        values: 1D array of readings, length T.
        lower_bounds: Scalar or length T array of lower thresholds.
        upper_bounds: Scalar or length T array of upper thresholds.
        start_step: First step to evaluate (default 480 for scored window).
        persistence_start: Readings outside required to trigger alarm (default 1).
        consecutive_inside_to_end: Consecutive inside readings to clear alarm (default 10).
        merge_gap_max: Max gap between episodes to merge (default 5).
        
    Returns:
        List of merged (start_step, end_step) integer tuples.
    """
    n = len(values)
    if n == 0 or start_step >= n:
        return []

    is_low = (values < lower_bounds) if isinstance(lower_bounds, np.ndarray) else (values < lower_bounds)
    is_high = (values > upper_bounds) if isinstance(upper_bounds, np.ndarray) else (values > upper_bounds)
    is_outside = (~np.isnan(values)) & (is_low | is_high)
    is_inside = (~np.isnan(values)) & (~is_low) & (~is_high)

    raw_episodes: List[Tuple[int, int]] = []
    active = False
    ep_start = 0
    last_outside = 0
    inside_count = 0
    outside_count = 0

    for t in range(start_step, n):
        if not active:
            if is_outside[t]:
                outside_count += 1
                if outside_count >= persistence_start:
                    active = True
                    ep_start = t - persistence_start + 1
                    last_outside = t
                    inside_count = 0
            else:
                outside_count = 0
        else:
            if is_outside[t]:
                last_outside = t
                inside_count = 0
            elif is_inside[t]:
                inside_count += 1
                if inside_count >= consecutive_inside_to_end:
                    active = False
                    raw_episodes.append((ep_start, last_outside))
                    inside_count = 0
                    outside_count = 0
            else:
                # Missing reading (NaN) interrupts consecutive streak inside
                inside_count = 0

    if active:
        raw_episodes.append((ep_start, last_outside))

    if not raw_episodes:
        return []

    # Merge episodes separated by <= merge_gap_max steps
    merged: List[Tuple[int, int]] = [raw_episodes[0]]
    for cur_s, cur_e in raw_episodes[1:]:
        prev_s, prev_e = merged[-1]
        gap = cur_s - prev_e - 1
        if gap <= merge_gap_max:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))

    return merged


class RawAlarmDetector:
    """Conventional threshold baseline detector specified in Appendix A.9.
    
    Reading outside [8-hour median +/- 3 x robust spread of reading levels over those 8 hours].
    Uses a causal eight-hour reference window (up to 480 past and current readings).
    Starts at 1 reading outside; ends after 10 consecutive inside; merge gap 5.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, mode: str = "rolling"):
        """Initialize detector with parameters from config/alarm.yaml.
        
        Args:
            config: Optional alarm configuration dictionary.
            mode: 'rolling' (causal 480-step window per Appendix A.9) or 'reference' (legacy static warmup).
        """
        if config is None:
            cfg_path = get_config_dir() / "alarm.yaml"
            config = load_yaml(cfg_path)
        self.config = config
        raw_cfg = config.get("raw_alarm", {})
        self.baseline_window_steps = int(raw_cfg.get("baseline_window_minutes", 480))
        self.robust_spread_mad_factor = float(raw_cfg.get("robust_spread_mad_factor", 1.4826))
        self.threshold_multiplier = float(raw_cfg.get("threshold_multiplier", 3.0))
        self.persistence_start = int(raw_cfg.get("persistence_start_readings", 1))
        self.consecutive_inside_to_end = int(raw_cfg.get("consecutive_inside_to_end", 10))
        self.episode_merge_gap_max = int(raw_cfg.get("episode_merge_gap_max", 5))
        self.mode = mode

    def _validate_observables(self, df: pd.DataFrame) -> None:
        """Enforce strict schema and truth isolation."""
        cols = set(df.columns)
        leaked = cols.intersection(PROHIBITED_TRUTH_COLUMNS)
        if leaked:
            raise ValueError(
                f"Truth isolation violation in RawAlarmDetector: prohibited ground-truth "
                f"columns detected in observable inputs: {sorted(leaked)}"
            )
        required = {"scenario_id", "group_id", "sensor_id", "timestamp", "value"}
        missing = required.difference(cols)
        if missing:
            raise ValueError(f"Missing required observable columns: {sorted(missing)}")

    def compute_sensor_bands(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute baseline median, robust spread, lower bound, and upper bound.
        
        In 'rolling' mode (Appendix A.9), computes causal 8-hour rolling estimates.
        In 'reference' mode (legacy), computes scalar bands from the 480-step warmup.
        """
        if self.mode == "rolling":
            return compute_causal_rolling_bands(
                values,
                window_steps=self.baseline_window_steps,
                min_periods=3,
                mad_factor=self.robust_spread_mad_factor,
                epsilon=1e-6,
                threshold_multiplier=self.threshold_multiplier,
            )
        elif self.mode == "reference":
            warmup = values[: self.baseline_window_steps]
            med, spread = compute_robust_spread(warmup, mad_factor=self.robust_spread_mad_factor)
            low = med - self.threshold_multiplier * spread
            high = med + self.threshold_multiplier * spread
            n = len(values)
            return np.full(n, med), np.full(n, spread), np.full(n, low), np.full(n, high)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'rolling' or 'reference'.")

    def detect_episodes(self, observables_df: pd.DataFrame, start_step: int = 480) -> pd.DataFrame:
        """Detect raw alarm episodes across all scenarios and sensors in observables_df.
        
        Args:
            observables_df: Layer 1 observable DataFrame (6 columns).
            start_step: Scored window start step (default 480).
            
        Returns:
            DataFrame with columns matching RawAlarmEpisode.
        """
        self._validate_observables(observables_df)

        episodes_list: List[Dict[str, Any]] = []

        # Process per scenario and sensor
        for (sid, gid, sens_id), grp in observables_df.groupby(
            ["scenario_id", "group_id", "sensor_id"], sort=False
        ):
            grp_sorted = grp.sort_values("timestamp")
            vals = grp_sorted["value"].to_numpy(dtype=float)
            timestamps = grp_sorted["timestamp"].to_numpy()

            meds, spreads, low, high = self.compute_sensor_bands(vals)

            raw_eps = detect_episodes_from_bounds(
                vals,
                lower_bounds=low,
                upper_bounds=high,
                start_step=start_step,
                persistence_start=self.persistence_start,
                consecutive_inside_to_end=self.consecutive_inside_to_end,
                merge_gap_max=self.episode_merge_gap_max,
            )

            for s_idx, e_idx in raw_eps:
                t_start = int(timestamps[s_idx])
                t_end = int(timestamps[e_idx])
                sub_vals = vals[s_idx : e_idx + 1]
                valid_sub = sub_vals[~np.isnan(sub_vals)]
                min_v = float(np.min(valid_sub)) if len(valid_sub) > 0 else np.nan
                max_v = float(np.max(valid_sub)) if len(valid_sub) > 0 else np.nan

                episodes_list.append({
                    "scenario_id": sid,
                    "group_id": gid,
                    "sensor_id": sens_id,
                    "start_step": t_start,
                    "end_step": t_end,
                    "duration": t_end - t_start + 1,
                    "median_base": float(meds[s_idx]),
                    "spread_base": float(spreads[s_idx]),
                    "lower_bound": float(low[s_idx]),
                    "upper_bound": float(high[s_idx]),
                    "min_value": min_v,
                    "max_value": max_v,
                })

        columns = [
            "scenario_id", "group_id", "sensor_id", "start_step", "end_step",
            "duration", "median_base", "spread_base", "lower_bound", "upper_bound",
            "min_value", "max_value"
        ]
        if not episodes_list:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(episodes_list)[columns]

    def predict(self, observables_df: pd.DataFrame, start_step: int = 480) -> pd.DataFrame:
        """Produce pointwise binary raw_alarm flags for all rows in observables_df.
        
        Args:
            observables_df: Layer 1 observable DataFrame.
            start_step: Scored window start step.
            
        Returns:
            DataFrame with original metadata and binary 'raw_alarm' flag.
        """
        self._validate_observables(observables_df)
        episodes_df = self.detect_episodes(observables_df, start_step=start_step)

        # Build interval lookup
        intervals = set()
        for row in episodes_df.itertuples(index=False):
            sid = row.scenario_id
            sens = row.sensor_id
            for t in range(int(row.start_step), int(row.end_step) + 1):
                intervals.add((sid, sens, t))

        flags = [
            int((row.scenario_id, row.sensor_id, int(row.timestamp)) in intervals)
            for row in observables_df.itertuples(index=False)
        ]

        out_df = observables_df[["scenario_id", "group_id", "sensor_id", "timestamp"]].copy()
        out_df["raw_alarm"] = flags
        return out_df


def assign_alarm_causes(
    episodes_df: pd.DataFrame,
    ground_truth_sensors_df: pd.DataFrame,
    ground_truth_events_df: pd.DataFrame,
    grace_steps: int = 5,
    min_step: int = 480,
    max_step: int = 1919,
) -> pd.DataFrame:
    """Assign ground-truth cause labels (EVENT, FAULT, BOTH, NEITHER) to raw alarm episodes.
    
    Appendix A.9:
    - Cause label for each raw alarm: EVENT, FAULT, BOTH, NEITHER, using 3 x sigma_k rule.
    - Evaluated over episode window extended by grace_steps (5 steps).
    
    Args:
        episodes_df: Output of RawAlarmDetector.detect_episodes().
        ground_truth_sensors_df: Layer 3 ground_truth_sensors table.
        ground_truth_events_df: Layer 3 ground_truth_events table.
        grace_steps: Grace window for overlap matching (default 5).
        min_step: Scored window start step (480).
        max_step: Scored window end step (1919).
        
    Returns:
        Copy of episodes_df with added columns 'has_event', 'has_fault', and 'cause_label'.
    """
    if episodes_df.empty:
        out = episodes_df.copy()
        out["has_event"] = pd.Series(dtype=bool)
        out["has_fault"] = pd.Series(dtype=bool)
        out["cause_label"] = pd.Series(dtype=str)
        return out

    out_df = episodes_df.copy()
    causes: List[str] = []
    has_event_list: List[bool] = []
    has_fault_list: List[bool] = []

    # Fast indexed lookups by (scenario_id, sensor_id) and (scenario_id, group_id)
    gts_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (sid, sens_id), sub in ground_truth_sensors_df.groupby(["scenario_id", "sensor_id"], sort=False):
        gts_map[(sid, sens_id)] = sub.set_index("timestamp")

    gte_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (sid, grp_id), sub in ground_truth_events_df.groupby(["scenario_id", "group_id"], sort=False):
        gte_map[(sid, grp_id)] = sub.set_index("timestamp")

    for row in out_df.itertuples(index=False):
        sid = row.scenario_id
        gid = row.group_id
        sens_id = row.sensor_id
        t_start = max(min_step, int(row.start_step) - grace_steps)
        t_end = min(max_step, int(row.end_step) + grace_steps)

        # 1. Fault check (sensor-level): sensor_untrusted == 1 or active fault severity
        has_f = False
        sens_gt = gts_map.get((sid, sens_id))
        if sens_gt is not None:
            slice_gts = sens_gt.loc[t_start:t_end]
            if not slice_gts.empty:
                has_f = bool((slice_gts["sensor_untrusted"] == 1).any())

        # 2. Event check (group-level): genuine_event == 1 (exceeds 3 x sigma_k)
        has_e = False
        grp_gt = gte_map.get((sid, gid))
        if grp_gt is not None:
            slice_gte = grp_gt.loc[t_start:t_end]
            if not slice_gte.empty:
                has_e = bool((slice_gte["genuine_event"] == 1).any())

        has_event_list.append(has_e)
        has_fault_list.append(has_f)

        if has_e and has_f:
            causes.append("BOTH")
        elif has_e and not has_f:
            causes.append("EVENT")
        elif not has_e and has_f:
            causes.append("FAULT")
        else:
            causes.append("NEITHER")

    out_df["has_event"] = has_event_list
    out_df["has_fault"] = has_fault_list
    out_df["cause_label"] = causes
    return out_df
