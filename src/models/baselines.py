"""Simple baseline detectors for sensor alarm and fault detection (P3.2).

Appendix A.9, A.14, and Appendix C:
- FixedThresholdDetector: flags readings exceeding a fixed threshold around baseline level.
- RobustZScoreDetector: flags readings with robust z-score > z_threshold.
- Both form episodes with merge gap 5 and can be calibrated on training/validation data
  to meet the false-alarm budget (at most 0.5 episodes per healthy sensor per day).
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.models.raw_alarm import (
    PROHIBITED_TRUTH_COLUMNS,
    compute_robust_spread,
    detect_episodes_from_bounds,
)


def extract_runs_and_merge(binary_mask: np.ndarray, start_step: int = 480, merge_gap_max: int = 5) -> List[Tuple[int, int]]:
    """Extract contiguous runs of True in binary_mask[start_step:] and merge gaps <= merge_gap_max."""
    n = len(binary_mask)
    if n == 0 or start_step >= n:
        return []

    raw: List[Tuple[int, int]] = []
    in_run = False
    run_s = 0

    for t in range(start_step, n):
        if binary_mask[t]:
            if not in_run:
                in_run = True
                run_s = t
        else:
            if in_run:
                in_run = False
                raw.append((run_s, t - 1))

    if in_run:
        raw.append((run_s, n - 1))

    if not raw:
        return []

    merged: List[Tuple[int, int]] = [raw[0]]
    for cur_s, cur_e in raw[1:]:
        prev_s, prev_e = merged[-1]
        gap = cur_s - prev_e - 1
        if gap <= merge_gap_max:
            merged[-1] = (prev_s, max(prev_e, cur_e))
        else:
            merged.append((cur_s, cur_e))

    return merged


class FixedThresholdDetector:
    """Fixed-threshold detector baseline.
    
    Flags deviations exceeding multiplier * robust_spread from reference median:
    |y(t) - median_base| > multiplier * spread_base.
    Runs are merged with gap <= 5.
    """

    def __init__(self, multiplier: float = 12.2, baseline_window_steps: int = 480, merge_gap_max: int = 5):
        """Initialize fixed threshold detector.
        
        Args:
            multiplier: Threshold multiplier k applied to robust spread (default 12.2 calibrated for 0.5 FAR).
            baseline_window_steps: Reference baseline window length in steps (default 480).
            merge_gap_max: Max gap for episode merging (default 5).
        """
        self.multiplier = float(multiplier)
        self.baseline_window_steps = int(baseline_window_steps)
        self.merge_gap_max = int(merge_gap_max)

    def _validate_observables(self, df: pd.DataFrame) -> None:
        cols = set(df.columns)
        leaked = cols.intersection(PROHIBITED_TRUTH_COLUMNS)
        if leaked:
            raise ValueError(
                f"Truth isolation violation in FixedThresholdDetector: prohibited ground-truth "
                f"columns detected in observable inputs: {sorted(leaked)}"
            )

    def detect_episodes(self, observables_df: pd.DataFrame, start_step: int = 480) -> pd.DataFrame:
        """Detect alarm episodes across all sensors in observables_df."""
        self._validate_observables(observables_df)

        episodes_list: List[Dict[str, Any]] = []

        for (sid, gid, sens_id), grp in observables_df.groupby(
            ["scenario_id", "group_id", "sensor_id"], sort=False
        ):
            grp_sorted = grp.sort_values("timestamp")
            vals = grp_sorted["value"].to_numpy(dtype=float)
            timestamps = grp_sorted["timestamp"].to_numpy()

            warmup = vals[:self.baseline_window_steps]
            med, spread = compute_robust_spread(warmup)
            threshold = self.multiplier * spread

            mask = (~np.isnan(vals)) & (np.abs(vals - med) > threshold)
            merged_eps = extract_runs_and_merge(mask, start_step=start_step, merge_gap_max=self.merge_gap_max)

            for s_idx, e_idx in merged_eps:
                t_start = int(timestamps[s_idx])
                t_end = int(timestamps[e_idx])
                episodes_list.append({
                    "scenario_id": sid,
                    "group_id": gid,
                    "sensor_id": sens_id,
                    "start_step": t_start,
                    "end_step": t_end,
                    "duration": t_end - t_start + 1,
                    "threshold_multiplier": self.multiplier,
                    "median_base": med,
                    "spread_base": spread,
                })

        columns = [
            "scenario_id", "group_id", "sensor_id", "start_step", "end_step",
            "duration", "threshold_multiplier", "median_base", "spread_base"
        ]
        if not episodes_list:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(episodes_list)[columns]

    def predict(self, observables_df: pd.DataFrame, start_step: int = 480) -> pd.DataFrame:
        """Produce pointwise binary flags."""
        self._validate_observables(observables_df)
        episodes_df = self.detect_episodes(observables_df, start_step=start_step)

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
        out_df["alarm_flag"] = flags
        return out_df

    @classmethod
    def calibrate_multiplier(
        cls,
        observables_df: pd.DataFrame,
        ground_truth_sensors_df: pd.DataFrame,
        target_far: float = 0.5,
        candidate_multipliers: Optional[List[float]] = None,
        start_step: int = 480,
    ) -> Tuple[float, Dict[float, float]]:
        """Calibrate threshold multiplier on training/validation data to meet target FAR.
        
        Args:
            observables_df: Observable readings.
            ground_truth_sensors_df: Ground truth sensors for identifying healthy sensors.
            target_far: Target false alarm rate (episodes / healthy sensor-day, default 0.5).
            candidate_multipliers: List of multipliers to evaluate.
            start_step: Scored window start step.
            
        Returns:
            (selected_multiplier, {multiplier: far_value})
        """
        if candidate_multipliers is None:
            candidate_multipliers = [3.0, 5.0, 8.0, 10.0, 11.0, 12.0, 12.2, 12.5, 13.0, 15.0, 18.0, 20.0]

        # Identify healthy sensors in scored window
        faulty_sensors = set()
        for (sid, sens), grp in ground_truth_sensors_df.groupby(["scenario_id", "sensor_id"]):
            scored_slice = grp[grp["timestamp"] >= start_step]
            if (scored_slice["sensor_untrusted"] == 1).any():
                faulty_sensors.add((sid, sens))

        all_sensors = observables_df[["scenario_id", "sensor_id"]].drop_duplicates()
        healthy_sensors = {
            (r.scenario_id, r.sensor_id)
            for r in all_sensors.itertuples(index=False)
            if (r.scenario_id, r.sensor_id) not in faulty_sensors
        }
        total_healthy_days = max(len(healthy_sensors) * 1.0, 1e-6)

        # Precompute sensor baseline stats
        sensor_cache = {}
        for (sid, sens), grp in observables_df.groupby(["scenario_id", "sensor_id"]):
            vals = grp.sort_values("timestamp")["value"].to_numpy(dtype=float)
            warmup = vals[:start_step]
            med, spread = compute_robust_spread(warmup)
            sensor_cache[(sid, sens)] = {
                "vals": vals,
                "med": med,
                "spread": spread,
                "is_healthy": (sid, sens) in healthy_sensors,
            }

        far_results: Dict[float, float] = {}
        selected_mult = candidate_multipliers[-1]

        for k in sorted(candidate_multipliers):
            fa_count = 0
            for (sid, sens), sdata in sensor_cache.items():
                if not sdata["is_healthy"]:
                    continue
                vals = sdata["vals"]
                med = sdata["med"]
                spread = sdata["spread"]
                mask = (~np.isnan(vals)) & (np.abs(vals - med) > k * spread)
                eps = extract_runs_and_merge(mask, start_step=start_step, merge_gap_max=5)
                fa_count += len(eps)

            far = fa_count / total_healthy_days
            far_results[k] = far
            if far <= target_far and selected_mult == candidate_multipliers[-1]:
                selected_mult = k

        return selected_mult, far_results


class RobustZScoreDetector:
    """Robust z-score detector baseline.
    
    Computes causal robust z-score z(t) = |y(t) - median(t)| / (1.4826 x MAD(t)).
    Flags z(t) > z_threshold.
    Runs are merged with gap <= 5.
    """

    def __init__(
        self,
        z_threshold: float = 5.0,
        window_steps: int = 480,
        mode: str = "rolling",
        merge_gap_max: int = 5,
        min_periods: int = 30,
    ):
        """Initialize robust z-score detector.
        
        Args:
            z_threshold: Z-score threshold (default 5.0 calibrated for 0.5 FAR).
            window_steps: Window size for rolling median/MAD (default 480).
            mode: 'rolling' (causal rolling window) or 'reference' (initial window [0, 480)).
            merge_gap_max: Max gap for episode merging (default 5).
            min_periods: Minimum valid readings in rolling window.
        """
        self.z_threshold = float(z_threshold)
        self.window_steps = int(window_steps)
        self.mode = mode
        self.merge_gap_max = int(merge_gap_max)
        self.min_periods = int(min_periods)

    def _validate_observables(self, df: pd.DataFrame) -> None:
        cols = set(df.columns)
        leaked = cols.intersection(PROHIBITED_TRUTH_COLUMNS)
        if leaked:
            raise ValueError(
                f"Truth isolation violation in RobustZScoreDetector: prohibited ground-truth "
                f"columns detected in observable inputs: {sorted(leaked)}"
            )

    def compute_z_scores(self, values: np.ndarray) -> np.ndarray:
        """Compute robust z-scores for sensor readings series."""
        if len(values) == 0:
            return np.array([], dtype=float)

        if self.mode == "reference":
            warmup = values[:self.window_steps]
            med, spread = compute_robust_spread(warmup)
            z = np.abs(values - med) / spread
            return z

        # Causal rolling window
        s_vals = pd.Series(values)
        roll_med = s_vals.rolling(self.window_steps, min_periods=self.min_periods).median()
        roll_mad = (s_vals - roll_med).abs().rolling(self.window_steps, min_periods=self.min_periods).median()
        roll_spread = np.maximum(1.4826 * roll_mad, 1e-6)
        z = ((s_vals - roll_med).abs() / roll_spread).to_numpy()
        return z

    def detect_episodes(self, observables_df: pd.DataFrame, start_step: int = 480) -> pd.DataFrame:
        """Detect alarm episodes across all sensors in observables_df."""
        self._validate_observables(observables_df)

        episodes_list: List[Dict[str, Any]] = []

        for (sid, gid, sens_id), grp in observables_df.groupby(
            ["scenario_id", "group_id", "sensor_id"], sort=False
        ):
            grp_sorted = grp.sort_values("timestamp")
            vals = grp_sorted["value"].to_numpy(dtype=float)
            timestamps = grp_sorted["timestamp"].to_numpy()

            z = self.compute_z_scores(vals)
            mask = (~np.isnan(z)) & (z > self.z_threshold)
            merged_eps = extract_runs_and_merge(mask, start_step=start_step, merge_gap_max=self.merge_gap_max)

            for s_idx, e_idx in merged_eps:
                t_start = int(timestamps[s_idx])
                t_end = int(timestamps[e_idx])
                sub_z = z[s_idx : e_idx + 1]
                valid_z = sub_z[~np.isnan(sub_z)]
                peak_z = float(np.max(valid_z)) if len(valid_z) > 0 else self.z_threshold

                episodes_list.append({
                    "scenario_id": sid,
                    "group_id": gid,
                    "sensor_id": sens_id,
                    "start_step": t_start,
                    "end_step": t_end,
                    "duration": t_end - t_start + 1,
                    "z_threshold": self.z_threshold,
                    "peak_z_score": peak_z,
                })

        columns = [
            "scenario_id", "group_id", "sensor_id", "start_step", "end_step",
            "duration", "z_threshold", "peak_z_score"
        ]
        if not episodes_list:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(episodes_list)[columns]

    def predict(self, observables_df: pd.DataFrame, start_step: int = 480) -> pd.DataFrame:
        """Produce pointwise binary flags."""
        self._validate_observables(observables_df)
        episodes_df = self.detect_episodes(observables_df, start_step=start_step)

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
        out_df["alarm_flag"] = flags
        return out_df

    @classmethod
    def calibrate_z_threshold(
        cls,
        observables_df: pd.DataFrame,
        ground_truth_sensors_df: pd.DataFrame,
        target_far: float = 0.5,
        window_steps: int = 480,
        mode: str = "rolling",
        candidate_thresholds: Optional[List[float]] = None,
        start_step: int = 480,
    ) -> Tuple[float, Dict[float, float]]:
        """Calibrate z_threshold on training/validation data to meet target FAR."""
        if candidate_thresholds is None:
            candidate_thresholds = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]

        faulty_sensors = set()
        for (sid, sens), grp in ground_truth_sensors_df.groupby(["scenario_id", "sensor_id"]):
            scored_slice = grp[grp["timestamp"] >= start_step]
            if (scored_slice["sensor_untrusted"] == 1).any():
                faulty_sensors.add((sid, sens))

        all_sensors = observables_df[["scenario_id", "sensor_id"]].drop_duplicates()
        healthy_sensors = {
            (r.scenario_id, r.sensor_id)
            for r in all_sensors.itertuples(index=False)
            if (r.scenario_id, r.sensor_id) not in faulty_sensors
        }
        total_healthy_days = max(len(healthy_sensors) * 1.0, 1e-6)

        # Precompute z-scores per sensor
        detector_helper = cls(window_steps=window_steps, mode=mode)
        z_cache = {}
        for (sid, sens), grp in observables_df.groupby(["scenario_id", "sensor_id"]):
            vals = grp.sort_values("timestamp")["value"].to_numpy(dtype=float)
            z = detector_helper.compute_z_scores(vals)
            z_cache[(sid, sens)] = {
                "z": z,
                "is_healthy": (sid, sens) in healthy_sensors,
            }

        far_results: Dict[float, float] = {}
        selected_thresh = candidate_thresholds[-1]

        for z_th in sorted(candidate_thresholds):
            fa_count = 0
            for (sid, sens), sdata in z_cache.items():
                if not sdata["is_healthy"]:
                    continue
                z = sdata["z"]
                mask = (~np.isnan(z)) & (z > z_th)
                eps = extract_runs_and_merge(mask, start_step=start_step, merge_gap_max=5)
                fa_count += len(eps)

            far = fa_count / total_healthy_days
            far_results[z_th] = far
            if far <= target_far and selected_thresh == candidate_thresholds[-1]:
                selected_thresh = z_th

        return selected_thresh, far_results
