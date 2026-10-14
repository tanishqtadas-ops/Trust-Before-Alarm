"""Feature Engine orchestrating causal feature calculation from raw observable data.

Accepts strictly the 6-column observable DataFrame:
  [scenario_id, group_id, sensor_id, variable_type, timestamp, value]

Produces:
  1. Sensor-level feature table (Heads A & B):
     4 metadata columns + 49 model features = 53 columns.
  2. Group-level feature table (Head C):
     3 metadata columns + 11 model features = 14 columns.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import get_config_dir, load_yaml
from src.features.sigma_hat import compute_sigma_hat
from src.features.sensor_features import compute_own_signal_features, compute_cross_sensor_features
from src.features.group_features import compute_group_features

# Strict 6-column observable schema
OBSERVABLE_SCHEMA = [
    "scenario_id",
    "group_id",
    "sensor_id",
    "variable_type",
    "timestamp",
    "value",
]

SENSOR_METADATA_COLS = [
    "scenario_id",
    "group_id",
    "sensor_id",
    "timestamp",
]

GROUP_METADATA_COLS = [
    "scenario_id",
    "group_id",
    "timestamp",
]

# Appendix B: 49 model features for sensor level
SENSOR_FEATURE_NAMES = [
    # Own-signal temporal (10)
    "diff_1",
    "absdiff_max_s",
    "absdiff_max_m",
    "slope_s",
    "slope_m",
    "slope_l",
    "slope_change",
    "dev_baseline",
    "dev_duration",
    "n_spikes_m",
    # Statistical (6)
    "noise_ratio_s",
    "noise_ratio_m",
    "noise_ratio_l",
    "noise_short_long",
    "kurtosis_m",
    "range_m",
    # Data quality (10)
    "cur_missing",
    "frac_missing_m",
    "frac_missing_l",
    "consec_missing",
    "longest_gap_m",
    "steps_since_valid",
    "repeat_run_len",
    "longest_repeat_m",
    "frac_zero_diff_m",
    "unique_ratio_m",
    # Cross-sensor (14)
    "gap",
    "gap_mean_m",
    "gap_slope_m",
    "gap_change_m",
    "gap_absmax_m",
    "gap_std_m",
    "gap_autocorr_m",
    "others_spread",
    "others_agreement",
    "diff_corr_m",
    "net_change_gap_m",
    "gain_slope_l",
    "gain_r2_l",
    "n_valid_others",
    # Long-term degradation (3)
    "noise_growth",
    "missing_growth",
    "gap_slope_xl",
    # Group/event joined onto sensor rows (6)
    "event_breadth_s",
    "event_breadth_m",
    "group_change_s",
    "group_change_m",
    "group_dispersion",
    "same_direction_frac",
]

# Appendix B: 11 model features for group level (Head C)
GROUP_FEATURE_NAMES = [
    "event_breadth_s",
    "event_breadth_m",
    "group_change_s",
    "group_change_m",
    "group_dispersion",
    "same_direction_frac",
    "group_slope_s",
    "group_dev_baseline",
    "n_valid_sensors",
    "group_peak_m",
    "group_noise_ratio",
]


class FeatureEngine:
    """End-to-end causal feature engine for Sensor Trust Intelligence."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FeatureEngine with configuration."""
        if config is None:
            cfg_path = get_config_dir() / "features.yaml"
            config = load_yaml(cfg_path)
        self.config = config

    def validate_observable_schema(self, df: pd.DataFrame) -> None:
        """Validate that input DataFrame strictly and exclusively adheres to the 6-column observable schema."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        actual_cols = list(df.columns)
        if actual_cols != OBSERVABLE_SCHEMA:
            # Check for ground-truth leakage or missing columns
            forbidden = [
                "true_value", "sensor_untrusted", "genuine_event", "fault_severity",
                "true_health", "f_spike", "f_offset_drift", "f_gain_error", "f_frozen",
                "f_high_noise", "f_dropout", "f_slow_degradation",
            ]
            leaked = [c for c in actual_cols if c in forbidden or c.startswith("f_") or "true" in c]
            if leaked:
                raise ValueError(f"Ground-truth leakage detected in feature engine input: {leaked}")
            raise ValueError(
                f"Invalid observable schema. Expected exactly {OBSERVABLE_SCHEMA}, got {actual_cols}"
            )

    def compute_scenario_features(
        self, scenario_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute sensor-level and group-level feature tables for a single scenario."""
        self.validate_observable_schema(scenario_df)

        scen_id = scenario_df["scenario_id"].iloc[0]
        groups = scenario_df["group_id"].unique()

        all_sensor_dfs: List[pd.DataFrame] = []
        all_group_dfs: List[pd.DataFrame] = []

        for gid in groups:
            gdf = scenario_df[scenario_df["group_id"] == gid]
            sensor_ids = sorted(gdf["sensor_id"].unique())

            # Pivot to (timestamp x sensor_id)
            pivot_df = gdf.pivot(index="timestamp", columns="sensor_id", values="value")
            pivot_df = pivot_df.sort_index()
            timestamps = pivot_df.index.to_numpy(dtype=int)
            n_steps = len(pivot_df)

            # Step 1: Compute sigma_hat for each sensor in the group
            sigma_hats: Dict[str, np.ndarray] = {}
            for sid in sensor_ids:
                vals = pivot_df[sid].to_numpy(dtype=float)
                sigma_hats[sid] = compute_sigma_hat(vals, config=self.config)

            # Step 2: Compute group-level features & Head C table for this group
            joined_group_df, head_c_df = compute_group_features(
                group_pivot_df=pivot_df,
                sensor_sigma_hats=sigma_hats,
                config=self.config,
            )

            # Build group output table
            group_out = pd.DataFrame({
                "scenario_id": scen_id,
                "group_id": gid,
                "timestamp": timestamps,
            })
            for col in GROUP_FEATURE_NAMES:
                group_out[col] = joined_group_df[col].to_numpy() if col in joined_group_df else head_c_df[col].to_numpy()
            all_group_dfs.append(group_out)

            # Step 3: Compute own-signal and cross-sensor features per sensor
            for sid in sensor_ids:
                s_vals = pivot_df[sid].to_numpy(dtype=float)
                sig = sigma_hats[sid]

                own_df = compute_own_signal_features(s_vals, sig, config=self.config)
                cross_df = compute_cross_sensor_features(
                    target_sensor_id=sid,
                    group_pivot_df=pivot_df,
                    sigma_hat=sig,
                    config=self.config,
                )

                # Assemble sensor record
                s_record = pd.DataFrame({
                    "scenario_id": scen_id,
                    "group_id": gid,
                    "sensor_id": sid,
                    "timestamp": timestamps,
                })

                # Add own-signal features
                for col in own_df.columns:
                    s_record[col] = own_df[col].to_numpy()

                # Add cross-sensor and degradation features
                for col in cross_df.columns:
                    s_record[col] = cross_df[col].to_numpy()

                # Add joined group/event features
                for col in [
                    "event_breadth_s", "event_breadth_m", "group_change_s",
                    "group_change_m", "group_dispersion", "same_direction_frac",
                ]:
                    s_record[col] = joined_group_df[col].to_numpy()

                # Ensure exact column ordering: SENSOR_METADATA_COLS + SENSOR_FEATURE_NAMES
                s_record = s_record[SENSOR_METADATA_COLS + SENSOR_FEATURE_NAMES]
                all_sensor_dfs.append(s_record)

        sensor_features_df = pd.concat(all_sensor_dfs, ignore_index=True) if all_sensor_dfs else pd.DataFrame()
        group_features_df = pd.concat(all_group_dfs, ignore_index=True) if all_group_dfs else pd.DataFrame()

        # Enforce column ordering
        if not group_features_df.empty:
            group_features_df = group_features_df[GROUP_METADATA_COLS + GROUP_FEATURE_NAMES]

        return sensor_features_df, group_features_df

    def compute_features(
        self, observables_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute sensor-level and group-level feature tables across one or more scenarios."""
        self.validate_observable_schema(observables_df)

        scenarios = observables_df["scenario_id"].unique()
        if len(scenarios) == 1:
            return self.compute_scenario_features(observables_df)

        sensor_list = []
        group_list = []
        for scen_id in scenarios:
            sub_df = observables_df[observables_df["scenario_id"] == scen_id]
            s_df, g_df = self.compute_scenario_features(sub_df)
            sensor_list.append(s_df)
            group_list.append(g_df)

        combined_sensor = pd.concat(sensor_list, ignore_index=True) if sensor_list else pd.DataFrame()
        combined_group = pd.concat(group_list, ignore_index=True) if group_list else pd.DataFrame()
        return combined_sensor, combined_group
