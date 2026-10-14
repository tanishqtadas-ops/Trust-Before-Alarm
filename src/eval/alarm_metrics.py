"""Evaluation metrics for alarms, baselines, and false-alarm budgets.

Appendix A.14, Appendix C, and config/eval.yaml:
- False-alarm budget: at most 0.5 episodes per healthy sensor per day
- Episode-level fault recall: fraction of fault episodes detected (5-reading grace)
- Row-level fault recall: fraction of untrusted steps flagged
- False-alarm rate (FAR): false-alarm episodes / healthy sensor-days
- Wilson 95% confidence intervals
- Cause breakdown: EVENT, FAULT, BOTH, NEITHER
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.models.raw_alarm import assign_alarm_causes


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score interval for a binomial proportion.
    
    Args:
        successes: Number of successes.
        trials: Total number of trials.
        confidence: Confidence level (default 0.95).
        
    Returns:
        (lower_bound, upper_bound)
    """
    if trials <= 0:
        return 0.0, 0.0
    if successes == 0:
        z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
        upper = min(1.0, float(z**2 / (trials + z**2)))
        return 0.0, upper
    if successes == trials:
        z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
        lower = max(0.0, float(trials / (trials + z**2)))
        return lower, 1.0
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    margin = (z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))) / denom
    lower = max(0.0, float(center - margin))
    upper = min(1.0, float(center + margin))
    return lower, upper


def evaluate_alarm_performance(
    alarm_episodes_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    ground_truth_sensors_df: pd.DataFrame,
    ground_truth_events_df: pd.DataFrame,
    fault_episodes_df: pd.DataFrame,
    grace_steps: int = 5,
    scored_start_step: int = 480,
    scored_end_step: int = 1919,
) -> Dict[str, Any]:
    """Compute comprehensive alarm and baseline metrics.
    
    Args:
        alarm_episodes_df: Detected alarm episodes (from RawAlarmDetector or Baselines).
        observables_df: Observable readings.
        ground_truth_sensors_df: Ground truth sensors table.
        ground_truth_events_df: Ground truth events table.
        fault_episodes_df: Ground truth fault episodes table.
        grace_steps: Overlap grace window (default 5 steps).
        scored_start_step: Scored window start (480).
        scored_end_step: Scored window end (1919).
        
    Returns:
        Dictionary of metrics including FAR, recall, precision, and cause breakdown.
    """
    # 1. Scored ground-truth fault episodes
    fe_scored = fault_episodes_df[fault_episodes_df["end_step"] >= scored_start_step].copy()
    total_fault_episodes = len(fe_scored)

    # 2. Healthy sensors during scored window
    faulty_sensors: Set[Tuple[str, str]] = set()
    for (sid, sens), grp in ground_truth_sensors_df.groupby(["scenario_id", "sensor_id"], sort=False):
        sub = grp[grp["timestamp"] >= scored_start_step]
        if (sub["sensor_untrusted"] == 1).any():
            faulty_sensors.add((sid, sens))

    all_sensors = observables_df[["scenario_id", "sensor_id"]].drop_duplicates()
    healthy_sensors = {
        (r.scenario_id, r.sensor_id)
        for r in all_sensors.itertuples(index=False)
        if (r.scenario_id, r.sensor_id) not in faulty_sensors
    }
    total_healthy_sensor_days = float(len(healthy_sensors))  # 1440 steps = 1 day

    # 3. Process alarm episodes: check healthy false alarms and fault overlaps
    total_alarm_episodes = len(alarm_episodes_df)
    fa_episodes_count = 0
    episodes_matching_fault = 0

    # Index alarm episodes by (scenario_id, sensor_id)
    alarm_map: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for row in alarm_episodes_df.itertuples(index=False):
        key = (row.scenario_id, row.sensor_id)
        if key not in alarm_map:
            alarm_map[key] = []
        alarm_map[key].append((int(row.start_step), int(row.end_step)))

    # Count false alarms on healthy sensors
    for (sid, sens), eps in alarm_map.items():
        if (sid, sens) in healthy_sensors:
            fa_episodes_count += len(eps)

    # Count detected fault episodes
    detected_fe_count = 0
    for _, f_row in fe_scored.iterrows():
        sid = f_row["scenario_id"]
        sens = f_row["sensor_id"]
        fs = max(scored_start_step, int(f_row["start_step"]))
        fe = min(scored_end_step, int(f_row["end_step"]))

        eps = alarm_map.get((sid, sens), [])
        is_detected = any(max(fs - grace_steps, ep_s) <= min(fe + grace_steps, ep_e) for ep_s, ep_e in eps)
        if is_detected:
            detected_fe_count += 1

    # Precision: alarm episodes that overlap any true fault on that sensor
    fe_map: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for _, f_row in fe_scored.iterrows():
        key = (f_row["scenario_id"], f_row["sensor_id"])
        if key not in fe_map:
            fe_map[key] = []
        fe_map[key].append((int(f_row["start_step"]), int(f_row["end_step"])))

    for (sid, sens), eps in alarm_map.items():
        f_intervals = fe_map.get((sid, sens), [])
        for ep_s, ep_e in eps:
            overlaps = any(
                max(fs - grace_steps, ep_s) <= min(fe + grace_steps, ep_e)
                for fs, fe in f_intervals
            )
            if overlaps:
                episodes_matching_fault += 1

    # 4. Row-level recall and false positive rate
    alarm_intervals = set()
    for row in alarm_episodes_df.itertuples(index=False):
        sid = row.scenario_id
        sens = row.sensor_id
        for t in range(int(row.start_step), int(row.end_step) + 1):
            alarm_intervals.add((sid, sens, t))

    # Evaluate across ground_truth_sensors scored window
    gts_scored = ground_truth_sensors_df[ground_truth_sensors_df["timestamp"] >= scored_start_step]
    untrusted_rows = (gts_scored["sensor_untrusted"] == 1).to_numpy()
    total_untrusted_steps = int(np.sum(untrusted_rows))
    total_trusted_steps = len(untrusted_rows) - total_untrusted_steps

    flagged_untrusted = 0
    flagged_trusted = 0
    for row in gts_scored.itertuples(index=False):
        is_flagged = (row.scenario_id, row.sensor_id, int(row.timestamp)) in alarm_intervals
        if is_flagged:
            if row.sensor_untrusted == 1:
                flagged_untrusted += 1
            else:
                flagged_trusted += 1

    # 5. Rates and intervals
    far = fa_episodes_count / max(total_healthy_sensor_days, 1e-6)
    ep_recall = detected_fe_count / max(total_fault_episodes, 1)
    rec_ci = wilson_interval(detected_fe_count, total_fault_episodes)

    row_recall = flagged_untrusted / max(total_untrusted_steps, 1)
    precision = episodes_matching_fault / max(total_alarm_episodes, 1)

    # 6. Cause label distribution
    labeled_episodes = assign_alarm_causes(
        alarm_episodes_df,
        ground_truth_sensors_df,
        ground_truth_events_df,
        grace_steps=grace_steps,
        min_step=scored_start_step,
        max_step=scored_end_step,
    )
    cause_counts = {"EVENT": 0, "FAULT": 0, "BOTH": 0, "NEITHER": 0}
    if not labeled_episodes.empty:
        for c, count in labeled_episodes["cause_label"].value_counts().items():
            if c in cause_counts:
                cause_counts[c] = int(count)

    return {
        "total_alarm_episodes": total_alarm_episodes,
        "healthy_sensor_days": total_healthy_sensor_days,
        "false_alarm_episodes": fa_episodes_count,
        "false_alarm_rate": far,
        "total_fault_episodes": total_fault_episodes,
        "detected_fault_episodes": detected_fe_count,
        "episode_recall": ep_recall,
        "episode_recall_ci95": rec_ci,
        "total_untrusted_steps": total_untrusted_steps,
        "flagged_untrusted_steps": flagged_untrusted,
        "row_recall": row_recall,
        "precision": precision,
        "cause_counts": cause_counts,
        "labeled_episodes": labeled_episodes,
    }
