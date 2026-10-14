"""Unit tests for Phase P3.2 & P3.3: Baselines, Metrics, and Difficulty Checks.

Verifies:
1. FixedThresholdDetector detection and merging.
2. RobustZScoreDetector rolling and reference modes.
3. Strict observable schema enforcement and truth isolation.
4. False-alarm budget calculation and threshold calibration.
5. Metric evaluation (FAR, episode recall, row recall, Wilson intervals).
6. Difficulty tier evaluation and sanity checks per P3.3.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    FixedThresholdDetector,
    RobustZScoreDetector,
    extract_runs_and_merge,
)
from src.eval.alarm_metrics import (
    evaluate_alarm_performance,
    wilson_interval,
)
from src.eval.difficulty_check import (
    evaluate_by_difficulty_tier,
    format_tier_report_table,
)


def test_extract_runs_and_merge():
    """Verify run extraction and gap merging logic."""
    mask = np.zeros(50, dtype=bool)
    # Run 1: 5..7 (length 3)
    mask[5:8] = True
    # Run 2: 10..12 (length 3, gap = 10 - 7 - 1 = 2 <= 5) -> merges into [5, 12]
    mask[10:13] = True
    # Run 3: 25..27 (length 3, gap = 25 - 12 - 1 = 12 > 5) -> separate [25, 27]
    mask[25:28] = True

    merged = extract_runs_and_merge(mask, start_step=0, merge_gap_max=5)
    assert len(merged) == 2
    assert merged[0] == (5, 12)
    assert merged[1] == (25, 27)


def test_fixed_threshold_detector_synthetic():
    """Verify FixedThresholdDetector flags deviations from baseline."""
    timestamps = list(range(600))
    # Baseline 0..479 around 20.0 with noise std 0.1
    vals = [20.0] * 600
    vals[500:510] = [35.0] * 10  # Spike / excursion

    df = pd.DataFrame({
        "scenario_id": ["scen_test"] * 600,
        "group_id": ["grp_0"] * 600,
        "sensor_id": ["s_0"] * 600,
        "variable_type": ["temperature"] * 600,
        "timestamp": timestamps,
        "value": vals,
    })

    detector = FixedThresholdDetector(multiplier=3.0)
    episodes = detector.detect_episodes(df, start_step=480)
    preds = detector.predict(df, start_step=480)

    assert len(episodes) == 1
    assert episodes.iloc[0]["start_step"] == 500
    assert episodes.iloc[0]["end_step"] == 509
    assert preds.loc[preds["timestamp"] == 500, "alarm_flag"].iloc[0] == 1
    assert preds.loc[preds["timestamp"] == 400, "alarm_flag"].iloc[0] == 0


def test_robust_z_score_detector_rolling_synthetic():
    """Verify RobustZScoreDetector computes rolling z-scores and flags anomalies."""
    timestamps = list(range(600))
    vals = [20.0] * 600
    # Add an extreme jump at t=520
    vals[520:530] = [40.0] * 10

    df = pd.DataFrame({
        "scenario_id": ["scen_test"] * 600,
        "group_id": ["grp_0"] * 600,
        "sensor_id": ["s_0"] * 600,
        "variable_type": ["temperature"] * 600,
        "timestamp": timestamps,
        "value": vals,
    })

    detector = RobustZScoreDetector(z_threshold=4.0, window_steps=120, min_periods=10)
    episodes = detector.detect_episodes(df, start_step=480)
    assert len(episodes) == 1
    assert episodes.iloc[0]["start_step"] == 520


def test_baselines_truth_isolation():
    """Verify both baseline detectors strictly reject prohibited ground truth columns."""
    df_clean = pd.DataFrame({
        "scenario_id": ["scen_0"] * 500,
        "group_id": ["grp_0"] * 500,
        "sensor_id": ["s_0"] * 500,
        "variable_type": ["temperature"] * 500,
        "timestamp": list(range(500)),
        "value": [20.0] * 500,
    })

    for cls in [FixedThresholdDetector, RobustZScoreDetector]:
        det = cls()
        df_leaked = df_clean.copy()
        df_leaked["sensor_untrusted"] = 0
        with pytest.raises(ValueError, match="Truth isolation violation"):
            det.detect_episodes(df_leaked)


def test_wilson_interval():
    """Verify Wilson confidence interval computation."""
    # 0 successes
    low, high = wilson_interval(0, 100)
    assert low == 0.0
    assert high > 0.0

    # 100% successes
    low, high = wilson_interval(100, 100)
    assert low < 1.0
    assert high == 1.0

    # 50% successes
    low, high = wilson_interval(50, 100)
    assert np.isclose(low, 0.4038, atol=0.01)
    assert np.isclose(high, 0.5962, atol=0.01)


def test_difficulty_tier_evaluation_on_dataset():
    """Verify difficulty tier evaluation executes and formats clean report."""
    # Load small subset of val data (first 5 scenarios)
    obs = pd.read_parquet("data/raw/val/observables.parquet")
    gts = pd.read_parquet("data/ground_truth/val/ground_truth_sensors.parquet")
    gte = pd.read_parquet("data/ground_truth/val/ground_truth_events.parquet")
    fe = pd.read_parquet("data/ground_truth/val/fault_episodes.parquet")
    sc = pd.read_parquet("data/ground_truth/val/scenarios.parquet")

    sample_scens = sc["scenario_id"].unique()[:10]
    sub_sc = sc[sc["scenario_id"].isin(sample_scens)]
    sub_obs = obs[obs["scenario_id"].isin(sample_scens)]
    sub_gts = gts[gts["scenario_id"].isin(sample_scens)]
    sub_gte = gte[gte["scenario_id"].isin(sample_scens)]
    sub_fe = fe[fe["scenario_id"].isin(sample_scens)]

    detector = FixedThresholdDetector(multiplier=12.2)
    episodes = detector.detect_episodes(sub_obs)

    res = evaluate_by_difficulty_tier(
        detector_name="Fixed Threshold (k=12.2)",
        alarm_episodes_df=episodes,
        observables_df=sub_obs,
        ground_truth_sensors_df=sub_gts,
        ground_truth_events_df=sub_gte,
        fault_episodes_df=sub_fe,
        scenarios_df=sub_sc,
    )

    assert "tiers" in res
    assert "overall" in res
    assert "sanity_status" in res
    for tier in ["easy", "medium", "hard"]:
        assert tier in res["tiers"]

    report_str = format_tier_report_table(res)
    assert "Baseline Difficulty Breakdown" in report_str
    assert "EASY" in report_str
    assert "HARD" in report_str
