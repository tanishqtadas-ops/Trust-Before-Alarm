"""Unit tests for Phase P3.1: Raw Alarm Detector and Cause Labeling.

Verifies:
1. Robust spread calculation (1.4826 x MAD, epsilon floor, NaN handling).
2. Alarm state machine (starts at 1 outside, clears after 10 inside).
3. Episode merging rule (gap <= 5 merges, gap > 5 separate).
4. Strict observable schema enforcement and ground-truth isolation.
5. Pointwise predictions vs episode intervals alignment.
6. Cause labeling 4-way classification (EVENT, FAULT, BOTH, NEITHER).
7. Execution on real dataset scenarios.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.raw_alarm import (
    RawAlarmDetector,
    assign_alarm_causes,
    compute_robust_spread,
    detect_episodes_from_bounds,
)


def test_compute_robust_spread_standard_normal():
    """Verify robust spread on Gaussian data approximates standard deviation."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=20.0, scale=2.0, size=10000)
    med, spread = compute_robust_spread(data)
    assert np.isclose(med, 20.0, atol=0.1)
    assert np.isclose(spread, 2.0, atol=0.1)


def test_compute_robust_spread_constant_and_nans():
    """Verify zero-noise flatline gets floored at positive epsilon and NaNs are ignored."""
    constant = np.array([25.0] * 500)
    med, spread = compute_robust_spread(constant)
    assert med == 25.0
    assert spread == 1e-6

    with_nans = np.array([np.nan, 10.0, 12.0, np.nan, 14.0, np.nan])
    med_nan, spread_nan = compute_robust_spread(with_nans)
    assert med_nan == 12.0
    assert spread_nan > 0


def test_state_machine_start_and_clear_rules():
    """Verify alarm starts at 1 reading outside and ends after 10 consecutive readings inside."""
    # 20 readings total:
    # t=0..4: inside (0.0)
    # t=5: outside (10.0) -> triggers alarm start at t=5
    # t=6..10: inside (5 readings inside -> alarm still active)
    # t=11: outside (10.0 -> alarm continues)
    # t=12..21: inside (10 consecutive inside readings -> clears alarm)
    # t=22..25: inside
    series = np.zeros(30)
    series[5] = 10.0
    series[11] = 10.0

    low = -1.0
    high = 1.0

    episodes = detect_episodes_from_bounds(
        series,
        lower_bounds=low,
        upper_bounds=high,
        start_step=0,
        persistence_start=1,
        consecutive_inside_to_end=10,
        merge_gap_max=0,  # No merging
    )

    # Should produce exactly 1 episode spanning from step 5 to step 11
    # because between step 5 and 11 there were only 5 inside readings (< 10)
    assert len(episodes) == 1
    start_step, end_step = episodes[0]
    assert start_step == 5
    assert end_step == 11


def test_episode_merging_gap():
    """Verify episodes merge when gap <= 5 and remain separate when gap > 5."""
    series = np.zeros(60)
    # Episode 1: outside at t=10, cleared at t=21 (ends at t=10)
    series[10] = 10.0
    # Case A: next outside at t=15 (gap = 15 - 10 - 1 = 4 <= 5) -> merges into [10, 15]
    # cleared at t=26
    # Case B: next outside at t=35 (gap = 35 - 15 - 1 = 19 > 5) -> separate episode [35, 35]
    series[15] = 10.0
    series[35] = 10.0

    episodes = detect_episodes_from_bounds(
        series,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        start_step=0,
        persistence_start=1,
        consecutive_inside_to_end=10,
        merge_gap_max=5,
    )

    assert len(episodes) == 2
    # First merged episode
    assert episodes[0] == (10, 15)
    # Second separate episode
    assert episodes[1] == (35, 35)


def test_raw_alarm_truth_isolation():
    """Verify detector accepts clean observables and strictly rejects ground truth columns."""
    detector = RawAlarmDetector()

    # Clean 6-column observable
    df_clean = pd.DataFrame({
        "scenario_id": ["scen_0"] * 600,
        "group_id": ["grp_0"] * 600,
        "sensor_id": ["s_0"] * 600,
        "variable_type": ["temperature"] * 600,
        "timestamp": list(range(600)),
        "value": [20.0] * 600,
    })

    # Predict must succeed
    preds = detector.predict(df_clean, start_step=480)
    assert "raw_alarm" in preds.columns

    # Leakage attempt with ground truth column
    for bad_col in ["sensor_untrusted", "genuine_event", "true_value", "fault_severity"]:
        df_leaked = df_clean.copy()
        df_leaked[bad_col] = 0
        with pytest.raises(ValueError, match="Truth isolation violation"):
            detector.predict(df_leaked)


def test_assign_alarm_causes_synthetic_quadrants():
    """Verify cause labeling strictly produces EVENT, FAULT, BOTH, NEITHER."""
    # Create 4 synthetic episodes on same sensor
    episodes_df = pd.DataFrame([
        {"scenario_id": "scen_1", "group_id": "grp_1", "sensor_id": "s_1", "start_step": 500, "end_step": 520},
        {"scenario_id": "scen_1", "group_id": "grp_1", "sensor_id": "s_1", "start_step": 600, "end_step": 620},
        {"scenario_id": "scen_1", "group_id": "grp_1", "sensor_id": "s_1", "start_step": 700, "end_step": 720},
        {"scenario_id": "scen_1", "group_id": "grp_1", "sensor_id": "s_1", "start_step": 800, "end_step": 820},
    ])

    timestamps = list(range(480, 1000))
    # GT sensors: fault active only in [600, 620] and [700, 720]
    gt_sensors = pd.DataFrame({
        "scenario_id": "scen_1",
        "group_id": "grp_1",
        "sensor_id": "s_1",
        "timestamp": timestamps,
        "sensor_untrusted": [1 if (600 <= t <= 620 or 700 <= t <= 720) else 0 for t in timestamps],
    })

    # GT events: event active only in [500, 520] and [700, 720]
    gt_events = pd.DataFrame({
        "scenario_id": "scen_1",
        "group_id": "grp_1",
        "timestamp": timestamps,
        "genuine_event": [1 if (500 <= t <= 520 or 700 <= t <= 720) else 0 for t in timestamps],
    })

    labeled = assign_alarm_causes(
        episodes_df,
        gt_sensors,
        gt_events,
        grace_steps=0,
    )

    causes = labeled["cause_label"].tolist()
    assert causes == ["EVENT", "FAULT", "BOTH", "NEITHER"]


def test_raw_alarm_detector_on_dataset():
    """Verify RawAlarmDetector executes correctly on a real exported scenario."""
    obs = pd.read_parquet("data/raw/val/observables.parquet")
    scen_id = obs["scenario_id"].iloc[0]
    scen_obs = obs[obs["scenario_id"] == scen_id]

    detector = RawAlarmDetector()
    episodes = detector.detect_episodes(scen_obs)
    preds = detector.predict(scen_obs)

    assert isinstance(episodes, pd.DataFrame)
    assert isinstance(preds, pd.DataFrame)
    assert len(preds) == len(scen_obs)
    assert set(preds["raw_alarm"].unique()).issubset({0, 1})
    # Scored window start step: steps < 480 should never be alarms
    warmup_preds = preds[preds["timestamp"] < 480]
    assert (warmup_preds["raw_alarm"] == 0).all()
