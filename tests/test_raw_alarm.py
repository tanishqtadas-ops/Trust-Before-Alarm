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


def test_raw_alarm_causality_and_rolling_reference():
    """Verify causal 8-hour reference window: future data at tau > t does NOT affect bounds at t."""
    from src.models.raw_alarm import compute_causal_rolling_bands

    np.random.seed(42)
    n = 1500
    orig = np.random.normal(20.0, 1.5, n)
    meds_orig, spreads_orig, low_orig, high_orig = compute_causal_rolling_bands(orig, window_steps=480)

    # Mutate all future data after step 1000
    mutated = orig.copy()
    mutated[1001:] += np.random.normal(50.0, 10.0, n - 1001)
    meds_mut, spreads_mut, low_mut, high_mut = compute_causal_rolling_bands(mutated, window_steps=480)

    # Verify past steps t <= 1000 are bitwise identical (using equal_nan=True for initial warm-up NaNs)
    assert np.array_equal(meds_orig[:1001], meds_mut[:1001], equal_nan=True), "Reference median violates causality!"
    assert np.array_equal(spreads_orig[:1001], spreads_mut[:1001], equal_nan=True), "Robust spread violates causality!"
    assert np.array_equal(low_orig[:1001], low_mut[:1001], equal_nan=True), "Lower bound violates causality!"
    assert np.array_equal(high_orig[:1001], high_mut[:1001], equal_nan=True), "Upper bound violates causality!"


def test_raw_alarm_persistence_and_clearing_streak():
    """Verify persistence of 1 reading, clearing on 10th consecutive inside, and streak interruptions."""
    # Build series:
    # t=500: outside
    # t=501..509 (9 inside readings): alarm must remain active
    # t=510: 10th consecutive inside reading: alarm clears, end_step=500
    series = np.zeros(600)
    series[500] = 10.0

    eps = detect_episodes_from_bounds(
        series,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        start_step=480,
        persistence_start=1,
        consecutive_inside_to_end=10,
        merge_gap_max=0,
    )
    assert len(eps) == 1
    assert eps[0] == (500, 500)

    # Interruption by outside reading at 8th step:
    # t=500: outside
    # t=501..507: 7 inside readings
    # t=508: outside again!
    # t=509..518: 10 consecutive inside readings
    series2 = np.zeros(600)
    series2[500] = 10.0
    series2[508] = 10.0
    eps2 = detect_episodes_from_bounds(
        series2,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        start_step=480,
        persistence_start=1,
        consecutive_inside_to_end=10,
        merge_gap_max=0,
    )
    assert len(eps2) == 1
    # Episode spans from first outside (500) to last outside (508)
    assert eps2[0] == (500, 508)

    # Interruption by NaN:
    # t=500: outside
    # t=501..505: 5 inside
    # t=506: NaN (missing) -> breaks inside streak!
    # t=507..515: 9 inside -> not enough to clear!
    # t=516: 10th inside after NaN -> clears!
    series3 = np.zeros(600)
    series3[500] = 10.0
    series3[506] = np.nan
    eps3 = detect_episodes_from_bounds(
        series3,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        start_step=480,
        persistence_start=1,
        consecutive_inside_to_end=10,
        merge_gap_max=0,
    )
    assert len(eps3) == 1
    assert eps3[0] == (500, 500)


def test_raw_alarm_merging_gap_boundary_5_vs_6():
    """Verify episodes separated by gap <= 5 merge, while gap == 6 remain separate."""
    # When consecutive_inside_to_end=1:
    # Episode 1 is outside at t=500 and ends at t=500 when t=501 is inside.
    # Case A: Next outside reading at t=506 -> gap = 506 - 500 - 1 = 5 <= 5 -> MUST MERGE
    series_gap5 = np.zeros(600)
    series_gap5[500] = 10.0
    series_gap5[506] = 10.0

    eps_gap5 = detect_episodes_from_bounds(
        series_gap5,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        start_step=480,
        persistence_start=1,
        consecutive_inside_to_end=1,
        merge_gap_max=5,
    )
    assert len(eps_gap5) == 1, f"Expected 1 merged episode for gap=5, got {eps_gap5}"
    assert eps_gap5[0] == (500, 506)

    # Case B: Next outside reading at t=507 -> gap = 507 - 500 - 1 = 6 > 5 -> MUST NOT MERGE
    series_gap6 = np.zeros(600)
    series_gap6[500] = 10.0
    series_gap6[507] = 10.0

    eps_gap6 = detect_episodes_from_bounds(
        series_gap6,
        lower_bounds=-1.0,
        upper_bounds=1.0,
        start_step=480,
        persistence_start=1,
        consecutive_inside_to_end=1,
        merge_gap_max=5,
    )
    assert len(eps_gap6) == 2, f"Expected 2 separate episodes for gap=6, got {eps_gap6}"
    assert eps_gap6[0] == (500, 500)
    assert eps_gap6[1] == (507, 507)


def test_raw_alarm_unit_invariance_affine():
    """Verify raw alarm detector is exactly unit-invariant under affine transformations."""
    np.random.seed(123)
    n = 800
    # Clean baseline readings around 20 C with diurnal wave and noise
    t_axis = np.arange(n)
    readings_c = 20.0 + 3.0 * np.sin(2 * np.pi * t_axis / 480) + np.random.normal(0, 0.3, n)
    # Inject 2 spikes well outside 3*robust_spread (> 10 C)
    readings_c[520] += 15.0
    readings_c[650] -= 15.0

    df_c = pd.DataFrame({
        "scenario_id": ["scen_c"] * n,
        "group_id": ["grp_1"] * n,
        "sensor_id": ["s_1"] * n,
        "timestamp": t_axis,
        "value": readings_c,
    })

    # Convert to Fahrenheit: F = 1.8 * C + 32.0
    readings_f = readings_c * 1.8 + 32.0
    df_f = pd.DataFrame({
        "scenario_id": ["scen_f"] * n,
        "group_id": ["grp_1"] * n,
        "sensor_id": ["s_1"] * n,
        "timestamp": t_axis,
        "value": readings_f,
    })

    # Also test arbitrary random affine scaling: Y = 4.2 * C - 15.7
    readings_aff = readings_c * 4.2 - 15.7
    df_aff = pd.DataFrame({
        "scenario_id": ["scen_aff"] * n,
        "group_id": ["grp_1"] * n,
        "sensor_id": ["s_1"] * n,
        "timestamp": t_axis,
        "value": readings_aff,
    })

    detector = RawAlarmDetector()
    eps_c = detector.detect_episodes(df_c)
    eps_f = detector.detect_episodes(df_f)
    eps_aff = detector.detect_episodes(df_aff)

    assert len(eps_c) == len(eps_f) == len(eps_aff) == 2

    for ec, ef, eaff in zip(eps_c.itertuples(), eps_f.itertuples(), eps_aff.itertuples()):
        # Exactly identical start and end steps
        assert ec.start_step == ef.start_step == eaff.start_step
        assert ec.end_step == ef.end_step == eaff.end_step
        # Baseline median scales affinely
        assert np.isclose(ef.median_base, ec.median_base * 1.8 + 32.0, rtol=1e-4)
        assert np.isclose(eaff.median_base, ec.median_base * 4.2 - 15.7, rtol=1e-4)
        # Robust spread scales linearly (invariant to offset)
        assert np.isclose(ef.spread_base, ec.spread_base * 1.8, rtol=1e-4)
        assert np.isclose(eaff.spread_base, ec.spread_base * 4.2, rtol=1e-4)


def test_raw_alarm_degenerate_and_missing_data():
    """Verify graceful handling of constant flatline, all-NaNs, and intermittent dropouts."""
    detector = RawAlarmDetector()

    # 1. Constant flatline: MAD is 0, floored at epsilon=1e-6, no false alarms
    df_const = pd.DataFrame({
        "scenario_id": ["scen_const"] * 600,
        "group_id": ["grp_1"] * 600,
        "sensor_id": ["s_1"] * 600,
        "timestamp": list(range(600)),
        "value": [22.0] * 600,
    })
    eps_const = detector.detect_episodes(df_const)
    assert len(eps_const) == 0

    # 2. All NaNs: no alarms triggered
    df_nans = pd.DataFrame({
        "scenario_id": ["scen_nans"] * 600,
        "group_id": ["grp_1"] * 600,
        "sensor_id": ["s_1"] * 600,
        "timestamp": list(range(600)),
        "value": [np.nan] * 600,
    })
    eps_nans = detector.detect_episodes(df_nans)
    assert len(eps_nans) == 0

    # 3. Intermittent NaN dropouts on normal series do not trigger false alarms
    np.random.seed(99)
    vals_clean = np.random.normal(20.0, 0.2, 600)
    # Inject missing bursts
    vals_clean[500:503] = np.nan
    vals_clean[550:555] = np.nan
    df_drop = pd.DataFrame({
        "scenario_id": ["scen_drop"] * 600,
        "group_id": ["grp_1"] * 600,
        "sensor_id": ["s_1"] * 600,
        "timestamp": list(range(600)),
        "value": vals_clean,
    })
    eps_drop = detector.detect_episodes(df_drop)
    assert len(eps_drop) == 0

