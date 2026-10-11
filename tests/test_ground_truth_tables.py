"""Tests for P1.8 ground truth tables, schemas, and formulas.

Covers:
  - Exact 6 tables, column names, column order, dtypes, and row counts.
  - Primary-key uniqueness across all six export tables.
  - All 7 fault-severity formulas, clipping boundaries, and the true_health invariant.
  - Decision 45 event activity semantics (no events, detectable event, undetectable event,
    same-sign overlap, opposite-sign cancellation, sub-epsilon tails, mutual exclusivity).
  - Multi-group event episodes and deterministic affected_groups serialization.
  - Deterministic JSON serialization of shape_params, unit_scale, group_sizes, quantisation, clip_limits.
  - Layer 1 observable schema isolation (strictly 6 columns, zero ground-truth leakage).
  - Observable row-stride parameter.
  - Seeded determinism across all 6 tables.
  - Zero-fault scenarios and multi-fault scenarios.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from src.sim.environment import EnvironmentSeries, generate_environment_series
from src.sim.events import (
    EventEpisode,
    EventGenerator,
    EventSeries,
    generate_step_exponential_recovery,
)
from src.sim.faults import (
    FaultEpisode,
    FaultSeries,
    FaultSimulator,
    FaultType,
    SensorFaultSeries,
    generate_dropout_fault,
    generate_frozen_fault,
    generate_gain_error_fault,
    generate_high_noise_fault,
    generate_offset_drift_fault,
    generate_slow_degradation_fault,
    generate_spike_fault,
)
from src.sim.profile import TemperatureProfile, TemperatureScenarioParams, TemperatureSensorParams
from src.sim.scenario import (
    ScenarioSimulator,
    VirtualScenario,
    detect_shared_cause_pair,
    simulate_scenario,
)
from src.sim.sensors import VirtualSensorGroup, VirtualSensorSimulator
from src.sim.units import UnitScaler
from src.utils.config import canonical_json, get_config_dir, load_yaml


# =============================================================================
# Exact Schema Definitions (P1.8 Authoritative Schemas)
# =============================================================================

EXPECTED_OBSERVABLES_COLUMNS = [
    "scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"
]

EXPECTED_GROUND_TRUTH_SENSORS_COLUMNS = [
    "scenario_id", "group_id", "sensor_id", "timestamp", "true_value",
    "sensor_untrusted", "f_spike", "f_offset_drift", "f_gain_error",
    "f_frozen", "f_high_noise", "f_dropout", "f_slow_degradation",
    "fault_severity", "true_health", "fault_id"
]

EXPECTED_GROUND_TRUTH_EVENTS_COLUMNS = [
    "scenario_id", "group_id", "timestamp", "genuine_event",
    "event_below_detectability", "event_id", "event_magnitude_sigma",
    "true_event_breadth", "true_group_value"
]

EXPECTED_FAULT_EPISODES_COLUMNS = [
    "scenario_id", "group_id", "sensor_id", "fault_id", "fault_type",
    "start_step", "end_step", "duration", "max_severity", "shape_params"
]

EXPECTED_EVENT_EPISODES_COLUMNS = [
    "scenario_id", "event_id", "event_type", "scope", "start_step",
    "end_step", "start_time_minutes", "magnitude", "abs_magnitude",
    "sign", "affected_sensors", "affected_groups", "sensor_attenuations",
    "sensor_delays", "shape_params", "is_detectable"
]

EXPECTED_SCENARIOS_COLUMNS = [
    "scenario_id", "split", "generator_version", "seed", "scenario_type",
    "difficulty", "hard_case_type", "sensor_count", "group_count",
    "group_sizes", "length", "sampling", "profile", "quantisation",
    "clip_limits", "unit_scale", "faulty_sensors_count", "faulty_fraction",
    "has_events", "event_count", "has_shared_cause"
]


# =============================================================================
# 1. Exact Columns, Order, and Data Types Tests
# =============================================================================

def test_six_tables_exact_columns_and_order():
    """Verify exact column names and exact column order for all six tables."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scenario_0001", seed=101)

    df_obs = scen.to_observable_dataframe()
    df_gt_s = scen.to_ground_truth_sensors_dataframe()
    df_gt_e = scen.to_ground_truth_events_dataframe()
    df_f_ep = scen.to_fault_episodes_dataframe()
    df_e_ep = scen.to_event_episodes_dataframe()
    df_scen = scen.to_scenario_dataframe()

    assert list(df_obs.columns) == EXPECTED_OBSERVABLES_COLUMNS
    assert len(df_obs.columns) == 6

    assert list(df_gt_s.columns) == EXPECTED_GROUND_TRUTH_SENSORS_COLUMNS
    assert len(df_gt_s.columns) == 16

    assert list(df_gt_e.columns) == EXPECTED_GROUND_TRUTH_EVENTS_COLUMNS
    assert len(df_gt_e.columns) == 9

    assert list(df_f_ep.columns) == EXPECTED_FAULT_EPISODES_COLUMNS
    assert len(df_f_ep.columns) == 10

    assert list(df_e_ep.columns) == EXPECTED_EVENT_EPISODES_COLUMNS
    assert len(df_e_ep.columns) == 16

    assert list(df_scen.columns) == EXPECTED_SCENARIOS_COLUMNS
    assert len(df_scen.columns) == 21


def test_table_row_counts_standard_scenario():
    """Verify row counts for standard single-group scenario."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scenario_0002", seed=202, scenario_type="standard")
    n_sensors = scen.total_sensors
    assert len(scen.groups) == 1
    t_steps = scen.environment.total_steps

    df_obs = scen.to_observable_dataframe(stride=1)
    df_gt_s = scen.to_ground_truth_sensors_dataframe()
    df_gt_e = scen.to_ground_truth_events_dataframe()
    df_scen = scen.to_scenario_dataframe()

    assert len(df_obs) == n_sensors * t_steps
    assert len(df_gt_s) == n_sensors * t_steps
    assert len(df_gt_e) == 1 * t_steps
    assert len(df_scen) == 1


def test_table_row_counts_two_groups_scenario():
    """Verify row counts for two-group scenario (one row per group per step in ground_truth_events)."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scenario_0003", seed=303, scenario_type="two_groups")
    n_sensors = scen.total_sensors
    assert len(scen.groups) == 2
    t_steps = scen.environment.total_steps

    df_obs = scen.to_observable_dataframe(stride=1)
    df_gt_s = scen.to_ground_truth_sensors_dataframe()
    df_gt_e = scen.to_ground_truth_events_dataframe()
    df_scen = scen.to_scenario_dataframe()

    assert len(df_obs) == n_sensors * t_steps
    assert len(df_gt_s) == n_sensors * t_steps
    assert len(df_gt_e) == 2 * t_steps
    assert set(df_gt_e["group_id"].unique()) == {"group_00", "group_01"}
    assert len(df_scen) == 1


def test_table_row_counts_single_sensor_scenario():
    """Verify row counts for single-sensor scenario."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scenario_0004", seed=404, scenario_type="single_sensor")
    assert scen.total_sensors == 1
    t_steps = scen.environment.total_steps

    df_obs = scen.to_observable_dataframe(stride=1)
    df_gt_s = scen.to_ground_truth_sensors_dataframe()
    df_gt_e = scen.to_ground_truth_events_dataframe()
    df_scen = scen.to_scenario_dataframe()

    assert len(df_obs) == 1 * t_steps
    assert len(df_gt_s) == 1 * t_steps
    assert len(df_gt_e) == 1 * t_steps
    assert len(df_scen) == 1


# =============================================================================
# 2. Primary-Key Uniqueness Tests
# =============================================================================

def test_primary_key_uniqueness():
    """Verify primary key uniqueness for all six tables."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scenario_0005", seed=505, scenario_type="two_groups")

    # 1. observables: (scenario_id, group_id, sensor_id, timestamp)
    df_obs = scen.to_observable_dataframe()
    assert not df_obs.duplicated(subset=["scenario_id", "group_id", "sensor_id", "timestamp"]).any()

    # 2. ground_truth_sensors: (scenario_id, group_id, sensor_id, timestamp)
    df_gt_s = scen.to_ground_truth_sensors_dataframe()
    assert not df_gt_s.duplicated(subset=["scenario_id", "group_id", "sensor_id", "timestamp"]).any()

    # 3. ground_truth_events: (scenario_id, group_id, timestamp)
    df_gt_e = scen.to_ground_truth_events_dataframe()
    assert not df_gt_e.duplicated(subset=["scenario_id", "group_id", "timestamp"]).any()

    # 4. fault_episodes: (scenario_id, group_id, sensor_id, fault_id)
    df_f_ep = scen.to_fault_episodes_dataframe()
    if len(df_f_ep) > 0:
        assert not df_f_ep.duplicated(subset=["scenario_id", "group_id", "sensor_id", "fault_id"]).any()

    # 5. event_episodes: (scenario_id, event_id)
    df_e_ep = scen.to_event_episodes_dataframe()
    if len(df_e_ep) > 0:
        assert not df_e_ep.duplicated(subset=["scenario_id", "event_id"]).any()

    # 6. scenarios: scenario_id
    df_scen = scen.to_scenario_dataframe()
    assert not df_scen.duplicated(subset=["scenario_id"]).any()


# =============================================================================
# 3. All Seven Fault Severities, Clipping, and Health Invariant Tests
# =============================================================================

def test_all_seven_fault_severity_formulas_and_health_invariant():
    """Verify severity formulas for all 7 fault types and true_health = 1 - severity."""
    yaml_cfg = load_yaml(get_config_dir() / "generator_a.yaml")
    rng = np.random.default_rng(42)
    t_steps = 1920
    sigma_k = 0.20

    # 1. Spike: severity = k / 40
    ep_spike = generate_spike_fault(
        sensor_id="s0", sigma_k=sigma_k, total_steps=t_steps, start_step=500,
        duration=100, config=yaml_cfg, rng=rng, sp_size_sigma=20.0
    )
    assert np.all(ep_spike.severity >= 0.0)
    assert np.all(ep_spike.severity <= 1.0)
    # At active spike sample with 20 sigma, severity = 20 / 40 = 0.5
    active_sev = ep_spike.severity[ep_spike.label_mask]
    assert np.allclose(active_sev, 20.0 / 40.0)

    # 2. Offset drift: severity = |bias| / (20 * sigma_k)
    ep_drift = generate_offset_drift_fault(
        sensor_id="s0", sigma_k=sigma_k, total_steps=t_steps, start_step=500,
        config=yaml_cfg, rng=rng, bias_mag_sigma=10.0, shape="sudden_constant",
        ramp_steps=1, plateau_steps=200
    )
    # |bias| = 10 * 0.20 = 2.0; severity = 2.0 / (20 * 0.20) = 0.50
    active_drift_sev = ep_drift.severity[500:700]
    assert np.allclose(active_drift_sev, 0.50)

    # 3. Gain error: severity = |g - 1| / 0.6
    ep_gain = generate_gain_error_fault(
        sensor_id="s0", total_steps=t_steps, start_step=500,
        duration=300, physical_xref=20.0, config=yaml_cfg, rng=rng, gain=1.30
    )
    # |1.30 - 1.0| / 0.6 = 0.50
    assert np.allclose(ep_gain.severity[500:800], 0.50)

    # 4. Frozen: 0.8 for last-reading, 1.0 for wrong-value
    ep_frozen_last = generate_frozen_fault(
        sensor_id="s0", sigma_k=sigma_k, total_steps=t_steps, start_step=500,
        duration=100, config=yaml_cfg, rng=rng, clean_base_value=20.0,
        is_wrong_value=False
    )
    assert np.allclose(ep_frozen_last.severity[500:600], 0.80)

    ep_frozen_wrong = generate_frozen_fault(
        sensor_id="s0", sigma_k=sigma_k, total_steps=t_steps, start_step=500,
        duration=100, config=yaml_cfg, rng=rng, clean_base_value=20.0,
        is_wrong_value=True
    )
    assert np.allclose(ep_frozen_wrong.severity[500:600], 1.00)

    # 5. High noise: (m - 1) / 9
    ep_noise = generate_high_noise_fault(
        sensor_id="s0", total_steps=t_steps, start_step=500, duration=200,
        config=yaml_cfg, rng=rng, multiplier=5.5, is_bursty=False
    )
    # (5.5 - 1.0) / 9.0 = 0.50
    assert np.allclose(ep_noise.severity[500:700], 0.50)

    # 6. Dropout: 0.3 + 0.7 * missing_frac_30
    ep_drop = generate_dropout_fault(
        sensor_id="s0", total_steps=t_steps, start_step=500, duration=150,
        config=yaml_cfg, rng=rng
    )
    assert np.all(ep_drop.severity >= 0.0)
    assert np.all(ep_drop.severity <= 1.0)

    # 7. Slow degradation: severity = d(t)
    ep_deg = generate_slow_degradation_fault(
        sensor_id="s0", sigma_k=sigma_k, total_steps=t_steps, start_step=500,
        duration=500, config=yaml_cfg, rng=rng, h_end=0.40
    )
    # d(t) rises linearly from 0 to 0.60; severity matches d(t)
    assert np.isclose(ep_deg.severity[500], 0.0)
    assert np.isclose(ep_deg.severity[999], 0.60, atol=0.01)

    # Check composite fault series health invariant
    series = FaultSeries(sensor_id="s0", total_steps=t_steps, episodes=[ep_gain, ep_deg])
    sev = series.fault_severity
    health = series.true_health
    assert np.all(sev >= 0.0) and np.all(sev <= 1.0)
    assert np.all(health >= 0.0) and np.all(health <= 1.0)
    assert np.allclose(health, 1.0 - sev)

    # Untrusted threshold for slow degradation: untrusted only when d >= 0.15
    deg_series = FaultSeries(sensor_id="s0", total_steps=t_steps, episodes=[ep_deg])
    for t in range(500, 1000):
        d_val = (t - 500) / 500.0 * 0.60
        if d_val >= 0.15:
            assert deg_series.sensor_untrusted[t] == 1
        else:
            assert deg_series.sensor_untrusted[t] == 0


# =============================================================================
# 4. Decision 45 Event Activity Semantics Tests
# =============================================================================

def test_event_activity_zero_events():
    """Verify event activity table with zero genuine events."""
    sim = ScenarioSimulator()
    # Force zero events
    scen = sim.simulate_scenario(scenario_id="scen_zero_ev", seed=10)
    # Manually clear events to test zero-event state
    for g in scen.groups.values():
        object.__setattr__(g, "events", None)
    scen.events = None

    df_gt_e = scen.to_ground_truth_events_dataframe()
    assert (df_gt_e["genuine_event"] == 0).all()
    assert (df_gt_e["event_below_detectability"] == 0).all()
    assert (df_gt_e["event_id"] == "").all()
    assert (df_gt_e["event_magnitude_sigma"] == 0.0).all()
    assert (df_gt_e["true_event_breadth"] == 0.0).all()


def test_event_activity_detectable_event():
    """Verify detectable event (> 3 sigma) produces genuine_event=1, event_below=0."""
    t_steps = 1920
    dt = 1.0
    t_minutes = np.arange(t_steps, dtype=float) * dt
    ep = EventEpisode(
        event_id="ev_det",
        event_type="pulse",
        start_step=600,
        start_time_minutes=600.0,
        magnitude=4.0,
        abs_magnitude=4.0,
        sign=1,
        scope="global",
        affected_sensors=("sensor_00", "sensor_01"),
        sensor_attenuations={"sensor_00": 1.0, "sensor_01": 1.0},
        sensor_delays={"sensor_00": 0.0, "sensor_01": 0.0},
        shape_params={"duration_minutes": 60.0},
        is_detectable=True,
        end_step=660,
    )
    s_params = [
        TemperatureSensorParams(sensor_id="sensor_00", sigma_k=0.15, static_offset=0.0, response_lag_tau=0.0, quantisation_step=0.01, is_noise_correlated=False, noise_correlation_rho=0.0),
        TemperatureSensorParams(sensor_id="sensor_01", sigma_k=0.20, static_offset=0.0, response_lag_tau=0.0, quantisation_step=0.01, is_noise_correlated=False, noise_correlation_rho=0.0),
    ]
    w = ep.generate_sensor_waveform("sensor_00", t_minutes)
    es = EventSeries(
        group_id="group_00",
        total_steps=t_steps,
        episodes=[ep],
        sensor_contributions={"sensor_00": w, "sensor_01": w},
        genuine_event=np.zeros(t_steps, dtype=int),
        event_below_detectability=np.zeros(t_steps, dtype=int),
        event_magnitude_sigma=np.zeros(t_steps, dtype=float),
        true_event_breadth=np.zeros(t_steps, dtype=float),
        active_event_ids=["" for _ in range(t_steps)],
        true_group_value=np.zeros(t_steps, dtype=float),
    )
    sim = ScenarioSimulator()
    g_events = sim._filter_events_for_group(es, s_params, "group_00")

    # In active window [600, 660)
    for t in range(600, 660):
        assert g_events.genuine_event[t] == 1
        assert g_events.event_below_detectability[t] == 0
        assert g_events.active_event_ids[t] == "ev_det"
        assert g_events.event_magnitude_sigma[t] >= 3.0
        assert g_events.true_event_breadth[t] == 1.0


def test_event_activity_sub_detectable_event():
    """Verify sub-detectable event (< 3 sigma, > epsilon) produces genuine_event=0, event_below=1."""
    t_steps = 1920
    dt = 1.0
    t_minutes = np.arange(t_steps, dtype=float) * dt
    # Magnitude 0.20 C on sensor with sigma = 0.15 C -> ratio = 1.33 < 3.0
    ep = EventEpisode(
        event_id="ev_sub",
        event_type="pulse",
        start_step=600,
        start_time_minutes=600.0,
        magnitude=0.20,
        abs_magnitude=0.20,
        sign=1,
        scope="global",
        affected_sensors=("sensor_00",),
        sensor_attenuations={"sensor_00": 1.0},
        sensor_delays={"sensor_00": 0.0},
        shape_params={"duration_minutes": 50.0},
        is_detectable=False,
        end_step=650,
    )
    s_params = [
        TemperatureSensorParams(sensor_id="sensor_00", sigma_k=0.15, static_offset=0.0, response_lag_tau=0.0, quantisation_step=0.01, is_noise_correlated=False, noise_correlation_rho=0.0),
    ]
    w = ep.generate_sensor_waveform("sensor_00", t_minutes)
    es = EventSeries(
        group_id="group_00",
        total_steps=t_steps,
        episodes=[ep],
        sensor_contributions={"sensor_00": w},
        genuine_event=np.zeros(t_steps, dtype=int),
        event_below_detectability=np.zeros(t_steps, dtype=int),
        event_magnitude_sigma=np.zeros(t_steps, dtype=float),
        true_event_breadth=np.zeros(t_steps, dtype=float),
        active_event_ids=["" for _ in range(t_steps)],
        true_group_value=np.zeros(t_steps, dtype=float),
    )
    sim = ScenarioSimulator()
    g_events = sim._filter_events_for_group(es, s_params, "group_00")

    for t in range(600, 650):
        assert g_events.genuine_event[t] == 0
        assert g_events.event_below_detectability[t] == 1
        assert g_events.active_event_ids[t] == "ev_sub"
        assert 0.0 < g_events.event_magnitude_sigma[t] < 3.0


def test_event_activity_opposite_sign_cancellation():
    """Verify opposite-sign cancellation preserves both active event IDs."""
    t_steps = 1920
    dt = 1.0
    t_minutes = np.arange(t_steps, dtype=float) * dt
    # Two simultaneous pulses of +2.0 C and -2.0 C
    ep1 = EventEpisode(
        event_id="ev_pos",
        event_type="pulse",
        start_step=700,
        start_time_minutes=700.0,
        magnitude=2.0,
        abs_magnitude=2.0,
        sign=1,
        scope="global",
        affected_sensors=("sensor_00",),
        sensor_attenuations={"sensor_00": 1.0},
        sensor_delays={"sensor_00": 0.0},
        shape_params={"duration_minutes": 30.0},
        is_detectable=True,
        end_step=730,
    )
    ep2 = EventEpisode(
        event_id="ev_neg",
        event_type="pulse",
        start_step=700,
        start_time_minutes=700.0,
        magnitude=-2.0,
        abs_magnitude=2.0,
        sign=-1,
        scope="global",
        affected_sensors=("sensor_00",),
        sensor_attenuations={"sensor_00": 1.0},
        sensor_delays={"sensor_00": 0.0},
        shape_params={"duration_minutes": 30.0},
        is_detectable=True,
        end_step=730,
    )
    w1 = ep1.generate_sensor_waveform("sensor_00", t_minutes)
    w2 = ep2.generate_sensor_waveform("sensor_00", t_minutes)
    w_sum = w1 + w2  # Exactly 0.0 in window

    s_params = [
        TemperatureSensorParams(sensor_id="sensor_00", sigma_k=0.15, static_offset=0.0, response_lag_tau=0.0, quantisation_step=0.01, is_noise_correlated=False, noise_correlation_rho=0.0),
    ]
    es = EventSeries(
        group_id="group_00",
        total_steps=t_steps,
        episodes=[ep1, ep2],
        sensor_contributions={"sensor_00": w_sum},
        genuine_event=np.zeros(t_steps, dtype=int),
        event_below_detectability=np.zeros(t_steps, dtype=int),
        event_magnitude_sigma=np.zeros(t_steps, dtype=float),
        true_event_breadth=np.zeros(t_steps, dtype=float),
        active_event_ids=["" for _ in range(t_steps)],
        true_group_value=np.zeros(t_steps, dtype=float),
    )
    sim = ScenarioSimulator()
    g_events = sim._filter_events_for_group(es, s_params, "group_00")

    # In active window [700, 730), both episodes are active and ordered deterministically
    for t in range(700, 730):
        assert "ev_neg" in g_events.active_event_ids[t]
        assert "ev_pos" in g_events.active_event_ids[t]
        assert g_events.active_event_ids[t] == "ev_neg,ev_pos"
        # Since net perturbation is 0, composite detectability is below threshold
        assert g_events.genuine_event[t] == 0
        assert g_events.event_below_detectability[t] == 1


def test_event_activity_sub_epsilon_tail():
    """Verify waveform tails below activity epsilon (1e-4) produce zero diagnostic magnitude."""
    t_steps = 1920
    dt = 1.0
    t_minutes = np.arange(t_steps, dtype=float) * dt
    # Exponential decay with small magnitude reaching below epsilon
    tau = 20.0
    mag = 1.0
    ep = EventEpisode(
        event_id="ev_tail",
        event_type="step_exponential_recovery",
        start_step=500,
        start_time_minutes=500.0,
        magnitude=mag,
        abs_magnitude=mag,
        sign=1,
        scope="global",
        affected_sensors=("sensor_00",),
        sensor_attenuations={"sensor_00": 1.0},
        sensor_delays={"sensor_00": 0.0},
        shape_params={"tau_minutes": tau},
        is_detectable=True,
        end_step=1000,
    )
    w = ep.generate_sensor_waveform("sensor_00", t_minutes)
    s_params = [
        TemperatureSensorParams(sensor_id="sensor_00", sigma_k=0.15, static_offset=0.0, response_lag_tau=0.0, quantisation_step=0.01, is_noise_correlated=False, noise_correlation_rho=0.0),
    ]
    es = EventSeries(
        group_id="group_00",
        total_steps=t_steps,
        episodes=[ep],
        sensor_contributions={"sensor_00": w},
        genuine_event=np.zeros(t_steps, dtype=int),
        event_below_detectability=np.zeros(t_steps, dtype=int),
        event_magnitude_sigma=np.zeros(t_steps, dtype=float),
        true_event_breadth=np.zeros(t_steps, dtype=float),
        active_event_ids=["" for _ in range(t_steps)],
        true_group_value=np.zeros(t_steps, dtype=float),
    )
    sim = ScenarioSimulator()
    g_events = sim._filter_events_for_group(es, s_params, "group_00")

    # At step 800: t - t_arr = 300 min; w(800) = 1.0 * exp(-300/20) = exp(-15) ~ 3e-7 < 1e-4
    assert abs(w[800]) < 1e-4
    assert g_events.active_event_ids[800] == ""
    assert g_events.event_magnitude_sigma[800] == 0.0
    assert g_events.genuine_event[800] == 0
    assert g_events.event_below_detectability[800] == 0
    assert g_events.true_event_breadth[800] == 0.0


def test_event_mutual_exclusivity():
    """Verify genuine_event and event_below_detectability are strictly mutually exclusive."""
    sim = ScenarioSimulator()
    for seed in [11, 22, 33]:
        scen = sim.simulate_scenario(scenario_id=f"scen_{seed}", seed=seed)
        df_e = scen.to_ground_truth_events_dataframe()
        sums = df_e["genuine_event"] + df_e["event_below_detectability"]
        assert (sums <= 1).all()


# =============================================================================
# 5. Multi-Group Events and Affected Groups Serialization Tests
# =============================================================================

def test_multi_group_event_affected_groups_serialization():
    """Verify affected_groups accurately reflects all impacted groups."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scen_mg", seed=777, scenario_type="two_groups")
    df_e_ep = scen.to_event_episodes_dataframe()

    if len(df_e_ep) > 0:
        for _, row in df_e_ep.iterrows():
            aff_sids = row["affected_sensors"].split(",")
            aff_groups = row["affected_groups"].split(",")
            # Verify every affected sensor belongs to one of the listed affected groups
            for sid in aff_sids:
                found_g = None
                for gid, g in scen.groups.items():
                    if sid in g.sensors:
                        found_g = gid
                        break
                assert found_g in aff_groups


# =============================================================================
# 6. JSON Serialization and Metadata Determinism Tests
# =============================================================================

def test_deterministic_json_serialization():
    """Verify shape_params, unit_scale, group_sizes, quantisation, and clip_limits parse and sort keys."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scen_json", seed=888)

    # Scenarios table JSON fields
    df_scen = scen.to_scenario_dataframe()
    row_scen = df_scen.iloc[0]

    # unit_scale
    unit_scale_dict = json.loads(row_scen["unit_scale"])
    assert "scale_type" in unit_scale_dict
    assert "multiplier" in unit_scale_dict
    assert "offset" in unit_scale_dict
    # Key order must be alphabetical
    assert list(unit_scale_dict.keys()) == sorted(unit_scale_dict.keys())

    # group_sizes
    gs_dict = json.loads(row_scen["group_sizes"])
    assert isinstance(gs_dict, dict)
    assert list(gs_dict.keys()) == sorted(gs_dict.keys())

    # clip_limits
    clip_dict = json.loads(row_scen["clip_limits"])
    assert "environment" in clip_dict and "sensor" in clip_dict
    assert list(clip_dict.keys()) == sorted(clip_dict.keys())

    # quantisation
    q_dict = json.loads(row_scen["quantisation"])
    assert len(q_dict) == scen.total_sensors
    assert list(q_dict.keys()) == sorted(q_dict.keys())

    # Fault episodes shape_params
    df_f_ep = scen.to_fault_episodes_dataframe()
    for _, f_row in df_f_ep.iterrows():
        sp = json.loads(f_row["shape_params"])
        assert isinstance(sp, dict)
        assert list(sp.keys()) == sorted(sp.keys())

    # Event episodes shape_params, attenuations, delays
    df_e_ep = scen.to_event_episodes_dataframe()
    for _, e_row in df_e_ep.iterrows():
        sp = json.loads(e_row["shape_params"])
        att = json.loads(e_row["sensor_attenuations"])
        delays = json.loads(e_row["sensor_delays"])
        assert isinstance(sp, dict)
        assert isinstance(att, dict)
        assert isinstance(delays, dict)
        assert list(sp.keys()) == sorted(sp.keys())
        assert list(att.keys()) == sorted(att.keys())
        assert list(delays.keys()) == sorted(delays.keys())


# =============================================================================
# 7. Layer 1 Observable Isolation and Stride Tests
# =============================================================================

def test_layer1_observable_isolation():
    """Verify observables table has strictly 6 columns and zero ground-truth leakage."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scen_iso", seed=999)
    df_obs = scen.to_observable_dataframe()

    assert list(df_obs.columns) == [
        "scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"
    ]
    forbidden_leakage_keywords = [
        "fault", "event", "truth", "untrusted", "health", "severity",
        "spike", "drift", "gain", "frozen", "noise", "dropout", "degradation"
    ]
    for col in df_obs.columns:
        for kw in forbidden_leakage_keywords:
            assert kw not in col.lower()


def test_observable_row_stride():
    """Verify observable stride parameter downsampling."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scen_stride", seed=1001)

    df_stride1 = scen.to_observable_dataframe(stride=1)
    df_stride5 = scen.to_observable_dataframe(stride=5)
    df_stride10 = scen.to_observable_dataframe(stride=10)

    n_sensors = scen.total_sensors
    assert len(df_stride1) == n_sensors * 1920
    assert len(df_stride5) == n_sensors * 384
    assert len(df_stride10) == n_sensors * 192

    # Check timestamps match stride
    s0_s1 = df_stride1[df_stride1["sensor_id"] == "sensor_00"]["timestamp"].values
    s0_s5 = df_stride5[df_stride5["sensor_id"] == "sensor_00"]["timestamp"].values
    assert np.allclose(s0_s1[::5], s0_s5)

    with pytest.raises(ValueError):
        scen.to_observable_dataframe(stride=0)


# =============================================================================
# 8. Seeded Determinism Across All Six Tables
# =============================================================================

def test_seeded_determinism_across_all_tables():
    """Verify running simulation twice with identical seed produces identical tables."""
    sim1 = ScenarioSimulator()
    scen1 = sim1.simulate_scenario(scenario_id="scen_det", seed=4242)

    sim2 = ScenarioSimulator()
    scen2 = sim2.simulate_scenario(scenario_id="scen_det", seed=4242)

    pd.testing.assert_frame_equal(scen1.to_observable_dataframe(), scen2.to_observable_dataframe())
    pd.testing.assert_frame_equal(scen1.to_ground_truth_sensors_dataframe(), scen2.to_ground_truth_sensors_dataframe())
    pd.testing.assert_frame_equal(scen1.to_ground_truth_events_dataframe(), scen2.to_ground_truth_events_dataframe())
    pd.testing.assert_frame_equal(scen1.to_fault_episodes_dataframe(), scen2.to_fault_episodes_dataframe())
    pd.testing.assert_frame_equal(scen1.to_event_episodes_dataframe(), scen2.to_event_episodes_dataframe())
    pd.testing.assert_frame_equal(scen1.to_scenario_dataframe(), scen2.to_scenario_dataframe())


# =============================================================================
# 9. Zero-Fault and Multi-Fault Scenarios Tests
# =============================================================================

def test_zero_fault_scenario_export():
    """Verify zero-fault scenario exports clean ground truth."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scen_zf", seed=5000, k_candidate=0)

    df_f_ep = scen.to_fault_episodes_dataframe()
    assert len(df_f_ep) == 0
    assert list(df_f_ep.columns) == EXPECTED_FAULT_EPISODES_COLUMNS

    df_gt_s = scen.to_ground_truth_sensors_dataframe()
    assert (df_gt_s["sensor_untrusted"] == 0).all()
    assert (df_gt_s["fault_severity"] == 0.0).all()
    assert (df_gt_s["true_health"] == 1.0).all()
    assert (df_gt_s["fault_id"] == "").all()
    for col in ["f_spike", "f_offset_drift", "f_gain_error", "f_frozen", "f_high_noise", "f_dropout", "f_slow_degradation"]:
        assert (df_gt_s[col] == 0).all()

    df_scen = scen.to_scenario_dataframe()
    assert df_scen["faulty_sensors_count"].iloc[0] == 0
    assert df_scen["faulty_fraction"].iloc[0] == 0.0


def test_multi_fault_simultaneous_episodes():
    """Verify multi-fault scenario joins active fault IDs deterministically."""
    sim = ScenarioSimulator()
    scen = sim.simulate_scenario(scenario_id="scen_mf", seed=6000, k_candidate=3, difficulty_tier="hard")

    df_gt_s = scen.to_ground_truth_sensors_dataframe()
    # Check if any sensor has multiple concurrent active faults
    multi_active_rows = df_gt_s[df_gt_s["fault_id"].str.contains(",")]
    if len(multi_active_rows) > 0:
        row = multi_active_rows.iloc[0]
        ids = row["fault_id"].split(",")
        assert len(ids) >= 2
        # Verify severities and true_health invariant on multi-fault row
        assert np.isclose(row["true_health"], 1.0 - row["fault_severity"])
