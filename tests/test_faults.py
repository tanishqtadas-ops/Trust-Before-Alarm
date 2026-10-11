"""Unit and integration tests for fault injectors (Appendix A.4, Decisions 1-6, 22-28).

Validates:
  1. Fault start step distribution: 10% present from step 0, 90% in [480, 1919].
  2. Spike fault: renewal process [10, 60], lengths {1, 2, 3}, size [3, 40] sigma_k,
     label active on spike samples + 4 following steps, severity k / 40.
  3. Offset drift: linear, saturating quadratic, stepped (K in {3, 4, 5}), sudden constant,
     ramp [60, 720], plateau [0, 600], final bias [1, 20] sigma_k, severity |bias| / (20 sigma_k).
  4. Gain error: standard vs subtle, duration [120, 1200], x_ref = true physical onset value,
     severity |g - 1| / 0.6.
  5. Frozen sensor: step 0 start uses initial healthy reading, 80% last reading, 20% wrong value,
     60% constant, 40% jitter, severity 0.8 vs 1.0.
  6. High noise: multiplier [2, 10], 70% continuous, 30% bursty [5, 30], severity (m - 1) / 9.
  7. Dropout: enter-gap prob [0.002, 0.03], geometric gaps capped at 100, severity 0.3 + 0.7 * missing_frac_30.
  8. Slow degradation: health falls to h_end [0.1, 0.7], noise (1 + 5d), persistent bias +/- 8 sigma_k * d,
     extra missing prob 0.15 d, untrusted once d >= 0.15, only labeled as slow_degradation.
  9. 11-step pipeline transformation order, gain-error reference, clipping, quantisation, unit scaling last.
 10. Strict Layer 1 vs Layer 3 data isolation.
 11. Determinism with fixed seeds.
"""

import numpy as np
import pandas as pd
import pytest

from src.sim.environment import generate_environment_series
from src.sim.faults import (
    FaultEpisode,
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
    sample_fault_start_step,
    sample_log_uniform,
)
from src.sim.profile import TemperatureProfile, TemperatureSensorParams
from src.sim.sensors import VirtualSensorSimulator, simulate_sensors
from src.sim.units import UnitScaler
from src.utils.config import get_config_dir, load_yaml


@pytest.fixture
def config():
    return load_yaml(get_config_dir() / "generator_a.yaml")


@pytest.fixture
def fault_sim(config):
    return FaultSimulator(config=config)


@pytest.fixture
def dummy_sensor_params():
    return TemperatureSensorParams(
        sensor_id="sensor_00",
        static_offset=0.2,
        sigma_k=0.15,
        response_lag_tau=1.5,
        quantisation_step=0.05,
        missing_rate=0.005,
        burst_max=3,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
    )


# =============================================================================
# 1. Start Step Distribution & Helper Tests
# =============================================================================

def test_sample_log_uniform():
    rng = np.random.default_rng(42)
    samples = [sample_log_uniform(1.0, 100.0, rng) for _ in range(1000)]
    assert all(1.0 <= s <= 100.0 for s in samples)
    # In log space, mean of log(s) should be near log(10) ~ 2.302
    log_mean = np.mean(np.log(samples))
    expected_log_mean = (np.log(1.0) + np.log(100.0)) / 2.0
    assert abs(log_mean - expected_log_mean) < 0.15


def test_fault_start_step_distribution():
    rng = np.random.default_rng(123)
    N = 2000
    starts = [sample_fault_start_step(total_steps=1920, warmup_steps=480, prob_present_from_start=0.10, rng=rng) for _ in range(N)]
    
    starts = np.array(starts)
    step0_frac = np.mean(starts == 0)
    # Expected ~10% step 0 starts
    assert 0.075 <= step0_frac <= 0.125

    post_warmup = starts[starts > 0]
    # Post-warmup must be strictly within [480, 1919]
    assert np.all(post_warmup >= 480)
    assert np.all(post_warmup < 1920)
    # Mean of post-warmup should be near (480 + 1919) / 2 = 1199.5
    assert abs(np.mean(post_warmup) - 1199.5) < 30.0


# =============================================================================
# 2. Spike Fault Tests
# =============================================================================

def test_spike_fault_properties(config, dummy_sensor_params):
    rng = np.random.default_rng(101)
    ep = generate_spike_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=600,
        duration=200,
        config=config,
        rng=rng,
        fault_id="spike_test",
    )

    assert ep.fault_type == FaultType.SPIKE
    assert ep.start_step == 600
    assert ep.end_step == 800
    assert ep.duration == 200

    # Spikes exist and are active
    spikes_info = ep.shape_params["spikes"]
    assert len(spikes_info) >= 2
    # First spike begins at start_step
    assert spikes_info[0]["step"] == 600

    # Check inter-spike intervals in [10, 60]
    for i in range(len(spikes_info) - 1):
        gap = spikes_info[i + 1]["step"] - (spikes_info[i]["step"] + spikes_info[i]["length"])
        assert 10 <= gap <= 60

    # Label mask active on spike samples + 4 following steps
    for sp in spikes_info:
        st = sp["step"]
        le = sp["length"]
        # Spike samples must have non-zero additive offset
        assert np.all(ep.additive_offset[st : st + le] != 0.0)
        # Label mask must be True for spike + 4 steps
        assert np.all(ep.label_mask[st : min(st + le + 4, 1920)])
        assert np.all(ep.untrusted_mask[st : min(st + le + 4, 1920)])
        # Severity matches k / 40
        expected_sev = min(sp["size_sigma"] / 40.0, 1.0)
        assert np.all(ep.severity[st : min(st + le + 4, 1920)] >= expected_sev - 1e-6)

    # Outside active label windows, label_mask is False
    assert not np.any(ep.label_mask[:600])
    assert not np.any(ep.label_mask[850:])


# =============================================================================
# 3. Offset Drift Fault Tests
# =============================================================================

def test_offset_drift_linear(config, dummy_sensor_params):
    rng = np.random.default_rng(202)
    ep = generate_offset_drift_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=500,
        config=config,
        rng=rng,
        fault_id="drift_linear",
        ramp_steps=100,
        plateau_steps=50,
    )

    assert ep.fault_type == FaultType.OFFSET_DRIFT
    assert ep.start_step == 500
    assert ep.end_step == 650
    final_bias = ep.shape_params["final_bias"]

    # At start, bias begins near 0
    assert abs(ep.additive_offset[500]) <= abs(final_bias) / 100.0 + 1e-6
    # At plateau, bias reaches final_bias
    assert np.allclose(ep.additive_offset[600:650], final_bias)

    # Label mask active throughout ramp and plateau
    assert np.all(ep.label_mask[500:650])
    assert not np.any(ep.label_mask[:500])
    assert not np.any(ep.label_mask[650:])


def test_offset_drift_saturating_quadratic(config, dummy_sensor_params):
    # Test parameter-free quadratic curve: B * (1 - (1 - u)^2)
    rng = np.random.default_rng(303)
    # Force saturating shape
    cfg_copy = dict(config)
    cfg_copy["faults"] = dict(config["faults"])
    cfg_copy["faults"]["offset_drift"] = dict(config["faults"]["offset_drift"])
    cfg_copy["faults"]["offset_drift"]["shape_probs"] = {
        "linear": 0.0,
        "saturating": 1.0,
        "stepped": 0.0,
        "sudden_constant": 0.0,
    }

    ep = generate_offset_drift_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=480,
        config=cfg_copy,
        rng=rng,
        ramp_steps=100,
        plateau_steps=20,
    )

    assert ep.shape_params["shape"] == "saturating"
    B = ep.shape_params["final_bias"]

    # Midpoint progress u = 0.5: quadratic formula (1 - 0.5^2) = 0.75 * B
    mid_offset = ep.additive_offset[480 + 50]
    expected_mid = B * 0.75
    assert abs(mid_offset - expected_mid) < 1e-4

    # Monotonically increasing towards B
    diffs = np.diff(ep.additive_offset[480:580])
    if B > 0:
        assert np.all(diffs >= -1e-9)
    else:
        assert np.all(diffs <= 1e-9)


def test_offset_drift_stepped(config, dummy_sensor_params):
    rng = np.random.default_rng(404)
    cfg_copy = dict(config)
    cfg_copy["faults"] = dict(config["faults"])
    cfg_copy["faults"]["offset_drift"] = dict(config["faults"]["offset_drift"])
    cfg_copy["faults"]["offset_drift"]["shape_probs"] = {
        "linear": 0.0,
        "saturating": 0.0,
        "stepped": 1.0,
        "sudden_constant": 0.0,
    }

    ep = generate_offset_drift_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=500,
        config=cfg_copy,
        rng=rng,
        ramp_steps=120,
        plateau_steps=30,
    )

    assert ep.shape_params["shape"] == "stepped"
    K = ep.shape_params["stepped_k"]
    assert K in [3, 4, 5]

    # Verify staircase plateaus: unique values during ramp should be <= K + 1
    ramp_vals = ep.additive_offset[500:620]
    unique_vals = np.unique(np.round(ramp_vals, 6))
    assert len(unique_vals) <= K + 1


def test_offset_drift_sudden_constant(config, dummy_sensor_params):
    rng = np.random.default_rng(505)
    cfg_copy = dict(config)
    cfg_copy["faults"] = dict(config["faults"])
    cfg_copy["faults"]["offset_drift"] = dict(config["faults"]["offset_drift"])
    cfg_copy["faults"]["offset_drift"]["shape_probs"] = {
        "linear": 0.0,
        "saturating": 0.0,
        "stepped": 0.0,
        "sudden_constant": 1.0,
    }

    ep = generate_offset_drift_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=600,
        config=cfg_copy,
        rng=rng,
        ramp_steps=100,
        plateau_steps=50,
    )

    assert ep.shape_params["shape"] == "sudden_constant"
    B = ep.shape_params["final_bias"]
    # All steps in [600, 750) must equal final_bias exactly
    assert np.allclose(ep.additive_offset[600:750], B)


# =============================================================================
# 4. Gain Error Fault Tests
# =============================================================================

def test_gain_error_fault(config, dummy_sensor_params):
    rng = np.random.default_rng(606)
    physical_xref = 23.45
    ep = generate_gain_error_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        total_steps=1920,
        start_step=700,
        duration=300,
        physical_xref=physical_xref,
        config=config,
        rng=rng,
        fault_id="gain_test",
    )

    assert ep.fault_type == FaultType.GAIN_ERROR
    assert ep.start_step == 700
    assert ep.end_step == 1000
    assert ep.gain is not None
    assert ep.gain_xref == physical_xref

    g = ep.gain
    # Check that g is within specified intervals
    is_standard = (0.55 <= g <= 0.85) or (1.15 <= g <= 1.60)
    is_subtle = (0.90 <= g <= 0.95) or (1.05 <= g <= 1.10)
    assert is_standard or is_subtle

    # Severity formula |g - 1| / 0.6
    expected_sev = min(abs(g - 1.0) / 0.6, 1.0)
    assert np.allclose(ep.severity[700:1000], expected_sev)
    assert not np.any(ep.severity[:700] > 0)


# =============================================================================
# 5. Frozen Sensor Fault Tests
# =============================================================================

def test_frozen_sensor_step0_initial_reading(config, dummy_sensor_params):
    rng = np.random.default_rng(707)
    clean_reading_step0 = 19.82
    ep = generate_frozen_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=0,
        duration=100,
        clean_base_value=clean_reading_step0,
        config=config,
        rng=rng,
        fault_id="frozen_step0",
    )

    assert ep.is_frozen
    assert ep.start_step == 0
    assert ep.end_step == 100

    if not ep.shape_params["is_wrong_value"]:
        # Frozen at clean reading
        assert ep.frozen_value == clean_reading_step0
        assert np.allclose(ep.severity[0:100], 0.8)
    else:
        # Frozen at wrong value
        assert abs(ep.frozen_value - clean_reading_step0) >= 5.0 * dummy_sensor_params.sigma_k
        assert np.allclose(ep.severity[0:100], 1.0)


# =============================================================================
# 6. High Noise Fault Tests
# =============================================================================

def test_high_noise_fault(config, dummy_sensor_params):
    rng = np.random.default_rng(808)
    ep = generate_high_noise_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        total_steps=1920,
        start_step=500,
        duration=200,
        config=config,
        rng=rng,
        fault_id="noise_test",
    )

    assert ep.fault_type == FaultType.HIGH_NOISE
    mult = ep.shape_params["multiplier"]
    assert 2.0 <= mult <= 10.0

    if not ep.shape_params["is_bursty"]:
        # Continuous: all active steps have noise_multiplier = mult
        assert np.allclose(ep.noise_multiplier[500:700], mult)
        assert np.allclose(ep.severity[500:700], min((mult - 1.0) / 9.0, 1.0))
    else:
        # Bursty: multiplier alternates between mult and 1.0
        active_mults = ep.noise_multiplier[500:700]
        assert np.all(np.isin(np.round(active_mults, 4), [1.0, round(mult, 4)]))
        assert np.any(np.isclose(active_mults, mult))

    # Outside episode: multiplier is exactly 1.0
    assert np.allclose(ep.noise_multiplier[:500], 1.0)
    assert np.allclose(ep.noise_multiplier[700:], 1.0)


# =============================================================================
# 7. Dropout Fault Tests
# =============================================================================

def test_dropout_fault(config, dummy_sensor_params):
    rng = np.random.default_rng(909)
    ep = generate_dropout_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        total_steps=1920,
        start_step=500,
        duration=300,
        config=config,
        rng=rng,
        fault_id="dropout_test",
    )

    assert ep.fault_type == FaultType.DROPOUT
    assert np.all(ep.label_mask[500:800])
    assert not np.any(ep.label_mask[:500])
    assert not np.any(ep.label_mask[800:])

    # Dropout creates missing readings within the episode
    missing_in_ep = ep.shape_params["total_missing_in_episode"]
    assert missing_in_ep >= 0
    # Outside episode, missing_mask is strictly False
    assert not np.any(ep.missing_mask[:500])
    assert not np.any(ep.missing_mask[800:])

    # Severity formula: 0.3 + 0.7 * missing_frac_30
    assert np.all(ep.severity[500:800] >= 0.3)
    assert np.all(ep.severity[500:800] <= 1.0)


# =============================================================================
# 8. Slow Degradation Fault Tests
# =============================================================================

def test_slow_degradation_fault(config, dummy_sensor_params):
    rng = np.random.default_rng(1010)
    ep = generate_slow_degradation_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=500,
        duration=600,
        config=config,
        rng=rng,
        fault_id="degradation_test",
    )

    assert ep.fault_type == FaultType.SLOW_DEGRADATION
    h_end = ep.shape_params["h_end"]
    assert 0.1 <= h_end <= 0.7
    final_damage = ep.shape_params["final_damage"]
    assert 0.3 <= final_damage <= 0.9

    # Damage increases monotonically
    # Check noise multiplier: (1 + 5d)
    assert ep.noise_multiplier[500] == 1.0
    expected_final_mult = 1.0 + 5.0 * final_damage
    assert abs(ep.noise_multiplier[1099] - expected_final_mult) < 0.1

    # Untrusted mask activates once d >= 0.15
    # Steps where d < 0.15 should be untrusted_mask = False
    assert not ep.untrusted_mask[500]
    assert ep.untrusted_mask[1099]

    # Severity matches d(t)
    assert np.allclose(ep.severity[500], 0.0)
    assert np.isclose(ep.severity[1099], final_damage, atol=0.05)


# =============================================================================
# 9. 11-Step Transduction Pipeline & Unit Scaling Test
# =============================================================================

def test_sensor_simulation_with_faults_and_unit_scaling(config, dummy_sensor_params):
    env_series = generate_environment_series(seed=42, config=config)
    sim = VirtualSensorSimulator(config=config)

    # Create a sensor fault series with spike and drift
    rng = np.random.default_rng(777)
    sp_ep = generate_spike_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=600,
        duration=100,
        config=config,
        rng=rng,
        fault_id="sp_1",
    )
    drift_ep = generate_offset_drift_fault(
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        total_steps=1920,
        start_step=800,
        config=config,
        rng=rng,
        fault_id="dr_1",
        ramp_steps=100,
        plateau_steps=50,
    )

    fault_series = SensorFaultSeries(
        sensor_id=dummy_sensor_params.sensor_id,
        total_steps=1920,
        episodes=[sp_ep, drift_ep],
    )

    # Unit scaler: Fahrenheit (multiplier=1.8, offset=32.0)
    fahrenheit_scaler = UnitScaler.fahrenheit()

    stream = sim.simulate_sensor(
        env_values=env_series.values,
        params=dummy_sensor_params,
        unit_scaler=fahrenheit_scaler,
        rng=rng,
        fault_series=fault_series,
    )

    # Canonical clean values must be in Celsius
    # Operational bounds [-10, 60] C strictly respected before unit scaling
    assert np.all(stream.canonical_clean_values >= -10.0)
    assert np.all(stream.canonical_clean_values <= 60.0)

    # Observable values must be in Fahrenheit (F = 1.8 * C + 32)
    valid_mask = ~stream.missing_mask
    expected_f = stream.canonical_values[valid_mask] * 1.8 + 32.0
    assert np.allclose(stream.values[valid_mask], expected_f)

    # Ground truth dataframe checks
    gt_df = stream.to_ground_truth_dataframe(scenario_id="sc_00", group_id="grp_00")
    assert len(gt_df) == 1920
    assert "true_value" in gt_df.columns
    assert "f_spike" in gt_df.columns
    assert "f_offset_drift" in gt_df.columns
    assert "sensor_untrusted" in gt_df.columns
    assert "fault_severity" in gt_df.columns
    assert "true_health" in gt_df.columns

    # Verify spike flag is active around step 600
    assert np.any(gt_df["f_spike"].values[600:700] == 1)
    # Verify drift flag is active around step 800
    assert np.any(gt_df["f_offset_drift"].values[800:950] == 1)


def test_gain_error_uses_true_physical_xref(config, dummy_sensor_params):
    # Verify Decision 5: x_ref is true physical onset value, NOT sensor reading with static offset
    sim = VirtualSensorSimulator(config=config)
    rng = np.random.default_rng(888)

    # Constant physical temperature at 25.0 C
    phys_vals = np.full(1920, 25.0, dtype=np.float64)
    # Give sensor a large static offset of +1.0 C
    params = TemperatureSensorParams(
        sensor_id="sensor_gain_test",
        static_offset=1.0,
        sigma_k=0.01,  # negligible noise
        response_lag_tau=0.0,
        quantisation_step=0.01,
        missing_rate=0.0,
        burst_max=3,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
    )

    # Gain g = 2.0 at start_step = 500
    # True physical x_ref at 500 is 25.0 C.
    # Unfaulted sensor reading would be 25.0 + 1.0 = 26.0 C.
    # With reading = x_ref + g * (x - x_ref):
    # reading = 25.0 + 2.0 * (26.0 - 25.0) = 25.0 + 2.0 * 1.0 = 27.0 C!
    # If the bug was present (using x_ref = with_offset = 26.0), reading would be 26.0 + 2.0 * (26.0 - 26.0) = 26.0 C.
    ep = FaultEpisode(
        fault_id="gain_xref_test",
        fault_type=FaultType.GAIN_ERROR,
        sensor_id=params.sensor_id,
        start_step=500,
        end_step=600,
        duration=100,
        shape_params={},
        additive_offset=np.zeros(1920),
        noise_multiplier=np.ones(1920),
        missing_mask=np.zeros(1920, dtype=bool),
        severity=np.ones(1920) * 0.5,
        label_mask=np.zeros(1920, dtype=bool),
        untrusted_mask=np.zeros(1920, dtype=bool),
        gain=2.0,
        gain_xref=25.0,
    )

    fs = SensorFaultSeries(sensor_id=params.sensor_id, total_steps=1920, episodes=[ep])
    stream = sim.simulate_sensor(
        env_values=phys_vals,
        params=params,
        unit_scaler=UnitScaler("celsius", 1.0, 0.0),
        rng=rng,
        noise=np.zeros(1920),
        fault_series=fs,
    )

    # In faulted interval [500, 600), value must be 27.0 C!
    assert np.allclose(stream.canonical_clean_values[500:600], 27.0)
    # Outside faulted interval, value is 26.0 C
    assert np.allclose(stream.canonical_clean_values[:500], 26.0)


# =============================================================================
# 10. Data Layer Isolation Tests
# =============================================================================

def test_data_layer_isolation(config):
    env_series = generate_environment_series(seed=12, config=config)
    rng = np.random.default_rng(34)
    sim = VirtualSensorSimulator(config=config)

    # Simulate group with a fault on sensor 0
    p = sim.profile.sample_sensor_params(num_sensors=4, rng=rng)
    fault_ep = generate_spike_fault(
        sensor_id=p[0].sensor_id,
        sigma_k=p[0].sigma_k,
        total_steps=1920,
        start_step=500,
        duration=100,
        config=config,
        rng=rng,
    )
    fs = SensorFaultSeries(sensor_id=p[0].sensor_id, total_steps=1920, episodes=[fault_ep])

    grp = sim.simulate_group(
        environment=env_series,
        sensor_params=p,
        rng=rng,
        fault_map={p[0].sensor_id: fs},
    )

    # Observable Layer 1 raw dataframe
    obs_df = grp.to_observable_dataframe(scenario_id="sc_iso")
    # Must contain strictly the 6 raw observable columns
    expected_cols = {"scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"}
    assert set(obs_df.columns) == expected_cols

    # Ground truth Layer 3 sensor dataframe
    gt_df = grp.to_ground_truth_sensors_dataframe(scenario_id="sc_iso")
    assert "true_value" in gt_df.columns
    assert "f_spike" in gt_df.columns
    assert "true_health" in gt_df.columns

    # Fault episodes dataframe
    ep_df = grp.fault_episodes_to_dataframe(scenario_id="sc_iso")
    assert len(ep_df) == 1
    assert ep_df.iloc[0]["fault_type"] == "spike"


# =============================================================================
# 11. Determinism Tests
# =============================================================================

def test_fault_determinism_identical_seed(config, dummy_sensor_params):
    f_sim = FaultSimulator(config=config)

    ep1 = f_sim.sample_fault(
        fault_type=FaultType.OFFSET_DRIFT,
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        rng=np.random.default_rng(999),
    )
    ep2 = f_sim.sample_fault(
        fault_type=FaultType.OFFSET_DRIFT,
        sensor_id=dummy_sensor_params.sensor_id,
        sigma_k=dummy_sensor_params.sigma_k,
        rng=np.random.default_rng(999),
    )

    assert ep1.start_step == ep2.start_step
    assert ep1.end_step == ep2.end_step
    assert ep1.shape_params["shape"] == ep2.shape_params["shape"]
    assert np.allclose(ep1.additive_offset, ep2.additive_offset)
    assert np.allclose(ep1.severity, ep2.severity)
