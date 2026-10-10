"""Focused unit and behavioral tests for virtual sensor simulation (P1.3).

Verifies:
  1. Sensor parameters loaded from configuration.
  2. Deterministic sensor outputs for identical seeds.
  3. Static offsets, Gaussian noise, first-order response lag, quantisation, and clipping.
  4. Unit scaling applied last, preserving canonical Celsius and ground truth separation.
  5. Missing-data rates and the exact 20% zero-missing sensor assignment.
  6. Maximum missing-burst length of 3 readings.
  7. Representation of missing readings as NaN (never substituted with zero).
  8. Mildly correlated noise factor model behavior.
  9. Strict separation of Layer 1 observable readings, canonical physical values, and environmental truth.
"""

import math
from typing import List
import numpy as np
import pytest

from src.sim.environment import EnvironmentSimulator, generate_environment_series
from src.sim.profile import TemperatureProfile, TemperatureSensorParams
from src.sim.sensors import (
    SensorStream,
    VirtualSensorGroup,
    VirtualSensorSimulator,
    apply_clipping,
    apply_quantisation,
    apply_response_lag,
    generate_group_noise,
    generate_missing_mask,
    simulate_sensors,
)
from src.sim.units import UnitScaler


# ==============================================================================
# 1. Parameter Loading and Configuration Tests
# ==============================================================================

def test_sensor_params_loaded_from_config():
    """Verify that virtual sensor simulation parameters are correctly loaded from YAML."""
    simulator = VirtualSensorSimulator()
    assert simulator.sampling_rate_minutes == 1.0
    assert simulator.total_steps == 1920
    assert simulator.warmup_steps == 480
    assert simulator.scored_steps == 1440
    assert simulator.sensor_clip_min == -10.0
    assert simulator.sensor_clip_max == 60.0

    profile = simulator.profile
    assert profile.sigma_k_min == 0.05
    assert profile.sigma_k_max == 0.30
    assert profile.static_offset_min == -1.0
    assert profile.static_offset_max == 1.0
    assert profile.response_lag_time_constant_min == 0.0
    assert profile.response_lag_time_constant_max == 6.0
    assert profile.response_lag_zero_fraction == 0.40
    assert profile.quantisation_values == [0.01, 0.05, 0.1, 0.25]
    assert profile.quantisation_probabilities == [0.25, 0.25, 0.35, 0.15]
    assert profile.missing_rate_min == 0.0
    assert profile.missing_rate_max == 0.01
    assert profile.zero_missing_sensor_fraction == 0.20
    assert profile.burst_max == 3
    assert profile.missing_rate_distribution == "uniform"


# ==============================================================================
# 2. Determinism Tests
# ==============================================================================

def test_sensor_determinism_identical_seed():
    """Verify that simulating virtual sensors with the same seed produces bit-for-bit identical results."""
    env = generate_environment_series(seed=101)

    group1 = simulate_sensors(environment=env, num_sensors=5, seed=202)
    group2 = simulate_sensors(environment=env, num_sensors=5, seed=202)

    assert group1.sensor_ids == group2.sensor_ids
    assert len(group1.sensors) == 5

    for s_id in group1.sensor_ids:
        s1 = group1.sensors[s_id]
        s2 = group2.sensors[s_id]

        # Sensor parameters match
        assert s1.params == s2.params

        # Missing masks match bit-for-bit
        np.testing.assert_array_equal(s1.missing_mask, s2.missing_mask)

        # Observable values match bit-for-bit, including NaNs
        np.testing.assert_allclose(s1.values, s2.values, equal_nan=True)

        # Canonical values match bit-for-bit, including NaNs
        np.testing.assert_allclose(s1.canonical_values, s2.canonical_values, equal_nan=True)

        # Clean canonical values match exactly
        np.testing.assert_array_equal(s1.canonical_clean_values, s2.canonical_clean_values)


def test_sensor_determinism_different_seed():
    """Verify that simulating virtual sensors with different seeds produces distinct outputs."""
    env = generate_environment_series(seed=101)

    group1 = simulate_sensors(environment=env, num_sensors=5, seed=202)
    group2 = simulate_sensors(environment=env, num_sensors=5, seed=999)

    # Clean values across sensors should differ due to different sampled noise and offsets
    s1_vals = group1.sensors["sensor_00"].canonical_clean_values
    s2_vals = group2.sensors["sensor_00"].canonical_clean_values
    assert not np.array_equal(s1_vals, s2_vals)


# ==============================================================================
# 3. Physical Transformations: Lag, Offset, Noise, Clipping, Quantisation
# ==============================================================================

def test_response_lag_filter():
    """Verify discrete first-order lag filter behavior (Decision 17)."""
    # 1. Zero lag (tau = 0.0): output must match input identically
    u = np.array([20.0, 22.0, 25.0, 18.0, 20.0])
    y_zero = apply_response_lag(u, tau=0.0, sampling_rate_minutes=1.0)
    np.testing.assert_array_equal(y_zero, u)

    # 2. Step response with tau = 5.0 min
    # Continuous step response: y(t) = 1 - exp(-t / tau)
    step_input = np.ones(50)
    tau = 5.0
    dt = 1.0
    y_step = apply_response_lag(step_input, tau=tau, sampling_rate_minutes=dt)

    # Equilibrium start: y[0] = step_input[0] = 1.0
    assert y_step[0] == 1.0

    # Step transition test from 0 to 10
    step_jump = np.zeros(60)
    step_jump[10:] = 10.0
    y_jump = apply_response_lag(step_jump, tau=5.0, sampling_rate_minutes=1.0)

    # Prior to jump, output is 0.0
    np.testing.assert_allclose(y_jump[:10], 0.0)

    # At step 10, jump occurs: y[10] = beta * 0 + (1 - beta) * 10
    beta = math.exp(-1.0 / 5.0)
    expected_10 = (1.0 - beta) * 10.0
    assert pytest.approx(y_jump[10], rel=1e-5) == expected_10

    # At t = 10 + tau = 15: y should reach ~63.2% of step
    assert 0.60 * 10.0 <= y_jump[15] <= 0.70 * 10.0

    # Eventually reaches 10.0
    assert pytest.approx(y_jump[-1], rel=1e-3) == 10.0


def test_static_sensor_offset():
    """Verify that static offset is uniformly and consistently applied across readings."""
    steps = 100
    env_const = np.full(steps, 25.0)

    # Sensor with static_offset = 0.75, tau = 0, sigma_k = 0, fine quantisation, no missing data
    params = TemperatureSensorParams(
        sensor_id="test_offset",
        sigma_k=0.0,
        static_offset=0.75,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.0,
        burst_max=3,
    )
    simulator = VirtualSensorSimulator()
    rng = np.random.default_rng(123)

    stream = simulator.simulate_sensor(
        env_values=env_const,
        params=params,
        unit_scaler=UnitScaler.celsius(),
        rng=rng,
        noise=np.zeros(steps),
    )

    expected = 25.0 + 0.75
    np.testing.assert_allclose(stream.canonical_clean_values, expected, atol=1e-5)
    np.testing.assert_allclose(stream.canonical_values, expected, atol=1e-5)


def test_gaussian_measurement_noise_statistics():
    """Verify that Gaussian noise follows configured sigma_k with zero mean."""
    steps = 20000
    sigma_k = 0.20
    rng = np.random.default_rng(42)

    params = TemperatureSensorParams(
        sensor_id="test_noise",
        sigma_k=sigma_k,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.0,
        burst_max=3,
    )
    simulator = VirtualSensorSimulator()
    env_zero = np.zeros(steps)

    stream = simulator.simulate_sensor(
        env_values=env_zero,
        params=params,
        unit_scaler=UnitScaler.celsius(),
        rng=rng,
    )

    clean = stream.canonical_clean_values
    sample_mean = float(np.mean(clean))
    sample_std = float(np.std(clean))

    # Mean within 3 * SE(mean) = 3 * 0.20 / sqrt(20000) ~ 0.0042
    assert abs(sample_mean) < 0.015, f"Sample mean {sample_mean} too far from 0.0"
    # Std within 4% of 0.20
    assert 0.190 <= sample_std <= 0.210, f"Sample std {sample_std} too far from {sigma_k}"


def test_correlated_noise_factor_model():
    """Verify that 1-factor model produces the configured mild correlation (Decision 16)."""
    steps = 30000
    rng = np.random.default_rng(101)

    # Create two correlated sensors with rho = 0.36 (expected corr = sqrt(0.36*0.36) = 0.36)
    # and one uncorrelated sensor
    s1 = TemperatureSensorParams(
        sensor_id="corr_01",
        sigma_k=0.20,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=True,
        noise_correlation_rho=0.36,
        missing_rate=0.0,
    )
    s2 = TemperatureSensorParams(
        sensor_id="corr_02",
        sigma_k=0.15,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=True,
        noise_correlation_rho=0.36,
        missing_rate=0.0,
    )
    s3 = TemperatureSensorParams(
        sensor_id="uncorr_03",
        sigma_k=0.25,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.0,
    )

    noise_map = generate_group_noise(steps=steps, sensor_params=[s1, s2, s3], rng=rng)

    n1 = noise_map["corr_01"]
    n2 = noise_map["corr_02"]
    n3 = noise_map["uncorr_03"]

    # Verify marginal standard deviations match sigma_k
    assert pytest.approx(np.std(n1), rel=0.05) == 0.20
    assert pytest.approx(np.std(n2), rel=0.05) == 0.15
    assert pytest.approx(np.std(n3), rel=0.05) == 0.25

    # Pairwise correlation between s1 and s2 should be close to 0.36 (in [0.2, 0.5])
    corr_12 = float(np.corrcoef(n1, n2)[0, 1])
    assert 0.31 <= corr_12 <= 0.41, f"Correlation between s1 and s2: {corr_12} expected ~0.36"

    # Pairwise correlation between s1 and uncorrelated s3 should be ~0.0
    corr_13 = float(np.corrcoef(n1, n3)[0, 1])
    assert abs(corr_13) < 0.03, f"Correlation between s1 and s3: {corr_13} expected ~0.0"


def test_sensor_bounds_clipping():
    """Verify that sensor readings are clipped strictly to [-10.0, 60.0] C."""
    # Extreme environmental excursions
    extreme_env = np.array([-30.0, -15.0, -10.0, 0.0, 25.0, 50.0, 60.0, 75.0, 100.0])
    simulator = VirtualSensorSimulator()
    rng = np.random.default_rng(123)

    params = TemperatureSensorParams(
        sensor_id="test_clip",
        sigma_k=0.05,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.0,
    )

    stream = simulator.simulate_sensor(
        env_values=extreme_env,
        params=params,
        unit_scaler=UnitScaler.celsius(),
        rng=rng,
        noise=np.zeros(len(extreme_env)),
    )

    # All canonical values must lie within [-10.0, 60.0]
    assert np.all(stream.canonical_clean_values >= -10.0)
    assert np.all(stream.canonical_clean_values <= 60.0)
    assert stream.canonical_clean_values[0] == -10.0
    assert stream.canonical_clean_values[-1] == 60.0


def test_sensor_quantisation_resolutions():
    """Verify that quantisation rounds to exact configured multiples for all allowed steps."""
    test_values = np.linspace(-5.0, 45.0, 500)
    valid_steps = [0.01, 0.05, 0.1, 0.25]

    for q in valid_steps:
        quantized = apply_quantisation(test_values, quantisation_step=q)

        # Every value divided by q must be an integer (within floating tolerance)
        multiples = quantized / q
        np.testing.assert_allclose(multiples, np.round(multiples), atol=1e-5)


# ==============================================================================
# 4. Healthy Missing Data Tests (Decision 14 & Decision 15)
# ==============================================================================

def test_healthy_missing_data_zero_rate_sensor_assignment():
    """Verify independent Bernoulli assignment with population proportion ~20% (Decision 14 corrected).
    
    Each sensor independently receives missing_rate == 0.0 with probability 0.20.
    In small groups (4-8 sensors), counts vary naturally. Across a sufficiently large sample,
    the empirical proportion converges to 20%.
    """
    profile = TemperatureProfile()

    # 1. Deterministic assignment for identical seeds
    batch_a1 = profile.sample_sensor_params(num_sensors=6, seed=777)
    batch_a2 = profile.sample_sensor_params(num_sensors=6, seed=777)
    rates_a1 = [s.missing_rate for s in batch_a1]
    rates_a2 = [s.missing_rate for s in batch_a2]
    assert rates_a1 == rates_a2

    # 2. Population-level proportion across a sufficiently large, fixed-seed sample (5,000 sensors)
    rng = np.random.default_rng(999)
    total_sensors = 5000
    all_sensors = profile.sample_sensor_params(num_sensors=total_sensors, rng=rng)

    zero_count = sum(1 for s in all_sensors if s.missing_rate == 0.0)
    zero_fraction = zero_count / total_sensors

    # Expected 0.20, SE = sqrt(0.20 * 0.80 / 5000) ~ 0.00565; 3.5-sigma window is [0.18, 0.22]
    assert 0.18 <= zero_fraction <= 0.22, f"Empirical zero-missing fraction {zero_fraction} outside [0.18, 0.22]"

    # 3. Across realistic group sizes (4 to 8 sensors), verify counts are not artificially locked
    group_sizes = [4, 5, 6, 7, 8]
    group_zero_counts = set()
    for n in group_sizes:
        batch = profile.sample_sensor_params(num_sensors=n, rng=rng)
        group_zero_counts.add(sum(1 for s in batch if s.missing_rate == 0.0))
    # Confirms variability across independent trials (not a single rigid integer)
    assert len(group_zero_counts) > 1


def test_healthy_missing_data_rates_distribution():
    """Verify that non-zero missing rates lie within Uniform[0.0, 0.01]."""
    profile = TemperatureProfile()
    rng = np.random.default_rng(555)

    batch = profile.sample_sensor_params(num_sensors=2000, rng=rng)
    non_zero_rates = [s.missing_rate for s in batch if s.missing_rate > 0.0]

    # Non-zero rate count ~ 80% of 2000 = 1600 (+/- 70)
    assert 1500 <= len(non_zero_rates) <= 1700

    for r in non_zero_rates:
        assert 0.0 < r <= 0.01

    # Mean of Uniform[0, 0.01] is 0.005
    mean_r = float(np.mean(non_zero_rates))
    assert 0.0045 <= mean_r <= 0.0055


def test_missing_data_burst_max_length():
    """Verify that contiguous missing runs NEVER exceed burst_max = 3 (Decision 14, 15)."""
    rng = np.random.default_rng(777)
    steps = 1920

    # Test across multiple missing rates from 0.001 to 0.01
    for rate in [0.001, 0.003, 0.005, 0.008, 0.01]:
        for _ in range(10):
            mask = generate_missing_mask(steps=steps, missing_rate=rate, burst_max=3, rng=rng)

            # Find lengths of all contiguous True runs
            if not np.any(mask):
                continue

            diffs = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            run_lengths = ends - starts

            assert np.all(run_lengths <= 3), f"Found burst of length > 3: {run_lengths[run_lengths > 3]}"
            assert np.all(run_lengths >= 1)


def test_missing_data_represented_as_nan_never_zero():
    """Verify that missing readings are represented as NaN and never substituted with 0.0."""
    env = generate_environment_series(seed=42)
    # Ensure env values are strictly non-zero positive temperatures
    assert np.all(env.values > 5.0)

    # Force a sensor with non-zero missing rate
    params = TemperatureSensorParams(
        sensor_id="test_nan",
        sigma_k=0.10,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.01,
        burst_max=3,
    )
    simulator = VirtualSensorSimulator()
    rng = np.random.default_rng(1234)

    stream = simulator.simulate_sensor(
        env_values=env.values,
        params=params,
        unit_scaler=UnitScaler.celsius(),
        rng=rng,
    )

    # Missing readings must be NaN
    missing_indices = np.where(stream.missing_mask)[0]
    assert len(missing_indices) > 0, "Expected some missing readings at 1% missing rate"

    for idx in missing_indices:
        assert np.isnan(stream.values[idx]), f"Expected NaN at missing step {idx}, got {stream.values[idx]}"
        assert np.isnan(stream.canonical_values[idx]), f"Expected NaN at missing step {idx}"
        # Crucial check: value must NOT be 0.0
        assert stream.values[idx] != 0.0

    # Non-missing readings must NOT be NaN
    valid_indices = np.where(~stream.missing_mask)[0]
    for idx in valid_indices:
        assert not np.isnan(stream.values[idx])
        assert not np.isnan(stream.canonical_values[idx])


def test_zero_missing_rate_sensor_has_no_nans():
    """Verify that a sensor with missing_rate == 0.0 has zero missing readings across the entire series."""
    env = generate_environment_series(seed=42)
    params = TemperatureSensorParams(
        sensor_id="zero_missing",
        sigma_k=0.10,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.0,
        burst_max=3,
    )
    simulator = VirtualSensorSimulator()
    rng = np.random.default_rng(54321)

    stream = simulator.simulate_sensor(
        env_values=env.values,
        params=params,
        unit_scaler=UnitScaler.celsius(),
        rng=rng,
    )

    assert stream.missing_count == 0
    assert stream.actual_missing_rate == 0.0
    assert not np.any(stream.missing_mask)
    assert not np.any(np.isnan(stream.values))
    assert not np.any(np.isnan(stream.canonical_values))


# ==============================================================================
# 5. Unit Scaling Applied Last and Layer Isolation Tests
# ==============================================================================

def test_unit_scaling_applied_last_fahrenheit():
    """Verify that Fahrenheit scaling is applied last and preserves canonical Celsius separation."""
    env = generate_environment_series(seed=88)
    fahrenheit_scaler = UnitScaler.fahrenheit()

    params = TemperatureSensorParams(
        sensor_id="f_sensor",
        sigma_k=0.10,
        static_offset=0.5,
        response_lag_tau=2.0,
        quantisation_step=0.05,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.005,
        burst_max=3,
    )
    simulator = VirtualSensorSimulator()
    rng = np.random.default_rng(777)

    stream = simulator.simulate_sensor(
        env_values=env.values,
        params=params,
        unit_scaler=fahrenheit_scaler,
        rng=rng,
    )

    # 1. Environmental truth is strictly canonical Celsius
    assert stream.environmental_values.shape == (1920,)
    assert -5.0 <= np.min(stream.environmental_values) <= np.max(stream.environmental_values) <= 50.0

    # 2. Canonical values are in Celsius [-10, 60] (where not NaN)
    valid_canon = stream.canonical_values[~stream.missing_mask]
    assert -10.0 <= np.min(valid_canon) <= np.max(valid_canon) <= 60.0

    # 3. Observable values are in Fahrenheit [14, 140] (where not NaN)
    valid_obs = stream.values[~stream.missing_mask]
    assert 14.0 <= np.min(valid_obs) <= np.max(valid_obs) <= 140.0

    # 4. Inverting observable values recovers canonical Celsius exactly
    inverted = fahrenheit_scaler.invert(stream.values)
    np.testing.assert_allclose(inverted, stream.canonical_values, equal_nan=True)


def test_unit_scaling_applied_last_affine():
    """Verify random affine scaling is applied last and preserves canonical Celsius separation."""
    env = generate_environment_series(seed=99)
    affine_scaler = UnitScaler.affine(multiplier=2.5, offset=-8.0)

    params = TemperatureSensorParams(
        sensor_id="aff_sensor",
        sigma_k=0.15,
        static_offset=-0.3,
        response_lag_tau=1.5,
        quantisation_step=0.1,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        missing_rate=0.008,
        burst_max=3,
    )
    simulator = VirtualSensorSimulator()
    rng = np.random.default_rng(888)

    stream = simulator.simulate_sensor(
        env_values=env.values,
        params=params,
        unit_scaler=affine_scaler,
        rng=rng,
    )

    # Inverting observable values recovers canonical Celsius exactly
    inverted = affine_scaler.invert(stream.values)
    np.testing.assert_allclose(inverted, stream.canonical_values, equal_nan=True)


def test_group_simulation_and_dataframe_schema():
    """Verify VirtualSensorGroup simulation and Layer 1 observable DataFrame export."""
    env = generate_environment_series(seed=12)
    group = simulate_sensors(environment=env, num_sensors=6, group_id="room_A", seed=34)

    assert group.group_id == "room_A"
    assert group.num_sensors == 6
    assert len(group.sensor_ids) == 6
    assert group.total_steps == 1920

    # Verify observable values matrix shape
    mat = group.values_matrix
    assert mat.shape == (6, 1920)

    # Verify canonical values matrix shape
    c_mat = group.canonical_matrix
    assert c_mat.shape == (6, 1920)

    # Convert to Layer 1 observable DataFrame
    df = group.to_observable_dataframe(scenario_id="sc_001")

    # Layer 1 schema check: ONLY observable columns present, NO ground truth columns!
    expected_cols = ["scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"]
    assert list(df.columns) == expected_cols

    # Length check: 6 sensors * 1920 steps = 11,520 rows
    assert len(df) == 6 * 1920
    assert df["scenario_id"].iloc[0] == "sc_001"
    assert df["group_id"].iloc[0] == "room_A"
    assert df["variable_type"].iloc[0] == "temperature"

    # Missing values represented as NaN in DataFrame
    assert df["value"].isna().sum() >= 0


def test_empty_and_edge_inputs():
    """Verify graceful handling of edge inputs in helper functions."""
    assert len(apply_response_lag(np.empty(0), tau=5.0)) == 0
    assert len(generate_missing_mask(steps=0, missing_rate=0.01, burst_max=3, rng=np.random.default_rng())) == 0

    # Negative tau treated as zero lag
    u = np.array([10.0, 20.0])
    np.testing.assert_array_equal(apply_response_lag(u, tau=-1.0), u)

    # Quantisation validation error
    with pytest.raises(ValueError, match="quantisation_step must be positive"):
        apply_quantisation(u, quantisation_step=-0.05)
