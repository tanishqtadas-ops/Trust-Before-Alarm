"""Unit tests for Temperature Profile and Unit Scaling (Task P1.1).

Validates:
1. UnitScaler transforms (Celsius, Fahrenheit, random affine), inverse mapping,
   sigma scaling, array handling, and serialization.
2. Configuration-driven parameter loading from config/generator_a.yaml.
3. Determinism: identical seeds yield bit-for-bit identical parameters.
4. Statistical distribution checks: sampled values respect all Appendix A.2 bounds.
5. Unit scaling applied last: canonical physical truth is preserved and convertible.
"""

import math
from typing import List
import numpy as np
import pytest

from src.sim.profile import (
    TemperatureProfile,
    TemperatureScenarioParams,
    TemperatureSensorParams,
)
from src.sim.units import UnitScaler, sample_unit_scaler
from src.utils.config import get_config_dir, load_yaml


# ==============================================================================
# UnitScaler Tests
# ==============================================================================

def test_unit_scaler_celsius():
    """Verify Celsius identity scaler."""
    scaler = UnitScaler.celsius()
    assert scaler.scale_type == "celsius"
    assert scaler.multiplier == 1.0
    assert scaler.offset == 0.0

    # Scalar application & inversion
    assert scaler.apply(25.0) == 25.0
    assert scaler.invert(25.0) == 25.0
    assert scaler.scale_std(0.2) == 0.2
    assert scaler.invert_std(0.2) == 0.2

    # Array application
    arr = np.array([-5.0, 0.0, 21.5, 50.0])
    np.testing.assert_allclose(scaler.apply(arr), arr)
    np.testing.assert_allclose(scaler.invert(arr), arr)


def test_unit_scaler_fahrenheit():
    """Verify standard Fahrenheit conversion: F = 1.8 * C + 32.0."""
    scaler = UnitScaler.fahrenheit()
    assert scaler.scale_type == "fahrenheit"
    assert scaler.multiplier == 1.8
    assert scaler.offset == 32.0

    # Exact physical reference points
    assert pytest.approx(scaler.apply(0.0)) == 32.0
    assert pytest.approx(scaler.apply(100.0)) == 212.0
    assert pytest.approx(scaler.apply(-40.0)) == -40.0
    assert pytest.approx(scaler.apply(37.0)) == 98.6

    # Inversion
    assert pytest.approx(scaler.invert(32.0)) == 0.0
    assert pytest.approx(scaler.invert(212.0)) == 100.0
    assert pytest.approx(scaler.invert(98.6)) == 37.0

    # Sigma scaling (offset must have 0 effect on standard deviation)
    sigma_c = 0.15
    sigma_f = scaler.scale_std(sigma_c)
    assert pytest.approx(sigma_f) == 0.15 * 1.8
    assert pytest.approx(scaler.invert_std(sigma_f)) == sigma_c

    # Bounds scaling
    scaled_bounds = scaler.scale_bounds((-5.0, 50.0))
    assert pytest.approx(scaled_bounds[0]) == 1.8 * (-5.0) + 32.0  # 23.0 F
    assert pytest.approx(scaled_bounds[1]) == 1.8 * 50.0 + 32.0   # 122.0 F


def test_unit_scaler_affine():
    """Verify arbitrary affine scaler: y = m * x + c."""
    m = 2.5
    c = -7.5
    scaler = UnitScaler.affine(multiplier=m, offset=c)
    assert scaler.scale_type == "affine"
    assert scaler.multiplier == 2.5
    assert scaler.offset == -7.5

    x = 10.0
    expected_y = 2.5 * 10.0 - 7.5  # 17.5
    assert pytest.approx(scaler.apply(x)) == expected_y
    assert pytest.approx(scaler.invert(expected_y)) == x

    # Standard deviation scale
    sigma = 0.5
    assert pytest.approx(scaler.scale_std(sigma)) == 2.5 * 0.5
    assert pytest.approx(scaler.invert_std(2.5 * 0.5)) == sigma

    # NaN preservation
    nan_arr = np.array([1.0, np.nan, 3.0])
    scaled_nan = scaler.apply(nan_arr)
    assert np.isnan(scaled_nan[1])
    assert pytest.approx(scaled_nan[0]) == 2.5 * 1.0 - 7.5


def test_unit_scaler_serialization():
    """Verify serialization to and from dictionary."""
    scaler = UnitScaler(scale_type="affine", multiplier=1.75, offset=4.2)
    d = scaler.to_dict()
    assert d == {"scale_type": "affine", "multiplier": 1.75, "offset": 4.2}

    restored = UnitScaler.from_dict(d)
    assert restored == scaler


def test_unit_scaler_invalid():
    """Verify validation checks on invalid inputs."""
    with pytest.raises(ValueError, match="strictly positive"):
        UnitScaler(scale_type="celsius", multiplier=0.0, offset=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        UnitScaler(scale_type="celsius", multiplier=-1.0, offset=0.0)

    with pytest.raises(ValueError, match="Unknown scale_type"):
        UnitScaler(scale_type="kelvin", multiplier=1.0, offset=273.15)


def test_sample_unit_scaler_sampling():
    """Verify sample_unit_scaler produces expected probability distribution and ranges."""
    cfg = load_yaml(get_config_dir() / "generator_a.yaml")
    unit_cfg = cfg["temperature_profile"]["unit_scaling"]

    n_samples = 10000
    rng = np.random.default_rng(12345)

    scalers = [sample_unit_scaler(unit_cfg, rng=rng) for _ in range(n_samples)]
    types = [s.scale_type for s in scalers]

    c_frac = types.count("celsius") / n_samples
    f_frac = types.count("fahrenheit") / n_samples
    a_frac = types.count("affine") / n_samples

    # Check nominal probabilities (70% C, 15% F, 15% Affine) within +/- 1.5%
    assert 0.68 <= c_frac <= 0.72, f"Celsius fraction {c_frac} outside expected range"
    assert 0.135 <= f_frac <= 0.165, f"Fahrenheit fraction {f_frac} outside expected range"
    assert 0.135 <= a_frac <= 0.165, f"Affine fraction {a_frac} outside expected range"

    # Verify all affine multipliers and offsets lie strictly in configured bounds
    for s in scalers:
        if s.scale_type == "affine":
            assert unit_cfg["affine_multiplier_min"] <= s.multiplier <= unit_cfg["affine_multiplier_max"]
            assert unit_cfg["affine_offset_min"] <= s.offset <= unit_cfg["affine_offset_max"]


# ==============================================================================
# TemperatureProfile Parameter & Config Tests
# ==============================================================================

def test_temperature_profile_init_matches_config():
    """Verify TemperatureProfile loads all Appendix A.2 parameters correctly from config."""
    profile = TemperatureProfile()

    # Base level (A.2: uniform 2 to 38 C)
    assert profile.base_level_min == 2.0
    assert profile.base_level_max == 38.0

    # Daily cycle (A.2: amplitude 0.5 to 4 C, 20% zero, 24h)
    assert profile.daily_cycle_amplitude_min == 0.5
    assert profile.daily_cycle_amplitude_max == 4.0
    assert profile.daily_cycle_zero_amplitude_prob == 0.20
    assert profile.daily_cycle_period_minutes == 1440.0

    # Slow wander (A.2: std 0.2 to 0.8 C, tau 120 to 360 min, 5-step MA)
    assert profile.slow_wander_std_min == 0.2
    assert profile.slow_wander_std_max == 0.8
    assert profile.slow_wander_correlation_time_min == 120.0
    assert profile.slow_wander_correlation_time_max == 360.0
    assert profile.slow_wander_moving_average_steps == 5

    # Sensor noise sigma_k (A.2: log-uniform 0.05 to 0.30 C, 20% correlated rho 0.2 to 0.5)
    assert profile.sigma_k_min == 0.05
    assert profile.sigma_k_max == 0.30
    assert profile.sensor_noise_correlated_fraction == 0.20
    assert profile.correlation_rho_min == 0.2
    assert profile.correlation_rho_max == 0.5

    # Static offset (A.2: uniform +/-1.0 C)
    assert profile.static_offset_min == -1.0
    assert profile.static_offset_max == 1.0

    # Response lag (A.2: tau 0 to 6 min, 40% zero)
    assert profile.response_lag_time_constant_min == 0.0
    assert profile.response_lag_time_constant_max == 6.0
    assert profile.response_lag_zero_fraction == 0.40

    # Quantisation (A.2: 0.01: 25%, 0.05: 25%, 0.1: 35%, 0.25: 15%)
    assert profile.quantisation_values == [0.01, 0.05, 0.1, 0.25]
    assert profile.quantisation_probabilities == [0.25, 0.25, 0.35, 0.15]

    # Bounds (A.2: env [-5, 50], sensor [-10, 60])
    assert profile.environment_clip_min == -5.0
    assert profile.environment_clip_max == 50.0
    assert profile.sensor_clip_min == -10.0
    assert profile.sensor_clip_max == 60.0


def test_temperature_profile_determinism():
    """Verify that identical seeds produce bit-for-bit identical outputs."""
    profile = TemperatureProfile()

    # Scenario parameters determinism
    p1 = profile.sample_scenario_params(seed=42)
    p2 = profile.sample_scenario_params(seed=42)
    assert p1 == p2
    assert p1.to_dict() == p2.to_dict()

    # Sensor parameters determinism
    s1 = profile.sample_sensor_params(num_sensors=6, rng=42)
    s2 = profile.sample_sensor_params(num_sensors=6, rng=42)
    assert s1 == s2
    for item1, item2 in zip(s1, s2):
        assert item1.to_dict() == item2.to_dict()

    # Different seeds yield different parameters
    p3 = profile.sample_scenario_params(seed=99)
    assert p1 != p3


def test_temperature_scenario_params_ranges():
    """Verify that sampled scenario parameters across multiple seeds respect all bounds."""
    profile = TemperatureProfile()
    n_scenarios = 1000

    zero_cycle_count = 0
    for seed in range(1, n_scenarios + 1):
        params = profile.sample_scenario_params(seed=seed)

        # Base level
        assert 2.0 <= params.base_level <= 38.0

        # Daily cycle
        assert 0.0 <= params.daily_cycle_phase < 2.0 * math.pi
        assert params.daily_cycle_period_minutes == 1440.0
        if params.daily_cycle_has_cycle:
            assert 0.5 <= params.daily_cycle_amplitude <= 4.0
        else:
            assert params.daily_cycle_amplitude == 0.0
            zero_cycle_count += 1

        # Slow wander
        assert 0.2 <= params.slow_wander_std <= 0.8
        assert 120.0 <= params.slow_wander_correlation_time <= 360.0
        assert params.slow_wander_ma_steps == 5

        # Bounds
        assert params.environment_clip_min == -5.0
        assert params.environment_clip_max == 50.0
        assert params.sensor_clip_min == -10.0
        assert params.sensor_clip_max == 60.0

        # Unit scaler
        assert params.unit_scaler.scale_type in ("celsius", "fahrenheit", "affine")

    # Zero amplitude cycle fraction ~ 20% (+/- 3.5%)
    zero_cycle_frac = zero_cycle_count / n_scenarios
    assert 0.165 <= zero_cycle_frac <= 0.235, f"Zero cycle fraction: {zero_cycle_frac}"


def test_temperature_sensor_params_ranges():
    """Verify that sampled sensor parameters across multiple seeds respect all bounds."""
    profile = TemperatureProfile()
    n_sensors = 2000

    # Sample in batches of 5 sensors
    all_sensors: List[TemperatureSensorParams] = []
    rng = np.random.default_rng(777)
    for _ in range(n_sensors // 5):
        batch = profile.sample_sensor_params(num_sensors=5, rng=rng)
        all_sensors.extend(batch)

    zero_lag_count = 0
    correlated_count = 0
    valid_quant_steps = {0.01, 0.05, 0.1, 0.25}

    for s in all_sensors:
        # Sensor noise sigma_k in [0.05, 0.30] C
        assert 0.05 <= s.sigma_k <= 0.30

        # Static healthy offset in [-1.0, 1.0] C
        assert -1.0 <= s.static_offset <= 1.0

        # Response lag tau in [0.0, 6.0] min
        assert 0.0 <= s.response_lag_tau <= 6.0
        if s.response_lag_tau == 0.0:
            zero_lag_count += 1

        # Quantisation step
        assert s.quantisation_step in valid_quant_steps

        # Noise correlation
        if s.is_noise_correlated:
            assert 0.2 <= s.noise_correlation_rho <= 0.5
            correlated_count += 1
        else:
            assert s.noise_correlation_rho == 0.0

    # Zero lag fraction ~ 40% (+/- 3.5%)
    zero_lag_frac = zero_lag_count / n_sensors
    assert 0.365 <= zero_lag_frac <= 0.435, f"Zero lag fraction: {zero_lag_frac}"

    # Correlated fraction ~ 20% (+/- 3.5%)
    corr_frac = correlated_count / n_sensors
    assert 0.165 <= corr_frac <= 0.235, f"Correlated fraction: {corr_frac}"


# ==============================================================================
# Unit Scaling Applied Last & Ground Truth Isolation Tests
# ==============================================================================

def test_unit_scaling_applied_last_principle():
    """Verify that unit scaling operates strictly as a final transformation.
    
    Physical ground truth is defined in canonical Celsius units [-5.0, 50.0] C.
    Observable readings are transformed via unit_scaler.apply().
    Inverting observable values restores canonical Celsius ground truth.
    """
    profile = TemperatureProfile()
    rng = np.random.default_rng(999)

    # 1. Physical truth simulated in canonical Celsius
    canonical_truth = np.array([20.0, 22.4, 25.1, 18.7, 30.2, 5.0])

    # 2. Test across all three scaler types
    scalers = [
        UnitScaler.celsius(),
        UnitScaler.fahrenheit(),
        sample_unit_scaler(profile.unit_scaling_config, rng=rng),
    ]

    for scaler in scalers:
        # Observable values are generated by applying unit scaling last
        observable_readings = profile.apply_unit_scaling(canonical_truth, scaler)

        if scaler.scale_type == "celsius":
            np.testing.assert_allclose(observable_readings, canonical_truth)
        elif scaler.scale_type == "fahrenheit":
            np.testing.assert_allclose(observable_readings, canonical_truth * 1.8 + 32.0)
        else:
            np.testing.assert_allclose(
                observable_readings, canonical_truth * scaler.multiplier + scaler.offset
            )

        # Inversion perfectly recovers canonical physical truth
        recovered_truth = profile.invert_unit_scaling(observable_readings, scaler)
        np.testing.assert_allclose(recovered_truth, canonical_truth, rtol=1e-12)


def test_temperature_profile_custom_config():
    """Verify TemperatureProfile handles custom config overrides cleanly."""
    custom_cfg = {
        "base_level_min": 10.0,
        "base_level_max": 20.0,
        "daily_cycle": {
            "amplitude_min": 1.0,
            "amplitude_max": 2.0,
            "zero_amplitude_prob": 0.0,
            "period_minutes": 1440,
        },
        "slow_wander": {
            "std_min": 0.3,
            "std_max": 0.5,
            "correlation_time_min": 150,
            "correlation_time_max": 250,
            "moving_average_steps": 5,
        },
        "sensor_noise": {
            "sigma_k_min": 0.1,
            "sigma_k_max": 0.2,
            "correlated_sensors_fraction": 0.0,
            "correlation_rho_min": 0.2,
            "correlation_rho_max": 0.5,
        },
        "static_offset_min": -0.5,
        "static_offset_max": 0.5,
        "response_lag": {
            "time_constant_min": 1.0,
            "time_constant_max": 3.0,
            "zero_lag_fraction": 0.0,
        },
        "quantisation": {
            "values": [0.05, 0.1],
            "probabilities": [0.5, 0.5],
        },
        "bounds": {
            "environment_clip_min": 0.0,
            "environment_clip_max": 40.0,
            "sensor_clip_min": -5.0,
            "sensor_clip_max": 45.0,
        },
        "unit_scaling": {
            "celsius_prob": 1.0,
            "fahrenheit_prob": 0.0,
            "random_affine_prob": 0.0,
            "affine_multiplier_min": 0.5,
            "affine_multiplier_max": 3.0,
            "affine_offset_min": -10.0,
            "affine_offset_max": 10.0,
        },
    }

    profile = TemperatureProfile(custom_cfg)
    params = profile.sample_scenario_params(seed=1)
    assert 10.0 <= params.base_level <= 20.0
    assert params.daily_cycle_has_cycle is True
    assert params.unit_scaler.scale_type == "celsius"


def test_sample_unit_scaler_invalid_prob_sum():
    """Verify ValueError when unit scaling probabilities do not sum to 1."""
    bad_cfg = {
        "celsius_prob": 0.5,
        "fahrenheit_prob": 0.2,
        "random_affine_prob": 0.1,  # sum = 0.8 != 1.0
        "affine_multiplier_min": 0.5,
        "affine_multiplier_max": 3.0,
        "affine_offset_min": -10.0,
        "affine_offset_max": 10.0,
    }
    with pytest.raises(ValueError, match="must sum to 1.0"):
        sample_unit_scaler(bad_cfg)


def test_temperature_profile_validation_errors():
    """Verify validation errors on misconfigured profile dictionaries."""
    # base_level_min > base_level_max
    bad_base = {
        "base_level_min": 50.0,
        "base_level_max": 20.0,
    }
    with pytest.raises(ValueError, match="base_level_min"):
        TemperatureProfile(bad_base)

    # num_sensors < 1
    profile = TemperatureProfile()
    with pytest.raises(ValueError, match="num_sensors must be >= 1"):
        profile.sample_sensor_params(num_sensors=0)

    # sensor_ids length mismatch
    with pytest.raises(ValueError, match="Length of sensor_ids"):
        profile.sample_sensor_params(num_sensors=3, sensor_ids=["s1", "s2"])


def test_unit_scaler_sequence_and_types():
    """Verify UnitScaler handles Python lists and preserves data types."""
    scaler = UnitScaler.affine(multiplier=2.0, offset=5.0)
    lst = [1.0, 2.0, 3.0]
    scaled = scaler.apply(lst)
    assert isinstance(scaled, np.ndarray)
    np.testing.assert_allclose(scaled, [7.0, 9.0, 11.0])

    inverted = scaler.invert(scaled)
    np.testing.assert_allclose(inverted, lst)

