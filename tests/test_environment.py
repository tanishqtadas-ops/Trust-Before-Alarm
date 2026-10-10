"""Unit tests for True Environment Simulation (Task P1.2).

Validates:
1. Exact series lengths: 1,920 total steps (480 warm-up, 1,440 scored).
2. Determinism: identical seeds yield bit-for-bit identical environment series.
3. Daily cycle component: amplitude, period (1,440 min), zero-amplitude behavior, phase bounds.
4. Slow random wander: stationary OU AR(1) properties and 5-step causal moving average.
5. Environmental bounds enforcement: values strictly within [-5.0, 50.0] C.
6. Separation of true physical environment (canonical Celsius) from observable sensor data.
"""

import math
import numpy as np
import pytest

from src.sim.environment import (
    EnvironmentSeries,
    EnvironmentSimulator,
    generate_environment_series,
)
from src.sim.profile import TemperatureProfile, TemperatureScenarioParams
from src.sim.units import UnitScaler


def test_environment_series_lengths_and_structure():
    """Verify total, warm-up, and scored lengths match Appendix A.1 / config (1920, 480, 1440)."""
    series = generate_environment_series(seed=101)

    assert isinstance(series, EnvironmentSeries)
    assert series.total_steps == 1920
    assert series.warmup_steps == 480
    assert series.scored_steps == 1440

    # Array shapes
    assert series.values.shape == (1920,)
    assert series.unclipped_values.shape == (1920,)
    assert series.daily_cycle.shape == (1920,)
    assert series.slow_wander.shape == (1920,)
    assert series.timestamps_minutes.shape == (1920,)

    # Slices
    assert len(series.warmup_values) == 480
    assert len(series.scored_values) == 1440
    np.testing.assert_array_equal(series.warmup_values, series.values[:480])
    np.testing.assert_array_equal(series.scored_values, series.values[480:])

    # Timestamps (1 reading per minute)
    np.testing.assert_array_equal(series.timestamps_minutes, np.arange(1920, dtype=float))

    # Serialization
    summary = series.to_dict()
    assert summary["total_steps"] == 1920
    assert summary["warmup_steps"] == 480
    assert summary["scored_steps"] == 1440
    assert "min_value" in summary
    assert "max_value" in summary
    assert "clipped_steps_count" in summary


def test_environment_determinism():
    """Verify identical seeds produce bit-for-bit identical environmental series."""
    simulator = EnvironmentSimulator()

    s1 = simulator.generate(seed=42)
    s2 = simulator.generate(seed=42)

    assert s1.base_level == s2.base_level
    np.testing.assert_array_equal(s1.values, s2.values)
    np.testing.assert_array_equal(s1.unclipped_values, s2.unclipped_values)
    np.testing.assert_array_equal(s1.daily_cycle, s2.daily_cycle)
    np.testing.assert_array_equal(s1.slow_wander, s2.slow_wander)

    # Different seed yields different series
    s3 = simulator.generate(seed=99)
    assert not np.array_equal(s1.values, s3.values)


def test_daily_cycle_zero_amplitude():
    """Verify daily cycle is identically zero when amplitude is 0.0."""
    cycle = EnvironmentSimulator.generate_daily_cycle(
        steps=1920, amplitude=0.0, period_minutes=1440.0, phase=1.0
    )
    assert np.all(cycle == 0.0)
    assert len(cycle) == 1920


def test_daily_cycle_periodicity_and_amplitude():
    """Verify daily cycle amplitude bounds and 24-hour (1,440-step) exact periodicity."""
    amp = 2.5
    period = 1440.0
    phase = 0.5
    steps = 2880  # Two full 24-hour cycles

    cycle = EnvironmentSimulator.generate_daily_cycle(
        steps=steps, amplitude=amp, period_minutes=period, phase=phase
    )

    # Amplitude bounds: strictly [-amp, amp]
    assert np.max(cycle) <= amp + 1e-12
    assert np.min(cycle) >= -amp - 1e-12

    # Periodicity: cycle[t] == cycle[t + 1440]
    np.testing.assert_allclose(cycle[:1440], cycle[1440:2880], atol=1e-12)

    # Phase check at t=0: value = amp * sin(phase)
    assert pytest.approx(cycle[0]) == amp * math.sin(phase)


def test_slow_wander_properties():
    """Verify slow wander OU AR(1) process and 5-step causal moving average."""
    rng = np.random.default_rng(2026)
    target_std = 0.5
    tau = 240.0
    ma_steps = 5
    steps = 50000

    wander = EnvironmentSimulator.generate_slow_wander(
        steps=steps,
        std=target_std,
        correlation_time_min=tau,
        ma_steps=ma_steps,
        rng=rng,
    )

    assert len(wander) == steps
    # Mean of mean-reverting process should be close to 0
    assert pytest.approx(float(np.mean(wander)), abs=0.05) == 0.0

    # Empirical std: Because tau = 240 is large compared to ma_steps = 5,
    # the 5-step MA retains > 95% of total variance.
    empirical_std = float(np.std(wander))
    assert 0.40 <= empirical_std <= 0.60, f"Empirical std {empirical_std} outside expected range"


def test_slow_wander_causal_moving_average():
    """Verify moving average operation on a simple deterministic sequence."""
    rng = np.random.default_rng(123)
    # With std=0, wander must be identically zero
    zero_wander = EnvironmentSimulator.generate_slow_wander(
        steps=100, std=0.0, correlation_time_min=120.0, ma_steps=5, rng=rng
    )
    assert np.all(zero_wander == 0.0)


def test_environmental_bounds_clipping():
    """Verify readings are strictly constrained within [-5.0, 50.0] C."""
    # Run across 100 random scenarios
    for seed in range(1, 101):
        series = generate_environment_series(seed=seed)
        assert np.all(series.values >= -5.0), f"Seed {seed} dropped below -5.0 C"
        assert np.all(series.values <= 50.0), f"Seed {seed} exceeded 50.0 C"

    # Test forced extreme values exceeding bounds
    unclipped_high = np.array([48.0, 49.5, 50.5, 55.0, 60.0])
    clipped_high = EnvironmentSimulator.clip_environment(unclipped_high, -5.0, 50.0)
    np.testing.assert_array_equal(clipped_high, [48.0, 49.5, 50.0, 50.0, 50.0])

    unclipped_low = np.array([-10.0, -7.0, -5.0, -2.0, 0.0])
    clipped_low = EnvironmentSimulator.clip_environment(unclipped_low, -5.0, 50.0)
    np.testing.assert_array_equal(clipped_low, [-5.0, -5.0, -5.0, -2.0, 0.0])


def test_custom_scenario_params_injection():
    """Verify passing custom TemperatureScenarioParams directly into generate()."""
    custom_params = TemperatureScenarioParams(
        variable_type="temperature",
        base_level=30.0,
        daily_cycle_amplitude=2.0,
        daily_cycle_has_cycle=True,
        daily_cycle_phase=0.0,
        daily_cycle_period_minutes=1440.0,
        slow_wander_std=0.4,
        slow_wander_correlation_time=180.0,
        slow_wander_ma_steps=5,
        environment_clip_min=-5.0,
        environment_clip_max=50.0,
        sensor_clip_min=-10.0,
        sensor_clip_max=60.0,
        unit_scaler=UnitScaler.celsius(),
    )

    simulator = EnvironmentSimulator()
    series = simulator.generate(seed=777, params=custom_params)

    assert series.base_level == 30.0
    assert series.params == custom_params
    assert len(series.values) == 1920


def test_custom_structure_config():
    """Verify EnvironmentSimulator handles custom scenario lengths cleanly."""
    custom_cfg = {
        "structure": {
            "sampling_rate_minutes": 2,
            "total_steps": 600,
            "warmup_steps": 200,
            "scored_steps": 400,
        },
        "temperature_profile": {
            "base_level_min": 15.0,
            "base_level_max": 25.0,
            "daily_cycle": {
                "amplitude_min": 1.0,
                "amplitude_max": 2.0,
                "zero_amplitude_prob": 0.0,
                "period_minutes": 720,
                "phase_min_rad": 0.0,
                "phase_max_rad": 6.283185307179586,
            },
            "slow_wander": {
                "std_min": 0.3,
                "std_max": 0.5,
                "correlation_time_min": 120,
                "correlation_time_max": 240,
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
                "values": [0.05],
                "probabilities": [1.0],
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
        },
    }

    sim = EnvironmentSimulator(config=custom_cfg)
    series = sim.generate(seed=55)

    assert series.total_steps == 600
    assert series.warmup_steps == 200
    assert series.scored_steps == 400
    assert len(series.values) == 600
    assert len(series.warmup_values) == 200
    assert len(series.scored_values) == 400
    # Timestamps stepped by 2 minutes
    assert series.timestamps_minutes[1] - series.timestamps_minutes[0] == 2.0


def test_invalid_structure_lengths():
    """Verify ValueError when warmup_steps + scored_steps != total_steps."""
    bad_cfg = {
        "structure": {
            "sampling_rate_minutes": 1,
            "total_steps": 1000,
            "warmup_steps": 300,
            "scored_steps": 600,  # 300 + 600 = 900 != 1000
        },
        "temperature_profile": {
            "base_level_min": 15.0,
            "base_level_max": 25.0,
            "daily_cycle": {
                "amplitude_min": 1.0,
                "amplitude_max": 2.0,
                "zero_amplitude_prob": 0.0,
                "period_minutes": 1440,
                "phase_min_rad": 0.0,
                "phase_max_rad": 6.283185307179586,
            },
            "slow_wander": {
                "std_min": 0.3,
                "std_max": 0.5,
                "correlation_time_min": 120,
                "correlation_time_max": 240,
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
                "values": [0.05],
                "probabilities": [1.0],
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
        },
    }

    with pytest.raises(ValueError, match="warmup_steps"):
        EnvironmentSimulator(config=bad_cfg)


def test_empty_steps_helper_functions():
    """Verify helper functions return empty arrays when steps <= 0."""
    cycle = EnvironmentSimulator.generate_daily_cycle(0, 2.0, 1440, 0.0)
    assert len(cycle) == 0

    rng = np.random.default_rng(1)
    wander = EnvironmentSimulator.generate_slow_wander(0, 0.5, 120, 5, rng)
    assert len(wander) == 0

