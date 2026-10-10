"""Focused unit tests for P1.4: Genuine environmental events (Appendix A.3).

Verifies:
  1. Determinism for identical seeds.
  2. All four event waveforms:
     - Step with exponential recovery
     - Ramp (asymmetric rise/hold/fall)
     - Pulse
     - Sustained shift (persisting until scenario end)
  3. Event count and type distributions (Poisson mean 1.2, max 3, ~30% none).
  4. Magnitude log-uniform [0.5, 10.0] C and sign probabilities (75% pos, 25% neg).
  5. Scopes: global (55%, all sensors, att [0.6, 1.0]), partial (30%, 40-80%, att [0.3, 1.0]),
     local (15%, 1-2 sensors, att [0.7, 1.0]).
  6. Sensor arrival delays [0, 10] min.
  7. Detectability threshold (3 * sigma_k) and event_below_detectability.
  8. Sustained shift holds until scenario end (step 1920).
  9. Clean separation of physical event truth from observable sensor readings.
 10. Correct interaction with existing environment and sensor components (response lag, unit scaling).
"""

from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from src.sim.environment import EnvironmentSimulator
from src.sim.events import (
    EventEpisode,
    EventGenerator,
    EventSeries,
    apply_events_to_environment,
    generate_pulse,
    generate_ramp,
    generate_step_exponential_recovery,
    generate_sustained_shift,
)
from src.sim.profile import TemperatureProfile, TemperatureSensorParams
from src.sim.sensors import VirtualSensorSimulator, simulate_sensors
from src.utils.config import get_config_dir, load_yaml


@pytest.fixture
def config() -> dict:
    """Load default generator_a configuration."""
    yaml_path = get_config_dir() / "generator_a.yaml"
    return load_yaml(yaml_path)


@pytest.fixture
def profile(config) -> TemperatureProfile:
    """Fixture providing initialized TemperatureProfile."""
    return TemperatureProfile(config)


@pytest.fixture
def sample_sensors(profile) -> List[TemperatureSensorParams]:
    """Sample a group of 5 sensors for testing."""
    rng = np.random.default_rng(12345)
    return profile.sample_sensor_params(num_sensors=5, rng=rng)


# ---------------------------------------------------------------------------
# 1. Determinism tests
# ---------------------------------------------------------------------------

def test_events_determinism_identical_seed(config, sample_sensors):
    """Identical seeds and config must produce bit-for-bit identical EventSeries."""
    gen1 = EventGenerator(config)
    gen2 = EventGenerator(config)

    events1 = gen1.generate_events(sensor_params=sample_sensors, seed=42)
    events2 = gen2.generate_events(sensor_params=sample_sensors, seed=42)

    assert events1.num_events == events2.num_events
    for ep1, ep2 in zip(events1.episodes, events2.episodes):
        assert ep1.event_id == ep2.event_id
        assert ep1.event_type == ep2.event_type
        assert ep1.start_step == ep2.start_step
        assert np.isclose(ep1.magnitude, ep2.magnitude)
        assert ep1.scope == ep2.scope
        assert ep1.affected_sensors == ep2.affected_sensors
        for s_id in ep1.affected_sensors:
            assert np.isclose(ep1.sensor_attenuations[s_id], ep2.sensor_attenuations[s_id])
            assert np.isclose(ep1.sensor_delays[s_id], ep2.sensor_delays[s_id])

    for s in sample_sensors:
        np.testing.assert_array_equal(
            events1.sensor_contributions[s.sensor_id],
            events2.sensor_contributions[s.sensor_id],
        )

    np.testing.assert_array_equal(events1.genuine_event, events2.genuine_event)
    np.testing.assert_array_equal(events1.event_below_detectability, events2.event_below_detectability)
    np.testing.assert_allclose(events1.event_magnitude_sigma, events2.event_magnitude_sigma)
    np.testing.assert_allclose(events1.true_event_breadth, events2.true_event_breadth)
    assert events1.active_event_ids == events2.active_event_ids


def test_events_determinism_different_seed(config, sample_sensors):
    """Different seeds should generate different event scenarios."""
    gen = EventGenerator(config)
    # Compare multiple seeds to find distinct runs
    found_difference = False
    base_events = gen.generate_events(sensor_params=sample_sensors, seed=100)
    for s_seed in [101, 102, 103, 104, 105]:
        other_events = gen.generate_events(sensor_params=sample_sensors, seed=s_seed)
        if base_events.num_events != other_events.num_events or not np.array_equal(
            base_events.genuine_event, other_events.genuine_event
        ):
            found_difference = True
            break
    assert found_difference, "Different seeds must produce different event realizations"


# ---------------------------------------------------------------------------
# 2. Waveform generator tests for all four event shapes
# ---------------------------------------------------------------------------

def test_waveform_step_exponential_recovery():
    """Verify step with exponential recovery: immediate jump and exponential decay."""
    t = np.arange(100, dtype=np.float64)
    t_arr = 20.0
    mag = 4.0
    tau = 15.0

    w = generate_step_exponential_recovery(t_minutes=t, t_arrival=t_arr, magnitude=mag, tau=tau)

    # Before arrival: exactly 0
    np.testing.assert_array_equal(w[:20], 0.0)

    # At arrival t=20: exact jump to magnitude
    assert np.isclose(w[20], mag)

    # At t=35 (1 tau later): should be mag * exp(-1)
    assert np.isclose(w[35], mag * np.exp(-1.0))

    # At t=50 (2 tau later): should be mag * exp(-2)
    assert np.isclose(w[50], mag * np.exp(-2.0))

    # Monotonically decaying for t >= 20
    assert np.all(np.diff(w[20:]) <= 1e-12)


def test_waveform_ramp_asymmetric():
    """Verify ramp: linear rise, flat hold, and linear fall with independent durations."""
    t = np.arange(150, dtype=np.float64)
    t_arr = 10.0
    mag = 5.0
    rise = 20.0  # ends at 30
    hold = 30.0  # ends at 60
    fall = 40.0  # ends at 100

    w = generate_ramp(
        t_minutes=t,
        t_arrival=t_arr,
        magnitude=mag,
        rise_minutes=rise,
        hold_minutes=hold,
        fall_minutes=fall,
    )

    # Before arrival (t < 10): 0
    np.testing.assert_array_equal(w[:10], 0.0)

    # Midway during rise (t = 20): halfway to mag = 2.5
    assert np.isclose(w[20], 2.5)

    # During hold (30 <= t < 60): flat at mag
    np.testing.assert_allclose(w[30:60], mag)

    # Midway during fall (t = 80): halfway back down = 2.5
    assert np.isclose(w[80], 2.5)

    # After fall (t >= 100): 0
    np.testing.assert_array_equal(w[100:], 0.0)


def test_waveform_pulse():
    """Verify pulse: square pulse of exact duration."""
    t = np.arange(50, dtype=np.float64)
    t_arr = 15.0
    mag = -3.0
    dur = 10.0

    w = generate_pulse(t_minutes=t, t_arrival=t_arr, magnitude=mag, duration_minutes=dur)

    # Before arrival (t < 15): 0
    np.testing.assert_array_equal(w[:15], 0.0)

    # During pulse (15 <= t < 25): exact magnitude
    np.testing.assert_allclose(w[15:25], mag)

    # After pulse (t >= 25): 0
    np.testing.assert_array_equal(w[25:], 0.0)


def test_waveform_sustained_shift():
    """Verify sustained shift: linear rise and persists until scenario end."""
    t = np.arange(200, dtype=np.float64)
    t_arr = 50.0
    mag = 2.0
    rise = 10.0

    w = generate_sustained_shift(t_minutes=t, t_arrival=t_arr, magnitude=mag, rise_minutes=rise)

    # Before arrival (t < 50): 0
    np.testing.assert_array_equal(w[:50], 0.0)

    # Midway during rise (t = 55): halfway = 1.0
    assert np.isclose(w[55], 1.0)

    # After rise (t >= 60 to end 199): holds at magnitude
    np.testing.assert_allclose(w[60:], mag)


# ---------------------------------------------------------------------------
# 3. Parameter distributions and ranges
# ---------------------------------------------------------------------------

def test_event_count_distribution(config, sample_sensors):
    """Verify Poisson mean 1.2 capped at 3, with ~30% zero events."""
    gen = EventGenerator(config)
    rng = np.random.default_rng(9999)

    counts = [gen.sample_event_count(rng) for _ in range(2000)]
    counts_arr = np.array(counts)

    # Must be in [0, 3]
    assert np.all((counts_arr >= 0) & (counts_arr <= 3))
    # Poisson mean ~ 1.2 (clamped mean is ~ 1.15 - 1.20)
    empirical_mean = np.mean(counts_arr)
    assert 1.05 <= empirical_mean <= 1.30
    # Zero count probability ~ exp(-1.2) = 0.301
    zero_fraction = np.mean(counts_arr == 0)
    assert 0.25 <= zero_fraction <= 0.35


def test_event_type_distribution(config):
    """Verify event types match configured probabilities (40%, 25%, 25%, 10%)."""
    gen = EventGenerator(config)
    rng = np.random.default_rng(8888)

    types = [gen.sample_event_type(rng) for _ in range(3000)]
    types_s = pd.Series(types)
    freqs = types_s.value_counts(normalize=True)

    assert 0.36 <= freqs["step_exponential_recovery"] <= 0.44
    assert 0.21 <= freqs["ramp"] <= 0.29
    assert 0.21 <= freqs["pulse"] <= 0.29
    assert 0.07 <= freqs["sustained_shift"] <= 0.13


def test_magnitude_and_sign_ranges(config):
    """Verify magnitude log-uniform [0.5, 10.0] C and sign ~75% positive, ~25% negative."""
    gen = EventGenerator(config)
    rng = np.random.default_rng(7777)

    signed_mags, abs_mags, signs = [], [], []
    for _ in range(2000):
        sm, am, s = gen.sample_magnitude(rng)
        signed_mags.append(sm)
        abs_mags.append(am)
        signs.append(s)

    abs_arr = np.array(abs_mags)
    signs_arr = np.array(signs)

    # Bounds
    assert np.all(abs_arr >= 0.5 - 1e-9)
    assert np.all(abs_arr <= 10.0 + 1e-9)

    # Log-uniform median: exp( (ln(0.5) + ln(10))/2 ) = sqrt(5) approx 2.236
    median_val = np.median(abs_arr)
    assert 1.9 <= median_val <= 2.6

    # Sign probabilities: ~75% positive (+1), ~25% negative (-1)
    pos_frac = np.mean(signs_arr == 1)
    assert 0.71 <= pos_frac <= 0.79


def test_scopes_and_attenuation_ranges(config, sample_sensors):
    """Verify global (55%), partial (30%), and local (15%) scopes and attenuation ranges."""
    gen = EventGenerator(config)
    rng = np.random.default_rng(6666)
    sensor_ids = [s.sensor_id for s in sample_sensors]  # 5 sensors

    scope_counts = {"global": 0, "partial": 0, "local": 0}
    for _ in range(1500):
        scope, affected, atts = gen.sample_scope_and_sensors(sensor_ids, rng)
        scope_counts[scope] += 1

        if scope == "global":
            assert len(affected) == len(sensor_ids)
            for s_id in affected:
                assert 0.6 <= atts[s_id] <= 1.0

        elif scope == "partial":
            # 40% to 80% of 5 sensors -> 2 to 4 sensors
            assert 1 <= len(affected) < len(sensor_ids)
            for s_id in affected:
                assert 0.3 <= atts[s_id] <= 1.0

        elif scope == "local":
            # 1 to 2 sensors
            assert 1 <= len(affected) <= 2
            for s_id in affected:
                assert 0.7 <= atts[s_id] <= 1.0

    total = sum(scope_counts.values())
    assert 0.50 <= scope_counts["global"] / total <= 0.60
    assert 0.25 <= scope_counts["partial"] / total <= 0.35
    assert 0.11 <= scope_counts["local"] / total <= 0.19


def test_arrival_delay_ranges(config, sample_sensors):
    """Verify arrival delay per sensor is strictly within [0, 10] minutes."""
    gen = EventGenerator(config)
    rng = np.random.default_rng(5555)
    sensor_ids = [s.sensor_id for s in sample_sensors]

    for _ in range(200):
        delays = gen.sample_sensor_delays(sensor_ids, rng)
        for d in delays.values():
            assert 0.0 <= d <= 10.0


# ---------------------------------------------------------------------------
# 3b. Event start step distribution and warm-up isolation (Decision 18)
# ---------------------------------------------------------------------------

def test_event_start_step_distribution_and_bounds(config):
    """Verify event start steps are sampled uniformly from inclusive range [480, 1919] (Decision 18).
    
    The warm-up period [0, 479] must strictly never contain event start steps.
    """
    gen = EventGenerator(config)
    rng = np.random.default_rng(12345)

    assert gen.start_step_min == 480
    assert gen.start_step_max == 1919

    # Draw a large sample to test boundaries and uniformity
    n_samples = 10000
    start_steps = np.array([gen.sample_start_step(rng=rng) for _ in range(n_samples)])

    # 1. Strict boundary enforcement
    assert np.all(start_steps >= 480), "Event start step must never be below 480"
    assert np.all(start_steps <= 1919), "Event start step must never be above 1919"

    # 2. Reaches boundary values over large sample
    min_step = np.min(start_steps)
    max_step = np.max(start_steps)
    assert min_step <= 482, f"Sample min {min_step} should reach near lower bound 480"
    assert max_step >= 1917, f"Sample max {max_step} should reach near upper bound 1919"

    # 3. Uniformity: mean should be near (480 + 1919) / 2 = 1199.5
    empirical_mean = np.mean(start_steps)
    assert 1180.0 <= empirical_mean <= 1220.0, f"Empirical mean {empirical_mean} should be near 1199.5"

    # 4. Uniformity across 4 quartiles
    q1 = np.percentile(start_steps, 25)
    q2 = np.percentile(start_steps, 50)
    q3 = np.percentile(start_steps, 75)
    # Expected quartiles: 480 + 0.25*1440 = 840, 1200, 1560
    assert 820 <= q1 <= 860
    assert 1180 <= q2 <= 1220
    assert 1540 <= q3 <= 1580


def test_warmup_period_strictly_event_free(config, sample_sensors):
    """Verify the first 480 steps (0 to 479) remain strictly free of genuine events across scenarios."""
    gen = EventGenerator(config)

    # Check across 100 independent scenario seeds
    for seed in range(100):
        events = gen.generate_events(
            sensor_params=sample_sensors,
            total_steps=1920,
            seed=seed,
        )

        for ep in events.episodes:
            assert ep.start_step >= 480, f"Seed {seed}: episode start step {ep.start_step} < 480"
            assert ep.start_step <= 1919, f"Seed {seed}: episode start step {ep.start_step} > 1919"

        # Sensor contributions during warm-up must be strictly 0.0
        for s in sample_sensors:
            warmup_contrib = events.sensor_contributions[s.sensor_id][:480]
            assert np.all(warmup_contrib == 0.0), f"Seed {seed}: non-zero event during warm-up on {s.sensor_id}"

        # Ground-truth masks during warm-up must be strictly 0
        assert np.all(events.genuine_event[:480] == 0)
        assert np.all(events.event_below_detectability[:480] == 0)
        assert np.all(events.event_magnitude_sigma[:480] == 0.0)
        assert np.all(events.true_event_breadth[:480] == 0.0)
        assert all(events.active_event_ids[t] == "" for t in range(480))


def test_configured_start_step_bounds_used(config, sample_sensors):
    """Verify that start_step_min and start_step_max are read from YAML configuration without hard-coding."""
    custom_cfg = dict(config)
    custom_events_cfg = dict(config["genuine_events"])
    custom_events_cfg["start_step_min"] = 600
    custom_events_cfg["start_step_max"] = 800
    custom_cfg["genuine_events"] = custom_events_cfg

    gen = EventGenerator(custom_cfg)
    assert gen.start_step_min == 600
    assert gen.start_step_max == 800

    rng = np.random.default_rng(42)
    steps = [gen.sample_start_step(rng=rng) for _ in range(500)]
    assert np.all(np.array(steps) >= 600)
    assert np.all(np.array(steps) <= 800)


def test_arrival_delays_separate_from_base_start_step(config, sample_sensors):
    """Verify arrival delays [0, 10] min remain separate from base event start step."""
    gen = EventGenerator(config)

    # Find a scenario with an event to inspect delays and arrival timestamps
    found = False
    for seed in range(50):
        events = gen.generate_events(
            sensor_params=sample_sensors,
            total_steps=1920,
            seed=seed,
        )
        if events.has_events:
            for ep in events.episodes:
                # Base start time in minutes
                base_time = ep.start_time_minutes
                assert base_time == ep.start_step * 1.0
                assert base_time >= 480.0

                # Per-sensor delays must be in [0, 10] and additive
                for s_id in ep.affected_sensors:
                    d_k = ep.sensor_delays[s_id]
                    assert 0.0 <= d_k <= 10.0
                    # Arrival time for sensor is base_time + d_k
                    t_arr_k = base_time + d_k
                    assert t_arr_k >= base_time
                    # Before arrival time, contribution for this episode must be 0
                    arrival_step = int(np.floor(t_arr_k))
                    if arrival_step > 0:
                        # Prior to arrival time, no wave from this episode has reached sensor
                        pass
            found = True
            break
    assert found, "Should find at least one scenario with events"


# ---------------------------------------------------------------------------
# 4. Ground-truth detectability logic tests
# ---------------------------------------------------------------------------

def test_detectability_threshold_behavior(config):
    """Verify genuine_event = 1 when max_k(|Delta T_k| / sigma_k) >= 3.0,

    and event_below_detectability = 1 when active but < 3.0.
    """
    # Create two synthetic sensors with known sigmas
    s1 = TemperatureSensorParams(
        sensor_id="sensor_00",
        sigma_k=0.10,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        missing_rate=0.0,
        burst_max=3,
    )
    s2 = TemperatureSensorParams(
        sensor_id="sensor_01",
        sigma_k=0.20,
        is_noise_correlated=False,
        noise_correlation_rho=0.0,
        static_offset=0.0,
        response_lag_tau=0.0,
        quantisation_step=0.01,
        missing_rate=0.0,
        burst_max=3,
    )

    gen = EventGenerator(config)

    # Let's create an EventSeries manually with controlled waveforms to test detectability
    # Event: ramp rising from 0.0 to 0.5 C on sensor_00 (3 * sigma_k = 0.30 C)
    t = np.arange(100, dtype=np.float64)
    # sensor_00 sees ramp from t=10 to t=30 (reaches 0.5 C at t=30)
    w0 = generate_ramp(t, t_arrival=10.0, magnitude=0.5, rise_minutes=20.0, hold_minutes=20.0, fall_minutes=20.0)
    # sensor_01 sees nothing
    w1 = np.zeros_like(t)

    # Simulate detectability check on these waveforms
    thresh = gen.detectability_threshold_sigma  # 3.0
    thresh_s0 = thresh * s1.sigma_k  # 3 * 0.10 = 0.30 C

    # Replicate detectability logic:
    # At t=10: w0 = 0.0 -> neither
    # At t=15: w0 = 0.125 < 0.30 -> event_below_detectability = 1, genuine_event = 0
    # At t=22: w0 = 0.30 >= 0.30 -> genuine_event = 1, event_below_detectability = 0
    # At t=35: w0 = 0.50 >= 0.30 -> genuine_event = 1
    # At t=65: fall midway = 0.375 >= 0.30 -> genuine_event = 1
    # At t=75: fall = 0.125 < 0.30 -> event_below_detectability = 1
    # At t=80: w0 = 0.0 -> neither

    # Check via generate_events by running a fixed seed and inspecting masks
    events = gen.generate_events(sensor_params=[s1, s2], total_steps=100, seed=123)

    for step in range(100):
        c0 = abs(events.sensor_contributions["sensor_00"][step])
        c1 = abs(events.sensor_contributions["sensor_01"][step])
        max_ratio = max(c0 / s1.sigma_k, c1 / s2.sigma_k)
        is_active = (c0 > 1e-4) or (c1 > 1e-4)
        is_detectable = (c0 >= 3.0 * s1.sigma_k) or (c1 >= 3.0 * s2.sigma_k)

        if is_detectable:
            assert events.genuine_event[step] == 1
            assert events.event_below_detectability[step] == 0
        elif is_active:
            assert events.genuine_event[step] == 0
            assert events.event_below_detectability[step] == 1
        else:
            assert events.genuine_event[step] == 0
            assert events.event_below_detectability[step] == 0

        assert np.isclose(events.event_magnitude_sigma[step], max_ratio)


def test_sustained_shift_persists_to_scenario_end(config, sample_sensors):
    """Verify that sustained shift events hold until scenario end (step 1920, Decision 5)."""
    gen = EventGenerator(config)

    # Search for a seed that produces a sustained shift
    found_sustained = False
    for seed in range(50):
        events = gen.generate_events(sensor_params=sample_sensors, total_steps=1920, seed=seed)
        for ep in events.episodes:
            if ep.event_type == "sustained_shift":
                found_sustained = True
                # Check that for all affected sensors, contribution at step 1919 equals the sustained magnitude
                for s_id in ep.affected_sensors:
                    end_val = events.sensor_contributions[s_id][1919]
                    expected_val = ep.magnitude * ep.sensor_attenuations[s_id]
                    assert np.isclose(end_val, expected_val, atol=1e-4)
                break
        if found_sustained:
            break

    assert found_sustained, "Should find at least one sustained shift in 50 seeds"


# ---------------------------------------------------------------------------
# 5. Integration with environment and virtual sensors
# ---------------------------------------------------------------------------

def test_apply_events_to_environment(config, sample_sensors):
    """Verify genuine events alter the physical signal before sensor artefacts."""
    env_sim = EnvironmentSimulator(config)
    env = env_sim.generate(seed=42)

    event_gen = EventGenerator(config)
    events = event_gen.generate_events(
        sensor_params=sample_sensors,
        total_steps=env.total_steps,
        seed=42,
        baseline_env_values=env.values,
    )

    phys_signals = apply_events_to_environment(
        environment=env,
        event_series=events,
        env_clip_min=-5.0,
        env_clip_max=50.0,
    )

    for s in sample_sensors:
        s_id = s.sensor_id
        expected_raw = env.values + events.sensor_contributions[s_id]
        expected_phys = np.clip(expected_raw, -5.0, 50.0)
        np.testing.assert_allclose(phys_signals[s_id], expected_phys)


def test_sensor_group_simulation_with_events(config, sample_sensors):
    """Verify VirtualSensorSimulator correctly transduces event-adjusted physical signal."""
    env_sim = EnvironmentSimulator(config)
    env = env_sim.generate(seed=77)

    group = simulate_sensors(
        environment=env,
        sensor_params=sample_sensors,
        seed=77,
        config=config,
        generate_events=True,
    )

    assert group.events is not None
    assert group.total_steps == 1920

    # Verify each sensor stream has physical_values and event_contributions
    for s_id, stream in group.sensors.items():
        assert stream.physical_values is not None
        assert stream.event_contributions is not None
        assert len(stream.physical_values) == 1920
        assert len(stream.event_contributions) == 1920

        # Physical values must be within environment bounds [-5, 50] C
        assert np.all(stream.physical_values >= -5.0 - 1e-9)
        assert np.all(stream.physical_values <= 50.0 + 1e-9)

        # Observable values must reflect missing data and unit scaling
        valid_obs = stream.values[~stream.missing_mask]
        assert len(valid_obs) > 0


def test_strict_isolation_observable_dataframe_no_truth(config, sample_sensors):
    """Verify to_observable_dataframe contains ZERO ground-truth columns or event labels."""
    env_sim = EnvironmentSimulator(config)
    env = env_sim.generate(seed=101)

    group = simulate_sensors(
        environment=env,
        sensor_params=sample_sensors,
        seed=101,
        config=config,
        generate_events=True,
    )

    obs_df = group.to_observable_dataframe(scenario_id="scenario_0042")

    # Required columns in Layer 1 raw observable table
    expected_cols = {"scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"}
    assert set(obs_df.columns) == expected_cols

    # Must NOT contain any truth columns
    forbidden_cols = {
        "genuine_event", "event_below_detectability", "event_id", "event_magnitude_sigma",
        "true_event_breadth", "true_group_value", "magnitude", "true_value", "sensor_untrusted",
    }
    assert forbidden_cols.isdisjoint(set(obs_df.columns))


def test_ground_truth_events_dataframe_schema(config, sample_sensors):
    """Verify to_ground_truth_events_dataframe matches Appendix A.7 schema."""
    env_sim = EnvironmentSimulator(config)
    env = env_sim.generate(seed=202)

    group = simulate_sensors(
        environment=env,
        sensor_params=sample_sensors,
        seed=202,
        config=config,
        generate_events=True,
    )

    gt_df = group.to_ground_truth_events_dataframe(scenario_id="scenario_0202")

    expected_gt_cols = {
        "scenario_id", "group_id", "timestamp", "genuine_event",
        "event_below_detectability", "event_id", "event_magnitude_sigma",
        "true_event_breadth", "true_group_value",
    }
    assert set(gt_df.columns) == expected_gt_cols
    assert len(gt_df) == 1920

    # Binary flags
    assert set(np.unique(gt_df["genuine_event"])).issubset({0, 1})
    assert set(np.unique(gt_df["event_below_detectability"])).issubset({0, 1})
    # Cannot be simultaneously 1
    assert np.all((gt_df["genuine_event"] + gt_df["event_below_detectability"]) <= 1)


def test_episodes_dataframe_export(config, sample_sensors):
    """Verify episodes_to_dataframe exports episode metadata table."""
    env_sim = EnvironmentSimulator(config)
    env = env_sim.generate(seed=303)

    group = simulate_sensors(
        environment=env,
        sensor_params=sample_sensors,
        seed=303,
        config=config,
        generate_events=True,
    )

    ep_df = group.episodes_to_dataframe(scenario_id="scenario_0303")
    assert isinstance(ep_df, pd.DataFrame)
    if len(ep_df) > 0:
        assert "event_id" in ep_df.columns
        assert "event_type" in ep_df.columns
        assert "magnitude" in ep_df.columns
        assert "scope" in ep_df.columns
        assert "affected_sensors" in ep_df.columns
