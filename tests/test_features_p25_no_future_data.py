"""Automated no-future-data test for P2.5.

Non-negotiable ML Rule:
Causal processing only: features at time t may use observations at or before t,
never future observations. Modifying future readings (t > t_0) MUST have zero effect
on features computed for timestamps t <= t_0.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engine import FeatureEngine, SENSOR_FEATURE_NAMES, GROUP_FEATURE_NAMES


@pytest.fixture
def synthetic_observables() -> pd.DataFrame:
    """Generate a clean 6-column observable DataFrame with 5 sensors and 800 steps."""
    np.random.seed(42)
    n_steps = 800
    sensor_ids = [f"sensor_{i:02d}" for i in range(5)]
    records = []

    for sid in sensor_ids:
        base = 22.0 + np.random.uniform(-0.5, 0.5)
        trend = np.sin(2 * np.pi * np.arange(n_steps) / 300)
        noise = np.random.normal(0, 0.1, n_steps)
        vals = base + trend + noise
        # occasional healthy missing
        vals[100:102] = np.nan
        vals[250] = np.nan

        for t in range(n_steps):
            records.append({
                "scenario_id": "scenario_causal_test",
                "group_id": "group_00",
                "sensor_id": sid,
                "variable_type": "temperature",
                "timestamp": t,
                "value": vals[t],
            })

    return pd.DataFrame(records)


def test_no_future_data_with_spikes(synthetic_observables):
    """Test that extreme future spikes at t > t_cutoff do not affect past features (t <= t_cutoff)."""
    engine = FeatureEngine()
    t_cutoff = 500

    # Past-only slice
    past_obs = synthetic_observables[synthetic_observables["timestamp"] <= t_cutoff].copy()
    sensor_past, group_past = engine.compute_features(past_obs)

    # Full data with extreme future spikes (+100 C)
    full_spiked = synthetic_observables.copy()
    mask_future = full_spiked["timestamp"] > t_cutoff
    full_spiked.loc[mask_future, "value"] += 100.0
    sensor_spiked, group_spiked = engine.compute_features(full_spiked)

    # Filter spiked results to past window
    sensor_spiked_past = sensor_spiked[sensor_spiked["timestamp"] <= t_cutoff].reset_index(drop=True)
    group_spiked_past = group_spiked[group_spiked["timestamp"] <= t_cutoff].reset_index(drop=True)

    # Sensor features comparison
    for feat in SENSOR_FEATURE_NAMES:
        past_vals = sensor_past[feat].to_numpy()
        future_vals = sensor_spiked_past[feat].to_numpy()
        np.testing.assert_allclose(
            past_vals,
            future_vals,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Future spikes leaked into past sensor feature: {feat}",
        )

    # Group features comparison
    for feat in GROUP_FEATURE_NAMES:
        past_vals = group_past[feat].to_numpy()
        future_vals = group_spiked_past[feat].to_numpy()
        np.testing.assert_allclose(
            past_vals,
            future_vals,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Future spikes leaked into past group feature: {feat}",
        )


def test_no_future_data_with_complete_future_dropout(synthetic_observables):
    """Test that complete future dropout (all NaNs at t > t_cutoff) does not alter past features."""
    engine = FeatureEngine()
    t_cutoff = 480

    past_obs = synthetic_observables[synthetic_observables["timestamp"] <= t_cutoff].copy()
    sensor_past, group_past = engine.compute_features(past_obs)

    # Full data with future replaced entirely by NaN
    full_nan = synthetic_observables.copy()
    mask_future = full_nan["timestamp"] > t_cutoff
    full_nan.loc[mask_future, "value"] = np.nan
    sensor_nan, group_nan = engine.compute_features(full_nan)

    sensor_nan_past = sensor_nan[sensor_nan["timestamp"] <= t_cutoff].reset_index(drop=True)
    group_nan_past = group_nan[group_nan["timestamp"] <= t_cutoff].reset_index(drop=True)

    for feat in SENSOR_FEATURE_NAMES:
        past_vals = sensor_past[feat].to_numpy()
        nan_vals = sensor_nan_past[feat].to_numpy()
        np.testing.assert_allclose(
            past_vals,
            nan_vals,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Future NaNs leaked into past sensor feature: {feat}",
        )

    for feat in GROUP_FEATURE_NAMES:
        past_vals = group_past[feat].to_numpy()
        nan_vals = group_nan_past[feat].to_numpy()
        np.testing.assert_allclose(
            past_vals,
            nan_vals,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Future NaNs leaked into past group feature: {feat}",
        )


def test_no_future_data_with_massive_future_event(synthetic_observables):
    """Test that an environmental heatwave injected in the future does not alter earlier features."""
    engine = FeatureEngine()
    t_cutoff = 350

    past_obs = synthetic_observables[synthetic_observables["timestamp"] <= t_cutoff].copy()
    sensor_past, group_past = engine.compute_features(past_obs)

    # Injected massive event in the future (+20 C ramp)
    full_event = synthetic_observables.copy()
    future_steps = full_event["timestamp"] - t_cutoff
    mask_future = full_event["timestamp"] > t_cutoff
    full_event.loc[mask_future, "value"] += np.minimum(20.0, 0.5 * future_steps[mask_future])

    sensor_event, group_event = engine.compute_features(full_event)
    sensor_event_past = sensor_event[sensor_event["timestamp"] <= t_cutoff].reset_index(drop=True)
    group_event_past = group_event[group_event["timestamp"] <= t_cutoff].reset_index(drop=True)

    # Verify bitwise / high-precision equality
    for feat in SENSOR_FEATURE_NAMES:
        np.testing.assert_allclose(
            sensor_past[feat].to_numpy(),
            sensor_event_past[feat].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Future event leaked into sensor feature: {feat}",
        )

    for feat in GROUP_FEATURE_NAMES:
        np.testing.assert_allclose(
            group_past[feat].to_numpy(),
            group_event_past[feat].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Future event leaked into group feature: {feat}",
        )
