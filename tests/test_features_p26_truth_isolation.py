"""Automated truth-isolation test for P2.6.

Non-negotiable ML Rules:
1. Feature computation physically accepts ONLY the 6-column observable schema:
   [scenario_id, group_id, sensor_id, variable_type, timestamp, value]
2. Ground-truth tables must NEVER be inputs to feature computation.
3. Feature tables must not expose raw value, timestamps or identifiers as model features.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engine import (
    FeatureEngine,
    OBSERVABLE_SCHEMA,
    SENSOR_METADATA_COLS,
    GROUP_METADATA_COLS,
    SENSOR_FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
)


@pytest.fixture
def isolated_observables() -> pd.DataFrame:
    """Create a completely isolated 6-column observable DataFrame with zero ground-truth."""
    np.random.seed(99)
    records = []
    for sid in ["s_01", "s_02", "s_03", "s_04"]:
        for t in range(500):
            records.append({
                "scenario_id": "isolated_test_scenario",
                "group_id": "group_alpha",
                "sensor_id": sid,
                "variable_type": "temperature",
                "timestamp": t,
                "value": 20.0 + 0.1 * np.sin(t / 20.0) + np.random.normal(0, 0.05),
            })
    df = pd.DataFrame(records)
    # Ensure exact column schema
    return df[OBSERVABLE_SCHEMA]


def test_feature_generation_with_only_observables(isolated_observables):
    """Test feature generation works when strictly and exclusively the 6-column observable table is provided."""
    # Verify input DataFrame has only the 6 observable columns
    assert list(isolated_observables.columns) == OBSERVABLE_SCHEMA

    engine = FeatureEngine()
    sensor_df, group_df = engine.compute_features(isolated_observables)

    # Output tables must not be empty
    assert not sensor_df.empty
    assert not group_df.empty
    assert len(sensor_df) == 4 * 500
    assert len(group_df) == 500


def test_ground_truth_leakage_rejected(isolated_observables):
    """Test that feature engine strictly rejects DataFrames containing ground-truth columns."""
    engine = FeatureEngine()

    leaked_columns = [
        "sensor_untrusted", "true_value", "genuine_event",
        "fault_severity", "true_health", "f_spike", "f_offset_drift",
    ]

    for col in leaked_columns:
        corrupt_df = isolated_observables.copy()
        corrupt_df[col] = 0.0
        with pytest.raises(ValueError, match="Ground-truth leakage detected"):
            engine.compute_features(corrupt_df)


def test_no_raw_value_or_identifiers_in_model_features(isolated_observables):
    """Test that raw value, variable_type, identifiers, and timestamps are NOT in model features."""
    engine = FeatureEngine()
    sensor_df, group_df = engine.compute_features(isolated_observables)

    # 1. Model features list must never include raw value, timestamp, or identifiers
    forbidden_features = [
        "value", "variable_type", "scenario_id", "group_id", "sensor_id", "timestamp",
    ]
    for forbidden in forbidden_features:
        assert forbidden not in SENSOR_FEATURE_NAMES, f"Forbidden column {forbidden} found in SENSOR_FEATURE_NAMES"
        assert forbidden not in GROUP_FEATURE_NAMES, f"Forbidden column {forbidden} found in GROUP_FEATURE_NAMES"

    # 2. Check metadata vs model feature separation in sensor_df
    expected_sensor_cols = SENSOR_METADATA_COLS + SENSOR_FEATURE_NAMES
    assert list(sensor_df.columns) == expected_sensor_cols
    assert "value" not in sensor_df.columns
    assert "variable_type" not in sensor_df.columns

    # 3. Check metadata vs model feature separation in group_df
    expected_group_cols = GROUP_METADATA_COLS + GROUP_FEATURE_NAMES
    assert list(group_df.columns) == expected_group_cols
    assert "value" not in group_df.columns
    assert "sensor_id" not in group_df.columns

    # 4. Extract clean model feature matrices without metadata
    X_sensor = sensor_df[SENSOR_FEATURE_NAMES]
    X_group = group_df[GROUP_FEATURE_NAMES]

    assert X_sensor.shape == (2000, 49)
    assert X_group.shape == (500, 11)

    # Ensure all data in model feature matrix are numeric floats
    assert np.issubdtype(X_sensor.dtypes.iloc[0], np.floating)
    assert np.issubdtype(X_group.dtypes.iloc[0], np.floating)
