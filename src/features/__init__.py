"""Feature extraction and engineering module (causal windows, sigma_hat, sensor & group features)."""

from src.features.windows import CausalWindows, causal_slice
from src.features.sigma_hat import compute_sigma_hat
from src.features.sensor_features import (
    compute_own_signal_features,
    compute_cross_sensor_features,
    compute_fast_rolling_slope,
    compute_rolling_mad_noise,
)
from src.features.group_features import compute_group_features
from src.features.engine import (
    FeatureEngine,
    OBSERVABLE_SCHEMA,
    SENSOR_METADATA_COLS,
    GROUP_METADATA_COLS,
    SENSOR_FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
)

__all__ = [
    "CausalWindows",
    "causal_slice",
    "compute_sigma_hat",
    "compute_own_signal_features",
    "compute_cross_sensor_features",
    "compute_fast_rolling_slope",
    "compute_rolling_mad_noise",
    "compute_group_features",
    "FeatureEngine",
    "OBSERVABLE_SCHEMA",
    "SENSOR_METADATA_COLS",
    "GROUP_METADATA_COLS",
    "SENSOR_FEATURE_NAMES",
    "GROUP_FEATURE_NAMES",
]
