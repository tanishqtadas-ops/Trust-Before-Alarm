"""Models module (rules, baselines, LightGBM heads, calibration, trust, alarm layer)."""

from src.models.raw_alarm import (
    RawAlarmDetector,
    RawAlarmEpisode,
    compute_robust_spread,
    detect_episodes_from_bounds,
    assign_alarm_causes,
)
from src.models.baselines import (
    FixedThresholdDetector,
    RobustZScoreDetector,
    extract_runs_and_merge,
)

__all__ = [
    "RawAlarmDetector",
    "RawAlarmEpisode",
    "compute_robust_spread",
    "detect_episodes_from_bounds",
    "assign_alarm_causes",
    "FixedThresholdDetector",
    "RobustZScoreDetector",
    "extract_runs_and_merge",
]
