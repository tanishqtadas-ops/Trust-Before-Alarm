"""Evaluation module (metrics, audit, slices, ablations, reports)."""

from src.eval.alarm_metrics import (
    evaluate_alarm_performance,
    wilson_interval,
)
from src.eval.difficulty_check import (
    evaluate_by_difficulty_tier,
    format_tier_report_table,
)

__all__ = [
    "evaluate_alarm_performance",
    "wilson_interval",
    "evaluate_by_difficulty_tier",
    "format_tier_report_table",
]
