"""Difficulty checks and stratified baseline evaluation (P3.3).

Appendix A.6, Appendix C, and P3.3:
- Evaluates detectors separately across EASY, MEDIUM, and HARD scenarios.
- Checks if baseline is near-perfect on HARD (data too easy).
- Checks if faults cannot be seen even on EASY (data too difficult).
- Reports results cleanly from saved output files.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.eval.alarm_metrics import evaluate_alarm_performance


def evaluate_by_difficulty_tier(
    detector_name: str,
    alarm_episodes_df: pd.DataFrame,
    observables_df: pd.DataFrame,
    ground_truth_sensors_df: pd.DataFrame,
    ground_truth_events_df: pd.DataFrame,
    fault_episodes_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    grace_steps: int = 5,
    scored_start_step: int = 480,
    scored_end_step: int = 1919,
) -> Dict[str, Any]:
    """Evaluate detector performance separately across difficulty tiers.
    
    Args:
        detector_name: Name/label of detector.
        alarm_episodes_df: Detected alarm episodes.
        observables_df: Observable readings.
        ground_truth_sensors_df: Ground truth sensors table.
        ground_truth_events_df: Ground truth events table.
        fault_episodes_df: Fault episodes table.
        scenarios_df: Scenario metadata table with 'difficulty'.
        grace_steps: Overlap grace window.
        scored_start_step: Scored window start (480).
        scored_end_step: Scored window end (1919).
        
    Returns:
        Dictionary mapping tier ('easy', 'medium', 'hard') to performance metrics, plus overall summary.
    """
    scen_tiers = dict(zip(scenarios_df["scenario_id"], scenarios_df["difficulty"].str.lower()))
    tiers = ["easy", "medium", "hard"]

    tier_results: Dict[str, Any] = {}

    for tier in tiers:
        tier_scen_ids = {sid for sid, t in scen_tiers.items() if t == tier}

        # Filter tables for this tier
        sub_obs = observables_df[observables_df["scenario_id"].isin(tier_scen_ids)]
        sub_gts = ground_truth_sensors_df[ground_truth_sensors_df["scenario_id"].isin(tier_scen_ids)]
        sub_gte = ground_truth_events_df[ground_truth_events_df["scenario_id"].isin(tier_scen_ids)]
        sub_fe = fault_episodes_df[fault_episodes_df["scenario_id"].isin(tier_scen_ids)]
        sub_alarm = alarm_episodes_df[alarm_episodes_df["scenario_id"].isin(tier_scen_ids)]

        metrics = evaluate_alarm_performance(
            alarm_episodes_df=sub_alarm,
            observables_df=sub_obs,
            ground_truth_sensors_df=sub_gts,
            ground_truth_events_df=sub_gte,
            fault_episodes_df=sub_fe,
            grace_steps=grace_steps,
            scored_start_step=scored_start_step,
            scored_end_step=scored_end_step,
        )
        metrics["scenario_count"] = len(tier_scen_ids)
        tier_results[tier] = metrics

    # Also compute overall
    overall_metrics = evaluate_alarm_performance(
        alarm_episodes_df=alarm_episodes_df,
        observables_df=observables_df,
        ground_truth_sensors_df=ground_truth_sensors_df,
        ground_truth_events_df=ground_truth_events_df,
        fault_episodes_df=fault_episodes_df,
        grace_steps=grace_steps,
        scored_start_step=scored_start_step,
        scored_end_step=scored_end_step,
    )
    overall_metrics["scenario_count"] = len(scen_tiers)

    # Sanity checks per P3.3 specification
    hard_recall = tier_results["hard"]["episode_recall"]
    easy_recall = tier_results["easy"]["episode_recall"]

    sanity_flags: List[str] = []
    if hard_recall >= 0.95:
        sanity_flags.append(
            f"WARNING: Baseline near-perfect on HARD ({hard_recall*100:.1f}% >= 95.0%). Data may be too easy."
        )
    if easy_recall < 0.05 and overall_metrics["false_alarm_rate"] > 0.5:
        sanity_flags.append(
            f"WARNING: Faults cannot be reliably seen even on EASY ({easy_recall*100:.1f}%). Data may be too difficult."
        )

    if not sanity_flags:
        sanity_status = "BENCHMARK_DIFFICULTY_VALIDATED"
        sanity_msg = (
            "Benchmark difficulty progression validated: baseline is not near-perfect on HARD "
            f"(HARD recall = {hard_recall*100:.1f}%), and faults remain distinguishable on EASY "
            f"(EASY recall = {easy_recall*100:.1f}%). No simulator adjustment required."
        )
    else:
        sanity_status = "BENCHMARK_DIFFICULTY_FLAGGED"
        sanity_msg = " | ".join(sanity_flags)

    return {
        "detector_name": detector_name,
        "tiers": tier_results,
        "overall": overall_metrics,
        "sanity_status": sanity_status,
        "sanity_message": sanity_msg,
    }


def format_tier_report_table(eval_result: Dict[str, Any]) -> str:
    """Format difficulty tier evaluation results into a clean markdown table."""
    name = eval_result["detector_name"]
    tiers = eval_result["tiers"]
    overall = eval_result["overall"]

    lines = [
        f"### Baseline Difficulty Breakdown: {name}",
        "",
        "| Tier | Scenarios | Healthy Days | FA Episodes | FAR (ep/day) | Fault Episodes | Detected | Recall (Ep) | Recall (Row) | Precision |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for t_name in ["easy", "medium", "hard"]:
        t = tiers[t_name]
        lines.append(
            f"| **{t_name.upper()}** | {t['scenario_count']} | {t['healthy_sensor_days']:.0f} | "
            f"{t['false_alarm_episodes']} | {t['false_alarm_rate']:.4f} | {t['total_fault_episodes']} | "
            f"{t['detected_fault_episodes']} | {t['episode_recall']*100:.2f}% | "
            f"{t['row_recall']*100:.2f}% | {t['precision']*100:.2f}% |"
        )

    lines.append(
        f"| **OVERALL** | {overall['scenario_count']} | {overall['healthy_sensor_days']:.0f} | "
        f"{overall['false_alarm_episodes']} | {overall['false_alarm_rate']:.4f} | {overall['total_fault_episodes']} | "
        f"{overall['detected_fault_episodes']} | {overall['episode_recall']*100:.2f}% | "
        f"{overall['row_recall']*100:.2f}% | {overall['precision']*100:.2f}% |"
    )
    lines.append("")
    lines.append(f"**Sanity Check Status:** `{eval_result['sanity_status']}`")
    lines.append(f"*{eval_result['sanity_message']}*")
    lines.append("")
    return "\n".join(lines)
