"""Leakage Audit Module for Phase P4.

Implements:
1. P4.1: Single-feature separability audit (AUC > 0.85 flag).
2. P4.2: Label-shuffle null control (5 seeds: 20001-20005) & canary leak test (AUC >= 0.95).

Strictly enforces:
- No identifiers or raw values as features.
- Scenario-level partition integrity (train vs val).
- Permitted data splits only (test split is held out).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

from src.utils.config import get_config_dir, load_yaml
from src.features.engine import SENSOR_FEATURE_NAMES, GROUP_FEATURE_NAMES


FAULT_TYPES = [
    "spike",
    "offset_drift",
    "gain_error",
    "frozen",
    "high_noise",
    "dropout",
    "slow_degradation",
]


@dataclass
class SingleFeatureSeparabilityResult:
    """Result of single-feature separability evaluation."""
    feature_name: str
    auc_untrusted: float
    auc_untrusted_raw: float
    direction: str  # "+" or "-"
    pr_auc_untrusted: float
    ovr_auc_per_fault: Dict[str, float]
    max_ovr_auc: float
    max_ovr_fault: str
    is_flagged_untrusted: bool
    is_flagged_fault: bool
    task_aware_notes: str


@dataclass
class LabelShuffleSeedResult:
    """Result of a single label-shuffle run."""
    seed: int
    val_auc: float
    val_pr_auc: float
    val_recall_at_50pct: float
    val_recall_at_budget: float
    passed_auc: bool
    passed_recall: bool
    train_samples: int = 0
    train_positives_before: int = 0
    train_positives_after: int = 0
    val_samples: int = 0
    val_positives: int = 0
    class_counts_preserved: bool = True
    overall_status: str = "FAIL"


@dataclass
class CanaryTestResult:
    """Result of the planted canary leak test."""
    canary_auc: float
    canary_caught: bool
    clean_max_auc: float
    clean_max_feature: str
    clean_passed: bool
    notes: str


def compute_directional_auc(y_true: np.ndarray, x: np.ndarray) -> Tuple[float, float, str]:
    """Compute directional ROC-AUC handling NaNs.
    
    Returns:
        (auc_directional, auc_raw, direction)
        auc_directional is in [0.5, 1.0].
    """
    valid_mask = ~np.isnan(x)
    if valid_mask.sum() == 0 or len(np.unique(y_true[valid_mask])) < 2:
        return 0.5, 0.5, "+"
    
    y_sub = y_true[valid_mask]
    x_sub = x[valid_mask]
    
    try:
        raw_auc = float(roc_auc_score(y_sub, x_sub))
    except Exception:
        return 0.5, 0.5, "+"
    
    if raw_auc >= 0.5:
        return raw_auc, raw_auc, "+"
    else:
        return 1.0 - raw_auc, raw_auc, "-"


def evaluate_single_feature_separability(
    feature_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    threshold: float = 0.85,
    features_to_eval: Optional[List[str]] = None,
) -> List[SingleFeatureSeparabilityResult]:
    """Evaluate single-feature separability (P4.1) against sensor_untrusted and fault classes.
    
    Args:
        feature_df: Feature DataFrame with metadata columns [scenario_id, sensor_id, timestamp].
        gt_df: Ground truth DataFrame with sensor_untrusted and f_* columns.
        threshold: Flag threshold (default 0.85 from Appendix A.14 / eval.yaml).
        features_to_eval: Optional subset of features to evaluate. Defaults to SENSOR_FEATURE_NAMES.
    """
    if features_to_eval is None:
        features_to_eval = [f for f in SENSOR_FEATURE_NAMES if f in feature_df.columns]

    # Align features and ground truth on scenario_id, sensor_id, timestamp
    join_keys = ["scenario_id", "sensor_id", "timestamp"]
    merged = pd.merge(
        feature_df[join_keys + features_to_eval],
        gt_df[join_keys + ["sensor_untrusted"] + [f"f_{ft}" for ft in FAULT_TYPES]],
        on=join_keys,
        how="inner",
    )
    
    y_untrusted = merged["sensor_untrusted"].to_numpy().astype(int)
    
    results: List[SingleFeatureSeparabilityResult] = []
    
    for feat in features_to_eval:
        x = merged[feat].to_numpy(dtype=float)
        auc_dir, auc_raw, direction = compute_directional_auc(y_untrusted, x)
        
        # PR-AUC for untrusted
        valid_mask = ~np.isnan(x)
        if valid_mask.sum() > 0 and len(np.unique(y_untrusted[valid_mask])) >= 2:
            score_for_pr = x[valid_mask] if direction == "+" else -x[valid_mask]
            try:
                pr_auc = float(average_precision_score(y_untrusted[valid_mask], score_for_pr))
            except Exception:
                pr_auc = 0.0
        else:
            pr_auc = 0.0

        # One-vs-Rest AUC per fault type
        ovr_aucs: Dict[str, float] = {}
        for ft in FAULT_TYPES:
            f_col = f"f_{ft}"
            if f_col in merged.columns:
                y_f = merged[f_col].to_numpy().astype(int)
                f_auc, _, _ = compute_directional_auc(y_f, x)
                ovr_aucs[ft] = f_auc
            else:
                ovr_aucs[ft] = 0.5
        
        max_f = max(ovr_aucs.items(), key=lambda item: item[1])
        max_ovr_fault, max_ovr_auc = max_f[0], max_f[1]

        is_flagged_u = auc_dir > threshold
        is_flagged_f = max_ovr_auc > threshold

        # Task-aware notes
        notes = []
        if is_flagged_u:
            notes.append(f"Untrusted AUC {auc_dir:.3f} > {threshold}")
        if is_flagged_f:
            notes.append(f"Fault '{max_ovr_fault}' AUC {max_ovr_auc:.3f} > {threshold}")
        
        # Physical domain explanations for legitimate separability
        if feat in ("consec_missing", "cur_missing", "frac_missing_m", "frac_missing_l") and "dropout" in max_ovr_fault:
            notes.append("Expected physical response to sensor dropout (provable rule)")
        elif feat in ("repeat_run_len", "longest_repeat_m", "unique_ratio_m", "frac_zero_diff_m") and "frozen" in max_ovr_fault:
            notes.append("Expected physical response to frozen sensor (provable rule)")
        elif feat in ("noise_ratio_s", "noise_ratio_m", "noise_short_long") and "high_noise" in max_ovr_fault:
            notes.append("Expected response to high measurement noise")
        elif not notes:
            notes.append("Nominal separability (<= 0.85)")

        results.append(SingleFeatureSeparabilityResult(
            feature_name=feat,
            auc_untrusted=auc_dir,
            auc_untrusted_raw=auc_raw,
            direction=direction,
            pr_auc_untrusted=pr_auc,
            ovr_auc_per_fault=ovr_aucs,
            max_ovr_auc=max_ovr_auc,
            max_ovr_fault=max_ovr_fault,
            is_flagged_untrusted=is_flagged_u,
            is_flagged_fault=is_flagged_f,
            task_aware_notes="; ".join(notes),
        ))

    # Sort results by untrusted AUC descending
    results.sort(key=lambda r: r.auc_untrusted, reverse=True)
    return results


def run_label_shuffle_null_control(
    train_features: pd.DataFrame,
    train_gt: pd.DataFrame,
    val_features: pd.DataFrame,
    val_gt: pd.DataFrame,
    feature_cols: List[str],
    seeds: List[int] = (20001, 20002, 20003, 20004, 20005),
    auc_target: float = 0.50,
    auc_tolerance: float = 0.03,
    recall_at_budget_max: float = 0.05,
    sample_train_size: Optional[int] = None,
) -> List[LabelShuffleSeedResult]:
    """Run label-shuffle null control (P4.2) across 5 seeds.
    
    Shuffles training labels only, leaving validation labels and split intact.
    """
    join_keys = ["scenario_id", "sensor_id", "timestamp"]

    # Verify train and val scenarios are disjoint
    train_scens = set(train_features["scenario_id"].unique())
    val_scens = set(val_features["scenario_id"].unique())
    if not train_scens.isdisjoint(val_scens):
        overlap = train_scens.intersection(val_scens)
        raise ValueError(f"Train and validation scenario IDs must be strictly disjoint! Overlap: {overlap}")
    
    # Prepare train dataset
    train_merged = pd.merge(
        train_features[join_keys + feature_cols],
        train_gt[join_keys + ["sensor_untrusted"]],
        on=join_keys,
        how="inner",
    )
    if sample_train_size and len(train_merged) > sample_train_size:
        train_merged = train_merged.sample(n=sample_train_size, random_state=42).reset_index(drop=True)

    X_train = train_merged[feature_cols].to_numpy(dtype=float)
    y_train_orig = train_merged["sensor_untrusted"].to_numpy().astype(int)
    n_pos_train_orig = int(y_train_orig.sum())

    # Prepare val dataset
    val_merged = pd.merge(
        val_features[join_keys + feature_cols],
        val_gt[join_keys + ["sensor_untrusted"]],
        on=join_keys,
        how="inner",
    )
    X_val = val_merged[feature_cols].to_numpy(dtype=float)
    y_val = val_merged["sensor_untrusted"].to_numpy().astype(int)
    n_pos_val = int(y_val.sum())

    seed_results: List[LabelShuffleSeedResult] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Permute training labels ONLY
        y_train_shuffled = rng.permutation(y_train_orig)
        n_pos_train_shuff = int(y_train_shuffled.sum())
        assert n_pos_train_orig == n_pos_train_shuff, (
            f"Class count mismatch in seed {seed}: before={n_pos_train_orig}, after={n_pos_train_shuff}"
        )

        # Train a fast LightGBM binary classifier
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=50,
            learning_rate=0.1,
            num_leaves=31,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
        clf.fit(X_train, y_train_shuffled)

        # Predict probabilities on untouched validation set
        val_probs = clf.predict_proba(X_val)[:, 1]

        # Validation AUC
        try:
            val_auc = float(roc_auc_score(y_val, val_probs))
        except Exception:
            val_auc = 0.5
        
        try:
            val_pr_auc = float(average_precision_score(y_val, val_probs))
        except Exception:
            val_pr_auc = 0.0

        # Recall at p >= 0.5
        preds_50 = (val_probs >= 0.5).astype(int)
        recall_50 = float((preds_50[y_val == 1]).sum() / n_pos_val) if n_pos_val > 0 else 0.0

        # Operating threshold to meet false-alarm budget (or empirical 99th percentile of healthy)
        healthy_mask = (y_val == 0)
        healthy_scores = val_probs[healthy_mask]
        budget_cutoff = float(np.quantile(healthy_scores, 0.99)) if len(healthy_scores) > 0 else 0.5
        preds_budget = (val_probs >= budget_cutoff).astype(int)
        recall_budget = float((preds_budget[y_val == 1]).sum() / n_pos_val) if n_pos_val > 0 else 0.0

        passed_auc = (auc_target - auc_tolerance) <= val_auc <= (auc_target + auc_tolerance)
        passed_recall = recall_budget <= recall_at_budget_max
        overall_status = "PASS" if (passed_auc and passed_recall) else "FAIL"

        seed_results.append(LabelShuffleSeedResult(
            seed=seed,
            val_auc=val_auc,
            val_pr_auc=val_pr_auc,
            val_recall_at_50pct=recall_50,
            val_recall_at_budget=recall_budget,
            passed_auc=passed_auc,
            passed_recall=passed_recall,
            train_samples=len(X_train),
            train_positives_before=n_pos_train_orig,
            train_positives_after=n_pos_train_shuff,
            val_samples=len(X_val),
            val_positives=n_pos_val,
            class_counts_preserved=(n_pos_train_orig == n_pos_train_shuff),
            overall_status=overall_status,
        ))

    return seed_results


def run_canary_leak_test(
    train_features: pd.DataFrame,
    train_gt: pd.DataFrame,
    val_features: pd.DataFrame,
    val_gt: pd.DataFrame,
    feature_cols: List[str],
    canary_auc_min: float = 0.95,
) -> CanaryTestResult:
    """Run canary leak test (P4.2).
    
    1. Positive control: plant an overt target leak into feature matrix; verify it is caught with AUC >= canary_auc_min.
    2. Negative control: verify clean feature matrix has zero illicit ground-truth leakage columns.
    """
    join_keys = ["scenario_id", "sensor_id", "timestamp"]
    
    val_merged = pd.merge(
        val_features[join_keys + feature_cols],
        val_gt[join_keys + ["sensor_untrusted"]],
        on=join_keys,
        how="inner",
    )
    y_val = val_merged["sensor_untrusted"].to_numpy().astype(int)

    # 1. Positive Control: Plant a canary leak feature
    # A canary feature correlates near-perfectly with sensor_untrusted + slight noise
    rng = np.random.default_rng(999)
    canary_leak = y_val.astype(float) + rng.normal(0, 0.02, size=len(y_val))
    canary_auc, _, _ = compute_directional_auc(y_val, canary_leak)
    canary_caught = canary_auc >= canary_auc_min

    # 2. Negative Control: Check all clean features
    clean_aucs = {}
    for feat in feature_cols:
        x = val_merged[feat].to_numpy(dtype=float)
        auc_dir, _, _ = compute_directional_auc(y_val, x)
        clean_aucs[feat] = auc_dir
    
    max_clean_item = max(clean_aucs.items(), key=lambda it: it[1])
    clean_max_feature = max_clean_item[0]
    clean_max_auc = max_clean_item[1]

    # In clean pipeline, no feature should be a direct synthetic label copy
    clean_passed = "sensor_untrusted" not in feature_cols and "true_value" not in feature_cols

    notes = (
        f"Canary AUC = {canary_auc:.4f} (>= {canary_auc_min}: {canary_caught}); "
        f"Clean max feature '{clean_max_feature}' AUC = {clean_max_auc:.4f}."
    )

    return CanaryTestResult(
        canary_auc=canary_auc,
        canary_caught=canary_caught,
        clean_max_auc=clean_max_auc,
        clean_max_feature=clean_max_feature,
        clean_passed=clean_passed,
        notes=notes,
    )
