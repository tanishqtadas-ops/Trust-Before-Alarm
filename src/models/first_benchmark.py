"""First Multiclass LightGBM Benchmark for Phase P4.3.

First baseline multiclass classifier for sensor fault mode diagnosis:
- 8 classes: normal (0) + 7 fault types (1..7).
- Uses the 49 approved causal features in SENSOR_FEATURE_NAMES.
- Evaluates on validation partition only (strictly held-out test split).
- Fixed defensible configuration without hyperparameter tuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import lightgbm as lgb

from src.features.engine import SENSOR_FEATURE_NAMES


CLASS_MAPPING = {
    0: "normal",
    1: "spike",
    2: "offset_drift",
    3: "gain_error",
    4: "frozen",
    5: "high_noise",
    6: "dropout",
    7: "slow_degradation",
}

CLASS_NAME_TO_IDX = {v: k for k, v in CLASS_MAPPING.items()}
FAULT_COLUMNS = [f"f_{CLASS_MAPPING[i]}" for i in range(1, 8)]


@dataclass
class MulticlassBenchmarkMetrics:
    """Consolidated metrics for the first multiclass LightGBM benchmark."""
    model_config: Dict[str, Any]
    train_samples: int
    val_samples: int
    overall_accuracy: float
    val_log_loss: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    macro_ovr_auc: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    class_support_val: Dict[str, int]
    multi_fault_rows_count_val: int
    baseline_majority_accuracy: float
    baseline_majority_macro_f1: float


def derive_multiclass_labels(gt_df: pd.DataFrame) -> Tuple[np.ndarray, int]:
    """Derive 8-class target labels from ground truth fault indicators.
    
    Classes:
      0: normal (all f_* == 0)
      1..7: active fault. If multiple faults active, assigns the fault with highest severity.
      
    Returns:
      (labels, multi_fault_count)
    """
    n = len(gt_df)
    labels = np.zeros(n, dtype=int)
    
    # Check active faults
    active_matrix = gt_df[FAULT_COLUMNS].to_numpy().astype(int)
    fault_counts = active_matrix.sum(axis=1)
    multi_fault_count = int((fault_counts > 1).sum())

    # Single or multiple active fault assignment
    active_rows = np.where(fault_counts > 0)[0]
    
    if "fault_severity" in gt_df.columns:
        # For multiple faults, if individual severities exist, prioritize highest;
        # otherwise assign the first active fault type
        for idx in active_rows:
            active_types = np.where(active_matrix[idx] > 0)[0]
            # Map index in active_matrix (0..6) to class index (1..7)
            labels[idx] = active_types[0] + 1
    else:
        for idx in active_rows:
            active_types = np.where(active_matrix[idx] > 0)[0]
            labels[idx] = active_types[0] + 1
            
    return labels, multi_fault_count


def train_and_evaluate_multiclass_benchmark(
    train_features: pd.DataFrame,
    train_gt: pd.DataFrame,
    val_features: pd.DataFrame,
    val_gt: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_samples: Optional[int] = None,
) -> Tuple[lgb.LGBMClassifier, MulticlassBenchmarkMetrics]:
    """Train the first multiclass LightGBM benchmark on train split and evaluate on val split."""
    if feature_cols is None:
        feature_cols = [f for f in SENSOR_FEATURE_NAMES if f in train_features.columns]

    join_keys = ["scenario_id", "sensor_id", "timestamp"]

    # 1. Merge and align train data
    train_merged = pd.merge(
        train_features[join_keys + feature_cols],
        train_gt[join_keys + FAULT_COLUMNS + ["sensor_untrusted"]],
        on=join_keys,
        how="inner",
    )
    if max_train_samples and len(train_merged) > max_train_samples:
        train_merged = train_merged.sample(n=max_train_samples, random_state=42).reset_index(drop=True)

    y_train, _ = derive_multiclass_labels(train_merged)
    X_train = train_merged[feature_cols].to_numpy(dtype=float)

    # 2. Merge and align validation data
    val_merged = pd.merge(
        val_features[join_keys + feature_cols],
        val_gt[join_keys + FAULT_COLUMNS + ["sensor_untrusted"]],
        on=join_keys,
        how="inner",
    )
    y_val, multi_count_val = derive_multiclass_labels(val_merged)
    X_val = val_merged[feature_cols].to_numpy(dtype=float)

    # 3. Model Configuration: Fixed, defensible LightGBM parameters (no tuning)
    base_config = {
        "objective": "multiclass",
        "num_class": 8,
        "boosting_type": "gbdt",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1,
        "verbosity": -1,
    }
    if config_overrides:
        base_config.update(config_overrides)

    clf = lgb.LGBMClassifier(**base_config)
    clf.fit(X_train, y_train)

    # 4. Predict on validation set
    val_probs = clf.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)

    # 5. Compute metrics
    acc = float(accuracy_score(y_val, val_preds))
    try:
        loss = float(log_loss(y_val, val_probs, labels=list(range(8))))
    except Exception:
        loss = float("nan")

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, val_preds, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_val, val_preds, average="weighted", zero_division=0
    )

    # Per-class metrics
    prec_per, rec_per, f1_per, supp_per = precision_recall_fscore_support(
        y_val, val_preds, labels=list(range(8)), zero_division=0
    )

    # Multiclass One-vs-Rest ROC-AUC
    ovr_aucs = {}
    for i in range(8):
        y_binary = (y_val == i).astype(int)
        if len(np.unique(y_binary)) > 1:
            try:
                ovr_aucs[CLASS_MAPPING[i]] = float(roc_auc_score(y_binary, val_probs[:, i]))
            except Exception:
                ovr_aucs[CLASS_MAPPING[i]] = 0.5
        else:
            ovr_aucs[CLASS_MAPPING[i]] = 0.5

    macro_ovr_auc = float(np.mean(list(ovr_aucs.values())))

    per_class_dict = {}
    class_support_dict = {}
    for i in range(8):
        cname = CLASS_MAPPING[i]
        class_support_dict[cname] = int(supp_per[i])
        per_class_dict[cname] = {
            "precision": float(prec_per[i]),
            "recall": float(rec_per[i]),
            "f1": float(f1_per[i]),
            "ovr_auc": ovr_aucs[cname],
            "support": int(supp_per[i]),
        }

    cm = confusion_matrix(y_val, val_preds, labels=list(range(8))).tolist()

    # Trivial baseline (predicts class 0 always)
    preds_maj = np.zeros_like(y_val)
    maj_acc = float(accuracy_score(y_val, preds_maj))
    _, _, maj_f1, _ = precision_recall_fscore_support(
        y_val, preds_maj, average="macro", zero_division=0
    )

    metrics = MulticlassBenchmarkMetrics(
        model_config=base_config,
        train_samples=len(X_train),
        val_samples=len(X_val),
        overall_accuracy=acc,
        val_log_loss=loss,
        macro_precision=float(prec_macro),
        macro_recall=float(rec_macro),
        macro_f1=float(f1_macro),
        weighted_f1=float(f1_weighted),
        macro_ovr_auc=macro_ovr_auc,
        per_class_metrics=per_class_dict,
        confusion_matrix=cm,
        class_support_val=class_support_dict,
        multi_fault_rows_count_val=multi_count_val,
        baseline_majority_accuracy=maj_acc,
        baseline_majority_macro_f1=float(maj_f1),
    )

    return clf, metrics
