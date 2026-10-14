"""Unit tests for Phase P4.3: First Multiclass LightGBM Benchmark."""

import numpy as np
import pandas as pd
import pytest

from src.models.first_benchmark import (
    CLASS_MAPPING,
    FAULT_COLUMNS,
    derive_multiclass_labels,
    train_and_evaluate_multiclass_benchmark,
)


def test_derive_multiclass_labels():
    """Verify derive_multiclass_labels maps 0..7 correctly."""
    n = 10
    gt_data = {
        "scenario_id": ["scen_0"] * n,
        "sensor_id": ["s0"] * n,
        "timestamp": list(range(n)),
        "fault_severity": [0.0, 0.5, 0.8, 0.2, 0.4, 0.9, 0.1, 0.3, 0.6, 0.7],
    }
    for col in FAULT_COLUMNS:
        gt_data[col] = np.zeros(n, dtype=int)

    # Row 0: normal
    # Row 1: spike (class 1)
    gt_data["f_spike"][1] = 1
    # Row 2: offset_drift (class 2)
    gt_data["f_offset_drift"][2] = 1
    # Row 3: gain_error (class 3)
    gt_data["f_gain_error"][3] = 1
    # Row 4: frozen (class 4)
    gt_data["f_frozen"][4] = 1
    # Row 5: high_noise (class 5)
    gt_data["f_high_noise"][5] = 1
    # Row 6: dropout (class 6)
    gt_data["f_dropout"][6] = 1
    # Row 7: slow_degradation (class 7)
    gt_data["f_slow_degradation"][7] = 1
    # Row 8: multi-fault (spike + dropout)
    gt_data["f_spike"][8] = 1
    gt_data["f_dropout"][8] = 1

    gt_df = pd.DataFrame(gt_data)
    labels, multi_count = derive_multiclass_labels(gt_df)

    assert labels[0] == 0  # normal
    assert labels[1] == 1  # spike
    assert labels[2] == 2  # offset_drift
    assert labels[3] == 3  # gain_error
    assert labels[4] == 4  # frozen
    assert labels[5] == 5  # high_noise
    assert labels[6] == 6  # dropout
    assert labels[7] == 7  # slow_degradation
    assert labels[8] in (1, 6)  # multi-fault assigned to active class
    assert multi_count == 1


def test_train_and_evaluate_multiclass_benchmark_synthetic():
    """Verify multiclass LightGBM trains and computes complete metrics without error."""
    np.random.seed(42)
    n_train = 400
    n_val = 200

    def make_data(n):
        scen_ids = [f"scen_{i//50}" for i in range(n)]
        sensor_ids = [f"s_{i%4}" for i in range(n)]
        timestamps = list(range(n))

        y = np.random.choice(8, size=n, p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.03, 0.04, 0.03])
        # Informative synthetic features
        f1 = y * 1.0 + np.random.normal(0, 0.5, n)
        f2 = np.random.normal(0, 1.0, n)

        feat_df = pd.DataFrame({
            "scenario_id": scen_ids,
            "sensor_id": sensor_ids,
            "timestamp": timestamps,
            "diff_1": f1,
            "slope_s": f2,
        })

        gt_dict = {
            "scenario_id": scen_ids,
            "sensor_id": sensor_ids,
            "timestamp": timestamps,
            "sensor_untrusted": (y > 0).astype(int),
        }
        for i in range(1, 8):
            gt_dict[f"f_{CLASS_MAPPING[i]}"] = (y == i).astype(int)

        gt_df = pd.DataFrame(gt_dict)
        return feat_df, gt_df

    tr_f, tr_g = make_data(n_train)
    val_f, val_g = make_data(n_val)

    clf, metrics = train_and_evaluate_multiclass_benchmark(
        train_features=tr_f,
        train_gt=tr_g,
        val_features=val_f,
        val_gt=val_g,
        feature_cols=["diff_1", "slope_s"],
        config_overrides={"n_estimators": 20, "num_leaves": 15},
    )

    assert 0.0 <= metrics.overall_accuracy <= 1.0
    assert metrics.macro_f1 > 0.0
    assert len(metrics.per_class_metrics) == 8
    assert len(metrics.confusion_matrix) == 8
    assert len(metrics.confusion_matrix[0]) == 8
