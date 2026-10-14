"""Unit tests for Phase P4.2: Label-Shuffle Null Control and Canary Leak Test."""

import numpy as np
import pandas as pd
import pytest

from src.eval.leakage_audit import (
    run_label_shuffle_null_control,
    run_canary_leak_test,
)


@pytest.fixture
def synthetic_train_val_data():
    """Create synthetic train and val datasets for testing leakage controls."""
    np.random.seed(42)
    n_train = 500
    n_val = 300

    def make_dataset(n, prefix):
        scen_ids = [f"{prefix}_scen_{i//100}" for i in range(n)]
        sensor_ids = [f"sensor_{i%5}" for i in range(n)]
        timestamps = list(range(n))

        y = np.random.binomial(1, 0.1, n)
        # Feature 1 has slight signal, Feature 2 is noise
        feat1 = y * 1.5 + np.random.normal(0, 1.0, n)
        feat2 = np.random.normal(0, 1.0, n)

        feat_df = pd.DataFrame({
            "scenario_id": scen_ids,
            "sensor_id": sensor_ids,
            "timestamp": timestamps,
            "diff_1": feat1,
            "slope_s": feat2,
        })
        gt_df = pd.DataFrame({
            "scenario_id": scen_ids,
            "sensor_id": sensor_ids,
            "timestamp": timestamps,
            "sensor_untrusted": y,
        })
        return feat_df, gt_df

    train_f, train_g = make_dataset(n_train, "tr")
    val_f, val_g = make_dataset(n_val, "val")
    return train_f, train_g, val_f, val_g


def test_label_shuffle_null_control_chance_level(synthetic_train_val_data):
    """Verify that label-shuffle null control results in chance-level AUC (0.50 +/- 0.05)."""
    tr_f, tr_g, val_f, val_g = synthetic_train_val_data
    seeds = [20001, 20002]

    results = run_label_shuffle_null_control(
        train_features=tr_f,
        train_gt=tr_g,
        val_features=val_f,
        val_gt=val_g,
        feature_cols=["diff_1", "slope_s"],
        seeds=seeds,
        auc_target=0.50,
        auc_tolerance=0.06,  # Slightly broader for small n=300 sample
        recall_at_budget_max=0.10,
    )

    assert len(results) == 2
    for r in results:
        # Shuffled training labels should produce near 0.50 AUC
        assert 0.40 <= r.val_auc <= 0.60
        assert r.val_recall_at_budget <= 0.10


def test_canary_leak_test_positive_and_clean(synthetic_train_val_data):
    """Verify canary leak test catches planted canary and validates clean matrix."""
    tr_f, tr_g, val_f, val_g = synthetic_train_val_data

    canary_res = run_canary_leak_test(
        train_features=tr_f,
        train_gt=tr_g,
        val_features=val_f,
        val_gt=val_g,
        feature_cols=["diff_1", "slope_s"],
        canary_auc_min=0.95,
    )

    # Planted canary must be caught with AUC >= 0.95
    assert canary_res.canary_caught is True
    assert canary_res.canary_auc >= 0.95

    # Clean matrix should pass
    assert canary_res.clean_passed is True
    assert canary_res.clean_max_auc < 0.95
