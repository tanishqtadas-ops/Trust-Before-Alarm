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
        # Class counts must be preserved
        assert r.class_counts_preserved is True
        assert r.train_positives_before == r.train_positives_after


def test_label_shuffle_preserves_class_counts(synthetic_train_val_data):
    """Verify that label-shuffle null control strictly preserves positive class counts."""
    tr_f, tr_g, val_f, val_g = synthetic_train_val_data
    seeds = [20001, 20002, 20003]

    results = run_label_shuffle_null_control(
        train_features=tr_f,
        train_gt=tr_g,
        val_features=val_f,
        val_gt=val_g,
        feature_cols=["diff_1", "slope_s"],
        seeds=seeds,
    )

    for r in results:
        assert r.train_samples == len(tr_f)
        assert r.train_positives_before == int(tr_g["sensor_untrusted"].sum())
        assert r.train_positives_after == r.train_positives_before
        assert r.class_counts_preserved is True
        assert r.val_samples == len(val_f)
        assert r.val_positives == int(val_g["sensor_untrusted"].sum())


def test_label_shuffle_rejects_overlapping_scenarios(synthetic_train_val_data):
    """Verify that run_label_shuffle_null_control raises ValueError if scenario IDs overlap."""
    tr_f, tr_g, val_f, val_g = synthetic_train_val_data
    # Create an intentional overlap
    bad_val_f = val_f.copy()
    bad_val_f.loc[0:10, "scenario_id"] = tr_f.loc[0, "scenario_id"]

    with pytest.raises(ValueError, match="strictly disjoint"):
        run_label_shuffle_null_control(
            train_features=tr_f,
            train_gt=tr_g,
            val_features=bad_val_f,
            val_gt=val_g,
            feature_cols=["diff_1", "slope_s"],
            seeds=[20001],
        )


def test_label_shuffle_strict_status_reporting(synthetic_train_val_data):
    """Verify that an AUC outside target range is strictly reported as FAIL."""
    tr_f, tr_g, val_f, val_g = synthetic_train_val_data
    # Force an impossible tolerance to guarantee failure
    results = run_label_shuffle_null_control(
        train_features=tr_f,
        train_gt=tr_g,
        val_features=val_f,
        val_gt=val_g,
        feature_cols=["diff_1", "slope_s"],
        seeds=[20001],
        auc_target=0.50,
        auc_tolerance=0.00001,  # impossibly narrow
    )
    assert len(results) == 1
    assert results[0].passed_auc is False
    assert results[0].overall_status == "FAIL"


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

