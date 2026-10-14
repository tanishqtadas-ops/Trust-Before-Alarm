"""Unit tests for Phase P4.1: Single-Feature Separability Audit."""

import numpy as np
import pandas as pd
import pytest

from src.features.engine import (
    SENSOR_FEATURE_NAMES,
    GROUP_FEATURE_NAMES,
)
from src.eval.leakage_audit import (
    evaluate_single_feature_separability,
    compute_directional_auc,
    SingleFeatureSeparabilityResult,
)


def test_forbidden_columns_not_in_model_features():
    """Verify strictly that no identifiers, timestamps, or raw values are in SENSOR_FEATURE_NAMES."""
    forbidden = {
        "scenario_id",
        "group_id",
        "sensor_id",
        "variable_type",
        "timestamp",
        "value",
        "true_value",
        "sensor_untrusted",
        "genuine_event",
    }
    for feat in SENSOR_FEATURE_NAMES:
        assert feat not in forbidden, f"Forbidden feature '{feat}' found in SENSOR_FEATURE_NAMES!"
        assert not feat.startswith("f_"), f"Ground truth fault flag '{feat}' found in SENSOR_FEATURE_NAMES!"

    for feat in GROUP_FEATURE_NAMES:
        assert feat not in forbidden, f"Forbidden feature '{feat}' found in GROUP_FEATURE_NAMES!"


def test_compute_directional_auc():
    """Verify compute_directional_auc computes two-sided separability properly."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # Positively correlated feature
    x_pos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    auc_pos, raw_pos, dir_pos = compute_directional_auc(y, x_pos)
    assert auc_pos == 1.0
    assert dir_pos == "+"

    # Negatively correlated feature
    x_neg = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    auc_neg, raw_neg, dir_neg = compute_directional_auc(y, x_neg)
    assert auc_neg == 1.0
    assert raw_neg == 0.0
    assert dir_neg == "-"

    # Uncorrelated noise
    np.random.seed(42)
    x_rand = np.random.normal(0, 1, 1000)
    y_rand = np.random.binomial(1, 0.1, 1000)
    auc_rand, _, _ = compute_directional_auc(y_rand, x_rand)
    assert 0.45 <= auc_rand <= 0.55


def test_single_feature_separability_synthetic():
    """Verify evaluate_single_feature_separability flags features exceeding 0.85 threshold."""
    np.random.seed(123)
    n = 200
    scen_ids = ["scen_01"] * n
    sensor_ids = ["s0"] * n
    timestamps = list(range(n))

    # Synthetic features: f_clean (low AUC), f_leaky (high AUC)
    y_untrusted = np.zeros(n, dtype=int)
    y_untrusted[150:] = 1  # Fault at the end

    f_clean = np.random.normal(20, 1, n)
    f_high = np.zeros(n)
    f_high[150:] = 10.0  # Perfect separator

    feat_df = pd.DataFrame({
        "scenario_id": scen_ids,
        "sensor_id": sensor_ids,
        "timestamp": timestamps,
        "diff_1": f_clean,
        "consec_missing": f_high,
    })

    gt_df = pd.DataFrame({
        "scenario_id": scen_ids,
        "sensor_id": sensor_ids,
        "timestamp": timestamps,
        "sensor_untrusted": y_untrusted,
        "f_spike": np.zeros(n, dtype=int),
        "f_offset_drift": np.zeros(n, dtype=int),
        "f_gain_error": np.zeros(n, dtype=int),
        "f_frozen": np.zeros(n, dtype=int),
        "f_high_noise": np.zeros(n, dtype=int),
        "f_dropout": y_untrusted,  # Dropout matches untrusted
        "f_slow_degradation": np.zeros(n, dtype=int),
    })

    results = evaluate_single_feature_separability(
        feature_df=feat_df,
        gt_df=gt_df,
        threshold=0.85,
        features_to_eval=["diff_1", "consec_missing"],
    )

    res_map = {r.feature_name: r for r in results}
    assert res_map["diff_1"].auc_untrusted < 0.85
    assert res_map["diff_1"].is_flagged_untrusted is False

    assert res_map["consec_missing"].auc_untrusted > 0.85
    assert res_map["consec_missing"].is_flagged_untrusted is True
    assert "dropout" in res_map["consec_missing"].task_aware_notes
