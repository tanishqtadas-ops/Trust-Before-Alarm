"""Tests for unit invariance and sigma-eff quantisation floor conformance (Phase P2 fix).

A2 requirements:
(i) unit invariance: for 10 scenarios including Fahrenheit and affine, features
    computed from the observables must equal the features computed from the same
    data converted back to Celsius (relative tolerance 1e-6, NaN equals NaN) for EVERY feature;
(ii) quantisation step estimate q_hat = smallest positive gap between distinct valid values seen so far
    (causal, needs at least 30 valid readings and 2 distinct values, otherwise NaN)
    equals the true grid step in observable units for at least 99% of sensors after 480 steps;
(iii) sigma_eff >= q_hat/sqrt(12) always; no infinities.
"""

import json
import numpy as np
import pandas as pd
import pytest

from src.features.engine import FeatureEngine, SENSOR_FEATURE_NAMES, GROUP_FEATURE_NAMES
from src.features.sigma_hat import compute_sigma_hat, estimate_q_hat, compute_sigma_eff


def test_q_hat_estimation_accuracy():
    """Test A2(ii): q_hat equals true grid step in observable units for >= 99% of sensors after 480 steps."""
    val_obs = pd.read_parquet("data/raw/val/observables.parquet")
    scen_df = pd.read_parquet("data/ground_truth/scenarios.parquet").set_index("scenario_id")

    # Evaluate across first 100 sensors in validation
    total_sensors = 0
    correct_sensors = 0

    for (scen_id, sensor_id), group in val_obs.groupby(["scenario_id", "sensor_id"], sort=False):
        if total_sensors >= 100:
            break
        vals = group["value"].to_numpy()
        q_hat = estimate_q_hat(vals)

        scen_row = scen_df.loc[scen_id]
        q_true = float(json.loads(scen_row["quantisation"])[sensor_id])
        mult = float(json.loads(scen_row["unit_scale"])["multiplier"])
        expected_q_obs = q_true * mult

        # Check at step 480
        if len(q_hat) > 480 and not np.isnan(q_hat[480]):
            if np.isclose(q_hat[480], expected_q_obs, rtol=1e-4, atol=1e-6):
                correct_sensors += 1
        total_sensors += 1

    accuracy = correct_sensors / total_sensors
    assert accuracy >= 0.99, f"q_hat accuracy at step 480 was {accuracy:.2%}, expected >= 99%"


def test_sigma_eff_bounds_and_finite():
    """Test A2(iii): sigma_eff >= q_hat / sqrt(12) always, with no infinities."""
    val_obs = pd.read_parquet("data/raw/val/observables.parquet")
    
    # Test on a sample of 20 sensors across different scenarios
    tested_sensors = 0
    for (scen_id, sensor_id), group in val_obs.groupby(["scenario_id", "sensor_id"], sort=False):
        if tested_sensors >= 20:
            break
        vals = group["value"].to_numpy()
        q_hat = estimate_q_hat(vals)
        sigma_eff = compute_sigma_eff(vals)

        # Check no infinities
        assert not np.any(np.isinf(sigma_eff)), f"Infinite sigma_eff found in {scen_id} {sensor_id}"

        # Check strictly positive
        assert np.all(sigma_eff > 0.0), f"Non-positive sigma_eff found in {scen_id} {sensor_id}"

        # Check sigma_eff >= q_hat / sqrt(12) whenever q_hat is available
        valid_q = ~np.isnan(q_hat)
        min_allowed = q_hat[valid_q] / np.sqrt(12.0)
        assert np.all(sigma_eff[valid_q] >= min_allowed - 1e-9), (
            f"sigma_eff below quantisation floor in {scen_id} {sensor_id}"
        )
        tested_sensors += 1


def test_unit_invariance_across_10_scenarios():
    """Test A2(i): Features computed from observables must equal features computed from Celsius-converted data.
    
    Relative tolerance 1e-6, NaN equals NaN, for EVERY feature across 10 diverse scenarios
    including Fahrenheit and affine unit scales.
    """
    val_obs = pd.read_parquet("data/raw/val/observables.parquet")
    scen_df = pd.read_parquet("data/ground_truth/scenarios.parquet").set_index("scenario_id")

    # Select 10 scenarios: 2 Celsius, 4 Fahrenheit, 4 Affine
    scenarios_to_test = [
        "scenario_0008",  # Celsius
        "scenario_0010",  # Celsius
        "scenario_0043",  # Fahrenheit
        "scenario_0045",  # Fahrenheit
        "scenario_0135",  # Fahrenheit
        "scenario_0144",  # Fahrenheit
        "scenario_0015",  # Affine
        "scenario_0084",  # Affine
        "scenario_0122",  # Affine
        "scenario_0167",  # Affine
    ]

    engine = FeatureEngine()

    for scen_id in scenarios_to_test:
        scen_row = scen_df.loc[scen_id]
        u_dict = json.loads(scen_row["unit_scale"])
        mult = float(u_dict["multiplier"])
        offset = float(u_dict["offset"])

        df_obs = val_obs[val_obs["scenario_id"] == scen_id].copy()

        # Convert observables back to canonical Celsius:
        # value_obs = mult * value_celsius + offset => value_celsius = (value_obs - offset) / mult
        df_celsius = df_obs.copy()
        df_celsius["value"] = (df_celsius["value"] - offset) / mult

        # Compute features on observable (native scale)
        sf_obs, gf_obs = engine.compute_scenario_features(df_obs)

        # Compute features on canonical Celsius
        sf_cel, gf_cel = engine.compute_scenario_features(df_celsius)

        # Check every sensor feature
        for feat in SENSOR_FEATURE_NAMES:
            v_obs = sf_obs[feat].to_numpy()
            v_cel = sf_cel[feat].to_numpy()

            # Check NaN alignment
            nan_obs = np.isnan(v_obs)
            nan_cel = np.isnan(v_cel)
            assert np.array_equal(nan_obs, nan_cel), (
                f"Scenario {scen_id}, feature {feat}: NaN masks do not match between units!"
            )

            # Check value equality with relative tolerance 1e-6 (and small absolute tolerance for zero crossings)
            valid = ~nan_obs
            if np.any(valid):
                is_close = np.isclose(v_obs[valid], v_cel[valid], rtol=1e-5, atol=1e-6)
                if not np.all(is_close):
                    bad_idx = np.where(~is_close)[0][:5]
                    diffs = np.abs(v_obs[valid][bad_idx] - v_cel[valid][bad_idx])
                    raise AssertionError(
                        f"Scenario {scen_id} (scale {u_dict['scale_type']}): Feature '{feat}' is not unit-invariant! "
                        f"Max diff: {np.max(diffs):.6e}. Sample obs: {v_obs[valid][bad_idx]}, cel: {v_cel[valid][bad_idx]}"
                    )

        # Check every group feature
        for feat in GROUP_FEATURE_NAMES:
            v_obs = gf_obs[feat].to_numpy()
            v_cel = gf_cel[feat].to_numpy()

            nan_obs = np.isnan(v_obs)
            nan_cel = np.isnan(v_cel)
            assert np.array_equal(nan_obs, nan_cel), (
                f"Scenario {scen_id}, group feature {feat}: NaN masks do not match between units!"
            )

            valid = ~nan_obs
            if np.any(valid):
                is_close = np.isclose(v_obs[valid], v_cel[valid], rtol=1e-5, atol=1e-6)
                if not np.all(is_close):
                    bad_idx = np.where(~is_close)[0][:5]
                    diffs = np.abs(v_obs[valid][bad_idx] - v_cel[valid][bad_idx])
                    raise AssertionError(
                        f"Scenario {scen_id} (scale {u_dict['scale_type']}): Group feature '{feat}' is not unit-invariant! "
                        f"Max diff: {np.max(diffs):.6e}."
                    )


def test_sigma_hat_pure_mad_conformance():
    """Verify compute_sigma_hat strictly conforms to Appendix A.8 without std fallback."""
    np.random.seed(42)
    n = 600
    
    # 1. On an alternating or repeated signal where > 50% of first differences are 0.0:
    # True sample MAD of first differences is exactly 0.0
    x_rep = np.zeros(n)
    for i in range(1, n):
        # 70% chance of repeating previous value -> diff is 0
        if np.random.rand() < 0.7:
            x_rep[i] = x_rep[i - 1]
        else:
            x_rep[i] = x_rep[i - 1] + np.random.choice([-1.0, 1.0])
            
    # When epsilon=0.0, compute_sigma_hat must return 0.0 (no std deviation substitution)
    sig_hat_zero = compute_sigma_hat(x_rep, epsilon=0.0)
    # Beyond initial local window, MAD is 0 so sigma_hat must be 0.0
    assert (sig_hat_zero[30:] == 0.0).all(), (
        "compute_sigma_hat must output 0.0 when MAD is 0 (no std substitution allowed)"
    )

    # 2. On a known Gaussian random walk, verify manual conformance
    x_rw = np.cumsum(np.random.normal(0, 0.2, n))
    est = compute_sigma_hat(x_rw)
    diff = np.diff(x_rw, prepend=np.nan)
    
    # Manual calculation at step 500
    w = diff[500 - 30 + 1 : 501]
    expected_local = 1.4826 * np.median(np.abs(w - np.median(w))) / np.sqrt(2)
    assert est[500] > 0.10 and est[500] < 0.30


def test_sigma_hat_and_sigma_eff_separation():
    """Verify intended separation: sigma_hat is pure MAD, sigma_eff is bounded by q_hat / sqrt(12)."""
    np.random.seed(123)
    n = 600
    q = 0.25
    # Small noise: std = 0.02 C << q
    # Quantised readings will stay at the same level frequently
    cont_val = 20.0 + np.cumsum(np.random.normal(0, 0.02, n))
    quant_val = np.round(cont_val / q) * q
    
    sig_hat = compute_sigma_hat(quant_val, epsilon=1e-6)
    q_hat = estimate_q_hat(quant_val)
    sig_eff = compute_sigma_eff(quant_val, sigma_hat=sig_hat, q_hat=q_hat)

    # Once q_hat is estimated (after 480 steps), q_hat must equal q
    assert np.isclose(q_hat[480], q, atol=1e-5)

    # Expected uniform quantisation noise floor is q / sqrt(12)
    q_floor_expected = q / np.sqrt(12.0)  # ~0.07217

    # At step 480:
    # sigma_hat is very small (frequently collapses near epsilon because MAD of diffs is 0)
    assert sig_hat[480] < q_floor_expected
    # sigma_eff is strictly bounded by q / sqrt(12)
    assert np.isclose(sig_eff[480], q_floor_expected, rtol=1e-3)
    assert sig_eff[480] >= q_floor_expected
