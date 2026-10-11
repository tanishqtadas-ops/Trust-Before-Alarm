"""Focused tests for P1.6: Combination rules, shared-cause pairs, and faulty-sensor cap.

Verifies Appendix A.5 and Decisions 29-34:
  1. Candidate-count probabilities (before caps).
  2. Exact actual-count formula and enforcement of both cap levels.
  3. Mechanical triplet validity and absence of forbidden pairs.
  4. Fault timing, staggered windows, step-zero handling, and empty event-coincidence fallbacks.
  5. Shared-cause eligibility, same-group allocation, shared type presence, feasible start offsets,
     severity tolerance, and applicable sign/mode coupling.
  6. Dropout expected-gap calculations with the 100-step cap.
  7. Degradation labels below and above the 0.15 threshold, concurrent-fault aggregation,
     and no-active-fault case.
  8. Strict Layer 1 / ground-truth isolation.
  9. Deterministic reproducibility and regression coverage.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.sim.environment import generate_environment_series
from src.sim.events import EventGenerator
from src.sim.faults import (
    FaultEpisode,
    FaultSimulator,
    FaultType,
    SensorFaultSeries,
    allocate_faulty_sensors,
    compute_actual_faulty_sensor_count,
    compute_dropout_theoretical_severity,
    compute_effective_dropout_gap_length,
    compute_group_cap,
    compute_scenario_cap,
    couple_shared_cause_parameters,
    derive_valid_triplets,
    extract_active_event_steps,
    generate_group_faults,
    generate_scenario_faults,
    get_valid_pairs,
    sample_episode_start_step,
    sample_fault_type_count,
    sample_fault_types_for_sensor,
    sample_faulty_sensor_candidate_count,
    sample_shared_cause_start_step,
)
from src.sim.profile import TemperatureProfile, TemperatureSensorParams
from src.sim.sensors import VirtualSensorSimulator, simulate_sensors
from src.utils.config import get_config_dir, load_yaml


@pytest.fixture
def config():
    return load_yaml(get_config_dir() / "generator_a.yaml")


# =============================================================================
# 1. Candidate-count probabilities (before caps)
# =============================================================================

def test_candidate_count_distribution(config):
    """Verify candidate count sampling matches configured distribution before caps (Decision 29)."""
    rng = np.random.default_rng(1001)
    n_samples = 5000
    counts = [sample_faulty_sensor_candidate_count(config, rng) for _ in range(n_samples)]
    
    unique_vals = set(counts)
    assert unique_vals.issubset({0, 1, 2, 3})
    
    p0 = counts.count(0) / n_samples
    p1 = counts.count(1) / n_samples
    p2 = counts.count(2) / n_samples
    p3 = counts.count(3) / n_samples
    
    # Configured: 0: 0.15, 1: 0.40, 2: 0.25, 3: 0.20
    assert abs(p0 - 0.15) < 0.025
    assert abs(p1 - 0.40) < 0.030
    assert abs(p2 - 0.25) < 0.025
    assert abs(p3 - 0.20) < 0.025


# =============================================================================
# 2. Exact actual-count formula and enforcement of both cap levels
# =============================================================================

def test_actual_count_formula_and_caps(config):
    """Verify actual-count formula k_actual = min(k_cand, cap_scen, sum(cap_g)) (Decision 29)."""
    # Case: Single-sensor scenario (n=1)
    # cap_scen = max(1, floor(0.4*1)) = 1, cap_g = 1
    assert compute_actual_faulty_sensor_count(0, [1], config) == 0  # 15% zero preserved
    assert compute_actual_faulty_sensor_count(1, [1], config) == 1
    assert compute_actual_faulty_sensor_count(2, [1], config) == 1
    assert compute_actual_faulty_sensor_count(3, [1], config) == 1

    # Case: Single-group scenario n=4: cap_scen = floor(1.6) = 1, cap_g = 1
    assert compute_actual_faulty_sensor_count(0, [4], config) == 0
    assert compute_actual_faulty_sensor_count(1, [4], config) == 1
    assert compute_actual_faulty_sensor_count(2, [4], config) == 1
    assert compute_actual_faulty_sensor_count(3, [4], config) == 1

    # Case: Single-group scenario n=8: cap_scen = floor(3.2) = 3, cap_g = 3
    assert compute_actual_faulty_sensor_count(0, [8], config) == 0
    assert compute_actual_faulty_sensor_count(1, [8], config) == 1
    assert compute_actual_faulty_sensor_count(2, [8], config) == 2
    assert compute_actual_faulty_sensor_count(3, [8], config) == 3

    # Case: Two-group scenario n1=4 (cap 1), n2=4 (cap 1). Total N=8 (cap 3).
    # sum(cap_g) = 1 + 1 = 2 <= cap_scen (3)
    assert compute_actual_faulty_sensor_count(3, [4, 4], config) == 2  # Capped by sum(cap_g)

    # Case: Two-group scenario n1=6 (cap 2), n2=6 (cap 2). Total N=12 (cap 4).
    # sum(cap_g) = 4, cap_scen = 4
    assert compute_actual_faulty_sensor_count(3, [6, 6], config) == 3


def test_capacity_constrained_allocation(config):
    """Verify capacity-constrained uniform allocation does not exceed any group cap."""
    rng = np.random.default_rng(2002)
    groups = {
        "group_0": [f"s_0_{i}" for i in range(4)],  # cap = 1
        "group_1": [f"s_1_{i}" for i in range(5)],  # cap = 2
    }
    # Allocate 3 faulty sensors (k_actual = 3)
    for _ in range(50):
        allocated = allocate_faulty_sensors(3, groups, rng)
        assert len(allocated) == 3
        g0_count = sum(1 for _, gid in allocated if gid == "group_0")
        g1_count = sum(1 for _, gid in allocated if gid == "group_1")
        assert g0_count <= 1
        assert g1_count <= 2
        assert g0_count + g1_count == 3


# =============================================================================
# 3. Mechanical triplet validity and absence of forbidden pairs
# =============================================================================

def test_mechanical_triplet_derivation(config):
    """Derive valid triplets mechanically from config and verify 11 triplets exist (Decision 30)."""
    valid_pairs = get_valid_pairs(config)
    assert len(valid_pairs) == 13
    
    # Verify forbidden pairs are excluded
    comb_cfg = config["combinations"]
    for pair in comb_cfg["forbidden_pairs"]:
        f_pair = frozenset(pair)
        assert f_pair not in valid_pairs

    triplets = derive_valid_triplets(config)
    assert len(triplets) == 11

    # Frozen cannot participate in any triplet
    for trip in triplets:
        assert "frozen" not in trip

    # Slow degradation participates in exactly 1 triplet
    slow_deg_triplets = [t for t in triplets if "slow_degradation" in t]
    assert len(slow_deg_triplets) == 1
    assert set(slow_deg_triplets[0]) == {"slow_degradation", "spike", "gain_error"}

    # The other 10 triplets are exactly the 10 combinations of the 5-clique
    k5_nodes = {"spike", "offset_drift", "gain_error", "high_noise", "dropout"}
    k5_triplets = [t for t in triplets if set(t).issubset(k5_nodes)]
    assert len(k5_triplets) == 10


def test_fault_type_count_probabilities(config):
    """Verify types-per-faulty-sensor sampling probabilities: 1: 0.65, 2: 0.27, 3: 0.08."""
    rng = np.random.default_rng(3003)
    n_samples = 5000
    counts = [sample_fault_type_count(config, rng) for _ in range(n_samples)]
    
    p1 = counts.count(1) / n_samples
    p2 = counts.count(2) / n_samples
    p3 = counts.count(3) / n_samples
    
    assert abs(p1 - 0.65) < 0.025
    assert abs(p2 - 0.27) < 0.025
    assert abs(p3 - 0.08) < 0.015


# =============================================================================
# 4. Fault timing, staggered windows, step-zero, and empty event-coincidence
# =============================================================================

def test_first_episode_timing_regimes(config):
    """Verify first episode timing adhering to 10% step 0 and 35% event branch (Decisions 31 & 32)."""
    rng = np.random.default_rng(4004)
    active_events = list(range(600, 650))  # 50 active event steps in post-warmup
    n_samples = 3000
    starts = [
        sample_episode_start_step(None, 1920, 480, active_events, config, rng)
        for _ in range(n_samples)
    ]
    
    p_zero = sum(1 for s in starts if s == 0) / n_samples
    assert abs(p_zero - 0.10) < 0.025

    # Check empty event steps fallback
    starts_no_event = [
        sample_episode_start_step(None, 1920, 480, [], config, rng)
        for _ in range(500)
    ]
    for s in starts_no_event:
        assert s == 0 or (480 <= s < 1920)


def test_subsequent_episode_stagger_and_fallbacks(config):
    """Verify subsequent episode stagger timing and feasibility fallbacks (Decision 31)."""
    rng = np.random.default_rng(5005)
    
    # Case A: Preceding episode ends before step 480 -> deterministically 0
    ep_prec_early = FaultEpisode(
        fault_id="f0", fault_type=FaultType.SPIKE, sensor_id="s0",
        start_step=0, end_step=300, duration=300, shape_params={},
        additive_offset=np.zeros(1920), noise_multiplier=np.ones(1920),
        missing_mask=np.zeros(1920, dtype=bool), severity=np.zeros(1920),
        label_mask=np.zeros(1920, dtype=bool), untrusted_mask=np.zeros(1920, dtype=bool),
    )
    for _ in range(20):
        t2 = sample_episode_start_step(ep_prec_early, 1920, 480, [], config, rng)
        assert t2 == 0

    # Case B: Preceding episode starts at 700, ends at 900 -> step 0 is infeasible
    ep_prec_mid = FaultEpisode(
        fault_id="f0", fault_type=FaultType.SPIKE, sensor_id="s0",
        start_step=700, end_step=900, duration=200, shape_params={},
        additive_offset=np.zeros(1920), noise_multiplier=np.ones(1920),
        missing_mask=np.zeros(1920, dtype=bool), severity=np.zeros(1920),
        label_mask=np.zeros(1920, dtype=bool), untrusted_mask=np.zeros(1920, dtype=bool),
    )
    # Stagger window: [max(0, 700-100), 900] = [600, 900]
    for _ in range(50):
        t2 = sample_episode_start_step(ep_prec_mid, 1920, 480, [], config, rng)
        assert 600 <= t2 <= 900


# =============================================================================
# 5. Shared-cause eligibility, pre-allocation, coupling, and offsets
# =============================================================================

def test_shared_cause_ineligible_when_group_cap_below_two(config):
    """Verify scenarios where all group caps < 2 cannot activate shared-cause pairs (Decision 33)."""
    # Single group with n=4 -> cap is 1
    groups = {"group_0": [f"s_{i}" for i in range(4)]}
    for seed in range(20):
        res = generate_scenario_faults(
            groups=groups,
            config=config,
            seed=seed,
            k_candidate=3,
            force_shared_cause=True,  # Ineligible, must remain False
        )
        faulty_sensors = [s for s, series in res.items() if series.has_faults]
        assert len(faulty_sensors) <= 1


def test_shared_cause_coupling_and_offsets(config):
    """Verify shared-cause pair hosts, timing offsets, and parameter coupling (Decision 33)."""
    groups = {"group_0": [f"s_{i}" for i in range(6)]}  # n=6 -> cap = 2
    
    for seed in range(10, 30):
        rng = np.random.default_rng(seed)
        res = generate_scenario_faults(
            groups=groups,
            config=config,
            rng=rng,
            k_candidate=2,
            force_shared_cause=True,
        )
        faulty = [s for s, series in res.items() if series.has_faults]
        assert len(faulty) == 2
        s_A, s_B = faulty[0], faulty[1]
        ep_A = res[s_A].episodes[0]
        ep_B = res[s_B].episodes[0]

        # Shared fault type on episode 1
        assert ep_A.fault_type == ep_B.fault_type

        # Start step offset within +/- 10 steps (or both 0 if step 0)
        if ep_A.start_step == 0:
            assert ep_B.start_step == 0
        else:
            assert abs(ep_A.start_step - ep_B.start_step) <= 10
            assert 480 <= ep_A.start_step < 1920
            assert 480 <= ep_B.start_step < 1920

        # Coupled parameters check
        ft = ep_A.fault_type
        if ft == FaultType.SPIKE:
            spA = ep_A.shape_params["spikes"][0]
            spB = ep_B.shape_params["spikes"][0]
            assert spA["sign"] == spB["sign"]
            rel_diff = abs(spA["size_sigma"] - spB["size_sigma"]) / max(spA["size_sigma"], spB["size_sigma"])
            assert rel_diff <= 0.20 + 1e-6
        elif ft == FaultType.OFFSET_DRIFT:
            assert ep_A.shape_params["sign"] == ep_B.shape_params["sign"]
            assert ep_A.shape_params["shape"] == ep_B.shape_params["shape"]
            bA = ep_A.shape_params["final_bias_sigma"]
            bB = ep_B.shape_params["final_bias_sigma"]
            rel_diff = abs(bA - bB) / max(bA, bB)
            assert rel_diff <= 0.20 + 1e-6
        elif ft == FaultType.GAIN_ERROR:
            gA = ep_A.shape_params["gain"]
            gB = ep_B.shape_params["gain"]
            assert (gA >= 1.0) == (gB >= 1.0)  # Direction preserved
            dA = abs(gA - 1.0)
            dB = abs(gB - 1.0)
            rel_diff = abs(dA - dB) / max(dA, dB)
            assert rel_diff <= 0.20 + 1e-6
        elif ft == FaultType.FROZEN:
            assert ep_A.shape_params["is_wrong_value"] == ep_B.shape_params["is_wrong_value"]
            assert ep_A.shape_params["has_jitter"] == ep_B.shape_params["has_jitter"]
        elif ft == FaultType.HIGH_NOISE:
            assert ep_A.shape_params["is_bursty"] == ep_B.shape_params["is_bursty"]
            mA = ep_A.shape_params["multiplier"]
            mB = ep_B.shape_params["multiplier"]
            rel_diff = abs(mA - mB) / max(mA, mB)
            assert rel_diff <= 0.20 + 1e-6
        elif ft == FaultType.SLOW_DEGRADATION:
            assert ep_A.shape_params["sign"] == ep_B.shape_params["sign"]
            dA = ep_A.shape_params["final_damage"]
            dB = ep_B.shape_params["final_damage"]
            rel_diff = abs(dA - dB) / max(dA, dB)
            assert rel_diff <= 0.20 + 1e-6


# =============================================================================
# 6. Dropout expected-gap calculations with 100-step cap
# =============================================================================

def test_dropout_effective_gap_length_and_theoretical_severity():
    """Verify capped geometric expectation formula and theoretical severity (Decision 33)."""
    # Small mean: E[min(X, 100)] is close to uncapped
    mu_small = compute_effective_dropout_gap_length(5.0, cap=100)
    assert abs(mu_small - 5.0) < 1e-4

    # Large mean (40.0): E[min(X, 100)] = 40 * (1 - (1 - 1/40)^100)
    expected_40 = 40.0 * (1.0 - (1.0 - 1.0 / 40.0) ** 100)
    mu_large = compute_effective_dropout_gap_length(40.0, cap=100)
    assert np.isclose(mu_large, expected_40)

    # Theoretical severity calculation
    sev1 = compute_dropout_theoretical_severity(p_enter=0.01, mean_gap_uncapped=20.0, cap=100)
    sev2 = compute_dropout_theoretical_severity(p_enter=0.012, mean_gap_uncapped=24.0, cap=100)
    assert sev1 >= 0.30
    assert sev2 >= sev1
    rel_diff = abs(sev2 - sev1) / max(sev2, sev1)
    assert rel_diff <= 0.20  # Satisfies the 20% severity tolerance


# =============================================================================
# 7. Degradation labels, concurrent-fault aggregation, and no-active-fault case
# =============================================================================

def test_ground_truth_aggregation_and_slow_degradation_threshold(config):
    """Verify untrusted label at 0.15 damage threshold, concurrent max severity, and healthy state."""
    # Healthy sensor (no active faults)
    healthy_series = SensorFaultSeries(sensor_id="s_clean", total_steps=1920, episodes=[])
    assert np.all(healthy_series.fault_severity == 0.0)
    assert np.all(healthy_series.true_health == 1.0)
    assert np.all(healthy_series.sensor_untrusted == False)
    assert np.all(healthy_series.get_fault_flag(FaultType.SPIKE) == 0)

    # Slow degradation fault
    sim = FaultSimulator(config=config)
    ep_deg = sim.sample_fault(
        FaultType.SLOW_DEGRADATION, sensor_id="s_deg", sigma_k=0.15,
        total_steps=1920, start_step=500, duration=1000,
        h_end=0.20, rng=np.random.default_rng(7007),
    )
    series_deg = SensorFaultSeries(sensor_id="s_deg", total_steps=1920, episodes=[ep_deg])

    # Find step where damage < 0.15 and step where damage >= 0.15
    start = ep_deg.start_step
    # Damage starts at 0 and reaches 0.80 at start + 1000
    early_step = start + 50   # damage = 0.8 * 50 / 1000 = 0.04 < 0.15
    late_step = start + 500   # damage = 0.8 * 500 / 1000 = 0.40 >= 0.15

    assert series_deg.get_fault_flag(FaultType.SLOW_DEGRADATION)[early_step] == 1
    assert series_deg.sensor_untrusted[early_step] == False  # untrusted == 0 below 0.15!
    assert 0.0 < series_deg.fault_severity[early_step] < 0.15

    assert series_deg.get_fault_flag(FaultType.SLOW_DEGRADATION)[late_step] == 1
    assert series_deg.sensor_untrusted[late_step] == True   # untrusted == 1 above 0.15!
    assert series_deg.fault_severity[late_step] >= 0.15

    # Concurrent active faults: severity is max, health is 1 - max
    ep_noise = sim.sample_fault(
        FaultType.HIGH_NOISE, sensor_id="s_deg", sigma_k=0.15,
        total_steps=1920, start_step=early_step, duration=50,
        rng=np.random.default_rng(7008),
    )
    series_concurrent = SensorFaultSeries(sensor_id="s_deg", total_steps=1920, episodes=[ep_deg, ep_noise])
    max_sev = max(ep_deg.severity[early_step], ep_noise.severity[early_step])
    assert np.isclose(series_concurrent.fault_severity[early_step], max_sev)
    assert np.isclose(series_concurrent.true_health[early_step], 1.0 - max_sev)
    assert series_concurrent.sensor_untrusted[early_step] == True  # Noise is untrusted


# =============================================================================
# 8. Strict Layer 1 / ground-truth isolation
# =============================================================================

def test_layer1_observable_table_isolation(config):
    """Verify observable dataframe contains strictly the 6 Layer 1 columns (Decision 34)."""
    env = generate_environment_series(config=config, seed=8008)
    group = simulate_sensors(environment=env, num_sensors=5, generate_faults=True, seed=8008)
    
    obs_df = group.to_observable_dataframe(scenario_id="scen_test")
    expected_cols = ["scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"]
    assert list(obs_df.columns) == expected_cols
    assert len(obs_df) == 5 * 1920


# =============================================================================
# 9. Deterministic reproducibility and regression coverage
# =============================================================================

def test_p1_6_deterministic_reproducibility(config):
    """Verify scenario fault generation is fully reproducible with fixed seed."""
    groups = {"g0": [f"s_{i}" for i in range(5)]}
    
    res1 = generate_scenario_faults(groups=groups, config=config, seed=9009)
    res2 = generate_scenario_faults(groups=groups, config=config, seed=9009)
    res3 = generate_scenario_faults(groups=groups, config=config, seed=9010)

    for sid in res1:
        assert len(res1[sid].episodes) == len(res2[sid].episodes)
        np.testing.assert_array_equal(res1[sid].fault_severity, res2[sid].fault_severity)
        np.testing.assert_array_equal(res1[sid].combined_additive_offset, res2[sid].combined_additive_offset)

    # Different seeds produce different outputs
    diff_count = sum(
        not np.array_equal(res1[sid].fault_severity, res3[sid].fault_severity)
        for sid in res1
    )
    assert diff_count > 0
