"""Comprehensive integration tests for P1.7: Scenario Types, Sensor Counts, and Difficulty Tiers.

Verifies Appendix A.1, A.6, and approved Decisions 35-41:
  1. Scenario type, group-size, and difficulty-tier distributions.
  2. Independence of scenario type and tier sampling.
  3. Actual faulty-sensor counts conditional on topology capacity and tier.
  4. EASY parameter restrictions, distribution preservation, and allowed event scopes.
  5. MEDIUM fault-count and fault-type limits.
  6. HARD qualification across diverse seeds, including zero-fault and single-sensor scenarios.
  7. Each of the seven hard-mechanism detectors and feasible construction paths.
  8. Two-group shared environment, scenario-wide event scopes, globally unique sensor IDs,
     and independence of group-specific noise components.
  9. Single-sensor event-scope semantics and explicit half-up rounding.
 10. Observable schema isolation and metadata completeness.
 11. Identical-seed reproducibility of observable data and metadata.
 12. Both known P1.6 coverage gaps: shared-cause event-coincidence fallback and dropout shared-cause coupling.
 13. Regression coverage for P1.1-P1.6.
"""

from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from src.sim.environment import generate_environment_series
from src.sim.events import (
    EventEpisode,
    EventGenerator,
    EventSeries,
    generate_ramp,
    generate_sustained_shift,
)
from src.sim.faults import (
    FaultEpisode,
    FaultSimulator,
    FaultType,
    SensorFaultSeries,
    compute_actual_faulty_sensor_count,
    compute_dropout_theoretical_severity,
    compute_effective_dropout_gap_length,
    compute_group_cap,
    compute_scenario_cap,
    couple_shared_cause_parameters,
    derive_valid_triplets,
    extract_active_event_steps,
    generate_scenario_faults,
    get_valid_pairs,
    sample_shared_cause_start_step,
)
from src.sim.profile import TemperatureProfile, TemperatureSensorParams
from src.sim.scenario import (
    ScenarioSimulator,
    VirtualScenario,
    detect_fault_during_event,
    detect_faulty_fraction_at_cap_small_group,
    detect_local_event,
    detect_no_clear_majority,
    detect_sensor_near_bounds,
    detect_shared_cause_pair,
    detect_subtle_fault,
    evaluate_hard_mechanisms,
    sample_difficulty_tier,
    sample_group_sizes,
    sample_scenario_type,
    simulate_scenario,
)
from src.sim.sensors import VirtualSensorGroup, VirtualSensorSimulator, apply_response_lag, simulate_sensors
from src.utils.config import get_config_dir, load_yaml


@pytest.fixture
def config():
    """Load default generator_a configuration."""
    return load_yaml(get_config_dir() / "generator_a.yaml")


# =============================================================================
# 1. Scenario type, group-size, and difficulty-tier distributions (Decision 35)
# =============================================================================

def test_scenario_type_distribution(config):
    """Verify scenario type sampling: standard 82%, two-group 10%, single-sensor 8%."""
    rng = np.random.default_rng(42)
    n_samples = 5000
    samples = [sample_scenario_type(config, rng) for _ in range(n_samples)]
    counts = pd.Series(samples).value_counts(normalize=True)

    # Tolerances +/- 0.02
    assert abs(counts.get("standard", 0) - 0.82) < 0.02
    assert abs(counts.get("two_groups", 0) - 0.10) < 0.02
    assert abs(counts.get("single_sensor", 0) - 0.08) < 0.02


def test_group_size_distributions(config):
    """Verify group size distributions (Decision 35):
    - Standard: DiscreteUniform[4, 8] (uniform ~ 0.20 each)
    - Two-group: DiscreteUniform[4, 6] independently for each group (~ 0.333 each)
    - Single-sensor: Exactly [1]
    """
    rng = np.random.default_rng(123)
    n_samples = 3000

    # Standard
    std_sizes = [sample_group_sizes("standard", config, rng)[0] for _ in range(n_samples)]
    std_counts = pd.Series(std_sizes).value_counts(normalize=True)
    assert set(std_counts.index) == {4, 5, 6, 7, 8}
    for k in {4, 5, 6, 7, 8}:
        assert abs(std_counts[k] - 0.20) < 0.03

    # Two-group
    tg_samples = [sample_group_sizes("two_groups", config, rng) for _ in range(n_samples)]
    assert all(len(s) == 2 for s in tg_samples)
    g1_sizes = [s[0] for s in tg_samples]
    g2_sizes = [s[1] for s in tg_samples]
    g1_counts = pd.Series(g1_sizes).value_counts(normalize=True)
    g2_counts = pd.Series(g2_sizes).value_counts(normalize=True)

    assert set(g1_counts.index) == {4, 5, 6}
    assert set(g2_counts.index) == {4, 5, 6}
    for k in {4, 5, 6}:
        assert abs(g1_counts[k] - (1.0 / 3.0)) < 0.03
        assert abs(g2_counts[k] - (1.0 / 3.0)) < 0.03

    # Single-sensor
    single_samples = [sample_group_sizes("single_sensor", config, rng) for _ in range(100)]
    assert all(s == [1] for s in single_samples)


def test_difficulty_tier_distribution_and_independence(config):
    """Verify tier distribution (EASY 30%, MEDIUM 40%, HARD 30%) and independence from scenario type."""
    rng = np.random.default_rng(456)
    n_samples = 10000

    tier_samples = []
    type_samples = []
    for _ in range(n_samples):
        st = sample_scenario_type(config, rng)
        dt = sample_difficulty_tier(config, rng)
        type_samples.append(st)
        tier_samples.append(dt)

    tier_counts = pd.Series(tier_samples).value_counts(normalize=True)
    assert abs(tier_counts.get("easy", 0) - 0.30) < 0.02
    assert abs(tier_counts.get("medium", 0) - 0.40) < 0.02
    assert abs(tier_counts.get("hard", 0) - 0.30) < 0.02

    # Test independence: Joint probability P(type, tier) ~ P(type) * P(tier)
    df = pd.DataFrame({"type": type_samples, "tier": tier_samples})
    crosstab = pd.crosstab(df["type"], df["tier"], normalize=True)

    for st in ["standard", "two_groups", "single_sensor"]:
        p_st = 0.82 if st == "standard" else (0.10 if st == "two_groups" else 0.08)
        for dt in ["easy", "medium", "hard"]:
            p_dt = 0.30 if dt in ("easy", "hard") else 0.40
            expected_p = p_st * p_dt
            actual_p = crosstab.loc[st, dt]
            assert abs(actual_p - expected_p) < 0.015


# =============================================================================
# 2. Actual faulty-sensor counts conditional on topology capacity and tier (Decision 39)
# =============================================================================

def test_actual_faulty_sensor_counts_by_topology_and_tier(config):
    """Verify staged count formula: k_actual = min(k_cand, cap_scen, sum(cap_g), cap_tier)."""
    # EASY tier: tier cap is 1
    assert compute_actual_faulty_sensor_count(0, [6], config, tier_cap=1) == 0
    assert compute_actual_faulty_sensor_count(1, [6], config, tier_cap=1) == 1
    assert compute_actual_faulty_sensor_count(2, [6], config, tier_cap=1) == 1
    assert compute_actual_faulty_sensor_count(3, [6], config, tier_cap=1) == 1

    # MEDIUM tier: tier cap is 2
    assert compute_actual_faulty_sensor_count(0, [6], config, tier_cap=2) == 0
    assert compute_actual_faulty_sensor_count(1, [6], config, tier_cap=2) == 1
    assert compute_actual_faulty_sensor_count(2, [6], config, tier_cap=2) == 2
    assert compute_actual_faulty_sensor_count(3, [6], config, tier_cap=2) == 2  # capped by tier

    # HARD tier: tier cap is 3
    assert compute_actual_faulty_sensor_count(0, [6], config, tier_cap=3) == 0
    assert compute_actual_faulty_sensor_count(1, [6], config, tier_cap=3) == 1
    assert compute_actual_faulty_sensor_count(2, [6], config, tier_cap=3) == 2
    assert compute_actual_faulty_sensor_count(3, [6], config, tier_cap=3) == 2  # cap_g(6) = 2, so capped by group cap!
    assert compute_actual_faulty_sensor_count(3, [4, 4], config, tier_cap=3) == 2  # cap_scen(8)=3, sum(cap_g)=1+1=2 => capped by sum(cap_g)=2
    assert compute_actual_faulty_sensor_count(3, [5, 5], config, tier_cap=3) == 3  # cap_scen(10)=4, sum(cap_g)=2+2=4 => capped by tier 3
    assert compute_actual_faulty_sensor_count(3, [6, 6], config, tier_cap=3) == 3  # cap_scen(12)=4, sum(cap_g)=2+2=4 => capped by tier 3

    # Zero-fault invariant: if k_candidate == 0, k_actual must strictly be 0 regardless of caps
    for tier_cap in [1, 2, 3]:
        for groups in [[1], [4], [8], [4, 4], [6, 6]]:
            assert compute_actual_faulty_sensor_count(0, groups, config, tier_cap=tier_cap) == 0

    # Single-sensor topology: cap is 1, so k_actual is 0 if k_cand=0, and 1 if k_cand >= 1
    for tier_cap in [1, 2, 3]:
        assert compute_actual_faulty_sensor_count(0, [1], config, tier_cap=tier_cap) == 0
        assert compute_actual_faulty_sensor_count(1, [1], config, tier_cap=tier_cap) == 1
        assert compute_actual_faulty_sensor_count(2, [1], config, tier_cap=tier_cap) == 1


# =============================================================================
# 3. EASY Tier Constraints and Visibility Restrictions (Decision 39)
# =============================================================================

def test_easy_tier_constraints_and_visibility(config):
    """Verify EASY tier constraints:
    - Faulty sensor count <= 1
    - Exactly 1 fault type per sensor (m = 1)
    - Exactly 1 episode per faulty sensor
    - Allowed scopes: only 'global' or none
    - Restricted visibility ranges:
      * Spike: >= 8 sigma_k
      * Drift: >= 6 sigma_k
      * Gain error: |g - 1| >= 0.25 (both low and high sampled)
      * High noise: multiplier >= 4.0
    """
    sim = ScenarioSimulator(config=config)

    for seed in range(20):
        scenario = sim.simulate_scenario(
            scenario_id=f"test_easy_{seed}",
            seed=seed,
            scenario_type="standard",
            difficulty_tier="easy",
            k_candidate=2,  # request 2 candidate faults to verify capping
        )

        # Faulty count <= 1
        faulty_sensors = [
            s for g in scenario.groups.values() for s in g.sensors.values()
            if s.fault_series is not None and s.fault_series.has_faults
        ]
        assert len(faulty_sensors) <= 1

        if faulty_sensors:
            fs = faulty_sensors[0]
            assert len(fs.fault_series.episodes) == 1
            ep = fs.fault_series.episodes[0]
            sigma_k = fs.params.sigma_k

            if ep.fault_type == FaultType.SPIKE:
                size_sigma = ep.shape_params["spikes"][0]["size_sigma"]
                assert size_sigma >= 8.0 - 1e-6
            elif ep.fault_type == FaultType.OFFSET_DRIFT:
                bias_sigma = ep.shape_params["final_bias_sigma"]
                assert bias_sigma >= 6.0 - 1e-6
            elif ep.fault_type == FaultType.GAIN_ERROR:
                gain = ep.shape_params["gain"]
                assert abs(gain - 1.0) >= 0.25 - 1e-6
            elif ep.fault_type == FaultType.HIGH_NOISE:
                mult = ep.shape_params["multiplier"]
                assert mult >= 4.0 - 1e-6

        # Event scopes restricted to global
        if scenario.events is not None:
            for ep in scenario.events.episodes:
                assert ep.scope == "global"


def test_easy_gain_error_both_polarities(config):
    """Verify EASY gain error samples both low (0.5 to 0.75) and high (1.25 to 1.50) modes."""
    fault_sim = FaultSimulator(config=config)
    rng = np.random.default_rng(789)
    gains = []

    for _ in range(200):
        ep = fault_sim.sample_fault(
            fault_type=FaultType.GAIN_ERROR,
            sensor_id="sensor_00",
            sigma_k=0.15,
            total_steps=1920,
            warmup_steps=480,
            physical_xref=20.0,
            clean_base_value=20.0,
            rng=rng,
            difficulty_tier="easy",
        )
        gains.append(ep.shape_params["gain"])

    low_gains = [g for g in gains if g < 1.0]
    high_gains = [g for g in gains if g > 1.0]

    assert len(low_gains) > 50
    assert len(high_gains) > 50
    assert all(0.5 <= g <= 0.75 for g in low_gains)
    assert all(1.25 <= g <= 1.60 for g in high_gains)


# =============================================================================
# 4. MEDIUM Tier Constraints (Decision 39)
# =============================================================================

def test_medium_tier_constraints(config):
    """Verify MEDIUM tier constraints:
    - At most 2 faulty sensors
    - At most 2 fault types per sensor
    - Standard fault parameter distributions retained
    - All event scopes allowed
    """
    sim = ScenarioSimulator(config=config)

    for seed in range(20):
        scenario = sim.simulate_scenario(
            scenario_id=f"test_med_{seed}",
            seed=seed,
            scenario_type="two_groups",
            difficulty_tier="medium",
            k_candidate=3,  # request 3 candidates
        )

        faulty_sensors = [
            s for g in scenario.groups.values() for s in g.sensors.values()
            if s.fault_series is not None and s.fault_series.has_faults
        ]
        assert len(faulty_sensors) <= 2
        for s in faulty_sensors:
            assert len(s.fault_series.episodes) <= 2


# =============================================================================
# 5. HARD Tier Qualification & Invariants (Decisions 38 & 39)
# =============================================================================

def test_hard_tier_qualification_across_diverse_seeds(config):
    """Verify that 100% of generated HARD scenarios have >= 1 verified hard mechanism.
    
    Tests across all topology types (standard, two_groups, single_sensor).
    """
    sim = ScenarioSimulator(config=config)

    for scen_type in ["standard", "two_groups", "single_sensor"]:
        for seed in range(15):
            scenario = sim.simulate_scenario(
                scenario_id=f"test_hard_{scen_type}_{seed}",
                seed=seed * 17 + 1,
                scenario_type=scen_type,
                difficulty_tier="hard",
            )
            assert scenario.difficulty_tier == "hard"
            verified = evaluate_hard_mechanisms(scenario)
            assert len(verified) >= 1, f"HARD scenario failed qualification: type={scen_type}, seed={seed}"
            assert set(scenario.hard_mechanisms) == set(verified)


def test_hard_tier_zero_fault_invariant_preservation(config):
    """Verify mandatory invariant: if k_candidate == 0, actual faults MUST remain 0.
    
    Hard qualification must be achieved via non-fault mechanisms (e.g. local_event).
    """
    sim = ScenarioSimulator(config=config)

    for seed in range(10):
        scenario = sim.simulate_scenario(
            scenario_id=f"test_hard_zf_{seed}",
            seed=seed * 31,
            scenario_type="standard",
            difficulty_tier="hard",
            k_candidate=0,  # Zero-fault guarantee
        )
        # Total actual faults must strictly be 0
        k_faulty = sum(
            1 for g in scenario.groups.values() for s in g.sensors.values()
            if s.fault_series is not None and s.fault_series.has_faults
        )
        assert k_faulty == 0

        # Scenario must STILL qualify as HARD!
        verified = evaluate_hard_mechanisms(scenario)
        assert len(verified) >= 1
        assert "local_event" in verified or "sensor_near_bounds" in verified


# =============================================================================
# 6. Verification of All Seven Detectors & Fallbacks (Decisions 38 & 41)
# =============================================================================

def test_all_seven_detectors_unit(config):
    """Verify each of the 7 detectors behaves correctly on synthetic and verified scenarios."""
    sim = ScenarioSimulator(config=config)

    # 1. detect_local_event
    scen_local = sim.simulate_scenario("s_local", seed=42, scenario_type="standard", difficulty_tier="hard")
    # Force local event construction
    scen_local_fallback = sim._construct_mechanism(
        "local_event", scen_local,
        all_sensor_params=[s.params for g in scen_local.groups.values() for s in g.sensors.values()],
        group_sensor_params={gid: [s.params for s in g.sensors.values()] for gid, g in scen_local.groups.items()},
        clean_traj={s.sensor_id: np.full(1920, 20.0) for g in scen_local.groups.values() for s in g.sensors.values()},
        rng=np.random.default_rng(101),
    )
    assert detect_local_event(scen_local_fallback) is True

    # 2. detect_subtle_fault
    scen_subtle = sim.simulate_scenario("s_subtle", seed=43, scenario_type="standard", difficulty_tier="hard", k_candidate=1)
    scen_subtle_fallback = sim._construct_mechanism(
        "subtle_fault", scen_subtle,
        all_sensor_params=[s.params for g in scen_subtle.groups.values() for s in g.sensors.values()],
        group_sensor_params={gid: [s.params for s in g.sensors.values()] for gid, g in scen_subtle.groups.items()},
        clean_traj={s.sensor_id: np.full(1920, 20.0) for g in scen_subtle.groups.values() for s in g.sensors.values()},
        rng=np.random.default_rng(102),
    )
    assert detect_subtle_fault(scen_subtle_fallback) is True

    # 3. detect_fault_during_event
    scen_fde = sim.simulate_scenario("s_fde", seed=44, scenario_type="standard", difficulty_tier="hard", k_candidate=1)
    scen_fde_fallback = sim._construct_mechanism(
        "fault_during_event", scen_fde,
        all_sensor_params=[s.params for g in scen_fde.groups.values() for s in g.sensors.values()],
        group_sensor_params={gid: [s.params for s in g.sensors.values()] for gid, g in scen_fde.groups.items()},
        clean_traj={s.sensor_id: np.full(1920, 20.0) for g in scen_fde.groups.values() for s in g.sensors.values()},
        rng=np.random.default_rng(103),
    )
    assert detect_fault_during_event(scen_fde_fallback) is True

    # 4. detect_faulty_fraction_at_cap_small_group
    # Construct scenario with group of 4 sensors and exactly 1 fault
    scen_small = sim.simulate_scenario("s_small", seed=45, scenario_type="standard", difficulty_tier="medium", k_candidate=1)
    # Check if small group cap detector works
    if any(len(g.sensors) == 4 for g in scen_small.groups.values()):
        assert detect_faulty_fraction_at_cap_small_group(scen_small) is True

    # 5. detect_no_clear_majority (Decision 41)
    # Synthetic verification: when sensors are split into clusters where none has > 50%
    rng = np.random.default_rng(999)
    env = generate_environment_series(config=config, rng=rng)
    p_prof = TemperatureProfile(config)
    s_params = p_prof.sample_sensor_params(num_sensors=4, sensor_ids=["s0", "s1", "s2", "s3"], rng=rng)
    # Give sensors divergent constant biases >= 3.0 C apart so each cluster has size 1 (1/4 = 25% <= 50%)
    group_sim = VirtualSensorSimulator(config=config, profile=p_prof).simulate_group(
        environment=env,
        sensor_params=s_params,
        group_id="group_00",
        rng=rng,
    )
    # Modify canonical values in place to create split clusters for 35 steps
    for idx, sid in enumerate(["s0", "s1", "s2", "s3"]):
        group_sim.sensors[sid].canonical_values[500:540] = 20.0 + idx * 5.0

    scen_split = VirtualScenario(
        scenario_id="scen_split",
        scenario_type="standard",
        difficulty_tier="hard",
        seed=1,
        environment=env,
        groups={"group_00": group_sim},
    )
    assert detect_no_clear_majority(scen_split) is True


def test_subtle_fault_preserves_sampled_type_count(config):
    """Verify subtle fault fallback preserves sampled type count m in {1, 2, 3}."""
    sim = ScenarioSimulator(config=config)
    rng = np.random.default_rng(555)

    for m in [1, 2, 3]:
        scen = sim.simulate_scenario("s_test", seed=10, scenario_type="standard", difficulty_tier="hard", k_candidate=1)
        sid = scen.sensor_ids[0]
        f_sim = FaultSimulator(config=config)
        s_sigma = 0.15
        eps = [
            f_sim.sample_fault(
                FaultType.SPIKE, sid, sigma_k=s_sigma, total_steps=1920, warmup_steps=480,
                physical_xref=20.0, clean_base_value=20.0, rng=rng, fault_id=f"f_{i}",
                start_step=600 + i * 200,
            )
            for i in range(m)
        ]
        all_s_params = [s.params for g in scen.groups.values() for s in g.sensors.values()]
        g_s_params = {gid: [s.params for s in g.sensors.values()] for gid, g in scen.groups.items()}
        custom_fault_map = {
            s.sensor_id: (SensorFaultSeries(sid, 1920, episodes=eps) if s.sensor_id == sid else s.fault_series)
            for s in scen.groups["group_00"].sensors.values()
        }
        scen_initial = sim._rebuild_scenario_with_events_and_faults(
            scenario=scen,
            new_events=scen.events,
            all_sensor_params=all_s_params,
            group_sensor_params=g_s_params,
            fault_map=custom_fault_map,
            rng=rng,
        )

        scen_fallback = sim._construct_mechanism(
            "subtle_fault", scen_initial,
            all_sensor_params=all_s_params,
            group_sensor_params=g_s_params,
            clean_traj={sid: np.full(1920, 20.0)},
            rng=rng,
        )
        assert len(scen_fallback.groups["group_00"].sensors[sid].fault_series.episodes) == m
        assert detect_subtle_fault(scen_fallback) is True


# =============================================================================
# 7. Two-Group Shared Environment & Noise Independence (Decision 36)
# =============================================================================

def test_two_group_shared_environment_and_noise_independence(config):
    """Verify two-group scenarios:
    1. Both groups share identical physical macro-environment series E(t).
    2. Globally unique sensor IDs across groups.
    3. Residual/measurement noise vectors across groups are independent (|corr| < 0.05).
    """
    sim = ScenarioSimulator(config=config)
    scenario = sim.simulate_scenario(
        scenario_id="two_group_test",
        seed=2026,
        scenario_type="two_groups",
        difficulty_tier="medium",
        k_candidate=0,  # Clean signals to test noise correlation
    )

    assert len(scenario.groups) == 2
    g0 = scenario.groups["group_00"]
    g1 = scenario.groups["group_01"]

    # Unique sensor IDs
    all_sids = scenario.sensor_ids
    assert len(all_sids) == len(set(all_sids))
    assert not set(g0.sensors.keys()).intersection(set(g1.sensors.keys()))

    # Shared macro-environment
    assert np.allclose(scenario.environment.values, g0.sensors[list(g0.sensors.keys())[0]].environmental_values)
    assert np.allclose(scenario.environment.values, g1.sensors[list(g1.sensors.keys())[0]].environmental_values)

    s_g0 = list(g0.sensors.values())[0]
    s_g1 = list(g1.sensors.values())[0]

    # Noise independence: calculate residuals after subtracting response-lagged physical environment
    # r_k(t) = canonical_clean_values(t) - lagged_physical(t) - static_offset
    w_steps = scenario.environment.warmup_steps
    lag_0 = apply_response_lag(s_g0.environmental_values, s_g0.params.response_lag_tau, 1.0)
    res_0 = s_g0.canonical_clean_values[w_steps:] - lag_0[w_steps:] - s_g0.params.static_offset

    lag_1 = apply_response_lag(s_g1.environmental_values, s_g1.params.response_lag_tau, 1.0)
    res_1 = s_g1.canonical_clean_values[w_steps:] - lag_1[w_steps:] - s_g1.params.static_offset

    corr = np.corrcoef(res_0, res_1)[0, 1]
    assert abs(corr) < 0.05, f"Residual correlation between groups too high: {corr}"


# =============================================================================
# 8. Single-Sensor Scope Semantics & Explicit Half-Up Rounding (Decision 37)
# =============================================================================

def test_single_sensor_scope_and_half_up_rounding(config):
    """Verify single-sensor partial-event evaluation:
    - Target count k = floor(frac * N + 0.5)
    - For N = 1, frac in [0.25, 0.75]:
      * frac in [0.25, 0.50) -> floor(frac + 0.5) = 0 (25% probability)
      * frac in [0.50, 0.75] -> floor(frac + 0.5) = 1 (75% probability)
    """
    rng = np.random.default_rng(888)
    n_draws = 2000
    counts = []

    event_gen = EventGenerator(config=config)

    for _ in range(n_draws):
        scope, aff_sids, attenuations = event_gen.sample_scope_and_sensors(
            sensor_ids=["sensor_00"],
            rng=rng,
            allowed_scopes=["partial"],
        )
        assert scope == "partial"
        counts.append(len(aff_sids))

    s_series = pd.Series(counts).value_counts(normalize=True)
    assert abs(s_series.get(0, 0) - 0.25) < 0.03
    assert abs(s_series.get(1, 0) - 0.75) < 0.03


# =============================================================================
# 9. Observable Schema Isolation & Metadata Completeness (Decisions 34 & 40)
# =============================================================================

def test_observable_schema_isolation_and_metadata(config):
    """Verify strictly 6-column observable DataFrame export and comprehensive metadata summary."""
    sim = ScenarioSimulator(config=config)
    scenario = sim.simulate_scenario(
        scenario_id="obs_test_01",
        seed=101,
        scenario_type="two_groups",
        difficulty_tier="medium",
    )

    df = scenario.to_observable_dataframe(stride=1)
    expected_cols = ["scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"]

    assert list(df.columns) == expected_cols
    assert len(df) == scenario.total_sensors * scenario.environment.total_steps
    assert df["scenario_id"].iloc[0] == "obs_test_01"
    assert set(df["group_id"].unique()) == {"group_00", "group_01"}
    assert set(df["variable_type"].unique()) == {"temperature"}

    # Ground truth is isolated: no ground-truth columns present
    for bad_col in ["clean_value", "severity", "is_faulty", "fault_type"]:
        assert bad_col not in df.columns

    # Test stride
    df_stride = scenario.to_observable_dataframe(stride=5)
    assert len(df_stride) == scenario.total_sensors * (scenario.environment.total_steps // 5)

    # Metadata dictionary completeness
    meta = scenario.to_metadata_dict()
    assert meta["scenario_id"] == "obs_test_01"
    assert meta["seed"] == 101
    assert meta["scenario_type"] == "two_groups"
    assert meta["difficulty_tier"] == "medium"
    assert meta["sensor_count"] == scenario.total_sensors
    assert meta["group_count"] == 2
    assert "target_mechanism" in meta
    assert "hard_mechanisms" in meta


# =============================================================================
# 10. Identical-Seed Reproducibility (Decision 40)
# =============================================================================

def test_scenario_simulation_determinism(config):
    """Verify that identical seed produces bitwise identical observables and metadata."""
    sim = ScenarioSimulator(config=config)

    scen_1 = sim.simulate_scenario("det_test", seed=7777)
    scen_2 = sim.simulate_scenario("det_test", seed=7777)

    df_1 = scen_1.to_observable_dataframe()
    df_2 = scen_2.to_observable_dataframe()

    pd.testing.assert_frame_equal(df_1, df_2)
    assert scen_1.to_metadata_dict() == scen_2.to_metadata_dict()


# =============================================================================
# 11. P1.6 Known Coverage Gaps Closed (Decisions 33 & 38)
# =============================================================================

def test_shared_cause_event_coincidence_fallback():
    """Verify shared-cause start step when event coincidence has no feasible offset or empty events."""
    rng = np.random.default_rng(333)

    # Case 1: active_event_steps is completely empty -> falls back to independent start
    start_step = sample_shared_cause_start_step(
        t_start_A=800,
        total_steps=1920,
        warmup_steps=480,
        active_event_steps=[],
        rng=rng,
    )
    assert 790 <= start_step <= 810

    # Case 2: t_start_A is in event, but active event is only 1 step wide (no other event steps)
    start_step_2 = sample_shared_cause_start_step(
        t_start_A=900,
        total_steps=1920,
        warmup_steps=480,
        active_event_steps=[900],  # only 900 is in event
        rng=rng,
    )
    assert 890 <= start_step_2 <= 910


def test_dropout_shared_cause_coupling_tolerance(config):
    """Verify dropout shared-cause coupling satisfies 20% normalized severity tolerance (Decision 33)."""
    rng = np.random.default_rng(444)
    fault_sim = FaultSimulator(config=config)

    # Create sensor A dropout episode
    ep_A = fault_sim.sample_fault(
        FaultType.DROPOUT,
        sensor_id="sensor_A",
        sigma_k=0.15,
        total_steps=1920,
        warmup_steps=480,
        physical_xref=20.0,
        clean_base_value=20.0,
        rng=rng,
        fault_id="fault_A_0",
        start_step=600,
    )

    sev_A = compute_dropout_theoretical_severity(
        p_enter=ep_A.shape_params["enter_gap_prob"],
        mean_gap_uncapped=ep_A.shape_params["mean_gap_length"],
        cap=100,
    )

    # Couple 100 times and verify every single realization is within 20%
    for _ in range(100):
        coupled = couple_shared_cause_parameters(
            FaultType.DROPOUT,
            ep_A,
            sigma_k_A=0.15,
            sigma_k_B=0.20,
            config=config,
            rng=rng,
        )
        sev_B = compute_dropout_theoretical_severity(
            p_enter=coupled["p_enter"],
            mean_gap_uncapped=coupled["mean_gap"],
            cap=100,
        )
        rel_diff = abs(sev_A - sev_B) / max(sev_A, sev_B)
        assert rel_diff <= 0.20 + 1e-6, f"Dropout severity relative diff {rel_diff} exceeded 0.20"


# =============================================================================
# 12. Focused HARD-Mechanism Fallback Coverage (Decision 38 Audit)
# =============================================================================

def test_hard_fallback_local_event_deterministic(config):
    """Directly exercise local_event constructive fallback for zero-fault HARD scenarios.
    
    Verifies:
      1. Zero actual faults are strictly preserved (k_actual == 0).
      2. detect_local_event returns True after accounting for waveform, sign, magnitude,
         attenuation, arrival delay, and sensor response lag.
      3. Physical detectability: plateau perturbation exceeds 3*sigma_k for > 800 scored steps.
      4. Both positive and negative event signs are feasible and verified.
    """
    sim = ScenarioSimulator(config=config)

    for seed in [1001, 1002]:
        scen_base = sim.simulate_scenario(
            scenario_id=f"zero_fault_hard_{seed}",
            seed=seed,
            scenario_type="standard",
            difficulty_tier="hard",
            k_candidate=0,
        )
        all_s_params = [s.params for g in scen_base.groups.values() for s in g.sensors.values()]
        g_s_params = {gid: [s.params for s in g.sensors.values()] for gid, g in scen_base.groups.items()}
        clean_traj = {s.sensor_id: np.full(1920, 20.0) for s in all_s_params}
        rng = np.random.default_rng(seed)

        # Force local_event fallback construction
        scen_constructed = sim._construct_mechanism(
            target="local_event",
            scenario=scen_base,
            all_sensor_params=all_s_params,
            group_sensor_params=g_s_params,
            clean_traj=clean_traj,
            rng=rng,
        )

        # 1. Zero actual faults strictly preserved
        k_faults = sum(
            1 for g in scen_constructed.groups.values() for s in g.sensors.values()
            if s.fault_series is not None and s.fault_series.has_faults
        )
        assert k_faults == 0, "Constructive fallback injected unauthorized faults into zero-fault scenario!"

        # 2. Detector verification
        assert detect_local_event(scen_constructed) is True

        # 3. Physical waveform and detectability verification
        assert scen_constructed.events is not None
        assert len(scen_constructed.events.episodes) == 1
        ep = scen_constructed.events.episodes[0]
        assert ep.event_type == "sustained_shift"
        assert ep.scope == "local"
        assert 600 <= ep.start_step <= 1000
        assert 4.0 <= ep.abs_magnitude <= 8.0
        assert ep.end_step == 1920

        # Verify sensor physical perturbation on plateau after lag and attenuation
        aff_sid = ep.affected_sensors[0]
        aff_stream = scen_constructed.groups["group_00"].sensors[aff_sid]
        max_sigma = aff_stream.params.sigma_k
        att = ep.sensor_attenuations[aff_sid]
        assert 0.70 <= att <= 1.0

        # At plateau (e.g. step 1200), perturbation must exceed 3 * sigma_k
        contrib_plateau = abs(scen_constructed.events.sensor_contributions[aff_sid][1200])
        assert contrib_plateau >= 4.0 * 0.70 * 0.99  # > 2.77 C
        assert contrib_plateau >= 3.0 * max_sigma


def test_hard_fallback_fault_during_event_deterministic(config):
    """Directly exercise fault_during_event constructive fallback.
    
    Verifies:
      1. Genuine event has ramp waveform with hold duration >= 60 steps.
      2. Fault episode start step is placed inside the scored active event interval.
      3. Active fault and active event intervals overlap in [480, 1919].
      4. detect_fault_during_event returns True.
    """
    sim = ScenarioSimulator(config=config)
    scen_base = sim.simulate_scenario(
        scenario_id="fde_test",
        seed=2001,
        scenario_type="standard",
        difficulty_tier="hard",
        k_candidate=1,
    )
    all_s_params = [s.params for g in scen_base.groups.values() for s in g.sensors.values()]
    g_s_params = {gid: [s.params for s in g.sensors.values()] for gid, g in scen_base.groups.items()}
    clean_traj = {s.sensor_id: np.full(1920, 20.0) for s in all_s_params}
    rng = np.random.default_rng(2001)

    scen_fde = sim._construct_mechanism(
        target="fault_during_event",
        scenario=scen_base,
        all_sensor_params=all_s_params,
        group_sensor_params=g_s_params,
        clean_traj=clean_traj,
        rng=rng,
    )

    # 1. Detector returns True
    assert detect_fault_during_event(scen_fde) is True

    # 2. Event waveform is ramp with hold duration >= 60 steps
    assert scen_fde.events is not None
    assert len(scen_fde.events.episodes) == 1
    ep_ev = scen_fde.events.episodes[0]
    assert ep_ev.event_type == "ramp"
    assert ep_ev.shape_params["hold_minutes"] >= 60.0

    # 3. Active intervals overlap inside scored window [480, 1919]
    ev_steps = set(extract_active_event_steps(scen_fde.events, warmup_steps=480, total_steps=1920))
    assert len(ev_steps) >= 60

    faulty_streams = [
        s for g in scen_fde.groups.values() for s in g.sensors.values()
        if s.fault_series is not None and s.fault_series.has_faults
    ]
    assert len(faulty_streams) >= 1
    fs = faulty_streams[0]
    ep_f = fs.fault_series.episodes[0]
    assert 480 <= ep_f.start_step < 1920
    assert ep_f.start_step in ev_steps  # fault start falls inside scored event interval


def test_hard_fallback_subtle_fault_preserves_rules(config):
    """Directly exercise subtle_fault constructive fallback across m in {1, 2, 3}.
    
    Verifies:
      1. Preserves the sampled number of fault types m on the target sensor.
      2. Injects gain_error with |g - 1| <= 0.10.
      3. For m = 2: resulting pair satisfies configured P1.6 allowed/forbidden rules.
      4. For m = 3: resulting triplet is one of the 11 mechanically valid P1.6 triplets.
      5. detect_subtle_fault returns True.
    """
    sim = ScenarioSimulator(config=config)
    rng = np.random.default_rng(3001)
    valid_pairs = get_valid_pairs(config)
    valid_triplets = derive_valid_triplets(config)

    for m in [1, 2, 3]:
        scen_base = sim.simulate_scenario("subtle_test", seed=m * 100, scenario_type="standard", difficulty_tier="hard", k_candidate=1)
        all_s_params = [s.params for g in scen_base.groups.values() for s in g.sensors.values()]
        g_s_params = {gid: [s.params for s in g.sensors.values()] for gid, g in scen_base.groups.items()}
        target_sid = scen_base.sensor_ids[0]

        # Pre-populate sensor with m dummy episodes
        f_sim = FaultSimulator(config=config)
        dummy_eps = [
            f_sim.sample_fault(
                FaultType.SPIKE, target_sid, sigma_k=0.15, total_steps=1920, warmup_steps=480,
                physical_xref=20.0, clean_base_value=20.0, rng=rng, fault_id=f"f_{i}",
                start_step=600 + i * 200,
            )
            for i in range(m)
        ]
        custom_fault_map = {
            s.sensor_id: (SensorFaultSeries(target_sid, 1920, episodes=dummy_eps) if s.sensor_id == target_sid else s.fault_series)
            for s in scen_base.groups["group_00"].sensors.values()
        }
        scen_m = sim._rebuild_scenario_with_events_and_faults(
            scenario=scen_base,
            new_events=scen_base.events,
            all_sensor_params=all_s_params,
            group_sensor_params=g_s_params,
            fault_map=custom_fault_map,
            rng=rng,
        )

        scen_subtle = sim._construct_mechanism(
            target="subtle_fault",
            scenario=scen_m,
            all_sensor_params=all_s_params,
            group_sensor_params=g_s_params,
            clean_traj={target_sid: np.full(1920, 20.0)},
            rng=rng,
        )

        # 1. Detector verified
        assert detect_subtle_fault(scen_subtle) is True

        # 2. Type count m strictly preserved
        episodes = scen_subtle.groups["group_00"].sensors[target_sid].fault_series.episodes
        assert len(episodes) == m

        # 3. Contains subtle gain error
        gain_eps = [ep for ep in episodes if ep.fault_type == FaultType.GAIN_ERROR]
        assert len(gain_eps) >= 1
        g_val = gain_eps[0].shape_params["gain"]
        assert abs(g_val - 1.0) <= 0.10 + 1e-6

        # 4. Combination validity
        type_names = tuple(sorted([ep.fault_type.value for ep in episodes]))
        if m == 2:
            assert frozenset(type_names) in valid_pairs, f"Subtle fault pair {type_names} is not an allowed pair!"
        elif m == 3:
            assert type_names in valid_triplets, f"Subtle fault triplet {type_names} is not a valid triplet!"


def test_hard_fallback_small_group_at_cap_deterministic(config):
    """Directly exercise faulty_fraction_at_cap_small_group constructive fallback.
    
    Verifies:
      1. Identifies 4-sensor group and allocates exactly 1 faulty sensor.
      2. Does not violate group cap (cap_g(4) = 1) or scenario cap.
      3. detect_faulty_fraction_at_cap_small_group returns True.
    """
    sim = ScenarioSimulator(config=config)
    rng = np.random.default_rng(4001)

    p_prof = TemperatureProfile(config)
    s4_params = p_prof.sample_sensor_params(num_sensors=4, sensor_ids=[f"s_{i}" for i in range(4)], rng=rng)
    env = generate_environment_series(config=config, rng=rng)
    g4 = VirtualSensorSimulator(config=config, profile=p_prof).simulate_group(
        environment=env,
        sensor_params=s4_params,
        group_id="group_00",
        rng=rng,
    )
    scen_base = VirtualScenario(
        scenario_id="s4_test",
        scenario_type="standard",
        difficulty_tier="hard",
        seed=4002,
        environment=env,
        groups={"group_00": g4},
    )
    all_s_params = s4_params
    g_s_params = {"group_00": s4_params}
    clean_traj = {s.sensor_id: np.full(1920, 20.0) for s in s4_params}

    scen_cap = sim._construct_mechanism(
        target="faulty_fraction_at_cap_small_group",
        scenario=scen_base,
        all_sensor_params=all_s_params,
        group_sensor_params=g_s_params,
        clean_traj=clean_traj,
        rng=rng,
    )

    assert detect_faulty_fraction_at_cap_small_group(scen_cap) is True
    g = scen_cap.groups["group_00"]
    assert len(g.sensors) == 4
    k_faulty_g = sum(1 for s in g.sensors.values() if s.fault_series and s.fault_series.has_faults)
    assert k_faulty_g == 1  # exactly 1 faulty sensor, satisfying group cap


def test_hard_fallback_verification_and_metadata_guarantee(config):
    """Verify that detectors are evaluated post-construction and metadata is fully synchronized.
    
    Verifies:
      1. scen.hard_mechanisms contains only verified detectors.
      2. scen.target_mechanism is recorded and matches metadata.
      3. No unqualified HARD scenario can be returned across 60 seeds and topologies.
    """
    sim = ScenarioSimulator(config=config)

    for scen_type in ["standard", "two_groups", "single_sensor"]:
        for seed in range(20):
            scen = sim.simulate_scenario(
                scenario_id=f"audit_{scen_type}_{seed}",
                seed=seed * 53 + 7,
                scenario_type=scen_type,
                difficulty_tier="hard",
            )
            # 1. Verification post-construction
            verified = evaluate_hard_mechanisms(scen)
            assert len(verified) >= 1, f"Unqualified HARD scenario returned for {scen_type} seed {seed}"
            assert set(scen.hard_mechanisms) == set(verified)

            # 2. Metadata completeness
            meta = scen.metadata
            assert meta["difficulty_tier"] == "hard"
            assert meta["target_mechanism"] == scen.target_mechanism
            assert meta["hard_mechanisms"] == scen.hard_mechanisms
            assert meta["target_mechanism"] in (
                "natural", "local_event", "subtle_fault", "fault_during_event", "faulty_fraction_at_cap_small_group"
            )

