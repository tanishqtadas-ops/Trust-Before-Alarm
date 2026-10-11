"""Stratum classification, Decision 47 Integer Apportionment, and Test Subset Partitioning.

This module implements:
1. Disjoint stratum classification across (difficulty_tier, fault_profile) with 22 fault profiles.
2. Decision 47 Stratified Biproportional Apportionment with exact integer costs and sequential
   lexicographic refinement.
3. Test set partitioning into test_full_resolution (stride 1) and test_downsampled (stride 5)
   enforcing canonical difficulty-tier parity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import LinearConstraint, milp

from src.sim.scenario import VirtualScenario, detect_shared_cause_pair


# The 22 approved fault profiles
APPROVED_FAULT_PROFILES = [
    "healthy",
    # 7 single_sensor_<type>
    "single_sensor_spike",
    "single_sensor_offset_drift",
    "single_sensor_gain_error",
    "single_sensor_frozen",
    "single_sensor_high_noise",
    "single_sensor_dropout",
    "single_sensor_slow_degradation",
    # 7 shared_cause_<type>
    "shared_cause_spike",
    "shared_cause_offset_drift",
    "shared_cause_gain_error",
    "shared_cause_frozen",
    "shared_cause_high_noise",
    "shared_cause_dropout",
    "shared_cause_slow_degradation",
    # 7 multi_independent_<type>
    "multi_independent_spike",
    "multi_independent_offset_drift",
    "multi_independent_gain_error",
    "multi_independent_frozen",
    "multi_independent_high_noise",
    "multi_independent_dropout",
    "multi_independent_slow_degradation",
]


@dataclass(frozen=True)
class ScenarioDescriptor:
    """Lightweight descriptor of a scenario for stratum classification and splitting."""
    scenario_id: str
    seed: int
    difficulty_tier: str
    fault_profile: str
    stratum_name: str


def find_shared_cause_fault_type(scenario: VirtualScenario) -> Optional[str]:
    """Find the shared fault type of the actual shared-cause pair in the scenario.
    
    Inspects scenario groups to locate the pair of sensors in the same group
    sharing Episode 1 fault type and start step within +/- 10 steps.
    """
    for gid in sorted(scenario.groups.keys()):
        group = scenario.groups[gid]
        faulty_sensors = [
            s for s in sorted(group.sensors.values(), key=lambda s: s.sensor_id)
            if s.fault_series is not None and len(s.fault_series.episodes) > 0
        ]
        if len(faulty_sensors) < 2:
            continue
        for i in range(len(faulty_sensors)):
            for j in range(i + 1, len(faulty_sensors)):
                s_i, s_j = faulty_sensors[i], faulty_sensors[j]
                ep_i = s_i.fault_series.episodes[0]
                ep_j = s_j.fault_series.episodes[0]
                if ep_i.fault_type == ep_j.fault_type:
                    if abs(ep_i.start_step - ep_j.start_step) <= 10:
                        ft = ep_i.fault_type
                        return ft.value if hasattr(ft, "value") else str(ft)
    return None


def classify_scenario_stratum(scenario: VirtualScenario) -> Tuple[str, str]:
    """Classify a VirtualScenario into (difficulty_tier, fault_profile).
    
    Preserves exact approved 22 fault profiles:
    - 'healthy': 0 faulty sensors.
    - 'single_sensor_<type>': exactly 1 faulty sensor; primary type from earliest episode.
    - 'shared_cause_<shared_type>': >= 2 faulty sensors with shared-cause pair;
      shared type obtained from internal pairing metadata / matched pair.
    - 'multi_independent_<type>': >= 2 faulty sensors without shared-cause pair;
      primary type from earliest episode of lowest-numbered faulty sensor.
    """
    tier = str(scenario.difficulty_tier)
    if tier not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unknown difficulty tier: {tier}")

    # Collect all faulty sensors across the scenario
    faulty_sensors = [
        s for g in scenario.groups.values() for s in g.sensors.values()
        if s.fault_series is not None and len(s.fault_series.episodes) > 0
    ]
    k_actual = len(faulty_sensors)

    if k_actual == 0:
        fault_profile = "healthy"
    elif k_actual == 1:
        s = faulty_sensors[0]
        # Sort episodes by start_step then fault_id
        sorted_eps = sorted(s.fault_series.episodes, key=lambda e: (e.start_step, e.fault_id))
        ft = sorted_eps[0].fault_type
        primary_type = ft.value if hasattr(ft, "value") else str(ft)
        fault_profile = f"single_sensor_{primary_type}"
    else:
        # k_actual >= 2: check if shared cause pair exists
        has_shared = detect_shared_cause_pair(scenario)
        if has_shared:
            shared_type = find_shared_cause_fault_type(scenario)
            if shared_type is None:
                # Fallback to metadata if available
                shared_type = scenario.metadata.get("shared_fault_type")
            if shared_type is None:
                raise ValueError(
                    f"Scenario {scenario.scenario_id} detected as shared cause, "
                    f"but failed to locate shared fault type."
                )
            fault_profile = f"shared_cause_{shared_type}"
        else:
            # Multi-independent: lowest-numbered faulty sensor and its earliest episode
            sorted_by_id = sorted(faulty_sensors, key=lambda s: s.sensor_id)
            lowest_sensor = sorted_by_id[0]
            sorted_eps = sorted(lowest_sensor.fault_series.episodes, key=lambda e: (e.start_step, e.fault_id))
            ft = sorted_eps[0].fault_type
            primary_type = ft.value if hasattr(ft, "value") else str(ft)
            fault_profile = f"multi_independent_{primary_type}"

    if fault_profile not in APPROVED_FAULT_PROFILES:
        raise ValueError(f"Classified fault profile '{fault_profile}' not in approved 22 profiles.")

    return tier, fault_profile


def extract_scenario_descriptor(scenario: VirtualScenario) -> ScenarioDescriptor:
    """Extract a lightweight ScenarioDescriptor from a VirtualScenario."""
    tier, profile = classify_scenario_stratum(scenario)
    stratum_name = f"{tier}__{profile}"
    return ScenarioDescriptor(
        scenario_id=str(scenario.scenario_id),
        seed=int(scenario.seed),
        difficulty_tier=tier,
        fault_profile=profile,
        stratum_name=stratum_name,
    )


def solve_decision_47_apportionment(
    stratum_names: Sequence[str],
    stratum_counts: Sequence[int],
    target_totals: Sequence[int] = (900, 300, 300),
    weights: Sequence[int] = (3, 1, 1),
) -> np.ndarray:
    """Solve Decision 47 Stratified Biproportional Apportionment via MILP and sequential refinement.
    
    Args:
        stratum_names: Sequence of unique stratum name strings.
        stratum_counts: Sequence of integer population counts n_j per stratum.
        target_totals: Target column totals (C_0^*, C_1^*, C_2^*), default (900, 300, 300).
        weights: Split weights (w_0, w_1, w_2), default (3, 1, 1).
        
    Returns:
        Exact integer allocation matrix x of shape (J, 3) where:
        - sum_c x_{j, c} == n_j for every stratum j
        - sum_j x_{j, c} == target_totals[c] for every split c
        - |x_{j, c} - T_{j, c}| < 1 (controlled rounding)
        - Lexicographically preferred solution among ties.
    """
    J = len(stratum_counts)
    W = sum(weights)  # 5
    C_targets = np.array(target_totals, dtype=int)
    w_arr = np.array(weights, dtype=int)
    counts = np.array(stratum_counts, dtype=int)

    if counts.sum() != C_targets.sum():
        raise ValueError(f"Total scenarios {counts.sum()} does not match target totals sum {C_targets.sum()}.")

    # Ideal rational quotas T_{j, c} = n_j * w_c / W
    # Floor allocations f_{j, c} = (n_j * w_c) // W
    # Remainder numerators I_{j, c} = (n_j * w_c) % W
    f = np.zeros((J, 3), dtype=int)
    I = np.zeros((J, 3), dtype=int)
    cost = np.zeros((J, 3), dtype=int)

    for j in range(J):
        for c in range(3):
            T_num = counts[j] * w_arr[c]
            f[j, c] = T_num // W
            I[j, c] = T_num % W
            # Integer cost coefficient: 5 - 2 * I[j, c]
            cost[j, c] = W - 2 * I[j, c]

    # Row supplies R_j in {0, 1, 2}
    R = counts - f.sum(axis=1)
    # Column demands K_c
    K = C_targets - f.sum(axis=0)

    if R.sum() != K.sum():
        raise ValueError(f"Row supply sum {R.sum()} != column demand sum {K.sum()}")

    # Setup transportation equality constraints: A_eq @ y == b_eq
    # y is a flat vector of length 3 * J representing y_{j, c} in {0, 1}
    num_vars = 3 * J
    A_rows = np.zeros((J, num_vars))
    for j in range(J):
        A_rows[j, j * 3 : j * 3 + 3] = 1.0

    A_cols = np.zeros((3, num_vars))
    for c in range(3):
        for j in range(J):
            A_cols[c, j * 3 + c] = 1.0

    A_eq = np.vstack([A_rows, A_cols])
    b_eq = np.concatenate([R, K]).astype(float)

    c_obj = cost.flatten().astype(float)
    integrality = np.ones(num_vars)
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)

    # 1. Primary solve: minimize total integer cost
    res = milp(
        c=c_obj,
        integrality=integrality,
        bounds=(lb, ub),
        constraints=LinearConstraint(A_eq, b_eq, b_eq),
    )
    if res.status != 0:
        raise RuntimeError(f"Primary MILP solve failed with status {res.status}: {res.message}")

    min_cost = int(round(res.fun))

    # 2. Sequential lexicographic refinement
    # Ordering Key(j, c) = (cost_{j, c}, -n_j, stratum_name_j, c)
    def cell_key(idx: int) -> Tuple[int, int, str, int]:
        j = idx // 3
        c = idx % 3
        return (cost[j, c], -counts[j], stratum_names[j], c)

    cell_order = sorted(range(num_vars), key=cell_key)

    # Augment constraint matrix with cost equality: c_obj @ y == min_cost
    A_all = np.vstack([A_eq, c_obj.reshape(1, -1)])
    b_all = np.concatenate([b_eq, [float(min_cost)]])

    cur_lb = lb.copy()
    cur_ub = ub.copy()

    # Iteratively fix variables to 1 if feasible, else 0
    for idx in cell_order:
        if cur_lb[idx] == cur_ub[idx]:
            continue
        test_lb = cur_lb.copy()
        test_lb[idx] = 1.0
        res_test = milp(
            c=np.zeros(num_vars),
            integrality=integrality,
            bounds=(test_lb, cur_ub),
            constraints=LinearConstraint(A_all, b_all, b_all),
        )
        if res_test.status == 0:
            cur_lb[idx] = 1.0
            cur_ub[idx] = 1.0
        else:
            cur_lb[idx] = 0.0
            cur_ub[idx] = 0.0

    y_sol = cur_lb.reshape((J, 3)).astype(int)
    x_sol = f + y_sol

    # Validation checks
    if not np.array_equal(x_sol.sum(axis=1), counts):
        raise ValueError("Controlled rounding row sums do not match stratum counts!")
    if not np.array_equal(x_sol.sum(axis=0), C_targets):
        raise ValueError("Controlled rounding column sums do not match target totals!")

    # Verify controlled rounding property: |x_{j, c} - T_{j, c}| < 1
    for j in range(J):
        for c in range(3):
            T_val = (counts[j] * w_arr[c]) / W
            if abs(x_sol[j, c] - T_val) >= 1.0:
                raise ValueError(
                    f"Controlled rounding violation at ({stratum_names[j]}, {c}): "
                    f"x={x_sol[j, c]}, T={T_val}"
                )

    return x_sol


def apportion_splits(
    scenarios: Sequence[Union[VirtualScenario, ScenarioDescriptor]],
    master_seed: int = 42,
) -> Dict[str, str]:
    """Assign whole scenarios to train, val, or test following Decision 47.
    
    Guarantees:
    - Exactly 900 train, 300 val, 300 test scenarios.
    - Whole-scenario assignment (no row splitting across splits).
    - Disjoint strata across (difficulty_tier, fault_profile).
    - Exact controlled rounding apportionment.
    - Deterministic per-stratum RNG permutation using SHA-256 derived seeds.
    
    Returns:
        Dict mapping scenario_id -> split name ('train', 'val', 'test').
    """
    # 1. Convert to ScenarioDescriptors if necessary
    descriptors: List[ScenarioDescriptor] = []
    for s in scenarios:
        if isinstance(s, ScenarioDescriptor):
            descriptors.append(s)
        elif isinstance(s, VirtualScenario):
            descriptors.append(extract_scenario_descriptor(s))
        else:
            raise TypeError(f"Expected VirtualScenario or ScenarioDescriptor, got {type(s)}")

    total_scenarios = len(descriptors)
    if total_scenarios != 1500:
        raise ValueError(f"Decision 47 requires exactly 1,500 scenarios, got {total_scenarios}.")

    # 2. Group scenarios by stratum
    strata_map: Dict[str, List[ScenarioDescriptor]] = {}
    for d in descriptors:
        strata_map.setdefault(d.stratum_name, []).append(d)

    # Sort stratum names canonically
    canonical_strata = sorted(strata_map.keys())
    stratum_counts = [len(strata_map[name]) for name in canonical_strata]

    # 3. Solve integer apportionment
    x_alloc = solve_decision_47_apportionment(
        stratum_names=canonical_strata,
        stratum_counts=stratum_counts,
        target_totals=(900, 300, 300),
        weights=(3, 1, 1),
    )

    # 4. Deterministic scenario assignment per stratum
    split_assignment: Dict[str, str] = {}
    split_names = ["train", "val", "test"]

    for j, s_name in enumerate(canonical_strata):
        s_list = strata_map[s_name]
        # Sort deterministically by seed
        s_list_sorted = sorted(s_list, key=lambda s: s.seed)

        # SHA-256 derived stratum seed
        hash_digest = hashlib.sha256(f"stratify_{master_seed}_{s_name}".encode("utf-8")).hexdigest()
        stratum_seed = int(hash_digest[:8], 16)
        rng = np.random.default_rng(stratum_seed)

        # Permute deterministically
        perm = rng.permutation(len(s_list_sorted))
        shuffled = [s_list_sorted[idx] for idx in perm]

        n_train = x_alloc[j, 0]
        n_val = x_alloc[j, 1]
        n_test = x_alloc[j, 2]

        idx_train = n_train
        idx_val = n_train + n_val

        train_scenarios = shuffled[:idx_train]
        val_scenarios = shuffled[idx_train:idx_val]
        test_scenarios = shuffled[idx_val:]

        for s in train_scenarios:
            split_assignment[s.scenario_id] = "train"
        for s in val_scenarios:
            split_assignment[s.scenario_id] = "val"
        for s in test_scenarios:
            split_assignment[s.scenario_id] = "test"

    # Verify global counts
    counts_by_split = {
        "train": sum(1 for v in split_assignment.values() if v == "train"),
        "val": sum(1 for v in split_assignment.values() if v == "val"),
        "test": sum(1 for v in split_assignment.values() if v == "test"),
    }
    if counts_by_split != {"train": 900, "val": 300, "test": 300}:
        raise RuntimeError(f"Global split counts mismatch: {counts_by_split}")

    return split_assignment


def partition_test_subsets(
    test_scenarios: Sequence[Union[VirtualScenario, ScenarioDescriptor]],
    master_seed: int = 42,
) -> Dict[str, str]:
    """Partition the 300 test scenarios into test_full_resolution and test_downsampled.
    
    Enforces the approved canonical difficulty-tier parity procedure:
    1. Realized counts M_easy, M_medium, M_hard for the 300 test scenarios.
    2. Process tiers in canonical order: 'easy', 'medium', 'hard'.
    3. Assign floor(M_d / 2) to each subset.
    4. If two tiers have odd counts, assign first odd remainder to full-resolution,
       and second odd remainder to downsampled.
    5. Deterministic sorting and permutation within each tier.
    
    Returns:
        Dict mapping scenario_id -> subset name ('test_full_resolution', 'test_downsampled').
    """
    descriptors: List[ScenarioDescriptor] = []
    for s in test_scenarios:
        if isinstance(s, ScenarioDescriptor):
            descriptors.append(s)
        elif isinstance(s, VirtualScenario):
            descriptors.append(extract_scenario_descriptor(s))
        else:
            raise TypeError(f"Expected VirtualScenario or ScenarioDescriptor, got {type(s)}")

    if len(descriptors) != 300:
        raise ValueError(f"Expected exactly 300 test scenarios, got {len(descriptors)}.")

    # Group by difficulty tier
    tier_map: Dict[str, List[ScenarioDescriptor]] = {"easy": [], "medium": [], "hard": []}
    for d in descriptors:
        if d.difficulty_tier not in tier_map:
            raise ValueError(f"Unknown tier {d.difficulty_tier}")
        tier_map[d.difficulty_tier].append(d)

    canonical_tiers = ["easy", "medium", "hard"]
    realized_counts = {t: len(tier_map[t]) for t in canonical_tiers}

    # Step 3 & 4: Parity assignment
    odd_tiers = [t for t in canonical_tiers if realized_counts[t] % 2 == 1]
    if len(odd_tiers) not in (0, 2):
        raise ValueError(f"Mathematical invariant violation: odd tiers count is {len(odd_tiers)}, expected 0 or 2.")

    n_full: Dict[str, int] = {}
    n_down: Dict[str, int] = {}

    for t in canonical_tiers:
        base_h = realized_counts[t] // 2
        n_full[t] = base_h
        n_down[t] = base_h

    if len(odd_tiers) == 2:
        first_odd, second_odd = odd_tiers[0], odd_tiers[1]
        n_full[first_odd] += 1
        n_down[second_odd] += 1

    # Verify per-subset totals
    if sum(n_full.values()) != 150 or sum(n_down.values()) != 150:
        raise RuntimeError(
            f"Test subset sizing failed: full={sum(n_full.values())}, down={sum(n_down.values())}"
        )

    # Step 5: Assign individual scenarios deterministically
    subset_assignment: Dict[str, str] = {}
    for t in canonical_tiers:
        scens = tier_map[t]
        scens_sorted = sorted(scens, key=lambda s: s.seed)

        tier_hash = hashlib.sha256(f"test_subset_{master_seed}_{t}".encode("utf-8")).hexdigest()
        tier_seed = int(tier_hash[:8], 16)
        rng = np.random.default_rng(tier_seed)

        perm = rng.permutation(len(scens_sorted))
        shuffled = [scens_sorted[idx] for idx in perm]

        full_part = shuffled[:n_full[t]]
        down_part = shuffled[n_full[t]:]

        for s in full_part:
            subset_assignment[s.scenario_id] = "test_full_resolution"
        for s in down_part:
            subset_assignment[s.scenario_id] = "test_downsampled"

    # Parity verification: difference at most 1 per tier
    for t in canonical_tiers:
        diff = abs(n_full[t] - n_down[t])
        if diff > 1:
            raise ValueError(f"Tier {t} parity violated: full={n_full[t]}, down={n_down[t]}, diff={diff}")

    return subset_assignment
