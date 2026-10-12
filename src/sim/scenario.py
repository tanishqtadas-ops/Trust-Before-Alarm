"""Scenario simulation and orchestration for sensor stream intelligence (P1.7).

Coordinates scenario types (standard, two-group, single-sensor), sensor counts,
and difficulty tiers (EASY, MEDIUM, HARD) adhering strictly to Appendix A.1, A.6,
and approved Decisions 35-41:
  - Decision 35: Discrete uniform group sizes ([4, 8] standard, [4, 6] two-group, 1 single);
    scenario type and difficulty tier independently sampled.
  - Decision 36: Two-group scenarios share scenario-level EnvironmentSeries E(t);
    events generated over total population N; sensor parameters and 1-factor noise
    independent per group; unit scaling sampled at scenario level, applied in sensor pipeline.
  - Decision 37: Single-sensor partial-event evaluation via explicit half-up rounding
    k_target = floor(frac * N + 0.5); 1-factor noise marginal variance sigma_0^2.
  - Decision 38: Deterministic HARD-mechanism guarantee via 4-stage algorithm:
    feasibility filtering, natural evaluation first, verified constructive fallback queue.
    Zero-fault strictly preserved (k_cand == 0 => k_actual == 0). P1.6 15% shared-cause
    probability strictly preserved (never forced as fallback).
  - Decision 39: Staged count k_actual = min(k_cand, cap_scen, sum(cap_g), cap_tier).
    EASY visibility sampled from original families restricted to allowed ranges.
  - Decision 40: VirtualScenario and ScenarioSimulator architecture; strict 6-column
    Layer 1 observable schema isolation; separate metadata summary.
  - Decision 41: Operational detector for no_clear_majority via 1D connected components
    (theta = 2.0 C, >= 30-step window, per-group evaluation, max cluster fraction <= 0.50).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.sim.environment import EnvironmentSeries, EnvironmentSimulator, generate_environment_series
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
    allocate_faulty_sensors,
    compute_actual_faulty_sensor_count,
    compute_group_cap,
    compute_scenario_cap,
    derive_valid_triplets,
    extract_active_event_steps,
    generate_scenario_faults,
    get_valid_pairs,
    sample_episode_start_step,
    sample_fault_type_count,
    sample_fault_types_for_sensor,
    sample_faulty_sensor_candidate_count,
    sample_log_uniform,
)
from src.sim.profile import TemperatureProfile, TemperatureScenarioParams, TemperatureSensorParams
from src.sim.sensors import VirtualSensorGroup, VirtualSensorSimulator, simulate_sensors
from src.utils.config import canonical_json, get_config_dir, load_yaml


# =============================================================================
# 1. Topology and Tier Sampling Utilities (Decisions 35, 37, 39)
# =============================================================================

def sample_scenario_type(config: Dict[str, Any], rng: np.random.Generator) -> str:
    """Sample scenario type according to configured probabilities (Decision 35).
    
    Standard: 82%, Two-group: 10%, Single-sensor: 8%.
    """
    st_cfg = config["structure"]["scenario_type_probs"]
    types = ["standard", "two_groups", "single_sensor"]
    probs = [float(st_cfg[t]) for t in types]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    return str(rng.choice(types, p=probs))


def sample_group_sizes(
    scenario_type: str,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> List[int]:
    """Sample group sizes for the scenario type (Decision 35).
    
    - Standard: DiscreteUniform[4, 8] (single group)
    - Two-group: DiscreteUniform[4, 6] independently for each of the two groups
    - Single-sensor: Exactly [1]
    """
    struct = config["structure"]
    if scenario_type == "standard":
        min_n = int(struct["sensors_per_group_min"])
        max_n = int(struct["sensors_per_group_max"])
        return [int(rng.integers(min_n, max_n + 1))]
    elif scenario_type == "two_groups":
        min_n = int(struct["two_groups_sensors_min"])
        max_n = int(struct["two_groups_sensors_max"])
        n1 = int(rng.integers(min_n, max_n + 1))
        n2 = int(rng.integers(min_n, max_n + 1))
        return [n1, n2]
    elif scenario_type == "single_sensor":
        return [int(struct.get("single_sensor_count", 1))]
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


def sample_difficulty_tier(config: Dict[str, Any], rng: np.random.Generator) -> str:
    """Sample difficulty tier independently from scenario type (Decision 35).
    
    EASY: 30%, MEDIUM: 40%, HARD: 30%.
    """
    tiers_cfg = config["difficulty_tiers"]
    tiers = ["easy", "medium", "hard"]
    probs = [float(tiers_cfg[t]["prob"]) for t in tiers]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    return str(rng.choice(tiers, p=probs))


# =============================================================================
# 2. Hard-Mechanism Detectors (Appendix A.6, Decisions 2-4, 10, 33, 41)
# =============================================================================

def detect_local_event(scenario: "VirtualScenario") -> bool:
    """Detector for mechanism 1: local_event.
    
    Returns True iff there is at least one genuine detectable event with scope == 'local'
    affecting >= 1 sensor, physically active during the scored window [480, 1919].
    """
    if scenario.events is None:
        return False
    for ep in scenario.events.episodes:
        if ep.scope == "local" and len(ep.affected_sensors) >= 1:
            if ep.start_step < scenario.environment.total_steps and ep.end_step > scenario.environment.warmup_steps:
                if ep.is_detectable:
                    return True
    return False


def detect_fault_during_event(scenario: "VirtualScenario") -> bool:
    """Detector for mechanism 2: fault_during_event.
    
    Returns True iff an active fault episode overlaps an active genuine event in [480, 1919].
    """
    if scenario.events is None:
        return False
    ev_steps = set(extract_active_event_steps(
        scenario.events,
        warmup_steps=scenario.environment.warmup_steps,
        total_steps=scenario.environment.total_steps,
    ))
    if not ev_steps:
        return False

    w_steps = scenario.environment.warmup_steps
    t_steps = scenario.environment.total_steps

    for group in scenario.groups.values():
        for stream in group.sensors.values():
            if stream.fault_series is not None:
                for ep in stream.fault_series.episodes:
                    # Scored portion of fault episode
                    ep_start = max(w_steps, ep.start_step)
                    ep_end = min(t_steps, ep.end_step)
                    if ep_end > ep_start:
                        if any(t in ev_steps for t in range(ep_start, ep_end)):
                            return True
    return False


def detect_subtle_fault(scenario: "VirtualScenario") -> bool:
    """Detector for mechanism 3: subtle_fault (Decision 10).
    
    Returns True iff at least one fault episode has type gain_error with |g - 1| <= 0.10.
    """
    for group in scenario.groups.values():
        for stream in group.sensors.values():
            if stream.fault_series is not None:
                for ep in stream.fault_series.episodes:
                    if ep.fault_type == FaultType.GAIN_ERROR:
                        g = float(ep.shape_params.get("gain", 1.0))
                        if abs(g - 1.0) <= 0.10 + 1e-6:
                            return True
    return False


def detect_shared_cause_pair(scenario: "VirtualScenario") -> bool:
    """Detector for mechanism 4: shared_cause_pair (Decision 33).
    
    Returns True iff a group contains a shared-cause pair (two sensors in same group
    sharing Episode 1 fault type, start step within +/- 10 steps, coupled params).
    """
    for group in scenario.groups.values():
        faulty_sensors = [
            s for s in group.sensors.values()
            if s.fault_series is not None and len(s.fault_series.episodes) > 0
        ]
        if len(faulty_sensors) < 2:
            continue
        for i in range(len(faulty_sensors)):
            for j in range(i + 1, len(faulty_sensors)):
                ep_i = faulty_sensors[i].fault_series.episodes[0]
                ep_j = faulty_sensors[j].fault_series.episodes[0]
                if ep_i.fault_type == ep_j.fault_type:
                    if abs(ep_i.start_step - ep_j.start_step) <= 10:
                        return True
    return False


def detect_faulty_fraction_at_cap_small_group(scenario: "VirtualScenario") -> bool:
    """Detector for mechanism 5: faulty_fraction_at_cap_small_group (Decision 2).
    
    Returns True iff there exists at least one group with n = 4 sensors having exactly
    k_actual_g = cap_g = 1 faulty sensor.
    """
    for group in scenario.groups.values():
        n_sensors = len(group.sensors)
        if n_sensors == 4:
            k_faulty = sum(
                1 for s in group.sensors.values()
                if s.fault_series is not None and s.fault_series.has_faults
            )
            if k_faulty == compute_group_cap(4):  # exactly 1
                return True
    return False


def detect_no_clear_majority(scenario: "VirtualScenario") -> bool:
    """Detector for mechanism 6: no_clear_majority (Decisions 3 & 41).
    
    Evaluates 1D connected-components on canonical Celsius readings with theta = 2.0 C.
    Returns True iff for at least 30 consecutive scored steps in at least one group,
    no cluster of valid sensor readings contains > 50% of valid sensors (f_max <= 0.50).
    """
    theta = 2.0  # Decision 41: distance threshold 2.0 C
    w_steps = scenario.environment.warmup_steps
    t_steps = scenario.environment.total_steps
    window_req = 30  # Decision 41: 30 consecutive scored steps

    for group in scenario.groups.values():
        sensor_streams = list(group.sensors.values())
        if len(sensor_streams) < 2:
            continue

        consecutive_steps = 0
        for t in range(w_steps, t_steps):
            # Collect valid canonical readings at step t
            valid_readings = [
                float(stream.canonical_values[t])
                for stream in sensor_streams
                if not np.isnan(stream.canonical_values[t])
            ]
            n_valid = len(valid_readings)
            if n_valid < 2:
                consecutive_steps = 0
                continue

            # 1D single-linkage / connected components
            sorted_r = np.sort(valid_readings)
            clusters = [[sorted_r[0]]]
            for val in sorted_r[1:]:
                if val - clusters[-1][-1] <= theta:
                    clusters[-1].append(val)
                else:
                    clusters.append([val])

            max_cluster_size = max(len(c) for c in clusters)
            max_frac = max_cluster_size / n_valid

            if max_frac <= 0.50 + 1e-6:
                consecutive_steps += 1
                if consecutive_steps >= window_req:
                    return True
            else:
                consecutive_steps = 0

    return False


def detect_sensor_near_bounds(scenario: "VirtualScenario") -> bool:
    """Detector for mechanism 7: sensor_near_bounds (Decision 4).
    
    Evaluates whether pre-clipping environment is within 2.0 C of [-5, 50] C bounds,
    OR any sensor pre-clipping reading is within max(3*sigma_k, 2.0 C) of [-10, 60] C bounds.
    """
    w_steps = scenario.environment.warmup_steps
    t_steps = scenario.environment.total_steps
    env_unclipped = scenario.environment.unclipped_values[w_steps:t_steps]

    # Environment bound check ([-5.0, 50.0] C, margin 2.0 C)
    if np.any(env_unclipped <= -3.0) or np.any(env_unclipped >= 48.0):
        return True

    # Sensor bound check ([-10.0, 60.0] C, margin max(3*sigma_k, 2.0 C))
    for group in scenario.groups.values():
        for stream in group.sensors.values():
            margin = max(3.0 * stream.params.sigma_k, 2.0)
            sensor_clean = stream.canonical_clean_values[w_steps:t_steps]
            if np.any(sensor_clean <= -10.0 + margin) or np.any(sensor_clean >= 60.0 - margin):
                return True

    return False


def evaluate_hard_mechanisms(scenario: "VirtualScenario") -> List[str]:
    """Run all 7 hard-mechanism detectors and return list of verified mechanisms."""
    active: List[str] = []
    if detect_local_event(scenario):
        active.append("local_event")
    if detect_fault_during_event(scenario):
        active.append("fault_during_event")
    if detect_subtle_fault(scenario):
        active.append("subtle_fault")
    if detect_shared_cause_pair(scenario):
        active.append("shared_cause_pair")
    if detect_faulty_fraction_at_cap_small_group(scenario):
        active.append("faulty_fraction_at_cap_small_group")
    if detect_no_clear_majority(scenario):
        active.append("no_clear_majority")
    if detect_sensor_near_bounds(scenario):
        active.append("sensor_near_bounds")
    return active


# =============================================================================
# 3. VirtualScenario Container (Decision 40)
# =============================================================================

@dataclass
class VirtualScenario:
    """Container for a fully simulated multi-group scenario (Decision 40).
    
    Holds:
      - Scenario topology and difficulty metadata
      - Physical macro-environment series E(t)
      - Scenario-level genuine EventSeries
      - Co-located VirtualSensorGroup instances
      - Strict 6-column Layer 1 observable dataframe exporter
      - Comprehensive scenario metadata dictionary
    """
    scenario_id: str
    scenario_type: str
    difficulty_tier: str
    seed: int
    environment: EnvironmentSeries
    groups: Dict[str, VirtualSensorGroup]
    events: Optional[EventSeries] = None
    target_mechanism: str = "none"
    hard_mechanisms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None

    @property
    def total_sensors(self) -> int:
        """Total sensor count across all groups in the scenario."""
        return sum(len(g.sensors) for g in self.groups.values())

    @property
    def sensor_ids(self) -> List[str]:
        """List of all sensor IDs across all groups."""
        return [s_id for g in self.groups.values() for s_id in g.sensors.keys()]

    def to_observable_dataframe(self, stride: int = 1) -> pd.DataFrame:
        """Export raw observable readings strictly in 6-column Layer 1 schema (Decision 34 & 40).
        
        Columns: scenario_id, group_id, sensor_id, variable_type, timestamp, value.
        Ground truth is strictly isolated and never included.
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        dfs = []
        for gid, group in sorted(self.groups.items()):
            df_g = group.to_observable_dataframe(scenario_id=self.scenario_id, stride=stride)
            dfs.append(df_g)
        cols = ["scenario_id", "group_id", "sensor_id", "variable_type", "timestamp", "value"]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            return df[cols]
        return pd.DataFrame(columns=cols)

    def to_observables_dataframe(self, stride: int = 1) -> pd.DataFrame:
        """Convenience alias for to_observable_dataframe."""
        return self.to_observable_dataframe(stride=stride)

    def to_ground_truth_sensors_dataframe(self, stride: int = 1) -> pd.DataFrame:
        """Export per-sensor per-step ground truth table (16 columns strictly).
        
        Columns: scenario_id, group_id, sensor_id, timestamp, true_value,
                 sensor_untrusted, f_spike, f_offset_drift, f_gain_error,
                 f_frozen, f_high_noise, f_dropout, f_slow_degradation,
                 fault_severity, true_health, fault_id
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        dfs = []
        for gid, group in sorted(self.groups.items()):
            df_g = group.to_ground_truth_sensors_dataframe(scenario_id=self.scenario_id, stride=stride)
            dfs.append(df_g)
        cols = [
            "scenario_id", "group_id", "sensor_id", "timestamp", "true_value",
            "sensor_untrusted", "f_spike", "f_offset_drift", "f_gain_error",
            "f_frozen", "f_high_noise", "f_dropout", "f_slow_degradation",
            "fault_severity", "true_health", "fault_id"
        ]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            return df[cols]
        return pd.DataFrame(columns=cols)

    def to_ground_truth_events_dataframe(self, stride: int = 1) -> pd.DataFrame:
        """Export per-group per-step ground truth events table (9 columns strictly).
        
        Columns: scenario_id, group_id, timestamp, genuine_event,
                 event_below_detectability, event_id, event_magnitude_sigma,
                 true_event_breadth, true_group_value
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        dfs = []
        for gid, group in sorted(self.groups.items()):
            df_g = group.to_ground_truth_events_dataframe(scenario_id=self.scenario_id, stride=stride)
            dfs.append(df_g)
        cols = [
            "scenario_id", "group_id", "timestamp", "genuine_event",
            "event_below_detectability", "event_id", "event_magnitude_sigma",
            "true_event_breadth", "true_group_value"
        ]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            return df[cols]
        return pd.DataFrame(columns=cols)

    def to_fault_episodes_dataframe(self) -> pd.DataFrame:
        """Export fault episodes ground truth catalog (10 columns strictly).
        
        Columns: scenario_id, group_id, sensor_id, fault_id, fault_type,
                 start_step, end_step, duration, max_severity, shape_params
        """
        dfs = []
        for gid, group in sorted(self.groups.items()):
            df_g = group.fault_episodes_to_dataframe(scenario_id=self.scenario_id)
            dfs.append(df_g)
        cols = [
            "scenario_id", "group_id", "sensor_id", "fault_id", "fault_type",
            "start_step", "end_step", "duration", "max_severity", "shape_params"
        ]
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            return df[cols]
        return pd.DataFrame(columns=cols)

    def fault_episodes_to_dataframe(self) -> pd.DataFrame:
        """Convenience alias for to_fault_episodes_dataframe."""
        return self.to_fault_episodes_dataframe()

    def to_event_episodes_dataframe(self) -> pd.DataFrame:
        """Export genuine event episodes catalog (16 columns strictly).
        
        Columns: scenario_id, event_id, event_type, scope, start_step, end_step,
                 start_time_minutes, magnitude, abs_magnitude, sign,
                 affected_sensors, affected_groups, sensor_attenuations,
                 sensor_delays, shape_params, is_detectable
        """
        cols = [
            "scenario_id", "event_id", "event_type", "scope", "start_step",
            "end_step", "start_time_minutes", "magnitude", "abs_magnitude",
            "sign", "affected_sensors", "affected_groups", "sensor_attenuations",
            "sensor_delays", "shape_params", "is_detectable"
        ]
        if self.events is None or len(self.events.episodes) == 0:
            return pd.DataFrame(columns=cols)

        rows = []
        for ep in self.events.episodes:
            impacted = [
                gid for gid, g in sorted(self.groups.items())
                if any(s_id in g.sensors for s_id in ep.affected_sensors)
            ]
            rows.append({
                "scenario_id": self.scenario_id,
                "event_id": ep.event_id,
                "event_type": ep.event_type,
                "scope": ep.scope,
                "start_step": int(ep.start_step),
                "end_step": int(ep.end_step),
                "start_time_minutes": float(ep.start_time_minutes),
                "magnitude": float(ep.magnitude),
                "abs_magnitude": float(ep.abs_magnitude),
                "sign": int(ep.sign),
                "affected_sensors": ",".join(ep.affected_sensors),
                "affected_groups": ",".join(impacted),
                "sensor_attenuations": canonical_json(ep.sensor_attenuations),
                "sensor_delays": canonical_json(ep.sensor_delays),
                "shape_params": canonical_json(ep.shape_params),
                "is_detectable": bool(ep.is_detectable),
            })
        return pd.DataFrame(rows)[cols]

    def event_episodes_to_dataframe(self) -> pd.DataFrame:
        """Convenience alias for to_event_episodes_dataframe."""
        return self.to_event_episodes_dataframe()

    def episodes_to_dataframe(self) -> pd.DataFrame:
        """Convenience alias for to_event_episodes_dataframe."""
        return self.to_event_episodes_dataframe()

    def to_scenario_record(self, split: str = "unassigned") -> Dict[str, Any]:
        """Export scenario-level record (21 columns strictly)."""
        k_faulty = sum(
            1 for g in self.groups.values() for s in g.sensors.values()
            if s.fault_series is not None and s.fault_series.has_faults
        )
        dt = float(self.environment.timestamps_minutes[1] - self.environment.timestamps_minutes[0]) if len(self.environment.timestamps_minutes) > 1 else 1.0

        if self.difficulty_tier == "hard":
            if self.target_mechanism != "none":
                hard_case = self.target_mechanism
            elif self.hard_mechanisms:
                hard_case = self.hard_mechanisms[0]
            else:
                hard_case = "natural"
        else:
            hard_case = "none"

        group_sizes_dict = {gid: len(g.sensors) for gid, g in sorted(self.groups.items())}
        quant_dict = {
            s_id: float(s.params.quantisation_step)
            for gid, g in sorted(self.groups.items())
            for s_id, s in sorted(g.sensors.items())
        }
        clip_limits_dict = {
            "environment": [
                float(self.environment.params.environment_clip_min),
                float(self.environment.params.environment_clip_max),
            ],
            "sensor": [
                float(self.environment.params.sensor_clip_min),
                float(self.environment.params.sensor_clip_max),
            ],
        }

        gen_version = "1.0.0"
        if self.config and "generator_version" in self.config:
            gen_version = str(self.config["generator_version"])

        return {
            "scenario_id": self.scenario_id,
            "split": str(split),
            "generator_version": gen_version,
            "seed": int(self.seed),
            "scenario_type": self.scenario_type,
            "difficulty": self.difficulty_tier,
            "hard_case_type": hard_case,
            "sensor_count": int(self.total_sensors),
            "group_count": int(len(self.groups)),
            "group_sizes": canonical_json(group_sizes_dict),
            "length": int(self.environment.total_steps),
            "sampling": float(dt),
            "profile": str(self.environment.params.variable_type),
            "quantisation": canonical_json(quant_dict),
            "clip_limits": canonical_json(clip_limits_dict),
            "unit_scale": canonical_json(self.environment.params.unit_scaler.to_dict()),
            "faulty_sensors_count": int(k_faulty),
            "faulty_fraction": float(k_faulty / self.total_sensors) if self.total_sensors > 0 else 0.0,
            "has_events": bool(self.events is not None and len(self.events.episodes) > 0),
            "event_count": int(len(self.events.episodes) if self.events is not None else 0),
            "has_shared_cause": bool(detect_shared_cause_pair(self)),
        }

    def to_scenario_dataframe(self, split: str = "unassigned") -> pd.DataFrame:
        """Export scenario-level table (21 columns strictly)."""
        cols = [
            "scenario_id", "split", "generator_version", "seed", "scenario_type",
            "difficulty", "hard_case_type", "sensor_count", "group_count",
            "group_sizes", "length", "sampling", "profile", "quantisation",
            "clip_limits", "unit_scale", "faulty_sensors_count", "faulty_fraction",
            "has_events", "event_count", "has_shared_cause"
        ]
        record = self.to_scenario_record(split=split)
        return pd.DataFrame([record])[cols]

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Export scenario-level metadata summary."""
        k_faulty = sum(
            1 for g in self.groups.values() for s in g.sensors.values()
            if s.fault_series is not None and s.fault_series.has_faults
        )
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "scenario_type": self.scenario_type,
            "difficulty_tier": self.difficulty_tier,
            "sensor_count": self.total_sensors,
            "group_count": len(self.groups),
            "group_sizes": {gid: len(g.sensors) for gid, g in self.groups.items()},
            "target_mechanism": self.target_mechanism,
            "hard_mechanisms": list(self.hard_mechanisms),
            "faulty_sensor_count": k_faulty,
            "has_shared_cause": detect_shared_cause_pair(self),
            "has_events": (self.events is not None and len(self.events.episodes) > 0),
            "event_count": len(self.events.episodes) if self.events is not None else 0,
            "unit_scale_type": getattr(self.environment.params.unit_scaler, "scale_type", "celsius"),
        }


# =============================================================================
# 4. ScenarioSimulator Orchestrator (Decisions 35-41)
# =============================================================================

class ScenarioSimulator:
    """Simulator for generating complete virtual scenarios."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ScenarioSimulator from configuration dictionary or generator_a.yaml."""
        if config is None:
            default_yaml = get_config_dir() / "generator_a.yaml"
            self.config = load_yaml(default_yaml)
        else:
            self.config = config

        self.profile = TemperatureProfile(self.config)
        self.sensor_simulator = VirtualSensorSimulator(config=self.config, profile=self.profile)
        self.event_generator = EventGenerator(config=self.config)

    def simulate_scenario(
        self,
        scenario_id: str,
        seed: int,
        scenario_type: Optional[str] = None,
        difficulty_tier: Optional[str] = None,
        k_candidate: Optional[int] = None,
    ) -> VirtualScenario:
        """Simulate a complete scenario adhering strictly to Decisions 35-41.
        
        Args:
            scenario_id: Unique scenario identifier string.
            seed: Integer random seed for reproducibility.
            scenario_type: Optional explicit scenario type ('standard', 'two_groups', 'single_sensor').
            difficulty_tier: Optional explicit difficulty tier ('easy', 'medium', 'hard').
            k_candidate: Optional candidate fault count before caps.
            
        Returns:
            Fully simulated VirtualScenario instance.
        """
        rng = np.random.default_rng(seed)

        # 1. Sample topology and difficulty tier independently (Decision 35)
        scen_type = scenario_type if scenario_type is not None else sample_scenario_type(self.config, rng)
        group_sizes = sample_group_sizes(scen_type, self.config, rng)
        tier = difficulty_tier if difficulty_tier is not None else sample_difficulty_tier(self.config, rng)

        # 2. Build sensor IDs globally unique across the scenario
        group_sensor_params: Dict[str, List[TemperatureSensorParams]] = {}
        all_sensor_params: List[TemperatureSensorParams] = []
        global_s_idx = 0

        for g_idx, n_g in enumerate(group_sizes):
            gid = f"group_{g_idx:02d}"
            s_ids = [f"sensor_{global_s_idx + i:02d}" for i in range(n_g)]
            global_s_idx += n_g
            params_g = self.profile.sample_sensor_params(num_sensors=n_g, sensor_ids=s_ids, rng=rng)
            group_sensor_params[gid] = params_g
            all_sensor_params.extend(params_g)

        # 3. Simulate physical macro-environment E(t) shared by all groups (Decision 36)
        env_series = generate_environment_series(config=self.config, rng=rng)

        # 4. Generate genuine events over the entire scenario sensor population (Decision 36)
        allowed_scopes = ["global"] if tier == "easy" else None
        event_series = self.event_generator.generate_events(
            sensor_params=all_sensor_params,
            total_steps=env_series.total_steps,
            sampling_rate_minutes=self.sensor_simulator.sampling_rate_minutes,
            rng=rng,
            group_id="scenario",
            baseline_env_values=env_series.values,
            allowed_scopes=allowed_scopes,
        )

        # Extract active event steps in scored window
        ev_steps = extract_active_event_steps(event_series, env_series.warmup_steps, env_series.total_steps)

        # 5. Clean physical trajectories for gain_error x_ref calculation
        clean_traj: Dict[str, np.ndarray] = {}
        for s_param in all_sensor_params:
            contrib = event_series.sensor_contributions.get(
                s_param.sensor_id, np.zeros(env_series.total_steps, dtype=np.float64)
            )
            clean_traj[s_param.sensor_id] = np.clip(
                env_series.values + contrib,
                self.sensor_simulator.env_clip_min,
                self.sensor_simulator.env_clip_max,
            )

        # 6. Generate scenario faults enforcing tier constraints (Decisions 29 & 39)
        fault_groups = {gid: [s.sensor_id for s in params_g] for gid, params_g in group_sensor_params.items()}
        sensor_sigmas = {s.sensor_id: s.sigma_k for s in all_sensor_params}

        fault_map = generate_scenario_faults(
            groups=fault_groups,
            sensor_sigmas=sensor_sigmas,
            clean_physical_trajectories=clean_traj,
            events=event_series,
            active_event_steps=ev_steps,
            total_steps=env_series.total_steps,
            warmup_steps=env_series.warmup_steps,
            config=self.config,
            rng=rng,
            k_candidate=k_candidate,
            difficulty_tier=tier,
            scenario_id=scenario_id,
        )

        # 7. Simulate sensor streams per group (independent measurement noise per group)
        groups_dict: Dict[str, VirtualSensorGroup] = {}
        for gid, params_g in group_sensor_params.items():
            g_events = self._filter_events_for_group(event_series, params_g, gid, environment=env_series)
            group_sim = self.sensor_simulator.simulate_group(
                environment=env_series,
                sensor_params=params_g,
                group_id=gid,
                rng=rng,
                events=g_events,
                fault_map={s.sensor_id: fault_map[s.sensor_id] for s in params_g},
            )
            groups_dict[gid] = group_sim

        scenario = VirtualScenario(
            scenario_id=scenario_id,
            scenario_type=scen_type,
            difficulty_tier=tier,
            seed=seed,
            environment=env_series,
            groups=groups_dict,
            events=event_series,
            target_mechanism="none",
            hard_mechanisms=[],
            config=self.config,
        )

        # 8. HARD Tier Evaluation and Deterministic Constructive Fallback (Decision 38)
        if tier == "hard":
            verified = evaluate_hard_mechanisms(scenario)
            if verified:
                scenario.target_mechanism = "natural"
                scenario.hard_mechanisms = verified
            else:
                scenario = self._apply_hard_fallback(
                    scenario=scenario,
                    all_sensor_params=all_sensor_params,
                    group_sensor_params=group_sensor_params,
                    clean_traj=clean_traj,
                    rng=rng,
                )

        scenario.metadata = scenario.to_metadata_dict()
        return scenario

    def _filter_events_for_group(
        self,
        event_series: EventSeries,
        params_g: Sequence[TemperatureSensorParams],
        group_id: str,
        environment: Optional[EnvironmentSeries] = None,
    ) -> EventSeries:
        """Filter scenario EventSeries to group-specific view adhering strictly to Decision 45."""
        g_sids = {s.sensor_id for s in params_g}
        g_contrib = {
            s.sensor_id: event_series.sensor_contributions.get(
                s.sensor_id, np.zeros(event_series.total_steps, dtype=np.float64)
            )
            for s in params_g
        }

        t_steps = event_series.total_steps
        dt = self.sensor_simulator.sampling_rate_minutes
        t_minutes = np.arange(t_steps, dtype=np.float64) * dt
        thresh = self.event_generator.detectability_threshold_sigma
        activity_eps = self.event_generator.activity_epsilon_celsius

        group_episodes = [
            ep for ep in event_series.episodes
            if any(sid in g_sids for sid in ep.affected_sensors)
        ]

        ep_waves: Dict[str, Dict[str, np.ndarray]] = {}
        for ep in group_episodes:
            ep_waves[ep.event_id] = {
                s.sensor_id: ep.generate_sensor_waveform(s.sensor_id, t_minutes)
                for s in params_g
            }

        g_gen_event = np.zeros(t_steps, dtype=int)
        g_below = np.zeros(t_steps, dtype=int)
        g_mag_sigma = np.zeros(t_steps, dtype=np.float64)
        g_breadth = np.zeros(t_steps, dtype=np.float64)
        g_active_ids: List[str] = ["" for _ in range(t_steps)]
        g_true_group_val = np.zeros(t_steps, dtype=np.float64)

        for t in range(t_steps):
            # Check individual episode waveforms on affected group sensors
            active_eps = [
                ep for ep in group_episodes
                if any(
                    abs(ep_waves[ep.event_id][s.sensor_id][t]) > activity_eps
                    for s in params_g if s.sensor_id in ep.affected_sensors
                )
            ]
            active_eps_sorted = sorted(active_eps, key=lambda e: (e.start_step, e.event_id))
            if active_eps_sorted:
                g_active_ids[t] = ",".join(e.event_id for e in active_eps_sorted)

            any_detectable = any(
                abs(g_contrib[s.sensor_id][t]) >= thresh * s.sigma_k
                for s in params_g if s.sensor_id in g_contrib
            )

            if len(active_eps_sorted) > 0:
                max_sigma_ratio = max(
                    (abs(g_contrib[s.sensor_id][t]) / s.sigma_k for s in params_g if s.sensor_id in g_contrib),
                    default=0.0
                )
                active_sensors = {
                    s.sensor_id for s in params_g
                    if any(abs(ep_waves[ep.event_id][s.sensor_id][t]) > activity_eps for ep in group_episodes)
                }
                g_mag_sigma[t] = max_sigma_ratio
                g_breadth[t] = len(active_sensors) / len(params_g) if len(params_g) > 0 else 0.0

                if any_detectable:
                    g_gen_event[t] = 1
                    g_below[t] = 0
                else:
                    g_gen_event[t] = 0
                    g_below[t] = 1
            else:
                g_mag_sigma[t] = 0.0
                g_breadth[t] = 0.0
                g_gen_event[t] = 0
                g_below[t] = 0

            if environment is not None:
                base_val = environment.values[t]
                phys_vals = [
                    np.clip(base_val + g_contrib[s.sensor_id][t], self.sensor_simulator.env_clip_min, self.sensor_simulator.env_clip_max)
                    for s in params_g if s.sensor_id in g_contrib
                ]
                g_true_group_val[t] = float(np.mean(phys_vals)) if phys_vals else float(base_val)
            else:
                phys_vals = [
                    g_contrib[s.sensor_id][t]
                    for s in params_g if s.sensor_id in g_contrib
                ]
                g_true_group_val[t] = float(np.mean(phys_vals)) if phys_vals else 0.0

        return EventSeries(
            group_id=group_id,
            total_steps=t_steps,
            episodes=group_episodes,
            sensor_contributions=g_contrib,
            genuine_event=g_gen_event,
            event_below_detectability=g_below,
            event_magnitude_sigma=g_mag_sigma,
            true_event_breadth=g_breadth,
            active_event_ids=g_active_ids,
            true_group_value=g_true_group_val,
        )

    def _apply_hard_fallback(
        self,
        scenario: VirtualScenario,
        all_sensor_params: List[TemperatureSensorParams],
        group_sensor_params: Dict[str, List[TemperatureSensorParams]],
        clean_traj: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> VirtualScenario:
        """Apply deterministic constructive fallback for HARD tier (Decision 38).
        
        Zero-fault scenarios strictly use local_event fallback.
        Faulty scenarios sample from constructible targets with priority queue.
        """
        k_faulty = sum(
            1 for g in scenario.groups.values() for s in g.sensors.values()
            if s.fault_series is not None and s.fault_series.has_faults
        )

        if k_faulty == 0:
            target_list = ["local_event"]
        else:
            targets = ["local_event", "subtle_fault", "fault_during_event"]
            if any(len(g.sensors) == 4 for g in scenario.groups.values()):
                targets.append("faulty_fraction_at_cap_small_group")
            primary_target = str(rng.choice(targets))
            target_list = [primary_target] + [t for t in targets if t != primary_target]

        for target in target_list:
            scen_modified = self._construct_mechanism(
                target=target,
                scenario=scenario,
                all_sensor_params=all_sensor_params,
                group_sensor_params=group_sensor_params,
                clean_traj=clean_traj,
                rng=rng,
            )
            verified = evaluate_hard_mechanisms(scen_modified)
            if verified:
                scen_modified.target_mechanism = target
                scen_modified.hard_mechanisms = verified
                scen_modified.metadata = scen_modified.to_metadata_dict()
                return scen_modified

        # Fail-safe: local_event is algebraically guaranteed
        scen_safe = self._construct_mechanism(
            target="local_event",
            scenario=scenario,
            all_sensor_params=all_sensor_params,
            group_sensor_params=group_sensor_params,
            clean_traj=clean_traj,
            rng=rng,
        )
        scen_safe.target_mechanism = "local_event"
        scen_safe.hard_mechanisms = evaluate_hard_mechanisms(scen_safe)
        scen_safe.metadata = scen_safe.to_metadata_dict()
        return scen_safe

    def _construct_mechanism(
        self,
        target: str,
        scenario: VirtualScenario,
        all_sensor_params: List[TemperatureSensorParams],
        group_sensor_params: Dict[str, List[TemperatureSensorParams]],
        clean_traj: Dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> VirtualScenario:
        """Construct a specific hard mechanism adhering to Section 3 rules."""
        t_steps = scenario.environment.total_steps
        w_steps = scenario.environment.warmup_steps
        dt = self.sensor_simulator.sampling_rate_minutes
        t_minutes = np.arange(t_steps, dtype=np.float64) * dt

        if target == "local_event":
            # Decision 38: Guaranteed local event via sustained_shift, log-uniform [4, 8] C, 75% pos / 25% neg
            start_step = int(rng.integers(600, 1000))
            rise_min = float(self.config["genuine_events"]["durations"]["sustained_rise_min"])
            rise_max = float(self.config["genuine_events"]["durations"]["sustained_rise_max"])
            rise = float(rng.uniform(rise_min, rise_max))

            # Magnitude: log-uniform in [4.0, 8.0] C
            abs_mag = sample_log_uniform(4.0, 8.0, rng)
            sign = 1.0 if rng.random() < 0.75 else -1.0
            signed_mag = sign * abs_mag

            # Affected sensors: 1 sensor if N=1; 1-2 sensors if N>=2
            n_total = len(all_sensor_params)
            k_aff = 1 if n_total == 1 else int(rng.integers(1, min(n_total, 2) + 1))
            aff_indices = rng.choice(n_total, size=k_aff, replace=False)
            aff_sids = [all_sensor_params[i].sensor_id for i in aff_indices]

            att_min = float(self.config["genuine_events"]["scope"]["local_attenuation_min"])
            att_max = float(self.config["genuine_events"]["scope"]["local_attenuation_max"])
            attenuations = {sid: float(rng.uniform(att_min, att_max)) for sid in aff_sids}
            delays = {sid: float(rng.uniform(0.0, 10.0)) for sid in aff_sids}

            # Build single sustained shift episode
            ep = EventEpisode(
                event_id="event_local_guaranteed",
                event_type="sustained_shift",
                start_step=start_step,
                start_time_minutes=float(start_step * dt),
                magnitude=signed_mag,
                abs_magnitude=abs_mag,
                sign=int(sign),
                scope="local",
                affected_sensors=tuple(aff_sids),
                sensor_attenuations=attenuations,
                sensor_delays=delays,
                shape_params={"rise_minutes": rise},
                is_detectable=True,
                end_step=t_steps,
            )

            # Generate waveforms
            contrib_map: Dict[str, np.ndarray] = {s.sensor_id: np.zeros(t_steps, dtype=np.float64) for s in all_sensor_params}
            for sid in aff_sids:
                t_arr = float(start_step * dt + delays[sid])
                s_mag = signed_mag * attenuations[sid]
                w = generate_sustained_shift(t_minutes, t_arr, s_mag, rise)
                contrib_map[sid] = w

            # Detectability masks
            thresh = self.event_generator.detectability_threshold_sigma
            gen_event_mask = np.zeros(t_steps, dtype=int)
            for t in range(t_steps):
                if any(abs(contrib_map[sid][t]) >= thresh * s.sigma_k for s in all_sensor_params for sid in aff_sids if s.sensor_id == sid):
                    gen_event_mask[t] = 1

            new_events = EventSeries(
                group_id="scenario",
                total_steps=t_steps,
                episodes=[ep],
                sensor_contributions=contrib_map,
                genuine_event=gen_event_mask,
                event_below_detectability=np.zeros(t_steps, dtype=int),
                event_magnitude_sigma=np.zeros(t_steps, dtype=np.float64),
                true_event_breadth=np.full(t_steps, float(k_aff) / float(n_total)),
                active_event_ids=["event_local_guaranteed" if gen_event_mask[t] else "" for t in range(t_steps)],
                true_group_value=np.zeros(t_steps, dtype=np.float64),
            )

            # Re-transduce sensor groups with new events
            return self._rebuild_scenario_with_events_and_faults(
                scenario=scenario,
                new_events=new_events,
                all_sensor_params=all_sensor_params,
                group_sensor_params=group_sensor_params,
                rng=rng,
            )

        elif target == "fault_during_event":
            # Decision 38: Guaranteed fault during ramp event with hold >= 60 steps
            t_ev = int(rng.integers(500, 1400))
            t_rise = float(rng.uniform(20.0, 60.0))
            t_hold = float(rng.uniform(60.0, 180.0))  # Hold duration >= 60 steps
            t_fall = float(rng.uniform(20.0, 60.0))
            abs_mag = sample_log_uniform(1.0, 5.0, rng)
            sign = 1.0 if rng.random() < 0.75 else -1.0
            signed_mag = sign * abs_mag

            n_total = len(all_sensor_params)
            aff_sids = [s.sensor_id for s in all_sensor_params]
            attenuations = {sid: float(rng.uniform(0.6, 1.0)) for sid in aff_sids}
            delays = {sid: float(rng.uniform(0.0, 10.0)) for sid in aff_sids}

            ep = EventEpisode(
                event_id="event_ramp_guaranteed",
                event_type="ramp",
                start_step=t_ev,
                start_time_minutes=float(t_ev * dt),
                magnitude=signed_mag,
                abs_magnitude=abs_mag,
                sign=int(sign),
                scope="global",
                affected_sensors=tuple(aff_sids),
                sensor_attenuations=attenuations,
                sensor_delays=delays,
                shape_params={"rise_minutes": t_rise, "hold_minutes": t_hold, "fall_minutes": t_fall},
                is_detectable=True,
                end_step=min(t_steps, int(np.ceil((t_ev * dt + 10.0 + t_rise + t_hold + t_fall) / dt))),
            )

            contrib_map = {s.sensor_id: np.zeros(t_steps, dtype=np.float64) for s in all_sensor_params}
            for sid in aff_sids:
                t_arr = float(t_ev * dt + delays[sid])
                s_mag = signed_mag * attenuations[sid]
                w = generate_ramp(t_minutes, t_arr, s_mag, t_rise, t_hold, t_fall)
                contrib_map[sid] = w

            gen_event_mask = np.zeros(t_steps, dtype=int)
            thresh = self.event_generator.detectability_threshold_sigma
            for t in range(t_steps):
                if any(abs(contrib_map[sid][t]) >= thresh * s.sigma_k for s in all_sensor_params):
                    gen_event_mask[t] = 1

            new_events = EventSeries(
                group_id="scenario",
                total_steps=t_steps,
                episodes=[ep],
                sensor_contributions=contrib_map,
                genuine_event=gen_event_mask,
                event_below_detectability=np.zeros(t_steps, dtype=int),
                event_magnitude_sigma=np.zeros(t_steps, dtype=np.float64),
                true_event_breadth=np.ones(t_steps),
                active_event_ids=["event_ramp_guaranteed" if gen_event_mask[t] else "" for t in range(t_steps)],
                true_group_value=np.zeros(t_steps, dtype=np.float64),
            )

            # Condition Episode 1 start on first faulty sensor to fall inside the scored plateau
            plateau_start = min(t_steps - 20, int(t_ev + np.ceil(t_rise / dt)))
            plateau_end = min(t_steps - 10, int(plateau_start + np.floor(t_hold / dt)))
            fault_start = int(rng.integers(plateau_start, plateau_end)) if plateau_end > plateau_start else plateau_start

            # Find first faulty sensor and update its fault series
            fault_map = self._extract_existing_fault_map(scenario)
            faulty_sids = [sid for sid, series in fault_map.items() if series.has_faults]
            if faulty_sids:
                target_sid = faulty_sids[0]
                ep_old = fault_map[target_sid].episodes[0]
                # Shift episode start to fault_start
                sim = FaultSimulator(config=self.config)
                s_sigma = next(s.sigma_k for s in all_sensor_params if s.sensor_id == target_sid)
                xref = scenario.environment.values[fault_start]
                ep_new = sim.sample_fault(
                    ep_old.fault_type, target_sid, sigma_k=s_sigma,
                    total_steps=t_steps, warmup_steps=w_steps,
                    physical_xref=xref, clean_base_value=xref,
                    rng=rng, fault_id=ep_old.fault_id, start_step=fault_start,
                )
                fault_map[target_sid].episodes[0] = ep_new

            return self._rebuild_scenario_with_events_and_faults(
                scenario=scenario,
                new_events=new_events,
                all_sensor_params=all_sensor_params,
                group_sensor_params=group_sensor_params,
                fault_map=fault_map,
                rng=rng,
            )

        elif target == "subtle_fault":
            # Decision 38: Guaranteed subtle fault preserving sampled type count m
            fault_map = self._extract_existing_fault_map(scenario)
            faulty_sids = [sid for sid, series in fault_map.items() if series.has_faults]
            if faulty_sids:
                target_sid = faulty_sids[0]
                old_episodes = fault_map[target_sid].episodes
                m = len(old_episodes)
                s_sigma = next(s.sigma_k for s in all_sensor_params if s.sensor_id == target_sid)
                sim = FaultSimulator(config=self.config)

                # Sample subtle gain: 50% in [0.90, 0.95], 50% in [1.05, 1.10]
                subtle_g = float(rng.uniform(0.90, 0.95)) if rng.random() < 0.5 else float(rng.uniform(1.05, 1.10))

                # Preserve m using mechanically verified valid combinations containing gain_error
                new_types = sample_fault_types_for_sensor(
                    type_count=m,
                    config=self.config,
                    rng=rng,
                    shared_fault_type="gain_error",
                    scenario_id=scenario.scenario_id,
                    sensor_id=target_sid,
                )

                new_episodes: List[FaultEpisode] = []
                for idx, ft_name in enumerate(new_types):
                    t_start = old_episodes[idx].start_step if idx < len(old_episodes) else int(rng.integers(600, 1600))
                    xref = scenario.environment.values[t_start]
                    kwargs: Dict[str, Any] = {}
                    if ft_name == "gain_error":
                        kwargs["gain"] = subtle_g
                    ep_new = sim.sample_fault(
                        FaultType(ft_name), target_sid, sigma_k=s_sigma,
                        total_steps=t_steps, warmup_steps=w_steps,
                        physical_xref=xref, clean_base_value=xref,
                        rng=rng, fault_id=f"fault_{target_sid}_{idx}", start_step=t_start,
                        **kwargs,
                    )
                    new_episodes.append(ep_new)

                fault_map[target_sid] = SensorFaultSeries(sensor_id=target_sid, total_steps=t_steps, episodes=new_episodes)

            return self._rebuild_scenario_with_events_and_faults(
                scenario=scenario,
                new_events=scenario.events,
                all_sensor_params=all_sensor_params,
                group_sensor_params=group_sensor_params,
                fault_map=fault_map,
                rng=rng,
            )

        elif target == "faulty_fraction_at_cap_small_group":
            # Re-run fault generation guaranteeing small group (n=4) receives exactly 1 fault
            small_gids = [gid for gid, g in scenario.groups.items() if len(g.sensors) == 4]
            if small_gids:
                target_gid = small_gids[0]
                fault_groups = {gid: [s.sensor_id for s in g.sensors.values()] for gid, g in scenario.groups.items()}
                sensor_sigmas = {s.sensor_id: s.sigma_k for s in all_sensor_params}
                k_curr = max(1, sum(1 for g in scenario.groups.values() for s in g.sensors.values() if s.fault_series and s.fault_series.has_faults))
                target_sensors = sorted(list(fault_groups[target_gid]))
                chosen_sensor = str(rng.choice(target_sensors))

                allocated = allocate_faulty_sensors(
                    k_actual=k_curr,
                    groups=fault_groups,
                    rng=rng,
                    pre_allocated_sensors=[chosen_sensor],
                    pre_allocated_group=target_gid,
                )
                fault_sim = FaultSimulator(config=self.config)
                valid_triplets = derive_valid_triplets(self.config)
                valid_pairs = get_valid_pairs(self.config)
                ev_steps = extract_active_event_steps(scenario.events, w_steps, t_steps) if scenario.events else []
                episodes_by_sensor: Dict[str, List[FaultEpisode]] = {s.sensor_id: [] for s in all_sensor_params}

                for sid, _ in allocated:
                    m = min(sample_fault_type_count(self.config, rng), 3)
                    types = sample_fault_types_for_sensor(
                        m, self.config, rng, valid_triplets=valid_triplets, valid_pairs=valid_pairs,
                        scenario_id=scenario.scenario_id, sensor_id=sid,
                    )
                    for idx, ft_name in enumerate(types):
                        prec = episodes_by_sensor[sid][-1] if episodes_by_sensor[sid] else None
                        t_start = sample_episode_start_step(prec, t_steps, w_steps, ev_steps, self.config, rng)
                        xref = clean_traj[sid][t_start] if sid in clean_traj else 20.0
                        ep = fault_sim.sample_fault(
                            FaultType(ft_name), sid, sigma_k=sensor_sigmas[sid],
                            total_steps=t_steps, warmup_steps=w_steps,
                            physical_xref=xref, clean_base_value=xref,
                            rng=rng, fault_id=f"fault_{sid}_{idx}", start_step=t_start,
                            difficulty_tier="hard",
                        )
                        episodes_by_sensor[sid].append(ep)

                new_fault_map = {
                    sid: SensorFaultSeries(sensor_id=sid, total_steps=t_steps, episodes=episodes_by_sensor[sid])
                    for sid in [s.sensor_id for s in all_sensor_params]
                }
                return self._rebuild_scenario_with_events_and_faults(
                    scenario=scenario,
                    new_events=scenario.events,
                    all_sensor_params=all_sensor_params,
                    group_sensor_params=group_sensor_params,
                    fault_map=new_fault_map,
                    rng=rng,
                )

        return scenario

    def _extract_existing_fault_map(self, scenario: VirtualScenario) -> Dict[str, SensorFaultSeries]:
        """Extract existing SensorFaultSeries mapping from scenario."""
        fault_map: Dict[str, SensorFaultSeries] = {}
        for group in scenario.groups.values():
            for stream in group.sensors.values():
                if stream.fault_series is not None:
                    fault_map[stream.sensor_id] = stream.fault_series
                else:
                    fault_map[stream.sensor_id] = SensorFaultSeries(
                        sensor_id=stream.sensor_id,
                        total_steps=scenario.environment.total_steps,
                        episodes=[],
                    )
        return fault_map

    def _rebuild_scenario_with_events_and_faults(
        self,
        scenario: VirtualScenario,
        new_events: Optional[EventSeries],
        all_sensor_params: List[TemperatureSensorParams],
        group_sensor_params: Dict[str, List[TemperatureSensorParams]],
        fault_map: Optional[Dict[str, SensorFaultSeries]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> VirtualScenario:
        """Re-transduce all sensor streams in the scenario with updated events and faults."""
        if rng is None:
            rng = np.random.default_rng(scenario.seed)

        active_fault_map = fault_map if fault_map is not None else self._extract_existing_fault_map(scenario)
        groups_dict: Dict[str, VirtualSensorGroup] = {}

        for gid, params_g in group_sensor_params.items():
            g_events = self._filter_events_for_group(new_events, params_g, gid, environment=scenario.environment) if new_events is not None else None
            group_sim = self.sensor_simulator.simulate_group(
                environment=scenario.environment,
                sensor_params=params_g,
                group_id=gid,
                rng=rng,
                events=g_events,
                fault_map={s.sensor_id: active_fault_map[s.sensor_id] for s in params_g},
            )
            groups_dict[gid] = group_sim

        return VirtualScenario(
            scenario_id=scenario.scenario_id,
            scenario_type=scenario.scenario_type,
            difficulty_tier=scenario.difficulty_tier,
            seed=scenario.seed,
            environment=scenario.environment,
            groups=groups_dict,
            events=new_events,
            target_mechanism=scenario.target_mechanism,
            hard_mechanisms=scenario.hard_mechanisms,
            metadata=scenario.metadata,
            config=self.config,
        )


def simulate_scenario(
    scenario_id: str,
    seed: int,
    scenario_type: Optional[str] = None,
    difficulty_tier: Optional[str] = None,
    k_candidate: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> VirtualScenario:
    """Convenience top-level function to simulate a complete virtual scenario (P1.7)."""
    simulator = ScenarioSimulator(config=config)
    return simulator.simulate_scenario(
        scenario_id=scenario_id,
        seed=seed,
        scenario_type=scenario_type,
        difficulty_tier=difficulty_tier,
        k_candidate=k_candidate,
    )
