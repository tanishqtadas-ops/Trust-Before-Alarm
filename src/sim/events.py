"""Genuine environmental events simulation for sensor stream intelligence.

Implements Appendix A.3 of the specification (Decisions 5, 18, 19, 20, 21):
  1. Four configured event types:
     - Step with exponential recovery (tau in [10, 120] min, 40%)
     - Ramp (rise/fall in [20, 180] min, hold in [30, 180] min, 25%, asymmetric)
     - Pulse (duration in [5, 30] min, 25%)
     - Sustained shift (rise in [5, 20] min, holds until scenario end, 10%)
  2. Event count: Poisson mean 1.2 capped at max 3 (~30% none)
  3. Magnitude: log-uniform [0.5, 10.0] C; 75% positive, 25% negative
  4. Scopes and attenuation:
     - Global: 55%, all sensors, attenuation in [0.6, 1.0]
     - Partial: 30%, 40% to 80% of sensors, attenuation in [0.3, 1.0]
     - Local: 15%, 1 to 2 sensors, attenuation in [0.7, 1.0]
  5. Arrival delay: uniform [0, 10] min per sensor
  6. Detectability: genuine_event = 1 while event contribution at >= 1 sensor exceeds 3 * sigma_k;
     else event_below_detectability = 1 while event is active in the physical environment.
  7. Physical signal alteration: Genuine events alter the physical temperature before sensor
     transduction (lag, offset, noise, quantisation, clipping, missing data, unit scaling).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.sim.environment import EnvironmentSeries
from src.sim.profile import TemperatureSensorParams
from src.utils.config import canonical_json, get_config_dir, load_yaml


def generate_step_exponential_recovery(
    t_minutes: np.ndarray,
    t_arrival: float,
    magnitude: float,
    tau: float,
) -> np.ndarray:
    """Generate step with exponential recovery event waveform (Decision 20).
    
    Delta T(t) = magnitude * exp(-(t - t_arrival) / tau) for t >= t_arrival,
    and 0 for t < t_arrival.
    
    Args:
        t_minutes: Elapsed time array in minutes, shape (T,).
        t_arrival: Wave arrival timestamp in minutes.
        magnitude: Peak signed temperature perturbation in Celsius.
        tau: Exponential recovery time constant in minutes.
        
    Returns:
        Array of event perturbations, shape (T,).
    """
    if len(t_minutes) == 0:
        return np.empty(0, dtype=np.float64)
    out = np.zeros_like(t_minutes, dtype=np.float64)
    mask = t_minutes >= t_arrival
    if np.any(mask):
        if tau > 0.0:
            out[mask] = magnitude * np.exp(-(t_minutes[mask] - t_arrival) / tau)
        else:
            out[mask] = magnitude
    return out


def generate_ramp(
    t_minutes: np.ndarray,
    t_arrival: float,
    magnitude: float,
    rise_minutes: float,
    hold_minutes: float,
    fall_minutes: float,
) -> np.ndarray:
    """Generate ramp event waveform with linear rise, flat hold, and linear fall (Decision 19).
    
    Phases:
      - t < t_0: 0.0
      - t_0 <= t < t_1: linear rise from 0 to magnitude over rise_minutes
      - t_1 <= t < t_2: flat hold at magnitude over hold_minutes
      - t_2 <= t < t_3: linear fall from magnitude to 0 over fall_minutes
      - t >= t_3: 0.0
      
    Args:
        t_minutes: Elapsed time array in minutes, shape (T,).
        t_arrival: Wave arrival timestamp in minutes (t_0).
        magnitude: Peak signed temperature perturbation in Celsius.
        rise_minutes: Duration of linear rise in minutes.
        hold_minutes: Duration of plateau hold in minutes.
        fall_minutes: Duration of linear return in minutes.
        
    Returns:
        Array of event perturbations, shape (T,).
    """
    if len(t_minutes) == 0:
        return np.empty(0, dtype=np.float64)
    out = np.zeros_like(t_minutes, dtype=np.float64)
    t_0 = float(t_arrival)
    t_1 = t_0 + max(0.0, float(rise_minutes))
    t_2 = t_1 + max(0.0, float(hold_minutes))
    t_3 = t_2 + max(0.0, float(fall_minutes))

    # Rise phase
    rise_mask = (t_minutes >= t_0) & (t_minutes < t_1)
    if np.any(rise_mask):
        if t_1 > t_0:
            out[rise_mask] = magnitude * (t_minutes[rise_mask] - t_0) / (t_1 - t_0)
        else:
            out[rise_mask] = magnitude

    # Hold phase
    hold_mask = (t_minutes >= t_1) & (t_minutes < t_2)
    out[hold_mask] = magnitude

    # Fall phase
    fall_mask = (t_minutes >= t_2) & (t_minutes < t_3)
    if np.any(fall_mask):
        if t_3 > t_2:
            out[fall_mask] = magnitude * (1.0 - (t_minutes[fall_mask] - t_2) / (t_3 - t_2))
        else:
            out[fall_mask] = 0.0

    return out


def generate_pulse(
    t_minutes: np.ndarray,
    t_arrival: float,
    magnitude: float,
    duration_minutes: float,
) -> np.ndarray:
    """Generate pulse event waveform (Decision 21).
    
    Delta T(t) = magnitude for t_arrival <= t < t_arrival + duration_minutes,
    and 0 elsewhere.
    
    Args:
        t_minutes: Elapsed time array in minutes, shape (T,).
        t_arrival: Wave arrival timestamp in minutes.
        magnitude: Signed pulse amplitude in Celsius.
        duration_minutes: Pulse duration in minutes.
        
    Returns:
        Array of event perturbations, shape (T,).
    """
    if len(t_minutes) == 0:
        return np.empty(0, dtype=np.float64)
    out = np.zeros_like(t_minutes, dtype=np.float64)
    t_end = float(t_arrival) + max(0.0, float(duration_minutes))
    mask = (t_minutes >= t_arrival) & (t_minutes < t_end)
    out[mask] = magnitude
    return out


def generate_sustained_shift(
    t_minutes: np.ndarray,
    t_arrival: float,
    magnitude: float,
    rise_minutes: float,
) -> np.ndarray:
    """Generate sustained shift event waveform (Decision 5).
    
    Linear ramp from 0 to magnitude over rise_minutes, then holds at elevated level
    until the scenario ends.
    
    Args:
        t_minutes: Elapsed time array in minutes, shape (T,).
        t_arrival: Wave arrival timestamp in minutes.
        magnitude: Signed sustained plateau amplitude in Celsius.
        rise_minutes: Rise time in minutes.
        
    Returns:
        Array of event perturbations, shape (T,).
    """
    if len(t_minutes) == 0:
        return np.empty(0, dtype=np.float64)
    out = np.zeros_like(t_minutes, dtype=np.float64)
    t_0 = float(t_arrival)
    t_1 = t_0 + max(0.0, float(rise_minutes))

    # Rise phase
    rise_mask = (t_minutes >= t_0) & (t_minutes < t_1)
    if np.any(rise_mask):
        if t_1 > t_0:
            out[rise_mask] = magnitude * (t_minutes[rise_mask] - t_0) / (t_1 - t_0)
        else:
            out[rise_mask] = magnitude

    # Sustained hold phase until scenario end
    hold_mask = t_minutes >= t_1
    out[hold_mask] = magnitude
    return out


@dataclass(frozen=True)
class EventEpisode:
    """Ground-truth metadata for a single genuine environmental event episode."""
    event_id: str
    event_type: str
    start_step: int
    start_time_minutes: float
    magnitude: float
    abs_magnitude: float
    sign: int
    scope: str
    affected_sensors: Tuple[str, ...]
    sensor_attenuations: Dict[str, float]
    sensor_delays: Dict[str, float]
    shape_params: Dict[str, float]
    is_detectable: bool
    end_step: int

    def generate_sensor_waveform(
        self,
        sensor_id: str,
        t_minutes: np.ndarray,
    ) -> np.ndarray:
        """Compute the signed physical temperature perturbation on a specific sensor."""
        if sensor_id not in self.affected_sensors or len(t_minutes) == 0:
            return np.zeros_like(t_minutes, dtype=np.float64)
        t_arr = self.start_time_minutes + self.sensor_delays.get(sensor_id, 0.0)
        s_mag = self.magnitude * self.sensor_attenuations.get(sensor_id, 1.0)
        e_type = self.event_type
        if e_type == "step_exponential_recovery":
            return generate_step_exponential_recovery(
                t_minutes=t_minutes,
                t_arrival=t_arr,
                magnitude=s_mag,
                tau=float(self.shape_params["tau_minutes"]),
            )
        elif e_type == "ramp":
            return generate_ramp(
                t_minutes=t_minutes,
                t_arrival=t_arr,
                magnitude=s_mag,
                rise_minutes=float(self.shape_params["rise_minutes"]),
                hold_minutes=float(self.shape_params["hold_minutes"]),
                fall_minutes=float(self.shape_params["fall_minutes"]),
            )
        elif e_type == "pulse":
            return generate_pulse(
                t_minutes=t_minutes,
                t_arrival=t_arr,
                magnitude=s_mag,
                duration_minutes=float(self.shape_params["duration_minutes"]),
            )
        elif e_type == "sustained_shift":
            return generate_sustained_shift(
                t_minutes=t_minutes,
                t_arrival=t_arr,
                magnitude=s_mag,
                rise_minutes=float(self.shape_params["rise_minutes"]),
            )
        return np.zeros_like(t_minutes, dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode metadata for ground-truth episode tables."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "start_step": self.start_step,
            "start_time_minutes": self.start_time_minutes,
            "magnitude": self.magnitude,
            "abs_magnitude": self.abs_magnitude,
            "sign": self.sign,
            "scope": self.scope,
            "affected_sensors": list(self.affected_sensors),
            "sensor_attenuations": dict(self.sensor_attenuations),
            "sensor_delays": dict(self.sensor_delays),
            "shape_params": dict(self.shape_params),
            "is_detectable": self.is_detectable,
            "end_step": self.end_step,
        }


@dataclass(frozen=True)
class EventSeries:
    """Group-level event contributions and ground-truth masks across all time steps.
    
    Attributes:
        group_id: Unique group identifier.
        total_steps: Scenario duration in steps.
        episodes: List of individual EventEpisode objects.
        sensor_contributions: Dict mapping sensor_id to event contribution array Delta T_k(t).
        genuine_event: Binary array (1 where event contribution at >= 1 sensor exceeds 3 * sigma_k).
        event_below_detectability: Binary array (1 where event is active but below 3 * sigma_k).
        event_magnitude_sigma: Array of max normalized event magnitude max_k(|Delta T_k(t)| / sigma_k).
        true_event_breadth: Array of fraction of sensors in group affected by active event at step t.
        active_event_ids: List of comma-separated active event IDs for each step.
        true_group_value: Ground-truth canonical temperature across the group at each step.
    """
    group_id: str
    total_steps: int
    episodes: List[EventEpisode]
    sensor_contributions: Dict[str, np.ndarray]
    genuine_event: np.ndarray
    event_below_detectability: np.ndarray
    event_magnitude_sigma: np.ndarray
    true_event_breadth: np.ndarray
    active_event_ids: List[str]
    true_group_value: np.ndarray

    @property
    def num_events(self) -> int:
        """Count of event episodes generated."""
        return len(self.episodes)

    @property
    def has_events(self) -> bool:
        """True if one or more event episodes are present."""
        return len(self.episodes) > 0

    def to_ground_truth_dataframe(
        self,
        scenario_id: str = "scenario_0000",
        sampling_rate_minutes: float = 1.0,
        stride: int = 1,
    ) -> pd.DataFrame:
        """Export group ground-truth per-step table as specified in Appendix A.7."""
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        timestamps = (np.arange(self.total_steps, dtype=np.float64) * sampling_rate_minutes)[::stride]
        cols = [
            "scenario_id", "group_id", "timestamp", "genuine_event",
            "event_below_detectability", "event_id", "event_magnitude_sigma",
            "true_event_breadth", "true_group_value"
        ]
        return pd.DataFrame({
            "scenario_id": scenario_id,
            "group_id": self.group_id,
            "timestamp": timestamps,
            "genuine_event": self.genuine_event[::stride],
            "event_below_detectability": self.event_below_detectability[::stride],
            "event_id": self.active_event_ids[::stride],
            "event_magnitude_sigma": self.event_magnitude_sigma[::stride],
            "true_event_breadth": self.true_event_breadth[::stride],
            "true_group_value": self.true_group_value[::stride],
        })[cols]

    def episodes_to_dataframe(
        self,
        scenario_id: str = "scenario_0000",
    ) -> pd.DataFrame:
        """Export episode ground-truth table as specified in Appendix A.7."""
        cols = [
            "scenario_id", "group_id", "event_id", "event_type", "scope",
            "start_step", "end_step", "start_time_minutes", "magnitude",
            "abs_magnitude", "sign", "affected_sensors", "sensor_attenuations",
            "sensor_delays", "shape_params", "is_detectable"
        ]
        if not self.episodes:
            return pd.DataFrame(columns=cols)
        rows = []
        for ep in self.episodes:
            rows.append({
                "scenario_id": scenario_id,
                "group_id": self.group_id,
                "event_id": ep.event_id,
                "event_type": ep.event_type,
                "scope": ep.scope,
                "start_step": ep.start_step,
                "end_step": ep.end_step,
                "start_time_minutes": ep.start_time_minutes,
                "magnitude": ep.magnitude,
                "abs_magnitude": ep.abs_magnitude,
                "sign": ep.sign,
                "affected_sensors": ",".join(ep.affected_sensors),
                "sensor_attenuations": canonical_json(ep.sensor_attenuations),
                "sensor_delays": canonical_json(ep.sensor_delays),
                "shape_params": canonical_json(ep.shape_params),
                "is_detectable": bool(ep.is_detectable),
            })
        return pd.DataFrame(rows)[cols]


class EventGenerator:
    """Generator for genuine environmental events driven strictly by configuration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize EventGenerator from configuration dictionary or default YAML.
        
        Args:
            config: Optional full configuration dictionary.
        """
        if config is None:
            default_yaml = get_config_dir() / "generator_a.yaml"
            self.config = load_yaml(default_yaml)
        else:
            self.config = config

        self._extract_config()

    def _extract_config(self) -> None:
        """Extract and validate all genuine event parameters from configuration."""
        events_cfg = self.config["genuine_events"]
        struct_cfg = self.config["structure"]
        bounds_cfg = self.config["temperature_profile"]["bounds"]

        # Structure
        self.sampling_rate_minutes = float(struct_cfg["sampling_rate_minutes"])
        self.total_steps = int(struct_cfg["total_steps"])
        self.warmup_steps = int(struct_cfg["warmup_steps"])
        self.scored_steps = int(struct_cfg["scored_steps"])

        # Environmental bounds
        self.env_clip_min = float(bounds_cfg["environment_clip_min"])
        self.env_clip_max = float(bounds_cfg["environment_clip_max"])

        # Count distribution
        self.count_poisson_mean = float(events_cfg["count_poisson_mean"])
        self.count_max = int(events_cfg["count_max"])

        # Event type probabilities
        self.type_probs = dict(events_cfg["type_probs"])
        total_type_prob = sum(self.type_probs.values())
        if not np.isclose(total_type_prob, 1.0, atol=1e-3):
            raise ValueError(f"Event type probabilities sum to {total_type_prob}, expected 1.0")

        # Magnitude and sign
        self.magnitude_min = float(events_cfg["magnitude_min"])
        self.magnitude_max = float(events_cfg["magnitude_max"])
        self.positive_sign_prob = float(events_cfg["positive_sign_prob"])
        self.negative_sign_prob = float(events_cfg["negative_sign_prob"])

        # Durations
        self.durations_cfg = dict(events_cfg["durations"])

        # Scope
        self.scope_cfg = dict(events_cfg["scope"])

        # Arrival delay
        self.arrival_delay_min = float(events_cfg["arrival_delay_min"])
        self.arrival_delay_max = float(events_cfg["arrival_delay_max"])

        # Detectability threshold
        self.detectability_threshold_sigma = float(events_cfg["detectability_threshold_sigma"])
        self.activity_epsilon_celsius = float(events_cfg.get("activity_epsilon_celsius", 1e-4))

        # Start step bounds (Decision 18: scored window [480, 1919] inclusive)
        self.start_step_min = int(events_cfg.get("start_step_min", self.warmup_steps))
        self.start_step_max = int(events_cfg.get("start_step_max", self.total_steps - 1))
        if self.start_step_min > self.start_step_max:
            raise ValueError(f"start_step_min ({self.start_step_min}) > start_step_max ({self.start_step_max})")

    def sample_event_count(self, rng: np.random.Generator) -> int:
        """Sample event count from Poisson distribution capped at count_max."""
        raw_count = int(rng.poisson(lam=self.count_poisson_mean))
        return min(raw_count, self.count_max)

    def sample_event_type(self, rng: np.random.Generator) -> str:
        """Sample event type from configured categorical probabilities."""
        types = list(self.type_probs.keys())
        probs = np.array([self.type_probs[t] for t in types], dtype=np.float64)
        probs = probs / np.sum(probs)
        return str(rng.choice(types, p=probs))

    def sample_magnitude(self, rng: np.random.Generator) -> Tuple[float, float, int]:
        """Sample log-uniform magnitude and sign.
        
        Returns:
            Tuple of (signed_magnitude, abs_magnitude, sign).
        """
        log_mag = rng.uniform(np.log(self.magnitude_min), np.log(self.magnitude_max))
        abs_mag = float(np.exp(log_mag))
        sign_val = 1 if rng.uniform(0.0, 1.0) < self.positive_sign_prob else -1
        signed_mag = float(sign_val * abs_mag)
        return signed_mag, abs_mag, sign_val

    def sample_scope_and_sensors(
        self,
        sensor_ids: Sequence[str],
        rng: np.random.Generator,
        allowed_scopes: Optional[Sequence[str]] = None,
    ) -> Tuple[str, List[str], Dict[str, float]]:
        """Sample event scope, affected sensors, and per-sensor attenuations.
        
        Args:
            sensor_ids: Sequence of sensor IDs.
            rng: Numpy Generator.
            allowed_scopes: Optional list of allowed scope names (e.g. ["global"] for EASY tier).
            
        Returns:
            Tuple of (scope_name, affected_sensor_ids, attenuation_dict).
        """
        all_scopes = ["global", "partial", "local"]
        all_probs = np.array([
            float(self.scope_cfg["global_prob"]),
            float(self.scope_cfg["partial_prob"]),
            float(self.scope_cfg["local_prob"]),
        ], dtype=np.float64)

        if allowed_scopes is not None:
            allowed_set = set(allowed_scopes)
            scopes = [s for s in all_scopes if s in allowed_set]
            if not scopes:
                scopes = ["global"]
                probs = np.array([1.0], dtype=np.float64)
            else:
                probs = np.array([all_probs[all_scopes.index(s)] for s in scopes], dtype=np.float64)
                probs = probs / np.sum(probs)
        else:
            scopes = all_scopes
            probs = all_probs / np.sum(all_probs)

        selected_scope = str(rng.choice(scopes, p=probs))

        n_sensors = len(sensor_ids)
        attenuations: Dict[str, float] = {}

        if selected_scope == "global":
            affected = list(sensor_ids)
            att_min = float(self.scope_cfg["global_attenuation_min"])
            att_max = float(self.scope_cfg["global_attenuation_max"])
            for s_id in affected:
                attenuations[s_id] = float(rng.uniform(att_min, att_max))

        elif selected_scope == "partial":
            frac = rng.uniform(
                float(self.scope_cfg["partial_sensors_fraction_min"]),
                float(self.scope_cfg["partial_sensors_fraction_max"]),
            )
            # Decision 37: Explicit half-up rounding floor(frac * N + 0.5)
            k_target = int(np.floor(frac * n_sensors + 0.5))
            if n_sensors > 1:
                k_count = min(n_sensors - 1, max(1, k_target))
                selected_indices = rng.choice(n_sensors, size=k_count, replace=False)
                affected = [sensor_ids[i] for i in selected_indices]
            else:
                # When n_sensors == 1: k_target is 0 with 25% prob (frac < 0.5) and 1 with 75% prob (frac >= 0.5)
                if k_target >= 1:
                    affected = [sensor_ids[0]]
                else:
                    affected = []

            att_min = float(self.scope_cfg["partial_attenuation_min"])
            att_max = float(self.scope_cfg["partial_attenuation_max"])
            for s_id in affected:
                attenuations[s_id] = float(rng.uniform(att_min, att_max))

        else:  # "local"
            k_min = int(self.scope_cfg["local_sensors_min"])
            k_max = int(self.scope_cfg["local_sensors_max"])
            k_target = int(rng.integers(k_min, k_max + 1))
            k_count = min(n_sensors, max(1, k_target))
            selected_indices = rng.choice(n_sensors, size=k_count, replace=False)
            affected = [sensor_ids[i] for i in selected_indices]
            att_min = float(self.scope_cfg["local_attenuation_min"])
            att_max = float(self.scope_cfg["local_attenuation_max"])
            for s_id in affected:
                attenuations[s_id] = float(rng.uniform(att_min, att_max))

        return selected_scope, affected, attenuations

    def sample_sensor_delays(
        self,
        affected_sensors: Sequence[str],
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Sample physical arrival delays per affected sensor in minutes."""
        delays: Dict[str, float] = {}
        for s_id in affected_sensors:
            delays[s_id] = float(rng.uniform(self.arrival_delay_min, self.arrival_delay_max))
        return delays

    def sample_durations(
        self,
        event_type: str,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Sample duration parameters specific to event type."""
        d_cfg = self.durations_cfg
        if event_type == "step_exponential_recovery":
            tau = float(rng.uniform(
                float(d_cfg["step_recovery_tau_min"]),
                float(d_cfg["step_recovery_tau_max"]),
            ))
            return {"tau_minutes": tau}
        elif event_type == "ramp":
            rise = float(rng.uniform(
                float(d_cfg["ramp_rise_fall_min"]),
                float(d_cfg["ramp_rise_fall_max"]),
            ))
            hold = float(rng.uniform(
                float(d_cfg["ramp_hold_min"]),
                float(d_cfg["ramp_hold_max"]),
            ))
            fall = float(rng.uniform(
                float(d_cfg["ramp_rise_fall_min"]),
                float(d_cfg["ramp_rise_fall_max"]),
            ))
            return {"rise_minutes": rise, "hold_minutes": hold, "fall_minutes": fall}
        elif event_type == "pulse":
            dur = float(rng.uniform(
                float(d_cfg["pulse_min"]),
                float(d_cfg["pulse_max"]),
            ))
            return {"duration_minutes": dur}
        elif event_type == "sustained_shift":
            rise = float(rng.uniform(
                float(d_cfg["sustained_rise_min"]),
                float(d_cfg["sustained_rise_max"]),
            ))
            return {"rise_minutes": rise}
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def sample_start_step(
        self,
        rng: Optional[Union[int, np.random.Generator]] = None,
        total_steps: Optional[Union[int, np.random.Generator]] = None,
        start_min: Optional[int] = None,
        start_max: Optional[int] = None,
    ) -> int:
        """Sample event start step uniformly from configured inclusive range [start_step_min, start_step_max].
        
        Decision 18 (approved engineering choice):
          Every genuine event's base start step is sampled uniformly from the inclusive range
          [start_step_min, start_step_max] (default [480, 1919] within the scored window).
          The warm-up period (steps 0 to 479) remains strictly free of genuine environmental events.
          
        Args:
            rng: Numpy Generator (or total_steps integer if passed positionally first).
            total_steps: Optional total scenario steps (or Generator if passed second).
            start_min: Optional override for minimum start step.
            start_max: Optional override for maximum start step.
            
        Returns:
            Sampled integer start step.
        """
        # Support both (rng, total_steps) and (total_steps, rng)
        if isinstance(rng, (int, np.integer)):
            t_steps: Optional[int] = int(rng)
            rng_gen = total_steps if isinstance(total_steps, np.random.Generator) else None
        elif isinstance(rng, np.random.Generator):
            rng_gen = rng
            t_steps = int(total_steps) if isinstance(total_steps, (int, np.integer)) else None
        else:
            rng_gen = total_steps if isinstance(total_steps, np.random.Generator) else None
            t_steps = None

        if rng_gen is None:
            raise ValueError("A numpy Generator must be passed to sample_start_step")

        if t_steps is not None and t_steps < self.start_step_min:
            s_min = 0 if start_min is None else int(start_min)
            s_max = max(0, t_steps - 1) if start_max is None else int(start_max)
        else:
            s_min = self.start_step_min if start_min is None else int(start_min)
            default_max = (
                min(self.start_step_max, t_steps - 1)
                if t_steps is not None
                else self.start_step_max
            )
            s_max = default_max if start_max is None else int(start_max)

        if s_min > s_max:
            raise ValueError(f"start_step_min ({s_min}) > start_step_max ({s_max})")
        return int(rng_gen.integers(s_min, s_max + 1))

    def generate_events(
        self,
        sensor_params: Sequence[TemperatureSensorParams],
        total_steps: Optional[int] = None,
        sampling_rate_minutes: Optional[float] = None,
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
        group_id: str = "group_00",
        baseline_env_values: Optional[np.ndarray] = None,
        allowed_scopes: Optional[Sequence[str]] = None,
    ) -> EventSeries:
        """Generate all genuine events and ground-truth masks for a group.
        
        Args:
            sensor_params: Sequence of TemperatureSensorParams in the group.
            total_steps: Optional scenario duration in steps (defaults to config).
            sampling_rate_minutes: Optional minutes per step (defaults to config).
            rng: Optional integer seed or numpy Generator.
            seed: Optional integer seed.
            group_id: Unique group identifier string.
            baseline_env_values: Optional unperturbed environment values, shape (T,).
            allowed_scopes: Optional list of allowed scope names (e.g. ["global"] for EASY tier).
            
        Returns:
            EventSeries instance.
        """
        # Resolve RNG
        if rng is None and seed is None:
            gen = np.random.default_rng()
        elif isinstance(rng, np.random.Generator):
            gen = rng
        elif isinstance(rng, (int, np.integer)):
            gen = np.random.default_rng(int(rng))
        elif seed is not None:
            gen = np.random.default_rng(int(seed))
        else:
            gen = np.random.default_rng()

        t_steps = total_steps if total_steps is not None else self.total_steps
        dt = sampling_rate_minutes if sampling_rate_minutes is not None else self.sampling_rate_minutes
        t_minutes = np.arange(t_steps, dtype=np.float64) * dt

        sensor_ids = [s.sensor_id for s in sensor_params]
        sensor_sigma_map = {s.sensor_id: s.sigma_k for s in sensor_params}

        # 1. Sample event count
        count = self.sample_event_count(rng=gen)

        episodes: List[EventEpisode] = []
        sensor_contributions: Dict[str, np.ndarray] = {
            s_id: np.zeros(t_steps, dtype=np.float64) for s_id in sensor_ids
        }
        episode_waveforms: List[Dict[str, np.ndarray]] = []

        for e_idx in range(count):
            e_id = f"event_{e_idx:03d}"
            e_type = self.sample_event_type(rng=gen)
            signed_mag, abs_mag, sign_val = self.sample_magnitude(rng=gen)
            scope, affected, attenuations = self.sample_scope_and_sensors(
                sensor_ids=sensor_ids,
                rng=gen,
                allowed_scopes=allowed_scopes,
            )
            delays = self.sample_sensor_delays(affected_sensors=affected, rng=gen)
            durations = self.sample_durations(event_type=e_type, rng=gen)
            start_step = self.sample_start_step(rng=gen, total_steps=t_steps)
            start_time_min = float(start_step * dt)

            # Determine end step
            max_delay = max(delays.values()) if len(delays) > 0 else 0.0
            if e_type == "step_exponential_recovery":
                tau = durations["tau_minutes"]
                end_step = min(t_steps, int(np.ceil((start_time_min + max_delay + 5.0 * tau) / dt)))
            elif e_type == "ramp":
                total_duration = durations["rise_minutes"] + durations["hold_minutes"] + durations["fall_minutes"]
                end_step = min(t_steps, int(np.ceil((start_time_min + max_delay + total_duration) / dt)))
            elif e_type == "pulse":
                end_step = min(t_steps, int(np.ceil((start_time_min + max_delay + durations["duration_minutes"]) / dt)))
            else:  # "sustained_shift"
                end_step = t_steps

            # Check if this episode reaches >= 3 * sigma_k on at least one affected sensor
            is_detectable = any(
                (attenuations[s_id] * abs_mag) >= self.detectability_threshold_sigma * sensor_sigma_map[s_id]
                for s_id in affected
            )

            ep = EventEpisode(
                event_id=e_id,
                event_type=e_type,
                start_step=start_step,
                start_time_minutes=start_time_min,
                magnitude=signed_mag,
                abs_magnitude=abs_mag,
                sign=sign_val,
                scope=scope,
                affected_sensors=tuple(affected),
                sensor_attenuations=attenuations,
                sensor_delays=delays,
                shape_params=durations,
                is_detectable=is_detectable,
                end_step=end_step,
            )
            episodes.append(ep)

            # Generate waveform per sensor for this episode
            ep_sensor_wave: Dict[str, np.ndarray] = {}
            for s_id in sensor_ids:
                w = ep.generate_sensor_waveform(s_id, t_minutes)
                ep_sensor_wave[s_id] = w
                sensor_contributions[s_id] += w

            episode_waveforms.append(ep_sensor_wave)

        # 2. Compute ground-truth masks per step
        genuine_event = np.zeros(t_steps, dtype=int)
        event_below_detectability = np.zeros(t_steps, dtype=int)
        event_magnitude_sigma = np.zeros(t_steps, dtype=np.float64)
        true_event_breadth = np.zeros(t_steps, dtype=np.float64)
        active_event_ids: List[str] = ["" for _ in range(t_steps)]
        true_group_value = np.zeros(t_steps, dtype=np.float64)

        thresh_sigma = self.detectability_threshold_sigma
        activity_eps = self.activity_epsilon_celsius

        for t in range(t_steps):
            # Check which episodes are active at step t (|w_{e,s}(t)| > activity_eps on any affected sensor in group)
            active_eps = []
            for ep_idx, ep in enumerate(episodes):
                ep_waves = episode_waveforms[ep_idx]
                if any(abs(ep_waves[s_id][t]) > activity_eps for s_id in ep.affected_sensors if s_id in sensor_ids):
                    active_eps.append(ep)

            # Deterministic sorting by (start_step, event_id)
            active_eps_sorted = sorted(active_eps, key=lambda e: (e.start_step, e.event_id))
            if active_eps_sorted:
                active_event_ids[t] = ",".join(e.event_id for e in active_eps_sorted)

            # Check composite event contribution across sensors
            max_sigma_ratio = 0.0
            any_detectable = False

            for s_param in sensor_params:
                c = abs(sensor_contributions[s_param.sensor_id][t])
                sigma = s_param.sigma_k
                ratio = c / sigma
                if ratio > max_sigma_ratio:
                    max_sigma_ratio = ratio
                if c >= thresh_sigma * sigma:
                    any_detectable = True

            # Threshold-gated diagnostic and breadth
            if len(active_eps_sorted) > 0:
                event_magnitude_sigma[t] = max_sigma_ratio
                active_sensors_in_group = {
                    s_id for s_id in sensor_ids
                    if any(abs(episode_waveforms[ep_idx][s_id][t]) > activity_eps for ep_idx, ep in enumerate(episodes))
                }
                true_event_breadth[t] = len(active_sensors_in_group) / len(sensor_params) if len(sensor_params) > 0 else 0.0

                if any_detectable:
                    genuine_event[t] = 1
                    event_below_detectability[t] = 0
                else:
                    genuine_event[t] = 0
                    event_below_detectability[t] = 1
            else:
                event_magnitude_sigma[t] = 0.0
                true_event_breadth[t] = 0.0
                genuine_event[t] = 0
                event_below_detectability[t] = 0

            # True group value at step t
            if baseline_env_values is not None:
                base_val = baseline_env_values[t]
                phys_vals = [
                    np.clip(base_val + sensor_contributions[s.sensor_id][t], self.env_clip_min, self.env_clip_max)
                    for s in sensor_params
                ]
                true_group_value[t] = float(np.mean(phys_vals))
            else:
                true_group_value[t] = float(np.mean([
                    sensor_contributions[s.sensor_id][t] for s in sensor_params
                ]))

        return EventSeries(
            group_id=group_id,
            total_steps=t_steps,
            episodes=episodes,
            sensor_contributions=sensor_contributions,
            genuine_event=genuine_event,
            event_below_detectability=event_below_detectability,
            event_magnitude_sigma=event_magnitude_sigma,
            true_event_breadth=true_event_breadth,
            active_event_ids=active_event_ids,
            true_group_value=true_group_value,
        )


def apply_events_to_environment(
    environment: EnvironmentSeries,
    event_series: EventSeries,
    env_clip_min: float = -5.0,
    env_clip_max: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Calculate the event-adjusted physical signal for each sensor in the group.
    
    The physical signal hitting sensor k is the baseline environment plus that sensor's
    event contribution, clipped to the canonical environmental bounds [-5, 50] C:
      u_phys,k(t) = clip(u_env(t) + Delta T_k(t), env_clip_min, env_clip_max)
      
    Args:
        environment: Baseline EnvironmentSeries from P1.2.
        event_series: EventSeries containing per-sensor event contributions.
        env_clip_min: Minimum environmental clipping bound.
        env_clip_max: Maximum environmental clipping bound.
        
    Returns:
        Dict mapping sensor_id to physical temperature array, shape (T,).
    """
    physical_signals: Dict[str, np.ndarray] = {}
    base_env = environment.values
    for sensor_id, contrib in event_series.sensor_contributions.items():
        raw_phys = base_env + contrib
        physical_signals[sensor_id] = np.clip(raw_phys, env_clip_min, env_clip_max)
    return physical_signals
