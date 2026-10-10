"""Virtual sensor simulation for sensor stream intelligence.

Simulates observable measurements for virtual sensors from the true environmental
temperature series (Appendix A.1, Appendix A.2, Decisions 14-17):
  1. Response lag: first-order physical lag (tau in [0, 6] min, 40% zero)
  2. Static offset: uniform [-1.0, 1.0] C per sensor
  3. Measurement noise: Gaussian noise sigma_k in [0.05, 0.30] C, with 20% of sensors
     mildly correlated (rho in [0.2, 0.5]) via a 1-factor model
  4. Sensor clipping: bounded to operational range [-10.0, 60.0] C
  5. Quantisation: resolution chosen from {0.01, 0.05, 0.1, 0.25} C
  6. Healthy missing data: exactly 20% of sensors with 0% rate, remaining 80% sampled
     from Uniform[0.0, 0.01], bursts up to 3 consecutive readings represented as NaN
  7. Unit scaling: applied last, converting canonical Celsius to observable units
     while strictly isolating ground truth and canonical values.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.sim.environment import EnvironmentSeries, EnvironmentSimulator
from src.sim.profile import (
    TemperatureProfile,
    TemperatureScenarioParams,
    TemperatureSensorParams,
)
from src.sim.units import UnitScaler
from src.utils.config import get_config_dir, load_yaml


def apply_response_lag(
    values: np.ndarray,
    tau: float,
    sampling_rate_minutes: float = 1.0,
) -> np.ndarray:
    """Apply first-order physical response lag filter (Decision 17).
    
    Continuous physical ODE: tau * dy/dt + y = u
    Discretized with sampling time dt:
      If tau <= 0.0: y[t] = u[t]
      If tau > 0.0:
        beta = exp(-dt / tau)
        y[0] = u[0]  (thermal equilibrium initialization)
        y[t] = beta * y[t-1] + (1 - beta) * u[t] for t >= 1
        
    Args:
        values: Environmental temperature array, shape (T,).
        tau: Sensor time constant in minutes.
        sampling_rate_minutes: Elapsed minutes per time step (dt).
        
    Returns:
        Lagged temperature array, shape (T,).
    """
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    if tau <= 0.0:
        return values.copy().astype(np.float64)

    dt = float(sampling_rate_minutes)
    beta = float(np.exp(-dt / tau))
    one_minus_beta = 1.0 - beta

    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for t in range(1, len(values)):
        out[t] = beta * out[t - 1] + one_minus_beta * values[t]
    return out


def generate_group_noise(
    steps: int,
    sensor_params: Sequence[TemperatureSensorParams],
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Generate Gaussian measurement noise for a group of sensors (Decision 16).
    
    Implements a 1-factor model for sensors with is_noise_correlated=True:
      - Correlated sensors share a common group latent factor xi[t] ~ N(0, 1).
      - For each correlated sensor k:
          eta_k[t] = sigma_k * (sqrt(rho_k) * xi[t] + sqrt(1 - rho_k) * z_k[t])
          where z_k[t] ~ N(0, 1) is sensor-specific idiosyncratic noise.
          Marginal Var(eta_k) = sigma_k^2 * (rho_k + 1 - rho_k) = sigma_k^2.
          Pairwise correlation between correlated sensors j and k is sqrt(rho_j * rho_k).
      - Uncorrelated sensors receive independent Gaussian noise:
          eta_k[t] = sigma_k * z_k[t].
          
    Args:
        steps: Duration in time steps.
        sensor_params: Sequence of TemperatureSensorParams.
        rng: Numpy Generator.
        
    Returns:
        Dictionary mapping sensor_id to noise array of shape (steps,).
    """
    if steps <= 0:
        return {s.sensor_id: np.empty(0, dtype=np.float64) for s in sensor_params}

    # Identify correlated sensors
    correlated_indices = [
        i for i, s in enumerate(sensor_params) if s.is_noise_correlated and s.noise_correlation_rho > 0.0
    ]

    # Sample group common noise factor if >= 2 correlated sensors exist
    if len(correlated_indices) >= 2:
        group_factor = rng.standard_normal(steps)
    else:
        group_factor = None

    noise_dict: Dict[str, np.ndarray] = {}
    for i, s in enumerate(sensor_params):
        sigma = s.sigma_k
        z_k = rng.standard_normal(steps)
        if s.is_noise_correlated and group_factor is not None:
            rho = s.noise_correlation_rho
            sqrt_rho = np.sqrt(rho)
            sqrt_one_minus_rho = np.sqrt(1.0 - rho)
            noise_dict[s.sensor_id] = sigma * (sqrt_rho * group_factor + sqrt_one_minus_rho * z_k)
        else:
            noise_dict[s.sensor_id] = sigma * z_k

    return noise_dict


def apply_clipping(
    values: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    """Clip temperature readings to operational sensor range [-10, 60] C."""
    return np.clip(values, clip_min, clip_max)


def apply_quantisation(
    values: np.ndarray,
    quantisation_step: float,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> np.ndarray:
    """Quantise continuous temperature values to discrete resolution.
    
    Args:
        values: Continuous temperature array.
        quantisation_step: Quantisation resolution step (e.g. 0.01, 0.05, 0.1, 0.25).
        clip_min: Optional lower bound clipping.
        clip_max: Optional upper bound clipping.
        
    Returns:
        Quantised temperature array.
    """
    if quantisation_step <= 0.0:
        raise ValueError(f"quantisation_step must be positive, got {quantisation_step}")
    quantized = np.round(values / quantisation_step) * quantisation_step
    # Clean floating precision representation
    quantized = np.round(quantized, decimals=6)
    if clip_min is not None and clip_max is not None:
        quantized = np.clip(quantized, clip_min, clip_max)
    return quantized


def generate_missing_mask(
    steps: int,
    missing_rate: float,
    burst_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate healthy missing data mask (Decision 14, Decision 15).
    
    Target missing count M = round(missing_rate * steps) is partitioned into
    burst lengths L drawn uniformly from {1, ..., burst_max}. Bursts are placed
    at non-adjacent starting positions (separated by at least 1 valid reading),
    guaranteeing that bursts never merge and consecutive missing readings strictly
    never exceed burst_max.
    
    Args:
        steps: Total duration in steps.
        missing_rate: Missing data rate in [0, 1]. Exactly 0.0 for 20% of sensors.
        burst_max: Maximum consecutive missing readings (configured as 3).
        rng: Numpy Generator.
        
    Returns:
        Boolean numpy array of shape (steps,) where True represents missing.
    """
    if steps <= 0 or missing_rate <= 0.0:
        return np.zeros(steps, dtype=bool)

    target_missing = int(np.round(missing_rate * steps))
    if target_missing <= 0:
        return np.zeros(steps, dtype=bool)
    if target_missing >= steps:
        return np.ones(steps, dtype=bool)

    burst_max = max(1, burst_max)

    # Partition target_missing into burst lengths L in [1, burst_max]
    remaining = target_missing
    burst_lengths: List[int] = []
    while remaining > 0:
        b_limit = min(remaining, burst_max)
        l = int(rng.integers(1, b_limit + 1))
        burst_lengths.append(l)
        remaining -= l

    mask = np.zeros(steps, dtype=bool)

    # Place bursts at non-adjacent starting indices to ensure bursts never merge
    for l in burst_lengths:
        valid_starts = []
        for s in range(steps - l + 1):
            left = max(0, s - 1)
            right = min(steps, s + l + 1)
            if not np.any(mask[left:right]):
                valid_starts.append(s)

        if valid_starts:
            idx = int(rng.choice(valid_starts))
            mask[idx : idx + l] = True
        else:
            # Fallback if non-adjacent slots exhausted: find positions where
            # placing length l does not create any run > burst_max
            fallback_starts = []
            for s in range(steps - l + 1):
                if not np.any(mask[s : s + l]):
                    test_mask = mask.copy()
                    test_mask[s : s + l] = True
                    diffs = np.diff(np.concatenate(([0], test_mask.view(np.int8), [0])))
                    starts = np.where(diffs == 1)[0]
                    ends = np.where(diffs == -1)[0]
                    if np.all(ends - starts <= burst_max):
                        fallback_starts.append(s)
            if fallback_starts:
                idx = int(rng.choice(fallback_starts))
                mask[idx : idx + l] = True
            else:
                # Place single isolated points where free
                free_indices = [i for i in range(steps) if not mask[i]]
                if free_indices:
                    idx = int(rng.choice(free_indices))
                    mask[idx] = True

    return mask


@dataclass(frozen=True)
class SensorStream:
    """Observable readings and canonical truth breakdown for a virtual sensor.
    
    Attributes:
        sensor_id: Unique sensor identifier string.
        values: Observable sensor readings after all transformations and unit scaling, shape (T,).
                Contains NaN for missing readings.
        canonical_values: Sensor physical values in canonical Celsius before unit scaling, shape (T,).
                          Contains NaN for missing readings.
        canonical_clean_values: Sensor physical values in canonical Celsius without missing data, shape (T,).
        environmental_values: True environmental series in canonical Celsius (unperturbed), shape (T,).
        missing_mask: Boolean array indicating missing readings (True = missing/NaN), shape (T,).
        params: TemperatureSensorParams used for this sensor.
        unit_scaler: UnitScaler applied to convert canonical to observable values.
        timestamps_minutes: Array of elapsed minutes, shape (T,).
        total_steps: Total duration of the simulation run in steps.
        warmup_steps: Number of initial warm-up steps.
        scored_steps: Number of evaluation/scored steps.
        physical_values: Event-adjusted physical temperature before sensor transduction, shape (T,).
        event_contributions: Genuine environmental event contribution Delta T_k(t), shape (T,).
    """
    sensor_id: str
    values: np.ndarray
    canonical_values: np.ndarray
    canonical_clean_values: np.ndarray
    environmental_values: np.ndarray
    missing_mask: np.ndarray
    params: TemperatureSensorParams
    unit_scaler: UnitScaler
    timestamps_minutes: np.ndarray
    total_steps: int
    warmup_steps: int
    scored_steps: int
    physical_values: Optional[np.ndarray] = None
    event_contributions: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.physical_values is None:
            object.__setattr__(self, "physical_values", self.environmental_values.copy())
        if self.event_contributions is None:
            object.__setattr__(self, "event_contributions", np.zeros_like(self.environmental_values))

    @property
    def warmup_values(self) -> np.ndarray:
        """Slice of observable readings during warm-up phase (0 to warmup_steps)."""
        return self.values[: self.warmup_steps]

    @property
    def scored_values(self) -> np.ndarray:
        """Slice of observable readings during scored phase (warmup_steps to total_steps)."""
        return self.values[self.warmup_steps :]

    @property
    def valid_count(self) -> int:
        """Count of valid (non-missing) readings."""
        return int(np.sum(~self.missing_mask))

    @property
    def missing_count(self) -> int:
        """Count of missing readings."""
        return int(np.sum(self.missing_mask))

    @property
    def actual_missing_rate(self) -> float:
        """Observed missing rate across all steps."""
        if self.total_steps == 0:
            return 0.0
        return self.missing_count / self.total_steps

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sensor stream metadata and summary statistics."""
        valid_vals = self.canonical_values[~self.missing_mask]
        return {
            "sensor_id": self.sensor_id,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "scored_steps": self.scored_steps,
            "valid_count": self.valid_count,
            "missing_count": self.missing_count,
            "actual_missing_rate": float(self.actual_missing_rate),
            "canonical_min": float(np.min(valid_vals)) if len(valid_vals) > 0 else float("nan"),
            "canonical_max": float(np.max(valid_vals)) if len(valid_vals) > 0 else float("nan"),
            "canonical_mean": float(np.mean(valid_vals)) if len(valid_vals) > 0 else float("nan"),
            "event_contribution_min": float(np.min(self.event_contributions)),
            "event_contribution_max": float(np.max(self.event_contributions)),
            "physical_min": float(np.min(self.physical_values)),
            "physical_max": float(np.max(self.physical_values)),
            "params": self.params.to_dict(),
            "unit_scaler": self.unit_scaler.to_dict(),
        }


@dataclass(frozen=True)
class VirtualSensorGroup:
    """Collection of sensor streams for a group of co-located sensors."""
    group_id: str
    sensors: Dict[str, SensorStream]
    environment: EnvironmentSeries
    scenario_params: TemperatureScenarioParams
    events: Optional[Any] = None

    @property
    def sensor_ids(self) -> List[str]:
        """List of sensor IDs in the group."""
        return list(self.sensors.keys())

    @property
    def num_sensors(self) -> int:
        """Number of sensors in the group."""
        return len(self.sensors)

    @property
    def total_steps(self) -> int:
        """Total duration of the simulation run in steps."""
        return self.environment.total_steps

    @property
    def values_matrix(self) -> np.ndarray:
        """2D array of observable values, shape (num_sensors, total_steps)."""
        return np.array([s.values for s in self.sensors.values()])

    @property
    def canonical_matrix(self) -> np.ndarray:
        """2D array of canonical Celsius values, shape (num_sensors, total_steps)."""
        return np.array([s.canonical_values for s in self.sensors.values()])

    def to_observable_dataframe(self, scenario_id: str = "scenario_0000") -> pd.DataFrame:
        """Format observable data as a Layer 1 raw table (observable only).
        
        Schema: (scenario_id, group_id, sensor_id, variable_type, timestamp, value)
        """
        rows = []
        var_type = self.scenario_params.variable_type
        for sensor_id, stream in self.sensors.items():
            for ts, val in zip(stream.timestamps_minutes, stream.values):
                rows.append({
                    "scenario_id": scenario_id,
                    "group_id": self.group_id,
                    "sensor_id": sensor_id,
                    "variable_type": var_type,
                    "timestamp": float(ts),
                    "value": float(val) if not np.isnan(val) else np.nan,
                })
        return pd.DataFrame(rows)

    def to_canonical_dataframe(self, scenario_id: str = "scenario_0000") -> pd.DataFrame:
        """Format canonical values in Celsius with missing readings as NaN."""
        rows = []
        var_type = self.scenario_params.variable_type
        for sensor_id, stream in self.sensors.items():
            for ts, val in zip(stream.timestamps_minutes, stream.canonical_values):
                rows.append({
                    "scenario_id": scenario_id,
                    "group_id": self.group_id,
                    "sensor_id": sensor_id,
                    "variable_type": var_type,
                    "timestamp": float(ts),
                    "value": float(val) if not np.isnan(val) else np.nan,
                })
        return pd.DataFrame(rows)

    def to_ground_truth_events_dataframe(
        self,
        scenario_id: str = "scenario_0000",
    ) -> pd.DataFrame:
        """Export group-level ground truth events table (Appendix A.7)."""
        dt = float(self.environment.timestamps_minutes[1] - self.environment.timestamps_minutes[0]) if len(self.environment.timestamps_minutes) > 1 else 1.0
        if self.events is not None:
            return self.events.to_ground_truth_dataframe(
                scenario_id=scenario_id,
                sampling_rate_minutes=dt,
            )
        timestamps = self.environment.timestamps_minutes
        return pd.DataFrame({
            "scenario_id": scenario_id,
            "group_id": self.group_id,
            "timestamp": timestamps,
            "genuine_event": np.zeros(self.total_steps, dtype=int),
            "event_below_detectability": np.zeros(self.total_steps, dtype=int),
            "event_id": ["" for _ in range(self.total_steps)],
            "event_magnitude_sigma": np.zeros(self.total_steps, dtype=float),
            "true_event_breadth": np.zeros(self.total_steps, dtype=float),
            "true_group_value": self.environment.values.copy(),
        })

    def episodes_to_dataframe(
        self,
        scenario_id: str = "scenario_0000",
    ) -> pd.DataFrame:
        """Export episode ground-truth table as specified in Appendix A.7."""
        if self.events is not None:
            return self.events.episodes_to_dataframe(scenario_id=scenario_id)
        return pd.DataFrame(columns=[
            "scenario_id", "group_id", "event_id", "event_type", "start_step",
            "start_time_minutes", "magnitude", "abs_magnitude", "sign", "scope",
            "affected_sensors", "sensor_attenuations", "sensor_delays", "shape_params",
            "is_detectable", "end_step",
        ])


class VirtualSensorSimulator:
    """Simulator for generating virtual sensor readings from true environment series."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        profile: Optional[TemperatureProfile] = None,
    ):
        """Initialize VirtualSensorSimulator.
        
        Args:
            config: Optional full configuration dictionary.
            profile: Optional TemperatureProfile instance.
        """
        if config is None:
            default_yaml = get_config_dir() / "generator_a.yaml"
            self.config = load_yaml(default_yaml)
        else:
            self.config = config

        self.profile = profile if profile is not None else TemperatureProfile(self.config)
        self._extract_config()

    def _extract_config(self) -> None:
        """Extract scenario structure and sensor bounds from configuration."""
        struct = self.config["structure"]
        self.sampling_rate_minutes = float(struct["sampling_rate_minutes"])
        self.total_steps = int(struct["total_steps"])
        self.warmup_steps = int(struct["warmup_steps"])
        self.scored_steps = int(struct["scored_steps"])

        bounds = self.config["temperature_profile"]["bounds"]
        self.sensor_clip_min = float(bounds["sensor_clip_min"])
        self.sensor_clip_max = float(bounds["sensor_clip_max"])
        self.env_clip_min = float(bounds["environment_clip_min"])
        self.env_clip_max = float(bounds["environment_clip_max"])

    def simulate_sensor(
        self,
        env_values: np.ndarray,
        params: TemperatureSensorParams,
        unit_scaler: UnitScaler,
        rng: np.random.Generator,
        noise: Optional[np.ndarray] = None,
        timestamps_minutes: Optional[np.ndarray] = None,
        warmup_steps: Optional[int] = None,
        scored_steps: Optional[int] = None,
        physical_values: Optional[np.ndarray] = None,
        event_contributions: Optional[np.ndarray] = None,
        baseline_env_values: Optional[np.ndarray] = None,
    ) -> SensorStream:
        """Simulate an individual sensor from physical values.
        
        Order of physical transformations:
          1. Response lag (tau)
          2. Static offset
          3. Noise (sigma_k, optionally correlated)
          4. Sensor clipping [-10, 60] C
          5. Quantisation
          6. Healthy missing data (NaN)
          7. Unit scaling (applied last)
          
        Args:
            env_values: Physical temperature array before sensor transduction, shape (T,).
            params: TemperatureSensorParams for this sensor.
            unit_scaler: UnitScaler for observable scaling.
            rng: Numpy Generator.
            noise: Optional pre-generated noise array (e.g. from group generator).
            timestamps_minutes: Optional array of timestamps in minutes.
            warmup_steps: Optional warm-up steps count.
            scored_steps: Optional scored steps count.
            physical_values: Optional physical temperature array (defaults to env_values).
            event_contributions: Optional genuine event contribution Delta T_k(t).
            baseline_env_values: Optional unperturbed baseline environment.
            
        Returns:
            SensorStream instance.
        """
        steps = len(env_values)
        if timestamps_minutes is None:
            timestamps = np.arange(steps, dtype=np.float64) * self.sampling_rate_minutes
        else:
            timestamps = timestamps_minutes

        w_steps = warmup_steps if warmup_steps is not None else self.warmup_steps
        s_steps = scored_steps if scored_steps is not None else self.scored_steps

        # 1. Response lag
        lagged = apply_response_lag(
            env_values,
            tau=params.response_lag_tau,
            sampling_rate_minutes=self.sampling_rate_minutes,
        )

        # 2. Static offset
        with_offset = lagged + params.static_offset

        # 3. Measurement noise
        if noise is not None:
            noisy = with_offset + noise
        else:
            sensor_noise = rng.standard_normal(steps) * params.sigma_k
            noisy = with_offset + sensor_noise

        # 4. Sensor clipping
        clipped = apply_clipping(
            noisy,
            clip_min=self.sensor_clip_min,
            clip_max=self.sensor_clip_max,
        )

        # 5. Quantisation
        quantized = apply_quantisation(
            clipped,
            quantisation_step=params.quantisation_step,
            clip_min=self.sensor_clip_min,
            clip_max=self.sensor_clip_max,
        )

        # Clean canonical values before missing data
        canonical_clean = quantized.copy()

        # 6. Healthy missing data (Decision 14, Decision 15)
        missing_mask = generate_missing_mask(
            steps=steps,
            missing_rate=params.missing_rate,
            burst_max=params.burst_max,
            rng=rng,
        )

        canonical_values = canonical_clean.copy()
        canonical_values[missing_mask] = np.nan

        # 7. Unit scaling applied last (P1.1 principle)
        observable_values = unit_scaler.apply(canonical_values)

        phys_vals = physical_values if physical_values is not None else env_values.copy()
        event_contrib = event_contributions if event_contributions is not None else np.zeros_like(env_values)
        base_env = baseline_env_values if baseline_env_values is not None else env_values.copy()

        return SensorStream(
            sensor_id=params.sensor_id,
            values=observable_values,
            canonical_values=canonical_values,
            canonical_clean_values=canonical_clean,
            environmental_values=base_env,
            missing_mask=missing_mask,
            params=params,
            unit_scaler=unit_scaler,
            timestamps_minutes=timestamps,
            total_steps=steps,
            warmup_steps=w_steps,
            scored_steps=s_steps,
            physical_values=phys_vals,
            event_contributions=event_contrib,
        )

    def simulate_group(
        self,
        environment: EnvironmentSeries,
        sensor_params: Optional[Sequence[TemperatureSensorParams]] = None,
        num_sensors: Optional[int] = None,
        group_id: str = "group_00",
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
        events: Optional[Any] = None,
        generate_events: bool = False,
    ) -> VirtualSensorGroup:
        """Simulate all sensors for a co-located group.
        
        Args:
            environment: True EnvironmentSeries from P1.2.
            sensor_params: Optional pre-sampled sensor parameters.
            num_sensors: Optional number of sensors to sample if sensor_params is None.
            group_id: Group identifier string.
            rng: Optional integer seed or numpy Generator.
            seed: Optional explicit integer seed.
            events: Optional pre-generated EventSeries instance.
            generate_events: If True and events is None, generates genuine events.
            
        Returns:
            VirtualSensorGroup instance.
        """
        gen = TemperatureProfile._resolve_rng(rng=rng, seed=seed)

        if sensor_params is None:
            n = num_sensors if num_sensors is not None else int(self.config["structure"]["sensors_per_group_min"])
            params_list = self.profile.sample_sensor_params(num_sensors=n, rng=gen)
        else:
            params_list = list(sensor_params)

        # Handle events
        active_events = events
        if active_events is None and generate_events:
            from src.sim.events import EventGenerator
            event_gen = EventGenerator(config=self.config)
            active_events = event_gen.generate_events(
                sensor_params=params_list,
                total_steps=environment.total_steps,
                sampling_rate_minutes=self.sampling_rate_minutes,
                rng=gen,
                group_id=group_id,
                baseline_env_values=environment.values,
            )

        # Generate group noise (handles 1-factor mild correlation)
        noise_map = generate_group_noise(
            steps=environment.total_steps,
            sensor_params=params_list,
            rng=gen,
        )

        streams: Dict[str, SensorStream] = {}
        for s_param in params_list:
            s_noise = noise_map[s_param.sensor_id]
            if active_events is not None:
                contrib = active_events.sensor_contributions.get(
                    s_param.sensor_id, np.zeros(environment.total_steps, dtype=np.float64)
                )
                raw_phys = environment.values + contrib
                phys_vals = np.clip(raw_phys, self.env_clip_min, self.env_clip_max)
            else:
                contrib = np.zeros(environment.total_steps, dtype=np.float64)
                phys_vals = environment.values.copy()

            stream = self.simulate_sensor(
                env_values=phys_vals,
                params=s_param,
                unit_scaler=environment.params.unit_scaler,
                rng=gen,
                noise=s_noise,
                timestamps_minutes=environment.timestamps_minutes,
                warmup_steps=environment.warmup_steps,
                scored_steps=environment.scored_steps,
                physical_values=phys_vals,
                event_contributions=contrib,
                baseline_env_values=environment.values,
            )
            streams[s_param.sensor_id] = stream

        return VirtualSensorGroup(
            group_id=group_id,
            sensors=streams,
            environment=environment,
            scenario_params=environment.params,
            events=active_events,
        )


def simulate_sensors(
    environment: EnvironmentSeries,
    sensor_params: Optional[Sequence[TemperatureSensorParams]] = None,
    num_sensors: Optional[int] = None,
    group_id: str = "group_00",
    seed: Optional[int] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
    config: Optional[Dict[str, Any]] = None,
    events: Optional[Any] = None,
    generate_events: bool = False,
) -> VirtualSensorGroup:
    """Convenience top-level function to simulate virtual sensors.
    
    Args:
        environment: EnvironmentSeries instance from P1.2.
        sensor_params: Optional pre-sampled sensor parameters.
        num_sensors: Optional sensor count if sampling new parameters.
        group_id: Group identifier string.
        seed: Optional integer seed for reproducibility.
        rng: Optional seed or numpy Generator.
        config: Optional configuration dictionary.
        events: Optional pre-generated EventSeries instance.
        generate_events: If True and events is None, generates genuine events.
        
    Returns:
        VirtualSensorGroup instance.
    """
    simulator = VirtualSensorSimulator(config=config)
    return simulator.simulate_group(
        environment=environment,
        sensor_params=sensor_params,
        num_sensors=num_sensors,
        group_id=group_id,
        rng=rng,
        seed=seed,
        events=events,
        generate_events=generate_events,
    )
