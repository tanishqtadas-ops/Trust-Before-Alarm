"""Variable profile definitions for synthetic stream generation.

Encapsulates physical variable configurations (Appendix A.2) and parameter
sampling for synthetic scenarios and virtual sensors. Unit scaling is applied last,
maintaining strict separation between canonical physical values (Celsius) and
observable readings.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from src.sim.units import UnitScaler, sample_unit_scaler
from src.utils.config import get_config_dir, load_yaml


@dataclass(frozen=True)
class TemperatureScenarioParams:
    """Scenario-level parameters sampled for a single simulation run.
    
    All physical parameters are specified in canonical Celsius before unit scaling.
    """
    variable_type: str
    base_level: float
    daily_cycle_amplitude: float
    daily_cycle_has_cycle: bool
    daily_cycle_phase: float
    daily_cycle_period_minutes: float
    slow_wander_std: float
    slow_wander_correlation_time: float
    slow_wander_ma_steps: int
    environment_clip_min: float
    environment_clip_max: float
    sensor_clip_min: float
    sensor_clip_max: float
    unit_scaler: UnitScaler

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scenario parameters to a dictionary."""
        d = asdict(self)
        d["unit_scaler"] = self.unit_scaler.to_dict()
        return d


@dataclass(frozen=True)
class TemperatureSensorParams:
    """Sensor-level parameters sampled for an individual sensor within a scenario.
    
    All physical parameters are specified in canonical Celsius before unit scaling.
    """
    sensor_id: str
    sigma_k: float
    static_offset: float
    response_lag_tau: float
    quantisation_step: float
    is_noise_correlated: bool
    noise_correlation_rho: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sensor parameters to a dictionary."""
        return asdict(self)


class VariableProfile(ABC):
    """Abstract base class for physical variable profiles."""

    def __init__(self, variable_type: str, config: Dict[str, Any]):
        self.variable_type = variable_type
        self.config = config

    @abstractmethod
    def sample_scenario_params(
        self,
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Sample scenario-level physical and scaling parameters."""
        pass

    @abstractmethod
    def sample_sensor_params(
        self,
        num_sensors: int,
        rng: Optional[Union[int, np.random.Generator]] = None,
        sensor_ids: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """Sample sensor-level parameters for a group of sensors."""
        pass

    @abstractmethod
    def apply_unit_scaling(
        self, values: Union[float, int, np.ndarray, Sequence[float]], scaler: UnitScaler
    ) -> Union[float, np.ndarray]:
        """Apply unit scaling last to convert canonical values to observable units."""
        pass


class TemperatureProfile(VariableProfile):
    """Temperature variable profile driven strictly by configuration (Appendix A.2).
    
    Holds parameters for:
      - Base temperature level (uniform 2 to 38 C)
      - Daily diurnal cycle (amplitude uniform 0.5 to 4 C, 20% zero, 24h period, random phase)
      - Slow wander (std 0.2 to 0.8 C, correlation time 120 to 360 min, 5-step MA)
      - Sensor noise sigma_k (log-uniform 0.05 to 0.30 C, 20% mildly correlated rho 0.2 to 0.5)
      - Static healthy offset (uniform +/-1.0 C)
      - Response lag (first-order tau 0 to 6 min, 40% zero)
      - Quantisation (0.01: 25%, 0.05: 25%, 0.1: 35%, 0.25: 15%)
      - Bounds (environment clipped to [-5, 50] C, sensor clipped to [-10, 60] C)
      - Unit scaling applied last (70% C, 15% F, 15% random affine)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TemperatureProfile from configuration dictionary or default YAML.
        
        Args:
            config: Optional dictionary containing temperature_profile config or full config.
        """
        if config is None:
            default_yaml = get_config_dir() / "generator_a.yaml"
            raw_cfg = load_yaml(default_yaml)
            cfg = raw_cfg["temperature_profile"]
        elif "temperature_profile" in config:
            cfg = config["temperature_profile"]
        else:
            cfg = config

        super().__init__(variable_type="temperature", config=cfg)
        self._validate_and_extract_config()

    def _validate_and_extract_config(self) -> None:
        """Validate and bind configuration parameters."""
        cfg = self.config

        # Base level
        self.base_level_min = float(cfg["base_level_min"])
        self.base_level_max = float(cfg["base_level_max"])
        if self.base_level_min > self.base_level_max:
            raise ValueError(f"base_level_min ({self.base_level_min}) > base_level_max ({self.base_level_max})")

        # Daily cycle
        dc = cfg["daily_cycle"]
        self.daily_cycle_amplitude_min = float(dc["amplitude_min"])
        self.daily_cycle_amplitude_max = float(dc["amplitude_max"])
        self.daily_cycle_zero_amplitude_prob = float(dc["zero_amplitude_prob"])

        # Required period explicit from configuration (no hardcoded fallback)
        if "period_minutes" in dc:
            self.daily_cycle_period_minutes = float(dc["period_minutes"])
        elif "period_hours" in dc:
            self.daily_cycle_period_minutes = float(dc["period_hours"]) * 60.0
        else:
            raise KeyError("daily_cycle configuration must specify 'period_minutes' or 'period_hours'")

        # Configured phase bounds representing standard full-cycle [0, 2pi) rad (Decision 12)
        if "phase_min_rad" in dc:
            self.daily_cycle_phase_min = float(dc["phase_min_rad"])
        elif "phase_min" in dc:
            self.daily_cycle_phase_min = float(dc["phase_min"])
        else:
            raise KeyError("daily_cycle configuration must specify 'phase_min_rad' or 'phase_min'")

        if "phase_max_rad" in dc:
            self.daily_cycle_phase_max = float(dc["phase_max_rad"])
        elif "phase_max" in dc:
            self.daily_cycle_phase_max = float(dc["phase_max"])
        else:
            raise KeyError("daily_cycle configuration must specify 'phase_max_rad' or 'phase_max'")

        if self.daily_cycle_phase_min > self.daily_cycle_phase_max:
            raise ValueError(
                f"daily_cycle phase_min ({self.daily_cycle_phase_min}) > phase_max ({self.daily_cycle_phase_max})"
            )

        # Slow wander
        sw = cfg["slow_wander"]
        self.slow_wander_std_min = float(sw["std_min"])
        self.slow_wander_std_max = float(sw["std_max"])
        self.slow_wander_correlation_time_min = float(sw["correlation_time_min"])
        self.slow_wander_correlation_time_max = float(sw["correlation_time_max"])
        self.slow_wander_moving_average_steps = int(sw["moving_average_steps"])

        # Sensor noise
        sn = cfg["sensor_noise"]
        self.sigma_k_min = float(sn["sigma_k_min"])
        self.sigma_k_max = float(sn["sigma_k_max"])
        self.sensor_noise_correlated_fraction = float(sn["correlated_sensors_fraction"])
        self.correlation_rho_min = float(sn["correlation_rho_min"])
        self.correlation_rho_max = float(sn["correlation_rho_max"])

        # Static offset
        self.static_offset_min = float(cfg["static_offset_min"])
        self.static_offset_max = float(cfg["static_offset_max"])

        # Response lag
        rl = cfg["response_lag"]
        self.response_lag_time_constant_min = float(rl["time_constant_min"])
        self.response_lag_time_constant_max = float(rl["time_constant_max"])
        self.response_lag_zero_fraction = float(rl["zero_lag_fraction"])

        # Quantisation
        q = cfg["quantisation"]
        self.quantisation_values = [float(v) for v in q["values"]]
        self.quantisation_probabilities = [float(p) for p in q["probabilities"]]
        total_q_prob = sum(self.quantisation_probabilities)
        if not np.isclose(total_q_prob, 1.0, atol=1e-5):
            raise ValueError(f"Quantisation probabilities must sum to 1.0, got {total_q_prob}")

        # Bounds
        b = cfg["bounds"]
        self.environment_clip_min = float(b["environment_clip_min"])
        self.environment_clip_max = float(b["environment_clip_max"])
        self.sensor_clip_min = float(b["sensor_clip_min"])
        self.sensor_clip_max = float(b["sensor_clip_max"])

        # Unit scaling
        self.unit_scaling_config = dict(cfg["unit_scaling"])

    @staticmethod
    def _resolve_rng(
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
    ) -> np.random.Generator:
        """Resolve optional integer seed or Generator into a numpy Generator."""
        if seed is not None:
            return np.random.default_rng(seed)
        if rng is None:
            return np.random.default_rng()
        if isinstance(rng, (int, np.integer)):
            return np.random.default_rng(int(rng))
        if isinstance(rng, np.random.Generator):
            return rng
        raise TypeError(f"Expected int or np.random.Generator, got {type(rng).__name__}")

    def sample_scenario_params(
        self,
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
    ) -> TemperatureScenarioParams:
        """Deterministically sample scenario-level parameters for temperature.
        
        Args:
            rng: Optional integer seed or numpy Generator.
            seed: Optional explicit integer seed.
            
        Returns:
            TemperatureScenarioParams instance.
        """
        gen = self._resolve_rng(rng=rng, seed=seed)

        # Base level: Uniform[base_level_min, base_level_max]
        base_level = float(gen.uniform(self.base_level_min, self.base_level_max))

        # Daily cycle: zero amplitude prob or Uniform[amplitude_min, amplitude_max]
        has_cycle = float(gen.random()) >= self.daily_cycle_zero_amplitude_prob
        if has_cycle:
            amplitude = float(gen.uniform(self.daily_cycle_amplitude_min, self.daily_cycle_amplitude_max))
        else:
            amplitude = 0.0

        # Daily cycle phase sampled from configured phase interval (Decision 12)
        phase = float(gen.uniform(self.daily_cycle_phase_min, self.daily_cycle_phase_max))

        # Slow wander
        wander_std = float(gen.uniform(self.slow_wander_std_min, self.slow_wander_std_max))
        wander_tau = float(
            gen.uniform(self.slow_wander_correlation_time_min, self.slow_wander_correlation_time_max)
        )

        # Unit scaler (sampled according to unit_scaling probabilities)
        unit_scaler = sample_unit_scaler(self.unit_scaling_config, rng=gen)

        return TemperatureScenarioParams(
            variable_type=self.variable_type,
            base_level=base_level,
            daily_cycle_amplitude=amplitude,
            daily_cycle_has_cycle=has_cycle,
            daily_cycle_phase=phase,
            daily_cycle_period_minutes=self.daily_cycle_period_minutes,
            slow_wander_std=wander_std,
            slow_wander_correlation_time=wander_tau,
            slow_wander_ma_steps=self.slow_wander_moving_average_steps,
            environment_clip_min=self.environment_clip_min,
            environment_clip_max=self.environment_clip_max,
            sensor_clip_min=self.sensor_clip_min,
            sensor_clip_max=self.sensor_clip_max,
            unit_scaler=unit_scaler,
        )

    def sample_sensor_params(
        self,
        num_sensors: int,
        rng: Optional[Union[int, np.random.Generator]] = None,
        sensor_ids: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
    ) -> List[TemperatureSensorParams]:
        """Deterministically sample sensor-level parameters for a group of sensors.
        
        Args:
            num_sensors: Number of sensors in the group.
            rng: Optional integer seed or numpy Generator.
            sensor_ids: Optional custom sensor ID strings.
            seed: Optional explicit integer seed.
            
        Returns:
            List of TemperatureSensorParams instances.
        """
        if num_sensors < 1:
            raise ValueError(f"num_sensors must be >= 1, got {num_sensors}")
        gen = self._resolve_rng(rng=rng, seed=seed)

        if sensor_ids is not None:
            if len(sensor_ids) != num_sensors:
                raise ValueError(
                    f"Length of sensor_ids ({len(sensor_ids)}) does not match num_sensors ({num_sensors})"
                )
            ids = list(sensor_ids)
        else:
            ids = [f"sensor_{i:02d}" for i in range(num_sensors)]

        sensors: List[TemperatureSensorParams] = []
        for s_id in ids:
            # Sensor noise sigma_k: log-uniform in [sigma_k_min, sigma_k_max]
            log_min = np.log(self.sigma_k_min)
            log_max = np.log(self.sigma_k_max)
            sigma_k = float(np.exp(gen.uniform(log_min, log_max)))

            # Static offset: uniform in [static_offset_min, static_offset_max]
            static_offset = float(gen.uniform(self.static_offset_min, self.static_offset_max))

            # Response lag: 40% zero, else uniform in [time_constant_min, time_constant_max]
            has_lag = float(gen.random()) >= self.response_lag_zero_fraction
            if has_lag:
                tau = float(gen.uniform(self.response_lag_time_constant_min, self.response_lag_time_constant_max))
            else:
                tau = 0.0

            # Quantisation: categorical choice from values with probabilities
            q_idx = gen.choice(len(self.quantisation_values), p=self.quantisation_probabilities)
            quantisation_step = float(self.quantisation_values[q_idx])

            # Mild noise correlation: 20% of sensors correlated with rho in [0.2, 0.5]
            is_correlated = float(gen.random()) < self.sensor_noise_correlated_fraction
            if is_correlated:
                rho = float(gen.uniform(self.correlation_rho_min, self.correlation_rho_max))
            else:
                rho = 0.0

            sensors.append(
                TemperatureSensorParams(
                    sensor_id=s_id,
                    sigma_k=sigma_k,
                    static_offset=static_offset,
                    response_lag_tau=tau,
                    quantisation_step=quantisation_step,
                    is_noise_correlated=is_correlated,
                    noise_correlation_rho=rho,
                )
            )

        return sensors

    def apply_unit_scaling(
        self, values: Union[float, int, np.ndarray, Sequence[float]], scaler: UnitScaler
    ) -> Union[float, np.ndarray]:
        """Apply unit scaling last to transform canonical Celsius values into observable units.
        
        Args:
            values: Canonical Celsius values.
            scaler: UnitScaler instance.
            
        Returns:
            Scaled observable values.
        """
        return scaler.apply(values)

    def invert_unit_scaling(
        self, scaled_values: Union[float, int, np.ndarray, Sequence[float]], scaler: UnitScaler
    ) -> Union[float, np.ndarray]:
        """Invert observable values back to canonical Celsius.
        
        Args:
            scaled_values: Scaled observable values.
            scaler: UnitScaler instance.
            
        Returns:
            Canonical Celsius values.
        """
        return scaler.invert(scaled_values)
