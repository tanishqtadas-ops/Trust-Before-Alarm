"""True environment simulation for sensor streams.

Generates the canonical physical environment time series in Celsius (Appendix A.2):
  1. Base temperature level (uniform 2 to 38 C)
  2. Daily diurnal cycle (amplitude uniform 0.5 to 4 C, 20% zero, period 24h, random phase)
  3. Slow random wander (stationary OU AR(1) process, std 0.2 to 0.8 C, correlation time 120 to 360 min,
     followed by 5-step moving average)
  4. Environmental bounds clipping (clipped to [-5, 50] C)

All simulation is conducted strictly in canonical Celsius before virtual sensor effects
(P1.3) and unit scaling (P1.1) are applied.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.sim.profile import TemperatureProfile, TemperatureScenarioParams
from src.utils.config import get_config_dir, load_yaml


@dataclass(frozen=True)
class EnvironmentSeries:
    """True environmental series and component breakdown in canonical Celsius.
    
    Attributes:
        values: Clipped environmental temperature series in canonical Celsius, shape (total_steps,).
        unclipped_values: Raw sum of base level, daily cycle, and slow wander before clipping.
        base_level: Constant base temperature level for the scenario in Celsius.
        daily_cycle: Diurnal sinusoidal cycle component, shape (total_steps,).
        slow_wander: Slow random wander component after 5-step moving average, shape (total_steps,).
        timestamps_minutes: Array of elapsed minutes, shape (total_steps,).
        params: Scenario parameters used to generate this environment series.
        total_steps: Total duration of the simulation run in steps.
        warmup_steps: Number of initial warm-up steps.
        scored_steps: Number of evaluation/scored steps.
    """
    values: np.ndarray
    unclipped_values: np.ndarray
    base_level: float
    daily_cycle: np.ndarray
    slow_wander: np.ndarray
    timestamps_minutes: np.ndarray
    params: TemperatureScenarioParams
    total_steps: int
    warmup_steps: int
    scored_steps: int

    @property
    def warmup_values(self) -> np.ndarray:
        """Slice of canonical environmental readings during warm-up phase (0 to warmup_steps)."""
        return self.values[: self.warmup_steps]

    @property
    def scored_values(self) -> np.ndarray:
        """Slice of canonical environmental readings during scored phase (warmup_steps to total_steps)."""
        return self.values[self.warmup_steps :]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata and summary statistics."""
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "scored_steps": self.scored_steps,
            "base_level": float(self.base_level),
            "min_value": float(np.min(self.values)),
            "max_value": float(np.max(self.values)),
            "mean_value": float(np.mean(self.values)),
            "std_value": float(np.std(self.values)),
            "clipped_steps_count": int(np.sum(self.values != self.unclipped_values)),
            "params": self.params.to_dict(),
        }


class EnvironmentSimulator:
    """Simulator for generating true canonical environmental time series driven by configuration."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        profile: Optional[TemperatureProfile] = None,
    ):
        """Initialize EnvironmentSimulator.
        
        Args:
            config: Optional full configuration dictionary. If None, loaded from config/generator_a.yaml.
            profile: Optional TemperatureProfile instance. If None, created from config.
        """
        if config is None:
            default_yaml = get_config_dir() / "generator_a.yaml"
            self.config = load_yaml(default_yaml)
        else:
            self.config = config

        self.profile = profile if profile is not None else TemperatureProfile(self.config)
        self._extract_structure_config()

    def _extract_structure_config(self) -> None:
        """Extract scenario duration and sampling parameters from configuration structure."""
        struct = self.config["structure"]
        self.sampling_rate_minutes = float(struct["sampling_rate_minutes"])
        self.total_steps = int(struct["total_steps"])
        self.warmup_steps = int(struct["warmup_steps"])
        self.scored_steps = int(struct["scored_steps"])

        if self.warmup_steps + self.scored_steps != self.total_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) + scored_steps ({self.scored_steps}) "
                f"!= total_steps ({self.total_steps})"
            )

    @staticmethod
    def generate_daily_cycle(
        steps: int,
        amplitude: float,
        period_minutes: float,
        phase: float,
        sampling_rate_minutes: float = 1.0,
    ) -> np.ndarray:
        """Generate diurnal sinusoidal temperature cycle.
        
        Args:
            steps: Number of time steps.
            amplitude: Cycle amplitude in Celsius (0.0 if zero-amplitude scenario).
            period_minutes: Cycle period in minutes (typically 1440 for 24h).
            phase: Cycle phase offset in radians.
            sampling_rate_minutes: Minutes per step (typically 1.0).
            
        Returns:
            Numpy array of cycle values of length steps.
        """
        if steps <= 0:
            return np.empty(0, dtype=np.float64)
        if amplitude == 0.0:
            return np.zeros(steps, dtype=np.float64)

        t_minutes = np.arange(steps, dtype=np.float64) * sampling_rate_minutes
        angular_freq = 2.0 * np.pi / period_minutes
        return amplitude * np.sin(angular_freq * t_minutes + phase)

    @staticmethod
    def generate_slow_wander(
        steps: int,
        std: float,
        correlation_time_min: float,
        ma_steps: int,
        rng: np.random.Generator,
        sampling_rate_minutes: float = 1.0,
    ) -> np.ndarray:
        """Generate slow random wander via discrete Ornstein-Uhlenbeck AR(1) process with moving average.
        
        Decision 13:
          The continuous OU process with correlation decay time tau is discretized with time step dt:
            alpha = exp(-dt / tau)
            sigma_innov = std * sqrt(1 - alpha^2)
            x_t = alpha * x_{t-1} + sigma_innov * epsilon_t
          Pre-generating (ma_steps - 1) warm-up points allows the causal moving average of length ma_steps
          to be fully evaluated from step 0 without edge attenuation.
          
        Args:
            steps: Number of time steps.
            std: Target stationary standard deviation in Celsius.
            correlation_time_min: Correlation decay time tau in minutes.
            ma_steps: Moving average window size in steps (typically 5).
            rng: Numpy random Generator.
            sampling_rate_minutes: Minutes per step (typically 1.0).
            
        Returns:
            Numpy array of smoothed slow wander of length steps.
        """
        if steps <= 0:
            return np.empty(0, dtype=np.float64)
        if std == 0.0:
            return np.zeros(steps, dtype=np.float64)

        dt = float(sampling_rate_minutes)
        tau = float(correlation_time_min)
        alpha = float(np.exp(-dt / tau))
        innov_scale = float(std * np.sqrt(max(0.0, 1.0 - alpha**2)))

        # Total points needed to form a full moving-average window from step 0
        warmup_points = max(0, ma_steps - 1)
        total_ou_steps = steps + warmup_points

        # Generate Gaussian innovations
        innovations = rng.normal(loc=0.0, scale=1.0, size=total_ou_steps)

        # Allocate AR(1) path
        x = np.empty(total_ou_steps, dtype=np.float64)
        # Stationary initial condition
        x[0] = rng.normal(loc=0.0, scale=std)

        # Step through AR(1) recursion
        for t in range(1, total_ou_steps):
            x[t] = alpha * x[t - 1] + innov_scale * innovations[t]

        # Apply causal moving average of window size ma_steps
        if ma_steps <= 1:
            return x[warmup_points:]

        # Fast cumulative sum causal rolling mean
        # Using 1D convolution with uniform weights
        kernel = np.ones(ma_steps, dtype=np.float64) / ma_steps
        # Valid convolution on x gives length (total_ou_steps - ma_steps + 1) == steps
        smoothed = np.convolve(x, kernel, mode="valid")
        return smoothed

    @staticmethod
    def clip_environment(
        values: np.ndarray, clip_min: float, clip_max: float
    ) -> np.ndarray:
        """Clip temperature readings to environmental bounds [-5, 50] C.
        
        Args:
            values: Unclipped temperature array.
            clip_min: Minimum environment bound.
            clip_max: Maximum environment bound.
            
        Returns:
            Clipped temperature array.
        """
        return np.clip(values, clip_min, clip_max)

    def generate(
        self,
        seed: Optional[int] = None,
        rng: Optional[Union[int, np.random.Generator]] = None,
        params: Optional[TemperatureScenarioParams] = None,
    ) -> EnvironmentSeries:
        """Deterministically generate the true environmental temperature series in canonical Celsius.
        
        Args:
            seed: Optional integer seed for reproducible generation.
            rng: Optional seed integer or numpy random Generator.
            params: Optional pre-sampled TemperatureScenarioParams. If None, sampled from profile.
            
        Returns:
            EnvironmentSeries containing values and component breakdown.
        """
        if rng is None and seed is not None:
            gen = np.random.default_rng(seed)
        elif isinstance(rng, (int, np.integer)):
            gen = np.random.default_rng(int(rng))
        elif isinstance(rng, np.random.Generator):
            gen = rng
        else:
            gen = np.random.default_rng()

        # 1. Obtain scenario parameters
        if params is None:
            scenario_params = self.profile.sample_scenario_params(rng=gen)
        else:
            scenario_params = params

        # 2. Daily cycle component
        daily_cycle = self.generate_daily_cycle(
            steps=self.total_steps,
            amplitude=scenario_params.daily_cycle_amplitude,
            period_minutes=scenario_params.daily_cycle_period_minutes,
            phase=scenario_params.daily_cycle_phase,
            sampling_rate_minutes=self.sampling_rate_minutes,
        )

        # 3. Slow random wander component
        slow_wander = self.generate_slow_wander(
            steps=self.total_steps,
            std=scenario_params.slow_wander_std,
            correlation_time_min=scenario_params.slow_wander_correlation_time,
            ma_steps=scenario_params.slow_wander_ma_steps,
            rng=gen,
            sampling_rate_minutes=self.sampling_rate_minutes,
        )

        # 4. Base level component
        base_level = scenario_params.base_level

        # 5. Unclipped raw environmental temperature
        unclipped = base_level + daily_cycle + slow_wander

        # 6. Environmental clipping to configured bounds
        clipped = self.clip_environment(
            unclipped,
            clip_min=scenario_params.environment_clip_min,
            clip_max=scenario_params.environment_clip_max,
        )

        timestamps = np.arange(self.total_steps, dtype=np.float64) * self.sampling_rate_minutes

        return EnvironmentSeries(
            values=clipped,
            unclipped_values=unclipped,
            base_level=base_level,
            daily_cycle=daily_cycle,
            slow_wander=slow_wander,
            timestamps_minutes=timestamps,
            params=scenario_params,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            scored_steps=self.scored_steps,
        )


def generate_environment_series(
    seed: Optional[int] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
    config: Optional[Dict[str, Any]] = None,
    params: Optional[TemperatureScenarioParams] = None,
) -> EnvironmentSeries:
    """Convenience top-level function to generate an EnvironmentSeries.
    
    Args:
        seed: Optional integer seed for reproducibility.
        rng: Optional seed or numpy Generator.
        config: Optional configuration dictionary.
        params: Optional pre-sampled TemperatureScenarioParams.
        
    Returns:
        EnvironmentSeries instance.
    """
    simulator = EnvironmentSimulator(config=config)
    return simulator.generate(seed=seed, rng=rng, params=params)
