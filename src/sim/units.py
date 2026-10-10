"""Unit scaling transformations for sensor stream variables.

Applies unit scaling (Celsius, Fahrenheit, random affine) to canonical physical
measurements. In the Sensor Trust Intelligence architecture, unit scaling is
applied last, after all environment simulation, bounds clipping, and fault injection
have been completed in canonical units.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np


@dataclass(frozen=True)
class UnitScaler:
    """Affine unit scaling transform: y = multiplier * x + offset.
    
    Attributes:
        scale_type: Scaling identifier ('celsius', 'fahrenheit', or 'affine').
        multiplier: Multiplicative scaling factor (m).
        offset: Additive offset (c).
    """
    scale_type: str
    multiplier: float
    offset: float

    def __post_init__(self):
        if self.multiplier <= 0.0:
            raise ValueError(f"Multiplier must be strictly positive, got {self.multiplier}")
        if self.scale_type not in ("celsius", "fahrenheit", "affine"):
            raise ValueError(
                f"Unknown scale_type: '{self.scale_type}'. Expected 'celsius', 'fahrenheit', or 'affine'."
            )

    @classmethod
    def celsius(cls) -> "UnitScaler":
        """Return canonical Celsius identity scaler (multiplier=1.0, offset=0.0)."""
        return cls(scale_type="celsius", multiplier=1.0, offset=0.0)

    @classmethod
    def fahrenheit(cls) -> "UnitScaler":
        """Return standard Fahrenheit scaler (multiplier=1.8, offset=32.0)."""
        return cls(scale_type="fahrenheit", multiplier=1.8, offset=32.0)

    @classmethod
    def affine(cls, multiplier: float, offset: float) -> "UnitScaler":
        """Return arbitrary affine scaler."""
        return cls(scale_type="affine", multiplier=float(multiplier), offset=float(offset))

    def apply(
        self, values: Union[float, int, np.ndarray, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        """Transform canonical Celsius value(s) to observable units: y = m * x + c.
        
        Args:
            values: Scalar float/int, numpy array, or sequence of floats.
            
        Returns:
            Scaled value(s) in observable units.
        """
        if isinstance(values, (int, float, np.floating, np.integer)):
            return float(self.multiplier * values + self.offset)
        arr = np.asarray(values, dtype=float)
        return self.multiplier * arr + self.offset

    def invert(
        self, scaled_values: Union[float, int, np.ndarray, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        """Invert observable units back to canonical Celsius: x = (y - c) / m.
        
        Args:
            scaled_values: Scalar float/int, numpy array, or sequence of floats.
            
        Returns:
            Canonical Celsius value(s).
        """
        if isinstance(scaled_values, (int, float, np.floating, np.integer)):
            return float((scaled_values - self.offset) / self.multiplier)
        arr = np.asarray(scaled_values, dtype=float)
        return (arr - self.offset) / self.multiplier

    def scale_std(
        self, sigma: Union[float, int, np.ndarray, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        """Scale standard deviation / noise: sigma_scaled = m * sigma (offset has no effect).
        
        Args:
            sigma: Canonical standard deviation.
            
        Returns:
            Scaled standard deviation in observable units.
        """
        if isinstance(sigma, (int, float, np.floating, np.integer)):
            return float(self.multiplier * sigma)
        arr = np.asarray(sigma, dtype=float)
        return self.multiplier * arr

    def invert_std(
        self, scaled_sigma: Union[float, int, np.ndarray, Sequence[float]]
    ) -> Union[float, np.ndarray]:
        """Invert scaled standard deviation back to canonical Celsius: sigma = sigma_scaled / m.
        
        Args:
            scaled_sigma: Scaled standard deviation.
            
        Returns:
            Canonical Celsius standard deviation.
        """
        if isinstance(scaled_sigma, (int, float, np.floating, np.integer)):
            return float(scaled_sigma / self.multiplier)
        arr = np.asarray(scaled_sigma, dtype=float)
        return arr / self.multiplier

    def scale_bounds(self, bounds: Tuple[float, float]) -> Tuple[float, float]:
        """Scale a (min_bound, max_bound) interval from canonical Celsius to observable units.
        
        Args:
            bounds: Tuple of (min_bound, max_bound).
            
        Returns:
            Tuple of (scaled_min, scaled_max).
        """
        min_val, max_val = bounds
        return (float(self.apply(min_val)), float(self.apply(max_val)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize UnitScaler configuration to a dictionary."""
        return {
            "scale_type": self.scale_type,
            "multiplier": float(self.multiplier),
            "offset": float(self.offset),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnitScaler":
        """Reconstruct a UnitScaler instance from a dictionary."""
        return cls(
            scale_type=str(data["scale_type"]),
            multiplier=float(data["multiplier"]),
            offset=float(data["offset"]),
        )


def sample_unit_scaler(
    unit_scaling_config: Dict[str, Any],
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> UnitScaler:
    """Sample a UnitScaler deterministically driven by configuration.
    
    Configuration requirements (Appendix A.2 & Decision 1):
        - celsius_prob: Probability of Celsius (default 0.70)
        - fahrenheit_prob: Probability of Fahrenheit (default 0.15)
        - random_affine_prob: Probability of random affine (default 0.15)
        - affine_multiplier_min: Log-uniform minimum multiplier (default 0.5)
        - affine_multiplier_max: Log-uniform maximum multiplier (default 3.0)
        - affine_offset_min: Uniform minimum additive offset (default -10.0)
        - affine_offset_max: Uniform maximum additive offset (default 10.0)
        
    Args:
        unit_scaling_config: Dictionary containing unit_scaling configuration.
        rng: Optional seed integer or numpy random Generator.
        
    Returns:
        Sampled UnitScaler instance.
    """
    if rng is None:
        generator = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        generator = np.random.default_rng(int(rng))
    else:
        generator = rng

    c_prob = float(unit_scaling_config.get("celsius_prob", 0.70))
    f_prob = float(unit_scaling_config.get("fahrenheit_prob", 0.15))
    a_prob = float(unit_scaling_config.get("random_affine_prob", 0.15))

    total_prob = c_prob + f_prob + a_prob
    if not np.isclose(total_prob, 1.0, atol=1e-5):
        raise ValueError(
            f"Unit scaling probabilities must sum to 1.0, got {total_prob} "
            f"(celsius={c_prob}, fahrenheit={f_prob}, affine={a_prob})"
        )

    probs = np.array([c_prob, f_prob, a_prob]) / total_prob
    scale_choice = generator.choice(["celsius", "fahrenheit", "affine"], p=probs)

    if scale_choice == "celsius":
        return UnitScaler.celsius()
    elif scale_choice == "fahrenheit":
        return UnitScaler.fahrenheit()
    else:
        mult_min = float(unit_scaling_config["affine_multiplier_min"])
        mult_max = float(unit_scaling_config["affine_multiplier_max"])
        offset_min = float(unit_scaling_config["affine_offset_min"])
        offset_max = float(unit_scaling_config["affine_offset_max"])

        # Multiplier sampled log-uniform in [mult_min, mult_max]
        log_mult = generator.uniform(np.log(mult_min), np.log(mult_max))
        multiplier = float(np.exp(log_mult))

        # Additive offset sampled uniform in [offset_min, offset_max]
        offset = float(generator.uniform(offset_min, offset_max))

        return UnitScaler.affine(multiplier=multiplier, offset=offset)
