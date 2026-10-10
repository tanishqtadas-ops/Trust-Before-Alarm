"""Simulation module for Sensor Trust Intelligence.

Contains variable profile definitions, unit scaling transformations,
environment simulation, sensor models, event generators, and fault injectors.
"""

from src.sim.environment import (
    EnvironmentSeries,
    EnvironmentSimulator,
    generate_environment_series,
)
from src.sim.profile import (
    TemperatureProfile,
    TemperatureScenarioParams,
    TemperatureSensorParams,
    VariableProfile,
)
from src.sim.sensors import (
    SensorStream,
    VirtualSensorGroup,
    VirtualSensorSimulator,
    apply_clipping,
    apply_quantisation,
    apply_response_lag,
    generate_group_noise,
    generate_missing_mask,
    simulate_sensors,
)
from src.sim.units import UnitScaler, sample_unit_scaler

__all__ = [
    "UnitScaler",
    "sample_unit_scaler",
    "VariableProfile",
    "TemperatureProfile",
    "TemperatureScenarioParams",
    "TemperatureSensorParams",
    "EnvironmentSeries",
    "EnvironmentSimulator",
    "generate_environment_series",
    "SensorStream",
    "VirtualSensorGroup",
    "VirtualSensorSimulator",
    "simulate_sensors",
    "apply_response_lag",
    "generate_group_noise",
    "apply_clipping",
    "apply_quantisation",
    "generate_missing_mask",
]
