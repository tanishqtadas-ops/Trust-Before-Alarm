"""Simulation module for Sensor Trust Intelligence.

Contains variable profile definitions, unit scaling transformations,
environment simulation, sensor models, event generators, and fault injectors.
"""

from src.sim.profile import (
    TemperatureProfile,
    TemperatureScenarioParams,
    TemperatureSensorParams,
    VariableProfile,
)
from src.sim.units import UnitScaler, sample_unit_scaler

__all__ = [
    "UnitScaler",
    "sample_unit_scaler",
    "VariableProfile",
    "TemperatureProfile",
    "TemperatureScenarioParams",
    "TemperatureSensorParams",
]
