"""Causal window configuration and sliding window utilities."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import get_config_dir, load_yaml


@dataclass(frozen=True)
class CausalWindows:
    """Window lengths for causal feature calculation (readings)."""
    short: int = 5
    medium: int = 30
    long: int = 120
    reference: int = 480
    causal_only: bool = True

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "CausalWindows":
        """Load window parameters from config/features.yaml or dictionary."""
        if config is None:
            cfg_path = get_config_dir() / "features.yaml"
            config = load_yaml(cfg_path)
        win_cfg = config.get("windows", {})
        return cls(
            short=int(win_cfg.get("short", 5)),
            medium=int(win_cfg.get("medium", 30)),
            long=int(win_cfg.get("long", 120)),
            reference=int(win_cfg.get("reference", 480)),
            causal_only=bool(win_cfg.get("causal_only", True)),
        )


def causal_slice(arr: Union[np.ndarray, pd.Series], t: int, window: int) -> np.ndarray:
    """Extract a causal window [max(0, t - window + 1), t] from arr.
    
    Guarantees no indices greater than t are accessed.
    """
    if isinstance(arr, pd.Series):
        arr = arr.to_numpy()
    start = max(0, t - window + 1)
    end = t + 1
    return arr[start:end]
