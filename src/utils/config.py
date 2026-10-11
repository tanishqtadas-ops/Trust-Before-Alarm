"""Configuration loader and hashing utilities for Sensor Trust Intelligence."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


def get_repo_root() -> Path:
    """Return the absolute Path to the repository root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "PROGRESS.md").exists() or (parent / "requirements.txt").exists():
            return parent
    return current.parent.parent.parent


def get_config_dir(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve and return the configuration directory path."""
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Config directory does not exist: {p}")
        return p
    default_dir = get_repo_root() / "config"
    return default_dir


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a single YAML configuration file."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML configuration at {path} must parse to a dictionary, got {type(data).__name__}")
    return data


def load_all_configs(config_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load all .yaml and .yml files from the config directory into a dictionary keyed by filename stem."""
    cfg_dir = get_config_dir(config_dir)
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"Config directory does not exist: {cfg_dir}")

    configs: Dict[str, Any] = {}
    yaml_files = sorted(list(cfg_dir.glob("*.yaml")) + list(cfg_dir.glob("*.yml")))

    for yml_file in yaml_files:
        stem = yml_file.stem
        configs[stem] = load_yaml(yml_file)

    return configs


def _normalize_for_serialization(obj: Any) -> Any:
    """Recursively convert all dictionary keys to strings so json.dumps sort_keys can compare them safely."""
    if isinstance(obj, dict):
        return {str(k): _normalize_for_serialization(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_serialization(item) for item in obj]
    return obj


def compute_config_hash(config_data: Union[Dict[str, Any], str, Path], length: int = 16) -> str:
    """Compute a deterministic SHA-256 hash of a config dictionary or YAML file.
    
    Args:
        config_data: Either a dictionary containing config data, or a path to a YAML file / config dir.
        length: Number of hex characters to return (default: 16).
        
    Returns:
        Hex string hash.
    """
    if isinstance(config_data, (str, Path)):
        p = Path(config_data).resolve()
        if p.is_dir():
            data = load_all_configs(p)
        elif p.is_file():
            data = load_yaml(p)
        else:
            raise FileNotFoundError(f"Path does not exist: {p}")
    else:
        data = config_data

    normalized = _normalize_for_serialization(data)
    # Canonical JSON string with sorted keys ensures deterministic hash
    canonical_repr = json.dumps(normalized, sort_keys=True, ensure_ascii=True, default=str)
    hasher = hashlib.sha256(canonical_repr.encode("utf-8"))
    digest = hasher.hexdigest()
    return digest[:length] if length else digest


def canonical_json(obj: Any) -> str:
    """Serialize object to deterministic JSON string with sorted keys and stringified types."""
    if obj is None:
        return "{}"
    if isinstance(obj, dict) and len(obj) == 0:
        return "{}"
    normalized = _normalize_for_serialization(obj)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=True, default=str)
