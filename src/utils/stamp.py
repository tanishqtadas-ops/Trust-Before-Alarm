"""Run stamp utility for Sensor Trust Intelligence.

Every run must carry a run id, seed, config hash, and timestamp,
written next to outputs to guarantee full auditability and reproducibility.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import uuid

from src.utils.config import compute_config_hash


@dataclass
class RunStamp:
    """Metadata stamp uniquely identifying a simulation, training, or evaluation run."""
    run_id: str
    seed: int
    config_hash: str
    timestamp: str
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert stamp to a dictionary representation."""
        data = asdict(self)
        if self.extra is None:
            data.pop("extra", None)
        return data


def generate_run_id(prefix: str = "run") -> str:
    """Generate a clean, timestamped unique run ID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uid = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short_uid}"


def create_run_stamp(
    seed: int,
    config_data: Union[Dict[str, Any], str, Path],
    run_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RunStamp:
    """Create a RunStamp object with computed config hash and UTC timestamp.
    
    Args:
        seed: Random seed for the run.
        config_data: Dict or path to YAML config file/directory.
        run_id: Optional custom run id (auto-generated if omitted).
        extra: Optional additional key-value metadata.
        
    Returns:
        RunStamp instance.
    """
    if run_id is None:
        run_id = generate_run_id()
    
    cfg_hash = compute_config_hash(config_data)
    ts = datetime.now(timezone.utc).isoformat()
    
    return RunStamp(
        run_id=run_id,
        seed=seed,
        config_hash=cfg_hash,
        timestamp=ts,
        extra=extra if extra else None,
    )


def save_run_stamp(
    stamp: RunStamp,
    output_dir: Union[str, Path],
    filename: str = "run_stamp.json",
) -> Path:
    """Write run stamp metadata to a JSON file in output_dir.
    
    Args:
        stamp: The RunStamp instance.
        output_dir: Target directory path.
        filename: Name of the stamp file (default: 'run_stamp.json').
        
    Returns:
        Path to the saved stamp file.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    stamp_file = out_path / filename

    with open(stamp_file, "w", encoding="utf-8") as f:
        json.dump(stamp.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")

    return stamp_file


def load_run_stamp(file_path: Union[str, Path]) -> RunStamp:
    """Load and parse a RunStamp from a JSON file."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Run stamp file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return RunStamp(
        run_id=data["run_id"],
        seed=int(data["seed"]),
        config_hash=data["config_hash"],
        timestamp=data["timestamp"],
        extra=data.get("extra"),
    )
