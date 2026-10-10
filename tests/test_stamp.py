"""Unit tests for the run stamp utility."""

from pathlib import Path
import pytest
from src.utils.stamp import (
    RunStamp,
    create_run_stamp,
    generate_run_id,
    load_run_stamp,
    save_run_stamp,
)


def test_generate_run_id():
    rid1 = generate_run_id()
    rid2 = generate_run_id(prefix="eval")
    assert rid1.startswith("run_")
    assert rid2.startswith("eval_")
    assert rid1 != rid2


def test_create_run_stamp():
    config_dict = {"sampling": 1, "seed": 42}
    stamp = create_run_stamp(seed=42, config_data=config_dict, run_id="test_run_001")
    assert stamp.run_id == "test_run_001"
    assert stamp.seed == 42
    assert len(stamp.config_hash) == 16
    assert stamp.timestamp is not None


def test_save_and_load_run_stamp(tmp_path: Path):
    config_dict = {"environment": {"base_level": [2, 38]}}
    stamp = create_run_stamp(
        seed=101,
        config_data=config_dict,
        run_id="custom_run",
        extra={"tier": "EASY", "n_sensors": 6},
    )

    out_file = save_run_stamp(stamp, tmp_path)
    assert out_file.exists()
    assert out_file.name == "run_stamp.json"

    loaded = load_run_stamp(out_file)
    assert loaded.run_id == stamp.run_id
    assert loaded.seed == stamp.seed
    assert loaded.config_hash == stamp.config_hash
    assert loaded.timestamp == stamp.timestamp
    assert loaded.extra == {"tier": "EASY", "n_sensors": 6}


def test_load_run_stamp_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_run_stamp(tmp_path / "missing.json")
