"""Unit tests for the configuration loader and hashing utilities."""

import pytest
from pathlib import Path
from src.utils.config import (
    load_yaml,
    load_all_configs,
    compute_config_hash,
    get_repo_root,
    get_config_dir,
)


def test_get_repo_root():
    root = get_repo_root()
    assert root.exists()
    assert (root / "PROGRESS.md").exists()
    assert (root / "requirements.txt").exists()


def test_load_yaml_valid(tmp_path: Path):
    test_yaml = tmp_path / "test.yaml"
    test_yaml.write_text("sampling: 1\nsensors:\n  min: 4\n  max: 8\n", encoding="utf-8")

    data = load_yaml(test_yaml)
    assert isinstance(data, dict)
    assert data["sampling"] == 1
    assert data["sensors"]["min"] == 4
    assert data["sensors"]["max"] == 8


def test_load_yaml_nonexistent(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml(missing_file)


def test_load_all_configs(tmp_path: Path):
    (tmp_path / "generator_a.yaml").write_text("seed: 42\n", encoding="utf-8")
    (tmp_path / "alarm.yaml").write_text("threshold: 3.0\n", encoding="utf-8")

    configs = load_all_configs(tmp_path)
    assert "generator_a" in configs
    assert "alarm" in configs
    assert configs["generator_a"]["seed"] == 42
    assert configs["alarm"]["threshold"] == 3.0


def test_compute_config_hash():
    data1 = {"b": 2, "a": 1, "nested": {"y": [1, 2], "x": True}}
    data2 = {"a": 1, "nested": {"x": True, "y": [1, 2]}, "b": 2}
    data3 = {"a": 1, "nested": {"x": False, "y": [1, 2]}, "b": 2}

    # Determinism regardless of key order
    hash1 = compute_config_hash(data1)
    hash2 = compute_config_hash(data2)
    hash3 = compute_config_hash(data3)

    assert len(hash1) == 16
    assert hash1 == hash2
    assert hash1 != hash3

    # Custom hash length
    hash_full = compute_config_hash(data1, length=64)
    assert len(hash_full) == 64
