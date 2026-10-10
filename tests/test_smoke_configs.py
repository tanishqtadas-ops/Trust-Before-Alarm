"""Smoke test for configuration loading and parameter verification.

Loads all configs from config/ and verifies their completeness against Appendix A.
Can be executed as a pytest test or run directly as a script to print parameters.
"""

from pathlib import Path
import pprint
import sys

# Ensure repository root is in sys.path when executed directly as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import compute_config_hash, get_config_dir, load_all_configs


def run_smoke_test():
    """Load and print all configuration parameters with validation checks."""
    cfg_dir = get_config_dir()
    print(f"\n========================================================")
    print(f" SMOKE TEST: Loading configuration files from: {cfg_dir}")
    print(f"========================================================")

    configs = load_all_configs(cfg_dir)
    expected_configs = ["generator_a", "features", "alarm", "eval"]

    for name in expected_configs:
        assert name in configs, f"Missing required configuration file: {name}.yaml"
        assert isinstance(configs[name], dict), f"Config {name} did not parse into a dictionary"
        assert len(configs[name]) > 0, f"Config {name} is empty"

    full_hash = compute_config_hash(configs, length=16)
    print(f"\n[OK] All {len(expected_configs)} configuration files loaded successfully.")
    print(f"[OK] Deterministic Configuration Hash (SHA-256): {full_hash}\n")

    for name in expected_configs:
        print(f"--- Configuration: {name}.yaml (top-level keys: {list(configs[name].keys())}) ---")
        pprint.pprint(configs[name], indent=2, compact=False, width=100)
        print()

    return configs, full_hash


def test_smoke_load_configs():
    """Pytest test to ensure smoke test executes without errors and validates all keys."""
    configs, cfg_hash = run_smoke_test()
    assert len(configs) >= 4
    assert len(cfg_hash) == 16

    # Verify key sections from Generator A
    gen_a = configs["generator_a"]
    assert gen_a["structure"]["total_scenarios"] == 1500
    assert gen_a["structure"]["sampling_rate_minutes"] == 1
    assert gen_a["temperature_profile"]["bounds"]["environment_clip_min"] == -5.0
    assert gen_a["genuine_events"]["count_poisson_mean"] == 1.2
    assert gen_a["faults"]["spike"]["size_sigma_max"] == 40.0

    # Verify key sections from Features
    feats = configs["features"]
    assert feats["sigma_hat"]["mad_scale_factor"] == 1.4826
    assert feats["windows"]["short"] == 5
    assert feats["windows"]["reference"] == 480

    # Verify key sections from Alarm
    alarm = configs["alarm"]
    assert alarm["raw_alarm"]["baseline_window_hours"] == 8
    assert alarm["raw_alarm"]["threshold_multiplier"] == 3.0
    assert alarm["provable_rules"]["frozen"]["consecutive_identical_readings"] == 20

    # Verify key sections from Eval
    eval_cfg = configs["eval"]
    assert eval_cfg["evaluation_settings"]["false_alarm_budget_per_sensor_day"] == 0.5
    assert eval_cfg["pass_marks"]["pm1_hard_beat_baseline_points"] == 10.0


if __name__ == "__main__":
    run_smoke_test()
