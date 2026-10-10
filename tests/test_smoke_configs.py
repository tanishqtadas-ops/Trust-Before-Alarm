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
    assert gen_a["temperature_profile"]["daily_cycle"]["period_minutes"] == 1440
    assert gen_a["temperature_profile"]["daily_cycle"]["phase_min_rad"] == 0.0
    assert gen_a["temperature_profile"]["daily_cycle"]["phase_max_rad"] == 6.283185307179586
    assert gen_a["temperature_profile"]["bounds"]["environment_clip_min"] == -5.0
    assert gen_a["genuine_events"]["count_poisson_mean"] == 1.2
    assert gen_a["faults"]["spike"]["size_sigma_max"] == 40.0
    assert gen_a["difficulty_tiers"]["hard"]["small_group_sensor_count"] == 4
    assert gen_a["difficulty_tiers"]["hard"]["no_clear_majority_cluster_max_fraction"] == 0.50
    assert gen_a["difficulty_tiers"]["hard"]["sensor_near_bounds"]["sigma_k_multiplier"] == 3.0
    assert gen_a["difficulty_tiers"]["hard"]["sensor_near_bounds"]["fixed_margin_celsius"] == 2.0
    assert gen_a["difficulty_tiers"]["hard"]["sensor_near_bounds"]["check_canonical_celsius"] is True
    assert gen_a["temperature_profile"]["unit_scaling"]["affine_offset_min"] == -10.0
    assert gen_a["temperature_profile"]["unit_scaling"]["affine_offset_max"] == 10.0
    assert gen_a["temperature_profile"]["unit_scaling"]["affine_offset_distribution"] == "uniform"
    assert gen_a["genuine_events"]["durations"]["sustained_hold_until_scenario_end"] is True
    assert gen_a["faults"]["sensor_untrusted_rule"] == "active_interval"
    assert gen_a["combinations"]["shared_cause_severity_relative_tolerance"] == 0.20
    assert "subtle_fault" in gen_a["difficulty_tiers"]["hard"]["mechanisms"]
    assert gen_a["structure"]["test_stride_5_count"] == 150
    assert gen_a["structure"]["max_faulty_sensors_fraction"] == 0.4

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
    assert alarm["trust_engine"]["status_thresholds"]["healthy_rule"] == "T > 0.8"
    assert alarm["trust_engine"]["status_thresholds"]["watch_rule"] == "0.5 <= T <= 0.8"
    assert alarm["trust_engine"]["status_thresholds"]["unreliable_rule"] == "T < 0.5"

    # Verify key sections from Eval
    eval_cfg = configs["eval"]
    assert eval_cfg["evaluation_settings"]["false_alarm_budget_per_sensor_day"] == 0.5
    assert eval_cfg["pass_marks"]["pm1_hard_beat_baseline_points"] == 10.0


if __name__ == "__main__":
    run_smoke_test()
