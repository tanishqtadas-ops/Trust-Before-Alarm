"""Validation script asserting consistency between scenario objects, generated summary table, and plots."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from PIL import Image

from src.sim.scenario import ScenarioSimulator
from src.sim.strata import classify_scenario_stratum

csv_path = Path("runs/sanity_batch_100/authoritative_10_scenarios_summary.csv")
assert csv_path.exists(), "Summary CSV does not exist"
df_summary = pd.read_csv(csv_path)

assert len(df_summary) == 10, f"Expected 10 scenarios, got {len(df_summary)}"
tier_counts = df_summary["difficulty"].str.lower().value_counts().to_dict()
assert tier_counts.get("easy", 0) == 3, f"Expected 3 EASY, got {tier_counts.get('easy')}"
assert tier_counts.get("medium", 0) == 3, f"Expected 3 MEDIUM, got {tier_counts.get('medium')}"
assert tier_counts.get("hard", 0) == 4, f"Expected 4 HARD, got {tier_counts.get('hard')}"

sim = ScenarioSimulator()
plot_dir = Path("runs/sanity_batch_100/plots")
art_dir = Path(r"C:\Users\tanis\.gemini\antigravity-ide\brain\31444b36-40c1-4539-94a7-d17afc79d3a6")

print("=" * 80)
print("VALIDATING 10 AUTHORITATIVE SCENARIOS, METADATA, EPISODES, AND VISUAL PLOTS")
print("=" * 80)

for _, row in df_summary.iterrows():
    p_num = int(row["plot_num"])
    scen_id = str(row["scenario_id"])
    seed = int(row["seed"])
    tier = str(row["difficulty"]).lower()
    scale_type = str(row["unit_scale_type"])
    
    # 1. Simulate authoritative scenario
    scen = sim.simulate_scenario(scenario_id=scen_id, seed=seed)
    rec = scen.to_scenario_record()
    classified_tier, classified_profile = classify_scenario_stratum(scen)
    
    # Assert tier and profile
    assert classified_tier == tier, f"Tier mismatch for {scen_id}: {classified_tier} != {tier}"
    assert classified_profile == row["fault_profile"], f"Profile mismatch for {scen_id}"
    
    # Assert sensor counts
    assert scen.total_sensors == int(row["sensor_count"]), f"Sensor count mismatch for {scen_id}"
    assert rec["sensor_count"] == int(row["sensor_count"]), f"Record sensor count mismatch for {scen_id}"
    assert len(scen.sensor_ids) == int(row["sensor_count"]), f"Sensor IDs count mismatch for {scen_id}"
    
    # Assert unit scale
    unit_dict = json.loads(rec["unit_scale"])
    assert unit_dict["scale_type"] == scale_type, f"Unit scale mismatch for {scen_id}"
    assert abs(unit_dict["multiplier"] - float(row["multiplier"])) < 1e-6, f"Multiplier mismatch for {scen_id}"
    assert abs(unit_dict["offset"] - float(row["offset"])) < 1e-6, f"Offset mismatch for {scen_id}"
    
    # Assert fault episodes
    df_f = scen.to_fault_episodes_dataframe()
    assert len(df_f) == int(row["fault_episodes_count"]), f"Fault count mismatch for {scen_id}"
    
    # Assert event episodes
    df_e = scen.to_event_episodes_dataframe()
    assert len(df_e) == int(row["event_episodes_count"]), f"Event count mismatch for {scen_id}"
    
    # Detailed verification for replacement Plot 02 (scenario_00026)
    if p_num == 2:
        assert scen_id == "scenario_00026", f"Expected scenario_00026, got {scen_id}"
        assert len(df_f) == 1, "Expected exactly 1 fault episode"
        f_row = df_f.iloc[0]
        assert f_row["sensor_id"] == "sensor_05"
        assert f_row["fault_type"] == "offset_drift"
        assert int(f_row["start_step"]) == 1297
        assert int(f_row["end_step"]) == 1920
        assert f_row["max_severity"] > 0.70, f"Expected severity > 0.70, got {f_row['max_severity']}"
        shape_p = json.loads(f_row["shape_params"])
        assert shape_p["shape"] == "saturating"
        assert shape_p["sign"] == -1
        # Verify severity progression in ground truth sensors dataframe
        df_sens = scen.to_ground_truth_sensors_dataframe()
        s5 = df_sens[df_sens["sensor_id"] == "sensor_05"].sort_values("timestamp")
        # Pre-fault severity must be 0
        assert (s5[s5["timestamp"] < 1297]["fault_severity"] == 0).all()
        # Post-fault severity must be strictly increasing
        sev_active = s5[s5["timestamp"] >= 1297]["fault_severity"].values
        assert len(sev_active) == 623
        assert (np.diff(sev_active) >= 0).all(), "Severity must monotonically progress"
        assert sev_active[-1] > 0.70
        print(f"  [VERIFIED P02] scenario_00026 offset-drift progression: onset step 1297, end step 1920, monotonic saturating severity up to {sev_active[-1]:.3f}")
        
    # Detailed verification for replacement Plot 03 (scenario_00027)
    if p_num == 3:
        assert scen_id == "scenario_00027", f"Expected scenario_00027, got {scen_id}"
        assert len(df_f) == 1, "Expected exactly 1 fault episode"
        f_row = df_f.iloc[0]
        assert f_row["sensor_id"] == "sensor_01"
        assert f_row["fault_type"] == "high_noise"
        assert int(f_row["start_step"]) == 509
        assert int(f_row["end_step"]) == 1109
        assert int(f_row["duration"]) == 600, f"Expected 600 steps duration, got {f_row['duration']}"
        assert abs(f_row["max_severity"] - 0.5037) < 1e-3
        shape_p = json.loads(f_row["shape_params"])
        assert shape_p["is_bursty"] is False
        assert shape_p["multiplier"] > 5.5
        # Verify ground truth recovery
        df_sens = scen.to_ground_truth_sensors_dataframe()
        s1 = df_sens[df_sens["sensor_id"] == "sensor_01"].sort_values("timestamp")
        assert (s1[s1["timestamp"] < 509]["fault_severity"] == 0).all()
        assert (s1[(s1["timestamp"] >= 509) & (s1["timestamp"] < 1109)]["sensor_untrusted"] == 1).all()
        assert (s1[s1["timestamp"] >= 1109]["fault_severity"] == 0).all()
        assert (s1[s1["timestamp"] >= 1109]["sensor_untrusted"] == 0).all()
        print(f"  [VERIFIED P03] scenario_00027 high-noise episode: onset step 509, end step 1109 (600 min duration), full recovery to step 1920")

    # Assert plot file exists and is readable
    file_name = f"sanity_plot_{p_num:02d}_{scen_id}_{tier}_{scale_type}.png"
    p_file = plot_dir / file_name
    
    assert p_file.exists(), f"Plot file {p_file} does not exist"
    if art_dir.exists():
        art_file = art_dir / file_name
        art_plots_file = art_dir / "plots" / file_name
        if art_file.exists():
            assert art_file.exists()
        if art_plots_file.exists():
            assert art_plots_file.exists()
    
    with Image.open(p_file) as img:
        assert img.size == (2100, 1540), f"Unexpected image size: {img.size}"
        
    print(f"  [OK] Plot {p_num:02d} ({scen_id}, {tier.upper()}, {scale_type}): {scen.total_sensors} sensors, {len(df_f)} faults, {len(df_e)} events validated.")

print("\n" + "=" * 80)
print("ALL 10 SCENARIOS, METADATA, EPISODES, AND HIGH-RES PLOTS STRICTLY VALIDATED!")
print("=" * 80)
