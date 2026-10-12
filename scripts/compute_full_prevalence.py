import sys
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/ground_truth")
PARTITIONS = ["train", "val", "test_full_resolution", "test_downsampled"]

print("=" * 80)
print("COMPUTING PREVALENCE METRICS ACROSS ALL FOUR DATASET PARTITIONS")
print("=" * 80)

partition_sensor_stats = {}
partition_event_stats = {}

fault_columns = [
    "f_spike", "f_offset_drift", "f_gain_error", "f_frozen",
    "f_high_noise", "f_dropout", "f_slow_degradation"
]

total_sensor_rows_full = 0
total_sensor_rows_scored = 0
total_untrusted_full = 0
total_untrusted_scored = 0

total_group_rows_full = 0
total_group_rows_scored = 0
total_genuine_event_full = 0
total_genuine_event_scored = 0
total_event_below_full = 0
total_event_below_scored = 0

fault_counts_scored = {fc: 0 for fc in fault_columns}
fault_counts_full = {fc: 0 for fc in fault_columns}

for part in PARTITIONS:
    p_dir = DATA_DIR / part
    sensors_file = p_dir / "ground_truth_sensors.parquet"
    events_file = p_dir / "ground_truth_events.parquet"
    
    assert sensors_file.exists(), f"Missing {sensors_file}"
    assert events_file.exists(), f"Missing {events_file}"
    
    df_sensors = pd.read_parquet(sensors_file)
    df_events = pd.read_parquet(events_file)
    
    # Scored filter: timestamp >= 480.0
    df_sensors_scored = df_sensors[df_sensors["timestamp"] >= 480.0]
    df_events_scored = df_events[df_events["timestamp"] >= 480.0]
    
    s_rows_full = len(df_sensors)
    s_rows_scored = len(df_sensors_scored)
    s_untrusted_full = int(df_sensors["sensor_untrusted"].sum())
    s_untrusted_scored = int(df_sensors_scored["sensor_untrusted"].sum())
    
    e_rows_full = len(df_events)
    e_rows_scored = len(df_events_scored)
    e_genuine_full = int(df_events["genuine_event"].sum())
    e_genuine_scored = int(df_events_scored["genuine_event"].sum())
    e_below_full = int(df_events["event_below_detectability"].sum())
    e_below_scored = int(df_events_scored["event_below_detectability"].sum())
    
    total_sensor_rows_full += s_rows_full
    total_sensor_rows_scored += s_rows_scored
    total_untrusted_full += s_untrusted_full
    total_untrusted_scored += s_untrusted_scored
    
    total_group_rows_full += e_rows_full
    total_group_rows_scored += e_rows_scored
    total_genuine_event_full += e_genuine_full
    total_genuine_event_scored += e_genuine_scored
    total_event_below_full += e_below_full
    total_event_below_scored += e_below_scored
    
    for fc in fault_columns:
        fault_counts_scored[fc] += int(df_sensors_scored[fc].sum())
        fault_counts_full[fc] += int(df_sensors[fc].sum())
        
    partition_sensor_stats[part] = {
        "rows_full": s_rows_full,
        "rows_scored": s_rows_scored,
        "untrusted_full": s_untrusted_full,
        "untrusted_scored": s_untrusted_scored,
        "untrusted_rate_full": s_untrusted_full / s_rows_full,
        "untrusted_rate_scored": s_untrusted_scored / s_rows_scored,
    }
    
    partition_event_stats[part] = {
        "rows_full": e_rows_full,
        "rows_scored": e_rows_scored,
        "genuine_full": e_genuine_full,
        "genuine_scored": e_genuine_scored,
        "genuine_rate_full": e_genuine_full / e_rows_full,
        "genuine_rate_scored": e_genuine_scored / e_rows_scored,
        "below_full": e_below_full,
        "below_scored": e_below_scored,
        "below_rate_full": e_below_full / e_rows_full,
        "below_rate_scored": e_below_scored / e_rows_scored,
    }
    
    print(f"\nPartition: {part}")
    print(f"  Sensor Rows: Full={s_rows_full:,}, Scored={s_rows_scored:,}")
    print(f"  Untrusted Sensor Steps: Full={s_untrusted_full:,} ({s_untrusted_full / s_rows_full * 100:.4f}%), Scored={s_untrusted_scored:,} ({s_untrusted_scored / s_rows_scored * 100:.4f}%)")
    print(f"  Group Rows: Full={e_rows_full:,}, Scored={e_rows_scored:,}")
    print(f"  Genuine Events: Full={e_genuine_full:,} ({e_genuine_full / e_rows_full * 100:.4f}%), Scored={e_genuine_scored:,} ({e_genuine_scored / e_rows_scored * 100:.4f}%)")
    print(f"  Below Detectability: Full={e_below_full:,} ({e_below_full / e_rows_full * 100:.4f}%), Scored={e_below_scored:,} ({e_below_scored / e_rows_scored * 100:.4f}%)")

print("\n" + "=" * 80)
print("POOLED TOTALS ACROSS ALL 4 PARTITIONS (1,500 SCENARIOS)")
print("=" * 80)
print(f"Total Sensor Rows: Full={total_sensor_rows_full:,}, Scored={total_sensor_rows_scored:,}")
print(f"Untrusted Sensor Steps (POOLED EXPORTED-ROW PREVALENCE):")
print(f"  Full Duration: {total_untrusted_full:,} / {total_sensor_rows_full:,} = {total_untrusted_full / total_sensor_rows_full * 100:.4f}%")
print(f"  Scored Window: {total_untrusted_scored:,} / {total_sensor_rows_scored:,} = {total_untrusted_scored / total_sensor_rows_scored * 100:.4f}%")
print(f"\nTotal Group Rows: Full={total_group_rows_full:,}, Scored={total_group_rows_scored:,}")
print(f"Genuine Events (POOLED EXPORTED-ROW PREVALENCE, TARGET 8-15%):")
print(f"  Full Duration: {total_genuine_event_full:,} / {total_group_rows_full:,} = {total_genuine_event_full / total_group_rows_full * 100:.4f}%")
print(f"  Scored Window: {total_genuine_event_scored:,} / {total_group_rows_scored:,} = {total_genuine_event_scored / total_group_rows_scored * 100:.4f}%")
print(f"\nEvents Below Detectability:")
print(f"  Full Duration: {total_event_below_full:,} / {total_group_rows_full:,} = {total_event_below_full / total_group_rows_full * 100:.4f}%")
print(f"  Scored Window: {total_event_below_scored:,} / {total_group_rows_scored:,} = {total_event_below_scored / total_group_rows_scored * 100:.4f}%")

print("\n" + "=" * 80)
print("ACTIVE FAULT LABELS IN SCORED WINDOW (POOLED)")
print("=" * 80)
for fc in fault_columns:
    c_scored = fault_counts_scored[fc]
    c_full = fault_counts_full[fc]
    pct_scored = c_scored / total_sensor_rows_scored * 100
    pct_full = c_full / total_sensor_rows_full * 100
    print(f"  {fc:<20}: Scored={c_scored:>7,} ({pct_scored:6.4f}%) | Full={c_full:>7,} ({pct_full:6.4f}%)")
