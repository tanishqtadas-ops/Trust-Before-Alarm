"""Regenerate the 10 sanity batch plots with 100% authoritative metadata and validate consistency."""

import os
import sys
import json
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.sim.scenario import ScenarioSimulator, VirtualScenario
from src.sim.strata import classify_scenario_stratum

RUNS_DIR = Path("runs/sanity_batch_100")
PLOTS_DIR = RUNS_DIR / "plots"
CURRENT_ARTIFACTS_DIR = Path(r"C:\Users\tanis\.gemini\antigravity-ide\brain\31444b36-40c1-4539-94a7-d17afc79d3a6")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
if CURRENT_ARTIFACTS_DIR.exists():
    (CURRENT_ARTIFACTS_DIR / "plots").mkdir(parents=True, exist_ok=True)

# Clean out previous plots in PLOTS_DIR to prevent obsolete files from lingering
for old_f in PLOTS_DIR.glob("sanity_plot_*.png"):
    old_f.unlink()

sim = ScenarioSimulator()

# The 10 authoritative scenarios: 3 EASY, 3 MEDIUM, 4 HARD
# Plots 02 and 03 replaced with high-quality EASY candidates:
# - Plot 02: scenario_00026 (seed 26) with smooth saturating offset drift reaching severity 0.701
# - Plot 03: scenario_00027 (seed 27) with 600-step (10h) 5.53x high noise episode and full baseline recovery
selected_seeds = [1, 26, 27, 6, 92, 9, 57, 14, 2, 13]

authoritative_records = []

for idx, seed in enumerate(selected_seeds, 1):
    scen_id = f"scenario_{seed:05d}"
    scen = sim.simulate_scenario(scenario_id=scen_id, seed=seed)
    tier, profile = classify_scenario_stratum(scen)
    rec = scen.to_scenario_record()
    unit_dict = json.loads(rec["unit_scale"])
    scale_type = unit_dict.get("scale_type", "unknown")
    mult = float(unit_dict.get("multiplier", 1.0))
    offset = float(unit_dict.get("offset", 0.0))
    
    df_obs = scen.to_observable_dataframe(stride=1)
    df_sens = scen.to_ground_truth_sensors_dataframe(stride=1)
    df_ev = scen.to_ground_truth_events_dataframe(stride=1)
    df_ep_fault = scen.to_fault_episodes_dataframe()
    df_ep_event = scen.to_event_episodes_dataframe()
    
    # Assert internal consistency
    assert len(df_obs) == scen.total_sensors * 1920
    assert len(df_sens) == scen.total_sensors * 1920
    
    fault_summary_list = []
    for _, f_row in df_ep_fault.iterrows():
        fault_summary_list.append(
            f"{f_row['sensor_id']} ({f_row['fault_type']} @ [{f_row['start_step']}, {f_row['end_step']}], "
            f"sev={f_row['max_severity']:.2f})"
        )
    fault_summary_str = "; ".join(fault_summary_list) if fault_summary_list else "None (Healthy)"
    
    event_summary_list = []
    for _, e_row in df_ep_event.iterrows():
        event_summary_list.append(
            f"{e_row['event_id']} ({e_row['event_type']} @ [{e_row['start_step']}, {e_row['end_step']}], "
            f"mag={e_row['magnitude_celsius']:+.2f}°C, {e_row['scope']})"
        )
    event_summary_str = "; ".join(event_summary_list) if event_summary_list else "None"
    
    group_sizes_str = ", ".join([f"{gid}: {len(sids)}" for gid, sids in sorted(scen.groups.items())])
    
    authoritative_records.append({
        "plot_num": idx,
        "scenario_id": scen_id,
        "seed": seed,
        "difficulty": tier.upper(),
        "topology": scen.topology_name,
        "group_count": len(scen.groups),
        "group_sizes": group_sizes_str,
        "sensor_count": scen.total_sensors,
        "sensor_ids": ", ".join(scen.sensor_ids),
        "unit_scale_type": scale_type,
        "multiplier": mult,
        "offset": offset,
        "fault_profile": profile,
        "hard_mechanism": scen.hard_mechanism or "none",
        "faulty_sensor_count": len(scen.faulty_sensors),
        "fault_episodes_count": len(df_ep_fault),
        "fault_summary": fault_summary_str,
        "event_episodes_count": len(df_ep_event),
        "event_summary": event_summary_str,
    })
    
    # Generate 3-panel figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    
    unit_label = f"Readings ({scale_type})"
    if scale_type == "celsius":
        unit_label = "Readings (°C)"
    elif scale_type == "fahrenheit":
        unit_label = "Readings (°F)"
    elif scale_type == "affine":
        unit_label = f"Affine Readings ({mult:.2f}x + {offset:.2f})"
        
    fig.suptitle(
        f"Sanity Plot {idx:02d}: {scen_id} (Seed {seed}) | Tier: {tier.upper()} | Profile: {profile} | Scale: {scale_type}\n"
        f"Topology: {scen.topology_name} ({scen.total_sensors} sensors, {len(scen.groups)} groups) | "
        f"Faults: {len(df_ep_fault)} episodes ({len(scen.faulty_sensors)} sensors) | "
        f"Events: {len(df_ep_event)} episodes",
        fontsize=12, fontweight='bold', y=0.98
    )
    
    # -------------------------------------------------------------
    # Panel 0: Raw Observable Sensor Readings
    # -------------------------------------------------------------
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, scen.total_sensors)))
    for s_idx, sid in enumerate(scen.sensor_ids):
        s_obs = df_obs[df_obs["sensor_id"] == sid].sort_values("timestamp")
        is_faulty = sid in scen.faulty_sensors
        line_style = '-' if not is_faulty else '--'
        alpha = 0.85 if is_faulty else 0.55
        lw = 1.4 if is_faulty else 1.0
        label_suffix = " (FAULTY)" if is_faulty else ""
        ax0.plot(s_obs["timestamp"], s_obs["value"], label=f"{sid}{label_suffix}", 
                 color=colors[s_idx % 10], linestyle=line_style, alpha=alpha, lw=lw)
        
    ax0.axvspan(0, 480, color='gray', alpha=0.12, label="Warmup Window (t < 480 min)")
    ax0.axvline(480, color='gray', linestyle='--', lw=1.2, alpha=0.8)
    ax0.set_ylabel(unit_label, fontsize=10, fontweight='semibold')
    ax0.grid(True, linestyle=':', alpha=0.6)
    ax0.legend(loc='upper right', fontsize=8, ncol=max(1, min(4, scen.total_sensors // 2)), framealpha=0.9)
    
    # -------------------------------------------------------------
    # Panel 1: Fault Severity & Ground Truth Injected Fault Bias
    # -------------------------------------------------------------
    if not df_ep_fault.empty:
        for s_idx, sid in enumerate(scen.sensor_ids):
            s_gt = df_sens[df_sens["sensor_id"] == sid].sort_values("timestamp")
            if s_gt["fault_severity"].max() > 0:
                ax1.plot(s_gt["timestamp"], s_gt["fault_severity"], 
                         label=f"{sid} severity (max={s_gt['fault_severity'].max():.2f})", 
                         color=colors[s_idx % 10], lw=1.5)
                # Shade untrusted intervals
                untrusted = s_gt["sensor_untrusted"].values
                ax1.fill_between(s_gt["timestamp"], 0, untrusted * 0.1, color=colors[s_idx % 10], alpha=0.2, step='post')
                
        # Annotate fault episode spans
        for _, f_row in df_ep_fault.iterrows():
            ax1.axvspan(f_row["start_step"], f_row["end_step"], color='red', alpha=0.08)
            ax1.text(f_row["start_step"], 0.85, f" {f_row['sensor_id']}: {f_row['fault_type']}", 
                     fontsize=8, color='darkred', rotation=0, verticalalignment='bottom')
    else:
        ax1.text(0.5, 0.5, "Healthy Scenario - Zero Sensor Faults Injected", 
                 transform=ax1.transAxes, ha='center', va='center', fontsize=11, color='green', fontweight='bold')
                 
    ax1.axvspan(0, 480, color='gray', alpha=0.12)
    ax1.axvline(480, color='gray', linestyle='--', lw=1.2, alpha=0.8)
    ax1.set_ylabel("Fault Severity [0, 1]", fontsize=10, fontweight='semibold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle=':', alpha=0.6)
    if not df_ep_fault.empty:
        ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
    # -------------------------------------------------------------
    # Panel 2: Ground Truth Environment & Genuine Event Activity
    # -------------------------------------------------------------
    env_ts = scen.environment.timestamps_minutes
    env_vals = scen.environment.values
    ax2.plot(env_ts, env_vals, color='navy', lw=1.2, label="Physical Environment E(t) (°C)")
    
    if not df_ev.empty:
        for gid in sorted(scen.groups.keys()):
            ev_g = df_ev[df_ev["group_id"] == gid].sort_values("timestamp")
            ax2.plot(ev_g["timestamp"], ev_g["true_group_value"], linestyle=':', lw=1.4, label=f"True {gid} Value (°C)")
            
            # Shade genuine event intervals
            gen_active = ev_g["genuine_event"].values
            if gen_active.any():
                ax2.fill_between(ev_g["timestamp"], ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else -10, 
                                 ax2.get_ylim()[1] if ax2.get_ylim()[1] != 1 else 50,
                                 where=gen_active.astype(bool), color='green', alpha=0.12, step='post')
                                 
        for _, e_row in df_ep_event.iterrows():
            ax2.axvspan(e_row["start_step"], e_row["end_step"], color='green', alpha=0.06)
            ax2.text(e_row["start_step"], ax2.get_ylim()[1] * 0.85 if ax2.get_ylim()[1] > 0 else 20, 
                     f" {e_row['event_id']}: {e_row['event_type']} ({e_row['magnitude_celsius']:+.1f}°C)", 
                     fontsize=8, color='darkgreen')
                     
    ax2.axvspan(0, 480, color='gray', alpha=0.12)
    ax2.axvline(480, color='gray', linestyle='--', lw=1.2, alpha=0.8)
    ax2.set_xlabel("Elapsed Time (physical minutes)", fontsize=10, fontweight='semibold')
    ax2.set_ylabel("True Physical Temp (°C)", fontsize=10, fontweight='semibold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    out_file = PLOTS_DIR / f"sanity_plot_{idx:02d}_{scen_id}_{tier}_{scale_type}.png"
    fig.savefig(out_file, dpi=140)
    if CURRENT_ARTIFACTS_DIR.exists():
        shutil.copyfile(out_file, CURRENT_ARTIFACTS_DIR / "plots" / out_file.name)
        shutil.copyfile(out_file, CURRENT_ARTIFACTS_DIR / out_file.name)
    plt.close(fig)
    print(f"Generated Plot {idx:02d}: {out_file.name}")

# Save authoritative summary CSV
df_auth = pd.DataFrame(authoritative_records)
df_auth.to_csv(RUNS_DIR / "authoritative_10_scenarios_summary.csv", index=False)
print(f"Authoritative summary saved to {RUNS_DIR / 'authoritative_10_scenarios_summary.csv'}")
