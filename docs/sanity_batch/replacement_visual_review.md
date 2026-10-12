# P1.11 Replacement Plots Direct Visual Review

This artifact provides direct visual inspection and authoritative programmatic verification for the two replacement scenarios selected from the 100-scenario sanity batch pool:
- **Plot 02:** `scenario_00026` (Seed 26, EASY) — Saturating Offset-Drift Progression
- **Plot 03:** `scenario_00027` (Seed 27, EASY) — 10-Hour Sustained High Noise & Full Baseline Recovery

---

## 1. Plot 02: `scenario_00026` (EASY Tier, Random Affine)

### Visual Rendering
![Plot 02: scenario_00026 (Saturating Offset Drift)](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_02_scenario_00026_easy_affine.png)
*Local file path:* [`runs/sanity_batch_100/plots/sanity_plot_02_scenario_00026_easy_affine.png`](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_02_scenario_00026_easy_affine.png)

### Why this Replaces `scenario_00019`:
- In `scenario_00019`, stepped drift began at step 1881 with a 661-step ramp window, which was truncated at scenario step 1920 before reaching the first discrete step. As a result, injected bias and ground-truth severity remained at exactly $0.00$.
- In `scenario_00026`, `sensor_05` undergoes an observable **saturating offset drift** starting at step 1297 and progressing across 623 consecutive steps through step 1920. Severity increases smoothly and monotonically from $0.002$ to **$0.701$**, with a $-14.12\,\sigma_k$ final bias ($-0.856^\circ\text{C}$ in canonical temperature, corresponding to $-1.29$ affine units).

### Programmatic Scenario Contract Verification:
- **Scenario ID & Seed:** `scenario_00026` (Seed 26)
- **Difficulty Tier:** `EASY` (classified profile: `single_sensor_offset_drift`)
- **Topology & Sensor Count:** `standard`, 6 sensors (`sensor_00` through `sensor_05` in `group_00`)
- **Unit Scaling:** Random Affine ($a = 1.511527$, $b = 0.207778$)
- **Fault Episode:**
  - Sensor ID: `sensor_05`
  - Fault Type: `offset_drift`
  - Start / End Step: $[1297, 1920]$ (active for 623 minutes)
  - Sampled Duration: 882 minutes (ramp: 680 steps, plateau: 202 steps)
  - Shape: `saturating` quadratic curve ($u \mapsto B(1 - (1-u)^2)$)
  - Max Severity: $0.7009$ (untrusted indicator $= 1$ on all 623 active steps)
- **Genuine Environmental Events:**
  - `event_000`: Ramp @ $[817, 1093]$ min, magnitude $+1.94^\circ\text{C}$, global scope
  - `event_001`: Ramp @ $[1093, 1351]$ min, magnitude $+4.94^\circ\text{C}$, global scope
  - `event_002`: Ramp @ $[1745, 1920]$ min, magnitude $-1.43^\circ\text{C}$, global scope

---

## 2. Plot 03: `scenario_00027` (EASY Tier, Random Affine)

### Visual Rendering
![Plot 03: scenario_00027 (10-Hour High Noise & Recovery)](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_03_scenario_00027_easy_affine.png)
*Local file path:* [`runs/sanity_batch_100/plots/sanity_plot_03_scenario_00027_easy_affine.png`](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_03_scenario_00027_easy_affine.png)

### Why this Replaces `scenario_00010`:
- In `scenario_00010`, high noise only started at step 1903, lasting a mere 17 minutes before scenario truncation.
- In `scenario_00027`, `sensor_01` undergoes continuous high noise ($5.53\times \sigma_k$, severity $0.504$) starting at step 509 (right after the 480-minute warmup) and persisting for **600 minutes (10 full hours)** until step 1109.
- At step 1109, the fault ceases cleanly. Over the remaining 811 minutes ($[1109, 1920]$), `sensor_01` returns to normal healthy behavior, tracking the other 4 sensors with identical baseline noise and zero untrusted flags.

### Programmatic Scenario Contract Verification:
- **Scenario ID & Seed:** `scenario_00027` (Seed 27)
- **Difficulty Tier:** `EASY` (classified profile: `single_sensor_high_noise`)
- **Topology & Sensor Count:** `standard`, 5 sensors (`sensor_00` through `sensor_04` in `group_00`)
- **Unit Scaling:** Random Affine ($a = 1.525377$, $b = -6.448873$)
- **Fault Episode:**
  - Sensor ID: `sensor_01`
  - Fault Type: `high_noise`
  - Start / End Step: $[509, 1109]$ (exactly 600 minutes / 10 hours duration)
  - Noise Multiplier: $5.5336\times \sigma_k$ (continuous mode, `is_bursty = False`)
  - Max Severity: $0.5037$ (untrusted indicator $= 1$ during $[509, 1109]$, strictly $0$ elsewhere)
  - Baseline Recovery: In steps $[1109, 1920]$, fault severity is strictly $0.000$ and untrusted flag is $0$.
- **Genuine Environmental Events:**
  - `event_000`: Step with Exponential Recovery @ $[931, 996]$ min, magnitude $+1.18^\circ\text{C}$, global scope

---

## 3. Tier Distribution & Programmatic Audit Confirmation

The overall sanity selection maintains the approved tier distribution:
- **3 EASY Scenarios:** Plot 01 (`scenario_00001`), Plot 02 (`scenario_00026`), Plot 03 (`scenario_00027`)
- **3 MEDIUM Scenarios:** Plot 04 (`scenario_00006`), Plot 05 (`scenario_00092`), Plot 06 (`scenario_00009`)
- **4 HARD Scenarios:** Plot 07 (`scenario_00057`), Plot 08 (`scenario_00014`), Plot 09 (`scenario_00002`), Plot 10 (`scenario_00013`)

All 10 scenarios pass all automated assertions in [`scratch/validate_sanity_plots.py`](file:///d:/CODE_APEX_HACKATHON/scratch/validate_sanity_plots.py) and the 163-test pytest suite.
