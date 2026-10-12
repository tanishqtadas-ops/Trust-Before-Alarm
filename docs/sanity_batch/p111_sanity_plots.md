# P1.11 Sanity Batch & Pooled Exported-Row Prevalence Report

This report provides the finalized, verified visual inspection plots for the 10 selected scenarios (3 EASY, 3 MEDIUM, 4 HARD) from the 100-scenario sanity batch (seeds 1 to 100), along with the authoritative recomputation of prevalence across the entire 1,500-scenario dataset from the parquet files across all four partitions.

- **Authoritative CSV Summary:** [`runs/sanity_batch_100/authoritative_10_scenarios_summary.csv`](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/authoritative_10_scenarios_summary.csv)
- **Local Plots Directory:** [`runs/sanity_batch_100/plots/`](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/)
- **Validation Script:** [`scratch/validate_sanity_plots.py`](file:///d:/CODE_APEX_HACKATHON/scratch/validate_sanity_plots.py) (All assertions PASSED)
- **Full Prevalence Script:** [`scratch/compute_full_prevalence.py`](file:///d:/CODE_APEX_HACKATHON/scratch/compute_full_prevalence.py) (All assertions PASSED)

---

## 1. Pooled Exported-Row Prevalence (All 4 Partitions)

The full 1,500-scenario dataset was partitioned in P1.9 across four distinct subsets: `train` (900 scenarios, stride 5), `val` (300 scenarios, stride 5), `test_full_resolution` (150 scenarios, stride 1), and `test_downsampled` (150 scenarios, stride 5).

Each table was loaded directly from `data/ground_truth/<partition>/ground_truth_sensors.parquet` and `ground_truth_events.parquet`.

> [!IMPORTANT]
> **Methodological Note on Metric Naming:**
> - The combined full-dataset metrics below represent **pooled exported-row prevalence** across all Parquet records in the storage layer.
> - Because partitions have different sampling strides (stride 1 for `test_full_resolution`, stride 5 for `train`, `val`, and `test_downsampled`), stride-1 scenarios contribute $5\times$ more exported rows per scenario than stride-5 scenarios. Consequently, pooled exported-row prevalence is **not** a uniform physical-time prevalence across the 1,500 scenarios.
> - The early 6–10% untrusted sensor-step heuristic is formally retired.
> - The locked formal specification requirement is the **8–15% full-duration genuine-event prevalence target** (Appendix A.3). The pooled full-duration rate of **12.3254%** (108,905 / 883,584 group rows) and every individual partition rate fall strictly within this 8–15% specification window.

### 1.1 Full-Dataset Pooled Exported-Row Totals (1,500 Scenarios)

| Metric | Scored Window ($t \ge 480$ min) | Full Duration ($t \in [0, 1919]$ min) | Exact Denominator Basis |
|---|:---:|:---:|---|
| **Sensor Untrusted Pooled Prevalence** | **4.3352%** (153,881 / 3,549,600) | **3.5191%** (166,553 / 4,732,800) | Exported Sensor Rows |
| **Group Genuine Event Pooled Prevalence** | **16.4338%** (108,905 / 662,688) | **12.3254%** (108,905 / 883,584) | Exported Group Rows (Target: 8–15%) |
| **Group Events Below Detectability** | **9.6949%** (64,247 / 662,688) | **7.2712%** (64,247 / 883,584) | Exported Group Rows |

> [!NOTE]
> **Exact Denominator Audit:**
> - Full-duration sensor rows sum to exactly **4,732,800** ($2,042,496 + 683,520 + 1,668,480 + 338,304$). Under the timestamp contract ($t \ge 480.0$), exactly 75% of rows fall in the scored window ($3/4$), yielding exactly **3,549,600** scored sensor rows.
> - Full-duration group rows sum to exactly **883,584** ($374,784 + 127,104 + 318,720 + 62,976$). The scored window contains exactly 75%, yielding exactly **662,688** scored group rows.
> - Genuine environmental events are strictly prohibited during warmup ($t < 480$) by Decision 18 (`start_step_min: 480`), meaning the positive numerator ($108,905$) is identical across both windows, while the denominator changes from 883,584 to 662,688.
> - Warmup untrusted sensor steps ($12,672 = 166,553 - 153,881$) stem from the 10% of fault episodes configured to start at step 0 (Decision 28).

### 1.2 Partition-Level Breakdown

| Partition | Scenarios | Stride | Sensor Untrusted (Scored) | Sensor Untrusted (Full) | Genuine Events (Scored) | Genuine Events (Full) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `train` | 900 | 5 | **4.1594%** (63,716 / 1,531,872) | **3.4069%** (69,586 / 2,042,496) | **17.1516%** (48,211 / 281,088) | **12.8637%** (48,211 / 374,784) |
| `val` | 300 | 5 | **3.9732%** (20,368 / 512,640) | **3.2902%** (22,489 / 683,520) | **16.1967%** (15,440 / 95,328) | **12.1475%** (15,440 / 127,104) |
| `test_full_resolution` | 150 | 1 | **4.7157%** (59,011 / 1,251,360) | **3.7693%** (62,890 / 1,668,480) | **15.3020%** (36,578 / 239,040) | **11.4765%** (36,578 / 318,720) |
| `test_downsampled` | 150 | 5 | **4.2510%** (10,786 / 253,728) | **3.4253%** (11,588 / 338,304) | **18.3689%** (8,676 / 47,232) | **13.7767%** (8,676 / 62,976) |

### 1.3 Active Fault Breakdown in Scored Window (Full Dataset)

| Fault Label | Scored Active Steps | Prevalence (% of 3,549,600) |
|---|:---:|:---:|
| `f_gain_error` | 48,431 | 1.3644% |
| `f_offset_drift` | 46,520 | 1.3106% |
| `f_slow_degradation` | 31,816 | 0.8963% |
| `f_dropout` | 25,694 | 0.7239% |
| `f_high_noise` | 16,633 | 0.4686% |
| `f_frozen` | 6,450 | 0.1817% |
| `f_spike` | 2,017 | 0.0568% |

### 1.4 Standalone Sanity Batch Measurements (100 Scenarios, Stride 1, Seeds 1–100)

Kept as a distinct, uniform physical-time result (1,920 minutes per scenario across all 100 scenarios):
- **Sensor Untrusted (Scored):** 35,118 / 865,440 = **4.0578%** (~4.06%)
- **Sensor Untrusted (Full Duration):** 39,238 / 1,153,920 = **3.4004%** (~3.40%)
- **Group Genuine Events (Scored):** 30,725 / 161,280 = **19.0507%** (~19.05%)
- **Group Genuine Events (Full Duration):** 30,725 / 215,040 = **14.2880%** (~14.29%, meets 8–15% target)

---

## 2. Replacement Visual Examples

Per audit instructions, the two weak visual examples from the sanity batch were replaced from the existing 100-scenario pool without altering generator distributions:

1. **Plot 02 Replacement (`scenario_00026`, Seed 26, EASY):**
   - *Previous (`scenario_00019`):* Stepped drift onset at step 1881 with ramp duration 661 steps cut off at step 1920 before step 1 transition, leaving severity at 0.00.
   - *Replacement (`scenario_00026`):* `sensor_05` undergoes saturating offset drift starting at step 1297 through scenario end (step 1920). Severity monotonically and continuously ramps from 0.002 up to **0.701**, producing a clearly observable downwards departure of $-14.12\,\sigma_k$ ($-0.86^\circ\text{C}$ canonical bias, $-1.29$ affine units) from the unaffected group sensors.
2. **Plot 03 Replacement (`scenario_00027`, Seed 27, EASY):**
   - *Previous (`scenario_00010`):* Single-sensor high noise started at step 1903 and lasted only 17 minutes.
   - *Replacement (`scenario_00027`):* `sensor_01` undergoes continuous high noise ($5.53\times \sigma_k$, max severity 0.504) starting at step 509 and persisting for **600 steps (10 hours)** until step 1109, after which it completely recovers and tracks normal healthy baseline through step 1920.

---

## 3. Authoritative 10-Scenario Summary Table

| Plot # | Scenario ID | Seed | Tier | Topology | Sensors | Group Sizes | Unit Scale | Fault Profile | Hard Case | Injected Faults | Genuine Events |
|:---:|---|:---:|:---:|---|:---:|---|---|---|---|---|---|
| **01** | `scenario_00001` | 1 | **EASY** | standard | **7** | `group_00: 7` | Celsius ($a=1.00, b=0.00$) | `healthy` | none | None (Healthy) | 2 events: Ramp @ [1848, 1920] (+1.22°C), Step @ [1108, 1238] (+1.44°C) |
| **02** | `scenario_00026` | 26 | **EASY** | standard | **6** | `group_00: 6` | Affine ($a=1.51, b=0.21$) | `single_sensor_offset_drift` | none | `sensor_05`: offset_drift @ [1297, 1920] (sev=0.70, saturating) | 3 events: Ramp @ 817m (+1.94°C), Ramp @ 1093m (+4.94°C), Ramp @ 1745m (-1.43°C) |
| **03** | `scenario_00027` | 27 | **EASY** | standard | **5** | `group_00: 5` | Affine ($a=1.53, b=-6.45$) | `single_sensor_high_noise` | none | `sensor_01`: high_noise @ [509, 1109] (sev=0.50, $5.53\times \sigma_k$, 10h episode) | 1 event: Step @ [931, 996] (+1.18°C) |
| **04** | `scenario_00006` | 6 | **MEDIUM** | standard | **6** | `group_00: 6` | Fahrenheit ($a=1.80, b=32.00$) | `multi_independent_dropout` | none | `sensor_02`: dropout @ [1787, 1920]; `sensor_04`: dropout @ [1538, 1920], spike @ [1557, 1747] | 2 events: Pulse @ 974m (+4.35°C), Step @ 1379m (+5.75°C) |
| **05** | `scenario_00092` | 92 | **MEDIUM** | standard | **8** | `group_00: 8` | Affine ($a=1.77, b=5.23$) | `single_sensor_dropout` | none | `sensor_06`: dropout @ [1005, 1172] (sev=1.00), frozen @ [1085, 1197] (sev=0.80) | 1 event: Step @ 589m (+0.84°C) |
| **06** | `scenario_00009` | 9 | **MEDIUM** | two_groups | **10** | `group_00: 6, group_01: 4` | Celsius ($a=1.00, b=0.00$) | `single_sensor_frozen` | none | `sensor_08`: frozen @ [1682, 1763] (sev=0.80) | 1 event: Step @ 1659m (+3.37°C) |
| **07** | `scenario_00057` | 57 | **HARD** | standard | **8** | `group_00: 8` | Fahrenheit ($a=1.80, b=32.00$) | `multi_independent_frozen` | `fault_during_event` | `sensor_03`: frozen @ [697, 1067]; `sensor_07`: slow_degradation @ [629, 1159], spike @ [1091, 1289] | 1 event: Ramp @ [662, 788] (+3.69°C) |
| **08** | `scenario_00014` | 14 | **HARD** | two_groups | **10** | `group_00: 5, group_01: 5` | Fahrenheit ($a=1.80, b=32.00$) | `multi_independent_gain_error` | `subtle_fault` | `sensor_03`: gain_error @ [738, 1254]; `sensor_08`: drift, spike, dropout; `sensor_09`: dropout | None |
| **09** | `scenario_00002` | 2 | **HARD** | standard | **4** | `group_00: 4` | Affine ($a=1.50, b=6.59$) | `healthy` | `local_event` | None (Healthy) | 1 event: Local sustained_shift @ [695, 1920] (-7.96°C) on sensors 00, 02 |
| **10** | `scenario_00013` | 13 | **HARD** | two_groups | **12** | `group_00: 6, group_01: 6` | Fahrenheit ($a=1.80, b=32.00$) | `shared_cause_dropout` | `natural` | `sensor_07`: dropout @ [0, 341], high_noise @ [0, 432]; `sensor_10`: dropout @ [0, 385], high_noise @ [0, 131], gain @ [0, 797] | None |

---

## 4. Visual Inspection Plots (10 Scenarios)

### Plot 01: `scenario_00001` (EASY, Celsius, Healthy Baseline)
![Plot 01: scenario_00001](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_01_scenario_00001_easy_celsius.png)
*Local file:* [sanity_plot_01_scenario_00001_easy_celsius.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_01_scenario_00001_easy_celsius.png)

---

### Plot 02: `scenario_00026` (EASY, Affine, Saturating Offset-Drift Progression) — REPLACEMENT
![Plot 02: scenario_00026](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_02_scenario_00026_easy_affine.png)
*Local file:* [sanity_plot_02_scenario_00026_easy_affine.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_02_scenario_00026_easy_affine.png)

---

### Plot 03: `scenario_00027` (EASY, Affine, 10-Hour High-Noise Episode & Recovery) — REPLACEMENT
![Plot 03: scenario_00027](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_03_scenario_00027_easy_affine.png)
*Local file:* [sanity_plot_03_scenario_00027_easy_affine.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_03_scenario_00027_easy_affine.png)

---

### Plot 04: `scenario_00006` (MEDIUM, Fahrenheit, Multi-Sensor Dropout & Spike)
![Plot 04: scenario_00006](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_04_scenario_00006_medium_fahrenheit.png)
*Local file:* [sanity_plot_04_scenario_00006_medium_fahrenheit.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_04_scenario_00006_medium_fahrenheit.png)

---

### Plot 05: `scenario_00092` (MEDIUM, Affine, Single-Sensor Dropout & Frozen)
![Plot 05: scenario_00092](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_05_scenario_00092_medium_affine.png)
*Local file:* [sanity_plot_05_scenario_00092_medium_affine.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_05_scenario_00092_medium_affine.png)

---

### Plot 06: `scenario_00009` (MEDIUM, Celsius, Two-Group Single-Sensor Frozen)
![Plot 06: scenario_00009](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_06_scenario_00009_medium_celsius.png)
*Local file:* [sanity_plot_06_scenario_00009_medium_celsius.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_06_scenario_00009_medium_celsius.png)

---

### Plot 07: `scenario_00057` (HARD, Fahrenheit, Multi-Fault During Genuine Event)
![Plot 07: scenario_00057](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_07_scenario_00057_hard_fahrenheit.png)
*Local file:* [sanity_plot_07_scenario_00057_hard_fahrenheit.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_07_scenario_00057_hard_fahrenheit.png)

---

### Plot 08: `scenario_00014` (HARD, Fahrenheit, Subtle Gain Error & Multi-Fault)
![Plot 08: scenario_00014](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_08_scenario_00014_hard_fahrenheit.png)
*Local file:* [sanity_plot_08_scenario_00014_hard_fahrenheit.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_08_scenario_00014_hard_fahrenheit.png)

---

### Plot 09: `scenario_00002` (HARD, Affine, Local Event with Zero Sensor Faults)
![Plot 09: scenario_00002](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_09_scenario_00002_hard_affine.png)
*Local file:* [sanity_plot_09_scenario_00002_hard_affine.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_09_scenario_00002_hard_affine.png)

---

### Plot 10: `scenario_00013` (HARD, Fahrenheit, Shared-Cause Pair Active From Step 0)
![Plot 10: scenario_00013](C:/Users/tanis/.gemini/antigravity-ide/brain/31444b36-40c1-4539-94a7-d17afc79d3a6/sanity_plot_10_scenario_00013_hard_fahrenheit.png)
*Local file:* [sanity_plot_10_scenario_00013_hard_fahrenheit.png](file:///d:/CODE_APEX_HACKATHON/runs/sanity_batch_100/plots/sanity_plot_10_scenario_00013_hard_fahrenheit.png)
