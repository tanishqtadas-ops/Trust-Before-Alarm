# Dataset Fault Support, Episode Counts, and Prevalence Report

This document provides the authoritative, cross-checked audit of fault support, scenario distribution, sensor-step counts, and environmental event prevalence across the entire 1,500-scenario dataset. All counts are directly computed from the Parquet tables under `data/ground_truth/` and metadata in `data/ground_truth/scenarios.parquet`.

> [!IMPORTANT]
> **Methodology & Terminology:**
> - **Episode Counts vs Sensor-Step Counts:** Episode counts quantify the discrete injection episodes in `fault_episodes.parquet` and `event_episodes.parquet`. Sensor-step counts quantify individual timestamped rows in `ground_truth_sensors.parquet` and `ground_truth_events.parquet`.
> - **Sampling Strides:** `train`, `val`, and `test_downsampled` are sampled at stride 5 (5-minute steps; 384 steps per sensor). `test_full_resolution` is sampled at stride 1 (1-minute steps; 1,920 steps per sensor). Row counts reflect exported table rows.
> - **Prevalence Denominators:** Full-duration rates use the entire 1,920-minute timeline ($t \in [0, 1919]$). Scored-window rates use only post-warmup timestamps ($t \ge 480.0$ min, exactly 75% of rows).
> - **Rare Classes:** Categories with fewer than 20 examples are explicitly cataloged for audit and review.

---

## 1. Partition Overview & Sampling Strides

| Partition | Scenarios | Sampling Stride | Sensor Rows (Full) | Sensor Rows (Scored) | Group Rows (Full) | Group Rows (Scored) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `train` | 900 | Stride 5 (5 min) | 2,042,496 | 1,531,872 | 374,784 | 281,088 |
| `val` | 300 | Stride 5 (5 min) | 683,520 | 512,640 | 127,104 | 95,328 |
| `test_full_resolution` | 150 | Stride 1 (1 min) | 1,668,480 | 1,251,360 | 318,720 | 239,040 |
| `test_downsampled` | 150 | Stride 5 (5 min) | 338,304 | 253,728 | 62,976 | 47,232 |
| **Total Pooled** | **1,500** | **Mixed (1 & 5)** | **4,732,800** | **3,549,600** | **883,584** | **662,688** |

---

## 2. Fault Episode Counts by Split and Difficulty Tier

Table below shows the number of fault injection episodes for each of the seven fault types, broken down by dataset split (`train`, `val`, `test`) and difficulty tier (`EASY`, `MEDIUM`, `HARD`).

| Fault Type | Split | EASY Episodes | MEDIUM Episodes | HARD Episodes | Total Episodes |
|---|---|:---:|:---:|:---:|:---:|
| `spike` | `train` | 33 | 100 | 76 | **209** |
| `spike` | `val` | 11 | 35 | 36 | **82** |
| `spike` | `test` | 11 | 35 | 31 | **77** |
| `offset_drift` | `train` | 40 | 87 | 80 | **207** |
| `offset_drift` | `val` | 13 | 30 | 20 | **63** |
| `offset_drift` | `test` | 14 | 34 | 23 | **71** |
| `gain_error` | `train` | 31 | 117 | 105 | **253** |
| `gain_error` | `val` | 10 | 35 | 36 | **81** |
| `gain_error` | `test` | 11 | 31 | 31 | **73** |
| `frozen` | `train` | 36 | 59 | 45 | **140** |
| `frozen` | `val` | 12 | 14 | 16 | **42** |
| `frozen` | `test` | 12 | 18 | 14 | **44** |
| `high_noise` | `train` | 37 | 72 | 67 | **176** |
| `high_noise` | `val` | 12 | 26 | 25 | **63** |
| `high_noise` | `test` | 12 | 26 | 21 | **59** |
| `dropout` | `train` | 36 | 102 | 80 | **218** |
| `dropout` | `val` | 12 | 34 | 24 | **70** |
| `dropout` | `test` | 12 | 36 | 28 | **76** |
| `slow_degradation` | `train` | 26 | 62 | 53 | **141** |
| `slow_degradation` | `val` | 9 | 17 | 18 | **44** |
| `slow_degradation` | `test` | 9 | 23 | 17 | **49** |

### 2.1 Distinct Scenarios Containing Each Fault Type

Because scenarios in MEDIUM and HARD tiers can contain multiple faulty sensors or multiple fault episodes, the number of distinct scenarios containing each fault type is reported below:

| Fault Type | Split | EASY Scenarios | MEDIUM Scenarios | HARD Scenarios | Total Scenarios |
|---|---|:---:|:---:|:---:|:---:|
| `spike` | `train` | 33 | 92 | 69 | **194** |
| `spike` | `val` | 11 | 33 | 31 | **75** |
| `spike` | `test` | 11 | 32 | 27 | **70** |
| `offset_drift` | `train` | 40 | 78 | 70 | **188** |
| `offset_drift` | `val` | 13 | 29 | 18 | **60** |
| `offset_drift` | `test` | 14 | 30 | 22 | **66** |
| `gain_error` | `train` | 31 | 104 | 97 | **232** |
| `gain_error` | `val` | 10 | 33 | 33 | **76** |
| `gain_error` | `test` | 11 | 28 | 29 | **68** |
| `frozen` | `train` | 36 | 55 | 41 | **132** |
| `frozen` | `val` | 12 | 13 | 14 | **39** |
| `frozen` | `test` | 12 | 17 | 13 | **42** |
| `high_noise` | `train` | 37 | 69 | 62 | **168** |
| `high_noise` | `val` | 12 | 24 | 22 | **58** |
| `high_noise` | `test` | 12 | 25 | 20 | **57** |
| `dropout` | `train` | 36 | 92 | 74 | **202** |
| `dropout` | `val` | 12 | 31 | 22 | **65** |
| `dropout` | `test` | 12 | 33 | 24 | **69** |
| `slow_degradation` | `train` | 26 | 55 | 46 | **127** |
| `slow_degradation` | `val` | 9 | 16 | 16 | **41** |
| `slow_degradation` | `test` | 9 | 22 | 16 | **47** |

---

## 3. Positive Sensor-Step Counts by Fault Flag & Partition

Sensor-step counts measure active rows in `ground_truth_sensors.parquet`. Stride 1 partitions (`test_full_resolution`) produce $5\times$ more rows per scenario than stride 5 partitions.

### 3.1 Scored Window ($t \ge 480.0$ min)

| Fault Flag | `train` (stride 5) | `val` (stride 5) | `test_full_resolution` (stride 1) | `test_downsampled` (stride 5) | Pooled Total | Pooled Rate (% of 3,549,600) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `f_spike` | 858 | 331 | 658 | 170 | **2,017** | **0.0568%** |
| `f_offset_drift` | 17,807 | 5,680 | 20,523 | 2,510 | **46,520** | **1.3106%** |
| `f_gain_error` | 20,328 | 6,815 | 17,621 | 3,667 | **48,431** | **1.3644%** |
| `f_frozen` | 3,111 | 667 | 2,126 | 546 | **6,450** | **0.1817%** |
| `f_high_noise` | 7,899 | 2,573 | 4,719 | 1,442 | **16,633** | **0.4686%** |
| `f_dropout` | 10,426 | 3,380 | 10,128 | 1,760 | **25,694** | **0.7239%** |
| `f_slow_degradation` | 13,691 | 4,201 | 11,517 | 2,407 | **31,816** | **0.8963%** |
| **`sensor_untrusted`** | **63,716** | **20,368** | **59,011** | **10,786** | **153,881** | **4.3352%** |

### 3.2 Full Duration ($t \in [0, 1919]$ min)

| Fault Flag | `train` (stride 5) | `val` (stride 5) | `test_full_resolution` (stride 1) | `test_downsampled` (stride 5) | Pooled Total | Pooled Rate (% of 4,732,800) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `f_spike` | 949 | 353 | 696 | 188 | **2,186** | **0.0462%** |
| `f_offset_drift` | 19,747 | 6,595 | 21,483 | 2,702 | **50,527** | **1.0676%** |
| `f_gain_error` | 22,197 | 7,101 | 18,581 | 3,859 | **51,738** | **1.0932%** |
| `f_frozen` | 3,327 | 740 | 2,258 | 571 | **6,896** | **0.1457%** |
| `f_high_noise` | 8,709 | 2,828 | 5,282 | 1,640 | **18,459** | **0.3900%** |
| `f_dropout` | 11,404 | 4,007 | 11,147 | 1,930 | **28,488** | **0.6019%** |
| `f_slow_degradation` | 14,657 | 4,517 | 11,772 | 2,468 | **33,414** | **0.7060%** |
| **`sensor_untrusted`** | **69,586** | **22,489** | **62,890** | **11,588** | **166,553** | **3.5191%** |

---

## 4. Genuine Environmental Events & Group-Step Prevalence

Genuine environmental events are injected at the group level and strictly begin after warmup ($t \ge 480$ min, Decision 18). Consequently, positive group-step numerators are identical between scored and full windows, while denominators change.

| Partition | Event Episodes | Genuine Steps (Scored) | Group Rows (Scored) | Scored Rate (%) | Genuine Steps (Full) | Group Rows (Full) | Full-Duration Rate (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `train` | 1,083 | 48,211 | 281,088 | 17.1516% | 48,211 | 374,784 | **12.8637%** |
| `val` | 351 | 15,440 | 95,328 | 16.1967% | 15,440 | 127,104 | **12.1475%** |
| `test_full_resolution` | 168 | 36,578 | 239,040 | 15.3020% | 36,578 | 318,720 | **11.4765%** |
| `test_downsampled` | 168 | 8,676 | 47,232 | 18.3689% | 8,676 | 62,976 | **13.7767%** |
| **Total Pooled** | **1,770** | **108,905** | **662,688** | **16.4338%** | **108,905** | **883,584** | **12.3254%** |

> [!NOTE]
> **Specification Conformance:** The pooled full-duration genuine-event rate is **12.3254%** (108,905 / 883,584 group rows), and each partition's individual rate (`train`: 12.86%, `val`: 12.15%, `test_full`: 11.48%, `test_down`: 13.78%) strictly satisfies the formal **8–15% full-duration genuine-event prevalence requirement** (Appendix A.3).

---

## 5. Rare Fault Classes & Low-Support Categories (< 20 Examples)

The table below catalogs every (fault type, split, difficulty tier) category containing fewer than 20 episode examples. This enables targeted monitoring during model evaluation.

| Fault Type | Split | Difficulty Tier | Episode Count | Distinct Scenarios | Notes & Evaluation Guidance |
|---|:---:|:---:|:---:|:---:|---|
| `spike` | `val` | `EASY` | **11** | 11 | Single-sensor spike profile in val split (EASY tier). |
| `spike` | `test` | `EASY` | **11** | 11 | Single-sensor spike profile in test split (EASY tier). |
| `offset_drift` | `val` | `EASY` | **13** | 13 | Single-sensor offset drift in val split (EASY tier). |
| `offset_drift` | `test` | `EASY` | **14** | 14 | Single-sensor offset drift in test split (EASY tier). |
| `gain_error` | `val` | `EASY` | **10** | 10 | Single-sensor gain error in val split (EASY tier). |
| `gain_error` | `test` | `EASY` | **11** | 11 | Single-sensor gain error in test split (EASY tier). |
| `frozen` | `val` | `EASY` | **12** | 12 | Single-sensor frozen in val split (EASY tier). |
| `frozen` | `val` | `MEDIUM` | **14** | 13 | Frozen episodes across medium scenarios in val split. |
| `frozen` | `val` | `HARD` | **16** | 14 | Frozen episodes in hard scenarios in val split. |
| `frozen` | `test` | `EASY` | **12** | 12 | Single-sensor frozen in test split (EASY tier). |
| `frozen` | `test` | `MEDIUM` | **18** | 17 | Frozen episodes across medium scenarios in test split. |
| `frozen` | `test` | `HARD` | **14** | 13 | Frozen episodes in hard scenarios in test split. |
| `high_noise` | `val` | `EASY` | **12** | 12 | Single-sensor high noise in val split (EASY tier). |
| `high_noise` | `test` | `EASY` | **12** | 12 | Single-sensor high noise in test split (EASY tier). |
| `dropout` | `val` | `EASY` | **12** | 12 | Single-sensor dropout in val split (EASY tier). |
| `dropout` | `test` | `EASY` | **12** | 12 | Single-sensor dropout in test split (EASY tier). |
| `slow_degradation` | `val` | `EASY` | **9** | 9 | Single-sensor slow degradation in val split (EASY tier). |
| `slow_degradation` | `val` | `MEDIUM` | **17** | 16 | Slow degradation episodes across medium scenarios in val split. |
| `slow_degradation` | `val` | `HARD` | **18** | 16 | Slow degradation episodes in hard scenarios in val split. |
| `slow_degradation` | `test` | `EASY` | **9** | 9 | Single-sensor slow degradation in test split (EASY tier). |
| `slow_degradation` | `test` | `HARD` | **17** | 16 | Slow degradation episodes in hard scenarios in test split. |

> [!TIP]
> **Context on Small Counts:** Because the dataset follows a 60/20/20 whole-scenario split (900 train, 300 val, 300 test) across 22 distinct fault profile strata, EASY-tier profiles naturally allocate ~30–40 scenarios to train, ~10–13 to val, and ~10–14 to test. These small counts are an expected mathematical consequence of rigorous stratum partitioning, not generator bugs. No documented specification failure exists solely from small sub-stratum sample sizes.

---

## 6. Reproducibility & PowerShell Verification Commands

Every figure in this report can be re-verified directly in PowerShell using the commands below.

### 6.1 Re-compute Full Prevalence Metrics Across All 4 Partitions
```powershell
.venv\Scripts\python.exe scripts/compute_full_prevalence.py
```

### 6.2 Query Fault Episode Counts by Fault Type and Split
```powershell
.venv\Scripts\python.exe -c "import pandas as pd; df = pd.concat([pd.read_parquet(f"data/ground_truth/{p}/fault_episodes.parquet") for p in ["train","val","test_full_resolution","test_downsampled"]]); sc = pd.read_parquet("data/ground_truth/scenarios.parquet")[["scenario_id","split"]]; df = df.merge(sc, on="scenario_id"); print(df.groupby(["fault_type", "split"]).size().unstack().fillna(0).astype(int))"
```

### 6.3 Query Positive Sensor-Step Counts in Scored Window
```powershell
.venv\Scripts\python.exe -c "import pandas as pd; cols = ["f_spike","f_offset_drift","f_gain_error","f_frozen","f_high_noise","f_dropout","f_slow_degradation","sensor_untrusted"]; parts = ["train","val","test_full_resolution","test_downsampled"]; data = {p: pd.read_parquet(f"data/ground_truth/{p}/ground_truth_sensors.parquet").query("timestamp >= 480")[cols].sum() for p in parts}; print(pd.DataFrame(data).astype(int))"
```

### 6.4 Query Genuine Event Prevalence
```powershell
.venv\Scripts\python.exe -c "import pandas as pd; parts = ["train","val","test_full_resolution","test_downsampled"]; print({p: f"{pd.read_parquet(f""data/ground_truth/{p}/ground_truth_events.parquet"")[""genuine_event""].sum():,} / {len(pd.read_parquet(f""data/ground_truth/{p}/ground_truth_events.parquet"")):,} ({pd.read_parquet(f""data/ground_truth/{p}/ground_truth_events.parquet"")[""genuine_event""].mean()*100:.2f}%)" for p in parts})"
```