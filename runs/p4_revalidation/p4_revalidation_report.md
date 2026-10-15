# Phase P4 Revalidation Report: Trust Before Alarm
**Date:** 2026-10-15  
**Execution Context:** Generator A Full Revalidation against Corrected P2 Feature Engine  
**Working Directory:** `D:\CODE_APEX_HACKATHON`  
**Base Commit:** `1efdc79` (`feat(features): enforce strict Appendix A.8 MAD conformance for sigma-hat and reconcile inventory`)  
**Current HEAD Commit:** `2723f7a` (`fix(alarm): ensure strict affine unit equivariance, reconcile evaluation counts, and clarify matching rule`)  
**Historical Tag:** `p2-fixed` preserved at `8dbfcd2` (not relied upon or moved)  

---

## Executive Summary & Status Recommendation

| Component | Target / Acceptance Criteria | Revalidation Outcome | Status Recommendation |
|---|---|---|---|
| **P4.1 Single-Feature Separability** | Clean feature AUC $\le 0.85$ on `sensor_untrusted`; per-fault OvR $> 0.85$ justified physically; zero illicit leakage | **Max Untrusted AUC = 0.6694**; 0 features flagged on untrusted; 16 flagged on fault OvR (all physically justified); 0 illicit features | **PASS** |
| **P4.2 Label-Shuffle Null Control** | Val AUC in $[0.4700, 0.5300]$ across 5 seeds; Recall @ budget $\le 0.05$ | **Mean Val AUC = 0.4671 $\pm$ 0.0423** (Range: $[0.4105, 0.5238]$; 2 passed, 3 failed); **Mean Recall @ Budget = 0.0253 $\le 0.05$** (All 5 passed) | **FAIL (Strict AUC)** |
| **P4.2 Canary Leak Test** | Planted canary caught with AUC $\ge 0.95$; Clean matrix max feature $< 0.85$ | **Canary AUC = 1.0000** (caught); **Clean Matrix Max Feature = 0.6694** ($< 0.85$) | **PASS** |
| **P4.3 First LightGBM Benchmark** | Quick multiclass GBDT benchmark (8 classes); meaningful lift over majority baseline; zero test split access | **Accuracy = 75.82%**, **Macro-F1 = 0.3247**, **Macro-AUC = 0.8595**, **Log Loss = 0.8950** (vs Majority Baseline F1 = 0.1223); 0 test accesses | **PASS (Benchmark Only)** |
| **P4.4 Consolidated Reporting** | Honest audit reporting without concealing failed controls or altering criteria | Revalidation report completed; failures explicitly documented; no criteria altered | **COMPLETED** |

> **Stop Condition & Gate Recommendation:**  
> - P4.1 and P4.3 successfully pass with improved metrics under the corrected P2 feature engine.  
> - P4.2 strictly fails the official AUC interval $[0.4700, 0.5300]$ (mean 0.4671; 3 of 5 seeds fail).  
> - In accordance with project instructions, **Phase P5 must NOT begin** until the user reviews and provides resolution on the P4.2 null-control AUC failure.

---

## 1. Starting State & Environment Verification

### 1.1 Git History and Feature-Engine Identity
- **Branch:** `master`
- **Current HEAD Commit:** `2723f7a`
- **Feature Engine Implementation:** Imports directly from `src/features/` which contains the corrected P2 feature engine from commit `1efdc79` and `8dbfcd2`:
  - Strict separation of `sigma_hat` (pure robust noise estimator: $1.4826 \times \text{MAD}(\Delta x) / \sqrt{2}$, 25th percentile of 30-step rolling estimates over 480 steps) from `sigma_eff`.
  - Removal of the incorrect standard-deviation fallback when MAD collapses to zero.
  - Bounded effective scale $\sigma_{\text{eff}} = \max(\sigma_{\text{hat}}, \hat{q}/\sqrt{12})$ using causal quantisation step $\hat{q}$ and cross-sensor fallback.
  - Strict unit-free normalization across all 49 sensor features and 11 group features.
- **Historical Tag Integrity:** The `p2-fixed` tag remains intact at `8dbfcd2` and was not checked out, moved, or rewritten.

### 1.2 Dataset Partition Integrity & Isolation
- **Dataset Generation:** Generator A 1,500 scenarios (seeds 1 to 1500).
- **Split Apportionment:**
  - **Train Split:** 900 scenarios (`scenario_0001` to `scenario_0900`), 1,531,872 scored rows ($t \ge 480$).
  - **Validation Split:** 300 scenarios (`scenario_0901` to `scenario_1200`), 512,640 scored rows ($t \ge 480$).
  - **Test Split:** 300 scenarios (`scenario_1201` to `scenario_1500`, 150 full-resolution + 150 downsampled).
- **Isolation Verification:**
  - Scenario IDs in Train and Validation are 100% disjoint (0 overlapping scenarios).
  - Test split was strictly held out and accessed 0 times across all audits and benchmarks.
  - Model feature matrices contain strictly 0 identifiers (`scenario_id`, `sensor_id`, `group_id`, `variable_type`), 0 timestamps, 0 raw measurement values, and 0 ground-truth columns.

---

## 2. Revalidation P4.1: Single-Feature Separability Audit

- **Evaluation Dataset:** Validation split (512,640 scored rows, 300 scenarios).
- **Features Evaluated:** All 49 causal, unit-free sensor features.
- **Threshold:** AUC $> 0.85$ (Appendix A.14 / `config/eval.yaml` investigation trigger).

### 2.1 Target: `sensor_untrusted` (General Sensor Health)
- **Flagged Features (AUC $> 0.85$):** **0 of 49 features** (0.00%).
- **Top 5 Features by Untrusted Separability:**
  1. `gap_absmax_m`: AUC = **0.6694**, PR-AUC = 0.0850 (Direction: `+`)
  2. `absdiff_max_m`: AUC = **0.6295**, PR-AUC = 0.1195 (Direction: `+`)
  3. `n_spikes_m`: AUC = **0.6180**, PR-AUC = 0.0920 (Direction: `+`)
  4. `frac_missing_l`: AUC = **0.6098**, PR-AUC = 0.1804 (Direction: `+`)
  5. `noise_ratio_l`: AUC = **0.6030**, PR-AUC = 0.0994 (Direction: `+`)

### 2.2 Target: Per-Fault One-vs-Rest (OvR) Flags
Exactly **16 features** exceed AUC 0.85 against specific fault classes. Every flagged feature is physically necessary and directly corresponds to the known physics of that fault:

| Feature Name | Target OvR Class | Max AUC | Direction | Physical Domain Justification |
|---|---|---|---|---|
| `repeat_run_len` | `ovr_frozen` | **0.9717** | `+` | Direct physical signature of frozen sensor reading repetition |
| `longest_repeat_m` | `ovr_frozen` | **0.9668** | `+` | Direct physical signature of frozen sensor reading repetition |
| `frac_zero_diff_m` | `ovr_frozen` | **0.9562** | `+` | Consecutive differences collapse to exactly zero during frozen fault |
| `absdiff_max_s` | `ovr_spike` | **0.9555** | `+` | Transient impulse produces extreme short-window difference |
| `absdiff_max_m` | `ovr_high_noise` | **0.9509** | `+` | High measurement noise elevates maximum difference over medium window |
| `n_spikes_m` | `ovr_high_noise` | **0.9344** | `+` | Elevated noise causes repeated threshold crossings counted as spikes |
| `noise_ratio_m` | `ovr_frozen` | **0.9172** | `-` | High-frequency noise collapses to zero when sensor freezes |
| `range_m` | `ovr_spike` | **0.9163** | `+` | Spike magnitude extends the dynamic range over medium window |
| `gap_absmax_m` | `ovr_spike` | **0.9111** | `+` | Single-sensor spike creates large instantaneous deviation from peer group |
| `kurtosis_m` | `ovr_spike` | **0.9105** | `+` | Heavy-tailed distribution caused by isolated extreme impulsive outliers |
| `noise_ratio_s` | `ovr_frozen` | **0.9067** | `-` | Short-window noise collapses to zero when sensor freezes |
| `noise_short_long` | `ovr_frozen` | **0.9032** | `-` | Ratio of short-to-long noise collapses when sensor freezes |
| `unique_ratio_m` | `ovr_frozen` | **0.8927** | `-` | Ratio of unique values drops sharply when values repeat identically |
| `gap_std_m` | `ovr_high_noise` | **0.8913** | `+` | Dispersed readings inflate group gap standard deviation |
| `noise_ratio_l` | `ovr_high_noise` | **0.8839** | `+` | Sustained elevated noise across 120-step long window |
| `noise_growth` | `ovr_high_noise` | **0.8839** | `+` | Ratio of long-window noise to baseline noise captures noise injection |

**P4.1 Finding:** Zero direct leakage, zero metadata proxies, zero illicit signals. All 16 flagged OvR features represent genuine physical fault signatures.

---

## 3. Revalidation P4.2: Label-Shuffle Null Control & Canary Test

### 3.1 Null-Control Protocol & Dataset Alignment
- **Training Dataset:** 300,000 rows sampled uniformly at random across all 900 training scenarios using `DataFrame.sample(n=300000, random_state=42)`.
- **Validation Dataset:** All 512,640 scored rows across all 300 validation scenarios ($t \ge 480$).
- **Label Shuffling:** Training labels `sensor_untrusted` permuted using `numpy.random.default_rng(seed)`. Validation labels strictly untouched.
- **Class Count Preservation:**
  - Train positives before shuffle: **12,427 / 300,000** (4.1423%).
  - Train positives after shuffle: **12,427 / 300,000** (4.1423%) (exact match).
  - Validation positives: **20,368 / 512,640** (3.9732%) (completely intact).
- **Model Configuration:** Default LightGBM binary classifier (`n_estimators=100`, `learning_rate=0.05`, `num_leaves=31`, `random_state=seed`, `class_weight='balanced'`).

### 3.2 Official Results Across Seeds 20001–20005

| Seed | Validation AUC | Target $[0.4700, 0.5300]$ | AUC Status | Val PR-AUC | Recall @ Budget ($\le 0.05$) | Recall Status | Overall Seed Status |
|---|---|---|---|---|---|---|---|
| **20001** | **0.5238** | $[0.4700, 0.5300]$ | **PASS** | 0.0542 | **0.0224** | **PASS** | **PASS** |
| **20002** | **0.4105** | $[0.4700, 0.5300]$ | **FAIL** | 0.0403 | **0.0214** | **PASS** | **FAIL** |
| **20003** | **0.5074** | $[0.4700, 0.5300]$ | **PASS** | 0.0534 | **0.0271** | **PASS** | **PASS** |
| **20004** | **0.4393** | $[0.4700, 0.5300]$ | **FAIL** | 0.0441 | **0.0267** | **PASS** | **FAIL** |
| **20005** | **0.4545** | $[0.4700, 0.5300]$ | **FAIL** | 0.0457 | **0.0289** | **PASS** | **FAIL** |

- **Mean Validation AUC:** **0.4671 $\pm$ 0.0423** (Range: $[0.4105, 0.5238]$).
- **AUC Acceptance Criterion:** **FAILED** (Mean 0.4671 is outside $[0.4700, 0.5300]$; 3 of 5 seeds failed).
- **Recall at False-Alarm Budget Criterion:** **PASSED** (Mean 0.0253 $\le 0.05$; all 5 seeds pass).

### 3.3 Deliberate Canary Leak Test
- **Planted Canary Feature:** Synthetic copy of `sensor_untrusted` injected into training and validation feature matrices.
- **Canary Test Result:** Planted canary was detected with AUC = **1.0000** (Requirement: $\ge 0.95$ $\rightarrow$ **PASSED**).
- **Clean Feature Matrix Check:** Maximum clean feature AUC = **0.6694** (Requirement: $< 0.85$ $\rightarrow$ **PASSED**).

### 3.4 Investigation of Sub-0.50 AUC Behavior
- **Evidence from Pure Noise Experiments:** When LightGBM is trained on synthetic independent Gaussian noise features with shuffled labels, it yields validation AUC = **0.5027**, confirming that LightGBM's split selection algorithm is intrinsically unbiased under the null hypothesis.
- **Mechanism of Ranking Inversion:** On real physical features, rare extreme feature values (e.g., `repeat_run_len > 10`, `n_spikes_m > 0`) represent $< 1\%$ of all observations. In clean unshuffled training data, these rare bins are almost 100% positive. However, when labels are randomly permuted under a 96% negative / 4% positive class imbalance, finite leaf splits on rare bins frequently sample positive fractions below the dataset average (often 0% to 2%). The tree leaves therefore predict *lower* probabilities for extreme feature values. In the validation set (where true labels are preserved), real faults concentrate heavily in these exact extreme bins. Consequently, true faults receive lower predicted scores than normal background data, systematically inverting the ranking and yielding AUC $< 0.50$.
- **Formal Status:** While this mechanism is supported by empirical diagnostics, per the official project specification (`eval.yaml`), this experiment must strictly be recorded as **FAILED** on AUC tolerance.

---

## 4. Revalidation P4.3: First Multiclass LightGBM Benchmark

### 4.1 Benchmark Protocol
- **Model:** Multiclass GBDT (`num_class=8`, `n_estimators=100`, `learning_rate=0.05`, `num_leaves=31`, `class_weight='balanced'`).
- **Target Mapping:** 8 mutually exclusive classes (0: `normal`, 1: `spike`, 2: `offset_drift`, 3: `gain_error`, 4: `frozen`, 5: `high_noise`, 6: `dropout`, 7: `slow_degradation`).
- **Data Splits:** Trained on 300,000 sampled training rows; evaluated on all 512,640 scored validation rows.

### 4.2 Benchmark Results vs Majority Baseline

| Metric | Trivial Majority Baseline | First LightGBM Benchmark (Corrected P2) | Absolute Lift |
|---|---|---|---|
| **Overall Accuracy** | **95.81%** | **75.82%** | -19.99% (Due to balanced weighting) |
| **Validation Log Loss** | N/A | **0.8950** | — |
| **Macro Precision** | 12.00% | **26.34%** | +14.34% |
| **Macro Recall** | 12.50% | **63.32%** | +50.82% |
| **Macro F1-Score** | **0.1223** | **0.3247** | **+0.2024 (+165.5% relative)** |
| **Weighted F1-Score** | 0.9410 | **0.8357** | -0.1053 |
| **Macro OvR ROC-AUC** | 0.5000 | **0.8595** | **+0.3595 (+71.9% relative)** |

### 4.3 Per-Class Performance Breakdown

| Class ID | Class Name | Validation Support | Class Prevalence | Precision | Recall | F1-Score | OvR ROC-AUC |
|---|---|---|---|---|---|---|---|
| 0 | `normal` | 491,138 | 95.8056% | 98.70% | 76.98% | **0.8650** | 0.8396 |
| 1 | `spike` | 331 | 0.0646% | 13.73% | 67.67% | **0.2282** | 0.9253 |
| 2 | `offset_drift` | 5,639 | 1.1000% | 4.09% | 34.62% | **0.0732** | 0.7370 |
| 3 | `gain_error` | 6,278 | 1.2246% | 4.90% | 33.00% | **0.0854** | 0.7401 |
| 4 | `frozen` | 667 | 0.1301% | 41.35% | 92.05% | **0.5706** | 0.9895 |
| 5 | `high_noise` | 2,142 | 0.4178% | 20.18% | 83.01% | **0.3247** | 0.9766 |
| 6 | `dropout` | 2,689 | 0.5245% | 11.28% | 47.04% | **0.1819** | 0.7593 |
| 7 | `slow_degradation` | 3,756 | 0.7327% | 16.49% | 72.15% | **0.2685** | 0.9089 |

### 4.4 Validation Confusion Matrix
```text
Rows: True Fault Class, Columns: Predicted Fault Class
                     normal    spike  offset_  gain_er   frozen  high_no  dropout  slow_de
            normal   378090     1257    43981    38414      824     6335     9382    12855
             spike       13      224        8       15        0       43       10       18
      offset_drift     1718       43     1952     1295       18      268      154      191
        gain_error     1859       37     1372     2072       21      178      324      415
            frozen       20        7        7        6      614        2        8        3
        high_noise      119       31       76       34        0     1778       30       74
           dropout      806       16      197      191        4       46     1265      164
  slow_degradation      462       17      126      233        4      161       43     2710
```

### 4.5 Benchmark Observations & Identified Limitations
1. **Strong Separation on Clear Physical Signatures:** Frozen sensors (Recall 92.05%, AUC 0.9895) and High Noise (Recall 83.01%, AUC 0.9766) are strongly separated even by this un-tuned preliminary model.
2. **Subtle Bias Fault Confusion:** `offset_drift` and `gain_error` exhibit mutual confusion and false alarms against normal drift, reflecting the fundamental difficulty of detecting subtle smooth deviations without dedicated group baseline modeling.
3. **Multi-Fault Co-occurrence Limitation:** 2,074 scored validation rows (~0.40%) contain multiple concurrent faults. In multiclass modeling, one fault must be arbitrarily chosen. This confirms the architectural decision in Phase P5 to move to Head B (7 separate binary models).

---

## 5. Quantitative Comparison: Historical P4 vs Corrected P2 Revalidation

| Metric / Experiment Item | Historical P4 (Pre-Correction, commit `9e1e446`) | Revalidation P4 (Corrected P2, commit `2723f7a`) | Impact of P2 Feature Correction |
|---|---|---|---|
| **Feature Engine Purity** | MAD with std fallback; unseparated $\sigma_{\text{eff}}$ | Pure MAD $\sigma_{\text{hat}}$; causal quantisation floor $\sigma_{\text{eff}}$ | Conforms to Appendix A.8; strict unit invariance |
| **Max Untrusted Separability AUC** | 0.6893 (`gap_absmax_m`) | **0.6694 (`gap_absmax_m`)** | Separability remains well below 0.85 threshold |
| **Flagged Untrusted Features (> 0.85)** | 0 features | **0 features** | Clean matrix confirmed |
| **Flagged OvR Fault Features (> 0.85)** | 16 features | **16 features** | Identical physical signatures |
| **Null Control Seed 20001 AUC** | 0.5106 (PASS) | **0.5238 (PASS)** | In-spec |
| **Null Control Seed 20002 AUC** | 0.4259 (FAIL) | **0.4105 (FAIL)** | Inverted rank on rare features |
| **Null Control Seed 20003 AUC** | 0.4851 (PASS) | **0.5074 (PASS)** | In-spec |
| **Null Control Seed 20004 AUC** | 0.4294 (FAIL) | **0.4393 (FAIL)** | Inverted rank on rare features |
| **Null Control Seed 20005 AUC** | 0.4566 (FAIL) | **0.4545 (FAIL)** | Inverted rank on rare features |
| **Null Control Mean AUC** | 0.4615 $\pm$ 0.0325 (FAIL) | **0.4671 $\pm$ 0.0423 (FAIL)** | Consistent behavior; officially FAILED |
| **Null Control Mean Recall @ Budget** | 0.0293 (PASS) | **0.0253 (PASS)** | Both comfortably $\le 0.05$ |
| **Canary Leak Detection AUC** | 1.0000 (PASS) | **1.0000 (PASS)** | Perfect canary sensitivity |
| **Multiclass Benchmark Accuracy** | 74.62% | **75.82%** | **+1.20% improvement** |
| **Multiclass Benchmark Log Loss** | 0.9029 | **0.8950** | **-0.0079 improvement** |
| **Multiclass Macro F1-Score** | 0.3110 | **0.3247** | **+0.0137 lift** |
| **Multiclass Macro OvR AUC** | 0.8521 | **0.8595** | **+0.0074 lift** |

### Key Takeaway from Comparison:
The corrected P2 feature engine (removing the artificial standard deviation substitution and introducing the rigorous quantisation-bounded effective scale) provides slightly higher numerical stability and feature consistency, resulting in modest performance gains across all multiclass metrics (+1.2% accuracy, +0.0137 Macro-F1, +0.0074 Macro-AUC, lower log loss). Concurrently, the null-control AUC behaves identically (mean ~0.467, 3 of 5 seeds failing), confirming that the sub-0.50 AUC is an inherent statistical property of tree leaf probability assignment on rare extreme features under heavy class imbalance, rather than an artifact of feature-engine bugs or data leakage.

---

## 6. Verification and Test Suite Execution

### 6.1 Focused P4 Tests
- `tests/test_p4_separability.py`: 3/3 passed (forbidden columns check, directional AUC logic, synthetic separability evaluation).
- `tests/test_p4_leakage_controls.py`: 5/5 passed (null-control chance level, class counts preservation, disjoint scenarios check, strict failure status reporting, canary leak test).
- `tests/test_p4_first_benchmark.py`: 2/2 passed (multiclass label mapping 0..7, benchmark training and evaluation metrics pipeline).
- **Result:** **10 of 10 focused P4 tests passed in 1.50s**.

### 6.2 Full Test Suite
- Full test suite running in `.venv` with `pytest`:
  - Simulator tests: environment, sensors, events, faults, strata, dataset parity.
  - Feature engine tests: windows, sigma_hat conformance, own-signal features, cross-sensor features, group features, causal no-future-data, truth isolation, unit invariance across 10 scenarios.
  - Baselines & Alarm tests: raw alarm, fixed threshold, robust z-score, affine unit invariance.
  - Leakage & benchmark tests: all 10 P4 tests.

---

## 7. Files Changed and Final Working-Tree Status

### 7.1 Files Created / Modified
- `runs/p4_revalidation/p4_revalidation_report.md` (This comprehensive revalidation report).
- `runs/p4_revalidation/p4_leakage_and_benchmark_report.md` (Raw benchmark output report).
- `runs/p4_revalidation/p4_evaluation_summary.json` (Structured JSON containing all raw evaluation metrics, seeds, and confusion matrices).
- `runs/p4_revalidation/single_feature_separability.csv` (Complete 49-feature separability audit results).
- `.gitignore` (Updated to unignore and track `runs/p4_revalidation/` report artifacts).
- `PROGRESS.md` (Updated with current status, P4 revalidation log, and outstanding issues).

### 7.2 Working Tree Status
- Working tree clean (once committed).
- No temporary scratch scripts left in tracked directories.
- Feature cache in `data/features/` regenerated with corrected P2 engine.
- Test split remains completely untouched.

---

## 8. Unresolved Issues and Recommendations

1. **P4.2 Label-Shuffle Null Control AUC Resolution (Outstanding Gate):**
   - **Recommendation:** Do NOT force a pass. Maintain the official status as **FAIL** on strict AUC $[0.4700, 0.5300]$ per `config/eval.yaml`.
   - **Options for Project Leadership / User:**
     - *Option A (Accept with Documented Exemption):* Acknowledge that the sub-0.50 AUC is an empirical consequence of tree leaf splitting on rare physical outliers under 96/4 imbalance (supported by passing pure noise tests, passing recall @ budget, and passing canary test), and grant a formal waiver to proceed to P5.
     - *Option B (Adjust Null Model Architecture):* Evaluate null control using a linear model (e.g., LogisticRegression) or constrain LightGBM min_child_samples to prevent leaf probability inversion on rare outliers.
     - *Option C (Widen Null Acceptance Tolerance):* Formally update `eval.yaml` acceptance interval to $[0.4000, 0.5500]$ for imbalanced GBDT null controls.

2. **P1.12 Clean-Checkout Portability (Outstanding Issue):**
   - Runtime dependencies on local `data_backup_p111/` for fault pairing and scenario split locking remain tracked as an open follow-up.

3. **Phase P5 Readiness:**
   - Feature caches are fully rebuilt and verified against the corrected P2 feature engine.
   - P5 must remain paused until user review of the P4.2 null-control outcome is complete.
