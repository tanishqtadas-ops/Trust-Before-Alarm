# Phase P4: Leakage Audits and First LightGBM Benchmark Report
*Generated on: 2026-10-15 13:34:58 UTC*
*Dataset: Generator A (Train: 1,531,872 scored rows, Val: 512,640 scored rows; Test split STRICTLY UNTOUCHED)*

## Executive Summary
- **P4.1 Single-Feature Separability:** 0 features flagged on `sensor_untrusted` (AUC > 0.85); exactly 16 features flagged on per-fault one-vs-rest (OvR) fault-type labels (> 0.85: 8 on `frozen`, 7 on `high_noise`, 1 on `spike`, 0 on other fault classes). All 16 flagged features reflect documented physical domain responses to sensor failure modes (e.g., `repeat_run_len` and `longest_repeat_m` for frozen sensors, `absdiff_max_m` and `n_spikes_m` for high noise, `kurtosis_m` for spikes); zero ground-truth leakage or illicit metadata proxies detected (**PASSED**).
- **P4.2 Label-Shuffle Null Control:**
  - **AUC Tolerance Status:** **FAILED** (Mean Val AUC = 0.4615 +/- 0.0325 outside required tolerance [0.4700, 0.5300]; 2/5 seeds passed, 3/5 seeds failed).
  - **Recall at Budget Status:** **PASSED** (All 5 seeds <= 0.05, mean recall @ budget = 0.0293 <= 0.05).
  - **Alignment & Leakage Audit:** Complete audit confirmed zero data leakage, zero join duplication, 100% disjoint scenario sets (exact 900 train vs 300 val scenarios from the 1,500 total), and exact class count preservation (12,427/300k train sample, 20,368/512k val).
  - **Root Cause Analysis:** Diagnostic experiments identified tree ranking inversion on rare extreme features under 96/4 class imbalance as an empirically supported statistical explanation for sub-0.50 AUC (pure noise features achieve AUC 0.5027; real features invert probabilities on rare physical outliers). However, this explanation is NOT proof that the null control passed. Per the project specification in `eval.yaml`, the P4.2 null control remains strictly **FAILED** on AUC tolerance.
- **P4.2 Canary Leak Test:** Planted canary was successfully caught with AUC = 1.0000 (>= 0.95: **PASSED**). Clean feature matrix contains zero synthetic ground-truth copies (max feature `gap_absmax_m` AUC = 0.6893 < 0.85 and < 0.95: **PASSED**).
- **P4.3 First LightGBM Benchmark:** 8-class GBDT benchmark achieved **74.62% accuracy**, **Macro-F1 0.3110**, and **Macro-AUC 0.8521** on validation set without tuning. Significant lift over trivial majority-class baseline (Macro-F1 0.1223). Test split strictly held out (0 accesses: **PASSED**).

## 1. P4.1 Single-Feature Separability Audit
Threshold: AUC > 0.85 (Appendix A.14 / eval.yaml)

### Threshold Specification Clarification
- `single_feature_auc_threshold: 0.85`: Used as an investigation trigger for clean feature evaluation. Any clean feature exceeding 0.85 must be justified by physical domain necessity.
- `canary_auc_min: 0.95`: Used as the sensitivity pass mark for catching planted ground-truth leakage (canary).
- These thresholds are distinct, intentional, and spec-compliant.

### Audit Summary and Authoritative Reference
- **Total Features Evaluated:** 49 engineered sensor features on 512,640 scored validation rows.
- **Features Flagged on Aggregate `sensor_untrusted` (AUC > 0.85):** **0 features** (Max AUC is `gap_absmax_m` at 0.6893, PR-AUC = 0.1060; well below the 0.85 threshold).
- **Features Flagged on Per-Fault One-vs-Rest Labels (AUC > 0.85):** **16 features** (all corresponding to expected physical dynamics of specific fault types: 8 on `frozen`, 7 on `high_noise`, 1 on `spike`, 0 on `offset_drift`, `gain_error`, `dropout`, or `slow_degradation`).
- **Authoritative Data Source:** The complete audit results for all 49 features across all target labels are recorded in [single_feature_separability.csv](file:///d:/CODE_APEX_HACKATHON/runs/p4_leakage_benchmark/single_feature_separability.csv).

### Top 15 Features by Untrusted Separability (AUC against `sensor_untrusted`)
| Rank | Feature Name | Direction | Untrusted AUC | PR-AUC | Max OvR Fault | Max Fault AUC | Task-Aware Explanation |
|---|---|---|---|---|---|---|---|
| 1 | `gap_absmax_m` | + | 0.6893 | 0.1060 | `high_noise` | 0.9075 | Fault 'high_noise' AUC 0.907 > 0.85 |
| 2 | `absdiff_max_m` | + | 0.6302 | 0.1258 | `high_noise` | 0.9356 | Fault 'high_noise' AUC 0.936 > 0.85 |
| 3 | `gap_std_m` | + | 0.6223 | 0.0963 | `high_noise` | 0.9170 | Fault 'high_noise' AUC 0.917 > 0.85 |
| 4 | `n_spikes_m` | + | 0.6180 | 0.0920 | `high_noise` | 0.9344 | Fault 'high_noise' AUC 0.934 > 0.85 |
| 5 | `frac_missing_l` | + | 0.6098 | 0.1804 | `slow_degradation` | 0.8386 | Nominal separability (<= 0.85) |
| 6 | `absdiff_max_s` | + | 0.6093 | 0.1351 | `frozen` | 0.9503 | Fault 'frozen' AUC 0.950 > 0.85 |
| 7 | `range_m` | + | 0.6088 | 0.1109 | `high_noise` | 0.9224 | Fault 'high_noise' AUC 0.922 > 0.85 |
| 8 | `noise_ratio_l` | + | 0.6016 | 0.0706 | `high_noise` | 0.8744 | Fault 'high_noise' AUC 0.874 > 0.85 |
| 9 | `noise_growth` | + | 0.6016 | 0.0706 | `high_noise` | 0.8744 | Fault 'high_noise' AUC 0.874 > 0.85 |
| 10 | `noise_ratio_m` | + | 0.5981 | 0.0730 | `frozen` | 0.9171 | Fault 'frozen' AUC 0.917 > 0.85 |
| 11 | `gain_r2_l` | - | 0.5934 | 0.0664 | `high_noise` | 0.7738 | Nominal separability (<= 0.85) |
| 12 | `missing_growth` | + | 0.5817 | 0.1440 | `slow_degradation` | 0.7753 | Nominal separability (<= 0.85) |
| 13 | `frac_missing_m` | + | 0.5765 | 0.1061 | `slow_degradation` | 0.7653 | Nominal separability (<= 0.85) |
| 14 | `longest_gap_m` | + | 0.5745 | 0.0934 | `slow_degradation` | 0.7554 | Nominal separability (<= 0.85) |
| 15 | `dev_duration` | - | 0.5703 | 0.0468 | `slow_degradation` | 0.6699 | Nominal separability (<= 0.85) |

### Complete Enumeration of All 16 Features Flagged for Investigation (AUC > 0.85 Against Fault-Type Labels)
The table below enumerates all 16 features from [single_feature_separability.csv](file:///d:/CODE_APEX_HACKATHON/runs/p4_leakage_benchmark/single_feature_separability.csv) that exceed the 0.85 threshold on one-vs-rest fault-type labels, ordered by Max OvR Fault AUC descending. Each feature's associated fault class, exact AUC values (both 4-decimal and full precision), untrusted AUC, direction, and physical domain justification are documented:

| Rank | Feature Name | Associated Fault Class | Target Evaluated | Max Fault AUC | Untrusted AUC | Dir | Assessment & Physical Specification Justification |
|---|---|---|---|---|---|---|---|
| 1 | `repeat_run_len` | `frozen` | `ovr_frozen` | 0.9717 (0.971694) | 0.5204 | - | Consecutive identical value count. Directly measures frozen sensor flatlines. Legitimate physical indicator; retained per spec. |
| 2 | `longest_repeat_m` | `frozen` | `ovr_frozen` | 0.9668 (0.966847) | 0.5507 | - | Max run of identical readings in medium window. Directly detects frozen state. Legitimate physical indicator; retained per spec. |
| 3 | `frac_zero_diff_m` | `frozen` | `ovr_frozen` | 0.9562 (0.956205) | 0.5497 | - | Fraction of successive first-differences equal to 0. Jumps to ~1.0 during sensor freeze. Legitimate physical indicator; retained per spec. |
| 4 | `absdiff_max_s` | `frozen` | `ovr_frozen` | 0.9503 (0.950340) | 0.6093 | + | Max absolute step difference in short window; collapses toward zero on frozen sensor. Legitimate physical indicator; retained per spec. |
| 5 | `absdiff_max_m` | `high_noise` | `ovr_high_noise` | 0.9356 (0.935596) | 0.6302 | + | Max absolute step difference in medium window. Elevated noise injects large point-to-point steps. Legitimate physical indicator; retained per spec. |
| 6 | `n_spikes_m` | `high_noise` | `ovr_high_noise` | 0.9344 (0.934385) | 0.6180 | + | Number of 3-sigma excursions in medium window. High noise frequently triggers threshold trips. Legitimate physical indicator; retained per spec. |
| 7 | `range_m` | `high_noise` | `ovr_high_noise` | 0.9224 (0.922438) | 0.6088 | + | Min-to-max envelope in medium window. High noise directly inflates observed spread. Legitimate physical indicator; retained per spec. |
| 8 | `noise_ratio_m` | `frozen` | `ovr_frozen` | 0.9171 (0.917106) | 0.5981 | + | Ratio of short-term noise to baseline noise. Collapses toward 0 when signal freezes. Legitimate physical indicator; retained per spec. |
| 9 | `gap_std_m` | `high_noise` | `ovr_high_noise` | 0.9170 (0.916971) | 0.6223 | + | Standard deviation of peer residual gap. Noise on target sensor inflates residual variance against peer median. Legitimate physical indicator; retained per spec. |
| 10 | `kurtosis_m` | `spike` | `ovr_spike` | 0.9105 (0.910531) | 0.5416 | + | Fourth standardized moment in medium window. Sharp isolated spikes generate extreme distribution tails. Legitimate physical indicator; retained per spec. |
| 11 | `gap_absmax_m` | `high_noise` | `ovr_high_noise` | 0.9075 (0.907481) | 0.6893 | + | Max absolute peer residual gap in medium window. Noise excursions deviate sharply from unaffected peers. Legitimate physical indicator; retained per spec. |
| 12 | `noise_ratio_s` | `frozen` | `ovr_frozen` | 0.9067 (0.906662) | 0.5653 | + | Short-window noise ratio. Drops to 0 when signal freezes. Legitimate physical indicator; retained per spec. |
| 13 | `noise_short_long` | `frozen` | `ovr_frozen` | 0.9032 (0.903197) | 0.5198 | + | Ratio of short-to-long window variance. Collapses during sustained frozen signal. Legitimate physical indicator; retained per spec. |
| 14 | `unique_ratio_m` | `frozen` | `ovr_frozen` | 0.8927 (0.892737) | 0.5547 | + | Fraction of unique values in medium window. Drops toward 1/N during sensor freeze. Legitimate physical indicator; retained per spec. |
| 15 | `noise_growth` | `high_noise` | `ovr_high_noise` | 0.8744 (0.874383) | 0.6016 | + | Ratio of current to historic noise. Detects sudden onset of noise faults. Legitimate physical indicator; retained per spec. |
| 16 | `noise_ratio_l` | `high_noise` | `ovr_high_noise` | 0.8744 (0.874383) | 0.6016 | + | Long-window noise ratio. Sustained elevation under continuous high noise injection. Legitimate physical indicator; retained per spec. |

## 2. P4.2 Label-Shuffle Null Control & Canary Leak Test

### Official Null-Control Benchmark Results (Seeds 20001 to 20005)
| Seed | Validation AUC | Target (0.50 +/- 0.03) | AUC Status | Val PR-AUC | Recall @ Budget (<= 0.05) | Recall Status | Overall Status |
|---|---|---|---|---|---|---|---|
| 20001 | 0.5106 | [0.4700, 0.5300] | PASS | 0.0595 | 0.0332 | PASS | PASS |
| 20002 | 0.4259 | [0.4700, 0.5300] | FAIL | 0.0419 | 0.0225 | PASS | FAIL |
| 20003 | 0.4851 | [0.4700, 0.5300] | PASS | 0.0535 | 0.0330 | PASS | PASS |
| 20004 | 0.4294 | [0.4700, 0.5300] | FAIL | 0.0457 | 0.0255 | PASS | FAIL |
| 20005 | 0.4566 | [0.4700, 0.5300] | FAIL | 0.0496 | 0.0322 | PASS | FAIL |

- **AUC Mean:** 0.4615 +/- 0.0325 [Min: 0.4259, Max: 0.5106]
- **AUC Acceptance Status:** **FAILED** (Mean 0.4615 is outside [0.4700, 0.5300]; 3 of 5 seeds failed)
- **Recall at Budget Status:** **PASSED** (Mean 0.0293 <= 0.05; all 5 seeds passed)

### Implementation, Data Alignment, and Partition Membership Audit
1. **Partition Membership & Isolation:** The dataset contains 1,500 scenarios in total (`scenario_0001` through `scenario_1500`). The training partition contains exactly 900 scenarios (not all 1,500 scenario IDs belong to train), the validation partition contains exactly 300 scenarios, and the held-out test partition contains the remaining 300 scenarios (150 full-resolution + 150 downsampled). Training (900 scenarios) and validation (300 scenarios) are strictly disjoint with 0 scenario overlap (`len(set(train_scenarios).intersection(val_scenarios)) == 0`).
2. **Validation Label Integrity:** Validation labels remain completely untouched and unpermuted; only training labels are shuffled.
3. **Feature-Label Alignment:** Feature rows and labels are aligned 1-to-1 on `[scenario_id, sensor_id, timestamp]` with 0 duplicate keys and identical row ordering.
4. **Class Count Preservation:** Training positive count before shuffle: 12,427 / 300,000 (4.1423%); after shuffle: 12,427 / 300,000 (4.1423%) (exact match). Validation positive count: 20,368 / 512,640 (3.9732%) (intact).
5. **Sampling Protocol:** The committed pipeline samples 300,000 rows uniformly at random across all 900 training scenarios via `DataFrame.sample(n=300000, random_state=42)`. An array prefix `X_tr[:100000]` was used only in transient exploratory scratch scripts.

### Diagnostic Experiments on the Sub-0.50 AUC Phenomenon

To investigate why validation AUC falls below 0.50 on the label-shuffle control, two separate diagnostic experiments were conducted:

#### A. Diagnostic 20-Seed Experiment (Protocol & Verified Details)
- **Objective:** Determine whether the sub-0.50 AUC in the 5-seed null control was random variance or a systematic statistical property across a broader seed ensemble.
- **Protocol Details:**
  - **Training Row Count:** 50,000 scored training rows.
  - **Training Sampling Method:** Uniform random row sampling across all 900 training scenarios via `train_m.sample(n=50000, random_state=42).reset_index(drop=True)`. Training positive prevalence: 2,121 positives / 50,000 rows (4.242%).
  - **Random Seed Range:** Seeds 20001 through 20020 inclusive (integers `range(20001, 20021)`).
  - **Shuffled-Label Procedure:** Permuted training labels using `rng = np.random.default_rng(seed); y_shuff = rng.permutation(y_tr_true)`. Validation labels remained 100% untouched.
  - **LightGBM Parameters:**
    ```python
    LGBMClassifier(
        objective="binary",
        n_estimators=50,
        learning_rate=0.1,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1
    )
    ```
  - **Validation Row Count:** Full untouched validation partition of 512,640 scored rows across all 300 validation scenarios (20,368 positive labels, 3.973% prevalence).
- **Reported Results:**
  - **Validation ROC-AUC Mean:** **0.4682**
  - **Standard Deviation:** **0.0269**
  - **Range:** Min **0.4045**, Max **0.5199**
  - **Distribution:** 18 of 20 seeds (90.0%) produced AUC < 0.50.

#### B. Diagnostic Five-Seed Full-Training-Set Experiment (Distinction from 20-Seed Run)
- **Distinction:** This experiment evaluates the official 5 seeds (20001–20005) on the **entire training partition without subsampling**, testing whether subsampling to 300k rows introduced an artifact.
- **Protocol Details:**
  - **Training Row Count:** All 1,531,872 scored training rows across all 900 training scenarios (63,716 positives, 4.159% prevalence).
  - **Random Seed Range:** Seeds 20001 to 20005.
  - **LightGBM Parameters:** `objective="binary"`, `n_estimators=100`, `learning_rate=0.05`, `num_leaves=31`, `random_state=seed`.
  - **Validation Row Count:** Full untouched validation partition of 512,640 scored rows.
- **Reported Results:**
  - **Validation ROC-AUC Mean:** **0.4554** (Min: 0.4069, Max: 0.4966).
  - 3 of 5 seeds failed the [0.4700, 0.5300] tolerance.
  - Demonstrates that the sub-0.50 AUC is identical on full-data training and not an artifact of 300k subsampling.

#### C. Statistical Mechanism: Tree-Based Outlier Ranking Inversion
- **Pure Noise Experiment:** Training on synthetic Gaussian noise features yields validation AUC = 0.5027 (in-spec), confirming that LightGBM's splitting logic is mathematically unbiased under null labels when features have symmetric distributions.
- **Physical Feature Inversion Mechanism:** Real sensor features contain rare extreme values (e.g. `repeat_run_len > 10`, `n_spikes_m > 0`, `cur_missing > 0`) representing physical fault manifestations with < 1% prevalence. In unshuffled data, these rows are almost 100% positive. When training labels are permuted across 96% negatives and 4% positives, finite leaf splits that isolate rare outlier feature values frequently sample positive rates below the 4.16% dataset average (often 0% to 2% by chance in small leaves), causing the tree to assign lower scores to outlier feature values.
- **Validation Ranking Inversion:** In the validation set (where true labels are preserved), actual faults concentrate heavily in these exact outlier bins. Because the null model assigns lower scores to outlier bins than to the normal bulk data, true positive cases receive lower probabilities than normal cases, inverting the validation ranking and systematically depressing AUC below 0.50.

#### D. Official Verdict on P4.2
- **Status:** **FAILED** on AUC tolerance.
- **Assessment:** The outlier-inversion mechanism is an **empirically supported statistical explanation**, but **NOT proof that the null control passed**. Per the project specification in `eval.yaml` (AUC range `[0.4700, 0.5300]`), the P4.2 label-shuffle null control remains strictly **FAILED**. No thresholds, model settings, datasets, or evaluation methodologies have been altered to force a pass.

### Canary Leak Test
- **Planted Canary AUC:** 1.0000 (Target >= 0.95: **PASS**)
- **Clean Matrix Max Feature:** `gap_absmax_m` (AUC = 0.6893 < 0.85 warning: **PASS**)
- **Audit Status:** Canary AUC = 1.0000 (>= 0.95: True); Clean max feature 'gap_absmax_m' AUC = 0.6893.


## 3. P4.3 First Multiclass LightGBM Benchmark

### Model Configuration
```yaml
{
  "objective": "multiclass",
  "num_class": 8,
  "boosting_type": "gbdt",
  "n_estimators": 100,
  "learning_rate": 0.05,
  "num_leaves": 31,
  "max_depth": -1,
  "min_child_samples": 20,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "random_state": 42,
  "class_weight": "balanced",
  "n_jobs": -1,
  "verbosity": -1
}
```

### Overall Performance Comparison
| Model | Accuracy | Log Loss | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro OvR AUC |
|---|---|---|---|---|---|---|---|
| **Trivial Majority Baseline** | 95.81% | N/A | 12.00% | 12.50% | 0.1223 | 0.9410 | 0.5000 |
| **First LightGBM Benchmark** | **74.62%** | **0.9029** | **25.29%** | **62.93%** | **0.3110** | **0.8277** | **0.8521** |

### Per-Class Detailed Breakdown (Validation Split)
| Class ID | Class Name | Support | Precision | Recall | F1-Score | OvR ROC-AUC |
|---|---|---|---|---|---|---|
| 0 | `normal` | 491,138 | 98.67% | 75.72% | 0.8569 | 0.8380 |
| 1 | `spike` | 331 | 11.04% | 64.35% | 0.1885 | 0.9106 |
| 2 | `offset_drift` | 5,639 | 3.96% | 34.37% | 0.0710 | 0.7335 |
| 3 | `gain_error` | 6,278 | 4.51% | 32.78% | 0.0792 | 0.7273 |
| 4 | `frozen` | 667 | 37.34% | 92.20% | 0.5315 | 0.9870 |
| 5 | `high_noise` | 2,142 | 19.29% | 84.08% | 0.3138 | 0.9790 |
| 6 | `dropout` | 2,689 | 10.63% | 48.72% | 0.1746 | 0.7379 |
| 7 | `slow_degradation` | 3,756 | 16.85% | 71.19% | 0.2725 | 0.9036 |

### Confusion Matrix (Rows: Ground Truth, Cols: Predicted)
```text
                     normal    spike  offset_  gain_er   frozen  high_no  dropout  slow_de
            normal   371903     1531    45405    41799      989     6802    10370    12339
             spike       24      213       18       12        0       33       20       11
      offset_drift     1648       49     1938     1344       22      303      155      180
        gain_error     1950       41     1216     2058       13      181      383      436
            frozen       23        6       11        5      615        0        4        3
        high_noise       79       54       77       30        0     1801       18       83
           dropout      837       18      147      193        5       34     1310      145
  slow_degradation      448       17      149      224        3      181       60     2674
```

### Limitations and Observations
1. **Extreme Class Imbalance:** Normal operation constitutes 96.03% of scored readings; individual faults range from 0.06% (spike) to 1.33% (gain error).
2. **Multi-Fault Overlap:** 2,074 scored validation rows (~0.40%) have multiple concurrent faults active. This single-label multiclass benchmark maps to the primary active fault. Phase P5 will transition to Head B (7 binary models) to handle multi-label co-occurrence gracefully.
3. **Zero Tuning Applied:** This first benchmark used default parameters without hyperparameter search, threshold calibration, or cross-validation. It serves strictly as a preliminary sanity benchmark.
4. **Split Discipline:** The test split (`data/raw/test/`) was strictly held out and accessed 0 times.
