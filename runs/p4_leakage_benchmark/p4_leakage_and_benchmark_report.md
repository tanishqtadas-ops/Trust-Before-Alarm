# Phase P4: Leakage Audits and First LightGBM Benchmark Report
*Generated on: 2026-10-15 13:34:58 UTC*
*Dataset: Generator A (Train: 1,531,872 scored rows, Val: 512,640 scored rows; Test split STRICTLY UNTOUCHED)*

## Executive Summary
- **P4.1 Single-Feature Separability:** 0 features flagged on `sensor_untrusted` (AUC > 0.85); 16 features flagged on per-fault OvR (> 0.85). All flagged features reflect documented physical domain properties (e.g. `consec_missing` for dropout, `repeat_run_len` for frozen); zero ground-truth leakage or illicit metadata proxies detected (**PASSED**).
- **P4.2 Label-Shuffle Null Control:**
  - **AUC Tolerance Status:** **FAILED** (Mean Val AUC = 0.4615 +/- 0.0325 outside required tolerance [0.4700, 0.5300]; 2/5 seeds passed, 3/5 seeds failed).
  - **Recall at Budget Status:** **PASSED** (All 5 seeds <= 0.05, mean recall @ budget = 0.0293 <= 0.05).
  - **Alignment & Leakage Audit:** Complete audit confirmed zero data leakage, zero join duplication, 100% disjoint scenario sets (900 train vs 300 val), and exact class count preservation (12,478/300k train, 20,368/512k val).
  - **Root Cause Analysis:** Empirical experiments confirmed that the sub-0.50 AUC is a tree ranking inversion artifact on rare extreme features under 96/4 class imbalance (pure noise features achieve AUC 0.5027; real features invert probabilities on rare physical outliers). However, per project specification (`eval.yaml`), the control must be reported as failed on the strict AUC criterion.
- **P4.2 Canary Leak Test:** Planted canary was successfully caught with AUC = 1.0000 (>= 0.95: **PASSED**). Clean feature matrix contains zero synthetic ground-truth copies (max feature `gap_absmax_m` AUC = 0.6893 < 0.85 and < 0.95: **PASSED**).
- **P4.3 First LightGBM Benchmark:** 8-class GBDT benchmark achieved **74.62% accuracy**, **Macro-F1 0.3110**, and **Macro-AUC 0.8521** on validation set without tuning. Significant lift over trivial majority-class baseline (Macro-F1 0.1223). Test split strictly held out (0 accesses: **PASSED**).

## 1. P4.1 Single-Feature Separability Audit
Threshold: AUC > 0.85 (Appendix A.14 / eval.yaml)

### Threshold Specification Clarification
- `single_feature_auc_threshold: 0.85`: Used as an investigation trigger for clean feature evaluation. Any clean feature exceeding 0.85 must be justified by physical domain necessity.
- `canary_auc_min: 0.95`: Used as the sensitivity pass mark for catching planted ground-truth leakage (canary).
- These thresholds are distinct, intentional, and spec-compliant.

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

### Features Flagged for Investigation (AUC > 0.85)
| Feature Name | Target Evaluated | Max AUC | Assessment & Specification Justification |
|---|---|---|---|
| `gap_absmax_m` | `ovr_high_noise` | 0.9075 | Fault 'high_noise' AUC 0.907 > 0.85. Legitimate physical indicator; retained per spec. |
| `absdiff_max_m` | `ovr_high_noise` | 0.9356 | Fault 'high_noise' AUC 0.936 > 0.85. Legitimate physical indicator; retained per spec. |
| `gap_std_m` | `ovr_high_noise` | 0.9170 | Fault 'high_noise' AUC 0.917 > 0.85. Legitimate physical indicator; retained per spec. |
| `n_spikes_m` | `ovr_high_noise` | 0.9344 | Fault 'high_noise' AUC 0.934 > 0.85. Legitimate physical indicator; retained per spec. |
| `absdiff_max_s` | `ovr_frozen` | 0.9503 | Fault 'frozen' AUC 0.950 > 0.85. Legitimate physical indicator; retained per spec. |
| `range_m` | `ovr_high_noise` | 0.9224 | Fault 'high_noise' AUC 0.922 > 0.85. Legitimate physical indicator; retained per spec. |
| `noise_ratio_l` | `ovr_high_noise` | 0.8744 | Fault 'high_noise' AUC 0.874 > 0.85. Legitimate physical indicator; retained per spec. |
| `noise_growth` | `ovr_high_noise` | 0.8744 | Fault 'high_noise' AUC 0.874 > 0.85. Legitimate physical indicator; retained per spec. |
| `noise_ratio_m` | `ovr_frozen` | 0.9171 | Fault 'frozen' AUC 0.917 > 0.85. Legitimate physical indicator; retained per spec. |
| `noise_ratio_s` | `ovr_frozen` | 0.9067 | Fault 'frozen' AUC 0.907 > 0.85. Legitimate physical indicator; retained per spec. |
| `unique_ratio_m` | `ovr_frozen` | 0.8927 | Fault 'frozen' AUC 0.893 > 0.85; Expected physical response to frozen sensor (provable rule). Legitimate physical indicator; retained per spec. |
| `longest_repeat_m` | `ovr_frozen` | 0.9668 | Fault 'frozen' AUC 0.967 > 0.85; Expected physical response to frozen sensor (provable rule). Legitimate physical indicator; retained per spec. |
| `frac_zero_diff_m` | `ovr_frozen` | 0.9562 | Fault 'frozen' AUC 0.956 > 0.85; Expected physical response to frozen sensor (provable rule). Legitimate physical indicator; retained per spec. |
| `kurtosis_m` | `ovr_spike` | 0.9105 | Fault 'spike' AUC 0.911 > 0.85. Legitimate physical indicator; retained per spec. |
| `repeat_run_len` | `ovr_frozen` | 0.9717 | Fault 'frozen' AUC 0.972 > 0.85; Expected physical response to frozen sensor (provable rule). Legitimate physical indicator; retained per spec. |
| `noise_short_long` | `ovr_frozen` | 0.9032 | Fault 'frozen' AUC 0.903 > 0.85. Legitimate physical indicator; retained per spec. |

## 2. P4.2 Label-Shuffle Null Control & Canary Leak Test

### Label-Shuffle Results (Seeds 20001 to 20005)
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

### Implementation and Data Alignment Audit
1. **Partition Isolation:** Training (900 scenarios) and validation (300 scenarios) are strictly disjoint (0 scenario overlap).
2. **Validation Label Integrity:** Validation labels remain completely untouched; only training labels are permuted.
3. **Feature-Label Alignment:** Feature rows and labels are aligned 1-to-1 on `[scenario_id, sensor_id, timestamp]` with 0 duplicate keys and identical row ordering.
4. **Class Count Preservation:** Training positive count before shuffle: 12,478 / 300,000 (4.1593%); after shuffle: 12,478 / 300,000 (4.1593%) (exact match). Validation positive count: 20,368 / 512,640 (3.9732%) (intact).
5. **Sampling Protocol:** The committed pipeline samples 300,000 rows uniformly at random across all 900 training scenarios via `DataFrame.sample(n=300000, random_state=42)`. An array prefix `X_tr[:100000]` was used only in exploratory scratch scripts. A separate evaluation using all 1,531,872 training rows was executed and produced the same outcome (mean AUC 0.4554; 3/5 seeds failed).
6. **Evaluation of Tree-Based Outlier Inversion:**
   - *Pure Noise Experiment:* Training on synthetic Gaussian noise features yields validation AUC = 0.5027 (in-spec), confirming LightGBM's splitting logic is unbiased under null labels.
   - *Physical Feature Inversion:* On real features, rare extreme values (e.g. `repeat_run_len > 10`, `n_spikes_m > 0`, `cur_missing > 0`) contain very few training samples (< 1%). In unshuffled data, these rows are nearly 100% positive. When labels are permuted across 96% zeros and 4% ones, finite leaf splits on outlier features frequently sample positive rates below the 4.16% dataset average (often 0% to 2%), causing the tree to assign lower scores to outlier feature values.
   - *Validation Inversion:* In the validation set (preserving true labels), actual faults concentrate heavily in these exact outlier bins. Because the null model assigns lower scores to outlier bins than to the normal bulk, true positive cases receive lower probabilities than normal cases, inverting the validation ranking and systematically depressing AUC below 0.50 (mean 0.4682 across 20 seeds).
   - *Verdict:* While the physical/statistical mechanism is confirmed as an artifact of tree splitting on imbalanced/outlier data rather than data leakage, the result fails the specification in `eval.yaml` ([0.4700, 0.5300]). The control is therefore reported as **FAILED** on AUC tolerance.

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
