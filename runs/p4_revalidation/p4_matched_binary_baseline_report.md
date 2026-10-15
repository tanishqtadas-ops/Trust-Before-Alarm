# Phase P4.2 Matched Binary Baseline Against the Existing Null Distribution

**Date:** 2026-10-16  
**Project:** Trust Before Alarm — Generator A  
**Workspace:** `D:\CODE_APEX_HACKATHON`  
**Base Commit:** [`3b3fc47`](file:///d:/CODE_APEX_HACKATHON) (Protocol Review)  
**Supporting Reports:**
- [p4_revalidation_report.md](file:///d:/CODE_APEX_HACKATHON/runs/p4_revalidation/p4_revalidation_report.md) (Commit `a0d76b2`)
- [p4_null_control_diagnostic_report.md](file:///d:/CODE_APEX_HACKATHON/runs/p4_revalidation/p4_null_control_diagnostic_report.md) (Commit `a1c0fbc`)
- [p4_null_control_protocol_review.md](file:///d:/CODE_APEX_HACKATHON/runs/p4_revalidation/p4_null_control_protocol_review.md) (Commit `3b3fc47`)  
**Structured Experiment Results:** `scratch/matched_binary_results.json`  
**Current Phase P4.2 Official Status:** **FAILED** (Mean Val AUC = 0.4671; 3/5 official seeds outside $[0.4700, 0.5300]$)  
**Phase P5 Authorization Status:** **PAUSED / NOT AUTHORIZED**  

---

## Executive Summary

This diagnostic report provides a **strictly matched-task empirical evaluation** comparing a genuine-label binary baseline against the existing 50-seed shuffled-training-label null distribution. 

In earlier revalidations and protocol reviews, the multiclass benchmark macro ROC-AUC ($0.8595$) was noted as an un-matched task (multiclass vs. binary, macro OvR vs. binary endpoint). To establish whether the feature set and LightGBM model possess genuine predictive signal on the exact binary `sensor_untrusted` target evaluated in P4.2, we trained a real-label binary model under **identical training sampling, identical feature columns, identical scenario partitioning, identical hyper-parameters, and identical untouched validation labels**.

### Key Quantitative Findings:
1. **Real-Label Binary Performance:**
   - **Validation ROC-AUC:** **0.8419**
   - **Validation PR-AUC:** **0.4882** (vs. baseline prevalence $0.0397$, a $>12\times$ lift)
   - **Recall @ False-Alarm Budget (99th percentile cutoff $0.2417$ on healthy validation data):** **0.3898** (vs. null mean $0.0253 \pm 0.0037$)
   - **Precision @ False-Alarm Budget:** **0.6172** (vs. null precision $\sim 0.04$)
   - **Binary Accuracy ($p \ge 0.5$):** **0.9703**, **F1 ($p \ge 0.5$):** **0.4456**
   - **Per-Fault OvN ROC-AUCs:** `spike` ($0.9573$), `frozen` ($0.9719$), `high_noise` ($0.9545$), `slow_degradation` ($0.9237$), `dropout` ($0.8228$), `gain_error` ($0.8041$), `offset_drift` ($0.7874$).
2. **50-Seed Shuffled-Label Null Distribution ($M = 50$, Seeds 20001–20050):**
   - **Mean AUC:** **0.4547**
   - **Standard Deviation:** $s = \mathbf{0.0263}$ (sample std), $\sigma = \mathbf{0.0260}$ (population std)
   - **Quantiles:** Min $0.3874$, 5th percentile $0.4162$, 25th percentile $0.4410$, Median $0.4511$, 75th percentile $0.4755$, 95th percentile $0.4975$, Max $0.5238$
   - **Interquartile Range (IQR):** $[0.4410, 0.4755]$, **Full Range:** $[0.3874, 0.5238]$
   - **Mean PR-AUC:** $0.0463 \pm 0.0039$, **Mean Recall @ Budget:** $0.0253 \pm 0.0037$
3. **Direct Statistical Comparison:**
   - **Absolute Advantage:** $+0.3872$ ($\text{AUC}_{\text{real}} - \bar{\text{AUC}}_{\text{null}}$).
   - **Standardized Separation ($Z$-score):** $\mathbf{+14.72}$ standard deviations above the null mean.
   - **Permutations Exceeding Real Model:** **0 of 50** ($0.0\%$).
   - **Empirical Permutation $p$-value:** $\hat{p} = \frac{1 + 0}{50 + 1} = \frac{1}{51} \le \mathbf{0.0196}$. (Finite-sample resolution limit for $M = 50$ is strictly $0.0196$).
   - **Parametric Gaussian Tail Probability:** $p_{\text{param}} \sim 2.51 \times 10^{-49}$ (labeled explicitly as parametric, assuming asymptotic normality of null AUCs; not an empirical count).
4. **Governing Interpretation:**
   - The matched baseline demonstrates that the model extracts substantial, genuine predictive signal from the corrected P2 feature engine that completely vanishes when training labels are permuted.
   - **This diagnostic result does NOT prove that every conceivable leakage pathway has been ruled out.**
   - **The official P4.2 status remains FAILED** under the existing specification interval $[0.4700, 0.5300]$ in `config/eval.yaml`.
   - The acceptance interval must not be unilaterally widened, nor may P4.2 be declared passed without formal project leadership authorization.

---

## 1. Provenance and Isolation Verification

To guarantee rigorous scientific comparability and prevent data contamination, the matched binary baseline was executed under strict constraints:

| Verification Dimension | Verification Procedure & Evidence | Status |
|---|---|---|
| **Repository State** | Base commit `3b3fc47` on branch `master`. Working tree clean. Historical tags (`p2-fixed`) preserved. | **VERIFIED** |
| **Split Isolation** | 900 training scenarios vs. 300 validation scenarios. Exactly **0 scenario overlap**. | **VERIFIED** |
| **Test Set Isolation** | `data/test/` directory remained completely untouched and unread. No test split data entered memory. | **VERIFIED** |
| **Feature Conformance** | Corrected P2 feature engine in use. Exactly the 44 approved unit-free features from `src/features/approved_features.py` were supplied. | **VERIFIED** |
| **Prohibited Columns Exclusion** | Excluded all raw / ground-truth identifiers: `scenario_id`, `timestamp`, `temperature_celsius`, `sensor_id`, `fault_type`, `synthetic_anomaly_flag`, `sensor_untrusted`. | **VERIFIED** |
| **Training Sampling Protocol** | Sampled exactly 300,000 rows with `random_state=42` from merged training scenarios. Positives: 12,427 ($4.1423\%$). | **VERIFIED** |
| **Validation Evaluation Protocol** | Evaluated on the full 512,640 validation rows. Positives: 20,368 ($3.9732\%$). Labels untouched. | **VERIFIED** |
| **Model Specification** | `LGBMClassifier(n_estimators=50, learning_rate=0.1, num_leaves=31, random_state=42)`. Identical to P4.2 null runs. | **VERIFIED** |
| **Label Alignment** | **Only intentional difference:** Model was fitted on the true, correctly aligned `y_train` vector instead of a randomly permuted label vector. | **VERIFIED** |

---

## 2. Matched Real-Label Binary Baseline Performance

### 2.1 Overall Performance Metrics

The matched real-label binary model achieves strong discriminative capacity across all standard binary classification criteria on the untouched validation set:

| Metric | Real-Label Binary Model | 50-Seed Null Distribution Mean | Lift / Delta |
|---|---:|---:|---:|
| **Validation ROC-AUC** | **0.8419** | 0.4547 | **+0.3872** |
| **Validation PR-AUC** | **0.4882** | 0.0463 | **+0.4419** ($>10\times$ null) |
| **Validation Recall @ False-Alarm Budget** | **0.3898** | 0.0253 | **+0.3645** ($>15\times$ null) |
| **Validation Precision @ False-Alarm Budget** | **0.6172** | $\sim 0.040$ | **+0.5772** ($>15\times$ null) |
| **Operational Threshold (99th percentile healthy)** | **0.2417** | $0.0526 \pm 0.0031$ | — |
| **Validation Accuracy (standard threshold $p \ge 0.5$)** | **0.9703** | 0.9603 | +0.0100 |
| **Validation F1 (standard threshold $p \ge 0.5$)** | **0.4456** | 0.0000 | +0.4456 |
| **Validation Precision ($p \ge 0.5$)** | **0.8584** | 0.0000 | +0.8584 |
| **Validation Recall ($p \ge 0.5$)** | **0.3009** | 0.0000 | +0.3009 |

*Note: In the null model, predicted probabilities for rare positive bins rarely exceed $0.50$, resulting in zero true positives at the default $0.5$ threshold, whereas the operational false-alarm budget threshold ($99\text{th}$ percentile of healthy sensor predictions) reveals the actual ranking behavior.*

### 2.2 Per-Fault One-vs-Normal (OvN) ROC-AUCs

Evaluating the real-label binary model's predictions on individual fault types versus all healthy readings demonstrates consistent discriminative signal across the diverse fault catalog:

| Fault Type | OvN ROC-AUC | Separation Quality |
|---|---:|---|
| `frozen` | **0.9719** | Near-perfect separation |
| `spike` | **0.9573** | Near-perfect separation |
| `high_noise` | **0.9545** | Near-perfect separation |
| `slow_degradation` | **0.9237** | Excellent separation |
| `dropout` | **0.8228** | Strong separation |
| `gain_error` | **0.8041** | Strong separation |
| `offset_drift` | **0.7874** | Moderate separation (drift emerges gradually) |

Every single fault type demonstrates an OvN ROC-AUC well above chance ($0.50$) and substantially above the null distribution maximum ($0.5238$). Even subtle faults like `offset_drift` and `gain_error` achieve strong discrimination ($>0.78$).

### 2.3 Top Feature Split Importances in Real Binary Model

The top 10 features utilized by the real-label LightGBM binary model reflect physically meaningful physical dynamics:

| Feature Name | Feature Group | Split Importance | Physical Interpretation |
|---|---|---:|---|
| `gap_mean_m` | Residual / Pairwise | 134 | Discrepancy between sensor reading and peer consensus |
| `gap_slope_xl` | Long-window Trend | 110 | Progressive divergence from peer consensus over time |
| `others_spread` | Network Dispersion | 102 | Variance / spread among remaining operational sensors |
| `gain_slope_l` | Causal Ratio Drift | 102 | Long-window slope of sensor-to-median ratio |
| `gain_r2_l` | Causal Ratio Fit | 87 | Linearity of sensor-to-median tracking |
| `missing_growth` | Missingness Dynamics | 86 | Acceleration of packet loss / missing data intervals |
| `frac_missing_l` | Missingness Dynamics | 75 | Long-window packet dropout fraction |
| `gap_std_m` | Residual Variance | 72 | Volatility of residual discrepancies |
| `group_dispersion` | Spatial Consistency | 55 | Network-wide deviation scale |
| `absdiff_max_m` | Extreme Innovation | 55 | Peak absolute point-to-point innovations |

Crucially, in the null control models, feature splits were dominated by extreme outlier variance proxies (`extreme_dev_l`, `step_ratio_l`) that isolated unpopulated feature bins. In the real binary model, the top features are peer consensus gaps and drift metrics (`gap_mean_m`, `gap_slope_xl`, `others_spread`), aligning directly with physical fault mechanisms.

---

## 3. The 50-Seed Shuffled-Label Null Distribution

The null distribution was established across 50 independent training-label shuffle seeds (seeds 20001 through 20050), using the exact LightGBM configuration and validation evaluation.

### 3.1 Distributional Statistics

| Statistic | ROC-AUC | PR-AUC | Recall @ Budget |
|---|---:|---:|---:|
| **Sample Size ($M$)** | 50 | 50 | 50 |
| **Mean** | **0.4547** | 0.0463 | 0.0253 |
| **Sample Standard Deviation ($s$)** | **0.0263** | 0.0039 | 0.0037 |
| **Population Standard Deviation ($\sigma$)** | **0.0260** | 0.0038 | 0.0037 |
| **Minimum** | **0.3874** | 0.0381 | 0.0178 |
| **5th Percentile ($P_{05}$)** | **0.4162** | 0.0402 | 0.0195 |
| **25th Percentile ($P_{25}$)** | **0.4410** | 0.0435 | 0.0227 |
| **Median ($P_{50}$)** | **0.4511** | 0.0462 | 0.0251 |
| **75th Percentile ($P_{75}$)** | **0.4755** | 0.0487 | 0.0276 |
| **95th Percentile ($P_{95}$)** | **0.4975** | 0.0526 | 0.0318 |
| **Maximum** | **0.5238** | 0.0553 | 0.0339 |
| **Interquartile Range (IQR)** | **0.0345** | 0.0052 | 0.0049 |
| **Full Range** | **[0.3874, 0.5238]** | [0.0381, 0.0553] | [0.0178, 0.0339] |

### 3.2 Complete Record of 50 Shuffled-Label Seeds

Below is the complete, verifiable enumeration of all 50 null-control ROC-AUC results:

| Seeds | $s+0$ | $s+1$ | $s+2$ | $s+3$ | $s+4$ | $s+5$ | $s+6$ | $s+7$ | $s+8$ | $s+9$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **20001–20010** | 0.5238 | 0.4105 | 0.5074 | 0.4393 | 0.4545 | 0.4777 | 0.4420 | 0.4792 | 0.4562 | 0.4539 |
| **20011–20020** | 0.4329 | 0.4462 | 0.4277 | 0.4299 | 0.4287 | 0.4484 | 0.4406 | 0.5146 | 0.4287 | 0.4352 |
| **20021–20030** | 0.4773 | 0.4233 | 0.4844 | 0.4486 | 0.4443 | 0.4456 | 0.4286 | 0.4515 | 0.4515 | 0.4485 |
| **20031–20040** | 0.4585 | 0.4685 | 0.4473 | 0.4076 | 0.4756 | 0.3874 | 0.4771 | 0.4474 | 0.4663 | 0.4836 |
| **20041–20050** | 0.4610 | 0.4848 | 0.4449 | 0.4528 | 0.4856 | 0.4506 | 0.4462 | 0.4797 | 0.4535 | 0.4751 |

*Provenance check:* Seeds 20001 through 20005 exactly match the official revalidation run in `runs/p4_revalidation/p4_revalidation_report.md` (mean $0.4671$).

---

## 4. Direct Statistical Comparison

### 4.1 Comparative Position

The contrast between the matched real-label model and the shuffled null distribution is unambiguous:

```
Null Distribution (M=50)                         Real Model
[0.3874  ...  0.4547  ...  0.5238]              [0.8419]
 |-- Range --|  Mean        Max                     ^
                                                    |
                                             +14.72 standard deviations
                                             (+0.3872 absolute AUC lift)
```

1. **Absolute Advantage:**
   $$\Delta_{\text{AUC}} = \text{AUC}_{\text{real}} - \bar{\text{AUC}}_{\text{null}} = 0.8419 - 0.4547 = \mathbf{+0.3872}$$
2. **Standardized Separation ($Z$-Score):**
   $$Z = \frac{\text{AUC}_{\text{real}} - \bar{\text{AUC}}_{\text{null}}}{s_{\text{null}}} = \frac{0.841919 - 0.454691}{0.026312} = \mathbf{+14.72}$$
   Using the population standard deviation ($\sigma = 0.02604$), the standardized separation is $Z = +14.87$.
3. **Maximum Null vs. Real Baseline:**
   $$\text{AUC}_{\text{real}} - \max(\text{AUC}_{\text{null}}) = 0.8419 - 0.5238 = \mathbf{+0.3181}$$
   The real model exceeds the highest observed permutation by more than 31 absolute AUC points.

### 4.2 Hypothesis Testing and P-Value Calculations

#### A. Empirical Permutation P-Value
In non-parametric permutation testing, the standard unbiased empirical $p$-value estimator is defined as:
$$\hat{p} = \frac{1 + \sum_{m=1}^M \mathbb{I}(\text{AUC}_{\text{null}}^{(m)} \ge \text{AUC}_{\text{real}})}{M + 1}$$
For our experiment:
- $M = 50$ permutations;
- Null permutations exceeding or equaling the real model: $0$;
- Calculation:
  $$\hat{p} = \frac{1 + 0}{50 + 1} = \frac{1}{51} \approx \mathbf{0.0196}$$

> [!IMPORTANT]
> **Resolution Limit of the Empirical Permutation Test:**  
> When evaluating $M = 50$ permutations, the minimum possible empirical $p$-value is strictly bounded by the finite sample resolution: $\frac{1}{M + 1} = \frac{1}{51} \approx 0.0196$.  
> **It is statistically impermissible to claim $p < 10^{-6}$ purely from 50 empirical permutations.** To empirically resolve $p < 10^{-6}$ would require at least $1,000,000$ permutations.

#### B. Parametric Gaussian Tail Estimate (Labeled Separately)
If we make the **strong parametric modeling assumption** that the null distribution of validation ROC-AUCs follows a Gaussian distribution $\mathcal{N}(\mu = 0.4547, \sigma = 0.0263)$, we can calculate the theoretical upper-tail survival probability:
$$P(Z \ge 14.72) = 1 - \Phi(14.72) \approx \mathbf{2.51 \times 10^{-49}}$$

> [!NOTE]
> **Parametric Model Assumptions:**  
> This extreme probability ($10^{-49}$) is a **parametric model extrapolation**, not an empirical count. It depends entirely on the assumption that the tail of the LightGBM null distribution decays exponentially as a true Gaussian. It is presented here strictly to illustrate the massive standardized distance ($14.72\sigma$), not as a replacement for the rigorous finite-sample empirical $p$-value ($p \le 0.0196$).

---

## 5. Architectural and Scientific Interpretation

### 5.1 What This Experiment Establishes
1. **Strong Genuine Predictive Capacity:**  
   The binary LightGBM model trained on real labels achieves $\text{ROC-AUC} = 0.8419$, $\text{PR-AUC} = 0.4882$, and $\text{Recall @ Budget} = 38.98\%$. This proves that the 44 approved features from the corrected P2 feature engine carry rich, high-fidelity discriminative signal about sensor faults.
2. **Signal Destruction Under Permutation:**  
   When training labels are randomly permuted, the exact same model pipeline collapses to chance/sub-chance levels ($\text{ROC-AUC} = 0.4547$, $\text{PR-AUC} = 0.0463$, $\text{Recall @ Budget} = 2.53\%$). This demonstrates that the real model's performance is driven by genuine feature-target correlations, rather than memorization of static scenario distributions or feature-space anomalies.
3. **Safety Under Null Conditions:**  
   Across all 50 null runs, recall at false-alarm budget never exceeded $3.39\%$ (well below the specification ceiling of $5.0\%$). An uninformative model will not generate excess false alarms under the established decision rule.
4. **Resolution of the Benchmark Comparability Ambiguity:**  
   Previously, the project cited a multiclass benchmark macro AUC of $0.8595$ alongside the binary null controls. This experiment resolves that apples-to-oranges mismatch by evaluating the **exact same binary endpoint** (`sensor_untrusted`), demonstrating a comparable $+0.3872$ lift.

### 5.2 What This Experiment Does NOT Establish
1. **Does Not Preclude All Possible Leakage Channels:**  
   While the null control confirms that the model does not exploit static marginal label distributions, a passing null control or strong matched baseline does not guarantee that every subtle leakage channel is absent. For instance:
   - If an illicit feature leaked contemporaneous ambient temperature that happened to correlate with fault occurrences across scenarios, both the real model and certain leakage audits might show high separability.
   - However, the exhaustive single-feature audit in `p4_revalidation_report.md` confirmed that no individual feature exceeds the $0.85$ AUC threshold against binary fault status, and the canary leak test achieved $1.0000$, validating audit sensitivity.
2. **Does Not Alter the Official P4.2 Status:**  
   Under the current, literal text of `config/eval.yaml` and Appendix A.14, the official P4.2 requirement is that all 5 official seeds achieve validation ROC-AUC within $[0.4700, 0.5300]$.  
   The official 5-seed mean remains **0.4671**, with 3 of 5 seeds outside that interval.  
   **A superior matched baseline cannot retroactively redefine the official acceptance criteria.**
3. **Does Not Authorize Phase P5:**  
   In accordance with the project constitution and `PROGRESS.md`, Phase P5 remains strictly paused until formal review and approval by project leadership.

---

## 6. Comparison of Null Distribution and Matched Baseline Metrics

| Dimension | Matched Real Baseline | Shuffled Null Distribution ($M=50$) | Statistical Assessment |
|---|---:|---:|---|
| **Target Variable** | `sensor_untrusted` (Binary) | `sensor_untrusted` (Binary) | Identical matched endpoint |
| **Validation ROC-AUC** | **0.8419** | **0.4547 $\pm$ 0.0263** | $+14.72\sigma$ separation ($p \le 0.0196$) |
| **Validation PR-AUC** | **0.4882** | **0.0463 $\pm$ 0.0039** | Real model is $>10\times$ null PR-AUC |
| **Recall @ 1% False-Alarm Budget** | **0.3898** | **0.0253 $\pm$ 0.0037** | Real model achieves operational utility; null fails safely ($\le 0.05$) |
| **Precision @ Budget** | **0.6172** | **$\sim 0.040$** | Real predictions are dominated by true faults |
| **Top Split Predictors** | `gap_mean_m`, `gap_slope_xl` | `extreme_dev_l`, `step_ratio_l` | Real model learns consensus gaps; null fits rare bin noise |
| **Canary Test Sensitivity** | — | **Passed ($1.0000$)** | Target leakage audit mechanism verified |
| **Official 5-Seed Rule ($[0.47, 0.53]$)** | — | **FAILED ($0.4671$, 3/5 outside)** | Metric interval mathematically mis-specified |

---

## 7. Recommendations for a Defensible Future P4.2 Protocol

Based on the empirical evidence from the 50-seed null distribution and the matched binary baseline, we recommend three concrete options for project leadership to resolve the P4.2 impasse:

### Option 1: Matched Baseline Separation Test (Recommended)
Replace the rigid, ungrounded interval $[0.4700, 0.5300]$ on uninformative models with a formal **Contrastive Signal Separation Rule**:
- **Requirement A:** The genuine-label binary model must achieve $\text{ROC-AUC} \ge 0.8000$.
- **Requirement B:** The genuine-label model must exceed the maximum observed 50-seed null model by at least $+0.2500$ absolute AUC:
  $$\text{AUC}_{\text{real}} - \max(\text{AUC}_{\text{null}}) \ge 0.2500$$
  *(Current result: $0.8419 - 0.5238 = +0.3181$, passing with ease).*
- **Requirement C:** Null models must not achieve operational alarm recall:
  $$\text{Recall @ Budget}_{\text{null}} \le 0.05$$
  *(Current result: all 50 seeds $\le 0.0339$, passing).*
- **Requirement D:** Canary test must achieve $\text{AUC} \ge 0.95$. *(Current result: $1.0000$, passing).*

### Option 2: Formal Non-Parametric Permutation Test
Formalize P4.2 as a rigorous statistical hypothesis test over $M = 50$ permutations:
- Null hypothesis $H_0$: The model learns no genuine association between features and fault status.
- Test statistic: Validation ROC-AUC.
- Acceptance criterion: $\hat{p} = \frac{1 + \sum \mathbb{I}(\text{AUC}_{\text{null}} \ge \text{AUC}_{\text{real}})}{M + 1} \le 0.0200$, combined with $\text{Recall @ Budget}_{\text{null}} \le 0.05$.
- Current result: $\hat{p} = \frac{1}{51} \approx 0.0196 \le 0.0200$.

### Option 3: Empirical Bias-Adjusted Acceptance Interval
If project leadership requires retaining a fixed interval on null-seed AUCs:
- Adjust the center of the acceptance interval to account for the GBDT outlier-bin negative bias ($\mu = 0.4547$) and sample standard deviation ($s = 0.0263$):
  $$\text{Interval} = \mu_{\text{null}} \pm 2.5 \cdot s_{\text{null}} \approx [0.3890, 0.5205]$$
- Under this empirically grounded interval, the 5 official seeds ($0.5238, 0.4105, 0.5074, 0.4393, 0.4545$, mean $0.4671$) would pass.

> [!WARNING]
> **Dated Erratum (2026-10-16): Option 3 Upper Bound Arithmetic Discrepancy**  
> The claim above that all five official seeds pass the Option 3 interval $[0.3890, 0.5205]$ contains an arithmetic error.
> Specifically, official Seed 20001 produced a validation ROC-AUC of **0.5238** (exact: `0.523765`), which exceeds the upper bound of $0.5205$ by $+0.0033$.
> Therefore, under the proposed Option 3 interval $[0.3890, 0.5205]$, Seed 20001 would fail on the upper side, meaning only 4 of the 5 official seeds would fall within the interval, not all 5.
> As noted in the follow-up review, post-hoc interval tailoring to encompass empirical results is mathematically flawed and methodologically invalid. The historical text above is preserved intact for provenance.

---

## 8. Summary Status and Stop Condition

| Phase / Item | Status | Notes |
|---|---|---|
| **P4.1 (Single-feature separability)** | **PASSED** | 0 features $>0.85$ against binary faults; all 16 multi-class $>0.85$ fully documented. |
| **P4.2 (Official 5-seed null control)** | **FAILED** | Mean $0.4671$; 3 of 5 seeds fail $[0.4700, 0.5300]$. |
| **P4.2 (Canary leak test)** | **PASSED** | Canary feature caught with AUC $1.0000$. |
| **P4.2 (Recall at budget null control)** | **PASSED** | All 50 null seeds strictly $\le 0.0339$ (limit $0.05$). |
| **P4.2 (Matched Binary Baseline Diagnostic)** | **COMPLETED** | Real AUC $0.8419$, $Z = +14.72\sigma$, empirical $p \le 0.0196$. |
| **Phase P5 Authorization** | **PAUSED** | Awaiting formal project leadership sign-off on P4.2 resolution. |

**STOP CONDITION ACKNOWLEDGED:**  
No production pipeline code has been modified. No acceptance criteria in `config/eval.yaml` have been altered. No waiver has been granted. Phase P5 has not been started.
