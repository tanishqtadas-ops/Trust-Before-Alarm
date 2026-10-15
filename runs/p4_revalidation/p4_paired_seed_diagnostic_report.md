# Phase P4.2 Paired-Seed Diagnostic and Report Reconciliation

**Project:** Trust Before Alarm — Generator A  
**Workspace:** `D:\CODE_APEX_HACKATHON`  
**Date:** 2026-10-16  
**Status:** Exploratory Diagnostic Report (Non-Production, Non-Confirmatory)  
**Deliverable Path:** `runs/p4_revalidation/p4_paired_seed_diagnostic_report.md`  

---

## 1. Executive Summary & Context

Following the independent Claude Sonnet audit of Phase P4.2, two critical technical questions were raised:
1. **Asymmetric Random Seed Allocation:** In the matched binary experiment, the genuine-label model was trained with LightGBM `random_state=42`, whereas the 50 null-control runs each used a different LightGBM seed tied directly to the permutation seed (`random_state=seed` for $seed \in [20001, 20050]$). Could internal LightGBM stochasticity (e.g., histogram binning, tie-breaking) account for any part of the sub-$0.50$ AUC drift or null dispersion?
2. **Historical Documentation Discrepancies:**
   - **Option 3 Arithmetic Error:** The proposed empirical interval $[0.3890, 0.5205]$ in `runs/p4_revalidation/p4_matched_binary_baseline_report.md` was claimed to pass all five official seeds. In fact, official Seed 20001 has AUC **0.5238**, strictly exceeding the upper bound of $0.5205$.
   - **`class_weight` Provenance:** Section 3.1 of `runs/p4_revalidation/p4_revalidation_report.md` described the null model as `class_weight='balanced'`, while the executed code in `src/eval/leakage_audit.py` used `class_weight=None`.

### Key Diagnostic Findings
1. **Holding LightGBM `random_state=42` Fixed Does Not Materially Change the Null Distribution:**
   - The 50-seed paired null distribution (with LightGBM `random_state=42` fixed across all permutations) has mean AUC **0.4528 $\pm$ 0.0238** (range $[0.3954, 0.5178]$).
   - This closely matches the unpaired null distribution (where LightGBM seed varied with permutation seed: mean **0.4547 $\pm$ 0.0263**, range $[0.3874, 0.5238]$).
   - The Pearson correlation between paired and unpaired runs is extremely high: **$r = 0.9177$**, with a mean pairwise difference of only **$-0.0019 \pm 0.0105$**.
   - Under paired runs, **41 of 50 seeds (82.0%)** still fail the official $[0.4700, 0.5300]$ criterion.
   - For the five official seeds (20001–20005), mean AUC is **0.4621**, and exactly the same **3 of 5 seeds fail** (Seeds 20002, 20004, 20005).
   - This proves conclusively that the sub-$0.50$ null behavior is driven by the interaction between label permutation and skewed physical feature distributions, not LightGBM's internal random seed.
2. **Separation of Genuine-Label Signal Remains Decisive:**
   - The genuine-label binary model achieves validation ROC-AUC **0.8419**, PR-AUC **0.4882**, and recall at false-alarm budget **0.3898** (at $61.72\%$ precision).
   - The separation against the paired null distribution is **$+0.3891$ absolute AUC** ($Z = +16.34$ standard deviations).
   - Zero of the 50 paired null runs equaled or exceeded the real model ($\hat{p} \le 0.0196$, empirical resolution limit $1/51$).
   - All 50 paired null runs strictly respected the false-alarm operational budget (mean recall at budget $0.0247 \le 0.05$; max $0.0324$).
3. **Historical Documentation Discrepancies Reconciled:**
   - A dated erratum has been attached to `runs/p4_revalidation/p4_matched_binary_baseline_report.md` documenting the Option 3 arithmetic error ($0.5238 > 0.5205$).
   - An evidence-based investigation confirmed that the mention of `class_weight='balanced'` in Section 3.1 of `runs/p4_revalidation/p4_revalidation_report.md` was solely a prose transcription typo copying P4.3 benchmark hyperparameters. The code executed in commit `a0d76b2` (`src/eval/leakage_audit.py`) has always used `class_weight=None`, verified down to 16 decimal places against `p4_evaluation_summary.json`.
4. **Preservation of Official Status:**
   - This experiment is exploratory and diagnostic only.
   - No production code has been modified.
   - No official acceptance thresholds in `config/eval.yaml` have been altered.
   - No post-hoc acceptance rule has been adopted.
   - Zero access was made to the held-out test split.

---

## 2. Experimental Methodology & Rigorous Controls

### 2.1 Non-Negotiable Boundaries
- **Strict Data Split Discipline:** Only training (`data/features/train/`) and validation (`data/features/val/`) splits were accessed. The held-out test split was untouched.
- **Scenario Isolation:** 900 training scenarios vs. 300 validation scenarios; verified zero scenario ID overlap ($900 \cap 300 = \emptyset$).
- **Untouched Validation Labels:** Validation features and labels were strictly frozen and unmodified across all genuine and null runs.
- **Pure Observable Features:** Evaluated strictly on the 49 approved observable sensor features (`SENSOR_FEATURE_NAMES`) from the corrected P2 feature engine.

### 2.2 Paired Diagnostic Protocol
To strictly isolate the effect of label permutation from model initialization randomness, the experiment was structured as follows:

| Component | Genuine-Label Baseline | Paired Null Control (Exploratory) | Unpaired Null Control (Historical Pipeline) |
|---|---|---|---|
| **Feature Engine** | Corrected P2 (`1efdc79` / `2723f7a`) | Corrected P2 (`1efdc79` / `2723f7a`) | Corrected P2 (`1efdc79` / `2723f7a`) |
| **Feature Set** | 49 `SENSOR_FEATURE_NAMES` | 49 `SENSOR_FEATURE_NAMES` | 49 `SENSOR_FEATURE_NAMES` |
| **Training Sample** | 300,000 rows (`random_state=42`) | 300,000 rows (`random_state=42`) | 300,000 rows (`random_state=42`) |
| **Validation Set** | 512,640 scored rows ($t \ge 480$) | 512,640 scored rows ($t \ge 480$) | 512,640 scored rows ($t \ge 480$) |
| **Training Labels** | Ground truth `sensor_untrusted` | Permuted via `default_rng(seed)` | Permuted via `default_rng(seed)` |
| **Permutation Seeds** | N/A | $20001, \dots, 20050$ ($M=50$) | $20001, \dots, 20050$ ($M=50$) |
| **LightGBM `random_state`** | **`42` (Fixed)** | **`42` (Strictly Fixed)** | **`seed` (Tied to Permutation)** |
| **LightGBM Hyperparameters** | `n_estimators=50, lr=0.1, num_leaves=31, class_weight=None` | `n_estimators=50, lr=0.1, num_leaves=31, class_weight=None` | `n_estimators=50, lr=0.1, num_leaves=31, class_weight=None` |

By fixing LightGBM's `random_state=42` across every paired null run, **the only factor that varies between null models (and between the null models and the genuine baseline) is the permutation of the training labels**.

---

## 3. Paired vs Unpaired Experimental Results

### 3.1 Comparative Null Distribution Statistics ($M = 50$ Permutations)

| Metric | Paired Null Control (`lgb_seed=42`) | Unpaired Null Control (`lgb_seed=seed`) | Paired Difference ($\Delta = \text{Paired} - \text{Unpaired}$) |
|---|---:|---:|---:|
| **Mean ROC-AUC** | **0.4528 $\pm$ 0.0238** | **0.4547 $\pm$ 0.0263** | **$-0.0019 \pm 0.0105$** |
| **Median ROC-AUC** | **0.4505** | **0.4511** | $-0.0006$ |
| **IQR [25th, 75th]** | **[0.4397, 0.4651]** | **[0.4410, 0.4755]** | — |
| **5th – 95th Percentile** | **[0.4135, 0.4916]** | **[0.4162, 0.4975]** | — |
| **Range [Min, Max]** | **[0.3954, 0.5178]** | **[0.3874, 0.5238]** | Max $|\Delta| = 0.0275$ |
| **Seeds in $[0.4700, 0.5300]$** | **9 of 50 (18.0%)** | **14 of 50 (28.0%)** | -5 seeds |
| **Mean PR-AUC** | **0.0463 $\pm$ 0.0035** | **0.0463 $\pm$ 0.0039** | $0.0000$ |
| **Mean Recall @ Budget** | **0.0247 $\pm$ 0.0035** | **0.0253 $\pm$ 0.0037** | $-0.0006$ |
| **Max Recall @ Budget** | **0.0324** | **0.0342** | $-0.0018$ |
| **Budget Compliance ($\le 0.05$)** | **50 of 50 (100.0%)** | **50 of 50 (100.0%)** | $0$ |
| **Mean Precision @ Budget** | **0.0926** | **0.0944** | $-0.0018$ |
| **Pearson Correlation ($r$)** | — | — | **$r = 0.9177$ ($p < 10^{-20}$)** |

### 3.2 Breakdown for the Five Official Seeds (20001–20005)

The original five official seeds are preserved without modification. The table below presents the side-by-side comparison:

| Seed | Official / Unpaired AUC (`lgb=seed`) | Official Interval $[0.47, 0.53]$ | Paired AUC (`lgb=42`) | Paired Interval $[0.47, 0.53]$ | Difference ($\Delta$) | Paired Rec @ Budget | Paired Prec @ Budget |
|---|---:|:---:|---:|:---:|---:|---:|---:|
| **20001** | **0.5238** | PASS | **0.4963** | PASS | $-0.0275$ | 0.0243 | 0.0912 |
| **20002** | **0.4105** | FAIL | **0.4070** | FAIL | $-0.0035$ | 0.0175 | 0.0674 |
| **20003** | **0.5074** | PASS | **0.5079** | PASS | $+0.0005$ | 0.0285 | 0.1056 |
| **20004** | **0.4393** | FAIL | **0.4416** | FAIL | $+0.0023$ | 0.0210 | 0.0800 |
| **20005** | **0.4545** | FAIL | **0.4578** | FAIL | $+0.0032$ | 0.0275 | 0.1023 |
| **Mean (5 Seeds)** | **0.4671 $\pm$ 0.0423** | **FAIL** (3/5 fail) | **0.4621 $\pm$ 0.0384** | **FAIL** (3/5 fail) | **$-0.0050$** | **0.0238** | **0.0893** |

#### Observations on Seed Breakdown
1. **Official Failure Remains Unaltered:** In both unpaired and paired runs, exactly the same **3 of 5 official seeds fail** the $[0.4700, 0.5300]$ criterion (Seeds 20002, 20004, 20005). The 5-seed mean remains firmly below $0.4700$ ($0.4621$ paired vs. $0.4671$ unpaired).
2. **Seed 20001 Movement:** Under paired evaluation (`lgb=42`), Seed 20001 shifts from $0.5238$ to $0.4963$ ($\Delta = -0.0275$). While this happens to bring it close to $0.50$, Seed 20002 remains deeply inverted at $0.4070$ and Seed 20004 at $0.4416$.
3. **Strong Stability Across Seeds:** For Seeds 20002, 20003, 20004, and 20005, the paired AUC changes by less than $0.0035$ from the unpaired run, corroborating the high correlation ($r = 0.9177$).

---

## 4. Signal Separation Analysis

### 4.1 Genuine-Label Model vs. Paired Null Distribution

| Metric | Genuine-Label Model | Paired Null Mean ($\pm 1s$) | Paired Null Range [Min, Max] | Margin / Lift |
|---|---:|---:|---:|---:|
| **Validation ROC-AUC** | **0.8419** | 0.4528 $\pm$ 0.0238 | [0.3954, 0.5178] | **$+0.3891$ lift** ($Z = +16.34\sigma$) |
| **Validation PR-AUC** | **0.4882** | 0.0463 $\pm$ 0.0035 | [0.0391, 0.0543] | **$+0.4419$ lift** (vs. prev 0.0397) |
| **Recall @ False-Alarm Budget** | **0.3898** | 0.0247 $\pm$ 0.0035 | [0.0175, 0.0324] | **$+0.3651$ operational recall** |
| **Precision @ False-Alarm Budget**| **0.6172** | 0.0926 $\pm$ 0.0125 | [0.0674, 0.1205] | **$+0.5246$ precision** |
| **Overall Binary Accuracy** | **0.9703** | ~0.9603 (chance) | — | $+0.0100$ |
| **Binary F1 ($p \ge 0.5$)** | **0.4456** | 0.0000 – 0.0001 | — | $+0.4455$ |

### 4.2 Statistical Significance & Empirical Permutation Test
- **Permutations Exceeding Real Model:** $0$ of $50$ paired null runs achieved an ROC-AUC greater than or equal to $0.8419$.
- **Empirical Permutation $p$-Value:**
  $$\hat{p} = \frac{1 + \sum_{m=1}^M \mathbb{I}(\text{AUC}_{\text{null}, m} \ge \text{AUC}_{\text{real}})}{M + 1} = \frac{1 + 0}{50 + 1} = \frac{1}{51} \approx 0.0196$$
  *(Note: $0.0196$ represents the exact finite-sample resolution limit for $M = 50$ permutations. It is an exploratory empirical bound, not a pre-registered confirmatory significance test).*
- **Parametric Gaussian Separation:** Under the assumption of normality for the null AUC estimator, the genuine model lies at $Z = \frac{0.8419 - 0.4528}{0.0238} = +16.34\sigma$, corresponding to a two-sided tail probability $p_{\text{param}} < 10^{-50}$.

---

## 5. Historical Documentation Reconciliation

### 5.1 Discrepancy 1: Option 3 Upper Bound Arithmetic Error

#### The Issue
In `runs/p4_revalidation/p4_matched_binary_baseline_report.md` Section 7.3, "Option 3: Empirical Bias-Adjusted Acceptance Interval" proposed:
$$\text{Interval} = \mu_{\text{null}} \pm 2.5 \cdot s_{\text{null}} \approx 0.4547 \pm 2.5(0.0263) = [0.3890, 0.5205]$$
The report asserted:
> *"Under this empirically grounded interval, the 5 official seeds (0.5238, 0.4105, 0.5074, 0.4393, 0.4545, mean 0.4671) would pass."*

#### Reconciled Facts & Arithmetic Proof
- **Exact Recorded Seed 20001 Value:** `0.5237649880596127` (rounded to `0.5238` in reports).
- **Upper Bound Calculation:** $\mu + 2.5 \cdot s = 0.454691 + 2.5 \cdot 0.026312 = 0.454691 + 0.065780 = 0.520471 \approx 0.5205$.
- **Discrepancy:** $0.523765 > 0.520471$ ($0.5238 > 0.5205$).
- **Conclusion:** Seed 20001 strictly exceeds the proposed upper bound by $+0.0033$. Under Option 3, only 4 of the 5 official seeds would fall within the interval. The claim that all five pass is mathematically false.
- **Remediation Action:** A dated warning callout has been appended to `runs/p4_revalidation/p4_matched_binary_baseline_report.md` documenting this arithmetic error and reiterating that post-hoc interval fitting is scientifically invalid.

---

### 5.2 Discrepancy 2: `class_weight='balanced'` vs. `class_weight=None`

#### The Issue
In Section 3.1 of `runs/p4_revalidation/p4_revalidation_report.md`, the model configuration was documented as:
> `Default LightGBM binary classifier (n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=seed, class_weight='balanced')`
However, the null-control execution in `src/eval/leakage_audit.py` (lines 267–275) instantiates LightGBM as:
> `lgb.LGBMClassifier(objective="binary", n_estimators=50, learning_rate=0.1, num_leaves=31, random_state=seed, n_jobs=-1, verbosity=-1)` (i.e., `class_weight=None`).

#### Investigation & Evidence
To determine whether this was a prose typo or a code provenance uncertainty, three independent lines of evidence were audited:
1. **Repository Git History:**
   - A search of the entire commit history (`git log -p -S "class_weight" src/eval/leakage_audit.py`) reveals that `class_weight` was **never** present in `src/eval/leakage_audit.py` at any point in time.
   - Commit `9e1e446` (the initial creation of `leakage_audit.py`) instantiated `LGBMClassifier` with `n_estimators=50, learning_rate=0.1, num_leaves=31, random_state=seed`.
2. **Benchmark Code Cross-Reference:**
   - In `src/models/first_benchmark.py` (line 151), the P4.3 multiclass LightGBM benchmark model is configured as:
     `num_class=8, n_estimators=100, learning_rate=0.05, num_leaves=31, class_weight='balanced'`.
   - When drafting `runs/p4_revalidation/p4_revalidation_report.md`, the author copied the hyperparameter description from Section 4.1 (P4.3 benchmark) into Section 3.1 (P4.2 null control).
3. **Empirical Bit-for-Bit Reproduction:**
   Both configurations were executed on Seed 20001 under identical sampled data:
   - **Config 1 (`50, 0.1, None`):** Validation AUC = **`0.5237649880596127`**
   - **Config 2 (`100, 0.05, 'balanced'`):** Validation AUC = **`0.4818568188288174`**
   - **Summary JSON Value (`p4_evaluation_summary.json`):** **`0.5237649880596127`**
   
   The executed code (`50, 0.1, None`) reproduces the official recorded result to **16 decimal places**. Config 2 produces an entirely different number ($0.4819$).

#### Conclusion
The discrepancy was purely a **prose transcription error** in the markdown report. The code executed in `a0d76b2` was identical to `src/eval/leakage_audit.py`. Historical experiment provenance is verified and reproducible. A dated erratum has been added to `runs/p4_revalidation/p4_revalidation_report.md`.

---

## 6. Conservative Interpretation & Technical Limitations

### 6.1 What This Diagnostic Establishes
1. **Internal LightGBM Seed is Not the Cause of Sub-0.50 Drift:**
   Holding `random_state=42` fixed across all 50 null runs produced an almost identical null distribution (mean $0.4528$ vs $0.4547$, correlation $r = 0.9177$). The negative rank correlation on extreme features is an intrinsic property of tree splits on permuted labels with severe class imbalance ($4\%$ positive).
2. **Real Signal Separation is Robust:**
   The real model ($0.8419$) remains $>16\sigma$ away from the null distribution regardless of whether LightGBM's seed is fixed or varied.

### 6.2 What This Diagnostic Does Not Establish (Strict Boundaries)
1. **Official P4.2 Remains FAILED:**
   The official project acceptance requirement defined in `config/eval.yaml` and Appendix A.14 is:
   $$\text{AUC} \in [0.4700, 0.5300] \quad \text{across 5 seeds}$$
   Under paired evaluation, 3 of 5 seeds still fail, and 41 of 50 seeds fail across the distribution. The test is **FAILED** under the governing specification.
2. **No Post-Hoc Threshold Shopping:**
   We strictly reject adjusting the acceptance interval (such as Option 3's $[0.3890, 0.5205]$ or any variant) simply to retroactively force existing experimental results to pass.
3. **Exploratory vs. Confirmatory:**
   The empirical $p$-value ($p \le 0.0196$) is an exploratory post-hoc diagnostic on $M = 50$ permutations. It cannot be retroactively substituted for the formal pre-registered confirmatory pass mark without project leadership authorization.

---

## 7. Artifact Provenance & Reproducibility

| Artifact Description | Path | Hash / Version / Details |
|---|---|---|
| **Diagnostic Script** | `scratch/run_paired_seed_diagnostic.py` | 100% reproducible execution script |
| **Diagnostic JSON Results** | `scratch/paired_seed_diagnostic_results.json` | Detailed per-seed metrics (paired & unpaired) |
| **Feature Tables** | `data/features/{train,val}/sensor_features_scored.parquet` | Corrected P2 feature cache |
| **Ground Truth Tables** | `data/ground_truth/{train,val}/ground_truth_sensors.parquet` | Scored rows ($t \ge 480, t \% 5 == 0$) |
| **Revalidation Report** | `runs/p4_revalidation/p4_revalidation_report.md` | Contains dated erratum on prose typo |
| **Matched Baseline Report**| `runs/p4_revalidation/p4_matched_binary_baseline_report.md` | Contains dated erratum on Option 3 |
| **Active Git Branch / Commit** | `master` / base commit `b493279` | Clean working tree; no production code modified |

---

## 8. Summary Status and Stop Condition

| Phase / Item | Official Status | Diagnostic Status | Notes |
|---|---|---|---|
| **P4.1 (Single-feature separability)** | **PASSED** | Confirmed | 0 features $>0.85$ on binary faults; all 16 multi-class $>0.85$ explained by physics. |
| **P4.2 (Official 5-seed null control)** | **FAILED** | **FAILED** | 3 of 5 seeds fail $[0.4700, 0.5300]$ under both unpaired ($0.4671$) and paired ($0.4621$). |
| **P4.2 (Canary leak test)** | **PASSED** | Confirmed | Canary caught with AUC $1.0000$. |
| **P4.2 (Recall at budget null control)** | **PASSED** | Confirmed | All 50 paired null runs strictly $\le 0.0324$ (budget limit $0.05$). |
| **P4.2 (Paired-Seed Diagnostic)** | **COMPLETED** | Informative | $r = 0.9177$, $\Delta = -0.0019$; proves seed allocation does not cause sub-0.50 drift. |
| **Documentation Reconciliation** | **COMPLETED** | Reconciled | Errata added for Option 3 arithmetic error and Section 3.1 prose typo. |
| **Phase P5 Authorization** | **PAUSED** | **PAUSED** | P5 has NOT been authorized. Test split remains untouched. |

---

**FINAL EXPLICIT STATUS:**  
`P4.2 REMAINS FAILED UNDER THE EXISTING CRITERION. P5 HAS NOT BEEN AUTHORIZED.`
