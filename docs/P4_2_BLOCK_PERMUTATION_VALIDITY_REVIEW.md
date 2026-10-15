# P4.2 Block-Permutation Methodology Review

**Document:** `docs/P4_2_BLOCK_PERMUTATION_VALIDITY_REVIEW.md`  
**Date:** 2026-10-16 02:10 UTC  
**Project:** Trust Before Alarm — Generator A  
**Workspace:** `D:\CODE_APEX_HACKATHON`  
**Target Commit Reviewed:** `c97b550`  
**Test Suite Verification:** Exit code 0, 238 passed in 151.38s on `c97b550`  
**Status:** Read-Only Review & Evidence-Gathering. P4.2 remains officially **FAILED**. Phase P5 is **NOT** authorized.

---

## Executive Summary & Final Verdict

This document provides a formal, evidence-based methodological review of the exploratory block-permutation null control executed in Step 6 of the P4.2 consolidated investigation (`scratch/run_block_permutation.py`, results in `scratch/block_permutation_results.json`).

### Review Verdict

```
BLOCK PERMUTATION VALIDITY NOT ESTABLISHED
```

### Key Rationale
1. **Physical Explanatory Power:** The block-permutation experiment demonstrably proved that preserving within-sensor temporal autocorrelation and contiguous episode duration eliminates the severe downward rank-inversion bias observed under row-wise label shuffling (shifting the null mean from $0.4528$ up to $0.5209$).
2. **Trajectory Length Discrepancy:** The earlier report's claim of "289-step sensor trajectories" was an off-by-one arithmetic error. Scored trajectories ($t \ge 480$ and $t \equiv 0 \pmod 5$) terminate at $t = 1915.0$ (not $1920.0$), yielding exactly **288** scored steps per sensor across all 5,319 training blocks and 1,780 validation blocks.
3. **Macro-Temporal Hazard Retention:** The observed block-null mean of $0.520882$ (+3.27 standard errors above $0.50$) is directly explained by the simulation's non-stationary temporal hazard rate. Fault episodes occur predominantly in the second half of scenarios ($t \ge 1000$ has double the fault frequency of $t < 1000$, $r = +0.9691$ between train and val). Because the block permutation preserved the simulation-time alignment of donor trajectories ($t = 480..1915$), LightGBM learned weak proxies for simulation time from time-trending features. Timestamp alone achieves validation ROC-AUC = **0.598378**; predicting the training temporal hazard curve achieves **0.596033**.
4. **Exchangeability Violations:** The algorithm performed a global unconstrained permutation across all 900 scenarios and 12 sensors. In 99.88% of assignments, trajectories crossed scenarios; in 86.76%, they crossed sensor identities. This mixed sensors with zero baseline faults (`sensor_11`) with high-fault sensors (`sensor_08`, 6.7%) and decoupled peer-consensus features from scenario-specific physics.
5. **Resolution Limits:** With $M = 50$ permutations, the empirical p-value resolution is strictly bounded by $\hat{p} = (0 + 1) / (50 + 1) = 1/51 \approx 0.0196$. The standardized separation $Z = +7.11\sigma$ is a descriptive metric in sample SD units, not a calibrated parametric Gaussian Z-test.

---

## Categorized Findings

### CONFIRMED FACTS
1. **Data Isolation:** Zero access to held-out test data occurred. Exactly four data files were read by the diagnostic scripts:
   - `data/features/train/sensor_features_scored.parquet`
   - `data/ground_truth/train/ground_truth_sensors.parquet`
   - `data/features/val/sensor_features_scored.parquet`
   - `data/ground_truth/val/ground_truth_sensors.parquet`
2. **Trajectory Length:** Every sensor trajectory in both training and validation consists of exactly **288** scored timestamps ($t \in [480.0, 1915.0]$, step 5.0). The full raw timeline contains **1,920** timestamps ($t \in [0.0, 1919.0]$, step 1.0).
3. **Trajectory Block Count:** The training set contains exactly **5,319** distinct `(scenario_id, sensor_id)` sensor blocks ($5,319 \times 288 = 1,531,872$ scored rows). The validation set contains **1,780** blocks ($1,780 \times 288 = 512,640$ scored rows).
4. **Permutation Scope:** In `scratch/run_block_permutation.py`, permutations were global across all 5,319 blocks:
   - Mean blocks assigned within the same scenario: 6.36 / 5,319 (**0.12%**).
   - Mean blocks assigned to the same `sensor_id`: 704.44 / 5,319 (**13.24%**).
   - Total training positive labels ($63,716$) were 100% conserved in full training before subsampling.
5. **Validation Integrity:** Validation labels were strictly untouched and evaluated using true ground truth ($N_{\text{val}} = 512,640$, $N_{\text{pos}} = 20,368$).
6. **Temporal Hazard Signal:** The probability of a sensor being untrusted rises monotonically over simulation time ($0.8\%$ at $t=480$ to $6.2\%$ at $t=1915$). Timestamp alone achieves validation ROC-AUC = **0.598378**; predicting the training marginal hazard $P(Y=1 \mid t)$ achieves **0.596033**.
7. **Empirical Distribution:** Across 50 block permutations:
   - Mean AUC: **0.520882**
   - Sample SD ($ddof=1$): **0.045175** (SE: $0.006389$)
   - Range: **[0.400891, 0.623103]**
   - Real-label binary baseline AUC: **0.841920** (separation: $+0.321038$ above null mean, $+0.218817$ above null maximum).
   - Exact empirical p-value: $\hat{p} = (0 + 1) / (50 + 1) = \mathbf{1/51 \approx 0.0196}$.

### HYPOTHESES
1. **Sub-0.50 Curing Mechanism:** Preserving within-sensor temporal autocorrelation and contiguous episode duration eliminates the micro-scale rank inversion caused by row-wise shuffling on sparse healthy outliers.
2. **Upward Null Drift Mechanism:** The block-null mean of $0.520882$ is primarily driven by LightGBM picking up macro-temporal simulation progression from time-correlated features (e.g., dispersion, missingness growth, drift) because donor trajectories retained their original timestamp alignment ($t=480..1915$).

### UNRESOLVED
1. Whether an alternative block-null construction (e.g., circular time-shifting within scenario, or within-sensor-type permutation with time-detrending) can produce a mean closer to $0.5000$.
2. Formal acceptance criteria: whether project leadership will adopt a contrastive signal-separation criterion or retain the strict $[0.4700, 0.5300]$ absolute band.

---

## Step 1: Verification of the Exact Block-Permutation Procedure

### 1.1 Algorithm Specification in `scratch/run_block_permutation.py`

The block-permutation script executes the following procedure:

```python
# 1. Sort training rows to guarantee deterministic timestamp order
train_merged = train_merged.sort_values(
    by=['scenario_id', 'sensor_id', 'timestamp']
).reset_index(drop=True)

# 2. Extract sensor trajectory blocks: 5,319 blocks of (scenario_id, sensor_id)
sensor_group_indices = train_merged.groupby(['scenario_id', 'sensor_id']).indices
sensor_list = list(sensor_group_indices.keys())
sensor_trajectories = {
    s_key: train_merged.loc[sensor_group_indices[s_key], 'sensor_untrusted'].to_numpy()
    for s_key in sensor_list
}

# 3. For each permutation seed in range(20001, 20051):
rng = np.random.default_rng(seed)
permuted_order = rng.permutation(len(sensor_list))

# 4. Assign donor trajectory to destination sensor block
y_train_block = np.zeros(len(train_merged), dtype=int)
for orig_idx, perm_idx in enumerate(permuted_order):
    orig_s_key = sensor_list[orig_idx]
    perm_s_key = sensor_list[perm_idx]
    row_indices = sensor_group_indices[orig_s_key]
    donor_traj = sensor_trajectories[perm_s_key]
    y_train_block[row_indices] = donor_traj

# 5. Subsample 300,000 rows with random_state=42 and fit LightGBM
train_sampled = train_block_df.sample(n=300000, random_state=42).reset_index(drop=True)
clf = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=50,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)
clf.fit(X_train_b, y_train_b)

# 6. Evaluate on untouched validation set with TRUE labels
val_probs = clf.predict_proba(X_val)[:, 1]
auc_b = float(roc_auc_score(y_val, val_probs))
```

### 1.2 Protocol Audit Table

| Audit Item | Protocol In Implementation | Verified Finding | Evidence Command & Output |
|---|---|---|---|
| **Trajectory Definition** | Scored label vector for 1 `(scenario_id, sensor_id)` block | Exact 288-step 1D array of `sensor_untrusted` | `unique=[288]` across all 5,319 blocks |
| **Source / Destination** | Global permutation of `sensor_list` | Assigned across all scenarios and sensors | Seed 20001: 0.23% same scen, 14.48% same sensor |
| **Scenario Identity** | Not preserved | Broken: 99.88% crossed scenario boundaries | Mean across 50 seeds: 6.36 / 5,319 (0.12%) same scen |
| **Sensor Identity** | Not preserved | Broken: 86.76% crossed sensor IDs | Mean across 50 seeds: 704.44 / 5,319 (13.24%) same sensor |
| **Variable Type** | Implicitly all temperature | All 12 sensors are temperature | `raw_train['variable_type'].unique() == ['temperature']` |
| **Internal Order** | Intact 1D vector assignment | Preserved: 100% sequential timestamp order | Assigned directly to sorted `row_indices` |
| **Episode Duration** | Preserved | Preserved: contiguous runs of 1s untouched | Vector copied without shuffling elements |
| **Positive Count** | Conserved in full training | 63,716 positives conserved exactly | `y_perm.sum() == 63716` (exact match: True) |
| **Validation Labels** | Untouched true ground truth | 100% untouched true validation labels | $N_{\text{val}} = 512,640$, $N_{\text{pos}} = 20,368$ |
| **Feature Set** | Identical 49 features | Exact match: `SENSOR_FEATURE_NAMES` | 49 observable features from `engine.py` |
| **Model Parameters** | Fixed `random_state=42` | Identical: `n_est=50, lr=0.1, leaves=31` | Freshly built model matches baseline |
| **Files Read** | Train & Val only | Zero test split access verified | See Section 1.3 below |

### 1.3 Exact Files Read During Diagnostic
The diagnostic executed in `scratch/run_block_permutation.py` read strictly these four files:
1. `data/features/val/sensor_features_scored.parquet` (43,153,303 bytes)
2. `data/ground_truth/val/ground_truth_sensors.parquet` (22,016,779 bytes)
3. `data/features/train/sensor_features_scored.parquet` (124,707,877 bytes)
4. `data/ground_truth/train/ground_truth_sensors.parquet` (69,895,862 bytes)

Zero test split files (`data/raw/test/`, `data/features/test/`, `data/ground_truth/test/`) were accessed or loaded.

---

## Step 2: Resolution of the Trajectory-Length Question

### 2.1 The 289 vs 288 Discrepancy
Earlier prose in `docs/P4_2_CONSOLIDATED_REPORT.md` and comments in `scratch/run_block_permutation.py` described "289-step sensor trajectories (t=480..1920, stride 5)".

Verification via actual data inspection reveals:

```powershell
.venv\Scripts\python.exe scratch/verify_block_permutation.py
```
**Actual Real Output:**
```
--- TRAIN GROUND TRUTH ---
Total train GT rows: 10,212,480
Train scenarios: 900
Train unique sensors: 12
Train timestamp min: 0.0, max: 1919.0, nunique: 1920

Train GT scored (t>=480, t%5==0) rows: 1,531,872
Train scored timestamps min: 480.0, max: 1915.0, nunique: 288

--- TRAJECTORY LENGTHS (TRAIN) ---
Full GT trajectory length summary: min=1920, max=1920, unique=[1920]
Scored GT trajectory length summary: min=288, max=288, unique=[288]

--- TRAIN FEATURES SCORED ---
Total train feat rows: 1,531,872
Train feat scenarios: 900
Train feat unique sensors: 12
Train feat timestamp min: 480, max: 1915, nunique: 288
Train feat trajectory length summary: min=288, max=288, unique=[288]
Merged train rows: 1,531,872
Merged train trajectory lengths: min=288, max=288, unique=[288]
```

### 2.2 Mathematical Explanation
1. **Raw Full Timeline:** Scenarios run from $t = 0.0$ to $t = 1919.0$ at 1-second resolution, containing exactly **1,920** timestamps.
2. **Scored Timeline:** Scored rows require $t \ge 480$ and $t \equiv 0 \pmod 5$.
   - The first scored timestamp is $t = 480.0$.
   - The maximum timestamp in the scenario is $1919.0$.
   - The largest multiple of 5 in $[480, 1919]$ is $t = 1915.0$ (not $1920.0$).
   - Number of steps:
     $$\frac{1915 - 480}{5} + 1 = \frac{1435}{5} + 1 = 287 + 1 = \mathbf{288}$$
3. **Execution Consequence:** In `scratch/run_block_permutation.py`:
   ```python
   if len(donor_traj) == len(row_indices):
       y_train_block[row_indices] = donor_traj
   ```
   Because every sensor trajectory in the dataset has length exactly 288, `len(donor_traj) == len(row_indices)` evaluated to `True` on 100% of blocks. No truncation or zero-padding ever took place. The earlier text was an off-by-one arithmetic error assuming $t=1920$ was present.

---

## Step 3: Validity Assessment of the Block Null and Exchangeability

### 3.1 Exchangeability Under the Null Hypothesis
A permutation test of the null hypothesis $H_0: P(Y \mid X) = P(Y)$ requires that under $H_0$, the assignment of labels to feature vectors is governed by an exchangeable probability distribution.
- **In Row-Wise Permutation:** Rows are assumed i.i.d. However, real sensor data contains strong temporal autocorrelation and multi-step fault episodes (average 48.7 rows/episode). Row-wise shuffling shatters this autocorrelation, producing artificial high-frequency noise labels that lead LightGBM to overfit healthy feature outliers, causing the observed sub-0.50 rank inversion (mean 0.4528).
- **In Block Permutation:** Intact trajectories are exchanged, preserving temporal autocorrelation and episode duration.

### 3.2 Demonstrable Behavior vs Statistical Appropriateness

| Dimension | What the Implementation Demonstrably Does | Statistical Appropriateness Assessment |
|---|---|---|
| **Temporal Autocorrelation** | Preserves 288-step sequential trajectory and contiguous episode durations intact. | **Appropriate:** Completely cures the severe downward rank-inversion bias of row-wise shuffling. |
| **Simulation Time Alignment** | Maps donor trajectory at identical simulation timestamps ($t = 480..1915$). | **Defective / Confounded:** Retains the macro-temporal hazard rate $P(Y=1 \mid t)$, allowing decision trees to exploit time-trending features. |
| **Scenario Boundaries** | Permutes globally across all 900 scenarios (99.88% cross-scenario). | **Partially Appropriate:** Decouples labels from genuine physical events in scenario B, but assigns labels to unrelated physical transient features. |
| **Sensor Identity** | Permutes across all 12 sensors (86.76% cross-sensor). | **Inappropriate:** Injects faults into sensors with zero baseline failure rate (`sensor_11`), distorting sensor-specific marginal distributions. |

### 3.3 Root Cause of the 0.520882 Mean (+3.27 SE Above 0.50)

In Step 6 of the consolidated report, the 50 block permutations yielded a mean AUC of **0.520882** with sample standard error **0.006389**, which is:
$$\frac{0.520882 - 0.500000}{0.006389} = \mathbf{+3.27\text{ standard errors above } 0.50}$$

To investigate this without assuming the cause, we examined the ground truth fault timing across simulation time:

```powershell
.venv\Scripts\python.exe scratch/investigate_fault_timing.py
```
**Actual Real Output:**
```
--- FAULT TIMING DISTRIBUTION ---
Train positive rate by timestamp: min=0.0083, max=0.0619, mean=0.0416
Val positive rate by timestamp: min=0.0079, max=0.0629, mean=0.0397
Correlation between train and val positive rates across time: 0.9691
Train positive rate: early (t<1000) = 0.0238, late (t>=1000) = 0.0516
Val positive rate: early (t<1000) = 0.0246, late (t>=1000) = 0.0483

--- FAULT EPISODE START TIMES ---
Train episodes count: 1344
Val episodes count: 445
Train episode start time summary: mean = 1145.77, std = 541.94, min = 0, 50% = 1198.0
Val episode start time summary:   mean = 1114.73, std = 525.74, min = 0, 50% = 1161.0
```

We then tested the predictive power of simulation time alone on the validation set:

```powershell
.venv\Scripts\python.exe scratch/test_time_hazard.py
```
**Actual Real Output:**
```
ROC-AUC on validation using timestamp alone as predictor: 0.598378
ROC-AUC on validation using train empirical P(Y=1 | timestamp): 0.596033

Top features correlated with timestamp:
  group_dispersion         : r = +0.0849
  others_spread            : r = +0.0796
  dev_duration             : r = -0.0274
  frac_missing_l           : r = +0.0258
  frac_missing_m           : r = +0.0176
  group_change_m           : r = +0.0160
  longest_gap_m            : r = +0.0150
```

### 3.4 Findings on the Upward Drift
1. **Timestamp Alone Has Discriminatory Power:** In both train and val, fault episodes concentrate in the second half of scenarios ($t \ge 1000$ has double the positive rate of $t < 1000$). The temporal hazard correlation between train and val is **$r = +0.9691$**. Using timestamp alone as a score achieves validation ROC-AUC = **0.598378**.
2. **Feature Leakage of Simulation Time:** Because donor trajectories remain pinned to their exact simulation timestamps ($t = 480..1915$), the permuted training set retains this macro-temporal hazard curve. Decision trees split on features that drift or accumulate with time (`group_dispersion`, `others_spread`, missingness, duration), effectively learning a weak proxy for simulation time.
3. **Conclusion on Mean 0.5209:** The block-null mean of $0.520882$ is not a statistical anomaly; it is the predictable consequence of preserving the non-stationary temporal hazard profile in a simulation where late-scenario timestamps have higher fault rates.

---

## Step 4: Audit of the Interpretation & Statistical Rigor

### 4.1 Empirical Separation vs Resolution Limits

```powershell
.venv\Scripts\python.exe scratch/verify_summary_print.py
```
**Actual Real Output:**
```
Mean AUC: 0.520882
Sample SD (ddof=1): 0.045175
Population SD (ddof=0): 0.044721
SE of Mean: 0.006389
Min AUC: 0.400891
Max AUC: 0.623103
Median AUC: 0.513772
Q2.5% AUC: 0.434533
Q97.5% AUC: 0.602451
Pass count: 24 out of 50
Real AUC: 0.841920
Diff: 0.321038
Z-score: 7.11
Permutations >= Real: 0
Empirical p-value: 0.0196
```

### 4.2 Critical Corrections to Statistical Claims
1. **Empirical P-Value Resolution:** In 50 permutations, exactly 0 permutations produced an AUC greater than or equal to the genuine model's AUC ($0.841920$). Under standard non-parametric permutation testing, the empirical p-value is:
   $$\hat{p} = \frac{r + 1}{M + 1} = \frac{0 + 1}{50 + 1} = \mathbf{\frac{1}{51} \approx 0.019608}$$
   **Correction:** It is invalid to report or imply an empirical p-value smaller than $1/51 \approx 0.0196$ (such as $p < 0.001$ or $p = 0.0000$) from a 50-permutation sample.
2. **Status of the Z-Score:** The standardized distance:
   $$Z = \frac{0.841920 - 0.520882}{0.045175} = \mathbf{+7.11\sigma}$$
   is a descriptive measure of distance in units of sample standard deviation.
   **Correction:** This must **not** be called a "formally calibrated Z-test" because:
   - The underlying permutation distribution across clustered time series is not established to be Gaussian.
   - Permutation iterations are mutually dependent (drawn from a finite permutation group).
   - The sample size ($M = 50$) is insufficient to calibrate tail probabilities at $7\sigma$.
3. **Data Leakage Terminology:** Absolute claims such as "the model is proven free of data leakage" are scientifically overbroad.
   **Correction:** All statements must be strictly grounded in the specific audits performed:
   - **Canary Leak Audit:** PASSED with AUC = 1.0000 on injected synthetic leak.
   - **Scenario Isolation:** Verified strictly disjoint scenario IDs (900 train vs 300 val).
   - **Feature Engine Causality:** Verified causal rolling windows with zero forward-looking operations.
   - **Validation Ground Truth:** Verified untouched true labels ($N_{\text{val}} = 512,640$, $N_{\text{pos}} = 20,368$).
   - No evidence of label leakage was detected across the audited pipelines.

---

## Step 5: Test Suite & Codebase Integrity

The full test suite was executed to completion on commit `c97b550`:

```powershell
.venv\Scripts\pytest -q
```
**Actual Real Output:**
```
........................................................................ [ 30%]
........................................................................ [ 60%]
........................................................................ [ 90%]
......................                                                   [100%]
238 passed in 151.38s (0:02:31)
```

**Audit Status:**
- Exit code: **0**
- Passed: **238**
- Failed: **0**
- Errors: **0**
- Duration: **151.38s**

---

## Methodological Summary & Recommendations

### Summary of What Was Learned
1. **Row-wise shuffling was flawed:** It decorrelated rare positive labels within continuous time-series fault episodes, creating extreme high-frequency label noise that triggered tree rank inversion on sparse healthy outliers (mean 0.4528).
2. **Block-permutation cured the downward bias:** By preserving contiguous episode durations, the null mean rose from $0.4528$ to $0.5209$.
3. **Block-permutation introduced an upward bias:** By maintaining strict simulation-clock alignment ($t = 480..1915$), the null models learned the real macro-temporal hazard rate ($P(Y=1 \mid t)$, which rises from 0.8% to 6.2%), shifting the mean to $0.5209$ (+3.27 SE above 0.50).
4. **Exchangeability is violated:** Unconstrained cross-scenario and cross-sensor swapping violates exchangeability by mixing heterogeneous sensor failure rates and decoupling scenario-level physical transients from fault labels.

### What Is Needed Before a Block-Permutation Null Could Be Formalized
If project leadership wishes to consider a block-permutation null control for a future specification amendment, the following methodological refinements would be strictly required:
1. **Circular Time-Shifting / Phase Randomization:** Shift label trajectories circularly in time within each sensor to destroy the macro-temporal hazard alignment while preserving autocorrelation and duration.
2. **Stratified Sensor-Type Assignment:** Restrict swaps within identical physical sensor locations/types to preserve baseline failure rates.
3. **Within-Scenario Permutation:** Evaluate permutations strictly within contemporaneous scenario sensor groups where applicable.
4. **Time-Detrended Evaluation:** Evaluate models conditioned on time-in-scenario to isolate pure sensor-level fault discrimination from global temporal hazard.

---

## Governance & Phase Authorization

> [!CAUTION]
> **MANDATORY GOVERNANCE DIRECTIVE**
> 1. Under the existing, unamended acceptance criterion defined in `config/eval.yaml` ($0.4700 \le \text{AUC} \le 0.5300$ across official seeds 20001–20005), **Phase P4.2 remains officially FAILED**.
> 2. The block-permutation null control was an exploratory diagnostic and **does not alter the official acceptance status**.
> 3. **Phase P5 has NOT been authorized** and must remain on hold until formal project-lead resolution is signed in `PROGRESS.md`.

---
*Report completed and verified on commit `c97b550` (working tree clean).*
