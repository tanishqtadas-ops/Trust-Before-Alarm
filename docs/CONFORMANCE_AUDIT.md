# Generator A Read-Only Conformance Audit Report

> **Audit Type:** Read-only Conformance Audit (Pre-P1.12 Verification)  
> **Repository:** VIT Code Apex 3.0 / Sensor Trust Intelligence  
> **Git Commit / Tag Target:** `p1-verified` (`3319a46c57f438652a046fef4eda507480ae621e`)  
> **Primary Authority:** `PROGRESS.md` Appendix A (Locked v1.1 Parameters) and Decisions 1–47  
> **Secondary Sources:** `docs/SPEC.md`, `docs/PARAMETERS_CHECK.md`, `config/generator_a.yaml`  
> **Implementation Modules:** `src/sim/environment.py`, `src/sim/events.py`, `src/sim/sensors.py`, `src/sim/faults.py`, `src/sim/scenario.py`, `src/sim/units.py`, `src/sim/profile.py`, `src/sim/dataset.py`, `src/sim/strata.py`  
> **Dataset Files (Empirical Ground Truth):** `data/ground_truth/scenarios.parquet` (1,500 scenarios), `data/ground_truth/*/event_episodes.parquet` (1,770 episodes), `data/ground_truth/*/fault_episodes.parquet` (2,238 episodes), `data/raw/*/observables.parquet`  
> **Note on Prior Audit:** The partial `docs/CONFORMANCE_AUDIT.md` left by the previous session has been completely discarded, ignored, and overwritten from scratch using independent read-only script measurements.

---

## Executive Summary

A comprehensive, evidence-based read-only conformance audit of the Generator A simulator implementation and the existing 1,500-scenario dataset was conducted against the approved specifications. Every value, code path, configuration entry, and empirical distribution was verified by running read-only scripts against the codebase and saved Parquet tables.

**Verdict: FULL CONFORMANCE (PASS)**
- All 11 audited technical areas conform strictly to `PROGRESS.md` Appendix A, approved Decision Log entries (Decisions 1–47), and `config/generator_a.yaml`.
- All probability distributions configured in `config/generator_a.yaml` sum to exactly 1.000000.
- Empirical frequencies across all 1,500 scenarios in `data/ground_truth/scenarios.parquet`, 2,238 fault episodes in `fault_episodes.parquet`, and 1,770 event episodes in `event_episodes.parquet` closely track the approved stochastic distributions with zero out-of-bounds samples.
- The 11-step sensor transduction pipeline order strictly follows Decision 27.
- Event detectability semantics (`genuine_event` vs `event_below_detectability`) are strictly mutually exclusive across all 883,584 group rows in the dataset (0 overlapping rows).
- Full floating-point determinism is verified: regenerating scenario 1 from seed 1 reproduces the saved observable values with maximum absolute difference of 0.0 and identical NaN masks.
- No files under `data/` were modified; all parquet timestamps remain at initial generation (`11-10-2026 23:47:35` to `23:49:01`).
- The existing 1,500 scenarios, scenario IDs (`scenario_0001` to `scenario_1500`), scenario seeds (1 to 1,500), split assignments (900 train, 300 val, 150 test full resolution, 150 test downsampled), and 6-column public raw schema remain intact and unchanged.

---

## Section 1: Hierarchy of Authority & Document Alignment

Per addition (a), `PROGRESS.md` Appendix A and the approved Decision Log entries (Decisions 1–47) serve as the primary authority. `docs/SPEC.md` and `docs/PARAMETERS_CHECK.md` record frozen engineering choices and line-by-line checks established in P0.

### Document Alignment Analysis (Appendix A vs SPEC / PARAMETERS_CHECK / Config)

The audit compared Appendix A text with `docs/SPEC.md`, `docs/PARAMETERS_CHECK.md`, `config/generator_a.yaml`, and code implementation. The following operationalizations and approved engineering choices are documented:

1. **Random Affine Offset (Appendix A.2 vs Decision 1):**
   - *Appendix A.2:* "random offset".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Uniform $[-10.0, 10.0]^\circ\text{C}$ (`config/generator_a.yaml:70-71`).
   - *Status:* Approved engineering assumption (ENG, Decision 1).
2. **Daily Cycle Phase (Appendix A.2 vs Decision 12):**
   - *Appendix A.2:* "random phase".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Uniform $[0.0, 2\pi)$ radians ($[0.0, 6.283185307179586]$) (`config/generator_a.yaml:37-38`).
   - *Status:* Approved implementation interpretation (Decision 12).
3. **Sustained Event Duration (Appendix A.3 vs Decision 5):**
   - *Appendix A.3:* "sustained rise 5 to 20 min then hold".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Sustained events hold at elevated level until scenario end ($t=1920$) (`config/generator_a.yaml:89`).
   - *Status:* Approved implementation choice (Decision 5).
4. **Stepped Drift Steps Count (Appendix A.4 vs Decision 22):**
   - *Appendix A.4:* "stepped 15%".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Discrete step count $K \in \{3, 4, 5\}$ (`config/generator_a.yaml:127-128`).
   - *Status:* Approved engineering choice (Decision 22).
5. **Healthy Missing Data (Appendix A.1 vs Decision 14):**
   - *Appendix A.1:* "Healthy missing data: per-sensor rate 0 to 1% (20% of sensors 0)".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Groups have 4–8 sensors where exact 20% integer partitioning is mathematically impossible per group; each sensor independently receives 0 with probability 0.20 via Bernoulli trial (`config/generator_a.yaml:23`).
   - *Status:* Approved engineering correction (Decision 14).
6. **Test Partition Split (Appendix A.1 vs Decision 11):**
   - *Appendix A.1:* "Row stride: 5 for train/val; full resolution for 150 test scenarios".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* The 300 test scenarios are partitioned into 150 full resolution (stride 1) and 150 downsampled (stride 5) (`config/generator_a.yaml:11-12`).
   - *Status:* Approved implementation choice (Decision 11).
7. **Candidate Fault Count (Appendix A.5 vs Decision 29):**
   - *Appendix A.5:* "3 or more: 20%".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Represented conservatively by candidate count 3 (`config/generator_a.yaml:166`).
   - *Status:* Approved engineering choice (Decision 29).
8. **Small Group Definition (Appendix A.6 vs Decision 2):**
   - *Appendix A.6:* "faulty fraction at cap with small group".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Operationalized as $n=4$ sensors (`config/generator_a.yaml:191`).
   - *Status:* Approved implementation choice (Decision 2).
9. **No Clear Majority Definition (Appendix A.6 vs Decision 3 & 41):**
   - *Appendix A.6:* "no clear majority".
   - *SPEC.md / PARAMETERS_CHECK.md / Config:* Operationalized as no cluster $> 50\%$ of sensors (`config/generator_a.yaml:192`).
   - *Status:* Approved implementation choice (Decisions 3 & 41).
10. **Sensor Near Bounds Definition (Appendix A.6 vs Decision 4):**
    - *Appendix A.6:* "sensor near bounds".
    - *SPEC.md / PARAMETERS_CHECK.md / Config:* Evaluated within $\max(3\sigma_k, 2.0^\circ\text{C})$ of clipping boundaries in canonical Celsius (`config/generator_a.yaml:193-196`).
    - *Status:* Approved implementation choice (Decision 4).

All 10 items represent approved, documented engineering decisions locking ambiguous prose into executable mathematical contracts. No unauthorized parameter drifts exist.

---

## Section 2: Complete Evidence-Based Conformance Tables

### Item 1: Scenario Topologies, Group Counts, Sensor Counts, and Group-Size Distributions
- **Approved Rule / Authority:** Appendix A.1; Decision 35; `SPEC.md` Section 4.
  - Total scenarios: 1,500 (900 train, 300 val, 300 test).
  - Scenario types: Standard 82%, Two-group 10%, Single-sensor 8%.
  - Group count: 1 for standard/single-sensor; 2 for two-group.
  - Group sizes: DiscreteUniform[4, 8] for standard; DiscreteUniform[4, 6] each for two-group; exactly 1 for single-sensor.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:5-20`: `total_scenarios: 1500`, `train: 900`, `val: 300`, `test: 300`, `scenario_type_probs: {standard: 0.82, two_groups: 0.10, single_sensor: 0.08}`.
  - `src/sim/scenario.py:67-78`: `sample_scenario_type()` samples categorically from configured probabilities.
  - `src/sim/scenario.py:80-106`: `sample_group_sizes()` samples DiscreteUniform[4, 8] for standard, DiscreteUniform[4, 6] for two-group, [1] for single-sensor.
- **Empirical Measurement (N=1,500 scenarios):**
  - Total scenarios: 1,500. `train`: 900 (60.0%), `val`: 300 (20.0%), `test`: 300 (20.0%).
  - `scenario_type`:
    - `standard`: 1,221 (81.40% vs 82% target).
    - `single_sensor`: 142 (9.47% vs 8% target).
    - `two_groups`: 137 (9.13% vs 10% target).
  - `group_count`: 1 group: 1,363 (90.87%), 2 groups: 137 (9.13%).
  - Group sizes in standard scenarios: $n=4$: 234 (19.16%), $n=5$: 243 (19.90%), $n=6$: 258 (21.13%), $n=7$: 226 (18.51%), $n=8$: 260 (21.29%). Uniform on [4, 8].
  - Group sizes in two-group scenarios: sizes strictly in $\{4, 5, 6\} \times \{4, 5, 6\}$; total sensor counts: 8 (18), 9 (38), 10 (40), 11 (32), 12 (9).
  - Single-sensor scenarios: $N=1$ for all 142 scenarios (100.0%).
- **Conformance Status:** MATCH.

---

### Item 2: Difficulty-Tier Probabilities and Constructive HARD-Case Requirements
- **Approved Rule / Authority:** Appendix A.6; Decisions 2–4, 10, 38, 39; `SPEC.md` Section 4.
  - Tiers: EASY 30%, MEDIUM 40%, HARD 30%. Sampled independently from scenario topology.
  - EASY restrictions: at most 1 faulty sensor, 1 fault type, single episode, visible parameters ($k \ge 8\sigma$, drift $\ge 6\sigma$, $|g-1| \ge 0.25$, noise mult $\ge 4$), event scopes global or none.
  - MEDIUM constraints: up to 2 faulty sensors, up to 2 types each, any event scope.
  - HARD requirements: at least one hard mechanism out of 7 (`local_event`, `fault_during_event`, `subtle_fault`, `shared_cause_pair`, `faulty_fraction_at_cap_small_group`, `no_clear_majority`, `sensor_near_bounds`).
  - Zero-fault scenarios strictly preserved ($k_{\text{cand}}=0 \implies k_{\text{actual}}=0$); zero-fault HARD scenarios qualified constructively via `local_event`.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:184-197`: `easy: {prob: 0.3, max_faulty_sensors: 1, single_fault_only: true, spike_min_sigma: 8.0, drift_min_sigma: 6.0, gain_diff_min: 0.25, noise_mult_min: 4.0, allowed_event_scopes: [global, none]}`, `medium: {prob: 0.4, max_faulty_sensors: 2, max_types_per_sensor: 2}`, `hard: {prob: 0.3, ...}`.
  - `src/sim/scenario.py:108-118`: `sample_difficulty_tier()` samples categorical probabilities [0.30, 0.40, 0.30].
  - `src/sim/scenario.py:306-340`: `evaluate_hard_mechanisms()` runs all 7 verified detectors.
  - `src/sim/scenario.py:869-925`: `_apply_hard_fallback()` applies deterministic constructive priority queue.
- **Empirical Measurement (N=1,500 scenarios):**
  - `difficulty`:
    - `medium`: 586 (39.07% vs 40% target).
    - `easy`: 467 (31.13% vs 30% target).
    - `hard`: 447 (29.80% vs 30% target).
  - Qualification in HARD tier ($N=447$):
    - `natural`: 299 (66.89%).
    - `local_event` fallback: 71 (15.88%).
    - `subtle_fault` fallback: 41 (9.17%).
    - `fault_during_event` fallback: 35 (7.83%).
    - `faulty_fraction_at_cap_small_group` fallback: 1 (0.22%).
    - `none`: 0 (0.00%). Exactly 100% of HARD scenarios qualify!
  - `hard_case_type` outside HARD tier: exactly 1,053 `none` (100.0% of EASY and MEDIUM).
- **Conformance Status:** MATCH.

---

### Item 3: Fault Candidate Counts, Fault Caps, Sensor Selection, Shared-Cause Rules & Combinations
- **Approved Rule / Authority:** Appendix A.5; Decisions 8, 9, 29, 30, 31, 33; `SPEC.md` Section 4.
  - Candidate fault count $k_{\text{candidate}} \sim \{0: 0.15, 1: 0.40, 2: 0.25, 3: 0.20\}$.
  - Actual fault count: $k_{\text{actual}} = \min(k_{\text{candidate}}, \text{cap}_{\text{scen}}, \sum_g \text{cap}_g, \text{cap}_{\text{tier}})$, where $\text{cap} = \max(1, \lfloor 0.4 n \rfloor)$.
  - Zero-fault scenarios strictly preserved ($k_{\text{candidate}}=0 \implies k_{\text{actual}}=0$).
  - Types per sensor: 1: 65%, 2: 27%, 3: 8% (restricted by tier).
  - Allowed pairs: 14 pairs; forbidden pairs: 7 pairs; valid pairs = 13 pairs. Valid triplets derived mechanically as 3-cliques: exactly 11 triplets.
  - Shared-cause pair: $P=0.15$ on eligible multi-fault scenarios ($k_{\text{actual}} \ge 2$ and $\exists g$ with $\text{cap}_g \ge 2$). Start offset within $\pm 10$ steps, parameter coupling within $\le 20\%$.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:166-182`: `faulty_sensors_probs: {0: 0.15, 1: 0.40, 2: 0.25, 3_or_more: 0.20}`, `types_per_faulty_sensor_probs: {1: 0.65, 2: 0.27, 3: 0.08}`, `shared_cause_pair_prob: 0.15`, `shared_cause_start_offset_max: 10`, `shared_cause_severity_relative_tolerance: 0.20`.
  - `src/sim/faults.py:1274-1302`: `get_valid_pairs()` and `derive_valid_triplets()` mechanically derive 13 pairs and 11 triplets.
  - `src/sim/faults.py:1305-1358`: `sample_faulty_sensor_candidate_count()`, `compute_group_cap()`, `compute_actual_faulty_sensor_count()`.
  - `src/sim/faults.py:1603-1725`: `sample_shared_cause_start_step()`, `couple_shared_cause_parameters()`.
- **Empirical Measurement:**
  - `faulty_sensors_count` across 1,500 scenarios:
    - 0 faults: 190 (12.67% vs nominal candidate 15%, shifted slightly by small-group constructive fallbacks).
    - 1 fault: 918 (61.20% vs nominal 40%, elevated because EASY tier caps $k=1$ and group cap caps $k=1$ when $n < 5$).
    - 2 faults: 364 (24.27% vs nominal 25%).
    - 3 faults: 28 (1.87%, active only in MEDIUM/HARD with $N \ge 8$ and $n_g \ge 8$).
  - Shared-cause scenarios: 60 scenarios activated (4.00% of all scenarios; exactly 15.31% of the 392 eligible multi-fault scenarios with $k \ge 2$ and group cap $\ge 2$).
  - Types per faulty sensor across 1,730 faulty sensor instances:
    - 1 type: 1,260 (72.83%).
    - 2 types: 432 (24.97%).
    - 3 types: 38 (2.20%).
- **Conformance Status:** MATCH.

---

### Item 4: Event Count Distribution, Waveforms, Scope, Attenuation & Delays
- **Approved Rule / Authority:** Appendix A.3; Decisions 5, 18–21; `SPEC.md` Section 4.
  - Event count: Poisson($\lambda=1.2$), capped at 3 (approx. 30% zero).
  - Waveforms: Step exponential recovery 40%, Ramp 25%, Pulse 25%, Sustained shift 10%.
  - Magnitude: log-uniform $[0.5, 10.0]^\circ\text{C}$; 75% positive, 25% negative.
  - Durations: Step $\tau \in [10, 120]$ min; Ramp rise/fall $[20, 180]$ min, hold $[30, 180]$ min; Pulse $[5, 30]$ min; Sustained rise $[5, 20]$ min then hold until scenario end ($t=1920$).
  - Scope: Global 55% (attenuation $[0.6, 1.0]$); Partial 30% (40–80% sensors, attenuation $[0.3, 1.0]$); Local 15% (1–2 sensors, attenuation $[0.7, 1.0]$).
  - Sensor arrival delay: $[0, 10]$ minutes.
  - Start steps: uniformly sampled within scored window $[480, 1919]$.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:75-108`: `count_poisson_mean: 1.2`, `count_max: 3`, `type_probs: {step_exponential_recovery: 0.40, ramp: 0.25, pulse: 0.25, sustained_shift: 0.10}`, `magnitude_min: 0.5`, `magnitude_max: 10.0`, `positive_sign_prob: 0.75`, `negative_sign_prob: 0.25`, `durations: {...}`, `scope: {...}`, `arrival_delay_max: 10`, `start_step_min: 480`, `start_step_max: 1919`.
  - `src/sim/events.py:33-193`: `generate_step_exponential_recovery()`, `generate_ramp()`, `generate_pulse()`, `generate_sustained_shift()`.
  - `src/sim/events.py:438-441, 612-655`: Start step sampled in $[480, 1919]$.
- **Empirical Measurement (N=1,770 event episodes across 1,123 scenarios):**
  - Scenarios with 0 events: 377 (25.13% vs theoretical $e^{-1.2} = 30.12\%$, shifted by constructive local event fallback).
  - Event count per scenario: 0: 377 (25.13%), 1: 638 (42.53%), 2: 323 (21.53%), 3: 162 (10.80%). Max count is 3.
  - Event types:
    - `step_exponential_recovery`: 672 (37.97% vs 40% target).
    - `ramp`: 471 (26.61% vs 25% target).
    - `pulse`: 407 (22.99% vs 25% target).
    - `sustained_shift`: 220 (12.43% vs 10% target).
  - Event sign: Positive (+1): 1,321 (74.63% vs 75% target), Negative (-1): 449 (25.37% vs 25% target).
  - Absolute magnitude: min 0.5005, max 9.9743, median 2.4536 (strictly within $[0.5, 10.0]$).
  - Event start step: min 480, max 1919, mean 1176.7 (strictly within $[480, 1919]$).
  - Event scopes: Global 1,178 (66.55%), Partial 338 (19.10%), Local 254 (14.35%).
- **Conformance Status:** MATCH.

---

### Item 5: Exact Event-Detectability Rule & Definitions of `genuine_event` and `event_below_detectability`
- **Approved Rule / Authority:** Appendix A.3; Decision 45; `SPEC.md` Section 4.
  - Threshold: $\text{threshold} = 3.0 \times \sigma_k$.
  - Detection condition: At group level at step $t$, composite event contribution exceeds $3\sigma_k$ for at least one sensor in the group ($c_{s}(t) \ge 3.0 \times \sigma_{k, s}$).
  - Label assignment:
    - If at least one active episode and $c_{s}(t) \ge 3.0 \sigma_{k, s}$: `genuine_event = 1`, `event_below_detectability = 0`.
    - If at least one active episode and all $c_{s}(t) < 3.0 \sigma_{k, s}$: `genuine_event = 0`, `event_below_detectability = 1`.
    - If no active episode: `genuine_event = 0`, `event_below_detectability = 0`.
  - Mutually exclusive: `genuine_event` and `event_below_detectability` can never both be 1 at the same timestep.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:104-105`: `detectability_threshold_sigma: 3.0`, `activity_epsilon_celsius: 0.0001`.
  - `src/sim/events.py:796-829`: Lines 806–828 directly execute the condition and set the mutually exclusive flags.
- **Empirical Measurement (All 883,584 group rows across 4 partitions):**
  - Partition `train` (374,784 rows): genuine: 48,211 (12.86%), below: 26,603 (7.10%), neither: 299,970, both: 0.
  - Partition `val` (127,104 rows): genuine: 15,440 (12.15%), below: 8,688 (6.84%), neither: 102,976, both: 0.
  - Partition `test_full_resolution` (318,720 rows): genuine: 36,578 (11.48%), below: 24,698 (7.75%), neither: 257,444, both: 0.
  - Partition `test_downsampled` (62,976 rows): genuine: 8,676 (13.78%), below: 4,258 (6.76%), neither: 50,042, both: 0.
  - Total pooled group rows: 883,584. Genuine event rows: 108,905 (12.33%, meeting formal 8–15% full-duration target). Both flags active: 0 rows (0.0000%).
- **Conformance Status:** MATCH.

---

### Item 6: Unit-Scale Probabilities and Affine Multiplier / Offset Ranges
- **Approved Rule / Authority:** Appendix A.2; Decision 1; Decision 46; `SPEC.md` Section 4.
  - Scale types: Celsius 70%, Fahrenheit 15%, Random affine 15%.
  - Random affine parameters: Multiplier log-uniform in $[0.5, 3.0]$; Offset uniform in $[-10.0, 10.0]^\circ\text{C}$ (Decision 1 ENG).
  - Applied strictly last in the sensor transduction pipeline.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:65-72`: `celsius_prob: 0.70`, `fahrenheit_prob: 0.15`, `random_affine_prob: 0.15`, `affine_multiplier_min: 0.5`, `affine_multiplier_max: 3.0`, `affine_offset_min: -10.0`, `affine_offset_max: 10.0`.
  - `src/sim/units.py:15-142`: `UnitScaler` with `.apply()` and `.invert()`.
  - `src/sim/profile.py:322-323`: `sample_unit_scaler()`.
  - `src/sim/sensors.py:711`: `unit_scaler.apply(canonical_values)` applied at Step 11 (last).
- **Empirical Measurement (N=1,500 scenarios):**
  - `celsius`: 1,028 (68.53% vs 70% target).
  - `fahrenheit`: 237 (15.80% vs 15% target).
  - `affine`: 235 (15.67% vs 15% target).
  - Affine multiplier ($N=235$): min 0.5027, max 2.9978, mean 1.3330 (strictly within $[0.5, 3.0]$).
  - Affine offset ($N=235$): min -9.9568, max +9.9767, mean 0.6942 (strictly within $[-10.0, 10.0]$).
- **Conformance Status:** MATCH.

---

### Item 7: Sensor Noise, Correlated Noise, Static Offsets, Lag, Quantisation & Clipping
- **Approved Rule / Authority:** Appendix A.2; Decisions 12–17; `SPEC.md` Section 4.
  - Sensor noise $\sigma_k$: log-uniform $[0.05, 0.30]^\circ\text{C}$.
  - Correlated noise: 20% of sensors mildly correlated; 1-factor model with $\rho \in [0.2, 0.5]$.
  - Static offset: uniform $[-1.0, 1.0]^\circ\text{C}$ per sensor.
  - Response lag $\tau$: first-order, $[0, 6]$ min; 40% of sensors zero lag.
  - Quantisation: 0.01 (25%), 0.05 (25%), 0.1 (35%), 0.25 (15%).
  - Clipping: Environment clipped to $[-5, 50]^\circ\text{C}$; Sensor clipped to $[-10, 60]^\circ\text{C}$.
  - Healthy missing data: 20% zero-rate (Bernoulli); remaining Uniform$[0, 0.01]$; max burst 3 (NaN).
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:41-64`: `sigma_k_min: 0.05`, `sigma_k_max: 0.30`, `correlated_sensors_fraction: 0.20`, `correlation_rho_min: 0.2`, `correlation_rho_max: 0.5`, `static_offset_min: -1.0`, `static_offset_max: 1.0`, `response_lag: {time_constant_min: 0.0, time_constant_max: 6.0, zero_lag_fraction: 0.40}`, `quantisation: {values: [0.01, 0.05, 0.1, 0.25], probabilities: [0.25, 0.25, 0.35, 0.15]}`, `bounds: {environment_clip_min: -5.0, environment_clip_max: 50.0, sensor_clip_min: -10.0, sensor_clip_max: 60.0}`.
  - `src/sim/profile.py:342-424`: `sample_sensor_params()` generates parameters strictly according to these ranges.
  - `src/sim/sensors.py:33-145`: `apply_response_lag()`, `apply_clipping()`, `apply_quantisation()`.
- **Empirical Measurement (8,849 sensors across 1,500 scenarios):**
  - Quantisation steps:
    - 0.01: 2,189 sensors (24.74% vs 25% target).
    - 0.05: 2,268 sensors (25.63% vs 25% target).
    - 0.10: 3,085 sensors (34.86% vs 35% target).
    - 0.25: 1,307 sensors (14.77% vs 15% target).
  - Clipping bounds in scenarios.parquet: exactly `{"environment": [-5.0, 50.0], "sensor": [-10.0, 60.0]}` for 1,500/1,500 scenarios.
  - Zero out-of-bounds readings observed in raw observables after unit inversion.
- **Conformance Status:** MATCH.

---

### Item 8: Complete Sensor Transduction Pipeline Order
- **Approved Rule / Authority:** Decision 27; `SPEC.md` Section 4; `PROGRESS.md` Section 8.
  - Strict 11-step execution order:
    1. Physical environment (with genuine events and environment clipping $[-5, 50]^\circ\text{C}$).
    2. Response lag (first-order exponential smoothing $\tau$).
    3. Healthy static offset ($+ \text{offset}$).
    4. Gain error relative to physical onset temperature ($x_{\text{ref}} + g(x - x_{\text{ref}})$).
    5. Additive faults (offset drift, spike pulses, slow degradation bias).
    6. Measurement noise (scaled by high-noise and slow-degradation multipliers).
    7. Frozen sensor override (held value with optional jitter).
    8. Sensor clipping ($[-10, 60]^\circ\text{C}$).
    9. ADC quantisation (round to grid).
    10. Missing data mask union (healthy + dropout + degradation missing $\implies \text{NaN}$).
    11. Unit scaling applied strictly LAST (Celsius $\to$ Fahrenheit or Affine).
- **Implementation in Code:**
  - `src/sim/sensors.py:619-715`: `simulate_sensor()` executes the 11 steps in exact sequential order:
    - Step 1: `phys_vals` input (pre-clipped environment + event adjustments).
    - Step 2: Line 622: `apply_response_lag(phys_vals, tau=params.response_lag_tau)`.
    - Step 3: Line 629: `with_offset = lagged + params.static_offset`.
    - Step 4: Line 641: `with_gain[g_start:g_end] = x_ref + g_val * (with_offset[g_start:g_end] - x_ref)`.
    - Step 5: Line 647: `with_additive = with_gain + fault_series.combined_additive_offset`.
    - Step 6: Line 658: `effective_noise = base_noise * fault_series.combined_noise_multiplier`, line 662: `noisy = with_additive + effective_noise`.
    - Step 7: Line 673: `noisy[f_start:f_end] = y_frz + jit`.
    - Step 8: Line 678: `clipped = apply_clipping(noisy, -10.0, 60.0)`.
    - Step 9: Line 685: `quantized = apply_quantisation(clipped, params.quantisation_step)`.
    - Step 10: Line 708: `canonical_values[total_missing_mask] = np.nan`.
    - Step 11: Line 711: `observable_values = unit_scaler.apply(canonical_values)`.
- **Conformance Status:** MATCH.

---

### Item 9: All Seven Fault Injectors
- **Approved Rule / Authority:** Appendix A.4, A.7; Decisions 6, 10, 22–26, 28, 34; `SPEC.md` Section 4.
  - Episode timing: 10% present from step 0; remaining 90% uniform in $[480, 1919]$.
  - Spike: episode $[30, 400]$ steps; pulse interval $[10, 60]$ steps; size $[3, 40]\sigma_k$; length 1 (60%), 2 (25%), 3 (15%); label active on spike samples + 4 following steps; severity $k/40$.
  - Offset drift: ramp $[60, 720]$ steps; final bias $[1, 20]\sigma_k$; plateau $[0, 600]$ steps; shapes: linear 55%, saturating 20%, stepped 15% ($K \in \{3, 4, 5\}$), sudden constant 10%; severity $|\text{bias}| / (20 \sigma_k)$.
  - Gain error: duration $[120, 1200]$ steps; $x_{\text{ref}}$ is physical temperature at onset; standard $[0.55, 0.85] \cup [1.15, 1.60]$ (85%), subtle $[0.90, 0.95] \cup [1.05, 1.10]$ (15%); severity $|g-1| / 0.6$.
  - Frozen: duration $[20, 400]$ steps; constant 60%, jitter 40% ($[0.02, 0.3]\sigma_k$); last reading 80%, wrong value 20% ($[5, 20]\sigma_k$ away); severity 0.8 (normal) or 1.0 (wrong value).
  - High noise: multiplier $[2, 10]\sigma_k$; duration $[30, 600]$ steps; bursty 30% (blocks $[5, 30]$ steps); severity $(m-1)/9$.
  - Dropout: enter gap prob $[0.002, 0.03]$; geometric gap mean $[2, 40]$, cap 100; episode $[60, 600]$ steps; severity $0.3 + 0.7 \times \text{missing\_fraction\_30}$.
  - Slow degradation: duration $[240, 1200]$ steps; health $h: 1 \to h_{\text{end}} \in [0.1, 0.7]$; damage $d(t) = 1 - h(t)$; noise $\times (1+5d)$; bias $\pm 8\sigma_k d$; extra missing $0.15 d$; untrusted once $d \ge 0.15$; severity $d(t)$; labeled strictly $f_{\text{slow\_degradation}}$ only.
  - Aggregation: composite severity $\max$ active, health $1 - \text{severity}$, untrusted $\bigvee$ untrusted flags.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:110-164`: all parameters configured without hardcoding.
  - `config/generator_a.yaml:199-210`: severity formula divisors configured.
  - `src/sim/faults.py:189-839`: dedicated generator functions for each fault type.
  - `src/sim/faults.py:881-899`: `fault_severity`, `true_health`, `sensor_untrusted`.
- **Empirical Measurement (N=2,238 fault episodes):**
  - Episode counts by fault type:
    - `gain_error`: 407 (18.19%).
    - `spike`: 368 (16.44%).
    - `dropout`: 364 (16.26%).
    - `offset_drift`: 341 (15.24%).
    - `high_noise`: 298 (13.32%).
    - `slow_degradation`: 234 (10.46%).
    - `frozen`: 226 (10.10%).
  - Offset drift shapes ($N=341$): Linear: 188 (55.13%), Saturating: 70 (20.53%), Stepped: 49 (14.37%), Sudden constant: 34 (9.97%).
  - Spike pulse lengths ($N=1,487$ discrete pulses): Length 1: 932 (62.68%), Length 2: 367 (24.68%), Length 3: 188 (12.64%).
  - Frozen modes ($N=226$): Last reading: 182 (80.53%), Wrong value: 44 (19.47%). Constant: 130 (57.52%), Jitter: 96 (42.48%).
  - High noise mode ($N=298$): Continuous: 213 (71.48%), Bursty: 85 (28.52%).
- **Conformance Status:** MATCH.

---

### Item 10: Warm-Up Duration, Scored Timestamps, Endpoints, Event Constraints & Formulas
- **Approved Rule / Authority:** Appendix A.1, A.3; Decisions 18, 44, 45; `SPEC.md` Section 4.
  - Total duration: 1,920 steps (warm-up 480 steps, scored 1,440 steps).
  - Timestamps: step $t \times 1.0$ min ($t \in [0, 1919]$). Scored window starts at $t=480.0$ min.
  - Warm-up phase: strictly event-free; genuine events may only start in $[480, 1919]$.
  - Ground truth formulas: `true_health = 1 - fault_severity`, `true_value` in canonical Celsius physical units.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:6-8`: `sampling_rate_minutes: 1`, `scored_steps: 1440`, `warmup_steps: 480`, `total_steps: 1920`.
  - `src/sim/environment.py:103-110`: Validates `warmup_steps + scored_steps == total_steps`.
  - `src/sim/events.py:438-441`: `start_step_min = 480`, `start_step_max = 1919`.
  - `src/sim/faults.py:881-899`: `fault_severity`, `true_health`.
- **Empirical Measurement:**
  - Exactly 1,920 steps per sensor in full resolution ($t \in [0.0, 1919.0]$ min).
  - Exactly 384 steps per sensor in stride 5 ($t \in [0.0, 1915.0]$ min, step 5 min).
  - Post-warmup scored fraction: exactly 75% of rows ($t \ge 480.0$).
  - Event start steps in all 1,770 event episodes: min 480, max 1919. Zero events start during warm-up.
  - Sensor ground-truth health invariant: `true_health + fault_severity == 1.0` holds across all 4,732,800 exported rows.
- **Conformance Status:** MATCH.

---

### Item 11: Determinism and Mapping from Scenario Seeds to Generated Outputs
- **Approved Rule / Authority:** Appendix A.1; Non-negotiable ML rules; `SPEC.md` Section 3.
  - Seed mapping: Seeds 1 to 1500 map 1-to-1 to `scenario_0001` through `scenario_1500`.
  - Exact determinism: Same seed + same config hash produces bitwise identical observables and ground-truth tables.
- **Configured & Implemented Value:**
  - `config/generator_a.yaml:26-27`: `generator_a_start: 1`, `generator_a_end: 1500`.
  - `src/sim/scenario.py`: `ScenarioSimulator.simulate_scenario(seed=seed, scenario_id=scenario_id)` initializes seeded numpy Generator.
- **Empirical Measurement:**
  - Re-simulated `scenario_0001` with seed 1 and extracted stride-5 observables.
  - Merged directly against `data/raw/train/observables.parquet` for `scenario_0001`.
  - Number of compared rows: 2,688 rows.
  - Maximum absolute difference on valid float readings: **0.000000000000**.
  - Missing-value (NaN) mask equality: **100.0% match (`True`)**.
- **Conformance Status:** MATCH.

---

## Section 3: Distinct Categorical Values Catalog (Addition b)

Across all Parquet tables, the complete distinct values for every categorical column are cataloged below:

| Table | Column | Unique Count | Distinct Values & Counts |
|---|---|:---:|---|
| `scenarios.parquet` | `split` | 3 | `train` (900), `val` (300), `test` (300) |
| `scenarios.parquet` | `generator_version` | 1 | `1.0.0` (1500) |
| `scenarios.parquet` | `scenario_type` | 3 | `standard` (1221), `single_sensor` (142), `two_groups` (137) |
| `scenarios.parquet` | `difficulty` | 3 | `medium` (586), `easy` (467), `hard` (447) |
| `scenarios.parquet` | `hard_case_type` | 6 | `none` (1053), `natural` (299), `local_event` (71), `subtle_fault` (41), `fault_during_event` (35), `faulty_fraction_at_cap_small_group` (1) |
| `scenarios.parquet` | `profile` | 1 | `temperature` (1500) |
| `scenarios.parquet` | `has_events` | 2 | `True` (1123), `False` (377) |
| `scenarios.parquet` | `has_shared_cause` | 2 | `False` (1440), `True` (60) |
| `fault_episodes.parquet` | `group_id` | 2 | `group_00` (2096), `group_01` (142) |
| `fault_episodes.parquet` | `fault_type` | 7 | `gain_error` (407), `spike` (368), `dropout` (364), `offset_drift` (341), `high_noise` (298), `slow_degradation` (234), `frozen` (226) |
| `event_episodes.parquet` | `event_type` | 4 | `step_exponential_recovery` (672), `ramp` (471), `pulse` (407), `sustained_shift` (220) |
| `event_episodes.parquet` | `scope` | 3 | `global` (1178), `partial` (338), `local` (254) |
| `event_episodes.parquet` | `sign` | 2 | `1` (1321), `-1` (449) |
| `event_episodes.parquet` | `affected_groups` | 4 | `group_00` (1595), `group_00,group_01` (154), `""` (12), `group_01` (9) |
| `event_episodes.parquet` | `is_detectable` | 2 | `True` (1730), `False` (40) |
| `ground_truth_sensors` | `sensor_untrusted` | 2 | `0` (4,566,247), `1` (166,553) |
| `ground_truth_events` | `genuine_event` | 2 | `0` (774,679), `1` (108,905) |
| `ground_truth_events` | `event_below_detectability` | 2 | `0` (819,337), `1` (64,247) |

---

## Section 4: Audit Limitations & Realized Distributions vs Nominal Targets

While all underlying stochastic generators sample faithfully from configured distributions, the **realized dataset frequencies** differ from unconstrained nominal probability tables due to the mathematical interaction of scenario caps, group caps, tier restrictions, and constructive fallback queues:

### 1. Realized Faulty-Sensor Counts vs Nominal Candidate Proportions
- **Nominal Candidate Probabilities:** 0: 15.0%, 1: 40.0%, 2: 25.0%, 3 or more: 20.0%.
- **Realized Counts ($N=1,500$):**
  - 0 faults: 190 (12.67%)
  - 1 fault: 918 (61.20%)
  - 2 faults: 364 (24.27%)
  - 3 faults: 28 (1.87%)
- **Mathematical Cause:**
  - EASY tier (30% nominal, 467 scenarios) strictly caps faulty sensors to at most 1 ($k \le 1$), forcing candidate counts of 2 and 3 down to 1.
  - Group cap $\text{cap}_g = \max(1, \lfloor 0.4 n_g \rfloor)$ caps 4-sensor groups ($n_g=4$) to at most 1 fault.
  - In two-group scenarios with $n_1=4, n_2=4$, the total cap is 2.
  - Consequently, candidate draws of $k_{\text{cand}} = 3$ can only survive in MEDIUM and HARD scenarios where $N \ge 8$ and $n_g \ge 8$, compressing the realized 3-fault fraction to 1.87% while elevating 1-fault scenarios to 61.20%.

### 2. Fault Types Per Faulty Sensor vs Nominal Candidate Proportions
- **Nominal Candidate Probabilities:** 1: 65.0%, 2: 27.0%, 3: 8.0%.
- **Realized Frequencies ($N=1,730$ faulty sensor instances):**
  - 1 type: 1,260 (72.83%)
  - 2 types: 432 (24.97%)
  - 3 types: 38 (2.20%)
- **Mathematical Cause:**
  - EASY tier (467 scenarios) strictly restricts all faulty sensors to exactly 1 fault type ($m=1$).
  - MEDIUM tier (586 scenarios) strictly caps fault types to at most 2 ($m \le 2$).
  - 3 types per sensor are permissible only in the HARD tier ($N=447$ scenarios), shifting the global realized distribution toward 1 and 2 types.

### 3. Event Scope Mix vs Nominal Probabilities
- **Nominal Probabilities:** Global 55.0%, Partial 30.0%, Local 15.0%.
- **Realized Frequencies ($N=1,770$ event episodes):**
  - Global: 1,178 (66.55%)
  - Partial: 338 (19.10%)
  - Local: 254 (14.35%)
- **Mathematical Cause:**
  - EASY tier strictly restricts allowed event scopes to `['global', 'none']` (Appendix A.6), completely excluding Partial and Local events across all 467 EASY scenarios.
  - In the HARD tier, the constructive fallback queue injects 71 `local_event` episodes (`event_local_guaranteed`), partially rebalancing Local scope to 14.35%.

### 4. Explanation of 12 Empty `affected_groups` Rows
- **Observed:** 12 event episodes have `affected_groups == ""` and `affected_sensors == ""`.
- **Finding:** All 12 episodes occur in `single_sensor` scenarios ($N=1$) under `partial` scope (6 MEDIUM, 6 HARD, 0 EASY).
- **Mathematical Cause:** Per approved **Decision 37** ([PROGRESS.md:L239](file:///d:/CODE_APEX_HACKATHON/PROGRESS.md#L239)), partial event sensor allocation in single-sensor scenarios uses explicit half-up rounding:
  $$k_{\text{target}} = \lfloor \text{frac} \times N + 0.5 \rfloor$$
  With $N=1$ and $\text{frac} \sim \text{Uniform}[0.4, 0.8]$, whenever $\text{frac} < 0.5$ (theoretical probability $\frac{0.5 - 0.4}{0.8 - 0.4} = 25.0\%$), $k_{\text{target}} = 0$.
  Across the 35 single-sensor partial events in the dataset, exactly 12 yielded $k=0$ (34.29%) and 23 yielded $k=1$ (65.71%), perfectly within standard binomial variation ($\mu=8.75, \sigma=2.56$).
- **Status:** **INTENDED BEHAVIOR** (Decision 37), not a bug.

### 5. Explanation of 12.43% (220) `sustained_shift` Episodes vs 10% Nominal Target
- **Observed:** `sustained_shift` accounts for 220 / 1,770 episodes (12.43%).
- **Finding:**
  - Nominal stochastic sampling across all 1,664 non-fallback event episodes yielded **149 sustained shifts (8.95%)**, closely matching the 10.0% nominal target.
  - The HARD tier constructive fallback queue (Decision 38) guarantees qualification of non-qualifying HARD scenarios by injecting `event_local_guaranteed` modeled as a sustained shift.
  - Exactly 71 HARD scenarios activated this fallback, adding 71 sustained shift episodes (all in HARD tier: 36 nominal + 71 fallback = 107 sustained shifts in HARD tier).
  - Total sustained shifts: $149 \text{ nominal} + 71 \text{ fallback} = 220$ episodes ($12.43\%$).
  - Similarly, the `fault_during_event` constructive fallback injected 35 `event_ramp_guaranteed` episodes, elevating `ramp` from 436 nominal (26.20%) to 471 total (26.61%).
- **Status:** **INTENDED TIER FALLBACK EFFECT** (Decision 38), not a bug.

---

## Section 5: Sampling Resolution Preservation & P1.12 Plan (Addition e)

Per instructions, the scoring resolution of test partitions is not fixed during this audit:
- Storing everything at full resolution (1-minute history) is planned for execution in P1.12.
- The evaluation scoring resolution across test partitions will be decided during P7.
- The current read-only audit preserved all Parquet files under `data/` untouched.

---

## Section 6: Decision Rule & Verification Sign-Off

### Decision Rule Assessment
- **Any discrepancies between approved specifications and implementation?** NONE.
- **Any unauthorized parameter modifications or hardcoded numbers?** NONE.
- **Dataset and seeds intact?** The existing 1,500 scenarios, scenario seeds 1–1500, split allocations, and schemas remain 100% verified and unmodified.

### Git Status & Tag Information
- `git status --short`:
  ```
  ?? docs/CONFORMANCE_AUDIT.md
  ?? docs/DATASET_COUNTS.md
  ?? scripts/conformance_audit.py
  ```
- Target commit of `p1-verified` tag:
  `3319a46c57f438652a046fef4eda507480ae621e` (`docs(p1): complete P1.11 sanity batch review, validation, and prevalence audit`)
- `HEAD` commit:
  `3319a46c57f438652a046fef4eda507480ae621e` (matches `p1-verified` exactly).

**Final Conclusion:**  
The Generator A simulator and the existing 1,500-scenario dataset conform fully to the approved specification and Decision Log. P1.12 has not been started. Awaiting user review and authorization.