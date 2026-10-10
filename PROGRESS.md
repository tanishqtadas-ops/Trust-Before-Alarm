# PROGRESS.md: Sensor Trust Intelligence

> **Every AI agent must read this whole file before doing anything, and update it after every task.**
> Project: VIT Code Apex 3.0, Track 4 (UST), Problem Statement 3: *AI-Based Sensor Trust, Drift and False-Alarm Detection*.
> Team: (fill in team name). Last updated: 2026-10-10.

## CURRENT STATUS (update this block after every task)

- **Current phase:** P1 (Generator A, batch mode)
- **Last completed task:** P1.3 (Virtual sensors: offset, noise, response lag, quantisation, clipping, healthy missing data)
- **Next task:** P1.4 (Genuine events: step/ramp/pulse/sustained, scope, attenuation, arrival delay, detectability labels)
- **Blocked on / open questions:** none
- **Last human-verified phase:** P0 (verified)

---

## 0. RULES FOR AI CODING AGENTS (mandatory)

1. **Read this file fully before starting any work.** If `docs/SPEC.md` exists, read it too. If it conflicts with this file, stop and ask the user.
2. **Work on the first unchecked task** of the current phase, unless the user says otherwise. Do one phase at a time. Do not jump ahead or build extra features.
3. **Never invent parameters or assumptions.** Every simulator and pipeline number must come from Appendix A or `docs/SPEC.md` and live in a config file, never hard-coded. If something is missing or ambiguous, **stop and ask the user**.
4. **After each task:** tick its checkbox, write a short entry in the Log (section 7: date, what was done, files created or changed, how it was verified, open issues), and update the CURRENT STATUS block. Do not mark a task done if tests fail.
5. **Never silently change a locked parameter or a completed task.** If a change seems necessary, ask the user first, and record it in the Decision Log (section 8) with the reason.
6. **Never show or write unmeasured results.** Every number in a report, table, plot or UI must be produced by code from saved output files. No hand-typed numbers. Placeholder cells say "not yet measured".
7. **Show the user every file you create** (path plus a one-line purpose) and the commands to run the tests.
8. **Before saying a phase is finished:** run all tests, run that phase's Definition of Done checks, and list the results honestly, including anything that failed or was skipped.
9. Keep this file concise. Do not paste large outputs into it. Link to output files instead.
10. **Generator B independence.** Generator B is specified in a separate file and built by a *different* agent in a separate folder. **Do not read, request, build or imitate Generator B.** You will only receive its frozen output files (read-only) for evaluation in P7/P8. Never change them, and never tune the model against them.
11. **Implementation choices vs parameters.** You may make small implementation choices that the spec leaves open (for example how to code a smoothing filter), but every such choice must be listed in the Decision Log as "implementation choice" with a one-line reason. You may **not** choose parameter values, thresholds or distributions yourself.

### Non-negotiable ML rules (violating any is a bug)

- The **model pipeline reads only Layer 1 (observable data)**. The feature code must physically accept only the raw table (`scenario_id, group_id, sensor_id, variable_type, timestamp, value`). Truth is stored in **separate files** and used only by evaluation code.
- **Causal only:** features at time t use readings at or before t. No future values, ever. There must be an automated test for this.
- **No `scenario_id`, `sensor_id`, `group_id`, `variable_type`, `timestamp`, or raw `value` as model features in v1.**
- **Split by whole scenarios**, never by rows. Stratify by difficulty tier and fault type. Seeds fixed (Appendix A).
- **The test set is used once, at the end.** No tuning, no calibration, no threshold selection on test data.
- **Group reference for cross-sensor features excludes the sensor being judged.**
- **Every output carries run id, seed and config hash.** Same seed plus same config gives identical files.
- Do not add algorithms (LSTM, Transformer, autoencoder, XGBoost and so on) unless the user approves. Core model is LightGBM.
- Do not use SMOTE.

---

## 1. PROJECT IN PLAIN WORDS

Imagine 6 thermometers in one room. Sometimes a thermometer breaks (stuck, jumping, slowly reading too high, silent). Sometimes the room really gets hot and all thermometers rise together. A simple alarm system either raises false alarms on broken sensors or misses real danger when it ignores a faulty one.

We are building a **software-only sensor intelligence system** that watches readings from several related sensors (v1: temperature) and, continuously, answers:

1. **Is this sensor trustworthy right now, and if not, what fault does it have?** (spike, offset drift, gain error, frozen, high noise, dropout, slow degradation; several can be true at once)
2. **Is something genuinely happening in the environment?** (a real event, which can coincide with a faulty sensor)
3. **Should this alarm be trusted, suppressed or escalated?** and what should be done (recalibrate, inspect, replace)

It explains every alert in plain language and is demonstrated through a dashboard where a judge can inject faults and events live.

**Key idea ("Trust Before Alarm"):** an abnormal reading is not automatically an alarm. We first ask whether the sensor can be trusted and whether the environment is really changing.

**No hardware.** We build a **simulator** (virtual world, virtual sensors, injected faults, genuine events). Because it injects the faults, it records exact **ground truth** (an answer key). Models never see the answer key; evaluation does.

**Official data:** the PS allows synthetic data, and organisers may release a development dataset plus a hidden test set. If an official dataset exists, we train and validate on it first (through a data adapter) and our simulator becomes extra training data and a stress-test lab.

## 2. ARCHITECTURE

```
Sources: Official dataset (if released) | Generator A (dev) | Generator B (built separately; robustness testing only)
   -> Data adapter -> common raw schema
   -> Causal windows -> Feature engine
   -> Sensor-level table (Heads A, B) | Group-level table (Head C)
   -> Head A: untrusted?   Head B: which faults? (7 yes/no models)   Head C: genuine event?
   -> Provable rules (exact frozen, obvious dropout) + ML scores
   -> Probability calibration
   -> Trust engine (smoothed score, status, recommended action)
   -> Trust-aware alarm layer (real / real + sensor flagged / suppressed / unverified)
   -> Explanations (top 3 reasons, plain sentences)
   -> API + Dashboard (live fault injection, "reveal truth", evidence tab)
```

**Four data layers:** (1) observable data, (2) derived features, (3) ground truth (hidden from models), (4) predictions.

## 3. SUGGESTED REPO LAYOUT

```
README.md   PROGRESS.md   docs/SPEC.md
config/        (all parameters: generator_a.yaml, generator_b.yaml, features.yaml, alarm.yaml, eval.yaml)
src/sim/       (environment, sensors, events, faults, labels, generator, streaming)
src/features/  (windows, sigma_hat, sensor_features, group_features)
src/models/    (rules, baselines, heads, calibration, trust, alarm, explain)
src/eval/      (metrics, audit, slices, ablations, reports)
src/api/       src/app/ (dashboard)
tests/   data/ (generated, git-ignored)   runs/ (run outputs: id, seed, config hash)
```

## 4. TASK TAGS

**[M]** must-have **[N]** nice-to-have **[S]** stretch. Do all [M] before any [N]; do [S] only if the user asks.

## 5. PHASES AND TASKS

Dates are our targets (official: Round 2 online 15 to 18 Oct, finale 24 Oct 2026).

### P0. Setup and spec lock (10 to 11 Oct)
- [x] P0.1 [M] Create repo layout, Python environment, `requirements.txt` (pandas, numpy, scipy, scikit-learn, lightgbm, pyarrow, pyyaml, pytest, matplotlib), config loader.
- [x] P0.2 [M] Run stamp utility: every run gets run id, seed, config hash, timestamp, written next to outputs.
- [x] P0.3 [M] Put Appendix A into `config/*.yaml` exactly (no hard-coded numbers elsewhere); write `docs/SPEC.md` from this file for the record.
- [x] P0.4 [M] Smoke test that loads configs and prints the parameters.
**Done when:** configs load, tests run, user confirms the parameters match Appendix A.

### P1. Generator A, batch mode (11 to 13 Oct)
- [x] P1.1 [M] Temperature variable profile (Appendix A.2), driven by config; unit scaling applied last.
- [x] P1.2 [M] True environment: daily cycle, slow random wander, bounds.
- [x] P1.3 [M] Virtual sensors: offset, noise (with optional correlation), response lag, quantisation, clipping, healthy missing data.
- [ ] P1.4 [M] Genuine events: step/ramp/pulse/sustained, scope (global/partial/local), attenuation, arrival delay, detectability labels.
- [ ] P1.5 [M] Seven fault injectors (A.4) with start/duration/severity sampling, "from the start" faults.
- [ ] P1.6 [M] Combination rules, shared-cause pairs, faulty-fraction cap (A.5).
- [ ] P1.7 [M] Scenario types (standard/two-group/single-sensor), sensor counts, difficulty tiers (A.6).
- [ ] P1.8 [M] Ground-truth tables saved **separately** from the raw table; true_health and severity formulas (A.7); episode tables and scenario table.
- [ ] P1.9 [M] Dataset writer (Parquet), 1,500 scenarios, split assignment 900/300/300 stratified; train/val stride 5, 150 test scenarios at full resolution.
- [ ] P1.10 [M] Tests: determinism (same seed identical), no truth columns in raw table, bounds respected, label consistency, prevalence report.
- [ ] P1.11 [M] Sanity batch: 100 scenarios, plot 10 (3 EASY, 3 MEDIUM, 4 HARD), prevalence check (untrusted 6 to 10%, event 8 to 15%).
**Done when:** the user opens the 10 plots and they look like believable sensor data; prevalence near targets; determinism test passes. **User must verify before P2.**

### P2. Feature engine (13 to 15 Oct)
- [ ] P2.1 [M] Causal windows (5/30/120/480) and sigma-hat estimator (A.8).
- [ ] P2.2 [M] Own-signal temporal, statistical, data-quality features (Appendix B).
- [ ] P2.3 [M] Cross-sensor features (exclude self; gap centring), single-sensor fallback (NaN).
- [ ] P2.4 [M] Group/event joined features and the group-level table for Head C.
- [ ] P2.5 [M] **No-future-data test:** changing future readings must not change current features.
- [ ] P2.6 [M] **Truth-isolation test:** feature code runs with only the raw table available.
**Done when:** both tests pass; feature tables saved; user reviews the feature list.

### P3. Raw alarm, baselines and sanity check (14 to 16 Oct)
- [ ] P3.1 [M] Raw alarm detector (A.9), episode merging, cause labels EVENT/FAULT/BOTH/NEITHER from truth.
- [ ] P3.2 [M] Simple baselines: fixed threshold and robust z-score; metrics at the false-alarm budget.
- [ ] P3.3 [M] Difficulty check: baselines per tier. If baselines are near-perfect on HARD, report to user (data too easy); if faults cannot be seen even on EASY, report (too hard). **Any simulator adjustment needs user approval, happens once, and is logged.**

### P4. Leakage audit and first model (15 to 17 Oct)
- [ ] P4.1 [M] Single-feature separability check (flag AUC > 0.85).
- [ ] P4.2 [M] Label-shuffle null control (5 shuffles, seeds 20001 to 20005) and canary leak test.
- [ ] P4.3 [M] Quick multiclass LightGBM as a first benchmark only.
- [ ] P4.4 [M] Report audit results honestly, including failures.
**Done when:** audit passes or issues are explained; user approves before P5.

### P5. Three heads and rules (16 to 18 Oct)
- [ ] P5.1 [M] Provable rules: frozen and dropout (A.10).
- [ ] P5.2 [M] Head A (untrusted), Head B (7 binary models), Head C (group event, group table).
- [ ] P5.3 [M] Class weights, grouped 5-fold CV by scenario, early stopping, small tuning grid.
- [ ] P5.4 [M] Ablations: rules only / ML only / rules plus ML; feature families; raw value versus none.

### P6. Calibration, trust engine, alarm layer (17 to 19 Oct)
- [ ] P6.1 [M] Calibration on out-of-fold predictions (A.11); reliability plot.
- [ ] P6.2 [M] Probability cutoffs chosen on validation to meet the false-alarm budget.
- [ ] P6.3 [M] Trust engine: smoothing, status, recommended action.
- [ ] P6.4 [M] Trust-aware alarm decisions (A.9 table); alarm metrics.

### P7. Evaluation suite (18 to 21 Oct)
- [ ] P7.1 [M] Metrics: per-fault precision/recall, macro-F1, PR-AUC, recall at the false-alarm budget, false-alarm episodes per healthy sensor-day, detection delay, calibration error, trust versus true health.
- [ ] P7.2 [M] Uncertainty: Wilson intervals, scenario-level bootstrap (1,000), thin-cell flags (n < 20).
- [ ] P7.3 [M] Per-difficulty results; worst-slice search (slices with at least 25 examples).
- [ ] P7.4 [M] Leave-one-fault-out; robustness curves (extra noise, extra missing); detectability curve.
- [ ] P7.5 [M] Auto-generated results tables from files (never hand-typed); pass-mark report (Appendix C).

### P8. Generator B evaluation (B is built elsewhere; evaluation 19 to 21 Oct)
Generator B is built by a different agent from a separate spec (target 13 to 16 Oct) and delivered as frozen, read-only files with a recorded hash. This agent does not build it.
- [ ] P8.1 [N] Receive B's output files; run the neutral schema and truth-consistency checker on them; record the file hash.
- [ ] P8.2 [N] Evaluate the finished model on B (500 test plus 200 stress scenarios) **once per major model version**; log every use in the Log.
- [ ] P8.3 [N] A-test versus B-test report (gap reported honestly, never "fixed" by changing B or tuning on B).

### P9. Explanations, streaming mode, API (20 to 22 Oct)
- [ ] P9.1 [N] Top-3 feature contributions to plain-language sentences (fixed templates).
- [ ] P9.2 [M] Streaming simulator (one reading at a time, 480-step healthy history first) with demo presets (A.13).
- [ ] P9.3 [N] API: send readings, receive trust, fault flags, event flag, confidence, reasons, alarm decision. Report latency and model size.

### P10. Dashboard (21 to 24 Oct)
Build in this order; three screens only.
- [ ] P10.1 [M] **Dashboard:** environment banner (Normal / Genuine Event), sensor cards (trust, status, fault tags), readings chart, alarm panel, "Why?" panel.
- [ ] P10.2 [M] **Injection panel** on the same screen: sensor, fault, severity, duration, event trigger; "reveal truth" toggle (disabled for official data).
- [ ] P10.3 [M] Alarm decision table; footer with run id, seed, data source.
- [ ] P10.4 [N] Recorded-demo mode; Present mode (large fonts).
- [ ] P10.5 [N] **Evidence tab** from result files; unmeasured cells show "not yet measured".
- [ ] P10.6 [S] Paired runs (same stream with and without the injection).
- Technology default: Streamlit with Plotly unless the user decides otherwise.

### P11. Official dataset adapter (when released)
- [ ] P11.1 [M] Adapter to the standard schema; handle unknown columns, units, sampling and sensor counts; group by variable type or by correlation when no groups are given.
- [ ] P11.2 [M] Train and validate on official data first; re-run the pipeline; report differences.

### P12. Final polish (22 to 24 Oct)
- [ ] P12.1 [M] README, run instructions, reproducibility check, demo script, backup recording.
- [ ] P12.2 [N] Public demo link if the user wants one.

### Stretch (only if the user asks)
- [ ] S1 Adaptive hard-case search (generate more of the cases the detector misses; evaluate only on independent data).
- [ ] S2 Humidity and pressure profiles; leave-one-variable-out test. (Placeholders only; not final.)

---

## 6. OPEN DECISIONS (user to answer)

- Final pass marks confirmation before evaluation (Appendix C is the draft).
- Who owns which phase. Whether to attempt stretch items. Whether to publish a public demo link.

## 7. LOG (newest first; one short entry per task)

Format: `YYYY-MM-DD | task id | what was done | files | how verified | open issues`

- 2026-10-10 | P1.3 | Implemented virtual sensor models (first-order response lag tau in [0, 6] min, static offset in [-1, 1] C, Gaussian measurement noise sigma_k in [0.05, 0.30] C with 1-factor mild correlation model rho in [0.2, 0.5], sensor clipping [-10, 60] C, quantisation in {0.01, 0.05, 0.1, 0.25} C, healthy missing data with 20% zero-rate assignment, bursts <= 3 as NaN, unit scaling applied last) | src/sim/sensors.py, src/sim/profile.py, src/sim/__init__.py, config/generator_a.yaml, tests/test_sensors.py, tests/test_smoke_configs.py, docs/SPEC.md, docs/PARAMETERS_CHECK.md, PROGRESS.md | pytest tests/ passed (56/56); smoke test verified; determinism, exact 20% zero-missing rate, max burst 3, and data layer isolation verified | none
- 2026-10-10 | P1.2 | Implemented true environmental temperature series in canonical Celsius (base level, daily cycle, slow random wander via stationary OU AR(1) with 5-step MA, bounds clipping [-5, 50] C) | src/sim/environment.py, src/sim/__init__.py, tests/test_environment.py, PROGRESS.md | pytest tests/ passed (38/38); verified 1,920 steps (480 warmup, 1,440 scored), determinism, daily cycle periodicity, AR(1) stationary variance, bounds [-5, 50] C | none
- 2026-10-10 | P1.1 | Corrected parameter fidelity: removed period fallback (made explicit in config), added configured daily-cycle phase bounds [0, 2pi) rad (Decision 12), made unit scaling probabilities strictly config-driven | config/generator_a.yaml, src/sim/profile.py, src/sim/units.py, tests/test_smoke_configs.py, tests/test_temperature_profile.py, docs/SPEC.md, docs/PARAMETERS_CHECK.md | pytest tests/ passed (27/27); verified configured period and phase interval strictly respected without fallbacks | none
- 2026-10-10 | P1.1 | Implemented temperature variable profile and unit scaling (Celsius, Fahrenheit, random affine) driven by config/generator_a.yaml, ensuring unit scaling applied last | src/sim/units.py, src/sim/profile.py, src/sim/__init__.py, tests/test_temperature_profile.py | pytest tests/ passed (25/25); verified determinism, Appendix A.2 parameter ranges, probability distributions, affine inversion, and ground truth isolation | none

- 2026-10-10 | P0.4 | Finalized remaining P0 engineering choices (Decisions 1-11: random affine offset ENG assumption, small group n=4, no clear majority <=50%, sensor near bounds in canonical Celsius); synced configs, docs, tests | config/generator_a.yaml, config/alarm.yaml, docs/SPEC.md, docs/PARAMETERS_CHECK.md, PROGRESS.md, tests/test_smoke_configs.py | pytest tests/ passed (10/10), standalone smoke test passed, all parameters verified against Appendix A | none
- 2026-10-10 | P0.4 | Implemented and executed smoke test loading all 4 YAML configs, verifying schema & keys, and printing parameters | tests/test_smoke_configs.py | pytest tests/ passed (10/10); python tests/test_smoke_configs.py exited 0 | none
- 2026-10-10 | P0.3 | Created generator_a.yaml, features.yaml, alarm.yaml, eval.yaml with exact Appendix A values; wrote docs/SPEC.md and docs/PARAMETERS_CHECK.md | config/*.yaml, docs/SPEC.md, docs/PARAMETERS_CHECK.md | all config keys verified against Appendix A text in docs/PARAMETERS_CHECK.md | none
- 2026-10-10 | P0.2 | Run stamp utility with run id, seed, config hash, UTC timestamp, save/load | src/utils/stamp.py, tests/test_stamp.py | pytest tests/test_stamp.py passed (4/4) | none
- 2026-10-10 | P0.1 | Created repo layout, Python 3.11 virtual environment, requirements.txt, config loader with SHA-256 hash | requirements.txt, .gitignore, README.md, src/**, tests/test_config.py, src/utils/config.py | pytest tests/test_config.py passed (5/5) | none

## 8. DECISION LOG (changes to locked parameters or plan; user approval required)

- 2026-10-10 | Decision 1 (ENG assumption, not official PS parameter): Unit scaling random affine additive offset range set to Uniform [-10.0, 10.0] C.
- 2026-10-10 | Decision 2 (implementation choice): "Small group" operationalized as exactly 4 sensors (n=4).
- 2026-10-10 | Decision 3 (implementation choice): "No clear majority" operational definition locked as no cluster of valid sensor readings containing > 50% of sensors (50/50 split in an even-sized group qualifies).
- 2026-10-10 | Decision 4 (implementation choice): "Sensor near bounds" operational definition locked as pre-clipping environment or sensor reading within max(3 x sigma_k, 2.0 C) of corresponding clipping boundary (-5 C / 50 C environment, -10 C / 60 C sensor) in canonical Celsius before unit scaling.
- 2026-10-10 | Decision 5 (implementation choice): Sustained environmental events hold at elevated level until scenario end (step 1920) after specified rise (5 to 20 min).
- 2026-10-10 | Decision 6 (implementation choice): Ground-truth sensor_untrusted defined by active fault-label intervals, including explicit spike window (samples + 4 steps) and slow degradation threshold (d >= 0.15).
- 2026-10-10 | Decision 7 (implementation choice): Trust status exact boundaries locked as HEALTHY when T > 0.8, WATCH when 0.5 <= T <= 0.8, and UNRELIABLE when T < 0.5.
- 2026-10-10 | Decision 8 (implementation choice): Max faulty sensors formula max(1, floor(0.4 * n)) acts as an upper bound cap on simultaneous faults that preserves the 15% zero-fault healthy scenarios.
- 2026-10-10 | Decision 9 (implementation choice): Shared-cause similarity tolerance defined as relative difference <= 20% on normalized fault severity (|s1 - s2| / max(s1, s2) <= 0.20).
- 2026-10-10 | Decision 10 (implementation choice): Subtle faults in HARD tier defined strictly from source ranges specified in Appendix A.4 (gain error g in [0.90, 0.95] or [1.05, 1.10]); no unapproved numerical thresholds added.
- 2026-10-10 | Decision 11 (implementation choice): Row stride 5 used for remaining 150 test scenarios, with 150 test scenarios evaluated at full resolution (stride 1).
- 2026-10-10 | Decision 12 (implementation choice): Daily-cycle random phase interval explicitly configured as [0.0, 6.283185307179586] radians ([0, 2pi)) in generator_a.yaml (operationalizing "random phase" in Appendix A.2) and period_minutes required explicitly without hard-coded fallbacks.
- 2026-10-10 | Decision 13 (implementation choice): Slow random wander generated via discrete Ornstein-Uhlenbeck AR(1) process with decay factor alpha = exp(-dt/tau) and innovation std = sigma * sqrt(1 - alpha^2), initialized in stationary distribution with (M-1) warm-up steps so the 5-step causal moving average is fully defined from step 0.
- 2026-10-10 | Decision 14 (approved engineering choice, user approved): Exactly 20% of virtual sensors have a missing-data rate of zero. For the remaining 80%, each sensor's missing-data rate is sampled independently from Uniform[0.0, 0.01] (0%–1%) and kept fixed for that sensor for the scenario. Missing bursts may last up to the configured maximum of 3 consecutive readings. Missing readings are represented consistently as NaN (never substituted with zero). Recorded in config/generator_a.yaml under structure.healthy_missing_data.
- 2026-10-10 | Decision 15 (implementation choice): Target missing readings M = round(r * T) are partitioned into burst lengths L ~ DiscreteUniform[1, burst_max] (burst_max=3) and placed at non-adjacent starting positions (buffered by at least 1 valid reading). This guarantees that bursts never merge and consecutive missing readings strictly never exceed 3.
- 2026-10-10 | Decision 16 (implementation choice): For the 20% of sensors designated as correlated in a group, Gaussian measurement noise is generated via a 1-factor model eta_{k, t} = sigma_k * (sqrt(rho_k) * xi_t + sqrt(1 - rho_k) * z_{k, t}) where xi_t ~ N(0, 1) is group-shared noise and z_{k, t} ~ N(0, 1) is idiosyncratic noise. This strictly preserves each sensor's marginal variance sigma_k^2 and produces pairwise correlation sqrt(rho_j * rho_k) in [0.2, 0.5] without requiring external covariance solvers.
- 2026-10-10 | Decision 17 (implementation choice): Discretized continuous-time physical lag tau * dy/dt + y = u with time step dt = 1.0 min via exponential smoothing parameter beta = exp(-dt / tau) for tau > 0 (beta = 0 for tau = 0), initialized in thermal equilibrium y[0] = u[0] to eliminate artificial startup transient at step 0.
- 2026-10-10 | P0.4 implementation choice: Added key string-normalization in compute_config_hash to guarantee deterministic comparison for mixed int/str dictionary keys.
- 2026-10-10 | P0.1 implementation choice: Used Python 3.11 for ML package compatibility; implemented config loader in src/utils/config.py with canonical JSON serialization for deterministic SHA-256 hash calculation.
- 2026-10-10 | Generator B parameters (old A.12) removed from this file so the agent building Generator A cannot see them; Generator B moves to a separate SPEC_B.md for a different agent. P8 rewritten accordingly. Robust spread in A.9 defined explicitly.
- 2026-10-10 | Raw alarm persistence changed from 3 readings to **1 reading** (spikes last only 1 to 3 readings; conventional alarms fire on first crossing). "Critical event" level dropped for v1.

---

## APPENDIX A. LOCKED v1.1 PARAMETERS

Basis tags: ENG engineering assumption, SYN synthetic-data design, PS problem statement. Sigma_k = a sensor's true noise (simulator only). All faults are sized in sigma_k.

### A.1 Scenarios and structure
| Parameter | Value |
|---|---|
| Sampling | 1 reading per minute |
| Scored steps / warm-up / total | 1,440 / 480 / 1,920 |
| Scenarios (Generator A) | 1,500: 900 train, 300 val, 300 test; stratified by difficulty and fault type |
| Row stride | 5 for train/val; full resolution for 150 test scenarios |
| Sensors per group | integer 4 to 8 |
| Scenario types | 82% standard, 10% two groups (4 to 6 sensors each), 8% single-sensor |
| Max simultaneous faulty sensors | max(1, floor(0.4 x n)); stress set (Generator B) up to 60% |
| Seeds | A: 1 to 1,500; B: 10,001 and up; shuffle controls: 20001 to 20005 |
| Healthy missing data | per-sensor rate 0 to 1% (20% of sensors 0), bursts up to 3 |

### A.2 Temperature profile (before unit scaling)
| Parameter | Value |
|---|---|
| Base level | uniform 2 to 38 C |
| Daily cycle | amplitude uniform 0.5 to 4 C (20% of scenarios zero); period 24 h; random phase |
| Slow wander | std 0.2 to 0.8 C; correlation time 120 to 360 min; then 5-step moving average |
| Sensor noise sigma_k | log-uniform 0.05 to 0.30 C; Gaussian; 20% of sensors mildly correlated (rho 0.2 to 0.5) |
| Static offset (healthy) | uniform +/-1.0 C per sensor |
| Response lag | first-order, time constant 0 to 6 min (40% of sensors zero) |
| Quantisation | 0.01 (25%), 0.05 (25%), 0.1 (35%), 0.25 (15%) |
| Bounds | environment clipped to [-5, 50] C; sensor range [-10, 60] C |
| Unit scaling (last) | 70% C, 15% F, 15% random affine (multiplier log-uniform 0.5 to 3, random offset) |

### A.3 Genuine events
| Parameter | Value |
|---|---|
| Count | Poisson mean 1.2, max 3 (about 30% of scenarios none) |
| Types | step with exponential recovery 40%, ramp 25%, pulse 25%, sustained shift 10% |
| Magnitude | log-uniform 0.5 to 10 C; 75% positive, 25% negative |
| Durations | recovery tau 10 to 120 min; ramp rise/fall 20 to 180 min, hold 30 to 180 min; pulse 5 to 30 min; sustained rise 5 to 20 min then hold |
| Scope | global 55% (attenuation 0.6 to 1.0); partial 30% (40 to 80% of sensors, attenuation 0.3 to 1.0); local 15% (1 to 2 sensors, attenuation 0.7 to 1.0) |
| Arrival delay per sensor | 0 to 10 min (plus lag) |
| Detectability | `genuine_event` = 1 while event contribution at >= 1 sensor exceeds 3 x sigma_k; else `event_below_detectability` = 1 (excluded or reported separately) |
| "Sensor is changing" (feature rule) | median of last 3 valid readings minus median of the 3 valid readings ending 15 steps earlier (short) or 60 steps earlier (medium); changing if abs change > 3 x sigma_hat |

### A.4 Faults (start uniform after warm-up; 10% of fault episodes present from step 0)
| Fault | Parameters |
|---|---|
| Spike | episode 30 to 400 steps (log-uniform); a spike every 10 to 60 steps; size 3 to 40 sigma_k (log-uniform), random sign; length 1 sample 60%, 2 samples 25%, 3 samples 15%; label active on spike samples plus 4 following steps |
| Offset drift | ramp 60 to 720 steps; final bias 1 to 20 sigma_k (log-uniform), random sign; plateau 0 to 600 steps; shape linear 55%, saturating 20%, stepped 15%, sudden constant offset 10% |
| Gain error | reading = x_ref + g (x - x_ref), x_ref = true value at fault start; g in 0.55 to 0.85 or 1.15 to 1.6 (85%), subtle 0.90 to 0.95 or 1.05 to 1.10 (15%); duration 120 to 1,200 steps |
| Frozen | 20 to 400 steps (log-uniform); 60% exactly constant, 40% jitter (std 0.02 to 0.3 sigma_k); frozen at last reading 80% or at a wrong value 5 to 20 sigma_k away 20% |
| High noise | multiplier 2 to 10 x sigma_k (log-uniform); 30 to 600 steps; 30% bursty (on/off blocks 5 to 30 steps) |
| Dropout | enter-gap probability 0.002 to 0.03 per step (log-uniform); gap length geometric, mean 2 to 40, cap 100; episode 60 to 600 steps |
| Slow degradation | health h falls 1 to h_end (0.1 to 0.7) over 240 to 1,200 steps; damage d = 1 - h; noise x(1 + 5d), bias +/-8 sigma_k x d, extra missing rate 0.15 d; **only** its own label (not also drift/noise/dropout); untrusted once d >= 0.15 |

### A.5 Combinations
- Faulty sensors per scenario: 0: 15%, 1: 40%, 2: 25%, 3 or more: 20% (subject to cap). Types per faulty sensor: 1: 65%, 2: 27%, 3: 8%.
- Allowed pairs: spike+drift, spike+noise, spike+gain, drift+noise, drift+gain, gain+noise, dropout with anything, degradation+spike, degradation+gain.
- Forbidden: frozen with spike/gain/noise/drift; degradation with drift/noise/dropout.
- Second fault starts within [first start - 100, first end]. 35% of fault episodes start inside an active event. Shared-cause pair in 15% of multi-fault scenarios (same fault type on two sensors, starts within +/-10 steps, similar size).

### A.6 Difficulty tiers (evaluation only; never a model input)
- EASY 30%: at most 1 faulty sensor, single clearly visible fault (spike >= 8 sigma, drift >= 6 sigma, |g-1| >= 0.25, noise multiplier >= 4), global events or none.
- MEDIUM 40%: up to 2 faulty sensors, up to 2 fault types each, any event scope.
- HARD 30%: at least one hard mechanism (local event; fault during event; subtle fault; shared-cause pair; faulty fraction at cap with small group; no clear majority; sensor near bounds).

### A.7 Ground truth saved separately (never given to models)
- Per sensor per step: true_value, sensor_untrusted, f_spike, f_offset_drift, f_gain_error, f_frozen, f_high_noise, f_dropout, f_slow_degradation, fault_severity, true_health, fault_id.
- Per group per step: genuine_event, event_below_detectability, event_id, event_magnitude_sigma, true_event_breadth, true_group_value.
- Episode tables (faults, events) and a scenario table (split, generator version, seed, difficulty, hard-case type, sensor count, length, sampling, profile, quantisation, clip limits, unit scale, faulty fraction).
- true_health = 1 - max active severity. Severity: spike k/40; drift abs(bias)/(20 sigma_k); gain abs(g-1)/0.6; frozen 0.8 (1.0 if frozen at wrong value); noise (m-1)/9; dropout 0.3 + 0.7 x missing fraction over last 30 steps; degradation d. All clipped to [0, 1].

### A.8 Features: building blocks
- **sigma_hat:** 1.4826 x MAD of first differences, divided by sqrt(2), taking the 25th percentile of rolling estimates over the 480-step reference.
- **Windows:** short 5, medium 30, long 120, reference 480 (readings). Causal only.
- **Gap centring:** gap features are measured relative to the sensor's own median gap over the 480-step reference (so static offsets are not faults).
- Group reference **excludes** the sensor being judged. With a single sensor, cross-sensor features are NaN.

### A.9 Raw alarm and trust-aware alarm layer
- **Raw alarm (the "conventional threshold" baseline):** reading outside [8-hour median +/- 3 x the sensor's own robust spread of its levels over those 8 hours, where robust spread = 1.4826 x MAD of the readings themselves (not of differences)]; **starts at 1 reading** outside the band; ends after 10 consecutive readings inside; episodes merge when gaps are 5 readings or fewer.
- **Cause label for each raw alarm (from truth):** EVENT, FAULT, BOTH, NEITHER, using the 3 x sigma_k rule on event and fault contributions.
- **Decision** (average calibrated probabilities over the first 5 readings of the episode; re-check every 5):
  | Real event? | Sensor trustworthy? | Decision |
  |---|---|---|
  | yes | yes | Real alarm |
  | yes | no | Real alarm, sensor flagged for inspection |
  | no | no | Suppressed (probable false alarm), sensor flagged |
  | no | yes | Unverified (lower priority) |
- Probability cutoffs start at 0.5 and are chosen on **validation** scenarios to meet the false-alarm budget.

### A.10 Provable rules
- **Frozen:** 20 identical consecutive readings AND more than 3 x the longest identical run in that sensor's previous 480 readings.
- **Dropout:** 5 or more consecutive missing, OR 20% or more missing in the last 30.

### A.11 Calibration, trust, status
- Calibration: isotonic when a fault type has 500 or more positives, otherwise sigmoid; on out-of-fold predictions (5 folds grouped by scenario) or a separate calibration set; never the test set.
- Trust: instant trust = 1 - p(untrusted); smoothing alpha 0.5 when falling, 0.05 when recovering (choose on validation from {0.3,0.5,0.7} x {0.02,0.05,0.1}).
- Status: above 0.8 HEALTHY, 0.5 to 0.8 WATCH, below 0.5 UNRELIABLE.
- Recommended action: stable constant bias, low noise: recalibrate; persistent noise/dropout/low trust: inspect; severe ongoing degradation: replace. Suggestions only.

### A.12 Generator B
Generator B is specified in a separate file (SPEC_B.md) and built by a different agent. It is intentionally **not** described here. Do not look for it, guess it, or make Generator A resemble it.

### A.13 Demo presets (streaming mode)
- Start with at least 480 steps of healthy history before the judge acts.
- Playback 1 second per simulated minute (adjustable up to 10x); chart shows the last 240 readings.
- Presets must be strong enough to be visible: Spike about 20 x sigma_k; Drift reaches about 10 x sigma_k within a few simulated minutes; Event magnitude about 5 C; durations shortened accordingly. Presets are for demonstration only and are labelled as such.
- "Reveal truth" only works on simulated data.

### A.14 Evaluation settings
- False-alarm episode: a run of flagged readings (merge gap 5) with no fault active (5-reading grace). **Budget: at most 0.5 episodes per healthy sensor per day.**
- Onset grace window 5 steps; detection delay = first step with calibrated p >= 0.5.
- Wilson 95% intervals on rates; 1,000 scenario-level bootstrap resamples; flag cells with n < 20; worst-slice search with at least 25 examples.
- Leakage audit: single-feature AUC above 0.85 is investigated; shuffle control must score near chance (AUC within 0.50 +/- 0.03 and recall at budget at most 0.05); a planted canary column must be caught (AUC >= 0.95).
- Speed and size: at most 50 ms per sensor reading (features plus all models) on a laptop CPU; model files at most 50 MB.

## APPENDIX B. FEATURES (about 48 per sensor row; about 11 per group row)

All names are generic (never "temperature_change").
- **Own-signal temporal (10):** diff_1, absdiff_max_s, absdiff_max_m, slope_s, slope_m, slope_l, slope_change, dev_baseline, dev_duration, n_spikes_m.
- **Statistical (6):** noise_ratio_s, noise_ratio_m, noise_ratio_l, noise_short_long, kurtosis_m, range_m.
- **Data quality (10):** cur_missing, frac_missing_m, frac_missing_l, consec_missing, longest_gap_m, steps_since_valid, repeat_run_len, longest_repeat_m, frac_zero_diff_m, unique_ratio_m (plus dist_to_bound when useful). Repeats are compared with the sensor's own normal repeat rate (healthy quantised sensors repeat values).
- **Cross-sensor (14):** gap, gap_mean_m, gap_slope_m, gap_change_m, gap_absmax_m, gap_std_m, gap_autocorr_m, others_spread, others_agreement, diff_corr_m, net_change_gap_m, gain_slope_l, gain_r2_l, n_valid_others.
- **Long-term degradation (3):** noise_growth, missing_growth, gap_slope_xl.
- **Group/event joined onto sensor rows (6):** event_breadth_s, event_breadth_m, group_change_s, group_change_m, group_dispersion, same_direction_frac.
- **Group table for Head C (about 11):** the six group/event features plus group_slope_s, group_dev_baseline, n_valid_sensors, group_peak_m, group_noise_ratio.

## APPENDIX C. PASS MARKS (DRAFT; user confirms before any test results exist)

At the false-alarm budget of 0.5 per healthy sensor per day:
- PM1: fault recall beats the best simple baseline by at least 10 points on HARD, 5 on MEDIUM, and is not worse on EASY.
- PM2: on EASY and MEDIUM (fault types with 20+ examples): recall at least 0.90 for rule-provable faults (frozen, dropout), at least 0.60 for every other fault type.
- PM3: at least 70% of fault-only raw alarms suppressed. PM4: at most 5% of event-only raw alarms wrongly suppressed.
- PM5: in BOTH cases, at least 60% get both flags right.
- PM6: recall drops by at most 20 points from Generator A to B. PM7: calibration error at most 0.05.
- PM8: trust versus true health, rank correlation at least 0.75. PM9: median detection delay at most 15 readings for faults above 5 sigma.
- PM10: leakage audit passes. PM11: speed and size limits met.
If a pass mark fails, report it honestly. Do not move the goalposts.

## HOW THE USER STARTS EACH SESSION (prompt template)

> Read PROGRESS.md (and docs/SPEC.md if it exists). Tell me the current phase and the next unchecked task, then do only that task. Follow Section 0 strictly. When finished, update PROGRESS.md (checkbox, Log entry, CURRENT STATUS) and tell me what to verify.
