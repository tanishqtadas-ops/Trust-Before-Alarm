# SPEC.md: Sensor Trust Intelligence Specification

> Frozen specification record derived from PROGRESS.md (v1.1) and finalized engineering choices.
> Project: VIT Code Apex 3.0, Track 4 (UST), Problem Statement 3: AI-Based Sensor Trust, Drift and False-Alarm Detection.
> Date: 2026-10-10.

## 1. Project Overview & Core Principle
Sensor Trust Intelligence is a software-only intelligence system for multi-sensor streams (v1: temperature) governed by the **"Trust Before Alarm"** principle: an abnormal reading is not automatically an alarm. The system evaluates whether the sensor is trustworthy, identifies active fault modes, verifies if an environmental event is occurring, and makes calibrated decisions to suppress false alarms or escalate genuine emergencies.

## 2. Architecture & Data Layers
### 2.1 Pipeline Flow
```
Sources: Official dataset (if released) | Generator A (dev) | Generator B (robustness only)
   -> Data adapter -> common raw schema
   -> Causal windows -> Feature engine
   -> Sensor-level table (Heads A, B) | Group-level table (Head C)
   -> Head A: untrusted?   Head B: which faults? (7 binary models)   Head C: genuine event?
   -> Provable rules (exact frozen, obvious dropout) + ML scores
   -> Probability calibration
   -> Trust engine (smoothed score, status, recommended action)
   -> Trust-aware alarm layer (real / real + sensor flagged / suppressed / unverified)
   -> Explanations (top 3 reasons, plain sentences)
   -> API + Dashboard (live fault injection, reveal truth, evidence tab)
```

### 2.2 Four Data Layers
1. **Layer 1: Observable Data:** Raw sensor readings (`scenario_id, group_id, sensor_id, variable_type, timestamp, value`). Only layer visible to feature extraction and models.
2. **Layer 2: Derived Features:** Causal windows, own-signal temporal, statistical, data quality, cross-sensor, and group aggregates.
3. **Layer 3: Ground Truth:** Separately stored answer key (`true_value, sensor_untrusted, f_*, fault_severity, true_health, genuine_event, event_*`). Strictly hidden from models; used solely by evaluation code.
4. **Layer 4: Predictions & Decisions:** Calibrated probabilities, trust score, operational status, alarm actions, explanations.

## 3. Non-Negotiable ML Rules
- Model pipeline reads exclusively Layer 1 (raw observable readings). Ground truth stored in separate files.
- Strictly causal: features at time $t$ use readings at or before $t$. Future data is prohibited.
- No `scenario_id`, `sensor_id`, `group_id`, `variable_type`, `timestamp`, or raw `value` as model features in v1.
- Splitting is performed strictly by whole scenarios stratified by difficulty and fault type.
- Fixed random seeds; test set evaluated strictly once at the end.
- Group cross-sensor references exclude the target sensor being evaluated.
- Every run carries run id, seed, config hash, and timestamp.
- Core ML model is LightGBM; SMOTE is prohibited.

## 4. Finalized P0 Engineering Choices & Operational Definitions
1. **Decision 1 — Random Affine Offset (Appendix A.2):** Range $[-10.0, 10.0]^\circ\text{C}$, Uniform distribution. Explicitly classified as an engineering assumption (ENG), not an official problem-statement parameter.
2. **Decision 2 — "Small Group" (Appendix A.6, HARD Tier):** Operationalized as exactly 4 sensors ($n=4$, lower bound of Appendix A.1 integer 4 to 8 sensors).
3. **Decision 3 — "No Clear Majority" (Appendix A.6, HARD Tier):** Operationalized as occurring when no cluster of valid sensor readings contains more than 50% of the sensors; an exact 50/50 split in an even-sized group qualifies.
4. **Decision 4 — "Sensor Near Bounds" (Appendix A.6, HARD Tier):** Operationalized as occurring when the relevant pre-clipping environment or sensor reading is within $\max(3\sigma_k, 2.0^\circ\text{C})$ of its corresponding clipping boundary ($-5.0^\circ\text{C}$ or $50.0^\circ\text{C}$ for environment; $-10.0^\circ\text{C}$ or $60.0^\circ\text{C}$ for sensor). This check is evaluated in canonical Celsius units before unit scaling (Fahrenheit or random affine).
5. **Decision 5 — Sustained Environmental Events (Appendix A.3):** Interpreted as holding at elevated level until scenario end ($t=1920$) after specified rise (5 to 20 min).
6. **Decision 6 — Ground-Truth `sensor_untrusted` Label (Appendix A.4, A.7):** Defined by active fault intervals, including explicit spike window (spike samples + 4 steps) and slow degradation threshold ($d \ge 0.15$).
7. **Decision 7 — Trust Status Exact Boundaries (Appendix A.11):**
   - HEALTHY: $T > 0.8$
   - WATCH: $0.5 \le T \le 0.8$
   - UNRELIABLE: $T < 0.5$
8. **Decision 8 — Simultaneous Fault Cap (Appendix A.1 vs A.5):** Formula $\max(1, \lfloor 0.4 \times n \rfloor)$ acts strictly as an upper bound cap on simultaneous faults; preserves 15% zero-fault healthy scenarios.
9. **Decision 9 — Shared-Cause Severity Similarity (Appendix A.5):** Defined as within $\pm 20\%$ relative tolerance on normalized fault severity ($|s_1 - s_2| / \max(s_1, s_2) \le 0.20$).
10. **Decision 10 — Subtle Faults (Appendix A.4, A.6):** Defined strictly from ranges specified in the source (subtle gain error $g \in [0.90, 0.95] \cup [1.05, 1.10]$).
11. **Decision 11 — Test Set Stride Split (Appendix A.1):** 150 test scenarios at full resolution (stride 1); remaining 150 test scenarios at stride 5 (matching train/val).
12. **Decision 12 — Daily Cycle Phase Interval (Appendix A.2):** Range $[0, 2\pi)$ radians ($[0.0, 6.283185307179586]$), Uniform distribution. Documented as the implementation interpretation of "random phase" in Appendix A.2, not as a numerical range explicitly stated by the original specification.
13. **Decision 13 — Slow Random Wander AR(1) Discretization (Appendix A.2):** Discrete Ornstein-Uhlenbeck AR(1) process with decay factor $\alpha = \exp(-\Delta t / \tau)$ and innovation standard deviation $\sigma_{\text{innov}} = \sigma \sqrt{1 - \alpha^2}$, initialized in the stationary distribution with $(M - 1)$ pre-steps to ensure the 5-step moving average is fully defined from step 0.
14. **Decision 14 — Healthy Missing-Data Distribution (Appendix A.1, Approved Engineering Choice & Correction):** Target population proportion 20% zero missing rate. Since groups contain 4–8 sensors (where exact 20% integer partition is mathematically impossible within every group), each sensor independently receives a zero missing-data rate with probability 0.20 via a seeded Bernoulli trial. For each remaining sensor, its missing-data rate is sampled independently from $\text{Uniform}[0.0, 0.01]$ ($0\%\text{--}1\%$) and kept fixed for that sensor for the scenario. Missing bursts may last up to the configured maximum of 3 consecutive readings. Missing readings are represented as `NaN` (never substituted with zero).
15. **Decision 15 — Healthy Missing-Burst Placement (Appendix A.1, Implementation Choice):** Target missing readings $M = \text{round}(r \times T)$ are partitioned into burst lengths $L \sim \text{DiscreteUniform}[1, \text{burst\_max}]$ ($\text{burst\_max}=3$) and placed at non-adjacent starting positions (buffered by at least 1 valid reading). This guarantees that bursts never merge and consecutive missing readings strictly never exceed 3.
16. **Decision 16 — Mildly Correlated Sensor Noise Factor Model (Appendix A.2, Implementation Choice):** For the 20% of sensors designated as correlated in a group, Gaussian measurement noise is generated via a 1-factor model $\eta_{k, t} = \sigma_k (\sqrt{\rho_k} \xi_t + \sqrt{1 - \rho_k} z_{k, t})$ where $\xi_t \sim \mathcal{N}(0, 1)$ is group-shared noise and $z_{k, t} \sim \mathcal{N}(0, 1)$ is idiosyncratic noise. This strictly preserves each sensor's marginal variance $\sigma_k^2$ and produces pairwise correlation $\sqrt{\rho_j \rho_k} \in [0.2, 0.5]$ without requiring external covariance solvers.
17. **Decision 17 — First-Order Response Lag Discretization (Appendix A.2, Implementation Choice):** Discretized continuous-time physical lag $\tau \frac{dy}{dt} + y = u$ with time step $\Delta t = 1.0$ min via exponential smoothing parameter $\beta = \exp(-\Delta t / \tau)$ for $\tau > 0$ ($\beta = 0$ for $\tau = 0$), initialized in thermal equilibrium $y[0] = u[0]$ to eliminate artificial startup transient at step 0.

