# SPEC.md: Sensor Trust Intelligence Specification

> Frozen specification record derived from PROGRESS.md (v1.1) and approved audit implementation choices.
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

## 4. Approved Implementation Choices (Post-Audit Review)
1. **Random Affine Offset (Appendix A.2):** Range $[-10.0, 10.0]^\circ\text{C}$, Uniform distribution. Recorded as an implementation choice, not an original Appendix A parameter.
2. **Sustained Environmental Events (Appendix A.3):** Interpreted as holding at elevated level until scenario end ($t=1920$) after specified rise (5 to 20 min).
3. **Ground-Truth `sensor_untrusted` Label (Appendix A.4, A.7):** Defined by active fault intervals, including explicit spike window (spike samples + 4 steps) and slow degradation threshold ($d \ge 0.15$).
4. **Trust Status Exact Boundaries (Appendix A.11):**
   - HEALTHY: $T > 0.8$
   - WATCH: $0.5 \le T \le 0.8$
   - UNRELIABLE: $T < 0.5$
5. **Simultaneous Fault Cap (Appendix A.1 vs A.5):** Formula $\max(1, \lfloor 0.4 \times n \rfloor)$ acts strictly as an upper bound cap on simultaneous faults; preserves 15% zero-fault healthy scenarios.
6. **Shared-Cause Severity Similarity (Appendix A.5):** Defined as within $\pm 20\%$ relative tolerance on normalized fault severity ($|s_1 - s_2| / \max(s_1, s_2) \le 0.20$).
7. **Subtle Faults (Appendix A.4, A.6):** Defined strictly from ranges specified in the source (subtle gain error $g \in [0.90, 0.95] \cup [1.05, 1.10]$).
8. **Test Set Stride Split (Appendix A.1):** 150 test scenarios at full resolution (stride 1); remaining 150 test scenarios at stride 5 (matching train/val).

## 5. Proposed Operational Definitions (Pending Approval)
- **"Small Group":** Confirmed as 4 sensors ($n=4$, lower bound of Appendix A.1 integer 4 to 8 sensors).
- **"No Clear Majority" (HARD tier):** Group split where exactly $\lfloor n/2 \rfloor$ sensors diverge from the remaining sensors, or dispersion is bimodal such that no cluster constitutes $> 50\%$ of sensors.
- **"Sensor Near Bounds" (HARD tier):** Sensor readings within $3\sigma_k$ or $\le 2^\circ\text{C}$ of physical clipping limits ($-5^\circ\text{C}$ / $50^\circ\text{C}$ environment; $-10^\circ\text{C}$ / $60^\circ\text{C}$ sensor).
