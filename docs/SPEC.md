# SPEC.md: Sensor Trust Intelligence Specification

> Frozen specification record derived from PROGRESS.md (v1.1).
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

## 4. Locked v1.1 Parameters (Appendix A Reference)
See `config/generator_a.yaml`, `config/features.yaml`, `config/alarm.yaml`, and `config/eval.yaml` for complete machine-readable parameters. Detailed audit mapping is catalogued in `docs/PARAMETERS_CHECK.md`.
