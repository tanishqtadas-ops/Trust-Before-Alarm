# Sensor Trust Intelligence

AI-Based Sensor Trust, Drift and False-Alarm Detection.
Project: VIT Code Apex 3.0, Track 4 (UST), Problem Statement 3.

## Overview
A software-only sensor intelligence system watching readings from related sensors to continuously answer:
1. **Is this sensor trustworthy right now, and if not, what fault does it have?** (spike, offset drift, gain error, frozen, high noise, dropout, slow degradation)
2. **Is something genuinely happening in the environment?** (a real event, which can coincide with a faulty sensor)
3. **Should this alarm be trusted, suppressed or escalated?** and what action is recommended (recalibrate, inspect, replace)

Key principle: **"Trust Before Alarm"** — an abnormal reading is not automatically an alarm. First verify sensor trust and environmental state.

## Repository Layout
```
README.md        Project overview and documentation
PROGRESS.md      Task tracker, rules, and decision log
docs/            Detailed project specs and architecture
config/          Configuration parameters (YAML)
src/
  sim/           Simulation environment, sensors, events, faults
  features/      Causal windowing, sigma-hat, and feature generation
  models/        Rules, baseline models, LightGBM heads, calibration, trust
  eval/          Evaluation metrics, audits, ablation studies, reports
  api/           REST/Streaming API
  app/           Dashboard
  utils/         Configuration and run tracking utilities
tests/           Automated test suite
data/            Generated synthetic datasets (git-ignored)
runs/            Run outputs, stamps, and artifacts (git-ignored)
```

## Setup & Environment
Ensure Python 3.11+ is installed.
```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

## Running Tests
```bash
pytest -v
```
