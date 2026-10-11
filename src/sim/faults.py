"""Fault injectors for sensor stream intelligence.

Implements Appendix A.4 and Appendix A.7 of the specification:
  1. Spike: episode 30 to 400 steps (log-uniform), inter-spike intervals 10 to 60 steps,
     size 3 to 40 sigma_k (log-uniform, random sign), length 1 to 3 samples (60%, 25%, 15%),
     label active on spike samples plus 4 following steps, severity k / 40.
  2. Offset drift: ramp 60 to 720 steps, final bias 1 to 20 sigma_k (log-uniform, random sign),
     plateau 0 to 600 steps, shapes linear (55%), saturating quadratic (20%), stepped (15%),
     sudden constant offset (10%), severity |bias| / (20 sigma_k).
  3. Gain error: duration 120 to 1,200 steps, standard gain in [0.55, 0.85] or [1.15, 1.60] (85%),
     subtle gain in [0.90, 0.95] or [1.05, 1.10] (15%), reference x_ref = true physical value
     at fault onset, reading = x_ref + g * (x - x_ref), severity |g - 1| / 0.6.
  4. Frozen: duration 20 to 400 steps (log-uniform), 60% constant vs 40% jitter (std 0.02 to 0.3 sigma_k),
     80% last reading vs 20% wrong value (5 to 20 sigma_k away), initial healthy reading for step 0 starts,
     severity 0.8 (normal) or 1.0 (wrong value).
  5. High noise: multiplier 2 to 10 x sigma_k (log-uniform), duration 30 to 600 steps,
     70% continuous vs 30% bursty (on/off blocks 5 to 30 steps), severity (m - 1) / 9.
  6. Dropout: episode 60 to 600 steps, enter-gap prob 0.002 to 0.03 (log-uniform),
     gap length geometric with mean 2 to 40 (capped at 100), severity 0.3 + 0.7 * (missing frac last 30).
  7. Slow degradation: duration 240 to 1,200 steps, health h falls linearly from 1 to h_end (0.1 to 0.7),
     damage d = 1 - h, noise x(1 + 5d), bias +/- 8 sigma_k * d (sign sampled once), extra missing prob 0.15 d,
     untrusted once d >= 0.15, labeled only as slow_degradation, severity d.

Engineering Choices (Decisions 22-28):
  - Decision 22: Stepped drift step count sampled uniformly from {3, 4, 5}.
  - Decision 23: Saturating drift modeled as parameter-free quadratic curve B * (1 - (1 - u)^2).
  - Decision 24: Inter-spike intervals sampled independently from [10, 60] steps (renewal process);
    first spike starts at t_start; overlapping label windows merged via logical OR.
  - Decision 25: Frozen sensor starting at step 0 uses initial healthy reading at step 0.
  - Decision 26: Slow degradation noise std scaled by (1 + 5d), bias sign sampled once at onset,
    extra missingness generated independently per step with probability 0.15 d.
  - Decision 27: Strict 11-step sensor transduction order with unit scaling applied last.
  - Decision 28: Fault start steps sampled uniformly from [480, 1919] for 90% post-warmup faults;
    10% start at step 0; durations truncated at scenario boundary step 1920 without resampling.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import get_config_dir, load_yaml


class FaultType(str, Enum):
    """Enumeration of the seven supported fault modes from Appendix A.4."""
    SPIKE = "spike"
    OFFSET_DRIFT = "offset_drift"
    GAIN_ERROR = "gain_error"
    FROZEN = "frozen"
    HIGH_NOISE = "high_noise"
    DROPOUT = "dropout"
    SLOW_DEGRADATION = "slow_degradation"


def sample_log_uniform(low: float, high: float, rng: np.random.Generator) -> float:
    """Sample a float from a log-uniform (reciprocal) distribution."""
    if low <= 0.0 or high <= 0.0:
        raise ValueError(f"Log-uniform bounds must be positive, got [{low}, {high}]")
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    log_low = float(np.log(low))
    log_high = float(np.log(high))
    return float(np.exp(rng.uniform(log_low, log_high)))


def sample_fault_start_step(
    total_steps: int = 1920,
    warmup_steps: int = 480,
    prob_present_from_start: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Sample fault start step adhering to Appendix A.4 and Decision 28.
    
    With probability prob_present_from_start (0.10), fault starts at step 0.
    With remaining probability (0.90), fault starts uniformly in [warmup_steps, total_steps - 1].
    
    Args:
        total_steps: Total duration of scenario in steps (default 1920).
        warmup_steps: Number of initial warm-up steps (default 480).
        prob_present_from_start: Probability of fault starting at step 0 (default 0.10).
        rng: Numpy Generator instance.
        
    Returns:
        Sampled start step integer.
    """
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < prob_present_from_start:
        return 0
    return int(rng.integers(warmup_steps, total_steps))


@dataclass
class FaultEpisode:
    """Representation of an individual fault episode on a virtual sensor.
    
    Carries complete time series of perturbations and ground truth indicators.
    """
    fault_id: str
    fault_type: FaultType
    sensor_id: str
    start_step: int
    end_step: int
    duration: int
    shape_params: Dict[str, Any]
    additive_offset: np.ndarray
    noise_multiplier: np.ndarray
    missing_mask: np.ndarray
    severity: np.ndarray
    label_mask: np.ndarray
    untrusted_mask: np.ndarray
    gain: Optional[float] = None
    gain_xref: Optional[float] = None
    is_frozen: bool = False
    frozen_value: Optional[float] = None
    frozen_jitter_std: float = 0.0

    @property
    def active_steps_count(self) -> int:
        """Number of scenario steps where the fault is physically active."""
        return max(0, self.end_step - self.start_step)

    @property
    def max_severity(self) -> float:
        """Peak severity observed across all steps."""
        if len(self.severity) == 0:
            return 0.0
        return float(np.max(self.severity))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fault episode metadata."""
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value if isinstance(self.fault_type, FaultType) else str(self.fault_type),
            "sensor_id": self.sensor_id,
            "start_step": int(self.start_step),
            "end_step": int(self.end_step),
            "duration": int(self.duration),
            "active_steps_count": int(self.active_steps_count),
            "max_severity": float(self.max_severity),
            "shape_params": self.shape_params,
        }


def generate_spike_fault(
    sensor_id: str,
    sigma_k: float,
    total_steps: int,
    start_step: int,
    duration: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
    fault_id: str = "fault_0",
) -> FaultEpisode:
    """Generate spike fault episode (Appendix A.4, Decision 24).
    
    A train of brief spikes occurring every 10 to 60 steps within the episode window.
    Label active on spike samples plus 4 following steps.
    """
    spike_cfg = config["faults"]["spike"]
    int_min = int(spike_cfg["interval_min"])
    int_max = int(spike_cfg["interval_max"])
    size_min = float(spike_cfg["size_sigma_min"])
    size_max = float(spike_cfg["size_sigma_max"])
    len_probs = {int(k): float(v) for k, v in spike_cfg["length_probs"].items()}
    lengths = list(len_probs.keys())
    probs = list(len_probs.values())
    label_following = int(spike_cfg["label_following_steps"])
    divisor = float(config["severity_formulas"]["spike_divisor"])

    end_step = min(start_step + duration, total_steps)
    additive_offset = np.zeros(total_steps, dtype=np.float64)
    severity = np.zeros(total_steps, dtype=np.float64)
    label_mask = np.zeros(total_steps, dtype=bool)
    untrusted_mask = np.zeros(total_steps, dtype=bool)

    t = start_step
    spike_events = []
    while t < end_step:
        sp_len = int(rng.choice(lengths, p=probs))
        sp_size_sigma = sample_log_uniform(size_min, size_max, rng)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        amp = sign * sp_size_sigma * sigma_k

        sp_actual_end = min(t + sp_len, end_step)
        if sp_actual_end > t:
            additive_offset[t:sp_actual_end] += amp

        # Label window extends for spike length plus configured following steps
        lbl_end = min(t + sp_len + label_following, total_steps)
        if lbl_end > t:
            label_mask[t:lbl_end] = True
            untrusted_mask[t:lbl_end] = True
            sp_sev = min(sp_size_sigma / divisor, 1.0)
            severity[t:lbl_end] = np.maximum(severity[t:lbl_end], sp_sev)

        spike_events.append({
            "step": int(t),
            "length": int(sp_len),
            "size_sigma": float(sp_size_sigma),
            "sign": int(sign),
            "amplitude": float(amp),
        })

        gap = int(rng.integers(int_min, int_max + 1))
        t = t + sp_len + gap

    shape_params = {
        "spike_count": len(spike_events),
        "spikes": spike_events,
    }

    return FaultEpisode(
        fault_id=fault_id,
        fault_type=FaultType.SPIKE,
        sensor_id=sensor_id,
        start_step=start_step,
        end_step=end_step,
        duration=duration,
        shape_params=shape_params,
        additive_offset=additive_offset,
        noise_multiplier=np.ones(total_steps, dtype=np.float64),
        missing_mask=np.zeros(total_steps, dtype=bool),
        severity=severity,
        label_mask=label_mask,
        untrusted_mask=untrusted_mask,
    )


def generate_offset_drift_fault(
    sensor_id: str,
    sigma_k: float,
    total_steps: int,
    start_step: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
    fault_id: str = "fault_0",
    ramp_steps: Optional[int] = None,
    plateau_steps: Optional[int] = None,
) -> FaultEpisode:
    """Generate offset drift fault episode (Appendix A.4, Decisions 1, 22, 23).
    
    Shapes: linear (55%), saturating quadratic (20%), stepped (15%), sudden constant (10%).
    """
    drift_cfg = config["faults"]["offset_drift"]
    if ramp_steps is None:
        ramp_steps = int(rng.integers(int(drift_cfg["ramp_steps_min"]), int(drift_cfg["ramp_steps_max"]) + 1))
    if plateau_steps is None:
        plateau_steps = int(rng.integers(int(drift_cfg["plateau_steps_min"]), int(drift_cfg["plateau_steps_max"]) + 1))

    duration = ramp_steps + plateau_steps
    end_step = min(start_step + duration, total_steps)

    bias_mag_sigma = sample_log_uniform(
        float(drift_cfg["final_bias_sigma_min"]),
        float(drift_cfg["final_bias_sigma_max"]),
        rng,
    )
    sign = 1.0 if rng.random() < 0.5 else -1.0
    final_bias = sign * bias_mag_sigma * sigma_k

    # Sample trajectory shape
    shapes = list(drift_cfg["shape_probs"].keys())
    probs = [float(drift_cfg["shape_probs"][s]) for s in shapes]
    shape = str(rng.choice(shapes, p=probs))

    stepped_steps_k: Optional[int] = None
    if shape == "stepped":
        k_min = int(drift_cfg.get("stepped_steps_min", 3))
        k_max = int(drift_cfg.get("stepped_steps_max", 5))
        stepped_steps_k = int(rng.integers(k_min, k_max + 1))

    divisor_sigma = float(config["severity_formulas"]["drift_divisor_sigma"])
    additive_offset = np.zeros(total_steps, dtype=np.float64)
    severity = np.zeros(total_steps, dtype=np.float64)
    label_mask = np.zeros(total_steps, dtype=bool)
    untrusted_mask = np.zeros(total_steps, dtype=bool)

    for t in range(start_step, end_step):
        dt = t - start_step
        if shape == "sudden_constant":
            b_t = final_bias
        elif dt < ramp_steps:
            u = dt / float(ramp_steps)
            if shape == "linear":
                b_t = final_bias * u
            elif shape == "saturating":
                # Decision 23: parameter-free quadratic curve B * (1 - (1 - u)^2)
                b_t = final_bias * (1.0 - (1.0 - u) ** 2)
            elif shape == "stepped":
                # Decision 22: discrete step count K in {3, 4, 5}
                k_val = stepped_steps_k if stepped_steps_k is not None else 4
                step_idx = int(np.floor(k_val * u))
                b_t = final_bias * (step_idx / float(k_val))
            else:
                b_t = final_bias * u
        else:
            b_t = final_bias

        additive_offset[t] = b_t
        label_mask[t] = True
        untrusted_mask[t] = True
        sev = min(abs(b_t) / (divisor_sigma * sigma_k), 1.0)
        severity[t] = sev

    shape_params = {
        "ramp_steps": int(ramp_steps),
        "plateau_steps": int(plateau_steps),
        "final_bias_sigma": float(bias_mag_sigma),
        "sign": int(sign),
        "final_bias": float(final_bias),
        "shape": shape,
        "stepped_k": stepped_steps_k,
    }

    return FaultEpisode(
        fault_id=fault_id,
        fault_type=FaultType.OFFSET_DRIFT,
        sensor_id=sensor_id,
        start_step=start_step,
        end_step=end_step,
        duration=duration,
        shape_params=shape_params,
        additive_offset=additive_offset,
        noise_multiplier=np.ones(total_steps, dtype=np.float64),
        missing_mask=np.zeros(total_steps, dtype=bool),
        severity=severity,
        label_mask=label_mask,
        untrusted_mask=untrusted_mask,
    )


def generate_gain_error_fault(
    sensor_id: str,
    total_steps: int,
    start_step: int,
    duration: int,
    physical_xref: float,
    config: Dict[str, Any],
    rng: np.random.Generator,
    fault_id: str = "fault_0",
) -> FaultEpisode:
    """Generate gain error fault episode (Appendix A.4, Decision 5).
    
    Applies reading = x_ref + g * (x - x_ref) where x_ref is the true event-adjusted
    physical temperature in canonical Celsius at fault onset.
    """
    gain_cfg = config["faults"]["gain_error"]
    is_standard = rng.random() < float(gain_cfg["standard_prob"])

    if is_standard:
        ranges = gain_cfg["standard_ranges"]
        tier = "standard"
    else:
        ranges = gain_cfg["subtle_ranges"]
        tier = "subtle"

    # Select low vs high range with 50/50 probability
    if rng.random() < 0.5:
        g = float(rng.uniform(float(ranges["low"][0]), float(ranges["low"][1])))
    else:
        g = float(rng.uniform(float(ranges["high"][0]), float(ranges["high"][1])))

    end_step = min(start_step + duration, total_steps)
    divisor = float(config["severity_formulas"]["gain_divisor"])
    sev_val = min(abs(g - 1.0) / divisor, 1.0)

    severity = np.zeros(total_steps, dtype=np.float64)
    label_mask = np.zeros(total_steps, dtype=bool)
    untrusted_mask = np.zeros(total_steps, dtype=bool)

    if end_step > start_step:
        severity[start_step:end_step] = sev_val
        label_mask[start_step:end_step] = True
        untrusted_mask[start_step:end_step] = True

    shape_params = {
        "gain": float(g),
        "tier": tier,
        "physical_xref": float(physical_xref),
    }

    return FaultEpisode(
        fault_id=fault_id,
        fault_type=FaultType.GAIN_ERROR,
        sensor_id=sensor_id,
        start_step=start_step,
        end_step=end_step,
        duration=duration,
        shape_params=shape_params,
        additive_offset=np.zeros(total_steps, dtype=np.float64),
        noise_multiplier=np.ones(total_steps, dtype=np.float64),
        missing_mask=np.zeros(total_steps, dtype=bool),
        severity=severity,
        label_mask=label_mask,
        untrusted_mask=untrusted_mask,
        gain=g,
        gain_xref=physical_xref,
    )


def generate_frozen_fault(
    sensor_id: str,
    sigma_k: float,
    total_steps: int,
    start_step: int,
    duration: int,
    clean_base_value: float,
    config: Dict[str, Any],
    rng: np.random.Generator,
    fault_id: str = "fault_0",
) -> FaultEpisode:
    """Generate frozen sensor fault episode (Appendix A.4, Decision 3, Decision 25).
    
    Freezes at last reading (80%) or wrong value 5 to 20 sigma_k away (20%).
    Constant (60%) or jitter with std 0.02 to 0.3 sigma_k (40%).
    For step 0 start, clean_base_value is initial healthy equilibrium reading at step 0.
    """
    frozen_cfg = config["faults"]["frozen"]
    is_wrong_value = rng.random() < float(frozen_cfg["frozen_at_wrong_value_prob"])
    has_jitter = rng.random() < float(frozen_cfg["jitter_prob"])

    if is_wrong_value:
        offset_mag_sigma = float(rng.uniform(
            float(frozen_cfg["wrong_value_offset_sigma_min"]),
            float(frozen_cfg["wrong_value_offset_sigma_max"]),
        ))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        offset = sign * offset_mag_sigma * sigma_k
        frozen_val = clean_base_value + offset
        sev_val = float(config["severity_formulas"]["frozen_severity_wrong_value"])
    else:
        offset_mag_sigma = 0.0
        sign = 1.0
        frozen_val = clean_base_value
        sev_val = float(config["severity_formulas"]["frozen_severity_normal"])

    if has_jitter:
        jitter_std_sigma = float(rng.uniform(
            float(frozen_cfg["jitter_std_sigma_min"]),
            float(frozen_cfg["jitter_std_sigma_max"]),
        ))
        jitter_std = jitter_std_sigma * sigma_k
    else:
        jitter_std_sigma = 0.0
        jitter_std = 0.0

    end_step = min(start_step + duration, total_steps)
    severity = np.zeros(total_steps, dtype=np.float64)
    label_mask = np.zeros(total_steps, dtype=bool)
    untrusted_mask = np.zeros(total_steps, dtype=bool)

    if end_step > start_step:
        severity[start_step:end_step] = sev_val
        label_mask[start_step:end_step] = True
        untrusted_mask[start_step:end_step] = True

    shape_params = {
        "is_wrong_value": bool(is_wrong_value),
        "frozen_value": float(frozen_val),
        "clean_base_value": float(clean_base_value),
        "wrong_value_offset_sigma": float(offset_mag_sigma),
        "sign": int(sign),
        "has_jitter": bool(has_jitter),
        "jitter_std_sigma": float(jitter_std_sigma),
    }

    return FaultEpisode(
        fault_id=fault_id,
        fault_type=FaultType.FROZEN,
        sensor_id=sensor_id,
        start_step=start_step,
        end_step=end_step,
        duration=duration,
        shape_params=shape_params,
        additive_offset=np.zeros(total_steps, dtype=np.float64),
        noise_multiplier=np.ones(total_steps, dtype=np.float64),
        missing_mask=np.zeros(total_steps, dtype=bool),
        severity=severity,
        label_mask=label_mask,
        untrusted_mask=untrusted_mask,
        is_frozen=True,
        frozen_value=frozen_val,
        frozen_jitter_std=jitter_std,
    )


def generate_high_noise_fault(
    sensor_id: str,
    total_steps: int,
    start_step: int,
    duration: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
    fault_id: str = "fault_0",
) -> FaultEpisode:
    """Generate high noise fault episode (Appendix A.4).
    
    Noise multiplier 2 to 10 x sigma_k (log-uniform).
    70% continuous, 30% bursty (on/off blocks 5 to 30 steps).
    """
    noise_cfg = config["faults"]["high_noise"]
    mult = sample_log_uniform(
        float(noise_cfg["multiplier_min"]),
        float(noise_cfg["multiplier_max"]),
        rng,
    )
    is_bursty = rng.random() < float(noise_cfg["bursty_prob"])

    end_step = min(start_step + duration, total_steps)
    noise_multiplier = np.ones(total_steps, dtype=np.float64)
    severity = np.zeros(total_steps, dtype=np.float64)
    label_mask = np.zeros(total_steps, dtype=bool)
    untrusted_mask = np.zeros(total_steps, dtype=bool)

    num_offset = float(config["severity_formulas"]["noise_offset"])
    den_divisor = float(config["severity_formulas"]["noise_divisor"])
    sev_val = min((mult - num_offset) / den_divisor, 1.0)

    if not is_bursty:
        if end_step > start_step:
            noise_multiplier[start_step:end_step] = mult
            severity[start_step:end_step] = sev_val
            label_mask[start_step:end_step] = True
            untrusted_mask[start_step:end_step] = True
    else:
        # Bursty on/off blocks
        blk_min = int(noise_cfg["burst_block_steps_min"])
        blk_max = int(noise_cfg["burst_block_steps_max"])
        t = start_step
        is_on = True
        while t < end_step:
            blk_len = int(rng.integers(blk_min, blk_max + 1))
            blk_end = min(t + blk_len, end_step)
            if is_on and blk_end > t:
                noise_multiplier[t:blk_end] = mult
                severity[t:blk_end] = sev_val
                label_mask[t:blk_end] = True
                untrusted_mask[t:blk_end] = True
            is_on = not is_on
            t = blk_end

    shape_params = {
        "multiplier": float(mult),
        "is_bursty": bool(is_bursty),
    }

    return FaultEpisode(
        fault_id=fault_id,
        fault_type=FaultType.HIGH_NOISE,
        sensor_id=sensor_id,
        start_step=start_step,
        end_step=end_step,
        duration=duration,
        shape_params=shape_params,
        additive_offset=np.zeros(total_steps, dtype=np.float64),
        noise_multiplier=noise_multiplier,
        missing_mask=np.zeros(total_steps, dtype=bool),
        severity=severity,
        label_mask=label_mask,
        untrusted_mask=untrusted_mask,
    )


def generate_dropout_fault(
    sensor_id: str,
    total_steps: int,
    start_step: int,
    duration: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
    fault_id: str = "fault_0",
) -> FaultEpisode:
    """Generate dropout fault episode (Appendix A.4).
    
    Enter-gap prob 0.002 to 0.03 per step.
    Gap length geometric with mean 2 to 40, capped at 100.
    Severity: 0.3 + 0.7 * (missing fraction over last 30 steps).
    """
    drop_cfg = config["faults"]["dropout"]
    p_enter = sample_log_uniform(
        float(drop_cfg["enter_gap_prob_min"]),
        float(drop_cfg["enter_gap_prob_max"]),
        rng,
    )
    mean_gap = float(rng.uniform(
        float(drop_cfg["gap_length_mean_min"]),
        float(drop_cfg["gap_length_mean_max"]),
    ))
    cap_gap = int(drop_cfg["gap_length_cap"])
    p_geom = 1.0 / mean_gap

    end_step = min(start_step + duration, total_steps)
    missing_mask = np.zeros(total_steps, dtype=bool)
    severity = np.zeros(total_steps, dtype=np.float64)
    label_mask = np.zeros(total_steps, dtype=bool)
    untrusted_mask = np.zeros(total_steps, dtype=bool)

    t = start_step
    while t < end_step:
        if rng.random() < p_enter:
            gap_len = int(rng.geometric(p_geom))
            gap_len = min(gap_len, cap_gap)
            gap_end = min(t + gap_len, end_step)
            missing_mask[t:gap_end] = True
            t = gap_end
        else:
            t += 1

    # Severity formula: 0.3 + 0.7 * (missing fraction over last 30 steps)
    sev_base = float(config["severity_formulas"]["dropout_base"])
    sev_slope = float(config["severity_formulas"]["dropout_slope"])
    window = int(config["severity_formulas"]["dropout_missing_window"])

    for step in range(start_step, end_step):
        label_mask[step] = True
        untrusted_mask[step] = True
        w_start = max(0, step - window + 1)
        w_slice = missing_mask[w_start:step + 1]
        frac = float(np.mean(w_slice)) if len(w_slice) > 0 else 0.0
        severity[step] = min(sev_base + sev_slope * frac, 1.0)

    shape_params = {
        "enter_gap_prob": float(p_enter),
        "mean_gap_length": float(mean_gap),
        "gap_cap": int(cap_gap),
        "total_missing_in_episode": int(np.sum(missing_mask[start_step:end_step])),
    }

    return FaultEpisode(
        fault_id=fault_id,
        fault_type=FaultType.DROPOUT,
        sensor_id=sensor_id,
        start_step=start_step,
        end_step=end_step,
        duration=duration,
        shape_params=shape_params,
        additive_offset=np.zeros(total_steps, dtype=np.float64),
        noise_multiplier=np.ones(total_steps, dtype=np.float64),
        missing_mask=missing_mask,
        severity=severity,
        label_mask=label_mask,
        untrusted_mask=untrusted_mask,
    )


def generate_slow_degradation_fault(
    sensor_id: str,
    sigma_k: float,
    total_steps: int,
    start_step: int,
    duration: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
    fault_id: str = "fault_0",
) -> FaultEpisode:
    """Generate slow degradation fault episode (Appendix A.4, Decision 4, Decision 26).
    
    Health h falls linearly from 1 to h_end (0.1 to 0.7) over duration.
    Damage d = 1 - h.
    Noise x(1 + 5d), bias +/- 8 sigma_k * d (sign sampled once), extra missing prob 0.15 d.
    Untrusted once d >= 0.15.
    Labeled ONLY as slow_degradation.
    """
    deg_cfg = config["faults"]["slow_degradation"]
    h_end = float(rng.uniform(float(deg_cfg["h_end_min"]), float(deg_cfg["h_end_max"])))
    sign = 1.0 if rng.random() < 0.5 else -1.0

    end_step = min(start_step + duration, total_steps)
    additive_offset = np.zeros(total_steps, dtype=np.float64)
    noise_multiplier = np.ones(total_steps, dtype=np.float64)
    missing_mask = np.zeros(total_steps, dtype=bool)
    severity = np.zeros(total_steps, dtype=np.float64)
    label_mask = np.zeros(total_steps, dtype=bool)
    untrusted_mask = np.zeros(total_steps, dtype=bool)

    noise_slope = float(deg_cfg["damage_noise_multiplier_slope"])
    bias_factor = float(deg_cfg["damage_bias_sigma_factor"])
    missing_slope = float(deg_cfg["damage_extra_missing_rate_slope"])
    untrusted_thresh = float(deg_cfg["untrusted_damage_threshold"])

    for t in range(start_step, end_step):
        u = (t - start_step) / float(duration)
        h_t = 1.0 - (1.0 - h_end) * u
        d_t = 1.0 - h_t

        noise_multiplier[t] = 1.0 + noise_slope * d_t
        additive_offset[t] = sign * bias_factor * sigma_k * d_t
        if rng.random() < missing_slope * d_t:
            missing_mask[t] = True

        label_mask[t] = True
        if d_t >= untrusted_thresh:
            untrusted_mask[t] = True
        severity[t] = min(d_t, 1.0)

    shape_params = {
        "h_end": float(h_end),
        "sign": int(sign),
        "final_damage": float(1.0 - h_end),
    }

    return FaultEpisode(
        fault_id=fault_id,
        fault_type=FaultType.SLOW_DEGRADATION,
        sensor_id=sensor_id,
        start_step=start_step,
        end_step=end_step,
        duration=duration,
        shape_params=shape_params,
        additive_offset=additive_offset,
        noise_multiplier=noise_multiplier,
        missing_mask=missing_mask,
        severity=severity,
        label_mask=label_mask,
        untrusted_mask=untrusted_mask,
    )


@dataclass
class SensorFaultSeries:
    """Aggregated fault time series for a single virtual sensor.
    
    Combines active fault episodes into composite perturbations and ground truth indicators.
    """
    sensor_id: str
    total_steps: int
    episodes: List[FaultEpisode] = field(default_factory=list)

    @property
    def has_faults(self) -> bool:
        """True if at least one fault episode is present."""
        return len(self.episodes) > 0

    @property
    def combined_additive_offset(self) -> np.ndarray:
        """Total additive perturbation in canonical Celsius across all episodes."""
        out = np.zeros(self.total_steps, dtype=np.float64)
        for ep in self.episodes:
            out += ep.additive_offset
        return out

    @property
    def combined_noise_multiplier(self) -> np.ndarray:
        """Composite noise multiplier across all episodes."""
        out = np.ones(self.total_steps, dtype=np.float64)
        for ep in self.episodes:
            out *= ep.noise_multiplier
        return out

    @property
    def combined_missing_mask(self) -> np.ndarray:
        """Union of fault-induced missing readings."""
        out = np.zeros(self.total_steps, dtype=bool)
        for ep in self.episodes:
            out |= ep.missing_mask
        return out

    @property
    def active_frozen_episode(self) -> Optional[FaultEpisode]:
        """Return the active frozen episode if any."""
        for ep in self.episodes:
            if ep.is_frozen:
                return ep
        return None

    @property
    def active_gain_episode(self) -> Optional[FaultEpisode]:
        """Return the active gain error episode if any."""
        for ep in self.episodes:
            if ep.gain is not None:
                return ep
        return None

    @property
    def fault_severity(self) -> np.ndarray:
        """Composite fault severity at each step (maximum across active faults, clipped to [0, 1])."""
        if not self.episodes:
            return np.zeros(self.total_steps, dtype=np.float64)
        severities = np.array([ep.severity for ep in self.episodes])
        return np.clip(np.max(severities, axis=0), 0.0, 1.0)

    @property
    def true_health(self) -> np.ndarray:
        """True health at each step: true_health = 1 - max active severity (Appendix A.7)."""
        return np.clip(1.0 - self.fault_severity, 0.0, 1.0)

    @property
    def sensor_untrusted(self) -> np.ndarray:
        """Boolean ground truth indicating whether sensor is untrusted at each step."""
        if not self.episodes:
            return np.zeros(self.total_steps, dtype=bool)
        masks = np.array([ep.untrusted_mask for ep in self.episodes])
        return np.any(masks, axis=0)

    def get_fault_flag(self, fault_type: Union[FaultType, str]) -> np.ndarray:
        """Get binary indicator array f_* for a specific fault type."""
        ft_val = fault_type.value if isinstance(fault_type, FaultType) else str(fault_type)
        matching = [ep for ep in self.episodes if (ep.fault_type.value if isinstance(ep.fault_type, FaultType) else str(ep.fault_type)) == ft_val]
        if not matching:
            return np.zeros(self.total_steps, dtype=int)
        masks = np.array([ep.label_mask for ep in matching])
        return np.any(masks, axis=0).astype(int)

    def get_fault_ids_per_step(self) -> List[str]:
        """Get comma-separated fault IDs active at each step."""
        res = []
        for t in range(self.total_steps):
            active_ids = [ep.fault_id for ep in self.episodes if ep.start_step <= t < ep.end_step]
            res.append(",".join(active_ids))
        return res

    def to_ground_truth_dataframe(
        self,
        scenario_id: str,
        group_id: str,
        timestamps_minutes: np.ndarray,
        true_values: np.ndarray,
    ) -> pd.DataFrame:
        """Export per-sensor per-step ground truth table (Appendix A.7 Layer 3)."""
        return pd.DataFrame({
            "scenario_id": scenario_id,
            "group_id": group_id,
            "sensor_id": self.sensor_id,
            "timestamp": timestamps_minutes,
            "true_value": true_values,
            "sensor_untrusted": self.sensor_untrusted.astype(int),
            "f_spike": self.get_fault_flag(FaultType.SPIKE),
            "f_offset_drift": self.get_fault_flag(FaultType.OFFSET_DRIFT),
            "f_gain_error": self.get_fault_flag(FaultType.GAIN_ERROR),
            "f_frozen": self.get_fault_flag(FaultType.FROZEN),
            "f_high_noise": self.get_fault_flag(FaultType.HIGH_NOISE),
            "f_dropout": self.get_fault_flag(FaultType.DROPOUT),
            "f_slow_degradation": self.get_fault_flag(FaultType.SLOW_DEGRADATION),
            "fault_severity": self.fault_severity,
            "true_health": self.true_health,
            "fault_id": self.get_fault_ids_per_step(),
        })

    def episodes_to_dataframe(
        self,
        scenario_id: str = "scenario_0000",
        group_id: str = "group_00",
    ) -> pd.DataFrame:
        """Export episode ground truth table (Appendix A.7)."""
        rows = []
        for ep in self.episodes:
            ft_str = ep.fault_type.value if isinstance(ep.fault_type, FaultType) else str(ep.fault_type)
            rows.append({
                "scenario_id": scenario_id,
                "group_id": group_id,
                "sensor_id": self.sensor_id,
                "fault_id": ep.fault_id,
                "fault_type": ft_str,
                "start_step": ep.start_step,
                "end_step": ep.end_step,
                "duration": ep.duration,
                "max_severity": ep.max_severity,
                "shape_params": ep.shape_params,
            })
        return pd.DataFrame(rows)


class FaultSimulator:
    """Simulator engine for sampling and generating fault episodes on virtual sensors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            default_yaml = get_config_dir() / "generator_a.yaml"
            self.config = load_yaml(default_yaml)
        else:
            self.config = config

    def sample_fault(
        self,
        fault_type: FaultType,
        sensor_id: str,
        sigma_k: float,
        total_steps: int = 1920,
        warmup_steps: int = 480,
        physical_xref: float = 20.0,
        clean_base_value: float = 20.0,
        rng: Optional[np.random.Generator] = None,
        fault_id: str = "fault_0",
        start_step: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> FaultEpisode:
        """Sample and generate a single fault episode of the specified fault type."""
        if rng is None:
            rng = np.random.default_rng()

        p_start0 = float(self.config["faults"]["prob_present_from_start"])
        if start_step is None:
            start_step = sample_fault_start_step(
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                prob_present_from_start=p_start0,
                rng=rng,
            )

        fault_cfg = self.config["faults"]

        if fault_type == FaultType.SPIKE:
            if duration is None:
                duration = int(sample_log_uniform(
                    float(fault_cfg["spike"]["episode_min"]),
                    float(fault_cfg["spike"]["episode_max"]),
                    rng,
                ))
            return generate_spike_fault(
                sensor_id=sensor_id,
                sigma_k=sigma_k,
                total_steps=total_steps,
                start_step=start_step,
                duration=duration,
                config=self.config,
                rng=rng,
                fault_id=fault_id,
            )

        elif fault_type == FaultType.OFFSET_DRIFT:
            return generate_offset_drift_fault(
                sensor_id=sensor_id,
                sigma_k=sigma_k,
                total_steps=total_steps,
                start_step=start_step,
                config=self.config,
                rng=rng,
                fault_id=fault_id,
            )

        elif fault_type == FaultType.GAIN_ERROR:
            if duration is None:
                duration = int(rng.integers(
                    int(fault_cfg["gain_error"]["duration_min"]),
                    int(fault_cfg["gain_error"]["duration_max"]) + 1,
                ))
            return generate_gain_error_fault(
                sensor_id=sensor_id,
                total_steps=total_steps,
                start_step=start_step,
                duration=duration,
                physical_xref=physical_xref,
                config=self.config,
                rng=rng,
                fault_id=fault_id,
            )

        elif fault_type == FaultType.FROZEN:
            if duration is None:
                duration = int(sample_log_uniform(
                    float(fault_cfg["frozen"]["duration_min"]),
                    float(fault_cfg["frozen"]["duration_max"]),
                    rng,
                ))
            return generate_frozen_fault(
                sensor_id=sensor_id,
                sigma_k=sigma_k,
                total_steps=total_steps,
                start_step=start_step,
                duration=duration,
                clean_base_value=clean_base_value,
                config=self.config,
                rng=rng,
                fault_id=fault_id,
            )

        elif fault_type == FaultType.HIGH_NOISE:
            if duration is None:
                duration = int(rng.integers(
                    int(fault_cfg["high_noise"]["duration_min"]),
                    int(fault_cfg["high_noise"]["duration_max"]) + 1,
                ))
            return generate_high_noise_fault(
                sensor_id=sensor_id,
                total_steps=total_steps,
                start_step=start_step,
                duration=duration,
                config=self.config,
                rng=rng,
                fault_id=fault_id,
            )

        elif fault_type == FaultType.DROPOUT:
            if duration is None:
                duration = int(rng.integers(
                    int(fault_cfg["dropout"]["episode_min"]),
                    int(fault_cfg["dropout"]["episode_max"]) + 1,
                ))
            return generate_dropout_fault(
                sensor_id=sensor_id,
                total_steps=total_steps,
                start_step=start_step,
                duration=duration,
                config=self.config,
                rng=rng,
                fault_id=fault_id,
            )

        elif fault_type == FaultType.SLOW_DEGRADATION:
            if duration is None:
                duration = int(rng.integers(
                    int(fault_cfg["slow_degradation"]["duration_min"]),
                    int(fault_cfg["slow_degradation"]["duration_max"]) + 1,
                ))
            return generate_slow_degradation_fault(
                sensor_id=sensor_id,
                sigma_k=sigma_k,
                total_steps=total_steps,
                start_step=start_step,
                duration=duration,
                config=self.config,
                rng=rng,
                fault_id=fault_id,
            )

        else:
            raise ValueError(f"Unknown fault type: {fault_type}")
