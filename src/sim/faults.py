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

import itertools
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.sim.profile import TemperatureSensorParams
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


def compute_effective_dropout_gap_length(mean_gap_uncapped: float, cap: int = 100) -> float:
    """Calculate effective expected gap length of capped geometric distribution (Decision 33).
    
    X ~ Geometric(p = 1 / mean_gap_uncapped) on {1, 2, ...}, capped at C:
    E[min(X, C)] = mean_gap_uncapped * [1 - (1 - 1/mean_gap_uncapped)^C]
    """
    if mean_gap_uncapped <= 0.0:
        return 0.0
    p = 1.0 / mean_gap_uncapped
    return float(mean_gap_uncapped * (1.0 - (1.0 - p) ** cap))


def compute_dropout_theoretical_severity(p_enter: float, mean_gap_uncapped: float, cap: int = 100) -> float:
    """Calculate theoretical expected normalized episode severity for dropout (Decision 33).
    
    In alternating renewal steady state:
      f_bar = (p_enter * mu_eff) / (1 + p_enter * mu_eff)
      s_episode = 0.3 + 0.7 * f_bar
    """
    mu_eff = compute_effective_dropout_gap_length(mean_gap_uncapped, cap=cap)
    expected_missing_frac = (p_enter * mu_eff) / (1.0 + p_enter * mu_eff)
    return float(0.3 + 0.7 * expected_missing_frac)


def sample_relative_tolerance_value(
    base_val: float,
    val_min: float,
    val_max: float,
    tolerance: float,
    rng: np.random.Generator,
) -> float:
    """Sample value within relative tolerance |v - v_base| / max(v, v_base) <= tolerance."""
    low = max(val_min, base_val * (1.0 - tolerance))
    high = min(val_max, base_val / (1.0 - tolerance))
    if low > high:
        return base_val
    return float(rng.uniform(low, high))


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
    sp_size_sigma: Optional[float] = None,
    sign: Optional[float] = None,
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
        cur_sp_size = sample_log_uniform(size_min, size_max, rng) if sp_size_sigma is None else float(sp_size_sigma)
        cur_sign = (1.0 if rng.random() < 0.5 else -1.0) if sign is None else float(sign)
        amp = cur_sign * cur_sp_size * sigma_k

        sp_actual_end = min(t + sp_len, end_step)
        if sp_actual_end > t:
            additive_offset[t:sp_actual_end] += amp

        # Label window extends for spike length plus configured following steps
        lbl_end = min(t + sp_len + label_following, total_steps)
        if lbl_end > t:
            label_mask[t:lbl_end] = True
            untrusted_mask[t:lbl_end] = True
            sp_sev = min(cur_sp_size / divisor, 1.0)
            severity[t:lbl_end] = np.maximum(severity[t:lbl_end], sp_sev)

        spike_events.append({
            "step": int(t),
            "length": int(sp_len),
            "size_sigma": float(cur_sp_size),
            "sign": int(cur_sign),
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
    bias_mag_sigma: Optional[float] = None,
    sign: Optional[float] = None,
    shape: Optional[str] = None,
    stepped_steps_k: Optional[int] = None,
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

    if bias_mag_sigma is None:
        bias_mag_sigma = sample_log_uniform(
            float(drift_cfg["final_bias_sigma_min"]),
            float(drift_cfg["final_bias_sigma_max"]),
            rng,
        )
    else:
        bias_mag_sigma = float(bias_mag_sigma)
    if sign is None:
        sign = 1.0 if rng.random() < 0.5 else -1.0
    else:
        sign = float(sign)
    final_bias = sign * bias_mag_sigma * sigma_k

    # Sample trajectory shape
    shapes = list(drift_cfg["shape_probs"].keys())
    probs = [float(drift_cfg["shape_probs"][s]) for s in shapes]
    if shape is None:
        shape = str(rng.choice(shapes, p=probs))

    if shape == "stepped" and stepped_steps_k is None:
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
    gain: Optional[float] = None,
) -> FaultEpisode:
    """Generate gain error fault episode (Appendix A.4, Decision 5).
    
    Applies reading = x_ref + g * (x - x_ref) where x_ref is the true event-adjusted
    physical temperature in canonical Celsius at fault onset.
    """
    gain_cfg = config["faults"]["gain_error"]
    if gain is not None:
        g = float(gain)
        tier = "subtle" if abs(g - 1.0) <= 0.10 else "standard"
    else:
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
    is_wrong_value: Optional[bool] = None,
    has_jitter: Optional[bool] = None,
    offset_mag_sigma: Optional[float] = None,
    sign: Optional[float] = None,
    jitter_std_sigma: Optional[float] = None,
) -> FaultEpisode:
    """Generate frozen sensor fault episode (Appendix A.4, Decision 3, Decision 25).
    
    Freezes at last reading (80%) or wrong value 5 to 20 sigma_k away (20%).
    Constant (60%) or jitter with std 0.02 to 0.3 sigma_k (40%).
    For step 0 start, clean_base_value is initial healthy equilibrium reading at step 0.
    """
    frozen_cfg = config["faults"]["frozen"]
    if is_wrong_value is None:
        is_wrong_value = rng.random() < float(frozen_cfg["frozen_at_wrong_value_prob"])
    else:
        is_wrong_value = bool(is_wrong_value)

    if has_jitter is None:
        has_jitter = rng.random() < float(frozen_cfg["jitter_prob"])
    else:
        has_jitter = bool(has_jitter)

    if is_wrong_value:
        if offset_mag_sigma is None:
            offset_mag_sigma = float(rng.uniform(
                float(frozen_cfg["wrong_value_offset_sigma_min"]),
                float(frozen_cfg["wrong_value_offset_sigma_max"]),
            ))
        else:
            offset_mag_sigma = float(offset_mag_sigma)
        if sign is None:
            sign = 1.0 if rng.random() < 0.5 else -1.0
        else:
            sign = float(sign)
        offset = sign * offset_mag_sigma * sigma_k
        frozen_val = clean_base_value + offset
        sev_val = float(config["severity_formulas"]["frozen_severity_wrong_value"])
    else:
        offset_mag_sigma = 0.0
        sign = 1.0
        frozen_val = clean_base_value
        sev_val = float(config["severity_formulas"]["frozen_severity_normal"])

    if has_jitter:
        if jitter_std_sigma is None:
            jitter_std_sigma = float(rng.uniform(
                float(frozen_cfg["jitter_std_sigma_min"]),
                float(frozen_cfg["jitter_std_sigma_max"]),
            ))
        else:
            jitter_std_sigma = float(jitter_std_sigma)
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
    multiplier: Optional[float] = None,
    is_bursty: Optional[bool] = None,
) -> FaultEpisode:
    """Generate high noise fault episode (Appendix A.4).
    
    Noise multiplier 2 to 10 x sigma_k (log-uniform).
    70% continuous, 30% bursty (on/off blocks 5 to 30 steps).
    """
    noise_cfg = config["faults"]["high_noise"]
    if multiplier is None:
        mult = sample_log_uniform(
            float(noise_cfg["multiplier_min"]),
            float(noise_cfg["multiplier_max"]),
            rng,
        )
    else:
        mult = float(multiplier)
    if is_bursty is None:
        is_bursty = rng.random() < float(noise_cfg["bursty_prob"])
    else:
        is_bursty = bool(is_bursty)

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
    p_enter: Optional[float] = None,
    mean_gap: Optional[float] = None,
) -> FaultEpisode:
    """Generate dropout fault episode (Appendix A.4).
    
    Enter-gap prob 0.002 to 0.03 per step.
    Gap length geometric with mean 2 to 40, capped at 100.
    Severity: 0.3 + 0.7 * (missing fraction over last 30 steps).
    """
    drop_cfg = config["faults"]["dropout"]
    if p_enter is None:
        p_enter = sample_log_uniform(
            float(drop_cfg["enter_gap_prob_min"]),
            float(drop_cfg["enter_gap_prob_max"]),
            rng,
        )
    else:
        p_enter = float(p_enter)
    if mean_gap is None:
        mean_gap = float(rng.uniform(
            float(drop_cfg["gap_length_mean_min"]),
            float(drop_cfg["gap_length_mean_max"]),
        ))
    else:
        mean_gap = float(mean_gap)
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
    h_end: Optional[float] = None,
    sign: Optional[float] = None,
) -> FaultEpisode:
    """Generate slow degradation fault episode (Appendix A.4, Decision 4, Decision 26).
    
    Health h falls linearly from 1 to h_end (0.1 to 0.7) over duration.
    Damage d = 1 - h.
    Noise x(1 + 5d), bias +/- 8 sigma_k * d (sign sampled once), extra missing prob 0.15 d.
    Untrusted once d >= 0.15.
    Labeled ONLY as slow_degradation.
    """
    deg_cfg = config["faults"]["slow_degradation"]
    if h_end is None:
        h_end = float(rng.uniform(float(deg_cfg["h_end_min"]), float(deg_cfg["h_end_max"])))
    else:
        h_end = float(h_end)
    if sign is None:
        sign = 1.0 if rng.random() < 0.5 else -1.0
    else:
        sign = float(sign)

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
        difficulty_tier: str = "medium",
        **kwargs: Any,
    ) -> FaultEpisode:
        """Sample and generate a single fault episode of the specified fault type (Decision 39)."""
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

        # Decision 39: Restricted visibility parameter sampling for EASY tier
        if difficulty_tier == "easy":
            tier_cfg = self.config.get("difficulty_tiers", {}).get("easy", {})
            if fault_type == FaultType.SPIKE and "sp_size_sigma" not in kwargs:
                min_sp = float(tier_cfg.get("spike_min_sigma", 8.0))
                max_sp = float(fault_cfg["spike"]["size_sigma_max"])
                kwargs["sp_size_sigma"] = sample_log_uniform(min_sp, max_sp, rng)
            elif fault_type == FaultType.OFFSET_DRIFT and "bias_mag_sigma" not in kwargs:
                min_drift = float(tier_cfg.get("drift_min_sigma", 6.0))
                max_drift = float(fault_cfg["offset_drift"]["final_bias_sigma_max"])
                kwargs["bias_mag_sigma"] = sample_log_uniform(min_drift, max_drift, rng)
            elif fault_type == FaultType.GAIN_ERROR and "gain" not in kwargs:
                # 50% low [0.55, 0.75] (|g-1| in [0.25, 0.45]), 50% high [1.25, 1.60] (|g-1| in [0.25, 0.60])
                if rng.random() < 0.5:
                    kwargs["gain"] = float(rng.uniform(0.55, 0.75))
                else:
                    kwargs["gain"] = float(rng.uniform(1.25, 1.60))
            elif fault_type == FaultType.HIGH_NOISE and "multiplier" not in kwargs:
                min_noise = float(tier_cfg.get("noise_mult_min", 4.0))
                max_noise = float(fault_cfg["high_noise"]["multiplier_max"])
                kwargs["multiplier"] = sample_log_uniform(min_noise, max_noise, rng)

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
                **kwargs,
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
                **kwargs,
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
                **kwargs,
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
                **kwargs,
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
                **kwargs,
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
                **kwargs,
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
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

    def get_valid_pairs(self) -> Set[FrozenSet[str]]:
        """Return valid 2-fault pairs mechanically derived from configuration."""
        return get_valid_pairs(self.config)

    def derive_valid_triplets(self) -> List[Tuple[str, str, str]]:
        """Return valid 3-fault triplets mechanically derived from configuration."""
        return derive_valid_triplets(self.config)

    def sample_candidate_count(self, rng: Optional[np.random.Generator] = None) -> int:
        """Sample candidate faulty sensor count k_candidate."""
        if rng is None:
            rng = np.random.default_rng()
        return sample_faulty_sensor_candidate_count(self.config, rng)

    def compute_actual_count(self, k_candidate: int, group_sensor_counts: Sequence[int]) -> int:
        """Calculate actual faulty sensor count k_actual enforcing both cap levels."""
        return compute_actual_faulty_sensor_count(k_candidate, group_sensor_counts, self.config)

    def sample_fault_type_count(self, rng: Optional[np.random.Generator] = None) -> int:
        """Sample number of fault types for a sensor (1, 2, or 3)."""
        if rng is None:
            rng = np.random.default_rng()
        return sample_fault_type_count(self.config, rng)

    def sample_fault_types_for_sensor(
        self,
        type_count: int,
        rng: Optional[np.random.Generator] = None,
        shared_fault_type: Optional[str] = None,
    ) -> List[str]:
        """Sample valid fault types for a sensor adhering to allowed/forbidden constraints."""
        if rng is None:
            rng = np.random.default_rng()
        return sample_fault_types_for_sensor(
            type_count=type_count,
            config=self.config,
            rng=rng,
            shared_fault_type=shared_fault_type,
        )

    def generate_scenario_faults(
        self,
        groups: Dict[str, Sequence[Union[TemperatureSensorParams, str]]],
        sensor_sigmas: Optional[Dict[str, float]] = None,
        clean_physical_trajectories: Optional[Dict[str, np.ndarray]] = None,
        events: Optional[Any] = None,
        active_event_steps: Optional[Sequence[int]] = None,
        total_steps: int = 1920,
        warmup_steps: int = 480,
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
        k_candidate: Optional[int] = None,
        force_shared_cause: Optional[bool] = None,
    ) -> Dict[str, SensorFaultSeries]:
        """Generate scenario-wide fault series adhering to all P1.6 constraints."""
        return generate_scenario_faults(
            groups=groups,
            sensor_sigmas=sensor_sigmas,
            clean_physical_trajectories=clean_physical_trajectories,
            events=events,
            active_event_steps=active_event_steps,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            config=self.config,
            rng=rng,
            seed=seed,
            k_candidate=k_candidate,
            force_shared_cause=force_shared_cause,
        )

    def generate_group_faults(
        self,
        sensor_params: Sequence[Union[TemperatureSensorParams, str]],
        group_id: str = "group_00",
        sensor_sigmas: Optional[Dict[str, float]] = None,
        clean_physical_trajectories: Optional[Dict[str, np.ndarray]] = None,
        events: Optional[Any] = None,
        active_event_steps: Optional[Sequence[int]] = None,
        total_steps: int = 1920,
        warmup_steps: int = 480,
        rng: Optional[Union[int, np.random.Generator]] = None,
        seed: Optional[int] = None,
        k_candidate: Optional[int] = None,
        force_shared_cause: Optional[bool] = None,
    ) -> Dict[str, SensorFaultSeries]:
        """Convenience wrapper for generating faults in a single-group scenario."""
        return generate_group_faults(
            sensor_params=sensor_params,
            group_id=group_id,
            sensor_sigmas=sensor_sigmas,
            clean_physical_trajectories=clean_physical_trajectories,
            events=events,
            active_event_steps=active_event_steps,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            config=self.config,
            rng=rng,
            seed=seed,
            k_candidate=k_candidate,
            force_shared_cause=force_shared_cause,
        )


def get_valid_pairs(config: Dict[str, Any]) -> Set[FrozenSet[str]]:
    """Derive valid 2-fault pairs mechanically from configuration (Decision 30).
    
    A pair is valid if and only if it is in allowed_pairs and NOT in forbidden_pairs.
    Returns a set of frozensets of size 2.
    """
    comb_cfg = config["combinations"]
    allowed = {frozenset(pair) for pair in comb_cfg["allowed_pairs"]}
    forbidden = {frozenset(pair) for pair in comb_cfg["forbidden_pairs"]}
    return allowed - forbidden


def derive_valid_triplets(config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Mechanically derive valid 3-fault triplets from valid pairs (Decision 30).
    
    A triplet {a, b, c} is valid iff all three pairwise sub-combinations
    {a, b}, {b, c}, and {a, c} are valid pairs.
    Returns sorted list of canonical tuples of length 3.
    """
    valid_pairs = get_valid_pairs(config)
    all_types = sorted([ft.value for ft in FaultType])
    triplets = []
    for a, b, c in itertools.combinations(all_types, 3):
        p1 = frozenset([a, b])
        p2 = frozenset([b, c])
        p3 = frozenset([a, c])
        if p1 in valid_pairs and p2 in valid_pairs and p3 in valid_pairs:
            triplets.append((a, b, c))
    return sorted(triplets)


def sample_faulty_sensor_candidate_count(
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> int:
    """Sample candidate count k_candidate of faulty sensors for a scenario (Decision 29).
    
    Probabilities from config['combinations']['faulty_sensors_probs']:
      0: 0.15, 1: 0.40, 2: 0.25, 3_or_more: 0.20
    Representative count 3 is used for '3_or_more'.
    """
    probs_dict = config["combinations"]["faulty_sensors_probs"]
    keys = [0, 1, 2, 3]
    probs = [
        float(probs_dict.get(0, probs_dict.get("0", 0.15))),
        float(probs_dict.get(1, probs_dict.get("1", 0.40))),
        float(probs_dict.get(2, probs_dict.get("2", 0.25))),
        float(probs_dict.get("3_or_more", probs_dict.get(3, probs_dict.get("3", 0.20)))),
    ]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    return int(rng.choice(keys, p=probs))


def compute_group_cap(group_sensor_count: int) -> int:
    """Calculate maximum faulty sensors allowed in a group: max(1, floor(0.4 * n_g))."""
    return max(1, int(np.floor(0.4 * group_sensor_count)))


def compute_scenario_cap(total_sensor_count: int) -> int:
    """Calculate maximum faulty sensors allowed in a scenario: max(1, floor(0.4 * N))."""
    return max(1, int(np.floor(0.4 * total_sensor_count)))


def compute_actual_faulty_sensor_count(
    k_candidate: int,
    group_sensor_counts: Sequence[int],
    config: Optional[Dict[str, Any]] = None,
    tier_cap: Optional[int] = None,
) -> int:
    """Calculate actual faulty sensor count k_actual enforcing scenario, group, and tier caps (Decisions 29 & 39).
    
    k_actual = min(k_candidate, cap_scenario, sum(cap_g), cap_tier)
    Note: When k_candidate = 0, k_actual = 0 (15% zero-fault preserved across all tiers).
    """
    if k_candidate <= 0:
        return 0
    total_sensors = sum(group_sensor_counts)
    cap_scenario = compute_scenario_cap(total_sensors)
    sum_group_caps = sum(compute_group_cap(ng) for ng in group_sensor_counts)
    effective_cap = min(cap_scenario, sum_group_caps)
    if tier_cap is not None:
        effective_cap = min(effective_cap, int(tier_cap))
    return min(int(k_candidate), effective_cap)


def allocate_faulty_sensors(
    k_actual: int,
    groups: Dict[str, Sequence[str]],
    rng: np.random.Generator,
    pre_allocated_sensors: Optional[Sequence[str]] = None,
    pre_allocated_group: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Allocate k_actual faulty sensors across groups with capacity constraints (Decision 29).
    
    Args:
        k_actual: Target number of faulty sensors.
        groups: Mapping from group_id to list of sensor_ids in that group.
        rng: Numpy Generator.
        pre_allocated_sensors: Optional pre-allocated sensor IDs (e.g., shared-cause pair).
        pre_allocated_group: Group ID of the pre-allocated sensors.
        
    Returns:
        List of (sensor_id, group_id) tuples for all allocated faulty sensors.
    """
    if k_actual <= 0:
        return []

    group_caps = {gid: compute_group_cap(len(s_ids)) for gid, s_ids in groups.items()}
    group_counts = {gid: 0 for gid in groups}

    allocated: List[Tuple[str, str]] = []
    assigned_sensor_ids = set()

    if pre_allocated_sensors and pre_allocated_group:
        for sid in pre_allocated_sensors:
            allocated.append((sid, pre_allocated_group))
            assigned_sensor_ids.add(sid)
            group_counts[pre_allocated_group] += 1

    while len(allocated) < k_actual:
        # Build pool of currently eligible sensors
        eligible = []
        for gid, s_ids in groups.items():
            if group_counts[gid] < group_caps[gid]:
                for sid in s_ids:
                    if sid not in assigned_sensor_ids:
                        eligible.append((sid, gid))
        if not eligible:
            break
        # Uniform draw from currently eligible pool
        idx = int(rng.integers(0, len(eligible)))
        chosen_sid, chosen_gid = eligible[idx]
        allocated.append((chosen_sid, chosen_gid))
        assigned_sensor_ids.add(chosen_sid)
        group_counts[chosen_gid] += 1

    return allocated


def sample_fault_type_count(
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> int:
    """Sample number of fault types for a faulty sensor: 1 (65%), 2 (27%), 3 (8%)."""
    probs_dict = config["combinations"]["types_per_faulty_sensor_probs"]
    keys = [1, 2, 3]
    probs = [
        float(probs_dict.get(1, probs_dict.get("1", 0.65))),
        float(probs_dict.get(2, probs_dict.get("2", 0.27))),
        float(probs_dict.get(3, probs_dict.get("3", 0.08))),
    ]
    total_p = sum(probs)
    probs = [p / total_p for p in probs]
    return int(rng.choice(keys, p=probs))


def sample_fault_types_for_sensor(
    type_count: int,
    config: Dict[str, Any],
    rng: np.random.Generator,
    shared_fault_type: Optional[str] = None,
    valid_triplets: Optional[List[Tuple[str, str, str]]] = None,
    valid_pairs: Optional[Set[FrozenSet[str]]] = None,
) -> List[str]:
    """Sample valid fault types for a sensor obeying pair/triplet constraints (Decisions 30 & 33).
    
    If shared_fault_type is provided, it is placed first in the returned list.
    """
    if valid_pairs is None:
        valid_pairs = get_valid_pairs(config)
    if valid_triplets is None:
        valid_triplets = derive_valid_triplets(config)

    all_types = sorted([ft.value for ft in FaultType])

    if type_count == 1:
        if shared_fault_type is not None:
            return [shared_fault_type]
        chosen = str(rng.choice(all_types))
        return [chosen]

    elif type_count == 2:
        if shared_fault_type is not None:
            candidate_pairs = [
                p for p in valid_pairs if shared_fault_type in p
            ]
            if not candidate_pairs:
                raise ValueError(f"No valid pairs contain shared fault type {shared_fault_type}")
            chosen_pair = candidate_pairs[int(rng.integers(0, len(candidate_pairs)))]
            other = next(iter(chosen_pair - {shared_fault_type}))
            return [shared_fault_type, other]
        else:
            pairs_list = sorted([tuple(sorted(list(p))) for p in valid_pairs])
            chosen = pairs_list[int(rng.integers(0, len(pairs_list)))]
            chosen_list = list(chosen)
            rng.shuffle(chosen_list)
            return chosen_list

    elif type_count == 3:
        if shared_fault_type is not None:
            candidate_triplets = [
                t for t in valid_triplets if shared_fault_type in t
            ]
            if not candidate_triplets:
                raise ValueError(f"No valid triplets contain shared fault type {shared_fault_type}")
            chosen_trip = candidate_triplets[int(rng.integers(0, len(candidate_triplets)))]
            others = [x for x in chosen_trip if x != shared_fault_type]
            rng.shuffle(others)
            return [shared_fault_type] + others
        else:
            chosen = valid_triplets[int(rng.integers(0, len(valid_triplets)))]
            chosen_list = list(chosen)
            rng.shuffle(chosen_list)
            return chosen_list

    else:
        raise ValueError(f"Unsupported fault type count: {type_count}")


def extract_active_event_steps(
    events: Any,
    warmup_steps: int = 480,
    total_steps: int = 1920,
) -> List[int]:
    """Extract time steps where a genuine environmental event is physically active (Decision 31)."""
    if events is None:
        return []

    # If already a list of ints or steps
    if isinstance(events, (list, tuple, np.ndarray)) and (len(events) == 0 or isinstance(events[0], (int, np.integer))):
        if isinstance(events, np.ndarray) and events.dtype == bool:
            return sorted([int(t) for t in np.where(events)[0] if warmup_steps <= t < total_steps])
        return sorted([int(s) for s in events if warmup_steps <= s < total_steps])

    # If EventSeries instance
    steps_set = set()
    if hasattr(events, "episodes"):
        for ep in events.episodes:
            start = getattr(ep, "start_step", 0)
            end = getattr(ep, "end_step", 0)
            for t in range(start, end):
                if warmup_steps <= t < total_steps:
                    steps_set.add(t)

    if hasattr(events, "genuine_event"):
        for t in np.where(events.genuine_event)[0]:
            if warmup_steps <= t < total_steps:
                steps_set.add(int(t))

    if hasattr(events, "event_below_detectability"):
        for t in np.where(events.event_below_detectability)[0]:
            if warmup_steps <= t < total_steps:
                steps_set.add(int(t))

    return sorted(list(steps_set))


def sample_episode_start_step(
    preceding_episode: Optional[FaultEpisode],
    total_steps: int,
    warmup_steps: int,
    active_event_steps: Sequence[int],
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> int:
    """Sample start step adhering to staggered windows and legal start regimes (Decisions 31 & 32)."""
    comb_cfg = config["combinations"]
    p_event = float(comb_cfg.get("start_inside_active_event_prob", 0.35))
    p_start0 = float(config["faults"].get("prob_present_from_start", 0.10))
    p_standard = max(0.0, 1.0 - p_start0 - p_event)

    active_event_post = sorted([s for s in active_event_steps if warmup_steps <= s < total_steps])

    if preceding_episode is None:
        # First episode on sensor
        branch = rng.random()
        if branch < p_start0:
            return 0
        elif branch < p_start0 + p_event:
            if active_event_post:
                return int(rng.choice(active_event_post))
            else:
                return int(rng.integers(warmup_steps, total_steps))
        else:
            return int(rng.integers(warmup_steps, total_steps))

    else:
        # Subsequent episode on same sensor
        offset_min = int(comb_cfg.get("second_fault_start_offset_min", -100))
        w_stagger_start = max(0, preceding_episode.start_step + offset_min)
        w_stagger_end = min(total_steps - 1, preceding_episode.end_step)
        if w_stagger_start > w_stagger_end:
            w_stagger_start = min(w_stagger_start, w_stagger_end)

        w_stagger_steps = set(range(w_stagger_start, w_stagger_end + 1))
        w_post_steps = sorted([s for s in w_stagger_steps if warmup_steps <= s < total_steps])
        has_zero = (0 in w_stagger_steps)

        # Case A: W_post is empty -> deterministically 0
        if not w_post_steps:
            return 0

        c_event = sorted([s for s in w_post_steps if s in active_event_post])

        # Case B: 0 not in W_stagger -> only post-warmup is feasible
        if not has_zero:
            if c_event:
                prob_event_cond = p_event / (p_event + p_standard) if (p_event + p_standard) > 0 else 0.5
                if rng.random() < prob_event_cond:
                    return int(rng.choice(c_event))
                else:
                    return int(rng.choice(w_post_steps))
            else:
                return int(rng.choice(w_post_steps))

        # Case C: Both 0 and W_post are feasible
        branch = rng.random()
        if branch < p_start0:
            return 0
        elif branch < p_start0 + p_event:
            if c_event:
                return int(rng.choice(c_event))
            else:
                return int(rng.choice(w_post_steps))
        else:
            return int(rng.choice(w_post_steps))


def sample_shared_cause_start_step(
    t_start_A: int,
    total_steps: int,
    warmup_steps: int,
    active_event_steps: Sequence[int],
    rng: np.random.Generator,
    max_offset: int = 10,
) -> int:
    """Sample start step for the second sensor in a shared-cause pair (Decision 33)."""
    if t_start_A == 0:
        return 0

    delta_min = max(-max_offset, warmup_steps - t_start_A)
    delta_max = min(max_offset, total_steps - 1 - t_start_A)
    if delta_min > delta_max:
        return t_start_A

    feasible_deltas = list(range(delta_min, delta_max + 1))
    event_set = set(active_event_steps)

    if t_start_A in event_set:
        event_deltas = [d for d in feasible_deltas if (t_start_A + d) in event_set]
        if event_deltas:
            chosen_delta = int(rng.choice(event_deltas))
            return t_start_A + chosen_delta

    chosen_delta = int(rng.choice(feasible_deltas))
    return t_start_A + chosen_delta


def couple_shared_cause_parameters(
    fault_type: Union[FaultType, str],
    ep_A: FaultEpisode,
    sigma_k_A: float,
    sigma_k_B: float,
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sample coupled parameters for sensor B within 20% relative tolerance of sensor A (Decision 33)."""
    comb_cfg = config["combinations"]
    tolerance = float(comb_cfg.get("shared_cause_severity_relative_tolerance", 0.20))
    ft_val = fault_type.value if isinstance(fault_type, FaultType) else str(fault_type)

    coupled: Dict[str, Any] = {}

    if ft_val == "spike":
        sp_cfg = config["faults"]["spike"]
        s_min = float(sp_cfg["size_sigma_min"])
        s_max = float(sp_cfg["size_sigma_max"])
        spikes = ep_A.shape_params.get("spikes", [])
        base_size = float(spikes[0]["size_sigma"]) if spikes else 10.0
        base_sign = float(spikes[0]["sign"]) if spikes else 1.0

        coupled["sp_size_sigma"] = sample_relative_tolerance_value(base_size, s_min, s_max, tolerance, rng)
        coupled["sign"] = base_sign

    elif ft_val == "offset_drift":
        drift_cfg = config["faults"]["offset_drift"]
        b_min = float(drift_cfg["final_bias_sigma_min"])
        b_max = float(drift_cfg["final_bias_sigma_max"])
        base_bias_sigma = float(ep_A.shape_params["final_bias_sigma"])
        base_sign = float(ep_A.shape_params["sign"])
        base_shape = str(ep_A.shape_params["shape"])
        base_stepped_k = ep_A.shape_params.get("stepped_k")

        coupled["bias_mag_sigma"] = sample_relative_tolerance_value(base_bias_sigma, b_min, b_max, tolerance, rng)
        coupled["sign"] = base_sign
        coupled["shape"] = base_shape
        coupled["stepped_steps_k"] = base_stepped_k

    elif ft_val == "gain_error":
        g_A = float(ep_A.shape_params["gain"])
        d_A = abs(g_A - 1.0)
        d_B = sample_relative_tolerance_value(d_A, 0.05, 0.60, tolerance, rng)
        if g_A >= 1.0:
            coupled["gain"] = 1.0 + d_B
        else:
            coupled["gain"] = 1.0 - d_B

    elif ft_val == "frozen":
        frz_cfg = config["faults"]["frozen"]
        is_wrong = bool(ep_A.shape_params["is_wrong_value"])
        has_jit = bool(ep_A.shape_params["has_jitter"])
        coupled["is_wrong_value"] = is_wrong
        coupled["has_jitter"] = has_jit

        if is_wrong:
            m_min = float(frz_cfg["wrong_value_offset_sigma_min"])
            m_max = float(frz_cfg["wrong_value_offset_sigma_max"])
            base_m = float(ep_A.shape_params["wrong_value_offset_sigma"])
            base_sign = float(ep_A.shape_params["sign"])
            coupled["offset_mag_sigma"] = sample_relative_tolerance_value(base_m, m_min, m_max, tolerance, rng)
            coupled["sign"] = base_sign

        if has_jit:
            j_min = float(frz_cfg["jitter_std_sigma_min"])
            j_max = float(frz_cfg["jitter_std_sigma_max"])
            base_j = float(ep_A.shape_params["jitter_std_sigma"])
            coupled["jitter_std_sigma"] = sample_relative_tolerance_value(base_j, j_min, j_max, tolerance, rng)

    elif ft_val == "high_noise":
        noise_cfg = config["faults"]["high_noise"]
        m_min = float(noise_cfg["multiplier_min"])
        m_max = float(noise_cfg["multiplier_max"])
        base_mult = float(ep_A.shape_params["multiplier"])
        is_bursty = bool(ep_A.shape_params["is_bursty"])

        coupled["multiplier"] = sample_relative_tolerance_value(base_mult, m_min, m_max, tolerance, rng)
        coupled["is_bursty"] = is_bursty

    elif ft_val == "dropout":
        drop_cfg = config["faults"]["dropout"]
        p_min = float(drop_cfg["enter_gap_prob_min"])
        p_max = float(drop_cfg["enter_gap_prob_max"])
        g_min = float(drop_cfg["gap_length_mean_min"])
        g_max = float(drop_cfg["gap_length_mean_max"])

        base_p = float(ep_A.shape_params["enter_gap_prob"])
        base_gap = float(ep_A.shape_params["mean_gap_length"])

        coupled["p_enter"] = sample_relative_tolerance_value(base_p, p_min, p_max, tolerance, rng)
        coupled["mean_gap"] = sample_relative_tolerance_value(base_gap, g_min, g_max, tolerance, rng)

    elif ft_val == "slow_degradation":
        base_final_d = float(ep_A.shape_params["final_damage"])
        base_sign = float(ep_A.shape_params["sign"])

        coupled_d = sample_relative_tolerance_value(base_final_d, 0.30, 0.90, tolerance, rng)
        coupled["h_end"] = 1.0 - coupled_d
        coupled["sign"] = base_sign

    return coupled


def generate_scenario_faults(
    groups: Dict[str, Sequence[Union[TemperatureSensorParams, str]]],
    sensor_sigmas: Optional[Dict[str, float]] = None,
    clean_physical_trajectories: Optional[Dict[str, np.ndarray]] = None,
    events: Optional[Any] = None,
    active_event_steps: Optional[Sequence[int]] = None,
    total_steps: int = 1920,
    warmup_steps: int = 480,
    config: Optional[Dict[str, Any]] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
    seed: Optional[int] = None,
    k_candidate: Optional[int] = None,
    force_shared_cause: Optional[bool] = None,
    difficulty_tier: str = "medium",
    tier_cap: Optional[int] = None,
    max_types_per_sensor: Optional[int] = None,
) -> Dict[str, SensorFaultSeries]:
    """Generate scenario-wide fault series adhering to all P1.6 and P1.7 tier constraints."""
    if config is None:
        default_yaml = get_config_dir() / "generator_a.yaml"
        config = load_yaml(default_yaml)

    if rng is None:
        rng = np.random.default_rng(seed)
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    sigmas = dict(sensor_sigmas) if sensor_sigmas is not None else {}
    groups_ids: Dict[str, List[str]] = {}

    for gid, items in groups.items():
        sids = []
        for item in items:
            if hasattr(item, "sensor_id"):
                sid = str(item.sensor_id)
                if sid not in sigmas:
                    sigmas[sid] = float(getattr(item, "sigma_k", 0.15))
            else:
                sid = str(item)
                if sid not in sigmas:
                    sigmas[sid] = 0.15
            sids.append(sid)
        groups_ids[gid] = sids

    all_sensor_ids = [sid for sids in groups_ids.values() for sid in sids]

    if active_event_steps is not None:
        ev_steps = extract_active_event_steps(active_event_steps, warmup_steps, total_steps)
    elif events is not None:
        ev_steps = extract_active_event_steps(events, warmup_steps, total_steps)
    else:
        ev_steps = []

    valid_triplets = derive_valid_triplets(config)
    valid_pairs = get_valid_pairs(config)

    if k_candidate is None:
        k_candidate = sample_faulty_sensor_candidate_count(config, rng)

    # Determine effective tier cap (Decision 39)
    if tier_cap is not None:
        eff_tier_cap = int(tier_cap)
    elif difficulty_tier == "easy":
        eff_tier_cap = 1
    elif difficulty_tier == "medium":
        eff_tier_cap = 2
    else:  # "hard"
        eff_tier_cap = 3

    group_counts = [len(groups_ids[gid]) for gid in groups_ids]
    k_actual = compute_actual_faulty_sensor_count(k_candidate, group_counts, config, tier_cap=eff_tier_cap)

    # If actual count is 0, return empty fault series for all sensors
    if k_actual == 0:
        return {
            sid: SensorFaultSeries(sensor_id=sid, total_steps=total_steps, episodes=[])
            for sid in all_sensor_ids
        }

    # Determine max types allowed per sensor in this tier
    if max_types_per_sensor is not None:
        eff_max_types = int(max_types_per_sensor)
    elif difficulty_tier == "easy":
        eff_max_types = 1
    elif difficulty_tier == "medium":
        eff_max_types = 2
    else:
        eff_max_types = 3

    def sample_types_count() -> int:
        raw_m = sample_fault_type_count(config, rng)
        return min(raw_m, eff_max_types)

    # Evaluate shared-cause eligibility (Decision 33)
    eligible_host_groups = [
        gid for gid, sids in groups_ids.items()
        if compute_group_cap(len(sids)) >= 2
    ]
    is_eligible_shared = (k_actual >= 2 and len(eligible_host_groups) > 0)

    if is_eligible_shared:
        if force_shared_cause is not None:
            is_shared = bool(force_shared_cause)
        else:
            p_shared = float(config["combinations"]["shared_cause_pair_prob"])
            is_shared = (rng.random() < p_shared)
    else:
        is_shared = False

    fault_sim = FaultSimulator(config=config)
    episodes_by_sensor: Dict[str, List[FaultEpisode]] = {sid: [] for sid in all_sensor_ids}

    if is_shared:
        host_group = str(rng.choice(eligible_host_groups))
        host_sensors = sorted(list(groups_ids[host_group]))
        chosen_pair = rng.choice(host_sensors, size=2, replace=False)
        s_A, s_B = str(chosen_pair[0]), str(chosen_pair[1])

        m_A = sample_types_count()
        m_B = sample_types_count()

        if max(m_A, m_B) == 3:
            triplet_types = sorted(list({t for trip in valid_triplets for t in trip}))
            shared_type = str(rng.choice(triplet_types))
        else:
            shared_type = str(rng.choice([ft.value for ft in FaultType]))

        types_A = sample_fault_types_for_sensor(
            m_A, config, rng, shared_fault_type=shared_type,
            valid_triplets=valid_triplets, valid_pairs=valid_pairs,
        )
        types_B = sample_fault_types_for_sensor(
            m_B, config, rng, shared_fault_type=shared_type,
            valid_triplets=valid_triplets, valid_pairs=valid_pairs,
        )

        # Allocate remaining k_actual - 2 sensors
        allocated = allocate_faulty_sensors(
            k_actual, groups_ids, rng,
            pre_allocated_sensors=[s_A, s_B], pre_allocated_group=host_group,
        )

        # Generate sensor A episodes
        t_start_A = sample_episode_start_step(None, total_steps, warmup_steps, ev_steps, config, rng)
        xref_A = clean_physical_trajectories[s_A][t_start_A] if clean_physical_trajectories and s_A in clean_physical_trajectories else 20.0
        ep_A1 = fault_sim.sample_fault(
            FaultType(shared_type), s_A, sigma_k=sigmas[s_A],
            total_steps=total_steps, warmup_steps=warmup_steps,
            physical_xref=xref_A, clean_base_value=xref_A,
            rng=rng, fault_id=f"fault_{s_A}_0", start_step=t_start_A,
            difficulty_tier=difficulty_tier,
        )
        episodes_by_sensor[s_A].append(ep_A1)
        for idx in range(1, len(types_A)):
            t_start = sample_episode_start_step(episodes_by_sensor[s_A][-1], total_steps, warmup_steps, ev_steps, config, rng)
            xref = clean_physical_trajectories[s_A][t_start] if clean_physical_trajectories and s_A in clean_physical_trajectories else 20.0
            ep = fault_sim.sample_fault(
                FaultType(types_A[idx]), s_A, sigma_k=sigmas[s_A],
                total_steps=total_steps, warmup_steps=warmup_steps,
                physical_xref=xref, clean_base_value=xref,
                rng=rng, fault_id=f"fault_{s_A}_{idx}", start_step=t_start,
                difficulty_tier=difficulty_tier,
            )
            episodes_by_sensor[s_A].append(ep)

        # Generate sensor B episodes (Episode 1 coupled to ep_A1)
        t_start_B = sample_shared_cause_start_step(t_start_A, total_steps, warmup_steps, ev_steps, rng)
        coupled_params = couple_shared_cause_parameters(shared_type, ep_A1, sigmas[s_A], sigmas[s_B], config, rng)
        xref_B = clean_physical_trajectories[s_B][t_start_B] if clean_physical_trajectories and s_B in clean_physical_trajectories else 20.0
        ep_B1 = fault_sim.sample_fault(
            FaultType(shared_type), s_B, sigma_k=sigmas[s_B],
            total_steps=total_steps, warmup_steps=warmup_steps,
            physical_xref=xref_B, clean_base_value=xref_B,
            rng=rng, fault_id=f"fault_{s_B}_0", start_step=t_start_B,
            difficulty_tier=difficulty_tier,
            **coupled_params,
        )
        episodes_by_sensor[s_B].append(ep_B1)
        for idx in range(1, len(types_B)):
            t_start = sample_episode_start_step(episodes_by_sensor[s_B][-1], total_steps, warmup_steps, ev_steps, config, rng)
            xref = clean_physical_trajectories[s_B][t_start] if clean_physical_trajectories and s_B in clean_physical_trajectories else 20.0
            ep = fault_sim.sample_fault(
                FaultType(types_B[idx]), s_B, sigma_k=sigmas[s_B],
                total_steps=total_steps, warmup_steps=warmup_steps,
                physical_xref=xref, clean_base_value=xref,
                rng=rng, fault_id=f"fault_{s_B}_{idx}", start_step=t_start,
                difficulty_tier=difficulty_tier,
            )
            episodes_by_sensor[s_B].append(ep)

        # Generate episodes for other allocated sensors
        for sid, _ in allocated:
            if sid in (s_A, s_B):
                continue
            m = sample_types_count()
            types = sample_fault_types_for_sensor(m, config, rng, valid_triplets=valid_triplets, valid_pairs=valid_pairs)
            for idx, ft_name in enumerate(types):
                prec = episodes_by_sensor[sid][-1] if episodes_by_sensor[sid] else None
                t_start = sample_episode_start_step(prec, total_steps, warmup_steps, ev_steps, config, rng)
                xref = clean_physical_trajectories[sid][t_start] if clean_physical_trajectories and sid in clean_physical_trajectories else 20.0
                ep = fault_sim.sample_fault(
                    FaultType(ft_name), sid, sigma_k=sigmas[sid],
                    total_steps=total_steps, warmup_steps=warmup_steps,
                    physical_xref=xref, clean_base_value=xref,
                    rng=rng, fault_id=f"fault_{sid}_{idx}", start_step=t_start,
                    difficulty_tier=difficulty_tier,
                )
                episodes_by_sensor[sid].append(ep)

    else:
        # Standard allocation without shared-cause
        allocated = allocate_faulty_sensors(k_actual, groups_ids, rng)
        for sid, _ in allocated:
            m = sample_types_count()
            types = sample_fault_types_for_sensor(m, config, rng, valid_triplets=valid_triplets, valid_pairs=valid_pairs)
            for idx, ft_name in enumerate(types):
                prec = episodes_by_sensor[sid][-1] if episodes_by_sensor[sid] else None
                t_start = sample_episode_start_step(prec, total_steps, warmup_steps, ev_steps, config, rng)
                xref = clean_physical_trajectories[sid][t_start] if clean_physical_trajectories and sid in clean_physical_trajectories else 20.0
                ep = fault_sim.sample_fault(
                    FaultType(ft_name), sid, sigma_k=sigmas[sid],
                    total_steps=total_steps, warmup_steps=warmup_steps,
                    physical_xref=xref, clean_base_value=xref,
                    rng=rng, fault_id=f"fault_{sid}_{idx}", start_step=t_start,
                    difficulty_tier=difficulty_tier,
                )
                episodes_by_sensor[sid].append(ep)

    return {
        sid: SensorFaultSeries(sensor_id=sid, total_steps=total_steps, episodes=episodes_by_sensor[sid])
        for sid in all_sensor_ids
    }


def generate_group_faults(
    sensor_params: Sequence[Union[TemperatureSensorParams, str]],
    group_id: str = "group_00",
    sensor_sigmas: Optional[Dict[str, float]] = None,
    clean_physical_trajectories: Optional[Dict[str, np.ndarray]] = None,
    events: Optional[Any] = None,
    active_event_steps: Optional[Sequence[int]] = None,
    total_steps: int = 1920,
    warmup_steps: int = 480,
    config: Optional[Dict[str, Any]] = None,
    rng: Optional[Union[int, np.random.Generator]] = None,
    seed: Optional[int] = None,
    k_candidate: Optional[int] = None,
    force_shared_cause: Optional[bool] = None,
) -> Dict[str, SensorFaultSeries]:
    """Convenience wrapper for generating faults in a single-group scenario."""
    return generate_scenario_faults(
        groups={group_id: sensor_params},
        sensor_sigmas=sensor_sigmas,
        clean_physical_trajectories=clean_physical_trajectories,
        events=events,
        active_event_steps=active_event_steps,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        config=config,
        rng=rng,
        seed=seed,
        k_candidate=k_candidate,
        force_shared_cause=force_shared_cause,
    )

