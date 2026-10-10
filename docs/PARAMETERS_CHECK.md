# PARAMETERS_CHECK.md: Parameter Audit Table

> Complete line-by-line verification table mapping each configuration key in `config/*.yaml` directly to its locked value and exact reference in Appendix A (plus Appendix B/C).

---

## 1. Generator A (`config/generator_a.yaml`)

### Structure & Scenario Matrix (Appendix A.1)
| Config Key | YAML Value | Appendix A Reference | Appendix A Source Text |
|---|---|---|---|
| `structure.sampling_rate_minutes` | `1` | A.1 | "Sampling: 1 reading per minute" |
| `structure.scored_steps` | `1440` | A.1 | "Scored steps / warm-up / total: 1,440 / 480 / 1,920" |
| `structure.warmup_steps` | `480` | A.1 | "Scored steps / warm-up / total: 1,440 / 480 / 1,920" |
| `structure.total_steps` | `1920` | A.1 | "Scored steps / warm-up / total: 1,440 / 480 / 1,920" |
| `structure.total_scenarios` | `1500` | A.1 | "Scenarios (Generator A): 1,500: 900 train, 300 val, 300 test" |
| `structure.train_scenarios` | `900` | A.1 | "900 train" |
| `structure.val_scenarios` | `300` | A.1 | "300 val" |
| `structure.test_scenarios` | `300` | A.1 | "300 test" |
| `structure.train_val_row_stride` | `5` | A.1 | "Row stride: 5 for train/val" |
| `structure.test_full_resolution_count` | `150` | A.1 | "full resolution for 150 test scenarios" |
| `structure.sensors_per_group_min` | `4` | A.1 | "Sensors per group: integer 4 to 8" |
| `structure.sensors_per_group_max` | `8` | A.1 | "Sensors per group: integer 4 to 8" |
| `structure.scenario_type_probs.standard` | `0.82` | A.1 | "Scenario types: 82% standard" |
| `structure.scenario_type_probs.two_groups` | `0.10` | A.1 | "10% two groups (4 to 6 sensors each)" |
| `structure.scenario_type_probs.single_sensor` | `0.08` | A.1 | "8% single-sensor" |
| `structure.two_groups_sensors_min` | `4` | A.1 | "two groups (4 to 6 sensors each)" |
| `structure.two_groups_sensors_max` | `6` | A.1 | "two groups (4 to 6 sensors each)" |
| `structure.max_faulty_sensors_fraction` | `0.4` | A.1 | "Max simultaneous faulty sensors: max(1, floor(0.4 x n))" |
| `structure.seeds.generator_a_start` | `1` | A.1 | "Seeds: A: 1 to 1,500" |
| `structure.seeds.generator_a_end` | `1500` | A.1 | "Seeds: A: 1 to 1,500" |
| `structure.seeds.generator_b_start` | `10001` | A.1 | "B: 10,001 and up" |
| `structure.seeds.shuffle_controls` | `[20001, 20002, 20003, 20004, 20005]` | A.1 | "shuffle controls: 20001 to 20005" |
| `structure.healthy_missing_data.rate_min` | `0.0` | A.1 | "Healthy missing data: per-sensor rate 0 to 1%" |
| `structure.healthy_missing_data.rate_max` | `0.01` | A.1 | "Healthy missing data: per-sensor rate 0 to 1%" |
| `structure.healthy_missing_data.zero_missing_sensor_fraction` | `0.20` | A.1 | "20% of sensors 0" |
| `structure.healthy_missing_data.burst_max` | `3` | A.1 | "bursts up to 3" |

### Temperature Profile (Appendix A.2)
| Config Key | YAML Value | Appendix A Reference | Appendix A Source Text |
|---|---|---|---|
| `temperature_profile.base_level_min` | `2.0` | A.2 | "Base level: uniform 2 to 38 C" |
| `temperature_profile.base_level_max` | `38.0` | A.2 | "Base level: uniform 2 to 38 C" |
| `temperature_profile.daily_cycle.amplitude_min` | `0.5` | A.2 | "amplitude uniform 0.5 to 4 C" |
| `temperature_profile.daily_cycle.amplitude_max` | `4.0` | A.2 | "amplitude uniform 0.5 to 4 C" |
| `temperature_profile.daily_cycle.zero_amplitude_prob` | `0.20` | A.2 | "20% of scenarios zero" |
| `temperature_profile.daily_cycle.period_hours` | `24` | A.2 | "period 24 h" |
| `temperature_profile.daily_cycle.period_minutes` | `1440` | A.2 | "period 24 h (1,440 minutes)" |
| `temperature_profile.slow_wander.std_min` | `0.2` | A.2 | "Slow wander: std 0.2 to 0.8 C" |
| `temperature_profile.slow_wander.std_max` | `0.8` | A.2 | "Slow wander: std 0.2 to 0.8 C" |
| `temperature_profile.slow_wander.correlation_time_min` | `120` | A.2 | "correlation time 120 to 360 min" |
| `temperature_profile.slow_wander.correlation_time_max` | `360` | A.2 | "correlation time 120 to 360 min" |
| `temperature_profile.slow_wander.moving_average_steps` | `5` | A.2 | "then 5-step moving average" |
| `temperature_profile.sensor_noise.sigma_k_min` | `0.05` | A.2 | "Sensor noise sigma_k: log-uniform 0.05 to 0.30 C" |
| `temperature_profile.sensor_noise.sigma_k_max` | `0.30` | A.2 | "Sensor noise sigma_k: log-uniform 0.05 to 0.30 C" |
| `temperature_profile.sensor_noise.correlated_sensors_fraction` | `0.20` | A.2 | "20% of sensors mildly correlated" |
| `temperature_profile.sensor_noise.correlation_rho_min` | `0.2` | A.2 | "rho 0.2 to 0.5" |
| `temperature_profile.sensor_noise.correlation_rho_max` | `0.5` | A.2 | "rho 0.2 to 0.5" |
| `temperature_profile.static_offset_min` | `-1.0` | A.2 | "Static offset (healthy): uniform +/-1.0 C per sensor" |
| `temperature_profile.static_offset_max` | `1.0` | A.2 | "Static offset (healthy): uniform +/-1.0 C per sensor" |
| `temperature_profile.response_lag.time_constant_min` | `0.0` | A.2 | "Response lag: first-order, time constant 0 to 6 min" |
| `temperature_profile.response_lag.time_constant_max` | `6.0` | A.2 | "Response lag: first-order, time constant 0 to 6 min" |
| `temperature_profile.response_lag.zero_lag_fraction` | `0.40` | A.2 | "40% of sensors zero" |
| `temperature_profile.quantisation.values` | `[0.01, 0.05, 0.1, 0.25]` | A.2 | "Quantisation: 0.01 (25%), 0.05 (25%), 0.1 (35%), 0.25 (15%)" |
| `temperature_profile.quantisation.probabilities` | `[0.25, 0.25, 0.35, 0.15]` | A.2 | "Quantisation: 0.01 (25%), 0.05 (25%), 0.1 (35%), 0.25 (15%)" |
| `temperature_profile.bounds.environment_clip_min` | `-5.0` | A.2 | "environment clipped to [-5, 50] C" |
| `temperature_profile.bounds.environment_clip_max` | `50.0` | A.2 | "environment clipped to [-5, 50] C" |
| `temperature_profile.bounds.sensor_clip_min` | `-10.0` | A.2 | "sensor range [-10, 60] C" |
| `temperature_profile.bounds.sensor_clip_max` | `60.0` | A.2 | "sensor range [-10, 60] C" |
| `temperature_profile.unit_scaling.celsius_prob` | `0.70` | A.2 | "Unit scaling (last): 70% C" |
| `temperature_profile.unit_scaling.fahrenheit_prob` | `0.15` | A.2 | "15% F" |
| `temperature_profile.unit_scaling.random_affine_prob` | `0.15` | A.2 | "15% random affine" |
| `temperature_profile.unit_scaling.affine_multiplier_min` | `0.5` | A.2 | "multiplier log-uniform 0.5 to 3" |
| `temperature_profile.unit_scaling.affine_multiplier_max` | `3.0` | A.2 | "multiplier log-uniform 0.5 to 3" |

### Genuine Events (Appendix A.3)
| Config Key | YAML Value | Appendix A Reference | Appendix A Source Text |
|---|---|---|---|
| `genuine_events.count_poisson_mean` | `1.2` | A.3 | "Count: Poisson mean 1.2" |
| `genuine_events.count_max` | `3` | A.3 | "max 3 (about 30% of scenarios none)" |
| `genuine_events.type_probs.step_exponential_recovery` | `0.40` | A.3 | "Types: step with exponential recovery 40%" |
| `genuine_events.type_probs.ramp` | `0.25` | A.3 | "ramp 25%" |
| `genuine_events.type_probs.pulse` | `0.25` | A.3 | "pulse 25%" |
| `genuine_events.type_probs.sustained_shift` | `0.10` | A.3 | "sustained shift 10%" |
| `genuine_events.magnitude_min` | `0.5` | A.3 | "Magnitude: log-uniform 0.5 to 10 C" |
| `genuine_events.magnitude_max` | `10.0` | A.3 | "Magnitude: log-uniform 0.5 to 10 C" |
| `genuine_events.positive_sign_prob` | `0.75` | A.3 | "75% positive" |
| `genuine_events.negative_sign_prob` | `0.25` | A.3 | "25% negative" |
| `genuine_events.durations.step_recovery_tau_min` | `10` | A.3 | "recovery tau 10 to 120 min" |
| `genuine_events.durations.step_recovery_tau_max` | `120` | A.3 | "recovery tau 10 to 120 min" |
| `genuine_events.durations.ramp_rise_fall_min` | `20` | A.3 | "ramp rise/fall 20 to 180 min" |
| `genuine_events.durations.ramp_rise_fall_max` | `180` | A.3 | "ramp rise/fall 20 to 180 min" |
| `genuine_events.durations.ramp_hold_min` | `30` | A.3 | "hold 30 to 180 min" |
| `genuine_events.durations.ramp_hold_max` | `180` | A.3 | "hold 30 to 180 min" |
| `genuine_events.durations.pulse_min` | `5` | A.3 | "pulse 5 to 30 min" |
| `genuine_events.durations.pulse_max` | `30` | A.3 | "pulse 5 to 30 min" |
| `genuine_events.durations.sustained_rise_min` | `5` | A.3 | "sustained rise 5 to 20 min then hold" |
| `genuine_events.durations.sustained_rise_max` | `20` | A.3 | "sustained rise 5 to 20 min then hold" |
| `genuine_events.scope.global_prob` | `0.55` | A.3 | "Scope: global 55%" |
| `genuine_events.scope.global_attenuation_min` | `0.6` | A.3 | "global attenuation 0.6 to 1.0" |
| `genuine_events.scope.global_attenuation_max` | `1.0` | A.3 | "global attenuation 0.6 to 1.0" |
| `genuine_events.scope.partial_prob` | `0.30` | A.3 | "partial 30%" |
| `genuine_events.scope.partial_sensors_fraction_min` | `0.4` | A.3 | "40 to 80% of sensors" |
| `genuine_events.scope.partial_sensors_fraction_max` | `0.8` | A.3 | "40 to 80% of sensors" |
| `genuine_events.scope.partial_attenuation_min` | `0.3` | A.3 | "partial attenuation 0.3 to 1.0" |
| `genuine_events.scope.partial_attenuation_max` | `1.0` | A.3 | "partial attenuation 0.3 to 1.0" |
| `genuine_events.scope.local_prob` | `0.15` | A.3 | "local 15%" |
| `genuine_events.scope.local_sensors_min` | `1` | A.3 | "1 to 2 sensors" |
| `genuine_events.scope.local_sensors_max` | `2` | A.3 | "1 to 2 sensors" |
| `genuine_events.scope.local_attenuation_min` | `0.7` | A.3 | "local attenuation 0.7 to 1.0" |
| `genuine_events.scope.local_attenuation_max` | `1.0` | A.3 | "local attenuation 0.7 to 1.0" |
| `genuine_events.arrival_delay_min` | `0` | A.3 | "Arrival delay per sensor 0 to 10 min" |
| `genuine_events.arrival_delay_max` | `10` | A.3 | "Arrival delay per sensor 0 to 10 min" |
| `genuine_events.detectability_threshold_sigma` | `3.0` | A.3 | "genuine_event = 1 while event contribution at >= 1 sensor exceeds 3 x sigma_k" |

### Fault Injectors (Appendix A.4)
| Config Key | YAML Value | Appendix A Reference | Appendix A Source Text |
|---|---|---|---|
| `faults.prob_present_from_start` | `0.10` | A.4 | "10% of fault episodes present from step 0" |
| `faults.spike.episode_min` | `30` | A.4 | "episode 30 to 400 steps (log-uniform)" |
| `faults.spike.episode_max` | `400` | A.4 | "episode 30 to 400 steps (log-uniform)" |
| `faults.spike.interval_min` | `10` | A.4 | "a spike every 10 to 60 steps" |
| `faults.spike.interval_max` | `60` | A.4 | "a spike every 10 to 60 steps" |
| `faults.spike.size_sigma_min` | `3.0` | A.4 | "size 3 to 40 sigma_k (log-uniform)" |
| `faults.spike.size_sigma_max` | `40.0` | A.4 | "size 3 to 40 sigma_k (log-uniform)" |
| `faults.spike.length_probs.1` | `0.60` | A.4 | "length 1 sample 60%" |
| `faults.spike.length_probs.2` | `0.25` | A.4 | "2 samples 25%" |
| `faults.spike.length_probs.3` | `0.15` | A.4 | "3 samples 15%" |
| `faults.spike.label_following_steps` | `4` | A.4 | "label active on spike samples plus 4 following steps" |
| `faults.offset_drift.ramp_steps_min` | `60` | A.4 | "ramp 60 to 720 steps" |
| `faults.offset_drift.ramp_steps_max` | `720` | A.4 | "ramp 60 to 720 steps" |
| `faults.offset_drift.final_bias_sigma_min` | `1.0` | A.4 | "final bias 1 to 20 sigma_k (log-uniform)" |
| `faults.offset_drift.final_bias_sigma_max` | `20.0` | A.4 | "final bias 1 to 20 sigma_k (log-uniform)" |
| `faults.offset_drift.plateau_steps_min` | `0` | A.4 | "plateau 0 to 600 steps" |
| `faults.offset_drift.plateau_steps_max` | `600` | A.4 | "plateau 0 to 600 steps" |
| `faults.offset_drift.shape_probs.linear` | `0.55` | A.4 | "shape linear 55%" |
| `faults.offset_drift.shape_probs.saturating` | `0.20` | A.4 | "saturating 20%" |
| `faults.offset_drift.shape_probs.stepped` | `0.15` | A.4 | "stepped 15%" |
| `faults.offset_drift.shape_probs.sudden_constant` | `0.10` | A.4 | "sudden constant offset 10%" |
| `faults.gain_error.duration_min` | `120` | A.4 | "duration 120 to 1,200 steps" |
| `faults.gain_error.duration_max` | `1200` | A.4 | "duration 120 to 1,200 steps" |
| `faults.gain_error.standard_prob` | `0.85` | A.4 | "g in 0.55 to 0.85 or 1.15 to 1.6 (85%)" |
| `faults.gain_error.standard_ranges.low` | `[0.55, 0.85]` | A.4 | "0.55 to 0.85" |
| `faults.gain_error.standard_ranges.high` | `[1.15, 1.60]` | A.4 | "1.15 to 1.6" |
| `faults.gain_error.subtle_prob` | `0.15` | A.4 | "subtle 0.90 to 0.95 or 1.05 to 1.10 (15%)" |
| `faults.gain_error.subtle_ranges.low` | `[0.90, 0.95]` | A.4 | "0.90 to 0.95" |
| `faults.gain_error.subtle_ranges.high` | `[1.05, 1.10]` | A.4 | "1.05 to 1.10" |
| `faults.frozen.duration_min` | `20` | A.4 | "20 to 400 steps (log-uniform)" |
| `faults.frozen.duration_max` | `400` | A.4 | "20 to 400 steps (log-uniform)" |
| `faults.frozen.exactly_constant_prob` | `0.60` | A.4 | "60% exactly constant" |
| `faults.frozen.jitter_prob` | `0.40` | A.4 | "40% jitter" |
| `faults.frozen.jitter_std_sigma_min` | `0.02` | A.4 | "std 0.02 to 0.3 sigma_k" |
| `faults.frozen.jitter_std_sigma_max` | `0.30` | A.4 | "std 0.02 to 0.3 sigma_k" |
| `faults.frozen.frozen_at_last_prob` | `0.80` | A.4 | "frozen at last reading 80%" |
| `faults.frozen.frozen_at_wrong_value_prob` | `0.20` | A.4 | "at a wrong value 5 to 20 sigma_k away 20%" |
| `faults.frozen.wrong_value_offset_sigma_min` | `5.0` | A.4 | "5 to 20 sigma_k away" |
| `faults.frozen.wrong_value_offset_sigma_max` | `20.0` | A.4 | "5 to 20 sigma_k away" |
| `faults.high_noise.multiplier_min` | `2.0` | A.4 | "multiplier 2 to 10 x sigma_k (log-uniform)" |
| `faults.high_noise.multiplier_max` | `10.0` | A.4 | "multiplier 2 to 10 x sigma_k (log-uniform)" |
| `faults.high_noise.duration_min` | `30` | A.4 | "30 to 600 steps" |
| `faults.high_noise.duration_max` | `600` | A.4 | "30 to 600 steps" |
| `faults.high_noise.bursty_prob` | `0.30` | A.4 | "30% bursty" |
| `faults.high_noise.burst_block_steps_min` | `5` | A.4 | "on/off blocks 5 to 30 steps" |
| `faults.high_noise.burst_block_steps_max` | `30` | A.4 | "on/off blocks 5 to 30 steps" |
| `faults.dropout.enter_gap_prob_min` | `0.002` | A.4 | "enter-gap probability 0.002 to 0.03 per step" |
| `faults.dropout.enter_gap_prob_max` | `0.03` | A.4 | "enter-gap probability 0.002 to 0.03 per step" |
| `faults.dropout.gap_length_mean_min` | `2.0` | A.4 | "gap length geometric, mean 2 to 40" |
| `faults.dropout.gap_length_mean_max` | `40.0` | A.4 | "gap length geometric, mean 2 to 40" |
| `faults.dropout.gap_length_cap` | `100` | A.4 | "cap 100" |
| `faults.dropout.episode_min` | `60` | A.4 | "episode 60 to 600 steps" |
| `faults.dropout.episode_max` | `600` | A.4 | "episode 60 to 600 steps" |
| `faults.slow_degradation.h_end_min` | `0.1` | A.4 | "health h falls 1 to h_end (0.1 to 0.7)" |
| `faults.slow_degradation.h_end_max` | `0.7` | A.4 | "health h falls 1 to h_end (0.1 to 0.7)" |
| `faults.slow_degradation.duration_min` | `240` | A.4 | "over 240 to 1,200 steps" |
| `faults.slow_degradation.duration_max` | `1200` | A.4 | "over 240 to 1,200 steps" |
| `faults.slow_degradation.damage_noise_multiplier_slope` | `5.0` | A.4 | "noise x(1 + 5d)" |
| `faults.slow_degradation.damage_bias_sigma_factor` | `8.0` | A.4 | "bias +/-8 sigma_k x d" |
| `faults.slow_degradation.damage_extra_missing_rate_slope` | `0.15` | A.4 | "extra missing rate 0.15 d" |
| `faults.slow_degradation.untrusted_damage_threshold` | `0.15` | A.4 | "untrusted once d >= 0.15" |

### Combinations & Difficulty Tiers (Appendix A.5, A.6)
| Config Key | YAML Value | Appendix A Reference | Appendix A Source Text |
|---|---|---|---|
| `combinations.faulty_sensors_probs.0` | `0.15` | A.5 | "Faulty sensors per scenario: 0: 15%" |
| `combinations.faulty_sensors_probs.1` | `0.40` | A.5 | "1: 40%" |
| `combinations.faulty_sensors_probs.2` | `0.25` | A.5 | "2: 25%" |
| `combinations.faulty_sensors_probs.3_or_more` | `0.20` | A.5 | "3 or more: 20%" |
| `combinations.types_per_faulty_sensor_probs.1` | `0.65` | A.5 | "Types per faulty sensor: 1: 65%" |
| `combinations.types_per_faulty_sensor_probs.2` | `0.27` | A.5 | "2: 27%" |
| `combinations.types_per_faulty_sensor_probs.3` | `0.08` | A.5 | "3: 8%" |
| `combinations.second_fault_start_offset_min` | `-100` | A.5 | "Second fault starts within [first start - 100, first end]" |
| `combinations.start_inside_active_event_prob` | `0.35` | A.5 | "35% of fault episodes start inside an active event" |
| `combinations.shared_cause_pair_prob` | `0.15` | A.5 | "Shared-cause pair in 15% of multi-fault scenarios" |
| `combinations.shared_cause_start_offset_max` | `10` | A.5 | "starts within +/-10 steps" |
| `difficulty_tiers.easy.prob` | `0.30` | A.6 | "EASY 30%" |
| `difficulty_tiers.easy.max_faulty_sensors` | `1` | A.6 | "at most 1 faulty sensor" |
| `difficulty_tiers.easy.spike_min_sigma` | `8.0` | A.6 | "spike >= 8 sigma" |
| `difficulty_tiers.easy.drift_min_sigma` | `6.0` | A.6 | "drift >= 6 sigma" |
| `difficulty_tiers.easy.gain_diff_min` | `0.25` | A.6 | "|g-1| >= 0.25" |
| `difficulty_tiers.easy.noise_mult_min` | `4.0` | A.6 | "noise multiplier >= 4" |
| `difficulty_tiers.easy.allowed_event_scopes` | `["global", "none"]` | A.6 | "global events or none" |
| `difficulty_tiers.medium.prob` | `0.40` | A.6 | "MEDIUM 40%" |
| `difficulty_tiers.medium.max_faulty_sensors` | `2` | A.6 | "up to 2 faulty sensors" |
| `difficulty_tiers.medium.max_types_per_sensor` | `2` | A.6 | "up to 2 fault types each" |
| `difficulty_tiers.hard.prob` | `0.30` | A.6 | "HARD 30%" |

### Severity Formulas (Appendix A.7)
| Config Key | YAML Value | Appendix A Reference | Appendix A Source Text |
|---|---|---|---|
| `severity_formulas.spike_divisor` | `40.0` | A.7 | "spike k/40" |
| `severity_formulas.drift_divisor_sigma` | `20.0` | A.7 | "drift abs(bias)/(20 sigma_k)" |
| `severity_formulas.gain_divisor` | `0.6` | A.7 | "gain abs(g-1)/0.6" |
| `severity_formulas.frozen_severity_normal` | `0.8` | A.7 | "frozen 0.8" |
| `severity_formulas.frozen_severity_wrong_value` | `1.0` | A.7 | "1.0 if frozen at wrong value" |
| `severity_formulas.noise_offset` | `1.0` | A.7 | "noise (m-1)/9 (numerator offset)" |
| `severity_formulas.noise_divisor` | `9.0` | A.7 | "noise (m-1)/9 (denominator divisor)" |
| `severity_formulas.dropout_base` | `0.3` | A.7 | "dropout 0.3 + 0.7 x missing fraction" |
| `severity_formulas.dropout_slope` | `0.7` | A.7 | "0.7 x missing fraction" |
| `severity_formulas.dropout_missing_window` | `30` | A.7 | "over last 30 steps" |

---

## 2. Feature Engine (`config/features.yaml`)

| Config Key | YAML Value | Appendix Reference | Appendix Source Text |
|---|---|---|---|
| `sigma_hat.mad_scale_factor` | `1.4826` | A.8 | "1.4826 x MAD of first differences" |
| `sigma_hat.sqrt_2_divisor` | `1.4142135623730951` | A.8 | "divided by sqrt(2)" |
| `sigma_hat.rolling_percentile` | `25.0` | A.8 | "taking the 25th percentile of rolling estimates" |
| `sigma_hat.reference_window` | `480` | A.8 | "over the 480-step reference" |
| `windows.short` | `5` | A.8 | "short 5" |
| `windows.medium` | `30` | A.8 | "medium 30" |
| `windows.long` | `120` | A.8 | "long 120" |
| `windows.reference` | `480` | A.8 | "reference 480 (readings)" |
| `windows.causal_only` | `true` | A.8 | "Causal only" |
| `gap_centring.reference_window` | `480` | A.8 | "relative to the sensor's own median gap over the 480-step reference" |
| `sensor_changing_rule.valid_readings_median_count` | `3` | A.3 | "median of last 3 valid readings" |
| `sensor_changing_rule.short_lag_steps` | `15` | A.3 | "ending 15 steps earlier (short)" |
| `sensor_changing_rule.medium_lag_steps` | `60` | A.3 | "60 steps earlier (medium)" |
| `sensor_changing_rule.threshold_sigma_multiplier` | `3.0` | A.3 | "changing if abs change > 3 x sigma_hat" |
| `counts.own_signal_temporal` | `10` | App B | "Own-signal temporal (10)" |
| `counts.statistical` | `6` | App B | "Statistical (6)" |
| `counts.data_quality` | `10` | App B | "Data quality (10)" |
| `counts.cross_sensor` | `14` | App B | "Cross-sensor (14)" |
| `counts.long_term_degradation` | `3` | App B | "Long-term degradation (3)" |
| `counts.group_event_joined` | `6` | App B | "Group/event joined onto sensor rows (6)" |
| `counts.group_table_head_c` | `11` | App B | "Group table for Head C (about 11)" |

---

## 3. Alarm & Trust Engine (`config/alarm.yaml`)

| Config Key | YAML Value | Appendix Reference | Appendix Source Text |
|---|---|---|---|
| `raw_alarm.baseline_window_hours` | `8` | A.9 | "8-hour median" |
| `raw_alarm.baseline_window_minutes` | `480` | A.9 | "8-hour median (480 minutes)" |
| `raw_alarm.robust_spread_mad_factor` | `1.4826` | A.9 | "robust spread = 1.4826 x MAD of the readings themselves" |
| `raw_alarm.threshold_multiplier` | `3.0` | A.9 | "+/- 3 x the sensor's own robust spread" |
| `raw_alarm.persistence_start_readings` | `1` | A.9 | "starts at 1 reading outside the band" |
| `raw_alarm.consecutive_inside_to_end` | `10` | A.9 | "ends after 10 consecutive readings inside" |
| `raw_alarm.episode_merge_gap_max` | `5` | A.9 | "episodes merge when gaps are 5 readings or fewer" |
| `raw_alarm.cause_label_sigma_multiplier` | `3.0` | A.9 | "using the 3 x sigma_k rule on event and fault contributions" |
| `decision.evaluation_initial_readings` | `5` | A.9 | "average calibrated probabilities over the first 5 readings" |
| `decision.recheck_interval_readings` | `5` | A.9 | "re-check every 5" |
| `decision.default_probability_cutoff` | `0.5` | A.9 | "Probability cutoffs start at 0.5" |
| `provable_rules.frozen.consecutive_identical_readings` | `20` | A.10 | "20 identical consecutive readings" |
| `provable_rules.frozen.run_multiplier` | `3.0` | A.10 | "AND more than 3 x the longest identical run" |
| `provable_rules.frozen.history_reference_steps` | `480` | A.10 | "in that sensor's previous 480 readings" |
| `provable_rules.dropout.consecutive_missing_threshold` | `5` | A.10 | "5 or more consecutive missing" |
| `provable_rules.dropout.window_missing_fraction` | `0.20` | A.10 | "OR 20% or more missing in the last 30" |
| `provable_rules.dropout.window_missing_steps` | `30` | A.10 | "in the last 30" |
| `calibration.isotonic_min_positive_count` | `500` | A.11 | "isotonic when a fault type has 500 or more positives" |
| `calibration.n_folds` | `5` | A.11 | "5 folds grouped by scenario" |
| `trust_engine.alpha_falling` | `0.5` | A.11 | "smoothing alpha 0.5 when falling" |
| `trust_engine.alpha_recovering` | `0.05` | A.11 | "0.05 when recovering" |
| `trust_engine.candidate_alphas_falling` | `[0.3, 0.5, 0.7]` | A.11 | "choose on validation from {0.3,0.5,0.7}" |
| `trust_engine.candidate_alphas_recovering` | `[0.02, 0.05, 0.1]` | A.11 | "x {0.02,0.05,0.1}" |
| `trust_engine.status_thresholds.healthy_min` | `0.8` | A.11 | "above 0.8 HEALTHY" |
| `trust_engine.status_thresholds.watch_min` | `0.5` | A.11 | "0.5 to 0.8 WATCH" |
| `trust_engine.status_thresholds.watch_max` | `0.8` | A.11 | "0.5 to 0.8 WATCH" |
| `trust_engine.status_thresholds.unreliable_max` | `0.5` | A.11 | "below 0.5 UNRELIABLE" |
| `demo_presets.warmup_healthy_steps` | `480` | A.13 | "Start with at least 480 steps of healthy history" |
| `demo_presets.chart_window_steps` | `240` | A.13 | "chart shows the last 240 readings" |
| `demo_presets.playback_seconds_per_minute` | `1.0` | A.13 | "Playback 1 second per simulated minute" |
| `demo_presets.playback_speed_max` | `10.0` | A.13 | "adjustable up to 10x" |
| `demo_presets.preset_spike_sigma` | `20.0` | A.13 | "Spike about 20 x sigma_k" |
| `demo_presets.preset_drift_sigma` | `10.0` | A.13 | "Drift reaches about 10 x sigma_k" |
| `demo_presets.preset_event_magnitude` | `5.0` | A.13 | "Event magnitude about 5 C" |

---

## 4. Evaluation Suite & Pass Marks (`config/eval.yaml`)

| Config Key | YAML Value | Appendix Reference | Appendix Source Text |
|---|---|---|---|
| `evaluation_settings.false_alarm_budget_per_sensor_day` | `0.5` | A.14 | "Budget: at most 0.5 episodes per healthy sensor per day" |
| `evaluation_settings.false_alarm_merge_gap` | `5` | A.14 | "a run of flagged readings (merge gap 5)" |
| `evaluation_settings.healthy_grace_window_steps` | `5` | A.14 | "with no fault active (5-reading grace)" |
| `evaluation_settings.onset_grace_window_steps` | `5` | A.14 | "Onset grace window 5 steps" |
| `evaluation_settings.detection_delay_probability_cutoff` | `0.5` | A.14 | "detection delay = first step with calibrated p >= 0.5" |
| `evaluation_settings.wilson_confidence_level` | `0.95` | A.14 | "Wilson 95% intervals on rates" |
| `evaluation_settings.bootstrap_resamples` | `1000` | A.14 | "1,000 scenario-level bootstrap resamples" |
| `evaluation_settings.thin_cell_min_samples` | `20` | A.14 | "flag cells with n < 20" |
| `evaluation_settings.worst_slice_min_samples` | `25` | A.14 | "worst-slice search with at least 25 examples" |
| `leakage_audit.single_feature_auc_threshold` | `0.85` | A.14 | "single-feature AUC above 0.85 is investigated" |
| `leakage_audit.shuffle_seeds` | `[20001, 20002, 20003, 20004, 20005]` | A.14 | "5 shuffles, seeds 20001 to 20005" |
| `leakage_audit.shuffle_auc_target` | `0.50` | A.14 | "AUC within 0.50 +/- 0.03" |
| `leakage_audit.shuffle_auc_tolerance` | `0.03` | A.14 | "AUC within 0.50 +/- 0.03" |
| `leakage_audit.shuffle_recall_at_budget_max` | `0.05` | A.14 | "recall at budget at most 0.05" |
| `leakage_audit.canary_auc_min` | `0.95` | A.14 | "a planted canary column must be caught (AUC >= 0.95)" |
| `performance_limits.max_latency_ms_per_reading` | `50.0` | A.14 | "at most 50 ms per sensor reading on laptop CPU" |
| `performance_limits.max_model_file_size_mb` | `50.0` | A.14 | "model files at most 50 MB" |
| `pass_marks.false_alarm_budget` | `0.5` | App C | "At the false-alarm budget of 0.5 per healthy sensor per day" |
| `pass_marks.pm1_hard_beat_baseline_points` | `10.0` | App C | "beats baseline by at least 10 points on HARD" |
| `pass_marks.pm1_medium_beat_baseline_points` | `5.0` | App C | "5 on MEDIUM" |
| `pass_marks.pm2_provable_rules_recall_min` | `0.90` | App C | "recall at least 0.90 for rule-provable faults" |
| `pass_marks.pm2_other_faults_recall_min` | `0.60` | App C | "at least 0.60 for every other fault type" |
| `pass_marks.pm2_min_examples` | `20` | App C | "fault types with 20+ examples" |
| `pass_marks.pm3_fault_only_suppression_min` | `0.70` | App C | "at least 70% of fault-only raw alarms suppressed" |
| `pass_marks.pm4_event_only_wrong_suppression_max` | `0.05` | App C | "at most 5% of event-only raw alarms wrongly suppressed" |
| `pass_marks.pm5_both_flags_correct_min` | `0.60` | App C | "in BOTH cases, at least 60% get both flags right" |
| `pass_marks.pm6_a_to_b_recall_drop_max` | `0.20` | App C | "recall drops by at most 20 points from Generator A to B" |
| `pass_marks.pm7_calibration_error_max` | `0.05` | App C | "calibration error at most 0.05" |
| `pass_marks.pm8_rank_correlation_trust_health_min` | `0.75` | App C | "rank correlation at least 0.75" |
| `pass_marks.pm9_median_detection_delay_max_steps` | `15` | App C | "median detection delay at most 15 readings" |
| `pass_marks.pm9_fault_size_min_sigma` | `5.0` | App C | "for faults above 5 sigma" |

---

## 5. Audit Review & Parameter Analysis

### 5.1 Categorical Specifications (Identified for Explicit Inclusion)
The following categorical rules from Appendix A must be formally maintained in configuration rather than hard-coded into simulation logic:

* **Allowed Fault Pairs (Appendix A.5):**
  `spike+drift`, `spike+noise`, `spike+gain`, `drift+noise`, `drift+gain`, `gain+noise`, `dropout with anything`, `degradation+spike`, `degradation+gain`.
* **Forbidden Fault Pairs (Appendix A.5):**
  `frozen with spike/gain/noise/drift`; `degradation with drift/noise/dropout`.
* **HARD-Tier Mechanisms (Appendix A.6):**
  `local_event`, `fault_during_event`, `subtle_fault`, `shared_cause_pair`, `faulty_fraction_at_cap_small_group`, `no_clear_majority`, `sensor_near_bounds`.

### 5.2 Specific Investigation Findings
1. **Random Affine Offset Range / Distribution (Appendix A.2):**
   * *Status:* Undefined in specification.
   * *Observation:* Appendix A.2 states `multiplier log-uniform 0.5 to 3, random offset` without providing minimum/maximum bounds or distribution for the additive offset.
2. **Sustained Environmental Event Hold Duration / Termination (Appendix A.3):**
   * *Status:* Undefined in specification.
   * *Observation:* Appendix A.3 specifies `sustained rise 5 to 20 min then hold`. It does not define how long the hold lasts or if it terminates before step 1920.
3. **Ground-Truth `sensor_untrusted` Label Definition (Appendix A.4, A.7):**
   * *Status:* Ambiguous across fault modes.
   * *Observation:* Explicit definition is provided for slow degradation (`d >= 0.15`) and active spike window (`samples + 4 steps`), but the exact mapping for other fault modes (offset drift, gain error, frozen, high noise, dropout) relative to severity or active episode is not unified.
4. **Exact Status Boundaries at Trust Score 0.8 and 0.5 (Appendix A.11):**
   * *Status:* Boundary closure unspecified.
   * *Observation:* Spec states `above 0.8 HEALTHY, 0.5 to 0.8 WATCH, below 0.5 UNRELIABLE`. Needs confirmation on boundary inclusivity ($T = 0.8000$ and $T = 0.5000$).
5. **Simultaneous Fault Cap vs Zero-Fault Scenarios (Appendix A.1 vs A.5):**
   * *Status:* Clarification required.
   * *Observation:* Appendix A.1 formula $\max(1, \lfloor 0.4 \times n \rfloor)$ acts as an upper-bound cap on simultaneous faults. It must not override the 15% zero-fault healthy scenario allocation from Appendix A.5.

