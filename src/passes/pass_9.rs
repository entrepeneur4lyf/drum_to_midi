//! Pass 9: Velocity Estimation

use crate::analysis::{DrumClass, MidiEvent};
use crate::audio::AudioState;
use crate::config::Config;
use crate::spectral::StftData;
use crate::DrumErrorResult;
use std::collections::HashMap;

/// Velocity feature weights for different drum classes
#[derive(Debug, Clone)]
struct VelocityWeights {
    pub slope: f32,
    pub hf: f32,
    pub energy: f32,
}

/// Contextual velocity adjustment rules
#[derive(Debug, Clone)]
struct VelocityContextRules {
    pub snare_backbeat_boost: f32,
    pub hat_pair_delta: i32,
    pub kick_double_delta: i32,
}

/// Extract velocity features from audio around an event
fn extract_velocity_features(
    event: &MidiEvent,
    audio: &[f32],
    sr: u32,
    stfts: &HashMap<(usize, usize), StftData>,
) -> VelocityFeatures {
    let frame_idx = (event.time_sec * sr as f32 / 512.0) as usize;
    let onset_window = (0.05 * sr as f32) as usize; // 50ms window

    // Extract audio segment around the event
    let start_sample = (event.time_sec * sr as f32) as usize;
    let end_sample = (start_sample + onset_window).min(audio.len());
    let segment = &audio[start_sample..end_sample];

    // Compute attack slope (rate of energy increase)
    let attack_slope = if segment.len() > 1 {
        let mut slope_sum = 0.0;
        for i in 1..segment.len() {
            let energy_prev = segment[i - 1].powi(2);
            let energy_curr = segment[i].powi(2);
            slope_sum += energy_curr - energy_prev;
        }
        slope_sum / segment.len() as f32
    } else {
        0.0
    };

    // Compute HF ratio (high frequency energy vs total)
    let hf_ratio = if let Some(stft) = stfts.get(&(1024, 128)) {
        let freq_bins = stft.freqs.len();
        let hf_start = freq_bins * 3 / 4; // Upper 25% of frequencies

        if frame_idx < stft.s.ncols() {
            let total_energy: f32 = stft.s.column(frame_idx).iter().map(|&x| x.norm_sqr()).sum();
            let hf_energy: f32 = stft
                .s
                .column(frame_idx)
                .iter()
                .skip(hf_start)
                .map(|&x| x.norm_sqr())
                .sum();

            if total_energy > 0.0 {
                hf_energy / total_energy
            } else {
                0.0
            }
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Compute early energy (first 10ms)
    let early_window = (0.01 * sr as f32) as usize; // 10ms
    let early_energy = if segment.len() >= early_window {
        segment[..early_window].iter().map(|&x| x * x).sum::<f32>() / early_window as f32
    } else {
        segment.iter().map(|&x| x * x).sum::<f32>() / segment.len().max(1) as f32
    };

    VelocityFeatures {
        attack_slope,
        hf_ratio,
        early_energy,
    }
}

/// Velocity features extracted from audio
#[derive(Debug, Clone)]
struct VelocityFeatures {
    pub attack_slope: f32,
    pub hf_ratio: f32,
    pub early_energy: f32,
}

/// Compute raw velocity score from features and weights
fn compute_velocity_score(features: &VelocityFeatures, weights: &VelocityWeights) -> f32 {
    weights.slope * features.attack_slope.abs()
        + weights.hf * features.hf_ratio
        + weights.energy * features.early_energy.sqrt()
}

/// Apply robust z-score normalization using median and MAD
fn robust_normalize_velocities(velocities: &mut [f32]) {
    if velocities.is_empty() {
        return;
    }

    // Compute median
    let mut sorted_vels = velocities.to_vec();
    sorted_vels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted_vels.len().is_multiple_of(2) {
        (sorted_vels[sorted_vels.len() / 2 - 1] + sorted_vels[sorted_vels.len() / 2]) / 2.0
    } else {
        sorted_vels[sorted_vels.len() / 2]
    };

    // Compute MAD (Median Absolute Deviation)
    let mut deviations: Vec<f32> = sorted_vels.iter().map(|&v| (v - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = if deviations.len().is_multiple_of(2) {
        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    // Apply robust z-score normalization
    let scale = mad * 1.482_602_2; // Consistency constant for normal distribution
    for velocity in velocities.iter_mut() {
        if scale > 0.0 {
            *velocity = (*velocity - median) / scale;
        } else {
            *velocity = 0.0;
        }
    }
}

/// Apply class-specific velocity ranges and gamma correction
fn apply_class_ranges_and_gamma(velocities: &mut [f32], drum_class: DrumClass, gamma: f32) {
    // Class-specific velocity ranges (typical MIDI velocity ranges)
    let (min_vel, max_vel) = match drum_class {
        DrumClass::Kick => (30, 110),
        DrumClass::Snare => (40, 120),
        DrumClass::HiHat => (20, 100),
        DrumClass::Tom => (35, 105),
        DrumClass::Cymbal => (25, 95),
        _ => (25, 100),
    };

    for velocity in velocities.iter_mut() {
        // Apply gamma correction
        *velocity = velocity.signum() * velocity.abs().powf(gamma);

        // Scale to class-specific range
        *velocity = min_vel as f32 + (max_vel - min_vel) as f32 * (*velocity * 0.5 + 0.5);

        // Clamp to valid MIDI range
        *velocity = velocity.max(1.0).min(127.0);
    }
}

/// Apply contextual velocity adjustments
fn apply_contextual_adjustments(
    midi_events: &mut [MidiEvent],
    context_rules: &VelocityContextRules,
) {
    // Create a copy of time and class information for context analysis
    let event_info: Vec<(f32, DrumClass, usize)> = midi_events
        .iter()
        .enumerate()
        .map(|(i, e)| (e.time_sec, e.drum_class, i))
        .collect();

    // Sort by time
    let mut sorted_info = event_info.clone();
    sorted_info.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for &(time, class, original_idx) in &sorted_info {
        match class {
            DrumClass::Snare => {
                // Check if this is a backbeat (beat 2 or 4 in 4/4)
                if midi_events[original_idx].grid_position.beat == 1
                    || midi_events[original_idx].grid_position.beat == 3
                {
                    midi_events[original_idx].velocity =
                        ((midi_events[original_idx].velocity as f32
                            * context_rules.snare_backbeat_boost) as u8)
                            .min(127);
                }
            }
            DrumClass::HiHat => {
                // Check for hat pairs and reduce velocity of second hat
                if let Some(prev_idx) = sorted_info.iter().position(|&(t, _, _)| t < time) {
                    let (_, prev_class, _prev_original_idx) = sorted_info[prev_idx];
                    if prev_class == DrumClass::HiHat {
                        let time_diff = time - sorted_info[prev_idx].0;
                        if time_diff < 0.2 {
                            // Within 200ms, likely a pair
                            midi_events[original_idx].velocity =
                                (midi_events[original_idx].velocity as i32
                                    + context_rules.hat_pair_delta)
                                    .max(1)
                                    .min(127) as u8;
                        }
                    }
                }
            }
            DrumClass::Kick => {
                // Check for kick doubles
                if let Some(prev_idx) = sorted_info.iter().position(|&(t, _, _)| t < time) {
                    let (_, prev_class, _) = sorted_info[prev_idx];
                    if prev_class == DrumClass::Kick {
                        let time_diff = time - sorted_info[prev_idx].0;
                        if time_diff < 0.15 {
                            // Within 150ms, likely a double
                            midi_events[original_idx].velocity =
                                (midi_events[original_idx].velocity as i32
                                    + context_rules.kick_double_delta)
                                    .max(1)
                                    .min(127) as u8;
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Estimate decay time for cymbals using exponential fitting
fn estimate_cymbal_decay(event: &MidiEvent, audio: &[f32], sr: u32) -> f32 {
    if event.drum_class != DrumClass::Cymbal {
        return 0.0;
    }

    let start_sample = (event.time_sec * sr as f32) as usize;
    let analysis_window = (2.0 * sr as f32) as usize; // 2 seconds
    let end_sample = (start_sample + analysis_window).min(audio.len());

    if end_sample <= start_sample {
        return 0.0;
    }

    let segment = &audio[start_sample..end_sample];

    // Simple exponential decay fitting
    // Fit: y = A * exp(-t/tau)
    // Take log and fit linear: log(y) = log(A) - t/tau

    let mut decay_points = Vec::new();
    for i in 0..segment.len() {
        let energy = segment[i].powi(2);
        if energy > 1e-6 {
            // Only consider significant energy
            decay_points.push((i as f32 / sr as f32, energy));
        }
    }

    if decay_points.len() < 10 {
        return 0.0;
    }

    // Simple linear regression on log(energy) vs time
    let n = decay_points.len() as f32;
    let sum_t: f32 = decay_points.iter().map(|(t, _)| t).sum();
    let sum_log_e: f32 = decay_points.iter().map(|(_, e)| e.ln()).sum();
    let sum_t_log_e: f32 = decay_points.iter().map(|(t, e)| t * e.ln()).sum();
    let sum_t2: f32 = decay_points.iter().map(|(t, _)| t * t).sum();

    let denominator = n * sum_t2 - sum_t * sum_t;
    if denominator.abs() < 1e-6 {
        return 0.0;
    }

    let slope = (n * sum_t_log_e - sum_t * sum_log_e) / denominator;

    // tau = -1/slope (since we have -t/tau)
    if slope >= 0.0 {
        0.0 // Not decaying
    } else {
        -1.0 / slope
    }
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 9: Velocity Estimation");

    let midi_events = &mut state.midi_events;

    if midi_events.is_empty() {
        return Ok(());
    }

    // Initialize velocity weights for different classes
    let mut class_weights = HashMap::new();
    class_weights.insert(
        DrumClass::Kick,
        VelocityWeights {
            slope: 0.6,
            hf: 0.0,
            energy: 0.4,
        },
    );
    class_weights.insert(
        DrumClass::Snare,
        VelocityWeights {
            slope: 0.4,
            hf: 0.4,
            energy: 0.2,
        },
    );
    class_weights.insert(
        DrumClass::HiHat,
        VelocityWeights {
            slope: 0.4,
            hf: 0.6,
            energy: 0.0,
        },
    );
    class_weights.insert(
        DrumClass::Tom,
        VelocityWeights {
            slope: 0.5,
            hf: 0.2,
            energy: 0.3,
        },
    );
    class_weights.insert(
        DrumClass::Cymbal,
        VelocityWeights {
            slope: 0.3,
            hf: 0.5,
            energy: 0.2,
        },
    );

    // Extract features and compute raw velocities for each event
    let mut raw_velocities = Vec::new();
    let mut event_features = Vec::new();

    for event in &*midi_events {
        let features = extract_velocity_features(event, &state.y, state.sr, &state.stfts);
        let weights = class_weights
            .get(&event.drum_class)
            .unwrap_or(&VelocityWeights {
                slope: 0.5,
                hf: 0.3,
                energy: 0.2,
            });

        let raw_velocity = compute_velocity_score(&features, weights);
        raw_velocities.push(raw_velocity);
        event_features.push((event.drum_class, features));
    }

    // Apply robust normalization
    robust_normalize_velocities(&mut raw_velocities);

    // Apply class-specific ranges and gamma correction
    for (i, (drum_class, _)) in event_features.iter().enumerate() {
        let mut class_velocities = vec![raw_velocities[i]];
        apply_class_ranges_and_gamma(&mut class_velocities, *drum_class, config.velocity.gamma);
        raw_velocities[i] = class_velocities[0];
    }

    // Apply contextual adjustments
    let context_rules = VelocityContextRules {
        snare_backbeat_boost: config.velocity.context.snare_backbeat_boost,
        hat_pair_delta: config.velocity.context.hat_pair_delta,
        kick_double_delta: config.velocity.context.kick_double_delta,
    };

    // Update MIDI events with computed velocities
    for (i, event) in midi_events.iter_mut().enumerate() {
        event.velocity = raw_velocities[i] as u8;

        // Estimate decay time for cymbals
        if event.drum_class == DrumClass::Cymbal {
            let _decay_time = estimate_cymbal_decay(event, &state.y, state.sr);
            // Store decay time in a way that can be accessed later (could add to MidiEvent if needed)
        }
    }

    // Apply contextual adjustments
    apply_contextual_adjustments(midi_events, &context_rules);

    Ok(())
}
