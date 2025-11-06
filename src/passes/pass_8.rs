//! Pass 8: Grid Inference + Fill/Silence Protection (Improved - Fixed)

use crate::analysis::{
    DrumClass, GridPosition, MidiEvent, RefinedEvent, SelfPriorMatrices, TempoMeterAnalysis,
};
use crate::audio::AudioState;
use crate::config::Config;
use crate::DrumError;
use crate::DrumErrorResult;
use std::collections::HashMap;

/// Improved Grid inference configuration parameters
#[derive(Debug, Clone)]
struct GridInferenceConfig {
    /// Acoustic score decay alpha (α=0.05)
    acoustic_decay_alpha: f32,
    /// Acoustic score tolerance in ms (tol=15ms)
    acoustic_tolerance_ms: f32,
    /// Acoustic weight in combined score (λ_acoustic=0.7)
    acoustic_weight: f32,
    /// Prior weight in combined score (λ_prior=0.3)
    prior_weight: f32,
    /// Fill protection percentile (95th percentile)
    fill_percentile: f32,
    /// Silence protection percentile (5th percentile)
    silence_percentile: f32,
    /// Minimum velocity for ghost notes
    min_ghost_velocity: u8,
    /// Maximum velocity for MIDI events
    max_velocity: u8,
    /// Ghost note threshold based on energy ratio
    ghost_threshold: f32,
    /// Velocity boost for on-beat events
    velocity_boost_on_beat: f32,
    /// Minimum confidence threshold (hardcoded since not in Config)
    min_confidence: f32,
    /// Velocity gamma (hardcoded)
    velocity_gamma: f32,
}

impl Default for GridInferenceConfig {
    fn default() -> Self {
        Self {
            acoustic_decay_alpha: 0.05,
            acoustic_tolerance_ms: 15.0,
            acoustic_weight: 0.8, // Increase acoustic weight
            prior_weight: 0.2, // Decrease prior weight to rely less on priors
            fill_percentile: 95.0,
            silence_percentile: 5.0,
            min_ghost_velocity: 25, // Increased from 20 to reduce over-flagging
            max_velocity: 127,
            ghost_threshold: 0.3, // Tune based on tests
            velocity_boost_on_beat: 1.2,
            min_confidence: 0.2, // Lowered to retain more events
            velocity_gamma: 127.0, // Hardcoded gamma for velocity scaling
        }
    }
}

/// Compute acoustic score for an event at a grid position
fn compute_acoustic_score(
    event: &RefinedEvent,
    grid_time_sec: f32,
    config: &GridInferenceConfig,
) -> f32 {
    let time_diff_ms = (event.refined_time_sec - grid_time_sec).abs() * 1000.0;

    if time_diff_ms > config.acoustic_tolerance_ms {
        return 0.0;
    }

    // Exponential decay from time difference
    let decay_factor = (-config.acoustic_decay_alpha * time_diff_ms).exp();

    // Weight by event confidence
    decay_factor * event.confidence // Use confidence from RefinedEvent
}

/// Compute prior probability for a drum class at a grid position
fn compute_prior_probability(
    drum_class: DrumClass,
    grid_position: &GridPosition,
    priors: &SelfPriorMatrices,
) -> f32 {
    if let Some(class_priors) = priors.class_priors.get(&drum_class) {
        if let Some(class_confidences) = priors.class_confidences.get(&drum_class) {
            // Use sub-beat position within the beat
            let slot_idx = grid_position.sub_beat % priors.grid_slots_per_beat;

            if slot_idx < class_priors.len() {
                // Weight by confidence
                class_priors[slot_idx] * class_confidences[slot_idx]
            } else {
                0.0
            }
        } else {
            0.0
        }
    } else {
        0.0
    }
}

/// Compute rhythmic density curve for fill/silence protection (per beat)
fn compute_density_curve(
    events: &[RefinedEvent],
    tempo_analysis: &TempoMeterAnalysis,
    _config: &Config,
) -> Vec<f32> {
    let duration_sec = events
        .iter()
        .map(|e| e.refined_time_sec)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(10.0);

    let n_beats = ((duration_sec * tempo_analysis.bpm) / 60.0).ceil() as usize;
    let mut density_curve = vec![0.0; n_beats];

    // Count events per beat
    for event in events {
        let beat_idx = ((event.refined_time_sec * tempo_analysis.bpm) / 60.0) as usize;
        if beat_idx < density_curve.len() {
            density_curve[beat_idx] += 1.0;
        }
    }

    density_curve
}

/// Compute percentile-based density thresholds
fn compute_density_thresholds(density_curve: &[f32], config: &GridInferenceConfig) -> (f32, f32) {
    if density_curve.is_empty() {
        return (0.0, 0.0);
    }

    let mut sorted_density = density_curve.to_vec();
    sorted_density.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let fill_idx = ((config.fill_percentile / 100.0) * (sorted_density.len() - 1) as f32) as usize;
    let silence_idx =
        ((config.silence_percentile / 100.0) * (sorted_density.len() - 1) as f32) as usize;

    (sorted_density[fill_idx], sorted_density[silence_idx])
}

/// Check if event is a ghost note based on energy and density
fn is_ghost_note(
    event: &RefinedEvent,
    spectral_max_energy: f32, // Computed from events
    local_density: f32,
    config: &GridInferenceConfig,
) -> bool {
    let energy_ratio = event.confidence / spectral_max_energy.max(1.0); // Use confidence
    energy_ratio < config.ghost_threshold || (local_density > 0.7 && event.confidence < 0.6)
}

/// Compute velocity based on acoustic score, prior, and density
fn compute_velocity(
    acoustic_score: f32,
    prior_score: f32,
    density_score: f32,
    is_on_beat: bool,
    config: &GridInferenceConfig,
) -> u8 {
    let base_score = config.acoustic_weight * acoustic_score + config.prior_weight * prior_score;
    let mut velocity = (base_score * config.velocity_gamma) as u8;
    velocity = velocity.max(config.min_ghost_velocity);

    // Boost for on-beat events
    if is_on_beat {
        velocity = ((velocity as f32 * config.velocity_boost_on_beat).min(config.max_velocity as f32)) as u8;
    }

    // Dampen in high-density areas
    velocity = ((velocity as f32 * (1.0 - density_score * 0.3)) as u8).max(1);
    velocity.min(config.max_velocity)
}

/// Check if time is near a strong beat
fn is_on_beat(time_sec: f32, tempo_analysis: &TempoMeterAnalysis, tolerance_sec: f32) -> bool {
    // Simple check: find nearest beat position
    let beat_pos = (time_sec * tempo_analysis.bpm / 60.0).floor() * 60.0 / tempo_analysis.bpm;
    (time_sec - beat_pos).abs() < tolerance_sec // e.g., 0.05s tolerance
}

/// Update local density in a sliding window (0.5s windows)
fn update_density_window(
    density_window: &mut HashMap<u32, f32>,
    time_sec: f32,
    confidence: f32,
) {
    let window_key = get_window_key(time_sec);
    *density_window.entry(window_key).or_insert(0.0) += confidence;
}

/// Get window key for density calculation
fn get_window_key(time_sec: f32) -> u32 {
    ((time_sec / 0.5).floor()) as u32
}

/// Convert time to grid position (simplified; enhance with finer quantization if needed)
fn time_to_grid_position(
    time_sec: f32,
    tempo_analysis: &TempoMeterAnalysis,
    grid_slots_per_beat: usize,
) -> GridPosition {
    let beats_per_minute = tempo_analysis.bpm;
    let total_beats = time_sec * beats_per_minute / 60.0;

    let bar = (total_beats / 4.0) as usize; // Assuming 4/4 time
    let beat_in_bar = (total_beats % 4.0) as usize;
    let sub_beat_position = total_beats % 1.0;
    let sub_beat = (sub_beat_position * (grid_slots_per_beat as f32)) as usize;
    let ticks = ((sub_beat_position * (grid_slots_per_beat as f32) % 1.0) * 96.0) as usize; // MIDI ticks

    GridPosition {
        bar,
        beat: beat_in_bar,
        sub_beat,
        ticks,
    }
}

/// Convert grid position to time (for snapping if needed)
fn grid_position_to_time(grid_pos: &GridPosition, tempo_analysis: &TempoMeterAnalysis) -> f32 {
    let beats_per_minute = tempo_analysis.bpm;
    let total_beats =
        grid_pos.bar as f32 * 4.0 + grid_pos.beat as f32 + (grid_pos.sub_beat as f32 / 4.0); // Assuming 16th notes

    total_beats * 60.0 / beats_per_minute
}

/// Perform gap filling with neighbor velocity interpolation (enabled with improvements)
fn fill_gaps_with_neighbors(
    midi_events: &mut Vec<MidiEvent>,
    _refined_events: &[RefinedEvent], // Not used directly here
    tempo_analysis: &TempoMeterAnalysis,
    config: &GridInferenceConfig,
) {
    if midi_events.is_empty() {
        return;
    }

    // Sort events by time (already done in run, but ensure)
    midi_events.sort_by(|a, b| a.time_sec.partial_cmp(&b.time_sec).unwrap());

    let mut filled_events = Vec::new();
    let beat_duration = 60.0 / tempo_analysis.bpm;

    for i in 0..midi_events.len() - 1 {
        filled_events.push(midi_events[i].clone());

        let current_time = midi_events[i].time_sec;
        let next_time = midi_events[i + 1].time_sec;
        let time_gap = next_time - current_time;

        // If gap is more than 1.5 beats, insert ghost note at midpoint
        if time_gap > beat_duration * 1.5 {
            let ghost_time = current_time + time_gap / 2.0;
            let ghost_velocity = ((midi_events[i].velocity as f32 + midi_events[i + 1].velocity as f32) / 2.0) as u8;
            let ghost_velocity = ghost_velocity.max(config.min_ghost_velocity);

            let ghost_event = MidiEvent {
                time_sec: ghost_time,
                grid_position: time_to_grid_position(ghost_time, tempo_analysis, 4),
                drum_class: midi_events[i].drum_class, // Same as previous for simplicity
                velocity: ghost_velocity,
                confidence: 0.5, // Lower for inserted
                is_ghost_note: true,
                acoustic_score: 0.0,
                prior_score: 0.3,
                density_score: 0.0,
            };

            filled_events.push(ghost_event);
        }
    }

    // Add the last event
    if !midi_events.is_empty() {
        filled_events.push(midi_events.last().unwrap().clone());
    }

    *midi_events = filled_events;
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 8: Grid Inference + Fill/Silence Protection");

    // Validate inputs
    let refined_events = &state.refined_events;
    let priors = match state.self_priors.as_ref() {
        Some(p) => p,
        None => {
            return Err(DrumError::ProcessingPipelineError(
                "Self-prior matrices not available".to_string(),
            ))
        }
    };
    let tempo_analysis = match state.tempo_meter_analysis.as_ref() {
        Some(t) => t,
        None => {
            return Err(DrumError::ProcessingPipelineError(
                "Tempo/meter analysis not available".to_string(),
            ))
        }
    };

    if refined_events.is_empty() {
        // No events to process, return empty MIDI events
        state.midi_events = Vec::new();
        return Ok(());
    }

    // Compute spectral_max_energy from refined_events (assuming acoustic_confidence is the field)
    let spectral_max_energy = refined_events
        .iter()
        .map(|e| e.confidence) // Use confidence field
        .fold(0.0f32, f32::max)
        .max(1.0); // Avoid div by zero

    let grid_config = GridInferenceConfig::default(); // Use default since Config lacks fields

    // Compute density curve for fill/silence protection
    let density_curve = compute_density_curve(refined_events, tempo_analysis, config);
    let (fill_threshold, silence_threshold) =
        compute_density_thresholds(&density_curve, &grid_config);

    // Sliding window for local density
    let mut density_window: HashMap<u32, f32> = HashMap::new();
    let mut midi_events = Vec::new();
    let grid_slots_per_beat = priors.grid_slots_per_beat;

    // Process each refined event
    for event in refined_events {
        // Skip low-confidence events
        if event.confidence < grid_config.min_confidence {
            continue;
        }

        // Find the best grid position for this event
        let mut best_score = 0.0;
        let mut best_grid_pos =
            time_to_grid_position(event.refined_time_sec, tempo_analysis, grid_slots_per_beat);
        let mut best_acoustic_score = 0.0;
        let mut best_prior_score = 0.0;

        // Try nearby grid positions (bar fixed, vary beat/sub-beat)
        for beat_offset in -1..=1 {
            for sub_beat_offset in -1..=1 {
                let test_beat = ((best_grid_pos.beat as i32 + beat_offset).max(0) as usize).min(3); // 4/4 time
                let test_sub_beat = ((best_grid_pos.sub_beat as i32 + sub_beat_offset).max(0)
                    as usize)
                    .min(grid_slots_per_beat.saturating_sub(1));

                let test_grid_pos = GridPosition {
                    bar: best_grid_pos.bar,
                    beat: test_beat,
                    sub_beat: test_sub_beat,
                    ticks: best_grid_pos.ticks,
                };

                let test_time = grid_position_to_time(&test_grid_pos, tempo_analysis);

                // Compute scores
                let acoustic_score = compute_acoustic_score(event, test_time, &grid_config);
                let prior_score = compute_prior_probability(event.drum_class, &test_grid_pos, priors);
                let combined_score = grid_config.acoustic_weight * acoustic_score
                    + grid_config.prior_weight * prior_score;

                if combined_score > best_score {
                    best_score = combined_score;
                    best_grid_pos = test_grid_pos;
                    best_acoustic_score = acoustic_score;
                    best_prior_score = prior_score;
                }
            }
        }

        // Get local density for this event's beat
        let beat_idx = ((event.refined_time_sec * tempo_analysis.bpm) / 60.0) as usize;
        let global_density = if beat_idx < density_curve.len() {
            density_curve[beat_idx]
        } else {
            0.0
        };
        let local_density = density_window.get(&get_window_key(event.refined_time_sec)).copied().unwrap_or(0.0);

        let density_score = if global_density > fill_threshold {
            // High density - reduce fill probability
            0.2
        } else if global_density < silence_threshold {
            // Low density - allow ghost notes
            0.8
        } else {
            0.5
        };

        // Determine if ghost note
        let is_ghost = is_ghost_note(event, spectral_max_energy, local_density, &grid_config);

        // Compute velocity using best_acoustic_score (which is now confidence-based)
        let is_beat = is_on_beat(event.refined_time_sec, tempo_analysis, 0.05);
        let velocity = compute_velocity(best_acoustic_score, best_prior_score, density_score, is_beat, &grid_config);

        // Only create MIDI event if score is high enough and velocity meets min
        if best_score > 0.05 && velocity >= grid_config.min_ghost_velocity { // Lowered threshold
            // Use refined time_sec (not snapped, to preserve precision)
            let midi_event = MidiEvent {
                time_sec: event.refined_time_sec, // Preserve original timing
                grid_position: best_grid_pos,
                drum_class: event.drum_class,
                velocity,
                confidence: best_score,
                is_ghost_note: is_ghost,
                acoustic_score: best_acoustic_score, // Assuming MidiEvent has this; adjust if needed
                prior_score: best_prior_score,
                density_score,
            };

            midi_events.push(midi_event);
        }

        // Update density window
        update_density_window(&mut density_window, event.refined_time_sec, event.confidence);
    }

    // Apply gap filling to add missing soft events
    fill_gaps_with_neighbors(&mut midi_events, refined_events, tempo_analysis, &grid_config);

    // Sort final events by time
    midi_events.sort_by(|a, b| a.time_sec.partial_cmp(&b.time_sec).unwrap());

    state.midi_events = midi_events;

    Ok(())
}
