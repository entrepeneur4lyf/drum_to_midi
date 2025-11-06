//! Pass 7: Self-Prior Construction

use crate::analysis::{ClassifiedEvent, DrumClass, PriorStats, SelfPriorMatrices};
use crate::audio::AudioState;
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};

/// Calculate grid position for an event within a bar
pub fn calculate_grid_position(
    event_time_sec: f32,
    tempo_bpm: f32,
    meter_beats_per_measure: usize,
    grid_slots_per_beat: usize,
    downbeat_time_sec: f32,
) -> Option<usize> {
    // Calculate time relative to downbeat
    let relative_time = event_time_sec - downbeat_time_sec;
    if relative_time < 0.0 {
        return None; // Event is before downbeat
    }

    // Calculate bar duration
    let bar_duration_sec = (60.0 / tempo_bpm) * meter_beats_per_measure as f32;

    if relative_time > bar_duration_sec {
        return None; // Event is after bar end
    }

    // Calculate position within bar (0.0 to 1.0)
    let bar_position = relative_time / bar_duration_sec;

    // Convert to grid slot
    let total_slots = meter_beats_per_measure * grid_slots_per_beat;
    let grid_slot = (bar_position * total_slots as f32).round() as usize;

    Some(grid_slot.min(total_slots - 1))
}

/// Accumulate event counts across bars for prior construction
pub fn accumulate_event_counts(
    events: &[ClassifiedEvent],
    tempo_bpm: f32,
    meter_beats_per_measure: usize,
    grid_slots_per_beat: usize,
    downbeat_positions: &[f32],
) -> std::collections::HashMap<DrumClass, Vec<f32>> {
    let mut class_counts = std::collections::HashMap::new();

    // Initialize count vectors for each drum class
    let total_slots = meter_beats_per_measure * grid_slots_per_beat;
    for drum_class in &[
        DrumClass::Kick,
        DrumClass::Snare,
        DrumClass::Rimshot,
        DrumClass::HiHat,
        DrumClass::Splash,
        DrumClass::Cowbell,
        DrumClass::Tom,
        DrumClass::Cymbal,
        DrumClass::Percussion,
    ] {
        class_counts.insert(*drum_class, vec![0.0; total_slots]);
    }

    // Process each event
    for event in events {
        // Find which bar this event belongs to
        let mut bar_index = 0;
        for (i, &downbeat) in downbeat_positions.iter().enumerate() {
            if event.time_sec >= downbeat {
                bar_index = i;
            } else {
                break;
            }
        }

        let downbeat_time = if bar_index < downbeat_positions.len() {
            downbeat_positions[bar_index]
        } else if !downbeat_positions.is_empty() {
            // Extrapolate for bars after the last detected downbeat
            let last_downbeat = *downbeat_positions.last().unwrap();
            let bar_duration = (60.0 / tempo_bpm) * meter_beats_per_measure as f32;
            last_downbeat + (bar_index - downbeat_positions.len() + 1) as f32 * bar_duration
        } else {
            0.0 // Default to start of track
        };

        // Calculate grid position
        if let Some(grid_slot) = calculate_grid_position(
            event.time_sec,
            tempo_bpm,
            meter_beats_per_measure,
            grid_slots_per_beat,
            downbeat_time,
        ) {
            if let Some(counts) = class_counts.get_mut(&event.drum_class) {
                if grid_slot < counts.len() {
                    counts[grid_slot] += 1.0;
                }
            }
        }
    }

    class_counts
}

/// Apply Gaussian smoothing to probability distributions
pub fn apply_gaussian_smoothing(
    counts: &[f32],
    sigma_beats: f32,
    grid_slots_per_beat: usize,
) -> Vec<f32> {
    let mut smoothed = vec![0.0; counts.len()];
    let sigma_slots = sigma_beats * grid_slots_per_beat as f32;

    // Gaussian kernel
    let kernel_size = (sigma_slots * 3.0).ceil() as usize * 2 + 1;
    let mut kernel = vec![0.0; kernel_size];
    let kernel_center = kernel_size / 2;

    for i in 0..kernel_size {
        let x = (i as f32 - kernel_center as f32) / sigma_slots;
        kernel[i] = (-0.5 * x * x).exp() / (sigma_slots * (2.0 * std::f32::consts::PI).sqrt());
    }

    // Normalize kernel
    let kernel_sum: f32 = kernel.iter().sum();
    for k in kernel.iter_mut() {
        *k /= kernel_sum;
    }

    // Apply convolution
    for i in 0..counts.len() {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for (k_idx, &kernel_val) in kernel.iter().enumerate() {
            let data_idx = i as isize + (k_idx as isize - kernel_center as isize);
            if data_idx >= 0 && data_idx < counts.len() as isize {
                sum += counts[data_idx as usize] * kernel_val;
                weight_sum += kernel_val;
            }
        }

        smoothed[i] = if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            counts[i]
        };
    }

    smoothed
}

/// Apply beta smoothing for confidence intervals on sparse data
pub fn apply_beta_smoothing(
    smoothed_counts: &[f32],
    alpha: f32,
    beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut priors = vec![0.0; smoothed_counts.len()];
    let mut confidences = vec![0.0; smoothed_counts.len()];

    // Find maximum count for normalization
    let max_count = smoothed_counts.iter().cloned().fold(0.0, f32::max);

    if max_count > 0.0 {
        for i in 0..smoothed_counts.len() {
            let normalized_count = smoothed_counts[i] / max_count;

            // Beta smoothing: posterior mean = (alpha + successes) / (alpha + beta + trials)
            // Here we treat normalized_count as the observed success rate
            let successes = normalized_count;
            let trials = 1.0; // Each position has one "trial"

            let posterior_mean = (alpha + successes) / (alpha + beta + trials);
            let _posterior_variance = (alpha + successes) * (beta + trials - successes)
                / ((alpha + beta + trials).powi(2) * (alpha + beta + trials + 1.0));

            priors[i] = posterior_mean;

            // Confidence based on evidence (more data = higher confidence)
            let evidence = successes.max(0.1); // At least some evidence
            confidences[i] = (evidence / (evidence + 1.0)).min(1.0);
        }
    } else {
        // No events found, use uniform prior with low confidence
        for i in 0..priors.len() {
            priors[i] = 1.0 / priors.len() as f32; // Uniform
            confidences[i] = 0.1; // Low confidence
        }
    }

    (priors, confidences)
}

/// Construct self-prior matrices for all drum classes
pub fn construct_self_priors(
    events: &[ClassifiedEvent],
    tempo_bpm: f32,
    meter_beats_per_measure: usize,
    downbeat_positions: &[f32],
    config: &Config,
) -> SelfPriorMatrices {
    let grid_slots_per_beat = config.self_prior.grid_slots_per_beat;

    // Accumulate event counts
    let mut class_counts = accumulate_event_counts(
        events,
        tempo_bpm,
        meter_beats_per_measure,
        grid_slots_per_beat,
        downbeat_positions,
    );

    // Apply smoothing and normalization for each class
    let mut class_priors = std::collections::HashMap::new();
    let mut class_confidences = std::collections::HashMap::new();
    let mut total_events_per_class = std::collections::HashMap::new();

    for (drum_class, counts) in class_counts.iter_mut() {
        let total_events = counts.iter().sum::<f32>();
        total_events_per_class.insert(*drum_class, total_events as usize);

        // Skip classes with insufficient events
        if total_events < config.self_prior.min_events_for_prior as f32 {
            let grid_size = meter_beats_per_measure * grid_slots_per_beat;
            class_priors.insert(*drum_class, vec![1.0 / grid_size as f32; grid_size]);
            class_confidences.insert(*drum_class, vec![0.1; grid_size]);
            continue;
        }

        // Get class-specific smoothing parameters
        let sigma_beats = config
            .self_prior
            .class_specific_sigmas
            .get(&drum_class.name().to_string())
            .copied()
            .unwrap_or(config.self_prior.smoothing_sigma_beats);

        // Apply Gaussian smoothing
        let smoothed = apply_gaussian_smoothing(counts, sigma_beats, grid_slots_per_beat);

        // Apply beta smoothing for confidence
        let (priors, confidences) = apply_beta_smoothing(
            &smoothed,
            config.self_prior.beta_smoothing_alpha,
            config.self_prior.beta_smoothing_beta,
        );

        // Normalize priors to sum to 1
        let prior_sum: f32 = priors.iter().sum();
        let normalized_priors: Vec<f32> = if prior_sum > 0.0 {
            priors.iter().map(|p| p / prior_sum).collect()
        } else {
            vec![1.0 / priors.len() as f32; priors.len()]
        };

        class_priors.insert(*drum_class, normalized_priors);
        class_confidences.insert(*drum_class, confidences);
    }

    SelfPriorMatrices {
        grid_slots_per_beat,
        class_priors,
        class_confidences,
        total_events_per_class,
        smoothing_sigma_beats: config.self_prior.smoothing_sigma_beats,
        beta_smoothing_alpha: config.self_prior.beta_smoothing_alpha,
        beta_smoothing_beta: config.self_prior.beta_smoothing_beta,
    }
}

/// Calculate prior construction statistics
pub fn calculate_prior_stats(
    priors: &SelfPriorMatrices,
    events: &[ClassifiedEvent],
    downbeat_positions: &[f32],
) -> PriorStats {
    let total_bars_analyzed = downbeat_positions.len().max(1);

    // Calculate events per bar
    let total_events = events.len();
    let events_per_bar_avg = total_events as f32 / total_bars_analyzed as f32;

    // Calculate class distribution
    let mut class_distribution = std::collections::HashMap::new();
    let mut class_counts = std::collections::HashMap::new();

    for event in events {
        *class_counts.entry(event.drum_class).or_insert(0) += 1;
    }

    for (drum_class, &count) in &class_counts {
        let percentage = count as f32 / total_events as f32;
        class_distribution.insert(*drum_class, percentage);
    }

    // Calculate grid coverage (percentage of grid positions with non-zero priors)
    let mut total_positions = 0;
    let mut covered_positions = 0;

    for priors_vec in priors.class_priors.values() {
        total_positions += priors_vec.len();
        covered_positions += priors_vec.iter().filter(|&&p| p > 0.01).count();
    }

    let grid_coverage_percent = if total_positions > 0 {
        (covered_positions as f32 / total_positions as f32) * 100.0
    } else {
        0.0
    };

    // Estimate smoothing effectiveness (rough heuristic)
    let smoothing_effectiveness = (grid_coverage_percent / 100.0).min(1.0);

    PriorStats {
        total_bars_analyzed,
        events_per_bar_avg,
        class_distribution,
        grid_coverage_percent,
        smoothing_effectiveness,
    }
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 7: Self-Prior Construction");

    // Get refined events from Pass 5
    if state.refined_events.is_empty() {
        println!("  No refined events found from Pass 5, skipping prior construction");
        return Ok(());
    }

    // Get tempo/meter analysis from Pass 6
    let tempo_meter = state.tempo_meter_analysis.as_ref().ok_or_else(|| {
        DrumError::ProcessingPipelineError("Pass 6 must be run before Pass 7".to_string())
    })?;

    // Convert refined events to classified events for prior construction
    let classified_events: Vec<ClassifiedEvent> = state
        .refined_events
        .iter()
        .map(|event| ClassifiedEvent {
            time_sec: event.refined_time_sec,
            frame_idx: event.refined_frame_idx,
            drum_class: event.drum_class,
            confidence: event.confidence,
            acoustic_confidence: event.confidence,
            prior_confidence: event.timing_confidence,
            features: event.features.clone(),
            alternative_classes: vec![],
        })
        .collect();

    if classified_events.len() < 4 {
        println!("  Insufficient events for prior construction, using defaults");
        return Ok(());
    }

    // Extract tempo and meter info
    let tempo_bpm = tempo_meter.bpm;
    let meter_beats_per_measure = match tempo_meter.meter.as_str() {
        "4/4" => 4,
        "3/4" => 3,
        "2/4" => 2,
        "5/4" => 5,
        "7/8" => 7,
        "12/8" => 12,
        _ => 4, // Default to 4/4
    };
    let downbeat_positions = &tempo_meter.downbeat_positions;

    println!(
        "  Constructing self-priors for {} events",
        classified_events.len()
    );
    println!(
        "  Using tempo: {:.1} BPM, meter: {}",
        tempo_bpm, tempo_meter.meter
    );

    // Construct self-prior matrices
    let self_priors = construct_self_priors(
        &classified_events,
        tempo_bpm,
        meter_beats_per_measure,
        downbeat_positions,
        config,
    );

    // Calculate statistics
    let prior_stats = calculate_prior_stats(&self_priors, &classified_events, downbeat_positions);

    // Print summary
    println!("  ✓ Self-prior construction complete");
    println!(
        "    Grid size: {} slots per beat",
        self_priors.grid_slots_per_beat
    );
    println!("    Bars analyzed: {}", prior_stats.total_bars_analyzed);
    println!("    Events per bar: {:.1}", prior_stats.events_per_bar_avg);
    println!(
        "    Grid coverage: {:.1}%",
        prior_stats.grid_coverage_percent
    );

    // Store results in state
    state.self_priors = Some(self_priors);
    state.prior_stats = Some(prior_stats);

    // Print class distribution
    println!("    Class distribution:");
    for (drum_class, &percentage) in &state.prior_stats.as_ref().unwrap().class_distribution {
        let event_count = state
            .self_priors
            .as_ref()
            .unwrap()
            .total_events_per_class
            .get(drum_class)
            .copied()
            .unwrap_or(0);
        println!(
            "      {}: {} events ({:.1}%)",
            drum_class.name(),
            event_count,
            percentage * 100.0
        );
    }

    Ok(())
}
