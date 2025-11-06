//! Pass 5: Class-Specific Timing Refinement

use crate::analysis::{
    ClassTimingStats, DrumClass, RefinedEvent, TimingStats,
};
use crate::audio::AudioState;
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};
use ndarray::Array2;

/// Class-specific frequency masking parameters
#[derive(Debug, Clone)]
struct FrequencyMask {
    fundamental_center_hz: Option<f32>,
    fundamental_width_hz: f32,
    secondary_bands_hz: Vec<(f32, f32)>, // (low, high) pairs
    weights: Vec<f32>,                   // Corresponding weights for each band
}

/// Get frequency mask for a specific drum class
fn get_frequency_mask(
    drum_class: DrumClass,
    tuning_info: &crate::analysis::TuningInfo,
) -> FrequencyMask {
    match drum_class {
        DrumClass::Kick => FrequencyMask {
            fundamental_center_hz: tuning_info.kick_hz,
            fundamental_width_hz: 40.0,
            secondary_bands_hz: vec![(2000.0, 4000.0)], // Click band
            weights: vec![1.0, 0.8],
        },
        DrumClass::Snare => FrequencyMask {
            fundamental_center_hz: Some(200.0), // Default snare body
            fundamental_width_hz: 70.0,
            secondary_bands_hz: vec![(2000.0, 5000.0)], // Snap band
            weights: vec![1.0, 0.9],
        },
        DrumClass::Tom => FrequencyMask {
            fundamental_center_hz: if !tuning_info.toms_hz.is_empty() {
                Some(tuning_info.toms_hz[0])
            } else {
                Some(120.0) // Default tom
            },
            fundamental_width_hz: 60.0,
            secondary_bands_hz: vec![], // No secondary bands for toms
            weights: vec![1.0],
        },
        DrumClass::HiHat => FrequencyMask {
            fundamental_center_hz: None, // Hi-hats are broadband
            fundamental_width_hz: 0.0,
            secondary_bands_hz: vec![(4000.0, 12000.0)], // High frequency broadband
            weights: vec![1.0],
        },
        DrumClass::Splash => FrequencyMask {
            fundamental_center_hz: None, // Splash is very high frequency
            fundamental_width_hz: 0.0,
            secondary_bands_hz: vec![
                (6000.0, 10000.0),  // Primary transient band
                (10000.0, 16000.0), // Extreme HF for sharp attack
            ],
            weights: vec![0.6, 0.4], // Emphasize primary transient band
        },
        DrumClass::Cowbell => FrequencyMask {
            fundamental_center_hz: Some(750.0), // Cowbell fundamental ~750Hz
            fundamental_width_hz: 200.0, // Narrower for precise transient detection
            secondary_bands_hz: vec![
                (1500.0, 3000.0),   // First harmonic transient
                (4000.0, 8000.0),   // Metallic transient overtones
            ],
            weights: vec![0.7, 0.5, 0.3], // Strong fundamental, moderate harmonics
        },
        DrumClass::Rimshot => FrequencyMask {
            fundamental_center_hz: Some(200.0), // Snare body component
            fundamental_width_hz: 50.0, // Very narrow for precise body detection
            secondary_bands_hz: vec![
                (3000.0, 6000.0),   // Rim snap transient
                (6000.0, 12000.0),  // High-frequency rim emphasis
            ],
            weights: vec![0.3, 0.7, 0.5], // De-emphasize body, emphasize rim transients
        },
        DrumClass::Cymbal => FrequencyMask {
            fundamental_center_hz: None, // Cymbals are broadband
            fundamental_width_hz: 0.0,
            secondary_bands_hz: vec![(2000.0, 8000.0)], // Broad cymbal range
            weights: vec![1.0],
        },
        DrumClass::Percussion => FrequencyMask {
            fundamental_center_hz: None,
            fundamental_width_hz: 0.0,
            secondary_bands_hz: vec![(1000.0, 4000.0)], // Mid-high percussion
            weights: vec![1.0],
        },
        DrumClass::Unknown => FrequencyMask {
            fundamental_center_hz: None,
            fundamental_width_hz: 0.0,
            secondary_bands_hz: vec![(100.0, 2000.0)], // Broad fallback
            weights: vec![1.0],
        },
    }
}

/// Compute tempo-adaptive search window
fn compute_tempo_adaptive_window(tempo_bpm: f32, base_window_ms: f32, max_window_ms: f32) -> f32 {
    // Scale window based on 16th note duration
    let sixteenth_note_sec = 60.0 / tempo_bpm / 4.0;
    let sixteenth_note_ms = sixteenth_note_sec * 1000.0;

    // Window is minimum of base window and 0.6 × 16th note duration
    (base_window_ms)
        .min(sixteenth_note_ms * 0.6)
        .min(max_window_ms)
}

/// Apply class-specific frequency masking to spectrogram
fn apply_frequency_mask(mag: &Array2<f32>, freqs: &[f32], mask: &FrequencyMask) -> Array2<f32> {
    let mut masked = Array2::<f32>::zeros(mag.raw_dim());

    // Apply fundamental frequency mask if present
    if let Some(center) = mask.fundamental_center_hz {
        let low = (center - mask.fundamental_width_hz / 2.0).max(0.0);
        let high = center + mask.fundamental_width_hz / 2.0;

        for i in 0..freqs.len() {
            if freqs[i] >= low && freqs[i] <= high {
                for t in 0..mag.shape()[1] {
                    masked[[i, t]] = mag[[i, t]] * mask.weights[0];
                }
            }
        }
    }

    // Apply secondary band masks
    for (band_idx, &(low, high)) in mask.secondary_bands_hz.iter().enumerate() {
        let weight_idx = if mask.fundamental_center_hz.is_some() {
            band_idx + 1
        } else {
            band_idx
        };

        if weight_idx >= mask.weights.len() {
            continue;
        }

        let weight = mask.weights[weight_idx];

        for i in 0..freqs.len() {
            if freqs[i] >= low && freqs[i] <= high {
                for t in 0..mag.shape()[1] {
                    masked[[i, t]] = mag[[i, t]] * weight;
                }
            }
        }
    }

    masked
}

/// Compute weighted onset strength using masked spectrogram
fn compute_weighted_onset_strength(masked_mag: &Array2<f32>) -> Vec<f32> {
    let mut onset_strength = vec![0.0; masked_mag.shape()[1]];

    for t in 1..masked_mag.shape()[1] {
        let mut frame_strength = 0.0;
        for f in 0..masked_mag.shape()[0] {
            let diff = masked_mag[[f, t]] - masked_mag[[f, t - 1]];
            if diff > 0.0 {
                frame_strength += diff;
            }
        }
        onset_strength[t] = frame_strength;
    }

    onset_strength
}

/// Find refined timing using SNR-guided peak selection
fn find_refined_timing(
    onset_strength: &[f32],
    original_frame: usize,
    search_window_frames: usize,
    min_snr_threshold: f32,
) -> Option<(usize, f32)> {
    let start = original_frame.saturating_sub(search_window_frames / 2);
    let end = (original_frame + search_window_frames / 2).min(onset_strength.len() - 1);

    if end <= start {
        return None;
    }

    // Find local peaks in the search window
    let mut best_peak = None;
    let mut best_snr = 0.0;

    for i in (start + 1)..end {
        // Check if this is a local maximum
        if onset_strength[i] > onset_strength[i - 1] && onset_strength[i] > onset_strength[i + 1] {
            // Compute SNR for this peak
            let window_size = 5.min((end - start) / 4);
            let snr_window_start = i.saturating_sub(window_size);
            let snr_window_end = (i + window_size).min(onset_strength.len() - 1);

            let mut surrounding_sum = 0.0;
            let mut count = 0;

            for j in snr_window_start..=snr_window_end {
                if j != i {
                    surrounding_sum += onset_strength[j];
                    count += 1;
                }
            }

            let noise_level = if count > 0 {
                surrounding_sum / count as f32
            } else {
                0.0
            };
            let snr = if noise_level > 0.0 {
                onset_strength[i] / noise_level
            } else {
                onset_strength[i] * 1000.0 // High SNR if no noise
            };

            // Update best peak if this has higher SNR and meets threshold
            if snr > min_snr_threshold && snr > best_snr {
                best_peak = Some(i);
                best_snr = snr;
            }
        }
    }

    best_peak.map(|frame| (frame, best_snr))
}

/// Apply drift limiting constraints
fn apply_drift_limits(
    original_time_sec: f32,
    refined_time_sec: f32,
    max_drift_ms: f32,
    tempo_bpm: f32,
) -> f32 {
    let drift_ms = (refined_time_sec - original_time_sec) * 1000.0;

    // Apply tempo-adaptive drift limiting
    let sixteenth_note_sec = 60.0 / tempo_bpm / 4.0;
    let tempo_adaptive_limit = (sixteenth_note_sec * 0.25 * 1000.0).min(max_drift_ms);

    let limited_drift = drift_ms
        .max(-tempo_adaptive_limit)
        .min(tempo_adaptive_limit);

    original_time_sec + limited_drift / 1000.0
}

/// Compute timing refinement statistics
fn compute_timing_stats(refined_events: &[RefinedEvent]) -> TimingStats {
    if refined_events.is_empty() {
        return TimingStats {
            total_events: 0,
            refined_events: 0,
            median_drift_ms: 0.0,
            mean_drift_ms: 0.0,
            max_drift_ms: 0.0,
            drift_std_ms: 0.0,
            events_within_5ms: 0,
            events_within_15ms: 0,
            events_within_30ms: 0,
            per_class_stats: std::collections::HashMap::new(),
        };
    }

    let drifts: Vec<f32> = refined_events.iter().map(|e| e.drift_ms.abs()).collect();

    // Sort for median calculation
    let mut sorted_drifts = drifts.clone();
    sorted_drifts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median_drift = sorted_drifts[sorted_drifts.len() / 2];
    let mean_drift = drifts.iter().sum::<f32>() / drifts.len() as f32;
    let max_drift = *drifts
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0);

    // Compute standard deviation
    let variance =
        drifts.iter().map(|d| (d - mean_drift).powi(2)).sum::<f32>() / drifts.len() as f32;
    let std_dev = variance.sqrt();

    // Count events within timing thresholds
    let within_5ms = drifts.iter().filter(|&&d| d <= 5.0).count();
    let within_15ms = drifts.iter().filter(|&&d| d <= 15.0).count();
    let within_30ms = drifts.iter().filter(|&&d| d <= 30.0).count();

    // Per-class statistics
    let mut per_class_stats = std::collections::HashMap::new();
    let mut class_drifts = std::collections::HashMap::new();

    for event in refined_events {
        class_drifts
            .entry(event.drum_class)
            .or_insert_with(Vec::new)
            .push(event.drift_ms.abs());
    }

    for (class, drifts) in class_drifts {
        if drifts.is_empty() {
            continue;
        }

        let mut sorted = drifts.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let class_median = sorted[sorted.len() / 2];
        let class_mean = drifts.iter().sum::<f32>() / drifts.len() as f32;
        let class_max = *drifts
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        let class_within_5ms = drifts.iter().filter(|&&d| d <= 5.0).count();
        let class_within_15ms = drifts.iter().filter(|&&d| d <= 15.0).count();

        per_class_stats.insert(
            class,
            ClassTimingStats {
                count: drifts.len(),
                median_drift_ms: class_median,
                mean_drift_ms: class_mean,
                max_drift_ms: class_max,
                events_within_5ms: class_within_5ms,
                events_within_15ms: class_within_15ms,
            },
        );
    }

    TimingStats {
        total_events: refined_events.len(),
        refined_events: refined_events.len(),
        median_drift_ms: median_drift,
        mean_drift_ms: mean_drift,
        max_drift_ms: max_drift,
        drift_std_ms: std_dev,
        events_within_5ms: within_5ms,
        events_within_15ms: within_15ms,
        events_within_30ms: within_30ms,
        per_class_stats,
    }
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 5: Class-Specific Timing Refinement");

    // Get classified events from Pass 4
    if state.classified_events.is_empty() {
        println!("  No classified events found from Pass 4, skipping timing refinement");
        return Ok(());
    }

    // Get whitened spectrogram from Pass 1
    let whitened = state.s_whitened.as_ref().ok_or_else(|| {
        DrumError::ProcessingPipelineError("Pass 1 must be run before Pass 5".to_string())
    })?;

    // Get primary STFT data
    let primary_stft = state
        .stfts
        .get(&(config.stft.n_fft, config.stft.hop_length))
        .ok_or_else(|| DrumError::ProcessingPipelineError("Primary STFT not found".to_string()))?;

    // Get tuning info from Pass 3
    let tuning_info = state.tuning_info.as_ref().ok_or_else(|| {
        DrumError::ProcessingPipelineError("Pass 3 must be run before Pass 5".to_string())
    })?;

    // Refine timing for each classified event
    println!(
        "  Refining timing for {} classified events...",
        state.classified_events.len()
    );

    let mut refined_events = Vec::new();
    let tempo_bpm = 120.0; // TODO: Get from analysis

    for event in &state.classified_events {
        // Get frequency mask for this drum class
        let mask = get_frequency_mask(event.drum_class, tuning_info);

        // Apply frequency masking
        let masked_mag = apply_frequency_mask(whitened, &primary_stft.freqs, &mask);

        // Compute weighted onset strength
        let onset_strength = compute_weighted_onset_strength(&masked_mag);

        // Compute tempo-adaptive search window
        let search_window_ms = compute_tempo_adaptive_window(
            tempo_bpm,
            config.timing_refinement.base_search_window_ms,
            config.timing_refinement.max_search_window_ms,
        );
        let search_window_frames = ((search_window_ms / 1000.0) * state.sr as f32
            / config.stft.hop_length as f32) as usize;

        // Find refined timing
        if let Some((refined_frame, snr)) = find_refined_timing(
            &onset_strength,
            event.frame_idx,
            search_window_frames,
            config.timing_refinement.min_snr_threshold,
        ) {
            // Convert refined frame to time
            let refined_time_sec =
                refined_frame as f32 * config.stft.hop_length as f32 / state.sr as f32;

            // Apply drift limiting
            let final_refined_time = apply_drift_limits(
                event.time_sec,
                refined_time_sec,
                config.timing_refinement.max_drift_ms,
                tempo_bpm,
            );

            let drift_ms = (final_refined_time - event.time_sec) * 1000.0;

            // Create refined event
            let refined_event = RefinedEvent {
                original_time_sec: event.time_sec,
                refined_time_sec: final_refined_time,
                frame_idx: event.frame_idx,
                refined_frame_idx: ((final_refined_time * state.sr as f32
                    / config.stft.hop_length as f32) as usize)
                    .min(whitened.shape()[1] - 1),
                drum_class: event.drum_class,
                confidence: event.confidence,
                timing_confidence: snr.min(1.0),
                drift_ms,
                snr_at_refined_time: snr,
                features: event.features.clone(),
            };

            refined_events.push(refined_event);
        } else {
            // If no refinement found, keep original timing
            let refined_event = RefinedEvent {
                original_time_sec: event.time_sec,
                refined_time_sec: event.time_sec,
                frame_idx: event.frame_idx,
                refined_frame_idx: event.frame_idx,
                drum_class: event.drum_class,
                confidence: event.confidence,
                timing_confidence: 0.0,
                drift_ms: 0.0,
                snr_at_refined_time: 0.0,
                features: event.features.clone(),
            };

            refined_events.push(refined_event);
        }
    }

    // Compute timing statistics
    let timing_stats = compute_timing_stats(&refined_events);

    // Print refinement summary
    println!("  ✓ Refined {} events", refined_events.len());
    println!("  Timing statistics:");
    println!("    Median drift: {:.1}ms", timing_stats.median_drift_ms);
    println!("    Mean drift: {:.1}ms", timing_stats.mean_drift_ms);
    println!("    Max drift: {:.1}ms", timing_stats.max_drift_ms);
    println!(
        "    Events within 5ms: {} ({:.1}%)",
        timing_stats.events_within_5ms,
        timing_stats.events_within_5ms as f32 / timing_stats.total_events as f32 * 100.0
    );
    println!(
        "    Events within 15ms: {} ({:.1}%)",
        timing_stats.events_within_15ms,
        timing_stats.events_within_15ms as f32 / timing_stats.total_events as f32 * 100.0
    );

    // Print per-class statistics
    println!("  Per-class timing statistics:");
    for (class, stats) in &timing_stats.per_class_stats {
        println!(
            "    {}: {} events, median drift {:.1}ms",
            class.name(),
            stats.count,
            stats.median_drift_ms
        );
    }

    // Store refined events in state
    state.refined_events = refined_events;

    println!("  ✓ Pass 5 complete");

    Ok(())
}
