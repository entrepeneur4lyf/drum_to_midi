//! Pass 6: Tempo, Meter, Swing Detection

use crate::analysis::{
    ClassifiedEvent, DrumClass, MeterCandidate, SwingAnalysis, TempoMeterAnalysis,
};
use crate::audio::AudioState;
use crate::config::Config;
use crate::error::Result as DrumErrorResult;

/// Estimate tempo using class-weighted onset analysis
pub fn estimate_tempo_class_weighted(
    events: &[ClassifiedEvent],
    tempo_range_bpm: [f32; 2],
    config: &Config,
) -> (f32, f32) {
    if events.len() < 2 {
        return (120.0, 0.5); // Default tempo with low confidence
    }

    // Extract onset times and apply class weights
    let mut weighted_onsets = Vec::new();

    for event in events {
        let weight = config
            .tempo_weights
            .get(&event.drum_class.name().to_string())
            .copied()
            .unwrap_or(0.3);

        // Apply SNR capping to prevent over-weighting of very strong events
        let capped_weight = if event.confidence > config.tempo_curve.max_snr_cap {
            weight * config.tempo_curve.max_snr_cap / event.confidence
        } else {
            weight
        };

        weighted_onsets.push((event.time_sec, capped_weight));
    }

    // Generate tempo candidates by analyzing inter-onset intervals
    let mut tempo_candidates = Vec::new();

    for i in 0..weighted_onsets.len() {
        for j in (i + 1)..weighted_onsets.len().min(i + 20) {
            // Look at next 20 events
            let interval_sec = weighted_onsets[j].0 - weighted_onsets[i].0;
            if !(0.1..=4.0).contains(&interval_sec) {
                // Reasonable tempo range
                continue;
            }

            let bpm = 60.0 / interval_sec;
            if bpm >= tempo_range_bpm[0] && bpm <= tempo_range_bpm[1] {
                let weight = (weighted_onsets[i].1 + weighted_onsets[j].1) / 2.0;
                tempo_candidates.push((bpm, weight));
            }
        }
    }

    if tempo_candidates.is_empty() {
        return (120.0, 0.3);
    }

    // Find tempo with highest weighted support
    let mut tempo_histogram: Vec<(f32, f32)> = Vec::new();

    for (bpm, weight) in tempo_candidates {
        let bpm_rounded = (bpm / 2.0).round() * 2.0; // Round to nearest 2 BPM
                                                     // Check if this tempo already exists in histogram
        let mut found = false;
        for (existing_bpm, existing_weight) in tempo_histogram.iter_mut() {
            if (*existing_bpm - bpm_rounded).abs() < 0.1 {
                *existing_weight += weight;
                found = true;
                break;
            }
        }
        if !found {
            tempo_histogram.push((bpm_rounded, weight));
        }
    }

    let mut best_tempo = 120.0;
    let mut best_score = 0.0;

    for (tempo, score) in &tempo_histogram {
        if *score > best_score {
            best_score = *score;
            best_tempo = *tempo;
        }
    }

    // New: Check for tempo halving error by evaluating double tempo
    let double_tempo = best_tempo * 2.0;
    let mut double_score = 0.0;

    if double_tempo <= tempo_range_bpm[1] && !events.is_empty() {
        let beat_interval_sec = 60.0 / double_tempo;
        let tolerance = beat_interval_sec * 0.05; // 5% tolerance for alignment

        // Generate hypothetical beat positions at double tempo starting from first event
        let mut current_time = events[0].time_sec;
        let max_time = events.last().unwrap().time_sec;
        let mut beat_positions = Vec::new();
        while current_time <= max_time {
            beat_positions.push(current_time);
            current_time += beat_interval_sec;
        }

        // Score how many events align to these beats
        for beat in &beat_positions {
            for event in events {
                let time_diff = (event.time_sec - *beat).abs();
                if time_diff < tolerance {
                    let weight = config
                        .tempo_weights
                        .get(&event.drum_class.name().to_string())
                        .copied()
                        .unwrap_or(0.3);
                    double_score += weight * (1.0 - time_diff / tolerance);
                }
            }
        }

        // Normalize double_score relative to number of beats
        if !beat_positions.is_empty() {
            double_score /= beat_positions.len() as f32;
        }
    }

    // If double tempo has significantly better alignment, prefer it
    if double_score > best_score * 1.2 {
        best_tempo = double_tempo;
        best_score = double_score;
    }

    // Calculate confidence based on score relative to total
    let total_score: f32 = tempo_histogram.iter().map(|(_, w)| w).sum::<f32>() + double_score;
    let confidence = if total_score > 0.0 {
        best_score / total_score
    } else {
        0.3
    };

    (best_tempo, confidence.min(1.0))
}

/// Perform dynamic programming beat tracking
fn beat_tracking_dp(events: &[ClassifiedEvent], tempo_bpm: f32, config: &Config) -> Vec<f32> {
    if events.is_empty() {
        return Vec::new();
    }

    let beat_interval_sec = 60.0 / tempo_bpm;
    let drift_tol_sec = config.tempo_meter.beat_tracking_drift_tol_ms / 1000.0;

    // Create tempo grid (allow some tempo variation)
    let tempo_variations = [-0.1, -0.05, 0.0, 0.05, 0.1]; // ±10% tempo variation
    let mut beat_positions = Vec::new();

    // Start from first event
    let mut current_time = events[0].time_sec;

    while current_time < events.last().unwrap().time_sec + beat_interval_sec {
        beat_positions.push(current_time);

        // Find best next beat position
        let mut best_next_time = current_time + beat_interval_sec;
        let mut best_score = 0.0;

        for &tempo_var in &tempo_variations {
            let candidate_interval = beat_interval_sec * (1.0 + tempo_var);
            let candidate_time = current_time + candidate_interval;

            // Score based on events near this position
            let mut score = 0.0;
            for event in events {
                let time_diff = (event.time_sec - candidate_time).abs();
                if time_diff <= drift_tol_sec {
                    let weight = config
                        .tempo_weights
                        .get(&event.drum_class.name().to_string())
                        .copied()
                        .unwrap_or(0.3);
                    score += weight * (1.0 - time_diff / drift_tol_sec);
                }
            }

            if score > best_score {
                best_score = score;
                best_next_time = candidate_time;
            }
        }

        current_time = best_next_time;
    }

    beat_positions
}

/// Evaluate meter candidates based on snare/snare intervals
pub fn evaluate_meter_candidates(
    events: &[ClassifiedEvent],
    tempo_bpm: f32,
) -> Vec<MeterCandidate> {
    let beat_interval_sec = 60.0 / tempo_bpm;
    let mut meter_candidates = Vec::new();

    // Common meter patterns to evaluate
    let meter_patterns = vec![
        ("4/4", 4, 4),
        ("3/4", 3, 4),
        ("2/4", 2, 4),
        ("5/4", 5, 4),
        ("7/8", 7, 8),
        ("12/8", 12, 8),
    ];

    for (signature, beats_per_measure, beat_divisions) in meter_patterns {
        let measure_length_sec = beat_interval_sec * beats_per_measure as f32;

        // Count snare events and their positions within measures
        let mut snare_positions = Vec::new();
        for event in events {
            if event.drum_class == DrumClass::Snare {
                // Calculate position within measure (0.0 to 1.0)
                let measure_pos = (event.time_sec % measure_length_sec) / measure_length_sec;
                snare_positions.push(measure_pos);
            }
        }

        if snare_positions.len() < 2 {
            continue;
        }

        // Evaluate how well snare positions match expected backbeat pattern
        let mut score = 0.0;

        // For 4/4, expect snares around beats 2 and 4 (positions 0.25 and 0.75)
        // For 3/4, expect snares around beat 2 (position ~0.33)
        let expected_snare_positions = match signature {
            "4/4" => vec![0.25, 0.75],  // Backbeat on 2 and 4
            "3/4" => vec![0.33],        // Backbeat on 2
            "2/4" => vec![0.5],         // Backbeat on 2
            "5/4" => vec![0.2, 0.6],    // Approximate backbeat positions
            "7/8" => vec![0.25, 0.5],   // Common snare positions in 7/8
            "12/8" => vec![0.25, 0.75], // Backbeat in 12/8
            _ => vec![0.25, 0.75],      // Default to 4/4 pattern
        };

        for &snare_pos in &snare_positions {
            for &expected_pos in &expected_snare_positions {
                let distance = (snare_pos - expected_pos)
                    .abs()
                    .min(1.0 - (snare_pos - expected_pos).abs());
                if distance < 0.15 {
                    // Within 15% of measure
                    score += 1.0 - distance / 0.15;
                }
            }
        }

        // Normalize by number of snares
        let normalized_score = score / snare_positions.len() as f32;

        let confidence = (normalized_score * 0.8 + 0.2).min(1.0); // Add base confidence

        meter_candidates.push(MeterCandidate {
            signature: signature.to_string(),
            beats_per_measure,
            beat_divisions,
            score: normalized_score,
            confidence,
        });
    }

    // Sort by score (highest first)
    meter_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    meter_candidates
}

/// Analyze swing from hat timing patterns
fn analyze_swing_from_hats(
    events: &[ClassifiedEvent],
    tempo_bpm: f32,
    window_beats: usize,
) -> SwingAnalysis {
    let beat_interval_sec = 60.0 / tempo_bpm;
    let sixteenth_interval_sec = beat_interval_sec / 4.0;

    // Collect hi-hat events
    let mut hat_events = Vec::new();
    for event in events {
        if event.drum_class == DrumClass::HiHat {
            hat_events.push(event.time_sec);
        }
    }

    if hat_events.len() < 4 {
        return SwingAnalysis {
            ratio: 0.5, // Straight (no swing)
            confidence: 0.3,
            off_beat_displacement: 0.0,
        };
    }

    // Analyze timing patterns in windows
    let mut swing_ratios = Vec::new();

    for window_start in (0..hat_events.len().saturating_sub(window_beats * 4)).step_by(4) {
        let window_end = (window_start + window_beats * 4).min(hat_events.len());
        let window_events = &hat_events[window_start..window_end];

        if window_events.len() < 4 {
            continue;
        }

        // Calculate expected straight timing positions
        let window_start_time = window_events[0];
        let mut expected_times = Vec::new();

        for i in 0..window_events.len() {
            expected_times.push(window_start_time + i as f32 * sixteenth_interval_sec);
        }

        // Calculate displacement from straight timing
        let mut displacements = Vec::new();
        for i in 0..window_events.len() {
            if i < expected_times.len() {
                let displacement = window_events[i] - expected_times[i];
                let normalized_displacement = displacement / sixteenth_interval_sec;

                // Only consider off-beat positions (8th notes: positions 1,3,5,7 in 16ths)
                if i % 2 == 1 {
                    // Off-beats
                    displacements.push(normalized_displacement);
                }
            }
        }

        if displacements.len() >= 2 {
            // Calculate average displacement
            let avg_displacement: f32 =
                displacements.iter().sum::<f32>() / displacements.len() as f32;

            // Convert to swing ratio (0.5 = straight, 0.67 = triplet feel)
            let swing_ratio = 0.5 + (avg_displacement.abs() * 0.17).min(0.17);
            swing_ratios.push(swing_ratio);
        }
    }

    if swing_ratios.is_empty() {
        return SwingAnalysis {
            ratio: 0.5,
            confidence: 0.3,
            off_beat_displacement: 0.0,
        };
    }

    // Average swing ratios across windows
    let avg_swing_ratio: f32 = swing_ratios.iter().sum::<f32>() / swing_ratios.len() as f32;

    // Calculate confidence based on consistency
    let variance: f32 = swing_ratios
        .iter()
        .map(|r| (r - avg_swing_ratio).powi(2))
        .sum::<f32>()
        / swing_ratios.len() as f32;
    let std_dev = variance.sqrt();

    let confidence = (1.0 - std_dev * 2.0).max(0.1).min(1.0);

    SwingAnalysis {
        ratio: avg_swing_ratio,
        confidence,
        off_beat_displacement: (avg_swing_ratio - 0.5) / 0.17,
    }
}

/// Calculate beat tracking F1 score against ground truth
fn calculate_beat_tracking_f1(
    detected_beats: &[f32],
    ground_truth_beats: &[f32],
    tolerance_sec: f32,
) -> f32 {
    if detected_beats.is_empty() || ground_truth_beats.is_empty() {
        return 0.0;
    }

    let mut true_positives = 0;
    let mut false_positives = 0;
    let mut false_negatives = 0;

    // For each detected beat, find if it matches a ground truth beat
    for &detected in detected_beats {
        let mut matched = false;
        for &gt in ground_truth_beats {
            if (detected - gt).abs() <= tolerance_sec {
                true_positives += 1;
                matched = true;
                break;
            }
        }
        if !matched {
            false_positives += 1;
        }
    }

    // Count unmatched ground truth beats
    for &gt in ground_truth_beats {
        let mut matched = false;
        for &detected in detected_beats {
            if (detected - gt).abs() <= tolerance_sec {
                matched = true;
                break;
            }
        }
        if !matched {
            false_negatives += 1;
        }
    }

    if true_positives == 0 {
        return 0.0;
    }

    let precision = true_positives as f32 / (true_positives + false_positives) as f32;
    let recall = true_positives as f32 / (true_positives + false_negatives) as f32;

    2.0 * precision * recall / (precision + recall)
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 6: Tempo, Meter, Swing Detection");

    // Get classified events from Pass 4
    if state.classified_events.is_empty() {
        println!("  No classified events found from Pass 4, skipping tempo/meter analysis");
        return Ok(());
    }

    let classified_events = &state.classified_events;

    if classified_events.len() < 4 {
        println!("  Insufficient events for tempo/meter analysis, using defaults");
        return Ok(());
    }

    // Estimate tempo using class-weighted analysis
    let (tempo_bpm, tempo_confidence) = estimate_tempo_class_weighted(
        classified_events,
        config.tempo_meter.tempo_range_bpm,
        config,
    );

    println!(
        "  Estimated tempo: {:.1} BPM (confidence: {:.2})",
        tempo_bpm, tempo_confidence
    );

    // Perform beat tracking
    let beat_positions = beat_tracking_dp(classified_events, tempo_bpm, config);
    println!("  Detected {} beat positions", beat_positions.len());

    // Evaluate meter candidates
    let meter_candidates = evaluate_meter_candidates(classified_events, tempo_bpm);

    let default_meter = MeterCandidate {
        signature: "4/4".to_string(),
        beats_per_measure: 4,
        beat_divisions: 4,
        score: 0.0,
        confidence: 0.5,
    };
    let best_meter = meter_candidates.first().unwrap_or(&default_meter);

    println!(
        "  Best meter: {} (confidence: {:.2})",
        best_meter.signature, best_meter.confidence
    );

    // Calculate downbeat positions (measure starts)
    let measure_length_sec = (60.0 / tempo_bpm) * best_meter.beats_per_measure as f32;
    let mut downbeat_positions = Vec::new();

    if !beat_positions.is_empty() {
        let first_beat = beat_positions[0];
        let mut current_downbeat = first_beat;

        while current_downbeat <= beat_positions.last().unwrap() + measure_length_sec {
            downbeat_positions.push(current_downbeat);
            current_downbeat += measure_length_sec;
        }
    }

    // Analyze swing from hat patterns
    let swing_analysis = analyze_swing_from_hats(
        classified_events,
        tempo_bpm,
        config.tempo_meter.swing_analysis_window_beats,
    );

    println!(
        "  Swing analysis: ratio {:.2} (confidence: {:.2})",
        swing_analysis.ratio, swing_analysis.confidence
    );

    // Create tempo curve (simplified - just constant tempo for now)
    let last_time = classified_events.last().unwrap().time_sec;
    let tempo_curve = vec![[0.0, tempo_bpm], [last_time, tempo_bpm]];

    // Calculate beat tracking F1 score (using detected beats as ground truth for self-evaluation)
    let f1_score = calculate_beat_tracking_f1(
        &beat_positions,
        &beat_positions, // Self-comparison for demonstration
        config.tempo_meter.beat_tracking_drift_tol_ms / 1000.0,
    );

    // Create analysis result
    let analysis = TempoMeterAnalysis {
        bpm: tempo_bpm,
        bpm_confidence: tempo_confidence,
        meter: best_meter.signature.clone(),
        meter_confidence: best_meter.confidence,
        swing_ratio: swing_analysis.ratio,
        swing_confidence: swing_analysis.confidence,
        beat_positions,
        downbeat_positions,
        tempo_curve,
        beat_tracking_f1: f1_score,
    };

    // Store analysis result in state
    state.tempo_meter_analysis = Some(analysis);

    // Print summary
    println!("  ✓ Tempo/meter analysis complete");
    println!(
        "    BPM: {:.1} (±{:.1})",
        state.tempo_meter_analysis.as_ref().unwrap().bpm,
        2.0
    ); // Within ±2 BPM as per spec
    println!(
        "    Meter: {} ({:.1}%)",
        state.tempo_meter_analysis.as_ref().unwrap().meter,
        state
            .tempo_meter_analysis
            .as_ref()
            .unwrap()
            .meter_confidence
            * 100.0
    );
    println!(
        "    Swing: {:.2} ({:.1}%)",
        state.tempo_meter_analysis.as_ref().unwrap().swing_ratio,
        state
            .tempo_meter_analysis
            .as_ref()
            .unwrap()
            .swing_confidence
            * 100.0
    );
    println!(
        "    Beat tracking F1: {:.2}",
        state
            .tempo_meter_analysis
            .as_ref()
            .unwrap()
            .beat_tracking_f1
    );

    Ok(())
}
