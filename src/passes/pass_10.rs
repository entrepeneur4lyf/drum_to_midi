//! Pass 10: Post-Processing & Export

use crate::analysis::{DrumClass, MidiEvent};
use crate::audio::AudioState;
use crate::config::Config;
use crate::DrumErrorResult;
use std::collections::HashMap;

/// Simple k-d tree implementation for spatial queries
#[derive(Debug)]
struct KdTree {
    points: Vec<MidiEvent>,
    indices: Vec<usize>,
}

impl KdTree {
    fn new(events: &[MidiEvent]) -> Self {
        let mut indices: Vec<usize> = (0..events.len()).collect();
        let points = events.to_vec();

        // Build k-d tree by recursively partitioning along time axis
        Self::build(&mut indices, &points, 0, 0, events.len());

        KdTree { points, indices }
    }

    fn build(indices: &mut [usize], points: &[MidiEvent], depth: usize, start: usize, end: usize) {
        if end - start <= 1 {
            return;
        }

        let axis = depth % 2; // Alternate between time (0) and confidence (1)
        let mid = (start + end) / 2;

        // Sort by the current axis
        indices[start..end].sort_by(|&a, &b| {
            let val_a = if axis == 0 {
                points[a].time_sec
            } else {
                points[a].confidence
            };
            let val_b = if axis == 0 {
                points[b].time_sec
            } else {
                points[b].confidence
            };
            val_a.partial_cmp(&val_b).unwrap()
        });

        // Recursively build subtrees
        Self::build(indices, points, depth + 1, start, mid);
        Self::build(indices, points, depth + 1, mid, end);
    }

    /// Find all points within a given radius of a query point
    fn range_query(
        &self,
        query_time: f32,
        query_confidence: f32,
        radius_time: f32,
        radius_conf: f32,
    ) -> Vec<usize> {
        let mut results = Vec::new();
        self.range_query_recursive(
            &self.indices,
            0,
            0,
            self.indices.len(),
            query_time,
            query_confidence,
            radius_time,
            radius_conf,
            &mut results,
        );
        results
    }

    fn range_query_recursive(
        &self,
        indices: &[usize],
        depth: usize,
        start: usize,
        end: usize,
        query_time: f32,
        query_confidence: f32,
        radius_time: f32,
        radius_conf: f32,
        results: &mut Vec<usize>,
    ) {
        if start >= end {
            return;
        }

        let axis = depth % 2;
        let mid_idx = (start + end) / 2;
        let point_idx = indices[mid_idx];
        let point = &self.points[point_idx];

        let point_val = if axis == 0 {
            point.time_sec
        } else {
            point.confidence
        };
        let query_val = if axis == 0 {
            query_time
        } else {
            query_confidence
        };

        // Check if current point is within range
        let time_diff = (point.time_sec - query_time).abs();
        let conf_diff = (point.confidence - query_confidence).abs();

        if time_diff <= radius_time && conf_diff <= radius_conf {
            results.push(point_idx);
        }

        // Decide which subtree to search first
        let diff = query_val - point_val;
        let (first_start, first_end, second_start, second_end) = if diff <= 0.0 {
            (start, mid_idx, mid_idx, end)
        } else {
            (mid_idx, end, start, mid_idx)
        };

        // Search first subtree
        self.range_query_recursive(
            indices,
            depth + 1,
            first_start,
            first_end,
            query_time,
            query_confidence,
            radius_time,
            radius_conf,
            results,
        );

        // Search second subtree only if necessary (pruning)
        let radius_val = if axis == 0 { radius_time } else { radius_conf };
        if diff.abs() <= radius_val {
            self.range_query_recursive(
                indices,
                depth + 1,
                second_start,
                second_end,
                query_time,
                query_confidence,
                radius_time,
                radius_conf,
                results,
            );
        }
    }
}

/// Event deduplication result
#[derive(Debug)]
struct DeduplicationResult {
    kept_events: Vec<MidiEvent>,
    removed_count: usize,
}

/// Quantization result
#[derive(Debug)]
struct QuantizationResult {
    quantized_events: Vec<MidiEvent>,
    max_drift_ms: f32,
}

/// Perform event deduplication with confidence-based selection
fn deduplicate_events(events: &[MidiEvent]) -> DeduplicationResult {
    let mut kept_events = Vec::new();
    let mut removed_count = 0;

    // Sort events by time for processing
    let mut sorted_events: Vec<(usize, &MidiEvent)> = events.iter().enumerate().collect();
    sorted_events.sort_by(|a, b| a.1.time_sec.partial_cmp(&b.1.time_sec).unwrap());

    let mut i = 0;
    while i < sorted_events.len() {
        let (current_idx, current_event) = sorted_events[i];
        let mut best_event_idx = current_idx;
        let mut max_confidence = current_event.confidence;

        // Look ahead for overlapping events
        let mut j = i + 1;
        while j < sorted_events.len() {
            let (_, next_event) = sorted_events[j];
            let time_diff = next_event.time_sec - current_event.time_sec;

            // Check if events are close enough to be considered duplicates
            let time_window = if current_event.drum_class == next_event.drum_class {
                0.002 // 2ms for same class
            } else {
                0.012 // 12ms for different classes
            };

            if time_diff > time_window {
                break; // No more overlapping events
            }

            // Keep the event with higher confidence
            if next_event.confidence > max_confidence {
                max_confidence = next_event.confidence;
                best_event_idx = sorted_events[j].0;
            }

            removed_count += 1;
            j += 1;
        }

        // Add the best event from this group
        kept_events.push(events[best_event_idx].clone());
        i = j; // Skip the processed group
    }

    DeduplicationResult {
        kept_events,
        removed_count,
    }
}

/// Apply masking based on acoustic properties using k-d tree for spatial queries
fn apply_acoustic_masking(events: &mut Vec<MidiEvent>, config: &Config) {
    if events.is_empty() {
        return;
    }

    // Build k-d tree for efficient spatial queries
    let kd_tree = KdTree::new(events);

    let mut to_remove = Vec::new();

    // Process each event and find potential conflicts using spatial queries
    for (i, event) in events.iter().enumerate() {
        if to_remove.contains(&i) {
            continue; // Already marked for removal
        }

        // Query for events within a small time window and confidence range
        let time_radius = 0.008; // 8ms time window
        let conf_radius = 0.3; // Confidence range

        let nearby_indices =
            kd_tree.range_query(event.time_sec, event.confidence, time_radius, conf_radius);

        // Check for snare/hat conflicts among nearby events
        for &nearby_idx in &nearby_indices {
            if nearby_idx == i || to_remove.contains(&nearby_idx) {
                continue;
            }

            let nearby_event = &events[nearby_idx];

            // Check for snare/hat conflicts
            let is_snare_hat_conflict = (event.drum_class == DrumClass::Snare
                && matches!(nearby_event.drum_class, DrumClass::HiHat))
                || (nearby_event.drum_class == DrumClass::Snare
                    && matches!(event.drum_class, DrumClass::HiHat));

            if is_snare_hat_conflict {
                // Use acoustic escape thresholds for masking decision
                let acoustic_escape = config.postprocess.acoustic_escape_threshold;

                // Prefer higher confidence event, or snare over hat in ties
                let should_remove_current = if event.confidence > nearby_event.confidence {
                    false // Keep current, remove nearby
                } else if nearby_event.confidence > event.confidence {
                    true // Remove current, keep nearby
                } else {
                    // Same confidence - prefer snare over hat
                    matches!(event.drum_class, DrumClass::HiHat)
                };

                if should_remove_current && event.confidence < acoustic_escape {
                    to_remove.push(i);
                    break; // Current event is being removed, no need to check more conflicts
                } else if !should_remove_current && nearby_event.confidence < acoustic_escape {
                    to_remove.push(nearby_idx);
                }
            }
        }
    }

    // Remove marked events (process in reverse order to maintain indices)
    to_remove.sort_unstable_by(|a, b| b.cmp(a));
    to_remove.dedup();

    for &idx in &to_remove {
        events.remove(idx);
    }

    println!(
        "Acoustic masking removed {} conflicting events using k-d tree",
        to_remove.len()
    );
}

/// Apply quantization with timing preservation
fn quantize_events(
    events: &[MidiEvent],
    tempo_analysis: &crate::analysis::TempoMeterAnalysis,
    config: &Config,
) -> QuantizationResult {
    let mut quantized_events = Vec::new();
    let mut max_drift_ms: f32 = 0.0;

    for event in events {
        let mut quantized_event = event.clone();

        // Calculate quantized time
        let beats_per_minute = tempo_analysis.bpm;
        let total_beats = event.time_sec * beats_per_minute / 60.0;

        // Quantize to 16th notes (4 slots per beat)
        let quantized_beats = (total_beats * 4.0).round() / 4.0;
        let quantized_time_sec = quantized_beats * 60.0 / beats_per_minute;

        // Apply strength-based quantization
        let original_time = event.time_sec;
        let quantized_time = quantized_time_sec;
        let drift_ms = (quantized_time - original_time).abs() * 1000.0;

        // Limit maximum drift
        let max_drift_limit = config.quantize.max_ms;
        let final_time = if drift_ms > max_drift_limit {
            // Preserve original timing if drift would be too large
            original_time
        } else {
            // Apply quantization with strength
            original_time + (quantized_time - original_time) * config.quantize.strength
        };

        quantized_event.time_sec = final_time;
        max_drift_ms = max_drift_ms.max(drift_ms);

        quantized_events.push(quantized_event);
    }

    QuantizationResult {
        quantized_events,
        max_drift_ms,
    }
}

/// Generate JSON bundle with complete analysis data
fn generate_json_bundle(state: &AudioState, output_dir: &std::path::Path) -> DrumErrorResult<()> {
    use serde_json;

    let json_path = output_dir.join("analysis.json");

    // Create comprehensive analysis bundle
    let analysis_bundle = serde_json::json!({
        "version": "1.1",
        "input_file": "processed_audio", // Would be actual filename in full implementation
        "processing_stats": {
            "total_events": state.midi_events.len(),
            "tempo_bpm": state.tempo_meter_analysis.as_ref().map(|t| t.bpm).unwrap_or(120.0),
            "meter": state.tempo_meter_analysis.as_ref().map(|t| t.meter.clone()).unwrap_or("4/4".to_string()),
            "swing_ratio": state.tempo_meter_analysis.as_ref().map(|t| t.swing_ratio).unwrap_or(0.5),
        },
        "midi_events": state.midi_events.iter().map(|event| {
            serde_json::json!({
                "time_sec": event.time_sec,
                "drum_class": event.drum_class.name(),
                "midi_note": event.drum_class.midi_note(),
                "velocity": event.velocity,
                "confidence": event.confidence,
                "grid_position": {
                    "bar": event.grid_position.bar,
                    "beat": event.grid_position.beat,
                    "sub_beat": event.grid_position.sub_beat,
                    "ticks": event.grid_position.ticks,
                },
                "is_ghost_note": event.is_ghost_note,
                "acoustic_score": event.acoustic_score,
                "prior_score": event.prior_score,
                "density_score": event.density_score,
            })
        }).collect::<Vec<_>>(),
        "analysis_metadata": {
            "tuning_info": state.tuning_info.as_ref().map(|t| {
                serde_json::json!({
                    "kick_hz": t.kick_hz,
                    "kick_confidence": t.kick_confidence,
                    "tom_frequencies": t.toms_hz,
                    "snare_body_hz": t.snare_body_hz,
                })
            }),
            "reverb_info": state.reverb_info.as_ref().map(|r| {
                serde_json::json!({
                    "rt60_ms": r.rt60_estimate_ms,
                    "strength": r.strength,
                })
            }),
            "prior_stats": state.prior_stats.as_ref().map(|p| {
                serde_json::json!({
                    "total_bars": p.total_bars_analyzed,
                    "events_per_bar_avg": p.events_per_bar_avg,
                    "grid_coverage_percent": p.grid_coverage_percent,
                })
            }),
        }
    });

    // Write JSON file
    let json_string = serde_json::to_string_pretty(&analysis_bundle)?;
    std::fs::write(json_path, json_string)?;

    Ok(())
}

/// Generate basic QA statistics
fn generate_qa_statistics(state: &AudioState, output_dir: &std::path::Path) -> DrumErrorResult<()> {
    let stats_path = output_dir.join("qa_stats.txt");

    let mut stats = String::new();
    stats.push_str("Drum-to-MIDI Processing Statistics\n");
    stats.push_str("==================================\n\n");

    stats.push_str(&format!("Total MIDI events: {}\n", state.midi_events.len()));

    // Count events by class
    let mut class_counts = HashMap::new();
    for event in &state.midi_events {
        *class_counts.entry(event.drum_class).or_insert(0) += 1;
    }

    stats.push_str("Events by drum class:\n");
    for (class, count) in class_counts {
        stats.push_str(&format!("  {}: {}\n", class.name(), count));
    }

    // Velocity statistics
    if !state.midi_events.is_empty() {
        let velocities: Vec<u8> = state.midi_events.iter().map(|e| e.velocity).collect();
        let avg_velocity =
            velocities.iter().map(|&v| v as f32).sum::<f32>() / velocities.len() as f32;
        let min_velocity = *velocities.iter().min().unwrap();
        let max_velocity = *velocities.iter().max().unwrap();

        stats.push_str("\nVelocity statistics:\n");
        stats.push_str(&format!("  Average: {:.1}\n", avg_velocity));
        stats.push_str(&format!(
            "  Range: {}-{} (MIDI)\n",
            min_velocity, max_velocity
        ));
    }

    // Tempo and timing info
    if let Some(tempo) = &state.tempo_meter_analysis {
        stats.push_str("\nTiming analysis:\n");
        stats.push_str(&format!("  Estimated BPM: {:.1}\n", tempo.bpm));
        stats.push_str(&format!("  Meter: {}\n", tempo.meter));
        stats.push_str(&format!("  Swing ratio: {:.2}\n", tempo.swing_ratio));
        stats.push_str(&format!(
            "  Beat tracking F1: {:.3}\n",
            tempo.beat_tracking_f1
        ));
    }

    // Ghost notes
    let ghost_count = state.midi_events.iter().filter(|e| e.is_ghost_note).count();
    stats.push_str(&format!(
        "\nGhost notes: {} ({:.1}%)\n",
        ghost_count,
        if state.midi_events.is_empty() {
            0.0
        } else {
            ghost_count as f32 / state.midi_events.len() as f32 * 100.0
        }
    ));

    std::fs::write(stats_path, stats)?;

    Ok(())
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    // Note: MIDI export and QA artifacts are handled by the main pipeline
    // This pass focuses on final cleanup and preparation

    let mut events = state.midi_events.clone();

    if events.is_empty() {
        return Ok(());
    }

    // 1. Event deduplication
    let dedup_result = deduplicate_events(&events);
    events = dedup_result.kept_events;
    println!(
        "Deduplication: removed {} duplicate events",
        dedup_result.removed_count
    );

    // 2. Apply acoustic masking
    apply_acoustic_masking(&mut events, config);
    println!("Acoustic masking: {} events after masking", events.len());

    // 3. Quantization with timing preservation
    if let Some(tempo_analysis) = &state.tempo_meter_analysis {
        let quant_result = quantize_events(&events, tempo_analysis, config);
        events = quant_result.quantized_events;
        println!("Quantization: max drift {:.1}ms", quant_result.max_drift_ms);
    }

    // Update state with processed events
    state.midi_events = events;

    // Note: JSON export and QA stats would be called from the main pipeline
    // but we implement the logic here for completeness

    Ok(())
}
