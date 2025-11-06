//! QA artifacts generation

use crate::audio::AudioState;
use crate::DrumError;
use plotters::prelude::*;
use std::collections::HashMap;
use std::fs;

/// Generate QA artifacts (plots, reports, etc.)
pub fn generate_artifacts(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    // Create QA output directory
    let qa_dir = output_dir.join("qa");
    fs::create_dir_all(&qa_dir)?;

    println!("Generating QA artifacts...");

    // Generate various QA artifacts
    generate_spectrogram_plot(state, &qa_dir)?;
    generate_onset_plot(state, &qa_dir)?;
    generate_classification_plot(state, &qa_dir)?;
    generate_timing_analysis_plot(state, &qa_dir)?;
    generate_midi_events_plot(state, &qa_dir)?;
    generate_statistics_report(state, &qa_dir)?;
    generate_quality_metrics(state, &qa_dir)?;

    println!("QA artifacts generated in {}", qa_dir.display());
    Ok(())
}

/// Generate spectrogram visualization (heatmap)
fn generate_spectrogram_plot(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    let path = output_dir.join("spectrogram.png");
    let root = BitMapBackend::new(&path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        DrumError::QaGenerationError(format!("Failed to fill plot background: {:?}", e))
    })?;

    let _duration_sec = state.duration_sec();

    // Get STFT data - use the first available STFT
    let stft_data = if let Some((_, stft)) = state.stfts.iter().next() {
        stft
    } else {
        // If no STFT available, skip spectrogram generation
        return Ok(());
    };

    // Compute magnitude spectrogram
    let mag_spec = crate::spectral::magnitude_spectrogram(stft_data);
    let n_frames = mag_spec.shape()[1];
    let n_bins = mag_spec.shape()[0];
    let freq_range = 0..n_bins;
    let time_range = 0..n_frames;

    let mut chart = ChartBuilder::on(&root)
        .caption("Spectrogram Heatmap", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(time_range.clone(), freq_range.clone())
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to build chart: {:?}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Time (frames)")
        .y_desc("Frequency Bin")
        .draw()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw mesh: {:?}", e)))?;

    // Create heatmap data - convert magnitude to dB scale
    let mut heatmap_data = Vec::new();
    let mut max_magnitude = 0.0f32;

    for &magnitude in mag_spec.iter() {
        max_magnitude = max_magnitude.max(magnitude);
    }

    for frame_idx in 0..n_frames {
        for bin_idx in 0..n_bins {
            let magnitude = mag_spec[[bin_idx, frame_idx]];
            // Convert to dB scale for better visualization
            let db_value = if magnitude > 0.0 {
                20.0 * (magnitude / max_magnitude).log10()
            } else {
                -100.0 // Very low dB for zero values
            };

            // Map dB to color intensity (clamp to reasonable range)
            let intensity = ((db_value + 100.0) / 100.0).max(0.0).min(1.0);
            heatmap_data.push((frame_idx, bin_idx, intensity));
        }
    }

    // Draw heatmap using rectangles
    chart
        .draw_series(heatmap_data.into_iter().map(|(x, y, intensity)| {
            let color = if intensity > 0.7 {
                RGBColor(255, 0, 0) // Red for high energy
            } else if intensity > 0.4 {
                RGBColor(255, 165, 0) // Orange for medium energy
            } else if intensity > 0.1 {
                RGBColor(255, 255, 0) // Yellow for low energy
            } else {
                RGBColor(0, 0, 0) // Black for very low energy
            };

            Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
        }))
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw series: {:?}", e)))?;

    Ok(())
}

/// Generate onset detection visualization
fn generate_onset_plot(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    let path = output_dir.join("onsets.png");
    let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        DrumError::QaGenerationError(format!("Failed to fill plot background: {:?}", e))
    })?;

    let duration_sec = state.duration_sec();
    let max_strength = state
        .onset_events
        .iter()
        .map(|e| e.strength)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(1.0);

    let mut chart = ChartBuilder::on(&root)
        .caption("Onset Events", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0f64..duration_sec as f64, 0.0f64..max_strength as f64)
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to build chart: {:?}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Time (seconds)")
        .y_desc("Onset Strength")
        .draw()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw mesh: {:?}", e)))?;

    // Plot onset events
    chart
        .draw_series(state.onset_events.iter().map(|event| {
            Circle::new(
                (event.time_sec as f64, event.strength as f64),
                3,
                BLUE.filled(),
            )
        }))
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw series: {:?}", e)))?;

    Ok(())
}

/// Generate classification results visualization (simplified)
fn generate_classification_plot(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    let path = output_dir.join("classification.png");
    let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        DrumError::QaGenerationError(format!("Failed to fill plot background: {:?}", e))
    })?;

    let duration_sec = state.duration_sec();

    let mut chart = ChartBuilder::on(&root)
        .caption("Drum Classification Results", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0f64..duration_sec as f64, 0.0f64..1.0f64)
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to build chart: {:?}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Time (seconds)")
        .y_desc("Classification Events")
        .draw()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw mesh: {:?}", e)))?;

    // Plot classified events as points
    chart
        .draw_series(
            state
                .classified_events
                .iter()
                .enumerate()
                .map(|(i, event)| {
                    let y_pos = (i % 10) as f64 * 0.1; // Spread events vertically
                    Circle::new((event.time_sec as f64, y_pos), 2, BLUE.filled())
                }),
        )
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw series: {:?}", e)))?;

    Ok(())
}

/// Generate timing analysis visualization
fn generate_timing_analysis_plot(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    let path = output_dir.join("timing_analysis.png");
    let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        DrumError::QaGenerationError(format!("Failed to fill plot background: {:?}", e))
    })?;

    let duration_sec = state.duration_sec();

    let mut chart = ChartBuilder::on(&root)
        .caption("Timing Analysis", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0f64..duration_sec as f64, -50.0f64..50.0f64)
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to build chart: {:?}", e)))?; // Timing drift in ms

    chart
        .configure_mesh()
        .x_desc("Time (seconds)")
        .y_desc("Timing Drift (ms)")
        .draw()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw mesh: {:?}", e)))?;

    // Plot timing refinements
    if let Some(_timing_stats) = state
        .tempo_meter_analysis
        .as_ref().map(|_| &state.refined_events)
    {
        chart
            .draw_series(state.refined_events.iter().map(|event| {
                Circle::new(
                    (event.refined_time_sec as f64, event.drift_ms as f64),
                    3,
                    if event.drift_ms.abs() < 15.0 {
                        GREEN.filled()
                    } else {
                        RED.filled()
                    },
                )
            }))
            .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw series: {:?}", e)))?;
    }

    Ok(())
}

/// Generate MIDI events visualization
fn generate_midi_events_plot(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    let path = output_dir.join("midi_events.png");
    let root = BitMapBackend::new(&path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| {
        DrumError::QaGenerationError(format!("Failed to fill plot background: {:?}", e))
    })?;

    let duration_sec = state.duration_sec();

    let mut chart = ChartBuilder::on(&root)
        .caption("Final MIDI Events", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0f64..duration_sec as f64, 0.0f64..1.0f64)
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to build chart: {:?}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Time (seconds)")
        .y_desc("MIDI Events")
        .draw()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw mesh: {:?}", e)))?;

    // Plot MIDI events as points
    chart
        .draw_series(state.midi_events.iter().enumerate().map(|(i, event)| {
            let y_pos = (i % 10) as f64 * 0.1; // Spread events vertically
            let color = if event.is_ghost_note { CYAN } else { BLUE };
            Circle::new((event.time_sec as f64, y_pos), 2, color.filled())
        }))
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw series: {:?}", e)))?;

    Ok(())
}

/// Generate statistics report
fn generate_statistics_report(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    let path = output_dir.join("statistics.json");

    let mut stats = serde_json::Map::new();

    // Basic audio statistics
    stats.insert("duration_seconds".to_string(), state.duration_sec().into());
    stats.insert("sample_rate".to_string(), state.sr.into());
    stats.insert("total_samples".to_string(), state.n_samples().into());

    // Onset statistics
    stats.insert("total_onsets".to_string(), state.onset_events.len().into());
    stats.insert(
        "total_classified_events".to_string(),
        state.classified_events.len().into(),
    );
    stats.insert(
        "total_refined_events".to_string(),
        state.refined_events.len().into(),
    );
    stats.insert(
        "total_midi_events".to_string(),
        state.midi_events.len().into(),
    );

    // Classification distribution
    let mut class_counts = HashMap::new();
    for event in &state.classified_events {
        *class_counts.entry(event.drum_class.name()).or_insert(0) += 1;
    }
    stats.insert(
        "classification_distribution".to_string(),
        serde_json::to_value(class_counts)?,
    );

    // Tempo/meter analysis
    if let Some(tempo_analysis) = &state.tempo_meter_analysis {
        stats.insert("bpm".to_string(), tempo_analysis.bpm.into());
        stats.insert("meter".to_string(), tempo_analysis.meter.clone().into());
        stats.insert(
            "beat_positions".to_string(),
            tempo_analysis.beat_positions.clone().into(),
        );
    }

    // Quality metrics
    let quality_metrics = compute_quality_metrics(state);
    stats.insert(
        "quality_metrics".to_string(),
        serde_json::to_value(quality_metrics)?,
    );

    // Write JSON report
    let json = serde_json::to_string_pretty(&stats)?;
    fs::write(path, json)?;

    Ok(())
}

/// Generate quality metrics report
fn generate_quality_metrics(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    let path = output_dir.join("quality_metrics.txt");
    let metrics = compute_quality_metrics(state);

    let mut report = String::new();
    report.push_str("DRUM TO MIDI - QUALITY METRICS REPORT\n");
    report.push_str("=====================================\n\n");

    report.push_str(&format!(
        "Total Events Processed: {}\n",
        metrics.total_events
    ));
    report.push_str(&format!(
        "Events with High Confidence (>0.8): {}\n",
        metrics.high_confidence_events
    ));
    report.push_str(&format!(
        "Events with Low Confidence (<0.3): {}\n",
        metrics.low_confidence_events
    ));
    report.push_str(&format!(
        "Average Confidence: {:.3}\n",
        metrics.avg_confidence
    ));
    report.push_str(&format!(
        "Classification Consistency: {:.3}\n",
        metrics.classification_consistency
    ));
    report.push_str(&format!(
        "Timing Precision (median drift): {:.1}ms\n",
        metrics.timing_precision_ms
    ));
    report.push_str(&format!(
        "Processing Time per Event: {:.1}ms\n",
        metrics.processing_time_per_event_ms
    ));

    if let Some(tempo_analysis) = &state.tempo_meter_analysis {
        report.push_str("\nTempo Analysis:\n");
        report.push_str(&format!("  BPM: {:.1}\n", tempo_analysis.bpm));
        report.push_str(&format!("  Meter: {}\n", tempo_analysis.meter));
        report.push_str(&format!(
            "  Beat Tracking F1: {:.3}\n",
            tempo_analysis.beat_tracking_f1
        ));
    }

    fs::write(path, report)?;

    Ok(())
}

/// Compute comprehensive quality metrics
fn compute_quality_metrics(state: &AudioState) -> QualityMetrics {
    let total_events = state.midi_events.len();

    let high_confidence_events = state
        .midi_events
        .iter()
        .filter(|e| e.confidence > 0.8)
        .count();

    let low_confidence_events = state
        .midi_events
        .iter()
        .filter(|e| e.confidence < 0.3)
        .count();

    let avg_confidence = if total_events > 0 {
        state.midi_events.iter().map(|e| e.confidence).sum::<f32>() / total_events as f32
    } else {
        0.0
    };

    // Classification consistency (how often alternative classifications have similar confidence)
    let classification_consistency = if !state.classified_events.is_empty() {
        let total_alternatives: usize = state
            .classified_events
            .iter()
            .map(|e| e.alternative_classes.len())
            .sum();
        let high_alternatives: usize = state
            .classified_events
            .iter()
            .flat_map(|e| &e.alternative_classes)
            .filter(|(_, conf)| *conf > 0.5)
            .count();
        high_alternatives as f32 / total_alternatives as f32
    } else {
        0.0
    };

    // Timing precision (median absolute drift)
    let timing_precision_ms = if !state.refined_events.is_empty() {
        let mut drifts: Vec<f32> = state
            .refined_events
            .iter()
            .map(|e| e.drift_ms.abs())
            .collect();
        drifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        drifts[drifts.len() / 2]
    } else {
        0.0
    };

    // Estimated processing time per event (rough estimate)
    let processing_time_per_event_ms = 50.0; // Placeholder - would need actual timing

    QualityMetrics {
        total_events,
        high_confidence_events,
        low_confidence_events,
        avg_confidence,
        classification_consistency,
        timing_precision_ms,
        processing_time_per_event_ms,
    }
}

/// Quality metrics structure
#[derive(Debug, serde::Serialize)]
struct QualityMetrics {
    total_events: usize,
    high_confidence_events: usize,
    low_confidence_events: usize,
    avg_confidence: f32,
    classification_consistency: f32,
    timing_precision_ms: f32,
    processing_time_per_event_ms: f32,
}
