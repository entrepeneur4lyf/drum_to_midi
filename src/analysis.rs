//! Analysis and feature extraction

use crate::audio::AudioState;
use serde::{Deserialize, Serialize};

/// Drum instrument classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrumClass {
    Kick,
    Snare,
    Rimshot,
    Tom,
    HiHat,
    Splash,
    Cowbell,
    Cymbal,
    Percussion,
    Unknown,
}

impl DrumClass {
    /// Get MIDI note number for standard drum mapping
    pub fn midi_note(&self) -> u8 {
        match self {
            DrumClass::Kick => 36,       // C2
            DrumClass::Snare => 38,      // D2
            DrumClass::Rimshot => 37,    // C#2 (side stick/rimshot)
            DrumClass::Tom => 45,        // A2 (floor tom)
            DrumClass::HiHat => 42,      // F#2 (closed hi-hat)
            DrumClass::Splash => 55,     // G3 (splash cymbal)
            DrumClass::Cowbell => 56,    // G#3 (cowbell)
            DrumClass::Cymbal => 49,     // C#3 (crash cymbal)
            DrumClass::Percussion => 39, // Eb2 (hand clap/percussion)
            DrumClass::Unknown => 0,     // No mapping
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            DrumClass::Kick => "kick",
            DrumClass::Snare => "snare",
            DrumClass::Rimshot => "rimshot",
            DrumClass::Tom => "tom",
            DrumClass::HiHat => "hi-hat",
            DrumClass::Splash => "splash",
            DrumClass::Cowbell => "cowbell",
            DrumClass::Cymbal => "cymbal",
            DrumClass::Percussion => "percussion",
            DrumClass::Unknown => "unknown",
        }
    }
}

/// Classification features for drum events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationFeatures {
    pub fundamental_hz: Option<f32>,
    pub spectral_centroid_hz: f32,
    pub spectral_rolloff_hz: f32,
    pub zero_crossing_rate: f32,
    pub low_energy_ratio: f32,  // Energy in 60-250Hz band
    pub mid_energy_ratio: f32,  // Energy in 250Hz-2kHz band
    pub high_energy_ratio: f32, // Energy in 2kHz+ band
    pub attack_time_ms: f32,
    pub decay_time_ms: f32,
    // Transient-specific features
    pub transient_energy_ratio: f32,     // Ratio of attack energy to total energy
    pub attack_spectral_centroid: f32,   // Centroid during attack phase only
    pub attack_hf_ratio: f32,            // HF content during attack vs sustained
    pub transient_sharpness: f32,        // Rate of level change at attack
    pub attack_to_sustain_ratio: f32,    // Attack peak vs sustained level
}

impl Default for ClassificationFeatures {
    fn default() -> Self {
        Self {
            fundamental_hz: None,
            spectral_centroid_hz: 1000.0,
            spectral_rolloff_hz: 3000.0,
            zero_crossing_rate: 0.1,
            low_energy_ratio: 0.4,
            mid_energy_ratio: 0.4,
            high_energy_ratio: 0.2,
            attack_time_ms: 5.0,
            decay_time_ms: 100.0,
            transient_energy_ratio: 0.3,
            attack_spectral_centroid: 2000.0,
            attack_hf_ratio: 0.5,
            transient_sharpness: 0.2,
            attack_to_sustain_ratio: 2.0,
        }
    }
}

/// Classified drum event with confidence scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifiedEvent {
    pub time_sec: f32,
    pub frame_idx: usize,
    pub drum_class: DrumClass,
    pub confidence: f32,
    pub acoustic_confidence: f32,
    pub prior_confidence: f32,
    pub features: ClassificationFeatures,
    pub alternative_classes: Vec<(DrumClass, f32)>, // Alternative classifications with scores
}

/// Refined timing event with sub-frame precision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinedEvent {
    pub original_time_sec: f32,
    pub refined_time_sec: f32,
    pub frame_idx: usize,
    pub refined_frame_idx: usize,
    pub drum_class: DrumClass,
    pub confidence: f32,
    pub timing_confidence: f32,
    pub drift_ms: f32,
    pub snr_at_refined_time: f32,
    pub features: ClassificationFeatures,
}

/// Final MIDI event with grid position and confidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidiEvent {
    pub time_sec: f32,
    pub grid_position: GridPosition,
    pub drum_class: DrumClass,
    pub velocity: u8, // MIDI velocity 0-127
    pub confidence: f32,
    pub is_ghost_note: bool, // True if inserted by gap filling
    pub acoustic_score: f32,
    pub prior_score: f32,
    pub density_score: f32,
}

/// Grid position for rhythmic placement
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GridPosition {
    pub bar: usize,      // Bar number (0-based)
    pub beat: usize,     // Beat within bar (0-based)
    pub sub_beat: usize, // Sub-beat division (0-based, e.g., 0,1,2,3 for 16th notes)
    pub ticks: usize,    // MIDI ticks within sub-beat for fine positioning
}

/// Timing refinement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    pub total_events: usize,
    pub refined_events: usize,
    pub median_drift_ms: f32,
    pub mean_drift_ms: f32,
    pub max_drift_ms: f32,
    pub drift_std_ms: f32,
    pub events_within_5ms: usize,
    pub events_within_15ms: usize,
    pub events_within_30ms: usize,
    pub per_class_stats: std::collections::HashMap<DrumClass, ClassTimingStats>,
}

/// Per-class timing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassTimingStats {
    pub count: usize,
    pub median_drift_ms: f32,
    pub mean_drift_ms: f32,
    pub max_drift_ms: f32,
    pub events_within_5ms: usize,
    pub events_within_15ms: usize,
}

/// Tempo and meter analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempoMeterAnalysis {
    pub bpm: f32,
    pub bpm_confidence: f32,
    pub meter: String, // e.g., "4/4", "3/4", "7/8"
    pub meter_confidence: f32,
    pub swing_ratio: f32, // 0.5 = straight, 0.67 = triplet feel, etc.
    pub swing_confidence: f32,
    pub beat_positions: Vec<f32>, // Time positions of beats in seconds
    pub downbeat_positions: Vec<f32>, // Time positions of downbeats (measure starts)
    pub tempo_curve: Vec<[f32; 2]>, // [(time_sec, bpm), ...]
    pub beat_tracking_f1: f32,
}

/// Beat tracking candidate
#[derive(Debug, Clone)]
pub struct BeatCandidate {
    pub interval_sec: f32,
    pub strength: f32,
    pub consistency: f32,
}

/// Meter candidate with evaluation score
#[derive(Debug, Clone)]
pub struct MeterCandidate {
    pub signature: String,
    pub beats_per_measure: usize,
    pub beat_divisions: usize, // e.g., 4 for quarter notes, 8 for eighth notes
    pub score: f32,
    pub confidence: f32,
}

/// Swing analysis results
#[derive(Debug, Clone)]
pub struct SwingAnalysis {
    pub ratio: f32,
    pub confidence: f32,
    pub off_beat_displacement: f32,
}

/// Self-prior probability matrices for grid inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfPriorMatrices {
    pub grid_slots_per_beat: usize,
    pub class_priors: std::collections::HashMap<DrumClass, Vec<f32>>,
    pub class_confidences: std::collections::HashMap<DrumClass, Vec<f32>>,
    pub total_events_per_class: std::collections::HashMap<DrumClass, usize>,
    pub smoothing_sigma_beats: f32,
    pub beta_smoothing_alpha: f32,
    pub beta_smoothing_beta: f32,
}

/// Prior construction statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorStats {
    pub total_bars_analyzed: usize,
    pub events_per_bar_avg: f32,
    pub class_distribution: std::collections::HashMap<DrumClass, f32>,
    pub grid_coverage_percent: f32,
    pub smoothing_effectiveness: f32,
}

/// Analysis results structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Analysis {
    pub version: String,
    pub bpm: f32,
    pub bpm_curve: Option<Vec<[f32; 2]>>,
    pub meter: String,
    pub swing_ratio: f32,
    pub tuning: TuningInfo,
    pub reverb_characteristics: ReverbInfo,
    pub beat_tracking_quality: QualityMetrics,
}

/// Tuning information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningInfo {
    pub kick_hz: Option<f32>,
    pub kick_confidence: f32,
    pub kick_coherence: f32,
    pub toms_hz: Vec<f32>,
    pub toms_confidence: Vec<f32>,
    pub toms_coherence: Vec<f32>,
    pub snare_body_hz: f32,
    pub snare_body_range_hz: [f32; 2],
}

/// Reverb characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverbInfo {
    pub rt60_estimate_ms: f32,
    pub strength: f32,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub f1: f32,
    pub precision: f32,
    pub recall: f32,
}

/// Export analysis results to JSON
pub fn export_analysis(
    state: &AudioState,
    output_dir: &std::path::Path,
) -> crate::DrumErrorResult<()> {
    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    // Generate filename
    let analysis_filename = "analysis.json";
    let analysis_path = output_dir.join(analysis_filename);

    // Build comprehensive analysis structure
    let analysis = build_analysis_results(state);

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&analysis)?;
    std::fs::write(&analysis_path, json)?;

    println!("Exported analysis results to {}", analysis_path.display());
    Ok(())
}

/// Build comprehensive analysis results structure
fn build_analysis_results(state: &AudioState) -> AnalysisResults {
    AnalysisResults {
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string(),
        audio_info: AudioInfo {
            duration_seconds: state.duration_sec(),
            sample_rate: state.sr,
            total_samples: state.n_samples(),
            channels: 1, // Always mono after processing
        },
        processing_pipeline: ProcessingPipeline {
            onset_events: state.onset_events.len(),
            classified_events: state.classified_events.len(),
            refined_events: state.refined_events.len(),
            midi_events: state.midi_events.len(),
        },
        tempo_analysis: state.tempo_meter_analysis.as_ref().map(|t| TempoAnalysis {
            bpm: t.bpm,
            bpm_confidence: t.bpm_confidence,
            meter: t.meter.clone(),
            meter_confidence: t.meter_confidence,
            swing_ratio: t.swing_ratio,
            swing_confidence: t.swing_confidence,
            beat_positions: t.beat_positions.clone(),
            downbeat_positions: t.downbeat_positions.clone(),
            tempo_curve: t.tempo_curve.clone(),
            beat_tracking_f1: t.beat_tracking_f1,
        }),
        tuning_analysis: state.tuning_info.as_ref().map(|t| TuningAnalysis {
            kick_hz: t.kick_hz,
            kick_confidence: t.kick_confidence,
            kick_coherence: t.kick_coherence,
            toms_hz: t.toms_hz.clone(),
            toms_confidence: t.toms_confidence.clone(),
            toms_coherence: t.toms_coherence.clone(),
            snare_body_hz: t.snare_body_hz,
            snare_body_range_hz: t.snare_body_range_hz,
        }),
        reverb_analysis: state.reverb_info.as_ref().map(|r| ReverbAnalysis {
            rt60_estimate_ms: r.rt60_estimate_ms,
            strength: r.strength,
        }),
        prior_analysis: state.prior_stats.as_ref().map(|p| PriorAnalysis {
            total_bars_analyzed: p.total_bars_analyzed,
            events_per_bar_avg: p.events_per_bar_avg,
            class_distribution: p.class_distribution.clone(),
            grid_coverage_percent: p.grid_coverage_percent,
            smoothing_effectiveness: p.smoothing_effectiveness,
        }),
        events: EventsSummary {
            onset_events: state
                .onset_events
                .iter()
                .map(|e| OnsetEventSummary {
                    time_sec: e.time_sec,
                    strength: e.strength,
                    snr: e.snr,
                    spectral_centroid_hz: e.spectral_centroid_hz,
                    quality_score: e.quality_score,
                })
                .collect(),
            classified_events: state
                .classified_events
                .iter()
                .map(|e| ClassifiedEventSummary {
                    time_sec: e.time_sec,
                    drum_class: e.drum_class,
                    confidence: e.confidence,
                    acoustic_confidence: e.acoustic_confidence,
                    prior_confidence: e.prior_confidence,
                    features: e.features.clone(),
                    alternative_classes: e.alternative_classes.clone(),
                })
                .collect(),
            refined_events: state
                .refined_events
                .iter()
                .map(|e| RefinedEventSummary {
                    original_time_sec: e.original_time_sec,
                    refined_time_sec: e.refined_time_sec,
                    drum_class: e.drum_class,
                    confidence: e.confidence,
                    timing_confidence: e.timing_confidence,
                    drift_ms: e.drift_ms,
                    snr_at_refined_time: e.snr_at_refined_time,
                })
                .collect(),
            midi_events: state
                .midi_events
                .iter()
                .map(|e| MidiEventSummary {
                    time_sec: e.time_sec,
                    grid_position: e.grid_position,
                    drum_class: e.drum_class,
                    velocity: e.velocity,
                    confidence: e.confidence,
                    is_ghost_note: e.is_ghost_note,
                    acoustic_score: e.acoustic_score,
                    prior_score: e.prior_score,
                    density_score: e.density_score,
                })
                .collect(),
        },
        quality_metrics: compute_overall_quality_metrics(state),
    }
}

/// Overall analysis results structure
#[derive(Debug, serde::Serialize)]
struct AnalysisResults {
    version: String,
    timestamp: String,
    audio_info: AudioInfo,
    processing_pipeline: ProcessingPipeline,
    tempo_analysis: Option<TempoAnalysis>,
    tuning_analysis: Option<TuningAnalysis>,
    reverb_analysis: Option<ReverbAnalysis>,
    prior_analysis: Option<PriorAnalysis>,
    events: EventsSummary,
    quality_metrics: OverallQualityMetrics,
}

/// Audio file information
#[derive(Debug, serde::Serialize)]
struct AudioInfo {
    duration_seconds: f32,
    sample_rate: u32,
    total_samples: usize,
    channels: usize,
}

/// Processing pipeline summary
#[derive(Debug, serde::Serialize)]
struct ProcessingPipeline {
    onset_events: usize,
    classified_events: usize,
    refined_events: usize,
    midi_events: usize,
}

/// Tempo and meter analysis
#[derive(Debug, serde::Serialize)]
struct TempoAnalysis {
    bpm: f32,
    bpm_confidence: f32,
    meter: String,
    meter_confidence: f32,
    swing_ratio: f32,
    swing_confidence: f32,
    beat_positions: Vec<f32>,
    downbeat_positions: Vec<f32>,
    tempo_curve: Vec<[f32; 2]>,
    beat_tracking_f1: f32,
}

/// Tuning analysis results
#[derive(Debug, serde::Serialize)]
struct TuningAnalysis {
    kick_hz: Option<f32>,
    kick_confidence: f32,
    kick_coherence: f32,
    toms_hz: Vec<f32>,
    toms_confidence: Vec<f32>,
    toms_coherence: Vec<f32>,
    snare_body_hz: f32,
    snare_body_range_hz: [f32; 2],
}

/// Reverb analysis results
#[derive(Debug, serde::Serialize)]
struct ReverbAnalysis {
    rt60_estimate_ms: f32,
    strength: f32,
}

/// Prior construction analysis
#[derive(Debug, serde::Serialize)]
struct PriorAnalysis {
    total_bars_analyzed: usize,
    events_per_bar_avg: f32,
    class_distribution: std::collections::HashMap<DrumClass, f32>,
    grid_coverage_percent: f32,
    smoothing_effectiveness: f32,
}

/// Events summary with detailed data
#[derive(Debug, serde::Serialize)]
struct EventsSummary {
    onset_events: Vec<OnsetEventSummary>,
    classified_events: Vec<ClassifiedEventSummary>,
    refined_events: Vec<RefinedEventSummary>,
    midi_events: Vec<MidiEventSummary>,
}

/// Onset event summary
#[derive(Debug, serde::Serialize)]
struct OnsetEventSummary {
    time_sec: f32,
    strength: f32,
    snr: f32,
    spectral_centroid_hz: f32,
    quality_score: f32,
}

/// Classified event summary
#[derive(Debug, serde::Serialize)]
struct ClassifiedEventSummary {
    time_sec: f32,
    drum_class: DrumClass,
    confidence: f32,
    acoustic_confidence: f32,
    prior_confidence: f32,
    features: ClassificationFeatures,
    alternative_classes: Vec<(DrumClass, f32)>,
}

/// Refined event summary
#[derive(Debug, serde::Serialize)]
struct RefinedEventSummary {
    original_time_sec: f32,
    refined_time_sec: f32,
    drum_class: DrumClass,
    confidence: f32,
    timing_confidence: f32,
    drift_ms: f32,
    snr_at_refined_time: f32,
}

/// MIDI event summary
#[derive(Debug, serde::Serialize)]
struct MidiEventSummary {
    time_sec: f32,
    grid_position: GridPosition,
    drum_class: DrumClass,
    velocity: u8,
    confidence: f32,
    is_ghost_note: bool,
    acoustic_score: f32,
    prior_score: f32,
    density_score: f32,
}

/// Overall quality metrics
#[derive(Debug, serde::Serialize)]
struct OverallQualityMetrics {
    total_events_processed: usize,
    average_confidence: f32,
    timing_precision_ms: f32,
    classification_accuracy_estimate: f32,
    processing_completeness: f32,
    ghost_notes_ratio: f32,
}

/// Compute overall quality metrics
fn compute_overall_quality_metrics(state: &AudioState) -> OverallQualityMetrics {
    let total_events = state.midi_events.len();

    let average_confidence = if total_events > 0 {
        state.midi_events.iter().map(|e| e.confidence).sum::<f32>() / total_events as f32
    } else {
        0.0
    };

    let timing_precision_ms = if !state.refined_events.is_empty() {
        let mut drifts: Vec<f32> = state
            .refined_events
            .iter()
            .map(|e| e.drift_ms.abs())
            .collect();
        drifts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        drifts[drifts.len() / 2] // median
    } else {
        0.0
    };

    // Estimate classification accuracy based on confidence scores
    let classification_accuracy_estimate = if !state.classified_events.is_empty() {
        let high_confidence = state
            .classified_events
            .iter()
            .filter(|e| e.confidence > 0.8)
            .count();
        high_confidence as f32 / state.classified_events.len() as f32
    } else {
        0.0
    };

    // Processing completeness (ratio of final MIDI events to initial onsets)
    let processing_completeness = if !state.onset_events.is_empty() {
        state.midi_events.len() as f32 / state.onset_events.len() as f32
    } else {
        0.0
    };

    // Ghost notes ratio
    let ghost_notes_ratio = if !state.midi_events.is_empty() {
        let ghost_notes = state.midi_events.iter().filter(|e| e.is_ghost_note).count();
        ghost_notes as f32 / state.midi_events.len() as f32
    } else {
        0.0
    };

    OverallQualityMetrics {
        total_events_processed: total_events,
        average_confidence,
        timing_precision_ms,
        classification_accuracy_estimate,
        processing_completeness,
        ghost_notes_ratio,
    }
}
