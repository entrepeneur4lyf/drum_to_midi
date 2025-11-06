//! Configuration system for the Drum-to-MIDI processor

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub version: String,
    pub audio: AudioConfig,
    pub stereo: StereoConfig,
    pub hpss: HpssConfig,
    pub stft: StftConfig,
    pub onset_fusion: OnsetFusionConfig,
    pub whitening: WhiteningConfig,
    pub thresholds: ThresholdConfig,
    pub onset_seeding: OnsetSeedingConfig,
    pub clustering: ClusteringConfig,
    pub reverb: ReverbConfig,
    pub classification: ClassificationConfig,
    pub timing_refinement: TimingRefinementConfig,
    pub tempo_meter: TempoMeterConfig,
    pub self_prior: SelfPriorConfig,
    pub tempo_weights: HashMap<String, f32>,
    pub tempo_curve: TempoCurveConfig,
    pub grid: GridConfig,
    pub fill_protection: FillProtectionConfig,
    pub velocity: VelocityConfig,
    pub quantize: QuantizeConfig,
    pub humanize: HumanizeConfig,
    pub postprocess: PostProcessConfig,
    pub export: ExportConfig,
    pub convergence: ConvergenceConfig,
    pub validation: ValidationConfig,
    pub edge_cases: EdgeCasesConfig,
    pub qa: QaConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            audio: AudioConfig::default(),
            stereo: StereoConfig::default(),
            hpss: HpssConfig::default(),
            stft: StftConfig::default(),
            onset_fusion: OnsetFusionConfig::default(),
            whitening: WhiteningConfig::default(),
            thresholds: ThresholdConfig::default(),
            onset_seeding: OnsetSeedingConfig::default(),
            clustering: ClusteringConfig::default(),
            reverb: ReverbConfig::default(),
            classification: ClassificationConfig::default(),
            timing_refinement: TimingRefinementConfig::default(),
            tempo_meter: TempoMeterConfig::default(),
            self_prior: SelfPriorConfig::default(),
            tempo_weights: default_tempo_weights(),
            tempo_curve: TempoCurveConfig::default(),
            grid: GridConfig::default(),
            fill_protection: FillProtectionConfig::default(),
            velocity: VelocityConfig::default(),
            quantize: QuantizeConfig::default(),
            humanize: HumanizeConfig::default(),
            postprocess: PostProcessConfig::default(),
            export: ExportConfig::default(),
            convergence: ConvergenceConfig::default(),
            validation: ValidationConfig::default(),
            edge_cases: EdgeCasesConfig::default(),
            qa: QaConfig::default(),
        }
    }
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub target_lufs: f32,
    pub true_peak_limit_db: f32,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            target_lufs: -16.0,
            true_peak_limit_db: -1.0,
        }
    }
}

/// Stereo processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StereoConfig {
    pub imbalance_ratio: f32,
    pub xcorr_min: f32,
    pub adaptive_window_sec: f32,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            imbalance_ratio: 1.5,
            xcorr_min: 0.6,
            adaptive_window_sec: 5.0,
        }
    }
}

/// HPSS (Harmonic/Percussive Source Separation) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HpssConfig {
    pub margin: usize,
    pub weights_by_style: HashMap<String, HashMap<String, f32>>,
}

impl Default for HpssConfig {
    fn default() -> Self {
        let mut weights_by_style = HashMap::new();
        let mut rock_weights = HashMap::new();
        rock_weights.insert("percussive".to_string(), 0.85);
        rock_weights.insert("harmonic".to_string(), 0.15);
        weights_by_style.insert("rock".to_string(), rock_weights);

        let mut pop_weights = HashMap::new();
        pop_weights.insert("percussive".to_string(), 0.80);
        pop_weights.insert("harmonic".to_string(), 0.20);
        weights_by_style.insert("pop".to_string(), pop_weights);

        let mut jazz_weights = HashMap::new();
        jazz_weights.insert("percussive".to_string(), 0.65);
        jazz_weights.insert("harmonic".to_string(), 0.35);
        weights_by_style.insert("jazz".to_string(), jazz_weights);

        let mut electronic_weights = HashMap::new();
        electronic_weights.insert("percussive".to_string(), 0.90);
        electronic_weights.insert("harmonic".to_string(), 0.10);
        weights_by_style.insert("electronic".to_string(), electronic_weights);

        let mut metal_weights = HashMap::new();
        metal_weights.insert("percussive".to_string(), 0.85);
        metal_weights.insert("harmonic".to_string(), 0.15);
        weights_by_style.insert("metal".to_string(), metal_weights);

        let mut default_weights = HashMap::new();
        default_weights.insert("percussive".to_string(), 0.80);
        default_weights.insert("harmonic".to_string(), 0.20);
        weights_by_style.insert("default".to_string(), default_weights);

        Self {
            margin: 2,
            weights_by_style,
        }
    }
}

/// STFT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: usize,
    pub window: String,
    pub multi_res_configs: Vec<(usize, usize)>,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            n_fft: 4096,
            hop_length: 512,
            window: "hann".to_string(),
            multi_res_configs: vec![(1024, 128), (2048, 256), (4096, 512)],
        }
    }
}

/// Onset fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct OnsetFusionConfig {
    pub weights: OnsetWeights,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OnsetWeights {
    pub flux: f32,
    pub complex_flux: f32,
    pub high: f32,
    pub transient: f32,
}

impl Default for OnsetWeights {
    fn default() -> Self {
        Self {
            flux: 0.4,
            complex_flux: 0.25,
            high: 0.2,
            transient: 0.15,
        }
    }
}

/// Spectral whitening configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhiteningConfig {
    pub percentile: f32,
    pub third_octave_smoothing: bool,
    pub adaptive: bool,
    pub adaptive_window_sec: f32,
}

impl Default for WhiteningConfig {
    fn default() -> Self {
        Self {
            percentile: 20.0,
            third_octave_smoothing: true,
            adaptive: false,
            adaptive_window_sec: 10.0,
        }
    }
}

/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ThresholdConfig {
    pub k_global: f32,
    pub adaptive_window_sec: f32,
    pub absolute_min: f32,
    pub refractory_ms_base: f32,
    pub flam_ms: [f32; 2],
    pub flam_spectral_tolerance_hz: f32,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            k_global: 1.4,
            adaptive_window_sec: 0.15,
            absolute_min: 0.01,
            refractory_ms_base: 20.0,
            flam_ms: [6.0, 12.0],
            flam_spectral_tolerance_hz: 200.0,
        }
    }
}

/// Onset seeding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OnsetSeedingConfig {
    pub filter_low_quality: bool,
    pub min_seed_snr: f32,
}

impl Default for OnsetSeedingConfig {
    fn default() -> Self {
        Self {
            filter_low_quality: false,
            min_seed_snr: 1.5,
        }
    }
}

/// Clustering configuration for tuning discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusteringConfig {
    pub method: String,
    pub min_samples: usize,
    pub priors: ClusteringPriors,
    pub confidence_thresholds: ClusteringThresholds,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            method: "gmm".to_string(),
            min_samples: 10,
            priors: ClusteringPriors::default(),
            confidence_thresholds: ClusteringThresholds::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusteringPriors {
    pub kick_hz_min: f32,
    pub kick_hz_max: f32,
    pub tom_min_spacing_hz: f32,
}

impl Default for ClusteringPriors {
    fn default() -> Self {
        Self {
            kick_hz_min: 35.0,
            kick_hz_max: 120.0,
            tom_min_spacing_hz: 24.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusteringThresholds {
    pub kick_min: f32,
    pub tom_min: f32,
}

impl Default for ClusteringThresholds {
    fn default() -> Self {
        Self {
            kick_min: 0.3,
            tom_min: 0.3,
        }
    }
}

/// Reverb masking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ReverbConfig {
    pub late_start_ms: f32,
    pub late_end_ms: f32,
    pub strength_a: f32,
    pub edge_cases: HashMap<String, ReverbEdgeCase>,
}

impl Default for ReverbConfig {
    fn default() -> Self {
        let mut edge_cases = HashMap::new();
        edge_cases.insert(
            "super_wet".to_string(),
            ReverbEdgeCase {
                late_start_ms: 100.0,
                late_end_ms: 220.0,
                strength_a: 0.9,
            },
        );

        Self {
            late_start_ms: 80.0,
            late_end_ms: 200.0,
            strength_a: 0.7,
            edge_cases,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReverbEdgeCase {
    pub late_start_ms: f32,
    pub late_end_ms: f32,
    pub strength_a: f32,
}

/// Classification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClassificationConfig {
    pub margins: ClassificationMargins,
    pub cowbell_attack_threshold: f32,
    pub rimshot_attack_threshold: f32,
    pub splash_centroid_min: f32,
    pub cowbell_freq_min: f32,
    pub cowbell_freq_max: f32,
    pub rimshot_hf_ratio_min: f32,
    pub splash_hf_ratio_min: f32,
    pub rimshot_max_decay_ms: f32,
    // Transient-specific parameters
    pub transient_energy_threshold: f32,
    pub attack_hf_threshold: f32,
    pub transient_sharpness_threshold: f32,
    pub attack_sustain_ratio_threshold: f32,
    pub splash_transient_hf_min: f32,
    pub cowbell_transient_mid_min: f32,
    pub rimshot_transient_sharpness_min: f32,
}

impl Default for ClassificationConfig {
    fn default() -> Self {
        Self {
            margins: ClassificationMargins::default(),
            cowbell_attack_threshold: 1.0,
            rimshot_attack_threshold: 1.5,
            splash_centroid_min: 6000.0,
            cowbell_freq_min: 500.0,
            cowbell_freq_max: 1000.0,
            rimshot_hf_ratio_min: 0.4,
            splash_hf_ratio_min: 0.7,
            rimshot_max_decay_ms: 80.0,
            // Transient-specific defaults
            transient_energy_threshold: 0.2,
            attack_hf_threshold: 0.6,
            transient_sharpness_threshold: 0.15,
            attack_sustain_ratio_threshold: 1.5,
            splash_transient_hf_min: 0.8,
            cowbell_transient_mid_min: 0.5,
            rimshot_transient_sharpness_min: 0.3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClassificationMargins {
    pub kick_dominance: f32,
    pub kick_centroid_tol_hz: f32,
    pub kick_hf_reject_ratio: f32,
    pub kick_exclusion_hf_ratio: f32,
    pub kick_exclusion_decay_max: f32,
    pub snare_dominance: f32,
    pub snare_decay_min: f32,
    pub snare_body_min: f32,
    pub hat_high_min: f32,
    pub hat_low_max: f32,
    pub hat_closed_decay_min: f32,
    pub hat_open_decay_max: f32,
    pub cymbal_decay_max: f32,
    pub crash_upper_min: f32,
    pub tom_centroid_tol_hz: f32,
}

impl Default for ClassificationMargins {
    fn default() -> Self {
        Self {
            kick_dominance: 1.3,
            kick_centroid_tol_hz: 150.0,
            kick_hf_reject_ratio: 0.30,
            kick_exclusion_hf_ratio: 0.60,
            kick_exclusion_decay_max: 1.1,
            snare_dominance: 1.5,
            snare_decay_min: 1.5,
            snare_body_min: 0.1,
            hat_high_min: 0.3,
            hat_low_max: 0.1,
            hat_closed_decay_min: 2.2,
            hat_open_decay_max: 2.0,
            cymbal_decay_max: 1.2,
            crash_upper_min: 0.5,
            tom_centroid_tol_hz: 30.0,
        }
    }
}

/// Timing refinement configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TimingRefinementConfig {
    pub base_search_window_ms: f32,
    pub max_search_window_ms: f32,
    pub min_snr_threshold: f32,
    pub max_drift_ms: f32,
}

impl Default for TimingRefinementConfig {
    fn default() -> Self {
        Self {
            base_search_window_ms: 20.0,
            max_search_window_ms: 50.0,
            min_snr_threshold: 1.5,
            max_drift_ms: 15.0,
        }
    }
}

/// Tempo and meter detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TempoMeterConfig {
    pub tempo_range_bpm: [f32; 2],
    pub beat_tracking_drift_tol_ms: f32,
    pub min_beat_tracking_f1: f32,
    pub swing_analysis_window_beats: usize,
    pub swing_ratio_tolerance: f32,
}

impl Default for TempoMeterConfig {
    fn default() -> Self {
        Self {
            tempo_range_bpm: [60.0, 200.0],
            beat_tracking_drift_tol_ms: 50.0,
            min_beat_tracking_f1: 0.70,
            swing_analysis_window_beats: 8,
            swing_ratio_tolerance: 0.05,
        }
    }
}

/// Self-prior construction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SelfPriorConfig {
    pub grid_slots_per_beat: usize,
    pub smoothing_sigma_beats: f32,
    pub beta_smoothing_alpha: f32,
    pub beta_smoothing_beta: f32,
    pub min_events_for_prior: usize,
    pub class_specific_sigmas: std::collections::HashMap<String, f32>,
}

impl Default for SelfPriorConfig {
    fn default() -> Self {
        let mut class_specific_sigmas = std::collections::HashMap::new();
        class_specific_sigmas.insert("kick".to_string(), 0.1); // Precise kick positions
        class_specific_sigmas.insert("snare".to_string(), 0.15); // Slightly more flexible snare
        class_specific_sigmas.insert("hat_closed".to_string(), 0.2); // More flexible hi-hats
        class_specific_sigmas.insert("hat_open".to_string(), 0.2);
        class_specific_sigmas.insert("tom".to_string(), 0.12); // Moderate tom flexibility
        class_specific_sigmas.insert("cymbal".to_string(), 0.25); // Most flexible cymbals

        Self {
            grid_slots_per_beat: 4, // 16th notes
            smoothing_sigma_beats: 0.15,
            beta_smoothing_alpha: 1.0,
            beta_smoothing_beta: 2.0,
            min_events_for_prior: 3,
            class_specific_sigmas,
        }
    }
}

/// Tempo curve configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TempoCurveConfig {
    pub max_snr_cap: f32,
    pub smoothing_window_beats: usize,
    pub median_filter_size: usize,
    pub min_subsample_interval_sec: f32,
    pub min_bpm_change: f32,
}

impl Default for TempoCurveConfig {
    fn default() -> Self {
        Self {
            max_snr_cap: 10.0,
            smoothing_window_beats: 4,
            median_filter_size: 5,
            min_subsample_interval_sec: 2.0,
            min_bpm_change: 2.0,
        }
    }
}

/// Grid inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GridConfig {
    pub assoc_tol_ms: f32,
    pub alpha_time: f32,
    pub lambda_acoustic: f32,
    pub lambda_prior: f32,
    pub beat_drift_percent: f32,
    pub prior_smoothing_beta: f32,
    pub class_thresholds: HashMap<String, f32>,
}

impl Default for GridConfig {
    fn default() -> Self {
        let mut class_thresholds = HashMap::new();
        class_thresholds.insert("kick".to_string(), -8.0);
        class_thresholds.insert("snare".to_string(), -8.0);
        class_thresholds.insert("hat_closed".to_string(), -10.0);
        class_thresholds.insert("hat_open".to_string(), -10.0);
        class_thresholds.insert("tom_low".to_string(), -9.0);
        class_thresholds.insert("tom_mid".to_string(), -9.0);
        class_thresholds.insert("tom_high".to_string(), -9.0);
        class_thresholds.insert("crash".to_string(), -9.0);
        class_thresholds.insert("ride".to_string(), -9.0);

        Self {
            assoc_tol_ms: 15.0,
            alpha_time: 0.05,
            lambda_acoustic: 0.7,
            lambda_prior: 0.3,
            beat_drift_percent: 2.0,
            prior_smoothing_beta: 0.5,
            class_thresholds,
        }
    }
}

/// Fill protection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FillProtectionConfig {
    pub window_sec: f32,
    pub fill_percentile: f32,
    pub silence_percentile: f32,
    pub gap_fill_radius_sec: f32,
    pub gap_fill_min_neighbors: usize,
    pub ghost_prior_threshold: f32,
    pub ghost_prior_threshold_by_style: HashMap<String, f32>,
}

impl Default for FillProtectionConfig {
    fn default() -> Self {
        let mut ghost_prior_threshold_by_style = HashMap::new();
        ghost_prior_threshold_by_style.insert("rock".to_string(), 0.7);
        ghost_prior_threshold_by_style.insert("pop".to_string(), 0.7);
        ghost_prior_threshold_by_style.insert("jazz".to_string(), 0.9);
        ghost_prior_threshold_by_style.insert("electronic".to_string(), 0.5);
        ghost_prior_threshold_by_style.insert("metal".to_string(), 0.65);
        ghost_prior_threshold_by_style.insert("hiphop".to_string(), 0.6);

        Self {
            window_sec: 1.0,
            fill_percentile: 95.0,
            silence_percentile: 5.0,
            gap_fill_radius_sec: 4.0,
            gap_fill_min_neighbors: 2,
            ghost_prior_threshold: 0.7,
            ghost_prior_threshold_by_style,
        }
    }
}

/// Velocity estimation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VelocityConfig {
    pub gamma: f32,
    pub min_samples_for_normalization: usize,
    pub weights: HashMap<String, VelocityWeights>,
    pub context: VelocityContextRules,
}

impl Default for VelocityConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert(
            "default".to_string(),
            VelocityWeights {
                slope: 0.5,
                hf: 0.3,
                energy: 0.2,
            },
        );
        weights.insert(
            "kick".to_string(),
            VelocityWeights {
                slope: 0.6,
                hf: 0.0,
                energy: 0.4,
            },
        );
        weights.insert(
            "snare".to_string(),
            VelocityWeights {
                slope: 0.4,
                hf: 0.4,
                energy: 0.2,
            },
        );
        weights.insert(
            "hat_closed".to_string(),
            VelocityWeights {
                slope: 0.4,
                hf: 0.6,
                energy: 0.0,
            },
        );
        weights.insert(
            "hat_open".to_string(),
            VelocityWeights {
                slope: 0.4,
                hf: 0.6,
                energy: 0.0,
            },
        );
        weights.insert(
            "tom_low".to_string(),
            VelocityWeights {
                slope: 0.5,
                hf: 0.2,
                energy: 0.3,
            },
        );
        weights.insert(
            "tom_mid".to_string(),
            VelocityWeights {
                slope: 0.5,
                hf: 0.2,
                energy: 0.3,
            },
        );
        weights.insert(
            "tom_high".to_string(),
            VelocityWeights {
                slope: 0.5,
                hf: 0.2,
                energy: 0.3,
            },
        );

        Self {
            gamma: 0.7,
            min_samples_for_normalization: 10,
            weights,
            context: VelocityContextRules::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityWeights {
    pub slope: f32,
    pub hf: f32,
    pub energy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VelocityContextRules {
    pub snare_backbeat_boost: f32,
    pub hat_pair_delta: i32,
    pub kick_double_delta: i32,
}

impl Default for VelocityContextRules {
    fn default() -> Self {
        Self {
            snare_backbeat_boost: 1.2,
            hat_pair_delta: 12,
            kick_double_delta: 10,
        }
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantizeConfig {
    pub strength: f32,
    pub max_ms: f32,
    pub preserve_flams: bool,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            strength: 0.5,
            max_ms: 12.0,
            preserve_flams: true,
        }
    }
}

/// Humanization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HumanizeConfig {
    pub enabled: bool,
    pub hat_jitter_ms: f32,
    pub long_run_threshold: usize,
    pub apply_to: Vec<String>,
}

impl Default for HumanizeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hat_jitter_ms: 3.0,
            long_run_threshold: 8,
            apply_to: vec!["hat_closed".to_string(), "hat_open".to_string()],
        }
    }
}

/// Post-processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PostProcessConfig {
    pub masking_window_ms: f32,
    pub snare_masking_threshold: i32,
    pub hat_hf_escape_ratio_min: f32,
    pub acoustic_escape_threshold: f32,
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self {
            masking_window_ms: 8.0,
            snare_masking_threshold: 90,
            hat_hf_escape_ratio_min: 0.70,
            acoustic_escape_threshold: 0.7,
        }
    }
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportConfig {
    pub ticks_per_beat: u32,
    pub channel: u8,
    pub emit_tempo_map: bool,
    pub max_cymbal_decay_sec: f32,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            ticks_per_beat: 480,
            channel: 9,
            emit_tempo_map: true,
            max_cymbal_decay_sec: 8.0,
        }
    }
}

/// Convergence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConvergenceConfig {
    pub max_iterations: usize,
    pub delta_f1_threshold: f32,
    pub method: String,
    pub confidence_weights: ConfidenceWeights,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            max_iterations: 2,
            delta_f1_threshold: 0.02,
            method: "weighted_log_sum".to_string(),
            confidence_weights: ConfidenceWeights::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConfidenceWeights {
    pub acoustic: f32,
    pub prior: f32,
}

impl Default for ConfidenceWeights {
    fn default() -> Self {
        Self {
            acoustic: 0.7,
            prior: 0.3,
        }
    }
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationConfig {
    pub tempo_range: [f32; 2],
    pub lambda_sum_tolerance: f32,
    pub min_events_per_track: usize,
    pub max_events_per_track: usize,
    pub min_beat_tracking_f1: f32,
    pub schema: ValidationSchema,
    // Drum stem validation
    pub harmonic_content_threshold: f32,
    pub melodic_frequency_threshold: f32,
    pub sustain_threshold: f32,
    pub percussion_score_min: f32,
    pub drum_stem_confidence_min: f32,
    pub max_issues_allowed: usize,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            tempo_range: [40.0, 240.0],
            lambda_sum_tolerance: 0.05,
            min_events_per_track: 10,
            max_events_per_track: 10000,
            min_beat_tracking_f1: 0.60,
            schema: ValidationSchema::default(),
            // Drum stem validation defaults
            harmonic_content_threshold: 0.3,
            melodic_frequency_threshold: 0.4,
            sustain_threshold: 0.25,
            percussion_score_min: 0.2,
            drum_stem_confidence_min: 0.6,
            max_issues_allowed: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidationSchema {
    pub enabled: bool,
    pub strict_mode: bool,
}

impl Default for ValidationSchema {
    fn default() -> Self {
        Self {
            enabled: true,
            strict_mode: false,
        }
    }
}

/// Edge cases configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct EdgeCasesConfig {
    pub blast_beats: BlastBeatsConfig,
    pub odd_meters: OddMetersConfig,
    pub electronic_drums: ElectronicDrumsConfig,
    pub extreme_rubato: ExtremeRubatoConfig,
    pub sparse_patterns: SparsePatternsConfig,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BlastBeatsConfig {
    pub refractory_ms_floor: f32,
    pub hat_pair_delta: i32,
}

impl Default for BlastBeatsConfig {
    fn default() -> Self {
        Self {
            refractory_ms_floor: 8.0,
            hat_pair_delta: 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OddMetersConfig {
    pub enabled: bool,
    pub candidates: Vec<String>,
}

impl Default for OddMetersConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            candidates: vec![
                "2/4".to_string(),
                "3/4".to_string(),
                "4/4".to_string(),
                "5/4".to_string(),
                "7/8".to_string(),
                "12/8".to_string(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ElectronicDrumsConfig {
    pub enabled: bool,
    pub refractory_ms_floor: f32,
    pub kick_attack_ms: f32,
}

impl Default for ElectronicDrumsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            refractory_ms_floor: 6.0,
            kick_attack_ms: 3.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExtremeRubatoConfig {
    pub max_drift_percent: f32,
    pub smoothing_window_bars: usize,
}

impl Default for ExtremeRubatoConfig {
    fn default() -> Self {
        Self {
            max_drift_percent: 10.0,
            smoothing_window_bars: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SparsePatternsConfig {
    pub boost_factor: f32,
    pub density_percentile_threshold: f32,
    pub min_hits_per_bar: usize,
}

impl Default for SparsePatternsConfig {
    fn default() -> Self {
        Self {
            boost_factor: 1.5,
            density_percentile_threshold: 10.0,
            min_hits_per_bar: 1,
        }
    }
}

/// QA artifacts configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QaConfig {
    pub generate_images: bool,
    pub image_formats: Vec<String>,
    pub save_pass_diffs: bool,
    pub confidence_heatmap_resolution: [usize; 2],
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            generate_images: true,
            image_formats: vec!["png".to_string()],
            save_pass_diffs: true,
            confidence_heatmap_resolution: [100, 16],
        }
    }
}

/// Validate configuration parameters
pub fn validate_config(config: &Config) -> anyhow::Result<()> {
    // Validate tempo range
    if config.validation.tempo_range[0] >= config.validation.tempo_range[1] {
        anyhow::bail!("tempo_range min must be < max");
    }

    // Normalize onset fusion weights
    let mut weights = config.onset_fusion.weights.clone();
    let total = weights.flux + weights.complex_flux + weights.high;
    if (total - 1.0).abs() > 0.01 {
        weights.flux /= total;
        weights.complex_flux /= total;
        weights.high /= total;
    }

    // Normalize grid lambda values
    let la = config.grid.lambda_acoustic;
    let lp = config.grid.lambda_prior;
    let total = la + lp;
    if (total - 1.0).abs() > 0.01 {
        // Note: This would modify the config, but we validate instead
        // In a real implementation, we'd return a normalized config
    }

    Ok(())
}

/// Load configuration from JSON file
pub fn load_config<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Config> {
    let content = std::fs::read_to_string(path)?;
    let config: Config = serde_json::from_str(&content)?;
    validate_config(&config)?;
    Ok(config)
}

/// Save configuration to JSON file
pub fn save_config<P: AsRef<std::path::Path>>(config: &Config, path: P) -> anyhow::Result<()> {
    let content = serde_json::to_string_pretty(config)?;
    std::fs::write(path, content)?;
    Ok(())
}

fn default_tempo_weights() -> HashMap<String, f32> {
    let mut weights = HashMap::new();
    weights.insert("kick".to_string(), 1.0);
    weights.insert("snare".to_string(), 1.0);
    weights.insert("rimshot".to_string(), 1.0); // Rimshots weighted like snare for tempo
    weights.insert("tom_low".to_string(), 0.6);
    weights.insert("tom_mid".to_string(), 0.6);
    weights.insert("tom_high".to_string(), 0.6);
    weights.insert("hat_closed".to_string(), 0.3);
    weights.insert("hat_open".to_string(), 0.3);
    weights.insert("splash".to_string(), 0.3); // Splash weighted like hats
    weights.insert("cowbell".to_string(), 0.4); // Cowbell has moderate tempo importance
    weights.insert("crash".to_string(), 0.2);
    weights.insert("ride".to_string(), 0.2);
    weights
}
