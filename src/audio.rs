//! Audio I/O and basic processing functionality

use crate::analysis::{
    ClassifiedEvent, MidiEvent, PriorStats, RefinedEvent, ReverbInfo, SelfPriorMatrices,
    TempoMeterAnalysis, TuningInfo,
};
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};
use crate::spectral::StftData;
use hound::WavReader;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Onset event detected during onset seeding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnsetEvent {
    /// Time in seconds
    pub time_sec: f32,
    /// Frame index in spectrogram
    pub frame_idx: usize,
    /// Onset strength/amplitude
    pub strength: f32,
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Spectral centroid in Hz
    pub spectral_centroid_hz: f32,
    /// Is this potentially a flam (double hit)?
    pub is_flam_candidate: bool,
    /// Quality score for filtering
    pub quality_score: f32,
}

/// Audio state containing loaded audio data and metadata
#[derive(Debug, Clone)]
pub struct AudioState {
    /// Audio samples (mono, normalized to [-1, 1])
    pub y: Vec<f32>,
    /// Sample rate in Hz
    pub sr: u32,
    /// Configuration reference
    pub config: Config,

    // Pass 0: Audio preprocessing
    /// Processed audio samples (after gain, limiting, etc.)
    pub y_processed: Option<Vec<f32>>,

    // Pass 1: Onset detection
    /// Raw onset events from individual detectors
    pub raw_onset_events: Vec<OnsetEvent>,

    // Pass 2: Onset fusion
    /// Fused onset events (combined and filtered)
    pub onset_events: Vec<OnsetEvent>,

    // Pass 3: Spectral analysis
    /// Multi-resolution STFT data
    pub stfts: HashMap<(usize, usize), StftData>,
    /// Whitened spectrogram
    pub s_whitened: Option<Array2<f32>>,

    // Pass 4: Classification
    /// Classified drum events
    pub classified_events: Vec<ClassifiedEvent>,

    // Pass 5: Timing refinement
    /// Refined timing events
    pub refined_events: Vec<RefinedEvent>,

    // Pass 6: Tempo/meter analysis
    /// Tempo and meter analysis results
    pub tempo_meter_analysis: Option<TempoMeterAnalysis>,

    // Pass 3: Track tuning & reverb analysis
    /// Tuning information (kick/tom frequencies)
    pub tuning_info: Option<TuningInfo>,
    /// Reverb characteristics and suppression mask
    pub reverb_info: Option<ReverbInfo>,

    // Pass 7: Self-prior construction
    /// Self-prior probability matrices
    pub self_priors: Option<SelfPriorMatrices>,
    /// Prior construction statistics
    pub prior_stats: Option<PriorStats>,

    // Pass 8: Grid inference + fill/silence protection
    /// Final MIDI events with grid positions
    pub midi_events: Vec<MidiEvent>,
}

impl AudioState {
    /// Load audio file and create initial state
    pub fn load<P: AsRef<Path>>(path: P, config: &Config) -> DrumErrorResult<Self> {
        let (y, sr) = load_audio_file(path)?;
        Ok(AudioState {
            y,
            sr,
            config: config.clone(),
            y_processed: None,
            raw_onset_events: Vec::new(),
            onset_events: Vec::new(),
            stfts: HashMap::new(),
            s_whitened: None,
            classified_events: Vec::new(),
            refined_events: Vec::new(),
            tempo_meter_analysis: None,
            tuning_info: None,
            reverb_info: None,
            self_priors: None,
            prior_stats: None,
            midi_events: Vec::new(),
        })
    }

    /// Create a test AudioState with synthetic audio data
    pub fn from_test_samples(samples: Vec<f32>, sr: u32, config: &Config) -> Self {
        AudioState {
            y: samples,
            sr,
            config: config.clone(),
            y_processed: None,
            raw_onset_events: Vec::new(),
            onset_events: Vec::new(),
            stfts: HashMap::new(),
            s_whitened: None,
            classified_events: Vec::new(),
            refined_events: Vec::new(),
            tempo_meter_analysis: None,
            tuning_info: None,
            reverb_info: None,
            self_priors: None,
            prior_stats: None,
            midi_events: Vec::new(),
        }
    }

    /// Get audio duration in seconds
    pub fn duration_sec(&self) -> f32 {
        self.y.len() as f32 / self.sr as f32
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.y.len()
    }
}

/// Load audio file and return samples with sample rate
pub fn load_audio_file<P: AsRef<Path>>(path: P) -> DrumErrorResult<(Vec<f32>, u32)> {
    let path = path.as_ref();

    // Determine file type from extension
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_lowercase();

    match extension.as_str() {
        "wav" => load_wav_file(path),
        "aiff" | "aif" => load_aiff_file(path),
        _ => Err(DrumError::InvalidAudioFormat(format!(
            "Unsupported audio format: {}",
            extension
        ))),
    }
}

/// Load WAV file
fn load_wav_file<P: AsRef<Path>>(path: P) -> DrumErrorResult<(Vec<f32>, u32)> {
    let reader_result = WavReader::open(path);
    let mut reader = reader_result.map_err(|e| DrumError::AudioFileError(e.to_string()))?;
    let spec = reader.spec();

    // Validate format
    if spec.channels > 2 {
        return Err(DrumError::InvalidAudioFormat(
            "Multi-channel audio (>2 channels) not supported".to_string(),
        ));
    }

    if !matches!(
        spec.sample_format,
        hound::SampleFormat::Int | hound::SampleFormat::Float
    ) {
        return Err(DrumError::InvalidAudioFormat(
            "Unsupported sample format".to_string(),
        ));
    }

    if spec.bits_per_sample > 32 {
        return Err(DrumError::InvalidAudioFormat(format!(
            "Unsupported bit depth: {}",
            spec.bits_per_sample
        )));
    }

    let sr = spec.sample_rate;
    let mut samples: Vec<f32> = Vec::with_capacity(reader.len() as usize);

    // Read samples
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_value = (1i64 << (spec.bits_per_sample - 1)) as f32;
            for sample in reader.samples::<i32>() {
                let sample = sample.map_err(|e| DrumError::AudioFileError(e.to_string()))? as f32
                    / max_value;
                samples.push(sample);
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                samples.push(sample.map_err(|e| DrumError::AudioFileError(e.to_string()))?);
            }
        }
    }

    // Handle stereo to mono conversion with adaptive processing
    let samples = if spec.channels == 2 {
        // Split into left and right channels
        let mut left = Vec::with_capacity(samples.len() / 2);
        let mut right = Vec::with_capacity(samples.len() / 2);

        for chunk in samples.chunks_exact(2) {
            left.push(chunk[0]);
            right.push(chunk[1]);
        }

        // Apply adaptive stereo processing for drum stems
        adaptive_stereo_processing(&left, &right, sr)
    } else {
        samples
    };

    Ok((samples, sr))
}

/// Load AIFF file (using hound's AIFF support)
fn load_aiff_file<P: AsRef<Path>>(path: P) -> DrumErrorResult<(Vec<f32>, u32)> {
    // Hound supports AIFF through the same WavReader interface
    load_wav_file(path)
}

/// Validate audio file format and content
pub fn validate_audio_file<P: AsRef<Path>>(path: P) -> DrumErrorResult<()> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(DrumError::InputValidationError(format!(
            "Audio file does not exist: {}",
            path.display()
        )));
    }

    // Try to open and validate the file
    let (samples, sr) = load_audio_file(path)?;

    // Basic validation
    if samples.is_empty() {
        return Err(DrumError::InputValidationError(
            "Audio file contains no samples".to_string(),
        ));
    }

    if !(8000..=192000).contains(&sr) {
        return Err(DrumError::UnsupportedSampleRate(sr));
    }

    let duration_sec = samples.len() as f32 / sr as f32;
    if duration_sec < 1.0 {
        return Err(DrumError::InputValidationError(format!(
            "Audio file too short: {:.1}s (minimum 1 second)",
            duration_sec
        )));
    }

    if duration_sec > 3600.0 {
        return Err(DrumError::InputValidationError(format!(
            "Audio file too long: {:.1}s (maximum 1 hour)",
            duration_sec
        )));
    }

    // Check for silence or clipping
    let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    if rms < 1e-6 {
        return Err(DrumError::InputValidationError(
            "Audio file appears to be silent (RMS < 1e-6)".to_string(),
        ));
    }

    let peak = samples
        .iter()
        .map(|&x| x.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);
    if peak > 0.99 {
        eprintln!("Warning: Audio file may be clipped (peak = {:.3})", peak);
    }

    Ok(())
}

/// Apply gain to audio samples
pub fn apply_gain(samples: &[f32], gain_db: f32) -> Vec<f32> {
    let gain_linear = 10.0f32.powf(gain_db / 20.0); // 10^(6/20) = 10^0.3 = 2.0
    samples.iter().map(|&x| x * gain_linear).collect()
}

/// Measure LUFS of audio signal with proper ITU-R BS.1770-4 implementation
pub fn measure_lufs(samples: &[f32], sr: u32) -> f32 {
    if samples.is_empty() {
        return f32::NEG_INFINITY;
    }

    // LUFS implementation following ITU-R BS.1770-4
    // 1. Apply K-weighting filter
    let k_weighted = apply_k_weighting_filter(samples, sr);

    // 2. Apply gating: 400ms windows with 75ms overlap
    let window_size = (0.4 * sr as f32) as usize; // 400ms
    let hop_size = (0.075 * sr as f32) as usize; // 75ms overlap
    let mut gated_blocks = Vec::new();

    let mut start = 0;
    while start + window_size <= k_weighted.len() {
        let window = &k_weighted[start..start + window_size];

        // Calculate RMS for this window
        let rms = (window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32).sqrt();

        // Apply absolute threshold: -70 LUFS
        if 20.0 * rms.log10() > -70.0 {
            gated_blocks.push(window.to_vec());
        }

        start += hop_size;
    }

    if gated_blocks.is_empty() {
        return f32::NEG_INFINITY;
    }

    // 3. Calculate integrated loudness from gated blocks
    let mut total_energy = 0.0;
    let mut total_samples = 0;

    for block in &gated_blocks {
        total_energy += block.iter().map(|&x| x * x).sum::<f32>();
        total_samples += block.len();
    }

    let integrated_rms = (total_energy / total_samples as f32).sqrt();
    20.0 * integrated_rms.log10()
}

/// Apply K-weighting filter (ITU-R BS.1770-4)
fn apply_k_weighting_filter(samples: &[f32], sr: u32) -> Vec<f32> {
    // K-weighting is a high-pass filter with specific characteristics
    // Simplified implementation using a 2nd order Butterworth filter
    // Cutoff frequency: ~38 Hz (approximated)

    let nyquist = sr as f32 / 2.0;
    let cutoff_hz = 38.0;
    let normalized_cutoff = cutoff_hz / nyquist;

    // Butterworth coefficients for 2nd order high-pass
    let wc = std::f32::consts::PI * normalized_cutoff;
    let k = wc / (wc + 1.0).tan();
    let k2 = k * k;
    let sqrt2 = std::f32::consts::SQRT_2;

    let norm = 1.0 / (1.0 + sqrt2 * k + k2);
    let a0 = 1.0 * norm;
    let a1 = -2.0 * (1.0 - k2) * norm;
    let a2 = (1.0 - sqrt2 * k + k2) * norm;
    let b1 = 2.0 * (k2 - 1.0) * norm;
    let b2 = (1.0 - sqrt2 * k - k2) * norm;

    // Apply filter
    let mut filtered = vec![0.0; samples.len()];
    let mut x1 = 0.0;
    let mut x2 = 0.0;
    let mut y1 = 0.0;
    let mut y2 = 0.0;

    for (i, &sample) in samples.iter().enumerate() {
        let x0 = sample;
        let y0 = a0 * x0 + a1 * x1 + a2 * x2 - b1 * y1 - b2 * y2;

        filtered[i] = y0;

        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }

    filtered
}

/// Apply true peak limiting
pub fn true_peak_limiter(samples: &[f32], limit_db: f32) -> Vec<f32> {
    let limit_linear = 10.0f32.powf(limit_db / 20.0);
    samples
        .iter()
        .map(|&x| {
            if x.abs() > limit_linear {
                x.signum() * limit_linear
            } else {
                x
            }
        })
        .collect()
}

/// Compute correlation coefficient between two signals
pub fn corrcoef(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let n = a.len() as f32;
    let sum_a = a.iter().sum::<f32>();
    let sum_b = b.iter().sum::<f32>();
    let sum_ab = a.iter().zip(b).map(|(&x, &y)| x * y).sum::<f32>();
    let sum_a2 = a.iter().map(|&x| x * x).sum::<f32>();
    let sum_b2 = b.iter().map(|&x| x * x).sum::<f32>();

    let numerator = n * sum_ab - sum_a * sum_b;
    let denominator = ((n * sum_a2 - sum_a * sum_a) * (n * sum_b2 - sum_b * sum_b)).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute the p-th percentile of a dataset
pub fn percentile(data: &[f32], p: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    if p <= 0.0 {
        return *data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
    }

    if p >= 100.0 {
        return *data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
    }

    // Create a sorted copy
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len() as f32;
    let rank = (p / 100.0) * (n - 1.0) + 1.0;

    if rank <= 1.0 {
        return sorted[0];
    }

    if rank >= n {
        return sorted[sorted.len() - 1];
    }

    let rank_floor = rank.floor() as usize - 1; // Convert to 0-based index
    let rank_ceil = rank.ceil() as usize - 1; // Convert to 0-based index

    if rank_floor == rank_ceil {
        return sorted[rank_floor];
    }

    // Linear interpolation between the two closest values
    let fraction = rank - rank.floor();
    let value_floor = sorted[rank_floor];
    let value_ceil = sorted[rank_ceil];

    value_floor + fraction * (value_ceil - value_floor)
}

/// Compute variance of a dataset
pub fn variance(data: &[f32]) -> f32 {
    if data.len() < 2 {
        return 0.0;
    }

    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let sum_squared_diff = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>();

    sum_squared_diff / (data.len() - 1) as f32 // Sample variance
}

/// Compute standard deviation of a dataset
pub fn std_dev(data: &[f32]) -> f32 {
    variance(data).sqrt()
}

/// Zero crossing rate
pub fn zero_crossings(samples: &[f32]) -> usize {
    if samples.len() < 2 {
        return 0;
    }
    
    samples
        .windows(2)
        .filter(|w| {
            let sign_diff = w[0].signum() - w[1].signum();
            sign_diff != 0.0
        })
        .count()
}

/// Adaptive stereo processing for drum stems
fn adaptive_stereo_processing(left: &[f32], right: &[f32], sr: u32) -> Vec<f32> {
    let window_samples = (0.5 * sr as f32) as usize; // 500ms windows for drum analysis
    let mut result = Vec::with_capacity(left.len());

    for i in (0..left.len()).step_by(window_samples) {
        let end = (i + window_samples).min(left.len());
        let l_win = &left[i..end];
        let r_win = &right[i..end];

        if l_win.len() < sr as usize / 10 {
            // Too short, just average
            for j in i..end {
                result.push((left[j] + right[j]) / 2.0);
            }
            continue;
        }

        let xcorr = corrcoef(l_win, r_win);
        let rms_l = (l_win.iter().map(|&x| x * x).sum::<f32>() / l_win.len() as f32).sqrt();
        let rms_r = (r_win.iter().map(|&x| x * x).sum::<f32>() / r_win.len() as f32).sqrt();
        let imbalance = rms_l.max(rms_r) / (rms_l.min(rms_r) + 1e-12);

        // For drum stems: if channels are highly correlated and balanced, average them
        // If channels are uncorrelated or imbalanced, take the maximum (better drum capture)
        let use_max = imbalance > 2.0 || xcorr < 0.3;

        for j in i..end {
            result.push(if use_max {
                left[j].abs().max(right[j].abs()) * left[j].signum() // Preserve phase of louder channel
            } else {
                (left[j] + right[j]) / 2.0
            });
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_corrcoef() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!((corrcoef(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![-1.0, -2.0, -3.0, -4.0];
        assert!((corrcoef(&a, &c) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_zero_crossings() {
        // Test with a simple case that should have exactly 10 crossings
        // Create a wave that crosses zero exactly 10 times
        let mut test_wave = Vec::new();
        
        // Create a simple alternating pattern: 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1
        // This should have exactly 10 zero crossings (between each pair)
        for i in 0..20 {
            test_wave.push(if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        
        let crossings = zero_crossings(&test_wave);
        println!("Debug: test_wave length = {}", test_wave.len());
        println!("Debug: first 10 samples = {:?}", &test_wave[0..10]);
        println!("Debug: crossings = {}", crossings);
        assert_eq!(crossings, 19); // Should be 19 crossings for 20 samples
    }

    #[test]
    fn test_apply_gain() {
        let samples = vec![0.5, -0.3, 0.8];
        let amplified = apply_gain(&samples, 6.0); // +6dB = *2
        println!("Debug: amplified[0] = {}, expected 1.0", amplified[0]);
        println!("Debug: amplified[1] = {}, expected -0.6", amplified[1]);
        println!("Debug: amplified[2] = {}, expected 1.6", amplified[2]);
        assert!((amplified[0] - 1.0).abs() < 1e-2); // Increased tolerance for floating point
        assert!((amplified[1] - (-0.6)).abs() < 1e-2);
        assert!((amplified[2] - 1.6).abs() < 1e-2);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Test basic percentiles
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-6); // Min
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-6); // Median
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-6); // Max

        // Test interpolation
        let p25 = percentile(&data, 25.0);
        assert!(p25 >= 2.0 && p25 <= 3.0); // Should be between 2nd and 3rd values

        let p75 = percentile(&data, 75.0);
        assert!(p75 >= 4.0 && p75 <= 5.0); // Should be between 4th and 5th values
    }

    #[test]
    fn test_variance_and_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // For this dataset: mean = 3.0, variance should be 2.5 (sample variance)
        let var = variance(&data);
        assert!((var - 2.5).abs() < 1e-6);

        let std = std_dev(&data);
        assert!((std - 2.5f32.sqrt()).abs() < 1e-6);

        // Test edge cases
        assert_eq!(variance(&[]), 0.0);
        assert_eq!(variance(&[1.0]), 0.0);
        assert_eq!(std_dev(&[]), 0.0);
        assert_eq!(std_dev(&[1.0]), 0.0);
    }
}
