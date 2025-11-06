//! Pass 0: Preflight & Normalization

use crate::audio::{apply_gain, measure_lufs, true_peak_limiter, AudioState};
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};
use crate::spectral::{magnitude_spectrogram, stft, inverse_stft};
use ndarray::Array2;

/// Drum stem validation result
#[derive(Debug)]
struct DrumStemValidation {
    is_drum_stem: bool,
    confidence: f32,
    issues: Vec<String>,
}

/// Validate that input is a drum stem, not a full mix
fn validate_drum_stem(
    audio: &[f32],
    sr: u32,
    config: &Config,
) -> DrumErrorResult<DrumStemValidation> {
    let mut issues = Vec::new();
    let mut confidence = 1.0;

    // Analyze only first 30 seconds to avoid excessive computation on large files
    let max_samples = (30.0 * sr as f32) as usize;
    let analysis_audio = if audio.len() > max_samples {
        &audio[0..max_samples]
    } else {
        audio
    };

    // Basic spectral analysis
    let stft_data = stft(analysis_audio, 2048, 512, "hann", sr);
    let mag_spec = magnitude_spectrogram(&stft_data);

    // 1. Check for melodic/harmonic content (piano, guitar, vocals)
    let harmonic_content = detect_harmonic_content(&mag_spec, &stft_data.freqs);
    if harmonic_content > config.validation.harmonic_content_threshold {
        issues.push(format!(
            "High harmonic content detected ({:.2}), suggests melodic instruments",
            harmonic_content
        ));
        confidence *= (1.0 - harmonic_content).max(0.1);
    }

    // 2. Check frequency range distribution
    let freq_distribution = analyze_frequency_distribution(&mag_spec, &stft_data.freqs);
    if freq_distribution.melodic_ratio > config.validation.melodic_frequency_threshold {
        issues.push(format!(
            "High melodic frequency content ({:.2}), suggests vocals/strings",
            freq_distribution.melodic_ratio
        ));
        confidence *= (1.0 - freq_distribution.melodic_ratio).max(0.1);
    }

    // 3. Check for sustained tones (non-percussive)
    let sustain_ratio = detect_sustained_tones(&mag_spec);
    if sustain_ratio > config.validation.sustain_threshold {
        issues.push(format!(
            "High sustained tone content ({:.2}), suggests sustained instruments",
            sustain_ratio
        ));
        confidence *= (1.0 - sustain_ratio).max(0.1);
    }

    // 4. Check percussion characteristics
    let percussion_score = analyze_percussion_characteristics(&mag_spec, &stft_data.freqs);
    if percussion_score < config.validation.percussion_score_min {
        issues.push(format!(
            "Low percussion characteristics ({:.2}), may not be drum-focused",
            percussion_score
        ));
        confidence *= percussion_score.max(0.1);
    }

    let is_drum_stem = confidence >= config.validation.drum_stem_confidence_min
        && issues.len() <= config.validation.max_issues_allowed;

    Ok(DrumStemValidation {
        is_drum_stem,
        confidence,
        issues,
    })
}

/// Detect harmonic/melodic content (piano, guitar, vocals)
fn detect_harmonic_content(mag_spec: &Array2<f32>, freqs: &[f32]) -> f32 {
    // Look for strong harmonics in melodic frequency ranges
    let mut harmonic_energy = 0.0;
    let mut total_energy = 0.0;

    for f_idx in 0..mag_spec.shape()[0] {
        let freq_energy: f32 = mag_spec.row(f_idx).iter().map(|&x| x * x).sum();
        total_energy += freq_energy;

        let freq = freqs[f_idx];

        // Check for melodic frequency ranges with strong harmonics
        if (80.0..1000.0).contains(&freq) {
            // Look for harmonic series patterns
            let fundamental = freq;
            let mut harmonic_strength = 0.0;

            // Check for 2nd, 3rd, 4th harmonics
            for harmonic in 2..=4 {
                let harmonic_freq = fundamental * harmonic as f32;
                if let Some(h_idx) = find_closest_freq_index(harmonic_freq, freqs) {
                    if h_idx < mag_spec.shape()[0] {
                        let h_energy: f32 = mag_spec.row(h_idx).iter().map(|&x| x * x).sum();
                        harmonic_strength += h_energy;
                    }
                }
            }

            if harmonic_strength > freq_energy * 0.3 {
                harmonic_energy += freq_energy;
            }
        }
    }

    if total_energy > 0.0 {
        harmonic_energy / total_energy
    } else {
        0.0
    }
}

/// Analyze frequency distribution for melodic vs percussive content
fn analyze_frequency_distribution(mag_spec: &Array2<f32>, freqs: &[f32]) -> FrequencyDistribution {
    let mut melodic_energy = 0.0;
    let _percussive_energy = 0.0;
    let mut total_energy = 0.0;

    for f_idx in 0..mag_spec.shape()[0] {
        let freq_energy: f32 = mag_spec.row(f_idx).iter().map(|&x| x * x).sum();
        total_energy += freq_energy;

        let freq = freqs[f_idx];

        if (200.0..4000.0).contains(&freq) {
            // Melodic range (vocals, strings, piano mid/high)
            melodic_energy += freq_energy;
        } else if (40.0..200.0).contains(&freq) || freq > 4000.0 {
            // Percussive range (kick, toms, cymbals)
            // _percussive_energy += freq_energy; // Not currently used
        }
    }

    let melodic_ratio = if total_energy > 0.0 {
        melodic_energy / total_energy
    } else {
        0.0
    };

    FrequencyDistribution { melodic_ratio }
}

/// Detect sustained tones (non-percussive)
fn detect_sustained_tones(mag_spec: &Array2<f32>) -> f32 {
    let n_frames = mag_spec.shape()[1];
    if n_frames < 10 {
        return 0.0;
    }

    let mut sustained_energy = 0.0;
    let mut total_energy = 0.0;

    // Check for frequencies that remain strong over time
    for f_idx in 0..mag_spec.shape()[0] {
        let freq_energy: Vec<f32> = mag_spec.row(f_idx).iter().map(|&x| x * x).collect();
        total_energy += freq_energy.iter().sum::<f32>();

        // Calculate autocorrelation to detect periodicity
        let mut sustained = false;
        let window_size = (n_frames / 4).max(10).min(n_frames);

        for start in 0..(n_frames - window_size) {
            let window: &[f32] = &freq_energy[start..start + window_size];
            let mean_energy = window.iter().sum::<f32>() / window.len() as f32;
            let variation =
                window.iter().map(|&x| (x - mean_energy).abs()).sum::<f32>() / window.len() as f32;

            if mean_energy > 0.01 && variation / mean_energy < 0.5 {
                sustained = true;
                break;
            }
        }

        if sustained {
            sustained_energy += freq_energy.iter().sum::<f32>();
        }
    }

    if total_energy > 0.0 {
        sustained_energy / total_energy
    } else {
        0.0
    }
}

/// Analyze percussion characteristics
fn analyze_percussion_characteristics(mag_spec: &Array2<f32>, freqs: &[f32]) -> f32 {
    let mut percussion_score = 0.0;
    let mut total_frames = 0;

    // Look for transient, broadband events typical of percussion
    for frame_idx in 0..mag_spec.shape()[1] {
        let frame: Vec<f32> = mag_spec.column(frame_idx).iter().map(|&x| x * x).collect();
        let total_frame_energy: f32 = frame.iter().sum();

        if total_frame_energy < 1e-6 {
            continue;
        }

        total_frames += 1;

        // Check for broadband energy (cymbals, snares)
        let high_freq_energy: f32 = freqs
            .iter()
            .zip(&frame)
            .filter(|(&f, _)| f > 2000.0)
            .map(|(_, &e)| e)
            .sum();

        let broadband_ratio = high_freq_energy / total_frame_energy;

        // Check for low-frequency transients (kick)
        let low_freq_energy: f32 = freqs
            .iter()
            .zip(&frame)
            .filter(|(&f, _)| (40.0..150.0).contains(&f))
            .map(|(_, &e)| e)
            .sum();

        let low_transient_ratio = low_freq_energy / total_frame_energy;

        // Percussion typically has either broadband or strong low-frequency content
        let frame_score = broadband_ratio.max(low_transient_ratio);
        percussion_score += frame_score;
    }

    if total_frames > 0 {
        percussion_score / total_frames as f32
    } else {
        0.0
    }
}

/// Find closest frequency index
fn find_closest_freq_index(target_freq: f32, freqs: &[f32]) -> Option<usize> {
    freqs
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| {
            (a - target_freq)
                .abs()
                .partial_cmp(&(b - target_freq).abs())
                .unwrap()
        })
        .map(|(idx, _)| idx)
}

#[derive(Debug)]
struct FrequencyDistribution {
    melodic_ratio: f32,
}

/// Apply HPSS (Harmonic/Percussive Source Separation)
fn apply_hpss(audio: &[f32], sr: u32, _style: &str, _config: &Config) -> Vec<f32> {
    // Get HPSS parameters from config (using defaults since config doesn't have these fields yet)
    let n_fft = 2048; // Default FFT size for HPSS
    let hop_length = 512; // Default hop length
    let beta = 2.0; // Default beta parameter
    let harmonic_kernel_size = 31; // Default horizontal kernel size (harmonic)
    let percussive_kernel_size = 31; // Default vertical kernel size (percussive)

    // Compute STFT
    let stft_data = stft(audio, n_fft, hop_length, "hann", sr);
    let mut s = stft_data.s.clone();

    // Compute magnitude spectrogram
    let mag = magnitude_spectrogram(&stft_data);

    // Apply median filtering for harmonic (horizontal) and percussive (vertical) separation
    let (harmonic_mask, percussive_mask) = compute_hpss_masks(&mag, harmonic_kernel_size, percussive_kernel_size, beta);

    // Apply masks to complex STFT
    for i in 0..s.shape()[0] {
        for j in 0..s.shape()[1] {
            let mag_val = mag[[i, j]];
            if mag_val > 0.0 {
                let scale = if percussive_mask[[i, j]] > harmonic_mask[[i, j]] {
                    percussive_mask[[i, j]]
                } else {
                    harmonic_mask[[i, j]]
                };
                s[[i, j]] *= scale;
            }
        }
    }

    // Reconstruct audio using inverse STFT
    inverse_stft(&s, n_fft, hop_length, "hann", sr)
}

/// Compute HPSS masks using median filtering
fn compute_hpss_masks(
    mag: &Array2<f32>,
    harmonic_kernel_size: usize,
    percussive_kernel_size: usize,
    beta: f32,
) -> (Array2<f32>, Array2<f32>) {
    let (n_freq, n_time) = mag.dim();

    // Initialize masks
    let mut harmonic_mask = Array2::<f32>::zeros((n_freq, n_time));
    let mut percussive_mask = Array2::<f32>::zeros((n_freq, n_time));

    // Apply median filtering
    for i in 0..n_freq {
        for j in 0..n_time {
            // Harmonic mask: horizontal median (time direction)
            let harmonic_median = median_filter_time(mag, i, j, harmonic_kernel_size);
            harmonic_mask[[i, j]] = (harmonic_median / (harmonic_median + mag[[i, j]] + 1e-8)).powf(beta);

            // Percussive mask: vertical median (frequency direction)
            let percussive_median = median_filter_freq(mag, i, j, percussive_kernel_size);
            percussive_mask[[i, j]] = (percussive_median / (percussive_median + mag[[i, j]] + 1e-8)).powf(beta);
        }
    }

    (harmonic_mask, percussive_mask)
}

/// Median filter in time direction (horizontal)
fn median_filter_time(mag: &Array2<f32>, freq_idx: usize, time_idx: usize, kernel_size: usize) -> f32 {
    let n_time = mag.shape()[1];
    let half_kernel = kernel_size / 2;

    let start = time_idx.saturating_sub(half_kernel);
    let end = (time_idx + half_kernel + 1).min(n_time);

    let mut values: Vec<f32> = (start..end)
        .map(|t| mag[[freq_idx, t]])
        .collect();

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    values[mid]
}

/// Median filter in frequency direction (vertical)
fn median_filter_freq(mag: &Array2<f32>, freq_idx: usize, time_idx: usize, kernel_size: usize) -> f32 {
    let n_freq = mag.shape()[0];
    let half_kernel = kernel_size / 2;

    let start = freq_idx.saturating_sub(half_kernel);
    let end = (freq_idx + half_kernel + 1).min(n_freq);

    let mut values: Vec<f32> = (start..end)
        .map(|f| mag[[f, time_idx]])
        .collect();

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    values[mid]
}



pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 0: Preflight & Normalization");

    // 1. Validate that this is actually a drum stem
    println!("  Validating drum stem content...");
    let validation = validate_drum_stem(&state.y, state.sr, config)?;

    if !validation.is_drum_stem {
        eprintln!("⚠️  WARNING: Input may not be an isolated drum stem!");
        eprintln!("   Confidence: {:.2}", validation.confidence);
        for issue in &validation.issues {
            eprintln!("   - {}", issue);
        }

        if validation.confidence < config.validation.drum_stem_confidence_min / 2.0 {
            return Err(DrumError::InputValidationError(
                "Input appears to be a full mix rather than isolated drums. This system requires isolated drum stems.".to_string()
            ));
        }
    } else {
        println!(
            "  ✓ Drum stem validation passed (confidence: {:.2})",
            validation.confidence
        );
    }

    // 2. Apply LUFS normalization
    println!("  Applying LUFS normalization...");
    let current_lufs = measure_lufs(&state.y, state.sr);
    let gain_db = config.audio.target_lufs - current_lufs;
    let mut processed = apply_gain(&state.y, gain_db);

    // 3. Apply true peak limiting
    println!("  Applying true peak limiting...");
    processed = true_peak_limiter(&processed, config.audio.true_peak_limit_db);

    // 4. Apply HPSS (placeholder for now)
    println!("  Applying HPSS...");
    let style = "default"; // TODO: Get from user hints
    processed = apply_hpss(&processed, state.sr, style, config);

    // Store processed audio
    state.y_processed = Some(processed);

    // 5. Handle stereo if needed
    if state.y.len().is_multiple_of(2) {
        // Check if we have stereo data (this is a simplification)
        // In practice, we'd need to track channel count from loading
        println!("  Stereo processing...");
        // For now, assume mono conversion already happened in audio loading
    }

    println!("  ✓ Pass 0 complete");
    Ok(())
}
