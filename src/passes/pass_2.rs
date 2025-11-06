//! Pass 2: High-Recall Onset Seeding

use crate::audio::{AudioState, OnsetEvent};
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};
use crate::spectral::StftData;
use ndarray::{s, Array2};

/// Compute spectral flux (positive differences only)
fn spectral_flux(mag: &Array2<f32>) -> Vec<f32> {
    let mut flux = vec![0.0; mag.shape()[1]];

    for t in 1..mag.shape()[1] {
        let mut frame_flux = 0.0;
        for f in 0..mag.shape()[0] {
            let diff = mag[[f, t]] - mag[[f, t - 1]];
            if diff > 0.0 {
                frame_flux += diff;
            }
        }
        flux[t] = frame_flux;
    }

    flux
}

/// Compute transient-focused flux emphasizing sharp attack characteristics
fn transient_flux(mag: &Array2<f32>, freqs: &[f32]) -> Vec<f32> {
    let mut flux = vec![0.0; mag.shape()[1]];
    
    // Focus on frequency bands where transients are most prominent
    let hf_start = freqs.iter().position(|&f| f >= 2000.0).unwrap_or(0);
    let vhf_start = freqs.iter().position(|&f| f >= 8000.0).unwrap_or(freqs.len());
    
    for t in 1..mag.shape()[1] {
        let mut frame_transient_flux = 0.0;
        
        // High frequency transients (2-8kHz) - rim shots, cymbal attacks
        for f in hf_start..vhf_start.min(mag.shape()[0]) {
            let diff = mag[[f, t]] - mag[[f, t - 1]];
            if diff > 0.0 {
                // Weight by frequency (higher frequencies get more weight for transients)
                let freq_weight = (freqs[f] / 1000.0).min(3.0);
                frame_transient_flux += diff * freq_weight;
            }
        }
        
        // Very high frequency transients (8kHz+) - splash, sharp attacks
        for f in vhf_start..mag.shape()[0] {
            let diff = mag[[f, t]] - mag[[f, t - 1]];
            if diff > 0.0 {
                // Strong weight for very high frequency transients
                let freq_weight = (freqs[f] / 2000.0).min(5.0);
                frame_transient_flux += diff * freq_weight;
            }
        }
        
        flux[t] = frame_transient_flux;
    }
    
    flux
}

/// Compute complex domain flux (preserves phase information)
fn complex_domain_flux(stft_data: &StftData) -> Vec<f32> {
    let mut flux = vec![0.0; stft_data.s.shape()[1]];

    for t in 1..stft_data.s.shape()[1] {
        let mut frame_flux = 0.0;
        for f in 0..stft_data.s.shape()[0] {
            let diff = (stft_data.s[[f, t]] - stft_data.s[[f, t - 1]]).norm();
            frame_flux += diff;
        }
        flux[t] = frame_flux;
    }

    flux
}

/// Compute band-limited flux for high frequencies (2-12kHz cymbal detection)
fn band_limited_flux(mag: &Array2<f32>, freqs: &[f32], low_freq: f32, high_freq: f32) -> Vec<f32> {
    // Find frequency indices for the band
    let start_idx = freqs.iter().position(|&f| f >= low_freq).unwrap_or(0);
    let end_idx = freqs
        .iter()
        .position(|&f| f >= high_freq)
        .unwrap_or(freqs.len());

    let band_mag = mag.slice(s![start_idx..end_idx, ..]).to_owned();
    spectral_flux(&band_mag)
}

/// Compute weighted envelope fusion of multiple onset signals
fn weighted_envelope_fusion(
    flux: &[f32],
    complex_flux: &[f32],
    high_flux: &[f32],
    transient_flux_signal: &[f32],
    weights: &(f32, f32, f32, f32),
) -> Vec<f32> {
    let mut fused = vec![0.0; flux.len()];

    for i in 0..flux.len() {
        fused[i] = weights.0 * flux[i] 
                 + weights.1 * complex_flux[i] 
                 + weights.2 * high_flux[i]
                 + weights.3 * transient_flux_signal[i];
    }

    fused
}

/// Compute adaptive threshold using rolling mean and std
fn adaptive_threshold(signal: &[f32], window_sec: f32, sr: u32, hop: usize, k: f32) -> Vec<f32> {
    let window_frames = (window_sec * sr as f32 / hop as f32) as usize;
    let mut thresholds = vec![0.0; signal.len()];

    for i in 0..signal.len() {
        let start = 0.max(i as i32 - window_frames as i32) as usize;
        let end = (signal.len()).min(i + window_frames + 1);

        let window: Vec<f32> = signal[start..end].to_vec();
        let mean = window.iter().sum::<f32>() / window.len() as f32;
        let variance =
            window.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / window.len() as f32;
        let std = variance.sqrt();

        thresholds[i] = mean + k * std;
    }

    thresholds
}

/// Find peaks with refractory period enforcement
fn find_peaks_with_refractory(
    signal: &[f32],
    thresholds: &[f32],
    min_distance_frames: usize,
) -> Vec<usize> {
    let mut peaks = Vec::new();
    let mut last_peak = 0;

    for i in 1..signal.len().saturating_sub(1) {
        if i < last_peak + min_distance_frames {
            continue;
        }

        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] && signal[i] > thresholds[i] {
            peaks.push(i);
            last_peak = i;
        }
    }

    peaks
}

/// Compute tempo-adaptive refractory period
fn tempo_adaptive_refractory(base_ms: f32, tempo_bpm: f32, sr: u32, hop: usize) -> usize {
    // Scale refractory period based on 16th note duration
    let sixteenth_note_sec = 60.0 / tempo_bpm / 4.0; // 16th note duration
    let refractory_sec = (base_ms / 1000.0).max(sixteenth_note_sec * 0.5); // Don't go below half a 16th note
    (refractory_sec * sr as f32 / hop as f32) as usize
}

/// Detect flam candidates using spectral centroid similarity
fn detect_flam_candidates(
    peaks: &[usize],
    mag: &Array2<f32>,
    freqs: &[f32],
    centroid_tolerance_hz: f32,
) -> Vec<bool> {
    let mut flam_candidates = vec![false; peaks.len()];

    for (i, &peak_idx) in peaks.iter().enumerate() {
        if i == 0 {
            continue;
        }

        let prev_peak_idx = peaks[i - 1];

        // Check if peaks are close enough to be potential flams (within 50ms)
        let time_diff_frames = peak_idx as i32 - prev_peak_idx as i32;
        if time_diff_frames > 50 {
            continue; // Too far apart
        }

        // Compute spectral centroids
        let centroid1 = compute_spectral_centroid(&mag.column(prev_peak_idx), freqs);
        let centroid2 = compute_spectral_centroid(&mag.column(peak_idx), freqs);

        // Check if centroids are similar (within tolerance)
        if (centroid1 - centroid2).abs() < centroid_tolerance_hz {
            flam_candidates[i] = true;
        }
    }

    flam_candidates
}

/// Compute spectral centroid
fn compute_spectral_centroid(mag_frame: &ndarray::ArrayView1<f32>, freqs: &[f32]) -> f32 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &mag) in mag_frame.iter().enumerate() {
        if i < freqs.len() {
            numerator += freqs[i] * mag;
            denominator += mag;
        }
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute SNR for onset quality assessment
fn compute_snr(signal: &[f32], peak_idx: usize, window_frames: usize) -> f32 {
    if peak_idx >= signal.len() {
        return 0.0;
    }

    let start = 0.max(peak_idx as i32 - window_frames as i32) as usize;
    let end = (signal.len()).min(peak_idx + window_frames + 1);

    let window: Vec<f32> = signal[start..end].to_vec();
    let peak_val = signal[peak_idx];

    // Simple SNR as peak value over mean of surrounding values
    let mean_surround = (window.iter().sum::<f32>() - peak_val) / (window.len() as f32 - 1.0);

    if mean_surround > 0.0 {
        20.0 * (peak_val / mean_surround).log10()
    } else {
        60.0 // High SNR if no surrounding noise
    }
}

/// Compute quality score for onset filtering
fn compute_quality_score(strength: f32, snr: f32, is_flam: bool) -> f32 {
    let mut score = strength * (snr / 20.0).min(3.0); // Cap SNR contribution

    if is_flam {
        score *= 0.7; // Reduce quality for flam candidates
    }

    score
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 2: High-Recall Onset Seeding");

    // Get whitened spectrogram from Pass 1
    let whitened = state.s_whitened.as_ref().ok_or_else(|| {
        DrumError::ProcessingPipelineError("Pass 1 must be run before Pass 2".to_string())
    })?;

    // Get primary STFT data
    let primary_stft = state
        .stfts
        .get(&(config.stft.n_fft, config.stft.hop_length))
        .ok_or_else(|| DrumError::ProcessingPipelineError("Primary STFT not found".to_string()))?;

    // Compute onset strength signals
    println!("  Computing onset strength signals...");

    // 1. Spectral flux
    let flux = spectral_flux(whitened);

    // 2. Complex domain flux
    let complex_flux = complex_domain_flux(primary_stft);

    // 3. High-frequency band-limited flux (2-12kHz for cymbals)
    let high_flux = band_limited_flux(whitened, &primary_stft.freqs, 2000.0, 12000.0);

    // 4. Transient-focused flux for sharp attack detection
    let transient_flux_signal = transient_flux(whitened, &primary_stft.freqs);

    // 5. Weighted envelope fusion
    let weights = (
        config.onset_fusion.weights.flux,
        config.onset_fusion.weights.complex_flux,
        config.onset_fusion.weights.high,
        config.onset_fusion.weights.transient,
    );
    let fused_onset = weighted_envelope_fusion(&flux, &complex_flux, &high_flux, &transient_flux_signal, &weights);

    // 5. Adaptive thresholding
    println!("  Computing adaptive thresholds...");
    let thresholds = adaptive_threshold(
        &fused_onset,
        config.thresholds.adaptive_window_sec,
        state.sr,
        config.stft.hop_length,
        config.thresholds.k_global,
    );

    // 6. Tempo-adaptive peak picking
    println!("  Finding onset peaks...");
    let tempo_bpm = 120.0; // TODO: Get from analysis or use default
    let refractory_frames = tempo_adaptive_refractory(
        config.thresholds.refractory_ms_base,
        tempo_bpm,
        state.sr,
        config.stft.hop_length,
    );

    let peak_frames = find_peaks_with_refractory(&fused_onset, &thresholds, refractory_frames);

    // 7. Flam detection
    println!("  Detecting flam candidates...");
    let flam_candidates = detect_flam_candidates(
        &peak_frames,
        whitened,
        &primary_stft.freqs,
        config.thresholds.flam_spectral_tolerance_hz,
    );

    // 8. Create onset events
    println!("  Creating onset events...");
    let mut onset_events = Vec::new();

    for (i, &frame_idx) in peak_frames.iter().enumerate() {
        let time_sec = frame_idx as f32 * config.stft.hop_length as f32 / state.sr as f32;
        let strength = fused_onset[frame_idx];
        let snr = compute_snr(&fused_onset, frame_idx, 10); // 10-frame window for SNR
        let spectral_centroid =
            compute_spectral_centroid(&whitened.column(frame_idx), &primary_stft.freqs);
        let is_flam_candidate = flam_candidates[i];
        let quality_score = compute_quality_score(strength, snr, is_flam_candidate);

        let event = OnsetEvent {
            time_sec,
            frame_idx,
            strength,
            snr,
            spectral_centroid_hz: spectral_centroid,
            is_flam_candidate,
            quality_score,
        };

        onset_events.push(event);
    }

    // Optional quality filtering
    if config.onset_seeding.filter_low_quality {
        onset_events.retain(|event| event.quality_score >= config.onset_seeding.min_seed_snr);
    }

    // Store results
    state.onset_events = onset_events;

    println!("  ✓ Detected {} onset events", state.onset_events.len());
    println!("  ✓ Pass 2 complete");

    Ok(())
}
