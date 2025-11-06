//! Spectral processing utilities (STFT, filtering, etc.)

use ndarray::Array2;
use rustfft::{num_complex::Complex32, FftPlanner};

/// STFT data structure
#[derive(Debug, Clone)]
pub struct StftData {
    pub s: Array2<Complex32>,
    pub freqs: Vec<f32>,
    pub times: Vec<f32>,
}

/// Compute STFT of audio signal
pub fn stft(y: &[f32], n_fft: usize, hop_length: usize, window: &str, sample_rate: u32) -> StftData {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let n_frames = (y.len() - n_fft) / hop_length + 1;
    let mut s = Array2::<Complex32>::zeros((n_fft / 2 + 1, n_frames));

    // Generate window function
    let window_fn = generate_window(window, n_fft);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;
        let end = start + n_fft;

        if end > y.len() {
            break;
        }

        // Apply window
        let mut frame: Vec<Complex32> = y[start..end]
            .iter()
            .zip(&window_fn)
            .map(|(&sample, &win)| Complex32::new(sample * win, 0.0))
            .collect();

        // FFT
        fft.process(&mut frame);

        // Store positive frequencies
        for (i, &val) in frame[..n_fft / 2 + 1].iter().enumerate() {
            s[[i, frame_idx]] = val;
        }
    }

    let freqs: Vec<f32> = (0..n_fft / 2 + 1)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let times: Vec<f32> = (0..n_frames)
        .map(|i| i as f32 * hop_length as f32 / sample_rate as f32)
        .collect();

    StftData { s, freqs, times }
}

/// Generate window function
fn generate_window(window_type: &str, size: usize) -> Vec<f32> {
    match window_type {
        "hann" => (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
            })
            .collect(),
        _ => vec![1.0; size], // Rectangular window as fallback
    }
}

/// Compute magnitude spectrogram
pub fn magnitude_spectrogram(stft_data: &StftData) -> Array2<f32> {
    stft_data.s.map(|c| c.norm())
}

/// Compute spectral flux
pub fn spectral_flux(mag_spec: &Array2<f32>) -> Vec<f32> {
    let mut flux = vec![0.0; mag_spec.shape()[1]];

    for t in 1..mag_spec.shape()[1] {
        let mut frame_flux = 0.0;
        for f in 0..mag_spec.shape()[0] {
            let diff = mag_spec[[f, t]] - mag_spec[[f, t - 1]];
            if diff > 0.0 {
                frame_flux += diff;
            }
        }
        flux[t] = frame_flux;
    }

    flux
}

/// Compute complex-domain spectral flux
pub fn complex_domain_flux(stft_data: &StftData) -> Vec<f32> {
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

/// Extract band-limited spectral flux
pub fn band_flux(stft_data: &StftData, freq_range: &[f32]) -> Vec<f32> {
    let freq_mask: Vec<bool> = stft_data
        .freqs
        .iter()
        .map(|&f| f >= freq_range[0] && f <= freq_range[1])
        .collect();

    let mut flux = vec![0.0; stft_data.s.shape()[1]];

    for t in 1..stft_data.s.shape()[1] {
        let mut frame_flux = 0.0;
        for (f, &in_band) in freq_mask.iter().enumerate() {
            if in_band {
                let diff = (stft_data.s[[f, t]] - stft_data.s[[f, t - 1]]).norm();
                frame_flux += diff;
            }
        }
        flux[t] = frame_flux;
    }

    flux
}

/// Compute inverse STFT of complex spectrogram
pub fn inverse_stft(s: &Array2<Complex32>, n_fft: usize, hop_length: usize, window: &str, _sample_rate: u32) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let n_frames = s.shape()[1];
    let expected_length = (n_frames - 1) * hop_length + n_fft;

    // Generate window function
    let window_fn = generate_window(window, n_fft);

    // Initialize output buffer
    let mut y = vec![0.0f32; expected_length];

    for frame_idx in 0..n_frames {
        // Get complex frame (only positive frequencies)
        let mut frame: Vec<Complex32> = s.column(frame_idx).iter().cloned().collect();

        // Add negative frequencies (conjugate symmetric)
        for i in 1..(n_fft / 2) {
            frame.push(frame[n_fft / 2 - i].conj());
        }

        // Apply inverse FFT
        ifft.process(&mut frame);

        // Apply window and overlap-add
        let start = frame_idx * hop_length;
        for i in 0..n_fft {
            if start + i < y.len() {
                // Scale by 1/n_fft for proper normalization
                let sample = frame[i].re / n_fft as f32;
                y[start + i] += sample * window_fn[i];
            }
        }
    }

    // Normalize by window compensation
    let window_sum = window_fn.iter().sum::<f32>();
    let norm_factor = hop_length as f32 / window_sum;

    for sample in &mut y {
        *sample *= norm_factor;
    }

    y
}

/// Smooth signal with 1/3 octave smoothing
pub fn smooth_1_3_octave(signal: &[f32], freqs: &[f32]) -> Vec<f32> {
    let mut smoothed = vec![0.0; signal.len()];

    // Third octave center frequencies
    let centers: Vec<f32> = (0..10)
        .map(|i| 31.25 * 2.0f32.powf(i as f32 / 3.0))
        .filter(|&f| f <= 20000.0)
        .collect();

    for &fc in &centers {
        let fl = fc / 2.0f32.powf(1.0 / 6.0);
        let fh = fc * 2.0f32.powf(1.0 / 6.0);

        let mut band_sum = 0.0;
        let mut count = 0;

        for (i, &freq) in freqs.iter().enumerate() {
            if freq >= fl && freq <= fh {
                band_sum += signal[i];
                count += 1;
            }
        }

        if count > 0 {
            let band_avg = band_sum / count as f32;
            for (i, &freq) in freqs.iter().enumerate() {
                if freq >= fl && freq <= fh {
                    smoothed[i] = band_avg;
                }
            }
        }
    }

    // Fill any gaps with interpolation
    for i in 0..smoothed.len() {
        if smoothed[i] == 0.0 && i > 0 && i < smoothed.len() - 1 {
            smoothed[i] = (smoothed[i - 1] + smoothed[i + 1]) / 2.0;
        }
    }

    smoothed
}
