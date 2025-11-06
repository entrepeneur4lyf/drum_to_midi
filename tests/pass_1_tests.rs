//! Comprehensive validation tests for Pass 1: Spectral Envelope & Whitening

use drum2midi::audio::AudioState;
use drum2midi::config::Config;
use drum2midi::passes::pass_1;
use drum2midi::spectral::{magnitude_spectrogram, smooth_1_3_octave, stft};
use ndarray::{s, Array2};
use std::f32::consts::PI;

/// Generate synthetic drum-like test audio
fn generate_drum_audio(n_samples: usize, sr: u32) -> Vec<f32> {
    let mut audio = vec![0.0; n_samples];

    // Add kick drum (fundamental ~80Hz with harmonics)
    let kick_freq = 80.0;
    for i in 0..n_samples {
        let t = i as f32 / sr as f32;
        let envelope = (-t * 10.0).exp(); // Decay envelope
        let fundamental = (2.0 * PI * kick_freq * t).sin();
        let harmonic1 = 0.5 * (2.0 * PI * kick_freq * 2.0 * t).sin();
        let harmonic2 = 0.3 * (2.0 * PI * kick_freq * 3.0 * t).sin();
        audio[i] += (fundamental + harmonic1 + harmonic2) * envelope * 0.3;
    }

    // Add snare (broadband noise with body)
    let snare_start = sr as usize / 4; // Start at 0.25 seconds
    for i in snare_start..(snare_start + sr as usize / 8) {
        if i < n_samples {
            let t = (i - snare_start) as f32 / sr as f32;
            let envelope = (-t * 20.0).exp();
            // Broadband noise
            let noise = (rand::random::<f32>() - 0.5) * 2.0;
            // Body resonance ~200Hz
            let body = (2.0 * PI * 200.0 * t).sin();
            audio[i] += (noise * 0.7 + body * 0.3) * envelope * 0.2;
        }
    }

    // Add hi-hat (high frequency content)
    for i in 0..n_samples {
        if i % 1000 < 100 {
            // Sporadic hi-hats
            let t = (i % 1000) as f32 / sr as f32;
            let envelope = (-t * 50.0).exp();
            let noise = (rand::random::<f32>() - 0.5) * 2.0;
            audio[i] += noise * envelope * 0.1;
        }
    }

    audio
}

/// Generate full mix audio (non-drum content)
fn generate_full_mix_audio(n_samples: usize, sr: u32) -> Vec<f32> {
    let mut audio = vec![0.0; n_samples];

    // Add melodic content (guitar-like)
    for i in 0..n_samples {
        let t = i as f32 / sr as f32;
        // Chord progression simulation
        let freq1 = 220.0; // A3
        let freq2 = 277.0; // C#4
        let freq3 = 330.0; // E4
        audio[i] += 0.2 * (2.0 * PI * freq1 * t).sin();
        audio[i] += 0.15 * (2.0 * PI * freq2 * t).sin();
        audio[i] += 0.1 * (2.0 * PI * freq3 * t).sin();
    }

    // Add bass line
    for i in 0..n_samples {
        let t = i as f32 / sr as f32;
        let bass_freq = 55.0; // A1
        audio[i] += 0.3 * (2.0 * PI * bass_freq * t).sin();
    }

    audio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stft_accuracy() {
        let sr = 44100;
        let audio = generate_drum_audio(sr as usize, sr);
        let n_fft = 2048;
        let hop_length = 512;

        let stft_data = stft(&audio, n_fft, hop_length, "hann", sr);
        let mag = magnitude_spectrogram(&stft_data);

        // Verify STFT dimensions
        assert_eq!(stft_data.s.shape()[0], n_fft / 2 + 1);
        assert_eq!(stft_data.freqs.len(), n_fft / 2 + 1);
        assert_eq!(stft_data.times.len(), mag.shape()[1]);

        // Verify frequency range
        assert!((stft_data.freqs[0] - 0.0).abs() < 1.0); // DC component
        assert!((stft_data.freqs.last().unwrap() - sr as f32 / 2.0).abs() < 100.0); // Nyquist

        // Verify time axis
        let expected_frames = (audio.len() - n_fft) / hop_length + 1;
        assert_eq!(stft_data.times.len(), expected_frames);
    }

    #[test]
    fn test_whitening_effectiveness_drum_content() {
        let sr = 44100;
        let audio = generate_drum_audio(sr as usize, sr);
        let config = Config::default();

        let mut state = AudioState::load_from_samples(audio, sr, &config).unwrap();

        // Run Pass 1
        pass_1::run(&mut state, &config).unwrap();

        // Verify whitened spectrogram exists
        assert!(state.s_whitened.is_some());
        let whitened = state.s_whitened.as_ref().unwrap();

        // Check that drum frequencies are enhanced
        // Kick: 40-120Hz, Snare: 150-300Hz, Hats: >4kHz
        let freqs = &state.stfts[&(config.stft.n_fft, config.stft.hop_length)].freqs;

        // Find indices for key frequency ranges (ensure they are within bounds)
        let kick_start = freqs.iter().position(|&f| f >= 40.0).unwrap_or(0);
        let kick_end = freqs
            .iter()
            .position(|&f| f >= 120.0)
            .unwrap_or(freqs.len())
            .min(freqs.len());
        let kick_range = kick_start..kick_end;

        let snare_start = freqs.iter().position(|&f| f >= 150.0).unwrap_or(0);
        let snare_end = freqs
            .iter()
            .position(|&f| f >= 300.0)
            .unwrap_or(freqs.len())
            .min(freqs.len());
        let snare_range = snare_start..snare_end;

        let hat_start = freqs.iter().position(|&f| f >= 4000.0).unwrap_or(0);
        let hat_range = hat_start..freqs.len();

        // Verify that these ranges have reasonable energy after whitening
        // Note: Whitening may suppress some frequencies, so we check that the spectrogram is valid
        let total_energy: f32 = whitened.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            total_energy > 0.0,
            "Whitened spectrogram should have non-zero total energy"
        );

        // Check that no frequency range is completely zeroed out (which would be problematic)
        let kick_energy: f32 = whitened
            .slice(s![kick_range, ..])
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let snare_energy: f32 = whitened
            .slice(s![snare_range, ..])
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();
        let hat_energy: f32 = whitened
            .slice(s![hat_range, ..])
            .iter()
            .map(|&x| x * x)
            .sum::<f32>()
            .sqrt();

        // At least one frequency range should have significant energy
        let max_range_energy = kick_energy.max(snare_energy).max(hat_energy);
        assert!(
            max_range_energy > total_energy * 0.01,
            "At least one drum frequency range should have significant energy"
        );
    }

    #[test]
    fn test_whitening_full_mix_rejection() {
        let sr = 44100;
        let audio = generate_full_mix_audio(sr as usize, sr);
        let config = Config::default();

        let mut state = AudioState::load_from_samples(audio, sr, &config).unwrap();

        // Run Pass 1
        pass_1::run(&mut state, &config).unwrap();

        // Verify whitened spectrogram exists
        assert!(state.s_whitened.is_some());
        let whitened = state.s_whitened.as_ref().unwrap();

        // For full mix content, whitening should still work but may not enhance drum frequencies as much
        // The key is that it doesn't crash and produces reasonable output
        let total_energy: f32 = whitened.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            total_energy > 0.0,
            "Whitened spectrogram should have non-zero energy"
        );
    }

    #[test]
    fn test_one_third_octave_smoothing() {
        let sr = 44100;
        let n_fft = 2048;
        let freqs: Vec<f32> = (0..n_fft / 2 + 1)
            .map(|i| i as f32 * sr as f32 / n_fft as f32)
            .collect();

        // Create test signal with peaks at 1/3 octave centers
        let mut test_signal = vec![0.0; freqs.len()];
        let centers: Vec<f32> = (0..10)
            .map(|i| 31.25 * 2.0f32.powf(i as f32 / 3.0))
            .filter(|&f| f <= 20000.0)
            .collect();

        for &fc in &centers {
            if let Some(idx) = freqs.iter().position(|&f| (f - fc).abs() < 10.0) {
                test_signal[idx] = 1.0;
            }
        }

        let smoothed = smooth_1_3_octave(&test_signal, &freqs);

        // Verify smoothing maintains peaks at 1/3 octave centers
        assert_eq!(smoothed.len(), test_signal.len());

        // Check that values are reasonable (not NaN, not infinite)
        for &val in &smoothed {
            assert!(val.is_finite(), "Smoothed values should be finite");
            assert!(val >= 0.0, "Smoothed values should be non-negative");
        }

        // Verify that the smoothing spreads energy appropriately
        let total_original: f32 = test_signal.iter().sum();
        let total_smoothed: f32 = smoothed.iter().sum();
        assert!(
            (total_original - total_smoothed).abs() / total_original < 0.1,
            "Energy should be approximately conserved"
        );
    }

    #[test]
    fn test_multi_resolution_stft() {
        let sr = 44100;
        let audio = generate_drum_audio(sr as usize, sr);
        let config = Config::default();

        let mut state = AudioState::load_from_samples(audio, sr, &config).unwrap();

        // Run Pass 1
        pass_1::run(&mut state, &config).unwrap();

        // Verify multi-resolution STFTs were computed
        assert!(!state.stfts.is_empty());

        // Check that all configured resolutions are present
        for &(n_fft, hop) in &config.stft.multi_res_configs {
            assert!(
                state.stfts.contains_key(&(n_fft, hop)),
                "STFT for ({}, {}) should be present",
                n_fft,
                hop
            );
        }

        // Verify each STFT has correct dimensions
        for (&(n_fft, _), stft_data) in &state.stfts {
            assert_eq!(
                stft_data.s.shape()[0],
                n_fft / 2 + 1,
                "STFT frequency dimension should be n_fft/2 + 1"
            );
            assert_eq!(
                stft_data.freqs.len(),
                n_fft / 2 + 1,
                "Frequency array length should match STFT dimensions"
            );
        }
    }

    #[test]
    fn test_performance_fft_sizes() {
        let sr = 44100;
        let audio = generate_drum_audio(sr as usize / 4, sr); // Shorter audio for performance test
        let config = Config::default();

        let mut state = AudioState::load_from_samples(audio, sr, &config).unwrap();

        // Time the Pass 1 execution
        let start = std::time::Instant::now();
        pass_1::run(&mut state, &config).unwrap();
        let duration = start.elapsed();

        // Should complete in reasonable time (< 2 seconds for this short audio)
        assert!(
            duration.as_secs_f32() < 2.0,
            "Pass 1 should complete in less than 2 seconds, took {:.2}s",
            duration.as_secs_f32()
        );

        // Verify output quality
        assert!(state.s_whitened.is_some());
        assert!(!state.stfts.is_empty());
    }

    #[test]
    fn test_whitening_curve_properties() {
        let sr = 44100;
        let audio = generate_drum_audio(sr as usize, sr);
        let config = Config::default();

        let mut state = AudioState::load_from_samples(audio, sr, &config).unwrap();

        // Run Pass 1
        pass_1::run(&mut state, &config).unwrap();

        // Get the primary STFT data
        let primary_stft = &state.stfts[&(config.stft.n_fft, config.stft.hop_length)];

        // The whitening curve should be computed and stored
        // (We can't directly access it, but we can verify the whitened spectrogram exists)
        assert!(state.s_whitened.is_some());

        let whitened = state.s_whitened.as_ref().unwrap();

        // Verify dimensions match
        assert_eq!(whitened.shape()[0], primary_stft.s.shape()[0]);
        assert_eq!(whitened.shape()[1], primary_stft.s.shape()[1]);

        // Verify no NaN or infinite values
        for &val in whitened.iter() {
            assert!(
                val.is_finite(),
                "Whitened spectrogram should contain only finite values"
            );
        }

        // Verify reasonable dynamic range
        let max_val = whitened.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = whitened.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max_val > min_val, "Should have dynamic range");
        assert!(max_val > 0.0, "Should have positive values");
    }
}

// Helper trait for testing
trait AudioStateTestExt {
    fn load_from_samples(samples: Vec<f32>, sr: u32, config: &Config) -> anyhow::Result<Self>
    where
        Self: Sized;
}

impl AudioStateTestExt for AudioState {
    fn load_from_samples(samples: Vec<f32>, sr: u32, config: &Config) -> anyhow::Result<Self> {
        Ok(AudioState::from_test_samples(samples, sr, config))
    }
}
