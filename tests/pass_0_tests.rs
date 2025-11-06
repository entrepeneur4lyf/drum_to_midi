//! Comprehensive validation tests for Pass 0: Preflight & Normalization

use drum2midi::audio::{apply_gain, corrcoef, measure_lufs, true_peak_limiter};

/// Generate synthetic test audio
fn generate_test_audio(n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_gain() {
        let samples = vec![0.5, -0.3, 0.8];
        let amplified = apply_gain(&samples, 6.0); // +6dB = *2
        assert!((amplified[0] - 1.0).abs() < 1e-6);
        assert!((amplified[1] - (-0.6)).abs() < 1e-6);
        assert!((amplified[2] - 1.6).abs() < 1e-6);
    }

    #[test]
    fn test_measure_lufs() {
        let audio = generate_test_audio(44100); // 1 second
        let lufs = measure_lufs(&audio, 44100);
        // Should be a reasonable negative dB value
        assert!(lufs < 0.0 && lufs > -50.0);
    }

    #[test]
    fn test_true_peak_limiting() {
        let mut audio = generate_test_audio(1000);
        // Add a peak above the limit
        audio[500] = 2.0; // Above 0 dBFS

        let limited = true_peak_limiter(&audio, -6.0); // -6 dB limit

        // Check that the peak is limited
        let limit_linear = 10.0_f32.powf(-6.0 / 20.0);
        for &sample in &limited {
            assert!(sample.abs() <= limit_linear * 1.01); // Small tolerance
        }
    }

    #[test]
    fn test_corrcoef_calculation() {
        // Test with identical signals
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = corrcoef(&a, &b);
        assert!(
            (corr - 1.0).abs() < 1e-6,
            "Identical signals should have correlation 1.0"
        );

        // Test with uncorrelated signals
        let c = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let d = vec![-1.0, 1.0, -1.0, 1.0, -1.0];
        let corr_uncorr = corrcoef(&c, &d);
        assert!(
            corr_uncorr.abs() < 0.1,
            "Uncorrelated signals should have low correlation"
        );
    }

    #[test]
    fn test_audio_processing_pipeline() {
        let mut audio = generate_test_audio(44100);
        let sr = 44100;

        // Test LUFS measurement
        let original_lufs = measure_lufs(&audio, sr);

        // Test gain application
        let gain_db = -16.0 - original_lufs;
        audio = apply_gain(&audio, gain_db);

        // Test limiting
        audio = true_peak_limiter(&audio, -1.0);

        // Verify final audio properties
        let final_lufs = measure_lufs(&audio, sr);
        assert!(
            (final_lufs - (-16.0)).abs() < 1.0,
            "Should be normalized to approximately -16 dB"
        );

        // Check no samples exceed limit
        let limit_linear = 10.0_f32.powf(-1.0 / 20.0);
        for &sample in &audio {
            assert!(sample.abs() <= limit_linear * 1.1);
        }
    }
}
