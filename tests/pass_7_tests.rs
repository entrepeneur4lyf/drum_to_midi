//! Tests for Pass 7: Self-Prior Construction

use drum2midi::analysis::{
    ClassificationFeatures, ClassifiedEvent, DrumClass, PriorStats, SelfPriorMatrices,
};
use drum2midi::config::Config;
use drum2midi::passes::pass_7;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> Config {
        Config::default()
    }

    fn create_test_events() -> Vec<ClassifiedEvent> {
        vec![
            ClassifiedEvent {
                time_sec: 0.0,
                frame_idx: 0,
                drum_class: DrumClass::Kick,
                confidence: 0.9,
                acoustic_confidence: 0.9,
                prior_confidence: 0.8,
                features: ClassificationFeatures {
                    fundamental_hz: Some(80.0),
                    spectral_centroid_hz: 1000.0,
                    spectral_rolloff_hz: 3000.0,
                    zero_crossing_rate: 0.1,
                    low_energy_ratio: 0.6,
                    mid_energy_ratio: 0.3,
                    high_energy_ratio: 0.1,
                    attack_time_ms: 5.0,
                    decay_time_ms: 100.0,
                    transient_energy_ratio: 0.7,
                    attack_spectral_centroid: 1200.0,
                    attack_hf_ratio: 0.2,
                    transient_sharpness: 0.6,
                    attack_to_sustain_ratio: 2.5,
                },
                alternative_classes: vec![],
            },
            ClassifiedEvent {
                time_sec: 2.0,
                frame_idx: 44100,
                drum_class: DrumClass::Snare,
                confidence: 0.85,
                acoustic_confidence: 0.85,
                prior_confidence: 0.8,
                features: ClassificationFeatures {
                    fundamental_hz: None,
                    spectral_centroid_hz: 2000.0,
                    spectral_rolloff_hz: 4000.0,
                    zero_crossing_rate: 0.15,
                    low_energy_ratio: 0.3,
                    mid_energy_ratio: 0.4,
                    high_energy_ratio: 0.3,
                    attack_time_ms: 3.0,
                    decay_time_ms: 80.0,
                    transient_energy_ratio: 0.6,
                    attack_spectral_centroid: 2500.0,
                    attack_hf_ratio: 0.4,
                    transient_sharpness: 0.7,
                    attack_to_sustain_ratio: 2.2,
                },
                alternative_classes: vec![],
            },
            ClassifiedEvent {
                time_sec: 4.0,
                frame_idx: 88200,
                drum_class: DrumClass::HiHat,
                confidence: 0.8,
                acoustic_confidence: 0.8,
                prior_confidence: 0.7,
                features: ClassificationFeatures {
                    fundamental_hz: None,
                    spectral_centroid_hz: 5000.0,
                    spectral_rolloff_hz: 8000.0,
                    zero_crossing_rate: 0.2,
                    low_energy_ratio: 0.1,
                    mid_energy_ratio: 0.2,
                    high_energy_ratio: 0.7,
                    attack_time_ms: 2.0,
                    decay_time_ms: 50.0,
                    transient_energy_ratio: 0.9,
                    attack_spectral_centroid: 5500.0,
                    attack_hf_ratio: 0.8,
                    transient_sharpness: 0.8,
                    attack_to_sustain_ratio: 3.5,
                },
                alternative_classes: vec![],
            },
        ]
    }

    #[test]
    fn test_grid_position_calculation() {
        let tempo_bpm = 120.0;
        let meter_beats_per_measure = 4;
        let grid_slots_per_beat = 4;
        let downbeat_time_sec = 0.0;

        // Test quarter note positions (every 0.5 seconds at 120 BPM)
        // 120 BPM = 2 beats/second = 0.5 sec/beat, 4 slots/beat = 8 slots/second
        assert_eq!(
            pass_7::calculate_grid_position(
                0.0,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(0)
        );
        assert_eq!(
            pass_7::calculate_grid_position(
                0.5,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(4)
        ); // 1 beat = 4 slots
        assert_eq!(
            pass_7::calculate_grid_position(
                1.0,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(8)
        ); // 2 beats = 8 slots
        assert_eq!(
            pass_7::calculate_grid_position(
                1.9,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(15)
        ); // Almost 4 beats

        // Test sixteenth note positions
        assert_eq!(
            pass_7::calculate_grid_position(
                0.125,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(1)
        ); // 0.25 beats = 1 slot
        assert_eq!(
            pass_7::calculate_grid_position(
                0.25,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(2)
        ); // 0.5 beats = 2 slots
        assert_eq!(
            pass_7::calculate_grid_position(
                0.375,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(3)
        ); // 0.75 beats = 3 slots
        assert_eq!(
            pass_7::calculate_grid_position(
                0.5,
                tempo_bpm,
                meter_beats_per_measure,
                grid_slots_per_beat,
                downbeat_time_sec
            ),
            Some(4)
        ); // 1 beat = 4 slots
    }

    #[test]
    fn test_event_accumulation() {
        let events = create_test_events();
        let tempo_bpm = 120.0;
        let meter_beats_per_measure = 4;
        let grid_slots_per_beat = 4;
        let downbeat_positions = vec![0.0];

        let counts = pass_7::accumulate_event_counts(
            &events,
            tempo_bpm,
            meter_beats_per_measure,
            grid_slots_per_beat,
            &downbeat_positions,
        );

        // Check that we have counts for each drum class
        assert!(counts.contains_key(&DrumClass::Kick));
        assert!(counts.contains_key(&DrumClass::Snare));
        assert!(counts.contains_key(&DrumClass::HiHat));

        // Check grid size (4 beats * 4 slots per beat = 16 slots)
        assert_eq!(counts[&DrumClass::Kick].len(), 16);
        assert_eq!(counts[&DrumClass::Snare].len(), 16);
        assert_eq!(counts[&DrumClass::HiHat].len(), 16);

        // Check that events are accumulated in correct positions
        // Kick at 0.0 -> grid position 0
        assert_eq!(counts[&DrumClass::Kick][0], 1.0);
        // Snare at 2.0 -> grid position 15 (2 seconds = 4 beats = 16 slots, clamped to 15)
        assert_eq!(counts[&DrumClass::Snare][15], 1.0);
        // HiHat at 4.0 -> beyond bar duration, should be ignored
    }

    #[test]
    fn test_gaussian_smoothing() {
        let counts = vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let sigma_beats = 0.15;
        let grid_slots_per_beat = 4;

        let smoothed = pass_7::apply_gaussian_smoothing(&counts, sigma_beats, grid_slots_per_beat);

        assert_eq!(smoothed.len(), counts.len());
        // The peak should still be at position 1, but smoothed
        assert!(smoothed[1] > smoothed[0]);
        assert!(smoothed[1] > smoothed[2]);

        // Check that values decrease with distance from peak
        for i in 2..smoothed.len() {
            assert!(smoothed[i] < smoothed[i - 1] || (smoothed[i] - smoothed[i - 1]).abs() < 0.001);
        }
    }

    #[test]
    fn test_beta_smoothing() {
        let smoothed_counts = vec![
            0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let alpha = 1.0;
        let beta = 2.0;

        let (priors, confidences) = pass_7::apply_beta_smoothing(&smoothed_counts, alpha, beta);

        assert_eq!(priors.len(), smoothed_counts.len());
        assert_eq!(confidences.len(), smoothed_counts.len());

        // Check that priors are in valid range (posterior means)
        for &p in &priors {
            assert!(p >= 0.0 && p <= 1.0);
        }

        // Check that confidence is higher for positions with data
        assert!(confidences[1] > confidences[0]);
        assert!(confidences[1] > 0.1); // Should be reasonably confident
    }

    #[test]
    fn test_self_prior_construction() {
        let events = create_test_events();
        let config = create_test_config();
        let tempo_bpm = 120.0;
        let meter_beats_per_measure = 4;
        let downbeat_positions = vec![0.0];

        let priors = pass_7::construct_self_priors(
            &events,
            tempo_bpm,
            meter_beats_per_measure,
            &downbeat_positions,
            &config,
        );

        // Check basic structure
        assert_eq!(
            priors.grid_slots_per_beat,
            config.self_prior.grid_slots_per_beat
        );
        assert!(priors.class_priors.contains_key(&DrumClass::Kick));
        assert!(priors.class_priors.contains_key(&DrumClass::Snare));
        assert!(priors.class_priors.contains_key(&DrumClass::HiHat));

        // Check that priors are normalized (sum to 1)
        for priors_vec in priors.class_priors.values() {
            let sum: f32 = priors_vec.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }

        // Check confidences are in valid range
        for confidences_vec in priors.class_confidences.values() {
            for &conf in confidences_vec {
                assert!(conf >= 0.0 && conf <= 1.0);
            }
        }
    }

    #[test]
    fn test_prior_statistics() {
        let events = create_test_events();
        let priors = SelfPriorMatrices {
            grid_slots_per_beat: 4,
            class_priors: std::collections::HashMap::new(),
            class_confidences: std::collections::HashMap::new(),
            total_events_per_class: std::collections::HashMap::new(),
            smoothing_sigma_beats: 0.15,
            beta_smoothing_alpha: 1.0,
            beta_smoothing_beta: 2.0,
        };
        let downbeat_positions = vec![0.0];

        let stats = pass_7::calculate_prior_stats(&priors, &events, &downbeat_positions);

        assert_eq!(stats.total_bars_analyzed, 1);
        assert_eq!(stats.events_per_bar_avg, 3.0);
        assert!(stats.class_distribution.contains_key(&DrumClass::Kick));
        assert!(stats.class_distribution.contains_key(&DrumClass::Snare));
        assert!(stats.class_distribution.contains_key(&DrumClass::HiHat));

        // Check that percentages sum to approximately 1
        let total_percentage: f32 = stats.class_distribution.values().sum();
        assert!((total_percentage - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_pattern_capture_different_styles() {
        // Test with rock pattern (kick on 1, snare on 2&4)
        let rock_events = vec![
            ClassifiedEvent {
                time_sec: 0.0,
                drum_class: DrumClass::Kick,
                confidence: 0.9,
                frame_idx: 0,
                acoustic_confidence: 0.9,
                prior_confidence: 0.8,
                features: ClassificationFeatures::default(),
                alternative_classes: vec![],
            },
            ClassifiedEvent {
                time_sec: 1.0,
                drum_class: DrumClass::Snare,
                confidence: 0.85,
                frame_idx: 22050,
                acoustic_confidence: 0.85,
                prior_confidence: 0.8,
                features: ClassificationFeatures::default(),
                alternative_classes: vec![],
            },
            ClassifiedEvent {
                time_sec: 2.0,
                drum_class: DrumClass::Kick,
                confidence: 0.9,
                frame_idx: 44100,
                acoustic_confidence: 0.9,
                prior_confidence: 0.8,
                features: ClassificationFeatures::default(),
                alternative_classes: vec![],
            },
            ClassifiedEvent {
                time_sec: 3.0,
                drum_class: DrumClass::Snare,
                confidence: 0.85,
                frame_idx: 66150,
                acoustic_confidence: 0.85,
                prior_confidence: 0.8,
                features: ClassificationFeatures::default(),
                alternative_classes: vec![],
            },
        ];

        let config = create_test_config();
        let tempo_bpm = 60.0; // 1 beat per second for simplicity
        let meter_beats_per_measure = 4;
        let downbeat_positions = vec![0.0];

        let priors = pass_7::construct_self_priors(
            &rock_events,
            tempo_bpm,
            meter_beats_per_measure,
            &downbeat_positions,
            &config,
        );

        // Should have captured the backbeat pattern
        assert!(priors.total_events_per_class[&DrumClass::Kick] >= 2);
        assert!(priors.total_events_per_class[&DrumClass::Snare] >= 2);
    }

    #[test]
    fn test_smoothing_preserves_features() {
        let events = create_test_events();
        let config = create_test_config();
        let tempo_bpm = 120.0;
        let meter_beats_per_measure = 4;
        let downbeat_positions = vec![0.0];

        let priors_no_smooth = {
            let mut config_no_smooth = config.clone();
            config_no_smooth.self_prior.smoothing_sigma_beats = 0.01; // Very sharp
            pass_7::construct_self_priors(
                &events,
                tempo_bpm,
                meter_beats_per_measure,
                &downbeat_positions,
                &config_no_smooth,
            )
        };

        let priors_smooth = pass_7::construct_self_priors(
            &events,
            tempo_bpm,
            meter_beats_per_measure,
            &downbeat_positions,
            &config,
        );

        // Smoothing should not eliminate the main peaks
        // (This is a basic check - more sophisticated validation would compare peak preservation)
        assert!(!priors_smooth.class_priors.is_empty());
        assert!(!priors_no_smooth.class_priors.is_empty());
    }

    #[test]
    fn test_performance_under_time_constraint() {
        let mut events = Vec::new();
        // Create a larger dataset to test performance
        for i in 0..100 {
            events.push(ClassifiedEvent {
                time_sec: i as f32 * 0.1,
                frame_idx: i * 4410,
                drum_class: if i % 3 == 0 {
                    DrumClass::Kick
                } else if i % 3 == 1 {
                    DrumClass::Snare
                } else {
                    DrumClass::HiHat
                },
                confidence: 0.8,
                acoustic_confidence: 0.8,
                prior_confidence: 0.7,
                features: ClassificationFeatures::default(),
                alternative_classes: vec![],
            });
        }

        let config = create_test_config();
        let tempo_bpm = 120.0;
        let meter_beats_per_measure = 4;
        let downbeat_positions = vec![0.0, 8.0, 16.0, 24.0];

        let start_time = std::time::Instant::now();
        let _priors = pass_7::construct_self_priors(
            &events,
            tempo_bpm,
            meter_beats_per_measure,
            &downbeat_positions,
            &config,
        );
        let elapsed = start_time.elapsed();

        // Should complete in under 500ms as per spec
        assert!(
            elapsed.as_millis() < 500,
            "Prior construction took {}ms, should be < 500ms",
            elapsed.as_millis()
        );
    }

    #[test]
    fn test_edge_case_insufficient_events() {
        let events = vec![ClassifiedEvent {
            time_sec: 0.0,
            drum_class: DrumClass::Kick,
            confidence: 0.9,
            frame_idx: 0,
            acoustic_confidence: 0.9,
            prior_confidence: 0.8,
            features: ClassificationFeatures::default(),
            alternative_classes: vec![],
        }];

        let config = create_test_config();
        let tempo_bpm = 120.0;
        let meter_beats_per_measure = 4;
        let downbeat_positions = vec![0.0];

        let priors = pass_7::construct_self_priors(
            &events,
            tempo_bpm,
            meter_beats_per_measure,
            &downbeat_positions,
            &config,
        );

        // Should still produce valid priors even with minimal events
        assert!(!priors.class_priors.is_empty());
        assert!(priors.class_confidences[&DrumClass::Kick]
            .iter()
            .all(|&c| c >= 0.0 && c <= 1.0));
    }
}
