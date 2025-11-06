//! Pass 4: Adaptive Instrument Classification

use crate::analysis::{ClassificationFeatures, ClassifiedEvent, DrumClass, TuningInfo};
use crate::audio::{AudioState, OnsetEvent};
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};
use ndarray::{Array2, s};

/// Extract fundamental frequency from spectral peak
fn extract_fundamental_frequency(
    mag_frame: &ndarray::ArrayView1<f32>,
    freqs: &[f32],
    tuning_info: &TuningInfo,
) -> Option<f32> {
    // Look for peaks in multiple frequency ranges
    let mut low_max_peak = 0.0;
    let mut low_fundamental = None;
    let mut cowbell_max_peak = 0.0;
    let mut cowbell_fundamental = None;

    for i in 0..mag_frame.len() {
        if i >= freqs.len() {
            break;
        }

        let freq = freqs[i];
        let magnitude = mag_frame[i];

        // Find local maxima
        if i > 0 && i < mag_frame.len() - 1
            && magnitude > mag_frame[i - 1]
                && magnitude > mag_frame[i + 1]
            {
                // Check low frequency range (60-400Hz for kick/toms)
                if (60.0..=400.0).contains(&freq) && magnitude > low_max_peak {
                    low_max_peak = magnitude;
                    low_fundamental = Some(freq);
                }
                
                // Check cowbell frequency range (500-1000Hz)
                if (500.0..=1000.0).contains(&freq) && magnitude > cowbell_max_peak {
                    cowbell_max_peak = magnitude;
                    cowbell_fundamental = Some(freq);
                }
            }
    }

    // Prioritize cowbell detection if we found a strong peak in that range
    if cowbell_fundamental.is_some() && cowbell_max_peak > low_max_peak * 0.6 {
        return cowbell_fundamental;
    }

    // Otherwise use low frequency detection
    if let Some(freq) = low_fundamental {
        // Check if it matches kick or tom frequencies from tuning
        if let Some(kick_freq) = tuning_info.kick_hz {
            if (freq - kick_freq).abs() < 20.0 {
                // Within 20Hz tolerance
                return Some(kick_freq);
            }
        }

        for &tom_freq in &tuning_info.toms_hz {
            if (freq - tom_freq).abs() < 30.0 {
                // Within 30Hz tolerance for toms
                return Some(tom_freq);
            }
        }
    }

    low_fundamental
}

/// Compute spectral centroid
fn compute_spectral_centroid(mag_frame: &ndarray::ArrayView1<f32>, freqs: &[f32]) -> f32 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..mag_frame.len().min(freqs.len()) {
        let magnitude = mag_frame[i].abs();
        numerator += freqs[i] * magnitude;
        denominator += magnitude;
    }

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute spectral rolloff (85th percentile energy frequency)
fn compute_spectral_rolloff(mag_frame: &ndarray::ArrayView1<f32>, freqs: &[f32]) -> f32 {
    let total_energy: f32 = mag_frame.iter().map(|&x| x * x).sum();
    if total_energy == 0.0 {
        return 0.0;
    }

    let target_energy = total_energy * 0.85;
    let mut cumulative_energy = 0.0;

    for i in 0..mag_frame.len().min(freqs.len()) {
        cumulative_energy += mag_frame[i] * mag_frame[i];
        if cumulative_energy >= target_energy {
            return freqs[i];
        }
    }

    freqs.last().copied().unwrap_or(0.0)
}

/// Compute zero crossing rate
fn compute_zero_crossing_rate(audio: &[f32], frame_start: usize, frame_size: usize) -> f32 {
    let start = frame_start;
    let end = (start + frame_size).min(audio.len());

    if end - start < 2 {
        return 0.0;
    }

    let mut crossings = 0;
    for i in start..end - 1 {
        if audio[i].signum() != audio[i + 1].signum() {
            crossings += 1;
        }
    }

    crossings as f32 / (end - start - 1) as f32
}

/// Compute multi-band energy ratios
fn compute_energy_ratios(mag_frame: &ndarray::ArrayView1<f32>, freqs: &[f32]) -> (f32, f32, f32) {
    let mut low_energy = 0.0;
    let mut mid_energy = 0.0;
    let mut high_energy = 0.0;

    for i in 0..mag_frame.len().min(freqs.len()) {
        let freq = freqs[i];
        let energy = mag_frame[i] * mag_frame[i];

        if freq < 250.0 {
            low_energy += energy;
        } else if freq < 2000.0 {
            mid_energy += energy;
        } else {
            high_energy += energy;
        }
    }

    let total_energy = low_energy + mid_energy + high_energy;
    if total_energy > 0.0 {
        (
            low_energy / total_energy,
            mid_energy / total_energy,
            high_energy / total_energy,
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Estimate attack and decay times
fn estimate_attack_decay_times(
    mag_frame: &ndarray::ArrayView1<f32>,
    hop_size: usize,
    sr: u32,
) -> (f32, f32) {
    // Simple estimation based on spectral energy distribution
    // This is a simplified implementation

    let max_energy_idx = mag_frame
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Convert to time domain estimate
    let attack_time_ms = (max_energy_idx as f32 * hop_size as f32 / sr as f32) * 1000.0;
    let decay_time_ms = 100.0; // Default decay estimate

    (attack_time_ms, decay_time_ms)
}

/// Analyze transient characteristics for drum classification
fn analyze_transient_characteristics(
    mag_frame: &ndarray::ArrayView1<f32>,
    freqs: &[f32],
    hop_size: usize,
    sr: u32,
) -> (f32, f32, f32, f32, f32) {
    // Define attack window (first 5-15ms)
    let attack_window_ms = 10.0;
    let attack_window_samples = (attack_window_ms / 1000.0 * sr as f32) as usize;
    let attack_window_frames = attack_window_samples / hop_size;
    
    // Extract attack phase (first few frames)
    let attack_frames = mag_frame.slice(s![0..attack_window_frames.min(mag_frame.len())]);
    let sustain_start_frame = (20.0 / 1000.0 * sr as f32 / hop_size as f32) as usize;
    let sustain_frames = mag_frame.slice(s![sustain_start_frame.min(mag_frame.len())..]);
    
    // Calculate attack energy vs total energy ratio
    let total_energy: f32 = mag_frame.map(|&x| x * x).sum();
    let attack_energy: f32 = attack_frames.map(|&x| x * x).sum();
    let transient_energy_ratio = if total_energy > 0.0 {
        attack_energy / total_energy
    } else {
        0.0
    };
    
    // Calculate attack spectral centroid (weighted by energy)
    let attack_spectral_centroid = if attack_energy > 0.0 {
        attack_frames.iter().enumerate().map(|(i, &mag)| {
            let freq_idx = i.min(freqs.len() - 1);
            let weight = mag * mag; // Energy weighting
            freqs[freq_idx] * weight
        }).sum::<f32>() / attack_energy
    } else {
        1000.0
    };
    
    // Calculate attack HF ratio vs sustained
    let hf_band_start = (4000.0 / (sr as f32 / 2.0) * freqs.len() as f32) as usize;
    let attack_hf_energy: f32 = attack_frames.slice(s![hf_band_start.min(attack_frames.len())..]).map(|&x| x * x).sum();
    let sustain_hf_energy: f32 = if !sustain_frames.is_empty() {
        sustain_frames.slice(s![hf_band_start.min(sustain_frames.len())..]).map(|&x| x * x).sum()
    } else {
        0.0
    };
    
    let attack_hf_ratio = if sustain_hf_energy > 0.0 {
        (attack_hf_energy / attack_energy) / (sustain_hf_energy / sustain_frames.map(|&x| x * x).sum())
    } else {
        1.0
    };
    
    // Calculate transient sharpness (rate of level change)
    let mut transient_sharpness = 0.0;
    if attack_frames.len() > 2 {
        for i in 1..attack_frames.len() {
            let level_change = (attack_frames[i] - attack_frames[i-1]).abs();
            transient_sharpness += level_change;
        }
        transient_sharpness /= attack_frames.len() as f32;
    }
    
    // Calculate attack-to-sustain ratio
    let attack_peak = attack_frames.iter().fold(0.0f32, |max, &x| max.max(x));
    let sustain_avg = if !sustain_frames.is_empty() {
        sustain_frames.iter().sum::<f32>() / sustain_frames.len() as f32
    } else {
        attack_peak * 0.1
    };
    let attack_to_sustain_ratio = if sustain_avg > 0.0 {
        attack_peak / sustain_avg
    } else {
        1.0
    };
    
    (
        transient_energy_ratio,
        attack_spectral_centroid,
        attack_hf_ratio.min(10.0), // Cap to prevent extreme values
        transient_sharpness,
        attack_to_sustain_ratio.min(20.0), // Cap to prevent extreme values
    )
}

/// Extract classification features for an onset event
fn extract_classification_features(
    event: &OnsetEvent,
    audio: &[f32],
    mag: &Array2<f32>,
    freqs: &[f32],
    tuning_info: &TuningInfo,
    hop_size: usize,
    sr: u32,
) -> ClassificationFeatures {
    let frame_idx = event.frame_idx;
    let mag_frame = mag.column(frame_idx);

    let fundamental_hz = extract_fundamental_frequency(&mag_frame, freqs, tuning_info);
    let spectral_centroid_hz = compute_spectral_centroid(&mag_frame, freqs);
    let spectral_rolloff_hz = compute_spectral_rolloff(&mag_frame, freqs);

    // Estimate frame size for ZCR computation
    let frame_size = hop_size * 2; // Rough estimate
    let zero_crossing_rate = compute_zero_crossing_rate(audio, frame_idx * hop_size, frame_size);

    let (low_energy_ratio, mid_energy_ratio, high_energy_ratio) =
        compute_energy_ratios(&mag_frame, freqs);

    let (attack_time_ms, decay_time_ms) = estimate_attack_decay_times(&mag_frame, hop_size, sr);
    
    // Analyze transient characteristics
    let (transient_energy_ratio, attack_spectral_centroid, attack_hf_ratio, transient_sharpness, attack_to_sustain_ratio) = 
        analyze_transient_characteristics(&mag_frame, freqs, hop_size, sr);

    ClassificationFeatures {
        fundamental_hz,
        spectral_centroid_hz,
        spectral_rolloff_hz,
        zero_crossing_rate,
        low_energy_ratio,
        mid_energy_ratio,
        high_energy_ratio,
        attack_time_ms,
        decay_time_ms,
        transient_energy_ratio,
        attack_spectral_centroid,
        attack_hf_ratio,
        transient_sharpness,
        attack_to_sustain_ratio,
    }
}

/// Hierarchical classification using rule-based system
fn classify_drum_event(
    features: &ClassificationFeatures,
    tuning_info: &TuningInfo,
    config: &Config,
) -> (DrumClass, f32, Vec<(DrumClass, f32)>) {
    let mut scores = std::collections::HashMap::new();

    // Rule 1: Fundamental frequency matching (strongest cue)
    if let Some(fundamental_freq) = features.fundamental_hz {
        // Check kick frequency match
        if let Some(kick_freq) = tuning_info.kick_hz {
            let kick_distance = (fundamental_freq - kick_freq).abs();
            if kick_distance < config.classification.margins.kick_centroid_tol_hz {
                scores.insert(
                    DrumClass::Kick,
                    0.9 * (1.0
                        - kick_distance / config.classification.margins.kick_centroid_tol_hz),
                );
            }
        }

        // Check tom frequency matches
        for &tom_freq in &tuning_info.toms_hz {
            let tom_distance = (fundamental_freq - tom_freq).abs();
            if tom_distance < config.classification.margins.tom_centroid_tol_hz {
                let score =
                    0.8 * (1.0 - tom_distance / config.classification.margins.tom_centroid_tol_hz);
                let current_score = scores.get(&DrumClass::Tom).unwrap_or(&0.0);
                scores.insert(DrumClass::Tom, current_score.max(score));
            }
        }

        // Check cowbell frequency match (500-1000Hz range)
        if fundamental_freq >= config.classification.cowbell_freq_min
            && fundamental_freq <= config.classification.cowbell_freq_max
        {
            let cowbell_score = 0.8 * (1.0 - ((fundamental_freq - 750.0).abs() / 250.0)); // Center around 750Hz
            let current_cowbell_score = scores.get(&DrumClass::Cowbell).unwrap_or(&0.0);
            scores.insert(
                DrumClass::Cowbell,
                cowbell_score.max(*current_cowbell_score),
            );
        }
    }

    // Rule 2: Spectral centroid ranges with new instruments
    let centroid = features.spectral_centroid_hz;
    if centroid < 150.0 {
        // Very low centroid - likely kick or low tom
        *scores.entry(DrumClass::Kick).or_insert(0.0) += 0.3;
        *scores.entry(DrumClass::Tom).or_insert(0.0) += 0.2;
    } else if centroid < 300.0 {
        // Mid-low centroid - snare, rimshot, or tom
        *scores.entry(DrumClass::Snare).or_insert(0.0) += 0.3;
        *scores.entry(DrumClass::Rimshot).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Tom).or_insert(0.0) += 0.3;
    } else if centroid < 2000.0 {
        // Mid centroid - cowbell, percussion, or cymbals
        *scores.entry(DrumClass::Cowbell).or_insert(0.0) += 0.4;
        *scores.entry(DrumClass::Percussion).or_insert(0.0) += 0.3;
        *scores.entry(DrumClass::Cymbal).or_insert(0.0) += 0.2;
    } else if centroid < 6000.0 {
        // High-mid centroid - hi-hats or cymbals (but not splash extreme)
        *scores.entry(DrumClass::HiHat).or_insert(0.0) += 0.4;
        *scores.entry(DrumClass::Cymbal).or_insert(0.0) += 0.3;
    }
    // Removed: Very high centroid splash rule - now only strict conjunction can classify splash

    // Rule 3: Energy distribution with refined thresholds
    let low_ratio = features.low_energy_ratio;
    let mid_ratio = features.mid_energy_ratio;
    let high_ratio = features.high_energy_ratio;

    if low_ratio > 0.6 {
        // Dominant low frequencies - kick or tom
        *scores.entry(DrumClass::Kick).or_insert(0.0) += 0.4;
        *scores.entry(DrumClass::Tom).or_insert(0.0) += 0.2;
    } else if mid_ratio > 0.5 {
        // Dominant mid frequencies - snare, rimshot, or cowbell
        *scores.entry(DrumClass::Snare).or_insert(0.0) += 0.3;
        *scores.entry(DrumClass::Rimshot).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Cowbell).or_insert(0.0) += 0.3;
    } else if high_ratio > 0.4 {
        // Dominant high frequencies - hi-hat or cymbal
        *scores.entry(DrumClass::HiHat).or_insert(0.0) += 0.3;
        *scores.entry(DrumClass::Cymbal).or_insert(0.0) += 0.3;
    }

    // Rule 4: Attack time with new thresholds
    let attack_ms = features.attack_time_ms;
    if attack_ms < 3.0 {
        // Very fast attack - hi-hat (removed splash scoring)
        *scores.entry(DrumClass::HiHat).or_insert(0.0) += 0.2;
    } else if attack_ms < 5.0 {
        // Fast attack - hi-hat or rimshot
        *scores.entry(DrumClass::HiHat).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Rimshot).or_insert(0.0) += 0.2;
    } else if attack_ms < 15.0 {
        // Medium attack - snare, rimshot, or cowbell
        *scores.entry(DrumClass::Snare).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Rimshot).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Cowbell).or_insert(0.0) += 0.2;
    } else {
        // Slower attack - kick or tom
        *scores.entry(DrumClass::Kick).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Tom).or_insert(0.0) += 0.2;
    }

    // Rule 5: Zero crossing rate (noisiness)
    let zcr = features.zero_crossing_rate;
    if zcr > 0.4 {
        // Very high ZCR - hi-hat (removed splash scoring)
        *scores.entry(DrumClass::HiHat).or_insert(0.0) += 0.2;
    } else if zcr > 0.3 {
        // High ZCR - noisy sound (hi-hat, snare wires, rimshot)
        *scores.entry(DrumClass::HiHat).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Snare).or_insert(0.0) += 0.2;
        *scores.entry(DrumClass::Rimshot).or_insert(0.0) += 0.2;
    }

    // Rule 6: Decay time for specific instruments (balanced approach)
    let decay_ms = features.decay_time_ms;
    
    // Check for cowbell first (most distinctive frequency signature with transient analysis)
    if let Some(fundamental_freq) = features.fundamental_hz {
        if fundamental_freq >= config.classification.cowbell_freq_min &&
           fundamental_freq <= config.classification.cowbell_freq_max &&
           features.attack_time_ms > config.classification.cowbell_attack_threshold &&
           centroid < 2000.0 &&
           mid_ratio > 0.4 &&
           features.transient_energy_ratio > config.classification.transient_energy_threshold &&
           features.attack_to_sustain_ratio > config.classification.attack_sustain_ratio_threshold {
            // Cowbell frequency range with appropriate attack, mid-frequency content, and strong transient characteristics
            *scores.entry(DrumClass::Cowbell).or_insert(0.0) += 0.7;
        }
    }
    
    // Check for splash (ultra-strict conjunction with vetoes)
    let is_splash_candidate = features.spectral_centroid_hz > 10000.0
        && features.attack_hf_ratio > 0.90
        && features.attack_time_ms < 2.0
        && features.decay_time_ms < 80.0    // HF short decay
        && features.decay_time_ms < 200.0   // Mid short decay
        && features.transient_energy_ratio > 0.5
        && features.attack_to_sustain_ratio > 4.0;

    // Explicit veto guards (ultra-conservative)
    let hat_veto = (features.spectral_centroid_hz < 9000.0) || (features.decay_time_ms >= 70.0) || features.zero_crossing_rate > 0.3;
    let crash_veto = (features.decay_time_ms >= 80.0) || (features.decay_time_ms >= 300.0);
    let bell_veto = features.fundamental_hz.is_some_and(|f| (500.0..=1000.0).contains(&f));

    // Splash classification: ultra-strict with vetoes
    if is_splash_candidate && !hat_veto && !crash_veto && !bell_veto {
        // Debug output for splash classifications
        static mut SPLASH_COUNT: usize = 0;
        unsafe {
            SPLASH_COUNT += 1;
            if SPLASH_COUNT <= 5 {
                eprintln!("SPLASH #{}: centroid={:.0}Hz, hf_ratio={:.2}, attack={:.1}ms, decay={:.0}ms, transient_energy={:.2}, attack_sustain={:.1}",
                    SPLASH_COUNT, features.spectral_centroid_hz, features.attack_hf_ratio,
                    features.attack_time_ms, features.decay_time_ms,
                    features.transient_energy_ratio, features.attack_to_sustain_ratio);
                eprintln!("  Veto status: hat={}, crash={}, bell={}",
                    hat_veto, crash_veto, bell_veto);
            }
        }
        *scores.entry(DrumClass::Splash).or_insert(0.0) += 0.8;
    }
    
    // Check for rimshot (combination of snare body + rim emphasis with transient focus)
    if features.attack_time_ms > config.classification.rimshot_attack_threshold && 
       decay_ms < config.classification.rimshot_max_decay_ms &&
       high_ratio > config.classification.rimshot_hf_ratio_min &&
       centroid > 150.0 && centroid < 4000.0 &&
       mid_ratio > 0.3 &&
       features.transient_sharpness > config.classification.rimshot_transient_sharpness_min &&
       features.attack_hf_ratio > config.classification.attack_hf_threshold {
        // Strong attack, short decay, high frequency content, sharp transient, strong attack HF - rimshot
        *scores.entry(DrumClass::Rimshot).or_insert(0.0) += 0.6;
    }

    // Find best classification
    let mut best_class = DrumClass::Unknown;
    let mut best_score = 0.0;

    for (class, score) in &scores {
        if *score > best_score {
            best_score = *score;
            best_class = *class;
        }
    }

    // Create alternatives list (sorted by score)
    let mut alternatives: Vec<(DrumClass, f32)> = scores.into_iter().collect();
    alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    (
        best_class,
        if best_score > 1.0 { 1.0 } else { best_score },
        alternatives,
    )
}

/// Compute confidence scores combining acoustic and prior evidence
fn compute_confidence_scores(
    acoustic_score: f32,
    drum_class: DrumClass,
    features: &ClassificationFeatures,
) -> (f32, f32, f32) {
    // Prior confidence based on feature consistency
    let mut prior_confidence: f32 = 0.5; // Base prior

    // Boost prior for events with strong transient characteristics
    if features.transient_energy_ratio > 0.3 {
        prior_confidence += 0.15;
    }
    
    if features.attack_to_sustain_ratio > 2.0 {
        prior_confidence += 0.1;
    }

    // Boost prior for events with clear fundamental frequency
    if features.fundamental_hz.is_some() {
        prior_confidence += 0.2;
    }

    // Boost prior for events with dominant energy in expected bands
    match drum_class {
        DrumClass::Kick if features.low_energy_ratio > 0.5 => prior_confidence += 0.2,
        DrumClass::Snare if features.mid_energy_ratio > 0.4 => prior_confidence += 0.2,
        DrumClass::Rimshot
            if features.high_energy_ratio > 0.5 && features.attack_time_ms < 10.0 =>
        {
            prior_confidence += 0.3
        }
        DrumClass::HiHat if features.high_energy_ratio > 0.3 => prior_confidence += 0.2,
        DrumClass::Splash
            if features.high_energy_ratio > 0.7 && features.spectral_centroid_hz > 6000.0 =>
        {
            prior_confidence += 0.4
        }
        DrumClass::Cowbell
            if features
                .fundamental_hz
                .is_some_and(|f| (500.0..=1000.0).contains(&f)) =>
        {
            prior_confidence += 0.3
        }
        DrumClass::Cymbal if features.high_energy_ratio > 0.4 => prior_confidence += 0.2,
        _ => {}
    }

    prior_confidence = prior_confidence.min(1.0);

    // Combined confidence (weighted average)
    let acoustic_weight = 0.7;
    let prior_weight = 0.3;
    let combined_confidence = acoustic_score * acoustic_weight + prior_confidence * prior_weight;

    (combined_confidence, acoustic_score, prior_confidence)
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 4: Adaptive Instrument Classification");

    // Get onset events from Pass 2
    if state.onset_events.is_empty() {
        println!("  No onset events found from Pass 2, skipping classification");
        return Ok(());
    }

    // Get whitened spectrogram from Pass 1
    let whitened = state.s_whitened.as_ref().ok_or_else(|| {
        DrumError::ProcessingPipelineError("Pass 1 must be run before Pass 4".to_string())
    })?;

    // Get primary STFT data
    let primary_stft = state
        .stfts
        .get(&(config.stft.n_fft, config.stft.hop_length))
        .ok_or_else(|| DrumError::ProcessingPipelineError("Primary STFT not found".to_string()))?;

    // Get tuning info from Pass 3
    let tuning_info = state.tuning_info.as_ref().ok_or_else(|| {
        DrumError::ProcessingPipelineError("Pass 3 must be run before Pass 4".to_string())
    })?;

    // Classify each onset event
    println!("  Classifying {} onset events...", state.onset_events.len());
    let mut classified_events = Vec::new();

    for (i, event) in state.onset_events.iter().enumerate() {
        // Extract classification features
        let features = extract_classification_features(
            event,
            &state.y,
            whitened,
            &primary_stft.freqs,
            tuning_info,
            config.stft.hop_length,
            state.sr,
        );

        // Perform hierarchical classification
        let (drum_class, acoustic_score, alternatives) =
            classify_drum_event(&features, tuning_info, config);

        // Debug: Log features for first 6 events
        if i < 6 {
            eprintln!("Event {}: centroid={:.0}Hz, hf_ratio={:.2}, attack={:.1}ms, decay={:.0}ms, fund={:.0}Hz, zcr={:.2}, label={}",
                i, features.spectral_centroid_hz, features.attack_hf_ratio,
                features.attack_time_ms, features.decay_time_ms,
                features.fundamental_hz.unwrap_or(0.0), features.zero_crossing_rate,
                drum_class.name());
        }

        // Compute confidence scores
        let (confidence, acoustic_confidence, prior_confidence) =
            compute_confidence_scores(acoustic_score, drum_class, &features);

        // Create classified event
        let classified_event = ClassifiedEvent {
            time_sec: event.time_sec,
            frame_idx: event.frame_idx,
            drum_class,
            confidence,
            acoustic_confidence,
            prior_confidence,
            features,
            alternative_classes: alternatives,
        };

        classified_events.push(classified_event);
    }

    // Store results in state
    state.classified_events = classified_events;
    println!("  ✓ Classified {} events", state.classified_events.len());

    // Print classification summary
    let mut class_counts = std::collections::HashMap::new();
    for event in &state.classified_events {
        *class_counts.entry(event.drum_class).or_insert(0) += 1;
    }

    println!("  Classification summary:");
    for (class, count) in class_counts {
        println!("    {}: {} events", class.name(), count);
    }

    println!("  ✓ Pass 4 complete");

    Ok(())
}
