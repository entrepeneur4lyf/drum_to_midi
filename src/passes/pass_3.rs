//! Pass 3: Track Tuning & Reverb Mask

use crate::analysis::{ReverbInfo, TuningInfo};
use crate::audio::{AudioState, OnsetEvent};
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};
use ndarray::Array2;
use rand::prelude::*;


/// Simple Gaussian component for GMM
#[derive(Debug, Clone)]
struct GaussianComponent {
    weight: f32,
    mean: f32,
    variance: f32,
}

/// Simple Gaussian Mixture Model for 1D data
struct GaussianMixtureModel {
    components: Vec<GaussianComponent>,
    max_iterations: usize,
    tolerance: f32,
}

impl GaussianMixtureModel {
    fn new(n_components: usize) -> Self {
        Self {
            components: vec![
                GaussianComponent {
                    weight: 1.0 / n_components as f32,
                    mean: 0.0,
                    variance: 1.0,
                };
                n_components
            ],
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }

    /// Fit GMM to 1D data
    fn fit(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        // Initialize components with k-means++
        self.initialize_components(data);

        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for _ in 0..self.max_iterations {
            // E-step: compute responsibilities
            let responsibilities = self.e_step(data);

            // M-step: update parameters
            self.m_step(data, &responsibilities);

            // Check convergence
            let log_likelihood = self.log_likelihood(data);
            if (log_likelihood - prev_log_likelihood).abs() < self.tolerance as f64 {
                break;
            }
            prev_log_likelihood = log_likelihood;
        }
    }

    fn initialize_components(&mut self, data: &[f32]) {
        let mut rng = thread_rng();

        // Initialize first component randomly
        let idx = rng.gen_range(0..data.len());
        self.components[0].mean = data[idx];
        self.components[0].variance =
            data.iter().map(|&x| (x - data[idx]).powi(2)).sum::<f32>() / data.len() as f32;

        // Initialize remaining components using k-means++ initialization
        for i in 1..self.components.len() {
            let mut distances = vec![0.0; data.len()];
            for (j, &x) in data.iter().enumerate() {
                let mut min_dist = f32::INFINITY;
                for component in &self.components[0..i] {
                    let dist = (x - component.mean).powi(2);
                    min_dist = min_dist.min(dist);
                }
                distances[j] = min_dist;
            }

            // Sample with probability proportional to squared distance
            let total_dist: f32 = distances.iter().sum();
            let mut r = rng.gen::<f32>() * total_dist;
            let mut idx = 0;
            for (j, &dist) in distances.iter().enumerate() {
                r -= dist;
                if r <= 0.0 {
                    idx = j;
                    break;
                }
            }

            self.components[i].mean = data[idx];
            self.components[i].variance = 1.0;
        }

        // Normalize weights
        let weight = 1.0 / self.components.len() as f32;
        for component in &mut self.components {
            component.weight = weight;
        }
    }

    fn e_step(&self, data: &[f32]) -> Array2<f32> {
        let n_samples = data.len();
        let n_components = self.components.len();
        let mut responsibilities = Array2::<f32>::zeros((n_samples, n_components));

        for i in 0..n_samples {
            let x = data[i];
            let mut total_prob = 0.0;

            // Compute weighted probabilities for each component
            for j in 0..n_components {
                let prob = self.components[j].weight * self.gaussian_pdf(x, &self.components[j]);
                responsibilities[[i, j]] = prob;
                total_prob += prob;
            }

            // Normalize responsibilities
            if total_prob > 0.0 {
                for j in 0..n_components {
                    responsibilities[[i, j]] /= total_prob;
                }
            }
        }

        responsibilities
    }

    fn m_step(&mut self, data: &[f32], responsibilities: &Array2<f32>) {
        let n_components = self.components.len();

        for j in 0..n_components {
            let mut weight_sum = 0.0;
            let mut mean_sum = 0.0;
            let mut var_sum = 0.0;

            for i in 0..data.len() {
                let resp = responsibilities[[i, j]];
                weight_sum += resp;
                mean_sum += resp * data[i];
            }

            if weight_sum > 0.0 {
                let new_mean = mean_sum / weight_sum;
                self.components[j].mean = new_mean;

                // Update variance
                for i in 0..data.len() {
                    let resp = responsibilities[[i, j]];
                    var_sum += resp * (data[i] - new_mean).powi(2);
                }

                self.components[j].variance = var_sum / weight_sum;
                self.components[j].weight = weight_sum / data.len() as f32;
            }
        }
    }

    fn log_likelihood(&self, data: &[f32]) -> f64 {
        data.iter()
            .map(|&x| {
                let prob: f32 = self
                    .components
                    .iter()
                    .map(|c| c.weight * self.gaussian_pdf(x, c))
                    .sum();
                if prob > 0.0 {
                    prob.ln() as f64
                } else {
                    f64::NEG_INFINITY
                }
            })
            .sum()
    }

    fn gaussian_pdf(&self, x: f32, component: &GaussianComponent) -> f32 {
        if component.variance <= 0.0 {
            return 0.0;
        }

        let diff = x - component.mean;
        let exponent = -0.5 * diff * diff / component.variance;
        let coeff = 1.0 / (2.0 * std::f32::consts::PI * component.variance).sqrt();

        coeff * exponent.exp()
    }

    /// Get the most likely component for each data point
    fn predict(&self, data: &[f32]) -> Vec<usize> {
        data.iter()
            .map(|&x| {
                let mut max_prob = 0.0;
                let mut best_component = 0;

                for (j, component) in self.components.iter().enumerate() {
                    let prob = component.weight * self.gaussian_pdf(x, component);
                    if prob > max_prob {
                        max_prob = prob;
                        best_component = j;
                    }
                }

                best_component
            })
            .collect()
    }

    /// Get component means (sorted by frequency)
    fn get_sorted_means(&self) -> Vec<f32> {
        let mut means: Vec<f32> = self.components.iter().map(|c| c.mean).collect();
        means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        means
    }
}

/// Safe ratio computation with EPS protection
fn safe_ratio(num: f32, den: f32, eps: f32) -> f32 {
    let d = if den.abs() < eps { eps } else { den };
    (num / d).clamp(-1e6, 1e6)
}

/// Sanitize NaN/Inf values
fn sanitize(x: f32, fallback: f32) -> f32 {
    if x.is_finite() { x } else { fallback }
}

/// Compute silhouette score for clustering quality (NaN-proof)
fn silhouette_score(data: &[f32], labels: &[usize], n_clusters: usize) -> f32 {
    const EPS: f32 = 1e-12;

    if n_clusters <= 1 || data.len() < 2 {
        return 0.0;
    }

    let mut scores = Vec::new();

    for i in 0..data.len() {
        let cluster_i = labels[i];
        let x_i = data[i];

        // Compute a(i): average distance to other points in same cluster
        let mut a_i = 0.0;
        let mut count_same_cluster = 0;

        for j in 0..data.len() {
            if i != j && labels[j] == cluster_i {
                a_i += (x_i - data[j]).abs();
                count_same_cluster += 1;
            }
        }

        if count_same_cluster > 0 {
            a_i = safe_ratio(a_i, count_same_cluster as f32, EPS);
        }

        // Compute b(i): minimum average distance to points in other clusters
        let mut b_i = f32::INFINITY;

        for c in 0..n_clusters {
            if c == cluster_i {
                continue;
            }

            let mut dist_sum = 0.0;
            let mut count_other_cluster = 0;

            for j in 0..data.len() {
                if labels[j] == c {
                    dist_sum += (x_i - data[j]).abs();
                    count_other_cluster += 1;
                }
            }

            if count_other_cluster > 0 {
                let avg_dist = safe_ratio(dist_sum, count_other_cluster as f32, EPS);
                b_i = b_i.min(avg_dist);
            }
        }

        // Compute silhouette score for this point
        if b_i > EPS {
            if a_i < b_i {
                scores.push(safe_ratio(b_i - a_i, b_i, EPS));
            } else {
                scores.push(0.0); // Point is closer to its own cluster than others
            }
        } else if a_i < EPS {
            scores.push(1.0); // Perfect clustering (no distance to own cluster)
        } else {
            scores.push(0.0); // Degenerate case
        }
    }

    if scores.is_empty() {
        0.0
    } else {
        let sum: f32 = scores.iter().sum();
        safe_ratio(sum, scores.len() as f32, EPS)
    }
}

/// Extract spectral peaks from onset events for tuning analysis
fn extract_spectral_peaks(
    onset_events: &[OnsetEvent],
    mag: &Array2<f32>,
    freqs: &[f32],
    tempo_bpm: f32,
    sr: u32,
    hop: usize,
) -> Vec<f32> {
    let mut spectral_peaks = Vec::new();

    // Tempo-adaptive window size (40ms min, 0.6 × 16th note max)
    let sixteenth_note_sec = 60.0 / tempo_bpm / 4.0;
    let window_frames = ((0.04f32).max(sixteenth_note_sec * 0.6) * sr as f32 / hop as f32) as usize;

    for event in onset_events {
        let frame_idx = event.frame_idx;

        // Extract spectrum around the onset
        let start_frame = frame_idx.saturating_sub(window_frames / 2);
        let end_frame = (frame_idx + window_frames / 2).min(mag.shape()[1] - 1);

        if end_frame > start_frame {
            // Average spectrum across the window
            let mut avg_spectrum = vec![0.0; mag.shape()[0]];
            let window_size = end_frame - start_frame + 1;

            for f in 0..mag.shape()[0] {
                for t in start_frame..=end_frame {
                    avg_spectrum[f] += mag[[f, t]];
                }
                avg_spectrum[f] /= window_size as f32;
            }

            // Find peaks in the spectrum (simple local maxima)
            for i in 1..avg_spectrum.len().saturating_sub(1) {
                if avg_spectrum[i] > avg_spectrum[i - 1] && avg_spectrum[i] > avg_spectrum[i + 1] {
                    // Only consider peaks above threshold and in drum frequency range
                    if avg_spectrum[i] > 0.01 && freqs[i] >= 30.0 && freqs[i] <= 500.0 {
                        spectral_peaks.push(freqs[i]);
                    }
                }
            }
        }
    }

    spectral_peaks
}

/// Perform GMM clustering on spectral peaks
fn cluster_spectral_peaks(peaks: &[f32], min_samples: usize) -> Option<(Vec<f32>, f32)> {
    if peaks.len() < min_samples {
        return None;
    }

    // Determine number of components based on sample count
    let n_components = if peaks.len() < 10 {
        1
    } else if peaks.len() < 30 {
        2
    } else {
        3
    };

    let mut gmm = GaussianMixtureModel::new(n_components);
    gmm.fit(peaks);

    // Get cluster centers (fundamental frequencies)
    let centers = gmm.get_sorted_means();

    // Compute silhouette score for quality assessment
    let labels = gmm.predict(peaks);
    let mut coherence = silhouette_score(peaks, &labels, n_components);

    // Ensure coherence is never NaN (can happen with degenerate clusters)
    if coherence.is_nan() {
        coherence = 0.0; // Default to poor clustering quality
    }

    Some((centers, coherence))
}

/// Estimate reverb characteristics from the audio
fn estimate_reverb_characteristics(state: &AudioState, onset_events: &[OnsetEvent]) -> ReverbInfo {
    // Simple reverb estimation based on decay analysis
    // This is a simplified implementation

    let mut rt60_estimate = 1.5; // Default 1.5 seconds
    let mut strength = 0.3; // Default moderate reverb

    if !onset_events.is_empty() {
        // Analyze decay patterns in the whitened spectrogram
        if let Some(whitened) = &state.s_whitened {
            // Look at late portions of the audio for reverb tail
            let late_start_frame = (state.duration_sec() * 0.7 * state.sr as f32 / 512.0) as usize;
            let late_end_frame = whitened.shape()[1]
                .min((state.duration_sec() * 0.9 * state.sr as f32 / 512.0) as usize);

            if late_end_frame > late_start_frame {
                // Compute RMS in late portion
                let mut late_energy = 0.0;
                let mut total_frames = 0;

                for t in late_start_frame..late_end_frame {
                    for f in 0..whitened.shape()[0] {
                        late_energy += whitened[[f, t]].powi(2);
                    }
                    total_frames += 1;
                }

                if total_frames > 0 {
                    late_energy =
                        (late_energy / (total_frames * whitened.shape()[0]) as f32).sqrt();

                    // Estimate RT60 based on late energy level
                    // Lower late energy suggests less reverb
                    rt60_estimate = (late_energy * 3.0).max(0.5).min(3.0);
                    strength = late_energy.min(0.8);
                }
            }
        }
    }

    ReverbInfo {
        rt60_estimate_ms: rt60_estimate * 1000.0,
        strength,
    }
}

/// Generate reverb suppression mask
fn generate_reverb_mask(reverb_info: &ReverbInfo, freqs: &[f32], n_frames: usize) -> Array2<f32> {
    let mut mask = Array2::<f32>::ones((freqs.len(), n_frames));

    // Apply frequency-dependent reverb suppression
    // Higher frequencies typically have less reverb
    for (f_idx, &freq) in freqs.iter().enumerate() {
        // Exponential decay model
        let freq_factor = if freq < 200.0 {
            1.0 // Low frequencies have more reverb
        } else if freq < 2000.0 {
            0.7 // Mid frequencies moderate reverb
        } else {
            0.3 // High frequencies less reverb
        };

        let suppression = 1.0 - (reverb_info.strength * freq_factor);

        for t in 0..n_frames {
            // Time-varying suppression (more suppression later in the track)
            let time_factor = 1.0 - (t as f32 / n_frames as f32) * 0.3;
            mask[[f_idx, t]] = suppression * time_factor;
        }
    }

    mask
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 3: Track Tuning & Reverb Mask");

    // Get onset events from Pass 2
    if state.onset_events.is_empty() {
        println!("  No onset events found from Pass 2, skipping tuning analysis");
        return Ok(());
    }

    // Get whitened spectrogram from Pass 1
    let whitened = state.s_whitened.as_ref().ok_or_else(|| {
        DrumError::ProcessingPipelineError("Pass 1 must be run before Pass 3".to_string())
    })?;

    // Get primary STFT data
    let primary_stft = state
        .stfts
        .get(&(config.stft.n_fft, config.stft.hop_length))
        .ok_or_else(|| DrumError::ProcessingPipelineError("Primary STFT not found".to_string()))?;

    // Extract spectral peaks from onset events
    println!(
        "  Extracting spectral peaks from {} onset events...",
        state.onset_events.len()
    );
    let tempo_bpm = 120.0; // TODO: Get from analysis
    let spectral_peaks = extract_spectral_peaks(
        &state.onset_events,
        whitened,
        &primary_stft.freqs,
        tempo_bpm,
        state.sr,
        config.stft.hop_length,
    );

    println!(
        "  Found {} spectral peaks for clustering",
        spectral_peaks.len()
    );

    // Perform GMM clustering
    let (tuning_info, _fallback_mode) = if let Some((centers, coherence)) =
        cluster_spectral_peaks(&spectral_peaks, config.clustering.min_samples)
    {
        println!("  Clustering coherence: {:.3}", coherence);

        // Check if clustering is reliable
        let reliable_clustering = coherence.is_finite() && coherence >= 0.0 && coherence <= 1.0;

        if reliable_clustering {
            // Interpret clusters as drum fundamentals
            let mut kick_hz = None;
            let mut toms_hz = Vec::new();

            for &center in &centers {
                if center >= config.clustering.priors.kick_hz_min
                    && center <= config.clustering.priors.kick_hz_max
                {
                    kick_hz = Some(center);
                } else if center > config.clustering.priors.kick_hz_max {
                    // Assume toms are higher frequency than kick
                    toms_hz.push(center);
                }
            }

            let toms_count = toms_hz.len();

            (TuningInfo {
                kick_hz,
                kick_confidence: coherence,
                kick_coherence: coherence,
                toms_hz,
                toms_confidence: vec![coherence; toms_count],
                toms_coherence: vec![coherence; toms_count],
                snare_body_hz: 200.0, // Default estimate
                snare_body_range_hz: [150.0, 250.0],
            }, false)
        } else {
            println!("  Warning: Clustering coherence unreliable ({:.3}), entering dry conservative fallback", coherence);
            (TuningInfo {
                kick_hz: None,
                kick_confidence: 0.0,
                kick_coherence: 0.0,
                toms_hz: Vec::new(),
                toms_confidence: Vec::new(),
                toms_coherence: Vec::new(),
                snare_body_hz: 200.0,
                snare_body_range_hz: [150.0, 250.0],
            }, true)
        }
    } else {
        println!("  Insufficient data for clustering, using defaults");
        (TuningInfo {
            kick_hz: None,
            kick_confidence: 0.0,
            kick_coherence: 0.0,
            toms_hz: Vec::new(),
            toms_confidence: Vec::new(),
            toms_coherence: Vec::new(),
            snare_body_hz: 200.0,
            snare_body_range_hz: [150.0, 250.0],
        }, true)
    };

    // Estimate reverb characteristics
    println!("  Estimating reverb characteristics...");
    let reverb_info = estimate_reverb_characteristics(state, &state.onset_events);

    println!(
        "  RT60 estimate: {:.1}ms, strength: {:.2}",
        reverb_info.rt60_estimate_ms, reverb_info.strength
    );

    // Generate reverb suppression mask
    println!("  Generating reverb suppression mask...");
    let _reverb_mask = generate_reverb_mask(&reverb_info, &primary_stft.freqs, whitened.shape()[1]);

    // Store results in state
    state.tuning_info = Some(tuning_info);
    state.reverb_info = Some(reverb_info);

    println!("  ✓ Tuning analysis complete");
    println!("  ✓ Reverb characteristics estimated");

    println!("  ✓ Pass 3 complete");

    Ok(())
}
