//! Pass 1: Spectral Envelope & Whitening

use crate::audio::AudioState;
use crate::config::Config;
use crate::error::{DrumError, Result as DrumErrorResult};
use crate::spectral::{magnitude_spectrogram, smooth_1_3_octave, stft};
use ndarray::{s, Array2, Axis};
use plotters::prelude::*;
use std::collections::HashMap;
use std::path::Path;

/// Compute percentile along an axis
fn percentile_axis(data: &Array2<f32>, p: f32, axis: Axis) -> Vec<f32> {
    let mut result = Vec::new();

    match axis {
        Axis(0) => {
            // Along frequency axis (columns)
            for col in 0..data.shape()[1] {
                let mut values: Vec<f32> = data.column(col).iter().cloned().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((p / 100.0) * (values.len() as f32 - 1.0)) as usize;
                result.push(values[idx]);
            }
        }
        Axis(1) => {
            // Along time axis (rows)
            for row in 0..data.shape()[0] {
                let mut values: Vec<f32> = data.row(row).iter().cloned().collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((p / 100.0) * (values.len() as f32 - 1.0)) as usize;
                result.push(values[idx]);
            }
        }
        _ => panic!("Unsupported axis"),
    }

    result
}

/// Compute robust whitening curve using percentile-based envelope estimation
fn compute_whitening_curve_robust(
    mag: &Array2<f32>,
    freqs: &[f32],
    config: &Config,
) -> (Array2<f32>, Array2<f32>) {
    // Calculate RMS per frame to find active regions
    let frame_rms: Vec<f32> = mag
        .outer_iter()
        .map(|col| {
            let sum_sq: f32 = col.iter().map(|&x| x * x).sum();
            (sum_sq / col.len() as f32).sqrt()
        })
        .collect();

    // Find frames above RMS threshold (30th percentile of frame RMS)
    let mut rms_sorted = frame_rms.clone();
    rms_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rms_threshold_idx = ((30.0 / 100.0) * (rms_sorted.len() as f32 - 1.0)) as usize;
    let rms_threshold = rms_sorted[rms_threshold_idx];

    let active_frames: Vec<usize> = frame_rms
        .iter()
        .enumerate()
        .filter(|(_, &rms)| rms > rms_threshold)
        .map(|(i, _)| i)
        .filter(|&i| i < mag.shape()[1]) // Ensure indices are within bounds
        .collect();

    // Compute percentile envelope from active frames
    let e_percentile = if active_frames.is_empty() {
        // If no frames are above threshold (e.g., silence), use all frames
        percentile_axis(mag, config.whitening.percentile, Axis(1))
    } else {
        percentile_axis(
            &mag.select(Axis(1), &active_frames),
            config.whitening.percentile,
            Axis(1),
        )
    };

    // Compute whitening weights
    let w_raw: Vec<f32> = e_percentile.iter().map(|&e| 1.0 / (e + 1e-8)).collect();

    // Smooth with 1/3 octave smoothing
    let w_smooth = smooth_1_3_octave(&w_raw, freqs);

    // Apply whitening to spectrogram
    let w_matrix = Array2::from_shape_vec(
        (mag.shape()[0], mag.shape()[1]),
        w_smooth.iter().cycle().take(mag.len()).cloned().collect(),
    )
    .unwrap();
    let s_whitened = mag * &w_matrix;

    (s_whitened, w_matrix)
}

/// Compute adaptive time-varying whitening curve
fn compute_whitening_curve_adaptive(
    mag: &Array2<f32>,
    freqs: &[f32],
    sr: u32,
    hop: usize,
    config: &Config,
) -> (Array2<f32>, Array2<f32>) {
    let window_frames = (config.whitening.adaptive_window_sec * sr as f32 / hop as f32) as usize;
    let mut w_time_varying = Array2::<f32>::zeros(mag.raw_dim());

    for i in 0..mag.shape()[1] {
        let start = 0.max(i as i32 - window_frames as i32 / 2) as usize;
        let end = (mag.shape()[1]).min(i + window_frames / 2);

        let mag_window = mag.slice(s![.., start..end]).to_owned();

        // Compute local percentile envelope
        let e_local = percentile_axis(&mag_window, config.whitening.percentile, Axis(1));

        // Compute local whitening weights
        let w_local: Vec<f32> = e_local.iter().map(|&e| 1.0 / (e + 1e-8)).collect();

        // Smooth locally
        let w_smooth_local = smooth_1_3_octave(&w_local, freqs);

        // Store in time-varying matrix
        for (f, &w) in w_smooth_local.iter().enumerate() {
            w_time_varying[[f, i]] = w;
        }
    }

    // Apply time-varying whitening
    let s_whitened = mag.clone() * &w_time_varying;

    (s_whitened, w_time_varying)
}

/// Generate QA plots for spectral envelope E(f) and whitening function W(f)
fn generate_qa_plots(freqs: &[f32], w_curve: &Array2<f32>, _config: &Config) -> DrumErrorResult<()> {
    // Create output directory if it doesn't exist
    let output_dir = "qa_output";
    std::fs::create_dir_all(output_dir).map_err(|e| {
        DrumError::QaGenerationError(format!("Failed to create QA output directory: {}", e))
    })?;

    // Plot whitening function W(f) - use first column for static, or average for time-varying
    let w_data: Vec<f32> = if w_curve.shape()[1] == 1 {
        // Static whitening - single column
        w_curve.column(0).iter().cloned().collect()
    } else {
        // Time-varying whitening - average across time
        w_curve
            .outer_iter()
            .map(|row| row.iter().sum::<f32>() / row.len() as f32)
            .collect()
    };

    // Create W(f) plot
    let w_plot_path = Path::new(output_dir).join("pass1_whitening_curve.png");
    let root = BitMapBackend::new(&w_plot_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to create plot: {}", e)))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Pass 1: Whitening Function W(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (freqs[0]..freqs[freqs.len() - 1]).log_scale(),
            w_data.iter().cloned().fold(f32::INFINITY, f32::min)
                ..w_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to build chart: {}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Whitening Weight")
        .draw()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to configure mesh: {}", e)))?;

    chart
        .draw_series(LineSeries::new(
            freqs.iter().zip(w_data.iter()).map(|(&f, &w)| (f, w)),
            &BLUE,
        ))
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw series: {}", e)))?;

    root.present()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to present plot: {}", e)))?;

    // Plot spectral envelope E(f) - this would be the inverse of W(f)
    let e_data: Vec<f32> = w_data.iter().map(|&w| 1.0 / (w + 1e-8)).collect();

    let e_plot_path = Path::new(output_dir).join("pass1_spectral_envelope.png");
    let root = BitMapBackend::new(&e_plot_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to create plot: {}", e)))?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Pass 1: Spectral Envelope E(f)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            (freqs[0]..freqs[freqs.len() - 1]).log_scale(),
            e_data.iter().cloned().fold(f32::INFINITY, f32::min)
                ..e_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        )
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to build chart: {}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Frequency (Hz)")
        .y_desc("Envelope Magnitude")
        .draw()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to configure mesh: {}", e)))?;

    chart
        .draw_series(LineSeries::new(
            freqs.iter().zip(e_data.iter()).map(|(&f, &e)| (f, e)),
            &RED,
        ))
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to draw series: {}", e)))?;

    root.present()
        .map_err(|e| DrumError::QaGenerationError(format!("Failed to present plot: {}", e)))?;

    Ok(())
}

pub fn run(state: &mut AudioState, config: &Config) -> DrumErrorResult<()> {
    println!("Pass 1: Spectral Envelope & Whitening");

    // Get audio to process (use processed audio if available, otherwise original)
    let audio_to_process = state.y_processed.as_ref().unwrap_or(&state.y);

    // Compute multi-resolution STFTs
    println!("  Computing multi-resolution STFTs...");
    let stft_configs = &config.stft.multi_res_configs;
    let mut stfts = HashMap::new();

    for &(n_fft, hop) in stft_configs {
        let stft_data = stft(audio_to_process, n_fft, hop, &config.stft.window, state.sr);
        stfts.insert((n_fft, hop), stft_data);
    }

    // Get primary STFT data for processing
    let primary_n_fft = config.stft.n_fft;
    let primary_hop = config.stft.hop_length;
    let primary_stft_data = stft(
        audio_to_process,
        primary_n_fft,
        primary_hop,
        &config.stft.window,
        state.sr,
    );

    // Ensure we have valid frequency data
    if primary_stft_data.freqs.is_empty() {
        return Err(DrumError::StftProcessingError(
            "STFT produced no frequency bins - check audio length".to_string(),
        ));
    }

    // Compute magnitude spectrogram
    let mag = magnitude_spectrogram(&primary_stft_data);

    // Compute whitening curve
    println!("  Computing spectral whitening...");
    let (s_whitened, w_curve) = if config.whitening.adaptive {
        println!("    Using adaptive time-varying whitening");
        compute_whitening_curve_adaptive(
            &mag,
            &primary_stft_data.freqs,
            state.sr,
            primary_hop,
            config,
        )
    } else {
        println!("    Using robust static whitening");
        compute_whitening_curve_robust(&mag, &primary_stft_data.freqs, config)
    };

    // Store results in state
    state.stfts = stfts;
    state.s_whitened = Some(s_whitened);

    println!(
        "  ✓ Whitening curve computed ({} frequency bins)",
        w_curve.shape()[0]
    );
    println!("  ✓ Spectrograms whitened and stored");

    // Generate QA plots if enabled
    if config.qa.generate_images {
        println!("  Generating QA plots...");
        if let Err(e) = generate_qa_plots(&primary_stft_data.freqs, &w_curve, config) {
            eprintln!("  Warning: Failed to generate QA plots: {}", e);
        } else {
            println!("  ✓ QA plots generated");
        }
    }

    println!("  ✓ Pass 1 complete");
    Ok(())
}
