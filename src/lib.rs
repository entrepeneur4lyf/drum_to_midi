//! Drum-to-MIDI Transcription System
//!
//! A deterministic, non-ML audio signal processing system that extracts
//! musically coherent MIDI from isolated drum recordings.

pub mod analysis;
pub mod audio;
pub mod config;
pub mod error;
pub mod midi;
pub mod passes;
pub mod qa;
pub mod spectral;

pub use audio::AudioState;
pub use config::Config;
pub use error::{DrumError, Result as DrumErrorResult};

use std::path::Path;

/// Main processing pipeline for drum-to-MIDI conversion
pub struct DrumToMidi {
    config: Config,
}

impl DrumToMidi {
    /// Create a new processor with the given configuration
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Process an audio file and generate MIDI output
    pub fn process<P: AsRef<Path>>(&self, input_path: P, output_dir: P) -> DrumErrorResult<()> {
        // Load audio
        let mut state = AudioState::load(input_path, &self.config)?;

        // Run all passes
        self.run_pipeline(&mut state)?;

        // Export results
        self.export_results(&state, output_dir)?;

        Ok(())
    }

    /// Execute the complete multi-pass pipeline
    fn run_pipeline(&self, state: &mut AudioState) -> DrumErrorResult<()> {
        // Pass 0: Preflight & Normalization
        passes::pass_0::run(state, &self.config)?;

        // Pass 1: Spectral Envelope & Whitening
        passes::pass_1::run(state, &self.config)?;

        // Pass 2: High-Recall Onset Seeding
        passes::pass_2::run(state, &self.config)?;

        // Pass 3: Track Tuning & Reverb Mask
        passes::pass_3::run(state, &self.config)?;

        // Pass 4: Adaptive Instrument Classification
        passes::pass_4::run(state, &self.config)?;

        // Pass 5: Class-Specific Timing Refinement
        passes::pass_5::run(state, &self.config)?;

        // Pass 6: Tempo, Meter, Swing Detection
        passes::pass_6::run(state, &self.config)?;

        // Pass 7: Self-Prior Construction
        passes::pass_7::run(state, &self.config)?;

        // Pass 8: Grid Inference + Fill/Silence Protection
        passes::pass_8::run(state, &self.config)?;

        // Pass 9: Velocity Estimation
        passes::pass_9::run(state, &self.config)?;

        // Pass 10: Post-Processing & Export
        passes::pass_10::run(state, &self.config)?;

        Ok(())
    }

    /// Export MIDI and analysis results
    fn export_results<P: AsRef<Path>>(
        &self,
        state: &AudioState,
        output_dir: P,
    ) -> DrumErrorResult<()> {
        midi::export_midi(state, output_dir.as_ref(), &self.config)?;
        analysis::export_analysis(state, output_dir.as_ref())?;
        qa::generate_artifacts(state, output_dir.as_ref())?;
        Ok(())
    }
}

/// Validate configuration and input files
pub fn validate_input<P: AsRef<Path>>(input_path: P, config: &Config) -> DrumErrorResult<()> {
    // Check input file exists and is valid audio
    audio::validate_audio_file(input_path)?;

    // Validate configuration
    config::validate_config(config)?;

    Ok(())
}
