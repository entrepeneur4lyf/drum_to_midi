//! Error types for the drum-to-MIDI system

use std::fmt;

/// Custom error type for drum-to-MIDI processing
#[derive(Debug, Clone)]
pub enum DrumError {
    /// E001: Invalid audio format (e.g., non-PCM WAV)
    InvalidAudioFormat(String),
    /// E002: Unsupported sample rate
    UnsupportedSampleRate(u32),
    /// E003: Configuration validation failed
    ConfigValidationFailed(String),
    /// E004: Insufficient events for clustering (< min_samples)
    InsufficientEventsForClustering(usize),
    /// E005: Audio file I/O error
    AudioFileError(String),
    /// E006: STFT processing error
    StftProcessingError(String),
    /// E007: Memory allocation error
    MemoryAllocationError(String),
    /// E008: Invalid configuration parameter
    InvalidConfigParameter(String),
    /// E009: Processing pipeline error
    ProcessingPipelineError(String),
    /// E010: MIDI export error
    MidiExportError(String),
    /// E011: Analysis export error
    AnalysisExportError(String),
    /// E012: QA artifact generation error
    QaGenerationError(String),
    /// E013: Input validation error
    InputValidationError(String),
    /// E014: Spectral processing error
    SpectralProcessingError(String),
    /// E015: Classification error
    ClassificationError(String),
}

impl fmt::Display for DrumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrumError::InvalidAudioFormat(msg) => {
                write!(f, "E001: Invalid audio format - {}", msg)
            }
            DrumError::UnsupportedSampleRate(sr) => {
                write!(f, "E002: Unsupported sample rate {} Hz", sr)
            }
            DrumError::ConfigValidationFailed(msg) => {
                write!(f, "E003: Configuration validation failed - {}", msg)
            }
            DrumError::InsufficientEventsForClustering(count) => {
                write!(
                    f,
                    "E004: Insufficient events for clustering ({} < min_samples)",
                    count
                )
            }
            DrumError::AudioFileError(msg) => {
                write!(f, "E005: Audio file I/O error - {}", msg)
            }
            DrumError::StftProcessingError(msg) => {
                write!(f, "E006: STFT processing error - {}", msg)
            }
            DrumError::MemoryAllocationError(msg) => {
                write!(f, "E007: Memory allocation error - {}", msg)
            }
            DrumError::InvalidConfigParameter(msg) => {
                write!(f, "E008: Invalid configuration parameter - {}", msg)
            }
            DrumError::ProcessingPipelineError(msg) => {
                write!(f, "E009: Processing pipeline error - {}", msg)
            }
            DrumError::MidiExportError(msg) => {
                write!(f, "E010: MIDI export error - {}", msg)
            }
            DrumError::AnalysisExportError(msg) => {
                write!(f, "E011: Analysis export error - {}", msg)
            }
            DrumError::QaGenerationError(msg) => {
                write!(f, "E012: QA artifact generation error - {}", msg)
            }
            DrumError::InputValidationError(msg) => {
                write!(f, "E013: Input validation error - {}", msg)
            }
            DrumError::SpectralProcessingError(msg) => {
                write!(f, "E014: Spectral processing error - {}", msg)
            }
            DrumError::ClassificationError(msg) => {
                write!(f, "E015: Classification error - {}", msg)
            }
        }
    }
}

impl std::error::Error for DrumError {}

// From implementations for common error types
impl From<std::io::Error> for DrumError {
    fn from(err: std::io::Error) -> Self {
        DrumError::AudioFileError(format!("File I/O error: {}", err))
    }
}

impl From<serde_json::Error> for DrumError {
    fn from(err: serde_json::Error) -> Self {
        DrumError::AnalysisExportError(format!("JSON serialization error: {}", err))
    }
}

impl From<anyhow::Error> for DrumError {
    fn from(err: anyhow::Error) -> Self {
        DrumError::ProcessingPipelineError(format!("Generic error: {}", err))
    }
}

// Note: Plotters errors are handled manually in the code due to complex type parameters

/// Result type alias for drum-to-MIDI operations
pub type Result<T> = std::result::Result<T, DrumError>;
