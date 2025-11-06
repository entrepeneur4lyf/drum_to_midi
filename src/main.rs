use clap::{Parser, Subcommand};
use drum2midi::{validate_input, Config, DrumToMidi};
use std::path::PathBuf;

/// Drum-to-MIDI Transcription System
#[derive(Parser)]
#[command(name = "drum2midi")]
#[command(about = "Convert isolated drum recordings to MIDI with high precision")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze audio file and generate MIDI output
    Analyze {
        /// Input audio file (WAV/AIFF)
        input: PathBuf,

        /// Output directory for results
        #[arg(short, long, default_value = "./output")]
        output: PathBuf,

        /// Custom configuration file
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Music style hint
        #[arg(long)]
        style: Option<String>,

        /// Expected BPM range (e.g., "80-140")
        #[arg(long)]
        bpm: Option<String>,

        /// Expected meter (e.g., "4/4")
        #[arg(long)]
        meter: Option<String>,

        /// Swing detection mode
        #[arg(long, default_value = "auto")]
        swing: String,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Quiet output
        #[arg(short, long)]
        quiet: bool,
    },
    /// Validate configuration file
    ValidateConfig {
        /// Configuration file to validate
        config: PathBuf,
    },
    /// Show default configuration
    ShowConfig,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Analyze {
            input,
            output,
            config,
            style,
            bpm,
            meter,
            swing,
            verbose,
            quiet,
        } => {
            if verbose && quiet {
                anyhow::bail!("Cannot specify both --verbose and --quiet");
            }

            // Load configuration
            let config = if let Some(config_path) = config {
                drum2midi::config::load_config(config_path)?
            } else {
                Config::default()
            };

            // Validate input
            validate_input(&input, &config)?;

            // Create processor
            let processor = DrumToMidi::new(config);

            // Process audio
            if !quiet {
                println!("Processing {}...", input.display());
            }

            processor.process(&input, &output)?;

            if !quiet {
                println!("Results saved to {}", output.display());
            }
        }
        Commands::ValidateConfig { config } => {
            let config = drum2midi::config::load_config(config)?;
            println!("Configuration is valid");
            if let Ok(json) = serde_json::to_string_pretty(&config) {
                println!("{}", json);
            }
        }
        Commands::ShowConfig => {
            let config = Config::default();
            let json = serde_json::to_string_pretty(&config)?;
            println!("{}", json);
        }
    }

    Ok(())
}
