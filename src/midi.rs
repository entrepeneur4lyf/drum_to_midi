//! MIDI export functionality

use crate::analysis::DrumClass;
use crate::audio::AudioState;
use crate::config::Config;
use midly::num::{u15, u24, u28, u4, u7};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, TrackEvent, TrackEventKind};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

/// Export MIDI file from processed audio state
pub fn export_midi(
    state: &AudioState,
    output_dir: &std::path::Path,
    config: &Config,
) -> crate::DrumErrorResult<()> {
    if state.midi_events.is_empty() {
        eprintln!("Warning: No MIDI events to export");
        return Ok(());
    }

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(output_dir)?;

    // Generate filename based on input audio (if available) or use default
    let midi_filename = "transcription.mid";
    let midi_path = output_dir.join(midi_filename);

    // Convert events to MIDI format
    let midi_data = convert_events_to_midi(state, config)?;

    // Write MIDI file
    let mut file = File::create(&midi_path)?;
    file.write_all(&midi_data)?;

    println!(
        "Exported {} MIDI events to {}",
        state.midi_events.len(),
        midi_path.display()
    );
    Ok(())
}

/// Convert MidiEvent vector to MIDI file bytes with proper note duration tracking
fn convert_events_to_midi(state: &AudioState, config: &Config) -> crate::DrumErrorResult<Vec<u8>> {
    // MIDI timing parameters
    let ppq = 960; // Pulses per quarter note (standard resolution)
    let tempo_bpm = state
        .tempo_meter_analysis
        .as_ref()
        .map(|t| t.bpm)
        .unwrap_or(120.0);

    // Convert BPM to microseconds per quarter note
    let tempo_uspq = (60_000_000.0 / tempo_bpm) as u32;

    // Sort events by time
    let mut events = state.midi_events.clone();
    events.sort_by(|a, b| a.time_sec.partial_cmp(&b.time_sec).unwrap());

    // Track active notes for duration calculation
    let mut active_notes: HashMap<(u8, DrumClass), u32> = HashMap::new(); // (midi_note, class) -> start_tick

    // Build MIDI track events
    let mut track_events = Vec::new();
    let mut current_tick = 0u32;

    // Add tempo meta event at the beginning
    track_events.push(TrackEvent {
        delta: u28::from(current_tick),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::from(tempo_uspq))),
    });

    // Add time signature if available
    if let Some(tempo_analysis) = &state.tempo_meter_analysis {
        // Parse meter string like "4/4" to get numerator/denominator
        let (numerator, denominator) = parse_meter_string(&tempo_analysis.meter);
        // TimeSignature expects different arguments - using simplified version for now
        track_events.push(TrackEvent {
            delta: u28::from(0), // At the same time as tempo
            kind: TrackEventKind::Meta(MetaMessage::TimeSignature(
                numerator,
                denominator,
                24, // MIDI clocks per metronome click
                8,  // 32nd notes per quarter note
            )),
        });
    }

    // Process events in chronological order
    for event in &events {
        // Convert time in seconds to MIDI ticks
        let event_tick = (event.time_sec * ppq as f32 * tempo_bpm / 60.0) as u32;
        let delta_ticks = event_tick - current_tick;
        current_tick = event_tick;

        // Skip ghost notes if configured to do so
        if event.is_ghost_note && !should_include_ghost_notes(config) {
            continue;
        }

        let note = event.drum_class.midi_note();
        let note_key = (note, event.drum_class);

        // Check if we need to end any previous note of the same class
        if let Some(start_tick) = active_notes.get(&note_key) {
            // Calculate duration and add note-off
            let duration_ticks = event_tick - start_tick;
            if duration_ticks > 0 {
                track_events.push(TrackEvent {
                    delta: u28::from(delta_ticks),
                    kind: TrackEventKind::Midi {
                        channel: u4::from(9),
                        message: MidiMessage::NoteOff {
                            key: u7::from(note),
                            vel: u7::from(0),
                        },
                    },
                });
                current_tick += delta_ticks;
            }
        }

        // Add note-on event
        track_events.push(TrackEvent {
            delta: if active_notes.contains_key(&note_key) {
                u28::from(0)
            } else {
                u28::from(delta_ticks)
            },
            kind: TrackEventKind::Midi {
                channel: u4::from(9),
                message: MidiMessage::NoteOn {
                    key: u7::from(note),
                    vel: u7::from(event.velocity),
                },
            },
        });

        // Record when this note started
        active_notes.insert(note_key, event_tick);
    }

    // Add final note-off events for any remaining active notes
    for (note_key, _start_tick) in active_notes {
        let (note, _) = note_key;
        let duration_ticks = (ppq as f32 * 0.1) as u32; // Default 100ms duration for hanging notes

        track_events.push(TrackEvent {
            delta: u28::from(duration_ticks),
            kind: TrackEventKind::Midi {
                channel: u4::from(9),
                message: MidiMessage::NoteOff {
                    key: u7::from(note),
                    vel: u7::from(0),
                },
            },
        });
    }

    // Add end of track meta event
    track_events.push(TrackEvent {
        delta: u28::from(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    // Create MIDI file
    let header = Header {
        format: Format::SingleTrack,
        timing: midly::Timing::Metrical(u15::from(ppq)),
    };

    let smf = Smf {
        header,
        tracks: vec![track_events],
    };

    // Serialize to bytes
    let mut bytes = Vec::new();
    smf.write(&mut bytes)
        .map_err(|e| anyhow::anyhow!("Failed to write MIDI data: {:?}", e))?;
    Ok(bytes)
}

/// Parse meter string like "4/4" into (numerator, denominator_log2)
fn parse_meter_string(meter: &str) -> (u8, u8) {
    let parts: Vec<&str> = meter.split('/').collect();
    if parts.len() == 2 {
        let numerator = parts[0].parse().unwrap_or(4);
        let denominator = parts[1].parse().unwrap_or(4);
        // MIDI time signature uses log2 of denominator
        let denominator_log2 = (denominator as f32).log2() as u8;
        (numerator, denominator_log2)
    } else {
        (4, 2) // Default to 4/4
    }
}

/// Determine if ghost notes should be included in MIDI export
fn should_include_ghost_notes(_config: &Config) -> bool {
    // In a full implementation, this would be configurable via config
    // For now, include ghost notes as they represent fill/silence protection
    true
}
