//! Offset between two versions of the same programme (e.g. language dubs).
//!
//! Dubs share the music & effects track and differ in dialogue, so the analysis drops the centre channel of
//! surround sources and the speech band, detects sharp onsets (landmarks: shots, slams, engine starts) in the
//! remaining bands, votes on the offset from landmark matches and refines each match by local cross-correlation.
//! Reference implementation and experimentation bench: `audio_offset.py`.

pub mod analysis;
pub mod config;
pub mod ffmpeg;
