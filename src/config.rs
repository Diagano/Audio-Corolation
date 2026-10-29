//! Config file `audio-offset.ini`: settings that rarely change, plus the window's last-used options.
//!
//! Format: `key = value` lines; lines starting with `#` are comments; a missing key takes its default.
//! Location: next to the executable when a file exists there (portable use), else the per-user config folder
//! (`%APPDATA%\audio-offset` on Windows, `$XDG_CONFIG_HOME/audio-offset` or `~/.config/audio-offset` elsewhere).

use std::path::PathBuf;

use crate::analysis::Params;

pub const FILE_NAME: &str = "audio-offset.ini";

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Folder holding ffmpeg and ffprobe; empty searches PATH
    pub ffmpeg_dir: String,
    /// Listening clip length and its lead before the cursor (s)
    pub clip_seconds: f64,
    pub clip_lead: f64,
    pub params: Params,
}

impl Default for Config {
    fn default() -> Self {
        Self { ffmpeg_dir: String::new(), clip_seconds: 4.0, clip_lead: 0.5, params: Params::default() }
    }
}

/// Config file location; see the module docs.
pub fn path() -> PathBuf {
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from)) {
        let portable = dir.join(FILE_NAME);
        if portable.is_file() {
            return portable;
        }
    }
    let base = if cfg!(windows) {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
    };
    base.unwrap_or_default().join("audio-offset").join(FILE_NAME)
}

impl Config {
    /// Reads `path()`; a missing file gives the defaults. Returns the config and a note per ignored line.
    pub fn load() -> (Self, Vec<String>) {
        match std::fs::read_to_string(path()) {
            Ok(text) => Self::parse(&text),
            Err(_) => (Self::default(), Vec::new()),
        }
    }

    pub fn parse(text: &str) -> (Self, Vec<String>) {
        let mut c = Self::default();
        let mut notes = Vec::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                notes.push(format!("line {}: expected key = value", n + 1));
                continue;
            };
            if let Err(e) = c.set(key.trim(), value.trim()) {
                notes.push(format!("line {}: {e}", n + 1));
            }
        }
        (c, notes)
    }

    fn set(&mut self, key: &str, v: &str) -> Result<(), String> {
        fn num<T: std::str::FromStr>(v: &str) -> Result<T, String> {
            v.parse().map_err(|_| format!("invalid value '{v}'"))
        }
        fn flag(v: &str) -> Result<bool, String> {
            match v {
                "true" | "yes" | "1" => Ok(true),
                "false" | "no" | "0" => Ok(false),
                _ => Err(format!("invalid value '{v}' (true / false)")),
            }
        }
        let p = &mut self.params;
        match key {
            "ffmpeg_dir" => self.ffmpeg_dir = v.to_string(),
            "clip_seconds" => self.clip_seconds = num(v)?,
            "clip_lead" => self.clip_lead = num(v)?,
            "sample_rate" => p.sr = num(v)?,
            "frame" => p.frame = num(v)?,
            "hop" => p.hop = num(v)?,
            "speech_low" => p.speech_lo = num(v)?,
            "speech_high" => p.speech_hi = num(v)?,
            "drop_centre" => p.keep_centre = !flag(v)?,
            "exclude_speech_band" => p.full_band = !flag(v)?,
            "landmarks_per_min" => p.lm_per_min = num(v)?,
            "min_gap" => p.lm_gap_s = num(v)?,
            "min_rise_db" => p.lm_min_rise = num::<f64>(v)? / 10.0,
            "min_similarity" => p.lm_min_sim = num(v)?,
            "start_min" => p.start_min = num(v)?,
            "speed" => p.speed = crate::analysis::parse_speed(v)?,
            "search_range" => {
                let s: f64 = num(v)?;
                p.max_lag_ms = (s > 0.0).then_some(s * 1000.0);
            }
            _ => return Err(format!("unknown key '{key}'")),
        }
        Ok(())
    }

    pub fn to_text(&self) -> String {
        let p = &self.params;
        format!(
            "# audio-offset configuration. key = value; '#' starts a comment line; a removed key takes its default.

# ---- Rarely changed ----
# Folder holding ffmpeg and ffprobe; empty searches PATH.
ffmpeg_dir = {}
# Analysis sample rate (Hz); bands above sample_rate / 2 are not analysed.
sample_rate = {}
# Analysis frame and hop (samples); time resolution is hop / sample_rate (16 ms by default).
frame = {}
hop = {}
# Band (Hz) treated as dialogue, excluded while exclude_speech_band is true.
speech_low = {}
speech_high = {}
# Listening clip length and its lead before the cursor (s).
clip_seconds = {}
clip_lead = {}

# ---- Analysis options: last used, also set from the window ----
drop_centre = {}
exclude_speech_band = {}
landmarks_per_min = {}
# Minimum spacing between landmarks (s).
min_gap = {}
# Minimum onset rise (dB summed over bands).
min_rise_db = {}
# Minimum band-signature similarity (0..1) for a landmark match to vote.
min_similarity = {}
# Analysis start (minutes into each file).
start_min = {}
# Offset search range (s, either direction); 0 searches the whole file.
search_range = {}
# File 2 playback speed relative to File 1: auto (tries PAL / NTSC ratios) or a ratio such as 25/23.976.
speed = {}
",
            self.ffmpeg_dir,
            p.sr,
            p.frame,
            p.hop,
            p.speech_lo,
            p.speech_hi,
            self.clip_seconds,
            self.clip_lead,
            !p.keep_centre,
            !p.full_band,
            p.lm_per_min,
            p.lm_gap_s,
            p.lm_min_rise * 10.0,
            p.lm_min_sim,
            p.start_min,
            p.max_lag_ms.map_or(0.0, |ms| ms / 1000.0),
            p.speed.map_or("auto".to_string(), |s| {
                let name = crate::analysis::speed_name(s);
                if name.contains('/') || name == "1" { name } else { s.to_string() }
            }),
        )
    }

    pub fn save(&self) -> Result<(), String> {
        let path = path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let tmp = path.with_extension("ini.tmp");
        std::fs::write(&tmp, self.to_text()).map_err(|e| format!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_round_trip() {
        let mut c = Config::default();
        c.ffmpeg_dir = r"C:\Program Files\ffmpeg\bin".into();
        c.params.keep_centre = true;
        c.params.max_lag_ms = Some(30000.0);
        c.params.lm_min_rise = 1.5;
        c.params.speed = Some(25.0 * 1001.0 / 24000.0);
        let (back, notes) = Config::parse(&c.to_text());
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(back, c);
    }

    #[test]
    fn bad_lines_are_reported_and_skipped() {
        let (c, notes) = Config::parse("hop = x\nnonsense\ncolour = red\nframe = 1024\n");
        assert_eq!(notes.len(), 3);
        assert_eq!(c.params.frame, 1024);
        assert_eq!(c.params.hop, Params::default().hop);
    }
}
