//! ffprobe / ffmpeg as external tools: source probing and sample-accurate mono decoding on each file's timeline.

use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread::JoinHandle;

use serde_json::Value;

/// Channel order of common layouts (ffprobe names); FC carries the dialogue.
const LAYOUT_CHANNELS: &[(&str, &str)] = &[
    ("3.0", "FL FR FC"),
    ("4.0", "FL FR FC BC"),
    ("5.0", "FL FR FC BL BR"),
    ("5.0(side)", "FL FR FC SL SR"),
    ("5.1", "FL FR FC LFE BL BR"),
    ("5.1(side)", "FL FR FC LFE SL SR"),
    ("6.1", "FL FR FC LFE BC SL SR"),
    ("7.1", "FL FR FC LFE BL BR SL SR"),
    ("7.1(wide)", "FL FR FC LFE BL BR FLC FRC"),
    ("7.1(wide-side)", "FL FR FC LFE FLC FRC SL SR"),
];

/// Where ffmpeg and ffprobe are found: a directory, or PATH when empty.
#[derive(Clone, Debug, Default)]
pub struct Tools {
    pub dir: PathBuf,
}

impl Tools {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The tool's executable: in `dir`, else the bare name for a PATH search.
    pub fn exe(&self, tool: &str) -> PathBuf {
        if self.dir.as_os_str().is_empty() {
            PathBuf::from(tool)
        } else {
            self.dir.join(format!("{tool}{}", std::env::consts::EXE_SUFFIX))
        }
    }

    fn command(&self, tool: &str) -> Command {
        let mut cmd = Command::new(self.exe(tool));
        cmd.stdin(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console window per call from the GUI
        }
        cmd
    }

    /// Version of ffmpeg and ffprobe, one line each, or why they cannot be run.
    pub fn check(&self) -> Result<String, String> {
        let mut lines = Vec::new();
        for tool in ["ffmpeg", "ffprobe"] {
            let out = self.command(tool).arg("-version").output().map_err(|e| format!("{tool}: {e}"))?;
            let text = String::from_utf8_lossy(&out.stdout);
            let first = text.lines().next().unwrap_or(tool);
            lines.push(first.split_whitespace().take(3).collect::<Vec<_>>().join(" "));
        }
        Ok(lines.join("\n"))
    }
}

/// One audio stream of a file.
#[derive(Clone, Debug)]
pub struct AudioTrack {
    pub codec: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub layout: String,
    pub language: String,
    pub title: String,
    /// Flagged as the default track
    pub default: bool,
    pub start: f64,
}

impl AudioTrack {
    /// "fre · ac3 5.1(side) · 48 kHz · default · Title"
    pub fn label(&self) -> String {
        let layout = if self.layout.is_empty() { format!("{} ch", self.channels) } else { self.layout.clone() };
        let mut parts = vec![format!("{} {layout}", self.codec), format!("{} kHz", self.sample_rate as f64 / 1000.0)];
        if !self.language.is_empty() {
            parts.insert(0, self.language.clone());
        }
        if self.default {
            parts.push("default".into());
        }
        if !self.title.is_empty() {
            parts.push(self.title.clone());
        }
        parts.join(" · ")
    }
}

/// A file with one of its audio tracks selected, and its timeline.
#[derive(Clone, Debug)]
pub struct Source {
    pub path: PathBuf,
    pub tracks: Vec<AudioTrack>,
    /// Selected track, 0-based among the audio tracks (ffmpeg's `a:N`)
    pub track: usize,
    /// Of the selected track
    pub codec: String,
    pub sample_rate: u32,
    pub layout: String,
    /// None without a video stream.
    pub frame_rate: Option<f64>,
    /// Native time analysed as 0: the video start (offsets are relative to each file's picture), else the selected
    /// track's start, not below 0 (codec priming before 0 is not content).
    pub origin: f64,
    pub format_start: f64,
    /// Seconds from origin to the end of the file; 0 when unknown.
    pub duration: f64,
    video_start: Option<f64>,
    end: Option<f64>,
}

impl Source {
    /// Selects audio track k (0-based; clamped); without video, origin and duration follow the track.
    pub fn select_track(&mut self, k: usize) {
        self.track = k.min(self.tracks.len() - 1);
        let t = &self.tracks[self.track];
        (self.codec, self.sample_rate, self.layout) = (t.codec.clone(), t.sample_rate, t.layout.clone());
        self.origin = self.video_start.unwrap_or(t.start.max(0.0));
        self.duration = self.end.map_or(0.0, |e| (e - self.origin).max(0.0));
    }
}

fn parse_rate(rate: &str) -> Option<f64> {
    let (num, den) = rate.split_once('/')?;
    let (num, den) = (num.parse::<f64>().ok()?, den.parse::<f64>().ok()?);
    (num != 0.0 && den != 0.0).then(|| num / den)
}

/// Probes path and selects audio `track` (1-based), else the track flagged default, else the first.
pub fn probe(tools: &Tools, path: &Path, track: Option<usize>) -> Result<Source, String> {
    let out = tools
        .command("ffprobe")
        .args(["-v", "error", "-print_format", "json", "-show_streams", "-show_format"])
        .arg(path)
        .output()
        .map_err(|e| format!("Cannot run ffprobe: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(format!("FFprobe failed on {}: {}", path.display(), err.trim()));
    }
    let json: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("FFprobe output: {e}"))?;
    let streams = json["streams"].as_array().cloned().unwrap_or_default();
    let num = |v: &Value| v.as_str().and_then(|s| s.parse::<f64>().ok());
    let start = |s: &Value| num(&s["start_time"]).unwrap_or(0.0);
    let tracks: Vec<AudioTrack> = streams
        .iter()
        .filter(|s| s["codec_type"] == "audio")
        .map(|s| AudioTrack {
            codec: s["codec_name"].as_str().unwrap_or("?").to_string(),
            sample_rate: num(&s["sample_rate"]).unwrap_or(0.0) as u32,
            channels: s["channels"].as_u64().unwrap_or(0) as u32,
            layout: s["channel_layout"].as_str().unwrap_or("").to_string(),
            language: s["tags"]["language"].as_str().unwrap_or("").to_string(),
            title: s["tags"]["title"].as_str().unwrap_or("").to_string(),
            default: s["disposition"]["default"].as_i64().unwrap_or(0) != 0,
            start: start(s),
        })
        .collect();
    if tracks.is_empty() {
        return Err(format!("No audio stream in {}", path.display()));
    }
    let k = match track {
        None => tracks.iter().position(|t| t.default).unwrap_or(0),
        Some(t) if (1..=tracks.len()).contains(&t) => t - 1,
        Some(t) => {
            return Err(format!("{} has {} audio track(s); track {t} does not exist", path.display(), tracks.len()))
        }
    };
    // Cover art is reported as a video stream; it carries no frame rate
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video" && s["disposition"]["attached_pic"].as_i64().unwrap_or(0) == 0);
    let frame_rate =
        video.and_then(|v| ["avg_frame_rate", "r_frame_rate"].iter().find_map(|k| parse_rate(v[*k].as_str()?)));
    let format_start = num(&json["format"]["start_time"]).unwrap_or(0.0);
    let mut src = Source {
        path: path.to_path_buf(),
        tracks,
        track: 0,
        codec: String::new(),
        sample_rate: 0,
        layout: String::new(),
        frame_rate,
        origin: 0.0,
        format_start,
        duration: 0.0,
        video_start: video.map(start),
        end: num(&json["format"]["duration"]).map(|d| format_start + d),
    };
    src.select_track(k);
    Ok(src)
}

/// Same file and same audio track.
pub fn same_track(a: &Source, b: &Source) -> bool {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    a.track == b.track && canon(&a.path) == canon(&b.path)
}

/// pan filter to mono without the centre channel; None (plain downmix) when the layout has no known centre.
pub fn downmix_filter(layout: &str, keep_centre: bool) -> Option<String> {
    let names: Vec<&str> = LAYOUT_CHANNELS.iter().find(|(l, _)| *l == layout)?.1.split(' ').collect();
    if keep_centre || !names.contains(&"FC") {
        return None;
    }
    Some(format!("pan=mono|c0<{}", names.iter().filter(|n| **n != "FC").copied().collect::<Vec<_>>().join("+")))
}

/// "[track 2 of 3 (fre, ac3), ]48000 Hz, 5.1(side) (centre dropped), 23.976 fps"
pub fn describe(src: &Source, mix: Option<&str>) -> String {
    let fps = src.frame_rate.map_or("no video".to_string(), |r| format!("{r:.3} fps"));
    let layout = if src.layout.is_empty() { "unknown layout" } else { &src.layout };
    let centre = if mix.is_some() { "centre dropped" } else { "plain downmix" };
    let track = if src.tracks.len() > 1 {
        let t = &src.tracks[src.track];
        let tag: Vec<&str> = [t.language.as_str(), t.codec.as_str()].into_iter().filter(|s| !s.is_empty()).collect();
        format!("track {} of {} ({}), ", src.track + 1, src.tracks.len(), tag.join(", "))
    } else {
        String::new()
    };
    format!("{track}{} Hz, {layout} ({centre}), {fps}", src.sample_rate)
}

/// Streams the selected audio track of a source, from `start` s after its origin, as mono f32 at `rate`.
/// Timestamps stay native (-copyts) and a second aresample pins output sample 0 to origin + start: codec priming
/// before it is trimmed and a seek landing on a later packet is padded, so the output is sample-accurate on the
/// timeline. The rate is converted first because first_pts counts samples at aresample's input rate.
pub struct Decoder {
    child: Child,
    stdout: ChildStdout,
    stderr: Option<JoinHandle<String>>,
    path: PathBuf,
    carry: Vec<u8>,
    buf: Vec<u8>,
}

impl Decoder {
    pub fn start(
        tools: &Tools,
        src: &Source,
        mix: Option<&str>,
        rate: u32,
        start: f64,
        duration: Option<f64>,
    ) -> Result<Self, String> {
        let t0 = src.origin + start;
        let mut cmd = tools.command("ffmpeg");
        cmd.args(["-nostdin", "-v", "error", "-copyts", "-ss", &format!("{:.6}", (t0 - src.format_start).max(0.0))]);
        if let Some(d) = duration {
            cmd.args(["-t", &d.to_string()]);
        }
        let resample =
            format!("aresample={rate},aresample=async=1:first_pts={}", (t0 * rate as f64).round_ties_even() as i64);
        let filters = mix.map_or(resample.clone(), |m| format!("{m},{resample}"));
        cmd.arg("-i").arg(&src.path);
        cmd.args(["-map", &format!("0:a:{}", src.track), "-af", &filters, "-ac", "1", "-ar", &rate.to_string()]);
        cmd.args(["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]);
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Cannot run ffmpeg: {e}"))?;
        let stdout = child.stdout.take().expect("piped stdout");
        let mut err_pipe = child.stderr.take().expect("piped stderr");
        // Drained on its own thread: a full stderr pipe would stall ffmpeg while stdout is being read
        let stderr = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = err_pipe.read_to_string(&mut text);
            text
        });
        Ok(Self { child, stdout, stderr: Some(stderr), path: src.path.clone(), carry: Vec::new(), buf: vec![0; 1 << 16] })
    }

    /// Appends the next decoded samples to `out`; false at the end of the stream.
    pub fn read(&mut self, out: &mut Vec<f32>) -> Result<bool, String> {
        let n = loop {
            match self.stdout.read(&mut self.buf) {
                Ok(n) => break n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(format!("Reading ffmpeg output: {e}")),
            }
        };
        if n == 0 {
            let status = self.child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
            let msg = self.stderr.take().map(|h| h.join().unwrap_or_default()).unwrap_or_default();
            if !status.success() {
                return Err(format!("FFmpeg decode failed on {}: {}", self.path.display(), msg.trim()));
            }
            return Ok(false);
        }
        self.carry.extend_from_slice(&self.buf[..n]);
        let whole = self.carry.len() / 4 * 4;
        out.extend(self.carry[..whole].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)));
        self.carry.drain(..whole);
        Ok(true)
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// ffmpeg command line that re-encodes the source's selected track through `filter` into "<name>.retimed.mka" next
/// to it: FLAC when `lossless`, else the source codec at a high bitrate (codecs ffmpeg does not encode well, such as
/// DTS, TrueHD and PCM: FLAC). Chapters are left out, as their times would not follow the new speed.
pub fn retime_command(tools: &Tools, src: &Source, filter: &str, lossless: bool) -> String {
    let channels = src.tracks[src.track].channels.max(1);
    let codec = match (lossless, src.codec.as_str()) {
        (false, "ac3") => "ac3 -b:a 640k".to_string(),
        (false, "eac3") => "eac3 -b:a 1536k".to_string(),
        (false, "aac") => format!("aac -b:a {}k", 96 * channels),
        (false, "opus") => format!("libopus -b:a {}k", 64 * channels),
        (false, "mp3") => "libmp3lame -b:a 320k".to_string(),
        _ => "flac".to_string(),
    };
    let stem = src.path.file_stem().map_or("audio".into(), |s| s.to_string_lossy().into_owned());
    let out = src.path.with_file_name(format!("{stem}.retimed.mka"));
    format!(
        "\"{}\" -i \"{}\" -map 0:a:{} -map_chapters -1 -af {filter} -c:a {codec} \"{}\"",
        tools.exe("ffmpeg").display(),
        src.path.display(),
        src.track,
        out.display()
    )
}

/// Whole decode of a span into memory; see [`Decoder`].
pub fn decode(
    tools: &Tools,
    src: &Source,
    mix: Option<&str>,
    rate: u32,
    start: f64,
    duration: Option<f64>,
) -> Result<Vec<f32>, String> {
    let mut dec = Decoder::start(tools, src, mix, rate, start, duration)?;
    let mut out = Vec::new();
    while dec.read(&mut out)? {}
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(codec: &str, channels: u32, track: usize) -> Source {
        let t = AudioTrack {
            codec: codec.into(),
            sample_rate: 48000,
            channels,
            layout: String::new(),
            language: String::new(),
            title: String::new(),
            default: false,
            start: 0.0,
        };
        let mut src = Source {
            path: PathBuf::from("/media/B_pal.mkv"),
            tracks: vec![t.clone(), t],
            track: 0,
            codec: String::new(),
            sample_rate: 0,
            layout: String::new(),
            frame_rate: None,
            origin: 0.0,
            format_start: 0.0,
            duration: 0.0,
            video_start: None,
            end: None,
        };
        src.select_track(track);
        src
    }

    #[test]
    fn retime_command_keeps_codec_or_uses_flac() {
        let tools = Tools::default();
        let ac3 = retime_command(&tools, &source("ac3", 6, 1), "atempo=0.959040959", false);
        assert_eq!(
            ac3,
            r#""ffmpeg" -i "/media/B_pal.mkv" -map 0:a:1 -map_chapters -1 -af atempo=0.959040959 -c:a ac3 -b:a 640k "/media/B_pal.retimed.mka""#
        );
        assert!(retime_command(&tools, &source("ac3", 6, 0), "x", true).contains("-c:a flac "));
        assert!(retime_command(&tools, &source("dts", 6, 0), "x", false).contains("-c:a flac "));
        assert!(retime_command(&tools, &source("aac", 2, 0), "x", false).contains("-c:a aac -b:a 192k "));
    }
}
