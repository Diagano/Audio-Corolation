//! Landmark analysis: non-speech band power -> sharp onsets (landmarks) -> offset votes -> per-landmark refinement.
//! Offsets are file1 - file2: positive means an event occurs later in file1.

use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::ffmpeg::{self, Decoder, Source, Tools};

pub const LM_LOW_HZ: f64 = 30.0; // lowest landmark band edge; below is rumble / DC
const ONSET_LAGS: [usize; 2] = [2, 10]; // frames over which a rise is measured: sharp hits, and slower attacks (engines)
const REFINE_CTX_S: f64 = 1.0; // half-width of the onset-curve segment cross-correlated per matched landmark
const REFINE_LAGS: i64 = 3; // lag search (frames) around the voted offset during refinement
const WEAK_RATIO: f64 = 2.0; // vote peak / runner-up below this is reported as a weak match
const WEAK_INLIERS: usize = 10; // fewer inlier landmarks than this is reported as a weak match
const MAX_PEAKS: usize = 10; // vote peaks examined for parts with their own offset (cuts, inserted scenes)
const MIN_PEAK_VOTES: f64 = 4.0; // smoothed votes a peak beyond the strongest needs to be examined
const PART_MIN_MATCHES: usize = 4; // matches a part needs; shorter runs at another offset are chance matches
// Share of its offset's matches a part needs: offsets that line up periodic content (steady musical tempo) match
// a little everywhere, a real part concentrates its matches
const PART_MIN_SHARE: f64 = 0.25;
// A part is dropped when a neighbouring part's offset has at least this share of its match count within its span:
// the stretch fits both (e.g. music only) and continuity wins; after a real cut the old offset hardly matches
const PART_RIVAL_SHARE: f64 = 0.5;
/// Playback speeds of File 2 relative to File 1 tried by default: PAL speed-up (25 fps from 24 / 23.976 film) and
/// NTSC slow-down (23.976 from 24), each way. Frame rates alone do not tell: telecine keeps the speed.
pub const SPEEDS: [(f64, &str); 7] = [
    (1.0, "1"),
    (25.0 / 24.0, "25/24"),
    (24.0 / 25.0, "24/25"),
    (25.0 * 1001.0 / 24000.0, "25/23.976"),
    (24000.0 / 1001.0 / 25.0, "23.976/25"),
    (1001.0 / 1000.0, "24/23.976"),
    (1000.0 / 1001.0, "23.976/24"),
];
const SPEED_TOL: f64 = 5e-4; // relative difference below which a frame-rate ratio and a speed are taken as equal
pub const CANCELLED: &str = "Cancelled";

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    /// Analysis sample rate (Hz)
    pub sr: u32,
    /// Analysis frame and hop (samples)
    pub frame: usize,
    pub hop: usize,
    /// Band (Hz) excluded as dialogue, unless `full_band`
    pub speech_lo: f64,
    pub speech_hi: f64,
    pub full_band: bool,
    /// Keep the centre (dialogue) channel of surround sources
    pub keep_centre: bool,
    /// Analysis start, minutes after each file's origin
    pub start_min: f64,
    /// Search limit for the offset; None searches all
    pub max_lag_ms: Option<f64>,
    /// Max landmarks per minute (strongest kept)
    pub lm_per_min: usize,
    /// Min spacing between landmarks (s)
    pub lm_gap_s: f64,
    /// Min onset strength: sum over bands of log10 power rise (1.0 = 10 dB in one band)
    pub lm_min_rise: f64,
    /// Min cosine similarity of band signatures for a landmark match to vote
    pub lm_min_sim: f64,
    /// File 2 playback speed relative to File 1; None tries SPEEDS
    pub speed: Option<f64>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            sr: 16000,
            frame: 512,
            hop: 256,
            speech_lo: 300.0,
            speech_hi: 3500.0,
            full_band: false,
            keep_centre: false,
            start_min: 0.0,
            max_lag_ms: None,
            lm_per_min: 20,
            lm_gap_s: 0.25,
            lm_min_rise: 1.0,
            lm_min_sim: 0.8,
            speed: None,
        }
    }
}

impl Params {
    pub fn validate(&self) -> Result<(), String> {
        if self.frame < 16 || self.hop == 0 || self.hop > self.frame {
            return Err("Frame must be at least 16 samples and hop between 1 and the frame size".into());
        }
        let nyq = self.sr as f64 / 2.0;
        if !self.full_band && !(LM_LOW_HZ < self.speech_lo && self.speech_lo < self.speech_hi && self.speech_hi < nyq) {
            return Err(format!("Speech band must satisfy {LM_LOW_HZ:.0} < low < high < {nyq:.0} Hz (sr/2)"));
        }
        if self.speed.is_some_and(|s| !(0.5..=2.0).contains(&s)) {
            return Err("Speed must lie between 0.5 and 2".into());
        }
        if self.band_ranges().is_empty() {
            return Err("No analysis bands: frame too short for the band layout".into());
        }
        Ok(())
    }

    /// Seconds per analysis frame (hop / sr).
    pub fn frame_s(&self) -> f64 {
        self.hop as f64 / self.sr as f64
    }

    fn frames(&self, seconds: f64) -> f64 {
        seconds * self.sr as f64 / self.hop as f64
    }

    /// FFT bin ranges of the landmark bands: three log-spaced bands each side of the speech band.
    fn band_ranges(&self) -> Vec<Range<usize>> {
        let nyq = self.sr as f64 / 2.0;
        let edges = if self.full_band {
            vec![geomspace(LM_LOW_HZ, nyq, 7)]
        } else {
            vec![geomspace(LM_LOW_HZ, self.speech_lo, 4), geomspace(self.speech_hi, nyq, 4)]
        };
        let bin_hz = self.sr as f64 / self.frame as f64;
        let bins = 0..self.frame / 2 + 1;
        edges
            .iter()
            .flat_map(|e| e.windows(2).map(|w| (w[0], w[1])).collect::<Vec<_>>())
            .filter_map(|(a, b)| {
                let inside: Vec<usize> = bins.clone().filter(|&k| k as f64 * bin_hz >= a && (k as f64 * bin_hz) < b).collect();
                Some(*inside.first()?..*inside.last()? + 1)
            })
            .collect()
    }
}

fn geomspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    let (la, lb) = (a.log10(), b.log10());
    let mut v: Vec<f64> = (0..n).map(|i| 10f64.powf(la + (lb - la) * i as f64 / (n - 1) as f64)).collect();
    v[0] = a;
    v[n - 1] = b;
    v
}

/// Streaming band power: Hann-windowed one-sided power per frame, summed per band, scaled so a band's value is
/// its share of the frame's mean square.
struct BandAnalyzer {
    frame: usize,
    hop: usize,
    window: Vec<f64>,
    scale: f64,
    fft: Arc<dyn Fft<f64>>,
    buf: Vec<Complex64>,
    scratch: Vec<Complex64>,
    bands: Vec<Range<usize>>,
    pending: Vec<f32>,
}

impl BandAnalyzer {
    fn new(p: &Params) -> Self {
        let n = p.frame;
        let window: Vec<f64> =
            (0..n).map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()).collect();
        let scale = 1.0 / (n as f64 * window.iter().map(|w| w * w).sum::<f64>());
        let fft = FftPlanner::new().plan_fft_forward(n);
        let scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];
        Self {
            frame: n,
            hop: p.hop,
            window,
            scale,
            fft,
            buf: vec![Complex64::default(); n],
            scratch,
            bands: p.band_ranges(),
            pending: Vec::new(),
        }
    }

    /// Consumes samples; appends one row (bands) of power per completed frame to `out`.
    fn push(&mut self, samples: &[f32], out: &mut Vec<f64>) {
        self.pending.extend_from_slice(samples);
        let mut pos = 0;
        while pos + self.frame <= self.pending.len() {
            for (i, c) in self.buf.iter_mut().enumerate() {
                *c = Complex64::new(self.pending[pos + i] as f64 * self.window[i], 0.0);
            }
            self.fft.process_with_scratch(&mut self.buf, &mut self.scratch);
            for band in &self.bands {
                let mut sum = 0.0;
                for k in band.clone() {
                    // bins other than DC and (even frame) Nyquist appear twice in the full spectrum
                    let twice = k >= 1 && k < self.frame.div_ceil(2);
                    sum += self.buf[k].norm_sqr() * if twice { 2.0 } else { 1.0 };
                }
                out.push(sum * self.scale);
            }
            pos += self.hop;
        }
        self.pending.drain(..pos);
    }
}

#[derive(Clone, Debug)]
pub struct Landmark {
    /// Analysis frame (sub-frame interpolated) after the analysis start
    pub frame: f64,
    /// Per-band rise, unit norm
    pub sig: Vec<f64>,
    pub strength: f64,
}

/// One analysed file.
#[derive(Clone, Debug)]
pub struct Track {
    /// Onset strength per frame (sum over bands of the log power rise)
    pub onset: Vec<f64>,
    /// Non-speech level per frame (dB re full scale), for display
    pub level_db: Vec<f32>,
    /// Sorted by frame
    pub landmarks: Vec<Landmark>,
}

#[derive(Clone, Debug)]
pub struct Match {
    /// File1 time (s after its origin)
    pub t1_s: f64,
    pub offset_ms: f64,
    /// Landmark indices in each track
    pub i1: usize,
    pub i2: usize,
    /// Vote peak whose offset the match was refined around
    pub peak: usize,
    /// Part the match belongs to
    pub part: usize,
    /// Within one frame of its part's median offset
    pub inlier: bool,
}

/// A stretch of File 1 time with one offset; versions with cuts or inserted scenes have several.
#[derive(Clone, Debug)]
pub struct Part {
    /// Median offset of its matches
    pub offset_ms: f64,
    /// First and last matched landmark (File 1 time, s)
    pub start_s: f64,
    pub end_s: f64,
    /// Index range into `Analysis::matches`
    pub matches: Range<usize>,
    pub inliers: usize,
}

/// Figures for the main part.
#[derive(Clone, Debug)]
pub struct Summary {
    pub median_ms: f64,
    pub p10_ms: f64,
    pub p90_ms: f64,
    pub inliers: usize,
    /// Slope of offset over time across inliers and its standard error (ms per hour); None below 4 inliers
    pub drift_ms_per_h: Option<(f64, f64)>,
    /// Speed refined by the residual drift, and its standard error; None below 4 inliers
    pub measured_speed: Option<(f64, f64)>,
    /// Its vote peak: offset (frames) and smoothed votes
    pub coarse: i64,
    pub peak: f64,
    /// Vote peak / strongest peak not explained by a part
    pub ratio: f64,
    pub weak: bool,
}

#[derive(Clone, Debug)]
pub struct Analysis {
    pub params: Params,
    /// As decoded; File 2 runs `speed` x File 1, so File 2 frame f lies at File 1 frame `speed` x f + offset
    pub tracks: [Track; 2],
    pub speed: f64,
    /// Offset votes (frames)
    pub votes: Vec<f64>,
    /// Vote peaks examined: (offset frames, smoothed votes), strongest first
    pub peaks: Vec<(i64, f64)>,
    /// Matches of all parts, part by part, in File 1 time order
    pub matches: Vec<Match>,
    /// In File 1 time order
    pub parts: Vec<Part>,
    /// The part with the most matches, described by `summary`
    pub main: usize,
    pub summary: Summary,
    /// Remarks on the inputs, e.g. frame rates that disagree with the measured speed
    pub notes: Vec<String>,
}

/// "25/23.976" for a speed of SPEEDS, else six decimals.
pub fn speed_name(speed: f64) -> String {
    SPEEDS.iter().find(|(s, _)| *s == speed).map_or(format!("{speed:.6}"), |(_, n)| n.to_string())
}

/// ffmpeg filter bringing audio that runs `speed` x to the reference speed: atempo keeps the pitch; asetrate shifts
/// it with the speed (undoing a PAL pitch rise) but takes whole sample rates, so a residual atempo makes it exact.
pub fn retime_filter(speed: f64, sample_rate: u32, undo_pitch: bool) -> String {
    if !undo_pitch {
        return format!("atempo={:.9}", 1.0 / speed);
    }
    let r = (sample_rate as f64 / speed + 0.5).floor() as u32;
    format!("asetrate={r},aresample={sample_rate},atempo={:.9}", sample_rate as f64 / (r as f64 * speed))
}

/// Both retiming filters, as the report line.
pub fn retime_hint(speed: f64, sample_rate: u32) -> String {
    format!(
        "{} (keeps pitch) or {} (also undoes a pitch shift)",
        retime_filter(speed, sample_rate, false),
        retime_filter(speed, sample_rate, true)
    )
}

/// "auto" -> None; a name of SPEEDS (exact value: 23.976 stands for 24000/1001), "a/b" or a number.
pub fn parse_speed(text: &str) -> Result<Option<f64>, String> {
    let text = text.trim();
    if text == "auto" {
        return Ok(None);
    }
    if let Some((s, _)) = SPEEDS.iter().find(|(_, n)| *n == text) {
        return Ok(Some(*s));
    }
    let (num, den) = text.split_once('/').unwrap_or((text, "1"));
    match (num.trim().parse::<f64>(), den.trim().parse::<f64>()) {
        (Ok(n), Ok(d)) if d != 0.0 => Ok(Some(n / d)),
        _ => Err(format!("'{text}' is not 'auto' or a ratio")),
    }
}

/// "h:mm:ss.s" or "m:ss.s"
pub fn fmt_time(s: f64) -> String {
    let sign = if s < 0.0 { "-" } else { "" };
    let s = (s.abs() * 10.0 + 0.5).floor() / 10.0;
    let (h, m, sec) = ((s / 3600.0).floor(), ((s % 3600.0) / 60.0).floor(), s % 60.0);
    if h > 0.0 {
        format!("{sign}{h}:{m:02}:{sec:04.1}")
    } else {
        format!("{sign}{m}:{sec:04.1}")
    }
}

fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate().fold(0, |best, (i, &c)| if c > v[best] { i } else { best })
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Linear-interpolated percentile of sorted data.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Sub-sample parabolic interpolation around a peak at idx.
fn refine_peak_parabola(c: &[f64], idx: usize) -> f64 {
    if idx == 0 || idx + 1 >= c.len() {
        return 0.0;
    }
    let (m1, c0, p1) = (c[idx - 1], c[idx], c[idx + 1]);
    let denom = m1 - 2.0 * c0 + p1;
    if denom != 0.0 {
        0.5 * (m1 - p1) / denom
    } else {
        0.0
    }
}

/// Per-band rise of log power, largest over ONSET_LAGS (frames x bands); levels relative to each band's median.
fn onset_strength(power: &[f64], nb: usize) -> Vec<f64> {
    let n = power.len() / nb;
    let mut level = vec![0.0; power.len()];
    for b in 0..nb {
        let mut col: Vec<f64> = (0..n).map(|t| power[t * nb + b]).collect();
        let med = median(&mut col) + 1e-20;
        for t in 0..n {
            level[t * nb + b] = (power[t * nb + b] / med + 1.0).log10();
        }
    }
    let mut rise = vec![0.0; power.len()];
    for lag in ONSET_LAGS {
        for i in lag * nb..power.len() {
            rise[i] = f64::max(rise[i], level[i] - level[i - lag * nb]);
        }
    }
    rise
}

/// Sharp onsets, strongest first, at least lm_gap apart, at most lm_per_min per minute.
fn detect_landmarks(rise: &[f64], nb: usize, p: &Params) -> (Vec<Landmark>, Vec<f64>) {
    let n = rise.len() / nb;
    let onset: Vec<f64> = rise.chunks_exact(nb).map(|r| r.iter().sum()).collect();
    let gap = (p.frames(p.lm_gap_s).round_ties_even() as usize).max(1);
    let frames_per_min = p.frames(60.0) as usize;
    let mut cand: Vec<usize> = (0..n).filter(|&t| onset[t] >= p.lm_min_rise).collect();
    cand.sort_by(|&a, &b| onset[b].total_cmp(&onset[a])); // stable: equal strengths keep time order
    let mut blocked = vec![false; n];
    let mut per_min = vec![0usize; n / frames_per_min.max(1) + 1];
    let mut picked = Vec::new();
    for idx in cand {
        let minute = idx / frames_per_min.max(1);
        if blocked[idx] || per_min[minute] >= p.lm_per_min {
            continue;
        }
        blocked[idx.saturating_sub(gap)..(idx + gap + 1).min(n)].fill(true);
        per_min[minute] += 1;
        picked.push(idx);
    }
    picked.sort_unstable();
    let landmarks = picked
        .into_iter()
        .map(|i| {
            let row = &rise[i * nb..(i + 1) * nb];
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt() + 1e-12;
            Landmark { frame: i as f64 + refine_peak_parabola(&onset, i), sig: row.iter().map(|x| x / norm).collect(), strength: onset[i] }
        })
        .collect();
    (landmarks, onset)
}

/// Index pairs of landmarks whose signatures' cosine similarity is at least lm_min_sim.
fn similar_pairs(l1: &[Landmark], l2: &[Landmark], min_sim: f64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, a) in l1.iter().enumerate() {
        for (j, b) in l2.iter().enumerate() {
            if a.sig.iter().zip(&b.sig).map(|(x, y)| x * y).sum::<f64>() >= min_sim {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// (speed, votes): the speed of SPEEDS (or the given one) whose offset votes (frames, file1 - speed x file2) form
/// the highest histogram peak; at a wrong speed the votes of one stretch smear over many offsets.
fn choose_speed(l1: &[Landmark], l2: &[Landmark], pairs: &[(usize, usize)], p: &Params) -> (f64, Vec<f64>) {
    let candidates: Vec<f64> = p.speed.map_or(SPEEDS.iter().map(|s| s.0).collect(), |s| vec![s]);
    let mut best: Option<(f64, f64, Vec<f64>)> = None;
    for speed in candidates {
        let votes: Vec<f64> = pairs.iter().map(|&(i, j)| l1[i].frame - speed * l2[j].frame).collect();
        let peak = vote_histogram(&votes, p).map_or(0.0, |(_, h)| h.into_iter().fold(0.0, f64::max));
        if best.as_ref().is_none_or(|b| peak > b.0) {
            best = Some((peak, speed, votes));
        }
    }
    let (_, speed, votes) = best.expect("at least one candidate speed");
    (speed, votes)
}

/// File 2 on File 1's time scale: landmark frames x speed, onset curve resampled.
fn scale_track(t: &Track, speed: f64) -> Track {
    let o = &t.onset;
    let n = ((o.len() - 1) as f64 * speed).floor() as usize + 1;
    let onset = (0..n)
        .map(|k| {
            let x = k as f64 / speed;
            let i = x.floor() as usize;
            if i + 1 >= o.len() { o[o.len() - 1] } else { (o[i + 1] - o[i]) * (x - i as f64) + o[i] }
        })
        .collect();
    let landmarks = t.landmarks.iter().map(|l| Landmark { frame: l.frame * speed, ..l.clone() }).collect();
    Track { onset, level_db: t.level_db.clone(), landmarks }
}

/// Remark when the frame-rate ratio and the speed disagree; None otherwise or without video.
fn rate_note(a: &Source, b: &Source, speed: f64, forced: bool) -> Option<String> {
    let (fr1, fr2) = (a.frame_rate?, b.frame_rate?);
    let ratio = fr2 / fr1;
    if (speed - ratio).abs() / ratio <= SPEED_TOL {
        return None;
    }
    Some(if forced {
        format!("speed set to {}, not measured; the frame-rate ratio is {ratio:.6}", speed_name(speed))
    } else if speed == 1.0 {
        format!("frame rates differ ({fr1:.3} vs {fr2:.3} fps) but the audio runs at the same speed (telecine or standards conversion)")
    } else {
        format!("audio speed {speed:.6} does not match the frame-rate ratio {ratio:.6} ({fr2:.3} / {fr1:.3} fps)")
    })
}

fn gap_frames(p: &Params) -> usize {
    (p.frames(p.lm_gap_s).round_ties_even() as usize).max(1)
}

/// Vote histogram within max_lag, summed over 3 bins to absorb +-1 frame jitter: (offset of bin 0, counts).
fn vote_histogram(votes: &[f64], p: &Params) -> Result<(i64, Vec<f64>), String> {
    let limit = p.max_lag_ms.map(|ms| p.frames(ms / 1000.0));
    let v: Vec<i64> = votes
        .iter()
        .filter(|x| limit.is_none_or(|l| x.abs() <= l))
        .map(|x| x.round_ties_even() as i64)
        .collect();
    let (Some(&lo), Some(&hi)) = (v.iter().min(), v.iter().max()) else {
        return Err("No landmark pairs matched".into());
    };
    let mut hist = vec![0.0; (hi - lo + 1) as usize];
    for x in &v {
        hist[(x - lo) as usize] += 1.0;
    }
    let smooth = (0..hist.len()).map(|i| hist[i.saturating_sub(1)..(i + 2).min(hist.len())].iter().sum()).collect();
    Ok((lo, smooth))
}

fn clear_around(hist: &mut [f64], k: i64, gap: usize) {
    let lo = (k - gap as i64).max(0) as usize;
    let hi = ((k + gap as i64 + 1).max(0) as usize).min(hist.len());
    if lo < hi {
        hist[lo..hi].fill(0.0);
    }
}

/// Histogram peaks, strongest first, at least lm_gap apart: (offset frames, votes).
fn vote_peaks(lo: i64, hist: &[f64], p: &Params) -> Vec<(i64, f64)> {
    let mut rest = hist.to_vec();
    let mut peaks = Vec::new();
    while peaks.len() < MAX_PEAKS {
        let k = argmax(&rest);
        if !peaks.is_empty() && rest[k] < MIN_PEAK_VOTES {
            break;
        }
        peaks.push((k as i64 + lo, rest[k]));
        clear_around(&mut rest, k as i64, gap_frames(p));
    }
    peaks
}

/// Strongest vote count away from the given offsets (frames).
fn runner_up(lo: i64, hist: &[f64], offsets: &[i64], p: &Params) -> f64 {
    let mut rest = hist.to_vec();
    for off in offsets {
        clear_around(&mut rest, off - lo, gap_frames(p));
    }
    rest.into_iter().fold(0.0, f64::max)
}

/// Landmarks of file1 with a file2 landmark at offset `coarse` (+-2 frames), each refined by cross-correlating
/// the onset curves around it.
fn refine_matches(t1: &Track, t2: &Track, coarse: i64, peak: usize, p: &Params) -> Vec<Match> {
    let (o1, o2) = (&t1.onset, &t2.onset);
    let frames2: Vec<f64> = t2.landmarks.iter().map(|l| l.frame).collect();
    let w = p.frames(REFINE_CTX_S).round_ties_even() as i64;
    let (lag_lo, lag_hi) = (coarse - REFINE_LAGS, coarse + REFINE_LAGS);
    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
    let mut out = Vec::new();
    for (i1, a) in t1.landmarks.iter().enumerate() {
        let target = a.frame - coarse as f64;
        let j = frames2.partition_point(|&t| t < target);
        let near = (j.saturating_sub(1)..(j + 1).min(frames2.len()))
            .filter(|&k| (target - frames2[k]).abs() <= 2.0)
            .min_by(|&x, &y| (target - frames2[x]).abs().total_cmp(&(target - frames2[y]).abs()));
        let Some(i2) = near else { continue };
        let c = a.frame.round_ties_even() as i64;
        if c - w < 0 || c + w > o1.len() as i64 || c - lag_hi - w < 0 || c - lag_lo + w > o2.len() as i64 {
            continue;
        }
        let seg1 = &o1[(c - w) as usize..(c + w) as usize];
        let m1 = mean(seg1);
        let corr: Vec<f64> = (lag_lo..=lag_hi)
            .map(|k| {
                let seg2 = &o2[(c - k - w) as usize..(c - k + w) as usize];
                let m2 = mean(seg2);
                seg1.iter().zip(seg2).map(|(x, y)| (x - m1) * (y - m2)).sum()
            })
            .collect();
        let idx = corr.iter().enumerate().fold(0, |best, (i, &c)| if c > corr[best] { i } else { best });
        if idx == 0 || idx == corr.len() - 1 {
            continue; // peak at the search edge: no reliable local match
        }
        let lag = (lag_lo + idx as i64) as f64 + refine_peak_parabola(&corr, idx);
        out.push(Match {
            t1_s: p.start_min * 60.0 + a.frame * p.frame_s(),
            offset_ms: lag * p.frame_s() * 1000.0,
            i1,
            i2,
            peak,
            part: 0,
            inlier: false,
        });
    }
    out
}

/// Splits matches into parts: runs in File 1 time sharing a vote peak. Dropped one at a time, so a part interrupted
/// by stray matches joins up again: runs shorter than PART_MIN_MATCHES (shortest and least supported first), then
/// runs holding less than PART_MIN_SHARE of their peak's matches (smallest share first), then runs a neighbour's
/// offset explains about as well (PART_RIVAL_SHARE; shortest first).
fn find_parts(mut m: Vec<Match>, n_peaks: usize) -> Vec<Vec<Match>> {
    m.sort_by(|a, b| a.t1_s.total_cmp(&b.t1_s).then(a.peak.cmp(&b.peak)));
    let mut support = vec![0usize; n_peaks];
    for x in &m {
        support[x.peak] += 1;
    }
    let mut keep = vec![true; m.len()];
    loop {
        let idx: Vec<usize> = (0..m.len()).filter(|&i| keep[i]).collect();
        if idx.is_empty() {
            return Vec::new();
        }
        let mut runs = Vec::new();
        let mut start = 0;
        for i in 1..=idx.len() {
            if i == idx.len() || m[idx[i]].peak != m[idx[start]].peak {
                runs.push((start, i));
                start = i;
            }
        }
        let sup = |s: usize| support[m[idx[s]].peak];
        let share = |&(s, e): &(usize, usize)| (e - s) as f64 / sup(s) as f64;
        let short = runs.iter().filter(|(s, e)| e - s < PART_MIN_MATCHES).min_by_key(|(s, e)| (e - s, sup(*s), *s));
        let scattered = runs
            .iter()
            .filter(|(s, e)| ((e - s) as f64) < PART_MIN_SHARE * sup(*s) as f64)
            .min_by(|a, b| share(a).total_cmp(&share(b)).then(a.0.cmp(&b.0)));
        let contested = || {
            runs.iter()
                .enumerate()
                .filter(|&(r, &(s, e))| {
                    let (t0, t1) = (m[idx[s]].t1_s, m[idx[e - 1]].t1_s);
                    let own = m[idx[s]].peak;
                    let rivals = [r.checked_sub(1), Some(r + 1)].into_iter().flatten().filter_map(|q| runs.get(q));
                    let best = rivals
                        .map(|&(rs, _)| m[idx[rs]].peak)
                        .filter(|&lab| lab != own)
                        .map(|lab| m.iter().filter(|x| x.peak == lab && x.t1_s >= t0 && x.t1_s <= t1).count())
                        .max()
                        .unwrap_or(0);
                    best as f64 >= PART_RIVAL_SHARE * (e - s) as f64
                })
                .map(|(_, run)| run)
                .min_by_key(|(s, e)| (e - s, *s))
        };
        let Some(&(s, e)) = short.or(scattered).or_else(contested) else {
            return runs.iter().map(|&(s, e)| idx[s..e].iter().map(|&i| m[i].clone()).collect()).collect();
        };
        for &i in &idx[s..e] {
            keep[i] = false;
        }
    }
}

/// Drift fit over (time s, offset ms): slope and standard error in ms per s.
fn drift(points: &[(f64, f64)]) -> Option<(f64, f64)> {
    (points.len() >= 4).then(|| {
        let n = points.len() as f64;
        let (mx, my) = (points.iter().map(|q| q.0).sum::<f64>() / n, points.iter().map(|q| q.1).sum::<f64>() / n);
        let sxx: f64 = points.iter().map(|q| (q.0 - mx).powi(2)).sum();
        let slope = points.iter().map(|q| (q.0 - mx) * (q.1 - my)).sum::<f64>() / sxx;
        let ssr: f64 = points.iter().map(|q| (q.1 - my - slope * (q.0 - mx)).powi(2)).sum();
        (slope, (ssr / (n - 2.0) / sxx).sqrt())
    })
}

/// Parts from match groups (inlier flags set per part), the main part's index and its summary.
fn build_parts(
    groups: Vec<Vec<Match>>,
    peaks: &[(i64, f64)],
    hist: (i64, &[f64]),
    speed: f64,
    p: &Params,
) -> (Vec<Match>, Vec<Part>, usize, Summary) {
    let step_ms = p.frame_s() * 1000.0;
    let (mut matches, mut parts) = (Vec::new(), Vec::new());
    for (k, mut group) in groups.into_iter().enumerate() {
        let mut offs: Vec<f64> = group.iter().map(|m| m.offset_ms).collect();
        let med = median(&mut offs);
        for m in group.iter_mut() {
            m.part = k;
            m.inlier = (m.offset_ms - med).abs() <= step_ms;
        }
        let range = matches.len()..matches.len() + group.len();
        let (start_s, end_s) = (group[0].t1_s, group[group.len() - 1].t1_s);
        let inliers = group.iter().filter(|m| m.inlier).count();
        matches.extend(group);
        parts.push(Part { offset_ms: med, start_s, end_s, matches: range, inliers });
    }
    let main = (0..parts.len()).fold(0, |best, k| if parts[k].matches.len() > parts[best].matches.len() { k } else { best });
    let own = &matches[parts[main].matches.clone()];
    let mut offs: Vec<f64> = own.iter().map(|m| m.offset_ms).collect();
    let median_ms = median(&mut offs); // offs is now sorted
    let inl: Vec<(f64, f64)> = own.iter().filter(|m| m.inlier).map(|m| (m.t1_s, m.offset_ms)).collect();
    let (coarse, peak) = peaks[own[0].peak];
    let part_peaks: Vec<i64> = parts.iter().map(|q| peaks[matches[q.matches.start].peak].0).collect();
    let runner = runner_up(hist.0, hist.1, &part_peaks, p);
    let ratio = if runner > 0.0 { peak / runner } else { f64::INFINITY };
    let summary = Summary {
        median_ms,
        p10_ms: percentile(&offs, 10.0),
        p90_ms: percentile(&offs, 90.0),
        inliers: inl.len(),
        drift_ms_per_h: drift(&inl).map(|(d, se)| (d * 3600.0, se * 3600.0)),
        // residual drift refines the speed: t1 = speed (1 + slope) t2 + offset
        measured_speed: drift(&inl).map(|(d, se)| (speed * (1.0 + d / 1000.0), speed * se / 1000.0)),
        coarse,
        peak,
        ratio,
        weak: ratio < WEAK_RATIO || inl.len() < WEAK_INLIERS,
    };
    (matches, parts, main, summary)
}

/// Decodes one source from the analysis start to its end and extracts its landmarks.
/// `progress` gets the decoded fraction; `stop` is polled between reads.
pub fn analyse_track(
    tools: &Tools,
    src: &Source,
    p: &Params,
    progress: &dyn Fn(f32),
    stop: &dyn Fn() -> bool,
) -> Result<Track, String> {
    let mix = ffmpeg::downmix_filter(&src.layout, p.keep_centre);
    let start = p.start_min * 60.0;
    let mut dec = Decoder::start(tools, src, mix.as_deref(), p.sr, start, None)?;
    let mut analyzer = BandAnalyzer::new(p);
    let nb = analyzer.bands.len();
    let total = ((src.duration - start) * p.sr as f64).max(1.0);
    let (mut samples, mut power, mut done) = (Vec::new(), Vec::new(), 0usize);
    while dec.read(&mut samples)? {
        if stop() {
            return Err(CANCELLED.into());
        }
        analyzer.push(&samples, &mut power);
        done += samples.len();
        samples.clear();
        progress((done as f64 / total).min(1.0) as f32);
    }
    if power.is_empty() {
        return Err(format!("No audio decoded from {}", src.path.display()));
    }
    let level_db = power.chunks_exact(nb).map(|r| (10.0 * (r.iter().sum::<f64>() + 1e-12).log10()) as f32).collect();
    let rise = onset_strength(&power, nb);
    let (landmarks, onset) = detect_landmarks(&rise, nb, p);
    Ok(Track { onset, level_db, landmarks })
}

/// Full analysis of two sources; both are decoded in parallel. `progress(file index, fraction)`.
pub fn analyse(
    tools: &Tools,
    sources: [&Source; 2],
    p: &Params,
    cancel: &AtomicBool,
    progress: &(dyn Fn(usize, f32) + Sync),
) -> Result<Analysis, String> {
    p.validate()?;
    let failed = AtomicBool::new(false); // one track failing stops the other
    let stop = || cancel.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed);
    let run = |i: usize| {
        let r = analyse_track(tools, sources[i], p, &|f| progress(i, f), &stop);
        if r.is_err() {
            failed.store(true, Ordering::Relaxed);
        }
        r
    };
    let (r1, r2) = std::thread::scope(|s| {
        let h = s.spawn(|| run(1));
        (run(0), h.join().unwrap_or_else(|_| Err("Analysis thread panicked".into())))
    });
    let tracks = match (r1, r2) {
        (Ok(a), Ok(b)) => [a, b],
        (Err(e), Err(other)) if e == CANCELLED => return Err(other),
        (Err(e), _) | (_, Err(e)) => return Err(e),
    };
    let pairs = similar_pairs(&tracks[0].landmarks, &tracks[1].landmarks, p.lm_min_sim);
    let (speed, votes) = choose_speed(&tracks[0].landmarks, &tracks[1].landmarks, &pairs, p);
    let scaled = scale_track(&tracks[1], speed);
    let (lo, hist) = vote_histogram(&votes, p)?;
    let peaks = vote_peaks(lo, &hist, p);
    let all: Vec<Match> = (0..peaks.len()).flat_map(|k| refine_matches(&tracks[0], &scaled, peaks[k].0, k, p)).collect();
    if !all.iter().any(|m| m.peak == 0) {
        return Err("No landmark matched at the voted offset".into());
    }
    let mut groups = find_parts(all.clone(), peaks.len());
    if groups.is_empty() {
        // no part qualifies: the strongest peak's matches form the only part
        groups = vec![all.into_iter().filter(|m| m.peak == 0).collect()];
    }
    let (matches, parts, main, summary) = build_parts(groups, &peaks, (lo, &hist), speed, p);
    let notes = rate_note(sources[0], sources[1], speed, p.speed.is_some()).into_iter().collect();
    Ok(Analysis { params: p.clone(), tracks, speed, votes, peaks, matches, parts, main, summary, notes })
}
