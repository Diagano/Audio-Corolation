//! Command-line front end; output matches `audio_offset.py --mode landmarks` for cross-checking.
//! Defaults come from the config file (see `config`); options override them.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use audio_offset::analysis::{self, fmt_time, Params};
use audio_offset::config::{self, Config};
use audio_offset::ffmpeg::{self, Tools};

const USAGE: &str = "usage: audio-offset-cli FILE1 FILE2 [--track1 N] [--track2 N] [--speed auto|RATIO] [--sr HZ] [--frame N] [--hop N] [--speech-band LO HI]
       [--full-band] [--keep-centre] [--offset MIN] [--max-lag MS] [--lm-per-min N] [--lm-gap S]
       [--lm-min-rise X] [--lm-min-sim X] [--ffmpeg-dir DIR]";

struct Args {
    files: [PathBuf; 2],
    /// 1-based; None takes the default track
    tracks: [Option<usize>; 2],
    params: Params,
    tools: Tools,
}

fn parse(args: &[String]) -> Result<Args, String> {
    let (cfg, notes) = Config::load();
    for n in notes {
        eprintln!("Warning: {}: {n}", config::path().display());
    }
    let mut p = cfg.params;
    let mut tools = Tools::new(&cfg.ffmpeg_dir);
    let mut files = Vec::new();
    let mut tracks = [None; 2];
    let mut it = args.iter();
    fn val<T: std::str::FromStr>(it: &mut std::slice::Iter<String>, name: &str) -> Result<T, String> {
        it.next().and_then(|v| v.parse().ok()).ok_or_else(|| format!("{name} needs a valid value"))
    }
    while let Some(a) = it.next() {
        match a.as_str() {
            "--track1" => tracks[0] = Some(val(&mut it, a)?),
            "--track2" => tracks[1] = Some(val(&mut it, a)?),
            "--speed" => p.speed = analysis::parse_speed(&val::<String>(&mut it, a)?)?,
            "--sr" => p.sr = val(&mut it, a)?,
            "--frame" => p.frame = val(&mut it, a)?,
            "--hop" => p.hop = val(&mut it, a)?,
            "--speech-band" => (p.speech_lo, p.speech_hi) = (val(&mut it, a)?, val(&mut it, a)?),
            "--full-band" => p.full_band = true,
            "--keep-centre" => p.keep_centre = true,
            "--offset" => p.start_min = val(&mut it, a)?,
            "--max-lag" => p.max_lag_ms = Some(val(&mut it, a)?),
            "--lm-per-min" => p.lm_per_min = val(&mut it, a)?,
            "--lm-gap" => p.lm_gap_s = val(&mut it, a)?,
            "--lm-min-rise" => p.lm_min_rise = val(&mut it, a)?,
            "--lm-min-sim" => p.lm_min_sim = val(&mut it, a)?,
            "--ffmpeg-dir" => tools = Tools::new(val::<String>(&mut it, a)?),
            "-h" | "--help" => return Err(USAGE.into()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}")),
            s => files.push(PathBuf::from(s)),
        }
    }
    let files = <[PathBuf; 2]>::try_from(files).map_err(|_| USAGE.to_string())?;
    p.validate()?;
    Ok(Args { files, tracks, params: p, tools })
}

fn run(args: &[String]) -> Result<(), String> {
    let Args { files, tracks, params: p, tools } = parse(args)?;
    let sources = [ffmpeg::probe(&tools, &files[0], tracks[0])?, ffmpeg::probe(&tools, &files[1], tracks[1])?];
    for (i, src) in sources.iter().enumerate() {
        let mix = ffmpeg::downmix_filter(&src.layout, p.keep_centre);
        println!("File{}: {}", i + 1, ffmpeg::describe(src, mix.as_deref()));
    }
    if ffmpeg::same_track(&sources[0], &sources[1]) {
        println!("Warning: both inputs are the same audio track");
    }
    let a = analysis::analyse(&tools, [&sources[0], &sources[1]], &p, &AtomicBool::new(false), &|_, _| {})?;
    for (i, t) in a.tracks.iter().enumerate() {
        let minutes = t.onset.len() as f64 * p.frame_s() / 60.0;
        println!("File{}: {} landmarks over {minutes:.1} min", i + 1, t.landmarks.len());
    }
    let step_ms = p.frame_s() * 1000.0;
    let s = &a.summary;
    println!("Vote peak: {:.0} votes at {:+.0} ms, {:.1}x the runner-up", s.peak, s.coarse as f64 * step_ms, s.ratio);
    println!("Matched landmarks: {} ({} within +-{step_ms:.0} ms of the median)", a.parts[a.main].matches.len(), s.inliers);
    println!("Estimated global offset (median, p10, p90): {:+.2}ms, {:+.2}ms, {:+.2}ms", s.median_ms, s.p10_ms, s.p90_ms);
    if let Some((drift, se)) = s.drift_ms_per_h {
        println!("Drift across inliers: {drift:+.2} +- {se:.2} ms/hour");
    }
    if s.weak {
        println!("Warning: weak match (few landmarks or ambiguous vote peak)");
    }
    if a.speed != 1.0 {
        let mut line = format!("Speed: File2 runs {:.6}x File1 ({})", a.speed, analysis::speed_name(a.speed));
        if let Some((m, se)) = s.measured_speed {
            line += &format!(", measured {m:.6} +- {se:.6}");
        }
        println!("{line}; offsets apply to File2 retimed to File1's speed");
        println!("Retime File2: {}", analysis::retime_hint(a.speed, sources[1].sample_rate));
    }
    for n in &a.notes {
        println!("Note: {n}");
    }
    if a.parts.len() > 1 {
        println!("Parts with their own offset: {} (lines above: the largest); File1 time, offset, landmarks", a.parts.len());
        for (k, part) in a.parts.iter().enumerate() {
            let mut line = format!(
                "  {} - {}  {:+.2}ms  {}",
                fmt_time(part.start_s),
                fmt_time(part.end_s),
                part.offset_ms,
                part.matches.len()
            );
            if k > 0 {
                let prev = &a.parts[k - 1];
                let d = part.offset_ms - prev.offset_ms;
                let place = if prev.end_s >= part.start_s {
                    format!("at {}", fmt_time(part.start_s))
                } else {
                    format!("between {} and {}", fmt_time(prev.end_s), fmt_time(part.start_s))
                };
                let what = if d > 0.0 { "missing in" } else { "extra in" };
                line += &format!("  ({:.1} ms {what} File2 {place})", d.abs());
            }
            println!("{line}");
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
