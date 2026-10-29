#![cfg_attr(windows, windows_subsystem = "windows")]
//! GUI: pick or drop two files, choose the analysis options, run it, inspect and listen to the alignment.
//! Rarely changed settings (ffmpeg folder, resolution, speech band) live in the config file; see `config`.
//! Usage: `audio-offset [FILE1 [FILE2]] [--analyse]`: files fill the slots (also when dropped on the program icon);
//! `--analyse` starts the analysis once both are set.

use std::num::NonZero;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

use audio_offset::analysis::{self, fmt_time, Analysis, Params};
use audio_offset::config::{self, Config};
use audio_offset::ffmpeg::{self, Source, Tools};
use eframe::egui::{self, Color32, RichText};
use egui_plot::{Bar, BarChart, HoverPosition, Legend, Line, Plot, Points, VLine};

const CLIP_RATE: u32 = 48_000;
const LANE_POINTS: usize = 2000; // envelope buckets drawn per lane across the visible range
const VOTE_WINDOW_S: f64 = 5.0; // vote histogram span either side of the peak
const FILE_COLORS: [Color32; 2] = [Color32::from_rgb(70, 140, 230), Color32::from_rgb(230, 140, 50)];
const MATCH_COLOR: Color32 = Color32::from_rgb(60, 180, 90);
const OUTLIER_COLOR: Color32 = Color32::from_rgb(210, 70, 70);
const ERROR_COLOR: Color32 = Color32::from_rgb(220, 80, 80);
const WARN_COLOR: Color32 = Color32::from_rgb(220, 180, 60);
const SEARCH_RANGES: [(f64, &str); 6] =
    [(0.0, "Whole file"), (5.0, "± 5 s"), (10.0, "± 10 s"), (30.0, "± 30 s"), (60.0, "± 1 min"), (300.0, "± 5 min")];
const MEDIA_EXT: &[&str] = &[
    "mkv", "mp4", "m4v", "mov", "avi", "ts", "m2ts", "mts", "mpg", "mpeg", "vob", "webm", "wmv", "mka", "m4a", "mp3",
    "ac3", "eac3", "dts", "thd", "aac", "flac", "wav", "ogg", "opus", "wma",
];

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Audio Offset")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 560.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let analyse = args.iter().any(|a| a == "--analyse");
    let files: Vec<PathBuf> = args.iter().filter(|a| !a.starts_with("--")).take(2).map(PathBuf::from).collect();
    eframe::run_native("Audio Offset", options, Box::new(move |_| Ok(Box::new(App::new(files, analyse)))))
}

struct Slot {
    path: PathBuf,
    probe: Result<Source, String>,
}

enum Msg {
    Progress(usize, f32),
    Done(Box<Result<Analysis, String>>),
}

struct Job {
    rx: Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    progress: [f32; 2],
}

/// A finished analysis and what the plots draw from it.
struct Shown {
    analysis: Analysis,
    sources: [Source; 2],
    /// Non-speech level per frame scaled to 0..1 per track
    norm: [Vec<f32>; 2],
    /// Landmark indices that are part of a match, per track
    matched: [Vec<bool>; 2],
    /// File 1 time between consecutive parts (midway between their outermost matches)
    cuts: Vec<f64>,
}

#[derive(Clone, Copy)]
enum Listen {
    File(usize),
    Both,
}

struct App {
    cfg: Config,
    saved_cfg: Config,
    cfg_notes: Vec<String>,
    tools_status: Result<String, String>,
    slots: [Option<Slot>; 2],
    slot_rects: [egui::Rect; 2],
    job: Option<Job>,
    result: Option<Result<Shown, String>>,
    cursor_s: Option<f64>,
    audio: Option<(rodio::MixerDeviceSink, rodio::Player)>,
    clip_rx: Option<Receiver<Result<Vec<f32>, String>>>,
    status: String,
    /// Start the analysis on the next frame (from `--analyse`)
    pending_start: bool,
    /// Retime commands write FLAC instead of the source codec
    retime_lossless: bool,
}

/// Level in dB mapped to 0..1 between the 5th percentile and the maximum.
fn normalise(db: &[f32]) -> Vec<f32> {
    let mut sorted = db.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let lo = sorted.get(sorted.len() / 20).copied().unwrap_or(0.0);
    let hi = sorted.last().copied().unwrap_or(1.0).max(lo + 1.0);
    db.iter().map(|v| ((v - lo) / (hi - lo)).clamp(0.0, 1.0)).collect()
}

impl Shown {
    fn new(analysis: Analysis, sources: [Source; 2]) -> Self {
        let norm = analysis.tracks.each_ref().map(|t| normalise(&t.level_db));
        let mut matched = analysis.tracks.each_ref().map(|t| vec![false; t.landmarks.len()]);
        for m in &analysis.matches {
            matched[0][m.i1] = true;
            matched[1][m.i2] = true;
        }
        let cuts = analysis.parts.windows(2).map(|w| (w[0].end_s + w[1].start_s) / 2.0).collect();
        Self { analysis, sources, norm, matched, cuts }
    }

    /// Part covering File 1 time x.
    fn part_at(&self, x: f64) -> usize {
        self.cuts.iter().take_while(|&&c| x >= c).count()
    }

    fn offset_s(&self, part: usize) -> f64 {
        self.analysis.parts[part].offset_ms / 1000.0
    }

    /// File 1 time range a part covers.
    fn span(&self, part: usize) -> (f64, f64) {
        let lo = if part == 0 { f64::NEG_INFINITY } else { self.cuts[part - 1] };
        (lo, self.cuts.get(part).copied().unwrap_or(f64::INFINITY))
    }
}

impl App {
    fn new(files: Vec<PathBuf>, analyse: bool) -> Self {
        let (cfg, cfg_notes) = Config::load();
        let tools_status = Tools::new(&cfg.ffmpeg_dir).check();
        let mut app = Self {
            saved_cfg: cfg.clone(),
            cfg,
            cfg_notes,
            tools_status,
            slots: [None, None],
            slot_rects: [egui::Rect::NOTHING; 2],
            job: None,
            result: None,
            cursor_s: None,
            audio: None,
            clip_rx: None,
            status: String::new(),
            pending_start: analyse,
            retime_lossless: false,
        };
        for (i, f) in files.into_iter().enumerate() {
            app.set_file(i, f);
        }
        app
    }

    fn tools(&self) -> Tools {
        Tools::new(&self.cfg.ffmpeg_dir)
    }

    fn save_config(&mut self) {
        if self.cfg != self.saved_cfg {
            match self.cfg.save() {
                Ok(()) => self.saved_cfg = self.cfg.clone(),
                Err(e) => self.status = format!("Saving config: {e}"),
            }
        }
    }

    /// Fills slot i. A file with several audio tracks put in File 1 while File 2 is empty also goes to File 2, with
    /// its next track: the usual case of two languages in one file.
    fn set_file(&mut self, i: usize, path: PathBuf) {
        let probe = ffmpeg::probe(&self.tools(), &path, None);
        if let (0, None, Ok(src)) = (i, &self.slots[1], &probe) {
            if src.tracks.len() > 1 {
                let mut other = src.clone();
                other.select_track(if src.track == 0 { 1 } else { 0 });
                self.slots[1] = Some(Slot { path: path.clone(), probe: Ok(other) });
            }
        }
        self.slots[i] = Some(Slot { path, probe });
        self.result = None;
        self.cursor_s = None;
    }

    fn sources(&self) -> Option<[Source; 2]> {
        let src = |i: usize| self.slots[i].as_ref().and_then(|s| s.probe.as_ref().ok()).cloned();
        Some([src(0)?, src(1)?])
    }

    fn poll(&mut self) {
        if let Some(job) = &mut self.job {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    Msg::Progress(i, f) => job.progress[i] = f,
                    Msg::Done(r) => {
                        let sources = self.sources();
                        self.result = Some((*r).and_then(|a| {
                            Ok(Shown::new(a, sources.ok_or("Files changed during the analysis")?))
                        }));
                        self.cursor_s = None;
                        self.job = None;
                        break;
                    }
                }
            }
        }
        if let Some(rx) = &self.clip_rx {
            if let Ok(r) = rx.try_recv() {
                self.clip_rx = None;
                match r {
                    Ok(samples) => self.play(samples),
                    Err(e) => self.status = e,
                }
            }
        }
    }

    fn start_job(&mut self, ctx: &egui::Context) {
        let Some(sources) = self.sources() else { return };
        self.save_config();
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tools, params, stop, ctx) = (self.tools(), self.cfg.params.clone(), cancel.clone(), ctx.clone());
        std::thread::spawn(move || {
            let progress = |i: usize, f: f32| {
                let _ = tx.send(Msg::Progress(i, f));
                ctx.request_repaint();
            };
            let r = analysis::analyse(&tools, [&sources[0], &sources[1]], &params, &stop, &progress);
            let _ = tx.send(Msg::Done(Box::new(r)));
            ctx.request_repaint();
        });
        self.job = Some(Job { rx, cancel, progress: [0.0; 2] });
        self.result = None;
        self.status.clear();
    }

    fn handle_drops(&mut self, ctx: &egui::Context) {
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).filter(|p| !p.as_os_str().is_empty()).collect()
        });
        if self.job.is_some() || dropped.is_empty() {
            return;
        }
        if dropped.len() >= 2 {
            let mut it = dropped.into_iter();
            self.set_file(0, it.next().unwrap());
            self.set_file(1, it.next().unwrap());
            return;
        }
        let pointer = ctx.input(|i| i.pointer.latest_pos());
        let target = pointer
            .and_then(|p| self.slot_rects.iter().position(|r| r.contains(p)))
            .or_else(|| self.slots.iter().position(|s| s.is_none()))
            .unwrap_or(1);
        self.set_file(target, dropped.into_iter().next().unwrap());
    }

    fn drop_overlay(&self, ctx: &egui::Context) {
        if ctx.input(|i| i.raw.hovered_files.is_empty()) {
            return;
        }
        let rect = ctx.content_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop")));
        painter.rect_filled(rect, 0.0, Color32::from_black_alpha(150));
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Drop two files, or one file onto its slot",
            egui::FontId::proportional(24.0),
            Color32::WHITE,
        );
    }

    fn listen(&mut self, which: Listen, ctx: &egui::Context) {
        let (Some(Ok(shown)), Some(cursor)) = (&self.result, self.cursor_s) else { return };
        let (tools, sources, offset) = (self.tools(), shown.sources.clone(), shown.offset_s(shown.part_at(cursor)));
        let speed = shown.analysis.speed;
        let (len, start) = (self.cfg.clip_seconds.max(0.5), cursor - self.cfg.clip_lead);
        let (tx, rx) = mpsc::channel();
        self.clip_rx = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            // File 1 time t is File 2 time (t - offset) / speed; File 2 is played retimed to File 1's speed, so
            // one of its seconds lasts `k` = speed output seconds
            let clip = |i: usize, at: f64, k: f64| -> Result<Vec<f32>, String> {
                // before the file start: silence, then the file from 0
                let n = (len * CLIP_RATE as f64) as usize;
                let pad_in = (-at).max(0.0);
                let mut s = vec![0.0; ((pad_in * k * CLIP_RATE as f64) as usize).min(n)];
                if pad_in * k < len {
                    let tempo = (k != 1.0).then(|| format!("atempo={}", 1.0 / k));
                    s.extend(ffmpeg::decode(&tools, &sources[i], tempo.as_deref(), CLIP_RATE, at.max(0.0), Some(len / k - pad_in))?);
                }
                s.resize(n, 0.0);
                Ok(s)
            };
            let at2 = (start - offset) / speed;
            let r = (|| {
                let (a, b) = match which {
                    Listen::File(0) => (clip(0, start, 1.0)?, None),
                    Listen::File(_) => (clip(1, at2, speed)?, None),
                    Listen::Both => (clip(0, start, 1.0)?, Some(clip(1, at2, speed)?)),
                };
                // stereo: both channels the same clip, or left = file 1 and right = file 2
                let right = b.as_ref().unwrap_or(&a);
                Ok(a.iter().zip(right).flat_map(|(l, r)| [*l, *r]).collect())
            })();
            let _ = tx.send(r);
            ctx.request_repaint();
        });
    }

    fn play(&mut self, samples: Vec<f32>) {
        if self.audio.is_none() {
            match rodio::DeviceSinkBuilder::open_default_sink() {
                Ok(sink) => {
                    let player = rodio::Player::connect_new(sink.mixer());
                    self.audio = Some((sink, player));
                }
                Err(e) => {
                    self.status = format!("Audio output: {e}");
                    return;
                }
            }
        }
        if let Some((_, player)) = &self.audio {
            player.clear();
            let buffer = rodio::buffer::SamplesBuffer::new(NonZero::new(2).unwrap(), NonZero::new(CLIP_RATE).unwrap(), samples);
            player.append(buffer);
            player.play();
        }
    }

    // ---------------- Panels ----------------

    fn files_ui(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let mut picked = None;
        let mut cleared = None;
        let mut track_pick = None;
        for (i, color) in FILE_COLORS.into_iter().enumerate() {
            let resp = egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("File {}", i + 1)).strong().color(color));
                    if ui.add_enabled(self.job.is_none(), egui::Button::new("Browse…")).clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .add_filter("Media", MEDIA_EXT)
                            .add_filter("All files", &["*"])
                            .pick_file()
                        {
                            picked = Some((i, p));
                        }
                    }
                    match &self.slots[i] {
                        None => {
                            ui.weak(if i == 0 { "Reference — drop a file here or browse" } else { "Other version — drop a file here or browse" });
                        }
                        Some(slot) => {
                            let name = slot.path.file_name().map_or(slot.path.display().to_string(), |n| n.to_string_lossy().into());
                            ui.label(RichText::new(name).strong()).on_hover_text(slot.path.display().to_string());
                            if ui.add_enabled(self.job.is_none(), egui::Button::new("✕").small()).on_hover_text("Remove").clicked() {
                                cleared = Some(i);
                            }
                        }
                    }
                });
                if let Some(slot) = &self.slots[i] {
                    match &slot.probe {
                        Ok(src) => {
                            ui.horizontal(|ui| {
                                ui.label("Audio track");
                                let current = format!("{}: {}", src.track + 1, src.tracks[src.track].label());
                                ui.add_enabled_ui(self.job.is_none() && src.tracks.len() > 1, |ui| {
                                    egui::ComboBox::from_id_salt(("track", i))
                                        .selected_text(current)
                                        .width(420.0)
                                        .show_ui(ui, |ui| {
                                            for (k, t) in src.tracks.iter().enumerate() {
                                                if ui.selectable_label(k == src.track, format!("{}: {}", k + 1, t.label())).clicked() {
                                                    track_pick = Some((i, k));
                                                }
                                            }
                                        })
                                        .response
                                        .on_hover_text("Audio track of this file taking part in the comparison");
                                });
                                if src.tracks.len() > 1 {
                                    ui.weak(format!("{} tracks", src.tracks.len()));
                                }
                            });
                            let mix = ffmpeg::downmix_filter(&src.layout, self.cfg.params.keep_centre);
                            ui.weak(format!("{}, {} — {}", src.codec, ffmpeg::describe(src, mix.as_deref()), fmt_time(src.duration)));
                        }
                        Err(e) => {
                            ui.colored_label(ERROR_COLOR, e);
                        }
                    }
                }
            });
            self.slot_rects[i] = resp.response.rect;
        }
        if let Some((i, p)) = picked {
            self.set_file(i, p);
        }
        if let Some(i) = cleared {
            self.slots[i] = None;
            self.result = None;
        }
        if let Some((i, k)) = track_pick {
            if let Some(Ok(src)) = self.slots[i].as_mut().map(|s| s.probe.as_mut()) {
                src.select_track(k);
                self.result = None;
                self.cursor_s = None;
            }
        }

        ui.horizontal(|ui| {
            if let Some(job) = &self.job {
                if ui.button("Cancel").clicked() {
                    job.cancel.store(true, Ordering::Relaxed);
                }
                for (i, f) in job.progress.iter().enumerate() {
                    ui.add(egui::ProgressBar::new(*f).desired_width(220.0).text(format!("File {}: {:.0} %", i + 1, f * 100.0)));
                }
            } else {
                let valid = self.cfg.params.validate();
                let ready = self.sources().is_some() && self.tools_status.is_ok() && valid.is_ok();
                let run = ui
                    .add_enabled(ready, egui::Button::new(RichText::new("▶  Analyse").strong()).min_size(egui::vec2(110.0, 0.0)))
                    .on_hover_text("Decode both files and find the offset");
                if run.clicked() {
                    self.start_job(ui.ctx());
                }
                if ui
                    .add_enabled(self.slots.iter().any(Option::is_some), egui::Button::new("⇅ Swap"))
                    .on_hover_text("Swap File 1 and File 2 (the offset changes sign)")
                    .clicked()
                {
                    self.slots.swap(0, 1);
                    self.result = None;
                }
                if let Err(e) = valid {
                    ui.colored_label(ERROR_COLOR, e);
                }
            }
            if let Some(src) = self.sources() {
                if ffmpeg::same_track(&src[0], &src[1]) {
                    ui.colored_label(WARN_COLOR, "Same file and audio track in both slots");
                }
                if let (Some(r1), Some(r2)) = (src[0].frame_rate, src[1].frame_rate) {
                    if (r1 - r2).abs() / r1.max(r2) > 1e-4 {
                        ui.colored_label(WARN_COLOR, format!("Frame rates differ ({r1:.3} vs {r2:.3} fps): the audio speed is measured"))
                            .on_hover_text("Speed-ups (PAL) change the audio speed; telecine and standards conversion keep it");
                    }
                }
            }
            if !self.status.is_empty() {
                ui.colored_label(WARN_COLOR, &self.status);
            }
        });
        ui.add_space(4.0);
    }

    fn options_ui(&mut self, ui: &mut egui::Ui) {
        let max_start = self.sources().map_or(600.0, |s| s[0].duration.min(s[1].duration) / 60.0);
        let enabled = self.job.is_none();
        let p = &mut self.cfg.params;
        ui.add_enabled_ui(enabled, |ui| {
            ui.heading("Options");
            ui.add_space(4.0);
            ui.label(RichText::new("Dialogue removal").strong());
            let mut drop_centre = !p.keep_centre;
            ui.checkbox(&mut drop_centre, "Drop centre channel").on_hover_text(
                "Surround mixes carry dialogue in the centre channel; dropping it leaves music & effects.\n\
                 No effect on stereo or mono sources.",
            );
            p.keep_centre = !drop_centre;
            let mut exclude = !p.full_band;
            ui.checkbox(&mut exclude, format!("Ignore speech band ({:.0}–{:.0} Hz)", p.speech_lo, p.speech_hi))
                .on_hover_text("Analyse only below and above the dialogue band. Band edges are set in the config file.");
            p.full_band = !exclude;

            ui.add_space(8.0);
            ui.label(RichText::new("Landmarks").strong());
            egui::Grid::new("landmarks").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("Density");
                ui.add(egui::Slider::new(&mut p.lm_per_min, 5..=60).suffix(" /min")).on_hover_text(
                    "Most landmarks kept per minute (strongest first).\n\
                     More catches quieter events but admits more dialogue onsets.",
                );
                ui.end_row();
                ui.label("Min rise");
                let mut db = p.lm_min_rise * 10.0;
                ui.add(egui::Slider::new(&mut db, 3.0..=30.0).step_by(0.5).suffix(" dB")).on_hover_text(
                    "How sharply the level must jump to count as an event.\n\
                     Lower finds softer events; higher keeps only hard hits (shots, slams).",
                );
                p.lm_min_rise = db / 10.0;
                ui.end_row();
                ui.label("Min spacing");
                ui.add(egui::Slider::new(&mut p.lm_gap_s, 0.1..=2.0).step_by(0.05).suffix(" s")).on_hover_text(
                    "Closest two landmarks may be. Larger spreads landmarks out; smaller resolves rapid events (gunfire).",
                );
                ui.end_row();
                ui.label("Match strictness");
                ui.add(egui::Slider::new(&mut p.lm_min_sim, 0.5..=0.99).step_by(0.01)).on_hover_text(
                    "How alike two landmarks' band signatures must be to vote.\n\
                     Higher rejects chance matches; lower tolerates mix differences between versions.",
                );
                ui.end_row();
            });

            ui.add_space(8.0);
            ui.label(RichText::new("Search").strong());
            egui::Grid::new("search").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("Range");
                let current = p.max_lag_ms.map_or(0.0, |ms| ms / 1000.0);
                let label = SEARCH_RANGES
                    .iter()
                    .find(|(s, _)| *s == current)
                    .map_or(format!("± {current} s"), |(_, l)| l.to_string());
                egui::ComboBox::from_id_salt("range")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        for (s, l) in SEARCH_RANGES {
                            if ui.selectable_label(current == s, l).clicked() {
                                p.max_lag_ms = (s > 0.0).then_some(s * 1000.0);
                            }
                        }
                    })
                    .response
                    .on_hover_text(
                        "Largest offset considered, either direction.\n\
                         Narrower rejects far-off false peaks; too narrow misses the true offset.",
                    );
                ui.end_row();
                ui.label("Speed");
                let label = p.speed.map_or("Auto".to_string(), analysis::speed_name);
                egui::ComboBox::from_id_salt("speed")
                    .selected_text(label)
                    .show_ui(ui, |ui| {
                        if ui.selectable_label(p.speed.is_none(), "Auto").clicked() {
                            p.speed = None;
                        }
                        for (s, name) in analysis::SPEEDS {
                            let what = match name {
                                "1" => "same speed",
                                "25/24" | "25/23.976" => "File 2 sped up (PAL)",
                                "24/25" | "23.976/25" => "File 1 sped up (PAL)",
                                "24/23.976" => "File 2 at 24, File 1 at 23.976",
                                _ => "File 2 at 23.976, File 1 at 24",
                            };
                            if ui.selectable_label(p.speed == Some(s), format!("{name}  —  {what}")).clicked() {
                                p.speed = Some(s);
                            }
                        }
                    })
                    .response
                    .on_hover_text(
                        "How fast File 2 plays relative to File 1. Auto tries these ratios and keeps the one\n\
                         whose votes agree best; frame rates alone do not tell (telecine keeps the speed).",
                    );
                ui.end_row();
                ui.label("Start at");
                ui.add(egui::DragValue::new(&mut p.start_min).range(0.0..=max_start.max(0.0)).speed(0.1).suffix(" min"))
                    .on_hover_text("Skip the opening of both files, e.g. differing studio logos or intros.");
                ui.end_row();
            });
            ui.add_space(8.0);
            if ui.button("Reset options").on_hover_text("Options back to their defaults; config file settings stay").clicked() {
                let d = Params::default();
                (p.keep_centre, p.full_band, p.lm_per_min, p.lm_min_rise) = (d.keep_centre, d.full_band, d.lm_per_min, d.lm_min_rise);
                (p.lm_gap_s, p.lm_min_sim, p.max_lag_ms, p.start_min) = (d.lm_gap_s, d.lm_min_sim, d.max_lag_ms, d.start_min);
                p.speed = d.speed;
            }
        });

        ui.add_space(12.0);
        ui.separator();
        ui.label(RichText::new("Configuration").strong());
        match &self.tools_status {
            Ok(v) => {
                let from = if self.cfg.ffmpeg_dir.is_empty() { "PATH".to_string() } else { self.cfg.ffmpeg_dir.clone() };
                ui.label(RichText::new("✔ ffmpeg found").color(MATCH_COLOR)).on_hover_text(format!("{v}\nfrom {from}"));
            }
            Err(e) => {
                ui.colored_label(ERROR_COLOR, format!("✘ {e}"));
                ui.weak("Install ffmpeg, add it to PATH, or choose its folder:");
            }
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("ffmpeg folder…").on_hover_text("Folder holding ffmpeg and ffprobe; stored in the config file").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.cfg.ffmpeg_dir = dir.display().to_string();
                    self.tools_status = self.tools().check();
                    self.save_config();
                }
            }
            if !self.cfg.ffmpeg_dir.is_empty() && ui.button("Use PATH").clicked() {
                self.cfg.ffmpeg_dir.clear();
                self.tools_status = self.tools().check();
                self.save_config();
            }
            if ui.button("Edit config").on_hover_text("Open the config file in the default editor").clicked() {
                self.save_config();
                if !config::path().exists() {
                    if let Err(e) = self.cfg.save() {
                        self.status = format!("Saving config: {e}");
                    }
                }
                let opener = if cfg!(windows) { "explorer" } else { "xdg-open" };
                if let Err(e) = std::process::Command::new(opener).arg(config::path()).spawn() {
                    self.status = format!("Opening config: {e}");
                }
            }
            if ui.button("Reload").on_hover_text("Re-read the config file after editing it").clicked() {
                let (cfg, notes) = Config::load();
                (self.saved_cfg, self.cfg, self.cfg_notes) = (cfg.clone(), cfg, notes);
                self.tools_status = self.tools().check();
            }
        });
        ui.weak(config::path().display().to_string());
        for n in &self.cfg_notes {
            ui.colored_label(WARN_COLOR, format!("Config {n}"));
        }
    }

    fn results_ui(&mut self, ui: &mut egui::Ui) {
        let tools = self.tools();
        let shown = match &self.result {
            None if self.job.is_some() => {
                ui.centered_and_justified(|ui| ui.spinner());
                return;
            }
            None => {
                ui.centered_and_justified(|ui| ui.weak("Choose or drop two files, then Analyse."));
                return;
            }
            Some(Err(e)) => {
                ui.colored_label(ERROR_COLOR, e);
                return;
            }
            Some(Ok(shown)) => shown,
        };
        let a = &shown.analysis;
        let s = &a.summary;
        let step_ms = a.params.frame_s() * 1000.0;
        let mut action = None;

        let parts = &a.parts;
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Offset  {:+.2} ms", s.median_ms)).size(22.0).strong());
            if parts.len() > 1 {
                ui.label(RichText::new(format!("largest of {} parts", parts.len())).color(WARN_COLOR));
            }
            let tip = if parts.len() > 1 { "Copy the parts table (tab separated)" } else { "Copy the offset (ms)" };
            if ui.button("Copy").on_hover_text(tip).clicked() {
                ui.ctx().copy_text(if parts.len() > 1 { parts_text(a) } else { format!("{:.2}", s.median_ms) });
            }
            ui.weak("File 1 − File 2: positive means events occur later in File 1");
        });
        let drift = s.drift_ms_per_h.map_or(String::new(), |(d, se)| format!(", drift {d:+.2} ± {se:.2} ms/h"));
        ui.label(format!(
            "{} matched landmarks ({} within ±{step_ms:.0} ms), 10–90 % {:+.1} … {:+.1} ms, vote peak {:.1}× the runner-up{drift}",
            parts[a.main].matches.len(),
            s.inliers,
            s.p10_ms,
            s.p90_ms,
            s.ratio,
        ));
        if s.weak {
            ui.colored_label(WARN_COLOR, "Weak match: few landmarks or an ambiguous vote peak — check by listening");
        }
        if a.speed != 1.0 {
            let measured = s.measured_speed.map_or(String::new(), |(m, se)| format!(", measured {m:.6} ± {se:.6}"));
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(format!("Speed: File 2 runs {:.6}× File 1 ({}){measured}", a.speed, analysis::speed_name(a.speed))).strong())
                    .on_hover_text("Offsets apply to File 2 retimed to File 1's speed; its audio must be re-encoded at that speed");
            });
            ui.horizontal_wrapped(|ui| {
                let src2 = &shown.sources[1];
                ui.label("Retime File 2 — copy ffmpeg command:");
                let modes = [
                    ("Keep pitch", false, "atempo: the speed changes, the pitch stays.\nFor File 2 audio whose pitch is already right."),
                    ("Undo pitch shift", true, "asetrate: speed and pitch change together, as PAL speed-ups did,\nso File 2 gets its original pitch back."),
                ];
                for (label, undo_pitch, what) in modes {
                    let filter = analysis::retime_filter(a.speed, src2.sample_rate, undo_pitch);
                    let cmd = ffmpeg::retime_command(&tools, src2, &filter, self.retime_lossless);
                    if ui.button(label).on_hover_text(format!("{what}\n\n{cmd}")).clicked() {
                        ui.ctx().copy_text(cmd);
                    }
                }
                ui.checkbox(&mut self.retime_lossless, "Lossless (FLAC)").on_hover_text(
                    "Write FLAC instead of re-encoding to the source codec (AC-3 640k, E-AC-3 1536k, AAC, ...).\n\
                     DTS, TrueHD and PCM sources always get FLAC.",
                );
                ui.weak("→ <File 2 name>.retimed.mka next to File 2; then mux it with the offset as delay");
            });
        }
        for n in &a.notes {
            ui.colored_label(WARN_COLOR, n);
        }
        if parts.len() > 1 {
            ui.label("The versions differ in content: each part below has its own offset (cut or inserted material between).")
                .on_hover_text("A part needs at least 4 matched landmarks; shorter differences are not resolved");
            egui::Grid::new("parts").striped(true).spacing([16.0, 2.0]).show(ui, |ui| {
                for h in ["Part", "File 1 time", "Offset", "Landmarks", "Change before it (File 2)"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for (k, part) in parts.iter().enumerate() {
                    ui.label(format!("{}{}", k + 1, if k == a.main { " (largest)" } else { "" }));
                    ui.label(format!("{} – {}", fmt_time(part.start_s), fmt_time(part.end_s)));
                    ui.label(format!("{:+.2} ms", part.offset_ms));
                    ui.label(part.matches.len().to_string());
                    ui.label(change_text(parts, k));
                    ui.end_row();
                }
            });
        }

        ui.horizontal(|ui| {
            match self.cursor_s {
                Some(c) => ui.label(format!("Cursor {}", fmt_time(c))),
                None => ui.weak("Click the lanes to place the cursor"),
            };
            let can = self.cursor_s.is_some() && self.clip_rx.is_none();
            let tips = ["Play File 1 at the cursor", "Play File 2 at the matching time"];
            for (i, tip) in tips.into_iter().enumerate() {
                if ui.add_enabled(can, egui::Button::new(format!("▶ File {}", i + 1))).on_hover_text(tip).clicked() {
                    action = Some(Listen::File(i));
                }
            }
            if ui
                .add_enabled(can, egui::Button::new("▶ Both"))
                .on_hover_text("File 1 in the left ear, File 2 in the right: aligned events sound as one")
                .clicked()
            {
                action = Some(Listen::Both);
            }
            if ui.add_enabled(self.audio.is_some(), egui::Button::new("■")).on_hover_text("Stop").clicked() {
                if let Some((_, player)) = &self.audio {
                    player.clear();
                }
            }
        });

        let h = ui.available_height();
        let clicked = lanes_plot(ui, shown, self.cursor_s, (h * 0.58).max(160.0));
        ui.add_space(6.0);
        let lower = (ui.available_height() - 4.0).max(120.0);
        ui.columns(2, |cols| {
            votes_plot(&mut cols[0], shown, lower);
            offsets_plot(&mut cols[1], shown, lower);
        });

        if clicked.is_some() {
            self.cursor_s = clicked;
        }
        if let Some(which) = action {
            self.listen(which, &ui.ctx().clone());
        }
    }
}

/// "3002.3 ms missing at 5:42.0" for the boundary before part k; empty for the first part.
fn change_text(parts: &[analysis::Part], k: usize) -> String {
    if k == 0 {
        return String::new();
    }
    let (prev, part) = (&parts[k - 1], &parts[k]);
    let d = part.offset_ms - prev.offset_ms;
    let place = if prev.end_s >= part.start_s {
        format!("at {}", fmt_time(part.start_s))
    } else {
        format!("between {} and {}", fmt_time(prev.end_s), fmt_time(part.start_s))
    };
    format!("{:.1} ms {} {place}", d.abs(), if d > 0.0 { "missing" } else { "extra" })
}

fn parts_text(a: &Analysis) -> String {
    let mut out = String::from("part\tstart\tend\toffset_ms\tlandmarks\tchange\n");
    for (k, p) in a.parts.iter().enumerate() {
        out += &format!(
            "{}\t{}\t{}\t{:.2}\t{}\t{}\n",
            k + 1,
            fmt_time(p.start_s),
            fmt_time(p.end_s),
            p.offset_ms,
            p.matches.len(),
            change_text(&a.parts, k)
        );
    }
    out
}

// ---------------- Plots ----------------

/// Lane envelope over the visible range: min/max per bucket, as one zig-zag polyline.
fn envelope_points(norm: &[f32], x_start: f64, frame_s: f64, base: f64, x0: f64, x1: f64) -> Vec<[f64; 2]> {
    let n = norm.len();
    if n == 0 {
        return Vec::new();
    }
    let k0 = (((x0 - x_start) / frame_s).floor().max(0.0) as usize).min(n - 1);
    let k1 = ((((x1 - x_start) / frame_s).ceil().max(0.0) as usize) + 1).min(n);
    if k1 <= k0 {
        return Vec::new();
    }
    let step = ((k1 - k0) / LANE_POINTS).max(1);
    let x = |k: usize| x_start + k as f64 * frame_s;
    if step == 1 {
        return (k0..k1).map(|k| [x(k), base + norm[k] as f64]).collect();
    }
    let mut pts = Vec::with_capacity(2 * (k1 - k0) / step + 2);
    for s in (k0..k1).step_by(step) {
        let seg = &norm[s..(s + step).min(k1)];
        let (lo, hi) = seg.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        pts.push([x(s), base + hi as f64]);
        pts.push([x(s), base + lo as f64]);
    }
    pts
}

fn time_plot(id: &str, height: f32) -> Plot<'static> {
    Plot::new(id)
        .height(height)
        .link_axis(egui::Id::unique("file1-time"), [true, false])
        .link_cursor(egui::Id::unique("file1-time"), [true, false])
        .allow_zoom([true, false])
        .allow_drag([true, false])
        .allow_scroll([true, false])
        .x_axis_formatter(|mark, _| fmt_time(mark.value))
}

/// File 1 above, File 2 below shifted part by part by its offset, both on File 1's time; matched landmarks tied
/// by lines, part boundaries marked. Returns the clicked time.
fn lanes_plot(ui: &mut egui::Ui, shown: &Shown, cursor: Option<f64>, height: f32) -> Option<f64> {
    let a = &shown.analysis;
    let fs = a.params.frame_s();
    let start = a.params.start_min * 60.0;
    let x2 = |part: usize| start + shown.offset_s(part); // File 1 time of File 2 frame 0, within a part
    let fs2 = fs * a.speed; // File 1 seconds per File 2 frame
    let base = [1.15, 0.0];
    let level_at = |lane: usize, frame: f64| {
        let norm = &shown.norm[lane];
        base[lane] + norm.get(frame.round() as usize).copied().unwrap_or(0.0) as f64
    };
    let resp = time_plot("lanes", height)
        .legend(Legend::default())
        .show_axes([true, false])
        .show_grid([true, false])
        .default_y_bounds(-0.05, 2.2)
        .label_formatter(|hover| match hover {
            HoverPosition::NearDataPoint { position, .. } | HoverPosition::Elsewhere { position } => Some(fmt_time(position.x)),
        })
        .show(ui, |pu| {
            let b = pu.plot_bounds();
            let (x0, x1) = (b.min()[0], b.max()[0]);
            let landmark_points = |lane: usize, xs: &dyn Fn(usize, f64) -> Option<f64>| {
                let (mut hit, mut miss) = (Vec::new(), Vec::new());
                for (i, l) in a.tracks[lane].landmarks.iter().enumerate() {
                    if let Some(x) = xs(i, l.frame).filter(|x| (x0..=x1).contains(x)) {
                        let pt = [x, level_at(lane, l.frame)];
                        if shown.matched[lane][i] { hit.push(pt) } else { miss.push(pt) }
                    }
                }
                (hit, miss)
            };
            let lane1 = envelope_points(&shown.norm[0], start, fs, base[0], x0, x1);
            pu.line(Line::new("File 1", lane1).color(FILE_COLORS[0]).width(1.0));
            let (hit1, miss1) = landmark_points(0, &|_, f| Some(start + f * fs));
            for k in 0..a.parts.len() {
                let (lo, hi) = shown.span(k);
                let (lo, hi) = (lo.max(x0), hi.min(x1));
                if lo < hi {
                    let lane2 = envelope_points(&shown.norm[1], x2(k), fs2, base[1], lo, hi);
                    pu.line(Line::new("File 2", lane2).color(FILE_COLORS[1]).width(1.0));
                }
            }
            // a File 2 landmark is drawn in the part whose span its shifted time falls in
            let (hit2, miss2) = landmark_points(1, &|_, f| {
                (0..a.parts.len()).map(|k| x2(k) + f * fs2).enumerate().find(|(k, x)| {
                    let (lo, hi) = shown.span(*k);
                    *x >= lo && *x < hi
                }).map(|(_, x)| x)
            });
            pu.points(Points::new("Unmatched landmarks", [miss1, miss2].concat()).color(Color32::GRAY).radius(2.0));
            pu.points(Points::new("Matched landmarks", hit1).color(FILE_COLORS[0]).radius(3.5));
            pu.points(Points::new("Matched landmarks", hit2).color(FILE_COLORS[1]).radius(3.5));
            for m in &a.matches {
                let (l1, l2) = (&a.tracks[0].landmarks[m.i1], &a.tracks[1].landmarks[m.i2]);
                let (xa, xb) = (start + l1.frame * fs, x2(m.part) + l2.frame * fs2);
                if xa.max(xb) >= x0 && xa.min(xb) <= x1 {
                    let color = if m.inlier { MATCH_COLOR } else { OUTLIER_COLOR };
                    pu.line(Line::new("Matches", vec![[xa, level_at(0, l1.frame)], [xb, level_at(1, l2.frame)]]).color(color).width(1.0));
                }
            }
            for &c in &shown.cuts {
                pu.vline(VLine::new("Part boundaries", c).color(WARN_COLOR).width(1.5));
            }
            if let Some(c) = cursor {
                pu.vline(VLine::new("Cursor", c).color(Color32::WHITE).width(1.0));
            }
            pu.pointer_coordinate()
        });
    if resp.response.clicked() { resp.inner.map(|p| p.x) } else { None }
}

/// Offset votes around the main peak: a clear single peak means a confident match.
fn votes_plot(ui: &mut egui::Ui, shown: &Shown, height: f32) {
    let a = &shown.analysis;
    let frame_ms = a.params.frame_s() * 1000.0;
    let half = (VOTE_WINDOW_S * 1000.0 / frame_ms) as i64;
    let coarse = a.summary.coarse;
    let mut hist = vec![0.0; (2 * half + 1) as usize];
    for v in &a.votes {
        let k = v.round_ties_even() as i64 - coarse + half;
        if (0..hist.len() as i64).contains(&k) {
            hist[k as usize] += 1.0;
        }
    }
    let bars: Vec<Bar> = hist
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0.0)
        .map(|(k, &c)| Bar::new((k as i64 - half + coarse) as f64 * frame_ms, c).width(frame_ms))
        .collect();
    ui.label(RichText::new("Votes").strong())
        .on_hover_text("Landmark pairs voting for each offset, ±5 s around the largest part's offset");
    Plot::new("votes")
        .height(height - 20.0)
        .x_axis_formatter(|mark, _| format!("{:.0} ms", mark.value))
        .show(ui, |pu| {
            pu.add(BarChart::new("Votes", bars).color(MATCH_COLOR));
            for part in &a.parts {
                pu.vline(VLine::new("Part offsets", part.offset_ms).color(Color32::WHITE).width(1.0));
            }
        });
}

/// Refined offset per matched landmark over File 1 time: flat means constant offset; a slope is drift, a step a cut.
fn offsets_plot(ui: &mut egui::Ui, shown: &Shown, height: f32) {
    let a = &shown.analysis;
    let frame_ms = a.params.frame_s() * 1000.0;
    let (lo, hi) = a.parts.iter().fold((f64::MAX, f64::MIN), |(lo, hi), p| (lo.min(p.offset_ms), hi.max(p.offset_ms)));
    let (inl, out): (Vec<_>, Vec<_>) = a.matches.iter().partition(|m| m.inlier);
    let pts = |ms: &[&analysis::Match]| ms.iter().map(|m| [m.t1_s, m.offset_ms]).collect::<Vec<_>>();
    ui.label(RichText::new("Offset over time").strong())
        .on_hover_text("Offset of each matched landmark: flat is a constant offset, a slope is drift, a step is a cut");
    time_plot("offsets", height - 20.0)
        .default_y_bounds(lo - 5.0 * frame_ms, hi + 5.0 * frame_ms)
        .y_axis_formatter(|mark, _| format!("{:.0} ms", mark.value))
        .show(ui, |pu| {
            for (k, part) in a.parts.iter().enumerate() {
                let (from, to) = shown.span(k);
                let (from, to) = (from.max(part.start_s), to.min(part.end_s));
                pu.line(Line::new("Part offsets", vec![[from, part.offset_ms], [to, part.offset_ms]]).color(Color32::WHITE).width(1.5));
            }
            pu.points(Points::new("Inliers", pts(&inl)).color(MATCH_COLOR).radius(3.0));
            pu.points(Points::new("Outliers", pts(&out)).color(OUTLIER_COLOR).radius(3.0));
        });
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll();
        self.handle_drops(&ctx);
        if std::mem::take(&mut self.pending_start) && self.tools_status.is_ok() && self.cfg.params.validate().is_ok() {
            self.start_job(&ctx);
        }
        egui::Panel::top("files").show(ui, |ui| self.files_ui(ui));
        egui::Panel::left("options").resizable(true).default_size(300.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.options_ui(ui));
        });
        egui::CentralPanel::default().show(ui, |ui| self.results_ui(ui));
        self.drop_overlay(&ctx);
    }

    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        self.save_config();
    }
}
