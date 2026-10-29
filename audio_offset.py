#!/usr/bin/env python3
"""
Alignment of two versions of the same programme (e.g. language dubs), outputs offset in ms.

Dubs share the music & effects track and differ in dialogue, so analysis uses the bands outside
the speech band and drops the centre (dialogue) channel of surround sources.
Modes:
  envelope   cross-correlate non-speech energy envelopes over spaced windows
  landmarks  detect sharp non-speech onsets (shots, slams, engine starts) over the whole file and
             vote on the offset from landmark matches with similar band signatures, then refine
             each matched landmark by local cross-correlation
"""
import os
import sys
import math
import json
import shutil
import argparse
import subprocess
from fractions import Fraction

try:
    import numpy as np
except ImportError:
    sys.exit("Missing required lib (numpy). Try `pip install numpy`")

# Relative frame-rate difference treated as equal (absorbs rational spellings such as 2997/125 vs 24000/1001)
FRAME_RATE_TOL = 1e-4

# Channel order of common layouts (ffprobe names); FC carries the dialogue
LAYOUT_CHANNELS = {
    "3.0": "FL FR FC",
    "4.0": "FL FR FC BC",
    "5.0": "FL FR FC BL BR",
    "5.0(side)": "FL FR FC SL SR",
    "5.1": "FL FR FC LFE BL BR",
    "5.1(side)": "FL FR FC LFE SL SR",
    "6.1": "FL FR FC LFE BC SL SR",
    "7.1": "FL FR FC LFE BL BR SL SR",
    "7.1(wide)": "FL FR FC LFE BL BR FLC FRC",
    "7.1(wide-side)": "FL FR FC LFE FLC FRC SL SR",
}

LM_LOW_HZ = 30.0        # lowest landmark band edge; below is rumble / DC
ONSET_LAGS = (2, 10)    # frames over which a rise is measured: sharp hits, and slower attacks such as engine starts
REFINE_CTX_S = 1.0      # half-width of the onset-curve segment cross-correlated per matched landmark
REFINE_LAGS = 3         # lag search (frames) around the voted offset during refinement
WEAK_RATIO = 2.0        # vote peak / runner-up below this is reported as a weak match
WEAK_INLIERS = 10       # fewer inlier landmarks than this is reported as a weak match
MAX_PEAKS = 10          # vote peaks examined for parts with their own offset (cuts, inserted scenes)
MIN_PEAK_VOTES = 4      # smoothed votes a peak beyond the strongest needs to be examined
PART_MIN_MATCHES = 4    # matches a part needs; shorter runs at another offset are chance matches
# Playback speeds of File2 relative to File1 tried in landmarks mode: PAL speed-up (25 fps from 24 / 23.976 film)
# and NTSC slow-down (23.976 from 24), each way. Frame rates alone do not tell: telecine keeps the speed.
SPEEDS = [(1.0, "1"), (25/24, "25/24"), (24/25, "24/25"), (25*1001/24000, "25/23.976"), (24000/1001/25, "23.976/25"),
          (1001/1000, "24/23.976"), (1000/1001, "23.976/24")]
# A part is dropped when a neighbouring part's offset has at least this share of its match count within its span:
# the stretch fits both (e.g. music only) and continuity wins; after a real cut the old offset hardly matches
PART_RIVAL_SHARE = 0.5
SPEED_TOL = 5e-4        # relative difference below which a frame-rate ratio and a speed are taken as equal
PART_MIN_SHARE = 0.25   # share of its offset's matches a part needs; offsets that line up periodic content
                        # (steady musical tempo) match a little everywhere, a real part concentrates its matches


# ---------------- Sources ----------------
def probe_source(path, track=None):
    """Return dict(path, track, n_tracks, language, codec, sample_rate, layout, frame_rate, origin, format_start) for
    one audio track of path: `track` (1-based), else the track flagged default, else the first; `track` in the
    result is 0-based among the audio tracks. frame_rate is None without video. origin is the native time analysed
    as 0: the video start (so the offset is relative to each file's picture), else the track's start, not below 0
    (codec priming before 0 is not content)."""
    res = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFprobe failed on {path}: {res.stderr.strip()}")
    probe = json.loads(res.stdout)
    streams = probe.get("streams", [])
    audios = [st for st in streams if st.get("codec_type") == "audio"]
    if not audios:
        raise RuntimeError(f"No audio stream in {path}")
    if track is None:
        k = next((i for i, st in enumerate(audios) if st.get("disposition", {}).get("default")), 0)
    elif 1 <= track <= len(audios):
        k = track - 1
    else:
        raise RuntimeError(f"{path} has {len(audios)} audio track(s); track {track} does not exist")
    audio = audios[k]
    # Cover art is reported as a video stream; it carries no frame rate
    video = next((st for st in streams if st.get("codec_type") == "video"
                  and not st.get("disposition", {}).get("attached_pic")), None)
    frame_rate = None
    if video is not None:
        for key in ("avg_frame_rate", "r_frame_rate"):
            rate = video.get(key, "0/0")
            if rate not in ("0/0", "0/1"):
                frame_rate = Fraction(rate)
                break
    start = lambda st: float(st.get("start_time", 0.0))
    origin = start(video) if video is not None else max(start(audio), 0.0)
    return {"path": path, "track": k, "n_tracks": len(audios), "language": audio.get("tags", {}).get("language", ""),
            "codec": audio.get("codec_name", "?"), "sample_rate": int(audio["sample_rate"]),
            "layout": audio.get("channel_layout", ""), "frame_rate": frame_rate, "origin": origin,
            "format_start": start(probe.get("format", {}))}

def check_rates(info1, info2):
    """Fail when the sources' video frame rates differ: the offset would drift (e.g. PAL 25 vs film 23.976)."""
    fr1, fr2 = info1["frame_rate"], info2["frame_rate"]
    if fr1 is not None and fr2 is not None:
        if abs(fr1 - fr2) / max(fr1, fr2) > FRAME_RATE_TOL:
            raise RuntimeError(f"Frame rates differ ({float(fr1):.3f} vs {float(fr2):.3f} fps): "
                               "a constant offset does not apply")
    else:
        print("Warning: frame rate not available for both files; rate match is not verified")

def downmix_filter(layout):
    """pan filter to mono without the centre channel; None (plain downmix) when the layout has no known centre."""
    names = LAYOUT_CHANNELS.get(layout, "").split()
    if args.keep_centre or "FC" not in names:
        return None
    return "pan=mono|c0<" + "+".join(n for n in names if n != "FC")

def describe(name, info, mix):
    fps = f"{float(info['frame_rate']):.3f} fps" if info["frame_rate"] is not None else "no video"
    centre = "centre dropped" if mix else "plain downmix"
    track = ""
    if info["n_tracks"] > 1:
        tag = ", ".join(t for t in (info["language"], info["codec"]) if t)
        track = f"track {info['track'] + 1} of {info['n_tracks']} ({tag}), "
    print(f"{name}: {track}{info['sample_rate']} Hz, {info['layout'] or 'unknown layout'} ({centre}), {fps}")

def decode_cmd(src, start, duration=None):
    """ffmpeg command: the source's audio track from start (s after its origin) -> mono float32 at args.sr on stdout.
    Timestamps stay native (-copyts) and a second aresample pins output sample 0 to origin + start: codec priming
    before it is trimmed and a seek landing on a later packet is padded, so the output is sample-accurate on the
    file's timeline. The rate is converted first because first_pts counts samples at aresample's input rate."""
    t0 = src["origin"] + start
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-ss", f"{max(t0 - src['format_start'], 0.0):.6f}"]
    if duration is not None:
        cmd += ["-t", str(duration)]
    filters = ([src["mix"]] if src["mix"] else []) + [f"aresample={args.sr}",
                                                      f"aresample=async=1:first_pts={round(t0 * args.sr)}"]
    return cmd + ["-i", src["path"], "-map", f"0:a:{src['track']}", "-af", ",".join(filters),
                  "-ac", "1", "-ar", str(args.sr), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]

def decode_window(src, start=0.0, duration=None):
    """Decode a window [start, start+duration) of a source to mono float32 at args.sr."""
    duration = duration or args.win_dur
    res = subprocess.run(decode_cmd(src, float(start), float(duration)), capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg decode failed: {res.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(res.stdout, dtype=np.float32)

def stream_band_power(src, bands):
    """Decode a source from args.offset to its end in chunks; return band power per frame (frames x bands)."""
    proc = subprocess.Popen(decode_cmd(src, args.offset * 60.0), stdout=subprocess.PIPE)
    chunk_bytes = 4 * args.sr * 30
    buf = np.zeros(0, dtype=np.float32)
    out = []
    while True:
        data = proc.stdout.read(chunk_bytes)
        if not data:
            break
        buf = np.concatenate([buf, np.frombuffer(data, dtype=np.float32)])
        n_frames = 1 + (len(buf) - args.frame) // args.hop
        if n_frames > 0:
            out.append(power_spectrum(buf[:(n_frames - 1)*args.hop + args.frame]) @ bands)
            buf = buf[n_frames*args.hop:]
    if proc.wait() != 0:
        raise RuntimeError(f"FFmpeg decode failed on {src['path']}")
    if not out:
        raise RuntimeError(f"No audio decoded from {src['path']}")
    return np.concatenate(out)

# ---------------- Spectral features ----------------
def power_spectrum(x):
    """Hann-windowed one-sided power per frame (args.frame / args.hop); bins sum to the frame's mean square."""
    if x.size < args.frame:
        x = np.pad(x, (0, args.frame - x.size))
    frames = np.lib.stride_tricks.sliding_window_view(x, args.frame)[::args.hop]
    w = np.hanning(args.frame)
    p = np.abs(np.fft.rfft(frames * w, axis=1))**2
    p[:, 1:(args.frame + 1)//2] *= 2  # all bins but DC and (even frame) Nyquist appear twice in the full spectrum
    return p / (args.frame * np.sum(w**2))

def bin_freqs():
    return np.fft.rfftfreq(args.frame, 1.0/args.sr)

def nonspeech_bins():
    """Mask of spectrum bins outside the speech band (all bins with --full-band)."""
    f = bin_freqs()
    if args.full_band:
        return np.ones(f.shape, dtype=bool)
    lo, hi = args.speech_band
    return (f < lo) | (f > hi)

def rms_envelope(x):
    """RMS envelope of the non-speech band, per frame (args.frame / args.hop)"""
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(power_spectrum(x)[:, nonspeech_bins()].sum(axis=1)).astype(np.float32)

def band_matrix():
    """Bin-to-band membership (bins x bands): three log-spaced landmark bands each side of the speech band."""
    f = bin_freqs()
    nyq = args.sr / 2.0
    if args.full_band:
        edges = [np.geomspace(LM_LOW_HZ, nyq, 7)]
    else:
        lo, hi = args.speech_band
        edges = [np.geomspace(LM_LOW_HZ, lo, 4), np.geomspace(hi, nyq, 4)]
    bands = [(f >= a) & (f < b) for e in edges for a, b in zip(e[:-1], e[1:])]
    return np.array([b for b in bands if b.any()], dtype=np.float64).T

# ---------------- Cross-Correlation ----------------
def refine_peak_parabola(corr, idx):
    """Sub-sample parabolic interpolation"""
    if idx<=0 or idx>=len(corr)-1:
        return 0.0
    c_m1, c0, c_p1 = float(corr[idx-1]), float(corr[idx]), float(corr[idx+1])
    denom = c_m1 - 2*c0 + c_p1
    return 0.5*(c_m1 - c_p1)/denom if denom!=0 else 0.0

def cross_correlation_offset_ms(env1, env2):
    """Returns offset in ms (float) and seconds (float)"""
    env1n = (env1 - env1.mean()) / (env1.std()+1e-12)
    env2n = (env2 - env2.mean()) / (env2.std()+1e-12)
    corr = np.correlate(env1n, env2n, mode="full")
    idx = int(np.argmax(corr))
    lags = np.arange(-len(env2)+1, len(env1))
    lag_bin = int(lags[idx])
    delta = refine_peak_parabola(corr, idx)
    lag_refined = lag_bin + delta
    offset_s = lag_refined * args.hop / float(args.sr)
    return offset_s*1000.0, offset_s  # ms, s

def offset_from_windows(src1, src2):
    """ Go over the file in windows offsets calculating offset"""
    offsets_ms = []
    first_envs = None
    start_first = None
    count_win = args.n_win
    i = -1
    while count_win != 0:
        i += 1
        start = args.offset * 60 + i*args.spacing
        try:
            x = decode_window(src1, start=start)
            y = decode_window(src2, start=start)
        except RuntimeError:
            break
        if x.size==0 or y.size==0:
            break
        env1, env2 = rms_envelope(x), rms_envelope(y)
        ''' Try to identify artifact of cross-correlation with short or mismatched windows compute mean RMS for this window
            Mostly silent / low-energy windows or Very different content between the files.
        '''
        rms1 = np.sqrt(np.mean(env1**2))
        rms2 = np.sqrt(np.mean(env2**2))
        # skip window if either track is below threshold
        if rms1 < args.silence_thresh or rms2 < args.silence_thresh:
            print(f"Skipping window {i} due to low energy ({rms1:.4f}, {rms2:.4f}), cross {cross_correlation_offset_ms(env1, env2)[0]:+.2f}ms")
            continue
        off_ms, off_s = cross_correlation_offset_ms(env1, env2)
        if abs(off_ms) > args.max_lag:
            print(f"Skipping window {i} due to excessive lag: {off_ms:+.2f} ms")
            continue
        count_win -= 1
        offsets_ms.append(off_ms)
        if first_envs is None:
            first_envs = (env1, env2)
            start_first = start
    if len(offsets_ms)==0:
        raise RuntimeError("No usable windows (silent, beyond --max-lag, or past end of file)")
    return offsets_ms, first_envs, start_first

# ---------------- Landmarks ----------------
def onset_strength(bp):
    """Per-band rise of log power, largest over ONSET_LAGS (frames x bands); levels relative to each band's median."""
    level = np.log10(bp / (np.median(bp, axis=0) + 1e-20) + 1.0)
    rise = np.zeros_like(level)
    for lag in ONSET_LAGS:
        rise[lag:] = np.maximum(rise[lag:], level[lag:] - level[:-lag])
    return rise

def detect_landmarks(bp):
    """Sharp onsets, strongest first, at least args.lm_gap apart, at most args.lm_per_min per minute.
    Returns (times in frames, sorted, sub-frame; signature per landmark: unit-norm band rise; onset curve)."""
    rise = onset_strength(bp)
    o = rise.sum(axis=1)
    gap = max(1, int(round(args.lm_gap * args.sr / args.hop)))
    frames_per_min = int(60 * args.sr / args.hop)
    blocked = np.zeros(len(o), dtype=bool)
    per_min = {}
    picked = []
    cand = np.flatnonzero(o >= args.lm_min_rise)
    for idx in cand[np.argsort(-o[cand], kind="stable")]:
        minute = idx // frames_per_min
        if blocked[idx] or per_min.get(minute, 0) >= args.lm_per_min:
            continue
        blocked[max(0, idx-gap):idx+gap+1] = True
        per_min[minute] = per_min.get(minute, 0) + 1
        picked.append(idx)
    picked = np.sort(np.array(picked, dtype=int))
    times = np.array([i + refine_peak_parabola(o, i) for i in picked])
    sig = rise[picked]
    return times, sig / (np.linalg.norm(sig, axis=1, keepdims=True) + 1e-12), o

def similar_pairs(sig1, sig2):
    """Index arrays (i, j) of landmark pairs whose signatures' cosine similarity is at least args.lm_min_sim."""
    I, J = [], []
    for start in range(0, len(sig1), 512):  # bounds the similarity matrix size
        i, j = np.nonzero(sig1[start:start+512] @ sig2.T >= args.lm_min_sim)
        I.append(start + i)
        J.append(j)
    return (np.concatenate(I), np.concatenate(J)) if I else (np.zeros(0, int), np.zeros(0, int))

def speed_name(speed):
    return next((n for s, n in SPEEDS if s == speed), f"{speed:.6f}")

def choose_speed(t1, t2, pairs):
    """(speed, votes): the speed of SPEEDS (or --speed) whose offset votes (frames, file1 - speed x file2) form the
    highest histogram peak; at a wrong speed the votes of one stretch smear over many offsets."""
    best = None
    for speed in ([args.speed] if args.speed is not None else [s for s, _ in SPEEDS]):
        votes = t1[pairs[0]] - speed * t2[pairs[1]]
        try:
            peak = vote_histogram(votes)[1].max()
        except RuntimeError:
            peak = 0.0
        if best is None or peak > best[0]:
            best = (peak, speed, votes)
    return best[1], best[2]

def scale_track(times, o, speed):
    """File2 landmark times and onset curve on File1's time scale: times x speed, curve resampled."""
    n = int(np.floor((len(o) - 1) * speed)) + 1
    return times * speed, np.interp(np.arange(n) / speed, np.arange(len(o)), o)

def retime_filter(speed, sr, undo_pitch):
    """ffmpeg filter bringing audio that runs `speed` x to the reference speed: atempo keeps the pitch; asetrate shifts
    it with the speed (undoing a PAL pitch rise) but takes whole sample rates, so a residual atempo makes it exact."""
    if not undo_pitch:
        return f"atempo={1/speed:.9f}"
    r = int(math.floor(sr / speed + 0.5))
    return f"asetrate={r},aresample={sr},atempo={sr / (r * speed):.9f}"

def rate_note(src1, src2, speed):
    """Note when the frame-rate ratio and the measured speed disagree; None otherwise or without video."""
    fr1, fr2 = src1["frame_rate"], src2["frame_rate"]
    if fr1 is None or fr2 is None:
        return None
    ratio = float(fr2 / fr1)
    if abs(speed - ratio) / ratio <= SPEED_TOL:
        return None
    if args.speed is not None:
        return f"speed set to {speed_name(speed)}, not measured; the frame-rate ratio is {ratio:.6f}"
    if speed == 1.0:
        return (f"frame rates differ ({float(fr1):.3f} vs {float(fr2):.3f} fps) but the audio runs at the same speed "
                "(telecine or standards conversion)")
    return f"audio speed {speed:.6f} does not match the frame-rate ratio {ratio:.6f} ({float(fr2):.3f} / {float(fr1):.3f} fps)"

def vote_histogram(votes):
    """Vote histogram within args.max_lag, summed over 3 bins to absorb +-1 frame jitter: (offset of bin 0, counts)."""
    if args.max_lag is not None:
        votes = votes[np.abs(votes) <= args.max_lag/1000.0 * args.sr / args.hop]
    if votes.size == 0:
        raise RuntimeError("No landmark pairs matched")
    v = np.round(votes).astype(int)
    lo = v.min()
    return lo, np.convolve(np.bincount(v - lo).astype(float), np.ones(3), mode="same")

def clear_around(hist, k, gap):
    hist[max(0, k-gap):k+gap+1] = 0

def vote_peaks(lo, hist):
    """Histogram peaks, strongest first, at least lm_gap apart: [(offset frames, votes)]."""
    gap = max(1, int(round(args.lm_gap * args.sr / args.hop)))
    rest = hist.copy()
    peaks = []
    while len(peaks) < MAX_PEAKS:
        k = int(np.argmax(rest))
        if peaks and rest[k] < MIN_PEAK_VOTES:
            break
        peaks.append((k + lo, rest[k]))
        clear_around(rest, k, gap)
    return peaks

def runner_up(lo, hist, offsets):
    """Strongest vote count away from the given offsets (frames)."""
    gap = max(1, int(round(args.lm_gap * args.sr / args.hop)))
    rest = hist.copy()
    for off in offsets:
        clear_around(rest, off - lo, gap)
    return rest.max()

def refine_matches(lm1, lm2, o1, o2, coarse, label):
    """Per landmark of file1 with a file2 landmark at offset coarse (+-2 frames): (file1 time s, offset ms, label),
    from cross-correlating the onset curves around it."""
    t2 = lm2[0]
    W = int(round(REFINE_CTX_S * args.sr / args.hop))
    lags = np.arange(coarse - REFINE_LAGS, coarse + REFINE_LAGS + 1)
    out = []
    for a in lm1[0]:
        j = np.searchsorted(t2, a - coarse)
        if not any(abs(a - coarse - t) <= 2 for t in t2[max(0, j-1):j+1]):
            continue
        c = int(round(a))
        if c - W < 0 or c + W > len(o1) or c - lags[-1] - W < 0 or c - lags[0] + W > len(o2):
            continue
        seg1 = o1[c-W:c+W] - o1[c-W:c+W].mean()
        corr = np.array([np.dot(seg1, o2[c-k-W:c-k+W] - o2[c-k-W:c-k+W].mean()) for k in lags])
        idx = int(np.argmax(corr))
        if idx in (0, len(lags)-1):  # peak at the search edge: no reliable local match
            continue
        lag = lags[idx] + refine_peak_parabola(corr, idx)
        out.append((args.offset*60 + a*args.hop/args.sr, lag*args.hop/args.sr*1000.0, label))
    return np.array(out).reshape(-1, 3)

def find_parts(matches):
    """Splits matches (rows: file1 time, offset ms, peak label) into parts: runs in file1 time sharing a peak.
    Dropped one at a time, so a part interrupted by stray matches joins up again: runs shorter than
    PART_MIN_MATCHES (shortest and least supported first), then runs holding less than PART_MIN_SHARE of their
    offset's matches (smallest share first), then runs a neighbour's offset explains about as well (PART_RIVAL_SHARE;
    shortest first). Returns the parts' rows in time order."""
    m = matches[np.lexsort((matches[:,2], matches[:,0]))]
    support = {lab: int(np.sum(m[:,2] == lab)) for lab in np.unique(m[:,2])}
    keep = np.ones(len(m), dtype=bool)
    while True:
        idx = np.flatnonzero(keep)
        if idx.size == 0:
            return []
        labels = m[idx, 2]
        bounds = [0] + [i for i in range(1, len(idx)) if labels[i] != labels[i-1]] + [len(idx)]
        runs = list(zip(bounds[:-1], bounds[1:]))
        short = [(e - s, support[labels[s]], s, e) for s, e in runs if e - s < PART_MIN_MATCHES]
        scattered = [((e - s) / support[labels[s]], s, e) for s, e in runs
                     if e - s < PART_MIN_SHARE * support[labels[s]]]
        if short:
            _, _, s, e = min(short)
        elif scattered:
            _, s, e = min(scattered)
        else:
            contested = []
            for r, (s, e) in enumerate(runs):
                inside = (m[:,0] >= m[idx[s], 0]) & (m[:,0] <= m[idx[e-1], 0])
                rivals = {labels[runs[q][0]] for q in (r - 1, r + 1) if 0 <= q < len(runs)} - {labels[s]}
                if max((np.sum(inside & (m[:,2] == lab)) for lab in rivals), default=0) >= PART_RIVAL_SHARE * (e - s):
                    contested.append((e - s, s, e))
            if not contested:
                return [m[idx[s:e]] for s, e in runs]
            _, s, e = min(contested)
        keep[idx[s:e]] = False

def landmark_offset(src1, src2):
    """Returns (speed, votes, histogram (lo, counts), peaks [(offset frames, votes)],
    parts [rows: time s, offset ms, label]); offsets apply to File2 retimed by speed (see choose_speed).
    Without any part of PART_MIN_MATCHES, the strongest peak's matches form the only part."""
    bands = band_matrix()
    lms, curves = [], []
    for name, src in (("File1", src1), ("File2", src2)):
        bp = stream_band_power(src, bands)
        times, sig, o = detect_landmarks(bp)
        print(f"{name}: {len(times)} landmarks over {len(o)*args.hop/args.sr/60:.1f} min")
        lms.append((times, sig))
        curves.append(o)
    speed, votes = choose_speed(lms[0][0], lms[1][0], similar_pairs(lms[0][1], lms[1][1]))
    times2, o2 = scale_track(lms[1][0], curves[1], speed)
    lm2 = (times2, lms[1][1])
    lo, hist = vote_histogram(votes)
    peaks = vote_peaks(lo, hist)
    matches = np.concatenate([refine_matches(lms[0], lm2, curves[0], o2, coarse, label)
                              for label, (coarse, _) in enumerate(peaks)])
    if not np.any(matches[:,2] == 0):
        raise RuntimeError("No landmark matched at the voted offset")
    parts = find_parts(matches) or [matches[matches[:,2] == 0]]
    return speed, votes, (lo, hist), peaks, parts

# ---------------- Plot ----------------
def plot_alignment(env1, env2, offset_s, title="Window alignment"):
    t1 = np.arange(len(env1))*args.hop/float(args.sr)
    t2 = np.arange(len(env2))*args.hop/float(args.sr)+offset_s
    plt.figure(figsize=(12,4))
    plt.plot(t1, env1, label="File1")
    plt.plot(t2, env2, label="File2 (shifted)")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS energy")
    plt.title(f"{title} (offset={offset_s*1000.0:+.2f} ms)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def fmt_time(s):
    """h:mm:ss.s or m:ss.s"""
    sign, s = ("-" if s < 0 else ""), math.floor(abs(s) * 10 + 0.5) / 10
    h, m, sec = int(s // 3600), int(s % 3600 // 60), s % 60
    return f"{sign}{h}:{m:02d}:{sec:04.1f}" if h else f"{sign}{m}:{sec:04.1f}"

def plot_landmarks(votes, coarse, parts, median_ms):
    frame_ms = args.hop/float(args.sr)*1000.0
    medians = [np.median(p[:,1]) for p in parts]
    near = votes[np.abs(votes - coarse) * frame_ms <= 5000.0] * frame_ms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.hist(near, bins=200)
    ax1.set_xlabel("Offset (ms)")
    ax1.set_ylabel("Votes")
    ax1.set_title("Votes within 5 s of the peak")
    for p in parts:
        ax2.plot(p[:,0]/60.0, p[:,1], ".")
    ax2.set_ylim(min(medians) - 5*frame_ms, max(medians) + 5*frame_ms)
    ax2.set_xlabel("File1 time (min)")
    ax2.set_ylabel("Offset (ms)")
    ax2.set_title(f"Matched landmarks (median {median_ms:+.2f} ms)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audio alignment of two versions (e.g. language dubs), in ms")
    p.add_argument("file1")
    p.add_argument("file2")
    p.add_argument("--track1", type=int, default=None,
                   help="audio track of file1 (1-based; default: the track flagged default, else the first)")
    p.add_argument("--track2", type=int, default=None, help="audio track of file2 (as --track1)")
    p.add_argument("--mode", choices=("envelope", "landmarks"), default="envelope",
                   help="envelope: windowed envelope cross-correlation; landmarks: whole-file onset matching")
    p.add_argument("--sr", type=int, default=16000, help="analysis sample rate (Hz)")
    p.add_argument("--frame", type=int, default=512, help="analysis frame size (samples)")
    p.add_argument("--hop", type=int, default=256, help="analysis hop size (samples)")
    p.add_argument("--speech-band", type=float, nargs=2, default=(300.0, 3500.0), metavar=("LO", "HI"),
                   help="band (Hz) excluded from analysis as dialogue")
    p.add_argument("--full-band", action="store_true", help="analyse all bands, speech included")
    p.add_argument("--keep-centre", action="store_true", help="keep the centre channel of surround sources")
    p.add_argument("--offset", type=int, default=0, help="start offset in min")
    p.add_argument("--max-lag", type=float, default=None,
                   help="Maximum allowed lag in ms (default: 10000 in envelope mode, unlimited in landmarks mode)")
    p.add_argument("--win-dur", type=float, default=60.0, help="envelope: window duration (s)")
    p.add_argument("--spacing", type=float, default=300.0, help="envelope: spacing between windows (s)")
    p.add_argument("--n-win", type=int, default=10, help="envelope: max number of accepted windows (silent / excessive-lag windows are not counted)")
    p.add_argument("--silence-thresh", type=float, default=0.01,
                   help="envelope: skip windows whose mean non-speech RMS is below this absolute level (full scale = 1.0)")
    p.add_argument("--lm-per-min", type=int, default=20, help="landmarks: max landmarks per minute (strongest kept)")
    p.add_argument("--lm-gap", type=float, default=0.25, help="landmarks: min spacing between landmarks (s)")
    p.add_argument("--lm-min-rise", type=float, default=1.0,
                   help="landmarks: min onset strength (sum over bands of log10 power rise; 1.0 = 10 dB in one band)")
    p.add_argument("--lm-min-sim", type=float, default=0.8,
                   help="landmarks: min cosine similarity of band signatures for a match to vote")
    p.add_argument("--speed", default="auto",
                   help="landmarks: File2 playback speed relative to File1, 'auto' (tries PAL / NTSC ratios) or a "
                        "ratio such as 25/23.976 or 1.0427")
    p.add_argument("--plot", action="store_true", help="plot the first window alignment (envelope) or votes and matches (landmarks)")
    args = p.parse_args()  # <-- global

    if not args.full_band and not LM_LOW_HZ < args.speech_band[0] < args.speech_band[1] < args.sr/2:
        p.error(f"--speech-band must satisfy {LM_LOW_HZ:.0f} < LO < HI < sr/2")
    if args.max_lag is None and args.mode == "envelope":
        args.max_lag = 10000.0
    if args.speed == "auto":
        args.speed = None
    else:
        try:
            named = {n: v for v, n in SPEEDS}  # exact values: 23.976 stands for 24000/1001
            num, _, den = args.speed.partition("/")
            args.speed = named[args.speed] if args.speed in named else float(num) / float(den or 1)
        except ValueError:
            p.error(f"--speed: '{args.speed}' is not 'auto' or a ratio")
        if not 0.5 <= args.speed <= 2.0:
            p.error("--speed must lie between 0.5 and 2")

    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        sys.exit(f"{' and '.join(missing)} not found in PATH")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("--plot requires matplotlib. Try `pip install matplotlib`")

    try:
        src1, src2 = (probe_source(f, t) for f, t in ((args.file1, args.track1), (args.file2, args.track2)))
        for name, src in (("File1", src1), ("File2", src2)):
            src["mix"] = downmix_filter(src["layout"])
            describe(name, src, src["mix"])
        if os.path.realpath(args.file1) == os.path.realpath(args.file2) and src1["track"] == src2["track"]:
            print("Warning: both inputs are the same audio track")
        if args.mode == "landmarks":
            speed, votes, (lo, hist), peaks, parts = landmark_offset(src1, src2)
        else:
            check_rates(src1, src2)
            offsets_ms, first_envs, start_first = offset_from_windows(src1, src2)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")

    step_ms = args.hop/float(args.sr)*1000.0
    if args.mode == "landmarks":
        main = max(range(len(parts)), key=lambda i: (len(parts[i]), -i))  # most matches, earliest on ties
        matches = parts[main]
        coarse, peak = peaks[int(matches[0,2])]
        runner = runner_up(lo, hist, [peaks[int(p[0,2])][0] for p in parts])
        offs = matches[:,1]
        median_ms = float(np.median(offs))
        inliers = matches[np.abs(offs - median_ms) <= step_ms]
        ratio = peak / runner if runner > 0 else float("inf")
        print(f"Vote peak: {peak:.0f} votes at {coarse*step_ms:+.0f} ms, {ratio:.1f}x the runner-up")
        print(f"Matched landmarks: {len(matches)} ({len(inliers)} within +-{step_ms:.0f} ms of the median)")
        print(f"Estimated global offset (median, p10, p90): {median_ms:+.2f}ms, "
              f"{np.percentile(offs, 10):+.2f}ms, {np.percentile(offs, 90):+.2f}ms")
        fit = None
        if len(inliers) >= 4:
            fit, cov = np.polyfit(inliers[:,0], inliers[:,1], 1, cov=True)  # slope in ms per s
            print(f"Drift across inliers: {fit[0]*3600:+.2f} +- {np.sqrt(cov[0,0])*3600:.2f} ms/hour")
        if ratio < WEAK_RATIO or len(inliers) < WEAK_INLIERS:
            print("Warning: weak match (few landmarks or ambiguous vote peak)")
        if speed != 1.0:
            line = f"Speed: File2 runs {speed:.6f}x File1 ({speed_name(speed)})"
            if fit is not None:  # residual drift refines the speed: t1 = speed (1 + slope) t2 + offset
                line += f", measured {speed * (1 + fit[0]/1000):.6f} +- {speed * np.sqrt(cov[0,0])/1000:.6f}"
            print(line + "; offsets apply to File2 retimed to File1's speed")
            sr2 = src2["sample_rate"]
            print(f"Retime File2: {retime_filter(speed, sr2, False)} (keeps pitch) or "
                  f"{retime_filter(speed, sr2, True)} (also undoes a pitch shift)")
        note = rate_note(src1, src2, speed)
        if note:
            print(f"Note: {note}")
        if len(parts) > 1:
            print(f"Parts with their own offset: {len(parts)} (lines above: the largest); File1 time, offset, landmarks")
            prev = None
            for p in parts:
                med = float(np.median(p[:,1]))
                line = f"  {fmt_time(p[0,0])} - {fmt_time(p[-1,0])}  {med:+.2f}ms  {len(p)}"
                if prev is not None:
                    d = med - prev[0]
                    where = (f"at {fmt_time(p[0,0])}" if prev[1] >= p[0,0]
                             else f"between {fmt_time(prev[1])} and {fmt_time(p[0,0])}")
                    line += f"  ({abs(d):.1f} ms {'missing in' if d > 0 else 'extra in'} File2 {where})"
                print(line)
                prev = (med, p[-1,0])
        if args.plot:
            plot_landmarks(votes, coarse, parts, median_ms)
    else:
        median_ms = np.median(offsets_ms)
        min_ms = np.min(offsets_ms)
        max_ms = np.max(offsets_ms)
        print(f"Estimated global offset (median, min, max): {median_ms:+.2f}ms, {min_ms:+.2f}ms, {max_ms:+.2f}ms")
        print("Offsets per window (ms):", [f"{o:+.2f}" for o in offsets_ms])
        print(f"(Raw resolution = hop/sr = {step_ms:.3f} ms; sub-sample interpolation applied)")
        if args.plot and first_envs is not None:
            env1, env2 = first_envs
            plot_alignment(env1, env2, offsets_ms[0]/1000.0, title=f"First window @ {start_first:.1f}s")
