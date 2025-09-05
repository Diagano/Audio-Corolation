#!/usr/bin/env python3
"""
Windowed audio alignment, outputs offset in ms.
"""
import math
import argparse

try:
    import ffmpeg
    import numpy as np
except ImportError as e:
    print("Missing required libs. Try `pip install ffmpeg-python librosa numpy`")
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Plotting requires matplotlib,  try pip install matplotlib")
    

def decode_window(path, start=0.0, duration=None):
    """Decode a window [start, start+duration) from path to mono float32 at args.sr."""
    duration = duration or args.win_dur
    try:
        out, _ = (
            ffmpeg
            .input(path, ss=float(start), t=float(duration))
            .output("pipe:", format="f32le", acodec="pcm_f32le", ac=1, ar=args.sr)
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg decode failed: {e.stderr.decode() if hasattr(e,'stderr') else str(e)}")
    return np.frombuffer(out, dtype=np.float32)

def rms_envelope(x):
    """Compute RMS envelope with args.frame / args.hop"""
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    n_frames = max(1, 1 + (len(x) - args.frame) // args.hop)
    env = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = x[i*args.hop:i*args.hop+args.frame]
        env[i] = math.sqrt(float(np.mean(frame.astype(np.float64)**2))) if frame.size>0 else 0.0
    return env

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

def offset_from_windows(file1, file2):
    """ Go over the file in windows offsets calculating offset"""
    offsets_ms = []
    first_envs = None
    start_first = None
    count_win = args.n_win
    i = 0
    while count_win != 0:
        i += 1
        start = args.offset * 60 + i*args.spacing
        try:
            x = decode_window(file1, start=start)
            y = decode_window(file2, start=start)
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
        count_win -= 1
        off_ms, off_s = cross_correlation_offset_ms(env1, env2)
        if abs(off_ms) > args.max_lag:
            print(f"Skipping window {i} due to excessive lag: {off_ms:+.2f} ms")
            continue
        offsets_ms.append(off_ms)
        if first_envs is None:
            first_envs = (env1, env2)
            start_first = start
    if len(offsets_ms)==0:
        raise RuntimeError("No valid windows decoded")
    return offsets_ms, first_envs, start_first

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

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Windowed audio alignment (ms)")
    p.add_argument("file1")
    p.add_argument("file2")
    p.add_argument("--sr", type=int, default=16000, help="analysis sample rate (Hz)")
    p.add_argument("--frame", type=int, default=512, help="frame size for RMS")
    p.add_argument("--hop", type=int, default=256, help="hop size for RMS")
    p.add_argument("--win-dur", type=float, default=60.0, help="window duration (s)")
    p.add_argument("--spacing", type=float, default=300.0, help="spacing between windows (s)")
    p.add_argument("--n-win", type=int, default=10, help="max number of windows")
    p.add_argument("--offset", type=int, default=0, help="start offset in min")
    p.add_argument("--silence-thresh", type=float, default=0.01, help="Skip windows with RMS below this threshold (silence? low energy?)")
    p.add_argument("--max-lag", type=float, default=10000.0, help="Maximum allowed lag in ms; windows exceeding this are skipped")
    p.add_argument("--plot", action="store_true", help="plot the first window alignment (seconds on x-axis)")
    args = p.parse_args()  # <-- global

    offsets_ms, first_envs, start_first = offset_from_windows(args.file1, args.file2)
    median_ms = np.median(offsets_ms)
    min_ms = np.min(offsets_ms)
    max_ms = np.max(offsets_ms)
    print(f"Estimated global offset (mediann, min, max): {median_ms:+.2f}ms, {min_ms:+.2f}ms, {max_ms:+.2f}ms")
    print("Offsets per window (ms):", [f"{o:+.2f}" for o in offsets_ms])
    step_ms = args.hop/float(args.sr)*1000.0
    print(f"(Raw resolution = hop/sr = {step_ms:.3f} ms; sub-sample interpolation applied)")

    if args.plot and first_envs is not None:
        env1, env2 = first_envs
        plot_alignment(env1, env2, offsets_ms[0]/1000.0, title=f"First window @ {start_first:.1f}s")
