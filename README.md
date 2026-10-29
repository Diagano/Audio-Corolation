Audio offset matching between two versions of the same programme, e.g. language dubs of a film.

Two implementations of the same landmark analysis, with identical output:
- `audio-offset` (Rust): GUI with file drop, options, alignment plots and A/B listening; `audio-offset-cli` for scripts.
  A single executable each; ffmpeg / ffprobe are used from PATH or a configured folder.
- `audio_offset.py`: reference and experimentation bench, also with the windowed envelope mode.

`examples/` holds synthetic test media with known offsets (cuts, speed changes, several tracks) and expected results.

Dubs share the music & effects (M&E) track and differ in dialogue, so analysis runs on what dialogue does not occupy:
- the centre channel of surround sources is dropped (`--keep-centre` keeps it);
- the speech band, 300–3500 Hz by default (`--speech-band LO HI`), is excluded (`--full-band` includes it).

Requires ffmpeg / ffprobe in PATH and numpy (matplotlib for `--plot`); `docker/` has a container with all of them.

Speed (landmarks mode): File 2 may play faster or slower than File 1 — PAL releases run 25/24 or 25/23.976 fast,
24 fps masters 24/23.976 against 23.976. Votes are collected with File 2's landmark times scaled by each of 1,
25/24, 25/23.976, 24/23.976 and their inverses; the speed with the highest vote peak wins (at a wrong speed one
stretch's votes smear over many offsets), and the residual drift refines it ("measured"). Offsets and parts then
apply to File 2 retimed to File 1's speed, and the report gives ffmpeg filters for that retiming (`atempo` keeps the
pitch, `asetrate` + `aresample` also undoes a PAL pitch rise); mkvmerge's stretch rescales timestamps only, so the
audio itself has to be re-encoded. `--speed` / the Speed option fixes the ratio instead. Frame rates do not decide
the speed (telecine and standards conversion keep it); a note reports when they disagree with it.
Envelope mode has no speed model: it aborts when the video frame rates differ.
Audio track: one per file takes part, `--track1` / `--track2` (1-based) in the scripts, a selector per file in the GUI;
default is the track flagged default, else the first. Both inputs may be the same file with different tracks (two
languages in one file); the GUI fills File 2 that way when a multi-track file goes into File 1 with File 2 empty.

Offsets are file1 − file2: positive means an event occurs later in file1. Each file is measured on its own timestamps
from its video start (audio start without video), sample-accurate: codec priming (e.g. AC-3) is trimmed and seeks
are exact, so the offset applies directly when muxing one file's audio against the other's picture.

## `--mode landmarks`
Matches background events (gunshots, door slams, engine starts) as fixed points, like terrain features on a map.
1. Decode each file once (streamed), compute power in three log-spaced bands below and three above the speech band.
2. Landmarks: frames where those bands rise sharply (over 32 ms and over 160 ms), relative to each band's median
   level, so gain differences between mixes cancel. Strongest first, at least `--lm-gap` apart, at most
   `--lm-per-min` per minute. Each landmark's signature is its per-band rise.
3. Every landmark pair across the files with signature similarity ≥ `--lm-min-sim` votes for its time difference.
   Dialogue onsets vote at random; shared M&E events agree on one offset, which forms the histogram peak.
4. Each landmark matched at that offset is refined by cross-correlating the onset curves around it (sub-frame).
5. Parts: versions with cuts or inserted scenes have one offset per stretch. The strongest vote peaks (up to 10) are
   refined as well; matches sorted by File 1 time form runs per offset, and a run is reported as a part when it has
   at least 4 matches and at least a quarter of its offset's matches. Offsets that line up periodic content (steady
   musical tempo) match a little everywhere and fail the second test. Each boundary is reported with the size of
   the change: "N ms missing in File2" (offset grows) or "extra in File2" (offset shrinks).

Reports, for the largest part, the median offset, its 10–90 % spread, the vote peak's margin over the strongest peak
no part explains, and the drift (slope of offset over time) with its standard error; then the parts, if several. A margin below 2× or fewer than 10 consistent landmarks is flagged as weak.
Search range is unlimited unless `--max-lag` is given.

## `--mode envelope` (default)
Cross-correlates non-speech RMS envelopes in `--n-win` windows of `--win-dur` s every `--spacing` s, skipping windows
below `--silence-thresh` (absolute RMS, full scale 1.0) or beyond `--max-lag` (default 10 s). Reports median/min/max.
Fast, but every window must be dominated by shared content; with dialogue-heavy material use landmarks.

## Limits
- Within a part the offset is constant; a part boundary is located to the spacing of the landmarks around it, and
  differing stretches with fewer than 4 matched landmarks are not resolved.
- Dialogue leaking outside the speech band (voice fundamentals, fricatives) and stereo-only dubs, where dialogue cannot
  be removed by channel, add random landmarks; they lower the vote margin but do not shift the peak.
- Precision is set by `--hop` / `--sr` (16 ms frames by default) and sub-frame interpolation; typically a few ms.

## Rust build
In the toolchain container (`docker/build.Dockerfile`), from the repository root:
```
docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/src" -v audio-offset-cargo:/cargo audio-offset-build \
    cargo build --release --target x86_64-pc-windows-gnu
```
Output: `target/x86_64-pc-windows-gnu/release/audio-offset.exe` (GUI) and `audio-offset-cli.exe`, statically linked
against the C runtime. `--no-default-features` builds only the CLI and library, without the GUI dependencies.

GUI usage: `audio-offset [FILE1 [FILE2]] [--analyse]`, or drop files on the window or the program icon.
Click the lanes to place the cursor; ▶ Both plays File 1 in the left ear and File 2 in the right.
When the speeds differ, two buttons copy a ready ffmpeg command that retimes File 2's selected track ("Keep pitch":
atempo; "Undo pitch shift": asetrate) into `<File 2 name>.retimed.mka` next to it, in the source codec at a high
bitrate or, with "Lossless", FLAC; mux that file with the reported offset as its delay.

## Config file
`audio-offset.ini`, used by the GUI and the CLI: next to the executable if present there (portable), else
`%APPDATA%\audio-offset\` (Windows) or `~/.config/audio-offset/`. It holds the rarely changed settings — ffmpeg
folder (empty: PATH), analysis sample rate / frame / hop, speech band edges, listening clip length — and the window's
last-used analysis options. The GUI writes it; each key is commented, and a removed key takes its default.
CLI options override it.
