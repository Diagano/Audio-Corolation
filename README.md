Basic audio file offset matching, using cross corellation over windowed RMS energy.

Steps:
1. Decode both files to raw PCM (Using ffmpeg-python, ffmpeg required in Path)
2. Convert to mono and normalize (removes channel/gain differences)
3. Compute a simple envelope (RMS energy per short window)
4. Cross-correlate the two envelopes to find the lag (offset)
5. Report alignment offset median/min/max in ms
6. Optionally plot the first envelope window (with offset applied)
