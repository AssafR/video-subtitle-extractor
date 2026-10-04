# Sync Timeline

The **Sync Timeline** tab is a GUI front-end for the bundled
[Sushi](../../backend/sushi) tool (v0.6.0). It re-times an existing subtitle
file so that it matches a different release of the same video. It performs no
OCR and is unrelated to the frame selection described in
[candidate-subtitle-frames.md](candidate-subtitle-frames.md).

## Inputs and output

Labels come from the `[TimelineSync]` section of `backend/interface/*.ini`.

| Input | Config key | Purpose |
|-------|------------|---------|
| Source Video | `TimelineSync/SushiSourceVideo` | Video the subtitles currently match |
| Source Subtitle | `TimelineSync/SushiSourceSubtitle` | `.srt` or `.ass` file to re-time |
| Destination Video | `TimelineSync/SushiDestinationVideo` | Video to sync the timeline to |

The output is written next to the destination video, named after it, with the
source subtitle's extension (`TimelineSyncInterface.output_path`).

## GUI flow

Code: `ui/timeline_sync_interface.py`, registered in `gui.py`.

1. Three file pickers store their paths in the config.
2. `run_button_clicked` verifies that all three files exist
   (`is_selected_file_exists`) and shows an `InfoBar` error otherwise.
3. It launches `python -u -m sushi --src <src> --dst <dst> --script <sub>
   --output <out>` with `cwd=./backend/`.
4. Output streams into a read-only log box with auto-scroll. Run and Stop
   buttons toggle visibility.

## How Sushi re-times subtitles

Code: `backend/sushi/__init__.py` (`run`), `backend/sushi/wav.py`,
defaults in `backend/sushi/__main__.py`.

**In plain words:** the subtitles are used *only for their timestamps*. The
subtitle text is never read; it is copied to the output unchanged. For each
line, Sushi takes the source-video audio between the line's start and end
times (the line's "audio fingerprint"), finds where that same sound occurs in
the destination video's audio, and moves the line by the difference. The
audio match is a waveform comparison (`cv2.matchTemplate`, `TM_SQDIFF_NORMED`),
so no speech recognition is involved.

**Core idea:** the dialogue audio of two releases of the same video is
almost identical, just offset in time (different intro, cut commercials,
different frame rate). Sushi never reads the subtitle *text*. It takes the
audio that plays while each subtitle is on screen in the source video, finds
where that same sound occurs in the destination video, and moves the subtitle
by the difference.

### 1. Audio

- Both videos are demuxed to WAV (ffmpeg) and downmixed to mono.
- Each is down-sampled to `--sample-rate` (default 12000 Hz) and stored as
  `uint8` (`WavStream`).
- `get_substream(start, end)` cuts the source audio for a subtitle's time
  span (the "pattern").
- `find_substream(pattern, center, window)` slides the pattern over the
  destination audio from `center - window` to `center + window` using
  `cv2.matchTemplate` with `TM_SQDIFF_NORMED`. The best position is the
  match; the match score is the line's `diff` (0 = identical, higher = worse).
- **shift = matched destination time - original subtitle time.**

### 2. Subtitles and gaps: building search groups

Parsed with `AssScript` / `SrtScript` and sorted by start time
(`prepare_search_groups`).

Some lines are excluded from the search and *linked* to a neighbour, so they
get that neighbour's shift later:

- comments (ASS `Comment` lines),
- lines whose midpoint is past the end of the source audio,
- zero-duration lines,
- lines with the same start and end as an earlier line (duplicates, e.g.
  signs plus dialogue).

Very short audio is unreliable for matching, so
`merge_short_lines_into_groups` merges short lines into one search group:

- A line longer than `--max-ts-duration` (default 1001/24000*10, about
  0.417 s) is its own group.
- A shorter line starts a group and absorbs following short lines while the
  **gap** between the group's end and the next line's start is below
  `--max-ts-distance` (same default, about 0.417 s).
- Merging never crosses a chapter boundary, if chapters are supplied.
- A group fully contained in an earlier, larger group is linked to it.

So a *gap* larger than about 0.4 s ends a group; smaller gaps are bridged
and the whole span, including the silence between lines, is used as one
audio pattern.

### 3. Searching and committing shifts (`calculate_shifts`)

For each group in order:

1. **Fast path.** Search +/-1.5 s around `group start + last committed
   shift`. If the match lands within `ALLOWED_ERROR` (0.01 s) of the previous
   shift, commit it.
2. **Normal path.** Otherwise search +/-`--window` (default 10 s). The
   pattern is also split in half and each half searched separately. The
   result is trusted only if both halves and the whole agree within 0.01 s.
   This rejects accidental matches.
3. **Not trusted yet.** The group is held as "uncommitted" and the next group
   is tried, searching around the last uncommitted shift too. When a later
   group is trusted, its shift is applied to all held groups, and a warning
   says those lines "will most likely be broken".
4. **Recovery.** If `--rewind-thresh` (default 5) groups in a row stay
   untrusted, Sushi widens the window to `--max-window` (default 30 s) and
   restarts from the first untrusted group.
5. **Out of range.** If the expected position is past the destination audio
   end, the remaining groups get no shift and are linked to earlier lines.

### 4. Refinement

- **Border fix** (`fix_near_borders`): lines at the start or end of a group
  whose match score is more than 5x worse (or 5x better) than the median are
  treated as broken and linked to the first good line.
- **Smoothing** (`smooth_events`): a running median of the shifts with
  `--smooth-radius` (default 3 lines) removes isolated outliers.
- **Grouping** (default on): consecutive lines whose shifts differ by at most
  0.01 s form a shift group. With chapters, groups follow chapters instead,
  and a chapter whose shifts have a standard deviation over 0.025 s is split
  back into automatic groups. Each group's lines get one weighted-average
  shift (weight 1 - diff), which keeps the lines' relative timing intact.
- **Keyframe snapping** (`snap_groups_to_keyframes`): only if keyframe files
  are given. The GUI does not pass them.
- Finally `apply_shift` is applied to every event and the script is saved.

### Example

The source video has a 24 s intro that the destination release lacks.
Subtitles (source timeline):

| # | Start | End | Text |
|---|-------|-----|------|
| 1 | 60.00 | 62.00 | "Where are you going?" |
| 2 | 62.20 | 62.50 | "Home." |
| 3 | 62.60 | 62.90 | "Why?" |
| 4 | 90.00 | 93.00 | "Because I said so." |

1. Lines 2 and 3 are shorter than 0.417 s, so they are merged. The gap from
   line 2's end (62.50) to line 3's start (62.60) is 0.1 s, under 0.417 s.
   Line 1 is long enough to be its own group, so the search groups are
   `[1]`, `[2, 3]` and `[4]`.
2. Group `[1]`: the pattern is the 2 s of source audio at 60-62 s. There is
   no committed shift yet, so the fast path searches 58.5-61.5 s plus the
   pattern length in the destination and fails. The normal path searches
   +/-10 s around 60 s, so 50-70 s. It matches at 36.00 s, which is
   shift = 36.00 - 60.00 = -24.00 s. Both halves agree, so it is committed.
3. Group `[2, 3]`: the pattern is 62.20-62.90. The fast path searches +/-1.5 s
   around 62.20 - 24.00 = 38.20 and finds 38.20, so the shift is -24.00 and
   the match is committed.
4. Group `[4]`: the same fast-path check commits -24.00.
5. All shifts are equal, so grouping makes one group and the average is
   -24.00. Final times: line 1 is 36.00-38.00, line 4 is 66.00-69.00.

If the destination also had a 3 s cut commercial after line 3, line 4's
fast path around 66.00 would miss. The normal path would find it at 63.00,
giving a shift of -27.00. That differs from the previous shift by more than
0.01 s, so line 4 starts a new shift group with its own shift.

## Observations and risks

- Sushi's tuning options (window sizes, smoothing, keyframes) are not exposed.
- The GUI passes no `--src-keyframes` / `--dst-keyframes`, so keyframe
  snapping is unused.
- The output name derives from the destination video, so an existing subtitle
  file there is overwritten without warning.
- The picker filter `"Subtitle File(*.srt, *.ass);;…"` uses commas inside the
  pattern, which Qt may not split into separate patterns (unverified).
- Sushi relies on external tools (ffmpeg, mkvtoolnix); the GUI does not check
  that they are available (not verified here).
- No progress indicator beyond raw log output, and no tests were found for
  this tab.

This analysis is based on reading the code; the feature was not run.
