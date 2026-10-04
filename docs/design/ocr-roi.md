# Selected-Area OCR Region of Interest

## Status

Design for the selected-area OCR optimization. The reusable `SubtitleROI` geometry helper exists in `backend/tools/roi.py` and is initialized on each `SubtitleArea` when extraction starts. Wiring it into the detection and OCR pipeline is follow-up work.

## Goals

- Avoid running text detection and recognition over frame regions that cannot contribute to the selected subtitle area.
- Preserve current selection, overflow-tolerance, confidence-filtering, cache, and output behavior as much as possible.
- Keep all coordinates outside the OCR crop in the existing full-frame coordinate system.
- Compute the crop geometry and coordinate transforms once per video, then reuse them for each frame.
- Fail clearly if a frame's dimensions differ from the dimensions used to construct the ROI.

## Current behavior

With a user-selected subtitle area, accurate-mode detection and comparison OCR in `backend/main.py` operate on full frames. The asynchronous OCR worker in `backend/tools/subtitle_ocr.py` also recognizes full frames, then filters recognized boxes against the selected area. Consequently, text elsewhere in a frame can be detected, recognized, logged as "Out of selection", and discarded.

The separate upper/lower-half crop (`frame_preprocess`) is a coarse preprocessing option. It does not crop to the custom selection rectangle.

## Design decisions

### One ROI geometry per video

Construct one `SubtitleROI` for each `SubtitleArea` from the video's frame dimensions, selected rectangle, and context padding. Its clipped crop bounds, crop origin, crop-to-frame offset, and frame-to-crop offset are calculated once. Decoded pixel data is different for every frame, so each frame still needs to be cropped using the stored bounds.

`SubtitleArea` objects are also created in the GUI before the video dimensions are available, so their constructors cannot initialize an ROI from selection coordinates alone. Each area's `initialize_roi()` is called by the per-video extractor once it knows the frame dimensions. This assumes video frames have stable dimensions. `SubtitleROI.crop()` checks the dimensions and raises an error rather than silently using stale bounds if they change.

### Context padding and frame clipping

Crop around the selection with context padding so text near or slightly outside the selected boundary can still be detected and recognized. Clip the crop bounds to the frame edges.

The padding amount and how it should relate to the configured overflow tolerance require empirical validation. The initial implementation should use one explicit, centralized padding policy; it should not add a GUI setting unless testing demonstrates that user-configurable padding is necessary.

Padding cannot guarantee preservation of every candidate accepted by the existing area-based overflow rule. The existing full-frame-coordinate filter remains authoritative after OCR.

### Bidirectional coordinate transforms

Use the same ROI instance to:

- Crop each full frame into ROI-local image coordinates.
- Translate crop-relative `(x, y)` points back to full-frame coordinates after detection/OCR.
- Translate full-frame points into ROI coordinates when needed.

Translate polygon vertices, not only axis-aligned boxes, so the helper works with OCR and detector outputs. Apply the reverse transform immediately after model inference. Caches, selection checks, subtitle output, and debug overlays continue to use full-frame coordinates.

### Share the crop across detection and recognition

For each processed frame, detection and recognition should use the same cropped image and the same ROI geometry. Do not independently recalculate crop bounds in the detector and recognizer. In accurate mode, the candidate-frame detector and the OCR used to compare subtitle text must follow the same ROI path as the later asynchronous OCR worker.

Where detection results are cached, store or otherwise preserve their coordinate-space contract. Prefer translating them to full-frame coordinates before caching, so consumers do not need to know whether results came from a crop.

### Preserve final selection behavior

After translating boxes to full-frame coordinates, retain the existing selection-overlap, allowed-deviation, and score checks. The crop is a performance optimization, not a change to what is accepted as a subtitle.

## Pipeline integration

1. Build the ROI once after the video dimensions and selected area are known.
2. Pass the ROI through the accurate-mode frame detector/comparison path and the asynchronous OCR worker.
3. For each frame, crop once and reuse the crop for the relevant detection and recognition calls.
4. Translate returned boxes back to full-frame coordinates before caching or downstream filtering.
5. Keep the upper/lower-half preprocessing behavior composable with the ROI. If both are applied, define the transform from the final model crop back to the original frame explicitly; do not mix coordinate origins.
6. Keep behavior unchanged when there is no custom selection area.

The current task setup passes only the first selected rectangle. Supporting multiple simultaneous selection rectangles is outside this optimization unless separately requested.

## Validation

### Unit-level checks

- Crop bounds and image shape for ordinary selections.
- Padding clipped at all four frame edges.
- Invalid, empty, or non-intersecting selections fail explicitly.
- Crop-relative points round-trip through `to_original()` and `to_relative()`.
- Polygon vertices and box corners translate consistently.
- Mismatched frame dimensions raise an explicit error.
- Combined upper/lower-half and custom-ROI transforms map coordinates back to the original frame correctly.

### Pipeline checks

- Fresh OCR and cached OCR follow the same full-frame coordinate contract.
- Boxes inside the selection remain accepted.
- Boxes outside the selection are not sent to recognition when excluded by the padded crop.
- Text crossing selection edges remains discoverable when covered by the padding and is still judged by the existing overflow filter.
- No-selection extraction remains unchanged.
- Compare a representative video before/after for subtitle text, timing, number of discarded out-of-selection OCR results, and processing time.

## Non-goals

- Changing Hebrew OCR confidence calculation or the OCR model.
- Changing subtitle timing, text post-processing, or the selection acceptance formula.
- Adding support for multiple selection rectangles.
- Exposing padding as a user-facing option before its value and need are established by testing.
