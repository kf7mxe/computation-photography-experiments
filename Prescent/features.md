# Prescent — Full Feature Specification & Implementation Audit

> This document captures the complete long-term vision, a breakdown of what each
> phase must deliver, and an honest audit of what has actually been implemented
> so far.  Keep this file updated as features are built.

---

## Long-Term Vision

Prescent is a **computational photography app** targeting Android first, then iOS
and Desktop.  Its core differentiator is an HDR / multi-exposure pipeline built
on top of OpenCV, combined with advanced camera modes (Night Sight, Astrophoto,
Focus Stacking, Photo Sphere, 3D Spatial) and eventual DSLR tethering on Desktop.

---

## Pages / Navigation Structure

| Page | Description | Status |
|------|-------------|--------|
| **Camera View** | Main viewfinder with shutter, gallery button (right), file-open button (left) | ⚠️ Partial |
| **HDR Processing** | Merge/convert bracketed exposures to HDR with tone mapping | ⚠️ Partial |
| **Gallery** | Browse & manage captured bracketed sets; launch HDR from gallery | ✅ Done |
| **Settings** | App-wide preferences (camera selection, defaults, theme) | ✅ Done |
| **Night Sight** | Multi-frame low-light stacking with star trail mode | ✅ Done |
| **Focus Stack** | Multi-focus capture with Laplacian sharpness blending | ✅ Done |
| **Spatial / 3D** | Stereo pair capture with depth map and anaglyph | ✅ Done |
| **Photo Sphere** | Multi-shot panorama stitching | ✅ Done |

---

## Camera View Page — Feature Requirements

### Controls Layout
- [ ] **Left of shutter** — "Open Images" button → file selector → HDR Processing page
- [ ] **Shutter button** — center, large, prominent
- [ ] **Right of shutter** — Gallery button → Gallery page
- [ ] **HDR Mode toggle** — switch between normal and HDR bracketing mode
- [ ] **Bracket count selector** — choose how many exposures to capture (e.g. 2–9)
- [ ] **EV offset control** — set the stop spread between exposures (e.g. ±1, ±2, ±3 EV)
- [ ] **Camera selector** — front/back/ultra-wide/telephoto
- [ ] **Settings shortcut** — top-bar settings icon

### Capture Behaviour
- [ ] In **HDR mode**, after pressing shutter, capture the set of bracketed images
- [ ] Save bracketed sets to a named folder so they can be linked as a group
- [ ] After saving, offer navigation to HDR Processing page
- [ ] Normal (non-HDR) single-shot capture

### Current Status
- ✅ Basic CameraX viewfinder rendered via `LifecycleCameraController`
- ✅ Bracketed exposure capture logic (sequential shots with EV compensation)
- ✅ `bracketCount` and `evOffset` signals wired up
- ✅ Shutter trigger signal drives bracket sequence
- ✅ File-open button (left of shutter) navigates to `HdrProcessingPage`
- ⚠️ Gallery button (right of shutter) — button exists but is a `TODO`
- ✅ HDR mode toggle wired to `cameraView`
- ✅ Captured bracket sets saved to `filesDir/brackets/<timestamp>/` (persistent)
- ✅ Front/back camera lens selector
- ✅ Settings page with persistent defaults (PersistentProperty)

---

## HDR Processing Page — Feature Requirements

### Step 1 — Image Alignment
- [x] **MTB Alignment**
- [x] Option to **skip alignment**
- [x] Choice of alignment method (MTB, ECC, Feature-based)
- [x] **Crop after alignment** toggle (auto-crop black borders)
- [x] **Ghosting removal** strength slider (0.0–1.0)
  - Per-pixel motion mask based on deviation from reference exposure
  - Morphological closing to fill gaps
  - Blends ghosted regions toward reference image

### Step 2 — Algorithm Selection (bottom sheet, slide-up)

Algorithms to include:

| Algorithm | OpenCV API | Status |
|-----------|-----------|--------|
| **Exposure Fusion (Mertens)** | `MergeMertens` | ✅ Implemented |
| **Tone Mapping — Reinhard** | `TonemapReinhard` + `MergeDebevec` | ✅ Pipeline fixed + full slider controls |
| **Tone Mapping — Drago** | `TonemapDrago` + `MergeDebevec` | ✅ Pipeline fixed + full slider controls |
| **Tone Mapping — Mantiuk** | `TonemapMantiuk` + `MergeDebevec` | ✅ Implemented + full slider controls |
| **Gradient Domain Processing** | Custom / Fattal et al. 2002 | ✅ Implemented |
| **iCam06 Model** | Custom perceptual model | ✅ Implemented |
| **Random (fun mode)** | Picks random algorithm + settings | ❌ Not implemented |

> **Clarification on algorithm naming:**
> - *Mertens* is an **Exposure Fusion** algorithm — no HDR radiance map needed.
> - *Reinhard* and *Drago* are **Tone Mapping operators** that should be
>   applied to an HDR radiance map created via `MergeDebevec` (with camera
>   response calibration via `CalibrateDebevec`), NOT directly on Mertens output.
>   The current code applies tone mapping on top of Mertens output — this is
>   **incorrect** and produces suboptimal results.
> - **Gradient Domain Processing** (Fattal et al.) — Laplacian pyramid gradient attenuation.
> - **iCam06** — simplified perceptual model (CAT02 chromatic adaptation + local adaptation + sigmoidal compression + gamut clamping).

### Step 3 — Per-Algorithm Settings

**Exposure Fusion (Mertens)**
- [x] Contrast Weight slider (0.0–2.0)
- [x] Saturation Weight slider (0.0–2.0)
- [x] Exposure Weight slider (0.0–2.0)

**Tone Mapping — Reinhard**
- [x] Gamma (0.1–5.0)
- [x] Intensity (0.0–2.0)
- [x] Light Adaptation (0.0–2.0)
- [x] Color Adaptation (0.0–2.0)

**Tone Mapping — Drago**
- [x] Saturation (0.0–2.0)
- [x] Bias (0.0–2.0)
- [x] Gamma (0.1–5.0)

**Tone Mapping — Mantiuk**
- [x] Saturation (0.0–2.0)
- [x] Scale (0.0–2.0)
- [x] Gamma (0.1–5.0)

**Gradient Domain Processing**
- [x] Alpha (gradient threshold) slider (0.01–0.5)
- [x] Beta (attenuation steepness) slider (0.1–1.0)
- [x] Color Saturation slider (0.0–1.0)

**iCam06**
- [x] Chromatic Adaptation strength slider (0.0–2.0)
- [x] Local Adaptation kernel size slider (0.1–5.0)
- [x] Color Saturation slider (0.0–2.0)

### Step 4 — Preview
- [x] Auto-generate thumbnail preview on settings change (debounced 600 ms)
- [ ] Preview image displayed, dimmed vs full result — ✅ done
- [ ] Activity indicator during processing — ✅ done
- [ ] **Bottom sheet** (slides up) that reveals all algorithm settings — ❌ currently flat scroll only

### Step 5 — Full Processing & Post-Processing
- [ ] "Process Full Image" button — ✅ exists
- [ ] After full process, show **post-processing tools**:
  - [ ] Crop
  - [ ] Image Upscaling (AI-based, e.g. ESRGAN or OpenCV super-res)
  - [ ] Brightness / Contrast / Saturation fine-tune
  - [ ] Sharpening / Clarity
- [x] Save final image to device gallery (MediaStore) — ✅ saves to Pictures/Prescent/

---

## Gallery Page — Feature Requirements

- [x] Browse all captured bracketed sets (grouped by capture session/folder)
- [x] Thumbnail grid view
- [x] Tap a set → open HDR Processing page with that set pre-loaded
- [x] Delete sets or individual images
- [x] Filter by All / Favorites / Recent
- [x] Mark/unmark favorites (persisted across restarts)
- [x] Single-image viewer (full-screen dialog with close button)
- [ ] Filter by HDR mode, Night Sight, etc.
- [ ] Single-image viewer with pinch-to-zoom

---

## Phase 3 — Specialized Photography Modes

### Night Sight / Astrophotography
- [x] Long-exposure frame stacking (capture N frames, align & average)
- [x] Configurable frame count (via `nightSightFrameCount`)
- [x] Noise reduction via temporal averaging
- [x] Dark frame subtraction (wired in processor, no capture UI yet)
- [x] Star trail mode (max-blend instead of average)
- [x] Brightness boost slider (0.5x–4.0x)
- [x] CLAHE local contrast enhancement
- [ ] Dark frame capture UI (currently must be passed programmatically)
- [ ] Manual shutter speed / ISO control

### Focus Stacking (Macro Photography)
- [x] Capture sequence at different focus distances (auto-sweep via CameraX `setFocusDistance`)
- [x] Laplacian-based sharpness map per pixel (local variance of Laplacian)
- [x] Blend sharpest regions into single deep-focus composite (weighted by sharpness)
- [x] Preview and export the stacked result
- [ ] Manual focus distance control per frame (currently auto-sweep 0→1)
- [ ] Real-time sharpness preview during capture

### Spatial / 3D Images (dual-lens devices)
- [x] Dual-shot capture (sequential, 300ms delay)
- [x] Feature matching for rectification (ORB + RANSAC homography)
- [x] Depth map generation via StereoSGBM disparity
- [x] Side-by-side output JPEG
- [x] Red-Cyan anaglyph output
- [ ] Simultaneous capture from two lenses (currently sequential)
- [ ] MPO format output
- [ ] 3D Photo Sphere mode (fuse wide + normal into immersive sphere)

### Photo Sphere
- [x] Multi-shot capture mode (accumulates frames in CameraPage)
- [x] Feature-matching stitch pipeline (ORB + RANSAC homography)
- [x] Auto-crop black borders after stitching (row/column scan)
- [x] Preview and export stitched panorama
- [ ] Gyroscope-guided capture overlay
- [ ] Equirectangular projection output
- [ ] XMP Photo Sphere metadata
- [ ] In-app sphere viewer

---

## Phase 4 — Desktop & Pro Features

### Desktop Application
- [ ] KMP / Compose Multiplatform Desktop target (JVM)
- [ ] Import from files and folder watch (drag-and-drop)
- [ ] Full RAW (DNG/CR3/ARW) support via LibRaw or dcraw
- [ ] Non-destructive edit history / sidecars
- [ ] Large-monitor layout with side panels

### DSLR / Tethered Shooting
- [ ] USB tethering (PTP/MTP via gPhoto2 or libgphoto2-JNI)
- [ ] Wi-Fi tethering where camera supports it
- [ ] Remote shutter control from the app
- [ ] Live View display
- [ ] Pull EXIF / metadata directly from camera sensor
- [ ] Trigger native AEB or manual bracketing via tether

### Pixel Binning Algorithms
- [ ] Software pixel binning (2×2, 4×4) for noise reduction
- [ ] Compare binned vs unbinned output
- [ ] Integrate into low-light pipeline

---

## Known Bugs / Issues (Phase 1 Audit)

1. **Reinhard/Drago pipeline is wrong** — both tone mappers are applied to the
   output of `MergeMertens` (already an 8-bit fused LDR image). The correct flow
   is: `CalibrateDebevec` → `MergeDebevec` (32-bit HDR radiance map) →
   `TonemapReinhard` or `TonemapDrago`.  Fix this before shipping.

2. **Bracket images not persisted** — files go to `cacheDir` and are subject to
   OS eviction. Implement permanent storage via `MediaStore` or
   `context.filesDir` with a metadata DB linking each bracketed set.

3. **Gallery button is a no-op** — the button exists on `CameraPage` but the
   `onClick` block is empty (`// TODO: Gallery Page`).

4. **Settings dialog is a no-op** — settings icon shown but no dialog wired up.

5. **No HDR-mode toggle** — the camera is always in bracketed capture mode;
   single-shot (non-HDR) is not wired up.

6. **`bracketCount`/`evOffset` snapshot bug** — `CameraPage` passes
   `bracketCount.value` and `evOffset.value` as snapshots at composition time.
   Changing these signals later won't update the already-composed `cameraView`.
   The `expect fun cameraView(...)` signature needs to accept `Signal<Int>` and
   `Signal<Float>` directly, or `cameraView` needs to be recomposed on change.

7. **Content URI images won't display** — `ImageRemote("file://...")` breaks
   for `content://` URIs returned by the file picker. Use
   `ImageRemote(uri.toString())` or copy to a temp file first.

---

## Algorithm Quick Reference

| User Term | Correct Name | OpenCV Classes | Phase |
|-----------|-------------|----------------|-------|
| Exposure Fusion | Mertens Fusion | `MergeMertens` | 1 ✅ |
| Tone Mapping | Reinhard | `CalibrateDebevec` + `MergeDebevec` + `TonemapReinhard` | 1 ✅ |
| Tone Mapping | Drago | `CalibrateDebevec` + `MergeDebevec` + `TonemapDrago` | 1 ✅ |
| Tone Mapping | Mantiuk | `CalibrateDebevec` + `MergeDebevec` + `TonemapMantiuk` | 1 ✅ |
| Gradient Domain Processing | Fattal HDR | Custom logarithmic gradient | 2 ✅ |
| iCam06 Model | iCam06 | Full custom perceptual model | 2/3 ✅ |
