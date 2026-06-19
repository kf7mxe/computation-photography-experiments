# Prescent Roadmap

This document outlines the long-term plan for the Prescent computational photography app.

## Phase 1: Core HDR & Camera Foundation (Completed/In Progress)
- [x] Android CameraX Viewfinder integration.
- [x] Bracketed exposure capture logic (adjustable count/offset).
- [x] Basic HDR workflow (MTB Alignment, Exposure Fusion via Mertens).
- [x] Tone Mapping pipeline fixed (CalibrateDebevec → MergeDebevec → Tonemap*).
- [x] HDR Mode Toggle — switch between normal single-shot and HDR bracketing mode.
- [x] Normal Single-Shot Capture — non-HDR photo capture and save.
- [x] Camera Selector — front/back lens selection (via ProcessCameraProvider).
- [x] Settings Page — app-wide preferences with PersistentProperty persistence.
- [x] Persistent Bracket Storage — save bracketed sets to `filesDir/brackets/`.
- [x] Gallery Page — browse bracketed sets, thumbnail grid, open in HDR, delete, favorites (persisted), filter by All/Favorites/Recent, single-image viewer dialog.
- [x] Save Final Image to Gallery — export processed HDR result to device MediaStore.
- [ ] Settings defaults propagated to CameraPage and HdrProcessingPage on entry.
- [ ] Gallery metadata database (replace filesystem scan).

## Phase 2: Advanced HDR Pipeline
- [x] Fix Tone Mapping Pipeline — CalibrateDebevec → MergeDebevec → Tonemap* for Reinhard/Drago/Mantiuk.
- [x] Mantiuk Tone Mapping — TonemapMantiuk with Saturation and Scale controls.
- [x] Per-Algorithm Settings — full slider controls for each algorithm (Reinhard: Gamma/Intensity/Light Adaptation/Color Adaptation; Drago: Saturation/Bias/Gamma; Mantiuk: Saturation/Scale/Gamma; Mertens: Contrast/Saturation/Exposure).
- [x] **Alignment Options** — MTB (existing), ECC (warpPerspective + findTransformECC), Feature-based (ORB + RANSAC homography), Skip.
- [x] **Ghosting Removal** — per-pixel motion mask based on deviation from reference exposure; strength slider.
- [x] **Crop After Alignment** — auto-crops black borders introduced by perspective warping.
- [x] **Fattal Gradient Domain** — Laplacian pyramid gradient attenuation (alpha threshold, beta steepness, color saturation).
- [x] **iCam06 Perceptual Model** — CAT02 chromatic adaptation, local adaptation via Gaussian, sigmoidal compression, XYZ color space pipeline.
- [ ] **Random Algorithm Mode** — picks random algorithm + settings.
- [ ] **Interactive Preview** — bottom sheet (slide-up) for algorithm settings.
- [ ] **Post-Processing Suite** — crop, AI upscaling, brightness/contrast/saturation, sharpening.

## Phase 3: Specialized Photography Modes
- [x] **Night Sight / Astrophotography:**
  - Long-exposure frame stacking (capture N frames, align & average)
  - Configurable frame count
  - Noise reduction via temporal averaging
  - Dark frame subtraction (astrophotography) — processor wired, UI pending
  - Star trail mode
  - Brightness boost / ISO controls
  - CLAHE local contrast enhancement
  - Manual shutter speed control where possible
- [x] **Focus Stacking (Macro Photography):**
  - Capture sequence at different focus distances (auto-sweep via CameraX setFocusDistance)
  - Laplacian-based sharpness map per region per frame
  - Blend sharpest regions into single deep-focus composite (weighted by sharpness)
  - Preview and export the stacked result
- [x] **Spatial / 3D Images (dual-lens devices):**
  - Capture two sequential shots (300ms delay for movement)
  - Feature matching for rectification (ORB + RANSAC homography)
  - Depth map generation via StereoSGBM disparity
  - Side-by-side and red-cyan anaglyph output
  - Preview and export
- [x] **Photo Sphere:**
  - Multi-shot capture mode (accumulates frames in CameraPage)
  - Feature-matching stitch pipeline (ORB + RANSAC homography)
  - Auto-crop black borders after stitching
  - Preview and export stitched panorama

## Phase 4: Desktop & Pro Features
- [ ] **Desktop Application:**
  - KMP / Compose Multiplatform Desktop target (JVM)
  - Import from files and folder watch (drag-and-drop)
  - Full RAW (DNG/CR3/ARW) support via LibRaw or dcraw
  - Non-destructive edit history / sidecars
  - Large-monitor layout with side panels
- [ ] **DSLR / Tethered Shooting:**
  - USB tethering (PTP/MTP via gPhoto2 or libgphoto2-JNI)
  - Wi-Fi tethering where camera supports it
  - Remote shutter control from the app
  - Live View display
  - Pull EXIF / metadata directly from camera sensor
  - Trigger native AEB or manual bracketing via tether
- [ ] **Pixel Binning Algorithms:**
  - Software pixel binning (2×2, 4×4) for noise reduction
  - Compare binned vs unbinned output
  - Integrate into low-light pipeline

## Technical Goals
- **Cross-Platform:** Port HDR processing logic (OpenCV) to iOS and Desktop.
- **Performance:** Optimize pixel-level operations using GPU acceleration where possible.
- **UI/UX:** Maintain a modern, interactive aesthetic across all modes.
- **Content URI Handling:** Properly resolve `content://` URIs from file pickers for image display and processing.
