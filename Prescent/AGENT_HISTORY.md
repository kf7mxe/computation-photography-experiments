
### 2026-06-18 20:11

**Session goal:** Fix all Phase 1 bugs identified in the audit; add Gallery page; update features.md.

**Files changed:**

- `features.md` *(new)* — Comprehensive feature specification and Phase 1 implementation audit. Documents all four pages, per-algorithm HDR settings, future phases (Night Sight, Focus Stacking, DSLR), known bugs, and algorithm naming clarification.

- `apps/src/androidMain/kotlin/…/views/HdrProcessor.android.kt` — **Fixed Reinhard/Drago pipeline**: now uses `CalibrateDebevec` → `MergeDebevec` to build a true 32-bit HDR radiance map before applying the tone mapper. Added `Mantiuk` tone mapper. Full-quality results now saved via `MediaStore` to the device's Pictures/Prescent folder (API 29+) or legacy storage below. Content URIs decoded correctly for images picked from the file picker. Preview still goes to cacheDir.

- `apps/src/commonMain/kotlin/…/views/CameraView.kt` — Changed `expect fun` signature: `bracketCount` and `evOffset` changed from value snapshots (`Int`, `Float`) to `Signal<Int>`, `Signal<Float>` so live changes always propagate to the camera implementation.

- `apps/src/androidMain/kotlin/…/views/CameraView.android.kt` — Updated to new Signal-based signature. Bracket images now saved to `filesDir/brackets/<timestamp>/` (persistent, not evicted) instead of `cacheDir`.

- `apps/src/commonMain/kotlin/…/views/HdrProcessor.kt` — Cleaned up unused import; updated default param doc comment.

- `apps/src/commonMain/kotlin/…/views/HdrProcessingPage.kt` — Rewrote using correct KiteUI patterns: `shownWhen` replaces reactive view-creation blocks; `toggleButton` + `equalTo()` used for algorithm/alignment pickers; content URI image display fixed; `Mantiuk` algorithm added; save-to-gallery success banner added.

- `apps/src/commonMain/kotlin/…/views/CameraPage.kt` — Passes `Signal` references to `cameraView` (not snapshots). Added working Settings dialog using `dialog { close -> }` pattern with bracket count + EV offset sliders. Added HDR mode toggle. Wired Gallery button to `GalleryPage`.

- `apps/src/commonMain/kotlin/…/views/GalleryPage.kt` *(new)* — Gallery page that scans `filesDir/brackets/` for captured bracket sets and allows opening any set directly in `HdrProcessingPage`. Uses `expect/actual` for platform-specific disk scanning.

- `apps/src/androidMain/kotlin/…/views/GalleryPage.android.kt` *(new)* — Android actual: scans `filesDir/brackets/` and returns sorted `BracketSet` list.

- `apps/src/jsMain/kotlin/…/views/CameraView.js.kt` — Updated to new Signal-based signature.
- `apps/src/jsMain/kotlin/…/views/HdrProcessor.js.kt` — Updated to full signature with all parameters.
- `apps/src/jsMain/kotlin/…/views/GalleryPage.js.kt` *(new)* — JS stub returning empty list.

- `apps/src/iosMain/kotlin/…/views/CameraView.ios.kt` — Updated to new Signal-based signature.
- `apps/src/iosMain/kotlin/…/views/HdrProcessor.ios.kt` — Updated to full signature with all parameters.
- `apps/src/iosMain/kotlin/…/views/GalleryPage.ios.kt` *(new)* — iOS stub returning empty list.

- `apps/src/commonMain/kotlin/…/App.kt` — Replaced placeholder "Home" nav item with "Gallery" (`GalleryPage`). Now has Camera + Gallery in the bottom nav.

### 2026-06-18 14:30
- Fixed compilation errors in CameraPage.kt, GalleryPage.kt, and HdrProcessingPage.kt
- **CameraPage.kt**: Added import for `com.lightningkite.reactive.extensions.equalTo`; fixed `shownWhen { }.align()` pattern by wrapping in `frame { atTopEnd... }` since `shownWhen` returns `CanAddSizing` not `CanAddAlignment`
- **GalleryPage.kt**: Replaced `recyclerView { children() }` with `frame { expanding.scrolling.col { forEach() } }` pattern due to `children` property shadowing the extension function on `Recycler2`; fixed `shownWhen { }.centered.expanding` chain by using `frame { centered.col }` pattern
- **HdrProcessingPage.kt**: Added import for `equalTo`; fixed multiple `shownWhen { }.centered` and `shownWhen { }.align()` patterns by wrapping in `frame { }` with appropriate alignment helpers (`atBottomCenter`, `atCenterEnd`, `centered`)
- Root cause: KiteUI 8.0.0-prerelease-379 has strict modifier chaining where `shownWhen` returns `CanAddSizing`, breaking the chain to `CanAddAlignment` needed for `align`/`centered`/`expanding`

### 2026-06-18 15:00
- Fixed image loading issue when using "Open images" file picker on Android
- **Root cause**: `FileReference.toString()` returns object identity string (e.g., `FileReference@12345`), not the underlying URI. Content:// URIs from the file picker also cannot be read directly by native HDR processing code.
- **Solution**: Created `copyFileReferencesToPaths()` expect/actual function that:
  - Copies content from `FileReference` URIs to actual files in the app's cache directory
  - Returns absolute file paths that can be read by both KiteUI image views and native OpenCV code
  - Platform-specific implementations in androidMain, iosMain, and jsMain
- **Files changed**:
  - `apps/src/commonMain/kotlin/com/kf7mxe/prescent/utils/fileUtils.kt` - expect declaration
  - `apps/src/androidMain/kotlin/com/kf7mxe/prescent/utils/fileUtils.android.kt` - Android implementation using ContentResolver to copy URI content
  - `apps/src/iosMain/kotlin/com/kf7mxe/prescent/utils/fileUtils.ios.kt` - iOS stub
  - `apps/src/jsMain/kotlin/com/kf7mxe/prescent/utils/fileUtils.js.kt` - JS stub
  - `apps/src/commonMain/kotlin/com/kf7mxe/prescent/views/CameraPage.kt` - Updated file picker button to use `copyFileReferencesToPaths()` before navigating to HdrProcessingPage

### 2026-06-18 22:00
**Session goal:** Implement Gallery filter/single-image-viewer, settings persistence, per-algorithm sliders.

**Files changed:**
- `apps/src/commonMain/kotlin/.../views/GalleryPage.kt` — Added filter bar (All/Favorites/Recent) with `toggleButton` + `equalTo()`. Added `filteredSets` Signal updated in a `reactive {}` block. Persisted favorites via `PersistentProperty` + `Json` serialization. Added single-image viewer dialog with close button on thumbnail tap.
- `apps/src/commonMain/kotlin/.../views/SettingsPage.kt` — Rewrote to use `PersistentProperty.lens()` for persistence of bracket count, EV offset, default algorithm, and alignment. No more `Signal(3)` hardcoded defaults — settings survive app restarts.
- `apps/src/commonMain/kotlin/com/kf7mxe/prescent/settings.kt` *(new)* — Shared top-level `PersistentProperty` instances for bracket count, EV offset, algorithm, and alignment, accessible from any page.
- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Loads persisted bracket count and EV offset via `load {}` block on page entry.
- `apps/src/commonMain/kotlin/.../views/HdrProcessingPage.kt` — Added per-algorithm sliders: Reinhard (Gamma/Intensity/Light Adaptation/Color Adaptation), Drago (Gamma/Saturation/Bias), Mantiuk (Gamma/Saturation/Scale). Loads persisted algorithm/alignment defaults on entry. Extended `processHdr()` call and reactive tracking with all new parameters.
- `apps/src/commonMain/kotlin/.../views/HdrProcessor.kt` — Extended `expect` signature with gamma, intensity, lightAdaptation, colorAdaptation, dragoBias, mantiukScale parameters.
- `apps/src/androidMain/kotlin/.../views/HdrProcessor.android.kt` — Updated `actual` signature; tone mappers now use per-algorithm parameters (Reinhard: setGamma/setIntensity/setLightAdaptation/setColorAdaptation; Drago: setGamma/setSaturation/setBias; Mantiuk: setGamma/setSaturation/setScale).
- `apps/src/iosMain/kotlin/.../views/HdrProcessor.ios.kt` — Updated stub signature.
- `apps/src/jsMain/kotlin/.../views/HdrProcessor.js.kt` — Updated stub signature.
- `apps/src/commonMain/kotlin/.../utils/numbers.kt` *(new)* — `Float.fmt()` extension for KMP-compatible float-to-string formatting (replaced JVM-only `String.format`).
- `ROADMAP.md` — Marked Phase 1 items as done; promoted per-algorithm sliders to done in Phase 2.
- `features.md` — Updated status of Gallery, Settings, per-algorithm sliders, and algorithm quick reference.

### 2026-06-18 23:30
**Session goal:** Implement ECC + Feature-based alignment, ghosting removal, crop after alignment.

**Files changed:**
- `apps/src/commonMain/kotlin/.../views/HdrProcessor.kt` — Extended expect signature with `ghostingStrength: Float` and `cropAfterAlignment: Boolean`.
- `apps/src/androidMain/kotlin/.../views/HdrProcessor.android.kt` — Major rewrite of alignment section: added `alignImages()` dispatcher that routes to `alignECC()` (uses `Video.findTransformECC` with `MOTION_HOMOGRAPHY` + `warpPerspective`), `alignFeature()` (uses ORB detector + BRUTEFORCE_HAMMING matcher + Lowe's ratio test + `Calib3d.findHomography` RANSAC + `warpPerspective`), and existing `alignMTB()`. Added `removeGhosting()`: computes per-pixel deviation from reference exposure, builds a ghost mask with morphological closing, blends ghosted regions toward the reference image, strength-controlled. Added `cropValidRegion()`: computes bounding rect of non-zero pixels across all aligned images, adds 5% margin, returns cropped sub-mats. Fixed `org.opencf` typo → `org.opencv.android.Utils`. Added `TermCriteria`, `MatOfDMatch`, `DMatch`, `KeyPoint`, `Size`, `Video` imports.
- `apps/src/commonMain/kotlin/.../views/HdrProcessingPage.kt` — Added `ghostingStrength` and `cropAfterAlignment` signals. Alignment picker extended to MTB/ECC/Feature/Skip. Added ghosting removal slider (0.0–1.0). Added crop-after-alignment switch. All new signals tracked in reactive preview block and passed to `processHdr()`.
- `apps/src/iosMain/kotlin/.../views/HdrProcessor.ios.kt` — Updated stub signature.
- `apps/src/jsMain/kotlin/.../views/HdrProcessor.js.kt` — Updated stub signature.
- `ROADMAP.md` — Advanced Alignment Options, Ghosting Removal, and Crop After Alignment marked as done.

### 2026-06-18 23:45
**Session goal:** Implement Fattal Gradient Domain Processing and iCam06 perceptual model.

**Files changed:**
- `apps/src/commonMain/kotlin/.../views/HdrProcessor.kt` — Extended expect signature with `fattalAlpha`, `fattalBeta`, `fattalColorSaturation`, `icam06ChromaticAdaptation`, `icam06LocalAdaptation`.
- `apps/src/androidMain/kotlin/.../views/HdrProcessor.android.kt` — Added `fattalToneMap()` and `icam06ToneMap()` private functions. Fattal uses Laplacian pyramid gradient attenuation: builds 5-level Gaussian pyramid, computes Laplacian + gradients at each level, attenuates Laplacian coefficients where gradient exceeds threshold α using power function (α/|∇|)^(1-β), reconstructs log luminance from attenuated pyramid, exponentiates, recombines with original color via saturation-weighted ratio. iCam06 converts RGB→XYZ, applies chromatic adaptation (blend X/Z toward Y), computes local adaptation luminance via Gaussian blur, applies sigmoidal compression Y_out = Y / (Y + Y_local^0.7), recombines XYZ, converts back to RGB with gamut clamping. Both wired into the Debevec HDR pipeline when algorithm is "Fattal" or "iCam06".
- `apps/src/commonMain/kotlin/.../views/HdrProcessingPage.kt` — Added `fattalAlpha`, `fattalBeta`, `fattalColorSaturation`, `icam06ChromaticAdaptation`, `icam06LocalAdaptation` signals. Extended algorithm list to include Fattal and iCam06. Added per-algorithm UI sections. All new signals tracked in reactive preview and passed to processHdr().
- `apps/src/iosMain/kotlin/.../views/HdrProcessor.ios.kt` — Updated stub signature.
- `apps/src/jsMain/kotlin/.../views/HdrProcessor.js.kt` — Updated stub signature.
- `ROADMAP.md` — Gradient Domain Processing and iCam06 Model marked as done.
- `features.md` — Tables and per-algorithm settings sections updated.

### 2026-06-18 23:00

**Session goal:** Implement Focus Stacking (Macro Photography) — Phase 3.

**Files changed:**

- `apps/src/commonMain/kotlin/.../views/CameraView.kt` — Extended `expect` signature with `isFocusStacking`, `focusStackFrameCount`, `focusStackCaptureTrigger`, `onFocusStackCaptured` parameters (all with defaults so existing callers don't break).

- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Updated `actual` signature with new focus stack params. Added `captureFocusStackFrames()`: sweeps focus from near (0) to far (1) via `Camera2CameraControl.setCaptureRequestOptions` with `LENS_FOCUS_DISTANCE` and `AF_MODE_OFF`, saves frames to `filesDir/focusstack/<timestamp>/`, 150ms inter-frame delay. Added focus stack trigger observer. Updated shutter trigger to delegate to `focusStackCaptureTrigger` in focus mode. Added imports for `Camera2CameraControl`, `CaptureRequestOptions`, `CaptureRequest`.

- `apps/src/iosMain/kotlin/.../views/CameraView.ios.kt` — Updated stub signature with focus stack params.

- `apps/src/jsMain/kotlin/.../views/CameraView.js.kt` — Updated stub signature with focus stack params.

- `apps/src/commonMain/kotlin/.../views/FocusStackProcessor.kt` *(new)* — `expect suspend fun processFocusStack(images, maxPreviewSize)`.

- `apps/src/androidMain/kotlin/.../views/FocusStackProcessor.android.kt` *(new)* — `actual` implementation: loads frames, MTB alignment, computes Laplacian per frame, builds sharpness weight maps via local variance of Laplacian over 15×15 neighborhoods, normalizes weights per-pixel across frames, weighted-blends all frames into single deep-focus composite, saves to cache (preview) or MediaStore Pictures/Prescent (full). Clean resource management (all Mats released).

- `apps/src/iosMain/kotlin/.../views/FocusStackProcessor.ios.kt` *(new)* — Stub returning null.

- `apps/src/jsMain/kotlin/.../views/FocusStackProcessor.js.kt` *(new)* — Stub returning null.

- `apps/src/commonMain/kotlin/.../views/FocusStackPage.kt` *(new)* — UI page: back button, frame strip preview, processing info card, auto-preview on entry (debounced 600ms), full process + save to gallery button with success banner. Follows same pattern as NightSightPage.

- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Added `isFocusStacking`, `focusStackFrameCount`, `focusStackCaptureTrigger` signals. Extended capture mode selector to include "Focus" (4th mode). Reactive sync now handles `"focus"` → `isFocusStacking=true`. Shutter button dispatches to `focusStackCaptureTrigger` in focus mode. `cameraView()` call now passes focus stack params with `onFocusStackCaptured` → `FocusStackPage`. Added focus stack info overlay showing frame count.

- `features.md` — Added Focus Stacking section with progress checkboxes. Updated Night Sight status to reflect actual implementation. Added Focus Stack and Night Sight to pages/status table.

- `ROADMAP.md` — Night Sight and Focus Stacking items marked as done (with notes on remaining UI polish).
