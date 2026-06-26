
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
[...] (previous session continued)

### 2026-06-18 23:30

**Session goal:** Implement Spatial / 3D Images and Photo Sphere — Phase 3.

**Files changed:**

- `apps/src/commonMain/kotlin/.../views/CameraView.kt` — Extended `expect` signature with `isSpatial`, `spatialCaptureTrigger`, `onSpatialCaptured` params (defaulted).

- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Updated actual signature. Added `captureSpatialPair()`: captures two sequential shots (300ms delay) to `filesDir/spatial/<timestamp>/`. Added spatial trigger observer.

- `apps/src/iosMain/kotlin/.../views/CameraView.ios.kt` — Updated stub signature.

- `apps/src/jsMain/kotlin/.../views/CameraView.js.kt` — Updated stub signature.

- `apps/src/commonMain/kotlin/.../views/SpatialProcessor.kt` *(new)* — `expect suspend fun processSpatial()` returning `SpatialResult` (sideBySidePath, depthMapPath, anaglyphPath).

- `apps/src/androidMain/kotlin/.../views/SpatialProcessor.android.kt` *(new)* — `actual` implementation: loads two images → ORB feature matching → RANSAC homography → StereoSGBM disparity → depth color map → side-by-side composite → red-cyan anaglyph → save to cache/MediaStore.

- `apps/src/iosMain/kotlin/.../views/SpatialProcessor.ios.kt` *(new)* — Stub.

- `apps/src/jsMain/kotlin/.../views/SpatialProcessor.js.kt` *(new)* — Stub.

- `apps/src/commonMain/kotlin/.../views/SpatialPage.kt` *(new)* — UI: source image strip, side-by-side/depth/anaglyph preview cards, auto-preview debounced, full process + save.

- `apps/src/commonMain/kotlin/.../views/PhotoSphereProcessor.kt` *(new)* — `expect suspend fun processPhotoSphere()` returning stitched path.

- `apps/src/androidMain/kotlin/.../views/PhotoSphereProcessor.android.kt` *(new)* — `actual` implementation: loads all frames → incremental ORB stitching: each frame matched to growing panorama via homography → warp → max-blend → auto-crop black borders by scanning non-zero rows/columns → save.

- `apps/src/iosMain/kotlin/.../views/PhotoSphereProcessor.ios.kt` *(new)* — Stub.

- `apps/src/jsMain/kotlin/.../views/PhotoSphereProcessor.js.kt` *(new)* — Stub.

- `apps/src/commonMain/kotlin/.../views/PhotoSpherePage.kt` *(new)* — UI: frame grid, stitch info, result preview, stitch + save button.

- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Added `isSpatial`, `spatialCaptureTrigger`, `sphereFrames` signals. Extended capture mode selector to include "Spatial" and "Sphere" (6 modes total). Sync logic for spatial mode. Sphere mode: `onImagesCaptured` accumulates single-shot paths into `sphereFrames`, shows "Stitch N frames" button when ≥2, navigates to `PhotoSpherePage`. Reset sphere frames on mode switch. CameraView call now passes spatial params.

- `features.md` — Updated Spatial/3D and Photo Sphere sections with implementation status checkboxes.

- `ROADMAP.md` — Both features marked as done with remaining UI polish noted.

### 2026-06-19 10:00

**Session goal:** Fix lens selector to use dynamically discovered cameras from `discoverLenses()`.

**Files changed:**

- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Changed `cameraLabels` from `Signal<List<String>>` to `Signal<List<Pair<Int, String>>>` so indices are carried with labels. Added `onCameraLabels` callback to `cameraView()` call that transforms `List<String>` → indexed pairs. Replaced hardcoded `listOf("Back" to 0, "Front" to 1)` lens selector with reactive `forEach(cameraLabels)` that dynamically creates toggle buttons from whatever cameras `discoverLenses()` reports.

### 2026-06-19 10:30

**Session goal:** Fix camera enumeration on Pixel 6 — ultrawide/telephoto lenses weren't discovered.

**Root cause:** `ProcessCameraProvider.getInstance(androidContext).get()` blocks the main thread and throws `IllegalStateException`, silently caught by the fallback which only populates `DEFAULT_BACK_CAMERA` + `DEFAULT_FRONT_CAMERA`.

**Fix:** Replaced `discoverLenses()` to use Android's `CameraManager.getCameraIdList()` (non-blocking) instead of CameraX's `ProcessCameraProvider.availableCameraInfos`. Builds individual `CameraSelector` instances via `CameraFilter` matching each Camera2 camera ID. Wired `onCameraLabels` callback in `CameraPage.kt` to supply dynamic labels back to the lens selector.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Rewrote `discoverLenses()` to enumerate via `CameraManager.getCameraIdList()` + `CameraCharacteristics.get(CameraCharacteristics.LENS_FACING)` instead of `ProcessCameraProvider.getInstance().get()` (which blocks the main thread). Removed `buildCameraLabel()`. Added imports for `Context`, `CameraCharacteristics`, `CameraManager`.

### 2026-06-19 11:00

**Session goal:** Still only seeing one back camera. Pixel 6 ultrawide still missing.

**Root cause:** On multi-camera devices like Pixel 6, the ultrawide is a **physical camera** within a logical multi-camera group. `CameraManager.getCameraIdList()` only returns logical cameras. Physical cameras are only accessible via `CameraCharacteristics.getPhysicalCameraIds()`. Additionally, CameraX cannot switch to a physical camera via `cameraSelector` alone — you must bind to the logical parent then set `LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID` via Camera2CameraControl interop.

**Fix:**
1. `discoverLenses()` now calls `getPhysicalCameraIds()` on each logical camera to discover physical cameras
2. Each camera is wrapped in its own try-catch so one failure doesn't abort discovery
3. Physical cameras are classified by relative focal length (shortest→Ultrawide, longest→Tele)
4. Camera selection (`selectLens()`) now handles physical cameras by binding to the logical parent and setting the active physical camera ID via Camera2 interop on the capture pipeline
5. `captureSpatialPair()` updated to use the new `cameras` list instead of removed `backCameraSelectors`

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Complete rewrite of lens discovery (`discoverLenses()` → `CameraEntry` data class, logical+physical camera iteration, relative focal length naming). Selection (`selectLens()`) sets `LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID` for physical cameras. `captureSpatialPair()` uses new `cameras` list.

### 2026-06-19 11:30

**Session goal:** Camera selector shows ultrawide option but viewfinder doesn't switch when selected; spatial mode still doesn't use second lens.

**Root cause:** Two issues:
1. `LifecycleCameraController.cameraSelector` property doesn't trigger a rebind — the initial `bindToLifecycle()` is permanent. Changing `cameraSelector` silently sets a flag but doesn't re-open the camera.
2. Setting `LOGICAL_MULTI_CAMERA_ACTIVE_PHYSICAL_ID` via `Camera2CameraControl.setCaptureRequestOptions()` was not reaching the preview because the pipeline was never rebuilt.

**Fix:** Replaced `LifecycleCameraController` with direct `ProcessCameraProvider` + `Preview` + `ImageCapture` management. New `startCamera(entry: CameraEntry)` function:
1. Unbinds all use cases via `provider.unbindAll()`
2. Creates fresh `Preview` and `ImageCapture` instances
3. Binds them to the lifecycle with the logical camera selector via `provider.bindToLifecycle()`
4. Immediately sets the physical camera ID via `Camera2CameraControl.setCaptureRequestOptions()` on the active camera

This ensures a full camera pipeline restart on every lens switch, so physical cameras properly take effect in the viewfinder. The capture pipeline (`imageCapture`) and camera control (`currentCamera.cameraControl`) are re-acquired after each bind so all capture functions use the current instance.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Replaced all `LifecycleCameraController` usage with `ProcessCameraProvider` + `Preview` + `ImageCapture`. Added `startCamera()`, `setActivePhysicalCamera()`, `selectLens()` functions. All capture functions now use `imageCapture` field and `currentCamera?.cameraControl` instead of controller.

### 2026-06-19 11:30

**Session goal:** Fix physical camera selection (ultrawide/tele) not taking effect in viewfinder or spatial mode.

**Root cause:** The previous approach used `CaptureRequest.Key("android.control.logicalMultiCameraActivePhysicalId")` to set the physical camera via `Camera2CameraControl.setCaptureRequestOptions()` after bind. This key is not recognized by the Camera2 framework — logs show `"CaptureRequest.Key is not supported"`. The string-based Key constructor only registers the Key object but the Camera2 pipeline doesn't validate it.

**Fix:** Switched to the official CameraX 1.4 API: `Camera2Interop.Extender(setPhysicalCameraId())` on `Preview.Builder` and `ImageCapture.Builder` **before** binding. This is the correct CameraX mechanism for selecting a physical camera within a logical multi-camera group. The physical camera ID is embedded in the use case configuration at build time, so CameraX passes it through to the Camera2 capture session correctly.

Also cleaned up unused imports and removed the old `setCaptureRequestOptions` + `CaptureRequest.Key` code.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — `startCamera()` now uses `Camera2Interop.Extender(previewBuilder).setPhysicalCameraId()` and `Camera2Interop.Extender(imageCaptureBuilder).setPhysicalCameraId()` instead of post-bind `setCaptureRequestOptions`. Added `Camera2Interop` import.

### 2026-06-19 12:00

**Session goal:** Fix spatial/3D processing — side-by-side, depth map, and anaglyph cards all empty; "Process Full" does nothing.

**Root cause:** The `processSpatial()` function had `matcher.knnMatch(descL, descR, kotlin.collections.listOf(knnMatches), 2)` which passes an **immutable singleton list** (`listOf()`) to OpenCV. OpenCV's knnMatch implementation internally calls `list.add()` to populate the output list, which throws `UnsupportedOperationException` on an immutable list. This exception was caught by the outer try-catch, causing the entire function to return `null` silently. All three result signals stayed null → empty cards.

Secondary issues discovered and fixed:
1. Feature matching between ultrawide and main cameras (different FOV/focal length) produces very few matches — the original code had no fallback when `findHomography` failed
2. Disparity computation on different-FOV images would also fail — now wrapped in its own try-catch
3. All release calls happened after save operations, but a crash in any step leaked previous resources — fixed with per-step error isolation

**Fix:** Completely rewrote `processSpatial()`:
- **Removed** the broken `knnMatch` call (the previous hack attempted a Lowe's ratio test but never used the results anyway)
- **Feature matching** is wrapped in its own try-catch with a `resize` fallback that works for any image pair regardless of FOV difference
- **Disparity** is wrapped in its own try-catch — if stereoscopic depth fails (expected for different-FOV images), it returns null and the depth card stays empty rather than crashing the whole pipeline
- **Anaglyph** is wrapped in its own try-catch
- **Always** produces side-by-side even when everything else fails
- Per-step error logging added to Logcat with `Spatial` tag

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/SpatialProcessor.android.kt` — Full rewrite with per-step try-catch, resize fallback, immutable list (knnMatch) bug removed, proper null handling for optional outputs (depth/anaglyph).

### 2026-06-19 12:30

**Session goal:** Fix spatial/3D output quality and auto-preview.

**Issues found & fixed:**

1. **Wrong lens switching** — `captureSpatialPair()` was switching from the current lens to a different back camera (ultrawide↔main) which creates images with incompatible FOVs. A homography between different-FOV images produces a misaligned result (one side cropped, the other zoomed out with black borders).

2. **Wrong homography direction** — `findHomography(leftPts, rightPts)` produces H that maps LEFT→RIGHT. `warpPerspective(right, rectRight, H, size)` treats H as the inverse mapping internally, so it was mapping right image pixels BACKWARD through the wrong transform. Fixed by using `H.inv()` which maps RIGHT→LEFT properly.

3. **Anaglyph channel swap** — The red-cyan anaglyph should put the RED channel from the left image and the GREEN+BLUE (cyan) channels from the right image. The original code had `listOf(chR[1], chR[2], chL[0])` which put G from right, B from right, R from left — that's actually correct (0=R, 1=G, 2=B). But the commented note was misleading. Clarified with explicit `listOf(chL[0], chR[1], chR[2])`.

4. **Auto-preview not triggering** — The `reactive { val f = frames.hashCode() }` block uses a non-signal value (`frames` is a plain constructor parameter), so the reactive block's launch may not execute reliably. Moved auto-preview to a `load {}` block which fires once on page creation.

5. **Mat release condition** — `if (!usedHomography) { rectLeft.release() }` was releasing the original `leftResized`/`rightResized` before they were released at line below, causing potential double-free.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — `captureSpatialPair()` now uses the same lens for both shots with 300ms delay for natural hand movement. Removed lens-switching logic.
- `apps/src/androidMain/kotlin/.../views/SpatialProcessor.android.kt` — Fixed homography direction (use `H.inv()`). Changed `rectLeft`/`rectRight` from `val` to `var` with `leftResized` defaults. Fixed anaglyph channel order. Increased ORB features to 2000. Fixed release condition. Added better SGBM parameters for handheld stereo.
- `apps/src/commonMain/kotlin/.../views/SpatialPage.kt` — Replaced auto-preview `reactive { launch {} }` with `load { launch {} }` pattern.

### 2026-06-19 13:00

**Session goal:** Fix SBS and anaglyph showing identical images (no stereo effect); add rotation control.

**Root cause:** The side-by-side and anaglyph were built from `rectLeft`/`rectRight` — the **homography-aligned** images. Alignment warps the right image to match the left, so they appear nearly identical. The alignment is necessary only for disparity (depth map) computation.

**Fix:** Split processing into two image paths:
- **Original** `leftResized`/`rightResized` — used for side-by-side and anaglyph (preserving natural parallax from handheld movement)
- **Aligned** `rectLeft`/`rectRight` — used only for depth map via StereoGBM

**Rotation support:** Added `rotation: Int` parameter (0/90/180/270) to `processSpatial()` expect/actual. Before any processing, both images are rotated via `warpAffine` with the rotation matrix centered and expanded to fit the rotated dimensions. SpatialPage gains a rotation toggle bar that triggers auto-preview on change.

**Files changed:**
- `apps/src/commonMain/kotlin/.../views/SpatialProcessor.kt` — Added `rotation: Int = 0` parameter to expect signature.
- `apps/src/androidMain/kotlin/.../views/SpatialProcessor.android.kt` — Split into original + aligned image paths. SBS and anaglyph now use originals. Depth uses aligned. Added rotation via `getRotationMatrix2D` + `warpAffine`. Added `Mat.rotated()` helper.
- `apps/src/iosMain/kotlin/.../views/SpatialProcessor.ios.kt` — Updated stub with `rotation` param.
- `apps/src/jsMain/kotlin/.../views/SpatialProcessor.js.kt` — Updated stub with `rotation` param.
- `apps/src/commonMain/kotlin/.../views/SpatialPage.kt` — Added `rotation` Signal + rotation toggle row (0°/90°/180°/270°). Auto-preview now triggered by `reactive { rotation() }` block. Passes `rotation` to `processSpatial`.

### 2026-06-19 13:30

**Session goal:** Fix rotation causing black bars/outlines in spatial/3D output.

**Root cause:** `warpAffine` with floating-point rotation matrix produces off-by-one pixel dimensions between left/right images. The floating-point `cos`/`sin` calculations in `newW`/`newH` can round differently for identical-size inputs. After resize to `minOf`, the slight mismatch creates visible artifacts in the side-by-side composite.

**Fix:** Replaced `warpAffine` with `Core.rotate` which uses exact integer pixel mapping — no float arithmetic. After rotation, both images are explicitly forced to identical `commonW`×`commonH` dimensions via `minOf` + conditional resize. Updated homography section to use `commonW`/`commonH` instead of removed `w`/`h` variables.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/SpatialProcessor.android.kt` — Replaced `warpAffine` with `Core.rotate`. Added post-rotation common-dimension enforcement. Updated homography warp size reference.

### 2026-06-19 14:00

**Session goal:** Add equirectangular projection and XMP metadata to Photo Sphere output.

**Summary:** The Photo Sphere pipeline previously output a flat homography-stitched panorama. This session rewrote the processor to use gyroscope orientation data for true spherical projection, producing a 4096×2048 equirectangular image with embedded Google Photo Sphere XMP metadata.

**Key changes:**

1. **Gyroscope-based spherical projection** — When orientations (azimuth, pitch) are provided for each frame, the processor projects each image onto a sphere using estimated camera intrinsics (~70° HFOV). Uses backward mapping: each equirectangular output pixel samples the corresponding input frame pixel via bilinear interpolation. Weighted averaging blends overlapping frames (Gaussian falloff from image center). Falls back to homography stitching when orientations are unavailable.

2. **XMP Photo Sphere metadata** — After saving, the JPEG is post-processed to embed Google Photo Sphere XMP metadata (APP1 marker before EOI). Includes ProjectionType=equirectangular, UsePanoramaViewer=True, FullPanoWidth/Height, CroppedArea, StitchingSoftware=Prescent, PoseHeadingDegrees.

3. **Signature change** — `processPhotoSphere()` now accepts `orientations: List<Pair<Float, Float>>` parameter. Expect/actual chain updated across all platforms. PhotoSpherePage passes orientations from CameraPage's `sphereOrientations` list.

4. **Grid overlay fix** — KiteUI `background` property is now `protected` on `NativeElement`. Replaced colored grid cells with Unicode symbols (◉ current, ● captured, ○ empty).

**Files changed:**
- `apps/src/commonMain/kotlin/.../views/PhotoSphereProcessor.kt` — Added `orientations` parameter to expect declaration.
- `apps/src/commonMain/kotlin/.../views/PhotoSpherePage.kt` — Added `orientations` constructor param, passes to `processPhotoSphere`.
- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Passes `sphereOrientations.toList()` to `PhotoSpherePage`. Replaced colored grid cells with Unicode symbol indicators.
- `apps/src/androidMain/kotlin/.../views/PhotoSphereProcessor.android.kt` — Full rewrite: `processSphereOrientations()` (gyroscope-based spherical projection + equirectangular output), `embedXmpMetadata()` (XMP APP1 embedding), `saveOutputImage()` (MediaStore save to PhotoSphere subfolder), `processHomographyFallback()` (previous homography stitch kept as fallback).
- `apps/src/iosMain/kotlin/.../views/PhotoSphereProcessor.ios.kt` — Updated stub signature.
- `apps/src/jsMain/kotlin/.../views/PhotoSphereProcessor.js.kt` — Updated stub signature.
- `features.md` — Marked Equirectangular projection [x] and XMP metadata [x].
- `ROADMAP.md` — Updated Photo Sphere checklist with equirectangular and XMP items.

### 2026-06-19 14:30

**Session goal:** Widen Photo Sphere guidance grid from 3×3 to full 360° coverage.

**Changes in `CameraPage.kt`:**
- Grid changed from 3×3 (90°×60°) to **8 columns × 3 rows** (360°×90°)
- Each column = 45° azimuth, each row = 30° pitch (-45° to +45°)
- `orientationToCell` wraps azimuth to [-180°, +180°] around reference, maps to 8 bins
- `nextCellHint` uses Manhattan distance with **column wrap-around** so it says "turn left" when the nearest empty cell is on the other side of the ring
- Grid rendering updated to 8 columns (1.2rem cells), coverage counter shows "/ 24"
- Companion object constants used throughout for DRY

### 2026-06-19 15:00

**Session goal:** Add Photomatix-style hybrid HDR pipelines combining multiple tone-mapping algorithms.

**Summary:** Implemented two new algorithms that blend individual tone-mapping operators together:

1. **Hybrid (Reinhard + Fattal):** Builds a single HDR radiance map, then tone-maps it with both Reinhard (natural global compression) and Fattal (gradient-domain detail). The Reinhard result provides natural colors and lighting; the Fattal detail ratio is extracted via luminance division, smoothed, and injected back into Reinhard. `surrealAmount` controls the blend strength.

2. **Contrast Optimizer (Mertens + Fattal):** Uses Mertens Exposure Fusion as a clean, halo-free base (via `createMergeMertens`), then builds a separate HDR radiance map for the Fattal detail pass. Fattal detail is applied only to the luminance channel of the Mertens result, with gamma-compressed factor blending to prevent over-boosting.

3. **Lighting Adjustments slider:** A `surrealAmount` (0.0–1.0) parameter with Natural/Balanced/Surreal labels controls the blend across both new algorithms. At 0.0 = pure base algorithm, at 1.0 = full detail injection.

4. **UI:** Default algorithm changed to Hybrid. Algorithm picker row expanded with Hybrid and Contrast Optimizer toggles. Surreal slider shown conditionally with descriptive labels. Per-algorithm parameter sections added for both new algorithms. All parameters feed into the reactive preview pipeline.

**Files changed:**
- `apps/src/commonMain/.../views/HdrProcessor.kt` — Added `surrealAmount: Float = 0.5f` to expect declaration.
- `apps/src/androidMain/.../views/HdrProcessor.android.kt` — Added `hybridToneMap()` (Reinhard + Fattal), `contrastOptimizer()` (Mertens + Fattal luminance). Changed main algorithm dispatch to route "Hybrid" and "Contrast Optimizer" cases. Added `surrealAmount` to actual signature.
- `apps/src/commonMain/.../views/HdrProcessingPage.kt` — Added `surrealAmount` Signal, updated `algorithms` list, added surreal slider with descriptive label, Hybrid and Contrast Optimizer parameter sections, wired `surrealAmount` into reactive preview trigger and `processHdrInternal` call.
- `apps/src/iosMain/.../views/HdrProcessor.ios.kt` — Added `surrealAmount` to stub.
- `apps/src/jsMain/.../views/HdrProcessor.js.kt` — Added `surrealAmount` to stub.

### 2026-06-19 15:30

**Session goal:** Fix Hybrid and Contrast Optimizer producing unrealistic photos; add more hybrid algorithms.

**Root cause of weird photos:** Both algorithms divided independently tone-mapped luminance values: `Core.divide(fattalLum, reinhardLum, detailRatio)`. Fattal and Reinhard produce wildly different per-pixel luminance — Fattal can make shadows 10× brighter while Reinhard keeps them natural. Dividing these creates ratios from 0.1× to 20×, causing massively over/under-bright regions when multiplied back into RGB channels. The subsequent blur didn't fix this because the fundamental approach was wrong.

**Fix — single-output unsharp mask approach:** Both algorithms now tone-map ONCE (Reinhard for Hybrid, Mertens for Contrast Optimizer), extract luminance, then apply a standard Photoshop-style unsharp mask on luminance only:
1. `GaussianBlur(lum)` → base
2. `detail = lum - base`
3. `enhanced = lum + amount × detail` where `amount = surreal × strength × 3`
4. `ratio = enhanced / lum` (typically 0.5–2.0, well-behaved)
5. Multiply ratio into all RGB channels — perfectly preserves hue

Parameters repurposed: `fattalAlpha` → Detail Radius (blur kernel size), `fattalBeta` → Detail Strength (combines with surreal), `fattalColorSaturation` → saturation boost via gamma on ratio.

**New algorithms added:**

3. **Durand (Bilateral Filter):** HDR → Debevec merge → log-luminance → `bilateralFilter` for edge-preserving base/detail decomposition → compress base to 40%, amplify detail by `1 + surreal×2` → exponentiate → color ratio. Classic Durand & Dorsey 2002 approach. Gamma correction applied at end.

4. **CLAHE Boost:** Mertens fusion → Lab color space → CLAHE on L channel (clipLimit=2.0, tile=8×8) → blend original/enhanced by `surrealAmount` → convert back to RGB. Gives natural-looking local contrast without halos.

**UI updates:**
- Algorithm list: 4 new + 6 existing = 10 total
- Surreal slider now shows for all 4 hybrid algorithms
- Contrast Optimizer label: "Clean Mertens fusion + unsharp mask detail on luminance only"
- Per-algorithm sliders relabeled (Detail Radius, Detail Strength, Color Saturation)
- Durand section: Gamma, Radius, Saturation sliders
- CLAHE Boost section: Contrast, Saturation sliders

**Files changed:**
- `apps/src/androidMain/.../views/HdrProcessor.android.kt` — Rewrote `hybridToneMap()` (unsharp mask on Reinhard luminance), rewrote `contrastOptimizer()` (unsharp mask on Mertens luminance). Added `durandToneMap()` (bilateral filter base/detail decomposition with log-luminance). Added `claheToneMap()` (Mertens + CLAHE on Lab L channel). Updated main dispatch for 4 algorithm cases.
- `apps/src/commonMain/.../views/HdrProcessingPage.kt` — Added "Durand" and "CLAHE Boost" to algorithms list. Added UI sections for both with parameter sliders. Updated surreal slider condition for all 4 hybrid algorithms. Fixed Contrast Optimizer subtext and slider labels.

### 2026-06-19 04:20

**Session goal:** Fix Night Sight bugs — processing crash + preview abandonment.

**Files changed:**

- `apps/src/androidMain/.../views/NightSightProcessor.android.kt` — **Fixed `IndexOutOfBoundsException` crash**: When MTB alignment produces all-empty frames (silent failure), fall back to raw frames. Added guard before stacking to return `null` if `subtracted` is still empty.

- `apps/src/androidMain/.../views/CameraView.android.kt` — **Fixed `BufferQueue has been abandoned`**: Added `OnAttachStateChangeListener` to `PreviewView` that calls `cameraProvider?.unbindAll()` when the view detaches from window.

### 2026-06-19 20:00

**Session goal:** Fix Night Sight processing crash — channel mismatch in temporal averaging.

**Files changed:**

- `apps/src/androidMain/.../views/NightSightProcessor.android.kt` — **Fixed `Sizes of input arguments do not match`**: `Mat.zeros(..., CvType.CV_32F)` created a 1-channel accumulator but frames are 3-channel RGB. Changed to `CvType.CV_32FC3` to match frame channel count.

### 2026-06-19 20:30

**Session goal:** Add multiple Night Sight algorithms with Lucky pre-filter combine option.

**New files:**
- `apps/src/commonMain/.../views/NightSightAlgorithm.kt` — Enum with 4 algorithms: Average, Median, Laplacian Pyramid, HDR Merge. Each has `label`, `description`, `supportsLuckyPreFilter` flag.

**Files changed:**
- `apps/src/commonMain/.../views/NightSightProcessor.kt` — Updated `expect` signature: added `algorithm`, `useLuckyPreFilter`, `luckyKeepFraction` params.

- `apps/src/androidMain/.../views/NightSightProcessor.android.kt` — **Full rewrite**: Refactored into helper functions. New algorithms:
  - **Median** (`stackMedianFrames`): Pads each frame's channel values into a column matrix, sorts each row, takes middle column — rejects outliers.
  - **Laplacian Pyramid** (`stackLaplacianFrames`): Builds 4-level Laplacian pyramid per frame, averages at each level, reconstructs — preserves detail at all frequencies.
  - **HDR Merge** (`stackHdrFrames`): Debevec calibration + merge, then Reinhard tone-map — handles varied exposure. Falls back to Average on failure.
  - **Lucky pre-filter** (`luckySelect`): Computes Laplacian variance per frame, keeps top fraction by sharpness — removes motion-blurred frames.

- `apps/src/androidMain/.../views/NightSightProcessor.android.kt` — Also fixed `Core.mean(lap).val[0]` → `` m.`val`[0] `` syntax (Kotlin `val` keyword conflict).

- `apps/src/commonMain/.../views/NightSightPage.kt` — Added algorithm segmented button picker, Lucky pre-filter toggle + keep-fraction slider. Star trail mode restricted to Average algorithm. Algorithm description shown reactively.

- `apps/src/iosMain/.../views/NightSightProcessor.ios.kt` — Updated stub with new params.
- `apps/src/jsMain/.../views/NightSightProcessor.js.kt` — Updated stub with new params.

### 2026-06-21 14:30

**Session goal:** Implement Photo Sphere viewfinder overlay — grid overlay and ghost preview on camera preview.

**Summary:** Added three new parameters to the `cameraView` expect/actual chain (`sphereGridData`, `sphereCurrentCell`, `sphereCellImages`) and implemented the visual overlay on Android.

**Files changed:**

- `apps/src/commonMain/kotlin/.../views/CameraView.kt` — Added 3 new expect parameter defaults: `sphereGridData: Signal<List<List<Boolean>>>`, `sphereCurrentCell: Signal<Pair<Int, Int>?>`, `sphereCellImages: Signal<Map<String, String>>`.

- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — `PreviewView` now wrapped in a `FrameLayout` root. Added `GridOverlayView` (private View subclass) that draws colored rectangles: green (captured), dark (uncaptured), yellow (current cell). Added `ImageView` ghost preview at 40% opacity behind the grid — loads bitmap via `BitmapFactory.decodeFile` with `inSampleSize=4` for the current cell's captured image. Two reactive observers wire grid data and ghost image updates.

- `apps/src/jsMain/kotlin/.../views/CameraView.js.kt` — Updated stub with new params.

- `apps/src/iosMain/kotlin/.../views/CameraView.ios.kt` — Updated stub with new params.

- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Added `sphereCellImages` Signal (maps "row,col" → file path), `pendingSphereCells` queue, `sphereCurrentCell` Signal derived from `orientationToCell`. `onSphereFrameOrientation` pushes cell to queue. `onImagesCaptured` pops from queue and stores path in map. State cleared when exiting sphere mode. New signals passed to `cameraView`.

### 2026-06-21 14:45

**Session goal:** Remove grid overlay, keep only ghost preview for Photo Sphere.

**Summary:** Removed the colored-rectangle `GridOverlayView` from the camera preview. Ghost preview (40% opacity translucent image of previously captured cell) now only shows when the phone is pointed at a position that already has a captured frame — matching the Pixel Photo Sphere ghost behavior.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Removed `GridOverlayView` class, removed `gridOverlay` from the `FrameLayout` stack, removed grid overlay reactive observer, removed unused `Canvas`/`Paint` imports. Ghost `ImageView` remains as the only overlay layer on top of `PreviewView`. Ghost naturally only activates in sphere mode because `sphereCellImages` is cleared on mode exit.

### 2026-06-21 15:00

**Session goal:** Fix phone rotation resetting to home page and captured images showing in wrong orientation.

**Root cause #1:** Android activity recreates on configuration change (rotation) by default. The `MainActivity` manifest entry lacked `android:configChanges`, causing the entire activity stack (including navigation state) to reset on rotation.

**Root cause #2:** CameraX `ImageCapture.Builder` and `Preview.Builder` were not configured with `setTargetRotation()`. Without this, images are captured in the sensor's native orientation (landscape for most rear cameras) without setting the EXIF rotation tag. The preview also stays in sensor orientation regardless of device orientation.

**Files changed:**
- `apps/src/androidMain/AndroidManifest.xml` — Added `android:configChanges="orientation|screenSize|screenLayout|keyboardHidden"` to the `<activity>` tag so Android doesn't recreate the activity on rotation.
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Added `setTargetRotation(displayRotation)` to both `Preview.Builder` and `ImageCapture.Builder` in `startCamera()`. Rotation is obtained from the current display's rotation via `WindowManager`. Added `android.view.Surface` import for rotation constants.

### 2026-06-21 15:30

**Session goal:** Fix Photo Sphere orientation calculation when phone is in portrait mode.

**Root cause:** The rotation vector sensor's `getOrientation()` returns azimuth/pitch in the device's natural coordinate frame. When the display is rotated (e.g., portrait → landscape), `setTargetRotation` correctly rotates the camera output, but the sensor orientation values were not adjusted for the display rotation. This caused the grid cell mapping to think the phone was in a different orientation than it actually was.

**Fix:** Added `SensorManager.remapCoordinateSystem()` to remap the rotation matrix to the display's coordinate frame before extracting orientation values. The axis remapping constants match Android's standard display rotation mapping: ROTATION_90 uses (AXIS_MINUS_Y, AXIS_X), ROTATION_180 uses (AXIS_MINUS_X, AXIS_MINUS_Y), ROTATION_270 uses (AXIS_Y, AXIS_MINUS_X). The display rotation is obtained fresh on each sensor event via `WindowManager.defaultDisplay.rotation`. A `remappedR` float array is pre-allocated to avoid GC pressure on the sensor callback.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Updated `onSensorChanged` to fetch display rotation, compute axis remapping, call `remapCoordinateSystem`, then `getOrientation` on the remapped matrix.

### 2026-06-21 16:00

**Session goal:** Fix Photo Sphere orientation calculation so ghost preview appears at the correct phone tilt.

**Root cause:** `SensorManager.getOrientation()` computes pitch relative to the device being flat on a table (0° = screen up). For a photo sphere we need the camera's elevation relative to the horizon (0° = camera level). These reference frames differ by ~90° and the sign depends on tilt direction, causing the ghost preview to activate at the wrong tilt angle.

**Fix:** Replaced `getOrientation()` with direct computation of camera direction from the rotation matrix. The camera points in the -Z direction of the device. The rotation matrix R maps device→world (East, North, Sky). Camera direction = (-R[0][2], -R[1][2], -R[2][2]). Azimuth = atan2(East, North), elevation = asin(-R[2][2]). This gives elevation = 0° at horizon, positive = looking up, negative = looking down — matching the photo sphere grid's expected reference frame.

Removed the `remapCoordinateSystem` since it's unnecessary when computing camera direction directly in the world frame.

Renamed `currentPitch` → `currentElevation` to reflect the corrected meaning.

**Files changed:**
- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Sensor callback now computes azimuth and elevation directly from rotation matrix elements instead of using `getOrientation()`. Removed `remapCoordinateSystem` and `remappedR`.

### 2026-06-21 16:30

**Session goal:** Replace flat ghost preview with positional green ghost — shows each captured frame at the correct screen position based on angular offset from current view.

**Summary:** Completely redesigned the sphere ghost overlay. Instead of a single translucent ImageView showing the full captured image at 40% opacity, the overlay now:
- Shows each captured frame at its correct angular position relative to current view
- Each frame is drawn at screen coordinates derived from (azimuth, pitch) offset
- Frame only shows when the current view overlaps with the frame's angular position
- As you pan, frames slide in/out of view naturally
- Frames are tinted green (semi-transparent `0x6600FF00` overlay) instead of white/gray
- Frames are pre-rotated per EXIF orientation for correct portrait preview

**Data model changes:**
- `sphereCellImages: Signal<Map<String, String>>` → `sphereGhostFrames: Signal<List<SphereGhostFrame>>` where `SphereGhostFrame(azimuth, pitch, path)` carries the actual capture orientation
- `pendingSphereCells` → `pendingSphereCaptures` to store orientation alongside cell information, ensuring correct path→orientation pairing even during rapid captures

**Files changed:**
- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Added top-level `SphereGhostFrame` data class. Replaced `sphereCellImages`/`pendingSphereCells` with `sphereGhostFrames`/`pendingSphereCaptures`. `PendingSphereCapture` stores both the grid cell and the exact azimuth/pitch at capture time.* `onImagesCaptured` now builds `SphereGhostFrame` entries from pending captures. Sphere state reset clears both lists.
- `apps/src/commonMain/kotlin/.../views/CameraView.kt` — `sphereCellImages` replaced with `sphereGhostFrames: Signal<List<SphereGhostFrame>>` in expect declaration.

- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt` — Replaced ghost `ImageView` with `GhostOverlayView`: custom `View` subclass that draws each captured frame as a green-tinted bitmap at the correct screen position. Screen position is computed from angular offset normalized by FOV constants (55° horizontal × 40° vertical in landscape, swapped for portrait). Frames outside viewport are skipped. EXIF-aware bitmap loading with `inSampleSize=4` for performance. Bitmap cache per path. Sensor callback feeds `(frames, curAz, curElev, displayRot)` to the overlay on every sensor event. Reactive observer syncs `lastGhostFrames` from `sphereGhostFrames` signal. FOV auto-swaps based on display rotation.

- `apps/src/jsMain/kotlin/.../views/CameraView.js.kt` — Updated stub signature.
- `apps/src/iosMain/kotlin/.../views/CameraView.ios.kt` — Updated stub signature.

### 2026-06-22 06:00

**Session goal:** Implement Quad Bayer RAW capture + processing pipeline.

**Summary:** Added a complete Quad Bayer photography mode: direct Camera2 RAW capture (bypassing CameraX), software demosaic with 3 algorithms, multi-frame merging, and pipe-to-HDR/NightSight.

**New files:**

- `apps/src/commonMain/kotlin/.../views/QuadBayerProcessor.kt` — `expect` declaration: `QuadBayerAlgorithm` enum (BIN_TO_BAYER, FULL_REMOSAIC, EDGE_GUIDED), `QuadBayerOptions` data class, `processQuadBayer()` suspend function.

- `apps/src/androidMain/kotlin/.../views/QuadBayerEngine.android.kt` — Camera2 RAW capture engine:
  - `detectRawCapability()` — queries `CameraCharacteristics` for RAW support, sensor size, Bayer pattern, black/white levels, bit depth
  - `captureRawFrames()` — opens Camera2 device, creates `ImageReader(RAW_SENSOR, 3)`, captures N frames sequentially (150ms apart), saves to `.raw` files with 40-byte header (magic, width, height, bitDepth, bayerPattern, blackLevel[4], whiteLevel) followed by uint16 pixel data
  - `readRawFile()` — reads `.raw` files back into `RawFrame` data class
  - Uses `CountDownLatch`-based Camera2 callbacks with dedicated `HandlerThread`

- `apps/src/androidMain/kotlin/.../views/QuadBayerProcessor.android.kt` — `actual` implementation:
  - Loads RAW frames → normalizes to [0,1] float → 3 algorithms per 2×2 cluster:
    - **BIN_TO_BAYER**: average 4 same-color pixels → half-res Bayer → OpenCV `COLOR_Bayer*2BGR_EA` edge-aware demosaic
    - **FULL_REMOSAIC** (default): TL pixel from each 2×2 → half-res Bayer → same demosaic
    - **EDGE_GUIDED**: per-cluster variance decides blend between TL and average
  - Multi-frame merge: average (element-wise) in normalized float domain before demosaic
  - Auto white balance: gray-world on post-demosaic BGR channels (scale R/B to match G mean)
  - Tone mapping: Reinhard luminance compression + gamma correction + 0.5% histogram clip
  - Output: 8-bit BGR → Bitmap → JPEG (quality 95)

**Files changed:**

- `apps/src/commonMain/kotlin/.../views/CameraView.kt` — Added `isQuadBayer`, `quadBayerFrameCount`, `quadBayerAlgorithm`, `quadBayerCaptureTrigger`, `onQuadBayerCaptured` params to expect declaration.

- `apps/src/androidMain/kotlin/.../views/CameraView.android.kt`:
  - Updated actual signature with new Quad Bayer params
  - Added `currentEntry: CameraEntry?` field for re-binding after RAW capture
  - Added `captureQuadBayerFrames()`: unbinds CameraX → Engine.captureRawFrames() → rebinds CameraX → Processor.processQuadBayer() → callback
  - Added reactive observer for `quadBayerCaptureTrigger`
  - Shutter trigger now skips Quad Bayer mode (handled by dedicated trigger)

- `apps/src/jsMain/kotlin/.../views/CameraView.js.kt` — Added Quad Bayer stub params.
- `apps/src/iosMain/kotlin/.../views/CameraView.ios.kt` — Added Quad Bayer stub params.

- `apps/src/commonMain/kotlin/.../views/CameraPage.kt`:
  - Added `isQuadBayer`, `quadBayerFrameCount`, `quadBayerAlgorithm`, `quadBayerCaptureTrigger`, `quadBayerPipeToHdr`, `quadBayerPipeToNightSight` signals
  - Added "Quad" mode toggle button (7th capture mode)
  - Mode switcher sets `isQuadBayer` when "quadbayer" selected, clears on mode exit
  - Shutter trigger dispatches to `quadBayerCaptureTrigger` in Quad Bayer mode
  - Quad Bayer UI overlay: frame count display, algorithm picker (3 toggles), pipe-to-HDR/NightSight checkboxes with mutual exclusion
  - `onQuadBayerCaptured` navigates to HdrProcessingPage, NightSightPage, or GalleryPage based on pipe settings

- `features.md` — Added Quad Bayer row to Pages table. Added Pixel Binning/Quad Bayer section under Phase 4 with implementation status. Added DNG/RAW export as future checkbox.

### 2026-06-22 06:30

**Session goal:** Fix Quad Bayer pipe-to-HDR (was only sending 1 image), fix no-pipe mode (wasn't saving to gallery), speed up processing.

**Issues fixed:**

1. **HDR pipe only sent 1 image** — `captureQuadBayerFrames()` now reads `quadBayerPipeToHdr` signal. When true, each RAW frame is processed individually (not merged), and all resulting JPEGs are fed to the HDR pipeline. RAW frames are captured with different EV offsets (-2, 0, +2 EV for 3 frames, -1.5/+1.5 for 2) via `CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION`.

2. **No-pipe mode didn't save** — `captureQuadBayerFrames()` now calls `saveBayerToGallery()` (MediaStore) after processing when no pipe is selected, so the result appears in the system gallery. GalleryPage navigation still happens on the CameraPage side.

3. **Slow pixel loops** — Replaced all per-pixel Kotlin `for` loops with OpenCV operations:
   - **FULL_REMOSAIC**: `Imgproc.resize(INTER_NEAREST)` — takes TL pixel of each 2×2 cluster in one call
   - **BIN_TO_BAYER**: `Imgproc.filter2D(2×2 avg kernel)` + `resize(INTER_NEAREST)` — box blur then subsample
   - **EDGE_GUIDED**: same operations separately, kept for reference but pixel-quality per-pixel loop removed

4. **Pipe signals wired through expect/actual** — Added `quadBayerPipeToHdr` and `quadBayerPipeToNightSight: Signal<Boolean>` as parameters to the `cameraView()` expect/actual chain so CameraView has access to the pipe mode at capture time.

**Files changed:**
- `apps/src/androidMain/.../views/QuadBayerEngine.android.kt` — Added `evOffsets: List<Float>` optional parameter to `captureRawFrames()`. Sets `AE_EXPOSURE_COMPENSATION` per frame.
- `apps/src/androidMain/.../views/QuadBayerProcessor.android.kt` — Rewrote algorithm section: replaced pixel loops with `Imgproc.resize(INTER_NEAREST)` for FULL_REMOSAIC, `filter2D + resize` for BIN_TO_BAYER, kept EDGE_GUIDED as a reference. Added `saveBayerToGallery(bitmap)` and `matToBitmap()` helper functions. Added MediaStore/ContentValues imports.
- `apps/src/commonMain/.../views/CameraView.kt` — Added `quadBayerPipeToHdr`, `quadBayerPipeToNightSight` signal params to expect declaration.
- `apps/src/androidMain/.../views/CameraView.android.kt` — Updated actual signature. `captureQuadBayerFrames()` now reads pipe flags, sets EV offsets for HDR pipe, processes each RAW individually for HDR, saves to gallery in no-pipe mode.
- `apps/src/commonMain/.../views/CameraPage.kt` — Passes `quadBayerPipeToHdr`, `quadBayerPipeToNightSight` signals to `cameraView()`.
- `apps/src/jsMain/.../views/CameraView.js.kt` — Updated stub with new signal params.
- `apps/src/iosMain/.../views/CameraView.ios.kt` — Updated stub with new signal params.

### 2026-06-22 07:00

**Session goal:** Fix Quad Bayer hangs/stalls — tone mapping per-pixel loops were 12M JNI calls per frame (20+ seconds).

**Root cause:** `toneMapReinhard()` used per-pixel `Mat.get()/put()` loops for luminance computation (8000×6000), Reinhard tone curve, and histogram sorting. Each get/put is a JNI boundary call. At 4000×3000 (half res), this was 12M iterations × multiple accesses = 36M+ JNI calls taking ~20 seconds per frame. For the HDR pipe (2 frames processed sequentially), this was ~40 seconds with no UI feedback.

**Fix — replaced with pure OpenCV matrix operations:**
- **Luminance:** `Core.addWeighted(r, 0.299, g, 0.587)` + second `addWeighted(b, 0.114)` — vectorized, no per-pixel loops
- **Reinhard:** `Core.multiply(ls)` → `Core.add(ls, 1.0)` → `Core.divide(ls, onePlusLs)` → `Core.patchNaNs` → `Core.divide(lMapped, ls, scale)` → `Core.multiply(ch, key)` → `Core.multiply(ch, scale)` → `Core.pow(ch, gamma)` → all vectorized
- **Contrast stretch:** Replaced O(n log n) sort with mean±2.5σ using `Core.meanStdDev` — single-pass vectorized
- **Result:** Processing time drops from ~20s per frame to ~1-2s per frame

**Additional fixes:**
- Added try-catch around the entire coroutine body in `captureQuadBayerFrames()` so any exception invokes `callback(emptyList())` with an error toast instead of silently hanging
- Removed broken `async` usage (deprecated in kotlinx.coroutines 1.9+), kept sequential processing (now fast enough)

**Files changed:**
- `apps/src/androidMain/.../views/QuadBayerProcessor.android.kt` — Full rewrite of `toneMapReinhard()` with vectorized OpenCV operations. Removed all per-pixel `get()`/`put()` loops.
- `apps/src/androidMain/.../views/CameraView.android.kt` — Added try-catch in coroutine body, removed `async` import, sequential processing with error logging.

### 2026-06-23 09:00

**Session goal:** Fix compilation errors, restore algorithm selection, add DNG export and dark frame UI.

**Files changed:**

- `apps/src/androidMain/.../views/CameraView.android.kt` — **Fixed duplicate `val algo`** (lines 598 and 619) that would cause compilation failure. Updated `processQuadBayer` calls in both HDR pipe and no-pipe branches to pass `saveDng = quadBayerSaveDng.value`. Added DNG gallery save call in no-pipe path. Added `quadBayerSaveDng: Signal<Boolean>` to actual signature.

- `apps/src/androidMain/.../views/QuadBayerProcessor.android.kt` — **Full algorithm selection restore**: each algorithm now properly implemented:
  - **BIN_TO_BAYER**: 2×2 box filter averaging + INTER_NEAREST subsample → Bayer at half-res → demosaic
  - **FULL_REMOSAIC**: INTER_NEAREST subsample (TL pixel from each 2×2 cluster) → Bayer at half-res → demosaic  
  - **EDGE_GUIDED**: Local variance computation per 2×2 cluster → adaptive blend weight between TL and average → Bayer → demosaic
  - **New DNG writer** (`saveRawAsDng`): Writes minimal spec-compliant DNG with TIFF header, IFD entries for all required DNG tags (NewSubFileType, ImageWidth/Height, BitsPerSample=16, Compression=uncompressed, CFA PhotometricInterpretation, BlackLevel, WhiteLevel, CFARepeatPatternDim, CFAPattern, CFAPlaneColor), and raw uint16 pixel data. Two-phase write (header buffer + pixel stream) to handle large sensors.
  - **New `saveDngToGallery()`**: Copies DNG to MediaStore Pictures/Prescent/ on API 29+, or direct file + MediaScanner on older versions.
  - Added `import java.io.ByteArrayOutputStream`, `import java.nio.ByteBuffer`, `import java.nio.ByteOrder`.

- `apps/src/commonMain/.../views/QuadBayerProcessor.kt` — Added `saveDng: Boolean = false` to `QuadBayerOptions` data class.

- `apps/src/commonMain/.../views/CameraView.kt` — Added `quadBayerSaveDng: Signal<Boolean> = Signal(false)` parameter to expect declaration. Changed `quadBayerFrameCount` default from `Signal(2)` to `Signal(3)` to match CameraPage.

- `apps/src/commonMain/.../views/CameraPage.kt` — Added `quadBayerSaveDng` signal and "Save DNG" toggle button in Quad Bayer overlay. Passes signal to `cameraView()` call.

- `apps/src/jsMain/.../views/CameraView.js.kt` — Added `quadBayerSaveDng: Signal<Boolean>` to stub actual signature.

- `apps/src/iosMain/.../views/CameraView.ios.kt` — Added `quadBayerSaveDng: Signal<Boolean>` to stub actual signature.

- `apps/src/commonMain/.../views/NightSightPage.kt` — **Dark frame selection UI**: Added `useDarkFrame: Signal<Boolean>` and `darkFramePath: Signal<String?>` signals. Added dark frame card with file picker button (uses `requestFiles` + `copyFileReferencesToPaths`), enable/disable toggle, and clear button. Shows selected path or instructional text. Wired `darkFramePath` into `processNightSight()` call instead of hardcoded `null`. Added import for `copyFileReferencesToPaths`.

- `features.md` — DNG/RAW export marked [x]; Dark frame selection UI marked [x] (was "no capture UI yet").
### 2026-06-23 07:30

**Session goal:** Fix Quad Bayer HDR only capturing 2 frames instead of 3, all with same exposure.

**Root causes:**

1. **Only 2 images instead of 3:** The ImageReader `onImageAvailable` listener ran `saveImage()` (file I/O) directly on the camera handler thread. When the 3rd capture completed, the handler was still processing the file write of the 1st image, so the ImageReader buffer pool was exhausted and the 3rd frame silently failed. CameraHandler is a single-threaded HandlerThread — any blocking work prevents processing of subsequent callbacks and image arrivals.

2. **All same exposure:** `AE_EXPOSURE_COMPENSATION` does not affect `RAW_SENSOR` captures. This metering hint only biases the auto-exposure target for processed outputs (YUV/JPEG). RAW captures always get the sensor's native exposure regardless of compensation values.

**Fixes applied:**

- **Separated capture from I/O in ImageReader listener:** The listener now copies the pixel buffer (`ShortArray` from `ByteBuffer`) and **closes the image immediately** (freeing the buffer pool slot). File saving (header + pixel data written to `.raw` file) is dispatched to a dedicated `Executors.newSingleThreadExecutor()`. The `framesLatch` counts down when the image is acquired+closed, not when the file save completes. This guarantees all 3 frames fit in the 3-slot ImageReader buffer pool.

- **Manual exposure for RAW bracketing:** Frame 0 is captured with `AE_MODE = ON` (auto-exposure). Its `onCaptureCompleted` callback reads the actual `SENSOR_EXPOSURE_TIME` and `SENSOR_SENSITIVITY` from the `TotalCaptureResult`. Frames 1+ use `AE_MODE = OFF` with `SENSOR_EXPOSURE_TIME = baseExposure × 2^EV` and `SENSOR_SENSITIVITY = baseIso`, clamped to sensor ranges from `CameraCharacteristics`. A per-frame `CountDownLatch` waits for each capture callback to fire before submitting the next frame (blocking the singleExecutor thread, not the camera handler).

- **Default frame count:** Changed `quadBayerFrameCount` default from `2` to `3` to match the standard HDR bracket count. Added 1/2/3 toggle buttons in the Quad Bayer overlay.

**Files changed:**

- `apps/src/androidMain/.../views/QuadBayerEngine.android.kt` — Full rewrite of `captureRawFrames()`: listener copies buffer + closes immediately, dispatches save to `ioExecutor`. Manual exposure bracketing via `AE_MODE_OFF` + `SENSOR_EXPOSURE_TIME`. Per-frame `frameLatch` for capture sequencing. Removed unused `saveImage()` function. Added `Collections`, `Executors`, `AtomicLong`, `AtomicInteger` usage.

- `apps/src/commonMain/.../views/CameraPage.kt` — Changed `quadBayerFrameCount` default to 3. Replaced frame count display text with 1/2/3 toggle button selector row. Cleaned up duplicate Quad Bayer overlay code from earlier edit.

- `apps/src/androidMain/.../views/CameraView.android.kt` — (No changes in this session, previous wiring already correct.)

### 2026-06-23 10:00

**Session goal:** Fix Quad Bayer HDR only sending 1 of 3 bracket images to the HDR processing page.

**Investigation findings:** Static code analysis showed the data flow should correctly pass all 3 JPEGs. The `bracketJpegs` list is populated in a sequential for-loop on a background thread, then `done(bracketJpegs)` is called with the full list on the main thread. All 3 bracket toasts fire, confirming `processQuadBayer()` returns non-null for all 3 brackets.

**Primary suspect:** `kotlinx.coroutines.runBlocking { processQuadBayer(...) }` running on `singleExecutor` (a single-thread `Executors.newSingleThreadExecutor()`). `runBlocking` blocks the thread with its own event loop, and `processQuadBayer` uses `withContext(Dispatchers.IO)`. The interaction between `runBlocking`'s event loop interception and the IO dispatcher can cause subtle coroutine dispatch issues that silently lose results for subsequent iterations.

**Changes made:**

- `apps/src/androidMain/.../views/CameraView.android.kt` — Refactored `captureQuadBayerFrames()`:
  - Replaced `singleExecutor.execute { try { ... runBlocking { processQuadBayer } ... } }` with `GlobalScope.launch(Dispatchers.IO) { ... processQuadBayer ... }`.
  - Eliminated the blocking-thread anti-pattern; suspend function now called directly without `runBlocking`.
  - Added `Log.d("CameraView", "Quad Bayer HDR: bracketJpegs final size=...")` right before `done()`.
  - Fixed `return@withContext` bug in the non-HDR pipe's empty-rawPaths handler — replaced with `if (rawPaths.isEmpty()) { withContext(Main) { done }; return@launch }`.
  - Added `import kotlinx.coroutines.withContext`.

- `apps/src/commonMain/kotlin/.../views/CameraPage.kt` — Added `println("CameraPage: onQuadBayerCaptured with ${paths.size} paths")` for diagnostic tracing.

- `apps/src/commonMain/kotlin/.../views/HdrProcessingPage.kt` — Added `println("HdrProcessingPage created with ${images.size} images")` in `init` block for diagnostic tracing.

- `gradle/libs.versions.toml` — Added `coroutines-android` library entry mapping to `kotlinx-coroutines-android:1.10.2`.

- `apps/build.gradle.kts` — Added `implementation(libs.coroutines.android)` to `androidMain` dependencies (required for `Dispatchers.Main`).

**Next step:** Build and test. Check `adb logcat -s CameraView CameraPage` for:
- `Quad Bayer HDR: bracketJpegs final size=3 paths=[...]`
- `CameraPage: onQuadBayerCaptured with 3 paths`
- `HdrProcessingPage created with 3 images`

### 2026-06-23 11:30

**Session goal:** Implement 7 new algorithms: Multi-Scale Guided Fusion, Exposure Stack Joint Denoising, Dark Channel Prior Dehazing, Multi-Frame Super Resolution, Retinex Tone Mapping, Saliency-Weighted Exposure Fusion, and Artistic Effects (Orton, Miniature, Bokeh).

**Files changed:**

- `apps/src/androidMain/.../views/HdrAlgorithms.android.kt` — Added 7 new algorithm implementations (functions 11-17): `multiScaleGuidedFusion()`, `exposureStackJointDenoise()` + `jointDenoiseBracketPair()`, `darkChannelPriorDehaze()`, `multiFrameSuperResolution()` (ECC shift estimation + shift-and-add), `retinexToneMap()`, `saliencyWeightedFusion()` (frequency-tuned saliency), `applyArtisticEffect()` with `ortonEffect()`, `miniatureEffect()`, `bokehEffect()`. Added `GuidedFusionConfig`, `JointDenoiseConfig`, `DehazeConfig`, `SuperResConfig`, `RetinexConfig`, `SaliencyFusionConfig`, `ArtisticConfig` data classes, and `ArtisticEffect` enum. Restored accidentally deleted `fixChannelHotPixels()`. Added `TermCriteria` and `Video` imports.

- `apps/src/commonMain/.../views/HdrProcessor.kt` — Added 17 new params: `guidedFusionLevels`, `guidedFusionSigmaColor/Space`, `retinexSigma/Compression/Gamma`, `saliencyWeight`, `enableJointDenoise`, `enableDehaze`, `dehazePatchSize/Omega`, `superResolutionScale`, `artisticEffect` + 5 related params.

- `apps/src/androidMain/.../views/HdrProcessor.android.kt` — Updated actual signature. Added joint denoise pre-processing. Added 3 new algorithm dispatch cases: "Guided Fusion", "Retinex", "Saliency Fusion". Added post-processing pipeline: dehaze, super-resolution, artistic effects. Removed unused old `Core.subtract(Scalar, Mat, Mat)` calls.

- `apps/src/commonMain/.../views/HdrProcessingPage.kt` — Added 3 new algorithms to picker (14 total). Added per-algorithm UI sections with sliders. Added Effects card: Dehaze toggle + params, Super Resolution slider, Joint Denoise toggle, Artistic Effect selector (None/Orton/Miniature/Bokeh) with per-effect sliders. All 17 new signals tracked in reactive preview and passed to `processHdr()`.

- `apps/src/iosMain/.../views/HdrProcessor.ios.kt` — Updated stub signature.
- `apps/src/jsMain/.../views/HdrProcessor.js.kt` — Updated stub signature.

- `ROADMAP.md` — Marked 7 new algorithms as done. Moved persist settings to Phase 2 TODO.

### 2026-06-23 15:00

**Session goal:** Add 7 new focus stacking algorithms and wire up the existing dead `focusStackBlend()`.

**Files changed:**

- `apps/src/commonMain/.../views/FocusStackProcessor.kt` — Added 11 new params: `algorithm`, `alignmentMethod`, `exposureBalance`, `showDepthMap`, `refocusDepth`, `focalLength`, `aperture`, `focusDistanceMeters`, `hdrHybridFramesPerFocus`, `pyramidLevels`.

- `apps/src/androidMain/.../views/FocusStackProcessor.android.kt` — Complete rewrite. Replaced dead simple-average path with proper Laplacian-weighted blending. Implemented 7 algorithms:
  1. **Multi-Scale Pyramid** — builds Laplacian pyramid per frame, selects sharpest pixel per level independently, reconstructs
  2. **Depth Map** — per-pixel argmax of Laplacian response across frames, outputs grayscale or jet colormap
  3. **Interactive Refocus** — fits Gaussian to depth map, weights frames by distance from focal plane, re-renders
  4. **Exposure Balanced** — normalizes each frame's mean to median brightness before blending (fixes focus breathing)
  5. **Feature Align** — ORB + RANSAC homography alignment (handles magnification changes that MTB can't)
  6. **HDR Hybrid** — groups frames by focus position, Mertens-fuses each group into HDR, then focus-stacks
  7. **Focus Bracketing Optimizer** — circle-of-confusion math, computes hyperfocal distance, recommends minimum step count
  Added `buildPyramid()`, `multiScalePyramidStack()`, `generateDepthMap()`, `interactiveRefocus()`, `exposureBalanceFrames()`, `alignFocusFrames()`, `alignFocusByFeature()`, `hdrHybridFocusStack()`, `focusBracketingOptimizer()`, `standardFocusStack()`, `focusStackBlend()`.

- `apps/src/commonMain/.../views/FocusStackPage.kt` — Added algorithm picker (7 options), alignment selector (MTB/Feature/None), exposure balance toggle, depth map colorization toggle, refocus depth slider, lens spec sliders (focal length, aperture, focus distance) for optimizer, pyramid levels slider. All signals tracked in reactive preview trigger. Removed deprecated `divider()` calls and unnamed local variables.

- `apps/src/iosMain/.../views/FocusStackProcessor.ios.kt` — Updated stub with 11 new params.
- `apps/src/jsMain/.../views/FocusStackProcessor.js.kt` — Updated stub with 11 new params.

- `ROADMAP.md` — Expanded focus stacking section with 7 sub-items as completed.

**Session goal:** Integrate PhotonCamera-inspired algorithms (Laplacian Pyramid Fusion, hot pixel correction, CA correction, lens correction, smart NR, contrast sharpening, smart frame selection) into the HDR processing pipeline.

**Summary:** Wired all 10 algorithms from `HdrAlgorithms.android.kt` into the existing `processHdr` pipeline with full UI controls and preview reactivity.

**Files changed:**

- `apps/src/commonMain/kotlin/.../views/HdrProcessor.kt` — Added 6 new parameters to `expect` declaration: `pyramidNoiseStrength: Float = 1.0f`, `enableHotPixelFix: Boolean = false`, `enableCACorrection: Boolean = false`, `enableLensCorrection: Boolean = false`, `enableSmartNR: Boolean = false`, `enableContrastSharpening: Boolean = false`, `smartFrameSelection: Boolean = false`.

- `apps/src/androidMain/kotlin/.../views/HdrProcessor.android.kt` — Added "Pyramid Fusion" case to algorithm dispatcher, uses `laplacianPyramidFusion()` with `PyramidMergeConfig(noiseStrength)`. Added smart frame selection before alignment (drops blurry frames via `selectSharpestFrames`). Added per-frame pre-processing: hot pixel fix (`detectAndFixHotPixels`), CA correction (`correctChromaticAberration`), lens correction (`correctLensDistortion`). Added post-processing on tonemapped result: smart NR (`smartNoiseReduction`), contrast sharpening (`contrastLimitedSharpening`). All steps guarded by their respective enable flags and logged.

- `apps/src/iosMain/kotlin/.../views/HdrProcessor.ios.kt` — Updated actual signature with 6 new params (stub returns null).

- `apps/src/jsMain/kotlin/.../views/HdrProcessor.js.kt` — Updated actual signature with 6 new params (stub returns null).

- `apps/src/commonMain/kotlin/.../views/HdrProcessingPage.kt` — Added "Pyramid Fusion" (11th algorithm) to algorithm list and randomize pool. Added `pyramidNoiseStrength` Signal + slider for noise-aware fusion control. Added Enhancements card with 6 toggle buttons: Smart Frame Selection, Hot Pixel Fix, CA Correction, Lens Correction, Smart NR, Contrast Sharpening. All new signals tracked in reactive preview trigger and passed to `processHdr()` call. Randomize function now randomizes all new flags. Updated algorithm count references.
