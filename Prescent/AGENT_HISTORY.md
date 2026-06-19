
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
