package com.kf7mxe.prescent.views

import com.lightningkite.kiteui.views.ElementWriter
import com.lightningkite.reactive.core.Signal

expect fun ElementWriter.cameraView(
    shutterTrigger: Signal<Int>,
    onImagesCaptured: (List<String>) -> Unit,
    bracketCount: Signal<Int>,
    evOffset: Signal<Float>,
    isHdrMode: Signal<Boolean>,
    cameraLens: Signal<Int>,
    onCameraLabels: ((List<String>) -> Unit)? = null,
    isNightSight: Signal<Boolean> = Signal(false),
    nightSightFrameCount: Signal<Int> = Signal(8),
    nightSightCaptureTrigger: Signal<Int> = Signal(0),
    onNightSightCaptured: ((List<String>) -> Unit)? = null,
    isFocusStacking: Signal<Boolean> = Signal(false),
    focusStackFrameCount: Signal<Int> = Signal(6),
    focusStackCaptureTrigger: Signal<Int> = Signal(0),
    onFocusStackCaptured: ((List<String>) -> Unit)? = null,
    isSpatial: Signal<Boolean> = Signal(false),
    spatialCaptureTrigger: Signal<Int> = Signal(0),
    onSpatialCaptured: ((List<String>) -> Unit)? = null,
    // Photo Sphere — fires live (azimuth, pitch) updates for guidance overlay
    onSphereOrientationUpdate: ((Pair<Float, Float>) -> Unit)? = null,
    // Photo Sphere — fires with the orientation of each captured frame
    onSphereFrameOrientation: ((Float, Float) -> Unit)? = null,
    // Photo Sphere — grid coverage state for overlay rendering
    sphereGridData: Signal<List<List<Boolean>>> = Signal(emptyList()),
    // Photo Sphere — current cell (row, col) being pointed at, or null
    sphereCurrentCell: Signal<Pair<Int, Int>?> = Signal(null),
    // Photo Sphere — captured frames with orientations for ghost preview
    sphereGhostFrames: Signal<List<SphereGhostFrame>> = Signal(emptyList())
)
