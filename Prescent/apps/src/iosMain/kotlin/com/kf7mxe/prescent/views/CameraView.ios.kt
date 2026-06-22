package com.kf7mxe.prescent.views

import com.lightningkite.kiteui.views.ElementWriter
import com.lightningkite.kiteui.views.direct.text
import com.lightningkite.reactive.core.Signal

actual fun ElementWriter.cameraView(
    shutterTrigger: Signal<Int>,
    onImagesCaptured: (List<String>) -> Unit,
    bracketCount: Signal<Int>,
    evOffset: Signal<Float>,
    isHdrMode: Signal<Boolean>,
    cameraLens: Signal<Int>,
    onCameraLabels: ((List<String>) -> Unit)?,
    isNightSight: Signal<Boolean>,
    nightSightFrameCount: Signal<Int>,
    nightSightCaptureTrigger: Signal<Int>,
    onNightSightCaptured: ((List<String>) -> Unit)?,
    isFocusStacking: Signal<Boolean>,
    focusStackFrameCount: Signal<Int>,
    focusStackCaptureTrigger: Signal<Int>,
    onFocusStackCaptured: ((List<String>) -> Unit)?,
    isSpatial: Signal<Boolean>,
    spatialCaptureTrigger: Signal<Int>,
    onSpatialCaptured: ((List<String>) -> Unit)?,
    onSphereOrientationUpdate: ((Pair<Float, Float>) -> Unit)?,
    onSphereFrameOrientation: ((Float, Float) -> Unit)?,
    sphereGridData: Signal<List<List<Boolean>>>,
    sphereCurrentCell: Signal<Pair<Int, Int>?>,
    sphereCellImages: Signal<Map<String, String>>
) {
    text("Camera View (iOS — coming soon)")
}
