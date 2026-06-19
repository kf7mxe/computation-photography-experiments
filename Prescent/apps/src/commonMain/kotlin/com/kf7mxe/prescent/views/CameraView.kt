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
    isNightSight: Signal<Boolean>,
    nightSightFrameCount: Signal<Int>,
    nightSightCaptureTrigger: Signal<Int>,
    onNightSightCaptured: ((List<String>) -> Unit)?,
    isFocusStacking: Signal<Boolean> = Signal(false),
    focusStackFrameCount: Signal<Int> = Signal(6),
    focusStackCaptureTrigger: Signal<Int> = Signal(0),
    onFocusStackCaptured: ((List<String>) -> Unit)? = null
)
