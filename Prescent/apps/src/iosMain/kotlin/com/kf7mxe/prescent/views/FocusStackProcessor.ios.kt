package com.kf7mxe.prescent.views

actual suspend fun processFocusStack(
    images: List<String>,
    maxPreviewSize: Int,
    algorithm: String,
    alignmentMethod: String,
    exposureBalance: Boolean,
    showDepthMap: Boolean,
    refocusDepth: Float,
    focalLength: Float,
    aperture: Float,
    focusDistanceMeters: Float,
    hdrHybridFramesPerFocus: Int,
    pyramidLevels: Int
): String? {
    println("FocusStack processing not yet available on iOS")
    return null
}
