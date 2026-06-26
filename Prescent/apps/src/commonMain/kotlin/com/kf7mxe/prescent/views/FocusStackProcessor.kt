package com.kf7mxe.prescent.views

expect suspend fun processFocusStack(
    images: List<String>,
    maxPreviewSize: Int = 0,
    algorithm: String = "Multi-Scale Pyramid",
    alignmentMethod: String = "MTB",
    exposureBalance: Boolean = false,
    showDepthMap: Boolean = false,
    refocusDepth: Float = 0.5f,
    focalLength: Float = 50.0f,
    aperture: Float = 2.8f,
    focusDistanceMeters: Float = 1.0f,
    hdrHybridFramesPerFocus: Int = 0,
    pyramidLevels: Int = 4
): String?
