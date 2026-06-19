package com.kf7mxe.prescent.views

data class SpatialResult(
    val sideBySidePath: String?,
    val depthMapPath: String?,
    val anaglyphPath: String?
)

expect suspend fun processSpatial(
    images: List<String>,
    maxPreviewSize: Int = 0,
    rotation: Int = 0
): SpatialResult?
