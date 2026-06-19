package com.kf7mxe.prescent.views

actual suspend fun processSpatial(
    images: List<String>,
    maxPreviewSize: Int,
    rotation: Int
): SpatialResult? {
    println("Spatial processing not yet available on JS")
    return null
}
