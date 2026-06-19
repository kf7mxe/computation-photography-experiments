package com.kf7mxe.prescent.views

actual suspend fun processPhotoSphere(
    images: List<String>,
    orientations: List<Pair<Float, Float>>,
    maxPreviewSize: Int
): String? {
    println("PhotoSphere processing not yet available on iOS")
    return null
}
