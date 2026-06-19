package com.kf7mxe.prescent.views

expect suspend fun processPhotoSphere(
    images: List<String>,
    orientations: List<Pair<Float, Float>> = listOf(),
    maxPreviewSize: Int = 0
): String?
