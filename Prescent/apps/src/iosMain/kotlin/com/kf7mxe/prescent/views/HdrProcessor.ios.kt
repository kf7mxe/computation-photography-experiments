package com.kf7mxe.prescent.views

actual suspend fun processHdr(
    images: List<String>,
    algorithm: String,
    alignment: String,
    contrastWeight: Float,
    saturationWeight: Float,
    exposureWeight: Float,
    gamma: Float,
    intensity: Float,
    lightAdaptation: Float,
    colorAdaptation: Float,
    dragoBias: Float,
    mantiukScale: Float,
    ghostingStrength: Float,
    cropAfterAlignment: Boolean,
    fattalAlpha: Float,
    fattalBeta: Float,
    fattalColorSaturation: Float,
    icam06ChromaticAdaptation: Float,
    icam06LocalAdaptation: Float,
    surrealAmount: Float,
    maxSize: Int
): String? = null
