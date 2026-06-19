package com.kf7mxe.prescent.views

expect suspend fun processHdr(
    images: List<String>,
    algorithm: String,
    alignment: String,
    contrastWeight: Float = 1.0f,
    saturationWeight: Float = 1.0f,
    exposureWeight: Float = 0.0f,
    gamma: Float = 1.0f,
    intensity: Float = 0.0f,
    lightAdaptation: Float = 0.0f,
    colorAdaptation: Float = 0.0f,
    dragoBias: Float = 0.85f,
    mantiukScale: Float = 0.75f,
    ghostingStrength: Float = 0.0f,
    cropAfterAlignment: Boolean = false,
    fattalAlpha: Float = 0.1f,
    fattalBeta: Float = 0.9f,
    fattalColorSaturation: Float = 0.5f,
    icam06ChromaticAdaptation: Float = 1.0f,
    icam06LocalAdaptation: Float = 1.0f,
    maxSize: Int = 0
): String?
