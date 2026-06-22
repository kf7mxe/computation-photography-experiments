package com.kf7mxe.prescent.views

actual suspend fun processNightSight(
    images: List<String>,
    algorithm: NightSightAlgorithm,
    useLuckyPreFilter: Boolean,
    luckyKeepFraction: Float,
    starTrail: Boolean,
    darkFramePath: String?,
    brightnessBoost: Float,
    maxPreviewSize: Int
): String? = null
