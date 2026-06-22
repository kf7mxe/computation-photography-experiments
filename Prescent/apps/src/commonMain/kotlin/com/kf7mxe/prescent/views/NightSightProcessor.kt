package com.kf7mxe.prescent.views

expect suspend fun processNightSight(
    images: List<String>,
    algorithm: NightSightAlgorithm = NightSightAlgorithm.AVERAGE,
    useLuckyPreFilter: Boolean = false,
    luckyKeepFraction: Float = 0.6f,
    starTrail: Boolean = false,
    darkFramePath: String? = null,
    brightnessBoost: Float = 1.5f,
    maxPreviewSize: Int = 0
): String?
