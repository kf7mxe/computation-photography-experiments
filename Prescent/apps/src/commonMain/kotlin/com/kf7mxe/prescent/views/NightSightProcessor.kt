package com.kf7mxe.prescent.views

import com.lightningkite.kiteui.views.ElementWriter
import com.lightningkite.reactive.core.Signal

expect suspend fun processNightSight(
    images: List<String>,
    starTrail: Boolean = false,
    darkFramePath: String? = null,
    brightnessBoost: Float = 1.5f,
    maxPreviewSize: Int = 0
): String?
