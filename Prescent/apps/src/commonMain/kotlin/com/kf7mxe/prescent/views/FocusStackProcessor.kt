package com.kf7mxe.prescent.views

expect suspend fun processFocusStack(
    images: List<String>,
    maxPreviewSize: Int = 0
): String?
