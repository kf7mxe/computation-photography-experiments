package com.kf7mxe.prescent.views

expect suspend fun processPhotoSphere(
    images: List<String>,
    orientations: List<Pair<Float, Float>> = listOf(),
    maxPreviewSize: Int = 0
): String?

/**
 * Computes the visual rotation angle (degrees) between two adjacent sphere frames
 * using feature matching. Returns null if matching fails.
 * Used to correct gyro drift in the photo sphere ghost overlay.
 */
expect suspend fun computeVisualRotation(prevPath: String, currentPath: String): Float?
