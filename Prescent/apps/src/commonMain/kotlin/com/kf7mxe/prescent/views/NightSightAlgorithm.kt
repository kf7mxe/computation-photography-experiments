package com.kf7mxe.prescent.views

enum class NightSightAlgorithm(
    val label: String,
    val description: String,
    val supportsLuckyPreFilter: Boolean = true
) {
    AVERAGE("Average", "Mean of aligned frames — simple noise reduction"),
    MEDIAN("Median", "Element-wise median — rejects outliers (moving objects, people)"),
    LAPLACIAN("Laplacian Pyramid", "Multi-scale pyramid blend — preserves detail at all frequencies", supportsLuckyPreFilter = false),
    HDR_MERGE("HDR Merge", "Debevec radiance map + Reinhard tone-map — handles varied exposure", supportsLuckyPreFilter = false)
}
