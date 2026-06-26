package com.kf7mxe.prescent.views

enum class QuadBayerAlgorithm(val label: String, val description: String) {
    BIN_TO_BAYER("Bin-to-Bayer", "Average 2×2 same-color clusters for best SNR, half resolution"),
    FULL_REMOSAIC("Full Remosaic", "One pixel per 2×2 cluster preserves detail, full resolution"),
    EDGE_GUIDED("Edge-Guided", "Adaptive per cluster: bin in smooth areas, remosaic at edges")
}

data class QuadBayerOptions(
    val algorithm: QuadBayerAlgorithm = QuadBayerAlgorithm.FULL_REMOSAIC,
    val frameCount: Int = 2,
    val denoise: Boolean = true,
    val pipeToHdr: Boolean = false,
    val pipeToNightSight: Boolean = false,
    val saveDng: Boolean = false,
    val smartSelection: Boolean = false
)

expect suspend fun processQuadBayer(
    rawFiles: List<String>,
    options: QuadBayerOptions = QuadBayerOptions(),
    maxPreviewSize: Int = 0
): String?
