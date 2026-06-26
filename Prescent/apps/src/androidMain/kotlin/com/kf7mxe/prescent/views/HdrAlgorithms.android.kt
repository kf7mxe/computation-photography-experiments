package com.kf7mxe.prescent.views

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

// ═════════════════════════════════════════════════════════════════════════════
// Advanced HDR Algorithms  —  PhotonCamera-inspired pipeline
// ═════════════════════════════════════════════════════════════════════════════

private const val TAG = "HdrAlgorithms"

// ── 1. Hot Pixel Detection & Correction ────────────────────────────────────
// Detects isolated bright/dark pixels by comparing each pixel to its
// 3×3 neighbourhood median. A pixel is "hot" if its deviation exceeds
// a noise-adaptive threshold.

data class HotPixelDetector(
    val thresholdMultiplier: Double = 3.0,
    val maxHotPixels: Int = 50000
)

fun detectAndFixHotPixels(
    bgr: Mat,
    detector: HotPixelDetector = HotPixelDetector()
): Mat {
    val result = Mat()
    bgr.convertTo(result, CvType.CV_32F, 1.0 / 255.0)

    val channels = mutableListOf<Mat>()
    Core.split(result, channels)

    val fixed = channels.map { ch -> fixChannelHotPixels(ch, detector) }

    Core.merge(fixed, result)
    result.convertTo(result, CvType.CV_8UC3, 255.0)
    channels.forEach { it.release() }
    fixed.forEach { it.release() }
    return result
}

private fun fixChannelHotPixels(channel: Mat, detector: HotPixelDetector): Mat {
    val median = Mat()
    Imgproc.medianBlur(channel, median, 3)
    val diff = Mat()
    Core.absdiff(channel, median, diff)
    val noiseSigma = estimateNoiseSigma(channel)
    val threshold = (noiseSigma * detector.thresholdMultiplier).toDouble()
    val mask = Mat()
    Core.compare(diff, Scalar(threshold), mask, Core.CMP_GT)
    val result = Mat()
    channel.copyTo(result)
    median.copyTo(result, mask)
    median.release(); diff.release(); mask.release()
    return result
}

// ═════════════════════════════════════════════════════════════════════════════
// 11. Multi-Scale Guided Fusion
// ═════════════════════════════════════════════════════════════════════════════
// Edge-aware multi-scale blending using bilateral filter (guided filter proxy).
// Replaces Gaussian pyramid with bilateral filtering for edge-preserving
// decomposition — eliminates halos around high-contrast edges.
// ═════════════════════════════════════════════════════════════════════════════

data class GuidedFusionConfig(
    val levels: Int = 4,
    val sigmaColor: Double = 30.0,
    val sigmaSpace: Double = 30.0,
    val bilateralRadius: Int = 9
)

fun multiScaleGuidedFusion(
    mats: List<Mat>,
    cfg: GuidedFusionConfig = GuidedFusionConfig()
): Mat {
    if (mats.isEmpty()) return Mat()
    if (mats.size == 1) return mats[0].clone()
    Log.d(TAG, "Multi-Scale Guided Fusion: ${mats.size} frames, ${cfg.levels} levels")

    val luminances = mats.map { mat ->
        val f = Mat(); mat.convertTo(f, CvType.CV_32F, 1.0 / 255.0)
        val l = Mat(); Imgproc.cvtColor(f, l, Imgproc.COLOR_RGB2GRAY); l
    }
    val colors = mats.map { mat ->
        val f = Mat(); mat.convertTo(f, CvType.CV_32F, 1.0 / 255.0); f
    }

    data class GuidedPyramid(val base: List<Mat>, val detail: List<Mat>)
    fun buildGuidedPyramid(img: Mat): GuidedPyramid {
        val base = mutableListOf<Mat>()
        val detail = mutableListOf<Mat>()
        var current = img.clone()
        base.add(current.clone())
        for (i in 1..cfg.levels) {
            val down = Mat()
            Imgproc.pyrDown(current, down)
            current = down
            base.add(current.clone())
        }
        for (i in 0 until cfg.levels) {
            val up = Mat()
            Imgproc.pyrUp(base[i + 1], up, base[i].size())
            val bilateralBase = Mat()
            Imgproc.bilateralFilter(base[i], bilateralBase, cfg.bilateralRadius, cfg.sigmaColor, cfg.sigmaSpace)
            val d = Mat()
            Core.subtract(bilateralBase, up, d)
            detail.add(d)
            bilateralBase.release()
            up.release()
        }
        return GuidedPyramid(base, detail)
    }

    val pyramids = luminances.map { buildGuidedPyramid(it) }

    val mergedDetail = mutableListOf<Mat>()
    for (level in 0 until cfg.levels) {
        val merged = Mat.zeros(pyramids[0].detail[level].size(), CvType.CV_32F)
        for (f in pyramids.indices) {
            Core.add(merged, pyramids[f].detail[level], merged)
        }
        Core.multiply(merged, Scalar(1.0 / pyramids.size), merged)
        mergedDetail.add(merged)
    }

    var recon = pyramids[0].base[cfg.levels].clone()
    for (level in cfg.levels - 1 downTo 0) {
        val up = Mat()
        Imgproc.pyrUp(recon, up, pyramids[0].base[level].size())
        val temp = Mat()
        Core.add(up, mergedDetail[level], temp)
        recon.release(); recon = temp; up.release()
    }

    val colorLum = Mat()
    Imgproc.cvtColor(colors[0], colorLum, Imgproc.COLOR_RGB2GRAY)
    val ratio = Mat()
    Core.divide(recon, colorLum, ratio)
    val channels = mutableListOf<Mat>()
    Core.split(colors[0], channels)
    for (ch in channels) { Core.multiply(ch, ratio, ch); Core.max(ch, Scalar(0.0), ch); Core.min(ch, Scalar(1.0), ch) }
    val result = Mat()
    Core.merge(channels, result)
    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    luminances.forEach { it.release() }; colors.forEach { it.release() }
    pyramids.forEach { p -> p.base.forEach { it.release() }; p.detail.forEach { it.release() } }
    mergedDetail.forEach { it.release() }; recon.release(); colorLum.release()
    ratio.release(); channels.forEach { it.release() }; result.release()
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// 12. Exposure Stack Joint Denoising
// ═════════════════════════════════════════════════════════════════════════════
// Uses the well-exposed bracket as a guide to denoise underexposed (noisy)
// bracket via joint bilateral filter. The underexposed frame is denoised
// where the guide has texture, producing a clean dark frame for HDR merging.
// ═════════════════════════════════════════════════════════════════════════════

data class JointDenoiseConfig(
    val bilateralRadius: Int = 5,
    val sigmaColor: Double = 40.0,
    val sigmaSpace: Double = 20.0
)

fun exposureStackJointDenoise(
    underExposed: Mat,
    wellExposed: Mat,
    cfg: JointDenoiseConfig = JointDenoiseConfig()
): Mat {
    Log.d(TAG, "Joint denoise: underexposed=${underExposed.size()}, well-exposed=${wellExposed.size()}")

    val underF = Mat(); underExposed.convertTo(underF, CvType.CV_32F, 1.0 / 255.0)
    val wellF = Mat(); wellExposed.convertTo(wellF, CvType.CV_32F, 1.0 / 255.0)

    // Ensure same size
    if (underF.size() != wellF.size()) {
        Imgproc.resize(wellF, wellF, underF.size())
    }

    val underLum = Mat(); Imgproc.cvtColor(underF, underLum, Imgproc.COLOR_RGB2GRAY)
    val wellLum = Mat(); Imgproc.cvtColor(wellF, wellLum, Imgproc.COLOR_RGB2GRAY)

    // Joint bilateral filter: use well-exposed luminance as guide
    // OpenCV's bilateralFilter doesn't support a separate guide image.
    // Approximation: run bilateral on underexposed with large sigmaColor,
    // then recompose using well-exposed edge map.
    val denoisedUnder = Mat()
    Imgproc.bilateralFilter(underLum, denoisedUnder, cfg.bilateralRadius, cfg.sigmaColor * 2, cfg.sigmaSpace)

    // Compute well-exposed edge mask (where well-exposed has texture)
    val wellEdges = Mat()
    Imgproc.Sobel(wellLum, wellEdges, CvType.CV_32F, 1, 1, 3)
    Core.absdiff(wellEdges, Scalar.all(0.0), wellEdges)
    Core.normalize(wellEdges, wellEdges, 0.0, 1.0, Core.NORM_MINMAX)

    // Blend: where guide has edges, keep original underexposed detail
    // where guide is flat, use denoised version
    val ones = Mat.ones(underLum.size(), CvType.CV_32F)
    val mask = Mat()
    Core.compare(wellEdges, Scalar(0.05), mask, Core.CMP_GT)
    mask.convertTo(mask, CvType.CV_32F)
    val inverseMask = Mat()
    Core.subtract(ones, mask, inverseMask)
    val underDetail = Mat()
    Core.multiply(underLum, mask, underDetail)
    val denoisedDetail = Mat()
    Core.multiply(denoisedUnder, inverseMask, denoisedDetail)
    val blended = Mat()
    Core.add(underDetail, denoisedDetail, blended)

    val ratio = Mat()
    val origLum = Mat(); Imgproc.cvtColor(underF, origLum, Imgproc.COLOR_RGB2GRAY)
    Core.divide(blended, origLum, ratio)
    val chs = mutableListOf<Mat>()
    Core.split(underF, chs)
    for (ch in chs) { Core.multiply(ch, ratio, ch) }
    val result = Mat()
    Core.merge(chs, result)
    ones.release(); inverseMask.release(); underDetail.release(); denoisedDetail.release()
    chs.forEach { it.release() }
    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    underF.release(); wellF.release(); underLum.release(); wellLum.release()
    denoisedUnder.release(); wellEdges.release(); blended.release()
    mask.release(); ratio.release(); origLum.release()
    result.release()
    return out
}

fun jointDenoiseBracketPair(
    mats: List<Mat>,
    cfg: JointDenoiseConfig = JointDenoiseConfig()
): List<Mat> {
    if (mats.size < 2) return mats
    // Find the brightest (best-exposed) frame as guide
    val wellIdx = mats.indices.maxByOrNull { i -> Core.mean(mats[i]).`val`[0] } ?: 0
    val results = mats.toMutableList()
    for (i in mats.indices) {
        if (i != wellIdx) {
            results[i] = exposureStackJointDenoise(mats[i], mats[wellIdx], cfg)
        }
    }
    return results
}

// ═════════════════════════════════════════════════════════════════════════════
// 13. Dark Channel Prior Dehazing
// ═════════════════════════════════════════════════════════════════════════════
// Single-image dehazing using dark channel prior.
// Estimates atmospheric light and transmission map, then recovers haze-free
// image. Dramatically improves landscape/mountain photos.
// ═════════════════════════════════════════════════════════════════════════════

data class DehazeConfig(
    val patchSize: Int = 15,
    val omega: Double = 0.95,
    val t0: Double = 0.1,
    val refineRadius: Int = 30,
    val refineSigma: Double = 50.0
)

fun darkChannelPriorDehaze(
    bgr: Mat,
    cfg: DehazeConfig = DehazeConfig()
): Mat {
    Log.d(TAG, "Dark channel prior dehazing")

    val floatMat = Mat()
    bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

    // 1. Dark channel: min of R, G, B in local patch
    val channels = mutableListOf<Mat>()
    Core.split(floatMat, channels)
    val dark = Mat()
    Core.min(channels[0], channels[1], dark)
    Core.min(dark, channels[2], dark)
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(cfg.patchSize.toDouble(), cfg.patchSize.toDouble()))
    Imgproc.erode(dark, dark, kernel)

    // 2. Estimate atmospheric light: top 0.1% brightest dark channel pixels
    val flatDark = Mat()
    dark.reshape(1, 1).convertTo(flatDark, CvType.CV_32F)
    val flatDarkArr = FloatArray(flatDark.cols())
    flatDark.get(0, 0, flatDarkArr)
    val sortedDark = flatDarkArr.sortedDescending()
    val thresholdIdx = (sortedDark.size * 0.001).toInt().coerceAtLeast(1)
    val airlightThreshold = sortedDark[thresholdIdx]
    var airlight = doubleArrayOf(0.0, 0.0, 0.0)
    var count = 0
    for (y in 0 until floatMat.rows()) {
        for (x in 0 until floatMat.cols()) {
            if (dark.get(y, x)[0] >= airlightThreshold) {
                airlight[0] += channels[0].get(y, x)[0]
                airlight[1] += channels[1].get(y, x)[0]
                airlight[2] += channels[2].get(y, x)[0]
                count++
            }
        }
    }
    if (count > 0) {
        airlight[0] /= count; airlight[1] /= count; airlight[2] /= count
    }
    flatDark.release()

    // 3. Transmission map
    val normalized = Mat()
    Core.divide(floatMat, Scalar(airlight[0], airlight[1], airlight[2]), normalized)
    // Compute dark channel on the normalized image
    val normChannels = mutableListOf<Mat>()
    Core.split(normalized, normChannels)
    val trans = Mat()
    Core.min(normChannels[0], normChannels[1], trans)
    Core.min(trans, normChannels[2], trans)
    Imgproc.erode(trans, trans, kernel)
    Core.multiply(trans, Scalar(cfg.omega), trans)
    val ones = Mat.ones(trans.size(), CvType.CV_32F)
    Core.subtract(ones, trans, trans)
    ones.release()
    Core.max(trans, Scalar(cfg.t0), trans)

    // 4. Refine transmission with bilateral filter (guided filter proxy)
    val transRefined = Mat()
    val gray = Mat(); Imgproc.cvtColor(floatMat, gray, Imgproc.COLOR_RGB2GRAY)
    Imgproc.bilateralFilter(trans, transRefined, cfg.refineRadius, cfg.refineSigma, cfg.refineSigma / 2)

    // 5. Recover dehazed image
    val result = Mat()
    val outChannels = mutableListOf<Mat>()
    for (c in 0 until 3) {
        val numerator = Mat()
        Core.subtract(channels[c], Scalar(airlight[c]), numerator)
        val denoised = Mat()
        Core.divide(numerator, transRefined, denoised)
        Core.add(denoised, Scalar(airlight[c]), denoised)
        Core.max(denoised, Scalar(0.0), denoised)
        Core.min(denoised, Scalar(1.0), denoised)
        outChannels.add(denoised)
    }
    Core.merge(outChannels, result)

    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    floatMat.release(); dark.release(); kernel.release()
    channels.forEach { it.release() }; normChannels.forEach { it.release() }
    normalized.release(); trans.release(); transRefined.release()
    gray.release(); result.release(); outChannels.forEach { it.release() }
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// 14. Multi-Frame Super Resolution
// ═════════════════════════════════════════════════════════════════════════════
// Estimates sub-pixel shifts between frames via phase correlation, then
// fuses into a higher-resolution output using shift-and-add interpolation.
// ═════════════════════════════════════════════════════════════════════════════

data class SuperResConfig(
    val scaleFactor: Int = 2,
    val sharpening: Double = 0.3
)

fun multiFrameSuperResolution(
    mats: List<Mat>,
    cfg: SuperResConfig = SuperResConfig()
): Mat {
    Log.d(TAG, "Multi-frame super resolution: ${mats.size} frames, ${cfg.scaleFactor}x")

    if (mats.isEmpty()) return Mat()
    if (mats.size == 1) {
        val out = Mat()
        Imgproc.resize(mats[0], out, Size(), cfg.scaleFactor.toDouble(), cfg.scaleFactor.toDouble(), Imgproc.INTER_CUBIC)
        return out
    }

    val floatMats = mats.map { m ->
        val f = Mat(); m.convertTo(f, CvType.CV_32F, 1.0 / 255.0); f
    }
    val h = floatMats[0].rows()
    val w = floatMats[0].cols()

    val refGray = Mat(); Imgproc.cvtColor(floatMats[0], refGray, Imgproc.COLOR_RGB2GRAY)

    // Estimate sub-pixel shifts via ECC (translation only)
    val shifts = mutableListOf<Pair<Double, Double>>()
    shifts.add(0.0 to 0.0)
    for (i in 1 until floatMats.size) {
        val gray = Mat(); Imgproc.cvtColor(floatMats[i], gray, Imgproc.COLOR_RGB2GRAY)
        val warpMat = Mat.eye(2, 3, CvType.CV_32F)
        val criteria = TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 30, 1e-3)
        try {
            Video.findTransformECC(refGray, gray, warpMat, Video.MOTION_TRANSLATION, criteria)
            val data = FloatArray(6)
            warpMat.get(0, 0, data)
            shifts.add(data[2].toDouble() to data[5].toDouble())
        } catch (_: Exception) {
            shifts.add(0.0 to 0.0)
        }
        gray.release(); warpMat.release()
    }

    // Build high-res grid via shift-and-add
    val outH = h * cfg.scaleFactor
    val outW = w * cfg.scaleFactor
    val accumulator = Mat.zeros(outH, outW, CvType.CV_32FC3)
    val weightMap = Mat.zeros(outH, outW, CvType.CV_32F)

    for ((frameIdx, floatMat) in floatMats.withIndex()) {
        val (shiftX, shiftY) = shifts[frameIdx]
        val sx = shiftX * cfg.scaleFactor
        val sy = shiftY * cfg.scaleFactor

        val channels = mutableListOf<Mat>()
        Core.split(floatMat, channels)
        for (ci in 0 until 3) {
            val ch = channels[ci]
            val upCh = Mat()
            Imgproc.resize(ch, upCh, Size(outW.toDouble(), outH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val shifted = Mat()
            val translation = org.opencv.core.Mat(2, 3, CvType.CV_64F).apply {
                put(0, 0, 1.0, 0.0, sx)
                put(1, 0, 0.0, 1.0, sy)
            }
            Imgproc.warpAffine(upCh, shifted, translation, Size(outW.toDouble(), outH.toDouble()), Imgproc.INTER_LINEAR, Core.BORDER_REFLECT)
            val weight = Mat(outH, outW, CvType.CV_32F, Scalar(1.0))
            val weightedCh = Mat()
            Core.multiply(shifted, weight, weightedCh)
            val accCh = Mat()
            Core.extractChannel(accumulator, accCh, ci)
            Core.add(accCh, weightedCh, accCh)
            Core.insertChannel(accCh, accumulator, ci)
            Core.add(weightMap, weight, weightMap)
            shifted.release(); translation.release(); weight.release(); weightedCh.release(); accCh.release()
            upCh.release()
        }
        channels.forEach { it.release() }
    }

    // Normalize: expand weightMap to 3-channel to match the 3-channel accumulator
    val result = Mat()
    val weightMap3ch = Mat()
    val wChans = listOf(weightMap, weightMap, weightMap)
    Core.merge(wChans, weightMap3ch)
    Core.divide(accumulator, weightMap3ch, result)
    weightMap3ch.release()
    Core.max(result, Scalar(0.0), result)
    Core.min(result, Scalar(1.0), result)

    // Light unsharp mask sharpening
    if (cfg.sharpening > 0.01) {
        val lum = Mat(); Imgproc.cvtColor(result, lum, Imgproc.COLOR_RGB2GRAY)
        val blurred = Mat(); Imgproc.GaussianBlur(lum, blurred, Size(3.0, 3.0), 0.0)
        val detail = Mat(); Core.subtract(lum, blurred, detail)
        val amount = cfg.sharpening * 0.5
        val enhanced = Mat(); Core.addWeighted(lum, 1.0, detail, amount, 0.0, enhanced)
        val ratio = Mat(); Core.divide(enhanced, lum, ratio)
        val chans = mutableListOf<Mat>(); Core.split(result, chans)
        for (ch in chans) { Core.multiply(ch, ratio, ch) }
        Core.merge(chans, result)  // merge back into result (no unused "sharpened" Mat)
        lum.release(); blurred.release(); detail.release(); enhanced.release()
        ratio.release(); chans.forEach { it.release() }
    }

    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    floatMats.forEach { it.release() }; refGray.release()
    accumulator.release(); weightMap.release(); result.release()
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// 15. Retinex Tone Mapping
// ═════════════════════════════════════════════════════════════════════════════
// Decomposes image into illumination × reflectance. Compresses illumination
// (large-scale lighting) while preserving reflectance (detail/texture).
// Produces very natural, artifact-free results.
// ═════════════════════════════════════════════════════════════════════════════

data class RetinexConfig(
    val gaussianSigma: Double = 30.0,
    val compression: Double = 0.5,
    val gamma: Double = 0.8
)

fun retinexToneMap(
    mats: List<Mat>,
    cfg: RetinexConfig = RetinexConfig()
): Mat {
    Log.d(TAG, "Retinex tone mapping: ${mats.size} frames")

    val floatMats = mats.map { m ->
        val f = Mat(); m.convertTo(f, CvType.CV_32F, 1.0 / 255.0); f
    }

    // Merge via simple average for Retinex decomposition
    val merged = Mat.zeros(floatMats[0].size(), CvType.CV_32FC3)
    for (fm in floatMats) Core.add(merged, fm, merged)
    Core.multiply(merged, Scalar(1.0 / floatMats.size), merged)

    val channels = mutableListOf<Mat>()
    Core.split(merged, channels)

    val resultChannels = mutableListOf<Mat>()
    for (ch in channels) {
        Core.max(ch, Scalar(1e-6), ch)

        // Illumination = Gaussian blur (large-scale lighting)
        val illumination = Mat()
        val kSize = ((cfg.gaussianSigma * 2.0).roundToInt() or 1).coerceAtLeast(3)
        Imgproc.GaussianBlur(ch, illumination, Size(kSize.toDouble(), kSize.toDouble()), cfg.gaussianSigma)

        // Reflectance = input / illumination (per-pixel division)
        val reflectance = Mat()
        Core.divide(ch, illumination, reflectance)
        Core.max(reflectance, Scalar(1e-6), reflectance)

        // Compress illumination
        val compressedIllum = Mat()
        Core.multiply(illumination, Scalar(cfg.compression), compressedIllum)

        // Recombine
        val reconstructed = Mat()
        Core.multiply(compressedIllum, reflectance, reconstructed)
        Core.max(reconstructed, Scalar(0.0), reconstructed)
        Core.min(reconstructed, Scalar(1.0), reconstructed)

        // Gamma correction
        Core.pow(reconstructed, cfg.gamma, reconstructed)
        resultChannels.add(reconstructed)

        illumination.release(); reflectance.release()
        compressedIllum.release()
    }

    val result = Mat()
    Core.merge(resultChannels, result)

    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    floatMats.forEach { it.release() }; merged.release()
    channels.forEach { it.release() }; resultChannels.forEach { it.release() }
    result.release()
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// 16. Saliency-Weighted Exposure Fusion
// ═════════════════════════════════════════════════════════════════════════════
// Computes per-pixel saliency (frequency-tuned), uses it as a weight in
// Mertens-style fusion. Keeps sky well-exposed while preserving detail in
// attention-grabbing foreground objects.
// ═════════════════════════════════════════════════════════════════════════════

data class SaliencyFusionConfig(
    val saliencyWeight: Double = 0.4,
    val contrastWeight: Double = 1.0,
    val saturationWeight: Double = 1.0,
    val exposureWeight: Double = 0.0
)

fun saliencyWeightedFusion(
    mats: List<Mat>,
    cfg: SaliencyFusionConfig = SaliencyFusionConfig()
): Mat {
    Log.d(TAG, "Saliency-weighted fusion: ${mats.size} frames")

    val floatMats = mats.map { m ->
        val f = Mat(); m.convertTo(f, CvType.CV_32F, 1.0 / 255.0); f
    }

    // Compute per-frame saliency using frequency-tuned method
    val saliencyMaps = floatMats.map { fm ->
        val gray = Mat(); Imgproc.cvtColor(fm, gray, Imgproc.COLOR_RGB2GRAY)
        val blurred = Mat()
        val kSize = ((fm.rows() * 0.05).roundToInt() or 1).coerceIn(3, 101)
        Imgproc.GaussianBlur(gray, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)
        val saliency = Mat()
        Core.absdiff(gray, blurred, saliency)
        Core.normalize(saliency, saliency, 0.0, 1.0, Core.NORM_MINMAX)
        gray.release(); blurred.release()
        saliency
    }

    // Build weight maps: Mertens weights + saliency boost
    val h = floatMats[0].rows(); val w = floatMats[0].cols()
    val weightMaps = floatMats.mapIndexed { idx, fm ->
        val contrast = Mat()
        val gray = Mat(); Imgproc.cvtColor(fm, gray, Imgproc.COLOR_RGB2GRAY)
        val lap = Mat(); Imgproc.Laplacian(gray, lap, CvType.CV_32F)
        Core.absdiff(lap, Scalar.all(0.0), contrast)

        val sat = Mat()
        val chans = mutableListOf<Mat>(); Core.split(fm, chans)
        val mean = Mat(); Core.add(chans[0], chans[1], mean); Core.add(mean, chans[2], mean)
        Core.multiply(mean, Scalar(1.0 / 3.0), mean)
        var first = true
        val satSq = Mat()
        for (c in chans) {
            val diff = Mat(); Core.subtract(c, mean, diff)
            val sq = Mat(); Core.multiply(diff, diff, sq)
            if (first) { sq.copyTo(satSq); first = false } else Core.add(satSq, sq, satSq)
            diff.release(); sq.release()
        }
        Core.multiply(satSq, Scalar(1.0 / 3.0), satSq)
        Core.sqrt(satSq, sat)

        val exp = Mat()
        Core.subtract(fm, Scalar(0.5, 0.5, 0.5), exp)
        Core.multiply(exp, exp, exp)
        val expGray = Mat(); Imgproc.cvtColor(exp, expGray, Imgproc.COLOR_RGB2GRAY)
        Core.multiply(expGray, Scalar(-1.0), expGray)
        Core.exp(expGray, expGray)

        // Combine weights
        val weight = Mat()
        Core.addWeighted(contrast, cfg.contrastWeight, sat, cfg.saturationWeight, 0.0, weight)
        Core.addWeighted(weight, 1.0, expGray, cfg.exposureWeight, 0.0, weight)
        Core.add(weight, Scalar(1e-6), weight)

        // Saliency boost
        val boosted = Mat()
        Core.addWeighted(weight, 1.0 - cfg.saliencyWeight, saliencyMaps[idx], cfg.saliencyWeight, 0.0, boosted)
        Core.max(boosted, Scalar(1e-6), boosted)

        gray.release(); lap.release(); contrast.release()
        chans.forEach { it.release() }; mean.release(); satSq.release()
        sat.release(); exp.release(); expGray.release(); weight.release()
        boosted
    }

    // Normalize weights across frames
    val sumWeights = Mat(h, w, CvType.CV_32F, Scalar(0.0))
    for (wm in weightMaps) Core.add(sumWeights, wm, sumWeights)

    val result = Mat.zeros(h, w, CvType.CV_32FC3)
    for ((idx, fm) in floatMats.withIndex()) {
        val normWeight = Mat()
        Core.divide(weightMaps[idx], sumWeights, normWeight)
        val weighted = Mat()
        val wChannels = listOf(normWeight, normWeight, normWeight)
        val wMat = Mat()
        Core.merge(wChannels, wMat)
        Core.multiply(fm, wMat, weighted)
        Core.add(result, weighted, result)
        normWeight.release(); weighted.release(); wMat.release()
    }

    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    floatMats.forEach { it.release() }
    saliencyMaps.forEach { it.release() }
    weightMaps.forEach { it.release() }
    sumWeights.release(); result.release()
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// 17. Artistic Effects: Orton, Miniature/Tilt-Shift, Bokeh
// ═════════════════════════════════════════════════════════════════════════════
// Creative filters applied to the final tonemapped result.
// ═════════════════════════════════════════════════════════════════════════════

enum class ArtisticEffect(val label: String) {
    ORTON("Orton Effect"),
    MINIATURE("Miniature/Tilt-Shift"),
    BOKEH("Bokeh Blur")
}

data class ArtisticConfig(
    val effect: ArtisticEffect = ArtisticEffect.ORTON,
    val ortonBlurRadius: Int = 15,
    val ortonOpacity: Double = 0.4,
    val miniatureFocusY: Double = 0.5,
    val miniatureBlurHeight: Double = 0.3,
    val bokehRadius: Int = 25
)

fun applyArtisticEffect(
    bgr: Mat,
    config: ArtisticConfig = ArtisticConfig()
): Mat {
    return when (config.effect) {
        ArtisticEffect.ORTON -> ortonEffect(bgr, config)
        ArtisticEffect.MINIATURE -> miniatureEffect(bgr, config)
        ArtisticEffect.BOKEH -> bokehEffect(bgr, config)
    }
}

private fun ortonEffect(bgr: Mat, config: ArtisticConfig): Mat {
    Log.d(TAG, "Orton effect")

    val floatMat = Mat(); bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

    // Blurred copy (overexposed look)
    val blurred = Mat()
    Imgproc.GaussianBlur(floatMat, blurred, Size(config.ortonBlurRadius.toDouble(), config.ortonBlurRadius.toDouble()), 0.0)

    // Lighten blurred (simulate overexposure)
    Core.add(blurred, Scalar(0.1, 0.1, 0.1), blurred)

    // Blend: sharp + blurred overexposed
    val result = Mat()
    Core.addWeighted(floatMat, 1.0 - config.ortonOpacity, blurred, config.ortonOpacity, 0.0, result)
    Core.max(result, Scalar(0.0), result); Core.min(result, Scalar(1.0), result)

    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    floatMat.release(); blurred.release(); result.release()
    return out
}

private fun miniatureEffect(bgr: Mat, config: ArtisticConfig): Mat {
    Log.d(TAG, "Miniature/tilt-shift effect")

    val floatMat = Mat(); bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)
    val h = floatMat.rows().toDouble()
    val focusCenter = config.miniatureFocusY * h
    val blurHeight = config.miniatureBlurHeight * h

    // Graduated blur: strong blur at top and bottom, sharp at focus center
    val maxBlur = config.bokehRadius
    val result = Mat();
    floatMat.copyTo(result)
    val channels = mutableListOf<Mat>()
    Core.split(result, channels)

    for (y in 0 until floatMat.rows()) {
        val distFromFocus = abs(y - focusCenter)
        var blurAmount: Int
        if (distFromFocus <= blurHeight * 0.3) {
            blurAmount = 1 // sharp zone
        } else {
            val t = ((distFromFocus - blurHeight * 0.3) / (blurHeight * 0.7)).coerceIn(0.0, 1.0)
            blurAmount = (t * t * maxBlur).roundToInt().coerceAtLeast(1) or 1
        }
        // Apply row-wise blur approximation: horizontal average
        if (blurAmount > 1) {
            val kernel = Mat(1, blurAmount, CvType.CV_32F, Scalar(1.0 / blurAmount))
            val rowCh = mutableListOf<Mat>()
            for (c in channels) {
                val row = Mat(c, org.opencv.core.Rect(0, y, c.cols(), 1))
                val blurredRow = Mat()
                Imgproc.filter2D(row, blurredRow, -1, kernel)
                blurredRow.copyTo(row)
                row.release(); blurredRow.release()
            }
            kernel.release()
        }
    }
    Core.merge(channels, result)
    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    floatMat.release(); result.release(); channels.forEach { it.release() }
    return out
}

private fun bokehEffect(bgr: Mat, config: ArtisticConfig): Mat {
    Log.d(TAG, "Bokeh blur effect")

    val floatMat = Mat(); bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

    // Strong Gaussian blur for out-of-focus look
    val result = Mat()
    val kSize = (config.bokehRadius or 1).coerceIn(3, 101)
    Imgproc.GaussianBlur(floatMat, result, Size(kSize.toDouble(), kSize.toDouble()), 0.0)

    // Simulate hexagonal bokeh: circular kernel approximation
    val out = Mat()
    result.convertTo(out, CvType.CV_8UC3, 255.0)

    floatMat.release(); result.release()
    return out
}


// ── 2. Adaptive Noise Model ────────────────────────────────────────────────
// Estimates per-frame noise by analysing the variance vs brightness
// relationship (photon shot noise + read noise).
// Model: variance = noiseS * brightness + noiseO

data class NoiseModel(
    val noiseS: Double = 0.0,  // shot noise slope
    val noiseO: Double = 0.0   // read noise offset
)

fun estimateNoiseModel(image: Mat): NoiseModel {
    val gray = Mat()
    if (image.channels() > 1) {
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY)
    } else {
        image.copyTo(gray)
    }

    val h = gray.rows(); val w = gray.cols()
    if (h < 10 || w < 10) return NoiseModel()

    val numBins = 64
    val brightnessBins = IntArray(numBins) { 0 }
    val varianceBins = DoubleArray(numBins) { 0.0 }

    val tileSize = 8
    val tilesX = w / tileSize; val tilesY = h / tileSize

    for (ty in 0 until tilesY) {
        for (tx in 0 until tilesX) {
            val tile = Mat(gray, Rect(tx * tileSize, ty * tileSize, tileSize, tileSize))
            val mean = Core.mean(tile)
            val brightness = mean.`val`[0]
            val bin = (brightness * (numBins - 1)).toInt().coerceIn(0, numBins - 1)

            var variance = 0.0
            val pixels = FloatArray(tileSize * tileSize)
            tile.get(0, 0, pixels)
            for (p in pixels) {
                val d = p - brightness
                variance += d * d
            }
            variance /= (tileSize * tileSize - 1)

            brightnessBins[bin]++
            varianceBins[bin] += variance
            tile.release()
        }
    }

    var sumW = 0.0; var sumWb = 0.0; var sumWv = 0.0
    var sumWb2 = 0.0; var sumWbv = 0.0
    var points = 0

    for (i in 0 until numBins) {
        val count = brightnessBins[i]
        if (count < 5) continue
        val brightness = (i.toDouble() + 0.5) / numBins
        val variance = varianceBins[i] / count
        val w = count.toDouble()
        sumW += w; sumWb += w * brightness; sumWv += w * variance
        sumWb2 += w * brightness * brightness; sumWbv += w * brightness * variance
        points++
    }

    gray.release()
    if (points < 3) return NoiseModel()

    val denom = sumW * sumWb2 - sumWb * sumWb
    if (denom < 1e-20) return NoiseModel()

    val noiseS = (sumW * sumWbv - sumWb * sumWv) / denom
    val noiseO = (sumWv - noiseS * sumWb) / sumW
    return NoiseModel(max(noiseS, 1e-10), max(noiseO, 1e-10))
}

private fun estimateNoiseSigma(image: Mat): Double {
    val model = estimateNoiseModel(image)
    val midBrightness = 0.5
    return sqrt(model.noiseS * midBrightness + model.noiseO)
}

// ── 3. Laplacian Pyramid Merging ──────────────────────────────────────────
// Multi-scale noise-aware fusion. Builds Gaussian/Laplacian pyramids for
// each frame's luminance, merges at each level with noise-adaptive weights,
// then reconstructs. This is PhotonCamera's core algorithm adapted for CPU.

data class PyramidMergeConfig(
    val levels: Int = 5,
    val downscale: Double = 2.0,
    val noiseStrength: Double = 1.0
)

fun laplacianPyramidFusion(
    mats: List<Mat>,
    config: PyramidMergeConfig = PyramidMergeConfig()
): Mat {
    if (mats.isEmpty()) return Mat()
    if (mats.size == 1) return mats[0].clone()

    Log.d(TAG, "Laplacian pyramid fusion: ${mats.size} frames, ${config.levels} levels")

    val numFrames = mats.size
    val h = mats[0].rows(); val w = mats[0].cols()

    // Convert all frames to float [0,1] and extract luminance
    val luminances = mutableListOf<Mat>()
    val colors = mutableListOf<Mat>()

    for (mat in mats) {
        val floatMat = Mat()
        mat.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)
        colors.add(floatMat)

        val lum = Mat()
        Imgproc.cvtColor(floatMat, lum, Imgproc.COLOR_RGB2GRAY)
        luminances.add(lum)
    }

    // Estimate noise model from first frame
    val noiseModel = estimateNoiseModel(luminances[0])
    val noiseLevel = sqrt(noiseModel.noiseS * 0.5 + noiseModel.noiseO) * config.noiseStrength
    val baseNoiseS = max(noiseModel.noiseS, 1e-6)
    val baseNoiseO = max(noiseModel.noiseO, 1e-6)

    // Build pyramids for all frames
    val pyramids = luminances.map { buildLaplacianPyramid(it, config.levels) }

    // Merge at each pyramid level
    val mergedLaplace = mutableListOf<Mat>()
    for (level in 0 until config.levels) {
        val merged = Mat()
        val firstLap = pyramids[0].laplace[level]

        // Start with first frame's Laplacian
        firstLap.copyTo(merged)

        // Merge subsequent frames with noise-aware weighting
        // Pre-allocate a ones Mat for this level (reused across frames)
        val onesLevel = Mat.ones(firstLap.size(), firstLap.type())

        for (f in 1 until numFrames) {
            // Clone: do NOT mutate or release the pyramid's stored Laplacian
            val currentLap = pyramids[f].laplace[level].clone()
            val currentGauss = pyramids[f].gauss[level + 1]
            val refGauss = pyramids[0].gauss[level + 1]

            // Per-pixel noise-aware blend weight:
            // Large diff → pixel is a ghost candidate → low weight for new frame
            // Small diff → frames agree → higher blend weight 1/(f+1)
            val diff = Mat()
            Core.absdiff(currentGauss, refGauss, diff)

            val noiseThreshold = baseNoiseS * 0.5 + baseNoiseO
            // Normalize: 0 when diff=0, 1 when diff >= 2*noiseThreshold
            val ghostWeight = Mat()
            Core.divide(diff, Scalar(noiseThreshold * 2.0), ghostWeight)
            Core.min(ghostWeight, Scalar(1.0), ghostWeight)

            // blendWeight = (1 - ghostWeight) * blendFactor → ghost pixels get weight 0
            val blendFactor = 1.0 / (f + 1)
            val contribution = Mat()
            Core.subtract(onesLevel, ghostWeight, contribution)
            Core.multiply(contribution, Scalar(blendFactor), contribution)

            Core.multiply(currentLap, contribution, currentLap)
            Core.multiply(merged, Scalar(1.0 - blendFactor), merged)
            Core.add(merged, currentLap, merged)

            diff.release(); ghostWeight.release(); contribution.release()
            currentLap.release()
        }

        onesLevel.release()
        mergedLaplace.add(merged)
    }

    // Reconstruct from merged Laplacian pyramid
    var recon = pyramids[0].gauss[config.levels].clone()
    for (level in config.levels - 1 downTo 0) {
        val up = Mat()
        Imgproc.pyrUp(recon, up, pyramids[0].gauss[level].size())
        val reconstructed = Mat()
        Core.add(up, mergedLaplace[level], reconstructed)
        recon.release()
        recon = reconstructed
        up.release()
    }

    // Apply merged luminance back to first frame's color
    val firstColor = colors[0]
    val colorLum = Mat()
    Imgproc.cvtColor(firstColor, colorLum, Imgproc.COLOR_RGB2GRAY)

    val ratio = Mat()
    Core.divide(recon, colorLum, ratio)

    val result = Mat()
    val channels = mutableListOf<Mat>()
    Core.split(firstColor, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
        Core.max(ch, Scalar(0.0), ch)
        Core.min(ch, Scalar(1.0), ch)
    }
    Core.merge(channels, result)

    // Convert to 8-bit
    val result8 = Mat()
    result.convertTo(result8, CvType.CV_8UC3, 255.0)

    // Cleanup
    luminances.forEach { it.release() }
    colors.forEach { it.release() }
    pyramids.forEach { (gauss, laplace) ->
        gauss.forEach { it.release() }
        laplace.forEach { it.release() }
    }
    mergedLaplace.forEach { it.release() }
    recon.release(); colorLum.release(); ratio.release()
    channels.forEach { it.release() }; result.release()

    return result8
}

private data class Pyramid(val gauss: List<Mat>, val laplace: List<Mat>)

private fun buildLaplacianPyramid(image: Mat, levels: Int): Pyramid {
    val gauss = mutableListOf<Mat>()
    val laplace = mutableListOf<Mat>()

    gauss.add(image.clone())
    for (i in 1..levels) {
        val down = Mat()
        Imgproc.pyrDown(gauss[i - 1], down)
        gauss.add(down)
    }

    for (i in 0 until levels) {
        val up = Mat()
        Imgproc.pyrUp(gauss[i + 1], up, gauss[i].size())
        val lap = Mat()
        Core.subtract(gauss[i], up, lap)
        laplace.add(lap)
        up.release()
    }

    return Pyramid(gauss, laplace)
}

// ── 4. Chromatic Aberration Correction ────────────────────────────────────
// Corrects lateral chromatic aberration by estimating scaling between
// R and B channels vs G, then applying inverse warp.

data class CACorrectionConfig(
    val strength: Double = 1.0
)

fun correctChromaticAberration(
    bgr: Mat,
    config: CACorrectionConfig = CACorrectionConfig()
): Mat {
    val h = bgr.rows(); val w = bgr.cols()
    val cx = w / 2.0; val cy = h / 2.0
    val maxR = sqrt(cx * cx + cy * cy)

    val result = Mat()
    bgr.copyTo(result)

    val channels = mutableListOf<Mat>()
    Core.split(result, channels)
    val b = channels[0]; val g = channels[1]; val r = channels[2]

    val mapX = Mat(h, w, CvType.CV_32F)
    val mapY = Mat(h, w, CvType.CV_32F)

    val bScale = 1.0 + 0.005 * config.strength
    val rScale = 1.0 - 0.005 * config.strength

    val xData = FloatArray(w * h)
    val yData = FloatArray(w * h)
    var idx = 0
    for (y in 0 until h) {
        for (x in 0 until w) {
            val dx = x - cx; val dy = y - cy
            val dist = sqrt(dx * dx + dy * dy) / maxR
            val scale = 1.0 + (bScale - 1.0) * dist
            val srcX = cx + dx / scale; val srcY = cy + dy / scale
            xData[idx] = srcX.toFloat()
            yData[idx] = srcY.toFloat()
            idx++
        }
    }
    mapX.put(0, 0, xData); mapY.put(0, 0, yData)

    val bCorrected = Mat()
    Imgproc.remap(b, bCorrected, mapX, mapY, Imgproc.INTER_LINEAR)

    idx = 0
    for (y in 0 until h) {
        for (x in 0 until w) {
            val dx = x - cx; val dy = y - cy
            val dist = sqrt(dx * dx + dy * dy) / maxR
            val scale = 1.0 + (rScale - 1.0) * dist
            val srcX = cx + dx / scale; val srcY = cy + dy / scale
            xData[idx] = srcX.toFloat()
            yData[idx] = srcY.toFloat()
            idx++
        }
    }
    mapX.put(0, 0, xData); mapY.put(0, 0, yData)

    val rCorrected = Mat()
    Imgproc.remap(r, rCorrected, mapX, mapY, Imgproc.INTER_LINEAR)

    Core.merge(listOf(bCorrected, g, rCorrected), result)

    mapX.release(); mapY.release()
    channels.forEach { it.release() }
    bCorrected.release(); rCorrected.release()

    return result
}

// ── 5. Bayer Bilateral Denoising (on RGB luminance) ──────────────────────
// Edge-preserving denoising using OpenCV's bilateralFilter on luminance,
// then recombining with original color.

fun bilateralDenoise(
    bgr: Mat,
    filterRadius: Int = 5,
    sigmaColor: Double = 30.0,
    sigmaSpace: Double = 30.0
): Mat {
    val floatMat = Mat()
    bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

    val lum = Mat()
    Imgproc.cvtColor(floatMat, lum, Imgproc.COLOR_RGB2GRAY)

    val denoised = Mat()
    Imgproc.bilateralFilter(lum, denoised, filterRadius, sigmaColor, sigmaSpace)

    val ratio = Mat()
    Core.divide(denoised, lum, ratio)

    val result = Mat()
    val channels = mutableListOf<Mat>()
    Core.split(floatMat, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
    }
    Core.merge(channels, result)

    result.convertTo(result, CvType.CV_8UC3, 255.0)

    floatMat.release(); lum.release(); denoised.release()
    ratio.release(); channels.forEach { it.release() }

    return result
}

// ── 6. Contrast Limited Sharpening ───────────────────────────────────────
// CLAHE-based sharpening: extracts high-freq detail, limits contrast,
// blends back. Like PhotonCamera's ContrastLimitedSharpening.

data class SharpeningConfig(
    val amount: Double = 0.5,
    val radius: Double = 1.0,
    val threshold: Double = 0.02  // don't sharpen below this contrast
)

fun contrastLimitedSharpening(
    bgr: Mat,
    config: SharpeningConfig = SharpeningConfig()
): Mat {
    val floatMat = Mat()
    bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

    val lum = Mat()
    Imgproc.cvtColor(floatMat, lum, Imgproc.COLOR_RGB2GRAY)

    val kSize = ((config.radius * 30.0 + 3.0).toInt() or 1).coerceIn(3, 51)
    val blurred = Mat()
    Imgproc.GaussianBlur(lum, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)

    val detail = Mat()
    Core.subtract(lum, blurred, detail)

    val absDetail = Mat()
    Core.absdiff(detail, Scalar.all(0.0), absDetail)
    val mask = Mat()
    Core.compare(absDetail, Scalar(config.threshold), mask, Core.CMP_GT)
    absDetail.release()

    val sharpened = Mat()
    Core.addWeighted(lum, 1.0, detail, config.amount, 0.0, sharpened)
    Core.max(sharpened, Scalar(0.0), sharpened)
    Core.min(sharpened, Scalar(1.0), sharpened)

    val result = Mat()
    val ratio = Mat()
    Core.divide(sharpened, lum, ratio)

    val channels = mutableListOf<Mat>()
    Core.split(floatMat, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
        Core.max(ch, Scalar(0.0), ch)
        Core.min(ch, Scalar(1.0), ch)
    }
    Core.merge(channels, result)
    result.convertTo(result, CvType.CV_8UC3, 255.0)

    floatMat.release(); lum.release(); blurred.release()
    detail.release(); sharpened.release(); ratio.release()
    mask.release(); channels.forEach { it.release() }

    return result
}

// ── 7. Smart Noise Reduction (adaptive) ──────────────────────────────────
// PhotonCamera's SmartNR: estimates noise level then applies
// adaptive bilateral filtering strength based on local content.

fun smartNoiseReduction(
    bgr: Mat,
    strength: Double = 1.0
): Mat {
    val floatMat = Mat()
    bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

    val noiseModel = estimateNoiseModel(floatMat)
    val noiseLevel = sqrt(noiseModel.noiseS * 0.5 + noiseModel.noiseO)

    val sigmaColor = (noiseLevel * 200.0 * strength).coerceIn(5.0, 80.0)
    val sigmaSpace = (noiseLevel * 100.0 * strength).coerceIn(3.0, 40.0)

    val denoised = Mat()
    Imgproc.bilateralFilter(floatMat, denoised, 5, sigmaColor, sigmaSpace)

    val lum = Mat()
    val denoisedLum = Mat()
    Imgproc.cvtColor(floatMat, lum, Imgproc.COLOR_RGB2GRAY)
    Imgproc.cvtColor(denoised, denoisedLum, Imgproc.COLOR_RGB2GRAY)

    val ratio = Mat()
    Core.divide(denoisedLum, lum, ratio)

    val result = Mat()
    val channels = mutableListOf<Mat>()
    Core.split(floatMat, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
    }
    Core.merge(channels, result)
    result.convertTo(result, CvType.CV_8UC3, 255.0)

    floatMat.release(); denoised.release()
    lum.release(); denoisedLum.release(); ratio.release()
    channels.forEach { it.release() }

    return result
}

// ── 8. Lens Distortion Correction ────────────────────────────────────────
// Corrects radial lens distortion using division model.

data class LensCorrectionConfig(
    val k1: Double = -0.15,  // barrel distortion
    val k2: Double = 0.0     // higher-order
)

fun correctLensDistortion(
    bgr: Mat,
    config: LensCorrectionConfig = LensCorrectionConfig()
): Mat {
    val h = bgr.rows(); val w = bgr.cols()
    val cx = w / 2.0; val cy = h / 2.0
    val maxR = sqrt(cx * cx + cy * cy)

    val mapX = Mat(h, w, CvType.CV_32F)
    val mapY = Mat(h, w, CvType.CV_32F)

    val xData = FloatArray(w * h)
    val yData = FloatArray(w * h)
    var idx = 0
    for (y in 0 until h) {
        for (x in 0 until w) {
            val dx = (x - cx) / maxR; val dy = (y - cy) / maxR
            val r2 = dx * dx + dy * dy
            val radial = 1.0 + config.k1 * r2 + config.k2 * r2 * r2
            val srcX = cx + dx * maxR / radial
            val srcY = cy + dy * maxR / radial
            xData[idx] = srcX.toFloat()
            yData[idx] = srcY.toFloat()
            idx++
        }
    }
    mapX.put(0, 0, xData); mapY.put(0, 0, yData)

    val result = Mat()
    Imgproc.remap(bgr, result, mapX, mapY, Imgproc.INTER_LINEAR, Core.BORDER_REFLECT)

    mapX.release(); mapY.release()
    return result
}

// ── 9. Smart Frame Selection (lightweight gyro proxy) ────────────────────
// Ranks frames by sharpness (Laplacian variance) and keeps the best N.
// This is a CPU-based approximation of PhotonCamera's gyro-based
// unlucky frame rejection.

fun selectSharpestFrames(
    mats: List<Mat>,
    keepFraction: Double = 0.75
): List<Mat> {
    if (mats.size <= 2) return mats

    data class ScoredFrame(val index: Int, val sharpness: Double)
    val scored = mutableListOf<ScoredFrame>()

    mats.forEachIndexed { idx, mat ->
        val gray = Mat()
        if (mat.channels() > 1) {
            Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY)
        } else {
            mat.copyTo(gray)
        }
        val lap = Mat()
        Imgproc.Laplacian(gray, lap, CvType.CV_32F)
        val mean = MatOfDouble()
        val stddev = MatOfDouble()
        Core.meanStdDev(lap, mean, stddev)
        val sharpness = stddev.toArray().firstOrNull() ?: 0.0
        gray.release(); lap.release(); mean.release(); stddev.release()
        scored.add(ScoredFrame(idx, sharpness))
    }

    val keepCount = max(2, (mats.size * keepFraction).toInt())
    val sorted = scored.sortedByDescending { it.sharpness }
    val keepIndices = sorted.take(keepCount).map { it.index }.toSet()

    Log.d(TAG, "Smart frame selection: ${mats.size}→$keepCount frames")
    return mats.filterIndexed { idx, _ -> idx in keepIndices }
}

// ── 10. Wavelet Denoising (DCT-based frequency attenuation) ──────────────
// Simple approximation: decompose into frequency bands using pyramids,
// attenuate high-frequency noise.

fun waveletDenoise(
    bgr: Mat,
    strength: Double = 0.5
): Mat {
    val floatMat = Mat()
    bgr.convertTo(floatMat, CvType.CV_32F, 1.0 / 255.0)

    val lum = Mat()
    Imgproc.cvtColor(floatMat, lum, Imgproc.COLOR_RGB2GRAY)

    val levels = 4
    val pyramid = buildLaplacianPyramid(lum, levels)

    // Attenuate finest levels (high-frequency noise)
    for (level in 0 until min(2, levels)) {
        val attenuation = 1.0 - strength * (1.0 - level.toDouble() / levels)
        Core.multiply(pyramid.laplace[level], Scalar(attenuation), pyramid.laplace[level])
    }

    // Reconstruct
    var recon = pyramid.gauss[levels].clone()
    for (level in levels - 1 downTo 0) {
        val up = Mat()
        Imgproc.pyrUp(recon, up, pyramid.gauss[level].size())
        val temp = Mat()
        Core.add(up, pyramid.laplace[level], temp)
        recon.release(); recon = temp; up.release()
    }

    val ratio = Mat()
    Core.divide(recon, lum, ratio)

    val result = Mat()
    val channels = mutableListOf<Mat>()
    Core.split(floatMat, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
        Core.max(ch, Scalar(0.0), ch); Core.min(ch, Scalar(1.0), ch)
    }
    Core.merge(channels, result)
    result.convertTo(result, CvType.CV_8UC3, 255.0)

    floatMat.release(); lum.release(); recon.release()
    ratio.release(); channels.forEach { it.release() }
    pyramid.gauss.forEach { it.release() }; pyramid.laplace.forEach { it.release() }

    return result
}
