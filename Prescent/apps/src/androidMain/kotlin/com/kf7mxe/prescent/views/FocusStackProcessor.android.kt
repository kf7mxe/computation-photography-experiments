package com.kf7mxe.prescent.views

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import com.lightningkite.kiteui.views.AndroidAppContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.calib3d.Calib3d
import org.opencv.core.*
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import java.io.File
import java.io.FileOutputStream
import kotlin.math.*

private const val TAG = "FocusStack"

actual suspend fun processFocusStack(
    images: List<String>,
    maxPreviewSize: Int,
    algorithm: String,
    alignmentMethod: String,
    exposureBalance: Boolean,
    showDepthMap: Boolean,
    refocusDepth: Float,
    focalLength: Float,
    aperture: Float,
    focusDistanceMeters: Float,
    hdrHybridFramesPerFocus: Int,
    pyramidLevels: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val isPreview = maxPreviewSize > 0
    Log.d(TAG, "Stack: ${images.size} frames, algo=$algorithm, align=$alignmentMethod")

    if (images.size < 2) return@withContext null

    if (algorithm == "Focus Bracketing Optimizer") {
        val msg = focusBracketingOptimizer(
            focalLength = focalLength,
            aperture = aperture,
            focusDistance = focusDistanceMeters,
            frameCount = images.size,
            stepsProvided = images.size
        )
        Log.d(TAG, msg)
        return@withContext null
    }

    // Decode
    val frameMats = mutableListOf<Mat>()
    for (path in images) {
        val opts = BitmapFactory.Options()
        if (isPreview) {
            opts.inJustDecodeBounds = true
            BitmapFactory.decodeFile(path, opts)
            val w = opts.outWidth; val h = opts.outHeight
            var s = 1; while (w / s > maxPreviewSize || h / s > maxPreviewSize) s *= 2
            opts.inSampleSize = s; opts.inJustDecodeBounds = false
        }
        val bmp = BitmapFactory.decodeFile(path, opts) ?: continue
        val rgba = Mat()
        Utils.bitmapToMat(bmp, rgba); bmp.recycle()
        val rgb = Mat()
        Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
        rgba.release()
        frameMats.add(rgb)
    }
    if (frameMats.size < 2) return@withContext null

    // Exposure balance (pre-alignment)
    val balancedMats = if (exposureBalance) {
        exposureBalanceFrames(frameMats)
    } else frameMats

    // Alignment
    val alignedFrames = alignFocusFrames(balancedMats, alignmentMethod)

    // Parse refocusDepth for interactive refocus
    val depthFraction = refocusDepth.toDouble().coerceIn(0.0, 1.0)

    val result8 = when (algorithm) {
        "Depth Map" -> generateDepthMap(alignedFrames, showDepthMap)
        "Interactive Refocus" -> interactiveRefocus(alignedFrames, depthFraction)
        "Exposure Balanced" -> standardFocusStack(alignedFrames, pyramidLevels, gaussianBlurWeightMaps = true)
        "Feature Align", "Feature" -> standardFocusStack(alignedFrames, pyramidLevels, gaussianBlurWeightMaps = true)
        "HDR Hybrid" -> if (hdrHybridFramesPerFocus > 1) {
            hdrHybridFocusStack(alignedFrames, hdrHybridFramesPerFocus)
        } else {
            Log.w(TAG, "HDR hybrid needs hdrHybridFramesPerFocus > 1, using standard stack")
            standardFocusStack(alignedFrames, pyramidLevels, gaussianBlurWeightMaps = true)
        }
        else -> multiScalePyramidStack(alignedFrames, pyramidLevels)
    }

    val bitmap = Bitmap.createBitmap(result8.cols(), result8.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(result8, bitmap)
    result8.release()

    frameMats.forEach { it.release() }
    // Balanced mats may share references with originals; only release unique ones
    balancedMats.forEach { bm ->
        if (frameMats.none { it === bm }) bm.release()
    }
    alignedFrames.forEach { it.release() }

    val filename = "focusstack_${System.currentTimeMillis()}.jpg"
    val savedPath = if (isPreview) {
        val f = File(context.cacheDir, filename)
        FileOutputStream(f).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        bitmap.recycle()
        f.absolutePath
    } else {
        saveToGallery(context, bitmap, "focusstack")
        bitmap.recycle()
        "$filename"
    }
    Log.d(TAG, "Complete: $savedPath")
    savedPath
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. Multi-Scale Pyramid Stack (Laplacian pyramid per frame, select per level)
// ═════════════════════════════════════════════════════════════════════════════

data class PyramidStack(val gauss: List<Mat>, val laplace: List<Mat>)

private fun buildPyramid(img: Mat, levels: Int): PyramidStack {
    val g = mutableListOf<Mat>()
    val l = mutableListOf<Mat>()
    g.add(img.clone())
    for (i in 1..levels) {
        val down = Mat(); Imgproc.pyrDown(g[i - 1], down); g.add(down)
    }
    for (i in 0 until levels) {
        val up = Mat(); Imgproc.pyrUp(g[i + 1], up, g[i].size())
        val lap = Mat(); Core.subtract(g[i], up, lap); l.add(lap); up.release()
    }
    return PyramidStack(g, l)
}

private fun multiScalePyramidStack(frames: List<Mat>, levels: Int): Mat {
    Log.d(TAG, "Multi-scale pyramid stack, $levels levels")
    val grays = frames.map { m -> val g = Mat(); Imgproc.cvtColor(m, g, Imgproc.COLOR_RGB2GRAY); g }
    val pyramids = grays.map { buildPyramid(it, levels) }

    // For each level, pick the sharpest Laplacian pixel across frames
    val mergedLaplace = mutableListOf<Mat>()
    for (level in 0 until levels) {
        val h = pyramids[0].laplace[level].rows()
        val w = pyramids[0].laplace[level].cols()
        val best = Mat.zeros(h, w, CvType.CV_32F)
        val bestIdx = Mat.zeros(h, w, CvType.CV_32S)

        for (fi in pyramids.indices) {
            val absLap = Mat()
            Core.absdiff(pyramids[fi].laplace[level], Scalar.all(0.0), absLap)
            val mask = Mat()
            Core.compare(absLap, best, mask, Core.CMP_GT)
            absLap.copyTo(best, mask)
            val idxMat = Mat(h, w, CvType.CV_32S, Scalar(fi.toDouble()))
            idxMat.copyTo(bestIdx, mask)
            absLap.release(); mask.release(); idxMat.release()
        }

        // Gather best Laplacian values
        val merged = Mat.zeros(h, w, CvType.CV_32F)
        for (fi in pyramids.indices) {
            val frameMask = Mat()
            Core.compare(bestIdx, Scalar(fi.toDouble()), frameMask, Core.CMP_EQ)
            frameMask.convertTo(frameMask, CvType.CV_32F, 1.0 / 255.0)
            val contribution = Mat()
            Core.multiply(pyramids[fi].laplace[level], frameMask, contribution)
            Core.add(merged, contribution, merged)
            frameMask.release(); contribution.release()
        }
        mergedLaplace.add(merged)
        best.release(); bestIdx.release()
    }

    // Reconstruct
    var recon = pyramids[0].gauss[levels].clone()
    for (level in levels - 1 downTo 0) {
        val up = Mat(); Imgproc.pyrUp(recon, up, pyramids[0].gauss[level].size())
        val temp = Mat(); Core.add(up, mergedLaplace[level], temp)
        recon.release(); recon = temp; up.release()
    }

    // Apply luminance ratio to color from first frame
    val firstColor = Mat(); frames[0].convertTo(firstColor, CvType.CV_32F, 1.0 / 255.0)
    val colorLum = Mat(); Imgproc.cvtColor(firstColor, colorLum, Imgproc.COLOR_RGB2GRAY)
    val ratio = Mat(); Core.divide(recon, colorLum, ratio)
    val ch = mutableListOf<Mat>(); Core.split(firstColor, ch)
    for (c in ch) { Core.multiply(c, ratio, c); Core.max(c, Scalar(0.0), c); Core.min(c, Scalar(1.0), c) }
    val resultF = Mat(); Core.merge(ch, resultF)
    val out = Mat(); resultF.convertTo(out, CvType.CV_8UC3, 255.0)

    grays.forEach { it.release() }
    pyramids.forEach { p -> p.gauss.forEach { it.release() }; p.laplace.forEach { it.release() } }
    mergedLaplace.forEach { it.release() }; recon.release()
    firstColor.release(); colorLum.release(); ratio.release()
    ch.forEach { it.release() }; resultF.release()
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. Depth Map Generation
// ═════════════════════════════════════════════════════════════════════════════

private fun generateDepthMap(frames: List<Mat>, showColor: Boolean): Mat {
    Log.d(TAG, "Generating depth map")
    val h = frames[0].rows(); val w = frames[0].cols()
    val grays = frames.map { m -> val g = Mat(); Imgproc.cvtColor(m, g, Imgproc.COLOR_RGB2GRAY); g }
    val numFrames = frames.size

    // For each pixel, find which frame has the max Laplacian response
    val laplacians = grays.map { g ->
        val lap = Mat(); Imgproc.Laplacian(g, lap, CvType.CV_32F)
        val absLap = Mat(); Core.absdiff(lap, Scalar.all(0.0), absLap); lap.release()
        Imgproc.GaussianBlur(absLap, absLap, Size(11.0, 11.0), 3.0)
        absLap
    }

    val depthIndex = Mat.zeros(h, w, CvType.CV_32F)
    val maxResp = Mat.zeros(h, w, CvType.CV_32F)
    for (fi in laplacians.indices) {
        val mask = Mat()
        Core.compare(laplacians[fi], maxResp, mask, Core.CMP_GT)
        val idxMat = Mat(h, w, CvType.CV_32F, Scalar((fi.toDouble() / (numFrames - 1).coerceAtLeast(1))))
        idxMat.copyTo(depthIndex, mask)
        laplacians[fi].copyTo(maxResp, mask)
        mask.release(); idxMat.release()
    }

    // Smooth depth map
    Imgproc.GaussianBlur(depthIndex, depthIndex, Size(31.0, 31.0), 8.0)

    if (!showColor) {
        // Grayscale depth map (white = near, black = far)
        val depth8 = Mat(); depthIndex.convertTo(depth8, CvType.CV_8UC1, 255.0)
        val out = Mat(); Imgproc.cvtColor(depth8, out, Imgproc.COLOR_GRAY2RGB)
        depth8.release()
        laplacians.forEach { it.release() }; grays.forEach { it.release() }
        depthIndex.release(); maxResp.release()
        return out
    }

    // Jet colormap
    val depth8 = Mat(); depthIndex.convertTo(depth8, CvType.CV_8UC1, 255.0)
    val colorMap = Mat(); Imgproc.applyColorMap(depth8, colorMap, Imgproc.COLORMAP_JET)
    depth8.release()
    laplacians.forEach { it.release() }; grays.forEach { it.release() }
    depthIndex.release(); maxResp.release()
    return colorMap
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. Interactive Refocus (blend weighted by distance from focal plane)
// ═════════════════════════════════════════════════════════════════════════════

private fun interactiveRefocus(frames: List<Mat>, focalDepth: Double): Mat {
    Log.d(TAG, "Interactive refocus at depth=$focalDepth")
    val numFrames = frames.size
    if (numFrames < 2) return frames[0].clone()

    val h = frames[0].rows(); val w = frames[0].cols()
    val grays = frames.map { m -> val g = Mat(); Imgproc.cvtColor(m, g, Imgproc.COLOR_RGB2GRAY); g }
    val laplacians = grays.map { g ->
        val lap = Mat(); Imgproc.Laplacian(g, lap, CvType.CV_32F)
        val absLap = Mat(); Core.absdiff(lap, Scalar.all(0.0), absLap); lap.release()
        Imgproc.GaussianBlur(absLap, absLap, Size(11.0, 11.0), 3.0)
        absLap
    }

    // Estimate per-pixel depth (same as depth map)
    val depthIndex = Mat.zeros(h, w, CvType.CV_32F)
    val maxResp = Mat.zeros(h, w, CvType.CV_32F)
    for (fi in laplacians.indices) {
        val mask = Mat()
        Core.compare(laplacians[fi], maxResp, mask, Core.CMP_GT)
        val idxMat = Mat(h, w, CvType.CV_32F, Scalar((fi.toDouble() / (numFrames - 1))))
        idxMat.copyTo(depthIndex, mask)
        laplacians[fi].copyTo(maxResp, mask)
        mask.release(); idxMat.release()
    }
    Imgproc.GaussianBlur(depthIndex, depthIndex, Size(25.0, 25.0), 6.0)

    // Weight each frame by Gaussian centered at focalDepth
    val sigma = 1.0 / (numFrames - 1).coerceAtLeast(1)
    val weights = mutableListOf<Mat>()
    var sumW = Mat.zeros(h, w, CvType.CV_32F)
    for (fi in frames.indices) {
        val idealDepth = fi.toDouble() / (numFrames - 1)
        val diff = Mat(); Core.subtract(depthIndex, Scalar(idealDepth), diff)
        val sq = Mat(); Core.multiply(diff, diff, sq)
        val wMat = Mat()
        val negHalfSigmaSq = -1.0 / (2.0 * sigma * sigma)
        Core.multiply(sq, Scalar(negHalfSigmaSq), wMat)
        Core.exp(wMat, wMat)
        Core.add(wMat, Scalar(1e-6), wMat)
        weights.add(wMat)
        Core.add(sumW, wMat, sumW)
        diff.release(); sq.release()
    }

    val resultF = Mat.zeros(h, w, CvType.CV_32FC3)
    for (fi in frames.indices) {
        val normW = Mat(); Core.divide(weights[fi], sumW, normW)
        val fFloat = Mat(); frames[fi].convertTo(fFloat, CvType.CV_32F, 1.0 / 255.0)
        val ch = mutableListOf<Mat>(); Core.split(fFloat, ch)
        for (c in ch) Core.multiply(c, normW, c)
        Core.merge(ch, fFloat)
        Core.add(resultF, fFloat, resultF)
        normW.release(); fFloat.release(); ch.forEach { it.release() }
    }
    Core.min(resultF, Scalar(1.0), resultF)
    val out = Mat(); resultF.convertTo(out, CvType.CV_8UC3, 255.0)

    grays.forEach { it.release() }; laplacians.forEach { it.release() }
    depthIndex.release(); maxResp.release()
    weights.forEach { it.release() }; sumW.release(); resultF.release()
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. Exposure-Balanced Focus Stack
// ═════════════════════════════════════════════════════════════════════════════

private fun exposureBalanceFrames(frames: MutableList<Mat>): MutableList<Mat> {
    Log.d(TAG, "Exposure balancing ${frames.size} frames")
    val means = frames.map { Core.mean(it).`val`[0] }
    val targetMean = means.sorted()[means.size / 2] // median brightness
    val result = mutableListOf<Mat>()
    for ((i, m) in frames.withIndex()) {
        val f = Mat(); m.convertTo(f, CvType.CV_32F, 1.0 / 255.0)
        val scale = (targetMean / means[i].coerceAtLeast(1e-6)).coerceIn(0.5, 2.0)
        Core.multiply(f, Scalar(scale, scale, scale), f)
        Core.max(f, Scalar(0.0), f); Core.min(f, Scalar(1.0), f)
        val out = Mat(); f.convertTo(out, CvType.CV_8UC3, 255.0)
        f.release(); m.release()
        result.add(out)
    }
    return result
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. Scale-Invariant Feature Alignment (ORB + homography)
// ═════════════════════════════════════════════════════════════════════════════

private fun alignFocusFrames(frames: List<Mat>, method: String): List<Mat> {
    if (method == "None" || method == "Skip") return frames.map { it.clone() }
    if (method == "Feature" || method == "Feature Align") {
        Log.d(TAG, "Feature-based alignment")
        return alignFocusByFeature(frames)
    }
    // MTB alignment (default)
    Log.d(TAG, "MTB alignment")
    val aligner = Photo.createAlignMTB()
    val aligned = mutableListOf<Mat>().also { out -> frames.forEach { _ -> out.add(Mat()) } }
    return try {
        aligner.process(frames, aligned)
        val valid = aligned.filter { !it.empty() }
        if (valid.size == frames.size) aligned else frames.map { it.clone() }
    } catch (e: Exception) {
        Log.e(TAG, "MTB failed", e)
        frames.map { it.clone() }
    }
}

private fun alignFocusByFeature(frames: List<Mat>): List<Mat> {
    val refIdx = frames.size / 2
    val orb = ORB.create(3000)
    val refGray = Mat(); Imgproc.cvtColor(frames[refIdx], refGray, Imgproc.COLOR_RGB2GRAY)
    val refKp = MatOfKeyPoint(); val refDesc = Mat()
    orb.detectAndCompute(refGray, Mat(), refKp, refDesc)
    val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
    val result = mutableListOf<Mat>()
    frames.forEachIndexed { i, m ->
        if (i == refIdx) { result.add(m.clone()); return@forEachIndexed }
        val gray = Mat(); Imgproc.cvtColor(m, gray, Imgproc.COLOR_RGB2GRAY)
        val kp = MatOfKeyPoint(); val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        val matches = MatOfDMatch()
        matcher.match(refDesc, desc, matches)
        val arr = matches.toArray()
        val good = if (arr.size >= 8) {
            val minDist = arr.minOf { it.distance }
            arr.filter { it.distance <= max(3.0f * minDist, 30.0f) }
        } else emptyList()
        if (good.size >= 8) {
            val refArr = Array(good.size) { Point() }
            val imgArr = Array(good.size) { Point() }
            val rkp = refKp.toArray(); val ikp = kp.toArray()
            good.forEachIndexed { idx, dm ->
                refArr[idx] = rkp[dm.queryIdx].pt
                imgArr[idx] = ikp[dm.trainIdx].pt
            }
            val refPts = MatOfPoint2f().apply { fromArray(*refArr) }
            val imgPts = MatOfPoint2f().apply { fromArray(*imgArr) }
            val mask = MatOfByte()
            val h = Calib3d.findHomography(imgPts, refPts, Calib3d.RANSAC, 5.0, mask)
            if (h != null) {
                val warped = Mat(); Imgproc.warpPerspective(m, warped, h, frames[refIdx].size()); result.add(warped)
                h.release()
            } else { result.add(m.clone()) }
            refPts.release(); imgPts.release(); mask.release()
        } else { result.add(m.clone()) }
        matches.release(); desc.release(); kp.release(); gray.release()
    }
    refGray.release(); refKp.release(); refDesc.release()
    return result
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. HDR + Focus Hybrid
// ═════════════════════════════════════════════════════════════════════════════
// Groups frames by focus position (framesPerFocus per group), merges each
// group with Mertens exposure fusion, then focus-stacks the results.

private fun hdrHybridFocusStack(frames: List<Mat>, framesPerFocus: Int): Mat {
    Log.d(TAG, "HDR hybrid: ${frames.size} frames, $framesPerFocus per focus")
    val numPositions = frames.size / framesPerFocus
    if (numPositions < 2) return standardFocusStack(frames, 4, true)

    val hdrFrames = mutableListOf<Mat>()
    for (pos in 0 until numPositions) {
        val start = pos * framesPerFocus
        val end = min(start + framesPerFocus, frames.size)
        val bracket = frames.subList(start, end)
        if (bracket.size < 2) { hdrFrames.add(bracket[0].clone()); continue }
        val merged = Mat()
        val merger = Photo.createMergeMertens(1.0f, 1.0f, 0.0f)
        merger.process(bracket, merged)
        val merged8 = Mat(); merged.convertTo(merged8, CvType.CV_8UC3, 255.0)
        merged.release()
        hdrFrames.add(merged8)
    }
    Log.d(TAG, "HDR hybrid: ${hdrFrames.size} HDR frames to stack")
    return multiScalePyramidStack(hdrFrames, 4)
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. Focus Bracketing Optimizer
// ═════════════════════════════════════════════════════════════════════════════

private fun focusBracketingOptimizer(
    focalLength: Float,
    aperture: Float,
    focusDistance: Float,
    frameCount: Int,
    stepsProvided: Int
): String {
    // Circle of confusion: typical for APS-C ~0.02mm, FF ~0.03mm
    val coc = 0.02 // mm (APS-C)
    val fl = focalLength // mm
    val fStop = aperture
    val hyperfocal = (fl * fl) / (fStop * coc) / 1000.0 // meters
    val nearLimit = (hyperfocal * focusDistance) / (hyperfocal + (focusDistance - fl / 1000.0))
    val farLimit = (hyperfocal * focusDistance) / (hyperfocal - (focusDistance - fl / 1000.0)).coerceAtLeast(1e-6)

    val recommendedSteps = if (hyperfocal.isFinite() && hyperfocal > 0) {
        ceil((farLimit - nearLimit) / (hyperfocal * 0.3)).toInt().coerceIn(2, 40)
    } else 10

    val msg = buildString {
        append("Bracketing optimizer: ")
        append("hyperfocal=${"%.2f".format(hyperfocal)}m, ")
        append("near=${"%.2f".format(nearLimit)}m, far=${"%.2f".format(farLimit)}m, ")
        append("recommended=$recommendedSteps steps")
    }
    Log.d(TAG, msg)
    return msg
}

// ═════════════════════════════════════════════════════════════════════════════
// Standard single-scale Laplacian stacking (weighted blend)
// ═════════════════════════════════════════════════════════════════════════════

private fun standardFocusStack(
    frames: List<Mat>,
    pyramidLevels: Int,
    gaussianBlurWeightMaps: Boolean
): Mat {
    Log.d(TAG, "Standard Laplacian stack (${frames.size} frames)")
    val grays = frames.map { m -> val g = Mat(); Imgproc.cvtColor(m, g, Imgproc.COLOR_RGB2GRAY); g }
    val laplacians = grays.map { g ->
        val lap = Mat(); Imgproc.Laplacian(g, lap, CvType.CV_32F)
        val absLap = Mat(); Core.absdiff(lap, Scalar.all(0.0), absLap); lap.release()
        Core.add(absLap, Scalar(1e-6), absLap)
        if (gaussianBlurWeightMaps) Imgproc.GaussianBlur(absLap, absLap, Size(15.0, 15.0), 4.0)
        absLap
    }

    val sumW = Mat.zeros(frames[0].size(), CvType.CV_32F)
    for (w in laplacians) Core.add(sumW, w, sumW)
    val normWeights = laplacians.map { w ->
        val n = Mat(); Core.divide(w, sumW, n); n
    }

    val result = focusStackBlend(frames, normWeights)
    grays.forEach { it.release() }; laplacians.forEach { it.release() }
    sumW.release(); normWeights.forEach { it.release() }
    return result
}

// ═════════════════════════════════════════════════════════════════════════════
// Blend helper (weighted sum of float frames)
// ═════════════════════════════════════════════════════════════════════════════

private fun focusStackBlend(frames: List<Mat>, weights: List<Mat>): Mat {
    val h = frames[0].rows(); val w = frames[0].cols()
    val resultFloat = Mat.zeros(h, w, CvType.CV_32FC3)
    for (i in frames.indices) {
        val frameFloat = Mat(); frames[i].convertTo(frameFloat, CvType.CV_32F, 1.0 / 255.0)
        val ch = mutableListOf<Mat>(); Core.split(frameFloat, ch)
        for (c in 0 until 3) Core.multiply(ch[c], weights[i], ch[c])
        Core.merge(ch, frameFloat)
        Core.add(resultFloat, frameFloat, resultFloat)
        ch.forEach { it.release() }; frameFloat.release()
    }
    Core.min(resultFloat, Scalar(1.0), resultFloat)
    val out = Mat(); resultFloat.convertTo(out, CvType.CV_8UC3, 255.0)
    resultFloat.release()
    return out
}

// ═════════════════════════════════════════════════════════════════════════════
// Save helper
// ═════════════════════════════════════════════════════════════════════════════

private fun saveToGallery(context: android.content.Context, bitmap: Bitmap, tag: String): String? {
    val filename = "${tag}_${System.currentTimeMillis()}.jpg"
    try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Prescent")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                context.contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                context.contentResolver.update(uri, values, null, null)
            }
        } else {
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "Prescent").also { it.mkdirs() }
            val f = File(dir, filename)
            FileOutputStream(f).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            android.media.MediaScannerConnection.scanFile(context, arrayOf(f.absolutePath), null, null)
        }
    } catch (e: Exception) { Log.e(TAG, "save failed", e) }
    val cacheFile = File(context.cacheDir, filename)
    FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
    return cacheFile.absolutePath
}
