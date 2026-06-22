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
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import java.io.File
import java.io.FileOutputStream
import kotlin.math.min

actual suspend fun processNightSight(
    images: List<String>,
    algorithm: NightSightAlgorithm,
    useLuckyPreFilter: Boolean,
    luckyKeepFraction: Float,
    starTrail: Boolean,
    darkFramePath: String?,
    brightnessBoost: Float,
    maxPreviewSize: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val isPreview = maxPreviewSize > 0
    Log.d("NightSight", "Processing ${images.size} frames, algo=$algorithm, lucky=$useLuckyPreFilter($luckyKeepFraction), starTrail=$starTrail")

    try {
        // 1. Load frames
        val frames = loadFrames(images, isPreview, maxPreviewSize)
        if (frames.size < 2) return@withContext null

        // 2. Load dark frame
        val darkFrame = loadDarkFrame(darkFramePath)

        // 3. Align frames using MTB
        val aligned = alignFrames(frames)
        var validFrames = aligned.filter { !it.empty() }
        if (validFrames.isEmpty()) {
            Log.w("NightSight", "All aligned frames empty, falling back to raw frames")
            validFrames = frames.map { it.clone() }
        }

        // 4. Dark frame subtract
        val subtracted = subtractDarkFrame(validFrames, darkFrame)
        if (subtracted.isEmpty()) {
            Log.e("NightSight", "No valid frames after subtraction, aborting")
            frames.forEach { it.release() }; aligned.forEach { it.release() }; darkFrame?.release()
            return@withContext null
        }

        // 5. Lucky pre-filter: select sharpest frames
        val workingFrames = if (useLuckyPreFilter && algorithm.supportsLuckyPreFilter) {
            luckySelect(subtracted, luckyKeepFraction)
        } else subtracted

        // 6. Stack by algorithm
        val stacked = when (algorithm) {
            NightSightAlgorithm.AVERAGE -> {
                if (starTrail) stackMaxFrames(workingFrames)
                else stackAverageFrames(workingFrames)
            }
            NightSightAlgorithm.MEDIAN -> stackMedianFrames(workingFrames)
            NightSightAlgorithm.LAPLACIAN -> stackLaplacianFrames(workingFrames)
            NightSightAlgorithm.HDR_MERGE -> stackHdrFrames(workingFrames)
        } ?: run {
            Log.e("NightSight", "Stacking returned null")
            frames.forEach { it.release() }; aligned.forEach { it.release() }; subtracted.forEach { it.release() }; workingFrames.forEach { it.release() }; darkFrame?.release()
            return@withContext null
        }

        // 7. Brightness boost
        val brightened = boostBrightness(stacked, brightnessBoost)

        // 8. CLAHE local contrast enhancement
        val enhanced = applyClahe(brightened)

        // 9. Save
        val savedPath = saveResult(enhanced, isPreview, starTrail)

        frames.forEach { it.release() }
        aligned.forEach { it.release() }
        subtracted.forEach { it.release() }
        workingFrames.forEach { if (it !== stacked && it !== brightened && it !== enhanced) it.release() }
        stacked.release()
        brightened.release()
        enhanced.release()
        darkFrame?.release()

        savedPath
    } catch (e: Exception) {
        Log.e("NightSight", "Processing failed", e)
        null
    }
}

// ── Helper: Load frames ────────────────────────────────────────────

private fun loadFrames(images: List<String>, isPreview: Boolean, maxPreviewSize: Int): List<Mat> {
    val result = mutableListOf<Mat>()
    for (path in images) {
        val opts = BitmapFactory.Options()
        if (isPreview) {
            opts.inJustDecodeBounds = true
            val bm = if (path.startsWith("/")) BitmapFactory.decodeFile(path, opts)
            else AndroidAppContext.applicationCtx.contentResolver.openInputStream(Uri.parse(path))?.use { BitmapFactory.decodeStream(it, null, opts) }
            val w = opts.outWidth; val h = opts.outHeight
            var s = 1; while (w / s > maxPreviewSize || h / s > maxPreviewSize) s *= 2
            opts.inSampleSize = s; opts.inJustDecodeBounds = false
        }
        val bitmap = decodeBitmap(path, opts) ?: continue
        val rgba = Mat(); Utils.bitmapToMat(bitmap, rgba); bitmap.recycle()
        val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_BGRA2BGR); rgba.release()
        result.add(rgb)
    }
    return result
}

private fun decodeBitmap(path: String, opts: BitmapFactory.Options): Bitmap? {
    return if (path.startsWith("/")) BitmapFactory.decodeFile(path, opts)
    else AndroidAppContext.applicationCtx.contentResolver.openInputStream(Uri.parse(path))
        ?.use { BitmapFactory.decodeStream(it, null, opts) }
}

private fun loadDarkFrame(darkFramePath: String?): Mat? {
    if (darkFramePath == null) return null
    return try {
        val opts = BitmapFactory.Options()
        val bm = decodeBitmap(darkFramePath, opts) ?: return null
        val rgba = Mat(); Utils.bitmapToMat(bm, rgba); bm.recycle()
        val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_BGRA2BGR); rgba.release()
        Log.d("NightSight", "Dark frame loaded")
        rgb
    } catch (e: Exception) {
        Log.e("NightSight", "Dark frame load failed", e)
        null
    }
}

// ── Helper: MTB alignment ──────────────────────────────────────────

private fun alignFrames(frames: List<Mat>): List<Mat> {
    val aligner = Photo.createAlignMTB()
    val aligned = mutableListOf<Mat>().also { out -> frames.forEach { _ -> out.add(Mat()) } }
    try { aligner.process(frames, aligned) } catch (e: Exception) {
        Log.e("NightSight", "Alignment failed, using raw frames")
        frames.forEachIndexed { i, m -> aligned[i] = m.clone() }
    }
    return aligned
}

// ── Helper: Dark frame subtract ────────────────────────────────────

private fun subtractDarkFrame(validFrames: List<Mat>, darkFrame: Mat?): List<Mat> {
    if (darkFrame == null) return validFrames
    return validFrames.map { mat ->
        val sub = Mat()
        Core.subtract(mat, darkFrame, sub)
        Core.max(sub, Scalar(0.0), sub)
        sub
    }
}

// ── Lucky pre-filter: select sharpest frames ───────────────────────

private fun luckySelect(frames: List<Mat>, keepFraction: Float): List<Mat> {
    val n = (frames.size * keepFraction).toInt().coerceAtLeast(1)
    if (n >= frames.size) return frames

    data class Sharpness(val index: Int, val score: Double)
    val scores = frames.mapIndexed { idx, mat ->
        Sharpness(idx, computeSharpness(mat))
    }
    scores.sortedByDescending { it.score }.take(n).forEach {
        Log.d("NightSight", "Lucky selected frame ${it.index} score=${it.score}")
    }
    val selected = scores.sortedByDescending { it.score }.take(n).map { frames[it.index].clone() }
    Log.d("NightSight", "Lucky filter: ${frames.size} -> ${selected.size} frames")
    return selected
}

private fun computeSharpness(mat: Mat): Double {
    val gray = Mat()
    Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY)
    val lap = Mat()
    Imgproc.Laplacian(gray, lap, CvType.CV_64F)
    val m = Core.mean(lap)
    val meanV = m.`val`[0]
    val squared = Mat()
    Core.subtract(lap, Scalar(meanV), squared)
    Core.multiply(squared, squared, squared)
    val variance = Core.mean(squared).`val`[0]
    gray.release(); lap.release(); squared.release()
    return variance
}

// ── Stack: Average ─────────────────────────────────────────────────

private fun stackAverageFrames(frames: List<Mat>): Mat {
    val result = Mat.zeros(frames[0].size(), CvType.CV_32FC3)
    val floatMats = frames.map { mat ->
        val f = Mat(); mat.convertTo(f, CvType.CV_32FC3, 1.0 / 255.0); f
    }
    for (fm in floatMats) Core.add(result, fm, result)
    Core.multiply(result, Scalar(1.0 / frames.size), result)
    val result8 = Mat(); result.convertTo(result8, CvType.CV_8UC3, 255.0)
    floatMats.forEach { it.release() }; result.release()
    return result8
}

// ── Stack: Max (star trail) ────────────────────────────────────────

private fun stackMaxFrames(frames: List<Mat>): Mat {
    val result = frames[0].clone()
    for (i in 1 until frames.size) {
        Core.max(result, frames[i], result)
    }
    return result
}

// ── Stack: Median ──────────────────────────────────────────────────

private fun stackMedianFrames(frames: List<Mat>): Mat {
    val n = frames.size
    val h = frames[0].rows()
    val w = frames[0].cols()
    val totalPixels = h * w

    val floatMats = frames.map { mat ->
        val f = Mat(); mat.convertTo(f, CvType.CV_32FC3, 1.0 / 255.0); f
    }

    val channels = mutableListOf<Mat>()
    for (c in 0 until 3) {
        val colMat = Mat(totalPixels, n, CvType.CV_32F)
        for (j in 0 until n) {
            val chJ = Mat()
            Core.extractChannel(floatMats[j], chJ, c)
            val flat = chJ.clone().reshape(1, totalPixels)
            flat.col(0).copyTo(colMat.col(j))
            flat.release(); chJ.release()
        }
        Core.sort(colMat, colMat, Core.SORT_EVERY_ROW or Core.SORT_ASCENDING)
        val medianCol = colMat.col(n / 2)
        val medianCh = medianCol.clone().reshape(1, h)
        channels.add(medianCh)
        colMat.release(); medianCol.release()
    }

    val median = Mat()
    Core.merge(channels, median)
    val result8 = Mat(); median.convertTo(result8, CvType.CV_8UC3, 255.0)
    floatMats.forEach { it.release() }; channels.forEach { it.release() }; median.release()
    return result8
}

// ── Stack: Laplacian Pyramid ───────────────────────────────────────

private fun stackLaplacianFrames(frames: List<Mat>): Mat {
    val levels = 4

    val floatMats = frames.map { mat ->
        val f = Mat(); mat.convertTo(f, CvType.CV_32FC3, 1.0 / 255.0); f
    }

    val result = try {
        // Build resize-based Laplacian pyramids and average each level
        val pyramids = floatMats.map { buildLaplacianPyramidResize(it, levels) }
        val blended = mutableListOf<Mat>()
        for (level in 0..levels) {
            val sum = Mat.zeros(pyramids[0][level].size(), CvType.CV_32FC3)
            for (p in pyramids) Core.add(sum, p[level], sum)
            Core.multiply(sum, Scalar(1.0 / pyramids.size), sum)
            blended.add(sum)
        }

        // Reconstruct
        var current = blended.last().clone()
        for (i in blended.size - 2 downTo 0) {
            val up = Mat()
            Imgproc.resize(current, up, blended[i].size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
            val reconstructed = Mat()
            Core.add(up, blended[i], reconstructed)
            current.release(); up.release()
            current = reconstructed
        }

        pyramids.forEach { level -> level.forEach { it.release() } }
        blended.forEach { it.release() }

        val result8 = Mat(); current.convertTo(result8, CvType.CV_8UC3, 255.0)
        current.release()
        result8
    } catch (e: Exception) {
        Log.e("NightSight", "Laplacian pyramid failed, falling back to average", e)
        floatMats.forEach { it.release() }
        return stackAverageFrames(frames)
    }

    floatMats.forEach { it.release() }
    return result
}

private fun buildLaplacianPyramidResize(mat: Mat, levels: Int): List<Mat> {
    val gaussian = mutableListOf<Mat>()
    gaussian.add(mat.clone())
    repeat(levels) {
        val prev = gaussian.last()
        val w = (prev.cols() + 1) / 2
        val h = (prev.rows() + 1) / 2
        val down = Mat()
        Imgproc.resize(prev, down, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        gaussian.add(down)
    }
    val laplacian = mutableListOf<Mat>()
    for (i in 0 until gaussian.size - 1) {
        val up = Mat()
        Imgproc.resize(gaussian[i + 1], up, gaussian[i].size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val lap = Mat()
        Core.subtract(gaussian[i], up, lap)
        laplacian.add(lap)
        up.release()
    }
    laplacian.add(gaussian.last().clone())
    gaussian.forEach { it.release() }
    return laplacian
}

// ── Stack: HDR Merge (Debevec + Reinhard) ─────────────────────────

private fun stackHdrFrames(frames: List<Mat>): Mat {
    val floatMats = frames.map { mat ->
        val f = Mat(); mat.convertTo(f, CvType.CV_32FC3, 1.0 / 255.0); f
    }

    val times = MatOfFloat(*FloatArray(floatMats.size) { 0.5f })
    try {
        val calibrate = Photo.createCalibrateDebevec()
        val response = Mat()
        calibrate.process(floatMats, response, times)

        val merge = Photo.createMergeDebevec()
        val hdr = Mat()
        merge.process(floatMats, hdr, times, response)
        response.release()

        val tonemap = Photo.createTonemapReinhard(1.5f, 0.8f, 0.0f)
        val ldr = Mat()
        tonemap.process(hdr, ldr)
        hdr.release()

        val result8 = Mat(); ldr.convertTo(result8, CvType.CV_8UC3, 255.0)
        floatMats.forEach { it.release() }; ldr.release()
        return result8
    } catch (e: Exception) {
        Log.e("NightSight", "HDR merge failed, falling back to average", e)
        floatMats.forEach { it.release() }
        return stackAverageFrames(frames)
    }
}

// ── Brightness boost ───────────────────────────────────────────────

private fun boostBrightness(stacked: Mat, boost: Float): Mat {
    val f = Mat(); stacked.convertTo(f, CvType.CV_32F, 1.0 / 255.0)
    Core.multiply(f, Scalar(boost.toDouble()), f)
    Core.min(f, Scalar(1.0), f)
    val result = Mat(); f.convertTo(result, CvType.CV_8UC3, 255.0)
    f.release()
    return result
}

// ── CLAHE enhancement ──────────────────────────────────────────────

private fun applyClahe(mat: Mat): Mat {
    val lab = Mat()
    Imgproc.cvtColor(mat, lab, Imgproc.COLOR_BGR2Lab)
    val labCh = mutableListOf<Mat>(); Core.split(lab, labCh)
    val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
    clahe.apply(labCh[0], labCh[0])
    Core.merge(labCh, lab)
    val result = Mat(); Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2BGR)
    mat.release(); lab.release(); labCh.forEach { it.release() }
    return result
}

// ── Save ───────────────────────────────────────────────────────────

private fun saveResult(enhanced: Mat, isPreview: Boolean, starTrail: Boolean): String? {
    val context = AndroidAppContext.applicationCtx
    val bitmap = Bitmap.createBitmap(enhanced.cols(), enhanced.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(enhanced, bitmap)

    val suffix = if (starTrail) "_trail" else ""
    val filename = "nightsight${suffix}_${System.currentTimeMillis()}.jpg"

    return if (isPreview) {
        val previewFile = File(context.cacheDir, filename)
        FileOutputStream(previewFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        bitmap.recycle()
        previewFile.absolutePath
    } else {
        try {
            saveToMediaStore(context, bitmap, filename)
        } catch (e: Exception) {
            Log.e("NightSight", "MediaStore save failed, saving to cache", e)
        }
        val cacheFile = File(context.cacheDir, filename)
        FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        bitmap.recycle()
        cacheFile.absolutePath
    }
}

private fun saveToMediaStore(context: android.content.Context, bitmap: Bitmap, filename: String) {
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
}
