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
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.DMatch
import org.opencv.core.KeyPoint
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.features2d.DescriptorMatcher
import org.opencv.features2d.ORB
import org.opencv.imgproc.Imgproc
import org.opencv.photo.AlignMTB
import org.opencv.photo.CalibrateDebevec
import org.opencv.photo.MergeDebevec
import org.opencv.photo.MergeMertens
import org.opencv.photo.Photo
import org.opencv.photo.TonemapDrago
import org.opencv.photo.TonemapMantiuk
import org.opencv.photo.TonemapReinhard
import org.opencv.video.Video
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

actual suspend fun processHdr(
    images: List<String>,
    algorithm: String,
    alignment: String,
    contrastWeight: Float,
    saturationWeight: Float,
    exposureWeight: Float,
    gamma: Float,
    intensity: Float,
    lightAdaptation: Float,
    colorAdaptation: Float,
    dragoBias: Float,
    mantiukScale: Float,
    ghostingStrength: Float,
    cropAfterAlignment: Boolean,
    fattalAlpha: Float,
    fattalBeta: Float,
    fattalColorSaturation: Float,
    icam06ChromaticAdaptation: Float,
    icam06LocalAdaptation: Float,
    maxSize: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val mats = mutableListOf<Mat>()
    val isPreview = maxSize > 0

    Log.d("HdrProcessor", "Starting HDR ($algorithm) with ${images.size} images, preview=$isPreview")

    try {
        for (imageUriString in images) {
            val options = BitmapFactory.Options()
            if (isPreview) {
                options.inJustDecodeBounds = true
                decodeBitmapFromSource(context, imageUriString, options)
                val w = options.outWidth; val h = options.outHeight
                var s = 1
                while (w / s > maxSize || h / s > maxSize) s *= 2
                options.inSampleSize = s
                options.inJustDecodeBounds = false
            }
            val bitmap = decodeBitmapFromSource(context, imageUriString, options)
                ?: run { Log.e("HdrProcessor", "Failed to decode $imageUriString"); return@withContext null }
            val rgbaMat = Mat()
            Utils.bitmapToMat(bitmap, rgbaMat)
            bitmap.recycle()
            val rgbMat = Mat()
            Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
            rgbaMat.release()
            mats.add(rgbMat)
        }

        if (mats.size < 2) return@withContext null

        val alignedMats = alignImages(mats, alignment)
        val croppedMats = if (cropAfterAlignment) cropValidRegion(alignedMats) else alignedMats
        val ghostFreeMats = if (ghostingStrength > 0.01f) removeGhosting(croppedMats, ghostingStrength) else croppedMats
        if (ghostFreeMats.size < 2) return@withContext null

        val resultMat = Mat()
        when (algorithm) {
            "Mertens" -> {
                val merger = Photo.createMergeMertens(contrastWeight, saturationWeight, exposureWeight)
                merger.process(ghostFreeMats, resultMat)
            }
            "Reinhard", "Drago", "Mantiuk", "Fattal", "iCam06" -> {
                val numImages = ghostFreeMats.size
                val times = MatOfFloat()
                val timeValues = FloatArray(numImages) { i ->
                    val evStep = 4.0f / (numImages - 1)
                    val ev = -2.0f + i * evStep
                    Math.pow(2.0, ev.toDouble()).toFloat()
                }
                times.fromArray(*timeValues)
                val calibrate = Photo.createCalibrateDebevec()
                val response = Mat()
                calibrate.process(ghostFreeMats, response, times)
                val merge = Photo.createMergeDebevec()
                val hdrMat = Mat()
                merge.process(ghostFreeMats, hdrMat, times, response)
                response.release()
                val rgb32f = Mat()
                hdrMat.convertTo(rgb32f, CvType.CV_32F)
                when (algorithm) {
                    "Reinhard" -> {
                        val tonemap = Photo.createTonemapReinhard().apply {
                            setGamma(gamma); setIntensity(intensity)
                            setLightAdaptation(lightAdaptation); setColorAdaptation(colorAdaptation)
                        }
                        tonemap.process(hdrMat, resultMat)
                    }
                    "Drago" -> {
                        val tonemap = Photo.createTonemapDrago().apply {
                            setGamma(gamma); setSaturation(saturationWeight); setBias(dragoBias)
                        }
                        tonemap.process(hdrMat, resultMat)
                    }
                    "Mantiuk" -> {
                        val tonemap = Photo.createTonemapMantiuk().apply {
                            setGamma(gamma); setSaturation(saturationWeight); setScale(mantiukScale)
                        }
                        tonemap.process(hdrMat, resultMat)
                    }
                    "Fattal" -> {
                        fattalToneMap(rgb32f, resultMat, fattalAlpha, fattalBeta, fattalColorSaturation)
                    }
                    "iCam06" -> {
                        icam06ToneMap(rgb32f, resultMat, icam06ChromaticAdaptation, icam06LocalAdaptation, saturationWeight)
                    }
                }
                rgb32f.release()
                hdrMat.release(); times.release()
            }
            else -> {
                val merger = Photo.createMergeMertens(contrastWeight, saturationWeight, exposureWeight)
                merger.process(ghostFreeMats, resultMat)
            }
        }
        if (resultMat.empty()) return@withContext null

        val final8bit = Mat()
        resultMat.convertTo(final8bit, CvType.CV_8UC3, 255.0)
        resultMat.release()
        val rgbFinal = Mat()
        Imgproc.cvtColor(final8bit, rgbFinal, Imgproc.COLOR_BGR2RGB)
        final8bit.release()

        val resultBitmap = Bitmap.createBitmap(rgbFinal.cols(), rgbFinal.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgbFinal, resultBitmap)
        rgbFinal.release()

        val savedPath = if (isPreview) {
            val previewFile = File(context.cacheDir, "hdr_preview_${System.currentTimeMillis()}.jpg")
            FileOutputStream(previewFile).use { out ->
                resultBitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
            previewFile.absolutePath
        } else {
            saveToGallery(context, resultBitmap, algorithm)
        }
        resultBitmap.recycle()
        Log.d("HdrProcessor", "HDR complete: $savedPath")
        savedPath
    } catch (e: Exception) {
        Log.e("HdrProcessor", "HDR processing failed", e)
        null
    } finally {
        mats.forEach { it.release() }
    }
}

// ── Alignment ─────────────────────────────────────────────────────────────

private fun alignImages(mats: List<Mat>, method: String): List<Mat> {
    if (method == "Skip" || mats.size < 2) return mats
    return when (method) {
        "ECC" -> alignECC(mats)
        "Feature" -> alignFeature(mats)
        else -> alignMTB(mats)
    }
}

private fun alignMTB(mats: List<Mat>): List<Mat> {
    Log.d("HdrProcessor", "Aligning with MTB")
    val aligner = Photo.createAlignMTB()
    val aligned = mutableListOf<Mat>().also { out -> mats.forEach { _ -> out.add(Mat()) } }
    return try {
        aligner.process(mats, aligned)
        if (aligned.none { it.empty() }) {
            mats.forEach { it.release() }; aligned
        } else {
            aligned.forEach { it.release() }; mats
        }
    } catch (e: Exception) {
        Log.e("HdrProcessor", "MTB alignment failed", e)
        aligned.forEach { it.release() }; mats
    }
}

private fun alignECC(mats: List<Mat>): List<Mat> {
    Log.d("HdrProcessor", "Aligning with ECC")
    val refIdx = mats.size / 2
    val refGray = Mat()
    Imgproc.cvtColor(mats[refIdx], refGray, Imgproc.COLOR_RGB2GRAY)
    val result = mutableListOf<Mat>()
    mats.forEachIndexed { i, mat ->
        if (i == refIdx) { result.add(mat.clone()); return@forEachIndexed }
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY)
        val warpMat = Mat.eye(3, 3, CvType.CV_32F)
        val criteria = TermCriteria(TermCriteria.COUNT + TermCriteria.EPS, 50, 1e-4)
        try {
            Video.findTransformECC(refGray, gray, warpMat, Video.MOTION_HOMOGRAPHY, criteria)
            val warped = Mat()
            Imgproc.warpPerspective(mat, warped, warpMat, mats[refIdx].size())
            result.add(warped)
        } catch (e: Exception) {
            Log.e("HdrProcessor", "ECC failed for image $i", e)
            result.add(mat.clone())
        }
        gray.release()
    }
    refGray.release()
    return result
}

private fun alignFeature(mats: List<Mat>): List<Mat> {
    Log.d("HdrProcessor", "Aligning with Feature-based matching")
    val refIdx = mats.size / 2
    val orb = ORB.create(2000)
    val refGray = Mat()
    Imgproc.cvtColor(mats[refIdx], refGray, Imgproc.COLOR_RGB2GRAY)
    val refKp = MatOfKeyPoint()
    val refDesc = Mat()
    orb.detectAndCompute(refGray, Mat(), refKp, refDesc)
    val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
    val result = mutableListOf<Mat>()
    mats.forEachIndexed { i, mat ->
        if (i == refIdx) { result.add(mat.clone()); return@forEachIndexed }
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY)
        val kp = MatOfKeyPoint()
        val desc = Mat()
        orb.detectAndCompute(gray, Mat(), kp, desc)
        val matches = MatOfDMatch()
        matcher.match(refDesc, desc, matches)
        val matchArr = matches.toArray()
        val goodMatches = if (matchArr.size >= 8) {
            val minDist = matchArr.minOf { it.distance }
            matchArr.filter { it.distance <= max(3.0f * minDist, 30.0f) }
        } else emptyList()
        if (goodMatches.size >= 8) {
            val refKpArr = refKp.toArray()
            val kpArr = kp.toArray()
            val refArr = Array(goodMatches.size) { Point() }
            val imgArr = Array(goodMatches.size) { Point() }
            goodMatches.forEachIndexed { idx, dm ->
                refArr[idx] = refKpArr[dm.queryIdx].pt
                imgArr[idx] = kpArr[dm.trainIdx].pt
            }
            val refPts = MatOfPoint2f().apply { fromArray(*refArr) }
            val imgPts = MatOfPoint2f().apply { fromArray(*imgArr) }
            val mask = MatOfByte()
            val homography = Calib3d.findHomography(imgPts, refPts, Calib3d.RANSAC, 5.0, mask)
            if (homography != null) {
                val warped = Mat()
                Imgproc.warpPerspective(mat, warped, homography, mats[refIdx].size())
                result.add(warped)
                homography.release()
            } else {
                result.add(mat.clone())
            }
            refPts.release(); imgPts.release(); mask.release()
        } else {
            result.add(mat.clone())
        }
        matches.release(); desc.release(); kp.release(); gray.release()
    }
    refGray.release(); refKp.release(); refDesc.release()
    return result
}

// ── Crop after alignment ──────────────────────────────────────────────────

private fun cropValidRegion(mats: List<Mat>): List<Mat> {
    if (mats.isEmpty()) return mats
    val h = mats[0].rows(); val w = mats[0].cols()
    val borderFrac = 0.05
    var top = Int.MAX_VALUE; var bottom = 0
    var left = Int.MAX_VALUE; var right = 0
    mats.forEach { mat ->
        val gray = Mat()
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_RGB2GRAY)
        val mask = Mat()
        Core.compare(gray, Scalar(0.0), mask, Core.CMP_NE)
        val r = Imgproc.boundingRect(mask)
        if (r.area() > 0) {
            left = minOf(left, r.x); top = minOf(top, r.y)
            right = maxOf(right, r.x + r.width); bottom = maxOf(bottom, r.y + r.height)
        }
        gray.release(); mask.release()
    }
    left = maxOf(left - (w * borderFrac).toInt(), 0)
    top = maxOf(top - (h * borderFrac).toInt(), 0)
    right = minOf(right + (w * borderFrac).toInt(), w - 1)
    bottom = minOf(bottom + (h * borderFrac).toInt(), h - 1)
    val cropW = right - left; val cropH = bottom - top
    if (cropW <= 0 || cropH <= 0) return mats
    val rect = org.opencv.core.Rect(left, top, cropW, cropH)
    return mats.map { Mat(it, rect) }
}

// ── Ghosting Removal ──────────────────────────────────────────────────────

private fun removeGhosting(mats: List<Mat>, strength: Float): List<Mat> {
    if (mats.size < 2) return mats
    Log.d("HdrProcessor", "Removing ghosting (strength=$strength)")
    val refIdx = mats.size / 2
    val refFloat = Mat()
    mats[refIdx].convertTo(refFloat, CvType.CV_32F, 1.0 / 255.0)
    val result = mutableListOf<Mat>()
    mats.forEachIndexed { i, mat ->
        if (i == refIdx) { result.add(mat.clone()); return@forEachIndexed }
        val imgFloat = Mat()
        mat.convertTo(imgFloat, CvType.CV_32F, 1.0 / 255.0)
        val diff = Mat()
        Core.absdiff(imgFloat, refFloat, diff)
        val diffGray = Mat()
        Imgproc.cvtColor(diff, diffGray, Imgproc.COLOR_RGB2GRAY)
        val ghostMask = Mat()
        Core.multiply(diffGray, Scalar(strength.toDouble()), ghostMask)
        Imgproc.threshold(ghostMask, ghostMask, 1.0, 1.0, Imgproc.THRESH_TRUNC)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(ghostMask, ghostMask, Imgproc.MORPH_CLOSE, kernel)
        val blended = Mat()
        val ones = Mat.ones(ghostMask.size(), CvType.CV_32F)
        val oneMinusMask = Mat()
        Core.subtract(ones, ghostMask, oneMinusMask)
        val cleanPart = Mat()
        Core.multiply(imgFloat, oneMinusMask, cleanPart)
        val refPart = Mat()
        Core.multiply(refFloat, ghostMask, refPart)
        Core.add(cleanPart, refPart, blended)
        val result8 = Mat()
        blended.convertTo(result8, CvType.CV_8UC3, 255.0)
        result.add(result8)
        ones.release()
        imgFloat.release(); diff.release(); diffGray.release()
        ghostMask.release(); blended.release(); cleanPart.release()
        refPart.release(); oneMinusMask.release()
    }
    refFloat.release()
    return result
}

// ── Fattal Gradient Domain Tone Mapping ──────────────────────────────────

private fun fattalToneMap(
    rgb32f: Mat, output: Mat,
    alpha: Float, beta: Float, colorSat: Float
) {
    Log.d("HdrProcessor", "Fattal tone map (alpha=$alpha, beta=$beta, saturation=$colorSat)")

    val eps = 1e-6
    val lum = Mat()
    Imgproc.cvtColor(rgb32f, lum, Imgproc.COLOR_RGB2GRAY)
    Core.add(lum, Scalar(eps), lum)
    val logLum = Mat()
    Core.log(lum, logLum)

    // Build Gaussian pyramid
    val maxLevel = 5
    val gPyr = mutableListOf<Mat>()
    gPyr.add(logLum.clone())
    for (i in 1..maxLevel) {
        val down = Mat()
        Imgproc.pyrDown(gPyr[i - 1], down)
        gPyr.add(down)
    }

    // Build Laplacian pyramid and attenuate
    val lPyr = mutableListOf<Mat>()
    for (i in 0 until maxLevel) {
        val up = Mat()
        Imgproc.pyrUp(gPyr[i + 1], up, gPyr[i].size())
        val lap = Mat()
        Core.subtract(gPyr[i], up, lap)

        // Compute gradient magnitude at this level
        val gx = Mat(); val gy = Mat()
        Imgproc.Sobel(gPyr[i], gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gPyr[i], gy, CvType.CV_32F, 0, 1, 3)
        val mag = Mat()
        Core.magnitude(gx, gy, mag)
        val maxVal = Core.minMaxLoc(mag).maxVal
        val threshold = alpha.toDouble() * maxVal

        // Build attenuation multiplier
        val atten = Mat()
        Core.divide(mag, Scalar(threshold), atten)
        Core.pow(atten, beta - 1.0, atten)
        val mask = Mat()
        Core.compare(mag, Scalar(threshold), mask, Core.CMP_LE)
        atten.setTo(Scalar(1.0), mask)

        Core.multiply(lap, atten, lap)
        lPyr.add(lap)
        gx.release(); gy.release(); mag.release(); atten.release(); mask.release(); up.release()
    }
    lPyr.add(gPyr[maxLevel].clone())

    // Reconstruct log luminance from attenuated pyramid
    var recon = lPyr[maxLevel].clone()
    for (i in maxLevel - 1 downTo 0) {
        val up = Mat()
        Imgproc.pyrUp(recon, up, gPyr[i].size())
        Core.add(up, lPyr[i], recon)
        up.release()
    }

    val outLum = Mat()
    Core.exp(recon, outLum)
    Core.subtract(outLum, Scalar(eps), outLum)

    // Recombine with color
    val invLum = Mat()
    Core.divide(1.0, lum, invLum)
    val ratio = Mat()
    Core.multiply(invLum, outLum, ratio)
    Core.pow(ratio, colorSat.toDouble(), ratio)

    val channels = mutableListOf<Mat>()
    Core.split(rgb32f, channels)
    for (c in channels) {
        Core.multiply(c, ratio, c)
    }
    Core.merge(channels, output)
    channels.forEach { it.release() }

    // Cleanup
    lum.release(); logLum.release(); recon.release()
    outLum.release(); invLum.release(); ratio.release()
    gPyr.forEach { it.release() }; lPyr.forEach { it.release() }
}

// ── iCam06 Perceptual Tone Mapping ───────────────────────────────────────

private fun icam06ToneMap(
    rgb32f: Mat, output: Mat,
    chromAdaptStrength: Float, localAdaptKernel: Float, colorSat: Float
) {
    Log.d("HdrProcessor", "iCam06 (chromAdapt=${chromAdaptStrength}, localAdaptKernel=${localAdaptKernel}, saturation=${colorSat})")

    val eps = 1e-6
    val rows = rgb32f.rows()
    val cols = rgb32f.cols()

    // RGB → XYZ conversion (sRGB D65)
    val channels = mutableListOf<Mat>()
    Core.split(rgb32f, channels)
    val r = channels[0]; val g = channels[1]; val b = channels[2]

    // Pre-allocate X, Y, Z as single-channel CV_32F (matching r/g/b from split)
    val X = Mat.zeros(rows, cols, CvType.CV_32F)
    val Y = Mat.zeros(rows, cols, CvType.CV_32F)
    val Z = Mat.zeros(rows, cols, CvType.CV_32F)

    Core.scaleAdd(r, 0.4124564, X, X)
    Core.scaleAdd(g, 0.3575761, X, X)
    Core.scaleAdd(b, 0.1804375, X, X)

    Core.scaleAdd(r, 0.2126729, Y, Y)
    Core.scaleAdd(g, 0.7151522, Y, Y)
    Core.scaleAdd(b, 0.0721750, Y, Y)

    Core.scaleAdd(r, 0.0193339, Z, Z)
    Core.scaleAdd(g, 0.1191920, Z, Z)
    Core.scaleAdd(b, 0.9503041, Z, Z)

    // Chromatic adaptation: blend X and Z toward Y
    val adaptedX = Mat()
    val adaptedZ = Mat()
    Core.addWeighted(X, chromAdaptStrength.toDouble(), Y, (1.0 - chromAdaptStrength), 0.0, adaptedX)
    Core.addWeighted(Z, chromAdaptStrength.toDouble(), Y, (1.0 - chromAdaptStrength), 0.0, adaptedZ)

    // Local adaptation luminance — Gaussian blur of Y
    val localAdapt = Mat()
    val kSize = ((localAdaptKernel * 15.0).toInt() or 1).coerceAtLeast(3)
    Imgproc.GaussianBlur(Y, localAdapt, Size(kSize.toDouble(), kSize.toDouble()), 0.0)
    Core.add(localAdapt, Scalar(eps), localAdapt)

    // Sigmoidal tone compression: Y_out = Y / (Y + localAdapt^0.7)
    val localPow = Mat()
    Core.pow(localAdapt, 0.7, localPow)
    val yDenom = Mat()
    Core.add(Y, localPow, yDenom)
    val yOut = Mat()
    Core.divide(Y, yDenom, yOut)

    // Scale adapted X, Z proportionally
    val yRatio = Mat()
    Core.divide(yOut, Y, yRatio)
    val xOut = Mat()
    val zOut = Mat()
    Core.multiply(adaptedX, yRatio, xOut)
    Core.multiply(adaptedZ, yRatio, zOut)

    // Color saturation boost
    Core.pow(yRatio, (1.0 - colorSat).toDouble(), yRatio)
    Core.multiply(xOut, yRatio, xOut)
    Core.multiply(yOut, yRatio, yOut)
    Core.multiply(zOut, yRatio, zOut)

    // XYZ → RGB — pre-allocate as single-channel CV_32F
    val outR = Mat.zeros(rows, cols, CvType.CV_32F)
    val outG = Mat.zeros(rows, cols, CvType.CV_32F)
    val outB = Mat.zeros(rows, cols, CvType.CV_32F)

    Core.scaleAdd(xOut, 3.2404542, outR, outR)
    Core.scaleAdd(yOut, -1.5371385, outR, outR)
    Core.scaleAdd(zOut, -0.4985314, outR, outR)

    Core.scaleAdd(xOut, -0.9692660, outG, outG)
    Core.scaleAdd(yOut, 1.8760108, outG, outG)
    Core.scaleAdd(zOut, 0.0415560, outG, outG)

    Core.scaleAdd(xOut, 0.0556434, outB, outB)
    Core.scaleAdd(yOut, -0.2040259, outB, outB)
    Core.scaleAdd(zOut, 1.0572252, outB, outB)

    // Clamp to [0,1]
    for (c in listOf(outR, outG, outB)) {
        Core.max(c, Scalar(0.0), c)
        Core.min(c, Scalar(1.0), c)
    }
    Core.merge(listOf(outR, outG, outB), output)

    channels.forEach { it.release() }
    X.release(); Y.release(); Z.release()
    adaptedX.release(); adaptedZ.release()
    localAdapt.release(); localPow.release(); yDenom.release()
    yOut.release(); yRatio.release(); xOut.release(); zOut.release()
    outR.release(); outG.release(); outB.release()
}

// ── Utility Functions ─────────────────────────────────────────────────────

private fun decodeBitmapFromSource(
    context: android.content.Context, uriString: String, options: BitmapFactory.Options
): Bitmap? = try {
    if (uriString.startsWith("/")) BitmapFactory.decodeFile(uriString, options)
    else {
        val uri = Uri.parse(uriString)
        context.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
    }
} catch (e: Exception) { Log.e("HdrProcessor", "decode failed for $uriString", e); null }

private fun saveToGallery(context: android.content.Context, bitmap: Bitmap, algorithm: String): String? {
    val filename = "prescent_hdr_${algorithm}_${System.currentTimeMillis()}.jpg"
    return try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Prescent")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null
            context.contentResolver.openOutputStream(uri)?.use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
            context.contentResolver.update(uri, values, null, null)
            val cacheFile = File(context.cacheDir, filename)
            FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            cacheFile.absolutePath
        } else {
            val picturesDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "Prescent").also { it.mkdirs() }
            val file = File(picturesDir, filename)
            FileOutputStream(file).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            android.media.MediaScannerConnection.scanFile(context, arrayOf(file.absolutePath), null, null)
            file.absolutePath
        }
    } catch (e: Exception) {
        Log.e("HdrProcessor", "saveToGallery failed", e)
        val cacheFile = File(context.cacheDir, filename)
        FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        cacheFile.absolutePath
    }
}
