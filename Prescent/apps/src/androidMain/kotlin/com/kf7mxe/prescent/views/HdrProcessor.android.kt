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
import kotlin.math.min

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
    surrealAmount: Float,
    maxSize: Int,
    pyramidNoiseStrength: Float,
    enableHotPixelFix: Boolean,
    enableCACorrection: Boolean,
    enableLensCorrection: Boolean,
    enableSmartNR: Boolean,
    enableContrastSharpening: Boolean,
    smartFrameSelection: Boolean,
    guidedFusionLevels: Int,
    guidedFusionSigmaColor: Float,
    guidedFusionSigmaSpace: Float,
    retinexSigma: Float,
    retinexCompression: Float,
    retinexGamma: Float,
    saliencyWeight: Float,
    enableJointDenoise: Boolean,
    enableDehaze: Boolean,
    dehazePatchSize: Int,
    dehazeOmega: Float,
    superResolutionScale: Int,
    artisticEffect: String,
    artisticOrtonBlurRadius: Int,
    artisticOrtonOpacity: Float,
    artisticMiniatureFocusY: Float,
    artisticMiniatureBlurHeight: Float,
    artisticBokehRadius: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val mats = mutableListOf<Mat>()
    val isPreview = maxSize > 0

    Log.d("HdrProcessor", "Starting HDR ($algorithm) with ${images.size} images, surreal=$surrealAmount, preview=$isPreview")

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

        // Smart frame selection: drop blurry frames
        val filteredMats = if (smartFrameSelection && mats.size > 2) {
            selectSharpestFrames(mats, 0.75)
        } else mats
        Log.d("HdrProcessor", "Smart selection: ${mats.size} → ${filteredMats.size} frames")

        // Joint denoise: use well-exposed frame as guide to denoise underexposed
        val denoisedMats = if (enableJointDenoise) {
            Log.d("HdrProcessor", "Applying joint denoise")
            jointDenoiseBracketPair(filteredMats)
        } else filteredMats

        // Per-frame pre-processing (hot pixel, CA, lens distortion)
        val preprocessed = denoisedMats.map { mat ->
            var m = mat
            if (enableHotPixelFix) {
                Log.d("HdrProcessor", "Applying hot pixel fix")
                m = detectAndFixHotPixels(m)
            }
            if (enableCACorrection) {
                Log.d("HdrProcessor", "Applying chromatic aberration correction")
                m = correctChromaticAberration(m)
            }
            if (enableLensCorrection) {
                Log.d("HdrProcessor", "Applying lens distortion correction")
                m = correctLensDistortion(m)
            }
            m
        }

        val alignedMats = alignImages(preprocessed, alignment)
        val croppedMats = if (cropAfterAlignment) cropValidRegion(alignedMats) else alignedMats
        if (croppedMats.size < 2) return@withContext null

        val resultMat = Mat()
        when (algorithm) {
            "Hybrid" -> {
                hybridToneMap(croppedMats, resultMat, surrealAmount,
                    gamma, intensity, lightAdaptation, colorAdaptation,
                    fattalAlpha, fattalBeta, fattalColorSaturation)
            }
            "Contrast Optimizer" -> {
                contrastOptimizer(croppedMats, resultMat, surrealAmount,
                    contrastWeight, saturationWeight, exposureWeight,
                    fattalAlpha, fattalBeta)
            }
            "Durand" -> {
                durandToneMap(croppedMats, resultMat, surrealAmount,
                    gamma, saturationWeight, fattalAlpha)
            }
            "CLAHE Boost" -> {
                claheToneMap(croppedMats, resultMat, surrealAmount,
                    contrastWeight, saturationWeight, exposureWeight)
            }
            "Mertens" -> {
                val merger = Photo.createMergeMertens(contrastWeight, saturationWeight, exposureWeight)
                merger.process(croppedMats, resultMat)
            }
            "Pyramid Fusion" -> {
                val cfg = PyramidMergeConfig(noiseStrength = pyramidNoiseStrength.toDouble())
                val pyramidResult = laplacianPyramidFusion(croppedMats, cfg)
                pyramidResult.convertTo(resultMat, CvType.CV_32FC3, 1.0 / 255.0)
                pyramidResult.release()
            }
            "Guided Fusion" -> {
                val cfg = GuidedFusionConfig(
                    levels = guidedFusionLevels,
                    sigmaColor = guidedFusionSigmaColor.toDouble(),
                    sigmaSpace = guidedFusionSigmaSpace.toDouble()
                )
                val guidedResult = multiScaleGuidedFusion(croppedMats, cfg)
                guidedResult.convertTo(resultMat, CvType.CV_32FC3, 1.0 / 255.0)
                guidedResult.release()
            }
            "Retinex" -> {
                val cfg = RetinexConfig(
                    gaussianSigma = retinexSigma.toDouble(),
                    compression = retinexCompression.toDouble(),
                    gamma = retinexGamma.toDouble()
                )
                val retinexResult = retinexToneMap(croppedMats, cfg)
                retinexResult.convertTo(resultMat, CvType.CV_32FC3, 1.0 / 255.0)
                retinexResult.release()
            }
            "Saliency Fusion" -> {
                val cfg = SaliencyFusionConfig(
                    saliencyWeight = saliencyWeight.toDouble(),
                    contrastWeight = contrastWeight.toDouble(),
                    saturationWeight = saturationWeight.toDouble(),
                    exposureWeight = exposureWeight.toDouble()
                )
                val salResult = saliencyWeightedFusion(croppedMats, cfg)
                salResult.convertTo(resultMat, CvType.CV_32FC3, 1.0 / 255.0)
                salResult.release()
            }
            "Reinhard", "Drago", "Mantiuk", "Fattal", "iCam06" -> {
                val numImages = croppedMats.size
                val times = MatOfFloat()
                val timeValues = FloatArray(numImages) { i ->
                    val evStep = 4.0f / (numImages - 1)
                    val ev = -2.0f + i * evStep
                    Math.pow(2.0, ev.toDouble()).toFloat()
                }
                times.fromArray(*timeValues)
                val calibrate = Photo.createCalibrateDebevec()
                val response = Mat()
                calibrate.process(croppedMats, response, times)
                val merge = Photo.createMergeDebevec()
                val hdrMat = Mat()
                merge.process(croppedMats, hdrMat, times, response)
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
                merger.process(croppedMats, resultMat)
            }
        }
        if (resultMat.empty()) return@withContext null

        // Ghost reduction: post-process on tonemapped float output (doesn't corrupt exposure brackets)
        val ghostFree = if (ghostingStrength > 0.01f) {
            val reduced = Mat()
            reduceGhostArtifacts(resultMat, reduced, ghostingStrength)
            resultMat.release()
            reduced
        } else resultMat

        // Post-processing on uint8 (smart NR, contrast sharpening, dehaze, artistic)
        var post8 = Mat()
        ghostFree.convertTo(post8, CvType.CV_8UC3, 255.0)
        ghostFree.release()

        if (enableSmartNR) {
            Log.d("HdrProcessor", "Applying smart noise reduction")
            post8 = smartNoiseReduction(post8)
        }
        if (enableContrastSharpening) {
            Log.d("HdrProcessor", "Applying contrast limited sharpening")
            post8 = contrastLimitedSharpening(post8)
        }
        if (enableDehaze) {
            Log.d("HdrProcessor", "Applying dark channel prior dehazing")
            post8 = darkChannelPriorDehaze(post8, DehazeConfig(
                patchSize = dehazePatchSize, omega = dehazeOmega.toDouble()
            ))
        }
        if (superResolutionScale > 1) {
            Log.d("HdrProcessor", "Applying super resolution ${superResolutionScale}x")
            val srList = listOf(post8)
            post8 = multiFrameSuperResolution(srList, SuperResConfig(scaleFactor = superResolutionScale))
        }
        if (artisticEffect != "None") {
            Log.d("HdrProcessor", "Applying artistic effect: $artisticEffect")
            val effect = when (artisticEffect) {
                "Orton" -> ArtisticEffect.ORTON
                "Miniature" -> ArtisticEffect.MINIATURE
                "Bokeh" -> ArtisticEffect.BOKEH
                else -> null
            }
            if (effect != null) {
                post8 = applyArtisticEffect(post8, ArtisticConfig(
                    effect = effect,
                    ortonBlurRadius = artisticOrtonBlurRadius,
                    ortonOpacity = artisticOrtonOpacity.toDouble(),
                    miniatureFocusY = artisticMiniatureFocusY.toDouble(),
                    miniatureBlurHeight = artisticMiniatureBlurHeight.toDouble(),
                    bokehRadius = artisticBokehRadius
                ))
            }
        }

        val final8bit = post8

        val resultBitmap = Bitmap.createBitmap(final8bit.cols(), final8bit.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(final8bit, resultBitmap)
        final8bit.release()

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

// ── Hybrid: Reinhard (base) + unsharp mask detail on luminance ────────────
//
// Correct approach: tone-map ONCE with Reinhard for natural colors, then
// apply unsharp mask on luminance only. This preserves hue while enhancing
// local contrast — the same technique Photoshop/Lightroom use for "Clarity".

private fun hybridToneMap(
    mats: List<Mat>, output: Mat,
    surrealAmount: Float,
    gamma: Float, intensity: Float, lightAdaptation: Float, colorAdaptation: Float,
    detailRadius: Float, detailStrength: Float, colorSaturation: Float
) {
    Log.d("HdrProcessor", "Hybrid tone map (surreal=$surrealAmount)")

    // 1. Build HDR radiance map
    val numImages = mats.size
    val times = MatOfFloat()
    val timeValues = FloatArray(numImages) { i ->
        Math.pow(2.0, (-2.0 + 4.0 * i / (numImages - 1)).toDouble()).toFloat()
    }
    times.fromArray(*timeValues)
    val calibrate = Photo.createCalibrateDebevec()
    val response = Mat()
    calibrate.process(mats, response, times)
    val merge = Photo.createMergeDebevec()
    val hdrMat = Mat()
    merge.process(mats, hdrMat, times, response)
    response.release(); times.release()

    // 2. Reinhard tone map → natural base
    val base = Mat()
    val tonemap = Photo.createTonemapReinhard().apply {
        setGamma(gamma); setIntensity(intensity)
        setLightAdaptation(lightAdaptation); setColorAdaptation(colorAdaptation)
    }
    tonemap.process(hdrMat, base)
    hdrMat.release()

    // 3. Extract luminance
    val lum = Mat()
    Imgproc.cvtColor(base, lum, Imgproc.COLOR_RGB2GRAY)

    // 4. Unsharp mask on luminance
    val kSize = ((detailRadius * 200.0 + 3.0).toInt() or 1).coerceIn(3, 101)
    val blurred = Mat()
    Imgproc.GaussianBlur(lum, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)

    val detail = Mat()
    Core.subtract(lum, blurred, detail)

    val amount = (surrealAmount * detailStrength * 3.0f).coerceAtMost(5.0f)
    val enhanced = Mat()
    Core.addWeighted(lum, 1.0, detail, amount.toDouble(), 0.0, enhanced)
    Core.max(enhanced, Scalar(1e-6), enhanced)

    // 5. ratio = enhanced / lum (hue-preserving multiplier)
    val ratio = Mat()
    Core.divide(enhanced, lum, ratio)

    // Saturation boost: gamma on ratio reduces its effect on color saturation
    Core.pow(ratio, (1.0 - colorSaturation * 0.4).toDouble(), ratio)

    // 6. Apply ratio to all RGB channels
    val channels = mutableListOf<Mat>()
    Core.split(base, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
    }
    Core.merge(channels, output)

    base.release(); lum.release(); blurred.release()
    detail.release(); enhanced.release(); ratio.release()
    channels.forEach { it.release() }
}

// ── Contrast Optimizer: Mertens (clean) + unsharp mask on luminance ────────
//
// Correct approach: Mertens fusion first (halo-free, photorealistic), then
// unsharp mask on luminance only for local contrast enhancement.

private fun contrastOptimizer(
    mats: List<Mat>, output: Mat,
    surrealAmount: Float,
    contrastWeight: Float, saturationWeight: Float, exposureWeight: Float,
    detailRadius: Float, detailStrength: Float
) {
    Log.d("HdrProcessor", "Contrast Optimizer (surreal=$surrealAmount)")

    // 1. Mertens exposure fusion → clean, halo-free base
    val base = Mat()
    val merger = Photo.createMergeMertens(contrastWeight, saturationWeight, exposureWeight)
    merger.process(mats, base)

    // 2. Extract luminance
    val lum = Mat()
    Imgproc.cvtColor(base, lum, Imgproc.COLOR_RGB2GRAY)

    // 3. Unsharp mask on luminance
    val kSize = ((detailRadius * 200.0 + 3.0).toInt() or 1).coerceIn(3, 101)
    val blurred = Mat()
    Imgproc.GaussianBlur(lum, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)

    val detail = Mat()
    Core.subtract(lum, blurred, detail)

    val amount = (surrealAmount * detailStrength * 2.5f).coerceAtMost(4.0f)
    val enhanced = Mat()
    Core.addWeighted(lum, 1.0, detail, amount.toDouble(), 0.0, enhanced)
    Core.max(enhanced, Scalar(1e-6), enhanced)

    // 4. ratio = enhanced / lum
    val ratio = Mat()
    Core.divide(enhanced, lum, ratio)

    // 5. Apply to all RGB channels
    val channels = mutableListOf<Mat>()
    Core.split(base, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
    }
    Core.merge(channels, output)

    base.release(); lum.release(); blurred.release()
    detail.release(); enhanced.release(); ratio.release()
    channels.forEach { it.release() }
}

// ── Durand (Bilateral Filter): edge-preserving base/detail decomposition ──
//
// Durand & Dorsey 2002: separates HDR log-luminance into base (large-scale)
// and detail (small-scale) using a bilateral filter. The base is compressed
// to fit display range while detail is preserved/amplified.

private fun durandToneMap(
    mats: List<Mat>, output: Mat,
    surrealAmount: Float,
    gamma: Float, saturationWeight: Float,
    detailRadius: Float
) {
    Log.d("HdrProcessor", "Durand bilateral filter (surreal=$surrealAmount)")

    val eps = 1e-6

    // 1. Build HDR radiance map
    val numImages = mats.size
    val times = MatOfFloat()
    val timeValues = FloatArray(numImages) { i ->
        Math.pow(2.0, (-2.0 + 4.0 * i / (numImages - 1)).toDouble()).toFloat()
    }
    times.fromArray(*timeValues)
    val calibrate = Photo.createCalibrateDebevec()
    val response = Mat()
    calibrate.process(mats, response, times)
    val merge = Photo.createMergeDebevec()
    val hdrMat = Mat()
    merge.process(mats, hdrMat, times, response)
    response.release(); times.release()

    val hdr32f = Mat()
    hdrMat.convertTo(hdr32f, CvType.CV_32F)

    // 2. Compute luminance
    val lum = Mat()
    Imgproc.cvtColor(hdr32f, lum, Imgproc.COLOR_RGB2GRAY)
    Core.add(lum, Scalar(eps), lum)

    // 3. Log luminance
    val logLum = Mat()
    Core.log(lum, logLum)

    // 4. Bilateral filter on log-luminance (edge-preserving base)
    val d = ((detailRadius * 60.0 + 5.0).toInt() or 1).coerceIn(5, 65)
    val sigmaColor = 30.0; val sigmaSpace = 30.0
    val baseLog = Mat()
    Imgproc.bilateralFilter(logLum, baseLog, d, sigmaColor, sigmaSpace)

    // 5. detail = logLum - baseLog
    val detailLog = Mat()
    Core.subtract(logLum, baseLog, detailLog)

    // 6. Compress base to ~40% of its range, amplify detail
    val compression = 0.4
    val detailBoost = 1.0 + surrealAmount * 2.0

    val compressedBase = Mat()
    Core.multiply(baseLog, Scalar(compression), compressedBase)

    val boostedDetail = Mat()
    Core.multiply(detailLog, Scalar(detailBoost), boostedDetail)

    val newLogLum = Mat()
    Core.add(compressedBase, boostedDetail, newLogLum)

    // 7. exp → linear luminance
    val newLum = Mat()
    Core.exp(newLogLum, newLum)

    // 8. Color ratio: newLum / lum, with saturation control
    val ratio = Mat()
    Core.divide(newLum, lum, ratio)

    // Saturation: pow(ratio, 1-sat) desaturates less when ratio varies
    Core.pow(ratio, (1.0 - saturationWeight * 0.3).toDouble(), ratio)

    val channels = mutableListOf<Mat>()
    Core.split(hdr32f, channels)
    for (ch in channels) {
        Core.multiply(ch, ratio, ch)
    }
    Core.merge(channels, output)

    // Apply gamma correction
    val gammaInv = 1.0 / gamma.toDouble()
    Core.pow(output, gammaInv, output)

    hdrMat.release(); hdr32f.release(); lum.release()
    logLum.release(); baseLog.release(); detailLog.release()
    compressedBase.release(); boostedDetail.release()
    newLogLum.release(); newLum.release(); ratio.release()
    channels.forEach { it.release() }
}

// ── CLAHE Boost: Mertens + adaptive histogram equalization on luminance ──
//
// Uses Contrast Limited Adaptive Histogram Equalization on the luminance
// channel of a Mertens fusion base. CLAHE enhances local contrast without
// halo artifacts. surrealAmount blends between original and CLAHE output.

private fun claheToneMap(
    mats: List<Mat>, output: Mat,
    surrealAmount: Float,
    contrastWeight: Float, saturationWeight: Float, exposureWeight: Float
) {
    Log.d("HdrProcessor", "CLAHE Boost (surreal=$surrealAmount)")

    // 1. Mertens exposure fusion → clean base
    val base = Mat()
    val merger = Photo.createMergeMertens(contrastWeight, saturationWeight, exposureWeight)
    merger.process(mats, base)

    // 2. Convert to 8-bit for CLAHE (CLAHE works on 8-bit integer)
    val base8 = Mat()
    base.convertTo(base8, CvType.CV_8UC3, 255.0)

    // 3. Extract luminance → Lab L channel
    val lab = Mat()
    Imgproc.cvtColor(base8, lab, Imgproc.COLOR_RGB2Lab)
    val labChannels = mutableListOf<Mat>()
    Core.split(lab, labChannels)
    val lOrig = labChannels[0].clone()

    // 4. Apply CLAHE on L channel
    val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
    val lEnhanced = Mat()
    clahe.apply(lOrig, lEnhanced)

    // 5. Blend original and CLAHE-enhanced by surrealAmount
    val lBlended = Mat()
    Core.addWeighted(lOrig, (1.0 - surrealAmount).toDouble(),
        lEnhanced, surrealAmount.toDouble(), 0.0, lBlended)

    // 6. Merge back and convert to RGB
    val resultChannels = listOf(lBlended, labChannels[1], labChannels[2])
    Core.merge(resultChannels, lab)
    val result = Mat()
    Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2RGB)

    // 7. Convert back to float [0, 1] for consistent pipeline
    result.convertTo(output, CvType.CV_32FC3, 1.0 / 255.0)

    base.release(); base8.release(); lab.release()
    lOrig.release(); lEnhanced.release(); lBlended.release()
    result.release()
    labChannels.forEach { it.release() }
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

// ── Ghost Artifact Reduction (post-process on tonemapped float output) ────

private fun reduceGhostArtifacts(rgb: Mat, output: Mat, strength: Float) {
    Log.d("HdrProcessor", "Reducing ghost artifacts (strength=$strength)")

    val kSize = (strength * 30.0 + 3.0).toInt() or 1
    val blurred = Mat()
    Imgproc.GaussianBlur(rgb, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)

    val blend = (strength.toDouble() * 0.4).coerceAtMost(0.6)
    Core.addWeighted(rgb, 1.0 - blend, blurred, blend, 0.0, output)

    blurred.release()
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

    val channels = mutableListOf<Mat>()
    Core.split(rgb32f, channels)
    val r = channels[0]; val g = channels[1]; val b = channels[2]

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

    val adaptedX = Mat()
    val adaptedZ = Mat()
    Core.addWeighted(X, chromAdaptStrength.toDouble(), Y, (1.0 - chromAdaptStrength), 0.0, adaptedX)
    Core.addWeighted(Z, chromAdaptStrength.toDouble(), Y, (1.0 - chromAdaptStrength), 0.0, adaptedZ)

    val localAdapt = Mat()
    val kSize = ((localAdaptKernel * 15.0).toInt() or 1).coerceAtLeast(3)
    Imgproc.GaussianBlur(Y, localAdapt, Size(kSize.toDouble(), kSize.toDouble()), 0.0)
    Core.add(localAdapt, Scalar(eps), localAdapt)

    val localPow = Mat()
    Core.pow(localAdapt, 0.7, localPow)
    val yDenom = Mat()
    Core.add(Y, localPow, yDenom)
    val yOut = Mat()
    Core.divide(Y, yDenom, yOut)

    val yRatio = Mat()
    Core.divide(yOut, Y, yRatio)
    val xOut = Mat()
    val zOut = Mat()
    Core.multiply(adaptedX, yRatio, xOut)
    Core.multiply(adaptedZ, yRatio, zOut)

    Core.pow(yRatio, (1.0 - colorSat).toDouble(), yRatio)
    Core.multiply(xOut, yRatio, xOut)
    Core.multiply(yOut, yRatio, yOut)
    Core.multiply(zOut, yRatio, zOut)

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
