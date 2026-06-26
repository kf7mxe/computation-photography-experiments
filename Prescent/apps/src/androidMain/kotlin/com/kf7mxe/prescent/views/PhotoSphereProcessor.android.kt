package com.kf7mxe.prescent.views

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.ExifInterface
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
import org.opencv.features2d.ORB
import org.opencv.features2d.DescriptorMatcher
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream
import java.io.ByteArrayOutputStream
import java.nio.charset.Charset
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.tan
import kotlin.math.atan2
import kotlin.math.sqrt

actual suspend fun processPhotoSphere(
    images: List<String>,
    orientations: List<Pair<Float, Float>>,
    maxPreviewSize: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val isPreview = maxPreviewSize > 0
    Log.d("PhotoSphere", "Processing ${images.size} images, orientations: ${orientations.size}")

    if (images.size < 2) return@withContext null

    try {
        // 1. Load all images and read EXIF rotation for each
        fun loadMat(path: String): Mat? {
            val opts = BitmapFactory.Options()
            if (isPreview) {
                opts.inJustDecodeBounds = true
                BitmapFactory.decodeFile(path, opts)
                val w = opts.outWidth; val h = opts.outHeight
                var s = 1; while (w / s > maxPreviewSize || h / s > maxPreviewSize) s *= 2
                opts.inSampleSize = s; opts.inJustDecodeBounds = false
            }
            val bitmap = BitmapFactory.decodeFile(path, opts) ?: return null
            val rgba = Mat(); Utils.bitmapToMat(bitmap, rgba); bitmap.recycle()
            val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_BGRA2BGR); rgba.release()
            return rgb
        }

        val exifRotations = images.map { path ->
            try {
                val exif = ExifInterface(path)
                exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } catch (_: Exception) { ExifInterface.ORIENTATION_NORMAL }
        }

        val mats = images.mapNotNull { loadMat(it) }
        if (mats.size < 2) { mats.forEach { it.release() }; return@withContext null }

        val useOrientations = orientations.size >= images.size

        val resultPath = if (useOrientations) {
            processSphereOrientations(mats, orientations, exifRotations, isPreview)
        } else {
            processHomographyFallback(mats, isPreview)
        }

        mats.forEach { it.release() }
        resultPath
    } catch (e: Exception) {
        Log.e("PhotoSphere", "Processing failed", e)
        null
    }
}

private fun processSphereOrientations(
    mats: List<Mat>,
    orientations: List<Pair<Float, Float>>,
    exifRotations: List<Int>,
    isPreview: Boolean
): String? {
    val context = AndroidAppContext.applicationCtx
    Log.d("PhotoSphere", "Using gyroscope orientations for spherical projection")

    val imW = mats[0].cols(); val imH = mats[0].rows()

    // Determine sensor-native dimensions from the first image's EXIF rotation
    val firstRot = exifRotations.getOrElse(0) { ExifInterface.ORIENTATION_NORMAL }
    val sensorW: Int; val sensorH: Int
    when (firstRot) {
        ExifInterface.ORIENTATION_ROTATE_90, ExifInterface.ORIENTATION_ROTATE_270 -> {
            sensorW = imH; sensorH = imW
        }
        else -> {
            sensorW = imW; sensorH = imH
        }
    }

    // Estimate camera intrinsics using the sensor's native dimensions
    val hfovDeg = 70.0
    val sensorLongPx = maxOf(sensorW, sensorH)
    val f = (sensorLongPx / 2.0) / tan(Math.toRadians(hfovDeg / 2.0))
    val cx_s = sensorW / 2.0
    val cy_s = sensorH / 2.0

    Log.d("PhotoSphere", "Sensor: ${sensorW}x$sensorH, image: ${imW}x$imH, first EXIF: $firstRot")

    // Equirectangular output (2:1)
    val eqW = 4096; val eqH = 2048
    val acc = Mat(eqH, eqW, CvType.CV_32FC3, Scalar(0.0, 0.0, 0.0))
    val weightSum = Mat(eqH, eqW, CvType.CV_32F, Scalar(0.0))

    // Precompute spherical direction vectors for all equirectangular pixels
    val vx = FloatArray(eqH * eqW)
    val vy = FloatArray(eqH * eqW)
    val vz = FloatArray(eqH * eqW)
    var idx = 0
    for (row in 0 until eqH) {
        val lat = Math.PI / 2.0 - (row.toDouble() / eqH) * Math.PI
        val cosLat = cos(lat)
        val sinLat = sin(lat)
        for (col in 0 until eqW) {
            val lon = (col.toDouble() / eqW) * 2.0 * Math.PI - Math.PI
            vx[idx] = (cos(lon) * cosLat).toFloat()
            vy[idx] = sinLat.toFloat()
            vz[idx] = (sin(lon) * cosLat).toFloat()
            idx++
        }
    }

    val refAz = orientations[0].first

    for (i in mats.indices) {
        val (azimuth, pitch) = orientations[i]
        val mat = mats[i]
        val exifRot = exifRotations.getOrElse(i) { ExifInterface.ORIENTATION_NORMAL }

        val theta = Math.toRadians((azimuth - refAz).toDouble())
        val phi = Math.toRadians(pitch.toDouble())

        val cosT = cos(theta).toFloat()
        val sinT = sin(theta).toFloat()
        val cosP = cos(phi).toFloat()
        val sinP = sin(phi).toFloat()

        // Camera axes in world space: X=right, Y=down, Z=optical axis
        // These map to sensor-native pixel coordinates:
        //   sensor_u = f * camX / camZ + cx_s
        //   sensor_v = f * camY_ / camZ + cy_s
        val xCamX = cosT; val xCamY = 0f; val xCamZ = -sinT
        val yCamX = sinP * sinT; val yCamY = -cosP; val yCamZ = sinP * cosT
        val zCamX = sinT * cosP; val zCamY = sinP; val zCamZ = cosT * cosP

        val mapXData = FloatArray(eqH * eqW)
        val mapYData = FloatArray(eqH * eqW)
        val weightData = FloatArray(eqH * eqW)

        var pIdx = 0
        for (row in 0 until eqH) {
            for (col in 0 until eqW) {
                val px = vx[pIdx]; val py = vy[pIdx]; val pz = vz[pIdx]
                val camX = px * xCamX + py * xCamY + pz * xCamZ
                val camY_ = px * yCamX + py * yCamY + pz * yCamZ
                val camZ = px * zCamX + py * zCamY + pz * zCamZ

                if (camZ > 0.001f) {
                    // Project to sensor-native coordinates
                    val sensorU = f * camX / camZ + cx_s
                    val sensorV = f * camY_ / camZ + cy_s

                    // Convert sensor → image coordinates using EXIF rotation
                    val imgU: Double
                    val imgV: Double
                    when (exifRot) {
                        ExifInterface.ORIENTATION_ROTATE_90 -> {
                            // 90° CW: image_u = sensor_v, image_v = sensorW - sensor_u
                            imgU = sensorV
                            imgV = sensorW.toDouble() - sensorU
                        }
                        ExifInterface.ORIENTATION_ROTATE_180 -> {
                            imgU = sensorW.toDouble() - sensorU
                            imgV = sensorH.toDouble() - sensorV
                        }
                        ExifInterface.ORIENTATION_ROTATE_270 -> {
                            // 90° CCW: image_u = sensorH - sensorV, image_v = sensor_u
                            imgU = sensorH.toDouble() - sensorV
                            imgV = sensorU
                        }
                        else -> {
                            imgU = sensorU
                            imgV = sensorV
                        }
                    }

                    if (imgU >= 0f && imgU < imW - 1 && imgV >= 0f && imgV < imH - 1) {
                        mapXData[pIdx] = imgU.toFloat()
                        mapYData[pIdx] = imgV.toFloat()
                        // Weight based on distance from sensor optical center (in sensor coords)
                        val dx = (sensorU - cx_s) / cx_s
                        val dy = (sensorV - cy_s) / cy_s
                        weightData[pIdx] = exp(-(dx * dx + dy * dy) * 2.0).toFloat()
                    } else {
                        mapXData[pIdx] = -1f
                        mapYData[pIdx] = -1f
                        weightData[pIdx] = 0f
                    }
                } else {
                    mapXData[pIdx] = -1f
                    mapYData[pIdx] = -1f
                    weightData[pIdx] = 0f
                }
                pIdx++
            }
        }

        val mapX = Mat(eqH, eqW, CvType.CV_32F)
        val mapY = Mat(eqH, eqW, CvType.CV_32F)
        val weightMap = Mat(eqH, eqW, CvType.CV_32F)
        mapX.put(0, 0, mapXData)
        mapY.put(0, 0, mapYData)
        weightMap.put(0, 0, weightData)

        val warped = Mat()
        Imgproc.remap(mat, warped, mapX, mapY, Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0, 0.0, 0.0))

        val warped32f = Mat()
        warped.convertTo(warped32f, CvType.CV_32FC3, 1.0)
        warped.release()

        val channels = mutableListOf<Mat>()
        Core.split(warped32f, channels)
        warped32f.release()

        for (ch in channels) {
            Core.multiply(ch, weightMap, ch)
        }

        val weighted = Mat()
        Core.merge(channels, weighted)
        channels.forEach { it.release() }

        Core.add(acc, weighted, acc)
        weighted.release()

        Core.add(weightSum, weightMap, weightSum)
        weightMap.release(); mapX.release(); mapY.release()
    }

    // Normalize accumulated RGB by weight: result = acc / weightSum (per-channel)
    val channels = mutableListOf<Mat>()
    Core.split(acc, channels)
    acc.release()

    val mask = Mat()
    Core.compare(weightSum, Scalar(0.01), mask, Core.CMP_GT)

    for (ch in channels) {
        val temp = Mat()
        Core.divide(ch, weightSum, temp)
        temp.copyTo(ch, mask)
        temp.release()
    }
    mask.release()
    weightSum.release()

    val result = Mat()
    Core.merge(channels, result)
    channels.forEach { it.release() }

    // Convert to RGBA for bitmap output
    val rgbResult = Mat(eqH, eqW, CvType.CV_8UC3)
    result.convertTo(rgbResult, CvType.CV_8UC3, 1.0)
    val rgbaResult = Mat()
    Imgproc.cvtColor(rgbResult, rgbaResult, Imgproc.COLOR_BGR2RGBA)
    rgbResult.release(); result.release()

    val bitmap = Bitmap.createBitmap(rgbaResult.cols(), rgbaResult.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(rgbaResult, bitmap)
    rgbaResult.release()

    val filename = "photosphere_${System.currentTimeMillis()}.jpg"
    val savedPath = saveOutputImage(bitmap, filename, isPreview)
    bitmap.recycle()

    // Embed XMP Photo Sphere metadata for full-size output
    if (!isPreview && savedPath != null) {
        embedXmpMetadata(savedPath, eqW, eqH, orientations.firstOrNull()?.first ?: 0f)
    }

    Log.d("PhotoSphere", "Equirectangular complete: $savedPath")
    return savedPath
}

private fun saveOutputImage(bitmap: Bitmap, filename: String, isPreview: Boolean): String? {
    val context = AndroidAppContext.applicationCtx
    return if (isPreview) {
        val previewFile = File(context.cacheDir, filename)
        FileOutputStream(previewFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        previewFile.absolutePath
    } else {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Prescent/PhotoSphere")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val uri = context.contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
                )
                if (uri != null) {
                    context.contentResolver.openOutputStream(uri)?.use {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)
                    }
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    context.contentResolver.update(uri, values, null, null)
                }
            } else {
                val dir = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                    "Prescent/PhotoSphere"
                ).also { it.mkdirs() }
                val f = File(dir, filename)
                FileOutputStream(f).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                android.media.MediaScannerConnection.scanFile(
                    context, arrayOf(f.absolutePath), null, null
                )
            }
        } catch (e: Exception) {
            Log.e("PhotoSphere", "save failed", e)
        }
        val cacheFile = File(context.cacheDir, filename)
        FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        cacheFile.absolutePath
    }
}

private fun embedXmpMetadata(jpegPath: String, fullW: Int, fullH: Int, headingDeg: Float) {
    try {
        val xmp = StringBuilder().apply {
            append("""<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>""")
            append("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">")
            append("<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">")
            append("<rdf:Description rdf:about=\"\" xmlns:GPano=\"http://ns.google.com/photos/1.0/panorama/\">")
            append("<GPano:ProjectionType>equirectangular</GPano:ProjectionType>")
            append("<GPano:UsePanoramaViewer>True</GPano:UsePanoramaViewer>")
            append("<GPano:CroppedAreaImageWidthPixels>$fullW</GPano:CroppedAreaImageWidthPixels>")
            append("<GPano:CroppedAreaImageHeightPixels>$fullH</GPano:CroppedAreaImageHeightPixels>")
            append("<GPano:FullPanoWidthPixels>$fullW</GPano:FullPanoWidthPixels>")
            append("<GPano:FullPanoHeightPixels>$fullH</GPano:FullPanoHeightPixels>")
            append("<GPano:CroppedAreaLeftPixels>0</GPano:CroppedAreaLeftPixels>")
            append("<GPano:CroppedAreaTopPixels>0</GPano:CroppedAreaTopPixels>")
            append("<GPano:InitialViewHeadingDegrees>$headingDeg</GPano:InitialViewHeadingDegrees>")
            append("<GPano:InitialHorizontalFOVDegrees>70.0</GPano:InitialHorizontalFOVDegrees>")
            append("<GPano:StitchingSoftware>Prescent</GPano:StitchingSoftware>")
            append("<GPano:PoseHeadingDegrees>$headingDeg</GPano:PoseHeadingDegrees>")
            append("</rdf:Description>")
            append("</rdf:RDF>")
            append("</x:xmpmeta>")
            append("<?xpacket end=\"w\"?>")
        }.toString()

        val xmpBytes = xmp.toByteArray(Charset.forName("UTF-8"))
        val xmpId = "http://ns.adobe.com/xap/1.0/".toByteArray(Charset.forName("UTF-8"))
        val app1Size = 2 + xmpId.size + 1 + xmpBytes.size
        val app1 = ByteArrayOutputStream()
        app1.write(0xFF); app1.write(0xE1)
        app1.write((app1Size shr 8) and 0xFF); app1.write(app1Size and 0xFF)
        app1.write(xmpId); app1.write(0x00)
        app1.write(xmpBytes)

        val jpegFile = File(jpegPath)
        val jpegBytes = jpegFile.readBytes()

        // Find JPEG SOS marker (0xFFDA) and insert APP1 before it
        // XMP must be in the header area between SOI and SOS
        var insertAt = -1
        var i = 2
        while (i < jpegBytes.size - 1) {
            val b0 = jpegBytes[i].toInt() and 0xFF
            val b1 = jpegBytes[i + 1].toInt() and 0xFF
            if (b0 == 0xFF && b1 == 0xDA) {
                insertAt = i
                break
            }
            if (b0 == 0xFF && b1 >= 0xE0 && b1 <= 0xEF) {
                // Skip this APP marker: FF En | size (2 bytes big-endian)
                val segSize = ((jpegBytes[i + 2].toInt() and 0xFF) shl 8) or (jpegBytes[i + 3].toInt() and 0xFF)
                i += 2 + segSize
            } else if (b0 == 0xFF && b1 != 0x00 && b1 != 0xFF) {
                // Skip other markers: FF XX | size (2 bytes big-endian)
                val segSize = ((jpegBytes[i + 2].toInt() and 0xFF) shl 8) or (jpegBytes[i + 3].toInt() and 0xFF)
                i += 2 + segSize
            } else {
                i++
            }
        }

        if (insertAt >= 0) {
            val out = ByteArrayOutputStream()
            out.write(jpegBytes, 0, insertAt)
            out.write(app1.toByteArray())
            out.write(jpegBytes, insertAt, jpegBytes.size - insertAt)
            jpegFile.writeBytes(out.toByteArray())
            Log.d("PhotoSphere", "XMP metadata embedded before SOS at offset $insertAt")
        } else {
            Log.w("PhotoSphere", "Could not find JPEG SOS marker")
        }
    } catch (e: Exception) {
        Log.e("PhotoSphere", "XMP embedding failed", e)
    }
}

private fun processHomographyFallback(mats: List<Mat>, isPreview: Boolean): String? {
    val context = AndroidAppContext.applicationCtx
    if (mats.size < 2) return null

    Log.d("PhotoSphere", "Fallback: homography stitching")
    val orb = ORB.create(2000)
    val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)

    var panorama = mats[0].clone()
    val maxPanoramaW = panorama.cols() * 4

    for (i in 1 until mats.size) {
        Log.d("PhotoSphere", "Stitching frame $i/${mats.size - 1}")
        val next = mats[i]

        val kpPano = MatOfKeyPoint(); val kpNext = MatOfKeyPoint()
        val descPano = Mat(); val descNext = Mat()
        orb.detectAndCompute(panorama, Mat(), kpPano, descPano)
        orb.detectAndCompute(next, Mat(), kpNext, descNext)

        val matches = MatOfDMatch()
        if (descPano.empty() || descNext.empty()) {
            kpPano.release(); kpNext.release(); descPano.release(); descNext.release(); matches.release()
            continue
        }
        matcher.match(descPano, descNext, matches)

        val matchList = matches.toList()
        val goodMatches = matchList.filter { it.distance < 50f }
        if (goodMatches.size < 10) {
            Log.w("PhotoSphere", "Frame $i: only ${goodMatches.size} good matches, skipping")
            kpPano.release(); kpNext.release(); descPano.release(); descNext.release(); matches.release()
            continue
        }

        val srcPts = MatOfPoint2f(); val dstPts = MatOfPoint2f()
        val kpPanoList = kpPano.toList(); val kpNextList = kpNext.toList()
        val srcArr = goodMatches.map { kpNextList[it.trainIdx].pt }.toTypedArray()
        val dstArr = goodMatches.map { kpPanoList[it.queryIdx].pt }.toTypedArray()
        srcPts.fromArray(*srcArr); dstPts.fromArray(*dstArr)

        val inlierMask = Mat()
        val H = Calib3d.findHomography(srcPts, dstPts, Calib3d.RANSAC, 5.0, inlierMask)
        if (H == null) {
            Log.w("PhotoSphere", "Frame $i: homography failed")
            kpPano.release(); kpNext.release(); descPano.release(); descNext.release(); matches.release()
            srcPts.release(); dstPts.release(); inlierMask.release()
            continue
        }

        val h = panorama.rows(); val w = maxPanoramaW
        val warped = Mat()
        Imgproc.warpPerspective(next, warped, H, Size(w.toDouble(), h.toDouble()))

        val extended = Mat(Size(w.toDouble(), h.toDouble()), panorama.type(), Scalar(0.0))
        val roi = extended.submat(0, h, 0, panorama.cols())
        panorama.copyTo(roi); roi.release()

        val mask = Mat()
        Core.inRange(warped, Scalar(1.0, 1.0, 1.0), Scalar(255.0, 255.0, 255.0), mask)
        warped.copyTo(extended, mask); mask.release()
        warped.release()

        panorama.release(); panorama = extended

        kpPano.release(); kpNext.release()
        descPano.release(); descNext.release()
        matches.release(); inlierMask.release(); H.release()
        srcPts.release(); dstPts.release()
    }

    // Crop black borders
    val grayPano = Mat()
    Imgproc.cvtColor(panorama, grayPano, Imgproc.COLOR_BGR2GRAY)
    val panoRows = grayPano.rows(); val panoCols = grayPano.cols()

    var top = 0; var bottom = panoRows - 1
    val rowSum = Mat()
    while (top < bottom) {
        Core.reduce(grayPano.row(top), rowSum, 0, Core.REDUCE_SUM, CvType.CV_64F)
        if (Core.sumElems(rowSum).`val`[0] > 1.0) break; top++
    }
    while (bottom > top) {
        Core.reduce(grayPano.row(bottom), rowSum, 0, Core.REDUCE_SUM, CvType.CV_64F)
        if (Core.sumElems(rowSum).`val`[0] > 1.0) break; bottom--
    }
    var left = 0; var right = panoCols - 1
    val colSum = Mat()
    while (left < right) {
        Core.reduce(grayPano.col(left), colSum, 1, Core.REDUCE_SUM, CvType.CV_64F)
        if (Core.sumElems(colSum).`val`[0] > 1.0) break; left++
    }
    while (right > left) {
        Core.reduce(grayPano.col(right), colSum, 1, Core.REDUCE_SUM, CvType.CV_64F)
        if (Core.sumElems(colSum).`val`[0] > 1.0) break; right--
    }
    grayPano.release(); rowSum.release(); colSum.release()

    val margin = 10
    val cropX = (left - margin).coerceAtLeast(0)
    val cropY = (top - margin).coerceAtLeast(0)
    val cropW = (right - left + margin * 2).coerceAtMost(panoCols - cropX)
    val cropH = (bottom - top + margin * 2).coerceAtMost(panoRows - cropY)
    var cropped = panorama
    if (cropW > 0 && cropH > 0) {
        cropped = Mat(panorama, Rect(cropX, cropY, cropW, cropH))
    }

    val bitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(cropped, bitmap)
    if (cropped !== panorama) cropped.release()
    panorama.release()

    val filename = "photosphere_pano_${System.currentTimeMillis()}.jpg"
    val savedPath = if (isPreview) {
        val previewFile = File(context.cacheDir, filename)
        FileOutputStream(previewFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        bitmap.recycle()
        previewFile.absolutePath
    } else {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Prescent")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                val uri = context.contentResolver.insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values
                )
                if (uri != null) {
                    context.contentResolver.openOutputStream(uri)?.use {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)
                    }
                    values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    context.contentResolver.update(uri, values, null, null)
                }
            } else {
                val dir = File(
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
                    "Prescent"
                ).also { it.mkdirs() }
                val f = File(dir, filename)
                FileOutputStream(f).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                android.media.MediaScannerConnection.scanFile(
                    context, arrayOf(f.absolutePath), null, null
                )
            }
        } catch (e: Exception) { Log.e("PhotoSphere", "save failed", e) }
        val cacheFile = File(context.cacheDir, filename)
        FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        bitmap.recycle()
        cacheFile.absolutePath
    }
    Log.d("PhotoSphere", "Homography stitch complete: $savedPath")
    return savedPath
}

actual suspend fun computeVisualRotation(prevPath: String, currentPath: String): Float? = withContext(Dispatchers.IO) {
    computeVisualRotationSync(prevPath, currentPath)
}

/**
 * Synchronous version of [computeVisualRotation] — call on [Dispatchers.IO].
 * Computes the visual rotation angle (in degrees) between two sphere images
 * using ORB feature matching + affine transform estimation.
 * Returns null if matching fails (too few features, motion blur, etc.).
 */
fun computeVisualRotationSync(prevPath: String, currentPath: String): Float? {
    val opts = BitmapFactory.Options().apply { inSampleSize = 2 }
    val bmp1 = BitmapFactory.decodeFile(prevPath, opts) ?: return null
    val bmp2 = BitmapFactory.decodeFile(currentPath, opts) ?: return null

    val m1 = Mat(); Utils.bitmapToMat(bmp1, m1); bmp1.recycle()
    val m2 = Mat(); Utils.bitmapToMat(bmp2, m2); bmp2.recycle()
    val g1 = Mat(); Imgproc.cvtColor(m1, g1, Imgproc.COLOR_BGRA2GRAY); m1.release()
    val g2 = Mat(); Imgproc.cvtColor(m2, g2, Imgproc.COLOR_BGRA2GRAY); m2.release()

    val orb = ORB.create(1500)
    val kp1 = MatOfKeyPoint(); val desc1 = Mat()
    val kp2 = MatOfKeyPoint(); val desc2 = Mat()
    orb.detectAndCompute(g1, Mat(), kp1, desc1)
    orb.detectAndCompute(g2, Mat(), kp2, desc2)
    g1.release(); g2.release()

    if (desc1.empty() || desc2.empty()) {
        kp1.release(); kp2.release(); desc1.release(); desc2.release()
        return null
    }

    val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
    val matches = MatOfDMatch()
    matcher.match(desc1, desc2, matches)

    val matchList = matches.toList()
    val good = matchList.filter { it.distance < 50f }
    if (good.size < 8) {
        kp1.release(); kp2.release(); desc1.release(); desc2.release(); matches.release()
        return null
    }

    val srcPts = MatOfPoint2f()
    val dstPts = MatOfPoint2f()
    val kpList1 = kp1.toList(); val kpList2 = kp2.toList()
    srcPts.fromArray(*good.map { kpList1[it.queryIdx].pt }.toTypedArray())
    dstPts.fromArray(*good.map { kpList2[it.trainIdx].pt }.toTypedArray())

    val inlierMask = Mat()
    val affine = Calib3d.estimateAffine2D(srcPts, dstPts, inlierMask, Calib3d.RANSAC, 3.0)

    kp1.release(); kp2.release(); desc1.release(); desc2.release(); matches.release()
    srcPts.release(); dstPts.release(); inlierMask.release()

    if (affine == null || affine.empty()) return null

    val a = affine.get(0, 0)[0]; val c = affine.get(1, 0)[0]
    val s = sqrt(a * a + c * c)
    if (s < 0.3 || s > 3.0) { affine.release(); return null }

    val rotDeg = Math.toDegrees(atan2(c / s, a / s))
    affine.release()
    return rotDeg.toFloat()
}

