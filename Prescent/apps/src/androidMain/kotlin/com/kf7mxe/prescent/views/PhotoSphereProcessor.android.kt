package com.kf7mxe.prescent.views

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
        // 1. Load all images
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
            val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB); rgba.release()
            return rgb
        }

        val mats = images.mapNotNull { loadMat(it) }
        if (mats.size < 2) { mats.forEach { it.release() }; return@withContext null }

        val useOrientations = orientations.size >= images.size

        val resultPath = if (useOrientations) {
            processSphereOrientations(mats, orientations, isPreview)
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
    isPreview: Boolean
): String? {
    val context = AndroidAppContext.applicationCtx
    Log.d("PhotoSphere", "Using gyroscope orientations for spherical projection")

    val imW = mats[0].cols(); val imH = mats[0].rows()

    // Estimate camera intrinsics (~70° horizontal FOV for phone main camera)
    val hfovDeg = 70.0
    val f = (imW / 2.0) / tan(Math.toRadians(hfovDeg / 2.0))
    val cx = imW / 2.0; val cy = imH / 2.0

    // Equirectangular output (2:1)
    val eqW = 4096; val eqH = 2048
    val acc = Mat(eqH, eqW, CvType.CV_32FC3, Scalar(0.0, 0.0, 0.0))
    val weightSum = Mat(eqH, eqW, CvType.CV_32F, Scalar(0.0))

    val refAz = orientations[0].first

    for (i in mats.indices) {
        val (azimuth, pitch) = orientations[i]
        val mat = mats[i]

        val theta = Math.toRadians((azimuth - refAz).toDouble())
        val phi = Math.toRadians(pitch.toDouble())

        val cosT = cos(theta); val sinT = sin(theta)
        val cosP = cos(phi); val sinP = sin(phi)

        // Camera axes in world space: X=right, Y=down, Z=optical axis
        val xCamX = cosT; val xCamY = 0.0; val xCamZ = -sinT
        val yCamX = sinP * sinT; val yCamY = -cosP; val yCamZ = sinP * cosT
        val zCamX = sinT * cosP; val zCamY = sinP; val zCamZ = cosT * cosP

        for (row in 0 until eqH) {
            for (col in 0 until eqW) {
                val lon = (col.toDouble() / eqW) * 2.0 * Math.PI - Math.PI
                val lat = Math.PI / 2.0 - (row.toDouble() / eqH) * Math.PI

                val vx = cos(lon) * cos(lat)
                val vy = sin(lat)
                val vz = sin(lon) * cos(lat)

                val camX = vx * xCamX + vy * xCamY + vz * xCamZ
                val camY_ = vx * yCamX + vy * yCamY + vz * yCamZ
                val camZ = vx * zCamX + vy * zCamY + vz * zCamZ

                if (camZ > 0.001) {
                    val imgU = f * camX / camZ + cx
                    val imgV = f * camY_ / camZ + cy

                    if (imgU >= 0.0 && imgU < imW - 1 && imgV >= 0.0 && imgV < imH - 1) {
                        val u1 = imgU.toInt(); val v1 = imgV.toInt()
                        val u2 = (u1 + 1).coerceAtMost(imW - 1)
                        val v2 = (v1 + 1).coerceAtMost(imH - 1)
                        val fx = imgU - u1; val fy = imgV - v1

                        val tl = mat.get(v1, u1)
                        val tr = mat.get(v1, u2)
                        val bl = mat.get(v2, u1)
                        val br = mat.get(v2, u2)

                        val r = tl[0] * (1.0 - fx) * (1.0 - fy) +
                                tr[0] * fx * (1.0 - fy) +
                                bl[0] * (1.0 - fx) * fy +
                                br[0] * fx * fy
                        val g = tl[1] * (1.0 - fx) * (1.0 - fy) +
                                tr[1] * fx * (1.0 - fy) +
                                bl[1] * (1.0 - fx) * fy +
                                br[1] * fx * fy
                        val b = tl[2] * (1.0 - fx) * (1.0 - fy) +
                                tr[2] * fx * (1.0 - fy) +
                                bl[2] * (1.0 - fx) * fy +
                                br[2] * fx * fy

                        val dx = (imgU - cx) / cx; val dy = (imgV - cy) / cy
                        val w = exp(-(dx * dx + dy * dy) * 2.0)

                        val cur = acc.get(row, col)
                        acc.put(row, col, floatArrayOf(
                            (cur[0] + r * w).toFloat(),
                            (cur[1] + g * w).toFloat(),
                            (cur[2] + b * w).toFloat()
                        ))
                        val ws = weightSum.get(row, col)[0]
                        weightSum.put(row, col, floatArrayOf((ws + w).toFloat()))
                    }
                }
            }
        }
    }

    // Normalize accumulated RGB by weight
    val result = Mat(eqH, eqW, CvType.CV_32FC3, Scalar(0.0, 0.0, 0.0))
    for (row in 0 until eqH) {
        for (col in 0 until eqW) {
            val w = weightSum.get(row, col)[0]
            if (w > 0.01) {
                val c = acc.get(row, col)
                result.put(row, col, floatArrayOf(
                    (c[0] / w).toFloat(), (c[1] / w).toFloat(), (c[2] / w).toFloat()
                ))
            }
        }
    }

    acc.release(); weightSum.release()

    // Convert to RGBA for bitmap output
    val rgbResult = Mat(eqH, eqW, CvType.CV_8UC3)
    result.convertTo(rgbResult, CvType.CV_8UC3, 1.0)
    val rgbaResult = Mat()
    Imgproc.cvtColor(rgbResult, rgbaResult, Imgproc.COLOR_RGB2RGBA)
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

        // Find JPEG EOI marker (0xFFD9) and insert APP1 before it
        var insertAt = -1
        for (i in jpegBytes.size - 2 downTo 0) {
            val b0 = jpegBytes[i].toInt() and 0xFF
            val b1 = jpegBytes[i + 1].toInt() and 0xFF
            if (b0 == 0xFF && b1 == 0xD9) {
                insertAt = i
                break
            }
        }

        if (insertAt >= 0) {
            val out = ByteArrayOutputStream()
            out.write(jpegBytes, 0, insertAt)
            out.write(app1.toByteArray())
            out.write(jpegBytes, insertAt, jpegBytes.size - insertAt)
            jpegFile.writeBytes(out.toByteArray())
            Log.d("PhotoSphere", "XMP metadata embedded")
        } else {
            Log.w("PhotoSphere", "Could not find JPEG EOI marker")
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
        if (descPano.empty() || descNext.empty()) continue
        matcher.match(descPano, descNext, matches)

        val matchList = matches.toList()
        val goodMatches = matchList.filter { it.distance < 50f }
        if (goodMatches.size < 10) {
            Log.w("PhotoSphere", "Frame $i: only ${goodMatches.size} good matches, skipping")
            continue
        }

        val srcPts = MatOfPoint2f(); val dstPts = MatOfPoint2f()
        val kpPanoList = kpPano.toList(); val kpNextList = kpNext.toList()
        val srcArr = goodMatches.map { kpNextList[it.trainIdx].pt }.toTypedArray()
        val dstArr = goodMatches.map { kpPanoList[it.queryIdx].pt }.toTypedArray()
        srcPts.fromArray(*srcArr); dstPts.fromArray(*dstArr)

        val inlierMask = Mat()
        val H = Calib3d.findHomography(srcPts, dstPts, Calib3d.RANSAC, 5.0, inlierMask)
        if (H == null) { Log.w("PhotoSphere", "Frame $i: homography failed"); continue }

        val h = panorama.rows(); val w = maxPanoramaW
        val warped = Mat()
        Imgproc.warpPerspective(next, warped, H, Size(w.toDouble(), h.toDouble()))

        val extended = Mat(Size(w.toDouble(), h.toDouble()), panorama.type(), Scalar(0.0))
        val roi = extended.submat(0, h, 0, panorama.cols())
        panorama.copyTo(roi); roi.release()

        val mask = Mat()
        Core.inRange(warped, Scalar(1.0, 1.0, 1.0), Scalar(255.0, 255.0, 255.0), mask)
        warped.copyTo(extended, mask); mask.release()

        panorama.release(); panorama = extended

        kpPano.release(); kpNext.release()
        descPano.release(); descNext.release()
        matches.release(); inlierMask.release(); H.release()
    }

    // Crop black borders
    val grayPano = Mat()
    Imgproc.cvtColor(panorama, grayPano, Imgproc.COLOR_RGB2GRAY)
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
