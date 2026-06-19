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

actual suspend fun processPhotoSphere(
    images: List<String>,
    maxPreviewSize: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val isPreview = maxPreviewSize > 0
    Log.d("PhotoSphere", "Processing ${images.size} images")

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
        if (mats.size < 2) return@withContext null

        // 2. Stitch images: start with first, then stitch each subsequent onto a panorama
        val orb = ORB.create(2000)
        val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)

        var panorama = mats[0].clone()
        val maxPanoramaW = panorama.cols() * 4

        for (i in 1 until mats.size) {
            Log.d("PhotoSphere", "Stitching frame $i/${mats.size - 1}")

            val next = mats[i]

            // Detect features
            val kpPano = MatOfKeyPoint(); val kpNext = MatOfKeyPoint()
            val descPano = Mat(); val descNext = Mat()
            orb.detectAndCompute(panorama, Mat(), kpPano, descPano)
            orb.detectAndCompute(next, Mat(), kpNext, descNext)

            // Match features
            val matches = MatOfDMatch()
            if (descPano.empty() || descNext.empty()) continue
            matcher.match(descPano, descNext, matches)

            val matchList = matches.toList()
            val goodMatches = matchList.filter { it.distance < 50f }
            if (goodMatches.size < 10) {
                Log.w("PhotoSphere", "Frame $i: only ${goodMatches.size} good matches, skipping")
                continue
            }

            // Find homography from next → panorama
            val srcPts = MatOfPoint2f()
            val dstPts = MatOfPoint2f()
            val kpPanoList = kpPano.toList()
            val kpNextList = kpNext.toList()
            val srcArr = goodMatches.map { kpNextList[it.trainIdx].pt }.toTypedArray()
            val dstArr = goodMatches.map { kpPanoList[it.queryIdx].pt }.toTypedArray()
            srcPts.fromArray(*srcArr)
            dstPts.fromArray(*dstArr)

            val inlierMask = Mat()
            val H = Calib3d.findHomography(srcPts, dstPts, Calib3d.RANSAC, 5.0, inlierMask)
            if (H == null) { Log.w("PhotoSphere", "Frame $i: homography failed"); continue }

            // Warp next onto panorama
            val h = panorama.rows(); val w = maxPanoramaW
            val warped = Mat()
            Imgproc.warpPerspective(next, warped, H, Size(w.toDouble(), h.toDouble()))

            // Create canvas for extended panorama
            val extended = Mat(Size(w.toDouble(), h.toDouble()), panorama.type(), Scalar(0.0))
            // Copy existing panorama
            val roi = extended.submat(0, h, 0, panorama.cols())
            panorama.copyTo(roi)
            roi.release()

            // Blend warped image
            val mask = Mat()
            Core.inRange(warped, Scalar(1.0, 1.0, 1.0), Scalar(255.0, 255.0, 255.0), mask)
            warped.copyTo(extended, mask)
            mask.release()

            panorama.release()
            panorama = extended

            kpPano.release(); kpNext.release()
            descPano.release(); descNext.release()
            matches.release(); inlierMask.release(); H.release()
        }

        // 3. Crop to content (remove black borders) — scan non-zero rows
        val grayPano = Mat(); Imgproc.cvtColor(panorama, grayPano, Imgproc.COLOR_RGB2GRAY)
        val rows = grayPano.rows(); val cols = grayPano.cols()
        var top = 0; var bottom = rows - 1
        val rowSum = Mat()
        while (top < bottom) {
            Core.reduce(grayPano.row(top), rowSum, 0, Core.REDUCE_SUM, CvType.CV_64F)
            if (Core.sumElems(rowSum).`val`[0] > 1.0) break
            top++
        }
        while (bottom > top) {
            Core.reduce(grayPano.row(bottom), rowSum, 0, Core.REDUCE_SUM, CvType.CV_64F)
            if (Core.sumElems(rowSum).`val`[0] > 1.0) break
            bottom--
        }
        var left = 0; var right = cols - 1
        val colSum = Mat()
        while (left < right) {
            Core.reduce(grayPano.col(left), colSum, 1, Core.REDUCE_SUM, CvType.CV_64F)
            if (Core.sumElems(colSum).`val`[0] > 1.0) break
            left++
        }
        while (right > left) {
            Core.reduce(grayPano.col(right), colSum, 1, Core.REDUCE_SUM, CvType.CV_64F)
            if (Core.sumElems(colSum).`val`[0] > 1.0) break
            right--
        }
        grayPano.release(); rowSum.release(); colSum.release()

        var cropped = panorama
        val margin = 10
        val cropX = (left - margin).coerceAtLeast(0)
        val cropY = (top - margin).coerceAtLeast(0)
        val cropW = (right - left + margin * 2).coerceAtMost(cols - cropX)
        val cropH = (bottom - top + margin * 2).coerceAtMost(rows - cropY)
        if (cropW > 0 && cropH > 0) {
            cropped = Mat(panorama, Rect(cropX, cropY, cropW, cropH))
        }

        // 4. Save
        val bitmap = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(cropped, bitmap)
        if (cropped !== panorama) cropped.release()
        panorama.release()
        mats.forEach { it.release() }

        val filename = "photosphere_${System.currentTimeMillis()}.jpg"

        val savedPath = if (isPreview) {
            val previewFile = File(context.cacheDir, filename)
            FileOutputStream(previewFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
            bitmap.recycle()
            previewFile.absolutePath
        } else {
            // Save to gallery
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
            } catch (e: Exception) { Log.e("PhotoSphere", "save failed", e) }
            val cacheFile = File(context.cacheDir, filename)
            FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            bitmap.recycle()
            cacheFile.absolutePath
        }
        Log.d("PhotoSphere", "Complete: $savedPath")
        savedPath
    } catch (e: Exception) {
        Log.e("PhotoSphere", "Processing failed", e)
        null
    }
}
