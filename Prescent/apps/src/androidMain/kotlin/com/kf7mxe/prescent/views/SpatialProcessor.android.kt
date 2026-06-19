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

actual suspend fun processSpatial(
    images: List<String>,
    maxPreviewSize: Int,
    rotation: Int
): SpatialResult? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val isPreview = maxPreviewSize > 0
    Log.d("Spatial", "Processing ${images.size} images, preview=$isPreview, rotation=$rotation")

    if (images.size < 2) return@withContext null

    try {
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

        var left = loadMat(images[0])
        var right = loadMat(images[1])
        if (left == null || right == null) {
            left?.release(); right?.release()
            Log.e("Spatial", "Failed to load images")
            return@withContext null
        }

        // Apply rotation before any processing — Core.rotate for exact 90/180/270
        if (rotation != 0) {
            val code = when (rotation) {
                90 -> Core.ROTATE_90_CLOCKWISE
                180 -> Core.ROTATE_180
                270 -> Core.ROTATE_90_COUNTERCLOCKWISE
                else -> null
            }
            if (code != null) {
                Core.rotate(left, left, code)
                Core.rotate(right, right, code)
            }
        }

        // Force both images to the exact same dimensions
        val commonW = minOf(left.cols(), right.cols())
        val commonH = minOf(left.rows(), right.rows())
        if (left.cols() != commonW || left.rows() != commonH) {
            val resized = Mat(); Imgproc.resize(left, resized, Size(commonW.toDouble(), commonH.toDouble()))
            left.release(); left = resized
        }
        if (right.cols() != commonW || right.rows() != commonH) {
            val resized = Mat(); Imgproc.resize(right, resized, Size(commonW.toDouble(), commonH.toDouble()))
            right.release(); right = resized
        }
        val leftResized = left; val rightResized = right

        // ── Aligned versions (only for depth map) ─────────────────────────
        var rectLeft: Mat = leftResized
        var rectRight: Mat = rightResized
        var usedHomography = false

        try {
            val orb = ORB.create(2000)
            val kpL = MatOfKeyPoint(); val kpR = MatOfKeyPoint()
            val descL = Mat(); val descR = Mat()
            orb.detectAndCompute(leftResized, Mat(), kpL, descL)
            orb.detectAndCompute(rightResized, Mat(), kpR, descR)

            if (descL.rows() >= 8 && descR.rows() >= 8) {
                val matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING)
                val matches = MatOfDMatch()
                matcher.match(descL, descR, matches)
                val matchList = matches.toList()
                val goodMatches = matchList.sortedBy { it.distance }.take(300.coerceAtMost(matchList.size))

                val srcPts = MatOfPoint2f()
                val dstPts = MatOfPoint2f()
                val srcArr = goodMatches.map { kpL.toList()[it.queryIdx].pt }.toTypedArray()
                val dstArr = goodMatches.map { kpR.toList()[it.trainIdx].pt }.toTypedArray()
                srcPts.fromArray(*srcArr); dstPts.fromArray(*dstArr)

                val inlierMask = Mat()
                val H = Calib3d.findHomography(srcPts, dstPts, Calib3d.RANSAC, 5.0, inlierMask)
                inlierMask.release()

                if (H != null) {
                    val Hinv = H.inv()
                    rectLeft = Mat(); leftResized.copyTo(rectLeft)
                    rectRight = Mat()
                    Imgproc.warpPerspective(rightResized, rectRight, Hinv, Size(commonW.toDouble(), commonH.toDouble()))
                    H.release(); Hinv.release()
                    usedHomography = true
                }
                kpL.release(); kpR.release(); descL.release(); descR.release(); matches.release()
            } else {
                kpL.release(); kpR.release(); descL.release(); descR.release()
            }
        } catch (e: Exception) {
            Log.e("Spatial", "Feature matching failed", e)
        }

        val fh = leftResized.rows(); val fw = leftResized.cols()

        // ── Depth map (uses aligned images) ──────────────────────────────
        val depthColor = if (usedHomography) {
            try {
                val grayL = Mat(); Imgproc.cvtColor(rectLeft, grayL, Imgproc.COLOR_RGB2GRAY)
                val grayR = Mat(); Imgproc.cvtColor(rectRight, grayR, Imgproc.COLOR_RGB2GRAY)
                val disparity = Mat()
                val sgbm = org.opencv.calib3d.StereoSGBM.create(0, 64, 5)
                sgbm.setMinDisparity(0); sgbm.setNumDisparities(64); sgbm.setBlockSize(5)
                sgbm.setP1(8 * 3 * 5 * 5); sgbm.setP2(32 * 3 * 5 * 5)
                sgbm.setUniquenessRatio(5); sgbm.setSpeckleWindowSize(200); sgbm.setSpeckleRange(32)
                sgbm.setMode(org.opencv.calib3d.StereoSGBM.MODE_SGBM)
                sgbm.compute(grayL, grayR, disparity)
                val dispNorm = Mat()
                Core.normalize(disparity, dispNorm, 0.0, 255.0, Core.NORM_MINMAX, CvType.CV_8U)
                val dc = Mat()
                Imgproc.applyColorMap(dispNorm, dc, Imgproc.COLORMAP_JET)
                grayL.release(); grayR.release(); disparity.release(); dispNorm.release()
                dc
            } catch (e: Exception) {
                Log.e("Spatial", "Disparity computation failed", e)
                null
            }
        } else null

        // ── Side-by-side composite (uses ORIGINAL unaligned images) ──────
        val sbs = Mat(Size(fw.toDouble() * 2, fh.toDouble()), leftResized.type())
        val leftRoi = sbs.submat(0, fh, 0, fw); leftResized.copyTo(leftRoi)
        val rightRoi = sbs.submat(0, fh, fw, fw * 2); rightResized.copyTo(rightRoi)

        // ── Red-cyan anaglyph (uses ORIGINAL unaligned images) ───────────
        val anaglyph = try {
            val chL = mutableListOf<Mat>(); Core.split(leftResized, chL)
            val chR = mutableListOf<Mat>(); Core.split(rightResized, chR)
            val a = Mat()
            Core.merge(listOf(chL[0], chR[1], chR[2]), a)
            chL.forEach { it.release() }; chR.forEach { it.release() }
            a
        } catch (e: Exception) {
            Log.e("Spatial", "Anaglyph creation failed", e)
            null
        }

        fun matToBitmap(mat: Mat): Bitmap {
            val bmp = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(mat, bmp)
            return bmp
        }

        val timestamp = System.currentTimeMillis()
        val cacheDir = context.cacheDir

        val sbsFile = File(cacheDir, "spatial_sbs_$timestamp.jpg")
        FileOutputStream(sbsFile).use { matToBitmap(sbs).compress(Bitmap.CompressFormat.JPEG, 90, it) }

        val depthFile = if (depthColor != null) {
            File(cacheDir, "spatial_depth_$timestamp.jpg").also { f ->
                FileOutputStream(f).use { matToBitmap(depthColor).compress(Bitmap.CompressFormat.JPEG, 90, it) }
            }
        } else null

        val anaglyphFile = if (anaglyph != null) {
            File(cacheDir, "spatial_anaglyph_$timestamp.jpg").also { f ->
                FileOutputStream(f).use { matToBitmap(anaglyph).compress(Bitmap.CompressFormat.JPEG, 90, it) }
            }
        } else null

        if (!isPreview) {
            fun saveToGallery(bmp: Bitmap, name: String) {
                try {
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        val values = ContentValues().apply {
                            put(MediaStore.Images.Media.DISPLAY_NAME, name)
                            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                            put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Prescent")
                            put(MediaStore.Images.Media.IS_PENDING, 1)
                        }
                        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                        if (uri != null) {
                            context.contentResolver.openOutputStream(uri)?.use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                            values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                            context.contentResolver.update(uri, values, null, null)
                        }
                    } else {
                        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "Prescent").also { it.mkdirs() }
                        FileOutputStream(File(dir, name)).use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
                        android.media.MediaScannerConnection.scanFile(context, arrayOf(File(dir, name).absolutePath), null, null)
                    }
                } catch (e: Exception) { Log.e("Spatial", "save $name failed", e) }
            }
            val sbsBmp = matToBitmap(sbs)
            saveToGallery(sbsBmp, "spatial_${timestamp}.jpg")
            sbsBmp.recycle()
        }

        sbs.release(); anaglyph?.release(); depthColor?.release()
        if (usedHomography) { rectLeft.release(); rectRight.release() }
        leftResized.release(); rightResized.release()

        SpatialResult(
            sideBySidePath = sbsFile.absolutePath,
            depthMapPath = depthFile?.absolutePath,
            anaglyphPath = anaglyphFile?.absolutePath
        )
    } catch (e: Exception) {
        Log.e("Spatial", "Processing failed", e)
        null
    }
}
