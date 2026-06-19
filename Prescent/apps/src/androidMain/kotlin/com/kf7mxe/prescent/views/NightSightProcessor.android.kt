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

actual suspend fun processNightSight(
    images: List<String>,
    starTrail: Boolean,
    darkFramePath: String?,
    brightnessBoost: Float,
    maxPreviewSize: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val isPreview = maxPreviewSize > 0
    Log.d("NightSight", "Processing ${images.size} frames, starTrail=$starTrail, darkFrame=$darkFramePath, boost=$brightnessBoost")

    try {
        // 1. Load frames
        val frames = mutableListOf<Mat>()
        for (path in images) {
            val opts = BitmapFactory.Options()
            if (isPreview) {
                opts.inJustDecodeBounds = true
                val bm = if (path.startsWith("/")) BitmapFactory.decodeFile(path, opts)
                else context.contentResolver.openInputStream(Uri.parse(path))?.use { BitmapFactory.decodeStream(it, null, opts) }
                val w = opts.outWidth; val h = opts.outHeight
                var s = 1; while (w / s > maxPreviewSize || h / s > maxPreviewSize) s *= 2
                opts.inSampleSize = s; opts.inJustDecodeBounds = false
            }
            val bitmap = if (path.startsWith("/")) BitmapFactory.decodeFile(path, opts)
            else context.contentResolver.openInputStream(Uri.parse(path))?.use { BitmapFactory.decodeStream(it, null, opts) }
                ?: continue
            val rgba = Mat(); Utils.bitmapToMat(bitmap, rgba); bitmap.recycle()
            val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB); rgba.release()
            frames.add(rgb)
        }
        if (frames.size < 2) return@withContext null

        // 2. Load dark frame if provided
        var darkFrame: Mat? = null
        if (darkFramePath != null) {
            try {
                val opts = BitmapFactory.Options()
                val bm = if (darkFramePath.startsWith("/")) BitmapFactory.decodeFile(darkFramePath, opts)
                else context.contentResolver.openInputStream(Uri.parse(darkFramePath))?.use { BitmapFactory.decodeStream(it, null, opts) }
                if (bm != null) {
                    val rgba = Mat(); Utils.bitmapToMat(bm, rgba); bm.recycle()
                    darkFrame = Mat(); Imgproc.cvtColor(rgba, darkFrame, Imgproc.COLOR_RGBA2RGB); rgba.release()
                    Log.d("NightSight", "Dark frame loaded")
                }
            } catch (e: Exception) {
                Log.e("NightSight", "Dark frame load failed", e)
            }
        }

        // 3. Align frames using MTB
        val aligner = Photo.createAlignMTB()
        val aligned = mutableListOf<Mat>().also { out -> frames.forEach { _ -> out.add(Mat()) } }
        try { aligner.process(frames, aligned) } catch (e: Exception) {
            Log.e("NightSight", "Alignment failed, using raw frames")
            frames.forEachIndexed { i, m -> aligned[i] = m.clone() }
        }
        val validFrames = aligned.filter { !it.empty() }

        // 4. Dark frame subtract
        val subtracted = if (darkFrame != null) {
            validFrames.map { mat ->
                val sub = Mat()
                Core.subtract(mat, darkFrame, sub)
                Core.max(sub, Scalar(0.0), sub)
                sub
            }
        } else validFrames

        // 5. Stack
        val stacked = if (starTrail && subtracted.size >= 2) {
            // Star trail: max blend — keeps bright spots across all frames
            val result = subtracted[0].clone()
            for (i in 1 until subtracted.size) {
                Core.max(result, subtracted[i], result)
            }
            result
        } else {
            // Temporal averaging: mean of all frames → reduces noise
            val result = Mat.zeros(subtracted[0].size(), CvType.CV_32F)
            val floatMats = subtracted.map { mat ->
                val f = Mat(); mat.convertTo(f, CvType.CV_32F, 1.0 / 255.0); f
            }
            for (fm in floatMats) Core.add(result, fm, result)
            Core.multiply(result, Scalar(1.0 / subtracted.size), result)
            val result8 = Mat(); result.convertTo(result8, CvType.CV_8UC3, 255.0)
            floatMats.forEach { it.release() }; result.release()
            result8
        }

        // 6. Brightness boost
        stacked.convertTo(stacked, CvType.CV_32F, 1.0 / 255.0)
        Core.multiply(stacked, Scalar(brightnessBoost.toDouble()), stacked)
        Core.min(stacked, Scalar(1.0), stacked)
        val brightened = Mat(); stacked.convertTo(brightened, CvType.CV_8UC3, 255.0)
        stacked.release()

        // 7. CLAHE local contrast enhancement on luminance channel
        val lab = Mat()
        Imgproc.cvtColor(brightened, lab, Imgproc.COLOR_RGB2Lab)
        val labCh = mutableListOf<Mat>(); Core.split(lab, labCh)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(labCh[0], labCh[0])
        Core.merge(labCh, lab)
        val enhanced = Mat(); Imgproc.cvtColor(lab, enhanced, Imgproc.COLOR_Lab2RGB)
        brightened.release(); lab.release(); labCh.forEach { it.release() }

        // 8. Save
        val bitmap = Bitmap.createBitmap(enhanced.cols(), enhanced.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(enhanced, bitmap)
        enhanced.release()
        frames.forEach { it.release() }
        aligned.forEach { it.release() }
        subtracted.forEach { it.release() }
        darkFrame?.release()

        val suffix = if (starTrail) "_trail" else ""
        val filename = "nightsight${suffix}_${System.currentTimeMillis()}.jpg"

        val savedPath = if (isPreview) {
            val previewFile = File(context.cacheDir, filename)
            FileOutputStream(previewFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
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
            } catch (e: Exception) { Log.e("NightSight", "save failed", e) }
            // Also save to cache for display
            val cacheFile = File(context.cacheDir, filename)
            FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            bitmap.recycle()
            cacheFile.absolutePath
        }
        Log.d("NightSight", "Complete: $savedPath")
        savedPath
    } catch (e: Exception) {
        Log.e("NightSight", "Processing failed", e)
        null
    }
}
