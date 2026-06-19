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

actual suspend fun processFocusStack(
    images: List<String>,
    maxPreviewSize: Int
): String? = withContext(Dispatchers.IO) {
    val context = AndroidAppContext.applicationCtx
    val isPreview = maxPreviewSize > 0
    Log.d("FocusStack", "Processing ${images.size} frames")

    try {
        // 1. Load frames
        val frames = mutableListOf<Mat>()
        for (path in images) {
            val opts = BitmapFactory.Options()
            if (isPreview) {
                opts.inJustDecodeBounds = true
                BitmapFactory.decodeFile(path, opts)
                val w = opts.outWidth; val h = opts.outHeight
                var s = 1; while (w / s > maxPreviewSize || h / s > maxPreviewSize) s *= 2
                opts.inSampleSize = s; opts.inJustDecodeBounds = false
            }
            val bitmap = BitmapFactory.decodeFile(path, opts) ?: continue
            val rgba = Mat(); Utils.bitmapToMat(bitmap, rgba); bitmap.recycle()
            val rgb = Mat(); Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB); rgba.release()
            frames.add(rgb)
        }
        if (frames.size < 2) return@withContext null

        // 2. Align frames using MTB
        val aligner = Photo.createAlignMTB()
        val aligned = mutableListOf<Mat>().also { out -> frames.forEach { _ -> out.add(Mat()) } }
        try { aligner.process(frames, aligned) } catch (e: Exception) {
            Log.e("FocusStack", "Alignment failed, using raw frames")
            frames.forEachIndexed { i, m -> aligned[i] = m.clone() }
        }
        val validFrames = aligned.filter { !it.empty() }

        // 3. Compute Laplacian variance per frame for sharpness metric
        val grayFrames = validFrames.map { mat ->
            val g = Mat(); Imgproc.cvtColor(mat, g, Imgproc.COLOR_RGB2GRAY); g
        }

        val laplacianMaps = grayFrames.map { gray ->
            val lap = Mat(); Imgproc.Laplacian(gray, lap, CvType.CV_32F); lap
        }

        // 4. Build focus weight maps: local variance of Laplacian over 15x15 neighborhoods
        val weightMaps = laplacianMaps.map { lap ->
            val mean = Mat(); val stddev = Mat()
            Imgproc.GaussianBlur(lap, mean, Size(15.0, 15.0), 0.0)
            val meanSq = Mat(); Core.pow(mean, 2.0, meanSq)
            val lapSq = Mat(); Core.pow(lap, 2.0, lapSq)
            val varMap = Mat(); Imgproc.GaussianBlur(lapSq, varMap, Size(15.0, 15.0), 0.0)
            Core.subtract(varMap, meanSq, varMap)
            Core.max(varMap, Scalar(1e-6), varMap)
            mean.release(); meanSq.release(); lapSq.release()
            varMap
        }

        // 5. Normalize weights across frames (per-pixel sum to 1)
        val weightSum = Mat.zeros(weightMaps[0].size(), CvType.CV_32F)
        for (wm in weightMaps) Core.add(weightSum, wm, weightSum)
        val normalizedWeights = weightMaps.map { wm ->
            val nw = Mat(); Core.divide(wm, weightSum, nw); nw
        }

        // 6. Weighted blend: accumulate (frame * weight) per color channel
        val h = validFrames[0].rows()
        val w = validFrames[0].cols()
        val resultFloat = Mat.zeros(h, w, CvType.CV_32FC3)
        for (i in validFrames.indices) {
            val frameFloat = Mat(); validFrames[i].convertTo(frameFloat, CvType.CV_32F, 1.0 / 255.0)
            val ch = mutableListOf<Mat>(); Core.split(frameFloat, ch)
            for (c in 0 until 3) {
                Core.multiply(ch[c], normalizedWeights[i], ch[c])
            }
            Core.merge(ch, frameFloat)
            Core.add(resultFloat, frameFloat, resultFloat)
            ch.forEach { it.release() }; frameFloat.release()
        }
        Core.min(resultFloat, Scalar(1.0), resultFloat)
        val result8 = Mat(); resultFloat.convertTo(result8, CvType.CV_8UC3, 255.0)
        resultFloat.release()

        // 7. Save
        val bitmap = Bitmap.createBitmap(result8.cols(), result8.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(result8, bitmap)
        result8.release()
        frames.forEach { it.release() }
        aligned.forEach { it.release() }
        grayFrames.forEach { it.release() }
        laplacianMaps.forEach { it.release() }
        weightMaps.forEach { it.release() }
        normalizedWeights.forEach { it.release() }
        weightSum.release()

        val filename = "focusstack_${System.currentTimeMillis()}.jpg"

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
            } catch (e: Exception) { Log.e("FocusStack", "save failed", e) }
            val cacheFile = File(context.cacheDir, filename)
            FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
            bitmap.recycle()
            cacheFile.absolutePath
        }
        Log.d("FocusStack", "Complete: $savedPath")
        savedPath
    } catch (e: Exception) {
        Log.e("FocusStack", "Processing failed", e)
        null
    }
}
