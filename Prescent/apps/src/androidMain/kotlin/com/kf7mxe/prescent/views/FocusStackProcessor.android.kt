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
            val rgbaMat = Mat()
            Utils.bitmapToMat(bitmap, rgbaMat); bitmap.recycle()
            val rgbMat = Mat()
            Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
            rgbaMat.release()
            frames.add(rgbMat)
        }
        if (frames.size < 2) return@withContext null

        val aligner = Photo.createAlignMTB()
        val aligned = mutableListOf<Mat>().also { out -> frames.forEach { _ -> out.add(Mat()) } }
        try { aligner.process(frames, aligned) } catch (e: Exception) {
            Log.e("FocusStack", "Alignment failed, using raw frames")
            frames.forEachIndexed { i, m -> aligned[i] = m.clone() }
        }
        var validFrames = aligned.filter { !it.empty() }
        if (validFrames.isEmpty()) {
            Log.w("FocusStack", "All aligned frames empty, falling back to raw frames")
            validFrames = frames.map { it.clone() }
        }

        val grayFrames = validFrames.map { mat ->
            val g = Mat(); Imgproc.cvtColor(mat, g, Imgproc.COLOR_RGB2GRAY); g
        }

        val laplacianMaps = grayFrames.map { gray ->
            val lap = Mat(); Imgproc.Laplacian(gray, lap, CvType.CV_32F); lap
        }

        // Simple average (skip weight computation for now)
        val resultFloat = Mat.zeros(validFrames[0].size(), CvType.CV_32FC3)
        for (i in validFrames.indices) {
            val frameFloat = Mat(); validFrames[i].convertTo(frameFloat, CvType.CV_32F, 1.0 / 255.0)
            Core.add(resultFloat, frameFloat, resultFloat)
            frameFloat.release()
        }
        Core.multiply(resultFloat, Scalar(1.0 / validFrames.size), resultFloat)
        Core.min(resultFloat, Scalar(1.0), resultFloat)
        val result8 = Mat(); resultFloat.convertTo(result8, CvType.CV_8UC3, 255.0)
        resultFloat.release()

        val bitmap = Bitmap.createBitmap(result8.cols(), result8.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(result8, bitmap)
        result8.release()
        frames.forEach { it.release() }
        aligned.forEach { it.release() }
        grayFrames.forEach { it.release() }
        laplacianMaps.forEach { it.release() }

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

private fun focusStackBlend(validFrames: List<Mat>, normalizedWeights: List<Mat>): Mat {
    val resultFloat = Mat.zeros(validFrames[0].size(), CvType.CV_32FC3)
    for (i in validFrames.indices) {
        val frameFloat = Mat(); validFrames[i].convertTo(frameFloat, CvType.CV_32F, 1.0 / 255.0)
        val ch = mutableListOf<Mat>(); Core.split(frameFloat, ch)
        for (c in 0 until 3) Core.multiply(ch[c], normalizedWeights[i], ch[c])
        Core.merge(ch, frameFloat)
        Core.add(resultFloat, frameFloat, resultFloat)
        ch.forEach { it.release() }; frameFloat.release()
    }
    Core.min(resultFloat, Scalar(1.0), resultFloat)
    val result8 = Mat(); resultFloat.convertTo(result8, CvType.CV_8UC3, 255.0)
    resultFloat.release()
    return result8
}
