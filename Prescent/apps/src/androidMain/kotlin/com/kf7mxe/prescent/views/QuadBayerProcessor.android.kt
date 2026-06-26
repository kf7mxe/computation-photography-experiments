package com.kf7mxe.prescent.views

import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import com.lightningkite.kiteui.views.AndroidAppContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

actual suspend fun processQuadBayer(
    rawFiles: List<String>,
    options: QuadBayerOptions,
    maxPreviewSize: Int
): String? = withContext(Dispatchers.IO) {
    Log.d("QuadBayer", "Processing ${rawFiles.size} raw frames, algo=${options.algorithm}")

    if (rawFiles.isEmpty()) return@withContext null

    try {
        val frames = rawFiles.map { QuadBayerEngine.readRawFile(it) }
        val refFrame = frames.first()
        val fullW = refFrame.width
        val fullH = refFrame.height
        val outW = fullW / 2
        val outH = fullH / 2

        // Step 1: load and normalize all frames to [0,1] float Bayer
        val normalized = mutableListOf<Mat>()
        for (frame in frames) {
            val raw16 = Mat(fullH, fullW, CvType.CV_16UC1)
            raw16.put(0, 0, frame.pixels)
            val f32 = Mat(fullH, fullW, CvType.CV_32FC1)
            raw16.convertTo(f32, CvType.CV_32FC1)
            raw16.release()
            val avgBlack = (frame.blackLevel[0] + frame.blackLevel[1]
                + frame.blackLevel[2] + frame.blackLevel[3]) / 4.0
            Core.subtract(f32, Scalar.all(avgBlack), f32)
            Core.multiply(f32, Scalar.all(1.0 / frame.whiteLevel), f32)
            Core.max(f32, Scalar.all(0.0), f32)
            Core.min(f32, Scalar.all(1.0), f32)
            normalized.add(f32)
        }

        // Step 2: merge multiple frames (average)
        val merged: Mat
        if (normalized.size > 1) {
            merged = Mat(normalized[0].size(), CvType.CV_32FC1, Scalar.all(0.0))
            for (m in normalized) Core.add(merged, m, merged)
            Core.divide(merged, Scalar.all(normalized.size.toDouble()), merged)
            normalized.forEach { if (it != merged) it.release() }
        } else {
            merged = normalized[0]
        }

        // Step 2b: optionally save merged RAW as DNG
        val dngPath: String? = if (options.saveDng) {
            try {
                val rawDir = File(rawFiles.first()).parentFile
                val dngFile = File(rawDir, "merged.dng")
                // Convert merged float back to uint16 and create a temp RawFrame
                val merged16 = Mat(fullH, fullW, CvType.CV_16UC1)
                merged.convertTo(merged16, CvType.CV_16UC1, 65535.0)
                val pixelArray = ShortArray(fullW * fullH)
                merged16.get(0, 0, pixelArray)
                merged16.release()
                val rawFrame = QuadBayerEngine.RawFrame(
                    width = fullW, height = fullH,
                    bitDepth = refFrame.bitDepth,
                    bayerPattern = refFrame.bayerPattern,
                    blackLevel = refFrame.blackLevel,
                    whiteLevel = refFrame.whiteLevel,
                    pixels = pixelArray
                )
                saveRawAsDng(rawFrame, dngFile.absolutePath)
                Log.d("QuadBayer", "DNG saved: ${dngFile.absolutePath}")
                dngFile.absolutePath
            } catch (e: Exception) {
                Log.e("QuadBayer", "DNG save failed", e)
                null
            }
        } else null

        val bayerCode = bayerPatternToCode(refFrame.bayerPattern)

        // Step 3: full-resolution demosaic first, then downscale.
        // Quad Bayer has all 4 colors in each 2×2 cluster — subsampling before
        // demosaic would pick only red pixels (all-Red Bayer → B&W output).
        val bayer16Full = Mat(fullH, fullW, CvType.CV_16UC1)
        merged.convertTo(bayer16Full, CvType.CV_16UC1, 65535.0)
        merged.release()

        val rgb16Full = Mat(fullH, fullW, CvType.CV_16UC3)
        Imgproc.cvtColor(bayer16Full, rgb16Full, bayerCode)
        bayer16Full.release()

        val rgb32f = Mat(outH, outW, CvType.CV_32FC3)
        val rgb16Half = Mat()
        Imgproc.resize(rgb16Full, rgb16Half, Size(outW.toDouble(), outH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        rgb16Full.release()
        rgb16Half.convertTo(rgb32f, CvType.CV_32FC3, 1.0 / 65535.0)
        rgb16Half.release()

        // Step 4: algorithm-specific post-processing on RGB
        when (options.algorithm) {
            QuadBayerAlgorithm.BIN_TO_BAYER -> {
                val kSize = ((outW.coerceAtMost(outH) * 0.02).toInt() or 1).coerceAtLeast(3)
                val blurred = Mat()
                Imgproc.GaussianBlur(rgb32f, blurred, Size(kSize.toDouble(), kSize.toDouble()), 0.0)
                Core.addWeighted(rgb32f, 0.7, blurred, 0.3, 0.0, rgb32f)
                blurred.release()
            }
            QuadBayerAlgorithm.FULL_REMOSAIC -> { /* full detail — no extra processing */ }
            QuadBayerAlgorithm.EDGE_GUIDED -> {
                val lum = Mat()
                Imgproc.cvtColor(rgb32f, lum, Imgproc.COLOR_BGR2GRAY)
                val blurred = Mat()
                Imgproc.GaussianBlur(lum, blurred, Size(5.0, 5.0), 0.0)
                val detail = Mat()
                Core.subtract(lum, blurred, detail)
                val channels = mutableListOf<Mat>()
                Core.split(rgb32f, channels)
                for (ch in channels) Core.addWeighted(ch, 1.0, detail, 0.15, 0.0, ch)
                Core.merge(channels, rgb32f)
                lum.release(); blurred.release(); detail.release()
                channels.forEach { it.release() }
            }
        }

        autoWhiteBalance(rgb32f)
        toneMapReinhard(rgb32f)

        if (maxPreviewSize > 0) {
            val scale = maxPreviewSize.toDouble() / maxOf(outW, outH).coerceAtLeast(1)
            if (scale < 1.0) {
                val scaled = Mat()
                Imgproc.resize(rgb32f, scaled, Size((outW * scale).toDouble(), (outH * scale).toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
                rgb32f.release()
                return@withContext saveToJpeg(scaled, "quadbayer")
            }
        }

        return@withContext saveToJpeg(rgb32f, "quadbayer")
    } catch (e: Exception) {
        Log.e("QuadBayer", "Processing failed", e)
        return@withContext null
    }
}

private fun bayerPatternToCode(pattern: Int): Int {
    return when (pattern) {
        0 -> Imgproc.COLOR_BayerRG2BGR_EA
        1 -> Imgproc.COLOR_BayerGR2BGR_EA
        2 -> Imgproc.COLOR_BayerGB2BGR_EA
        3 -> Imgproc.COLOR_BayerBG2BGR_EA
        else -> Imgproc.COLOR_BayerRG2BGR_EA
    }
}

private fun autoWhiteBalance(bgr: Mat) {
    val channels = mutableListOf<Mat>()
    Core.split(bgr, channels)
    val bAvg = MatOfDouble()
    val gAvg = MatOfDouble()
    val rAvg = MatOfDouble()
    Core.meanStdDev(channels[0], bAvg, MatOfDouble())
    Core.meanStdDev(channels[1], gAvg, MatOfDouble())
    Core.meanStdDev(channels[2], rAvg, MatOfDouble())

    val meanB = bAvg.let { it.get(0, 0)[0] }
    val meanG = gAvg.let { it.get(0, 0)[0] }
    val meanR = rAvg.let { it.get(0, 0)[0] }

    if (meanG > 0.01) {
        if (meanR > 0.01) Core.multiply(channels[2], Scalar.all(meanG / meanR), channels[2])
        if (meanB > 0.01) Core.multiply(channels[0], Scalar.all(meanG / meanB), channels[0])
    }

    Core.merge(channels, bgr)
    channels.forEach { it.release() }
}

private fun toneMapReinhard(bgr: Mat) {
    val channels = mutableListOf<Mat>()
    Core.split(bgr, channels)
    val r = channels[2]; val g = channels[1]; val b = channels[0]

    // Luminance: 0.299*R + 0.587*G + 0.114*B (all OpenCV ops, no per-pixel loops)
    val luminance = Mat()
    Core.addWeighted(r, 0.299, g, 0.587, 0.0, luminance)
    Core.addWeighted(luminance, 1.0, b, 0.114, 0.0, luminance)

    val meanLum = Core.mean(luminance).`val`[0]
    val key = if (meanLum > 0.001) 0.18 / meanLum else 0.18
    val gamma = 1.0 / 2.2

    // Scaled luminance: Ls = L * key
    val ls = Mat()
    Core.multiply(luminance, Scalar.all(key), ls)

    // Reinhard tone mapping: Lt = Ls * (1 + Ls) / (1 + Ls)
    // Simplified: Lmapped = Ls / (1 + Ls) (typical Reinhard with Lwhite=inf)
    // For Lwhite=1: Lmapped = Ls * (1 + Ls) / (1 + Ls) which simplifies when Ls is small
    val onePlusLs = Mat()
    Core.add(ls, Scalar.all(1.0), onePlusLs)
    val lMapped = Mat()
    Core.divide(ls, onePlusLs, lMapped)
    Core.patchNaNs(lMapped, 0.0)

    // Scale = Lmapped / Ls (per-pixel, avoiding div-by-zero)
    val scale = Mat()
    Core.divide(lMapped, ls, scale)
    Core.patchNaNs(scale, 0.0)

    for (ch in listOf(r, g, b)) {
        Core.multiply(ch, Scalar.all(key), ch)
        Core.multiply(ch, scale, ch)
        Core.pow(ch, gamma, ch)
        Core.min(ch, Scalar.all(1.0), ch)
        Core.max(ch, Scalar.all(0.0), ch)
    }

    // Clean intermediates
    luminance.release(); ls.release(); onePlusLs.release(); lMapped.release(); scale.release()

    Core.merge(channels, bgr)
    channels.forEach { it.release() }

    // Contrast stretch via mean±2.5σ (avoids expensive sort)
    val gray = Mat()
    Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)
    val mean = MatOfDouble()
    val stddev = MatOfDouble()
    Core.meanStdDev(gray, mean, stddev)
    val m = mean.get(0, 0)[0]
    val s = stddev.get(0, 0)[0]
    val lowClip = (m - 2.5 * s).coerceAtLeast(0.0)
    val highClip = (m + 2.5 * s).coerceAtMost(1.0)
    val range = (highClip - lowClip).coerceAtLeast(0.001)
    gray.release()

    Core.subtract(bgr, Scalar.all(lowClip), bgr)
    Core.divide(bgr, Scalar.all(range), bgr)
    Core.min(bgr, Scalar.all(1.0), bgr)
    Core.max(bgr, Scalar.all(0.0), bgr)
}

fun saveBayerToGallery(bitmap: Bitmap): String? {
    val context = AndroidAppContext.applicationCtx
    val filename = "prescent_quadbayer_${System.currentTimeMillis()}.jpg"
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
        Log.e("QuadBayer", "saveToGallery failed", e)
        val cacheFile = File(context.cacheDir, filename)
        FileOutputStream(cacheFile).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        cacheFile.absolutePath
    }
}

fun saveDngToGallery(dngFile: java.io.File) {
    val context = AndroidAppContext.applicationCtx
    val filename = "prescent_quadbayer_${System.currentTimeMillis()}.dng"
    try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val values = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/x-adobe-dng")
                put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Prescent")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
            val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            if (uri != null) {
                context.contentResolver.openOutputStream(uri)?.use { out ->
                    dngFile.inputStream().use { inp -> inp.copyTo(out) }
                }
                values.clear(); values.put(MediaStore.Images.Media.IS_PENDING, 0)
                context.contentResolver.update(uri, values, null, null)
                Log.d("QuadBayer", "DNG saved to gallery: $filename")
            }
        } else {
            val picturesDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "Prescent")
                .also { it.mkdirs() }
            val dest = File(picturesDir, filename)
            dngFile.inputStream().use { inp -> dest.outputStream().use { out -> inp.copyTo(out) } }
            android.media.MediaScannerConnection.scanFile(context, arrayOf(dest.absolutePath), null, null)
            Log.d("QuadBayer", "DNG saved: ${dest.absolutePath}")
        }
    } catch (e: Exception) {
        Log.e("QuadBayer", "DNG gallery save failed", e)
    }
}

private fun saveToJpeg(rgb32f: Mat, subDir: String): String {
    val context = AndroidAppContext.applicationCtx
    val dir = File(context.filesDir, "$subDir/${System.currentTimeMillis()}").also { it.mkdirs() }
    val outputFile = File(dir, "result.jpg")

    val bgr8u = Mat(rgb32f.size(), CvType.CV_8UC3)
    rgb32f.convertTo(bgr8u, CvType.CV_8UC3, 255.0)
    rgb32f.release()

    val bitmap = Bitmap.createBitmap(bgr8u.cols(), bgr8u.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(bgr8u, bitmap)
    bgr8u.release()

    FileOutputStream(outputFile).use { out ->
        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
    }
    bitmap.recycle()

    Log.d("QuadBayer", "Saved: ${outputFile.absolutePath}")
    return outputFile.absolutePath
}

fun saveRawAsDng(frame: QuadBayerEngine.RawFrame, outputPath: String) {
    val w = frame.width; val h = frame.height
    val pixels = frame.pixels
    val pixelCount = w * h
    val rawDataSize = pixelCount * 2

    val CFA_RGGB = byteArrayOf(0, 1, 1, 2)
    val CFA_GRBG = byteArrayOf(1, 0, 2, 1)
    val CFA_GBRG = byteArrayOf(1, 2, 0, 1)
    val CFA_BGGR = byteArrayOf(2, 1, 1, 0)
    val cfaPattern = when (frame.bayerPattern) {
        0 -> CFA_RGGB; 1 -> CFA_GRBG; 2 -> CFA_GBRG; 3 -> CFA_BGGR; else -> CFA_RGGB
    }

    val buf = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN)

    // Tag type codes
    val T_BYTE = 1; val T_SHORT = 3; val T_LONG = 4

    // Write TIFF header
    buf.put(0x49); buf.put(0x49) // Little-endian
    buf.putShort(42) // TIFF magic
    val ifdOffsetPos = buf.position()
    buf.putInt(0) // placeholder for IFD offset (fixup after entries)

    // Collect non-inline tag data in a separate buffer
    val tagData = ByteArrayOutputStream()

    fun addInline(tag: Int, type: Int, count: Int, data: ByteArray) {
        buf.putShort(tag.toShort()); buf.putShort(type.toShort()); buf.putInt(count)
        buf.put(data)
        while (buf.position() % 4 != 0) buf.put(0) // pad to 4 bytes
    }

    fun addPtr(tag: Int, type: Int, count: Int, data: ByteArray) {
        buf.putShort(tag.toShort()); buf.putShort(type.toShort()); buf.putInt(count)
        buf.putInt(0) // placeholder for offset; fix up later
    }

    // Save IFD start position
    val ifdStart = buf.position()
    val entryCountPos = buf.position()
    buf.putShort(0) // placeholder entry count
    var numEntries = 0

    // Track entries that need offset fixup
    data class PtrEntry(val valueOffsetPos: Int, val data: ByteArray)
    val ptrEntries = mutableListOf<PtrEntry>()

    fun entry(tag: Int, type: Int, count: Int, data: ByteArray) {
        numEntries++
        if (data.size <= 4) {
            addInline(tag, type, count, data)
        } else {
            val pos = buf.position()
            addPtr(tag, type, count, data)
            ptrEntries.add(PtrEntry(pos, data))
        }
    }

    // Write IFD entries
    entry(0x00FE, T_LONG, 1, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(0).array()) // NewSubFileType=0
    entry(0x0100, T_LONG, 1, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(w).array()) // ImageWidth
    entry(0x0101, T_LONG, 1, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(h).array()) // ImageLength
    entry(0x0102, T_SHORT, 1, ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(16).array()) // BitsPerSample
    entry(0x0103, T_SHORT, 1, ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(1).array()) // Compression=uncompressed
    entry(0x0106, T_SHORT, 1, ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(32803.toShort()).array()) // CFA
    entry(0x0111, T_LONG, 1, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(0).array()) // StripOffsets (fixup later)
    entry(0x0115, T_SHORT, 1, ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(1).array()) // SamplesPerPixel
    entry(0x0116, T_LONG, 1, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(h).array()) // RowsPerStrip
    entry(0x0117, T_LONG, 1, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(rawDataSize).array()) // StripByteCounts
    entry(0x011C, T_SHORT, 1, ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(1).array()) // PlanarConfig=chunky
    // BlackLevel: 4 SHORTs
    val blBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
    for (i in 0..3) blBuf.putShort(frame.blackLevel[i].toShort())
    entry(0xC6D2, T_SHORT, 4, blBuf.array()) // BlackLevel
    entry(0xC6D3, T_LONG, 1, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(frame.whiteLevel).array()) // WhiteLevel
    entry(0xC6D5, T_SHORT, 2, ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putShort(2).putShort(2).array()) // CFA repeat
    entry(0xC6D6, T_BYTE, 4, cfaPattern) // CFAPattern
    entry(0xC6D7, T_BYTE, 3, byteArrayOf(0, 1, 2)) // CFA plane color

    // Fixup IFD offset and entry count
    val ifdStartActual = 8 // TIFF header is 8 bytes
    buf.putInt(ifdOffsetPos, ifdStartActual)
    buf.putShort(entryCountPos, numEntries.toShort())

    // Next IFD offset = 0 (no more IFDs)
    buf.putInt(0)

    // Write non-inline tag data and fixup offsets
    val tagDataStart = buf.position()
    for (pe in ptrEntries) {
        buf.putInt(pe.valueOffsetPos, tagDataStart + tagData.size())
        tagData.write(pe.data)
    }

    // Pad tag data to even boundary
    if (tagData.size() % 2 != 0) tagData.write(0)

    // Write tag data
    buf.put(tagData.toByteArray())

    // Fixup StripOffsets
    val rawDataStart = buf.position()
    // The strip offset was entry 6 (index 6). Find it in ptrEntries
    // Actually let me just scan the buffer for the StripOffsets entry
    // Simplier: we know StripOffsets is entry index 6 (0-indexed in the entry list)
    // But we need to find it via the tag ID
    // Let me just set it using the position

    // Fixup StripOffsets value
    val stripOffsetsPos = ifdStart + 2 + 6 * 12 + 8
    buf.putInt(stripOffsetsPos, buf.position())

    // Write header + tags to file
    val fos = FileOutputStream(outputPath)
    fos.write(buf.array(), 0, buf.position())

    // Write raw pixel data separately (doesn't need to fit in buf)
    val pixelBuf = ByteBuffer.allocate(rawDataSize).order(ByteOrder.LITTLE_ENDIAN)
    pixelBuf.asShortBuffer().put(pixels)
    fos.write(pixelBuf.array())
    fos.close()
}

fun matToBitmap(rgb32f: Mat): Bitmap {
    val bgr8u = Mat(rgb32f.size(), CvType.CV_8UC3)
    rgb32f.convertTo(bgr8u, CvType.CV_8UC3, 255.0)
    val bitmap = Bitmap.createBitmap(bgr8u.cols(), bgr8u.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(bgr8u, bitmap)
    bgr8u.release()
    return bitmap
}
