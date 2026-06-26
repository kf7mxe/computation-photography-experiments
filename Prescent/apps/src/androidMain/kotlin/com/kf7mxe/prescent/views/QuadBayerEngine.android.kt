package com.kf7mxe.prescent.views

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.core.Camera
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

data class RawSensorInfo(
    val supportsRaw: Boolean = false,
    val pixelArraySize: Size = Size(0, 0),
    val bayerPattern: Int = -1,
    val whiteLevel: Int = 4095,
    val blackLevel: IntArray = intArrayOf(64, 64, 64, 64),
    val bitDepth: Int = 10,
    val physicalSize: android.util.SizeF? = null
)

data class ManualExposure(val exposureTimeNs: Long, val iso: Int)

object QuadBayerEngine {
    private const val TAG = "QuadBayerEngine"
    private const val HEADER_MAGIC = 0x59414251
    private const val HEADER_SIZE = 40
    private const val TIMEOUT_SECONDS = 8L

    data class RawFrame(
        val width: Int,
        val height: Int,
        val bitDepth: Int,
        val bayerPattern: Int,
        val blackLevel: IntArray,
        val whiteLevel: Int,
        val pixels: ShortArray
    )

    fun detectRawCapability(cameraManager: CameraManager, cameraId: String): RawSensorInfo {
        return try {
            val chars = cameraManager.getCameraCharacteristics(cameraId)
            val capabilities = chars.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            val supportsRaw = capabilities?.contains(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW
            ) ?: false
            if (!supportsRaw) return RawSensorInfo()

            val pixelArraySize = chars.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
                ?: return RawSensorInfo()

            val whiteLevel = chars.get(CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL) ?: 4095
            val blackPattern = chars.get(CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN)
            val blackLevel = if (blackPattern != null) {
                intArrayOf(
                    blackPattern.getOffsetForIndex(0, 0),
                    blackPattern.getOffsetForIndex(1, 0),
                    blackPattern.getOffsetForIndex(0, 1),
                    blackPattern.getOffsetForIndex(1, 1)
                )
            } else intArrayOf(64, 64, 64, 64)

            val cfa = chars.get(CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT)
                ?: CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB

            val bitDepth = when {
                whiteLevel <= 255 -> 8
                whiteLevel <= 1023 -> 10
                whiteLevel <= 4095 -> 12
                whiteLevel <= 16383 -> 14
                else -> 16
            }
            val physicalSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

            RawSensorInfo(
                supportsRaw = true,
                pixelArraySize = pixelArraySize,
                bayerPattern = cfa,
                whiteLevel = whiteLevel,
                blackLevel = blackLevel,
                bitDepth = bitDepth,
                physicalSize = physicalSize
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to detect RAW capability for $cameraId", e)
            RawSensorInfo()
        }
    }

    fun getCameraId(camera: Camera): String? {
        return try {
            Camera2CameraInfo.from(camera.cameraInfo).cameraId
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get Camera2 camera ID", e)
            null
        }
    }

    fun getSensorInfo(context: Context, cameraId: String): SensorBounds {
        return try {
            val mgr = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val chars = mgr.getCameraCharacteristics(cameraId)
            val expRange = chars.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            val isoRange = chars.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            SensorBounds(
                exposureRange = expRange ?: Range(1000L, 1000000000L),
                isoRange = isoRange ?: Range(100, 3200)
            )
        } catch (_: Exception) {
            SensorBounds()
        }
    }

    fun measureAutoExposure(context: Context, cameraId: String, info: RawSensorInfo, bounds: SensorBounds): ManualExposure? {
        val manager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val thread = HandlerThread("quadbayer-meter").apply { start() }
        val handler = Handler(thread.looper)
        val discardListener = ImageReader.OnImageAvailableListener { r -> r.acquireNextImage()?.close() }
        try {
            val device = openCamera(manager, cameraId, handler) ?: return null
            val reader = ImageReader.newInstance(info.pixelArraySize.width, info.pixelArraySize.height,
                ImageFormat.RAW_SENSOR, 1)
            reader.setOnImageAvailableListener(discardListener, handler)
            val session = createSession(device, reader, handler) ?: run { reader.close(); device.close(); return null }

            var resultExp = 0L; var resultIso = 0
            val latch = CountDownLatch(1)
            val req = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
            req.addTarget(reader.surface)
            req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            session.capture(req.build(), object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, result: TotalCaptureResult) {
                    val exp = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
                    val iso = result.get(CaptureResult.SENSOR_SENSITIVITY)
                    if (exp != null && exp > 0L) resultExp = exp
                    if (iso != null && iso > 0) resultIso = iso
                    latch.countDown()
                }
                override fun onCaptureFailed(s: CameraCaptureSession, r: CaptureRequest, f: CaptureFailure) {
                    latch.countDown()
                }
            }, handler)
            latch.await(3, TimeUnit.SECONDS)
            Thread.sleep(100)

            session.close(); reader.close(); device.close()
            val exp = resultExp.coerceIn(bounds.exposureRange.lower, bounds.exposureRange.upper)
            val iso = resultIso.coerceIn(bounds.isoRange.lower, bounds.isoRange.upper)
            return ManualExposure(exp, iso)
        } catch (e: Exception) {
            Log.e(TAG, "Metering failed", e)
            return null
        } finally { thread.quitSafely() }
    }

    fun captureRawFrames(
        context: Context,
        cameraId: String,
        count: Int,
        outputDir: File,
        info: RawSensorInfo,
        manualExposure: ManualExposure? = null
    ): List<String> {
        val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraThread = HandlerThread("quadbayer-camera").apply { start() }
        val cameraHandler = Handler(cameraThread.looper)
        val ioExecutor = Executors.newSingleThreadExecutor()

        val paths = Collections.synchronizedList(mutableListOf<String>())
        val framesLatch = CountDownLatch(count)

        try {
            val w = info.pixelArraySize.width
            val h = info.pixelArraySize.height
            val device = openCamera(cameraManager, cameraId, cameraHandler) ?: return emptyList()
            val reader = ImageReader.newInstance(w, h, ImageFormat.RAW_SENSOR, count.coerceAtMost(3))

            reader.setOnImageAvailableListener({ rdr ->
                val image = rdr.acquireNextImage() ?: return@setOnImageAvailableListener
                val buffer = image.planes[0].buffer
                val shorts = ShortArray(image.width * image.height)
                buffer.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shorts)
                val idx = paths.size; val iw = image.width; val ih = image.height
                image.close()
                ioExecutor.execute {
                    val file = File(outputDir, "frame_$idx.raw")
                    FileOutputStream(file).use { out ->
                        val h = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
                        h.putInt(HEADER_MAGIC); h.putInt(iw); h.putInt(ih)
                        h.putInt(info.bitDepth); h.putInt(info.bayerPattern)
                        h.putInt(info.blackLevel[0]); h.putInt(info.blackLevel[1])
                        h.putInt(info.blackLevel[2]); h.putInt(info.blackLevel[3])
                        h.putInt(info.whiteLevel)
                        out.write(h.array())
                        val pb = ByteBuffer.allocate(shorts.size * 2).order(ByteOrder.LITTLE_ENDIAN)
                        pb.asShortBuffer().put(shorts); out.write(pb.array())
                    }
                    paths.add(file.absolutePath)
                    Log.d(TAG, "Saved $idx: ${file.absolutePath} ($iw x $ih)")
                }
                framesLatch.countDown()
            }, cameraHandler)

            val session = createSession(device, reader, cameraHandler) ?: run {
                reader.close(); device.close(); return emptyList()
            }

            for (i in 0 until count) {
                val req = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE)
                req.addTarget(reader.surface)
                req.set(CaptureRequest.CONTROL_CAPTURE_INTENT, CaptureRequest.CONTROL_CAPTURE_INTENT_STILL_CAPTURE)
                req.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY)

                if (manualExposure != null) {
                    req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                    req.set(CaptureRequest.SENSOR_EXPOSURE_TIME, manualExposure.exposureTimeNs)
                    req.set(CaptureRequest.SENSOR_SENSITIVITY, manualExposure.iso)
                    Log.d(TAG, "Frame $i: manual exp=${manualExposure.exposureTimeNs} iso=${manualExposure.iso}")
                } else {
                    req.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                    if (i > 0) req.set(CaptureRequest.CONTROL_AE_LOCK, true)
                }

                session.capture(req.build(), object : CameraCaptureSession.CaptureCallback() {
                    override fun onCaptureFailed(s: CameraCaptureSession, r: CaptureRequest, f: CaptureFailure) {
                        Log.e(TAG, "Frame $i failed: ${f.reason}")
                    }
                }, cameraHandler)
                if (i < count - 1) Thread.sleep(300)
            }

            val allDone = framesLatch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)
            if (!allDone) Log.w(TAG, "Only ${paths.size}/$count frames captured")

            ioExecutor.shutdown()
            ioExecutor.awaitTermination(5, TimeUnit.SECONDS)
            session.close()
            reader.close()
            device.close()
            return paths.toList()
        } catch (e: Exception) {
            Log.e(TAG, "RAW capture failed", e)
            ioExecutor.shutdownNow()
            return paths.toList()
        } finally {
            cameraThread.quitSafely()
        }
    }

    private fun openCamera(manager: CameraManager, cameraId: String, handler: Handler): CameraDevice? {
        val latch = CountDownLatch(1)
        var device: CameraDevice? = null
        manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(d: CameraDevice) { device = d; latch.countDown() }
            override fun onDisconnected(d: CameraDevice) { d.close(); latch.countDown() }
            override fun onError(d: CameraDevice, e: Int) { d.close(); latch.countDown() }
        }, handler)
        return if (latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)) device else null
    }

    private fun createSession(device: CameraDevice, reader: ImageReader, handler: Handler): CameraCaptureSession? {
        val latch = CountDownLatch(1)
        var session: CameraCaptureSession? = null
        device.createCaptureSession(listOf(reader.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(s: CameraCaptureSession) { session = s; latch.countDown() }
                override fun onConfigureFailed(s: CameraCaptureSession) { latch.countDown() }
            }, handler)
        return if (latch.await(TIMEOUT_SECONDS, TimeUnit.SECONDS)) session else null
    }

    fun readRawFile(path: String): RawFrame {
        val bytes = File(path).readBytes()
        val header = ByteBuffer.wrap(bytes, 0, HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        val magic = header.getInt()
        if (magic != HEADER_MAGIC) throw IllegalArgumentException("Invalid .raw file: $path")
        val width = header.getInt(); val height = header.getInt()
        val bitDepth = header.getInt(); val bayerPattern = header.getInt()
        val blackLevel = intArrayOf(header.getInt(), header.getInt(), header.getInt(), header.getInt())
        val whiteLevel = header.getInt()
        val pixelCount = width * height
        val pixels = ShortArray(pixelCount)
        ByteBuffer.wrap(bytes, HEADER_SIZE, bytes.size - HEADER_SIZE)
            .order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(pixels)
        return RawFrame(width, height, bitDepth, bayerPattern, blackLevel, whiteLevel, pixels)
    }
}

data class SensorBounds(
    val exposureRange: Range<Long> = Range(1000L, 1000000000L),
    val isoRange: Range<Int> = Range(100, 3200)
)
