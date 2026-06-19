package com.kf7mxe.prescent.views

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.view.View
import android.view.ViewGroup
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.context.reactive
import java.io.File
import java.util.concurrent.Executors

@SuppressLint("UnsafeOptInUsageError")
actual fun ElementWriter.cameraView(
    shutterTrigger: Signal<Int>,
    onImagesCaptured: (List<String>) -> Unit,
    bracketCount: Signal<Int>,
    evOffset: Signal<Float>,
    isHdrMode: Signal<Boolean>,
    cameraLens: Signal<Int>,
    onCameraLabels: ((List<String>) -> Unit)?,
    isNightSight: Signal<Boolean>,
    nightSightFrameCount: Signal<Int>,
    nightSightCaptureTrigger: Signal<Int>,
    onNightSightCaptured: ((List<String>) -> Unit)?,
    isFocusStacking: Signal<Boolean>,
    focusStackFrameCount: Signal<Int>,
    focusStackCaptureTrigger: Signal<Int>,
    onFocusStackCaptured: ((List<String>) -> Unit)?,
    isSpatial: Signal<Boolean>,
    spatialCaptureTrigger: Signal<Int>,
    onSpatialCaptured: ((List<String>) -> Unit)?,
    onSphereOrientationUpdate: ((Pair<Float, Float>) -> Unit)?,
    onSphereFrameOrientation: ((Float, Float) -> Unit)?
) {
    val androidContext = AndroidAppContext.applicationCtx
    val mainExecutor = ContextCompat.getMainExecutor(androidContext)
    val singleExecutor = Executors.newSingleThreadExecutor()

    val previewView = PreviewView(androidContext).apply {
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
    }

    val kiteContext = (this as ViewWriter).context
    val element = object : NativeElement(kiteContext) {
        override val native: View = previewView
    }
    this.write(element) {}

    if (ContextCompat.checkSelfPermission(androidContext, Manifest.permission.CAMERA)
        == PackageManager.PERMISSION_DENIED
    ) {
        Log.d("CameraView", "Requesting camera permissions")
        AndroidAppContext.requestPermissions(Manifest.permission.CAMERA) { result ->
            Log.d("CameraView", "Permission result: ${result.accepted}")
        }
    }

    // ── Rotation sensor for Photo Sphere guidance ───────────────────────
    var currentAzimuth = 0f
    var currentPitch = 0f
    try {
        val sensorManager = androidContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (rotationSensor != null) {
            sensorManager.registerListener(object : SensorEventListener {
                val rotationMatrix = FloatArray(9)
                val orientation = FloatArray(3)
                override fun onSensorChanged(event: SensorEvent) {
                    SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values)
                    SensorManager.getOrientation(rotationMatrix, orientation)
                    currentAzimuth = Math.toDegrees(orientation[0].toDouble()).toFloat()
                    currentPitch = Math.toDegrees(orientation[1].toDouble()).toFloat()
                    onSphereOrientationUpdate?.invoke(currentAzimuth to currentPitch)
                }
                override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) {}
            }, rotationSensor, SensorManager.SENSOR_DELAY_GAME)
        }
    } catch (e: Exception) {
        Log.e("CameraView", "Rotation sensor init failed", e)
    }

    data class CameraEntry(
        val index: Int,
        val label: String,
        val isPhysical: Boolean = false,
        val logicalSelector: CameraSelector,
        val physicalCameraId: String? = null,
    )

    val cameras = mutableListOf<CameraEntry>()

    fun discoverLenses() {
        cameras.clear()
        val seenIds = mutableSetOf<String>()
        try {
            val cameraManager = androidContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val logicalIds = cameraManager.cameraIdList.toList()
            Log.d("CameraView", "CameraManager logical IDs: $logicalIds")

            val logicalToPhysical = mutableMapOf<String, Set<String>>()
            for (logicalId in logicalIds) {
                try {
                    val chars = cameraManager.getCameraCharacteristics(logicalId)
                    val physicalIds = chars.physicalCameraIds
                    logicalToPhysical[logicalId] = physicalIds
                } catch (_: Exception) {
                    logicalToPhysical[logicalId] = emptySet()
                }
            }
            Log.d("CameraView", "Logical->Physical map: $logicalToPhysical")

            var nextIndex = 0

            for (logicalId in logicalIds) {
                if (logicalId in seenIds) continue
                try {
                    val chars = cameraManager.getCameraCharacteristics(logicalId)
                    val lensFacing = chars.get(CameraCharacteristics.LENS_FACING) ?: continue
                    val logicalSel = CameraSelector.Builder()
                        .addCameraFilter(CameraFilter { camerasList ->
                            camerasList.filter { Camera2CameraInfo.from(it).cameraId == logicalId }
                        }).build()
                    val label = when (lensFacing) {
                        CameraCharacteristics.LENS_FACING_BACK -> "Back"
                        CameraCharacteristics.LENS_FACING_FRONT -> "Front"
                        else -> "Cam"
                    }
                    cameras.add(CameraEntry(nextIndex, label, logicalSelector = logicalSel))
                    seenIds.add(logicalId); nextIndex++
                } catch (e: Exception) {
                    Log.e("CameraView", "Skipping logical camera $logicalId: ${e.message}")
                }
            }

            val physToLogical = mutableMapOf<String, String>()
            for ((logicalId, physicalIds) in logicalToPhysical) {
                for (physId in physicalIds) {
                    physToLogical[physId] = logicalId
                }
            }

            data class PhysInfo(val id: String, val focal: Float, val facing: Int)
            val allPhysicals = mutableListOf<PhysInfo>()
            for (physId in physToLogical.keys) {
                if (physId in seenIds) continue
                try {
                    val physChars = cameraManager.getCameraCharacteristics(physId)
                    val facing = physChars.get(CameraCharacteristics.LENS_FACING) ?: continue
                    val focal = physChars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                        ?.firstOrNull() ?: 5.0f
                    allPhysicals.add(PhysInfo(physId, focal, facing))
                    seenIds.add(physId)
                } catch (_: Exception) {}
            }

            val backPhys = allPhysicals.filter { it.facing == CameraCharacteristics.LENS_FACING_BACK }.sortedBy { it.focal }
            val frontPhys = allPhysicals.filter { it.facing == CameraCharacteristics.LENS_FACING_FRONT }

            fun logicalSelectorFor(physId: String): CameraSelector? {
                val logicalId = physToLogical[physId] ?: return null
                val idx = logicalIds.indexOf(logicalId)
                if (idx < 0) return null
                return try {
                    CameraSelector.Builder()
                        .addCameraFilter(CameraFilter { cl ->
                            cl.filter { Camera2CameraInfo.from(it).cameraId == logicalId }
                        }).build()
                } catch (_: Exception) { null }
            }

            for ((idx, phys) in backPhys.withIndex()) {
                val suffix = when {
                    backPhys.size == 1 -> "Wide"
                    idx == 0 -> "Ultrawide"
                    idx == backPhys.lastIndex -> "Tele"
                    else -> "Cam ${idx + 1}"
                }
                val sel = logicalSelectorFor(phys.id) ?: continue
                cameras.add(CameraEntry(nextIndex, "Back $suffix",
                    isPhysical = true, logicalSelector = sel, physicalCameraId = phys.id))
                nextIndex++
            }
            for (phys in frontPhys) {
                val sel = logicalSelectorFor(phys.id) ?: continue
                cameras.add(CameraEntry(nextIndex, "Front Wide",
                    isPhysical = true, logicalSelector = sel, physicalCameraId = phys.id))
                nextIndex++
            }
        } catch (e: Exception) {
            Log.e("CameraView", "Lens discovery via CameraManager failed", e)
        }
        if (cameras.isEmpty()) {
            cameras.add(CameraEntry(0, "Back", logicalSelector = CameraSelector.DEFAULT_BACK_CAMERA))
            cameras.add(CameraEntry(1, "Front", logicalSelector = CameraSelector.DEFAULT_FRONT_CAMERA))
        }
        onCameraLabels?.invoke(cameras.map { it.label })
        Log.d("CameraView", "Discovered ${cameras.size} cameras: ${cameras.map { it.label }}")
    }
    discoverLenses()

    var cameraProvider: ProcessCameraProvider? = null
    var imageCapture: ImageCapture? = null
    var currentCamera: Camera? = null

    fun startCamera(entry: CameraEntry) {
        val provider = cameraProvider ?: return
        val lifecycleOwner = AndroidAppContext.activityCtx ?: return
        Log.d("CameraView", "startCamera: ${entry.label}")
        provider.unbindAll()

        val previewBuilder = Preview.Builder()
        val imageCaptureBuilder = ImageCapture.Builder()

        // For physical cameras, set the physical camera ID via Camera2Interop
        if (entry.isPhysical && entry.physicalCameraId != null) {
            try {
                Camera2Interop.Extender(previewBuilder)
                    .setPhysicalCameraId(entry.physicalCameraId)
                Camera2Interop.Extender(imageCaptureBuilder)
                    .setPhysicalCameraId(entry.physicalCameraId)
                Log.d("CameraView", "Physical camera ID set on use cases: ${entry.physicalCameraId}")
            } catch (e: Exception) {
                Log.e("CameraView", "Could not set physical camera ID on use cases", e)
            }
        }

        val previewUseCase = previewBuilder.build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }
        val imageCaptureUseCase = imageCaptureBuilder.build()
        imageCapture = imageCaptureUseCase
        currentCamera = provider.bindToLifecycle(
            lifecycleOwner, entry.logicalSelector, previewUseCase, imageCaptureUseCase
        )
        Log.d("CameraView", "Camera bound: ${entry.label}")
    }

    // Init camera provider asynchronously
    ProcessCameraProvider.getInstance(androidContext).apply {
        addListener({
            cameraProvider = get()
            val initialEntry = cameras.firstOrNull() ?: return@addListener
            startCamera(initialEntry)
        }, mainExecutor)
    }

    @Suppress("DEPRECATION")
    fun selectLens(index: Int) {
        val entry = cameras.getOrNull(index) ?: return
        Log.d("CameraView", "selectLens: $index ${entry.label}")
        startCamera(entry)
    }

    var lastLensIndex = -1
    reactive {
        val lensIdx = cameraLens()
        if (lensIdx != lastLensIndex) {
            lastLensIndex = lensIdx
            selectLens(lensIdx)
        }
    }

    fun captureSingleShot() {
        Log.d("CameraView", "captureSingleShot")
        onSphereFrameOrientation?.invoke(currentAzimuth, currentPitch)
        val ic = imageCapture ?: return
        val captureDir = File(androidContext.filesDir, "captures/${System.currentTimeMillis()}").also { it.mkdirs() }
        val file = File(captureDir, "shot.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
        ic.takePicture(
            outputOptions, mainExecutor,
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    Log.d("CameraView", "Single shot saved: ${file.absolutePath}")
                    onImagesCaptured(listOf(file.absolutePath))
                }
                override fun onError(exc: ImageCaptureException) {
                    Log.e("CameraView", "Single shot failed", exc)
                    onImagesCaptured(emptyList())
                }
            }
        )
    }

    fun captureBracket() {
        val count = bracketCount.value
        val offset = evOffset.value
        Log.d("CameraView", "captureBracket: count=$count, offset=$offset")
        val capturedFiles = mutableListOf<String>()
        val cameraInfo = currentCamera?.cameraInfo
        val step = cameraInfo?.exposureState?.exposureCompensationStep?.toFloat() ?: 1.0f
        val offsets = if (count <= 1) listOf(0f)
        else (0 until count).map { i -> -offset + (i * 2 * offset / (count - 1)) }
        val bracketDir = File(androidContext.filesDir, "brackets/${System.currentTimeMillis()}").also { it.mkdirs() }
        val control = currentCamera?.cameraControl
        var currentIdx = 0
        fun takeNext() {
            if (currentIdx >= offsets.size) {
                Log.d("CameraView", "Bracket sequence complete: $capturedFiles")
                control?.setExposureCompensationIndex(0)
                onImagesCaptured(capturedFiles)
                return
            }
            val offsetEv = offsets[currentIdx]
            val index = if (step != 0f) (offsetEv / step).toInt() else 0
            val captureAction: () -> Unit = {
                val ic = imageCapture
                if (ic != null) {
                    val file = File(bracketDir, "shot_${currentIdx}.jpg")
                    val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                    ic.takePicture(
                        outputOptions, mainExecutor,
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                capturedFiles.add(file.absolutePath)
                                currentIdx++; takeNext()
                            }
                            override fun onError(exc: ImageCaptureException) {
                                currentIdx++; takeNext()
                            }
                        }
                    )
                }
            }
            if (control != null) {
                control.setExposureCompensationIndex(index).addListener(captureAction, mainExecutor)
            } else captureAction()
        }
        takeNext()
    }

    // ── Night Sight Multi-Frame Capture ─────────────────────────────────
    fun captureNightSightFrames(frameCount: Int, callback: (List<String>) -> Unit) {
        Log.d("CameraView", "Night sight capture: $frameCount frames")
        val capturedFiles = mutableListOf<String>()
        val control = currentCamera?.cameraControl
        val nsDir = File(androidContext.filesDir, "nightsight/${System.currentTimeMillis()}").also { it.mkdirs() }
        var currentIdx = 0

        fun takeNext() {
            if (currentIdx >= frameCount) {
                control?.setExposureCompensationIndex(0)
                Log.d("CameraView", "Night sight complete: $capturedFiles")
                callback(capturedFiles)
                return
            }
            val evTarget = 2.0f
            val step = currentCamera?.cameraInfo?.exposureState?.exposureCompensationStep?.toFloat() ?: 1.0f
            val index = if (step != 0f) (evTarget / step).toInt() else 0
            val captureAction: () -> Unit = {
                val ic = imageCapture
                if (ic != null) {
                    val file = File(nsDir, "frame_${currentIdx}.jpg")
                    val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                    ic.takePicture(
                        outputOptions, mainExecutor,
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                capturedFiles.add(file.absolutePath)
                                currentIdx++
                                singleExecutor.execute {
                                    Thread.sleep(150)
                                    mainExecutor.execute { takeNext() }
                                }
                            }
                            override fun onError(exc: ImageCaptureException) {
                                currentIdx++; takeNext()
                            }
                        }
                    )
                }
            }
            if (control != null) {
                control.setExposureCompensationIndex(index).addListener(captureAction, mainExecutor)
            } else captureAction()
        }
        takeNext()
    }

    // ── Focus Stack Multi-Frame Capture ─────────────────────────────────
    fun captureFocusStackFrames(frameCount: Int, callback: (List<String>) -> Unit) {
        Log.d("CameraView", "Focus stack capture: $frameCount frames")
        val capturedFiles = mutableListOf<String>()
        val control = currentCamera?.cameraControl
        val fsDir = File(androidContext.filesDir, "focusstack/${System.currentTimeMillis()}").also { it.mkdirs() }
        var currentIdx = 0

        fun takeNext() {
            if (currentIdx >= frameCount) {
                control?.setExposureCompensationIndex(0)
                Log.d("CameraView", "Focus stack complete: $capturedFiles")
                callback(capturedFiles)
                return
            }
            val focusDistance = currentIdx.toFloat() / (frameCount - 1).coerceAtLeast(1)
            val captureAction: () -> Unit = {
                val ic = imageCapture
                if (ic != null) {
                    val file = File(fsDir, "frame_${currentIdx}.jpg")
                    val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                    ic.takePicture(
                        outputOptions, mainExecutor,
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                capturedFiles.add(file.absolutePath)
                                currentIdx++
                                singleExecutor.execute {
                                    Thread.sleep(150)
                                    mainExecutor.execute { takeNext() }
                                }
                            }
                            override fun onError(exc: ImageCaptureException) {
                                currentIdx++; takeNext()
                            }
                        }
                    )
                }
            }
            if (control != null) {
                try {
                    val camera2Control = Camera2CameraControl.from(control)
                    val options = CaptureRequestOptions.Builder()
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                        .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, focusDistance)
                        .build()
                    camera2Control.setCaptureRequestOptions(options).addListener(captureAction, mainExecutor)
                } catch (e: Exception) {
                    Log.e("CameraView", "Focus distance control not supported, capturing without focus change", e)
                    captureAction()
                }
            } else captureAction()
        }
        takeNext()
    }

    // ── Spatial / 3D Stereo Pair Capture ───────────────────────────────
    // Uses the same camera for both shots — stereo effect comes from natural
    // hand movement between captures (like Google Motion Photos).
    fun captureSpatialPair(callback: (List<String>) -> Unit) {
        Log.d("CameraView", "Spatial pair capture (same lens)")
        val capturedFiles = mutableListOf<String>()
        val spDir = File(androidContext.filesDir, "spatial/${System.currentTimeMillis()}").also { it.mkdirs() }
        var currentShot = 0
        val totalShots = 2

        fun captureShot() {
            val ic = imageCapture
            if (ic == null) {
                callback(capturedFiles)
                return
            }
            val label = if (currentShot == 0) "left" else "right"
            val file = File(spDir, "${label}.jpg")
            capturedFiles.add(file.absolutePath)
            val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
            ic.takePicture(
                outputOptions, mainExecutor,
                object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                        onDone()
                    }
                    override fun onError(exc: ImageCaptureException) {
                        onDone()
                    }
                    fun onDone() {
                        currentShot++
                        if (currentShot >= totalShots) {
                            Log.d("CameraView", "Spatial pair complete: $capturedFiles")
                            callback(capturedFiles)
                        } else {
                            // Same lens — small delay for natural hand movement
                            singleExecutor.execute {
                                Thread.sleep(300)
                                mainExecutor.execute { captureShot() }
                            }
                        }
                    }
                }
            )
        }
        captureShot()
    }

    // ── Shutter trigger observer ────────────────────────────────────────
    var isInitialRun = true
    reactive {
        shutterTrigger()
        if (isInitialRun) { isInitialRun = false; return@reactive }
        Log.d("CameraView", "Shutter trigger received")
        if (isSpatial()) {
            Log.d("CameraView", "Spatial mode — handled via spatialCaptureTrigger")
        } else if (isNightSight()) {
            Log.d("CameraView", "Night sight mode — handled via nightSightCaptureTrigger")
        } else if (isFocusStacking()) {
            Log.d("CameraView", "Focus stack mode — handled via focusStackCaptureTrigger")
        } else if (isHdrMode.value) {
            captureBracket()
        } else {
            captureSingleShot()
        }
    }

    // ── Night sight capture trigger observer ────────────────────────────
    var nsInitial = true
    reactive {
        nightSightCaptureTrigger()
        if (nsInitial) { nsInitial = false; return@reactive }
        if (isNightSight()) {
            captureNightSightFrames(nightSightFrameCount.value) { frames ->
                onNightSightCaptured?.invoke(frames)
            }
        }
    }

    // ── Focus stack capture trigger observer ────────────────────────────
    var fsInitial = true
    reactive {
        focusStackCaptureTrigger()
        if (fsInitial) { fsInitial = false; return@reactive }
        if (isFocusStacking()) {
            captureFocusStackFrames(focusStackFrameCount.value) { frames ->
                onFocusStackCaptured?.invoke(frames)
            }
        }
    }

    // ── Spatial capture trigger observer ────────────────────────────────
    var spInitial = true
    reactive {
        spatialCaptureTrigger()
        if (spInitial) { spInitial = false; return@reactive }
        if (isSpatial()) {
            captureSpatialPair { frames ->
                onSpatialCaptured?.invoke(frames)
            }
        }
    }
}
