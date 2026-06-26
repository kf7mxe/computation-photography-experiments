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
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.media.ExifInterface
import android.view.Surface
import android.view.View
import android.widget.Toast
import android.view.ViewGroup
import android.widget.FrameLayout
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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.atan

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
    onSphereFrameOrientation: ((Float, Float) -> Unit)?,
    sphereGridData: Signal<List<List<Boolean>>>,
    sphereCurrentCell: Signal<Pair<Int, Int>?>,
    sphereGhostFrames: Signal<List<SphereGhostFrame>>,
    sphereDriftCorrection: Signal<Float>,
    isQuadBayer: Signal<Boolean>,
    quadBayerFrameCount: Signal<Int>,
    quadBayerAlgorithm: Signal<Int>,
    quadBayerCaptureTrigger: Signal<Int>,
    quadBayerPipeToHdr: Signal<Boolean>,
    quadBayerPipeToNightSight: Signal<Boolean>,
    quadBayerSaveDng: Signal<Boolean>,
    quadBayerSmartSelection: Signal<Boolean>,
    onQuadBayerCaptured: ((List<String>) -> Unit)?
) {
    val androidContext = AndroidAppContext.applicationCtx
    val mainExecutor = ContextCompat.getMainExecutor(androidContext)
    val singleExecutor = Executors.newSingleThreadExecutor()

    var cameraProvider: ProcessCameraProvider? = null

    val previewView = PreviewView(androidContext).apply {
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
        addOnAttachStateChangeListener(object : android.view.View.OnAttachStateChangeListener {
            override fun onViewAttachedToWindow(v: android.view.View) {}
            override fun onViewDetachedFromWindow(v: android.view.View) {
                Log.d("CameraView", "PreviewView detached, unbinding camera")
                cameraProvider?.unbindAll()
            }
        })
    }

    val ghostOverlay = GhostOverlayView(androidContext).apply {
        layoutParams = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.MATCH_PARENT
        )
    }

    val rootLayout = FrameLayout(androidContext).apply {
        layoutParams = ViewGroup.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT
        )
        addView(previewView)
        addView(ghostOverlay)
    }

    val kiteContext = (this as ViewWriter).context
    val element = object : NativeElement(kiteContext) {
        override val native: View = rootLayout
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
    var currentElevation = 0f
    var lastGhostFrames: List<SphereGhostFrame> = emptyList()
    var currentHfovDeg = 60f  // updated each time camera starts
    var currentVfovDeg = 45f
    val display = androidContext.getSystemService(Context.WINDOW_SERVICE).let { (it as android.view.WindowManager).defaultDisplay }
    try {
        val sensorManager = androidContext.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        // Game rotation vector uses gyro+accel only — no magnetometer drift in heading
        var rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        if (rotationSensor == null) rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
        if (rotationSensor != null) {
            sensorManager.registerListener(object : SensorEventListener {
                val R = FloatArray(9)
                val remappedR = FloatArray(9)
                override fun onSensorChanged(event: SensorEvent) {
                    SensorManager.getRotationMatrixFromVector(R, event.values)
                    // Remap sensor axes to display orientation
                    val rot = display.rotation
                    when (rot) {
                        Surface.ROTATION_90 -> SensorManager.remapCoordinateSystem(
                            R, SensorManager.AXIS_Y, SensorManager.AXIS_MINUS_X, remappedR)
                        Surface.ROTATION_270 -> SensorManager.remapCoordinateSystem(
                            R, SensorManager.AXIS_MINUS_Y, SensorManager.AXIS_X, remappedR)
                        Surface.ROTATION_180 -> SensorManager.remapCoordinateSystem(
                            R, SensorManager.AXIS_MINUS_X, SensorManager.AXIS_MINUS_Y, remappedR)
                        else -> System.arraycopy(R, 0, remappedR, 0, 9)
                    }
                    val camUp = -remappedR[8].toDouble()
                    val camEast = -remappedR[2].toDouble()
                    val camNorth = -remappedR[5].toDouble()
                    currentAzimuth = Math.toDegrees(Math.atan2(camEast, camNorth)).toFloat()
                    currentElevation = Math.toDegrees(Math.asin(camUp)).toFloat()
                    onSphereOrientationUpdate?.invoke(currentAzimuth to currentElevation)
                    ghostOverlay.update(lastGhostFrames, currentAzimuth, currentElevation, display.rotation, currentHfovDeg, currentVfovDeg, sphereDriftCorrection.value)
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

    var imageCapture: ImageCapture? = null
    var currentCamera: Camera? = null
    var currentEntry: CameraEntry? = null

    fun startCamera(entry: CameraEntry) {
        currentEntry = entry
        val provider = cameraProvider ?: return
        val lifecycleOwner = AndroidAppContext.activityCtx ?: return
        Log.d("CameraView", "startCamera: ${entry.label}")
        provider.unbindAll()

        val previewBuilder = Preview.Builder()
        val imageCaptureBuilder = ImageCapture.Builder()

        val displayRotation = AndroidAppContext.activityCtx?.windowManager?.defaultDisplay?.rotation ?: Surface.ROTATION_0
        previewBuilder.setTargetRotation(displayRotation)
        imageCaptureBuilder.setTargetRotation(displayRotation)

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

        // Read actual FOV from camera characteristics
        try {
            val cameraManager = androidContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val camId = if (entry.isPhysical) entry.physicalCameraId
                else Camera2CameraInfo.from(currentCamera!!.cameraInfo).cameraId
            val chars = cameraManager.getCameraCharacteristics(camId!!)
            val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            if (focalLengths != null && focalLengths.isNotEmpty()) {
                val focalMM = focalLengths[0].toDouble()
                // Sensor size is either physical or we estimate from active array
                var sensorW = 0.0; var sensorH = 0.0
                val physSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                if (physSize != null) {
                    sensorW = physSize.width.toDouble()
                    sensorH = physSize.height.toDouble()
                } else {
                    // Fallback: estimate from active array size and typical 1.4µm pixel pitch
                    val activeSize = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                    if (activeSize != null) {
                        val pixelPitchMM = 0.0014
                        sensorW = activeSize.width().toDouble() * pixelPitchMM
                        sensorH = activeSize.height().toDouble() * pixelPitchMM
                    }
                }
                if (sensorW > 0 && focalMM > 0) {
                    currentHfovDeg = Math.toDegrees(2.0 * atan(sensorW / (2.0 * focalMM))).toFloat()
                    currentVfovDeg = if (sensorH > 0) Math.toDegrees(2.0 * atan(sensorH / (2.0 * focalMM))).toFloat()
                    else currentHfovDeg * 0.75f
                }
                Log.d("CameraView", "FOV: ${currentHfovDeg.toInt()}°×${currentVfovDeg.toInt()}° from f=${focalMM}mm sensor=${"%.2f".format(sensorW)}mm×${"%.2f".format(sensorH)}mm")
            }
        } catch (e: Exception) {
            Log.d("CameraView", "FOV from characteristics failed (using default): ${e.message}")
        }
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
        onSphereFrameOrientation?.invoke(currentAzimuth, currentElevation)
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

    // ── Quad Bayer RAW Capture ──────────────────────────────────────────
    var quadBayerCapturing = false
    fun captureQuadBayerFrames(callback: (List<String>) -> Unit) {
        if (quadBayerCapturing) {
            Log.w("CameraView", "Quad Bayer: already capturing, ignoring trigger")
            return
        }
        quadBayerCapturing = true
        val framesPerBracket = quadBayerFrameCount.value.coerceIn(1, 3)
        val pipeHdr = quadBayerPipeToHdr.value
        val pipeNight = quadBayerPipeToNightSight.value

        fun done(result: List<String>) {
            quadBayerCapturing = false
            callback(result)
        }

        val entry = currentEntry ?: run {
            Log.e("CameraView", "Quad Bayer: no current entry")
            done(emptyList()); return
        }
        val camManager = androidContext.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val cameraId = currentCamera?.let { QuadBayerEngine.getCameraId(it) }
            ?: entry.physicalCameraId
            ?: run {
            Log.e("CameraView", "Quad Bayer: no active camera")
            done(emptyList()); return
        }

        val info = QuadBayerEngine.detectRawCapability(camManager, cameraId)
        if (!info.supportsRaw) {
            Log.e("CameraView", "RAW not supported on camera $cameraId")
            Toast.makeText(androidContext, "RAW not supported on this camera", Toast.LENGTH_SHORT).show()
            done(emptyList()); return
        }

        Log.d("CameraView", "Quad Bayer capture: $framesPerBracket frames/bracket, sensor ${info.pixelArraySize}")
        cameraProvider?.unbindAll()

        GlobalScope.launch(Dispatchers.IO) {
            try {
                if (pipeHdr && framesPerBracket >= 1) {
                    // ── HDR pipe: 3 brackets × N frames each ──
                    val bracketEVs = listOf(-2f, 0f, 2f)
                    val algo = QuadBayerAlgorithm.entries.getOrElse(
                        quadBayerAlgorithm.value
                    ) { QuadBayerAlgorithm.FULL_REMOSAIC }

                    val bounds = QuadBayerEngine.getSensorInfo(androidContext, cameraId)

                    withContext(Dispatchers.Main) {
                        Toast.makeText(androidContext, "Quad Bayer: metering scene...", Toast.LENGTH_SHORT).show()
                    }
                    val meter = QuadBayerEngine.measureAutoExposure(androidContext, cameraId, info, bounds)

                    if (meter == null) {
                        withContext(Dispatchers.Main) {
                            startCamera(entry)
                            Toast.makeText(androidContext, "Quad Bayer: metering failed", Toast.LENGTH_LONG).show()
                            done(emptyList())
                        }
                        return@launch
                    }

                    val bracketJpegs = mutableListOf<String>()
                    val totalBrackets = bracketEVs.size

                    withContext(Dispatchers.Main) {
                        Toast.makeText(androidContext, "Quad Bayer HDR: capturing $totalBrackets brackets...", Toast.LENGTH_SHORT).show()
                    }

                    for ((bi, ev) in bracketEVs.withIndex()) {
                        val ratio = Math.pow(2.0, ev.toDouble())
                        val targetExp = (meter.exposureTimeNs * ratio).toLong()
                            .coerceIn(bounds.exposureRange.lower, bounds.exposureRange.upper)
                        val targetIso = meter.iso.coerceIn(bounds.isoRange.lower, bounds.isoRange.upper)

                        val bracketDir = File(androidContext.cacheDir,
                            "quadbayer/${System.currentTimeMillis()}_bracket${bi}")
                            .also { it.mkdirs() }

                        val rawPaths = QuadBayerEngine.captureRawFrames(
                            androidContext, cameraId, framesPerBracket, bracketDir, info,
                            ManualExposure(targetExp, targetIso)
                        )

                        if (rawPaths.size < framesPerBracket) {
                            Log.w("CameraView", "Bracket $bi: only ${rawPaths.size}/$framesPerBracket frames")
                        }

                        if (rawPaths.isNotEmpty()) {
                            val result = processQuadBayer(rawPaths, QuadBayerOptions(algorithm = algo, saveDng = quadBayerSaveDng.value, smartSelection = quadBayerSmartSelection.value))
                            if (result != null) {
                                bracketJpegs.add(result)
                                withContext(Dispatchers.Main) {
                                    Toast.makeText(androidContext, "Quad Bayer HDR: bracket ${bi+1}/$totalBrackets done", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    }

                    Log.d("CameraView", "Quad Bayer HDR: bracketJpegs final size=${bracketJpegs.size} paths=${bracketJpegs}")
                    withContext(Dispatchers.Main) {
                        startCamera(entry)
                        if (bracketJpegs.size >= 2) {
                            Toast.makeText(androidContext, "Quad Bayer HDR: ${bracketJpegs.size} brackets ready", Toast.LENGTH_SHORT).show()
                            done(bracketJpegs)
                        } else {
                            Toast.makeText(androidContext, "Quad Bayer HDR: bracket capture failed (got ${bracketJpegs.size})", Toast.LENGTH_LONG).show()
                            done(emptyList())
                        }
                    }
                } else {
                    // ── Single bracket (no pipe): N frames, auto-exposure, merge into 1 JPEG ──
                    val outputDir = File(androidContext.cacheDir, "quadbayer/${System.currentTimeMillis()}")
                        .also { it.mkdirs() }

                    val rawPaths = QuadBayerEngine.captureRawFrames(
                        androidContext, cameraId, framesPerBracket, outputDir, info
                    )

                    if (rawPaths.isEmpty()) {
                        withContext(Dispatchers.Main) {
                            startCamera(entry)
                            Toast.makeText(androidContext, "RAW capture failed", Toast.LENGTH_SHORT).show()
                            done(emptyList())
                        }
                        return@launch
                    }

                    withContext(Dispatchers.Main) {
                        startCamera(entry)
                        Toast.makeText(androidContext, "Processing ${rawPaths.size} RAW frames...", Toast.LENGTH_SHORT).show()
                    }

                    try {
                        val algo = QuadBayerAlgorithm.entries.getOrElse(
                            quadBayerAlgorithm.value
                        ) { QuadBayerAlgorithm.FULL_REMOSAIC }

                        val saveDng = quadBayerSaveDng.value
                        val result = processQuadBayer(rawPaths, QuadBayerOptions(algorithm = algo, saveDng = saveDng, smartSelection = quadBayerSmartSelection.value))
                        withContext(Dispatchers.Main) {
                            if (result != null) {
                                if (!pipeNight) {
                                    val bitmap = BitmapFactory.decodeFile(result)
                                    if (bitmap != null) {
                                        saveBayerToGallery(bitmap)
                                        if (saveDng) {
                                            try {
                                                val dngFile = File(outputDir, "merged.dng")
                                                if (dngFile.exists()) {
                                                    saveDngToGallery(dngFile)
                                                }
                                            } catch (e: Exception) {
                                                Log.e("CameraView", "DNG gallery save failed", e)
                                            }
                                        }
                                        bitmap.recycle()
                                    }
                                }
                                done(listOf(result))
                            } else {
                                Toast.makeText(androidContext, "Quad Bayer processing failed", Toast.LENGTH_LONG).show()
                                done(emptyList())
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("CameraView", "Quad Bayer processing crash", e)
                        withContext(Dispatchers.Main) {
                            Toast.makeText(androidContext, "Quad Bayer error: ${e.message}", Toast.LENGTH_LONG).show()
                            done(emptyList())
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("CameraView", "Quad Bayer capture failed", e)
                withContext(Dispatchers.Main) {
                    startCamera(entry)
                    Toast.makeText(androidContext, "Quad Bayer error: ${e.message}", Toast.LENGTH_LONG).show()
                    done(emptyList())
                }
            }
        }
    }

    // ── Sphere ghost frames sync & overlay update ───────────────────────
    reactive {
        lastGhostFrames = sphereGhostFrames()
        ghostOverlay.update(lastGhostFrames, currentAzimuth, currentElevation, display.rotation, currentHfovDeg, currentVfovDeg, sphereDriftCorrection())
    }

    // ── Shutter trigger observer ────────────────────────────────────────
    var isInitialRun = true
    reactive {
        shutterTrigger()
        if (isInitialRun) { isInitialRun = false; return@reactive }
        Log.d("CameraView", "Shutter trigger received")
        if (isQuadBayer()) {
            Log.d("CameraView", "Quad Bayer mode — handled via quadBayerCaptureTrigger")
        } else if (isSpatial()) {
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

    // ── Quad Bayer capture trigger observer ────────────────────────────
    var qbInitial = true
    reactive {
        quadBayerCaptureTrigger()
        if (qbInitial) { qbInitial = false; return@reactive }
        if (isQuadBayer()) {
            captureQuadBayerFrames { frames ->
                onQuadBayerCaptured?.invoke(frames)
            }
        }
    }
}

    private class GhostOverlayView(context: Context) : View(context) {
        private var fovHorizontal: Float = 60f
        private var fovVertical: Float = 45f
        private var frames: List<SphereGhostFrame> = emptyList()
        private var currentAzimuth: Float = 0f
        private var currentPitch: Float = 0f
        private var currentDrift: Float = 0f
        private var displayRotation: Int = Surface.ROTATION_0
        private val bitmapCache = mutableMapOf<String, Bitmap?>()
        private val ghostPaint = Paint().apply { isFilterBitmap = true; alpha = 90 }
        private val tintPaint = Paint().apply { color = (0x4400AA00).toInt(); style = Paint.Style.FILL }

        init { setWillNotDraw(false) }

        fun update(frames: List<SphereGhostFrame>, curAz: Float, curPitch: Float, displayRot: Int, hfov: Float = fovHorizontal, vfov: Float = fovVertical, drift: Float = 0f) {
            this.frames = frames
            this.currentAzimuth = curAz
            this.currentPitch = curPitch
            this.currentDrift = drift
            this.displayRotation = displayRot
            this.fovHorizontal = hfov
            this.fovVertical = vfov
            if (frames.isEmpty()) bitmapCache.clear()
            invalidate()
        }

        override fun onDraw(canvas: Canvas) {
            super.onDraw(canvas)
            if (frames.isEmpty()) return

            val w = width.toFloat()
            val h = height.toFloat()

            val fovH: Float
            val fovV: Float
            when (displayRotation) {
                Surface.ROTATION_90, Surface.ROTATION_270 -> {
                    fovH = fovVertical
                    fovV = fovHorizontal
                }
                else -> {
                    fovH = fovHorizontal
                    fovV = fovVertical
                }
            }

            for (frame in frames) {
                val trueAzAtCapture = frame.azimuth + frame.driftAtCapture
                val trueAzNow = currentAzimuth + currentDrift
                var dAz = (trueAzAtCapture - trueAzNow) % 360f
                if (dAz < -180f) dAz += 360f
                if (dAz >= 180f) dAz -= 360f
                val dPitch = frame.pitch - currentPitch
                if (abs(dAz) > fovH || abs(dPitch) > fovV) continue

                val cx = w / 2f + (dAz / fovH) * w
                val cy = h / 2f - (dPitch / fovV) * h

                val fl = cx - w / 2f
                val ft = cy - h / 2f
                val fr = cx + w / 2f
                val fb = cy + h / 2f

                val bmp = bitmapCache.getOrPut(frame.path) {
                    try {
                        val o = BitmapFactory.Options().apply { inSampleSize = 4 }
                        val raw = BitmapFactory.decodeFile(frame.path, o) ?: return@getOrPut null
                        val exif = ExifInterface(frame.path)
                        val rot = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
                        when (rot) {
                            ExifInterface.ORIENTATION_ROTATE_90 -> {
                                val m = Matrix().apply { postRotate(90f) }
                                val r = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                                if (r != raw) raw.recycle(); r
                            }
                            ExifInterface.ORIENTATION_ROTATE_180 -> {
                                val m = Matrix().apply { postRotate(180f) }
                                val r = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                                if (r != raw) raw.recycle(); r
                            }
                            ExifInterface.ORIENTATION_ROTATE_270 -> {
                                val m = Matrix().apply { postRotate(270f) }
                                val r = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                                if (r != raw) raw.recycle(); r
                            }
                            else -> raw
                        }
                    } catch (e: Exception) { null }
                }
                if (bmp == null) continue

                canvas.drawBitmap(bmp, Rect(0, 0, bmp.width, bmp.height), RectF(fl, ft, fr, fb), ghostPaint)
                canvas.drawRect(fl, ft, fr, fb, tintPaint)
            }
        }
    }