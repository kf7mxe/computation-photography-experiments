package com.kf7mxe.prescent.views

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.hardware.camera2.CaptureRequest
import android.util.Log
import android.view.View
import android.view.ViewGroup
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.LifecycleCameraController
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
    isNightSight: Signal<Boolean>,
    nightSightFrameCount: Signal<Int>,
    nightSightCaptureTrigger: Signal<Int>,
    onNightSightCaptured: ((List<String>) -> Unit)?,
    isFocusStacking: Signal<Boolean>,
    focusStackFrameCount: Signal<Int>,
    focusStackCaptureTrigger: Signal<Int>,
    onFocusStackCaptured: ((List<String>) -> Unit)?
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

    val cameraController = LifecycleCameraController(androidContext).apply {
        AndroidAppContext.activityCtx?.let {
            Log.d("CameraView", "Binding to lifecycle")
            bindToLifecycle(it)
        } ?: Log.e("CameraView", "Activity context is null")
        cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        previewView.controller = this
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

    val availableLenses = mutableListOf<Pair<Int, CameraSelector>>()
    fun discoverLenses() {
        availableLenses.clear()
        try {
            val cameraProvider = ProcessCameraProvider.getInstance(androidContext).get()
            cameraProvider.availableCameraInfos.forEach { info ->
                val lensFacing = info.lensFacing
                val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
                when (lensFacing) {
                    CameraSelector.LENS_FACING_BACK -> availableLenses.add(0 to selector)
                    CameraSelector.LENS_FACING_FRONT -> availableLenses.add(1 to selector)
                    else -> {}
                }
            }
        } catch (e: Exception) {
            Log.e("CameraView", "Lens discovery failed", e)
        }
        if (availableLenses.isEmpty()) {
            availableLenses.add(0 to CameraSelector.DEFAULT_BACK_CAMERA)
            availableLenses.add(1 to CameraSelector.DEFAULT_FRONT_CAMERA)
        }
        Log.d("CameraView", "Discovered ${availableLenses.size} lenses")
    }
    discoverLenses()

    fun selectLens(index: Int) {
        val selector = availableLenses.getOrNull(index)?.second ?: CameraSelector.DEFAULT_BACK_CAMERA
        cameraController.cameraSelector = selector
        Log.d("CameraView", "Switched to lens index $index")
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
        val captureDir = File(androidContext.filesDir, "captures/${System.currentTimeMillis()}").also { it.mkdirs() }
        val file = File(captureDir, "shot.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
        cameraController.takePicture(
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
        val camera = cameraController.cameraInfo
        val step = camera?.exposureState?.exposureCompensationStep?.toFloat() ?: 1.0f
        val offsets = if (count <= 1) listOf(0f)
        else (0 until count).map { i -> -offset + (i * 2 * offset / (count - 1)) }
        val bracketDir = File(androidContext.filesDir, "brackets/${System.currentTimeMillis()}").also { it.mkdirs() }
        var currentIdx = 0
        fun takeNext() {
            if (currentIdx >= offsets.size) {
                Log.d("CameraView", "Bracket sequence complete: $capturedFiles")
                cameraController.cameraControl?.setExposureCompensationIndex(0)
                onImagesCaptured(capturedFiles)
                return
            }
            val offsetEv = offsets[currentIdx]
            val index = if (step != 0f) (offsetEv / step).toInt() else 0
            val control = cameraController.cameraControl
            val captureAction = {
                val file = File(bracketDir, "shot_${currentIdx}.jpg")
                val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                cameraController.takePicture(
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
            if (control != null) {
                control.setExposureCompensationIndex(index).addListener({ captureAction() }, mainExecutor)
            } else captureAction()
        }
        takeNext()
    }

    // ── Night Sight Multi-Frame Capture ─────────────────────────────────
    fun captureNightSightFrames(frameCount: Int, callback: (List<String>) -> Unit) {
        Log.d("CameraView", "Night sight capture: $frameCount frames")
        val capturedFiles = mutableListOf<String>()
        // Use positive EV to brighten each frame
        val control = cameraController.cameraControl
        val nsDir = File(androidContext.filesDir, "nightsight/${System.currentTimeMillis()}").also { it.mkdirs() }
        var currentIdx = 0

        fun takeNext() {
            if (currentIdx >= frameCount) {
                control?.setExposureCompensationIndex(0)
                Log.d("CameraView", "Night sight complete: $capturedFiles")
                callback(capturedFiles)
                return
            }
            // Boost EV: +2 for night sight (brighter frames)
            val evTarget = 2.0f
            val step = cameraController.cameraInfo?.exposureState?.exposureCompensationStep?.toFloat() ?: 1.0f
            val index = if (step != 0f) (evTarget / step).toInt() else 0
            val captureAction = {
                val file = File(nsDir, "frame_${currentIdx}.jpg")
                val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                cameraController.takePicture(
                    outputOptions, mainExecutor,
                    object : ImageCapture.OnImageSavedCallback {
                        override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                            capturedFiles.add(file.absolutePath)
                            currentIdx++
                            // Small delay between frames to reduce camera shake
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
            if (control != null) {
                control.setExposureCompensationIndex(index).addListener({ captureAction() }, mainExecutor)
            } else captureAction()
        }
        takeNext()
    }

    // ── Focus Stack Multi-Frame Capture ─────────────────────────────────
    fun captureFocusStackFrames(frameCount: Int, callback: (List<String>) -> Unit) {
        Log.d("CameraView", "Focus stack capture: $frameCount frames")
        val capturedFiles = mutableListOf<String>()
        val control = cameraController.cameraControl
        val fsDir = File(androidContext.filesDir, "focusstack/${System.currentTimeMillis()}").also { it.mkdirs() }
        var currentIdx = 0

        fun takeNext() {
            if (currentIdx >= frameCount) {
                control?.setExposureCompensationIndex(0)
                Log.d("CameraView", "Focus stack complete: $capturedFiles")
                callback(capturedFiles)
                return
            }
            // Sweep focus from near (0) to far (1) across the sequence
            val focusDistance = currentIdx.toFloat() / (frameCount - 1).coerceAtLeast(1)
            val captureAction = {
                val file = File(fsDir, "frame_${currentIdx}.jpg")
                val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                cameraController.takePicture(
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
            // Set focus distance before capturing (via Camera2 interop)
            if (control != null) {
                try {
                    val camera2Control = Camera2CameraControl.from(control)
                    val options = CaptureRequestOptions.Builder()
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                        .setCaptureRequestOption(CaptureRequest.LENS_FOCUS_DISTANCE, focusDistance)
                        .build()
                    camera2Control.setCaptureRequestOptions(options).addListener({ captureAction() }, mainExecutor)
                } catch (e: Exception) {
                    Log.e("CameraView", "Focus distance control not supported, capturing without focus change", e)
                    captureAction()
                }
            } else captureAction()
        }
        takeNext()
    }

    // ── Shutter trigger observer ────────────────────────────────────────
    var isInitialRun = true
    reactive {
        shutterTrigger()
        if (isInitialRun) { isInitialRun = false; return@reactive }
        Log.d("CameraView", "Shutter trigger received")
        if (isNightSight()) {
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
}
