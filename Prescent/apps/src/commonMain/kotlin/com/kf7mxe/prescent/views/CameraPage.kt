package com.kf7mxe.prescent.views

import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.GlobalNavigator
import com.kf7mxe.prescent.bracketCountStore
import com.kf7mxe.prescent.evOffsetStore
import com.kf7mxe.prescent.utils.copyFileReferencesToPaths
import com.kf7mxe.prescent.utils.fmt
import com.lightningkite.kiteui.*
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.context.invoke
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.context.reactive
import com.lightningkite.reactive.extensions.equalTo
import kotlinx.coroutines.launch

class CameraPage : Page, FullscreenPage {

    val bracketCount = Signal(3)
    val evOffset = Signal(2.0f)
    val shutterTrigger = Signal(0)
    val isHdrMode = Signal(true)
    val cameraLens = Signal(0)
    val cameraLabels = Signal(listOf(0 to "Back", 1 to "Front"))

    // Night Sight
    val isNightSight = Signal(false)
    val nightSightFrameCount = Signal(8)
    val nightSightCaptureTrigger = Signal(0)

    // Focus Stack
    val isFocusStacking = Signal(false)
    val focusStackFrameCount = Signal(6)
    val focusStackCaptureTrigger = Signal(0)

    // Spatial / 3D
    val isSpatial = Signal(false)
    val spatialCaptureTrigger = Signal(0)

    // Photo Sphere (accumulates single shots)
    val sphereFrames = Signal<List<String>>(listOf())

    // Capture mode: "hdr", "single", "night", "focus", "spatial", "sphere"
    val captureMode = Signal("hdr")

    override fun ElementWriter.CanAddTheme.render() {
        col {
            // ── Top Bar ───────────────────────────────────────────────────
            padded.row {
                centered.expanding.h2 { content = "Prescent" }

                button {
                    icon(Icon.settings, "Settings")
                    onClick { GlobalNavigator.main.navigate(SettingsPage()) }
                }
            }

            load {
                bracketCountStore().toIntOrNull()?.let { bracketCount.value = it }
                evOffsetStore().toFloatOrNull()?.let { evOffset.value = it }
            }

            // ── Capture Mode Selector ─────────────────────────────────────
            padded.row {
                listOf("hdr" to "HDR", "single" to "Photo", "night" to "Night", "focus" to "Focus", "spatial" to "Spatial", "sphere" to "Sphere").forEach { (mode, label) ->
                    expanding.toggleButton {
                        text(label); checked bind captureMode.equalTo(mode)
                    }
                }
            }

            // Sync captureMode → individual signals
            reactive {
                when (captureMode()) {
                    "hdr" -> { isHdrMode.value = true; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = false }
                    "single" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = false }
                    "night" -> { isHdrMode.value = false; isNightSight.value = true; isFocusStacking.value = false; isSpatial.value = false }
                    "focus" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = true; isSpatial.value = false }
                    "spatial" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = true }
                    "sphere" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = false }
                }
            }

            // Reset sphere frames when leaving sphere mode
            reactive {
                if (captureMode() != "sphere" && sphereFrames().isNotEmpty()) {
                    sphereFrames.value = listOf()
                }
            }

            // ── Camera Viewfinder ─────────────────────────────────────────
            expanding.frame {
                cameraView(
                    shutterTrigger = shutterTrigger,
                    onImagesCaptured = { paths ->
                        when {
                            isHdrMode.value && paths.size > 1 -> GlobalNavigator.main.navigate(HdrProcessingPage(paths))
                            captureMode.value == "sphere" && paths.isNotEmpty() -> {
                                sphereFrames.value = sphereFrames.value + paths
                            }
                        }
                    },
                    bracketCount = bracketCount,
                    evOffset = evOffset,
                    isHdrMode = isHdrMode,
                    cameraLens = cameraLens,
                    isNightSight = isNightSight,
                    nightSightFrameCount = nightSightFrameCount,
                    nightSightCaptureTrigger = nightSightCaptureTrigger,
                    onNightSightCaptured = { frames ->
                        GlobalNavigator.main.navigate(NightSightPage(frames))
                    },
                    isFocusStacking = isFocusStacking,
                    focusStackFrameCount = focusStackFrameCount,
                    focusStackCaptureTrigger = focusStackCaptureTrigger,
                    onFocusStackCaptured = { frames ->
                        GlobalNavigator.main.navigate(FocusStackPage(frames))
                    },
                    isSpatial = isSpatial,
                    spatialCaptureTrigger = spatialCaptureTrigger,
                    onSpatialCaptured = { frames ->
                        GlobalNavigator.main.navigate(SpatialPage(frames))
                    },
                    onCameraLabels = { labels ->
                        cameraLabels.value = labels.mapIndexed { idx, l -> idx to l }
                    }
                )

                shownWhen { isHdrMode() && !isNightSight() }.frame {
                    atTopEnd.card.padded.col {
                        text { ::content { "${bracketCount()} shots" } }
                        text { ::content { "±${evOffset().fmt()} EV" } }
                    }
                }

                shownWhen { isNightSight() }.frame {
                    atTopEnd.card.padded.col {
                        text { ::content { "${nightSightFrameCount()} frames" } }
                    }
                }

                shownWhen { isFocusStacking() }.frame {
                    atTopEnd.card.padded.col {
                        text { ::content { "${focusStackFrameCount()} focus steps" } }
                    }
                }

                shownWhen { isSpatial() }.frame {
                    atTopEnd.card.padded.col {
                        text("Stereo pair")
                    }
                }

                shownWhen { captureMode() == "sphere" }.frame {
                    atTopEnd.card.padded.col {
                        text { ::content { "${sphereFrames().size} captured" } }
                    }
                }
            }

            // ── Bottom Controls ────────────────────────────────────────────
            padded.row {
                button {
                    icon(Icon.upload, "Open images")
                    onClick {
                        launch {
                            val files = context.requestFiles(listOf("image/*"))
                            if (files.isNotEmpty()) {
                                val paths = copyFileReferencesToPaths(files)
                                if (paths.isNotEmpty()) {
                                    GlobalNavigator.main.navigate(HdrProcessingPage(paths))
                                }
                            }
                        }
                    }
                }

                centered.expanding.important.button {
                    icon(Icon.add, "Capture")
                    onClick {
                        when {
                            isNightSight.value -> nightSightCaptureTrigger.value += 1
                            isFocusStacking.value -> focusStackCaptureTrigger.value += 1
                            isSpatial.value -> spatialCaptureTrigger.value += 1
                            else -> shutterTrigger.value += 1
                        }
                    }
                }

                button {
                    icon(Icon.list, "Gallery")
                    onClick { GlobalNavigator.main.navigate(GalleryPage()) }
                }
            }

            // ── Sphere: Stitch All button when frames accumulated ────────
            shownWhen { captureMode() == "sphere" && sphereFrames().size >= 2 }.frame {
                padded.important.button {
                    text { ::content { "Stitch ${sphereFrames().size} frames" } }
                    onClick {
                        GlobalNavigator.main.navigate(PhotoSpherePage(sphereFrames.value))
                    }
                }
            }

            // ── Lens Selector ─────────────────────────────────────────────
            padded.row {
                text("Lens:")
                forEach(cameraLabels) { (idx, label) ->
                    toggleButton {
                        text(label); checked bind cameraLens.equalTo(idx)
                    }
                }
            }
        }
    }
}
