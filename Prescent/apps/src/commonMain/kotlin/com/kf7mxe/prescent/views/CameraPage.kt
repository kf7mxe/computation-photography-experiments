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

    // Night Sight
    val isNightSight = Signal(false)
    val nightSightFrameCount = Signal(8)
    val nightSightCaptureTrigger = Signal(0)

    // Focus Stack
    val isFocusStacking = Signal(false)
    val focusStackFrameCount = Signal(6)
    val focusStackCaptureTrigger = Signal(0)

    // Capture mode: "hdr", "single", "night", "focus"
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
                listOf("hdr" to "HDR", "single" to "Photo", "night" to "Night", "focus" to "Focus").forEach { (mode, label) ->
                    expanding.toggleButton {
                        text(label)
                        checked bind captureMode.equalTo(mode)
                    }
                }
            }

            // Sync captureMode → individual signals
            reactive {
                when (captureMode()) {
                    "hdr" -> { isHdrMode.value = true; isNightSight.value = false; isFocusStacking.value = false }
                    "single" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = false }
                    "night" -> { isHdrMode.value = false; isNightSight.value = true; isFocusStacking.value = false }
                    "focus" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = true }
                }
            }

            // ── Camera Viewfinder ─────────────────────────────────────────
            expanding.frame {
                cameraView(
                    shutterTrigger = shutterTrigger,
                    onImagesCaptured = { paths ->
                        if (isHdrMode.value && paths.size > 1) {
                            GlobalNavigator.main.navigate(HdrProcessingPage(paths))
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
                            else -> shutterTrigger.value += 1
                        }
                    }
                }

                button {
                    icon(Icon.list, "Gallery")
                    onClick { GlobalNavigator.main.navigate(GalleryPage()) }
                }
            }

            // ── Lens Selector ─────────────────────────────────────────────
            padded.row {
                text("Lens:")
                listOf("Back" to 0, "Front" to 1).forEach { (label, idx) ->
                    toggleButton {
                        text(label); checked bind cameraLens.equalTo(idx)
                    }
                }
            }
        }
    }
}
