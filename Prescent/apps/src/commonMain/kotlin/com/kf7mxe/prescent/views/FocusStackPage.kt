package com.kf7mxe.prescent.views

import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.GlobalNavigator
import com.kf7mxe.prescent.utils.fmt
import com.lightningkite.kiteui.*
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.context.reactive
import com.lightningkite.reactive.core.Constant
import com.lightningkite.reactive.core.Reactive
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.extensions.equalTo
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class FocusStackPage(val frames: List<String> = listOf()) : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Focus Stack")

    val isProcessing = Signal(false)
    val previewPath = Signal<String?>(null)
    val resultPath = Signal<String?>(null)
    val saveSuccess = Signal(false)

    val algorithm = Signal("Multi-Scale Pyramid")
    val alignmentMethod = Signal("MTB")
    val exposureBalance = Signal(false)
    val showDepthMap = Signal(false)
    val refocusDepth = Signal(0.5f)
    val focalLength = Signal(50.0f)
    val aperture = Signal(2.8f)
    val focusDistanceMeters = Signal(1.0f)
    val pyramidLevels = Signal(4.0f)

    private val algorithms = listOf(
        "Multi-Scale Pyramid",
        "Depth Map",
        "Interactive Refocus",
        "Exposure Balanced",
        "Feature Align",
        "HDR Hybrid",
        "Focus Bracketing Optimizer"
    )

    private var previewJob: Job? = null

    override fun ElementWriter.CanAddTheme.render() {
        col {
            padded.row {
                button {
                    icon(Icon.arrowBack, "Back")
                    onClick { GlobalNavigator.main.goBack() }
                }
                centered.expanding.h2 { content = "Focus Stack" }
            }

            if (frames.isEmpty()) {
                centered.expanding.text("No frames captured. Go back and capture in Focus Stack mode.")
                return@col
            }

            expanding.scrolling.col {
                // ── Input frame strip ─────────────────────────────────────
                padded.scrolling.row {
                    frames.forEach { path ->
                        sizedBox(SizeConstraints(width = 4.rem, height = 4.rem)).image {
                            source = ImageRemote(if (path.startsWith("/")) "file://$path" else path)
                        }
                    }
                }

                // ── Algorithm Picker ──────────────────────────────────────
                card.padded.col {
                    h2 { content = "Algorithm" }
                    subtext("Choose your focus stacking algorithm")
                    row {
                        algorithms.forEach { algo ->
                            expanding.toggleButton {
                                text(algo); checked bind algorithm.equalTo(algo)
                            }
                        }
                    }

                    // Wavelet sharpness levels
                    shownWhen { algorithm() == "Multi-Scale Pyramid" }.col {
                        text { ::content { "Pyramid Levels: ${pyramidLevels().toInt()}" } }
                        slider { max = 6.0f; min = 2.0f; step = 1.0f; value.bind(pyramidLevels) }
                        subtext("More levels = finer detail selection but slower")
                    }

                    // Depth map
                    shownWhen { algorithm() == "Depth Map" }.col {
                        row { toggleButton { text("Colorized Depth Map"); checked bind showDepthMap } }
                    }

                    // Interactive refocus
                    shownWhen { algorithm() == "Interactive Refocus" }.col {
                        text { ::content { "Focal Plane: ${refocusDepth().fmt()}" } }
                        slider { range(0.0f, 1.0f, 0.02f); value.bind(refocusDepth) }
                        subtext("0 = near, 1 = far. Tap to refocus on a region.")
                    }

                    // Lens specs for optimizer
                    shownWhen { algorithm() == "Focus Bracketing Optimizer" }.col {
                        col {
                            text { ::content { "Focal Length: ${focalLength().fmt()}mm" } }
                            slider { max = 200.0f; min = 10.0f; step = 5.0f; value.bind(focalLength) }
                        }
                        col {
                            text { ::content { "Aperture: f/${aperture().fmt()}" } }
                            slider { max = 22.0f; min = 1.4f; step = 0.1f; value.bind(aperture) }
                        }
                        col {
                            text { ::content { "Focus Distance: ${focusDistanceMeters().fmt()}m" } }
                            slider { max = 50.0f; min = 0.1f; step = 0.1f; value.bind(focusDistanceMeters) }
                        }
                    }
                }

                // ── Alignment ─────────────────────────────────────────────
                card.padded.col {
                    h2 { content = "Alignment" }
                    row {
                        listOf("MTB", "Feature", "None").forEach { opt ->
                            expanding.toggleButton {
                                text(opt); checked bind alignmentMethod.equalTo(opt)
                            }
                        }
                    }
                    subtext("Feature-based handles magnification changes (focus breathing) better than MTB")
                }

                // ── Advanced ──────────────────────────────────────────────
                card.padded.col {
                    h2 { content = "Advanced" }
                    col {
                        row { toggleButton { text("Exposure Balance"); checked bind exposureBalance } }
                        subtext("Normalizes brightness shifts caused by focus breathing")
                    }
                }

                // ── Preview & Result ──────────────────────────────────────
                sizedBox(SizeConstraints(minHeight = 25.rem)).frame {
                    image {
                        reactive {
                            val full = resultPath()
                            val prev = previewPath()
                            when {
                                full != null -> {
                                    source = ImageRemote(if (full.startsWith("/")) "file://$full" else full)
                                    visible = true; opacity = 1.0
                                }
                                prev != null -> {
                                    source = ImageRemote(if (prev.startsWith("/")) "file://$prev" else prev)
                                    visible = true; opacity = 0.7
                                }
                                else -> visible = false
                            }
                        }
                    }

                    shownWhen { isProcessing() && previewPath() == null }.frame {
                        atBottomCenter.activityIndicator()
                    }

                    shownWhen {
                        !isProcessing() && previewPath() == null && resultPath() == null
                    }.frame {
                        centered.text("Press \"Process\" to blend frames")
                    }

                    shownWhen { isProcessing() && previewPath() != null }.frame {
                        atCenterEnd.padded.text("Updating...")
                    }
                }

                // ── Save Success ──────────────────────────────────────────
                shownWhen { saveSuccess() }.frame {
                    centered.card.padded.text("✓ Saved to gallery!")
                }

                // ── Process Button ────────────────────────────────────────
                padded.important.button {
                    text { ::content { if (isProcessing()) "Processing..." else "Process Full Image" } }
                    ::enabled { !isProcessing() }
                    onClick { processInternal(fullSize = true) }
                }
            }

            // Auto-preview on page load + parameter change (debounced)
            reactive {
                frames.hashCode()
                algorithm()
                alignmentMethod()
                exposureBalance()
                showDepthMap()
                refocusDepth()
                previewJob?.cancel()
                previewJob = launch {
                    delay(800)
                    processInternal(fullSize = false)
                }
            }
        }
    }

    private suspend fun processInternal(fullSize: Boolean) {
        if (isProcessing.value) return
        isProcessing.value = true
        if (fullSize) saveSuccess.value = false

        try {
            val result = processFocusStack(
                frames,
                maxPreviewSize = if (fullSize) 0 else 1024,
                algorithm = algorithm.value,
                alignmentMethod = alignmentMethod.value,
                exposureBalance = exposureBalance.value,
                showDepthMap = showDepthMap.value,
                refocusDepth = refocusDepth.value,
                focalLength = focalLength.value,
                aperture = aperture.value,
                focusDistanceMeters = focusDistanceMeters.value,
                hdrHybridFramesPerFocus = 0,
                pyramidLevels = pyramidLevels.value.toInt()
            )
            if (fullSize) {
                resultPath.value = result
                if (result != null) saveSuccess.value = true
            } else {
                previewPath.value = result
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            isProcessing.value = false
        }
    }
}
