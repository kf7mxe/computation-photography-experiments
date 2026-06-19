package com.kf7mxe.prescent.views

import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.GlobalNavigator
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

class SpatialPage(val frames: List<String> = listOf()) : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Spatial / 3D")

    val isProcessing = Signal(false)
    val sideBySidePath = Signal<String?>(null)
    val depthMapPath = Signal<String?>(null)
    val anaglyphPath = Signal<String?>(null)
    val saveSuccess = Signal(false)
    val rotation = Signal(0)

    private var previewJob: Job? = null

    override fun ElementWriter.CanAddTheme.render() {
        col {
            padded.row {
                button {
                    icon(Icon.arrowBack, "Back")
                    onClick { GlobalNavigator.main.goBack() }
                }
                centered.expanding.h2 { content = "Spatial / 3D" }
            }

            if (frames.size < 2) {
                centered.expanding.text("Need at least 2 images.")
                return@col
            }

            // ── Rotation control ──────────────────────────────────────────
            padded.row {
                text("Rotate:")
                listOf(0 to "0\u00b0", 90 to "90\u00b0", 180 to "180\u00b0", 270 to "270\u00b0").forEach { (deg, label) ->
                    toggleButton {
                        text(label); checked bind rotation.equalTo(deg)
                    }
                }
            }

            expanding.scrolling.col {
                // ── Source image pair ─────────────────────────────────────
                padded.scrolling.row {
                    frames.forEach { path ->
                        sizedBox(SizeConstraints(width = 5.rem, height = 5.rem)).image {
                            source = ImageRemote(if (path.startsWith("/")) "file://$path" else path)
                        }
                    }
                }

                // ── Results ───────────────────────────────────────────────
                card.padded.col {
                    h2 { content = "Side-by-Side" }
                    sizedBox(SizeConstraints(minHeight = 15.rem)).image {
                        reactive {
                            val path = sideBySidePath()
                            if (path != null) {
                                source = ImageRemote(if (path.startsWith("/")) "file://$path" else path)
                            }
                        }
                    }
                }

                card.padded.col {
                    h2 { content = "Depth Map" }
                    sizedBox(SizeConstraints(minHeight = 15.rem)).image {
                        reactive {
                            val path = depthMapPath()
                            if (path != null) {
                                source = ImageRemote(if (path.startsWith("/")) "file://$path" else path)
                            }
                        }
                    }
                }

                card.padded.col {
                    h2 { content = "Red-Cyan Anaglyph" }
                    sizedBox(SizeConstraints(minHeight = 15.rem)).image {
                        reactive {
                            val path = anaglyphPath()
                            if (path != null) {
                                source = ImageRemote(if (path.startsWith("/")) "file://$path" else path)
                            }
                        }
                    }
                }

                // ── Save Success ──────────────────────────────────────────
                shownWhen { saveSuccess() }.frame {
                    centered.card.padded.text("\u2713 Saved to gallery!")
                }

                // ── Process Button ────────────────────────────────────────
                padded.important.button {
                    text { ::content { if (isProcessing()) "Processing..." else "Process Full" } }
                    ::enabled { !isProcessing() }
                    onClick { processInternal(fullSize = true) }
                }
            }

            // Auto-preview whenever rotation changes
            reactive {
                rotation()
                previewJob?.cancel()
                previewJob = launch {
                    delay(600)
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
            val result = processSpatial(
                frames,
                maxPreviewSize = if (fullSize) 0 else 1024,
                rotation = rotation.value
            )
            if (result != null) {
                sideBySidePath.value = result.sideBySidePath
                depthMapPath.value = result.depthMapPath
                anaglyphPath.value = result.anaglyphPath
                if (fullSize) saveSuccess.value = true
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            isProcessing.value = false
        }
    }
}
