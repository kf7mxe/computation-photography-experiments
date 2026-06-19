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
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class PhotoSpherePage(
    val frames: List<String> = listOf(),
    val orientations: List<Pair<Float, Float>> = listOf()
) : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Photo Sphere")

    val isProcessing = Signal(false)
    val previewPath = Signal<String?>(null)
    val resultPath = Signal<String?>(null)
    val saveSuccess = Signal(false)

    private var previewJob: Job? = null

    override fun ElementWriter.CanAddTheme.render() {
        col {
            padded.row {
                button {
                    icon(Icon.arrowBack, "Back")
                    onClick { GlobalNavigator.main.goBack() }
                }
                centered.expanding.h2 { content = "Photo Sphere" }
            }

            if (frames.isEmpty()) {
                centered.expanding.text("No images to stitch.")
                return@col
            }

            expanding.scrolling.col {
                // ── Input frame grid ──────────────────────────────────────
                padded.scrolling.row {
                    frames.forEach { path ->
                        sizedBox(SizeConstraints(width = 4.rem, height = 4.rem)).image {
                            source = ImageRemote(if (path.startsWith("/")) "file://$path" else path)
                        }
                    }
                }

                card.padded.col {
                    h2 { content = "Stitch Info" }
                    text { ::content { "${frames.size} frames to stitch" } }
                }

                // ── Result Preview ────────────────────────────────────────
                sizedBox(SizeConstraints(minHeight = 20.rem)).frame {
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
                        centered.text("Press \"Stitch All\" to create panorama")
                    }
                }

                // ── Save Success ──────────────────────────────────────────
                shownWhen { saveSuccess() }.frame {
                    centered.card.padded.text("✓ Saved to gallery!")
                }

                // ── Stitch Button ─────────────────────────────────────────
                padded.important.button {
                    text { ::content { if (isProcessing()) "Stitching..." else "Stitch All" } }
                    ::enabled { !isProcessing() }
                    onClick { processInternal(fullSize = true) }
                }
            }

            // Auto-preview
            reactive {
                val f = frames.hashCode()
                if (frames.size >= 2) {
                    previewJob?.cancel()
                    previewJob = launch {
                        delay(600)
                        processInternal(fullSize = false)
                    }
                }
            }
        }
    }

    private suspend fun processInternal(fullSize: Boolean) {
        if (isProcessing.value) return
        isProcessing.value = true
        if (fullSize) saveSuccess.value = false

        try {
            val result = processPhotoSphere(
                frames,
                orientations = orientations,
                maxPreviewSize = if (fullSize) 0 else 1024
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
