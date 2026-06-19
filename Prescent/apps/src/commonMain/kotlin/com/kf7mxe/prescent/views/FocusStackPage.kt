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
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

class FocusStackPage(val frames: List<String> = listOf()) : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Focus Stack")

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

                card.padded.col {
                    h2 { content = "Processing Info" }
                    text { ::content { "${frames.size} frames captured at different focus distances" } }
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

            // Auto-preview on page load (debounced)
            reactive {
                val f = frames.hashCode()
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
            val result = processFocusStack(
                frames,
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
