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

class NightSightPage(val frames: List<String> = listOf()) : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Night Sight")

    val nsAlgorithm = Signal(NightSightAlgorithm.AVERAGE)
    val useLucky = Signal(false)
    val luckyFraction = Signal(0.6f)
    val starTrail = Signal(false)
    val brightnessBoost = Signal(1.5f)
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
                centered.expanding.h2 { content = "Night Sight" }
            }

            if (frames.isEmpty()) {
                centered.expanding.text("No frames captured. Go back and capture in Night mode.")
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

                // ── Algorithm Selection ──────────────────────────────────
                card.padded.col {
                    h2 { content = "Algorithm" }
                    scrolling.row {
                        NightSightAlgorithm.entries.forEach { algo ->
                            toggleButton {
                                text(algo.label)
                                checked bind nsAlgorithm.equalTo(algo)
                            }
                        }
                    }
                    text { ::content { nsAlgorithm().description } }
                }

                // ── Lucky Pre-filter ─────────────────────────────────────
                reactive {
                    val algo = nsAlgorithm()
                    if (algo.supportsLuckyPreFilter) {
                        card.padded.col {
                            h2 { content = "Combine: Lucky Frame Selection" }
                            text("Selects sharpest frames before stacking — removes motion blur")
                            row {
                                text("Enable")
                                switch { checked bind useLucky }
                            }
                            shownWhen { useLucky() }.col {
                                text { ::content { "Keep sharpest ${(luckyFraction() * 100).toInt()}% of frames" } }
                                slider { range(0.2f, 1.0f, 0.05f); value.bind(luckyFraction) }
                            }
                        }
                    } else {
                        useLucky.value = false
                    }
                }

                // ── Stack Settings ──────────────────────────────────────
                card.padded.col {
                    h2 { content = "Stack Settings" }

                    col {
                        text { ::content { "Brightness Boost: ${brightnessBoost().fmt()}x" } }
                        slider { range(0.5f, 4.0f, 0.25f); value.bind(brightnessBoost) }
                    }

                    reactive {
                        val algo = nsAlgorithm()
                        shownWhen { algo == NightSightAlgorithm.AVERAGE }.row {
                            text("Star Trail Mode")
                            switch { checked bind starTrail }
                        }
                        if (algo != NightSightAlgorithm.AVERAGE) starTrail.value = false
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
                        centered.text("Adjust settings for preview")
                    }

                    shownWhen { isProcessing() && previewPath() != null }.frame {
                        atCenterEnd.padded.text("Updating...")
                    }
                }

                // ── Save Success ──────────────────────────────────────────
                shownWhen { saveSuccess() }.frame {
                    centered.card.padded.text("✓ Saved to gallery!")
                }

                // ── Process Full Button ───────────────────────────────────
                padded.important.button {
                    text { ::content { if (isProcessing()) "Processing..." else "Process Full Image" } }
                    ::enabled { !isProcessing() }
                    onClick { processInternal(fullSize = true) }
                }
            }

            // Auto-preview on settings change
            reactive {
                nsAlgorithm()
                useLucky()
                luckyFraction()
                starTrail()
                brightnessBoost()

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
            val result = processNightSight(
                frames,
                algorithm = nsAlgorithm.value,
                useLuckyPreFilter = useLucky.value && nsAlgorithm.value.supportsLuckyPreFilter,
                luckyKeepFraction = luckyFraction.value,
                starTrail = starTrail.value,
                darkFramePath = null,
                brightnessBoost = brightnessBoost.value,
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
