package com.kf7mxe.prescent.views

import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.GlobalNavigator
import com.kf7mxe.prescent.algorithmStore
import com.kf7mxe.prescent.alignmentStore
import com.lightningkite.kiteui.*
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.context.invoke
import com.lightningkite.reactive.context.reactive
import com.lightningkite.reactive.context.await
import com.lightningkite.reactive.core.Constant
import com.lightningkite.reactive.core.Reactive
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.extensions.equalTo
import com.kf7mxe.prescent.utils.fmt
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Routable("/hdr-processing")
class HdrProcessingPage(val images: List<String> = listOf()) : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("HDR Processing")

    val alignmentOption = Signal("MTB")
    val tonemapAlgorithm = Signal("Mertens")
    val ghostingStrength = Signal(0.0f)
    val cropAfterAlignment = Signal(false)

    // Mertens parameters
    val contrastWeight = Signal(1.0f)
    val saturationWeight = Signal(1.0f)
    val exposureWeight = Signal(0.0f)

    // Reinhard parameters
    val gamma = Signal(1.0f)
    val intensity = Signal(0.0f)
    val lightAdaptation = Signal(0.0f)
    val colorAdaptation = Signal(0.0f)

    // Drago parameters
    val dragoBias = Signal(0.85f)

    // Mantiuk parameters
    val mantiukScale = Signal(0.75f)

    // Fattal parameters
    val fattalAlpha = Signal(0.1f)
    val fattalBeta = Signal(0.9f)
    val fattalColorSaturation = Signal(0.5f)

    // iCam06 parameters
    val icam06ChromaticAdaptation = Signal(1.0f)
    val icam06LocalAdaptation = Signal(1.0f)

    val processing = Signal(false)
    val previewProcessing = Signal(false)
    val resultImagePath = Signal<String?>(null)
    val previewImagePath = Signal<String?>(null)
    val saveSuccess = Signal(false)

    private var previewJob: Job? = null

    private val algorithms = listOf("Mertens", "Reinhard", "Drago", "Mantiuk", "Fattal", "iCam06")

    override fun ElementWriter.CanAddTheme.render() {
        col {
            // ── Top Bar ───────────────────────────────────────────────────
            padded.row {
                button {
                    icon(Icon.arrowBack, "Back")
                    onClick { GlobalNavigator.main.goBack() }
                }
                centered.expanding.h2 { content = "HDR Processing" }
            }

            if (images.isEmpty()) {
                centered.expanding.text("No images selected. Go back and choose images.")
                return@col
            }

            load {
                algorithmStore().takeIf { it.isNotBlank() }?.let { tonemapAlgorithm.value = it }
                alignmentStore().takeIf { it.isNotBlank() }?.let { alignmentOption.value = it }
            }

            expanding.scrolling.col {

                // ── Input image strip ─────────────────────────────────────
                padded.row {
                    images.forEach { path ->
                        // Use the raw path as-is; content:// URIs are fine here
                        sizedBox(SizeConstraints(width = 5.rem, height = 5.rem)).image {
                            source = ImageRemote(
                                if (path.startsWith("/")) "file://$path" else path
                            )
                        }
                    }
                }

                // ── Algorithm Picker ──────────────────────────────────────
                card.padded.col {
                    h2 { content = "Algorithm" }

                    row {
                        algorithms.forEach { alg ->
                            expanding.toggleButton {
                                text(alg)
                                checked bind tonemapAlgorithm.equalTo(alg)
                            }
                        }
                    }

                    // ── Per-algorithm settings ────────────────────────────
                    // Mertens (Exposure Fusion)
                    shownWhen { tonemapAlgorithm() == "Mertens" }.col {
                        h2 { content = "Exposure Fusion Settings" }
                        col {
                            text { ::content { "Contrast: ${contrastWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(contrastWeight) }
                        }
                        col {
                            text { ::content { "Saturation: ${saturationWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(saturationWeight) }
                        }
                        col {
                            text { ::content { "Exposure: ${exposureWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(exposureWeight) }
                        }
                    }

                    // Reinhard
                    shownWhen { tonemapAlgorithm() == "Reinhard" }.col {
                        h2 { content = "Reinhard Tone Mapping" }
                        col {
                            text { ::content { "Gamma: ${gamma().fmt()}" } }
                            slider { range(0.1f, 5.0f, 0.1f); value.bind(gamma) }
                        }
                        col {
                            text { ::content { "Intensity: ${intensity().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(intensity) }
                        }
                        col {
                            text { ::content { "Light Adaptation: ${lightAdaptation().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(lightAdaptation) }
                        }
                        col {
                            text { ::content { "Color Adaptation: ${colorAdaptation().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(colorAdaptation) }
                        }
                    }

                    // Drago
                    shownWhen { tonemapAlgorithm() == "Drago" }.col {
                        h2 { content = "Drago Tone Mapping" }
                        col {
                            text { ::content { "Gamma: ${gamma().fmt()}" } }
                            slider { range(0.1f, 5.0f, 0.1f); value.bind(gamma) }
                        }
                        col {
                            text { ::content { "Saturation: ${saturationWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(saturationWeight) }
                        }
                        col {
                            text { ::content { "Bias: ${dragoBias().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(dragoBias) }
                        }
                    }

                    // Mantiuk
                    shownWhen { tonemapAlgorithm() == "Mantiuk" }.col {
                        h2 { content = "Mantiuk Tone Mapping" }
                        col {
                            text { ::content { "Gamma: ${gamma().fmt()}" } }
                            slider { range(0.1f, 5.0f, 0.1f); value.bind(gamma) }
                        }
                        col {
                            text { ::content { "Saturation: ${saturationWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(saturationWeight) }
                        }
                        col {
                            text { ::content { "Scale: ${mantiukScale().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(mantiukScale) }
                        }
                    }

                    // Fattal Gradient Domain
                    shownWhen { tonemapAlgorithm() == "Fattal" }.col {
                        h2 { content = "Fattal Gradient Domain" }
                        col {
                            text { ::content { "Alpha (threshold): ${fattalAlpha().fmt()}" } }
                            slider { range(0.01f, 0.5f, 0.01f); value.bind(fattalAlpha) }
                        }
                        col {
                            text { ::content { "Beta (attenuation): ${fattalBeta().fmt()}" } }
                            slider { range(0.1f, 1.0f, 0.05f); value.bind(fattalBeta) }
                        }
                        col {
                            text { ::content { "Color Saturation: ${fattalColorSaturation().fmt()}" } }
                            slider { range(0.0f, 1.0f, 0.05f); value.bind(fattalColorSaturation) }
                        }
                    }

                    // iCam06 Perceptual
                    shownWhen { tonemapAlgorithm() == "iCam06" }.col {
                        h2 { content = "iCam06 Perceptual Model" }
                        col {
                            text { ::content { "Chromatic Adaptation: ${icam06ChromaticAdaptation().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(icam06ChromaticAdaptation) }
                        }
                        col {
                            text { ::content { "Local Adaptation: ${icam06LocalAdaptation().fmt()}" } }
                            slider { range(0.1f, 5.0f, 0.1f); value.bind(icam06LocalAdaptation) }
                        }
                        col {
                            text { ::content { "Color Sat: ${saturationWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(saturationWeight) }
                        }
                    }
                }

                // ── Alignment Settings ────────────────────────────────────
                card.padded.col {
                    h2 { content = "Alignment" }
                    row {
                        listOf("MTB", "ECC", "Feature", "Skip").forEach { opt ->
                            expanding.toggleButton {
                                text(opt)
                                checked bind alignmentOption.equalTo(opt)
                            }
                        }
                    }
                    col {
                        text { ::content { "Ghosting Removal: ${ghostingStrength().fmt()}" } }
                        slider { range(0.0f, 1.0f, 0.05f); value.bind(ghostingStrength) }
                    }
                    row {
                        text("Crop After Alignment")
                        switch { checked bind cropAfterAlignment }
                    }
                }

                // ── Preview & Result area ─────────────────────────────────
                sizedBox(SizeConstraints(minHeight = 25.rem)).frame {
                    image {
                        reactive {
                            val full = resultImagePath()
                            val prev = previewImagePath()
                            when {
                                full != null -> {
                                    source = ImageRemote(if (full.startsWith("/")) "file://$full" else full)
                                    visible = true
                                    opacity = 1.0
                                }
                                prev != null -> {
                                    source = ImageRemote(if (prev.startsWith("/")) "file://$prev" else prev)
                                    visible = true
                                    opacity = 0.7
                                }
                                else -> visible = false
                            }
                        }
                    }

                    shownWhen { processing() || (previewProcessing() && previewImagePath() == null) }.frame {
                        atBottomCenter.activityIndicator()
                    }

                    shownWhen {
                        !processing() && !previewProcessing()
                            && previewImagePath() == null && resultImagePath() == null
                    }.frame {
                        centered.text("Adjust settings to preview")
                    }

                    // Processing/preview label overlay
                    shownWhen { previewProcessing() && previewImagePath() != null }.frame {
                        atCenterEnd.padded.text("Updating preview…")
                    }
                }

                // ── Save Success Banner ───────────────────────────────────
                shownWhen { saveSuccess() }.frame {
                    centered.card.padded.text("✓ Saved to gallery!")
                }

                // ── Process Button ────────────────────────────────────────
                padded.important.button {
                    text { ::content { if (processing()) "Processing…" else "Process & Save Full Image" } }
                    ::enabled { !processing() }
                    onClick { processHdrInternal(fullSize = true) }
                }
            }

            reactive {
                tonemapAlgorithm()
                alignmentOption()
                contrastWeight()
                saturationWeight()
                exposureWeight()
                gamma()
                intensity()
                lightAdaptation()
                colorAdaptation()
                dragoBias()
                mantiukScale()
                ghostingStrength()
                cropAfterAlignment()
                fattalAlpha()
                fattalBeta()
                fattalColorSaturation()
                icam06ChromaticAdaptation()
                icam06LocalAdaptation()

                previewJob?.cancel()
                previewJob = launch {
                    delay(600)
                    processHdrInternal(fullSize = false)
                }
            }
        }
    }

    private suspend fun processHdrInternal(fullSize: Boolean) {
        if (fullSize) {
            if (processing.value) return
            processing.value = true
            saveSuccess.value = false
        } else {
            previewProcessing.value = true
        }

        try {
            val result = processHdr(
                images,
                tonemapAlgorithm.value,
                alignmentOption.value,
                contrastWeight.value,
                saturationWeight.value,
                exposureWeight.value,
                gamma = gamma.value,
                intensity = intensity.value,
                lightAdaptation = lightAdaptation.value,
                colorAdaptation = colorAdaptation.value,
                dragoBias = dragoBias.value,
                mantiukScale = mantiukScale.value,
                ghostingStrength = ghostingStrength.value,
                cropAfterAlignment = cropAfterAlignment.value,
                fattalAlpha = fattalAlpha.value,
                fattalBeta = fattalBeta.value,
                fattalColorSaturation = fattalColorSaturation.value,
                icam06ChromaticAdaptation = icam06ChromaticAdaptation.value,
                icam06LocalAdaptation = icam06LocalAdaptation.value,
                maxSize = if (fullSize) 0 else 1024
            )
            if (fullSize) {
                resultImagePath.value = result
                if (result != null) saveSuccess.value = true
            } else {
                previewImagePath.value = result
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            if (fullSize) processing.value = false
            else previewProcessing.value = false
        }
    }
}
