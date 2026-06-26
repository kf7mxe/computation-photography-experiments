package com.kf7mxe.prescent.views

import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.GlobalNavigator
import com.kf7mxe.prescent.algorithmStore
import com.kf7mxe.prescent.alignmentStore
import com.kf7mxe.prescent.smartFrameSelectionStore
import com.kf7mxe.prescent.hotPixelFixStore
import com.kf7mxe.prescent.caCorrectionStore
import com.kf7mxe.prescent.lensCorrectionStore
import com.kf7mxe.prescent.smartNRStore
import com.kf7mxe.prescent.contrastSharpeningStore
import com.kf7mxe.prescent.jointDenoiseStore
import com.kf7mxe.prescent.dehazeStore
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
    init { println("HdrProcessingPage created with ${images.size} images: $images") }
    override val title: Reactive<String> get() = Constant("HDR Processing")

    val alignmentOption = Signal("MTB")
    val tonemapAlgorithm = Signal("Hybrid")
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

    // Hybrid / Contrast Optimizer shared
    val surrealAmount = Signal(0.5f)

    // Pyramid Fusion params
    val pyramidNoiseStrength = Signal(1.0f)

    // Pre/Post-processing toggles
    val smartFrameSelection = Signal(false)
    val enableHotPixelFix = Signal(false)
    val enableCACorrection = Signal(false)
    val enableLensCorrection = Signal(false)
    val enableSmartNR = Signal(false)
    val enableContrastSharpening = Signal(false)

    // Guided Fusion params
    val guidedFusionLevels = Signal(4.0f)
    val guidedFusionSigmaColor = Signal(30.0f)
    val guidedFusionSigmaSpace = Signal(30.0f)

    // Retinex params
    val retinexSigma = Signal(30.0f)
    val retinexCompression = Signal(0.5f)
    val retinexGamma = Signal(0.8f)

    // Saliency Fusion params
    val saliencyWeight = Signal(0.4f)

    // Pre/post toggles
    val enableJointDenoise = Signal(false)
    val enableDehaze = Signal(false)
    val dehazePatchSize = Signal(15.0f)
    val dehazeOmega = Signal(0.95f)
    val superResolutionScale = Signal(0.0f)
    val artisticEffect = Signal("None")
    val artisticOrtonBlurRadius = Signal(15.0f)
    val artisticOrtonOpacity = Signal(0.4f)
    val artisticMiniatureFocusY = Signal(0.5f)
    val artisticMiniatureBlurHeight = Signal(0.3f)
    val artisticBokehRadius = Signal(25.0f)

    val processing = Signal(false)
    val previewProcessing = Signal(false)
    val resultImagePath = Signal<String?>(null)
    val previewImagePath = Signal<String?>(null)
    val saveSuccess = Signal(false)

    private var previewJob: Job? = null
    private val mountTrigger = Signal(0)
    private var previewGuard = false

    private val algorithms = listOf("Hybrid", "Contrast Optimizer", "Durand", "CLAHE Boost", "Mertens", "Pyramid Fusion", "Guided Fusion", "Retinex", "Saliency Fusion", "Reinhard", "Drago", "Mantiuk", "Fattal", "iCam06")

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
                smartFrameSelection.value = smartFrameSelectionStore().toBoolean()
                enableHotPixelFix.value = hotPixelFixStore().toBoolean()
                enableCACorrection.value = caCorrectionStore().toBoolean()
                enableLensCorrection.value = lensCorrectionStore().toBoolean()
                enableSmartNR.value = smartNRStore().toBoolean()
                enableContrastSharpening.value = contrastSharpeningStore().toBoolean()
                enableJointDenoise.value = jointDenoiseStore().toBoolean()
                enableDehaze.value = dehazeStore().toBoolean()
                delay(200)
                mountTrigger.value = 1
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

                    // ── Lighting Adjustments (surreal↔natural) ──────────
                    shownWhen {
                        tonemapAlgorithm() == "Hybrid" ||
                        tonemapAlgorithm() == "Contrast Optimizer" ||
                        tonemapAlgorithm() == "Durand" ||
                        tonemapAlgorithm() == "CLAHE Boost"
                    }.col {
                        h2 { content = "Lighting Adjustments" }
                        col {
                            text { ::content {
                                val s = surrealAmount()
                                when {
                                    s < 0.25 -> "Natural"
                                    s < 0.45 -> "Balanced → Natural"
                                    s < 0.55 -> "Balanced"
                                    s < 0.75 -> "Surreal → Balanced"
                                    else -> "Surreal"
                                }
                            } }
                            slider { range(0.0f, 1.0f, 0.05f); value.bind(surrealAmount) }
                        }
                    }

                    // ── Per-algorithm settings ────────────────────────────

                    // Hybrid (Reinhard + unsharp mask on luminance)
                    shownWhen { tonemapAlgorithm() == "Hybrid" }.col {
                        h2 { content = "Hybrid: Base + Detail" }
                        subtext("Reinhard tone mapping with unsharp mask detail enhancement on luminance")
                        col {
                            text { ::content { "Gamma: ${gamma().fmt()}" } }
                            slider { range(0.1f, 5.0f, 0.1f); value.bind(gamma) }
                        }
                        col {
                            text { ::content { "Intensity: ${intensity().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(intensity) }
                        }
                        col {
                            text { ::content { "Detail Radius: ${fattalAlpha().fmt()}" } }
                            slider { range(0.01f, 0.5f, 0.01f); value.bind(fattalAlpha) }
                        }
                        col {
                            text { ::content { "Detail Strength: ${fattalBeta().fmt()}" } }
                            slider { range(0.1f, 1.0f, 0.05f); value.bind(fattalBeta) }
                        }
                        col {
                            text { ::content { "Color Saturation: ${fattalColorSaturation().fmt()}" } }
                            slider { range(0.0f, 1.0f, 0.05f); value.bind(fattalColorSaturation) }
                        }
                    }

                    // Contrast Optimizer (Mertens + unsharp mask on luminance)
                    shownWhen { tonemapAlgorithm() == "Contrast Optimizer" }.col {
                        h2 { content = "Contrast Optimizer" }
                        subtext("Clean Mertens fusion + unsharp mask detail on luminance only")
                        col {
                            text { ::content { "Contrast: ${contrastWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(contrastWeight) }
                        }
                        col {
                            text { ::content { "Saturation: ${saturationWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(saturationWeight) }
                        }
                        col {
                            text { ::content { "Detail Radius: ${fattalAlpha().fmt()}" } }
                            slider { range(0.01f, 0.5f, 0.01f); value.bind(fattalAlpha) }
                        }
                        col {
                            text { ::content { "Detail Strength: ${fattalBeta().fmt()}" } }
                            slider { range(0.1f, 1.0f, 0.05f); value.bind(fattalBeta) }
                        }
                    }

                    // Durand (Bilateral Filter)
                    shownWhen { tonemapAlgorithm() == "Durand" }.col {
                        h2 { content = "Durand Bilateral Filter" }
                        subtext("Edge-preserving base/detail in log-luminance. Compresses large contrast while retaining fine detail.")
                        col {
                            text { ::content { "Gamma: ${gamma().fmt()}" } }
                            slider { range(0.1f, 5.0f, 0.1f); value.bind(gamma) }
                        }
                        col {
                            text { ::content { "Radius: ${fattalAlpha().fmt()}" } }
                            slider { range(0.01f, 0.5f, 0.01f); value.bind(fattalAlpha) }
                        }
                        col {
                            text { ::content { "Saturation: ${saturationWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(saturationWeight) }
                        }
                    }

                    // CLAHE Boost
                    shownWhen { tonemapAlgorithm() == "CLAHE Boost" }.col {
                        h2 { content = "CLAHE Boost" }
                        subtext("Mertens fusion + CLAHE on luminance for natural local contrast")
                        col {
                            text { ::content { "Contrast: ${contrastWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(contrastWeight) }
                        }
                        col {
                            text { ::content { "Saturation: ${saturationWeight().fmt()}" } }
                            slider { range(0.0f, 2.0f, 0.1f); value.bind(saturationWeight) }
                        }
                    }
                    // Pyramid Fusion (Laplacian multi-scale)
                    shownWhen { tonemapAlgorithm() == "Pyramid Fusion" }.col {
                        h2 { content = "Pyramid Fusion" }
                        subtext("Multi-scale Laplacian pyramid merging with noise-aware weights. PhotonCamera-inspired.")
                        col {
                            text { ::content { "Noise Strength: ${pyramidNoiseStrength().fmt()}" } }
                            slider { range(0.0f, 3.0f, 0.1f); value.bind(pyramidNoiseStrength) }
                        }
                    }
                    // Guided Fusion (edge-aware multi-scale)
                    shownWhen { tonemapAlgorithm() == "Guided Fusion" }.col {
                        h2 { content = "Guided Fusion" }
                        subtext("Edge-aware multi-scale blending via bilateral filter. No halos on high-contrast edges.")
                        col {
                            text { ::content { "Levels: ${guidedFusionLevels().fmt()}" } }
                            slider { max = 6.0f; min = 2.0f; step = 1.0f; value.bind(guidedFusionLevels) }
                        }
                        col {
                            text { ::content { "Sigma Color: ${guidedFusionSigmaColor().fmt()}" } }
                            slider { max = 80.0f; min = 5.0f; step = 5.0f; value.bind(guidedFusionSigmaColor) }
                        }
                        col {
                            text { ::content { "Sigma Space: ${guidedFusionSigmaSpace().fmt()}" } }
                            slider { max = 80.0f; min = 5.0f; step = 5.0f; value.bind(guidedFusionSigmaSpace) }
                        }
                    }
                    // Retinex Tone Mapping
                    shownWhen { tonemapAlgorithm() == "Retinex" }.col {
                        h2 { content = "Retinex Tone Mapping" }
                        subtext("Decomposes into illumination × reflectance. Compresses lighting, preserves detail — very natural results.")
                        col {
                            text { ::content { "Sigma: ${retinexSigma().fmt()}" } }
                            slider { max = 80.0f; min = 5.0f; step = 1.0f; value.bind(retinexSigma) }
                        }
                        col {
                            text { ::content { "Compression: ${retinexCompression().fmt()}" } }
                            slider { range(0.1f, 1.0f, 0.05f); value.bind(retinexCompression) }
                        }
                        col {
                            text { ::content { "Gamma: ${retinexGamma().fmt()}" } }
                            slider { range(0.3f, 1.5f, 0.05f); value.bind(retinexGamma) }
                        }
                    }
                    // Saliency-Weighted Fusion
                    shownWhen { tonemapAlgorithm() == "Saliency Fusion" }.col {
                        h2 { content = "Saliency Fusion" }
                        subtext("Weights each pixel by visual saliency. Keeps sky well-exposed while preserving attention-grabbing foreground detail.")
                        col {
                            text { ::content { "Saliency Weight: ${saliencyWeight().fmt()}" } }
                            slider { range(0.0f, 1.0f, 0.05f); value.bind(saliencyWeight) }
                        }
                    }
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

                // ── Enhancements (pre/post processing) ──────────────────────
                card.padded.col {
                    h2 { content = "Enhancements" }
                    subtext("Pre-process each frame before alignment and post-process the final result")
                    col {
                        row {
                            toggleButton {
                                text("Smart Frame Selection"); checked bind smartFrameSelection
                            }
                        }
                        subtext("Drop blurry frames before processing")
                    }
                    col {
                        row {
                            toggleButton {
                                text("Hot Pixel Fix"); checked bind enableHotPixelFix
                            }
                            toggleButton {
                                text("CA Correction"); checked bind enableCACorrection
                            }
                        }
                        row {
                            toggleButton {
                                text("Lens Correction"); checked bind enableLensCorrection
                            }
                            toggleButton {
                                text("Smart NR"); checked bind enableSmartNR
                            }
                        }
                        row {
                            toggleButton {
                                text("Contrast Sharpening"); checked bind enableContrastSharpening
                            }
                        }
                    }
                }

                // ── Effects (dehaze, super-res, artistic) ────────────────────
                card.padded.col {
                    h2 { content = "Effects" }
                    subtext("Post-processing filters applied to the final result")
                    col {
                        row { toggleButton { text("Dehaze"); checked bind enableDehaze } }
                        shownWhen { enableDehaze() }.col {
                            subtext("Dark channel prior — removes haze from landscapes")
                            col {
                                text { ::content { "Patch Size: ${dehazePatchSize().toInt()}" } }
                                slider { max = 31.0f; min = 7.0f; step = 2.0f; value.bind(dehazePatchSize) }
                            }
                            col {
                                text { ::content { "Strength: ${dehazeOmega().fmt()}" } }
                                slider { range(0.5f, 1.0f, 0.05f); value.bind(dehazeOmega) }
                            }
                        }
                    }
                    col {
                        text { ::content { "Super Resolution: ${if (superResolutionScale() < 1.5f) "Off" else "${superResolutionScale().toInt()}x"}" } }
                        slider { range(0.0f, 4.0f, 1.0f); value.bind(superResolutionScale) }
                        subtext("0 = off, 2-4 = upscale factor. 2x recommended for best quality.")
                    }
                    col {
                        row { toggleButton { text("Joint Denoise"); checked bind enableJointDenoise } }
                        shownWhen { enableJointDenoise() }.col {
                            subtext("Uses well-exposed frame as guide to denoise dark brackets")
                        }
                    }
                    col {
                        h3 { content = "Artistic Effect" }
                        row {
                            listOf("None", "Orton", "Miniature", "Bokeh").forEach { opt ->
                                expanding.toggleButton {
                                    text(opt); checked bind artisticEffect.equalTo(opt)
                                }
                            }
                        }
                        shownWhen { artisticEffect() == "Orton" }.col {
                            col {
                                text { ::content { "Blur Radius: ${artisticOrtonBlurRadius().toInt()}" } }
                                slider { max = 41.0f; min = 3.0f; step = 2.0f; value.bind(artisticOrtonBlurRadius) }
                            }
                            col {
                                text { ::content { "Opacity: ${artisticOrtonOpacity().fmt()}" } }
                                slider { range(0.1f, 1.0f, 0.05f); value.bind(artisticOrtonOpacity) }
                            }
                        }
                        shownWhen { artisticEffect() == "Miniature" }.col {
                            col {
                                text { ::content { "Focus Y: ${artisticMiniatureFocusY().fmt()}" } }
                                slider { range(0.0f, 1.0f, 0.05f); value.bind(artisticMiniatureFocusY) }
                            }
                            col {
                                text { ::content { "Blur Height: ${artisticMiniatureBlurHeight().fmt()}" } }
                                slider { range(0.1f, 0.8f, 0.05f); value.bind(artisticMiniatureBlurHeight) }
                            }
                        }
                        shownWhen { artisticEffect() == "Bokeh" }.col {
                            col {
                                text { ::content { "Blur Radius: ${artisticBokehRadius().toInt()}" } }
                                slider { max = 71.0f; min = 3.0f; step = 2.0f; value.bind(artisticBokehRadius) }
                            }
                        }
                    }
                }

                // ── Randomize Button ─────────────────────────────────────────
                padded.important.button {
                    text("Surprise Me!")
                    onClick { randomizeSettings() }
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
                smartFrameSelection()
                enableHotPixelFix()
                enableCACorrection()
                enableLensCorrection()
                enableSmartNR()
                enableContrastSharpening()
                enableJointDenoise()
                enableDehaze()

                smartFrameSelectionStore.value = smartFrameSelection.value.toString()
                hotPixelFixStore.value = enableHotPixelFix.value.toString()
                caCorrectionStore.value = enableCACorrection.value.toString()
                lensCorrectionStore.value = enableLensCorrection.value.toString()
                smartNRStore.value = enableSmartNR.value.toString()
                contrastSharpeningStore.value = enableContrastSharpening.value.toString()
                jointDenoiseStore.value = enableJointDenoise.value.toString()
                dehazeStore.value = enableDehaze.value.toString()
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
                surrealAmount()
                pyramidNoiseStrength()
                smartFrameSelection()
                enableHotPixelFix()
                enableCACorrection()
                enableLensCorrection()
                enableSmartNR()
                enableContrastSharpening()
                guidedFusionLevels()
                guidedFusionSigmaColor()
                guidedFusionSigmaSpace()
                retinexSigma()
                retinexCompression()
                retinexGamma()
                saliencyWeight()
                enableJointDenoise()
                enableDehaze()
                dehazePatchSize()
                dehazeOmega()
                superResolutionScale()
                artisticEffect()
                artisticOrtonBlurRadius()
                artisticOrtonOpacity()
                artisticMiniatureFocusY()
                artisticMiniatureBlurHeight()
                artisticBokehRadius()
                mountTrigger()

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
            if (previewGuard) return
            previewGuard = true
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
                surrealAmount = surrealAmount.value,
                maxSize = if (fullSize) 0 else 1024,
                pyramidNoiseStrength = pyramidNoiseStrength.value,
                enableHotPixelFix = enableHotPixelFix.value,
                enableCACorrection = enableCACorrection.value,
                enableLensCorrection = enableLensCorrection.value,
                enableSmartNR = enableSmartNR.value,
                enableContrastSharpening = enableContrastSharpening.value,
                smartFrameSelection = smartFrameSelection.value,
                guidedFusionLevels = guidedFusionLevels.value.toInt(),
                guidedFusionSigmaColor = guidedFusionSigmaColor.value,
                guidedFusionSigmaSpace = guidedFusionSigmaSpace.value,
                retinexSigma = retinexSigma.value,
                retinexCompression = retinexCompression.value,
                retinexGamma = retinexGamma.value,
                saliencyWeight = saliencyWeight.value,
                enableJointDenoise = enableJointDenoise.value,
                enableDehaze = enableDehaze.value,
                dehazePatchSize = dehazePatchSize.value.toInt(),
                dehazeOmega = dehazeOmega.value,
                superResolutionScale = superResolutionScale.value.toInt(),
                artisticEffect = artisticEffect.value,
                artisticOrtonBlurRadius = artisticOrtonBlurRadius.value.toInt(),
                artisticOrtonOpacity = artisticOrtonOpacity.value,
                artisticMiniatureFocusY = artisticMiniatureFocusY.value,
                artisticMiniatureBlurHeight = artisticMiniatureBlurHeight.value,
                artisticBokehRadius = artisticBokehRadius.value.toInt()
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
            else { previewProcessing.value = false; previewGuard = false }
        }
    }

    private fun randomizeSettings() {
        val rng = kotlin.random.Random
        val hybridAlgorithms = listOf("Hybrid", "Contrast Optimizer", "Durand", "CLAHE Boost", "Pyramid Fusion", "Guided Fusion", "Retinex", "Saliency Fusion")
        val singleAlgorithms = listOf("Mertens", "Reinhard", "Drago", "Mantiuk", "Fattal", "iCam06")

        tonemapAlgorithm.value = if (rng.nextFloat() < 0.5f) {
            hybridAlgorithms[rng.nextInt(hybridAlgorithms.size)]
        } else {
            singleAlgorithms[rng.nextInt(singleAlgorithms.size)]
        }

        alignmentOption.value = listOf("MTB", "ECC", "Feature", "Skip")[rng.nextInt(4)]
        ghostingStrength.value = rng.nextFloat() * 0.8f
        cropAfterAlignment.value = rng.nextBoolean()

        contrastWeight.value = 0.5f + rng.nextFloat() * 1.5f
        saturationWeight.value = 0.5f + rng.nextFloat() * 1.5f
        exposureWeight.value = rng.nextFloat() * 1.5f
        gamma.value = 0.5f + rng.nextFloat() * 3.0f
        intensity.value = rng.nextFloat() * 1.5f
        lightAdaptation.value = rng.nextFloat() * 1.5f
        colorAdaptation.value = rng.nextFloat() * 1.5f
        dragoBias.value = 0.3f + rng.nextFloat() * 1.2f
        mantiukScale.value = 0.2f + rng.nextFloat() * 1.5f
        fattalAlpha.value = 0.02f + rng.nextFloat() * 0.4f
        fattalBeta.value = 0.2f + rng.nextFloat() * 0.7f
        fattalColorSaturation.value = rng.nextFloat() * 0.8f
        icam06ChromaticAdaptation.value = 0.5f + rng.nextFloat() * 1.5f
        icam06LocalAdaptation.value = 0.5f + rng.nextFloat() * 4.0f
        surrealAmount.value = rng.nextFloat()
        pyramidNoiseStrength.value = 0.5f + rng.nextFloat() * 2.5f
        smartFrameSelection.value = rng.nextBoolean()
        enableHotPixelFix.value = rng.nextBoolean()
        enableCACorrection.value = rng.nextBoolean()
        enableLensCorrection.value = rng.nextBoolean()
        enableSmartNR.value = rng.nextBoolean()
        enableContrastSharpening.value = rng.nextBoolean()
        guidedFusionLevels.value = (2 + rng.nextInt(4)).toFloat()
        guidedFusionSigmaColor.value = 10f + rng.nextFloat() * 50f
        guidedFusionSigmaSpace.value = 10f + rng.nextFloat() * 50f
        retinexSigma.value = 5f + rng.nextFloat() * 50f
        retinexCompression.value = 0.2f + rng.nextFloat() * 0.6f
        retinexGamma.value = 0.5f + rng.nextFloat() * 0.8f
        saliencyWeight.value = rng.nextFloat() * 0.8f
        enableJointDenoise.value = rng.nextBoolean()
        enableDehaze.value = rng.nextBoolean()
        dehazePatchSize.value = (7 + rng.nextInt(20) * 2).toFloat()
        dehazeOmega.value = 0.7f + rng.nextFloat() * 0.25f
        superResolutionScale.value = if (rng.nextBoolean()) 0f else (2 + rng.nextInt(2)).toFloat()
        artisticEffect.value = listOf("None", "Orton", "Miniature", "Bokeh")[rng.nextInt(4)]
        artisticOrtonBlurRadius.value = (5 + rng.nextInt(30) * 2).toFloat()
        artisticOrtonOpacity.value = 0.2f + rng.nextFloat() * 0.6f
        artisticMiniatureFocusY.value = 0.2f + rng.nextFloat() * 0.6f
        artisticMiniatureBlurHeight.value = 0.2f + rng.nextFloat() * 0.4f
        artisticBokehRadius.value = (5 + rng.nextInt(30) * 2).toFloat()
    }
}
