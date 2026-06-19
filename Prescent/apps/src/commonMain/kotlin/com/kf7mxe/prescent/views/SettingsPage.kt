package com.kf7mxe.prescent.views

import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.GlobalNavigator
import com.kf7mxe.prescent.bracketCountStore
import com.kf7mxe.prescent.evOffsetStore
import com.kf7mxe.prescent.algorithmStore
import com.kf7mxe.prescent.alignmentStore
import com.kf7mxe.prescent.utils.fmt
import com.lightningkite.kiteui.*
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.reactive.core.Constant
import com.lightningkite.reactive.core.Reactive
import com.lightningkite.reactive.extensions.equalTo

@Routable("/settings")
class SettingsPage : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Settings")

    val defaultBracketCount = bracketCountStore.lens(
        get = { it.toIntOrNull() ?: 3 },
        modify = { _, v -> v.toString() }
    )
    val defaultEvOffset = evOffsetStore.lens(
        get = { it.toFloatOrNull() ?: 2.0f },
        modify = { _, v -> v.toString() }
    )
    val defaultAlgorithm = algorithmStore.lens(
        get = { it },
        modify = { _, v -> v }
    )
    val defaultAlignment = alignmentStore.lens(
        get = { it },
        modify = { _, v -> v }
    )

    private val algorithms = listOf("Mertens", "Reinhard", "Drago", "Mantiuk")

    override fun ElementWriter.CanAddTheme.render() {
        col {
            padded.row {
                button {
                    icon(Icon.arrowBack, "Back")
                    onClick { GlobalNavigator.main.goBack() }
                }
                centered.expanding.h2 { content = "Settings" }
            }

            expanding.scrolling.col {
                card.padded.col {
                    h2 { content = "Default Capture Settings" }

                    col {
                        text { ::content { "Default Bracket Count: ${defaultBracketCount()}" } }
                        slider {
                            range(2f, 9f, 1f)
                            value bind defaultBracketCount.lens(
                                get = { it.toFloat() },
                                modify = { _, v -> v.toInt() }
                            )
                        }
                    }

                    col {
                        text { ::content { "Default EV Offset: ±${defaultEvOffset()}" } }
                        slider {
                            range(0.5f, 4.0f, 0.5f)
                            value bind defaultEvOffset
                        }
                    }
                }

                card.padded.col {
                    h2 { content = "Default HDR Algorithm" }
                    algorithms.forEach { alg ->
                        toggleButton {
                            text(alg)
                            checked bind defaultAlgorithm.equalTo(alg)
                        }
                    }
                }

                card.padded.col {
                    h2 { content = "Default Alignment" }
                    row {
                        listOf("MTB", "Skip").forEach { opt ->
                            expanding.toggleButton {
                                text(opt)
                                checked bind defaultAlignment.equalTo(opt)
                            }
                        }
                    }
                }

                card.padded.col {
                    h2 { content = "About" }
                    text { ::content { "Prescent — Computational Photography" } }
                    subtext { content = "HDR, Night Sight, Focus Stacking & more" }
                }
            }
        }
    }
}
