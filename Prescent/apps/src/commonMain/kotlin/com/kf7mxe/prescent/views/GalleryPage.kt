package com.kf7mxe.prescent.views

import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.GlobalNavigator
import com.kf7mxe.prescent.now
import com.lightningkite.kiteui.*
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.reactive.PersistentProperty
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.kiteui.views.l2.*
import com.lightningkite.reactive.context.invoke
import com.lightningkite.reactive.core.Constant
import com.lightningkite.reactive.core.Reactive
import com.lightningkite.reactive.core.Signal
import com.lightningkite.reactive.context.reactive
import com.lightningkite.reactive.extensions.equalTo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json

@Routable("/gallery")
class GalleryPage : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Gallery")

    data class BracketSet(val folderName: String, val paths: List<String>, val timestamp: Long)

    val bracketSets: Signal<List<BracketSet>> = Signal(emptyList())

    private val favoriteFoldersStore = PersistentProperty("galleryFavorites", "[]")
    val favoriteFolders: Signal<Set<String>> = Signal(emptySet())

    val filterMode = Signal("all")
    val filteredSets = Signal(emptyList<BracketSet>())

    private val oneWeekMs = 7L * 24 * 60 * 60 * 1000L

    override fun ElementWriter.CanAddTheme.render() {
        col {
            // ── Top Bar ────────────────────────────────────────────────────
            padded.row {
                button {
                    icon(Icon.arrowBack, "Back")
                    onClick { GlobalNavigator.main.goBack() }
                }
                centered.expanding.h2 { content = "Gallery" }
            }

            loadBracketSets()

            // ── Compute filtered sets reactively ───────────────────────────
            reactive {
                val mode = filterMode()
                val favs = favoriteFolders()
                val all = bracketSets()
                val nowMs = now().toEpochMilliseconds()
                filteredSets.value = when (mode) {
                    "favorites" -> all.filter { it.folderName in favs }
                    "recent" -> all.filter { it.timestamp > 0 && nowMs - it.timestamp < oneWeekMs }
                    else -> all
                }
            }

            // ── Filter Bar ──────────────────────────────────────────────────
            padded.row {
                listOf("all" to "All", "favorites" to "Favorites", "recent" to "Recent").forEach { (value, label) ->
                    expanding.toggleButton {
                        text(label)
                        checked bind filterMode.equalTo(value)
                    }
                }
            }

            // ── Empty States ───────────────────────────────────────────────
            shownWhen { bracketSets().isEmpty() }.frame {
                centered.col {
                    text("No bracketed sets found.")
                    text("Capture images in HDR mode or import them via the camera screen.")
                }
            }

            shownWhen { bracketSets().isNotEmpty() && filteredSets().isEmpty() }.frame {
                centered.col {
                    text("No sets match the current filter.")
                }
            }

            // ── Bracketed Sets List ────────────────────────────────────────
            shownWhen { filteredSets().isNotEmpty() }.frame {
                expanding.scrolling.col {
                    forEach(filteredSets) { bracketSet ->
                        val folderName = bracketSet.folderName
                        card.padded.col {
                            // Thumbnail row
                            scrolling.row {
                                bracketSet.paths.forEach { path ->
                                    sizedBox(SizeConstraints(width = 4.rem, height = 4.rem)).button {
                                        image {
                                            source = ImageRemote(
                                                if (path.startsWith("/")) "file://$path" else path
                                            )
                                        }
                                        onClick {
                                            dialog { close ->
                                                frame {
                                                    image {
                                                        source = ImageRemote(
                                                            if (path.startsWith("/")) "file://$path" else path
                                                        )
                                                    }
                                                    atTopStart.button {
                                                        icon(Icon.close, "Close")
                                                        onClick { close() }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            row {
                                expanding.col {
                                    text { ::content { bracketSet.folderName } }
                                    text { ::content { "${bracketSet.paths.size} image(s)" } }
                                }

                                button {
                                    text {
                                        ::content {
                                            if (folderName in favoriteFolders()) "★" else "☆"
                                        }
                                    }
                                    onClick {
                                        val current = favoriteFolders.value
                                        favoriteFolders.value = if (folderName in favoriteFolders()) {
                                            current - folderName
                                        } else {
                                            current + folderName
                                        }
                                        persistFavorites()
                                    }
                                }

                                val paths = bracketSet.paths
                                important.button {
                                    text("Open in HDR")
                                    onClick {
                                        GlobalNavigator.main.navigate(HdrProcessingPage(paths))
                                    }
                                }

                                button {
                                    icon(Icon.delete, "Delete")
                                    onClick {
                                        dialog { close ->
                                            card.padded.col {
                                                h2 { content = "Delete Bracket Set" }
                                                text("Delete ${bracketSet.folderName} and all ${bracketSet.paths.size} images?")
                                                row {
                                                    expanding.button {
                                                        text("Cancel")
                                                        onClick { close() }
                                                    }
                                                    expanding.important.button {
                                                        text("Delete")
                                                        onClick {
                                                            deleteBracketSet(bracketSet)
                                                            close()
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun persistFavorites() {
        favoriteFoldersStore.value = Json.encodeToString(favoriteFolders.value.toList())
    }

    private fun ElementWriter.loadBracketSets() {
        load {
            loadFavorites()
            val sets = loadBracketSetsFromDisk()
            bracketSets.value = sets
        }
    }

    private suspend fun loadFavorites() {
        try {
            val raw = favoriteFoldersStore()
            if (!raw.isNullOrBlank() && raw != "[]") {
                favoriteFolders.value = Json.decodeFromString<List<String>>(raw).toSet()
            }
        } catch (e: Exception) { }
    }

    private fun deleteBracketSet(bracketSet: BracketSet) {
        deleteBracketSetFromDisk(bracketSet)
        bracketSets.value = bracketSets.value.filter { it.folderName != bracketSet.folderName }
    }
}

expect fun loadBracketSetsFromDisk(): List<GalleryPage.BracketSet>
expect fun deleteBracketSetFromDisk(bracketSet: GalleryPage.BracketSet)
