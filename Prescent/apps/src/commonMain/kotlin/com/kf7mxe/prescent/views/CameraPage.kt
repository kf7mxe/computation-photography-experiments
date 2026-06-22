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
import kotlin.math.abs

data class SphereGhostFrame(val azimuth: Float, val pitch: Float, val path: String)

class CameraPage : Page, FullscreenPage {

    val bracketCount = Signal(3)
    val evOffset = Signal(2.0f)
    val shutterTrigger = Signal(0)
    val isHdrMode = Signal(true)
    val cameraLens = Signal(0)
    val cameraLabels = Signal(listOf(0 to "Back", 1 to "Front"))

    val isNightSight = Signal(false)
    val nightSightFrameCount = Signal(8)
    val nightSightCaptureTrigger = Signal(0)

    val isFocusStacking = Signal(false)
    val focusStackFrameCount = Signal(6)
    val focusStackCaptureTrigger = Signal(0)

    val isSpatial = Signal(false)
    val spatialCaptureTrigger = Signal(0)

    val sphereFrames = Signal<List<String>>(listOf())
    val sphereOrientations = mutableListOf<Pair<Float, Float>>()
    val sphereCurrentOrientation = Signal(0f to 0f)
    var sphereRefAz: Float? = null
    val sphereGhostFrames = Signal<List<SphereGhostFrame>>(emptyList())
    private val pendingSphereCaptures = mutableListOf<PendingSphereCapture>()
    val sphereCurrentCell = Signal<Pair<Int, Int>?>(null)

    val captureMode = Signal("hdr")

    // Grid: 3 rows (pitch) × 8 cols (azimuth), full 360° × 90°
    // Each column = 45° azimuth, each row = 30° pitch (-45° to +45°)
    companion object {
        const val GRID_ROWS = 3
        const val GRID_COLS = 8
        const val COL_DEG = 45f
        const val PITCH_MIN = -45f
        const val PITCH_STEP = 30f
    }

    data class GridCell(val row: Int, val col: Int)
    private data class PendingSphereCapture(val cell: GridCell, val azimuth: Float, val pitch: Float)
    fun orientationToCell(azimuth: Float, pitch: Float): GridCell? {
        val refAz = sphereRefAz ?: return null
        var dAz = (azimuth - refAz) % 360f
        if (dAz < -180f) dAz += 360f
        if (dAz >= 180f) dAz -= 360f
        val col = ((dAz + 180f) / COL_DEG).toInt().coerceIn(0, GRID_COLS - 1)
        val row = ((pitch - PITCH_MIN) / PITCH_STEP).toInt().coerceIn(0, GRID_ROWS - 1)
        return GridCell(row, col)
    }

    val sphereGrid = Signal<List<List<Boolean>>>(List(GRID_ROWS) { List(GRID_COLS) { false } })

    fun recomputeGrid() {
        val grid = MutableList(GRID_ROWS) { MutableList(GRID_COLS) { false } }
        for ((az, pitch) in sphereOrientations) {
            val cell = orientationToCell(az, pitch) ?: return
            grid[cell.row][cell.col] = true
        }
        sphereGrid.value = grid.map { it.toList() }
    }

    fun nextCellHint(currentAz: Float, currentPitch: Float): String {
        val grid = sphereGrid.value
        val currentCell = orientationToCell(currentAz, currentPitch) ?: return "Point phone at scene"
        val total = grid.sumOf { row -> row.count { it } }
        if (total >= GRID_ROWS * GRID_COLS) return "All covered!"
        // Find closest empty cell (Manhattan distance with wrap-around for columns)
        var bestDist = Int.MAX_VALUE
        var bestR = currentCell.row; var bestC = currentCell.col
        for (r in 0 until GRID_ROWS) for (c in 0 until GRID_COLS) {
            if (!grid[r][c]) {
                val dRow = abs(r - currentCell.row)
                val dCol = minOf(abs(c - currentCell.col), GRID_COLS - abs(c - currentCell.col))
                val dist = dRow + dCol
                if (dist < bestDist) { bestDist = dist; bestR = r; bestC = c }
            }
        }
        val dirs = mutableListOf<String>()
        if (bestR < currentCell.row) dirs.add("down")
        if (bestR > currentCell.row) dirs.add("up")
        val dCol = bestC - currentCell.col
        val wrapCol = if (dCol > 0 && dCol < GRID_COLS / 2) dCol
            else if (dCol < 0 && -dCol < GRID_COLS / 2) dCol
            else if (dCol > 0) dCol - GRID_COLS
            else dCol + GRID_COLS
        if (wrapCol < 0) dirs.add("left")
        if (wrapCol > 0) dirs.add("right")
        return "Move ${dirs.joinToString("+")}"
    }

    override fun ElementWriter.CanAddTheme.render() {
        col {
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

            padded.row {
                listOf("hdr" to "HDR", "single" to "Photo", "night" to "Night", "focus" to "Focus", "spatial" to "Spatial", "sphere" to "Sphere").forEach { (mode, label) ->
                    expanding.toggleButton {
                        text(label); checked bind captureMode.equalTo(mode)
                    }
                }
            }

            reactive {
                when (captureMode()) {
                    "hdr" -> { isHdrMode.value = true; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = false }
                    "single" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = false }
                    "night" -> { isHdrMode.value = false; isNightSight.value = true; isFocusStacking.value = false; isSpatial.value = false }
                    "focus" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = true; isSpatial.value = false }
                    "spatial" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = true }
                    "sphere" -> { isHdrMode.value = false; isNightSight.value = false; isFocusStacking.value = false; isSpatial.value = false }
                }
            }

            reactive {
                if (captureMode() != "sphere" && sphereFrames().isNotEmpty()) {
                    sphereFrames.value = listOf()
                    sphereOrientations.clear()
                    sphereRefAz = null
                    sphereGhostFrames.value = emptyList()
                    pendingSphereCaptures.clear()
                    sphereCurrentCell.value = null
                }
            }

            expanding.frame {
                cameraView(
                    shutterTrigger = shutterTrigger,
                    onImagesCaptured = { paths ->
                        when {
                            isHdrMode.value && paths.size > 1 -> GlobalNavigator.main.navigate(HdrProcessingPage(paths))
                            captureMode.value == "sphere" && paths.isNotEmpty() -> {
                                sphereFrames.value = sphereFrames.value + paths
                                val pending = pendingSphereCaptures.removeFirstOrNull()
                                if (pending != null) {
                                    sphereGhostFrames.value = sphereGhostFrames.value + SphereGhostFrame(pending.azimuth, pending.pitch, paths.first())
                                }
                            }
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
                    },
                    isSpatial = isSpatial,
                    spatialCaptureTrigger = spatialCaptureTrigger,
                    onSpatialCaptured = { frames ->
                        GlobalNavigator.main.navigate(SpatialPage(frames))
                    },
                    onCameraLabels = { labels ->
                        cameraLabels.value = labels.mapIndexed { idx, l -> idx to l }
                    },
                    onSphereOrientationUpdate = { (az, pitch) ->
                        sphereCurrentOrientation.value = az to pitch
                        sphereCurrentCell.value = orientationToCell(az, pitch)?.let { it.row to it.col }
                    },
                    onSphereFrameOrientation = { az, pitch ->
                        if (sphereRefAz == null) sphereRefAz = az
                        sphereOrientations.add(az to pitch)
                        recomputeGrid()
                        val cell = orientationToCell(az, pitch)
                        if (cell != null) pendingSphereCaptures.add(PendingSphereCapture(cell, az, pitch))
                    },
                    sphereGridData = sphereGrid,
                    sphereCurrentCell = sphereCurrentCell,
                    sphereGhostFrames = sphereGhostFrames,
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

                shownWhen { isSpatial() }.frame {
                    atTopEnd.card.padded.col {
                        text("Stereo pair")
                    }
                }

                // ── Sphere guidance overlay ────────────────────────────────
                shownWhen { captureMode() == "sphere" }.frame {
                    atBottomStart.padded.row {
                        col {
                            (0 until GRID_ROWS).forEach { r ->
                                row {
                                    (0 until GRID_COLS).forEach { c ->
                                        sizedBox(SizeConstraints(width = 1.2.rem, height = 1.2.rem)).frame {
                                            centered.text {
                                                reactive {
                                                    val grid = sphereGrid()
                                                    val captured = grid.getOrNull(r)?.getOrNull(c) ?: false
                                                    val cur = sphereCurrentOrientation()
                                                    val curCell = orientationToCell(cur.first, cur.second)
                                                    val isCurrent = curCell != null && curCell.row == r && curCell.col == c
                                                    content = when {
                                                        isCurrent -> "\u25C9"
                                                        captured -> "\u25CF"
                                                        else -> "\u25CB"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        padded.col {
                            reactive {
                                val cur = sphereCurrentOrientation()
                                text(nextCellHint(cur.first, cur.second))
                            }
                            reactive {
                                val total = sphereGrid.value.sumOf { row -> row.count { it } }
                                text("$total / ${GRID_ROWS * GRID_COLS} covered")
                            }
                            text { ::content { "${sphereFrames().size} shots" } }
                        }
                    }
                }

                shownWhen { captureMode() == "sphere" }.frame {
                    atTopEnd.card.padded.col {
                        text { ::content { "${sphereFrames().size} captured" } }
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
                            isSpatial.value -> spatialCaptureTrigger.value += 1
                            else -> shutterTrigger.value += 1
                        }
                    }
                }

                button {
                    icon(Icon.list, "Gallery")
                    onClick { GlobalNavigator.main.navigate(GalleryPage()) }
                }
            }

            shownWhen { captureMode() == "sphere" && sphereFrames().size >= 2 }.frame {
                padded.important.button {
                    text { ::content { "Stitch ${sphereFrames().size} frames" } }
                    onClick {
                        GlobalNavigator.main.navigate(
                            PhotoSpherePage(
                                frames = sphereFrames.value,
                                orientations = sphereOrientations.toList()
                            )
                        )
                    }
                }
            }

            padded.row {
                text("Lens:")
                forEach(cameraLabels) { (idx, label) ->
                    toggleButton {
                        text(label); checked bind cameraLens.equalTo(idx)
                    }
                }
            }
        }
    }
}
