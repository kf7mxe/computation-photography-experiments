package com.kf7mxe.prescent

import com.lightningkite.kiteui.reactive.PersistentProperty

val bracketCountStore = PersistentProperty("settings_bracketCount", "3")
val evOffsetStore = PersistentProperty("settings_evOffset", "2.0")
val algorithmStore = PersistentProperty("settings_algorithm", "Mertens")
val alignmentStore = PersistentProperty("settings_alignment", "MTB")

// Enhancement toggle persistence
val smartFrameSelectionStore = PersistentProperty("settings_smartFrameSelection", "false")
val hotPixelFixStore = PersistentProperty("settings_hotPixelFix", "false")
val caCorrectionStore = PersistentProperty("settings_caCorrection", "false")
val lensCorrectionStore = PersistentProperty("settings_lensCorrection", "false")
val smartNRStore = PersistentProperty("settings_smartNR", "false")
val contrastSharpeningStore = PersistentProperty("settings_contrastSharpening", "false")
val jointDenoiseStore = PersistentProperty("settings_jointDenoise", "false")
val dehazeStore = PersistentProperty("settings_dehaze", "false")
