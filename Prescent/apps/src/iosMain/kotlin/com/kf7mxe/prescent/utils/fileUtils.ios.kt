package com.kf7mxe.prescent.utils

import com.lightningkite.kiteui.FileReference

actual suspend fun copyFileReferencesToPaths(fileRefs: List<FileReference>): List<String> {
    // iOS not yet implemented
    return emptyList()
}
