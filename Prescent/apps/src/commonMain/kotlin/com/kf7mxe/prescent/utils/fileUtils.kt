package com.kf7mxe.prescent.utils

import com.lightningkite.kiteui.FileReference

/**
 * Copies [FileReference] objects to actual files in the app's cache directory
 * and returns the list of absolute file paths.
 *
 * This is needed because [FileReference] on Android wraps a [android.net.Uri]
 * which may be a content:// URI that cannot be read directly by native code.
 */
expect suspend fun copyFileReferencesToPaths(fileRefs: List<FileReference>): List<String>
