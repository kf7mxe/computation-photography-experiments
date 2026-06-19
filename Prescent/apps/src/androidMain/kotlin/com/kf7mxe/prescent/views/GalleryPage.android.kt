package com.kf7mxe.prescent.views

import android.util.Log
import com.lightningkite.kiteui.views.AndroidAppContext
import java.io.File

actual fun loadBracketSetsFromDisk(): List<GalleryPage.BracketSet> {
    val context = AndroidAppContext.applicationCtx
    val bracketsRoot = File(context.filesDir, "brackets")
    if (!bracketsRoot.exists() || !bracketsRoot.isDirectory) return emptyList()

    return bracketsRoot.listFiles()
        ?.filter { it.isDirectory }
        ?.sortedByDescending { it.name } // newest first
        ?.mapNotNull { dir ->
            val images = dir.listFiles()
                ?.filter { it.isFile && it.extension.lowercase() in listOf("jpg", "jpeg", "png") }
                ?.sortedBy { it.name }
                ?.map { it.absolutePath }
                ?: emptyList()
            val timestamp = dir.name.toLongOrNull() ?: 0L
            if (images.isNotEmpty()) GalleryPage.BracketSet(dir.name, images, timestamp) else null
        }
        ?: emptyList()
}

actual fun deleteBracketSetFromDisk(bracketSet: GalleryPage.BracketSet) {
    val context = AndroidAppContext.applicationCtx
    val dir = File(context.filesDir, "brackets/${bracketSet.folderName}")
    if (dir.exists()) {
        dir.deleteRecursively()
        Log.d("GalleryPage", "Deleted bracket set: ${bracketSet.folderName}")
    }
}
