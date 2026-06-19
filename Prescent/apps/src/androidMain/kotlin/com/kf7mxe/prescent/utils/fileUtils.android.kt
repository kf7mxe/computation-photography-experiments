package com.kf7mxe.prescent.utils

import com.lightningkite.kiteui.FileReference
import com.lightningkite.kiteui.mimeType
import com.lightningkite.kiteui.views.AndroidAppContext
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

actual suspend fun copyFileReferencesToPaths(fileRefs: List<FileReference>): List<String> {
    val context = AndroidAppContext.applicationCtx
    val cacheDir = File(context.cacheDir, "hdr_input").also { it.mkdirs() }
    val result = mutableListOf<String>()

    fileRefs.forEach { fileRef ->
        val uri = fileRef.uri
        val extension = when (fileRef.mimeType()) {
            "image/jpeg" -> "jpg"
            "image/png" -> "png"
            "image/webp" -> "webp"
            else -> "jpg"
        }
        val destFile = File(cacheDir, "${UUID.randomUUID()}.$extension")

        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(destFile).use { output ->
                input.copyTo(output)
            }
        }

        if (destFile.exists() && destFile.length() > 0) {
            result.add(destFile.absolutePath)
        }
    }

    return result
}
