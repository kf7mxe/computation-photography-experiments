package com.kf7mxe.prescent.views

// Gallery/disk access not available in the browser build
actual fun loadBracketSetsFromDisk(): List<GalleryPage.BracketSet> = emptyList()
actual fun deleteBracketSetFromDisk(bracketSet: GalleryPage.BracketSet) = Unit
