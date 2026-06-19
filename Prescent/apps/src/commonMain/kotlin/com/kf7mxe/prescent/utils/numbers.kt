package com.kf7mxe.prescent.utils

import kotlin.math.abs

fun Float.fmt(): String {
    val abs = abs(this)
    val whole = abs.toInt()
    val frac = ((abs - whole) * 10 + 0.5f).toInt()
    val sign = if (this < 0) "-" else ""
    return "$sign$whole.$frac"
}

fun Int.fmt(): String = toString()
