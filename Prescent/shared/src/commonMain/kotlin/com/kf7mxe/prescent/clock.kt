package com.kf7mxe.prescent

import com.lightningkite.services.data.ZonedDateTime
import com.lightningkite.services.data.nowLocal
import kotlin.time.Clock
import kotlin.time.Instant

private var prescentClock: Clock = Clock.System

val Clock.Companion.prescent: Clock get() = prescentClock

@RequiresOptIn("Meant to only be used in tests.")
annotation class TestOnly

@TestOnly
fun setPrescentClockForTesting(clock: Clock) {
    println("WARN!! prescent clock is being set to $clock.")
    prescentClock = clock
}

fun now(): Instant = Clock.prescent.now()
fun nowLocal(): ZonedDateTime = Clock.prescent.nowLocal()