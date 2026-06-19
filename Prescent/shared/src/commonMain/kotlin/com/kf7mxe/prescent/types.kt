package com.kf7mxe.prescent

import kotlinx.serialization.Serializable

@Serializable
enum class AppPlatform {
    iOS,
    Android,
    Web,
    Desktop,
    ;

    companion object
}