package com.kf7mxe.prescent

import com.lightningkite.kiteui.Build
import com.lightningkite.kiteui.Platform
import com.lightningkite.kiteui.current
import com.lightningkite.kiteui.exceptions.ExceptionHandler
import com.lightningkite.kiteui.exceptions.installLsError
import com.lightningkite.kiteui.exceptions.installSmartHandlers
import com.lightningkite.kiteui.models.*
import com.lightningkite.kiteui.navigation.PageNavigator
import com.lightningkite.kiteui.views.ViewWriter
import com.lightningkite.kiteui.views.buttonTheme
import com.lightningkite.kiteui.views.card
import com.lightningkite.kiteui.views.centered
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.kiteui.views.exceptionMessage
import com.lightningkite.kiteui.views.l2.appNav
import com.lightningkite.kiteui.views.l2.dialog
import com.kf7mxe.prescent.extensions.toAppPlatform
import com.kf7mxe.prescent.sdk.currentSession
import com.kf7mxe.prescent.views.*
import com.kf7mxe.prescent.sdk.installLoggedOutErrors
import com.kf7mxe.prescent.sdk.selectedApi
import com.kf7mxe.prescent.utils.fcmSetup
import com.kf7mxe.prescent.utils.notificationPermissions
import com.kf7mxe.prescent.utils.requestNotificationPermissions
import com.kf7mxe.prescent.views.GalleryPage
import com.kf7mxe.prescent.views.HomePage
import com.kf7mxe.prescent.views.checkAppVersion
import com.lightningkite.reactive.context.await
import com.lightningkite.reactive.context.invoke
import com.lightningkite.reactive.context.reactiveSuspending
import com.lightningkite.reactive.core.AppScope
import com.lightningkite.reactive.core.Signal
import com.lightningkite.services.database.Query
import com.lightningkite.services.database.condition
import com.lightningkite.services.database.eq
import kotlinx.coroutines.launch

val defaultTheme = Theme.flat2("default", Angle(0.55f))// brandBasedExperimental("bsa", normalBack = Color.white)
val appTheme = Signal(defaultTheme)

// Notification Items
val fcmToken: Signal<String?> = Signal(null)
val setFcmToken = { token: String -> fcmToken.value = token } // This is for iOS. It is used in the iOS app. Do not remove.

fun ViewWriter.app(navigator: PageNavigator, dialog: PageNavigator) {
    context.exceptionHandlers.installSmartHandlers()
    context.exceptionHandlers.installLsError()
    context.exceptionHandlers.installLoggedOutErrors()

    AppScope.reactiveSuspending {
        if (currentSession() == null) return@reactiveSuspending
        val permission = notificationPermissions()
        when (permission) {
            false -> {}

            true -> {
                fcmSetup()
            }

            null -> {
                context.confirmDanger(
                    "Send notifications?",
                    "Prescent would like to send you notifications.",
                    "Allow"
                ) {
                    requestNotificationPermissions()
                }
            }
        }
    }

    checkAppVersion()

    return appNav(navigator, dialog) {
        appName = "Prescent"
        ::navItems {
            listOf(
                NavLink(title = "Camera", icon = Icon.home) { CameraPage() },
                NavLink(title = "Gallery", icon = Icon.list) { GalleryPage() },
            )
        }

        ::exists {
            navigator.currentPage() !is FullscreenPage
        }
    }
}

interface FullscreenPage


