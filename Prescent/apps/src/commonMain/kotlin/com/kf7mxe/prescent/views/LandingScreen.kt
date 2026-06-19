package com.kf7mxe.prescent.views

import com.lightningkite.kiteui.Routable
import com.lightningkite.kiteui.navigation.Page
import com.lightningkite.kiteui.navigation.pageNavigator
import com.lightningkite.kiteui.views.*
import com.lightningkite.kiteui.views.centered
import com.lightningkite.kiteui.views.direct.*
import com.lightningkite.kiteui.views.direct.activityIndicator
import com.kf7mxe.prescent.FullscreenPage
import com.kf7mxe.prescent.sdk.currentSession
import com.lightningkite.reactive.context.await
import com.lightningkite.reactive.core.Constant
import com.lightningkite.reactive.core.Reactive
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Routable("/")
class LandingPage : Page, FullscreenPage {
    override val title: Reactive<String> get() = Constant("Loading")
    override fun ElementWriter.CanAddTheme.render() {
        launch {
//            if (currentSession.await() != null) {
                context.pageNavigator.reset(CameraPage())
//            } else {
//                context.pageNavigator.reset(LoginPage())
//            }
        }
        frame {
            centered.activityIndicator()
        }
    }
}