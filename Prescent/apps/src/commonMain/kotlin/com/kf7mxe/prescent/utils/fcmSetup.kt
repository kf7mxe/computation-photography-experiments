package com.kf7mxe.prescent.utils


expect fun fcmSetup(): Unit

expect suspend fun requestNotificationPermissions(): Unit

expect suspend fun notificationPermissions(): Boolean?
