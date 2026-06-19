package com.kf7mxe.prescent.sdk

import com.lightningkite.lightningserver.db.*
import kotlinx.serialization.builtins.*

open class CachedApi(val uncached: Api) {
	open val appReleases = ModelCache(uncached.appRelease, com.kf7mxe.prescent.AppRelease.serializer())
	open val users = ModelCache(uncached.user, com.kf7mxe.prescent.User.serializer())
	open val sessions = ModelCache(uncached.userAuth, com.lightningkite.lightningserver.sessions.Session.serializer(com.kf7mxe.prescent.User.serializer(), com.kf7mxe.prescent.User.ID.serializer()))
	open val totpSecrets = ModelCache(uncached.userAuth.totp, com.lightningkite.lightningserver.sessions.TotpSecret.serializer())
	open val passwordSecrets = ModelCache(uncached.userAuth.password, com.lightningkite.lightningserver.sessions.PasswordSecret.serializer())
	open val fcmTokens = ModelCache(uncached.fcmToken, com.kf7mxe.prescent.FcmToken.serializer())
	open val organizations = ModelCache(uncached.organization, com.kf7mxe.prescent.Organization.serializer())
	open val memberships = ModelCache(uncached.membership, com.kf7mxe.prescent.Membership.serializer())
}
