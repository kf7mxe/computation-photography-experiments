package com.kf7mxe.prescent.sdk

import com.kf7mxe.prescent.User

class UserSession(
    val api: Api,
    val userId: User.ID,
) : CachedApi(api) {

}
