---
name: Lightning Server Development
description: IMPORTANT - Read this skill proactively when working on any file that imports from com.lightningkite.lightningserver. This skill covers Lightning Server framework, Kotlin server development, building APIs, database operations, authentication, and testing patterns.
version: 1.2.0
---

# Lightning Server Development Skill

You are an expert Lightning Server developer. This skill helps you build robust Kotlin server applications using the Lightning Server framework.

**⚠️ IMPORTANT:** Proactively read this skill whenever you open or work on a file that imports from `com.lightningkite.lightningserver.*`. The patterns and lifecycle details here are critical for correct implementation.

## Framework Overview

Lightning Server is a Kotlin-based server framework for building APIs across multiple serverless platforms. It provides:
- Type-safe endpoint definitions with auto-generated documentation
- Database abstractions (MongoDB, Postgres, JSON files)
- Caching abstractions (Redis, Memcached, DynamoDB)
- File storage abstractions (S3, Azure, local)
- Authentication & authorization (email, SMS, OAuth, password, OTP)
- WebSocket support
- Background tasks and scheduled jobs
- Multi-platform SDK generation (TypeScript, Kotlin)
- OpenAPI documentation generation

**Version:** 5.x
**Main Branch:** master

## REST-First Methodology

**⚠️ KEY INSIGHT:** Before creating custom endpoints, ask if ModelRestEndpoints can handle it:

1. **Can REST create + `interceptCreate` handle validation?** - Use for booking validation, conflict detection, subscription restrictions
2. **Can REST modify + `updateRestrictions` handle field rules?** - Use for phase transitions, state changes
3. **Can REST query handle the read?** - Client can compute complex views from multiple queries
4. **Can `postChange` handle side effects?** - Use for cascade updates, notifications

**Custom endpoints are ONLY needed for:**
- Hardware integrations (door unlocks, IoT) - external systems dictate the interface
- External API callbacks (webhooks) - third party dictates the contract
- Truly atomic multi-model operations where eventual consistency isn't acceptable

**Example - Reservation Booking (NO custom endpoint needed):**
```kotlin
// All business logic in interceptCreate - no custom booking endpoint!
.interceptCreate { reservation ->
    // 1. Conflict detection
    val conflicts = table.find(condition {
        (it.booth eq reservation.booth) and
        (it.at lt reservation.endsAt) and
        (it.endsAt gt reservation.at) and
        it.cancelledAt.eq(null)
    }).toList()
    if (conflicts.isNotEmpty()) {
        throw BadRequestException("Time slot conflicts with existing reservation")
    }

    // 2. Subscription restrictions
    val restrictions = getRestrictions(reservation.user)
    restrictions?.let { r ->
        if (duration < r.minimumDuration) throw BadRequestException("Duration too short")
        if (reservationsToday >= r.perDay) throw BadRequestException("Daily limit reached")
    }

    // 3. Set expiration for tentative booking
    reservation.copy(expireAt = Clock.System.now() + 15.minutes)
}

// Background task handles expiration - no custom endpoint needed
val expireStale = path.path("scheduled/expire") bind ScheduledTask(frequency = 1.minutes) {
    table.updateMany(
        condition { it.expireAt.notNull and (it.expireAt lt now) and it.cancelledAt.eq(null) },
        modification { it.cancelledAt assign now }
    )
}

// Confirm booking = just remove expireAt via REST PATCH
// Cancel booking = just set cancelledAt via REST PATCH
```

## Core Concepts

### 1. ServerBuilder Pattern

All Lightning Server applications use the `ServerBuilder` pattern:

```kotlin
object MyServer : ServerBuilder() {
    // Settings
    val database = setting("database", Database.Settings())
    val cache = setting("cache", Cache.Settings())

    // Endpoints
    val hello = path.get bind HttpHandler {
        HttpResponse.plainText("Hello World!")
    }
}
```

### 2. Endpoint Definition

Endpoints use a fluent path-building syntax:

```kotlin
// Simple GET
val root = path.get bind HttpHandler { /* ... */ }

// With path parameter
val getUser = path.path("users").arg<String>("id").get bind HttpHandler { request ->
    val id = request.path.arg1  // Type-safe access
    // ...
}

// Multiple arguments
val getUserPost = path.path("users").arg<String>("userId")
    .path("posts").arg<Int>("postId").get bind HttpHandler { request ->
    val userId = request.path.arg1  // String
    val postId = request.path.arg2  // Int
    // ...
}
```

### 3. Typed Endpoints

Use typed endpoints for auto-documentation and SDK generation. There are two approaches:

#### Using .api() Helper (Recommended for simple cases)

```kotlin
val createPost = path.path("posts").post.api(
    summary = "Create a blog post",
    description = "Creates a new blog post with the provided data",
    auth = noAuth,  // or auth<User>()
    errorCases = listOf(
        LSError(http = 400, detail = "invalid-input", message = "Title required")
    ),
    successCode = HttpStatus.Created,
    implementation = { input: CreatePostRequest ->
        // Type-safe implementation
        CreatePostResponse(...)
    }
)
```

#### Using ApiHttpHandler Directly (For custom paths or detail endpoints)

```kotlin
// Custom endpoint on ModelRestEndpoints detail path
val unlockDoor = rest.detailPath.path("unlock").post bind ApiHttpHandler(
    summary = "Unlock Door",
    description = "Unlocks the door for an active reservation",
    auth = UserAuth.require(),
    implementation = { _: Unit ->
        // Access path parameters from parent paths
        val reservationId = request.arg1  // From rest.detailPath

        // Implementation is automatically suspend - can call suspend functions directly
        val reservation = info.table().get(reservationId)
            ?: throw BadRequestException("Not found")

        // Validate state
        if (reservation.checkedInAt == null) {
            throw BadRequestException("Not checked in")
        }

        // Call suspend functions directly
        unifiClient.unlockDoor(reservation.lockId, durationSeconds = 60)

        Unit
    }
)

// With path parameter and input
val updateSettings = path.path("settings").arg<String>("key").post bind ApiHttpHandler(
    summary = "Update Setting",
    auth = UserAuth.require(),
    implementation = { newValue: String ->
        val settingKey = request.arg1  // Type-safe access to path arg
        settingsService.update(settingKey, newValue)
    }
)
```

**Key ApiHttpHandler features:**
- Implementation block is **automatically suspend** - no need to wrap in `runBlocking` or `coroutineScope`
- Access path parameters via `request.arg1`, `request.arg2`, etc. (type-safe based on `.arg<T>()` definitions)
- Input type is the type parameter (`{ input: InputType -> ... }`)
- Return value becomes the response body
- Can call suspend functions directly (database queries, external API calls, etc.)

### 4. Database Operations

**⚠️ IMPORTANT: Use ModelRestEndpoints for CRUD Operations**

For standard CRUD operations, use the pre-built `ModelRestEndpoints` rather than manually creating database endpoints:

```kotlin
// Define your model with @GenerateDataClassPaths
@Serializable
@GenerateDataClassPaths
data class Post(
    override val _id: Uuid = Uuid.random(),
    val title: String,
    val content: String,
    val authorId: Uuid,
    val createdAt: Instant = Clock.System.now()
) : HasId<Uuid>

// Set up ModelInfo with auth and permissions
val postInfo = database.modelInfo(
    auth = UserAuth.require() or AuthRequirement.None,
    permissions = {
        val user = authOrNull?.fetch()
        val isOwner = condition { it.authorId eqNn user?._id }

        ModelPermissions(
            create = if (user != null) Condition.Always else Condition.Never,
            read = Condition.Always,
            update = isOwner,
            delete = isOwner
        )
    }
)

// Create REST endpoints automatically (provides list, get, create, update, delete, query)
val posts = path.path("posts").path("rest") module ModelRestEndpoints(postInfo)

// Optional: Add WebSocket updates for real-time changes
val postsWithWs = path.path("posts").path("rest") include
    ModelRestEndpoints(postInfo) + ModelRestUpdatesWebsocket(postInfo)
```

#### Signal Composition and Lifecycle Hooks

ModelInfo supports powerful signal composition for validation, side effects, and data integrity:

```kotlin
val projectInfo = database.modelInfo(
    auth = UserAuth.require(),
    permissions = { permissions(this) },
    signals = { table ->
        table
            // Validate changes before they're applied
            .interceptChange(::interceptChange)

            // Validate creates before insertion
            .interceptCreates { validateProjects(it) }

            // React after changes are persisted
            .postChange { old, new -> handleProjectChange(old, new) }

            // Clean up after deletion
            .postDelete { deleteRelatedData(it) }

            // Process images automatically
            .interceptImagesForProcessing(
                MediaPreviewOptions.CorrectOddFeatures,
                MediaPreviewOptions(sizeInPixels = 200, type = MediaType.Image.JPEG)
            ) { it.avatar }

            // Maintain denormalized fields
            .denormalize(
                Project_organization,
                Project_organizationName,
                organizationTable,
                Organization_name,
                null
            )
    }
)
```

## REST Endpoint Lifecycle

**⚠️ CRITICAL:** Understanding the full lifecycle is essential for placing business logic in the right hooks.

### Full Lifecycle for REST Operations

#### POST (Create) Lifecycle

When a client calls `POST /model/rest`:

```
1. Request arrives at ModelRestEndpoints.insert
   ↓
2. Authentication checked (via info.auth)
   ↓
3. info.table(auth) called → builds Table with all wrappers
   ↓
4. Table.interceptCreate runs
   ├─ Can modify the object being created
   ├─ Can throw exceptions to reject creation
   ├─ ⚠️ Denormalized fields NOT populated yet!
   └─ Must fetch related records directly if needed
   ↓
5. Database insert occurs (actual write)
   ↓
6. Denormalization runs (.denormalize() calculations)
   ├─ Denormalized fields now populated
   └─ Source records queried and values copied
   ↓
7. Permissions.create condition checked
   ├─ If condition fails, insertion is rolled back
   └─ Returns ForbiddenException
   ↓
8. Table.postCreate runs
   ├─ Object is now in database with all fields
   ├─ Safe to trigger side effects
   └─ Cannot stop creation (already committed)
   ↓
9. Response sent to client with created object
```

**Example:**
```kotlin
signals = { table ->
    table
        .interceptCreate { reservation ->
            // 1. Validation - check conflicts
            val conflicts = info.table().find(condition {
                (it.booth eq reservation.booth) and
                (it.at lt reservation.endsAt) and
                (it.endsAt gt reservation.at) and
                it.cancelledAt.eq(null)
            }).toList()
            if (conflicts.isNotEmpty()) {
                throw BadRequestException("Time slot conflicts")
            }

            // 2. Business logic - check subscription limits
            val subscription = getActiveSubscription(reservation.user)
            subscription?.let {
                val limit = getPlanLimit(it.plan)
                val todayCount = countReservationsToday(reservation.user)
                if (todayCount >= limit) {
                    throw BadRequestException("Daily limit reached")
                }
            }

            // 3. Modify object - set expiration
            reservation.copy(expireAt = Clock.System.now() + 15.minutes)
        }
        .postCreate { created ->
            // 4. Side effects - send email notification
            emailService.sendConfirmation(created.user, created)
        }
}
```

#### PATCH (Update) Lifecycle

When a client calls `PATCH /model/rest/{id}`:

```
1. Request arrives with Modification<T>
   ↓
2. Authentication checked (via info.auth)
   ↓
3. info.table(auth) called → builds Table with all wrappers
   ↓
4. Table.interceptUpdate runs (if using interceptChangePerInstance)
   ├─ Has access to BOTH old and new values
   ├─ Can modify the Modification being applied
   ├─ Can throw exceptions to reject update
   └─ Expensive - fetches existing record first
   ↓
5. Table.interceptChange / interceptModification runs
   ├─ Can validate or modify the Modification
   ├─ Can throw exceptions to reject update
   └─ Cheaper - doesn't need existing record
   ↓
6. Permissions.updateRestrictions checked
   ├─ Field-level restrictions enforced
   ├─ e.g., it.author.cannotBeModified()
   └─ Throws ForbiddenException if violated
   ↓
7. Database update occurs (actual write)
   ↓
8. Denormalization runs (if source fields changed)
   ├─ Updates denormalized fields
   └─ May trigger cascade updates to other tables
   ↓
9. Permissions.update condition checked
   ├─ Checked against OLD value
   ├─ If condition fails, update is rolled back
   └─ Returns ForbiddenException
   ↓
10. Table.postChange runs
    ├─ Has access to both old and new values
    ├─ Object already updated in database
    ├─ Safe to trigger side effects
    └─ Cannot stop update (already committed)
    ↓
11. Response sent to client with updated object
```

**Example:**
```kotlin
permissions = {
    val user = auth.fetch()
    ModelPermissions(
        // ...
        update = condition { it.user eq user._id },  // Can only update your own
        updateRestrictions = updateRestrictions {
            it.user.cannotBeModified()      // Field restriction
            it.createdAt.cannotBeModified() // Field restriction
        }
    )
},
signals = { table ->
    table
        .interceptChange { mod ->
            // Validate modifications before they're applied
            mod.vet(Reservation_booth) { fieldMod ->
                when (fieldMod) {
                    is Modification.Assign -> {
                        // Check if new booth is available
                        val newBooth = BoothEndpoints.info.table().get(fieldMod.value)
                            ?: throw BadRequestException("Booth not found")
                        if (!newBooth.available) {
                            throw BadRequestException("Booth not available")
                        }
                    }
                    else -> {}
                }
            }
            mod
        }
        .postChange { old, new ->
            // Side effect - if booth changed, send notification
            if (old.booth != new.booth) {
                notificationService.send(new.user, "Booth changed")
            }
        }
}
```

### Hook Reference Guide

#### interceptCreate / interceptCreates

**When it runs:** Before database insert, after auth check

**What it can do:**
- ✅ Modify the object being created (return modified copy)
- ✅ Throw exceptions to reject creation
- ✅ Validate business rules (uniqueness, conflicts, quotas)
- ✅ Set computed fields (expiration, defaults)
- ✅ Query database for validation
- ❌ Cannot access denormalized fields (not populated yet)

**Use cases:**
- Conflict detection (time slot conflicts, unique constraints)
- Subscription/quota validation (daily limits, plan restrictions)
- Setting expiration times (tentative bookings)
- Validating related records exist
- Enforcing business rules

**Example:**
```kotlin
.interceptCreate { reservation ->
    // Validate: No conflicts
    val conflicts = findConflicts(reservation)
    if (conflicts.isNotEmpty()) throw BadRequestException("Conflict")

    // Validate: Subscription limits
    val subscription = getSubscription(reservation.user)
    validateLimits(subscription, reservation)

    // Modify: Set expiration
    reservation.copy(expireAt = now() + 15.minutes)
}
```

#### interceptChange / interceptModification

**When it runs:** Before database update, after auth check, before updateRestrictions

**What it can do:**
- ✅ Modify the Modification being applied
- ✅ Throw exceptions to reject update
- ✅ Validate field changes with mod.vet()
- ✅ Query database for validation
- ❌ Cannot access old/new values (use interceptChangePerInstance for that)

**Use cases:**
- Field-level validation (enum transitions, format checks)
- Validating modifications before application
- Transforming modifications (normalize data)
- Enforcing state machine rules

**Example:**
```kotlin
.interceptChange { mod ->
    mod.vet(Reservation_status) { fieldMod ->
        when (fieldMod) {
            is Modification.Assign -> {
                // Validate status transitions
                if (!isValidTransition(currentStatus, fieldMod.value)) {
                    throw BadRequestException("Invalid status transition")
                }
            }
            else -> {}
        }
    }
    mod
}
```

#### interceptChangePerInstance (expensive!)

**When it runs:** Before database update, fetches existing record first

**What it can do:**
- ✅ Access both old value and modification
- ✅ Modify the Modification based on old value
- ✅ Throw exceptions to reject update
- ⚠️ Expensive - requires extra database read

**Use cases:**
- Validation that requires old value
- State machine transitions
- Audit trail generation

**Example:**
```kotlin
.interceptChangePerInstance { old, mod ->
    // Can access old value for validation
    if (old.status == Status.Completed) {
        throw BadRequestException("Cannot modify completed reservation")
    }
    mod
}
```

#### updateRestrictions (in ModelPermissions)

**When it runs:** After interceptChange, before database update

**What it can do:**
- ✅ Block specific fields from being modified
- ✅ Enforced automatically by framework
- ❌ Cannot throw custom exceptions (framework handles it)

**Use cases:**
- Prevent modification of immutable fields (author, createdAt)
- Enforce field-level access control
- Prevent tampering with computed fields

**Example:**
```kotlin
ModelPermissions(
    // ...
    updateRestrictions = updateRestrictions {
        it.user.cannotBeModified()           // User cannot change
        it.createdAt.cannotBeModified()      // Created timestamp locked
        it.organizationName.cannotBeModified() // Denormalized field locked
    }
)
```

#### postChange

**When it runs:** After successful database update, after denormalization

**What it can do:**
- ✅ Access both old and new values
- ✅ Trigger side effects (notifications, cascade updates)
- ✅ Update denormalized data in other tables
- ✅ Log changes for audit trail
- ❌ Cannot stop the update (already committed)
- ❌ Should not throw exceptions (may leave inconsistent state)

**Use cases:**
- Send notifications (email, push, WebSocket)
- Cascade updates to denormalized fields in other tables
- Trigger workflows based on changes
- Audit logging
- Cache invalidation

**Example:**
```kotlin
.postChange { old, new ->
    // Cascade update denormalized fields
    if (old.name != new.name) {
        RelatedTable.updateMany(
            condition { it.projectId eq new._id },
            modification { it.projectName assign new.name }
        )
    }

    // Send notifications
    if (old.status != new.status) {
        notificationService.send(new.user, "Status changed to ${new.status}")
    }
}
```

#### postCreate

**When it runs:** After successful database insert, after denormalization

**What it can do:**
- ✅ Access the created object (with all fields populated)
- ✅ Trigger side effects (notifications, related record creation)
- ✅ Create related records
- ❌ Cannot stop the creation (already committed)
- ❌ Should not throw exceptions (may leave inconsistent state)

**Use cases:**
- Send welcome emails
- Create related records (default settings, initial data)
- Trigger external systems
- Analytics tracking

**Example:**
```kotlin
.postCreate { created ->
    // Send confirmation email
    emailService.sendWelcome(created.email)

    // Create default settings
    SettingsTable.insertOne(Settings(userId = created._id))
}
```

#### postDelete

**When it runs:** After successful database deletion

**What it can do:**
- ✅ Access the deleted object
- ✅ Clean up related records
- ✅ Trigger external cleanup
- ❌ Cannot stop the deletion (already committed)
- ❌ Should not throw exceptions (may leave inconsistent state)

**Use cases:**
- Delete related records (cascade delete)
- Clean up uploaded files
- Notify external systems
- Audit logging

**Example:**
```kotlin
.postDelete { deleted ->
    // Clean up related records
    RelatedTable.deleteMany(condition { it.projectId eq deleted._id })

    // Delete uploaded files
    deleted.avatarFile?.let { fileService.delete(it) }
}
```

### Decision Tree: Which Hook to Use?

**Need to validate before creation?**
→ Use `interceptCreate`

**Need to validate before update?**
→ Use `interceptChange` (or `interceptModification`)

**Need old value for validation?**
→ Use `interceptChangePerInstance` (expensive!)

**Need to prevent specific fields from being modified?**
→ Use `updateRestrictions` in `ModelPermissions`

**Need to send notifications after change?**
→ Use `postChange` or `postCreate`

**Need to cascade updates to other tables?**
→ Use `postChange`

**Need to clean up after deletion?**
→ Use `postDelete`

### Example: Complete Reservation System

Putting it all together for a complete reservation system:

```kotlin
val reservationInfo = database.modelInfo(
    auth = UserAuth.require(),
    permissions = {
        val user = auth.fetch()
        ModelPermissions(
            create = condition { it.user eq user._id },
            read = condition { it.user eq user._id },
            update = condition { it.user eq user._id },
            delete = condition { it.user eq user._id },
            updateRestrictions = updateRestrictions {
                it.user.cannotBeModified()      // Can't change owner
                it.createdAt.cannotBeModified() // Can't change timestamp
                it.booth.cannotBeModified()     // Can't change booth after creation
            }
        )
    },
    signals = { table ->
        table
            // 1. VALIDATION ON CREATE
            .interceptCreate { reservation ->
                // Check for time slot conflicts
                val conflicts = info.table().find(condition {
                    (it.booth eq reservation.booth) and
                    (it.at lt reservation.endsAt) and
                    (it.endsAt gt reservation.at) and
                    it.cancelledAt.eq(null)
                }).toList()
                if (conflicts.isNotEmpty()) {
                    throw BadRequestException("Time slot conflicts with existing reservation")
                }

                // Check subscription limits (if subscription exists)
                val subscription = findActiveSubscription(reservation.user, reservation.organization)
                subscription?.let { sub ->
                    val restrictions = getPlanRestrictions(sub.plan)
                    val duration = reservation.endsAt - reservation.at

                    if (duration < restrictions.minimumDuration) {
                        throw BadRequestException("Duration too short")
                    }
                    if (duration > restrictions.maximumDuration) {
                        throw BadRequestException("Duration too long")
                    }

                    val todayCount = countReservationsToday(reservation.user, reservation.booth)
                    if (todayCount >= restrictions.perDay) {
                        throw BadRequestException("Daily limit reached")
                    }
                }

                // Set expiration for tentative booking
                reservation.copy(expireAt = Clock.System.now() + 15.minutes)
            }

            // 2. VALIDATION ON UPDATE
            .interceptChange { mod ->
                // Validate status transitions
                mod.vet(Reservation_status) { fieldMod ->
                    when (fieldMod) {
                        is Modification.Assign -> {
                            if (!isValidStatusTransition(fieldMod.value)) {
                                throw BadRequestException("Invalid status transition")
                            }
                        }
                        else -> {}
                    }
                }
                mod
            }

            // 3. SIDE EFFECTS AFTER CHANGE
            .postChange { old, new ->
                // Send notification if status changed
                if (old.status != new.status) {
                    notificationService.send(
                        new.user,
                        "Reservation status changed to ${new.status}"
                    )
                }

                // If confirmed (expireAt cleared), send confirmation email
                if (old.expireAt != null && new.expireAt == null) {
                    emailService.sendConfirmation(new.user, new)
                }
            }

            // 4. SIDE EFFECTS AFTER CREATE
            .postCreate { created ->
                // Send tentative booking notification
                emailService.sendTentativeBooking(created.user, created)
            }

            // 5. CLEANUP AFTER DELETE
            .postDelete { deleted ->
                // Cancel any related services
                if (deleted.checkedInAt != null) {
                    serviceIntegration.releaseResources(deleted.booth)
                }
            }
    }
)
```

### Common Pitfalls

**❌ Accessing denormalized fields in interceptCreate:**
```kotlin
.interceptCreate { request ->
    // ❌ WRONG - proposedBy is not set yet!
    if (request.proposedBy == request.proposedTo) {
        throw BadRequestException("Cannot propose to yourself")
    }

    // ✅ CORRECT - fetch source data directly
    val fromReservation = ReservationTable.get(request.fromReservation)
        ?: throw BadRequestException("Not found")
    if (fromReservation.user == toReservation.user) {
        throw BadRequestException("Cannot propose to yourself")
    }
}
```

**❌ Throwing exceptions in postChange:**
```kotlin
.postChange { old, new ->
    // ❌ WRONG - may leave inconsistent state
    if (new.invalidField) {
        throw BadRequestException("Invalid")
    }

    // ✅ CORRECT - validate in interceptChange instead
}
```

**❌ Forgetting updateRestrictions:**
```kotlin
// ❌ WRONG - client can modify createdAt!
ModelPermissions(
    update = condition { it.user eq user._id }
)

// ✅ CORRECT - lock immutable fields
ModelPermissions(
    update = condition { it.user eq user._id },
    updateRestrictions = updateRestrictions {
        it.createdAt.cannotBeModified()
        it.user.cannotBeModified()
    }
)
```

**⚠️ CRITICAL: Signal Hook Execution Order**

Understanding the exact execution order is crucial for correct implementation:

1. **interceptCreates / interceptChange** - Validation (can throw exceptions)
   - ⚠️ **Denormalized fields are NOT populated yet!**
   - Must fetch source records directly if needed
2. **Database write occurs** - Record inserted/updated
3. **Denormalization updates** - `.denormalize()` calculations run, fields populated
4. **postChange / postCreate / postDelete** - Side effects (cascade updates, notifications)

**Example showing timing:**

```kotlin
.denormalize2(
    BoothDividerOpenRequest_fromReservation,
    ReservationEndpoints.info.table(),
    DenormalizationCalculation(BoothDividerOpenRequest_proposedBy, Uuid.fromLongs(0, 0)) { it.user },
    DenormalizationCalculation(BoothDividerOpenRequest_from, Instant.DISTANT_PAST) { it.at }
).interceptCreate { request ->
    // ⚠️ request.proposedBy and request.from are NOT set yet!
    // Denormalization hasn't run - still have default values

    // ✅ Must fetch source directly:
    val fromReservation = ReservationEndpoints.info.table().get(request.fromReservation)
        ?: throw BadRequestException("fromReservation not found")

    val proposedBy = fromReservation.user  // Get from source, not request.proposedBy
    val timeRange = fromReservation.at .. fromReservation.endsAt

    // Validate and potentially modify the request
    val toReservation = findToReservation(request.toBooth, timeRange)
    request.copy(toReservation = toReservation._id, proposedTo = toReservation.user)
}
```

**Key patterns:**
- **interceptChange**: Validate modifications before applying
- **interceptCreates**: Validate before insertion (e.g., check quotas, uniqueness, related records exist)
  - Can modify the object being created by returning a modified copy
  - Denormalized fields not yet populated - fetch source data if needed
- **postChange**: Cascade updates to denormalized data
- **postDelete**: Clean up related records
- **denormalize**: Auto-maintain denormalized fields (updates automatically on source changes)

**Optional Validation Pattern:**

A common pattern is to only enforce restrictions when certain conditions are met (e.g., user has a subscription):

```kotlin
.interceptCreate { reservation ->
    // Get booth and location info
    val booth = BoothEndpoints.info.table().get(reservation.booth)
        ?: throw BadRequestException("Booth not found")

    val location = LocationEndpoints.info.table().get(booth.location)
        ?: throw BadRequestException("Location not found")

    // Get restrictions (might be null if no subscription)
    val subscription = findActiveSubscription(reservation.user, booth.organization, booth.location)
    val restrictions = subscription?.let { sub ->
        getPlanRestrictions(sub.plan, booth.boothType)
    }

    // Only validate if restrictions exist
    restrictions?.let { restr ->
        // Validate duration
        val duration = reservation.endsAt - reservation.at
        if (duration < restr.minimumDuration) {
            throw BadRequestException("Duration too short")
        }
        if (duration > restr.maximumDuration) {
            throw BadRequestException("Duration too long")
        }

        // Validate per-day limit
        val dayStart = reservation.at.toLocalDateTime(location.timeZone).date.atStartOfDayIn(location.timeZone)
        val reservationsToday = info.table().find(
            condition {
                (it.user eq reservation.user) and
                (it.booth eq reservation.booth) and
                (it.at gte dayStart) and
                it.cancelledAt.eq(null)
            }
        ).toList().size

        if (reservationsToday >= restr.perDay) {
            throw BadRequestException("Daily limit reached")
        }
    }

    reservation
}
```

This gives you:
- `GET /posts/rest` - List with pagination, sorting, filtering
- `GET /posts/rest/{id}` - Get by ID
- `POST /posts/rest` - Create
- `PUT /posts/rest/{id}` - Update
- `DELETE /posts/rest/{id}` - Delete
- `POST /posts/rest/query` - Advanced querying
- `WS /posts/rest/watch` - Real-time updates (if WebSocket added)

**Manual Database Operations (use only when needed)**

Use low-level database operations for custom business logic beyond simple CRUD:

```kotlin
val posts = database().table<Post>()

// Insert
posts.insertOne(Post(title = "Hello", content = "World"))

// Query
posts.find(condition { it.title eq "Hello" }).toList()

// Update
posts.updateOne(
    condition { it._id eq id },
    modification { it.title assign "Updated" }
)

// Delete
posts.deleteMany(condition { it.authorId eq userId })

// Complex queries
posts.find(
    condition = condition {
        (it.title.contains("Kotlin")) and (it.createdAt gt yesterday)
    },
    orderBy = listOf(SortPart(Post.path.createdAt, false)),
    skip = page * pageSize,
    limit = pageSize
).toList()
```

Use manual operations when you need:
- Custom business logic beyond CRUD
- Complex queries not supported by ModelRestEndpoints
- Special validation or transformation logic
- Aggregations or computed fields

#### Modification Validation with mod.vet()

Validate modifications before they're applied to the database using `mod.vet()`:

```kotlin
fun interceptChange(mod: Modification<Project>): Modification<Project> {
    // Validate specific field modifications
    mod.vet(Project_projectTags) { fieldMod ->
        when (fieldMod) {
            is Modification.Assign -> {
                if (fieldMod.value.any { it != it.trim() })
                    throw BadRequestException("All tags must be trimmed")
                if (fieldMod.value.any { it != it.lowercase() })
                    throw BadRequestException("All tags must be lowercase")
            }
            is Modification.SetAppend -> {
                if (fieldMod.items.any { it != it.trim() })
                    throw BadRequestException("All tags must be trimmed")
                if (fieldMod.items.any { it != it.lowercase() })
                    throw BadRequestException("All tags must be lowercase")
            }
            else -> {}
        }
    }
    return mod
}

// Use in signals
signals = { table ->
    table.interceptChange(::interceptChange)
}
```

**Common Modification types:**
- `Modification.Assign` - Setting a value directly
- `Modification.SetAppend` / `Modification.SetRemove` - Modifying sets
- `Modification.ListAppend` / `Modification.ListRemove` - Modifying lists
- `Modification.Increment` / `Modification.Decrement` - Math operations

**Cascade Updates in postChange:**

```kotlin
context(runtime: ServerRuntime)
suspend fun postChange(old: Project, new: Project) {
    // Update denormalized fields across tables
    if (old.name != new.name) {
        Server.tasks.info.baseTable()
            .updateManyIgnoringResult(
                condition { it.project eq new._id },
                modification { it.projectName assign new.name }
            )
        Server.timeEntries.info.baseTable()
            .updateManyIgnoringResult(
                condition { it.project eq new._id },
                modification { it.projectName assign new.name }
            )
    }
}

context(runtime: ServerRuntime)
suspend fun postDelete(project: Project) {
    // Clean up related records
    Server.tasks.info.baseTable()
        .deleteManyIgnoringOld(condition { it.project eq project._id })
}
```

**Best practices:**
- Use `interceptChange` for validation (can throw exceptions to prevent changes)
- Use `postChange` for side effects (cascade updates, notifications, logging)
- Use `postDelete` for cleanup (delete related records, notify systems)
- Use `updateManyIgnoringResult` for performance (skips fetching updated records)
- Use `deleteManyIgnoringOld` for performance (skips fetching deleted records)

### 5. Authentication

Define a PrincipalType for your user model:

```kotlin
object UserAuth: PrincipalType<User, Uuid> {
    override val idSerializer = Uuid.serializer()
    override val subjectSerializer = User.serializer()
    override val name = "User"

    context(server: ServerRuntime)
    override suspend fun fetch(id: Uuid): User =
        database().table<User>().get(id) ?: throw NotFoundException()
}
```

Set up ModelInfo with permissions:

```kotlin
val userInfo: ModelInfo<User?, User, Uuid> = database.modelInfo(
    auth = UserAuth.require() or AuthRequirement.None,
    permissions = {
        val user = authOrNull?.fetch()
        val self = condition { it._id eqNn user?._id }
        val admin = if (user?.isSuperUser == true) Condition.Always else Condition.Never

        ModelPermissions(
            create = Condition.Never,
            read = Condition.Always,
            update = self or admin,
            delete = admin
        )
    }
)
```

Configure proof methods:

```kotlin
val pins = PinHandler(cache, "pins")
val proofEmail = path.path("proof").path("email") module
    EmailProofEndpoints(pins, email, { to, pin ->
        Email(subject = "Login Code", to = listOf(EmailAddressWithName(to)),
              plainText = "Your PIN is $pin")
    })
val proofPassword = path.path("proof").path("password") module
    PasswordProofEndpoints(database, cache)
```

Set up AuthEndpoints:

```kotlin
val auth = path.path("auth") module object: AuthEndpoints<User, Uuid>(
    principal = UserAuth,
    database = database
) {
    context(server: ServerRuntime)
    override suspend fun requiredProofStrengthFor(subject: User): Int = 5

    context(server: ServerRuntime)
    override suspend fun sessionExpiration(subject: User): Instant? = null
}
```

#### Advanced: Auth Caching with AuthCacheKey

Cache expensive authentication-derived data to avoid repeated database queries:

```kotlin
object UserAuth : PrincipalType<User, Uuid> {
    // ... other fields ...

    // Define what to pre-cache on session creation
    override val precache: List<AuthCacheKey<User, *>> = listOf(IsSuperUserCache, MembershipCache)

    // Simple cache example
    object IsSuperUserCache : AuthCacheKey<User, Boolean> {
        override val id: String = "super-user"
        override val serializer: KSerializer<Boolean> = Boolean.serializer()
        override val expireAfter: Duration = 5.minutes

        context(_: ServerRuntime)
        override suspend fun calculate(input: Authentication<User>): Boolean =
            input.fetch().isSuperUser

        // Extension functions for easy access
        context(_: ServerRuntime) suspend fun Authentication<User>.isSuperUser() = get(IsSuperUserCache)
        context(_: ServerRuntime) suspend fun AuthAccess<User>.isSuperUser() = auth.isSuperUser()
    }

    // Complex cache with related data
    object MembershipCache : AuthCacheKey<User, Set<ActiveMembership>> {
        override val id: String = "memberships"
        override val serializer = SetSerializer(ActiveMembership.serializer())
        override val expireAfter = 5.minutes

        context(_: ServerRuntime)
        override suspend fun calculate(input: Authentication<User>): Set<ActiveMembership> {
            val memberships = input.fetch().memberships
            val activeOrgs = OrganizationTable.getMany(memberships.map { it.organization })
                .filter { it.subscriptionActive }
                .map { it._id }
                .toSet()
            return memberships.map {
                ActiveMembership(it, it.organization in activeOrgs)
            }.toSet()
        }
    }

    // Derived caches (transform existing cache data efficiently)
    private fun <T, R : Any> AuthCacheKey<User, Set<T>>.derive(
        id: String,
        transform: (T) -> R?
    ): AuthCacheKey<User, Set<R>> = object : AuthCacheKey<User, Set<R>> {
        override val id = id
        override val serializer = SetSerializer(serializerOrContextual<R>())
        override val expireAfter = this@derive.expireAfter
        override val localOnly = true  // Not stored on token

        context(_: ServerRuntime)
        override suspend fun calculate(input: Authentication<User>) =
            input[this@derive].mapNotNullTo(HashSet(), transform)
    }

    // Chain derived caches
    val memberships = MembershipCache.derive("memberships-only") { it.membership }
    val activeMemberships = MembershipCache.derive("active") { it.takeIf { it.active }?.membership }
    val organizations = memberships.derive("orgs") { it.organization }
    val activeOrganizations = activeMemberships.derive("active-orgs") { it.organization }

    // Convenience extensions
    context(_: ServerRuntime) suspend fun AuthAccess<User>.activeOrganizations() = auth[activeOrganizations]
}
```

**Use in permissions:**

```kotlin
permissions = {
    if (auth.isSuperUser()) return@permissions ModelPermissions.allowAll()
    val orgs = auth.activeOrganizations()
    val orgCondition = condition { it.organization inside orgs }

    ModelPermissions(
        create = orgCondition,
        read = orgCondition,
        update = orgCondition,
        delete = orgCondition
    )
}
```

**Key benefits:**
- Pre-cached values stored on session token (fast access)
- Derived caches avoid redundant database queries
- `localOnly` caches save token space (computed on-demand)
- Automatic invalidation on cache expiry

#### Auto-User-Creation Pattern for Signup Flows

**⚠️ IMPORTANT**: For email-based authentication, you typically don't need custom signup endpoints. Users are automatically created on first login via `fetchByProperty`.

**Automatic User Creation on First Login:**

```kotlin
object UserAuth : PrincipalType<User, Uuid> {
    // ... other fields ...

    context(server: ServerRuntime)
    override suspend fun fetchByProperty(property: String, value: String): User? = when (property) {
        "email" -> UserEndpoints.info.table()
            .run {
                // Automatically creates user if doesn't exist
                findOne(condition { it.email eq value.toEmailAddress() })
                    ?: insertOne(User(email = value.toEmailAddress()))
            }
        else -> super.fetchByProperty(property, value)
    }
}
```

**This means:**
- When a user authenticates via email for the first time, a User record is created automatically
- No need for dedicated "signup" endpoints - just use existing EmailProofEndpoints
- User is created with minimal fields (just email), additional profile info added later

**Progressive Profile Completion Pattern:**

After auto-creation, guide users to complete their profile:

**Backend:**
```kotlin
@Serializable
data class User(
    override val _id: Uuid = Uuid.random(),
    val email: EmailAddress,
    val name: String = "No Name Specified",  // Default value
    val phone: String? = null,                // Optional
    val emergencyContact: String? = null,     // Optional
    val role: UserRole = UserRole.User,
) : HasId<Uuid>
```

**Frontend (KiteUI):**
```kotlin
// After successful authentication, check profile completion
reactive {
    val session = currentSession()
    if(session == null) {
        pageNavigator.reset(LandingPage())
    } else {
        val user = session.api.userAuth.getSelf()
        val needsProfileCompletion = user.name == "No Name Specified" ||
                                     user.phone == null ||
                                     user.emergencyContact == null
        if (needsProfileCompletion) {
            pageNavigator.reset(ProfileCompletionScreen(user.email.raw))
        }
    }
}
```

**Updating User Profile:**
```kotlin
// In ProfileCompletionScreen
onClick {
    launch {
        val session = currentSession.awaitNotNull()

        // Modify user with ModelCache helper
        session.users[session.userId].modify(modification {
            it.name assign fullName.value
            it.phone assign phone.value
            it.emergencyContact assign emergencyContact.value
        })

        pageNavigator.reset(HomePage())
    }
}
```

**Key imports for modification:**
```kotlin
import com.lightningkite.lightningdb.modification
import com.lightningkite.lskiteuistarter.name      // Auto-generated
import com.lightningkite.lskiteuistarter.phone     // Auto-generated
import com.lightningkite.lskiteuistarter.emergencyContact  // Auto-generated
```

**Why This Pattern Works:**
- ✅ No redundant signup endpoints (reuse existing auth)
- ✅ Users can authenticate immediately (frictionless onboarding)
- ✅ Profile completed progressively (better UX)
- ✅ Standard REST endpoints for updates (consistent API)

### 6. File Handling

```kotlin
val files = setting("files", PublicFileSystem.Settings())

// Upload (early binding)
val uploadEarly = path.path("upload") module
    UploadEarlyEndpoint(files, database, Runtime.Constant(listOf()))

// Get signed URL
val getFile = path.path("files").arg<String>("path").get bind HttpHandler {
    val filePath = it.arg1
    val fileRef = files().root.then(filePath)
    HttpResponse.plainText(fileRef.signedUrl)
}
```

### 7. WebSockets

```kotlin
val topic = path.path("topic").topic(Message.serializer())

val socket = path.path("ws") bind WebSocketHandler(
    willConnect = { Uuid.random().toString() },
    didConnect = {
        subscribe(topic)
        send(WelcomeMessage())
    },
    messageFromClient = {
        topic.send(Message(currentState, it.content))
    },
    topicHandlers = {
        topic bind { send(it.value) }
    },
    disconnect = {
        println("Disconnected: $currentState")
    }
)
```

### 8. Background Tasks

```kotlin
// Define task
val emailTask = path.path("tasks").path("email") bind Task { input: EmailRequest ->
    println("Sending email to ${input.to}")
    delay(1000)
    email().send(Email(subject = input.subject, to = listOf(EmailAddressWithName(input.to)),
                       plainText = input.body))
}

// Invoke task
val sendEmail = path.path("send-email").post bind HttpHandler { request ->
    emailTask.invoke(EmailRequest(request.body!!.text()))
    HttpResponse.plainText("Email queued")
}

// Scheduled task
val cleanup = path.path("scheduled-cleanup") bind ScheduledTask(
    frequency = 1.hours
) {
    println("Running cleanup...")
    database().table<OldData>().deleteMany(condition {
        it.createdAt lt Clock.System.now() - 30.days
    })
}

// Tentative reservation expiration pattern
val expireStaleReservations = path.path("scheduled").path("expire-reservations") bind ScheduledTask(
    frequency = 1.minutes
) {
    val now = Clock.System.now()
    val expiredCount = ReservationEndpoints.info.table().updateMany(
        condition {
            // Find tentative (has expireAt) reservations that have expired
            it.expireAt.notNull and
            (it.expireAt lt now) and
            it.cancelledAt.eq(null)
        },
        modification {
            it.cancelledAt assign now  // Mark as cancelled
        }
    )
    if (expiredCount > 0) {
        println("Expired $expiredCount stale tentative reservations")
    }
}
```

### 9. Caching

```kotlin
val cache = setting("cache", Cache.Settings())

// Set with expiration
cache().set("key", "value", expire = 5.minutes)

// Get
val value = cache().get<String>("key")

// Remove
cache().remove("key")

// Cache-aside pattern
suspend fun getExpensiveData(id: String): Data {
    val cached = cache().get<Data>("data:$id")
    if (cached != null) return cached

    val fresh = database().table<Data>().get(id)
    cache().set("data:$id", fresh, expire = 10.minutes)
    return fresh
}
```

### 9.5. Migrations and Startup Tasks

**Idempotent Migrations with `doOnce`:**

```kotlin
context(_: ServerRuntime)
suspend fun runMigrations() {
    // doOnce ensures migration only runs once, tracked in database
    doOnce("migrate-categories", database) {
        logging("migrate-categories") {
            migrateCategories()
        }
    }
    doOnce("migrate-icons", database) { migrateIcons() }
}

context(_: ServerRuntime)
suspend fun migrateCategories() {
    val projects = database().table<Project>().all().toList()
    for (project in projects) {
        database().table<Project>().updateOne(
            condition { it._id eq project._id },
            modification { it.categories assign project.legacyCategories.toSet() }
        )
    }
}
```

**Startup Initialization:**

```kotlin
// In Server object - runs once on server startup
init {
    path.path("setUpAdmins") bind startupOnce(database) {
        defaultData()  // Creates default admin users, organizations, etc.
    }
}

context(_: ServerRuntime)
suspend fun defaultData() {
    val adminUser = database().table<User>().insertOne(
        User(email = "admin@example.com", isSuperUser = true)
    )
    database().table<Organization>().insertOne(
        Organization(name = "Default Org")
    )
}
```

**Benefits:**
- `doOnce` tracks completion in database - safe for restarts
- Runs automatically on server startup
- Can be triggered manually via endpoint or CLI
- Idempotent - safe to call multiple times

### 10. Testing

**Practical Testing Patterns from Real Projects:**

```kotlin
class ServerTest {
    init {
        TestSettings  // Initialize test database, settings
    }

    @Test
    fun basicTest(): Unit = runBlocking {
        // Direct table access for test setup
        val org = Server.organizations.info.table()
            .insertOne(Organization(name = "Test"))!!

        val user = Server.users.info.table()
            .insertOne(User(
                email = "${Random.nextInt()}@test.com",
                memberships = setOf(Membership(org._id, UserRole.Owner))
            ))!!

        val project = Server.projects.info.table()
            .insertOne(Project(
                name = "Test Project",
                organization = org._id
            ))!!

        // Test endpoint with .test() helper
        val result = Server.projects.someEndpoint.test(user, ProjectInput(...))

        // Verify denormalized fields work correctly
        assertEquals("Test Project", result.name)
        assertEquals(org.name, result.organizationName)
    }

    @Test
    fun testFcmTokenRegistration(): Unit = runBlocking {
        val user = createTestUser()
        Server.fcmTokens.registerEndpoint.test(user, "some-fcm-token")
        assertEquals(user._id, Server.fcmTokens.info.table().get("some-fcm-token")!!.user)
    }
}
```

**⚠️ CRITICAL: Build Server Once Per Test Suite**

When writing tests, ensure `Server.build()` is only called once across all tests to avoid `DuplicateRegistrationError`. Create a shared `TestHelper`:

```kotlin
// TestHelper.kt - shared across all test files
object TestHelper {
    val testRunner by lazy { TestRunner(Server.build()) }
}

// In your test file
class ServerTest {
    init {
        JsonFileDatabase  // Ensure mock implementations are loaded
    }

    @Test
    fun testEndpoint() = runBlocking {
        with(TestHelper.testRunner) {
            val response = Server.someEndpoint.test()
            assertEquals("expected", response.body!!.text())
        }
    }
}
```

**Test Method Signatures**

For basic `HttpHandler` endpoints:
```kotlin
// No path args
Server.endpoint.test(
    queryParameters = QueryParameters(listOf("key" to "value")),
    body = TypedData.text("content", MediaType.Text.Plain)
)

// With path args
Server.endpoint.test(
    "pathArg1",
    42,  // pathArg2
    queryParameters = QueryParameters.EMPTY
)
```

For `ApiHttpHandler` endpoints:
```kotlin
// No path args
Server.typedEndpoint.test(auth = null, input = RequestData(...))

// With path args
Server.typedEndpoint.test("pathArg", auth = null, input = RequestData(...))
```

**Common Testing Pitfalls**

⚠️ **Duplicate UploadEarlyEndpoint Declarations**

If you create multiple instances of `UploadEarlyEndpoint` (e.g., in different modules or endpoints), they will have **conflicting declarations for how `ServerFile` is serialized**. This causes runtime serialization errors that manifest as `500 Internal Server Error` responses in tests, even though the code compiles successfully.

**Solution:** Only instantiate `UploadEarlyEndpoint` once in your server definition:

```kotlin
object Server : ServerBuilder() {
    // ✅ Good - single instance
    val uploadEarly = path.path("upload") module
        UploadEarlyEndpoint(files, database, Runtime.Constant(listOf()))

    // ❌ Bad - creates duplicate with conflicting ServerFile serialization
    // val anotherUpload = path.path("upload2") module
    //     UploadEarlyEndpoint(files, database, Runtime.Constant(listOf()))
}
```

If you need multiple upload endpoints, reuse the same `UploadEarlyEndpoint` instance or use different endpoint patterns.

## Testing Infrastructure Setup

**Setting Up a Complete Testing Environment:**

When setting up testing infrastructure for multi-project development, follow these patterns to avoid conflicts and streamline workflows:

### Port Configuration

**Pattern:** Use isolated ports for each project to avoid conflicts. Document ports in testing configuration files.

```json
// testing/settings.testing.json
{
  "general": {
    "publicUrl": "http://localhost:8082",  // Backend port
    "wsUrl": "ws://localhost:8082",
    "debug": true
  },
  "ktorRunConfig": {
    "host": "0.0.0.0",
    "port": 8082  // Backend port
  }
}
```

```javascript
// apps/vite.config.mjs (for frontend)
export default {
  server: {
    port: 8942,  // Frontend port
    proxy: {
      '/api': {
        target: 'http://localhost:8082',  // Backend port
        ws: true
      }
    }
  }
}
```

**Shell Scripts:**
- Update all testing scripts with correct ports
- Create start-backend.sh, start-frontend.sh, stop-all.sh, start-all.sh
- Document in testing/README.md

### Debug Admin Token Feature

**Pattern:** Auto-generate admin session token when debug mode is enabled for testing.

```kotlin
import com.lightningkite.lightningserver.definition.generalSettings
import kotlin.uuid.Uuid

object Server : ServerBuilder() {
    // ... other code ...

    // Debug admin token - prints on server startup when debug=true
    val debugAdminToken = path.path("debug-admin-token") bind StartupTask {
        if (generalSettings().debug) {
            // Create session for a specific admin user (ID: 0L, 10L)
            val token = UserAuth.session.createSession(Uuid.fromLongs(0L, 10L)).second
            println("Admin token: '$token'")
        }
    }
}
```

**Key points:**
- Import is `com.lightningkite.lightningserver.definition.generalSettings`, NOT `settings.generalSettings`
- Use `StartupTask` to run code once on server startup
- Token is printed to console for easy capture and use in testing

**Testing workflow:**
1. Start backend server
2. Script captures printed token from logs (e.g., `grep "Admin token:"`)
3. Save to `.admin-token` file
4. Browser testing script injects token into localStorage

### Custom Endpoints on ModelRestEndpoints

**Pattern:** Add custom endpoints to the REST path structure using `bind ApiHttpHandler`.

```kotlin
object PendingInputEndpoints : ServerBuilder() {
    val info = Server.database.modelInfo<User, PendingInput, Uuid>(...)

    val rest = path include ModelRestEndpoints(info)

    // Custom endpoint: POST /pendingInput/respond
    val respond = path.path("respond").post bind ApiHttpHandler(
        summary = "Respond to Input",
        description = "Provides a response to a pending input request",
        auth = UserAuth.require(),
        implementation = { request: RespondInputRequest ->
            // Validation
            val input = info.table().get(request.inputId)
                ?: throw NotFoundException("Input request not found")

            // Security check
            val isAdmin = auth.userRole() >= UserRole.Admin
            if (input.ownerId != auth.id && !isAdmin) {
                throw ForbiddenException("You do not own this input request")
            }

            // Business logic
            if (input.status != InputStatus.PENDING) {
                throw BadRequestException("Input already resolved")
            }

            // Update database
            info.table().updateOneById(request.inputId, modification<PendingInput> {
                it.status assign InputStatus.RESOLVED
                it.response assign request.response
                it.resolvedAt assign Clock.System.now()
            })

            Unit
        }
    )
}
```

**Request data class in shared models:**
```kotlin
@Serializable
data class RespondInputRequest(
    val inputId: Uuid,
    val response: String
)
```

**After adding custom endpoint:**
1. Regenerate SDK: `./gradlew :server:generateSdk`
2. Generated method will be available in frontend SDK (e.g., `api.pendingInput.respondToInput(...)`)
3. Frontend can call the endpoint through type-safe SDK

### Common Pitfalls

**❌ Wrong generalSettings import:**
```kotlin
// ❌ WRONG - compile error
import com.lightningkite.lightningserver.settings.generalSettings

// ✅ CORRECT
import com.lightningkite.lightningserver.definition.generalSettings
```

**❌ Frontend MJS reference mismatch:**

When using Kotlin/JS with Vite, the generated JavaScript bundle name is based on the **project name**, not the template name.

```html
<!-- index.html -->
<!-- ❌ WRONG - if project name is "claude-coordinator" -->
<script src="/ls-kiteui-starter-apps.mjs" type="module"></script>

<!-- ✅ CORRECT - must match project name from settings.gradle.kts -->
<script src="/claude-coordinator-apps.mjs" type="module"></script>
```

**Debugging frontend blank page:**
1. Check browser console for "Failed to load url /xxx-apps.mjs" errors
2. Verify project name in `settings.gradle.kts` (rootProject.name)
3. Update index.html to reference `/${projectName}-apps.mjs`
4. Generated MJS file will match the rootProject.name

**❌ Calling non-existent SDK methods:**

After adding custom endpoints, you MUST regenerate the SDK before using them in the frontend:

```bash
# Always regenerate after changing server endpoints
./gradlew :server:generateSdk
```

SDK generation creates type-safe methods in the frontend API client. Without regeneration, you'll get compilation errors when trying to call new endpoints.

### Browser Testing with Chrome MCP

For end-to-end testing, use Claude's Chrome MCP tools rather than separate Playwright/Selenium setups:

```bash
# testing/prepare-browser-test.sh
#!/bin/bash

# Start backend
./testing/start-backend.sh

# Wait for backend
while ! curl -s http://localhost:8082 > /dev/null; do
    sleep 1
done

# Capture admin token from logs
TOKEN=$(grep "Admin token:" server.log | cut -d"'" -f2)
echo "$TOKEN" > testing/.admin-token

# Start frontend
./testing/start-frontend.sh

echo "Ready for browser testing at http://localhost:8942"
```

**Testing workflow:**
1. Run prepare script to start servers and capture token
2. Use Chrome MCP tools to navigate and test
3. Inject token into localStorage for authenticated testing
4. Take screenshots to verify UI rendering

## Common Patterns

### REST PATCH with Modifications

**⚠️ CRITICAL:** REST PATCH endpoints ALWAYS use `Modification<T>` wrapper, never raw values.

**To update a field via PATCH:**
```bash
# Set a field to a value
PATCH /model/rest/{id}
{"fieldName": {"Assign": "newValue"}}

# Set an optional field to null (clear it)
PATCH /model/rest/{id}
{"fieldName": {"Assign": null}}

# Increment a number
PATCH /model/rest/{id}
{"count": {"Increment": 5}}

# Multiple fields at once
PATCH /model/rest/{id}
{
  "Chain": [
    {"firstName": {"Assign": "John"}},
    {"age": {"Increment": 1}}
  ]
}
```

**Common Use Cases:**

**Confirm a tentative reservation (clear expireAt):**
```bash
PATCH /reservation/rest/{id}
{"expireAt": {"Assign": null}}
```

**Cancel a reservation (set cancelledAt):**
```bash
PATCH /reservation/rest/{id}
{"cancelledAt": {"Assign": "2025-12-20T00:00:00Z"}}
```

**Check in to a session (set checkedInAt):**
```bash
PATCH /reservation/rest/{id}
{"checkedInAt": {"Assign": "2025-12-20T14:00:00Z"}}
```

**See full Modification types documentation:**
- `/Users/jivie/Projects/lightning-server/docs/use-as-client.md` - Lines 127-205
- `Assign`, `Increment`, `ListAppend`, `SetAppend`, `Chain`, etc.

**Why this matters:**
- ❌ `{"expireAt": null}` → JSON parse error
- ✅ `{"expireAt": {"Assign": null}}` → Works correctly
- Error message is cryptic because it comes from JSON parser itself
- Always wrap field updates in Modification objects

### Organizing Endpoints

Group related endpoints into ServerBuilder objects:

```kotlin
object ApiEndpoints : ServerBuilder() {
    val posts = path.path("posts") include PostsEndpoints
    val comments = path.path("comments") include CommentsEndpoints
}

object Server : ServerBuilder() {
    val api = path.path("api") include ApiEndpoints
}
```

### Error Handling

Use standard exceptions:

```kotlin
throw BadRequestException("Invalid input")
throw NotFoundException("Resource not found")
throw UnauthorizedException("Auth required")
throw ForbiddenException("Access denied")
```

### Accessing Services

Services are accessed through settings:

```kotlin
object Server : ServerBuilder() {
    val database = setting("database", Database.Settings())

    val endpoint = path.get bind HttpHandler {
        val db = database()
        // Use db...
    }
}
```

### Path Reference Pattern

Always store endpoint references for testing and internal calls:

```kotlin
object Server : ServerBuilder() {
    val createPost = path.path("posts").post bind ApiHttpHandler { ... }
    val getPost = path.path("posts").arg<Uuid>("id").get bind ApiHttpHandler { ... }

    // Can reference: Server.createPost, Server.getPost
}
```

## Troubleshooting

### Common Import Errors

When you see "Unresolved reference" errors for database operations or datetime utilities, you're likely missing imports:

```kotlin
// Database operation imports
import com.lightningkite.services.database.get
import com.lightningkite.services.database.find
import com.lightningkite.services.database.lt
import com.lightningkite.services.database.lte
import com.lightningkite.services.database.gte
import com.lightningkite.services.database.eq
import com.lightningkite.services.database.neq
import com.lightningkite.services.database.condition
import com.lightningkite.services.database.modification

// DateTime operation imports (kotlinx.datetime, not kotlin.time)
import kotlinx.datetime.DateTimeUnit
import kotlinx.datetime.toLocalDateTime
import kotlinx.datetime.atStartOfDayIn
import kotlinx.datetime.plus
import kotlinx.datetime.minus

// Flow operations
import kotlinx.coroutines.flow.toList
```

**Common mistakes:**
- Using `kotlin.time.DateTimeUnit` instead of `kotlinx.datetime.DateTimeUnit`
- Using `.value` on `DayOfWeek` (doesn't exist) - use `.ordinal` instead (0-6 where Monday=0)
- Forgetting to import database query operators like `lt`, `gte`, etc.

### Type Inference Issues with DataClassPath

If you see "Cannot infer type parameter 'ROOT'" errors, ensure you're using generated path constants correctly:

```kotlin
// ✅ Correct - using generated path constant
condition { it.user eq userId }

// ❌ Wrong - missing context or incorrect path usage
condition { User_email eq "test@example.com" }  // Need it.email, not User_email
```

### Denormalization Not Working

If denormalized fields aren't updating:
1. Check that denormalize is in the `signals` block
2. Verify the source table and field are correct
3. Remember: fields aren't populated in `interceptCreate` - they update after insert

## Build System

Lightning Server uses Gradle with Kotlin Multiplatform:

### Common Commands

```bash
# Build all modules
./gradlew build

# Run tests
./gradlew check

# Run demo server
./gradlew :demo:run --args="serve"

# Generate SDK
./gradlew :demo:run --args="sdk"

# Publish to local Maven
./gradlew publishToMavenLocal
```

### Module Structure

Projects typically have paired modules:
- `module` - JVM-only code (server implementation)
- `module-shared` - Multiplatform code (shared models, DTOs)

## Deployment

### Engines

Lightning Server supports multiple engines:

- `engine-local` - For unit testing
- `engine-ktor` - Ktor HTTP server (dev/prod)
- `engine-netty` - Netty HTTP server
- `engine-jdk-server` - Pure JDK HTTP server
- `engine-aws-serverless` - AWS Lambda with Terraform generation

### AWS Deployment

The AWS engine auto-generates Terraform:

```kotlin
fun main() {
    val built = Server.build()
    AwsHandler(built).apply {
        settings.loadFromFile(KFile("settings.json"))
        // Generates terraform/ directory
    }
}
```

### Settings Management

First run generates `settings.json`:

```json
{
  "database": {
    "url": "mongodb://localhost:27017/mydb"
  },
  "cache": {
    "url": "redis://localhost:6379"
  }
}
```

## Best Practices

1. **Use ModelRestEndpoints for CRUD** - Don't manually create database CRUD endpoints; use ModelRestEndpoints
2. **Settings File Works Out-of-Box** - Generated settings should allow immediate running
3. **Use Service Abstractions** - Don't depend on specific implementations
4. **Test with Mocks** - Use JsonFileDatabase, RAM cache for tests
5. **Store Endpoint References** - Keep constants for all endpoints
6. **Group Endpoints Logically** - Use ServerBuilder objects
7. **Type Safety** - Use @GenerateDataClassPaths on all database models
8. **Document Typed Endpoints** - Add summaries and descriptions
9. **Read This Skill First** - When working on Lightning Server files, read this skill to understand patterns

## Anti-Patterns

❌ **Don't manually create CRUD endpoints** - Use ModelRestEndpoints instead
❌ **Don't create multiple UploadEarlyEndpoint instances** - Causes ServerFile serialization conflicts
❌ **Don't call Server.build() multiple times in tests** - Use shared TestHelper with lazy initialization
❌ **Don't assume denormalized fields are set in interceptCreate** - They're not populated yet!
❌ Don't access database implementations directly
❌ Don't hardcode configuration
❌ Don't skip endpoint reference storage
❌ Don't forget @GenerateDataClassPaths
❌ Don't use plain HttpHandler for APIs (use typed endpoints)
❌ Don't test against real services
❌ Don't write manual list/get/create/update/delete endpoints when ModelRestEndpoints can do it
❌ Don't use `kotlin.time.DateTimeUnit` - use `kotlinx.datetime.DateTimeUnit`

## Key Files to Reference

- `demo/src/main/kotlin/.../Server.kt` - Comprehensive example
- `docs/setup.md` - Project setup
- `docs/endpoints.md` - Endpoint patterns
- `docs/typed-endpoints.md` - Typed API docs
- `docs/database.md` - Database usage
- `docs/authentication.md` - Auth setup

## Getting Help

When stuck:
1. **Read this skill** - Most common patterns are documented here
2. Check the demo server for examples
3. Review relevant docs in `/docs`
4. Look at existing endpoint implementations
5. Check test files for usage patterns
6. Examine the CLAUDE.md file for project-specific guidance

## Usage

**IMPORTANT:** Invoke this skill proactively when you:
- Open any file with `import com.lightningkite.lightningserver.*`
- Work on Lightning Server endpoints
- Set up authentication
- Work with databases
- Handle file uploads
- Implement WebSockets
- Create background tasks
- Write tests
- Deploy to AWS
- See compilation errors related to Lightning Server

Simply say: "Help me build a Lightning Server endpoint" or "How do I set up auth in Lightning Server?"
