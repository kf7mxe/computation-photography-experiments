---
name: Service Abstractions
description: This skill should be used when the user asks about Service Abstractions library, infrastructure service management, or database/cache/file system configurations in Kotlin
version: 1.0.0
---

# Service Abstractions Skill

This skill provides expertise in using the Service Abstractions Kotlin Multiplatform library for infrastructure service management.

## When to Use This Skill

Invoke this skill when working with:
- Service Abstractions library code
- Database, cache, file system, or other infrastructure service configurations
- Type-safe database queries using generated DataClassPath
- Settings-based service instantiation
- Multiplatform Kotlin projects using service abstractions

## Core Concepts

### Settings Pattern

Service Abstractions uses a settings-based configuration where services are instantiated from serializable configuration objects:

```kotlin
@Serializable
data class ServerSettings(
    val database: Database.Settings = Database.Settings("mongodb://localhost"),
    val cache: Cache.Settings = Cache.Settings("redis://localhost:6379"),
    val files: PublicFileSystem.Settings = PublicFileSystem.Settings("s3://bucket/path")
)

// Instantiate services with a context
val context = SettingContext(
    internalSerializersModule = module,
    otel = otel,
    // ... other shared resources
)

val db = settings.database("main-db", context)
val cache = settings.cache("app-cache", context)
val files = settings.files("uploads", context)
```

### Service Interface

All services implement the `Service` interface:

```kotlin
interface Service {
    val name: String
    val context: SettingContext
    suspend fun connect()
    suspend fun disconnect()  // Critical for serverless (Lambda, SnapStart)
    val healthCheckFrequency: Duration
    suspend fun healthCheck(): HealthStatus
}
```

**Important**: Always call `connect()` after instantiation and `disconnect()` when done, especially in serverless environments.

### URL-Based Configuration

Services are configured using URL schemes:

**In-Memory/Testing:**
- `ram://` or `ram` - In-memory implementation

**Databases:**
- `mongodb://host:port/database`
- `postgresql://host:port/database`
- `jsonfile://path/to/file.json`

**Caches:**
- `redis://host:port`
- `memcached://host:port`
- `dynamodb://region/table-name`

**File Systems:**
- `file://path/to/directory`
- `s3://bucket-name/prefix`

**Testing Suffix:**
- URLs ending in `-test` (e.g., `mongodb-test://`) start ephemeral instances for testing

## Database Usage

### Defining Models

```kotlin
@GenerateDataClassPaths  // Enables type-safe queries
@Serializable
data class User(
    override val _id: UUID = UUID.random(),
    @Index(unique = true) val email: String,
    @Index val age: Int,
    val name: String,
    val friends: List<UUID>,
    val metadata: Map<String, String> = emptyMap()
) : HasId<UUID>
```

**Key Annotations:**
- `@GenerateDataClassPaths` - Generates field path objects for type-safe queries
- `@Index` - Creates database index (supports `unique` parameter)
- Model must implement `HasId<T>` with `_id` field

### Code Generation

After defining models, run KSP to generate DataClassPath objects:

```bash
./gradlew :your-module:kspCommonMainKotlinMetadata
```

This generates `User.path.email`, `User.path.age`, etc. for type-safe queries.

### Getting a Table

```kotlin
val db: Database = settings.database("my-db", context)
await db.connect()

val userTable: Table<User, UUID> = db.collection<User, UUID>()
// Or with explicit name:
val userTable = db.collection<User, UUID>("users")
```

### CRUD Operations

```kotlin
// Insert
val user = User(email = "john@example.com", age = 30, name = "John")
userTable.insert(user)
userTable.insertMany(listOf(user1, user2))

// Find
val allUsers = userTable.findAll()
val adults = userTable.find(condition = User.path.age greaterThan 18)
val john = userTable.get(userId)  // By ID
val byEmail = userTable.findOne(condition = User.path.email eq "john@example.com")

// Update
userTable.updateOne(
    condition = User.path.email eq "john@example.com",
    modification = User.path.age assign 31
)

userTable.updateMany(
    condition = User.path.age lessThan 18,
    modification = User.path.metadata.mapSet("status", "minor")
)

// Delete
userTable.deleteOne(condition = User.path.email eq "john@example.com")
userTable.deleteMany(condition = User.path.age lessThan 13)
```

### Type-Safe Query Conditions

The generated DataClassPath objects provide type-safe query builders:

```kotlin
// Comparison
User.path.age eq 30
User.path.age notEq 30
User.path.age greaterThan 18
User.path.age lessThan 65
User.path.age greaterThanOrEqual 21
User.path.age lessThanOrEqual 64

// String operations
User.path.email contains "@example.com"
User.path.name startsWith "John"
User.path.name endsWith "Smith"
User.path.email regex ".*@example\\.com"

// Logical operations
(User.path.age greaterThan 18) and (User.path.age lessThan 65)
(User.path.email eq "john@example.com") or (User.path.email eq "jane@example.com")
not(User.path.age lessThan 18)

// List operations
User.path.friends.any eq friendId
User.path.friends.all greaterThan someId
User.path.friends.sizesEquals 5

// Existence checks
User.path.metadata.exists()
User.path.metadata.notExists()

// Map operations
User.path.metadata.mapGet("status") eq "active"
```

### Type-Safe Modifications

```kotlin
// Assignment
User.path.age assign 31

// Increment/decrement
User.path.age plusAssign 1
User.path.age minusAssign 1

// List operations
User.path.friends.addAll(listOf(id1, id2))
User.path.friends.removeAll(listOf(id3))
User.path.friends.dropFirst()
User.path.friends.dropLast()

// Map operations
User.path.metadata.mapSet("status", "active")
User.path.metadata.mapRemove("oldKey")

// Combine multiple modifications
listOf(
    User.path.age assign 31,
    User.path.metadata.mapSet("updated", "true")
)
```

### Aggregations

```kotlin
// Count by age
val countByAge: Map<Int, Int> = userTable.groupAggregate(
    aggregate = Aggregate.Count,
    groupBy = User.path.age
)

// Sum ages by email domain
val sumAgesByDomain = userTable.groupAggregate(
    aggregate = Aggregate.Sum(User.path.age),
    groupBy = User.path.email  // Would need to extract domain in real code
)

// Average age
val avgAge = userTable.aggregate(
    aggregate = Aggregate.Average(User.path.age)
)
```

### Sorting and Limits

```kotlin
// Sort by age descending
val sorted = userTable.find(
    condition = Condition.Always(),
    orderBy = listOf(SortPart(User.path.age, false))  // false = descending
)

// Limit and skip
val page = userTable.find(
    condition = Condition.Always(),
    orderBy = listOf(SortPart(User.path.age)),
    skip = 20,
    limit = 10
)
```

## Cache Usage

```kotlin
val cache: Cache = settings.cache("my-cache", context)
await cache.connect()

// Set with expiration
cache.set("key", "value", Duration.parse("1h"))

// Get
val value: String? = cache.get("key")

// Delete
cache.remove("key")

// Add (only if not exists)
cache.add("key", "value", Duration.parse("30m"))

// Clear all
cache.clear()
```

## File System Usage

```kotlin
val fs: PublicFileSystem = settings.files("uploads", context)
await fs.connect()

// Upload
val fileId = fs.upload(
    path = FileRef.Known("users/avatar.jpg"),
    content = byteArray
)

// Download
val content: ByteArray = fs.get(FileRef.Known("users/avatar.jpg"))

// Delete
fs.delete(FileRef.Known("users/avatar.jpg"))

// List
val files = fs.list(FileRef.Known("users/"))

// Get public URL (if supported)
val url = fs.url(FileRef.Known("users/avatar.jpg"))
```

## Testing Patterns

### Shared Test Suites

Service Abstractions uses shared test suites to ensure all implementations satisfy the same contract:

```kotlin
// In your test file
class MyMongoDbTests : ConditionTests() {
    override val database: Database = Database.Settings(
        "mongodb-test://localhost"
    )(
        "test-db",
        testContext
    )

    init {
        runBlocking { database.connect() }
    }
}
```

The abstract test class (`ConditionTests`, `CacheTests`, etc.) contains comprehensive tests that all implementations must pass.

### Test Databases

Use `-test` suffix URLs for ephemeral test instances:
- `mongodb-test://` - Starts temporary MongoDB
- `redis-test://` - Starts temporary Redis
- `ram://` - In-memory (no setup needed)

## Common Patterns

### Service Lifecycle in Applications

```kotlin
class MyApplication(settings: ServerSettings) {
    private val context = SettingContext(...)
    private val database = settings.database("main", context)
    private val cache = settings.cache("main", context)

    suspend fun start() {
        database.connect()
        cache.connect()
    }

    suspend fun stop() {
        cache.disconnect()
        database.disconnect()
    }
}
```

### Serverless (AWS Lambda, SnapStart)

Services support disconnect/reconnect for serverless environments:

```kotlin
class LambdaHandler {
    private val db = settings.database("lambda-db", context)

    init {
        runBlocking { db.connect() }
    }

    // Called before snapshot
    fun beforeCheckpoint() {
        runBlocking { db.disconnect() }
    }

    // Called after restore
    fun afterRestore() {
        runBlocking { db.connect() }
    }
}
```

### Custom Serialization

Register custom serializers in the context:

```kotlin
val module = SerializersModule {
    contextual(MyCustomType::class, MyCustomTypeSerializer)
}

val context = SettingContext(
    internalSerializersModule = module,
    // ...
)
```

## Module Structure

- **basis/** - Core interfaces (Setting, Service, SettingContext)
- **database/** - Database abstraction + InMemoryDatabase
- **database-mongodb/** - MongoDB implementation
- **database-postgres/** - PostgreSQL implementation
- **database-jsonfile/** - JSON file database
- **cache/** - Cache abstraction + MapCache
- **cache-redis/** - Redis implementation
- **cache-memcached/** - Memcached implementation
- **cache-dynamodb/** - DynamoDB implementation
- **files/** - File system abstraction
- **files-s3/** - S3 implementation
- **email/** - Email service abstraction
- **email-javasmtp/** - SMTP implementation
- **sms/** - SMS abstraction
- **sms-twilio/** - Twilio implementation

## Common Pitfalls

1. **Forgetting to call connect()**: Services must be connected before use
2. **Not generating DataClassPath**: Run KSP after adding/changing models
3. **Missing @GenerateDataClassPaths**: Required for type-safe queries
4. **Not implementing HasId**: Database models must have `_id` field
5. **Incorrect URL schemes**: Check supported schemes for each service
6. **Missing SerializersModule**: Custom types need registered serializers
7. **Platform compatibility**: Not all implementations support all platforms (e.g., MongoDB is JVM-only)

## Build Commands

```bash
# Generate DataClassPath for models
./gradlew :your-module:kspCommonMainKotlinMetadata

# Run tests
./gradlew jvmTest                    # Fast JVM tests
./gradlew allTests                   # All platform tests
./gradlew :module:jvmTest            # Specific module

# Build
./gradlew build                      # All modules
./gradlew :module:build              # Specific module

# Publish locally (for testing)
./gradlew publishToMavenLocal
```

## OpenTelemetry Integration

Services automatically integrate with OpenTelemetry when provided in context:

```kotlin
val otel = OpenTelemetry.noop()  // Or real implementation
val context = SettingContext(
    otel = otel,
    // ...
)

// All database/cache operations are automatically traced
```

## Health Checks

All services support health checks:

```kotlin
val status: HealthStatus = database.healthCheck()

when (status) {
    is HealthStatus.Healthy -> println("Service is healthy")
    is HealthStatus.Unhealthy -> println("Service unhealthy: ${status.message}")
}

// Services declare their preferred check frequency
val frequency: Duration = database.healthCheckFrequency
```

## Best Practices

1. **Use Settings objects**: Makes configuration serializable and testable
2. **Leverage type-safe queries**: Use generated DataClassPath instead of raw conditions
3. **Add indexes**: Use `@Index` annotation on frequently queried fields
4. **Test with shared suites**: Extend abstract test classes to ensure compatibility
5. **Handle disconnect**: Implement proper lifecycle management for serverless
6. **Use ram:// for tests**: Fast in-memory implementations for unit tests
7. **Register serializers early**: Set up SerializersModule before creating context
8. **Check platform support**: Verify your target platforms support the implementation

## Example: Complete Application

```kotlin
@Serializable
data class AppSettings(
    val database: Database.Settings = Database.Settings("mongodb://localhost"),
    val cache: Cache.Settings = Cache.Settings("redis://localhost:6379")
)

@GenerateDataClassPaths
@Serializable
data class Post(
    override val _id: UUID = UUID.random(),
    @Index val authorId: UUID,
    val title: String,
    val content: String,
    @Index val createdAt: Long = Clock.System.now().toEpochMilliseconds()
) : HasId<UUID>

class BlogService(settings: AppSettings) {
    private val context = SettingContext(...)
    private val database = settings.database("blog", context)
    private val cache = settings.cache("blog", context)

    private lateinit var posts: Table<Post, UUID>

    suspend fun start() {
        database.connect()
        cache.connect()
        posts = database.collection()
    }

    suspend fun createPost(authorId: UUID, title: String, content: String): Post {
        val post = Post(authorId = authorId, title = title, content = content)
        posts.insert(post)
        return post
    }

    suspend fun getPostsByAuthor(authorId: UUID): List<Post> {
        return posts.find(
            condition = Post.path.authorId eq authorId,
            orderBy = listOf(SortPart(Post.path.createdAt, ascending = false))
        )
    }

    suspend fun getPost(id: UUID): Post? {
        // Try cache first
        val cached = cache.get<Post>("post:$id")
        if (cached != null) return cached

        // Fall back to database
        val post = posts.get(id)
        if (post != null) {
            cache.set("post:${post._id}", post, Duration.parse("1h"))
        }
        return post
    }

    suspend fun stop() {
        cache.disconnect()
        database.disconnect()
    }
}
```

## Quick Reference

### Must Remember
- Call `connect()` after service instantiation
- Run KSP after model changes: `./gradlew :module:kspCommonMainKotlinMetadata`
- Models need `@GenerateDataClassPaths` and `HasId<T>`
- Use `-test` suffix URLs for ephemeral test services
- Check platform compatibility for implementations

### Query Cheat Sheet
```kotlin
// Conditions
field eq value
field notEq value
field greaterThan value
field lessThan value
field.any eq value        // List contains
field.exists()            // Field exists

// Modifications
field assign value
field plusAssign 1
field.addAll(list)
field.mapSet("key", "val")

// Combining
condition1 and condition2
condition1 or condition2
not(condition)
```
