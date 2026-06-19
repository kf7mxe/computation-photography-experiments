---
name: Gradle Library Source Reader
description: This skill should be used when you need to read source code from Maven/Gradle dependencies to understand how a library works internally
version: 1.0.0
---

# Gradle Library Source Reader Skill

This skill allows you to download and extract source JARs from Maven Central and other repositories to inspect library internals.

## How It Works

Maven artifacts typically have a `-sources.jar` classifier that contains the original source code. This skill downloads and extracts these to `~/.extractlibs/src/<artifact>-<version>/` for inspection.

## Usage

To download and extract library sources, run the following bash command:

```bash
# Format: group:artifact:version
~/.extractlibs/download-lib-sources.sh "io.ktor:ktor-client-core:3.3.2"
```

The script will:
1. Download the sources JAR from Maven Central (or other configured repos)
2. Extract it to `~/.extractlibs/src/<artifact>-<version>/`
3. Report what was extracted

## After Extraction

Once extracted, you can search and read the library source code:

```bash
# Find all Kotlin files
find ~/.extractlibs/src/ktor-client-core-3.3.2 -name "*.kt"

# Search for specific patterns
grep -r "suspend fun" ~/.extractlibs/src/ktor-client-core-3.3.2/

# Read specific files
cat ~/.extractlibs/src/ktor-client-core-3.3.2/io/ktor/client/HttpClient.kt
```

## Common Libraries

Here are coordinates for common Kotlin/Java libraries:

### Kotlin Standard Libraries
- `org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2`
- `org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0`
- `org.jetbrains.kotlinx:kotlinx-datetime:0.7.1`
- `org.jetbrains.kotlinx:kotlinx-io-core:0.8.0`

### Ktor
- `io.ktor:ktor-client-core:3.3.2`
- `io.ktor:ktor-client-cio:3.3.2`
- `io.ktor:ktor-server-core:3.3.2`
- `io.ktor:ktor-server-netty:3.3.2`

### Databases
- `org.mongodb:mongodb-driver-kotlin-coroutine:5.6.1`
- `org.jetbrains.exposed:exposed-core:0.61.0`
- `org.postgresql:postgresql:42.7.8`

### Caching
- `io.lettuce:lettuce-core:6.7.1.RELEASE`
- `com.googlecode.xmemcached:xmemcached:2.4.8`

### AWS
- `software.amazon.awssdk:s3:2.38.5`
- `software.amazon.awssdk:dynamodb:2.38.5`

### Other
- `com.stripe:stripe-java:28.3.0`
- `com.google.firebase:firebase-admin:9.5.0`

## Troubleshooting

**Sources not found**: Not all libraries publish source JARs. This is common for:
- Very old libraries
- Some proprietary libraries
- Some Android-specific libraries

**Wrong version**: Make sure to specify the exact version. Use Maven Central to find available versions:
https://search.maven.org/

## Workflow Example

1. User asks about how Ktor's HttpClient handles retries
2. Download sources: `~/.extractlibs/download-lib-sources.sh "io.ktor:ktor-client-core:3.3.2"`
3. Search for relevant code: `grep -r "retry" ~/.extractlibs/src/ktor-client-core-3.3.2/`
4. Read the specific files found
5. Explain to user based on actual source code