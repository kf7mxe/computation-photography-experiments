---
name: KiteUI Theming
description: This skill should be used when the user asks about KiteUI themes, custom colors, fonts, semantic styling, or troubleshooting theme issues (colors not applying, fonts not loading, buttons not styled correctly)
version: 1.0.0
---

# KiteUI Theming Skill

You are an expert in KiteUI theming. This skill covers creating custom themes, adding fonts, styling semantics, and troubleshooting common theme issues.

## Quick Reference: Common Issues

### Button Text Not Visible
**Problem**: Text inside `important.button` doesn't show.
**Solution**: Use `text {}` not heading elements like `h6 {}` inside buttons.

### Custom Font Not Loading
**Solution**:
1. Place font files in `apps/src/commonMain/resources/fonts/{fontname}/normal.ttf`
2. Run `./gradlew :apps:kiteuiResourcesCommon`
3. Reference as `Resources.fontsFontname` (auto-generated)

### Theme.flat2 accentHue Ignored
**Explanation**: `accentHue` only applies when `baseBrightness` is 0.35-0.65. For light/dark themes, create a custom theme instead.

### Semantic Override Not Working
**Solution**: Use `it.withBack(...)` not `it.copy(...)` in semantic overrides.

### HeaderSemantic Applies to All Headings
**Important**: HeaderSemantic automatically styles h1, h2, h3, h4, h5, h6 elements AND the navigation header bar from `appNav`.

If headings aren't showing your HeaderSemantic styling:
1. Verify the override uses `withBack()` correctly
2. Check that heading text contrasts with background color
3. Ensure font is loading if using custom font

### Semantic Override Affects Other Components
**Problem**: Global SelectedSemantic override bleeds into nav bar.
**Solution**: Create a custom semantic (e.g., `StepDotSemantic`) instead of overriding global semantics. See "Global Semantic Override Bleeding" section below.

### Prefer Custom Semantics Over Inline Styles
**Rule**: Never use `themeModify` or anonymous styling. Always create a named custom semantic.

---

## Creating a Custom Theme

### Minimal Theme Template

```kotlin
import com.lightningkite.kiteui.models.*
import com.lightningkite.reactive.core.Signal

object MyColors {
    val primary = Color.fromHex(0x2D3B2D)     // No # prefix - use hex literal
    val background = Color.fromHex(0xF5F3F0)
    val text = Color.fromHex(0x1A1A1A)
    val border = Color.fromHex(0xD4CFC8)
}

val myTheme = Theme(
    id = "my-theme",
    font = FontAndStyle(),                        // System font
    elevation = 1.dp,                             // Shadow depth
    cornerRadii = CornerRadii.Constant(8.dp),    // Border radius
    gap = 1.rem,                                  // Element spacing
    outlineWidth = 0.px,
    foreground = MyColors.text,
    background = MyColors.background,
    outline = MyColors.border,
    semanticOverrides = SemanticOverrides(
        ImportantSemantic.override {
            it.withBack(
                foreground = Color.white,
                background = MyColors.primary,
            )
        },
    ),
)

val appTheme = Signal<Theme>(myTheme)
```

---

## Custom Fonts

### Font File Structure

Place fonts in `apps/src/commonMain/resources/fonts/`:

```
fonts/
├── inter/
│   ├── normal.ttf      # Required
│   ├── bold.ttf        # Optional
│   └── italic.ttf      # Optional
└── bebasneue/
    └── normal.ttf
```

Variable fonts work - just name them `normal.ttf`.

### Generate Resources

```bash
./gradlew :apps:kiteuiResourcesCommon
```

This generates `Resources.fontsInter`, `Resources.fontsBebasneue`, etc.

### Use in Theme

```kotlin
val myTheme = Theme(
    font = FontAndStyle(
        font = Resources.fontsInter,
        weight = 400,                    // 400=normal, 600=semibold, 700=bold
        lineSpacingMultiplier = 1.5,     // NOT lineHeight!
        additionalLetterSpacing = 0.px,
        size = 1.rem,
    ),
    // ...
)
```

### FontAndStyle Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `font` | `Font` | Font resource (e.g., `Resources.fontsInter`) |
| `weight` | `Int` | 100-900 (400=normal, 700=bold) |
| `italic` | `Boolean` | Italic style |
| `allCaps` | `Boolean` | Uppercase transform |
| `lineSpacingMultiplier` | `Double` | Line height multiplier (default 1.4) |
| `additionalLetterSpacing` | `Dimension` | Extra letter spacing |
| `size` | `Dimension` | Font size |

**Common Mistake**: There is no `lineHeight` parameter. Use `lineSpacingMultiplier`.

---

## Semantic Overrides

### Why Semantic, Not Visual Hierarchy?

KiteUI uses **semantic meaning** (important, danger, warning) instead of **visual hierarchy** (primary, secondary, tertiary). This is intentional:

**Benefits**:
1. **Resilient to redesign** - Change "important" from filled→outlined→elevated without touching code
2. **Meaningful code** - `danger.button` clearly indicates purpose, `tertiary.red.button` doesn't
3. **Theme flexibility** - Different brands can interpret "important" differently while keeping the same codebase
4. **Better accessibility** - Semantic meaning helps screen readers
5. **Forces good UX thinking** - Developers must consider "what does this button do?" not just "what should it look like?"

**Example**:
```kotlin
// Semantic - clear intent
important.button { text { content = "Save Changes" } }
danger.button { text { content = "Delete Account" } }

// Visual hierarchy - meaning unclear
primary.button { text { content = "Save Changes" } }
tertiary.red.button { text { content = "Delete Account" } }
```

When you retheme, semantic styling adapts naturally to the new design language.

### Understanding Semantics

Semantics are applied via modifiers, not element types:

| Semantic | Applied Via | Purpose |
|----------|-------------|---------|
| `ImportantSemantic` | `.important.button` | Primary actions |
| `CardSemantic` | `.card.col` | Card backgrounds |
| `FieldSemantic` | `.field()` | Input containers |
| `HeaderSemantic` | `h1`, `h2`, `h3`, `h4`, `h5`, `h6` + nav bar | All heading elements |
| `DangerSemantic` | `.danger.button` | Destructive actions |

### Override Pattern

**Always use `withBack()` or `withoutBack()`**, never `copy()`:

```kotlin
semanticOverrides = SemanticOverrides(
    ImportantSemantic.override { theme ->
        val bg = MyColors.primary
        theme.withBack(
            background = bg,
            foreground = bg.highlight(1f),
        )
    },
)
```

### Critical: Derive Themes, Don't Set Them

**Semantic overrides should _derive_ from the incoming theme state, not force absolute values.** This ensures semantics compose correctly when combined (e.g., `important.card.button`, `selected.radioToggleButton`).

#### The Problem with Absolute Values

```kotlin
// ❌ DANGEROUS - forces absolute values, ignores context
SelectedSemantic.override {
    it.withBack(
        background = MyColors.primary,  // Always green, ignores parent
        foreground = Color.white,       // Hardcoded, doesn't adapt
    )
}
```

Problems:
1. `important.radioToggleButton` looks identical to `radioToggleButton` (both green)
2. If nested in a `card`, the card's white background + inherited white foreground = invisible text
3. Semantic combinations become unpredictable landmines

#### The Correct Pattern: Derive from Incoming State

```kotlin
// ✅ SAFE - derives from current state
SelectedSemantic.override {
    val existing = it.background.closestColor().toHSP()
    // Derive a "selected" version: increase saturation, adjust brightness
    val bg = existing.copy(saturation = 0.8f, brightness = 0.4f).toRGB()
    it.withBack(
        background = bg,
        foreground = bg.highlight(1f),  // Always readable
    )
}

// ✅ SAFE - cards derive foreground from their background
CardSemantic.override {
    val bg = it.background.closestColor().toHSP()
        .copy(brightness = it.background.closestColor().toHSP().brightness + 0.1f)
        .toRGB()
    it.withBack(
        background = bg,
        foreground = bg.highlight(1f),
    )
}
```

#### Key Functions for Deriving Themes

| Function | Purpose |
|----------|---------|
| `it.background.closestColor()` | Get the current background as a Color |
| `color.toHSP()` | Convert to HSP (hue, saturation, perceived brightness) for manipulation |
| `hspColor.copy(...)` | Modify hue, saturation, or brightness |
| `hspColor.toRGB()` | Convert back to Color |
| `color.highlight(1f)` | Get maximum contrast color (black or white) |
| `color.highlight(0.5f)` | Get medium contrast color |

#### Example: Theme.flat2 Pattern

See how `ImportantSemantic` in Theme.flat2 derives from context:

```kotlin
ImportantSemantic.override {
    val existing = it.background.closestColor().toHSP()
    if (abs(existing.brightness - 0.5f) > brightnessStep * 3) {
        // Far from mid-brightness: shift toward 0.5
        val b = existing.copy(brightness = 0.5f).toRGB()
        it.withBack(
            foreground = b.highlight(1f),
            background = b,
            outline = b,
        )
    } else {
        // Near mid-brightness: use accent hue instead
        val b = HSPColor(hue = accentHue, saturation = saturation, brightness = 0.5f).toRGB()
        it.withBack(
            foreground = b.highlight(1f),
            background = b,
            outline = b,
        )
    }
}
```

This makes `important` adapt to whatever context it's in, rather than forcing a single look.

#### Summary

1. **Read incoming state**: Use `it.background.closestColor()` to see what you're working with
2. **Derive new values**: Transform the existing colors rather than replacing them entirely
3. **Always pair foreground with background**: Use `.highlight(1f)` to ensure readability
4. **Test combinations**: Verify that `semantic1.semantic2.element` works correctly

### withBack() Parameters

```kotlin
theme.withBack(
    cascading = true,                           // Children inherit
    font = FontAndStyle(...),                   // Typography
    elevation = 2.dp,                           // Shadow
    cornerRadii = CornerRadii.Constant(12.dp), // Radius
    gap = 1.rem,                                // Internal spacing
    foreground = Color.white,                   // Text color
    outline = MyColors.border,                  // Border color
    outlineWidth = 2.dp,                        // Border width
    background = MyColors.primary,              // Background
)
```

---

## Common Semantic Patterns

### Primary Buttons

```kotlin
ImportantSemantic.override {
    it.withBack(
        foreground = Color.white,
        background = MyColors.primary,
        cornerRadii = CornerRadii.Constant(8.dp),
        font = FontAndStyle(weight = 600),
    )
}
```

### Cards

```kotlin
CardSemantic.override {
    it.withBack(
        background = Color.white,
        outline = MyColors.border,
        outlineWidth = 1.dp,
        cornerRadii = CornerRadii.Constant(12.dp),
        elevation = 2.dp,
    )
}
```

### Input Fields

```kotlin
FieldSemantic.override {
    it.withBack(
        background = Color.white,
        outline = MyColors.border,
        outlineWidth = 2.dp,
        cornerRadii = CornerRadii.Constant(8.dp),
    )
}
```

### Headings (h1, h2, h3, h4, h5, h6)

```kotlin
HeaderSemantic.override {
    it.withBack(
        foreground = Color.white,
        background = MyColors.primary,
        font = FontAndStyle(
            font = Resources.fontsBebasneue,
            additionalLetterSpacing = 0.03125.rem,
            size = 1.5.rem,
        ),
    )
}
```

**Note**: This applies to ALL heading elements (h1-h6) and the navigation bar.

---

## Theme.flat2 (Quick Prototyping)

For quick prototyping only:

```kotlin
val quickTheme = Theme.flat2(
    id = "quick",
    hue = Angle(0.35f),        // Main hue (0-1)
    saturation = 0.3f,
    baseBrightness = 0.5f,     // MUST be 0.35-0.65 for accentHue to work
    accentHue = Angle(0.08f),
)
```

**Warning**: `accentHue` only affects `ImportantSemantic` when `baseBrightness` is 0.35-0.65. For dark/light themes, buttons use the main `hue`.

**Recommendation**: Create custom themes for production apps.

---

## Troubleshooting

### Theme Changes Not Appearing

1. Ensure `jsViteDev --continuous` is running
2. Wait for "BUILD SUCCESSFUL" in terminal
3. For font changes: `./gradlew :apps:kiteuiResourcesCommon`
4. Hard refresh browser (Cmd+Shift+R)

### Colors Not Applying

1. Check `Color.fromHex(0xABCDEF)` - no # prefix
2. Verify semantic is correct (`.important.button` not just `.button`)
3. Check semantic override uses `withBack()` not `copy()`
4. **CRITICAL**: Ensure alpha channel is set! `Color` constructor defaults to `alpha = 0f` (transparent)
   - Use `Color.fromHex(0xRRGGBB)` which automatically sets `alpha = 1f`
   - OR use `Color(alpha = 1f, red = ..., green = ..., blue = ...)` with explicit alpha
   - Transparent colors (alpha = 0) will be invisible even if RGB values are correct

### Font Not Rendering

1. Font file exists at `resources/fonts/{name}/normal.ttf`
2. Resources generated: `./gradlew :apps:kiteuiResourcesCommon`
3. Using `Resources.fontsFontname` not a string

### Button Text Invisible

Using the wrong element inside button:
```kotlin
// Wrong
important.button { h6 { content = "Click" } }

// Correct
important.button { text { content = "Click" } }
```

### Creating Pill/Capsule Shapes

For decorative elements like step indicators with non-square dimensions:

```kotlin
// Step dot pill shape (24px × 8px)
selected.sizeConstraints(width = 1.5.rem, height = 0.5.rem).frame {
    padding = 0.px  // Remove theme padding
}

// In theme:
SelectedSemantic.override {
    it.withBack(
        background = MyColors.primary,
        cornerRadii = CornerRadii.ForceConstant(999.rem),  // Capsule shape
    )
}
```

**Key Points**:
- Use `.frame { padding = 0.px }` to remove default theme padding
- Use very high `ForceConstant` corner radius (999.rem) for capsule shapes
- `sizeConstraints` with `frame` respects width/height for non-square elements

### ⚠️ Global Semantic Override Bleeding

**Problem**: Global semantic overrides can unexpectedly affect other components.

**Example**: Overriding `SelectedSemantic` with extreme corner radius affects the nav bar:

```kotlin
// ❌ This bleeds into NavSemantic (which uses SelectedSemantic internally)
SelectedSemantic.override {
    it.withBack(
        cornerRadii = CornerRadii.ForceConstant(999.rem),  // Affects nav tabs!
    )
}
```

**Solution**: Create isolated custom semantics for specific use cases:

```kotlin
// ✅ Create a custom semantic for step dots
data object StepDotSemantic : Semantic("step-dot") {
    override fun default(theme: Theme): ThemeAndBack = theme.withBack(
        background = MyColors.primary,
        cornerRadii = CornerRadii.RatioOfSize(0.5f),  // Pill shape
    )
}

// Create a ViewWriter extension for easy use
inline val ViewWriter.stepDot: ViewWriter get() = StepDotSemantic.onNext

// Usage
stepDot.sizeConstraints(width = 1.5.rem, height = 0.5.rem).frame { padding = 0.px }
```

**Rule**: If a semantic override might affect other components, create a custom semantic instead.

### Custom Semantics vs Inline Styles

**Always prefer custom semantics over inline/anonymous styling:**

```kotlin
// ❌ Don't use themeModify or inline styles
themeModify { theme ->
    theme.withBack(background = Color.red)
}.button { /* ... */ }

// ✅ Create a custom semantic
data object AlertSemantic : Semantic("alert") {
    override fun default(theme: Theme): ThemeAndBack = theme.withBack(
        background = Color.red,
        foreground = Color.white,
    )
}

// Use in theme overrides or directly
AlertSemantic.onNext.button { /* ... */ }
```

**Why semantics over inline styles:**
1. **Reusability** - Use the same styling across multiple components
2. **Consistency** - All alert buttons look the same automatically
3. **Maintainability** - Change styling in one place
4. **Theme support** - Can be overridden per-theme
5. **Debugging** - Semantic names appear in dev tools

---

## Creating Custom Semantics

**Important**: Only create custom semantics when existing ones don't fit semantically. Check available semantics in KiteUI source (`/Users/jivie/Projects/kiteui/library/src/commonMain/kotlin/com/lightningkite/kiteui/models/Theme.kt`) first.

Available built-in semantics include: ImportantSemantic, DangerSemantic, WarningSemantic, SelectedSemantic, UnselectedSemantic, CardSemantic, FieldSemantic, ButtonSemantic, HeaderSemantic, and many more.

### When to Create Custom Semantics

Create custom semantics only when:
- No existing semantic fits the use case semantically
- You need styling that's conceptually different from existing semantics
- The element represents a unique UI pattern not covered by built-in semantics

### How to Create

```kotlin
data object MyCustomSemantic : Semantic("my-custom") {
    override fun default(theme: Theme): ThemeAndBack = theme[ImportantSemantic]
}
```

Then override in theme:

```kotlin
semanticOverrides = SemanticOverrides(
    MyCustomSemantic.override {
        it.withBack(
            background = MyColors.custom,
            cornerRadii = CornerRadii.RatioOfSize(0.5f),
        )
    },
)
```

Use in views with `onNext(ThemeDerivation { it[MyCustomSemantic] })` or create a view modifier extension.

**Best Practice**: Prefer reusing existing semantics when semantically appropriate:
- Use `SelectedSemantic` for active/selected states
- Use `ImportantSemantic` for primary actions
- Use `DangerSemantic` for destructive actions (not for decoration)
- Use `CardSemantic` for card-like containers

## Complete Theme Example

```kotlin
object AppColors {
    val primary = Color.fromHex(0x2D3B2D)
    val primaryLight = Color.fromHex(0x4A5D4A)
    val background = Color.fromHex(0xF5F3F0)
    val cardBackground = Color.white
    val text = Color.fromHex(0x1A1A1A)
    val textSecondary = Color.fromHex(0x6B6B6B)
    val border = Color.fromHex(0xD4CFC8)
    val error = Color.fromHex(0xC62828)
}

val appTheme = Signal<Theme>(Theme(
    id = "my-app",
    font = FontAndStyle(
        font = Resources.fontsInter,
        weight = 400,
        lineSpacingMultiplier = 1.5,
        size = 1.rem,
    ),
    elevation = 1.dp,
    cornerRadii = CornerRadii.Constant(12.dp),
    gap = 1.rem,
    outlineWidth = 0.px,
    foreground = AppColors.text,
    background = AppColors.background,
    outline = AppColors.border,
    semanticOverrides = SemanticOverrides(
        ImportantSemantic.override {
            it.withBack(
                foreground = Color.white,
                background = AppColors.primary,
                cornerRadii = CornerRadii.Constant(8.dp),
                font = FontAndStyle(
                    font = Resources.fontsInter,
                    weight = 600,
                ),
            )
        },
        CardSemantic.override {
            it.withBack(
                background = AppColors.cardBackground,
                outline = AppColors.border,
                outlineWidth = 1.dp,
                cornerRadii = CornerRadii.Constant(12.dp),
                elevation = 1.dp,
            )
        },
        FieldSemantic.override {
            it.withBack(
                background = AppColors.cardBackground,
                outline = AppColors.border,
                outlineWidth = 2.dp,
                cornerRadii = CornerRadii.Constant(8.dp),
            )
        },
        HeaderSemantic.override {
            it.withBack(
                foreground = Color.white,
                background = AppColors.primary,
                font = FontAndStyle(
                    font = Resources.fontsBebasneue,
                    additionalLetterSpacing = 0.03125.rem,
                    size = 1.5.rem,
                ),
            )
        },
        DangerSemantic.override {
            it.withBack(
                foreground = Color.white,
                background = AppColors.error,
            )
        },
    ),
))
```

---

## Theme Properties Reference

| Property | Type | Example |
|----------|------|---------|
| `id` | `String` | `"my-theme"` |
| `font` | `FontAndStyle` | `FontAndStyle(weight = 400)` |
| `elevation` | `Dimension` | `2.dp` |
| `cornerRadii` | `CornerRadii` | `CornerRadii.Constant(8.dp)` |
| `gap` | `Dimension` | `1.rem` |
| `outlineWidth` | `Dimension` | `1.dp` |
| `foreground` | `Paint` | `Color.fromHex(0x1A1A1A)` |
| `background` | `Paint` | `Color.fromHex(0xF5F3F0)` |
| `outline` | `Paint` | `Color.fromHex(0xD4CFC8)` |

## Dimension Units

```kotlin
8.dp      // Density-independent (consistent physical size)
1.rem     // Relative to root font (typically 16px)
16.px     // Raw pixels (avoid for responsive design)
```

Use `rem` for typography, `dp` for spacing/sizing.

---

## Advanced Features

### CornerRadii Types

KiteUI offers several corner radius strategies:

```kotlin
// AdaptiveToSpacing - adapts to parent spacing context
// Uses min(value, parent's gap) - good for tab bars where buttons touch (gap=0)
cornerRadii = CornerRadii.AdaptiveToSpacing(8.dp)
cornerRadii = CornerRadii.Constant(8.dp)  // Alias for AdaptiveToSpacing

// Fixed - always this exact value, never adapts to context
// Use for elements that should NEVER have rounded corners (like app shell)
cornerRadii = CornerRadii.Fixed(0.px)
cornerRadii = CornerRadii.ForceConstant(8.dp)  // Alias for Fixed

// Circular - ratio of element size (0.5 = circle)
cornerRadii = CornerRadii.RatioOfSize(0.5f)

// Capsule/Pill - use very high radius
cornerRadii = CornerRadii.Fixed(999.rem)  // Creates pill/capsule shape

// Per-corner - boolean on/off per corner (mobile limitation)
cornerRadii = CornerRadii.PerCorner(
    value = 8.dp,
    topLeft = true,
    topRight = true,
    bottomLeft = false,
    bottomRight = false,
)

// Ratio of spacing
cornerRadii = CornerRadii.RatioOfSpacing(0.5f)
```

**Aliases**: `Constant` is an alias for `AdaptiveToSpacing`, and `ForceConstant` is an alias for `Fixed`. Prefer the explicit names for clarity.

**App Shell Pattern**: Use `CornerRadii.Fixed(0.px)` for outer container semantics (like `OuterSemantic`, `MainContentSemantic`) to prevent unwanted rounding on the app shell:

```kotlin
OuterSemantic.override {
    it.withBack(
        cascading = false,
        cornerRadii = CornerRadii.Fixed(0.px),  // Never round the outer shell
        gap = 1.px,
        padding = Edges.ZERO,
        background = Color.gray(0.3f)
    )
},
MainContentSemantic.override {
    it.withBack(
        cascading = false,
        cornerRadii = CornerRadii.Fixed(0.px)  // Keep content area square
    )
},
```

**Capsule/Pill Shapes**: Use `CornerRadii.ForceConstant(999.rem)` (or any very large value) to create capsule shapes. The browser automatically limits the radius to create fully rounded ends, making pills from rectangles and circles from squares.

**Tab Bar Example**:
```kotlin
// Buttons with gap=0 won't have visible radii between them
row(gap = 0.dp) {
    repeat(4) {
        important.button {
            // Constant(8.dp) becomes 0dp between buttons
            // but stays 8dp on outer edges
        }
    }
}
```

### Paint Types Beyond Color

```kotlin
// Solid color (most common)
background = Color.fromHex(0x2D3B2D)

// Fading animation (for loading states)
background = FadingColor(
    Color.fromHex(0xAAAAAA),
    Color.fromHex(0xBBBBBB),
)

// Linear gradient
background = LinearGradient(
    // See KiteUI source for full API
)

// Radial gradient
background = RadialGradient(
    // See KiteUI source for full API
)
```

### Double Important Pattern

Apply `important` twice for extra emphasis:

```kotlin
// Important dialog background
important.col {
    h1 { content = "Critical Action" }

    // Button needs to stand out even more
    important.button {
        text { content = "Confirm" }
    }
    // This button uses the accent color (double important)
}

// Or: REALLY important single button
important.important.button {
    text { content = "Emergency Stop" }
}
```

### Auto-Applied Semantics

Some elements automatically apply semantics:

| Element | Semantic | Can Override |
|---------|----------|--------------|
| `h1`, `h2`, `h3`, `h4`, `h5`, `h6` | `HeaderSemantic` | No |
| `subtext` | `SubtextSemantic` | No |
| Navigation bar (from `appNav`) | `HeaderSemantic` | No |
| `radioToggleButton` (checked) | `SelectedSemantic` | Yes, in theme |
| `radioToggleButton` (unchecked) | `UnselectedSemantic` | Yes, in theme |
| `toggleButton` (checked) | `SelectedSemantic` | Yes, in theme |
| `toggleButton` (unchecked) | `UnselectedSemantic` | Yes, in theme |

Other semantics require explicit modifiers (`.important`, `.card`, `.field()`).

### Styling Radio Toggle Buttons

`radioToggleButton` and `toggleButton` automatically use `SelectedSemantic` when checked and `UnselectedSemantic` when unchecked. Style them via theme overrides:

```kotlin
semanticOverrides = SemanticOverrides(
    // Selected state - filled with primary color
    SelectedSemantic.override {
        it.withBack(
            background = MyColors.primary,
            foreground = Color.white,
            outline = MyColors.primary,
            outlineWidth = 2.dp,
            cornerRadii = CornerRadii.Constant(8.dp),
        )
    },
    // Unselected state - transparent with border
    UnselectedSemantic.override {
        it.withBack(
            background = Color.transparent,
            foreground = MyColors.text,
            outline = MyColors.border,
            outlineWidth = 2.dp,
            cornerRadii = CornerRadii.Constant(8.dp),
        )
    },
)
```

**Usage with `.equalTo()`**:

```kotlin
val selectedActivity = Signal(0)
val activities = listOf("Endurance", "Hill Climb", "HIIT", "Recovery")

row {
    activities.forEachIndexed { index, label ->
        expanding.radioToggleButton {
            centered.text { content = label }
            checked bind selectedActivity.equalTo(index)
        }
    }
}
```

**Key Points**:
- Use `expanding` modifier to make all buttons equal width in a row
- Use `.equalTo()` extension for clean bidirectional binding
- The button automatically toggles appearance based on `checked` state
- `radioToggleButton` cannot be deselected by clicking again (radio behavior)
- `toggleButton` can be toggled on/off (checkbox behavior)
