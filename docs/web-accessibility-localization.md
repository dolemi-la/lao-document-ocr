# Web accessibility and localization

The web UI ships with a dependency-free Lao/English localization layer and baseline keyboard/screen-reader support.

## Languages

Supported UI locales:

- Lao (`lo`)
- English (`en`)

The app chooses:

1. a previously saved locale, when browser storage is available
2. otherwise Lao when the browser's preferred language starts with `lo`
3. otherwise English

The language can be switched at any time from the header.

The selected locale updates:

```html
<html lang="lo">
```

or:

```html
<html lang="en">
```

so assistive technology receives the correct document language.

The locale is saved in local storage when permitted. If storage is blocked, language switching still works for the current session.

## Message catalog

UI strings live in:

```text
apps/web/src/i18n.ts
```

Both locales implement the same typed message interface, so TypeScript compilation catches missing message keys.

Backend error details can still arrive in English because API validation errors are currently not localized.

## Keyboard access

The upload control keeps the native `<input type="file">` in the accessibility/focus tree.

It is visually hidden rather than disabled with pointer events.

Keyboard focus on the file input produces a visible focus ring around the drop zone.

Buttons use `:focus-visible` outlines.

## Screen readers

The UI includes:

- a main heading referenced by `aria-labelledby`
- a polite live region for OCR engine readiness
- a polite atomic live region for job/upload/conversion updates
- an alert role for backend/health errors
- `aria-busy` during conversion
- `aria-pressed` on language controls
- decorative icons/status dots hidden from assistive technology

## Motion

The stylesheet honors:

```css
@media (prefers-reduced-motion: reduce)
```

and reduces transitions/animations to effectively instantaneous behavior.

## Lao typography

The default font stack prefers:

```text
Noto Sans Lao
Noto Sans
system-ui
```

When `lang="lo"` is active, large headings use Lao-friendly line height and normal letter spacing instead of the tighter Latin display treatment.

## Current accessibility scope

This is a baseline, not a formal WCAG certification.

Before a public hosted release, manually test at minimum:

- keyboard-only upload and conversion
- browser zoom at 200%
- narrow/mobile viewport
- VoiceOver/NVDA screen-reader flow
- high-contrast/forced-colors modes
- Lao text rendering on target browsers

Future UI features should preserve native semantic controls instead of replacing them with div-based custom widgets.
