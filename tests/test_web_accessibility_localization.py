from pathlib import Path

ROOT = Path(__file__).parents[1]
APP = ROOT / "apps" / "web" / "src" / "App.tsx"
I18N = ROOT / "apps" / "web" / "src" / "i18n.ts"
STYLES = ROOT / "apps" / "web" / "src" / "styles.css"


def test_web_has_lao_and_english_message_catalogs() -> None:
    content = I18N.read_text(encoding="utf-8")
    assert 'export type Locale = "lo" | "en";' in content
    assert "en: {" in content
    assert "lo: {" in content
    assert "ປ່ຽນເອກະສານພາສາລາວ" in content
    assert "Turn Lao scans and PDFs" in content


def test_locale_updates_document_language_and_is_remembered() -> None:
    content = APP.read_text(encoding="utf-8")
    i18n = I18N.read_text(encoding="utf-8")
    assert "document.documentElement.lang = locale" in content
    assert "persistLocale(locale)" in content
    assert "lao-document-ocr-locale" in i18n
    assert "navigator.languages" in i18n


def test_live_regions_and_accessible_status_are_present() -> None:
    content = APP.read_text(encoding="utf-8")
    assert 'role="status"' in content
    assert 'aria-live="polite"' in content
    assert 'aria-atomic="true"' in content
    assert 'role="alert"' in content
    assert "aria-busy={busy}" in content


def test_file_input_remains_keyboard_focusable() -> None:
    app = APP.read_text(encoding="utf-8")
    styles = STYLES.read_text(encoding="utf-8")
    assert 'className="fileInput"' in app
    assert 'aria-describedby="file-help"' in app
    assert ".dropzone:focus-within" in styles
    assert ".fileInput {" in styles
    assert "pointer-events: none" not in styles.split(".fileInput {", 1)[1].split("}", 1)[0]


def test_language_toggle_is_accessible() -> None:
    content = APP.read_text(encoding="utf-8")
    assert 'role="group"' in content
    assert 'aria-pressed={locale === "lo"}' in content
    assert 'aria-pressed={locale === "en"}' in content
    assert 'lang="lo"' in content
    assert 'lang="en"' in content


def test_styles_support_focus_and_reduced_motion() -> None:
    content = STYLES.read_text(encoding="utf-8")
    assert "button:focus-visible" in content
    assert "@media (prefers-reduced-motion: reduce)" in content
    assert 'html[lang="lo"] h2' in content
