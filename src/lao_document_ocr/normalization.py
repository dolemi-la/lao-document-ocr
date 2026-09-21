import re
import unicodedata

_HORIZONTAL_SPACE = re.compile(r"[\t\u00a0 ]+")
_EXCESS_BLANKS = re.compile(r"\n{3,}")


def normalize_lao_text(text: str) -> str:
    """Normalize OCR text without guessing or rewriting Lao words."""
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        lines.append(_HORIZONTAL_SPACE.sub(" ", line).strip())
    return _EXCESS_BLANKS.sub("\n\n", "\n".join(lines)).strip()
