"""Phone-number handling for the demonstration recipient.

The number lives in ``.env`` (``EMERGENCY_CONTACT_NUMBER``) and is read at
runtime.  It is never written into Python source, never logged in full and
never stored in a config file - :func:`mask_phone` is what every log line, the
debug overlay and the call log use.

Validation is deliberately strict: a *malformed* number must fail loudly at
startup rather than silently becoming a call to nobody.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "RecipientError",
    "is_placeholder",
    "mask_phone",
    "normalize_phone",
]

#: E.164 allows up to 15 digits; 8 is a permissive lower bound that still
#: rejects obvious typos and local formats.
_MIN_DIGITS = 8
_MAX_DIGITS = 15

_SEPARATORS = re.compile(r"[\s\-()./]+")
_PLACEHOLDER_CHARS = set("xX*#")


class RecipientError(ValueError):
    """Raised when the configured recipient cannot be used."""


def normalize_phone(raw: Optional[str]) -> str:
    """Return ``raw`` in E.164 form (``+<digits>``).

    Raises :class:`RecipientError` when the value is empty, a placeholder
    (``+91XXXXXXXXXX``) or not in international format - a national number is
    ambiguous and guessing a country code would risk calling the wrong person.
    """
    if raw is None:
        raise RecipientError("no recipient configured")
    text = str(raw).strip()
    if not text:
        raise RecipientError("recipient is empty")
    if any(ch in _PLACEHOLDER_CHARS for ch in text):
        raise RecipientError(
            f"recipient {text!r} still contains placeholder characters - "
            "set a real demonstration number in .env"
        )

    cleaned = _SEPARATORS.sub("", text)
    if cleaned.startswith("00"):
        cleaned = "+" + cleaned[2:]
    if not cleaned.startswith("+"):
        raise RecipientError(
            f"recipient {text!r} is not in international format - "
            "use E.164, e.g. +15550100"
        )

    digits = cleaned[1:]
    if not digits.isdigit():
        raise RecipientError(f"recipient {text!r} contains non-digit characters")
    if not _MIN_DIGITS <= len(digits) <= _MAX_DIGITS:
        raise RecipientError(
            f"recipient {text!r} has {len(digits)} digits; "
            f"E.164 allows {_MIN_DIGITS}-{_MAX_DIGITS}"
        )
    return "+" + digits


def is_placeholder(raw: Optional[str]) -> bool:
    """True when ``raw`` looks like an unfilled template value."""
    if raw is None:
        return True
    text = str(raw).strip()
    return not text or any(ch in _PLACEHOLDER_CHARS for ch in text)


def mask_phone(number: Optional[str], keep_prefix: int = 3, keep_tail: int = 4) -> str:
    """Mask the middle of a number.

    ``+15550100`` (a fictional 555 number) becomes ``+15*0100``.  Used everywhere
    the recipient is displayed or logged.  A value that cannot be parsed is
    masked completely rather than leaked.
    """
    if not number:
        return "(none)"
    text = str(number).strip()
    try:
        normalized = normalize_phone(text)
    except RecipientError:
        return "(unconfigured)"
    visible = len(normalized) - 1
    hidden = visible - keep_prefix - keep_tail
    if hidden <= 0:
        # too short to mask meaningfully - show the tail only
        return "*" * keep_prefix + normalized[-keep_tail:]
    return f"{normalized[:keep_prefix]}{'*' * hidden}{normalized[-keep_tail:]}"
