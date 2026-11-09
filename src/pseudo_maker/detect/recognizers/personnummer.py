from __future__ import annotations

import re
from typing import Iterable

from .base import Span, luhn_check

# Accepts YYMMDD[-+]XXXX, YYMMDDXXXX, YYYYMMDD[-]XXXX, YYYYMMDDXXXX
# Bounded by non-digit / start-end so we don't grab parts of longer numbers.
_PATTERNS = [
    re.compile(r"(?<!\d)(\d{8})[-]?(\d{4})(?!\d)"),       # 12-digit form
    re.compile(r"(?<!\d)(\d{6})([-+]?)(\d{4})(?!\d)"),    # 10-digit form
]


def _normalize_to_10(value: str) -> str | None:
    """Strip separators, return YYMMDDXXXX (10 digits) or None if shape wrong."""
    digits = re.sub(r"[-+]", "", value)
    if len(digits) == 12:
        digits = digits[2:]
    if len(digits) != 10:
        return None
    return digits


def has_valid_date(value: str) -> bool:
    d = _normalize_to_10(value)
    if not d:
        return False
    mm, dd = int(d[2:4]), int(d[4:6])
    # Samordningsnummer: day is real_day + 60
    if dd > 60:
        dd -= 60
    return 1 <= mm <= 12 and 1 <= dd <= 31


def is_valid_personnummer(value: str) -> bool:
    if not has_valid_date(value):
        return False
    d = _normalize_to_10(value)
    # The 9-digit body for Luhn is YYMMDD + 3-digit serial
    return luhn_check(d[:9] + d[9])


# A personnummer-shaped value with a wrong checksum is still identifying
# (typos, test data, partially redacted copies). Flag it when the shape is
# unambiguous (YYYYMMDD-NNNN with a 19xx/20xx year) or a cue word precedes it.
_CUE = re.compile(
    r"(?i)(?:personnummer|personnr|pers\.?\s?nr|pnr|p-nr|samordningsnummer|"
    r"personal identity number|identity number|personal number|civic number|swedish id)\W{0,12}$"
)
_TWELVE_WITH_SEP = re.compile(r"^(19|20)\d{6}-\d{4}$")


class PersonnummerRecognizer:
    name = "PERSONNUMMER"

    def find(self, text: str) -> Iterable[Span]:
        seen: set[tuple[int, int]] = set()
        for pat in _PATTERNS:
            for m in pat.finditer(text):
                start, end = m.start(), m.end()
                if (start, end) in seen:
                    continue
                value = m.group(0)
                if is_valid_personnummer(value):
                    seen.add((start, end))
                    yield Span(start, end, self.name, value, 1.0, "regex")
                elif has_valid_date(value) and (
                    _TWELVE_WITH_SEP.match(value) or _CUE.search(text[max(0, start - 60):start])
                ):
                    seen.add((start, end))
                    yield Span(start, end, self.name, value, 0.8, "regex")
