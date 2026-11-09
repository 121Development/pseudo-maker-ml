from __future__ import annotations

import re
from typing import Iterable

from .base import Span

# Swedish postal codes are written NNN NN (with a space) in all formal contexts
# we care about (CVs, contracts, letters). Requiring the space keeps us from
# matching inside standard identifiers such as "ISO 18788:2015" or "ISO 27001".
# Also accept the country-prefixed variants SE-NNN NN and S-NNN NN.
_PATTERN = re.compile(r"(?<![\w-])(?:SE-|S-)?(\d{3})\s(\d{2})(?!\d)")

# If the postal-code-shaped match is preceded (within a short window) by a
# standards-token like ISO/IEC/EN/SS/RFC/ISBN/DIN, it's almost certainly a
# standard reference, not a postal code. Reject those.
_STANDARDS_PREFIX = re.compile(
    r"\b(?:ISO|IEC|EN|SS|ASTM|DIN|BS|RFC|ISBN|ANSI|IEEE|UL|CE|NIST|PCI)\b[\s/\-]*\Z",
    re.IGNORECASE,
)


class PostalCodeSeRecognizer:
    name = "POSTAL_CODE"

    def find(self, text: str) -> Iterable[Span]:
        for m in _PATTERN.finditer(text):
            first = int(m.group(1))
            if not (100 <= first <= 989):
                continue
            # Look back up to 24 chars for a standards-reference prefix.
            left = text[max(0, m.start() - 24):m.start()]
            if _STANDARDS_PREFIX.search(left):
                continue
            yield Span(m.start(), m.end(), self.name, m.group(0), 0.7, "regex")


# UK postcodes: "SW1A 1AA", "M1 1AE", "BS1 1AA", "EC2R 8AH".
_UK = re.compile(r"(?<![\w-])([A-Z]{1,2}\d[A-Z\d]?)[ \t]?(\d[A-Z]{2})(?![\w-])")


class UkPostcodeRecognizer:
    name = "POSTAL_CODE"

    def find(self, text: str) -> Iterable[Span]:
        for m in _UK.finditer(text):
            yield Span(m.start(), m.end(), self.name, m.group(0), 0.85, "regex")
