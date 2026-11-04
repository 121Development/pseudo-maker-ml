from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    value: str
    confidence: float = 1.0
    source: str = "regex"


@dataclass(frozen=True)
class Candidate:
    """A possible entity the pipeline did NOT replace (shown as "ambiguous" in the UI)."""
    start: int
    end: int
    type: str
    value: str
    score: float
    reason: str


class Recognizer(Protocol):
    name: str

    def find(self, text: str) -> Iterable[Span]: ...


def luhn_check(digits: str) -> bool:
    if not digits.isdigit() or not digits:
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def luhn_checksum(digits: str) -> int:
    """Return the Luhn check digit that, appended, makes `digits + check` valid."""
    if not digits.isdigit():
        raise ValueError("digits must be numeric")
    total = 0
    parity = (len(digits) + 1) % 2
    for i, ch in enumerate(digits):
        n = int(ch)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return (10 - total % 10) % 10
