"""Post-processing shared by every NER backend (spaCy, KB-BERT, Rampart).

Model output goes through the same hard-coded filters, so blocklists and
Swedish location rules apply identically whichever model found the span.
"""
from __future__ import annotations

import re

from .blocklist import (
    is_blocklisted_company,
    is_blocklisted_person,
    is_country_or_continent,
    is_job_title,
)
from .recognizers.base import Span


_CITY_SHAPE = re.compile(r"^[A-ZÅÄÖ][a-zåäöéüè]+$")
_STREET_SUFFIXES = (
    "gatan", "vägen", "stigen", "torget", "gränd", "gränden", "allén",
    "esplanaden", "leden", "platsen", "plan", "kajen", "backen",
    "parken", "kullen", "åsen", "gången", "rondellen",
)


def _is_street_token(t: str) -> bool:
    lower = t.lower()
    return any(lower.endswith(suffix) for suffix in _STREET_SUFFIXES)


def _classify_location(text: str) -> str:
    """Return 'CITY' for a single city-like token, 'STREET' otherwise."""
    t = text.strip()
    # "Region Jönköpings län", "Örebro kommun", "Västra Götalandsregionen": a region, not a street.
    if re.search(r"(?i)\b(län|kommun|region)\b|regionen$", t):
        return "CITY"
    # Street name ending (Slåttervallsgatan, Storgatan, Kungsvägen, …) wins
    # over single-token-city detection.
    if _is_street_token(t):
        return "STREET"
    if " " not in t and "," not in t and not any(c.isdigit() for c in t):
        if _CITY_SHAPE.match(t):
            return "CITY"
    return "STREET"


_LEADING_NOISE = re.compile(
    r"^(?:(?:mainly|primarily|mostly|främst|huvudsakligen|ofta|endast|bara)\s+"
    r"|(?:mr|mrs|ms|dr|prof|professor|fru|herr|doktor|docent)\.?\s+)+",
    re.IGNORECASE,
)

_CONJUNCTION = re.compile(r"\s+(?:och|and|samt|eller|or|&)\s+")


def split_conjunctions(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """'Ebba och Hugo' → two spans. Used for PERSON spans only (companies keep '&')."""
    out, cur = [], start
    for m in _CONJUNCTION.finditer(text, start, end):
        out.append((cur, m.start()))
        cur = m.end()
    out.append((cur, end))
    return [(a, b) for a, b in out if b > a]


def _strip_leading_noise(text: str) -> tuple[str, int]:
    """Strip a leading adverb like 'mainly ' / 'huvudsakligen '.

    Returns (new_text, offset_adjustment) so callers can adjust the span start.
    """
    m = _LEADING_NOISE.match(text)
    if m:
        return text[m.end():], m.end()
    return text, 0


_NAME_PARTICLES = {"von", "af", "de", "van", "der", "la", "le", "du", "di", "da", "bin", "al", "el"}


def _should_reject_person(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if is_blocklisted_person(t) or is_job_title(t):
        return True
    # Names are capitalised; "years of experience" is not a person.
    tokens = t.split()
    if len(tokens) > 1 and any(
        tok[0].islower() and tok.lower() not in _NAME_PARTICLES for tok in tokens
    ):
        return True
    # Slash inside a name almost always means a role fragment was captured
    # ("OWNER/FOUNDER"). Safer to drop than try to salvage.
    if "/" in t:
        return True
    # Single lowercase token → probably a common noun mis-tagged.
    if " " not in t and t == t.lower():
        return True
    # Single all-uppercase token of role-word-like length → often a title fragment.
    if " " not in t and t.isupper() and len(t) <= 10:
        return True
    return False


def _should_reject_company(text: str) -> bool:
    t = text.strip()
    if not t:
        return True
    if "/" in t:
        return True
    # Tech products / jargon ("Teams", "Excel", "Slack") are tagged ORG a lot.
    if is_blocklisted_person(t) or is_blocklisted_company(t):
        return True
    return False


_CAPS_WORD = re.compile(r"\b[A-ZÅÄÖÉÜ]{3,}(?:-[A-ZÅÄÖÉÜ]{2,})*\b")


def _titlecase_caps(text: str) -> str:
    def fix(m: re.Match) -> str:
        w = m.group(0)
        return "-".join(p[0] + p[1:].lower() for p in w.split("-"))
    out = _CAPS_WORD.sub(fix, text)
    assert len(out) == len(text)
    return out


def postprocess(
    entity_type: str,
    text: str,
    start: int,
    end: int,
    *,
    confidence: float,
    source: str = "ner",
) -> Span | None:
    """Clean one model span and map LOCATION → CITY/STREET. None = rejected.

    `text` is the full original text; `start`/`end` index into it.
    """
    # Trim whitespace and punctuation the model may include at the edges.
    while start < end and (text[start].isspace() or text[start] in "\"'([{,;:–—-•*"):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in "\"')]},;:.–—-•*"):
        end -= 1
    span_text = text[start:end]
    stripped, offset = _strip_leading_noise(span_text)
    if offset:
        span_text = stripped
        start += offset
    if not span_text.strip():
        return None

    if entity_type == "PERSON":
        if _should_reject_person(span_text):
            return None
    elif entity_type == "COMPANY":
        if _should_reject_company(span_text):
            return None
    elif entity_type in ("LOCATION", "CITY", "STREET"):
        if is_country_or_continent(span_text):
            return None
        if entity_type == "LOCATION":
            entity_type = _classify_location(span_text)
    return Span(start=start, end=end, type=entity_type, value=span_text, confidence=confidence, source=source)


def note_rejected(
    rejected: list | None,
    entity_type: str,
    text: str,
    start: int,
    end: int,
    score: float,
    reason: str,
) -> None:
    """Record a model span that was not accepted, for the UI's ambiguous list.

    Only plausible entities are kept: they must contain a capital letter, and
    filtered persons are skipped (those are mostly pronouns, titles, weekdays).
    """
    if rejected is None:
        return
    from .recognizers.base import Candidate

    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    value = text[start:end]
    if len(value) < 2 or not any(c.isupper() for c in value):
        return
    if entity_type == "LOCATION":
        entity_type = _classify_location(value)
    rejected.append(Candidate(start, end, entity_type, value, round(float(score), 3), reason))


# Model spans scoring between this and the recognizer's threshold are shown
# as ambiguous instead of being dropped silently.
REJECT_FLOOR = 0.2


def filter_reason(entity_type: str, value: str) -> str:
    """Human-readable reason why postprocess() rejected a span."""
    v = value.strip()
    if entity_type in ("LOCATION", "CITY", "STREET") and is_country_or_continent(v):
        return "country (kept by default)"
    if entity_type == "COMPANY":
        if is_job_title(v):
            return "looks like a job title"
        if is_blocklisted_company(v) or is_blocklisted_person(v):
            return "product, platform or public agency (kept by default)"
    return "filtered by rules"
