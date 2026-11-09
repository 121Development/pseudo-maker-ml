"""Post-processing shared by every NER backend (spaCy, KB-BERT, Rampart).

Model output goes through the same hard-coded filters, so blocklists and
Swedish location rules apply identically whichever model found the span.
"""
from __future__ import annotations

import re

from .. import swedish as sv
from .blocklist import (
    is_blocklisted_company,
    is_blocklisted_person,
    is_country_or_continent,
    is_currency,
    is_job_title,
    is_label_word,
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


# Punctuation and Markdown syntax a model may include at a span's edges.
_EDGE_PUNCT_START = "\"'([{,;:–—-•*#|>_`~"
_EDGE_PUNCT_END = "\"')]},;:.–—-•*#|>_`~"

_LINE_START = re.compile(r"(?:^|\n|\|)[ \t]*(?:[#>*\-+•]+[ \t]*|\d+[.)][ \t]*)*$")


def _at_line_start(text: str, start: int) -> bool:
    """True if only whitespace / list bullets / heading or table markup precede `start` on its line."""
    return bool(_LINE_START.search(text[max(0, start - 40):start]))


_EDGE_WORDS = r"(?:the|a|an|and|or|of|at|in|on|for|to|och|eller|i|på|vid|hos|från|till|en|ett|den|det)"
_EDGE_WORDS_START = re.compile(rf"^(?:{_EDGE_WORDS}\s+)+")
_EDGE_WORDS_END = re.compile(rf"(?:\s+{_EDGE_WORDS})+$")

# Lowercase nouns that are part of Swedish organisation names
# ("Uppsala universitet", "Örebro kommun", "Karolinska sjukhuset").
_ORG_LOWER = {
    "universitet", "universitetet", "högskola", "högskolan", "kommun", "kommuns", "region", "regionen",
    "län", "läns", "stad", "stads", "sjukhus", "sjukhuset", "skola", "skolan", "gymnasium", "förening",
    "föreningen", "kyrka", "kyrkan", "församling", "institutet", "akademi", "akademin", "bibliotek",
    "museum", "museet", "tingsrätt", "hovrätt", "förvaltningsrätt", "polismyndighet", "länsstyrelse",
    "bostäder", "fastigheter", "bygg", "konsult", "konsulter", "advokatbyrå", "revisionsbyrå",
}

_COMPANY_CONNECTORS = {
    "i", "och", "&", "på", "av", "för", "of", "and", "the", "for", "in", "de", "du", "la", "le",
    "von", "van", "der", "den", "det", "y", "e", "a", "an", "at", "zu", "und", "et", "&co", "co",
}

_MARKUP = re.compile(r"[#|*_>`~]")


def mask_markup(text: str) -> str:
    """Replace Markdown syntax with spaces (same length) before a model reads it."""
    return _MARKUP.sub(" ", text)


def _known_any_name(word: str) -> bool:
    """Known Swedish first name/surname, or an English first name (also genitive)."""
    for w in (word, re.sub(r"(?:'s|’s|s)$", "", word)):
        if w and (sv.is_known_name(w) or sv.name_gender(w) is not None):
            return True
    return False


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
    while start < end and (text[start].isspace() or text[start] in _EDGE_PUNCT_START):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in _EDGE_PUNCT_END):
        end -= 1
    span_text = text[start:end]
    stripped, offset = _strip_leading_noise(span_text)
    if offset:
        span_text = stripped
        start += offset
    # Trim function words a model glued on: "the University of Edinburgh and".
    m = _EDGE_WORDS_START.match(span_text)
    if m and m.end() < len(span_text):
        start += m.end()
        span_text = span_text[m.end():]
    m = _EDGE_WORDS_END.search(span_text)
    if m and m.start() > 0:
        end -= len(span_text) - m.start()
        span_text = span_text[:m.start()]
    if not any(c.isalnum() for c in span_text):
        return None  # markup only ("###", "|")

    words = span_text.split()
    # Names, companies and places start with a capital letter (or a digit, for
    # English addresses): "clarify ownership", "of SEK", "regards" are not entities.
    first_alpha = next((c for c in span_text if c.isalpha()), "")
    if entity_type in ("PERSON", "COMPANY", "LOCATION", "CITY", "STREET") and first_alpha.islower() \
            and not span_text[0].isdigit():
        return None
    # Countries, regions and currencies are not anonymized, whatever the model calls them.
    if is_country_or_continent(span_text) or is_currency(span_text):
        return None
    # Form/CV field labels ("Arbetsadress", "Thesis:") and all-label spans.
    if all(is_label_word(w) for w in words):
        return None
    # A single unknown word at the start of a line or table cell is far more
    # often a heading or label than a name ("Kontaktuppgifter", "Thesis: …").
    if entity_type == "PERSON" and len(words) == 1 and _at_line_start(text, start) \
            and not _known_any_name(span_text):
        return None

    if entity_type == "PERSON":
        if _should_reject_person(span_text):
            return None
    elif entity_type == "COMPANY":
        if _should_reject_company(span_text):
            return None
        # Company names are capitalised word by word, apart from connectors:
        # "Message to international" / "Administrativ samordnare" are headings.
        if any(w[0].islower() and w.lower() not in _COMPANY_CONNECTORS and w.lower().rstrip(".,") not in _ORG_LOWER
               for w in words[1:]):
            return None
    elif entity_type in ("LOCATION", "CITY", "STREET"):
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
