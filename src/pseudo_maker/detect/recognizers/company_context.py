"""Company names found from their context rather than a list (source="pattern").

Rampart has no company label and KB-BERT misses many English company names,
so these rules cover the common shapes in CVs, contracts and meeting notes:

  - an industry word at the end:     "Pinewood Analytics", "Greystone Properties"
  - a brand that is a domain:        "Booking.com", "Expedia.com"
  - a work-context cue before it:    "interned at Rolls-Royce", "joined Netlight"

A person followed by a company in parentheses ("Kenji Watanabe (Toyota)") is
handled in the pipeline, since it needs the person spans.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Iterable

from ... import models

from .. import blocklist
from ... import swedish as sv
from .base import Span
from .places import SWEDISH_PLACES, WORLD_CITIES

_PLACES = frozenset(p.lower() for p in SWEDISH_PLACES + WORLD_CITIES)

# One capitalised name word: "Rolls-Royce", "McKinsey", "Nando's", "KPMG".
_NAME_WORD = r"[A-ZÅÄÖÉ][\wåäöéü&]*(?:-[A-ZÅÄÖÉ\w][\wåäöéü]*)*(?:'s)?"

# Generic words that end many company names. A department or job ("Customer
# Services", "Data Analytics") has a generic word first; see _GENERIC_FIRST.
_INDUSTRY = (
    "Analytics|Properties|Property|Consulting|Consultants|Logistics|Engineering|Systems|"
    "Solutions|Technologies|Technology|Labs|Partners|Capital|Holdings|Ventures|Industries|"
    "Software|Media|Studios|Networks|Pharmaceuticals|Pharma|Foods|Energy|Insurance|"
    "Associates|Enterprises|Investments|Advisors|Advisory|Robotics|Motors|Airlines|"
    "Construction|Developments|Estates|Healthcare|Therapeutics|Biosciences|Interactive|"
    "Trust|Foundation|Brewery|Bakery|Builders|Recruitment|Security|Telecom|Finance"
)
_SUFFIXED = re.compile(rf"(?<![\w-])((?:{_NAME_WORD}[ \t]+){{1,3}}(?:{_INDUSTRY}))(?![\w-])")
_GENERIC_FIRST = frozenset("""
customer customers financial business data digital it human legal technical professional
social health group head senior junior global local national international public private
corporate regional general new our your their my his her this that these those all
information software network security energy property real investment marketing sales
research clinical medical risk asset wealth capital supply product strategic management
""".split())

_DOMAIN = re.compile(r"(?<![\w@./:-])([A-Z][A-Za-z0-9-]+\.(?:com|se|io|ai|net|org|co\.uk|nu|app))(?![\w/.@-])")

# A capitalised name after a work verb: "worked at X", "joined X", "left X".
_CUE = re.compile(
    r"(?:\b(?:work(?:s|ed|ing)?|intern(?:s|ed|ing)?|employed|hired|consultant|engineer|developer|"
    r"manager|analyst|internship|position|role|job|career|years?|months?|volunteer(?:s|ed|ing)?)"
    r"[ \t]+(?:at|for|by|with)"
    r"|\b(?:join(?:s|ed|ing)?|left|leaving|founded|co-founded)"
    r"|\b(?:arbetar|arbetade|jobbar|jobbade|anställd|anställdes|praktiserade|konsult)[ \t]+(?:på|hos|vid))"
    rf"[ \t]+({_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,2}})"
)


@lru_cache(maxsize=1)
def _common_words() -> frozenset[str] | None:
    """Whole words in the models' vocabularies, lowercased.

    A word the tokenizers keep whole in lowercase ("christmas", "home",
    "ledig", "kommunen") is an everyday word, not a company name; brands
    like "Toyota" or "Netlight" are split into pieces. None if no model is
    installed: the context rules then stay off.
    """
    words: set[str] = set()
    for name in (models.KB_NER, models.RAMPART):
        d = models.model_dir(name)
        if d is None:
            continue
        vocab = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
        # Only entries stored in lowercase: KB-BERT is cased, so "Toyota" is a name there.
        words.update(w for w in vocab if len(w) > 1 and w.isalpha() and w.islower())
    return frozenset(words) if words else None


def _common(word: str) -> bool:
    common = _common_words()
    return common is not None and _plain(word).lower() in common


_EXAMPLE_DOMAINS = {"example.com", "example.org", "example.net", "test.com"}


def _plain(word: str) -> str:
    return word[:-2] if word.endswith("'s") else word


def _rejected(value: str) -> bool:
    """Words that are not company names whatever their context."""
    words = [_plain(w) for w in value.split()]
    joined = " ".join(words)
    low = joined.lower()
    if low in _PLACES or blocklist.is_country_or_continent(joined) or blocklist.is_currency(joined):
        return True
    if blocklist.is_blocklisted_company(joined) or blocklist.is_job_title(joined):
        return True
    if all(blocklist.is_label_word(w) or blocklist.is_blocklisted_person(w) for w in words):
        return True
    # Everyday words: "worked at Christmas", "Maria (Ledig)", "Customer Services".
    if all(_common(w) for w in words):
        return True
    # Starting with a known first name, it's a person ("joined Ben Carter").
    if sv.is_known_first_name(words[0]):
        return True
    return False


class ContextCompanyRecognizer:
    name = "COMPANY"

    def find(self, text: str) -> Iterable[Span]:
        for m in _SUFFIXED.finditer(text):
            value = m.group(1)
            head = value.rsplit(None, 1)[0]
            if value.split()[0].lower() in _GENERIC_FIRST or _rejected(head):
                continue
            yield Span(m.start(1), m.end(1), self.name, value, 0.8, "pattern")
        for m in _DOMAIN.finditer(text):
            if m.group(1).lower() in _EXAMPLE_DOMAINS:
                continue
            yield Span(m.start(1), m.end(1), self.name, m.group(1), 0.8, "pattern")
        if _common_words() is None:
            return
        for m in _CUE.finditer(text):
            value = m.group(1)
            if _rejected(value):
                continue
            yield Span(m.start(1), m.end(1), self.name, value, 0.7, "pattern")


_PAREN = re.compile(rf"[ \t]*\(({_NAME_WORD}(?:[ \t]+{_NAME_WORD}){{0,2}})\)")


def paren_company(text: str, pos: int) -> Span | None:
    """A company in parentheses starting at `pos` (right after a person's name)."""
    m = _PAREN.match(text, pos)
    if not m or _common_words() is None or _rejected(m.group(1)):
        return None
    return Span(m.start(1), m.end(1), "COMPANY", m.group(1), 0.75, "pattern")
