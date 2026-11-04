"""Detection pipeline: NER models first, hard-coded rules as the fallback layer.

  1. Models — KB-BERT (Swedish, ONNX) for persons/companies/places, Rampart
     (English, ONNX) on English-looking paragraphs. spaCy sv_core_news_lg is
     used instead of KB-BERT when the KB model is not installed.
  2. Rules — checksum/regex recognizers, Swedish company/street/case-number
     patterns, a known-company list. Always on; they catch structured IDs and
     edge cases the models miss, and win overlaps with model spans.
  3. Post-passes — house numbers, name propagation, genitives, aliases.
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Sequence

from .. import models
from .. import swedish as sv
from .recognizers.bankgiro_plusgiro import BankgiroRecognizer, PlusgiroRecognizer
from .recognizers.base import Candidate, Span
from .recognizers.company_street_se import (
    CompanySeRecognizer,
    KnownCompanySeRecognizer,
    StreetSeRecognizer,
)
from .recognizers.case_number_se import CaseNumberSeRecognizer
from .recognizers.credit_card import CreditCardRecognizer
from .recognizers.date_age import AgeRecognizer, DateRecognizer
from .recognizers.email import EmailRecognizer
from .recognizers.iban import IbanRecognizer
from .recognizers.ip_url import IpRecognizer, UrlRecognizer
from .recognizers.license_plate_se import LicensePlateSeRecognizer
from .recognizers.orgnr import OrgnrRecognizer
from .recognizers.passport import PassportRecognizer
from .recognizers.places import KnownPlaceRecognizer
from .recognizers.personnummer import PersonnummerRecognizer
from .recognizers.phone_se import PhoneSeRecognizer
from .recognizers.postal_code_se import PostalCodeSeRecognizer
from .recognizers.vat import VatRecognizer


def _default_regex_recognizers():
    return [
        # Order matters slightly: high-precision first so overlaps prefer them.
        PersonnummerRecognizer(),
        OrgnrRecognizer(),
        CaseNumberSeRecognizer(),
        IbanRecognizer(),
        CreditCardRecognizer(),
        VatRecognizer(),
        BankgiroRecognizer(),
        PlusgiroRecognizer(),
        EmailRecognizer(),
        UrlRecognizer(),
        IpRecognizer(),
        PhoneSeRecognizer(),
        LicensePlateSeRecognizer(),
        PassportRecognizer(),
        PostalCodeSeRecognizer(),
        DateRecognizer(),
        AgeRecognizer(),
        StreetSeRecognizer(),
        CompanySeRecognizer(),
        KnownCompanySeRecognizer(),
        KnownPlaceRecognizer(),
    ]


NerMode = Literal["auto", "kb", "spacy", "off"]
EnglishMode = Literal["auto", "on", "off"]

log = logging.getLogger(__name__)
_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        log.warning(msg)


def _make_swedish_ner(mode: NerMode, spacy_model: str):
    if mode == "off":
        return None
    if mode in ("auto", "kb") and models.runtime_available() and models.available(models.KB_NER):
        from .kb_ner import KbSwedishNerRecognizer
        return KbSwedishNerRecognizer()
    if mode == "kb":
        raise FileNotFoundError("KB-BERT model not installed; run `pseudo-maker fetch-models`")
    if not _spacy_model_installed(spacy_model):
        if mode == "spacy":
            raise FileNotFoundError(f"spaCy model {spacy_model} not installed")
        _warn_once("No Swedish NER model available (KB-BERT or spaCy); using rules only.")
        return None
    from .ner_swedish import SwedishNerRecognizer
    if mode == "auto":
        _warn_once("KB-BERT model not installed; falling back to spaCy. Run `pseudo-maker fetch-models`.")
    return SwedishNerRecognizer(spacy_model)


def _spacy_model_installed(spacy_model: str) -> bool:
    try:
        import spacy.util
    except ImportError:
        return False
    return bool(spacy.util.is_package(spacy_model))


def _make_spacy_assist(spacy_model: str):
    if not _spacy_model_installed(spacy_model):
        return None
    from .ner_swedish import SwedishNerRecognizer
    return SwedishNerRecognizer(spacy_model)


def _make_english_ner(mode: EnglishMode):
    if mode == "off":
        return None
    if models.runtime_available() and models.available(models.RAMPART):
        from .rampart_ner import RampartRecognizer
        return RampartRecognizer()
    if mode == "on":
        raise FileNotFoundError("Rampart model not installed; run `pseudo-maker fetch-models`")
    _warn_once("Rampart model not installed; English text uses the Swedish model only.")
    return None


class DetectionPipeline:
    def __init__(
        self,
        *,
        use_ner: bool = True,
        ner: NerMode = "auto",
        english_ner: EnglishMode = "auto",
        ner_model: str = "sv_core_news_lg",
    ):
        if not use_ner:
            ner, english_ner = "off", "off"
        self.regex_recognizers = _default_regex_recognizers()
        self.ner = _make_swedish_ner(ner, ner_model)
        self.english_ner = _make_english_ner(english_ner)
        self.english_mode = english_ner
        # Rampart has no company label. When KB-BERT is the main model and
        # spaCy is installed too, spaCy also reads English paragraphs: on the
        # held-out set it catches English company names the others miss.
        self.english_assist = None
        if english_ner != "off" and self.ner is not None and type(self.ner).__name__ == "KbSwedishNerRecognizer":
            self.english_assist = _make_spacy_assist(ner_model)

    def detect(self, text: str) -> list[Span]:
        return self.detect_with_candidates(text)[0]

    def detect_with_candidates(self, text: str) -> tuple[list[Span], list[Candidate]]:
        """Accepted spans, plus ambiguous candidates that were *not* accepted.

        Candidates are model near-misses (low score, or dropped by a filter)
        and capitalised words that are known names but weren't detected. They
        are not filtered against the spans here; see `filter_candidates`.
        """
        spans: list[Span] = []
        rejected: list[Candidate] = []
        for r in self.regex_recognizers:
            spans.extend(r.find(text))
        if self.ner is not None:
            spans.extend(self.ner.find(text, rejected))
        english = self.english_mode == "on" or sv.looks_english(text)
        if self.english_ner is not None and english:
            spans.extend(self.english_ner.find(text, rejected))
        if self.english_assist is not None and english:
            spans.extend(self.english_assist.find(text, rejected))
        spans = [_extend_street_number(text, s) for s in spans]
        spans = _resolve_overlaps(spans)
        rejected.extend(_known_name_candidates(text))
        return spans, rejected


_CAP_WORD = re.compile(r"(?<![\w-])[A-ZÅÄÖ][a-zåäöéü]{2,}(?:-[A-ZÅÄÖ][a-zåäöéü]+)?(?![\w-])")
_SENTENCE_START = re.compile(r"(?:^|[.!?:;•\-–—\n\t(\"]|\d)\s*$")


def _known_name_candidates(text: str) -> list[Candidate]:
    """Capitalised mid-sentence words that are known Swedish first names/surnames."""
    from .blocklist import is_blocklisted_person, is_country_or_continent

    out = []
    for m in _CAP_WORD.finditer(text):
        word = m.group(0)
        if _SENTENCE_START.search(text[max(0, m.start() - 3):m.start()]) or m.start() == 0:
            continue
        if not sv.is_known_name(word) or is_blocklisted_person(word) or is_country_or_continent(word):
            continue
        out.append(Candidate(m.start(), m.end(), "PERSON", word, 0.5, "known first name or surname"))
    return out


def filter_candidates(spans: Sequence[Span], candidates: Sequence[Candidate]) -> list[Candidate]:
    """Drop candidates that overlap an accepted span; among overlapping
    candidates keep the highest-scoring one."""
    kept: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: (-c.score, c.start)):
        if any(c.start < s.end and s.start < c.end for s in spans):
            continue
        if any(c.start < k.end and k.start < c.end for k in kept):
            continue
        kept.append(c)
    return sorted(kept, key=lambda c: c.start)


_HOUSE_NUMBER = re.compile(r"[ \t]+\d{1,4}(?:[A-Za-z]|[ \t][A-Z])?(?![\wåäö])")


def propagate_person_names(texts: Sequence[str], spans_per_text: list[list[Span]]) -> list[list[Span]]:
    """Find further mentions of names that NER caught somewhere in the document.

    spaCy is inconsistent: it may tag "Johanna Ek" but miss "Ek noterade ..."
    two sentences later, or tag "MOHAMMED AL-HASSAN" but miss "Fatima
    Al-Hassan". Every name part of a detected PERSON is searched for across
    the whole document (Title Case or ALL CAPS, optional genitive -s). A known
    first name directly before a found surname is included in the span.
    """
    from .. import swedish as sv
    from .blocklist import is_blocklisted_person

    parts: set[str] = set()
    for spans in spans_per_text:
        for s in spans:
            if s.type != "PERSON":
                continue
            for tok in s.value.split():
                if (tok.lower() in sv.NAME_PARTICLES or len(tok) < 2 or is_blocklisted_person(tok)
                        or not tok[0].isupper()):
                    continue
                if tok.isupper():
                    tok = "-".join(p[:1] + p[1:].lower() for p in tok.split("-"))
                parts.add(tok)
    if not parts:
        return spans_per_text

    alts = "|".join(re.escape(p) for p in sorted(parts, key=len, reverse=True))
    alts_upper = "|".join(re.escape(p.upper()) for p in sorted(parts, key=len, reverse=True))
    pat = re.compile(rf"(?<![\w\-])(?:{alts}|{alts_upper})(?:s|S)?(?![\w\-])")
    first_before = re.compile(r"([A-ZÅÄÖÉ][a-zåäöéü]+(?:-[A-ZÅÄÖÉ][a-zåäöéü]+)?)[ \t]+$")

    out: list[list[Span]] = []
    for text, spans in zip(texts, spans_per_text):
        new = list(spans)
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            if any(start < s.end and s.start < end for s in new):
                continue
            fm = first_before.search(text, 0, start)
            if fm and sv.is_known_first_name(fm.group(1)) and not any(
                fm.start(1) < s.end and s.start < start for s in new
            ):
                start = fm.start(1)
            new.append(Span(start, end, "PERSON", text[start:end], 0.75, "ner"))
        new.sort(key=lambda s: s.start)
        out.append(new)
    return out


def _extend_street_number(text: str, span: Span) -> Span:
    """NER often tags just 'Storgatan' in 'Storgatan 12B'; pull in the number."""
    if span.type != "STREET" or span.source != "ner" or any(c.isdigit() for c in span.value):
        return span
    m = _HOUSE_NUMBER.match(text, span.end)
    if not m:
        return span
    end = m.end()
    return Span(span.start, end, span.type, text[span.start:end], span.confidence, span.source)


def _resolve_overlaps(spans: Sequence[Span]) -> list[Span]:
    """When two spans overlap, keep the higher-priority one.

    Priority: regex IDs first; then (models and name patterns alike) longer >
    shorter, higher score, earlier.
    """
    def priority(s: Span) -> tuple:
        return (
            0 if s.source == "regex" else 1,   # checksum/regex IDs win; patterns compete with models
            -(s.end - s.start),                # longer wins
            0 if s.source == "pattern" else 1, # same span: explicit lists/patterns beat models
            -s.confidence,                     # then the more confident model
            s.start,                           # earlier wins
        )
    sorted_spans = sorted(spans, key=priority)
    accepted: list[Span] = []
    for s in sorted_spans:
        if not any(_overlaps(s, a) for a in accepted):
            accepted.append(s)
    accepted.sort(key=lambda s: s.start)
    return accepted


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end
