"""Swedish NER via spaCy sv_core_news_lg.

Maps spaCy entity labels to pseudo-maker types, and applies a post-filter
that drops obvious false positives:

  - Single-token lowercase PERSON matches are almost always wrong (spaCy
    confuses common English nouns with Swedish given names).
  - Anything in the PERSON blocklist (e.g. "site", "turnover", "owner").
  - Spans containing "/" or matching a role word ("OWNER/FOUNDER").
  - LOC spans matching a country/continent name are passed through unchanged
    (not anonymized) — they're rarely identifying on their own.
  - LOC spans are split into CITY (single capitalized token, no digits/comma)
    vs STREET (anything else), so the substitution layer can pick the right
    fake generator.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

from .ner_postprocess import (  # noqa: F401  (re-exported for tests/back-compat)
    _classify_location,
    _should_reject_company,
    _should_reject_person,
    _strip_leading_noise,
    _titlecase_caps,
    mask_markup,
    filter_reason,
    note_rejected,
    postprocess,
)
from .recognizers.base import Span

_LABEL_MAP = {
    "PRS": "PERSON",
    "PER": "PERSON",
    "ORG": "COMPANY",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
}


@lru_cache(maxsize=1)
def _load_model(model_name: str = "sv_core_news_lg"):
    import spacy
    return spacy.load(model_name)


class SwedishNerRecognizer:
    name = "NER"

    def __init__(self, model_name: str = "sv_core_news_lg"):
        self._model_name = model_name
        self._nlp = None

    def _ensure(self):
        if self._nlp is None:
            self._nlp = _load_model(self._model_name)
        return self._nlp

    def find(self, text: str, rejected: list | None = None) -> Iterable[Span]:
        return self.find_many([text], [rejected])[0]

    def find_many(self, texts: list[str], rejected: list[list | None] | None = None) -> list[list[Span]]:
        """Batched find() via nlp.pipe."""
        rejected = rejected or [None] * len(texts)
        nlp = self._ensure()
        # spaCy misses names written in capitals ("MOHAMMED AL-HASSAN"), common
        # in Swedish headings and forms. Title-case all-caps words before NER;
        # same length, so offsets map straight back to the original text.
        docs = nlp.pipe((_titlecase_caps(mask_markup(t)) for t in texts), batch_size=64)
        return [list(self._spans(t, doc, rej)) for t, doc, rej in zip(texts, docs, rejected)]

    def _spans(self, text: str, doc, rejected: list | None) -> Iterable[Span]:
        for ent in doc.ents:
            mapped = _LABEL_MAP.get(ent.label_)
            if not mapped:
                continue

            span = postprocess(mapped, text, ent.start_char, ent.end_char, confidence=0.85)
            if span is not None:
                yield span
            elif mapped != "PERSON":
                note_rejected(rejected, mapped, text, ent.start_char, ent.end_char, 0.85,
                              filter_reason(mapped, text[ent.start_char:ent.end_char]))
