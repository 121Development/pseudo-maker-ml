"""English (and other Latin-script) PII NER with Rampart, via ONNX.

nationaldesignstudio/rampart: MiniLM token classifier with 35 BIO labels
(GIVEN_NAME, SURNAME, STREET_NAME, BUILDING_NUMBER, CITY, PHONE, …).
The tokenizer is uncased and strips accents, so names are only accepted if
the original text is capitalised, and the model is not used for companies
(it has no ORG label).
"""
from __future__ import annotations

from typing import Iterable

from .. import models
from .. import swedish as sv
from .models.onnx_tagger import WordPrediction, load_tagger
from .ner_postprocess import (
    REJECT_FLOOR,
    filter_reason,
    mask_markup,
    note_rejected,
    postprocess,
    split_conjunctions,
)
from .recognizers.base import Span

# Rampart type → (group key, pseudo-maker type). Types in the same group that
# sit next to each other merge: GIVEN_NAME + SURNAME → one PERSON.
_TYPE_MAP = {
    "GIVEN_NAME": ("PERSON", "PERSON"),
    "SURNAME": ("PERSON", "PERSON"),
    "STREET_NAME": ("STREET", "STREET"),
    "BUILDING_NUMBER": ("STREET", "STREET"),
    "SECONDARY_ADDRESS": ("STREET", "STREET"),
    "CITY": ("CITY", "CITY"),
    "PHONE": ("PHONE", "PHONE"),
    "EMAIL": ("EMAIL", "EMAIL"),
    "URL": ("URL", "URL"),
    "PASSPORT": ("PASSPORT", "PASSPORT"),
    "TAX_ID": ("ID", "ID_NUMBER"),
    "BANK_ACCOUNT": ("ID", "ID_NUMBER"),
    "ROUTING_NUMBER": ("ID", "ID_NUMBER"),
    "GOVERNMENT_ID": ("ID", "ID_NUMBER"),
    "DRIVERS_LICENSE": ("ID", "ID_NUMBER"),
    # STATE / ZIP_CODE: dropped (Swedish postal codes are handled by regex).
}

_JOIN_GAP = set(" \t-'.,")


def decode_bio(text: str, words: list[WordPrediction]) -> list[tuple[int, int, str, float]]:
    """BIO words → (start, end, pseudo-maker type, mean score) entities."""
    out: list[tuple[int, int, str, float]] = []
    cur: list | None = None  # [start, end, group, type, scores]
    for w in words:
        if w.label == "O":
            if cur:
                out.append(cur)
                cur = None
            continue
        tag, _, rtype = w.label.partition("-")
        mapped = _TYPE_MAP.get(rtype)
        if mapped is None:
            if cur:
                out.append(cur)
                cur = None
            continue
        group, ptype = mapped
        gap = text[cur[1]:w.start] if cur else ""
        joinable = cur is not None and cur[2] == group and all(c in _JOIN_GAP for c in gap) and len(gap) <= 3
        if joinable and (tag == "I" or group in ("PERSON", "STREET")):
            cur[1] = w.end
            cur[4].append(w.score)
        else:
            if cur:
                out.append(cur)
            cur = [w.start, w.end, group, ptype, [w.score]]
    if cur:
        out.append(cur)
    return [(s, e, t, sum(sc) / len(sc)) for s, e, _g, t, sc in out]


class RampartRecognizer:
    name = "RAMPART"

    def __init__(self, model_dir: str | None = None, *, min_score: float = 0.4):
        d = model_dir or models.model_dir(models.RAMPART)
        if d is None:
            raise FileNotFoundError("Rampart model not installed (run `pseudo-maker fetch-models`)")
        self._model_dir = str(d)
        self.min_score = min_score

    def find(self, text: str, rejected: list | None = None) -> Iterable[Span]:
        """Yield accepted spans; append near-misses to `rejected` if given."""
        tagger = load_tagger(self._model_dir)
        for start, end, ptype, score in decode_bio(text, tagger.tag(mask_markup(text))):
            value = text[start:end]
            if score < self.min_score:
                if score >= REJECT_FLOOR and ptype in ("PERSON", "CITY", "STREET", "ID_NUMBER"):
                    note_rejected(rejected, ptype, text, start, end, score, "low model confidence")
                continue
            # Uncased model: require a capital letter for names and places.
            if ptype in ("PERSON", "CITY", "STREET") and not any(c.isupper() for c in value):
                continue
            # A lone capitalised word (table header, heading) needs either a
            # confident prediction or to be a known name.
            if ptype == "PERSON" and " " not in value.strip() and score < 0.8 and not sv.is_known_name(value):
                note_rejected(rejected, ptype, text, start, end, score, "single word, low confidence")
                continue
            # Rampart strips accents, so "Lön" looks like the name "Lon" to it.
            # A lone word with å/ä/ö must be a known name to count as a person.
            if (ptype == "PERSON" and " " not in value.strip() and any(c in "åäöÅÄÖ" for c in value)
                    and not sv.is_known_name(value)):
                continue
            pieces = split_conjunctions(text, start, end) if ptype == "PERSON" else [(start, end)]
            for a, b in pieces:
                span = postprocess(ptype, text, a, b, confidence=round(score, 3))
                if span is not None:
                    yield span
                elif ptype != "PERSON":
                    note_rejected(rejected, ptype, text, a, b, score, filter_reason(ptype, text[a:b]))
