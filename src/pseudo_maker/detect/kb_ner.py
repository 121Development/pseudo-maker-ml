"""Swedish NER with KB-BERT (KBLab/bert-base-swedish-cased-ner), via ONNX.

The model is trained on SUC 3.0 and tags words with IO labels (no B-/I-):
PER/PRS (person), ORG, LOC, plus TME, MSR, EVN, WRK, OBJ and mixed labels
such as ORG/PRS. Neighbouring words with the same label form one entity.
"""
from __future__ import annotations

import re
from typing import Iterable

from .. import models
from .. import swedish as sv
from .models.onnx_tagger import WordPrediction, load_tagger
from .ner_postprocess import (
    REJECT_FLOOR,
    _titlecase_caps,
    mask_markup,
    filter_reason,
    note_rejected,
    postprocess,
    split_conjunctions,
)
from .recognizers.base import Span

_LEGAL_OR_PUBLIC = re.compile(
    r"\b(AB|HB|KB|Aktiebolag|kommun|region|landsting|län|stad|universitet|högskola|"
    r"myndighet|verket|förening|stiftelse)\b",
    re.IGNORECASE,
)

# What may sit between two words of one entity: "Anna-Karin", "H&M", "S:t", one
# space. Wider gaps (signature lines, table-like layouts) separate entities.
_BRIDGE = re.compile(r"(?: ?[\-&:'] ?| )?")


def _map_label(label: str, text: str) -> str | None:
    if label in ("PER", "PRS"):
        return "PERSON"
    if label in ("ORG", "OBJ/ORG"):
        return "COMPANY"
    if label == "LOC":
        return "LOCATION"
    if label in ("ORG/PRS", "PRS/WRK", "LOC/PRS"):
        parts = [p for p in re.split(r"[\s\-]+", text) if p]
        if parts and all(sv.is_known_name(p) or p.lower() in sv.NAME_PARTICLES for p in parts):
            return "PERSON"
        return "COMPANY" if label == "ORG/PRS" else ("LOCATION" if label == "LOC/PRS" else None)
    if label == "LOC/ORG":
        return "COMPANY" if _LEGAL_OR_PUBLIC.search(text) else "LOCATION"
    return None  # TME, MSR, EVN, WRK, OBJ: dates/amounts are handled by regex


def group_io(text: str, words: list[WordPrediction]) -> list[tuple[int, int, str, float]]:
    """Merge neighbouring same-label words into (start, end, label, mean score)."""
    groups: list[list[WordPrediction]] = []
    for w in words:
        if w.label == "O":
            continue
        if groups:
            prev = groups[-1][-1]
            if prev.label == w.label and _BRIDGE.fullmatch(text[prev.end:w.start]):
                groups[-1].append(w)
                continue
        groups.append([w])
    return [
        (g[0].start, g[-1].end, g[0].label, sum(x.score for x in g) / len(g))
        for g in groups
    ]


class KbSwedishNerRecognizer:
    name = "NER"

    def __init__(self, model_dir: str | None = None, *, min_score: float = 0.5):
        d = model_dir or models.model_dir(models.KB_NER)
        if d is None:
            raise FileNotFoundError("KB-BERT NER model not installed (run `pseudo-maker fetch-models`)")
        self._model_dir = str(d)
        self.min_score = min_score

    def find(self, text: str, rejected: list | None = None) -> Iterable[Span]:
        """Yield accepted spans; append near-misses to `rejected` if given."""
        return self.find_many([text], [rejected])[0]

    def find_many(self, texts: list[str], rejected: list[list | None] | None = None) -> list[list[Span]]:
        """Batched find(): one model call per batch of paragraphs."""
        rejected = rejected or [None] * len(texts)
        tagger = load_tagger(self._model_dir)
        # KB-BERT is cased: ALL-CAPS names ("MOHAMMED AL-HASSAN") look like
        # acronyms to it. Title-case them first (same length → same offsets).
        tagged = tagger.tag_many([_titlecase_caps(mask_markup(t)) for t in texts])
        return [list(self._spans(t, words, rej)) for t, words, rej in zip(texts, tagged, rejected)]

    def _spans(self, text: str, words, rejected: list | None) -> Iterable[Span]:
        for start, end, label, score in group_io(text, words):
            mapped = _map_label(label, text[start:end])
            if mapped is None:
                continue
            if score < self.min_score:
                if score >= REJECT_FLOOR:
                    note_rejected(rejected, mapped, text, start, end, score, "low model confidence")
                continue
            pieces = split_conjunctions(text, start, end) if mapped == "PERSON" else [(start, end)]
            for a, b in pieces:
                span = postprocess(mapped, text, a, b, confidence=round(score, 3))
                if span is not None:
                    yield span
                elif mapped != "PERSON":
                    note_rejected(rejected, mapped, text, a, b, score, filter_reason(mapped, text[a:b]))
