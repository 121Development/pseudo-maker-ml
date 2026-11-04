"""KB-BERT and Rampart recognizers. Skipped when the ONNX models are not installed."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

from pseudo_maker import models
from pseudo_maker.detect.kb_ner import group_io
from pseudo_maker.detect.models.onnx_tagger import WordPrediction
from pseudo_maker.detect.rampart_ner import decode_bio
from pseudo_maker.substitute.fakes import fake_id_number
from pseudo_maker.swedish import looks_english

HAS_KB = models.runtime_available() and models.available(models.KB_NER)
HAS_RAMPART = models.runtime_available() and models.available(models.RAMPART)
needs_kb = pytest.mark.skipif(not HAS_KB, reason="KB-BERT model not installed")
needs_rampart = pytest.mark.skipif(not HAS_RAMPART, reason="Rampart model not installed")


def _w(text: str, word: str, label: str, score: float = 0.99, after: int = 0) -> WordPrediction:
    i = text.index(word, after)
    return WordPrediction(i, i + len(word), label, score)


# --- decoding (no model needed) ------------------------------------------------

def test_group_io_merges_hyphen_names_and_splits_on_labels():
    t = "Anna-Karin Lindqvist på Volvo"
    words = [_w(t, "Anna", "PER"), _w(t, "-", "PER"), _w(t, "Karin", "PER"),
             _w(t, "Lindqvist", "PER"), _w(t, "på", "O"), _w(t, "Volvo", "ORG")]
    groups = group_io(t, words)
    assert [(t[s:e], lab) for s, e, lab, _ in groups] == [("Anna-Karin Lindqvist", "PER"), ("Volvo", "ORG")]


def test_decode_bio_joins_given_and_surname_and_street_parts():
    t = "John Miller at 42 Baker Street, Leeds"
    words = [_w(t, "John", "B-GIVEN_NAME"), _w(t, "Miller", "B-SURNAME"), _w(t, "at", "O"),
             _w(t, "42", "B-BUILDING_NUMBER"), _w(t, "Baker", "B-STREET_NAME"),
             _w(t, "Street", "I-STREET_NAME"), _w(t, ",", "O"), _w(t, "Leeds", "B-CITY")]
    out = [(t[s:e], typ) for s, e, typ, _ in decode_bio(t, words)]
    assert out == [("John Miller", "PERSON"), ("42 Baker Street", "STREET"), ("Leeds", "CITY")]


def test_looks_english():
    assert looks_english("Please contact John Miller at the office")
    assert not looks_english("Anna arbetar på Volvo och bor i Malmö")
    assert looks_english("Per Åkesson")  # no stop words → let both models look


def test_fake_id_number_keeps_shape():
    fake = fake_id_number("AB-1234-xy", random.Random(0))
    assert re.fullmatch(r"[A-Z]{2}-\d{4}-[a-z]{2}", fake) and fake != "AB-1234-xy"


# --- KB-BERT -------------------------------------------------------------------

@needs_kb
def test_kb_offsets_with_swedish_letters_and_caps():
    from pseudo_maker.detect.kb_ner import KbSwedishNerRecognizer

    text = "Klient: MOHAMMED AL-HASSAN. Per Åkessons kontor ligger i Västerås."
    found = {(s.type, s.value) for s in KbSwedishNerRecognizer().find(text)}
    assert ("PERSON", "MOHAMMED AL-HASSAN") in found
    assert ("PERSON", "Per Åkessons") in found
    assert ("CITY", "Västerås") in found
    for s in KbSwedishNerRecognizer().find(text):
        assert text[s.start:s.end] == s.value


@needs_kb
def test_kb_splits_names_joined_by_och():
    from pseudo_maker.detect.kb_ner import KbSwedishNerRecognizer

    values = [s.value for s in KbSwedishNerRecognizer().find("De har två barn, Ebba och Hugo.")]
    assert "Ebba" in values and "Hugo" in values
    assert not any("och" in v for v in values)


@needs_kb
def test_kb_skips_products_and_agencies():
    from pseudo_maker.detect.kb_ner import KbSwedishNerRecognizer

    text = "Rapporten skickades via Teams till Skatteverket och Trafikverket."
    assert list(KbSwedishNerRecognizer().find(text)) == []


# --- Rampart -------------------------------------------------------------------

@needs_rampart
def test_rampart_finds_english_pii():
    from pseudo_maker.detect.rampart_ner import RampartRecognizer

    text = "Please contact John Miller at 42 Baker Street, Manchester."
    found = {(s.type, s.value) for s in RampartRecognizer().find(text)}
    assert ("PERSON", "John Miller") in found
    assert ("STREET", "42 Baker Street") in found
    assert ("CITY", "Manchester") in found


@needs_rampart
def test_rampart_ignores_lone_low_confidence_word():
    from pseudo_maker.detect.rampart_ner import RampartRecognizer

    assert list(RampartRecognizer().find("Belopp")) == []


@needs_kb
@needs_rampart
def test_pipeline_runs_rampart_only_on_english(monkeypatch):
    from pseudo_maker.detect.pipeline import DetectionPipeline

    p = DetectionPipeline(ner="kb", english_ner="auto")
    calls: list[str] = []
    real = p.english_ner.find
    monkeypatch.setattr(p.english_ner, "find", lambda t, *a: calls.append(t) or real(t, *a))
    p.detect("Anna arbetar på Volvo och bor i Malmö.")
    assert calls == []
    p.detect("Please call John Miller at the office.")
    assert calls == ["Please call John Miller at the office."]


@needs_kb
@needs_rampart
def test_english_cv_docx_round_trip(tmp_path: Path):
    from docx import Document

    from pseudo_maker import orchestrate

    doc = Document()
    doc.add_heading("ANNA SVENSSON – CURRICULUM VITAE", level=1)
    doc.add_paragraph("Anna has ten years of experience in logistics at Volvo Cars in Gothenburg.")
    doc.add_paragraph("References: Emma Thompson, Head of HR at Unilever, 15 Oak Avenue, Bristol.")
    src = tmp_path / "cv.docx"
    doc.save(str(src))
    cleaned, mapping, restored = tmp_path / "c.docx", tmp_path / "m.json", tmp_path / "r.docx"
    orchestrate.clean_docx(src, cleaned, mapping)
    text = "\n".join(p.text for p in Document(str(cleaned)).paragraphs)
    for leak in ("ANNA SVENSSON", "Anna ", "Emma Thompson", "Unilever", "Oak Avenue", "Bristol", "Gothenburg"):
        assert leak not in text, (leak, text)
    orchestrate.restore_docx(cleaned, restored, mapping)
    assert [p.text for p in Document(str(restored)).paragraphs] == [p.text for p in doc.paragraphs]


# --- gold-set regression -------------------------------------------------------

@needs_kb
@needs_rampart
@pytest.mark.parametrize("gold,minimum", [("ner_gold.jsonl", 0.97), ("ner_heldout.jsonl", 0.93)])
def test_gold_set_leak_free_recall(gold: str, minimum: float):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import eval_ner

    items = eval_ner.load_gold(root / "tests" / "data" / gold)
    stats, _ = eval_ner.evaluate(eval_ner.CONFIGS["kb+rampart(auto)+rules"](), items)
    recall = stats["all"]["leakfree"] / stats["all"]["gold"]
    precision = stats["all"]["tp"] / stats["all"]["pred"]
    assert recall >= minimum, recall
    assert precision >= 0.95, precision
