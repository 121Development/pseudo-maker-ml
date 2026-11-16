"""Context rules for company names (needed once spaCy is not used)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pseudo_maker import models
from pseudo_maker.detect.recognizers.company_context import ContextCompanyRecognizer, paren_company

HAS_MODELS = models.runtime_available() and models.available(models.KB_NER) and models.available(models.RAMPART)
needs_models = pytest.mark.skipif(not HAS_MODELS, reason="ONNX models not installed")


def _found(text: str) -> list[str]:
    return [s.value for s in ContextCompanyRecognizer().find(text)]


def test_industry_suffix_and_domain_brands():
    assert _found("Please forward it to Pinewood Analytics.") == ["Pinewood Analytics"]
    assert _found("Rent is paid to Greystone Properties.") == ["Greystone Properties"]
    assert _found("Analyst, Booking.com, Amsterdam.") == ["Booking.com"]
    # Departments and example domains are not companies.
    assert _found("Our Data Analytics team and Customer Services.") == []
    assert _found("See Example.com for details.") == []


@needs_models
def test_work_cues():
    assert _found("She interned at Rolls-Royce in Derby.") == ["Rolls-Royce"]
    assert _found("Moa worked as a consultant at Netlight in Stockholm.") == ["Netlight"]
    assert _found("Marcus Lee left Tesla in 2021.") == ["Tesla"]


@needs_models
@pytest.mark.parametrize("text", [
    "She worked at Christmas and over Easter.",
    "After two years at University she joined Marketing.",
    "They left London for Paris in May.",
    "Hon arbetar på Måndagar och jobbar hos Kommunen.",
    "He joined Ben Carter for lunch.",
])
def test_work_cues_skip_everyday_words_places_and_people(text):
    assert _found(text) == []


@needs_models
def test_company_in_parentheses_after_person():
    t = "Kenji Watanabe (Toyota) and Lucia Ferreira (Galp) agreed."
    assert paren_company(t, t.index(" (Toyota")).value == "Toyota"
    for t, name in (("Anna (HR) called.", "Anna"), ("Lisa (Stockholm) came.", "Lisa"),
                    ("Maria (Ledig) och Johan.", "Maria")):
        assert paren_company(t, len(name)) is None, t


@needs_models
@pytest.mark.parametrize("gold,minimum", [
    ("ner_gold.jsonl", 0.98), ("ner_heldout.jsonl", 0.97), ("ner_heldout2.jsonl", 0.95),
])
def test_scores_without_spacy(gold: str, minimum: float):
    """KB-BERT + Rampart + rules, no spaCy: the setup the .NET port uses."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import eval_ner

    items = eval_ner.load_gold(root / "tests" / "data" / gold)
    stats, _ = eval_ner.evaluate(eval_ner.CONFIGS["kb+rampart(auto)+rules-noassist"](), items)
    assert stats["all"]["leakfree"] / stats["all"]["gold"] >= minimum
    assert stats["all"]["tp"] / stats["all"]["pred"] >= 0.97
