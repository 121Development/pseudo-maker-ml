"""Regressions from a review of a synthetic Swedish/English CV
(tests/data/fake_cv_swedish_english.md): clean → shorten → restore."""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from pseudo_maker import models, orchestrate
from pseudo_maker.detect.recognizers.company_street_se import CompanySeRecognizer, StreetSeRecognizer
from pseudo_maker.detect.recognizers.personnummer import PersonnummerRecognizer
from pseudo_maker.detect.recognizers.postal_code_se import UkPostcodeRecognizer
from pseudo_maker.restore.restore import restore
from pseudo_maker.substitute.fakes import fake_phone
from pseudo_maker.substitute.mapper import Mapper
from pseudo_maker.swedish import phone_key, render_phone

CV = Path(__file__).parent / "data" / "fake_cv_swedish_english.md"
HAS_MODELS = models.runtime_available() and models.available(models.KB_NER) and models.available(models.RAMPART)
needs_models = pytest.mark.skipif(not HAS_MODELS, reason="KB-BERT and Rampart models not installed")


def _values(recognizer, text):
    return [s.value for s in recognizer.find(text)]


# --- rules ------------------------------------------------------------------------

def test_personnummer_with_bad_checksum_is_detected_with_cue_or_12_digit_form():
    text = "Personnummer: 19890417-2840. Pnr 890417-2841. Order 8904172840. Telefon 0701234567."
    found = _values(PersonnummerRecognizer(), text)
    assert "19890417-2840" in found and "890417-2841" in found
    assert "8904172840" not in found and "0701234567" not in found


def test_uk_postcodes_and_english_addresses():
    assert _values(UkPostcodeRecognizer(), "London, SW1A 1AA. Bristol BS1 1AA. M1 1AE") == ["SW1A 1AA", "BS1 1AA", "M1 1AE"]
    found = _values(StreetSeRecognizer(), "Office: 14 Example Orchard Lane, London. I have 3 Close friends.")
    assert found == ["14 Example Orchard Lane"]


def test_company_pattern_does_not_swallow_person_before_på():
    text = "Följde upp kundärenden med Fatima Norén på Fiktiva Kundbolaget AB."
    assert _values(CompanySeRecognizer(), text) == ["Fiktiva Kundbolaget AB"]


# --- phones -------------------------------------------------------------------------

def test_phone_formats_share_one_key_and_render_in_each_format():
    same = ["070-000 01 21", "+46 70 000 01 21", "+46700000121", "+46 (0)70 000 01 21", "0046 70 000 01 21"]
    assert {phone_key(v) for v in same} == {"0700000121"}
    assert render_phone("+46 (0)70 000 01 21", "0739876543") == "+46 (0)73 987 65 43"
    assert phone_key("+44 7700 900121") == phone_key("+44 (0)7700 900121") == "+447700900121"


def test_same_number_in_different_formats_gets_consistent_fakes():
    m = Mapper(seed=1)
    a = m.map("PHONE", "070-000 01 21")
    b = m.map("PHONE", "+46 70 000 01 21")
    c = m.map("PHONE", "+46700000121")
    assert a is b is c
    fakes = {m.fake_for_alias(a, v) for v in ("070-000 01 21", "+46 70 000 01 21", "+46700000121")}
    assert {phone_key(f) for f in fakes} == {phone_key(a.fake)} and len(fakes) == 3


def test_foreign_phone_keeps_country_code_and_first_digit():
    rng = random.Random(0)
    for _ in range(20):
        f = fake_phone("+44 7700 900121", rng)
        assert f.startswith("+44 7") and len(f) == len("+44 7700 900121")


# --- names / urls ---------------------------------------------------------------------

def test_shorter_name_form_links_to_full_name():
    m = Mapper(seed=0)
    full = m.map("PERSON", "Emilia Åström Lind")
    short = m.map("PERSON", "Emilia Lind")
    first = m.map("PERSON", "Emilia")
    assert short is full and first is full
    f = full.fake.split()
    assert m.fake_for_alias(full, "Emilia Lind") == f"{f[0]} {f[-1]}"
    assert m.fake_for_alias(full, "Emilia") == f[0]


def test_url_path_with_person_name_gets_fake_name():
    m = Mapper(seed=0)
    person = m.map("PERSON", "Emilia Åström Lind")
    url = m.map("URL", "https://example.com/portfolio/emilia-lind")
    first, *_, last = (p.lower() for p in person.fake.split())
    fold = str.maketrans("åäöé", "aaoe")
    assert url.fake.endswith(f"/portfolio/{first.translate(fold)}-{last.translate(fold)}")
    assert "emilia" not in url.fake


def test_hyphenated_names_never_get_numeric_suffix():
    for seed in range(20):
        m = Mapper(seed=seed)
        m.reserve(["Karl-Johan Nyström", "Lars-Erik Holm", "Nils-Åke Ek", "Per-Olof Berg", "Jan-Erik Ek"])
        for name in ("Karl-Johan Nyström", "Lars-Erik Holm", "Nils-Åke Ek", "Per-Olof Berg", "Jan-Erik Ek"):
            assert not re.search(r"_\d+$", m.map("PERSON", name).fake)


# --- restore counts -----------------------------------------------------------------------

def test_restore_counts_each_replacement_once():
    entries = [{"id": "p", "type": "PERSON", "original": "Rebecca O'Neill", "fake": "Ester Jonsson",
                "fake_aliases": {"Rebecca": "Ester"}}]
    out, report = restore("Ester Jonsson och Ester Jonsson. Ester ringde.", entries)
    assert out == "Rebecca O'Neill och Rebecca O'Neill. Rebecca ringde."
    assert report.exact[0]["count"] == 3


# --- end to end on the CV ----------------------------------------------------------------

REAL_VALUES = [
    "Emilia", "Åström", "Lind", "Sofia Bergström", "Henrik Östlund", "Amina El-Sayed", "Karl-Johan Nyström",
    "Rebecca O'Neill", "Johan Åkesson", "Fatima Norén", "Maria Sjöström", "Lars-Erik Holm", "Åsa Wennerström",
    "Björn Håkansson", "Nils-Åke Ek", "Chen Wei", "James Whitmore", "Charlotte Hughes", "19890417-2840",
    "070-000 01 21", "+46 70 000 01 21", "+46700000121", "+46 (0)70 000 02 03", "070-0000206", "+44 7700 900121",
    "Exempelbjörksvägen 18 B", "Fiktionsgatan 24", "117 43", "SW1A 1AA", "14 Example Orchard Lane",
    "Fiktiva Kundbolaget AB", "Exempel Operations AB", "emilia-astrom-lind", "emilia-lind",
    "sofia.bergstrom@example.com",
]
NOT_ENTITIES = ["clarify ownership", "of SEK", "regards", "degree", "Sweden", "United Kingdom", "Thesis",
                "Bostadsadress", "Kontaktuppgifter", "###", "Learning Lead", "ECTS"]


@pytest.fixture(scope="module")
def cleaned_cv(tmp_path_factory):
    d = tmp_path_factory.mktemp("cv")
    cleaned, mapping = d / "cv_CLEANED.md", d / "cv_MAPPING.json"
    orchestrate.clean_text(CV, cleaned, mapping, ner="kb", english_ner="auto")
    return d, cleaned, mapping


@needs_models
def test_cv_has_no_leaks_and_keeps_markdown(cleaned_cv):
    _, cleaned, mapping = cleaned_cv
    orig, cl = CV.read_text(encoding="utf-8"), cleaned.read_text(encoding="utf-8")
    leaks = [v for v in REAL_VALUES if re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", cl)]
    assert leaks == []
    for a, b in zip(orig.splitlines(), cl.splitlines()):
        assert a.count("|") == b.count("|") and a.startswith("#") == b.startswith("#"), (a, b)
    originals = {e["original"] for e in json.loads(mapping.read_text())["entries"]}
    assert not originals & set(NOT_ENTITIES), originals & set(NOT_ENTITIES)


@needs_models
def test_cv_shortened_then_restored_matches_original_lines_and_audit_counts(cleaned_cv):
    d, cleaned, mapping = cleaned_cv
    olines = CV.read_text(encoding="utf-8").splitlines(keepends=True)
    lines = cleaned.read_text(encoding="utf-8").splitlines(keepends=True)
    appendix = next(i for i, x in enumerate(olines) if x.startswith("## Bilaga"))
    keep = [i for i in range(len(lines)) if not (8 <= i <= 23 or i >= appendix)]
    short = d / "short.md"
    short.write_text("".join(lines[i] for i in keep), encoding="utf-8")
    restored = d / "short_RESTORED.md"
    report = orchestrate.restore_text_file(short, restored, mapping)
    assert restored.read_text(encoding="utf-8").splitlines(keepends=True) == [olines[i] for i in keep]

    # Audit count = substitutions the clean step made in the kept lines.
    session, _ = orchestrate.preview(CV, ner="kb", english_ner="auto")
    starts = [0]
    for line in olines:
        starts.append(starts[-1] + len(line))
    kept_ranges = [(starts[i], starts[i + 1]) for i in keep]
    expected = Counter(m.entity_id for m in session.mapped_spans
                       if any(a <= m.span.start < b for a, b in kept_ranges))
    assert sum(h["count"] for h in report.exact) == sum(expected.values())


def test_function_words_and_currencies_are_trimmed_or_rejected():
    from pseudo_maker.detect.ner_postprocess import postprocess

    t = "She studied at the University of Edinburgh and later. A portfolio of SEK 12 million."
    i = t.index("the University")
    span = postprocess("COMPANY", t, i, t.index(" later"), confidence=0.9)
    assert span is not None and span.value == "University of Edinburgh"
    j = t.index("of SEK")
    assert postprocess("COMPANY", t, j, j + len("of SEK"), confidence=0.9) is None
    k = "Rekommendation vid Uppsala universitet."
    assert postprocess("COMPANY", k, k.index("Uppsala"), k.index("."), confidence=0.9).value == "Uppsala universitet"
