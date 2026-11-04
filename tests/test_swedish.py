"""Swedish-specific behaviour: grammar, name shape, number formats, restore."""
from __future__ import annotations

import pytest

import json
import random
import re
from pathlib import Path

from docx import Document
from stdnum import iban
from stdnum.eu import vat

from pseudo_maker import orchestrate
from pseudo_maker.detect.recognizers.case_number_se import CaseNumberSeRecognizer
from pseudo_maker.detect.recognizers.company_street_se import (
    CompanySeRecognizer,
    KnownCompanySeRecognizer,
    StreetSeRecognizer,
)
from pseudo_maker.detect.recognizers.date_age import AgeRecognizer, DateRecognizer
from pseudo_maker.detect.recognizers.license_plate_se import LicensePlateSeRecognizer
from pseudo_maker.detect.recognizers.orgnr import is_valid_orgnr
from pseudo_maker.detect.recognizers.personnummer import is_valid_personnummer
from pseudo_maker.detect.recognizers.base import luhn_check
from pseudo_maker.substitute import fakes
from pseudo_maker.substitute.mapper import Mapper
from pseudo_maker.substitute.valid_format import (
    fake_iban_se,
    fake_orgnr,
    fake_personnummer,
    fake_plusgiro,
)
from pseudo_maker.swedish import genitive, name_gender


def _values(recognizer, text):
    return [s.value for s in recognizer.find(text)]


# --- grammar ---------------------------------------------------------------

def test_genitive_rules():
    assert genitive("Anna") == "Annas"
    assert genitive("Lars") == "Lars"
    assert genitive("Max") == "Max"
    assert genitive("ANNA") == "ANNAS"


@pytest.mark.usefixtures("needs_ner")
def test_genitive_coalesces_with_base_name_in_any_order(tmp_path: Path):
    # Genitive appears *before* the full name in the document.
    src = tmp_path / "a.txt"
    src.write_text("Annas chef ringde. Senare kom Anna Berg.", encoding="utf-8")
    out, mapping = tmp_path / "c.txt", tmp_path / "m.json"
    orchestrate.clean_text(src, out, mapping)
    persons = [e for e in json.loads(mapping.read_text())["entries"] if e["type"] == "PERSON"]
    assert len(persons) == 1
    fake_first = persons[0]["fake"].split()[0]
    assert out.read_text(encoding="utf-8").startswith(genitive(fake_first) + " chef")

    m2 = Mapper(seed=1)
    full2 = m2.map("PERSON", "Per Åkesson")
    gen2 = m2.map("PERSON", "Per Åkessons")
    assert gen2 is full2
    fake_gen = m2.fake_for_alias(full2, "Per Åkessons")
    assert fake_gen == genitive(full2.fake)


def test_hyphenated_first_name_part_is_alias():
    m = Mapper(seed=3)
    full = m.map("PERSON", "Anna-Karin Lindqvist")
    short = m.map("PERSON", "Anna")
    surname = m.map("PERSON", "Lindqvist")
    assert short is full and surname is full
    # surname alias maps to the fake *surname*, not the fake first name
    assert m.fake_for_alias(full, "Lindqvist") == full.fake.split()[-1]


# --- fake names ------------------------------------------------------------

def test_fake_person_keeps_gender_shape_and_particles():
    rng = random.Random(0)
    for original in ("Anna Andersson", "Karin Berg", "Maria Lind"):
        fake = fakes.fake_person(original, rng)
        assert name_gender(fake.split()[0]) in ("F", None), fake
        assert len(fake.split()) == 2
    for original in ("Per Åkesson", "Lars Nilsson"):
        fake = fakes.fake_person(original, rng)
        assert name_gender(fake.split()[0]) in ("M", None), fake
    fake = fakes.fake_person("Anna-Karin Lindqvist", rng)
    assert "-" in fake.split()[0]
    fake = fakes.fake_person("Erik von Platen", rng)
    assert fake.split()[1] == "von"
    assert fakes.fake_person("ANNA BERG", rng).isupper()


def test_fake_person_never_reuses_real_name_parts():
    rng = random.Random(0)
    for _ in range(200):
        fake = fakes.fake_person("Anna-Karin Lindqvist", rng)
        parts = {p.lower() for p in re.split(r"[\s\-]+", fake)}
        assert not parts & {"anna", "karin", "lindqvist"}, fake


def test_fake_person_keeps_genitive_behaviour():
    rng = random.Random(0)
    for _ in range(50):
        assert not fakes.fake_person("Mohammed", rng).endswith("s")
        assert fakes.fake_person("Anna Fors", rng)[-1] in "sxz"


# --- companies / streets / emails -------------------------------------------

def test_company_suffix_and_publ():
    rng = random.Random(0)
    assert fakes.fake_company("Volvo AB (publ)", rng).endswith(" AB (publ)")
    assert not re.search(r"\b(AB|HB|KB)$", fakes.fake_company("Skanska", rng))
    m = Mapper(seed=0)
    a = m.map("COMPANY", "Nordström Bygg AB (publ)")
    b = m.map("COMPANY", "Nordström Bygg AB")
    assert a is b
    assert m.fake_for_alias(a, "Nordström Bygg AB") == a.fake.replace(" (publ)", "")


def test_company_pattern_and_known_companies():
    text = "Mellan Nordström Bygg AB (publ) och\nLind & Son HB. Mötet med Volvo Personvagnar och ICA Gruppen."
    assert _values(CompanySeRecognizer(), text) == ["Nordström Bygg AB (publ)", "Lind & Son HB"]
    assert set(_values(KnownCompanySeRecognizer(), text)) == {"Volvo Personvagnar", "ICA Gruppen"}


def test_street_with_house_number():
    text = "Bor på Storgatan 12B, 211 34 Malmö, kontor på Drottninggatan 5 i Stockholm och Sankt Eriksgatan 5, 3 tr."
    assert _values(StreetSeRecognizer(), text) == [
        "Storgatan 12B", "Drottninggatan 5", "Sankt Eriksgatan 5, 3 tr.",
    ]
    rng = random.Random(0)
    fake = fakes.fake_address("Storgatan 12B", rng)
    assert re.fullmatch(r"\S+ \d+[A-D]", fake), fake
    assert not any(c.isdigit() for c in fakes.fake_address("Storgatan", rng))


def test_email_follows_fake_person_and_keeps_webmail_domain():
    m = Mapper(seed=0)
    person = m.map("PERSON", "Anna-Karin Lindqvist")
    email = m.map("EMAIL", "anna-karin.lindqvist@gmail.com")
    first, last = person.fake.lower().split()
    assert email.fake.endswith("@gmail.com")
    assert email.fake.startswith(first.translate(str.maketrans("åäöé", "aaoe")))
    work = m.map("EMAIL", "anna@nordstrombygg.se")
    assert not work.fake.endswith("nordstrombygg.se")


# --- numbers -----------------------------------------------------------------

def test_phone_keeps_format_and_prefix():
    rng = random.Random(0)
    for original in ("070-123 45 67", "08-555 123 45", "+46 8 508 123 45", "+46 70 123 45 67"):
        fake = fakes.fake_phone(original, rng)
        assert re.sub(r"\d", "#", fake) == re.sub(r"\d", "#", original)
        assert fake[:4] == original[:4] or fake[:3] == original[:3]
        assert fake != original


def test_personnummer_keeps_gender_age_and_kind():
    rng = random.Random(0)
    for _ in range(100):
        f = fake_personnummer("19850312-2387", rng)       # woman (8 even)
        assert is_valid_personnummer(f) and int(f[-2]) % 2 == 0
        assert 1982 <= int(f[:4]) <= 1988
        f = fake_personnummer("840722-6375", rng)         # man (7 odd)
        assert is_valid_personnummer(f) and int(f[-2]) % 2 == 1
        f = fake_personnummer("870374-1234", rng)         # samordningsnummer
        assert is_valid_personnummer(f) and int(f[4:6]) > 60
        f = fake_personnummer("250101+2344", rng)         # 100+ years old
        assert "+" in f and is_valid_personnummer(f)


def test_orgnr_keeps_legal_form_digit():
    rng = random.Random(0)
    for _ in range(50):
        f = fake_orgnr("556677-8899", rng)
        assert f[0] == "5" and is_valid_orgnr(f)
        assert fake_orgnr("969999-1234", rng)[0] == "9"


def test_iban_vat_plusgiro_fakes_are_valid():
    rng = random.Random(0)
    for _ in range(50):
        f = fake_iban_se(rng, "SE45 5000 0000 0583 9825 7466")
        assert iban.is_valid(f) and f.count(" ") == 5
        assert vat.is_valid(fakes.fake_vat("SE556036079301", rng))
        pg = fake_plusgiro("45 67 89-7", rng)
        assert re.fullmatch(r"\d\d \d\d \d\d-\d", pg) and luhn_check(re.sub(r"\D", "", pg))


def test_case_numbers():
    text = "Dnr KS 2024/123. Diarienummer: 3.1.2-4567/2023. Mål nr T 1234-23. Vi vann målet 2-1."
    assert _values(CaseNumberSeRecognizer(), text) == ["KS 2024/123", "3.1.2-4567/2023", "T 1234-23"]
    fake = fakes.fake_case_number("T 1234-23", random.Random(0))
    assert fake.startswith("T ") and fake.endswith("-23") and fake != "T 1234-23"


# --- dates / ages / plates -------------------------------------------------------

def test_swedish_date_formats_preserved():
    rng = random.Random(0)
    text = "den 1:a mars 2025, 15 januari 2025, 3 feb. 2024"
    assert _values(DateRecognizer(), text) == ["1:a mars 2025", "15 januari 2025", "3 feb. 2024"]
    assert re.fullmatch(r"\d{1,2} [a-zåäö]+ 2025|\d{1,2} [a-zåäö]+ 202[45]", fakes.fake_date("15 januari 2025", rng))
    assert re.fullmatch(r"\d{1,2} [a-z]{3}\. 202[345]", fakes.fake_date("3 feb. 2024", rng))
    assert re.fullmatch(r"\d{1,2}:[ae] [a-zåäö]+ 202[45]", fakes.fake_date("1:a mars 2025", rng))


def test_age_needs_age_cue_not_duration():
    text = "Har arbetat 3 år. Hon är 45 år gammal, en 39-årig kvinna, ålder: 52, Per (61 år)."
    assert _values(AgeRecognizer(), text) == ["45 år gammal", "39-årig", "ålder: 52", "(61 år)"]


def test_currency_is_not_a_license_plate():
    assert _values(LicensePlateSeRecognizer(), "Avgift SEK 500 och EUR 250, bil ABC 123.") == ["ABC 123"]


# --- end to end ---------------------------------------------------------------------

@pytest.mark.usefixtures("needs_ner")
def test_caps_names_and_propagated_surnames(tmp_path: Path):
    src = tmp_path / "a.txt"
    src.write_text(
        "Klient: MOHAMMED AL-HASSAN.\nHandläggare Johanna Ek ringde. Ek noterade att Fatima Al-Hassan var med.",
        encoding="utf-8",
    )
    out, mapping = tmp_path / "c.txt", tmp_path / "m.json"
    orchestrate.clean_text(src, out, mapping)
    cleaned = out.read_text(encoding="utf-8")
    for leak in ("MOHAMMED", "AL-HASSAN", "Al-Hassan", " Ek ", "Ek noterade", "Johanna"):
        assert leak not in cleaned, (leak, cleaned)
    restored = tmp_path / "r.txt"
    orchestrate.restore_text_file(out, restored, mapping)
    assert restored.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


@pytest.mark.usefixtures("needs_ner")
def test_restore_llm_reply_with_new_genitives_and_bare_names(tmp_path: Path):
    src = tmp_path / "a.txt"
    src.write_text("Anna-Karin Lindqvist anställs av Nordström Bygg AB. Chef är Per Åkesson.", encoding="utf-8")
    out, mapping = tmp_path / "c.txt", tmp_path / "m.json"
    orchestrate.clean_text(src, out, mapping)
    e = {x["original"]: x["fake"] for x in json.loads(mapping.read_text())["entries"]}
    p, boss, co = e["Anna-Karin Lindqvist"], e["Per Åkesson"], e["Nordström Bygg AB"]
    reply = tmp_path / "reply.txt"
    reply.write_text(
        f"{genitive(p.split()[-1])} lön. {p.split()[0]} rapporterar till {genitive(boss)}. {genitive(co)} vd.",
        encoding="utf-8",
    )
    restored = tmp_path / "r.txt"
    orchestrate.restore_text_file(reply, restored, mapping)
    assert restored.read_text(encoding="utf-8") == (
        "Lindqvists lön. Anna-Karin rapporterar till Per Åkessons. Nordström Bygg ABs vd."
    )


def test_docx_restore_keeps_run_formatting(tmp_path: Path):
    doc = Document()
    p = doc.add_paragraph()
    p.add_run("Chef är ")
    p.add_run("Per Åkesson").bold = True
    p.add_run(", tel 070-123 45 67")
    src = tmp_path / "in.docx"
    doc.save(str(src))
    cleaned, mapping, restored = tmp_path / "c.docx", tmp_path / "m.json", tmp_path / "r.docx"
    orchestrate.clean_docx(src, cleaned, mapping)
    orchestrate.restore_docx(cleaned, restored, mapping)
    para = Document(str(restored)).paragraphs[0]
    assert para.text == "Chef är Per Åkesson, tel 070-123 45 67"
    assert [r.text for r in para.runs if r.bold] == ["Per Åkesson"]


def test_family_members_share_fake_surname():
    m = Mapper(seed=0)
    son = m.map("PERSON", "Mohammed Al-Hassan")
    mother = m.map("PERSON", "Fatima Al-Hassan")
    assert son is not mother
    assert son.fake.split()[-1] == mother.fake.split()[-1]
    assert son.fake != mother.fake

pytestmark = pytest.mark.usefixtures("ner_backend")


def test_unrelated_people_get_different_fake_surnames():
    m = Mapper(seed=0)
    fakes_ = [m.map("PERSON", n).fake for n in (
        "Anna-Karin Lindqvist", "Per Åkesson", "Erik Svensson", "Maria Holm", "Johan Berg",
        "Sara Nyberg", "Lars Nilsson", "Karin Wallin", "Olle Ek", "Ebba Strand",
    )]
    surnames = [f.split()[-1] for f in fakes_]
    assert len(set(surnames)) == len(surnames), fakes_
