"""End-to-end clean → restore round-trip tests."""
from __future__ import annotations

import pytest

import json
from pathlib import Path

from docx import Document

from pseudo_maker import orchestrate


def _docx_text(path: Path) -> str:
    doc = Document(str(path))
    parts: list[str] = []
    for p in doc.paragraphs:
        parts.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    parts.append(p.text)
    for section in doc.sections:
        for hf in (section.header, section.footer):
            for p in hf.paragraphs:
                parts.append(p.text)
    return "\n".join(parts)


def test_docx_clean_then_restore_recovers_original(sample_docx, tmp_path):
    cleaned = tmp_path / "cleaned.docx"
    mapping = tmp_path / "mapping.json"
    orchestrate.clean_docx(sample_docx, cleaned, mapping, use_ner=True, seed=42)

    # The cleaned doc should not contain the original sensitive values.
    cleaned_text = _docx_text(cleaned)
    original_values = [e["original"] for e in json.loads(mapping.read_text())["entries"]]
    for orig in original_values:
        assert orig not in cleaned_text, f"Leaked original value: {orig!r}"

    # Restore: pretend the LLM returned the cleaned doc verbatim.
    restored = tmp_path / "restored.docx"
    orchestrate.restore_docx(cleaned, restored, mapping)
    restored_text = _docx_text(restored)
    original_text = _docx_text(sample_docx)
    # Whitespace tolerant compare
    assert _normalize(restored_text) == _normalize(original_text)


def test_docx_format_preserved(sample_docx, tmp_path):
    cleaned = tmp_path / "cleaned.docx"
    mapping = tmp_path / "mapping.json"
    orchestrate.clean_docx(sample_docx, cleaned, mapping, use_ner=True, seed=42)

    orig_doc = Document(str(sample_docx))
    new_doc = Document(str(cleaned))
    assert len(orig_doc.paragraphs) == len(new_doc.paragraphs)
    assert len(orig_doc.tables) == len(new_doc.tables)
    # Bold runs survive (we had `Anna Andersson` as bold)
    new_bold_runs = [r for p in new_doc.paragraphs for r in p.runs if r.bold]
    assert len(new_bold_runs) >= 1


@pytest.mark.usefixtures("needs_ner")
def test_text_clean_then_restore(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text(
        "Anna Andersson har personnummer 800101-1231 och arbetar hos Volvo AB. "
        "Telefon 070-123 45 67. Mail anna@example.se.",
        encoding="utf-8",
    )
    cleaned = tmp_path / "cleaned.txt"
    mapping = tmp_path / "mapping.json"
    orchestrate.clean_text(src, cleaned, mapping, use_ner=True, seed=42)

    cleaned_text = cleaned.read_text(encoding="utf-8")
    for orig in ("Anna Andersson", "800101-1231", "anna@example.se"):
        assert orig not in cleaned_text

    restored = tmp_path / "restored.txt"
    orchestrate.restore_text_file(cleaned, restored, mapping)
    assert restored.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


@pytest.mark.usefixtures("needs_ner")
def test_fuzzy_restore_recovers_lightly_modified_fakes(tmp_path):
    src = tmp_path / "input.txt"
    src.write_text("Anna Andersson har telefonnummer 070-123 45 67.", encoding="utf-8")
    cleaned = tmp_path / "cleaned.txt"
    mapping = tmp_path / "mapping.json"
    orchestrate.clean_text(src, cleaned, mapping, use_ner=True, seed=42)

    # Mutate one character of the fake person name to simulate LLM rewrite
    cleaned_text = cleaned.read_text(encoding="utf-8")
    entries = json.loads(mapping.read_text())["entries"]
    person = next(e for e in entries if e["type"] == "PERSON")
    fake = person["fake"]
    if len(fake) > 4:
        # Swap one letter
        mutated = fake[:-1] + ("a" if fake[-1] != "a" else "e")
        cleaned_text = cleaned_text.replace(fake, mutated)
        cleaned.write_text(cleaned_text, encoding="utf-8")

        restored = tmp_path / "restored.txt"
        report = orchestrate.restore_text_file(cleaned, restored, mapping, fuzzy_threshold=85)
        out = restored.read_text(encoding="utf-8")
        # The original person name should be back even though the fake was mutated
        assert "Anna Andersson" in out or any(h["original"] == "Anna Andersson" for h in report.fuzzy)


def _normalize(s: str) -> str:
    return " ".join(s.split())

pytestmark = pytest.mark.usefixtures("ner_backend")
