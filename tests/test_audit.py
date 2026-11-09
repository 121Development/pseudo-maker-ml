"""Tests for the restore audit log."""
from __future__ import annotations

import json
from pathlib import Path

from pseudo_maker import orchestrate
from pseudo_maker.restore.restore import RestoreReport


def test_write_audit_log_matches_schema(tmp_path: Path):
    report = RestoreReport(
        exact=[
            {"entry_id": "ent_0001", "type": "PERSON", "fake": "Karin Berg",
             "original": "Anna Andersson", "count": 5},
            {"entry_id": "ent_0002", "type": "PERSONNUMMER", "fake": "770319-5678",
             "original": "800101-1231", "count": 1},
        ],
        fuzzy=[
            {"entry_id": "ent_0003", "type": "PERSON", "fake": "Bo Andersson",
             "matched_text": "Bo Anderssoa", "original": "Bo Andersson", "score": 91},
        ],
        missing=[
            {"entry_id": "ent_0004", "type": "EMAIL", "fake": "test@example.com",
             "original": "real@client.se"},
        ],
    )
    audit_path = tmp_path / "doc_AUDIT.json"
    orchestrate.write_audit_log(
        report,
        audit_path,
        restored_document="doc_RESTORED.docx",
        mapping_file="doc_MAPPING.json",
    )
    payload = json.loads(audit_path.read_text(encoding="utf-8"))

    assert payload["version"] == 2
    assert payload["restored_document"] == "doc_RESTORED.docx"
    assert payload["mapping_used"] == "doc_MAPPING.json"
    assert payload["summary"]["exact_count"] == 2
    assert payload["summary"]["fuzzy_count"] == 1
    assert payload["summary"]["not_found_in_input_count"] == 1
    # total = 5 + 1 exact occurrences + 1 fuzzy = 7
    assert payload["summary"]["total_replacements"] == 7
    assert payload["exact"][0]["type"] == "PERSON"
    assert payload["fuzzy"][0]["type"] == "PERSON"
    assert payload["not_found_in_input"][0]["type"] == "EMAIL"


def test_cli_restore_writes_audit_log(tmp_path: Path):
    """End-to-end: clean → restore via orchestrate writes an audit file."""
    src = tmp_path / "input.txt"
    src.write_text(
        "Anna Andersson har personnummer 800101-1231. Mail anna@example.se.",
        encoding="utf-8",
    )
    cleaned = tmp_path / "input_CLEANED.txt"
    mapping = tmp_path / "input_MAPPING.json"
    orchestrate.clean_text(src, cleaned, mapping, use_ner=True, seed=42)

    restored = tmp_path / "input_RESTORED.txt"
    report = orchestrate.restore_text_file(cleaned, restored, mapping)

    audit = tmp_path / "input_AUDIT.json"
    orchestrate.write_audit_log(
        report,
        audit,
        restored_document=restored.name,
        mapping_file=mapping.name,
    )
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["summary"]["exact_count"] >= 2  # at least person + personnummer + email
    # Every exact entry should carry a type (the change we added upstream).
    for hit in payload["exact"]:
        assert hit["type"], f"missing type in {hit}"
