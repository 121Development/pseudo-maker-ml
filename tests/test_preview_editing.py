"""Preview-time editing used by the web UI: ambiguous candidates, custom terms,
live fake edits and the document view."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from pseudo_maker import orchestrate

pytestmark = pytest.mark.usefixtures("ner_backend")

TEXT = (
    "Projekt Blåval leds av Anna Berg. Rapporten skickades till Skatteverket.\n"
    "Annas team träffas i Malmö. PROJEKT BLÅVAL är hemligt."
)


def _session(tmp_path: Path, text: str = TEXT):
    src = tmp_path / "in.txt"
    src.write_text(text, encoding="utf-8")
    session, _ = orchestrate.preview(src)
    return session


def _finalize(session, tmp_path: Path, excluded=()):
    out, mapping = tmp_path / "out.txt", tmp_path / "m.json"
    orchestrate.finalize(session, out, mapping, excluded_entity_ids=set(excluded))
    return out.read_text(encoding="utf-8"), json.loads(mapping.read_text())["entries"]


def test_add_term_replaces_every_occurrence_case_insensitively(tmp_path):
    session = _session(tmp_path)
    assert orchestrate.add_term(session, "Blåval", "CUSTOM") == 2
    text, entries = _finalize(session, tmp_path)
    assert "Blåval" not in text and "BLÅVAL" not in text
    assert any(e["type"] == "CUSTOM" for e in entries)
    assert orchestrate.add_term(session, "Blåval", "CUSTOM") == 0  # already replaced


def test_add_term_covering_an_existing_span_replaces_it(tmp_path):
    session = _session(tmp_path, "Vi anlitade Volvo Personvagnar Torslanda igen.")
    view = orchestrate.document_view(session)[0]
    assert "Volvo Personvagnar" in [view["text"][s["start"]:s["end"]] for s in view["spans"]]
    added = orchestrate.add_term(session, "Volvo Personvagnar Torslanda", "COMPANY")
    assert added == 1
    view = orchestrate.document_view(session)[0]
    covered = [view["text"][s["start"]:s["end"]] for s in view["spans"]]
    assert covered == ["Volvo Personvagnar Torslanda"]


def test_add_term_round_trips(tmp_path):
    session = _session(tmp_path)
    orchestrate.add_term(session, "Blåval", "CUSTOM")
    out, mapping = tmp_path / "out.txt", tmp_path / "m.json"
    orchestrate.finalize(session, out, mapping)
    restored = tmp_path / "r.txt"
    orchestrate.restore_text_file(out, restored, mapping)
    assert restored.read_text(encoding="utf-8") == TEXT


def test_agency_is_a_candidate_and_can_be_added(tmp_path, ner_backend):
    if ner_backend == "off":
        pytest.skip("candidates come from the NER models")
    session = _session(tmp_path)
    values = {c["value"] for c in orchestrate.candidates(session)}
    assert "Skatteverket" in values
    orchestrate.add_term(session, "Skatteverket", "COMPANY")
    assert "Skatteverket" not in {c["value"] for c in orchestrate.candidates(session)}
    text, _ = _finalize(session, tmp_path)
    assert "Skatteverket" not in text


def test_set_fake_updates_aliases_in_document_view(tmp_path, ner_backend):
    if ner_backend == "off":
        pytest.skip("needs an NER model to find the person")
    session = _session(tmp_path)
    person = next(e for e in orchestrate.entities(session) if e.original == "Anna Berg")
    orchestrate.set_fake(session, person.id, "Zelda Holm")
    view = orchestrate.document_view(session)[0]
    fakes = {view["text"][s["start"]:s["end"]]: s["fake"] for s in view["spans"]}
    assert fakes["Anna Berg"] == "Zelda Holm"
    assert fakes["Annas"] == "Zeldas"
    text, _ = _finalize(session, tmp_path)
    assert "Zeldas team" in text


def test_server_preview_endpoints(tmp_path):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from pseudo_maker.server.app import create_app

    client = TestClient(create_app())
    up = client.post("/api/clean/upload", files={"file": ("a.txt", io.BytesIO(TEXT.encode()))}).json()
    assert {"entities", "candidates", "document"} <= set(up)
    sid = up["session_id"]
    r = client.post("/api/clean/add-term", data={"session_id": sid, "term": "Blåval", "type": "CUSTOM"})
    assert r.status_code == 200 and r.json()["added"] == 2
    bad = client.post("/api/clean/add-term", data={"session_id": sid, "term": "x", "type": "NOPE"})
    assert bad.status_code == 400
    eid = r.json()["entities"][0]["id"]
    r = client.post("/api/clean/set-fake", data={"session_id": sid, "entity_id": eid, "fake": "Testvärde"})
    assert r.status_code == 200
    done = client.post("/api/clean/finalize", data={"session_id": sid}).json()
    cleaned = client.get(done["cleaned_url"]).text
    assert "Blåval" not in cleaned and "Testvärde" in cleaned
