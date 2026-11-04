"""High-level clean/restore orchestration over docx and text formats.

The clean flow is split into two phases so the UI can present a preview:

  1. preview(...)  → runs detection + mapping but does NOT write anything.
                     Returns a CleanSession (holding in-memory doc + spans +
                     mapper) and a list of PreviewEntity for the UI.
  2. finalize(...) → applies the session's replacements, skipping any entity
                     IDs the user excluded. Writes cleaned doc + mapping.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .detect.pipeline import DetectionPipeline, propagate_person_names
from .detect.recognizers.base import Span
from .io import docx_io, text_io
from .mapping.store import read_mapping, write_mapping
from .restore.restore import RestoreReport, restore as restore_text
from .substitute.mapper import Mapper


@dataclass
class MappedSpan:
    span: Span
    entity_id: str
    fake: str


@dataclass
class PreviewEntity:
    id: str
    type: str
    original: str
    fake: str
    occurrences: int
    aliases: list[str]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "original": self.original,
            "fake": self.fake,
            "occurrences": self.occurrences,
            "aliases": self.aliases,
        }


@dataclass
class CleanSession:
    input_path: Path
    kind: str  # "docx" | "text"
    mapper: Mapper
    doc: Any = None                                  # docx Document (when kind=="docx")
    handles: list[docx_io.ParagraphHandle] = field(default_factory=list)
    mapped_per_handle: list[list[MappedSpan]] = field(default_factory=list)
    text: str = ""                                   # full text (when kind=="text")
    mapped_spans: list[MappedSpan] = field(default_factory=list)


def _is_docx(path: Path) -> bool:
    return path.suffix.lower() == ".docx"


def is_docx_path(path: Path) -> bool:
    return _is_docx(path)


def _collect_preview_entities(mapper: Mapper) -> list[PreviewEntity]:
    return [
        PreviewEntity(
            id=e.id,
            type=e.type,
            original=e.original,
            fake=e.fake,
            occurrences=e.occurrences,
            aliases=sorted(e.aliases),
        )
        for e in mapper.all_entries()
    ]


def make_pipeline(*, use_ner: bool = True, ner: str | None = None, english_ner: str | None = None) -> DetectionPipeline:
    """Build the detection pipeline. Modes default to $PSEUDO_MAKER_NER /
    $PSEUDO_MAKER_ENGLISH_NER, else "auto" (use whatever models are installed)."""
    import os

    return DetectionPipeline(
        use_ner=use_ner,
        ner=ner or os.environ.get("PSEUDO_MAKER_NER", "auto"),
        english_ner=english_ner or os.environ.get("PSEUDO_MAKER_ENGLISH_NER", "auto"),
    )


def _map_spans(mapper: Mapper, spans: list[Span]) -> list[MappedSpan]:
    """Map spans to entries, full names first so short forms coalesce.

    Without this, "Anna" or "Annas" appearing *before* "Anna-Karin Lindqvist"
    in the document would get a separate, unrelated fake. Persons go before
    emails so email addresses can be rebuilt from the person's fake name.
    """
    type_rank = {"PERSON": 0, "COMPANY": 1}

    def order(i: int) -> tuple:
        s = spans[i]
        return (type_rank.get(s.type, 2), -len(s.value.split()), -len(s.value), i)

    mapper.reserve(s.value for s in spans)
    entries: dict[int, object] = {}
    for i in sorted(range(len(spans)), key=order):
        entries[i] = mapper.map(spans[i].type, spans[i].value)
    return [
        MappedSpan(span=s, entity_id=entries[i].id, fake=mapper.fake_for_alias(entries[i], s.value))
        for i, s in enumerate(spans)
    ]


def preview_docx(
    input_path: Path,
    *,
    use_ner: bool = True,
    seed: int | None = 42,
    ner: str | None = None,
    english_ner: str | None = None,
) -> tuple[CleanSession, list[PreviewEntity]]:
    pipeline = make_pipeline(use_ner=use_ner, ner=ner, english_ner=english_ner)
    mapper = Mapper(seed=seed)
    doc, handles = docx_io.load(input_path)

    spans_per_handle = [
        pipeline.detect(h.full_text) if h.full_text.strip() else [] for h in handles
    ]
    spans_per_handle = propagate_person_names([h.full_text for h in handles], spans_per_handle)
    flat = [s for spans in spans_per_handle for s in spans]
    flat_mapped = _map_spans(mapper, flat)
    mapped_per_handle: list[list[MappedSpan]] = []
    pos = 0
    for spans in spans_per_handle:
        mapped_per_handle.append(flat_mapped[pos:pos + len(spans)])
        pos += len(spans)

    session = CleanSession(
        input_path=input_path,
        kind="docx",
        mapper=mapper,
        doc=doc,
        handles=handles,
        mapped_per_handle=mapped_per_handle,
    )
    return session, _collect_preview_entities(mapper)


def preview_text(
    input_path: Path,
    *,
    use_ner: bool = True,
    seed: int | None = 42,
    ner: str | None = None,
    english_ner: str | None = None,
) -> tuple[CleanSession, list[PreviewEntity]]:
    pipeline = make_pipeline(use_ner=use_ner, ner=ner, english_ner=english_ner)
    mapper = Mapper(seed=seed)
    text = text_io.read_text(input_path)

    spans = propagate_person_names([text], [pipeline.detect(text)])[0]
    mapped_spans = _map_spans(mapper, spans)

    session = CleanSession(
        input_path=input_path,
        kind="text",
        mapper=mapper,
        text=text,
        mapped_spans=mapped_spans,
    )
    return session, _collect_preview_entities(mapper)


def finalize(
    session: CleanSession,
    output_path: Path,
    mapping_path: Path,
    *,
    excluded_entity_ids: set[str] | None = None,
) -> None:
    excluded = excluded_entity_ids or set()

    def resolve(m: MappedSpan) -> str:
        """Re-derive the fake from the (possibly overridden) mapper."""
        entry = session.mapper.get_entry(m.entity_id)
        return session.mapper.fake_for_alias(entry, m.span.value)

    if session.kind == "docx":
        for handle, mapped_list in zip(session.handles, session.mapped_per_handle):
            repls = [
                (m.span.start, m.span.end, resolve(m))
                for m in mapped_list
                if m.entity_id not in excluded
            ]
            if repls:
                docx_io.apply_paragraph_replacements(handle, repls)
        docx_io.save(session.doc, output_path)
    else:
        repls = [
            (m.span.start, m.span.end, resolve(m))
            for m in session.mapped_spans
            if m.entity_id not in excluded
        ]
        new_text = text_io.apply_replacements(session.text, repls)
        text_io.write_text(output_path, new_text)

    kept = [e for e in session.mapper.all_entries() if e.id not in excluded]
    write_mapping(mapping_path, kept, source_file=session.input_path.name)


def preview(input_path: Path, **kwargs) -> tuple[CleanSession, list[PreviewEntity]]:
    if _is_docx(input_path):
        return preview_docx(input_path, **kwargs)
    return preview_text(input_path, **kwargs)


# Single-shot clean helpers (used by the CLI when no preview step is desired).

def clean_docx(
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    *,
    use_ner: bool = True,
    seed: int | None = 42,
    ner: str | None = None,
    english_ner: str | None = None,
) -> Mapper:
    session, _ = preview_docx(input_path, use_ner=use_ner, seed=seed, ner=ner, english_ner=english_ner)
    finalize(session, output_path, mapping_path)
    return session.mapper


def clean_text(
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    *,
    use_ner: bool = True,
    seed: int | None = 42,
    ner: str | None = None,
    english_ner: str | None = None,
) -> Mapper:
    session, _ = preview_text(input_path, use_ner=use_ner, seed=seed, ner=ner, english_ner=english_ner)
    finalize(session, output_path, mapping_path)
    return session.mapper


# Restore flow (unchanged).

def restore_docx(
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    *,
    fuzzy_threshold: int = 92,
) -> RestoreReport:
    entries = read_mapping(mapping_path)
    doc, handles = docx_io.load(input_path)
    full_report = RestoreReport()
    seen_entry_ids_exact: set[str] = set()
    seen_entry_ids_fuzzy: set[str] = set()
    for handle in handles:
        if not handle.full_text.strip():
            continue
        new_text, report = restore_text(handle.full_text, entries, fuzzy_threshold=fuzzy_threshold)
        if new_text != handle.full_text:
            # Replace only the changed pieces so run formatting (bold, italic, …)
            # around and inside each name survives.
            docx_io.apply_paragraph_replacements(handle, _diff_replacements(handle.full_text, new_text))
        for hit in report.exact:
            seen_entry_ids_exact.add(hit["entry_id"])
            full_report.exact.append(hit)
        for hit in report.fuzzy:
            seen_entry_ids_fuzzy.add(hit["entry_id"])
            full_report.fuzzy.append(hit)
    docx_io.save(doc, output_path)
    matched_ids = seen_entry_ids_exact | seen_entry_ids_fuzzy
    for entry in entries:
        if entry["id"] not in matched_ids:
            full_report.missing.append({
                "entry_id": entry["id"],
                "fake": entry["fake"],
                "original": entry["original"],
                "type": entry["type"],
            })
    return full_report


def _diff_replacements(old: str, new: str) -> list[tuple[int, int, str]]:
    """Minimal (start, end, replacement) edits turning `old` into `new`."""
    import difflib

    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    return [
        (i1, i2, new[j1:j2])
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    ]


AUDIT_SCHEMA_VERSION = 1


def write_audit_log(
    report: RestoreReport,
    audit_path: Path,
    *,
    restored_document: str,
    mapping_file: str,
) -> None:
    """Persist a structured log of the restore pass alongside the restored doc."""
    total = sum(int(hit.get("count", 1)) for hit in report.exact) + len(report.fuzzy)
    payload = {
        "version": AUDIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "restored_document": restored_document,
        "mapping_used": mapping_file,
        "summary": {
            "exact_count": len(report.exact),
            "fuzzy_count": len(report.fuzzy),
            "missing_count": len(report.missing),
            "total_replacements": total,
        },
        "exact": list(report.exact),
        "fuzzy": list(report.fuzzy),
        "missing": list(report.missing),
    }
    Path(audit_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def restore_text_file(
    input_path: Path,
    output_path: Path,
    mapping_path: Path,
    *,
    fuzzy_threshold: int = 92,
) -> RestoreReport:
    entries = read_mapping(mapping_path)
    text = text_io.read_text(input_path)
    new_text, report = restore_text(text, entries, fuzzy_threshold=fuzzy_threshold)
    text_io.write_text(output_path, new_text)
    return report
