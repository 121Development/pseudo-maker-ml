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

import re

from .detect.pipeline import DetectionPipeline, filter_candidates, propagate_person_names
from .detect.recognizers.base import Candidate, Span
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
    confidence: float = 1.0          # lowest confidence among its spans
    sources: list[str] = field(default_factory=list)  # regex / pattern / ner / manual

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "original": self.original,
            "fake": self.fake,
            "occurrences": self.occurrences,
            "aliases": self.aliases,
            "confidence": round(self.confidence, 3),
            "sources": self.sources,
            "uncertain": self.confidence < UNCERTAIN_BELOW,
        }


# Entities whose weakest span scores below this are flagged "uncertain" in the UI.
UNCERTAIN_BELOW = 0.7


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
    candidates: list[list[Candidate]] = field(default_factory=list)  # per text, not replaced

    def texts(self) -> list[str]:
        return [h.full_text for h in self.handles] if self.kind == "docx" else [self.text]

    def mapped_lists(self) -> list[list[MappedSpan]]:
        """Mutable per-text span lists (the text kind has a single one)."""
        return self.mapped_per_handle if self.kind == "docx" else [self.mapped_spans]


def _is_docx(path: Path) -> bool:
    return path.suffix.lower() == ".docx"


def is_docx_path(path: Path) -> bool:
    return _is_docx(path)


def _collect_preview_entities(session: "CleanSession") -> list[PreviewEntity]:
    """Entities with at least one span, with occurrence counts from the spans."""
    counts: dict[str, int] = {}
    conf: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    for mapped in session.mapped_lists():
        for m in mapped:
            counts[m.entity_id] = counts.get(m.entity_id, 0) + 1
            conf[m.entity_id] = min(conf.get(m.entity_id, 1.0), m.span.confidence)
            sources.setdefault(m.entity_id, set()).add(m.span.source)
    return [
        PreviewEntity(
            id=e.id,
            type=e.type,
            original=e.original,
            fake=e.fake,
            occurrences=counts[e.id],
            aliases=sorted(e.aliases),
            confidence=conf[e.id],
            sources=sorted(sources[e.id]),
        )
        for e in session.mapper.all_entries()
        if counts.get(e.id)
    ]


def entities(session: "CleanSession") -> list[PreviewEntity]:
    return _collect_preview_entities(session)


def candidates(session: "CleanSession") -> list[dict]:
    """Ambiguous findings not replaced by default, grouped by (type, value)."""
    groups: dict[tuple[str, str], dict] = {}
    for cands in session.candidates:
        for c in cands:
            key = (c.type, c.value.lower())
            g = groups.get(key)
            if g is None:
                groups[key] = {"type": c.type, "value": c.value, "score": c.score,
                               "reason": c.reason, "occurrences": 1}
            else:
                g["occurrences"] += 1
                g["score"] = max(g["score"], c.score)
    out = sorted(groups.values(), key=lambda g: (-g["score"], g["value"].lower()))
    for i, g in enumerate(out, 1):
        g["id"] = f"cand_{i:03d}"
    return out


def document_view(session: "CleanSession") -> list[dict]:
    """The document as paragraphs with substitution and candidate offsets."""
    mapper = session.mapper
    out = []
    for text, mapped, cands in zip(session.texts(), session.mapped_lists(), session.candidates):
        spans = []
        for m in sorted(mapped, key=lambda m: m.span.start):
            entry = mapper.get_entry(m.entity_id)
            spans.append({
                "start": m.span.start, "end": m.span.end, "entity_id": m.entity_id,
                "fake": mapper.fake_for_alias(entry, m.span.value),
            })
        out.append({
            "text": text,
            "spans": spans,
            "candidates": [{"start": c.start, "end": c.end, "type": c.type, "value": c.value}
                           for c in sorted(cands, key=lambda c: c.start)],
        })
    return out


def set_fake(session: "CleanSession", entity_id: str, new_fake: str) -> None:
    session.mapper.override_fake(entity_id, new_fake)


def add_term(session: "CleanSession", term: str, entity_type: str) -> int:
    """Replace every occurrence of a user-supplied term. Returns occurrences added.

    Matching is whole-word and case-insensitive; for word-like types a
    trailing genitive -s is included ("Annas"). An occurrence inside an
    existing substitution is skipped; one that fully covers existing
    substitutions replaces them ("Volvo" → user adds "Volvo Personvagnar AB").
    """
    term = " ".join(term.split())
    if not term:
        raise ValueError("term must not be empty")
    entity_type = entity_type.upper()
    genitive = r"s?" if entity_type in ("PERSON", "COMPANY", "CITY", "STREET") and term[-1].isalpha() else ""
    pattern = re.compile(
        r"(?<![\w])" + r"\s+".join(re.escape(w) for w in term.split()) + genitive + r"(?![\w])",
        re.IGNORECASE,
    )
    found: list[tuple[int, int, int, str]] = []  # (text index, start, end, value)
    for ti, (text, mapped) in enumerate(zip(session.texts(), session.mapped_lists())):
        for mt in pattern.finditer(text):
            a, b = mt.start(), mt.end()
            overlapping = [m for m in mapped if a < m.span.end and m.span.start < b]
            if any(not (a <= m.span.start and m.span.end <= b) or (m.span.start, m.span.end) == (a, b)
                   for m in overlapping):
                continue  # already replaced, inside or partly overlapping a substitution
            found.append((ti, a, b, mt.group(0)))
    if not found:
        return 0

    mapper = session.mapper
    mapper.reserve(v for *_, v in found)
    # Base forms before genitives so "Annas" links to "Anna".
    found.sort(key=lambda f: (f[3].lower().endswith("s") and len(f[3]) > len(term), f[0], f[1]))
    lists = session.mapped_lists()
    for ti, a, b, value in found:
        mapped = lists[ti]
        mapped[:] = [m for m in mapped if not (a <= m.span.start and m.span.end <= b)]
        entry = mapper.map(entity_type, value)
        span = Span(a, b, entity_type, value, 1.0, "manual")
        mapped.append(MappedSpan(span=span, entity_id=entry.id, fake=mapper.fake_for_alias(entry, value)))
        session.candidates[ti] = [c for c in session.candidates[ti] if not (c.start < b and a < c.end)]
    for mapped in lists:
        mapped.sort(key=lambda m: m.span.start)
    return len(found)


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

    detected = [
        pipeline.detect_with_candidates(h.full_text) if h.full_text.strip() else ([], []) for h in handles
    ]
    spans_per_handle = propagate_person_names([h.full_text for h in handles], [d[0] for d in detected])
    cands = [filter_candidates(sp, d[1]) for sp, d in zip(spans_per_handle, detected)]
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
        candidates=cands,
    )
    return session, _collect_preview_entities(session)


def _shift(item, offset: int):
    """Move a Span or Candidate found in a line to whole-text offsets."""
    import dataclasses

    return dataclasses.replace(item, start=item.start + offset, end=item.end + offset)


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

    # Detect line by line, like paragraphs in a .docx: the language check
    # and the models then see one paragraph / table row / list item at a time
    # instead of a whole mixed Swedish-English document.
    lines = [(m.start(), m.group()) for m in re.finditer(r"[^\n]+", text)]
    detected = [pipeline.detect_with_candidates(line) if line.strip() else ([], []) for _, line in lines]
    per_line = propagate_person_names([line for _, line in lines], [d[0] for d in detected])
    spans: list[Span] = []
    cands: list[Candidate] = []
    for (offset, _line), line_spans, (_, line_cands) in zip(lines, per_line, detected):
        spans.extend(_shift(s, offset) for s in line_spans)
        cands.extend(_shift(c, offset) for c in filter_candidates(line_spans, line_cands))
    mapped_spans = _map_spans(mapper, spans)

    session = CleanSession(
        input_path=input_path,
        kind="text",
        mapper=mapper,
        text=text,
        mapped_spans=mapped_spans,
        candidates=[cands],
    )
    return session, _collect_preview_entities(session)


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

    used = {m.entity_id for mapped in session.mapped_lists() for m in mapped}
    kept = [e for e in session.mapper.all_entries() if e.id in used and e.id not in excluded]
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
            prev = next((h for h in full_report.exact if h["entry_id"] == hit["entry_id"]), None)
            if prev:
                prev["count"] += hit["count"]   # one line per mapping, summed over paragraphs
            else:
                full_report.exact.append(dict(hit))
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


AUDIT_SCHEMA_VERSION = 2


def write_audit_log(
    report: RestoreReport,
    audit_path: Path,
    *,
    restored_document: str,
    mapping_file: str,
) -> None:
    """Persist a structured log of the restore pass alongside the restored doc.

    Schema v2: `not_found_in_input` (was `missing` in v1) lists mapping
    entries whose fake never appears in the restored input. That is expected
    when the LLM shortened the text; it is not a failed restoration.
    """
    total = sum(int(hit.get("count", 1)) for hit in report.exact) + len(report.fuzzy)
    payload = {
        "version": AUDIT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "restored_document": restored_document,
        "mapping_used": mapping_file,
        "summary": {
            "exact_count": len(report.exact),
            "fuzzy_count": len(report.fuzzy),
            "not_found_in_input_count": len(report.missing),
            "total_replacements": total,
        },
        "exact": list(report.exact),
        "fuzzy": list(report.fuzzy),
        "not_found_in_input": list(report.missing),
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
