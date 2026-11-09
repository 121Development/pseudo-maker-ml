"""Restore: replace fake values in an LLM output with the originals.

Strategy:
  1. For every mapping entry, search for `fake` and any `aliases` (exact match).
     For PERSON entries, also try the first-name slice of the fake to catch
     short-form mentions ("Erik" inside "Erik Lindqvist" → "Anna" of original).
  2. Sort all matches by length descending so longer matches win when nested
     (e.g. "Erik Lindqvist" before just "Erik").
  3. Apply non-overlapping replacements to the text.
  4. For mapping entries with zero exact matches, run rapidfuzz partial-ratio
     against tokens of the document. If a token has a high enough similarity
     score (>= threshold), replace it and flag as fuzzy in the report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from rapidfuzz import fuzz

from .. import swedish as sv

# Word-like types where an LLM may freely add a Swedish genitive -s.
_GENITIVE_TYPES = {"PERSON", "COMPANY", "CITY", "STREET"}


@dataclass
class RestoreReport:
    exact: list[dict] = field(default_factory=list)   # {entry_id, fake, original, count}
    fuzzy: list[dict] = field(default_factory=list)   # {entry_id, matched_text, original, score}
    missing: list[dict] = field(default_factory=list) # {entry_id, fake, original, type}


def _person_part_pairs(entry: dict) -> list[tuple[str, str]]:
    """(fake part, original part) for each name part of a PERSON entry.

    'Eva-Lena Berg' ← 'Anna-Karin Lindqvist' gives
    [('Eva-Lena', 'Anna-Karin'), ('Berg', 'Lindqvist'), ('Eva', 'Anna'), ('Lena', 'Karin')].
    LLMs often refer to people by surname or first name alone even if the
    source never did.
    """
    if entry.get("type") != "PERSON":
        return []
    o_tokens, f_tokens = entry["original"].split(), entry["fake"].split()
    if len(o_tokens) != len(f_tokens) or len(o_tokens) < 2:
        return []
    out = []
    for o, f in zip(o_tokens, f_tokens):
        if o.lower() in sv.NAME_PARTICLES or o == f:
            continue
        out.append((f, o))
        o_sub, f_sub = o.split("-"), f.split("-")
        if len(o_sub) == len(f_sub) > 1:
            out.extend((fs, os_) for os_, fs in zip(o_sub, f_sub) if fs != os_)
    return out


def _unambiguous_part_candidates(entries: Sequence[dict]) -> dict[str, list[tuple[str, str]]]:
    """Per entry id: name-part candidates whose fake part points to exactly one original."""
    owners: dict[str, set[tuple[str, str]]] = {}
    for e in entries:
        for fake_part, orig_part in _person_part_pairs(e):
            owners.setdefault(fake_part, set()).add((e["id"], orig_part))
    full_fakes = {e["fake"] for e in entries}
    out: dict[str, list[tuple[str, str]]] = {}
    for fake_part, owner_set in owners.items():
        # Family members share a surname: still unambiguous if every owner
        # maps the fake part back to the same original part.
        if len({orig for _, orig in owner_set}) == 1 and fake_part not in full_fakes:
            eid, orig_part = min(owner_set)
            out.setdefault(eid, []).append((fake_part, orig_part))
    return out


def _candidates_for(entry: dict) -> list[tuple[str, str]]:
    """Return [(search_string_in_llm_output, original_to_restore_to)] pairs.

    Always includes the full fake → full original. Also includes each
    fake-side alias the mapper recorded (e.g. "Karin" → "Anna" when the
    source doc referenced the person by first name only).
    """
    out: list[tuple[str, str]] = [(entry["fake"], entry["original"])]
    fake_aliases: dict = entry.get("fake_aliases") or {}
    for orig_alias, fake_alias in fake_aliases.items():
        if fake_alias and fake_alias != entry["fake"]:
            out.append((fake_alias, orig_alias))
    if entry.get("type") in _GENITIVE_TYPES:
        # The LLM may write "Karin Bergs" even if the source never had a genitive.
        seen = {f for f, _ in out}
        for fake, orig in list(out):
            g = sv.genitive(fake)
            if g != fake and g not in seen:
                out.append((g, sv.genitive(orig)))
                seen.add(g)
    return out


def _find_matches(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    out = []
    pat = re.compile(r"(?<!\w)" + re.escape(needle) + r"(?!\w)")
    for m in pat.finditer(text):
        out.append((m.start(), m.end()))
    return out


def restore(
    text: str,
    entries: Sequence[dict],
    *,
    fuzzy_threshold: int = 92,
) -> tuple[str, RestoreReport]:
    report = RestoreReport()
    spans: list[tuple[int, int, str, str]] = []  # (start, end, replacement, entry_id)

    matched_entry_ids: set[str] = set()
    ranked: list[tuple[int, tuple[int, int, str, str]]] = []
    for entry in entries:
        local_count = 0
        for search, replacement in _candidates_for(entry):
            # Rank 0 = an entity's full fake, 1 = a short form/alias. On an
            # exact tie the full fake wins (a company called "Berg" beats the
            # surname-only alias "Berg" of some person).
            rank = 0 if search == entry["fake"] else 1
            for start, end in _find_matches(text, search):
                ranked.append((rank, (start, end, replacement, entry["id"])))
                local_count += 1
        if local_count > 0:
            matched_entry_ids.add(entry["id"])
            report.exact.append({
                "entry_id": entry["id"],
                "type": entry.get("type", ""),
                "fake": entry["fake"],
                "original": entry["original"],
                "count": local_count,
            })

    spans = [sp for _, sp in sorted(ranked, key=lambda r: r[0])]  # stable: ties keep rank order
    out = _apply_spans(text, spans)

    # Fuzzy fallback for unmatched entries
    unmatched = [e for e in entries if e["id"] not in matched_entry_ids]
    if unmatched:
        # Tokenize the *output* (post-exact) into multi-word windows
        # Single-word tokens are quick; we use process.extractOne against fake names.
        # For simplicity we look at word-level windows of size 1..4.
        words = re.findall(r"\S+", out)
        # Never fuzzy-match a word that is itself a real value: it may have
        # just been restored by the exact pass (or by another paragraph).
        real = {e["original"].lower() for e in entries}
        real |= {p for e in entries for p in re.split(r"[\s\-]+", e["original"].lower()) if p}
        for entry in unmatched:
            fake = entry["fake"]
            best = None  # (score, start_word_idx, n_words)
            n_fake_words = max(1, len(fake.split()))
            for n in {1, n_fake_words}:
                for i in range(len(words) - n + 1):
                    candidate = " ".join(words[i : i + n])
                    if candidate.lower() in real:
                        continue
                    score = fuzz.ratio(fake, candidate)
                    if score >= fuzzy_threshold and (best is None or score > best[0]):
                        best = (score, i, n, candidate)
            if best:
                _score, _i, _n, candidate = best
                # Replace only the first occurrence
                pat = re.compile(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)")
                out, n_subs = pat.subn(entry["original"], out, count=1)
                if n_subs > 0:
                    report.fuzzy.append({
                        "entry_id": entry["id"],
                        "type": entry.get("type", ""),
                        "fake": entry["fake"],
                        "matched_text": candidate,
                        "original": entry["original"],
                        "score": _score,
                    })
                    matched_entry_ids.add(entry["id"])

    # Last: bare first names / surnames ("Bergs lön", "Eva svarade"). Runs after
    # fuzzy so a partial hit doesn't stop the full name from being fuzzy-matched.
    # Parts equal to any original name part are skipped, so text that was just
    # restored is never rewritten again.
    original_parts = {
        p for e in entries if e.get("type") == "PERSON"
        for p in re.split(r"[\s\-]+", e["original"]) if p
    }
    part_spans: list[tuple[int, int, str, str]] = []
    for eid, pairs in _unambiguous_part_candidates(entries).items():
        entry = next(e for e in entries if e["id"] == eid)
        count = 0
        for fake_part, orig_part in pairs:
            variants = [(fake_part, orig_part)]
            g = sv.genitive(fake_part)
            if g != fake_part:
                variants.append((g, sv.genitive(orig_part)))
            for search, repl in variants:
                if search in original_parts:
                    continue
                for start, end in _find_matches(out, search):
                    part_spans.append((start, end, repl, eid))
                    count += 1
        if count:
            hit = next((h for h in report.exact if h["entry_id"] == eid), None)
            if hit:
                hit["count"] += count
            else:
                report.exact.append({
                    "entry_id": eid,
                    "type": entry.get("type", ""),
                    "fake": entry["fake"],
                    "original": entry["original"],
                    "count": count,
                })
            matched_entry_ids.add(eid)
    out = _apply_spans(out, part_spans)

    for entry in entries:
        if entry["id"] not in matched_entry_ids:
            report.missing.append({
                "entry_id": entry["id"],
                "fake": entry["fake"],
                "original": entry["original"],
                "type": entry["type"],
            })

    return out, report


def _apply_spans(text: str, spans: list[tuple[int, int, str, str]]) -> str:
    """Apply non-overlapping replacements, preferring longer then earlier matches."""
    spans = sorted(spans, key=lambda s: (-(s[1] - s[0]), s[0]))
    accepted: list[tuple[int, int, str, str]] = []
    for s in spans:
        if not any(_overlap(s, a) for a in accepted):
            accepted.append(s)
    accepted.sort(key=lambda s: s[0], reverse=True)
    for start, end, repl, _eid in accepted:
        text = text[:start] + repl + text[end:]
    return text


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]
