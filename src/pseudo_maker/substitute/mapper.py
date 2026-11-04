"""Stable input→fake mapping with alias coalescing."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import re

from .. import swedish as sv
from .fakes import generate_fake

# Types whose values are words and therefore take Swedish genitive -s.
GENITIVE_TYPES = {"PERSON", "COMPANY", "CITY", "STREET"}
ALIAS_TYPES = {"PERSON", "COMPANY"}


def _normalize(value: str) -> str:
    return " ".join(value.split()).lower()


_PUBL = re.compile(r"\s*\(publ\)\s*$", re.IGNORECASE)


def _company_core(value: str) -> str:
    """'Volvo AB (publ)' and 'Volvo AB' are the same company."""
    return _normalize(_PUBL.sub("", value))


def _parts(value: str) -> list[str]:
    """Name parts, split on whitespace and hyphens: 'Anna-Karin Lindqvist' → [anna, karin, lindqvist]."""
    return [p for p in re.split(r"[\s\-]+", value.lower()) if p]


@dataclass
class MappingEntry:
    id: str
    type: str
    original: str
    fake: str
    aliases: set[str] = field(default_factory=set)
    # Map original-side alias → fake-side rendering used in the cleaned doc.
    # Lets restore know "Karin" alone in the LLM output should become "Anna".
    fake_aliases: dict[str, str] = field(default_factory=dict)
    occurrences: int = 0
    # alias → base alias for genitive forms ("Annas" → "Anna"); not persisted.
    genitive_of: dict[str, str] = field(default_factory=dict)


class Mapper:
    """Owns the (type, normalized_original) → MappingEntry table for one document."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._by_key: dict[tuple[str, str], MappingEntry] = {}
        self._used_fakes: set[str] = set()
        # Real values (and name parts) in the document; fakes must never equal
        # one, or restore can't tell a fake from a real value.
        self._reserved: set[str] = set()
        self._next_id = 1

    def _new_id(self) -> str:
        eid = f"ent_{self._next_id:04d}"
        self._next_id += 1
        return eid

    _NOT_RESERVED = {
        "ab", "hb", "kb", "(publ)", "aktiebolag", "ltd", "inc", "llc", "plc", "gmbh", "ag",
        "och", "and", "&", "i", "på", "of",
    }

    def reserve(self, originals) -> None:
        """Register every real value of the document before mapping starts.

        Name-like parts (≥ 3 letters, not legal suffixes or particles) are
        reserved too, so a fake never reuses a real first name or surname.
        """
        for o in originals:
            self._reserved.add(o.lower())
            for p in re.split(r"[\s\-]+", o.lower()):
                if len(p) >= 3 and p.isalpha() and p not in self._NOT_RESERVED and p not in sv.NAME_PARTICLES:
                    self._reserved.add(p)

    def _collides(self, candidate: str, original: str) -> bool:
        if candidate == original or candidate in self._used_fakes:
            return True
        c = candidate.lower()
        if c in self._reserved:
            return True
        return any(p in self._reserved for p in re.split(r"[\s\-]+", c) if p)

    def _unique_fake(self, entity_type: str, original: str, **hints) -> str:
        for _ in range(50):
            candidate = generate_fake(entity_type, original, self.rng, **hints)
            if entity_type in GENITIVE_TYPES:
                bad = self._collides(candidate, original)   # word-like: check name parts too
            else:
                bad = candidate in self._used_fakes or candidate == original
            if not bad:
                self._used_fakes.add(candidate)
                return candidate
        # Last-resort suffix to guarantee uniqueness
        candidate = f"{candidate}_{self.rng.randint(1000, 9999)}"
        self._used_fakes.add(candidate)
        return candidate

    def _find_existing(self, entity_type: str, text: str) -> MappingEntry | None:
        """Exact entry for `text`, or the unique entry `text` is a single-part alias of."""
        entry = self._by_key.get((entity_type, _normalize(text)))
        if entry is not None:
            return entry
        if entity_type == "COMPANY":
            core = _company_core(text)
            for (t, _), e in self._by_key.items():
                if t == "COMPANY" and _company_core(e.original) == core:
                    return e
        if entity_type in ALIAS_TYPES and len(text.split()) == 1:
            want = text.lower()
            candidates = {
                id(e): e for (t, _), e in self._by_key.items()
                if t == entity_type and (want in _parts(e.original) or want in _normalize(e.original).split())
            }
            if len(candidates) == 1:
                return next(iter(candidates.values()))
        return None

    def _attach_alias(self, entry: MappingEntry, alias: str, key: tuple[str, str]) -> MappingEntry:
        entry.aliases.add(alias)
        entry.fake_aliases[alias] = self.fake_for_alias(entry, alias)
        entry.occurrences += 1
        self._by_key[key] = entry
        return entry

    def _email_name_hint(self, email: str) -> tuple[str, str] | None:
        """(original, fake) of the PERSON whose name the email local part is built from."""
        local = sv.ascii_fold(email.partition("@")[0]).lower()
        local_parts = set(p for p in re.split(r"[._\-+]+", local) if p)
        best = None
        for (t, _), e in self._by_key.items():
            if t != "PERSON":
                continue
            parts = [sv.ascii_fold(p) for p in _parts(e.original) if p not in sv.NAME_PARTICLES]
            hits = sum(1 for p in parts if p in local_parts)
            if hits >= 2 or (hits == 1 and len(local_parts) == 1):
                if best is None or hits > best[0]:
                    best = (hits, e)
        return (best[1].original, best[1].fake) if best else None

    def map(self, entity_type: str, original: str) -> MappingEntry:
        """Return the entry for this (type, original). Creates one if first seen.

        Swedish-aware coalescing:
          * A single name part ("Anna", "Lindqvist", or "Anna" from
            "Anna-Karin Lindqvist") reuses the unique entry containing it.
          * A genitive ("Annas", "Per Åkessons", "Volvos") reuses the entry of
            its base form, and its fake gets the genitive too ("Karins").
        """
        key = (entity_type, _normalize(original))
        if key in self._by_key:
            entry = self._by_key[key]
            entry.occurrences += 1
            return entry

        if entity_type in ALIAS_TYPES:
            existing = self._find_existing(entity_type, original)
            if existing is not None:
                return self._attach_alias(existing, original, key)

        if entity_type in GENITIVE_TYPES:
            base = sv.strip_genitive(original)
            if base:
                base_entry = self._find_existing(entity_type, base)
                if (
                    base_entry is None
                    and entity_type == "PERSON"
                    and not sv.is_known_name(original.split()[-1])
                    and sv.is_known_name(base.split()[-1])
                ):
                    # "Annas" seen before any "Anna": create the base entry.
                    base_entry = self.map(entity_type, base)
                    base_entry.occurrences -= 1
                if base_entry is not None:
                    base_entry.genitive_of[original] = base
                    return self._attach_alias(base_entry, original, key)

        if entity_type == "PERSON" and len(original.split()) >= 2:
            relative = self._same_surname(original)
            if relative is not None:
                return self._new_relative(original, key, relative)

        hints = {}
        if entity_type == "EMAIL":
            hint = self._email_name_hint(original)
            if hint:
                hints["name_hint"] = hint
        entry = MappingEntry(
            id=self._new_id(),
            type=entity_type,
            original=original,
            fake=self._unique_fake(entity_type, original, **hints),
            occurrences=1,
        )
        self._by_key[key] = entry
        return entry

    def _same_surname(self, original: str) -> MappingEntry | None:
        """An existing multi-part PERSON with the same surname (family member)."""
        last = original.split()[-1].lower()
        for (t, _), e in self._by_key.items():
            tokens = e.original.split()
            if t == "PERSON" and len(tokens) >= 2 and tokens[-1].lower() == last:
                return e
        return None

    def _new_relative(self, original: str, key: tuple[str, str], relative: MappingEntry) -> MappingEntry:
        """New person sharing `relative`'s fake surname: 'Fatima Al-Hassan' next to
        'Mohammed Al-Hassan' keeps the family link in the cleaned text."""
        fake_last = sv.match_case(original.split()[-1], relative.fake.split()[-1])
        for _ in range(50):
            head = generate_fake("PERSON", " ".join(original.split()[:-1]) + " X", self.rng).split()[:-1]
            candidate = sv.match_case(original, " ".join(head + [fake_last]))
            if candidate not in self._used_fakes and not any(
                p in self._reserved for p in re.split(r"[\s\-]+", " ".join(head).lower()) if p
            ):
                break
        self._used_fakes.add(candidate)
        entry = MappingEntry(id=self._new_id(), type="PERSON", original=original, fake=candidate, occurrences=1)
        self._by_key[key] = entry
        return entry

    def fake_for_alias(self, entry: MappingEntry, alias_text: str) -> str:
        """Derive the fake-side rendering of an alias of `entry`.

        'Anna' (of 'Anna-Karin Lindqvist' → 'Eva-Lena Berg')  → 'Eva'
        'Lindqvist'                                            → 'Berg'
        'Annas'                                                → 'Evas'
        """
        if _normalize(alias_text) == _normalize(entry.original):
            return entry.fake
        base = entry.genitive_of.get(alias_text)
        if base is not None:
            return sv.genitive(self.fake_for_alias(entry, base))
        if entry.type == "COMPANY" and _company_core(alias_text) == _company_core(entry.original):
            fake_core = _PUBL.sub("", entry.fake)
            return fake_core + alias_text[len(_PUBL.sub("", alias_text)):] if _PUBL.search(alias_text) else fake_core
        if entry.type in ALIAS_TYPES and len(alias_text.split()) == 1:
            orig_tokens = entry.original.split()
            fake_tokens = entry.fake.split()
            want = alias_text.lower()
            for i, tok in enumerate(orig_tokens):
                if i >= len(fake_tokens):
                    break
                if tok.lower() == want:
                    return sv.match_case(alias_text, fake_tokens[i])
                sub = tok.lower().split("-")
                if want in sub:
                    fake_sub = fake_tokens[i].split("-")
                    j = sub.index(want)
                    piece = fake_sub[j] if len(fake_sub) == len(sub) else fake_tokens[i]
                    return sv.match_case(alias_text, piece)
            if entry.type == "PERSON" and fake_tokens:
                return fake_tokens[0]
        return entry.fake

    def get_entry(self, entity_id: str) -> MappingEntry:
        for entry in self._by_key.values():
            if entry.id == entity_id:
                return entry
        raise KeyError(f"unknown entity_id: {entity_id}")

    def override_fake(self, entity_id: str, new_fake: str) -> None:
        """Replace an entry's fake with a user-supplied value.

        Drops the old fake from ``_used_fakes``, records the new one, and for
        PERSON entries re-derives ``fake_aliases`` so single-name aliases in
        the cleaned doc still round-trip via the refreshed first-name slice.
        Raises ``KeyError`` if no entry with the given id is tracked.
        """
        new_fake = (new_fake or "").strip()
        if not new_fake:
            raise ValueError("new_fake must be a non-empty string")

        for entry in self._by_key.values():
            if entry.id == entity_id:
                self._used_fakes.discard(entry.fake)
                entry.fake = new_fake
                self._used_fakes.add(new_fake)
                if entry.aliases:
                    entry.fake_aliases = {
                        alias: self.fake_for_alias(entry, alias)
                        for alias in entry.aliases
                    }
                return
        raise KeyError(f"unknown entity_id: {entity_id}")

    def all_entries(self) -> list[MappingEntry]:
        seen_ids: set[str] = set()
        out: list[MappingEntry] = []
        for entry in self._by_key.values():
            if entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)
            out.append(entry)
        return out
