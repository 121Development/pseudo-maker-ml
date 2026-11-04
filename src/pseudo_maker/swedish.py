"""Swedish language helpers shared by detection, substitution and restore."""
from __future__ import annotations

import random
import re
from functools import lru_cache


# --- Genitive -------------------------------------------------------------

def genitive(word: str) -> str:
    """Swedish genitive: 'Anna' → 'Annas', but 'Lars' / 'Max' / 'Ruiz' stay as-is."""
    if not word:
        return word
    if word[-1].lower() in "sxz":
        return word
    return word + ("S" if word.isupper() else "s")


def strip_genitive(word: str) -> str | None:
    """Return the base form if `word` *could* be a genitive ('Annas' → 'Anna'), else None."""
    if len(word) > 2 and word[-1] in "sS" and word[-2].isalpha():
        return word[:-1]
    return None


# --- Names ----------------------------------------------------------------

@lru_cache(maxsize=1)
def _name_lists() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    from faker.providers.person.sv_SE import Provider as P
    female = {n.lower() for n in P.first_names_female}
    male = {n.lower() for n in P.first_names_male}
    female |= _EXTRA_FEMALE
    male |= _EXTRA_MALE
    last = {n.lower() for n in P.last_names}
    return frozenset(female), frozenset(male), frozenset(last)


# Common Swedish names that Faker's sv_SE lists lack.
_EXTRA_FEMALE = {
    "anna-karin", "ann-sofie", "ann-christin", "eva-lena", "britt-marie",
    "karin", "kerstin", "birgitta", "margareta", "ingrid", "marie", "lena",
    "malin", "ulla", "inger", "gunilla", "agneta", "ebba", "elsa", "majken",
}
_EXTRA_MALE = {
    "lars-erik", "per-olof", "jan-erik", "karl-johan", "carl", "per", "lars",
    "nils", "olof", "anders", "jan", "bengt", "göran", "sven", "åke", "leif",
    "mats", "magnus", "johan", "pelle", "kalle", "nisse", "janne",
}

NAME_PARTICLES = {"von", "af", "de", "van", "der", "la", "le", "du", "di", "da"}


def is_known_first_name(name: str) -> bool:
    f, m, _ = _name_lists()
    n = name.lower()
    return n in f or n in m or all(p in f or p in m for p in n.split("-") if p)


def is_known_name(name: str) -> bool:
    _, _, last = _name_lists()
    return is_known_first_name(name) or name.lower() in last


def name_gender(first: str) -> str | None:
    """'F', 'M' or None if unknown/ambiguous. Hyphenated names use the first part."""
    f, m, _ = _name_lists()
    n = first.lower()
    if n not in f and n not in m:
        n = n.split("-")[0]
    in_f, in_m = n in f, n in m
    if in_f and not in_m:
        return "F"
    if in_m and not in_f:
        return "M"
    return None


def _title_word(word: str) -> str:
    """'[REDACTED' → '[Redacted', 'KARLSSON' → 'Karlsson' (first letter kept upper)."""
    for i, c in enumerate(word):
        if c.isalpha():
            return word[:i + 1] + word[i + 1:].lower()
    return word


def match_case(template: str, value: str) -> str:
    """Render `value` in the same casing style as `template` (UPPER / lower / as-is)."""
    letters = [c for c in template if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return value.upper()
    if letters and all(c.islower() for c in letters):
        return value.lower()
    v_letters = [c for c in value if c.isalpha()]
    if len(v_letters) > 1 and all(c.isupper() for c in v_letters):
        # Fake was generated for an ALL-CAPS original; render in Title Case here.
        return re.sub(r"[^\s\-]+", lambda m: _title_word(m.group(0)), value)
    return value


# --- Digit-shape helpers --------------------------------------------------

def reshape_digits(template: str, digits: str) -> str:
    """Pour `digits` into the digit positions of `template`, keeping all separators.

    reshape_digits("070-123 45 67", "0709876543") → "070-987 65 43"
    """
    it = iter(digits)
    return "".join(next(it, "0") if c.isdigit() else c for c in template)


def random_digits(n: int, rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(n))


ASCII_FOLD = str.maketrans({"å": "a", "ä": "a", "ö": "o", "é": "e", "ü": "u",
                            "Å": "A", "Ä": "A", "Ö": "O", "É": "E", "Ü": "U"})


def ascii_fold(s: str) -> str:
    return s.translate(ASCII_FOLD)


SV_MONTHS = ["januari", "februari", "mars", "april", "maj", "juni", "juli",
             "augusti", "september", "oktober", "november", "december"]
SV_MONTHS_SHORT = ["jan", "feb", "mar", "apr", "maj", "jun", "jul", "aug",
                   "sep", "okt", "nov", "dec"]


def parse_month(token: str) -> int | None:
    t = token.lower().rstrip(".")
    if t in SV_MONTHS:
        return SV_MONTHS.index(t) + 1
    if t in SV_MONTHS_SHORT:
        return SV_MONTHS_SHORT.index(t) + 1
    return None




# --- Language guess ---------------------------------------------------------

_EN_STOP = frozenset(
    "the and of to is are was were with for in on at by from this that be have has "
    "will would please our your their his her it as an or not we you they".split()
)
_SV_STOP = frozenset(
    "och att det som är på med för en ett av till den har inte om var vid "
    "jag vi ni de han hon sin sitt sina ska kan också eller samt där när".split()
)
_WORD = re.compile(r"[A-Za-zÅÄÖåäöÉé']+")


def looks_english(text: str) -> bool:
    """Cheap stop-word vote: True if the text reads as English rather than Swedish.

    Short texts with no stop words (headings, names, table cells) return
    True so the English model also gets a look at them.
    """
    words = [w.lower() for w in _WORD.findall(text)]
    en = sum(w in _EN_STOP for w in words)
    sv = sum(w in _SV_STOP for w in words)
    if en == 0 and sv == 0:
        return True
    return en >= sv
