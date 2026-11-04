"""Per-entity fake-value generators."""
from __future__ import annotations

import random
import string
import re
from datetime import datetime, timedelta

from faker import Faker

from .. import swedish as sv
from . import valid_format as vf

# Faker is seeded per-mapper; this module is stateless.
_fake_se = Faker("sv_SE")
_fake_en = Faker("en_US")


_COMPANY_SUFFIX = re.compile(
    r"\s+(AB(?:\s*\(publ\))?|HB|KB|Aktiebolag(?:\s*\(publ\))?|Handelsbolag|"
    r"Kommanditbolag|ek\.\s?för\.?|ekonomisk förening|i likvidation|"
    r"Ltd\.?|Limited|Inc\.?|LLC|LLP|plc|PLC|GmbH|AG|Oy|Oyj|ASA|A/S|B\.V\.|N\.V\.|S\.A\.|Corp\.?|Corporation)$",
    re.IGNORECASE,
)


def _preserve_company_suffix(original: str, generated: str) -> str:
    """Give the fake exactly the legal-form suffix of the original (or none)."""
    base = _COMPANY_SUFFIX.sub("", generated).strip()
    m = _COMPANY_SUFFIX.search(original.strip())
    return f"{base} {m.group(1)}" if m else base


# Real Swedish double first names, so hyphenated originals get natural fakes.
_DOUBLE_FEMALE = [
    "Anna-Karin", "Anna-Lena", "Ann-Sofie", "Ann-Christin", "Ann-Charlotte",
    "Eva-Lena", "Eva-Karin", "Britt-Marie", "Maj-Britt", "Anna-Maria",
    "Lena-Maria", "Ann-Marie", "Karin-Sofia", "Inga-Lill", "Gun-Britt",
]
_DOUBLE_MALE = [
    "Lars-Erik", "Per-Olof", "Jan-Erik", "Karl-Johan", "Carl-Johan", "Sven-Erik",
    "Lars-Göran", "Per-Anders", "Nils-Erik", "Jan-Olof", "Karl-Erik", "Bengt-Göran",
    "Hans-Åke", "Per-Erik", "Lars-Olof",
]


def _first_name(gender: str | None, hyphenated: bool, rng: random.Random | None = None) -> str:
    if hyphenated:
        pool = _DOUBLE_FEMALE if gender == "F" else _DOUBLE_MALE if gender == "M" else _DOUBLE_FEMALE + _DOUBLE_MALE
        return (rng or random).choice(pool)
    if gender == "F":
        return _fake_se.first_name_female()
    if gender == "M":
        return _fake_se.first_name_male()
    return _fake_se.first_name()


def _last_name(hyphenated: bool) -> str:
    name = _fake_se.last_name()
    if hyphenated:
        name = f"{name}-{_fake_se.last_name()}"
    return name


_SXZ_SURNAMES = [
    "Lindfors", "Wikfors", "Bergfors", "Hedfors", "Rehnfors", "Fors", "Malmros",
    "Ros", "Lantz", "Frantz", "Schultz", "Fritz", "Wentz", "Kranz",
]
_SXZ_CITIES = ["Borås", "Västerås", "Vännäs", "Robertsfors", "Bollnäs", "Kramfors", "Hagfors"]


def _ends_sxz(word: str) -> bool:
    return bool(word) and word[-1].lower() in "sxz"


def fake_person(original: str, rng: random.Random) -> str:
    """Swedish fake name with the same shape as the original.

    Keeps: number of name parts, hyphenation per part, particles such as
    'von'/'af', the gender of the first name (so 'hon'/'han' in the
    surrounding text still fits), and ALL-CAPS styling.
    """
    Faker.seed(rng.randint(0, 2**31))
    tokens = original.split()
    if not tokens:
        return _fake_se.name()
    gender = sv.name_gender(tokens[0])
    single_token_is_surname = (
        len(tokens) == 1 and not sv.is_known_first_name(tokens[0]) and sv.is_known_name(tokens[0])
    )
    original_parts = {p.lower() for p in re.split(r"[\s\-]+", original) if p}
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok.lower() in sv.NAME_PARTICLES:
            out.append(tok)
            continue
        hyph = "-" in tok
        for attempt in range(20):
            if i == 0 and not single_token_is_surname:
                cand = _first_name(gender, hyph, rng)
            elif i < len(tokens) - 1 and sv.is_known_first_name(tok):
                cand = _first_name(gender, hyph, rng)  # middle name
            else:
                cand = _last_name(hyph)
            # Never reuse any part of the real name ("Anna-Karin" → "Anna-Lena" leaks "Anna").
            if {p.lower() for p in cand.split("-")} & original_parts:
                continue
            # The last part must take the genitive the same way as the original
            # ("Mohammed" → "Mats" would make "Mohammeds" and "Mohammed" collide).
            if i == len(tokens) - 1 and _ends_sxz(cand) != _ends_sxz(tok):
                if _ends_sxz(tok) and attempt >= 10 and (i > 0 or single_token_is_surname):
                    # Faker has few such surnames; fall back to real Swedish ones.
                    cand = rng.choice(_SXZ_SURNAMES)
                    if hyph:
                        cand = f"{_last_name(False)}-{cand}"
                    if not {p.lower() for p in cand.split("-")} & original_parts:
                        break
                continue
            break
        out.append(cand)
    return sv.match_case(original, " ".join(out))


def fake_company(original: str, rng: random.Random) -> str:
    Faker.seed(rng.randint(0, 2**31))
    name = _fake_se.company()
    return sv.match_case(original, _preserve_company_suffix(original, name))


_HOUSE_NUMBER = re.compile(r"\s+(\d{1,4})([A-Za-z]|\s[A-Z])?(?![\wåäö])(.*)$")


def fake_address(original: str, rng: random.Random) -> str:
    """Street or mixed-address. Used for STREET and the legacy ADDRESS type.

    'Storgatan' → 'Björkvägen'; 'Storgatan 12B' → 'Björkvägen 7A';
    'Storgatan 12, 3 tr' → 'Björkvägen 41, 3 tr'.
    """
    Faker.seed(rng.randint(0, 2**31))
    if "\n" in original:
        return _fake_se.address().replace("\n", ", ")
    street = _fake_se.street_name()
    lead = re.match(r"^(\d{1,4}[A-Za-z]?)\s+", original)
    if lead:  # English style: "15 Oak Avenue" → "37 Björkvägen"
        return f"{rng.randint(1, 99 if len(lead.group(1)) <= 2 else 199)} {sv.match_case(original[lead.end():], street)}"
    m = _HOUSE_NUMBER.search(original)
    if not m:
        return sv.match_case(original, street)
    num = str(rng.randint(1, 99 if len(m.group(1)) <= 2 else 199))
    letter = ""
    if m.group(2):
        sep = " " if m.group(2).startswith(" ") else ""
        letter = sep + rng.choice("ABCD")
    return sv.match_case(original, f"{street} {num}{letter}{m.group(3)}")


def fake_city(original: str, rng: random.Random) -> str:
    """Swedish city name. Used for single-token LOC matches."""
    Faker.seed(rng.randint(0, 2**31))
    city = _fake_se.city()
    for _ in range(20):
        if _ends_sxz(city) == _ends_sxz(original.strip()):
            break
        city = _fake_se.city()
    else:
        if _ends_sxz(original.strip()):
            city = rng.choice([c for c in _SXZ_CITIES if c.lower() != original.strip().lower()])
    return sv.match_case(original, city)


def fake_phone(original: str, rng: random.Random) -> str:
    """Same format as the original; keeps the country code and the area/mobile prefix.

    '070-123 45 67' → '070-842 19 03', '+46 8 555 123 45' → '+46 8 731 904 22'.
    """
    digits = re.sub(r"\D", "", original)
    if original.lstrip().startswith("+46"):
        keep = 3 + (2 if digits[2:3] == "7" else 1)   # 46 + '7x' mobile or area digit
    else:
        keep = 3 if digits[1:2] == "7" else 2         # '07x' mobile or '0x' area
    new = digits[:keep] + sv.random_digits(len(digits) - keep, rng)
    return sv.reshape_digits(original, new)


_PUBLIC_MAIL_DOMAINS = {
    "gmail.com", "hotmail.com", "hotmail.se", "outlook.com", "outlook.se",
    "live.se", "live.com", "yahoo.com", "yahoo.se", "icloud.com", "me.com",
    "telia.com", "bredband.net", "spray.se", "msn.com", "protonmail.com", "proton.me",
}


def _name_parts(name: str) -> list[str]:
    return [
        sv.ascii_fold(p).lower()
        for p in re.split(r"[\s\-]+", name)
        if p and p.lower() not in sv.NAME_PARTICLES
    ]


def fake_email(
    original: str,
    rng: random.Random,
    name_hint: tuple[str, str] | None = None,
) -> str:
    """Fake email that keeps the shape of the original.

    * Public webmail domains (gmail.com, hotmail.se, telia.com, …) are kept —
      they don't identify anyone. Other domains get a fake '<surname>.<tld>'.
    * `name_hint` = (original person name, fake person name). When the local
      part is built from the person's name it is rewritten part-by-part:
      'anna-karin.lindqvist' + (Anna-Karin Lindqvist → Eva-Lena Berg)
      → 'eva-lena.berg'.
    """
    Faker.seed(rng.randint(0, 2**31))
    local, _, domain = original.partition("@")
    new_local = None
    if name_hint:
        orig_parts, fake_parts = _name_parts(name_hint[0]), _name_parts(name_hint[1])
        if len(orig_parts) == len(fake_parts):
            table = dict(zip(orig_parts, fake_parts))
            pat = re.compile("|".join(re.escape(p) for p in sorted(table, key=len, reverse=True)))
            folded = sv.ascii_fold(local).lower()
            candidate = pat.sub(lambda m: table[m.group(0)], folded)
            if candidate != folded:
                new_local = candidate
    if new_local is None:
        parts = _name_parts(_fake_se.name())
        sep_m = re.search(r"[._\-]", local)
        if sep_m and len(parts) >= 2:
            new_local = f"{parts[0]}{sep_m.group(0)}{parts[-1]}"
        elif re.fullmatch(r"[a-z]{1,3}", local.lower()):
            new_local = "".join(p[0] for p in parts)[: len(local)]
        else:
            new_local = parts[0]
    if domain.lower() in _PUBLIC_MAIL_DOMAINS:
        new_domain = domain
    else:
        tld = domain.rsplit(".", 1)[-1] if "." in domain else "se"
        new_domain = f"{sv.ascii_fold(_fake_se.last_name()).lower()}.{tld}"
    return f"{new_local}@{new_domain}"


def fake_url(original: str, rng: random.Random) -> str:
    Faker.seed(rng.randint(0, 2**31))
    # Preserve scheme and any path
    m = re.match(r"^(https?://)([^/]+)(.*)$", original, re.IGNORECASE)
    if not m:
        return _fake_en.url()
    scheme, _host, path = m.groups()
    return f"{scheme}{_fake_en.domain_name()}{path}"


def fake_ip(original: str, rng: random.Random) -> str:
    Faker.seed(rng.randint(0, 2**31))
    if ":" in original:
        return _fake_en.ipv6()
    return _fake_en.ipv4_private()


def fake_vat(original: str, rng: random.Random) -> str:
    cc = original[:2].upper() if original[:2].isalpha() else "SE"
    if cc == "SE":
        # Swedish VAT = SE + organisationsnummer + 01
        body = vf.fake_orgnr(re.sub(r"\D", "", original)[:10], rng)
        return "SE" + sv.reshape_digits(original[2:], body + "01")
    digits = re.sub(r"\D", "", original)
    return cc + sv.reshape_digits(original[2:], sv.random_digits(len(digits), rng))


_LONG_DATE = re.compile(r"^(\d{1,2})(:[ae])?(\s+)([A-Za-zåäöÅÄÖ]+\.?)(?:(\s+)(\d{4}))?$")


def fake_date(original: str, rng: random.Random) -> str:
    """Shift the date by ±90 days, keeping the exact input format.

    Handles ISO / numeric forms and Swedish long forms ('15 januari 2025',
    '3 mars', '1:a feb. 2024').
    """
    delta = timedelta(days=rng.choice([d for d in range(-90, 91) if d != 0]))
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d.%m.%Y", "%Y%m%d"):
        try:
            dt = datetime.strptime(original, fmt)
            return (dt + delta).strftime(fmt)
        except ValueError:
            continue
    m = _LONG_DATE.match(original.strip())
    if m:
        day, ordinal, sp1, month_tok, sp2, year = m.groups()
        month = sv.parse_month(month_tok)
        if month:
            try:
                dt = datetime(int(year) if year else 2000, month, int(day)) + delta
            except ValueError:
                dt = None
            if dt:
                short = month_tok.lower().rstrip(".") in sv.SV_MONTHS_SHORT and month_tok.lower().rstrip(".") not in sv.SV_MONTHS
                name = (sv.SV_MONTHS_SHORT if short else sv.SV_MONTHS)[dt.month - 1]
                if month_tok.endswith("."):
                    name += "."
                if month_tok.isupper():
                    name = name.upper()
                elif month_tok[0].isupper():
                    name = name.capitalize()
                if ordinal:
                    ordinal = ":a" if dt.day % 10 in (1, 2) and dt.day not in (11, 12) else ":e"
                out = f"{dt.day}{ordinal or ''}{sp1}{name}"
                if year:
                    out += f"{sp2}{dt.year}"
                return out
    Faker.seed(rng.randint(0, 2**31))
    return _fake_se.date()


def fake_age(original: str, rng: random.Random) -> str:
    m = re.search(r"\d+", original)
    if not m:
        return original
    n = int(m.group(0))
    new_n = max(0, n + rng.choice([-2, -1, 1, 2]))
    return original[: m.start()] + str(new_n) + original[m.end():]


def fake_postal_code(original: str, rng: random.Random) -> str:
    return vf.fake_postal_code(rng, original)


def fake_personnummer(original: str, rng: random.Random) -> str:
    return vf.fake_personnummer(original, rng)


def fake_orgnr(original: str, rng: random.Random) -> str:
    return vf.fake_orgnr(original, rng)


def fake_bankgiro(original: str, rng: random.Random) -> str:
    return vf.fake_bankgiro(original, rng)


def fake_plusgiro(original: str, rng: random.Random) -> str:
    return vf.fake_plusgiro(original, rng)


def fake_credit_card(original: str, rng: random.Random) -> str:
    return vf.fake_credit_card(rng, original)


def fake_iban(original: str, rng: random.Random) -> str:
    return vf.fake_iban_se(rng, original)


def fake_license_plate(original: str, rng: random.Random) -> str:
    return vf.fake_license_plate(rng, original)


def fake_passport(original: str, rng: random.Random) -> str:
    return vf.fake_passport(rng, original)


def fake_case_number(original: str, rng: random.Random) -> str:
    """Same shape; keeps letter prefixes ('KS', 'T') and 4-digit years, randomises the rest."""
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        if len(tok) == 4 and tok[:2] in ("19", "20"):
            return tok
        # Court numbers end in a 2-digit year: 'T 1234-23'
        if len(tok) == 2 and m.end() == len(original) and original[m.start() - 1] == "-":
            return tok
        return str(rng.randint(1, 9)) + sv.random_digits(len(tok) - 1, rng)
    return re.sub(r"\d+", repl, original)


def fake_id_number(original: str, rng: random.Random) -> str:
    """Generic ID (tax id, account, licence …): same letters/digits/separators."""
    return "".join(
        rng.choice(string.ascii_uppercase) if c.isalpha() and c.isupper()
        else rng.choice(string.ascii_lowercase) if c.isalpha()
        else str(rng.randint(0, 9)) if c.isdigit()
        else c
        for c in original
    )


# Dispatch table — one entry per entity type
GENERATORS = {
    "PERSON": fake_person,
    "COMPANY": fake_company,
    "STREET": fake_address,
    "CITY": fake_city,
    "ADDRESS": fake_address,  # backward-compat alias
    "POSTAL_CODE": fake_postal_code,
    "PHONE": fake_phone,
    "EMAIL": fake_email,
    "URL": fake_url,
    "IP": fake_ip,
    "PERSONNUMMER": fake_personnummer,
    "ORGNR": fake_orgnr,
    "BANKGIRO": fake_bankgiro,
    "PLUSGIRO": fake_plusgiro,
    "CREDIT_CARD": fake_credit_card,
    "IBAN": fake_iban,
    "VAT": fake_vat,
    "LICENSE_PLATE": fake_license_plate,
    "PASSPORT": fake_passport,
    "CASE_NUMBER": fake_case_number,
    "ID_NUMBER": fake_id_number,
    # Mixed case on purpose: an ALL-CAPS occurrence then gets "[REDACTED-123]",
    # which restore can tell apart.
    "CUSTOM": lambda original, rng: sv.match_case(original, f"[Redacted-{rng.randint(100, 999)}]"),
    "DATE": fake_date,
    "AGE": fake_age,
}


def generate_fake(entity_type: str, original: str, rng: random.Random, **hints) -> str:
    gen = GENERATORS.get(entity_type)
    if gen is None:
        # Unknown type: opaque token fallback
        return f"[{entity_type}_{rng.randint(1000, 9999)}]"
    return gen(original, rng, **hints) if hints else gen(original, rng)
