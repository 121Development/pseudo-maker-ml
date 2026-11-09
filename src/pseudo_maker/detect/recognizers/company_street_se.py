"""Pattern-based company names and street addresses (source="pattern").

Pattern spans compete with model spans by length (unlike checksum/regex IDs,
which always win), so "Sandvik Coromant" from the model beats "Sandvik" here.

These complement spaCy NER, which often misses the legal suffix
("Nordström Bygg AB (publ)") or the house number ("Storgatan 12B").
"""
from __future__ import annotations

import re
from typing import Iterable

from .base import Span

_CAP = r"[A-ZÅÄÖÉ][\wåäöÅÄÖéü&.\-]*"
_COMPANY = re.compile(
    rf"\b({_CAP}(?:[ \t]+(?:{_CAP}|&|och|i))*?)[ \t]+"
    r"(AB|HB|KB|Aktiebolag|Handelsbolag|Kommanditbolag|Ltd\.?|Limited|Inc\.?|LLC|LLP|plc|PLC|"
    r"GmbH|AG|Oy|Oyj|ASA|A/S|B\.V\.|N\.V\.|S\.A\.|Corp\.?|Corporation|Solicitors)(?![\w])([ \t]*\(publ\))?"
)

# Capitalised words that start sentences / clauses in contracts and are not
# part of the company name: "Mellan Volvo AB och ...", "Bolaget Volvo AB".
_LEADING_STOPWORDS = {
    "mellan", "med", "och", "hos", "från", "till", "för", "av", "den", "det",
    "detta", "denna", "dessa", "bolaget", "företaget", "avtalet", "enligt",
    "där", "som", "vid", "på", "i", "om", "att", "nedan", "ovan", "parterna",
    "kunden", "leverantören", "arbetsgivaren", "uppdragsgivaren", "säljaren",
    "köparen", "hyresvärden", "hyresgästen", "vi", "jag", "han", "hon",
    "dotterbolaget", "moderbolaget", "koncernen", "firma", "firman", "kund",
    "leverantör", "part", "anställd", "vd", "ordförande", "styrelsen", "the",
    "and", "with", "between", "at", "for", "inom", "under", "efter", "innan",
    "samt", "eller", "även", "också", "när", "då", "så", "men", "både",
}

_STREET_SUFFIX = (
    r"(?:gatan|gata|vägen|väg|stigen|torget|torg|gränd|gränden|allén|allé|"
    r"esplanaden|leden|platsen|plan|kajen|backen|backe|parken|liden|gången|"
    r"strand|stranden|kroken|slingan|ringen|bryggan)"
)
_STREET_PREFIX = r"(?:(?:Sankt|S:t|Gamla|Norra|Södra|Östra|Västra|Lilla|Stora|Nya|Övre|Nedre)[ \t]+)?"
_STREET = re.compile(
    rf"\b{_STREET_PREFIX}[A-ZÅÄÖ][a-zåäöéü]*{_STREET_SUFFIX}"
    r"[ \t]+\d{1,4}(?:[A-Za-z]|[ \t][A-Z])?(?![\wåäö])(?:[ \t]*,[ \t]*\d{1,2}[ \t]*tr\.?)?"
)


class CompanySeRecognizer:
    name = "COMPANY"

    def find(self, text: str) -> Iterable[Span]:
        for m in _COMPANY.finditer(text):
            words = m.group(1).split()
            # Drop leading stopwords: "Mellan Nordström Bygg" → "Nordström Bygg".
            while words and words[0].lower() in _LEADING_STOPWORDS:
                words.pop(0)
            if not words or words[-1].lower() in ("och", "i", "på", "&"):
                continue
            start = text.index(words[0], m.start(1))
            end = m.end()
            yield Span(start, end, self.name, text[start:end], 0.9, "pattern")


# English-style addresses: "14 Example Orchard Lane", "22 Sample Quay Road", "221B Baker Street".
_EN_STREET = re.compile(
    r"\b\d{1,5}[A-Za-z]?[ \t]+(?:[A-Z][a-z'’]+[ \t]+){1,4}"
    r"(?:Street|St\.|Road|Rd\.|Lane|Ln\.|Avenue|Ave\.|Court|Ct\.|Close|Drive|Way|Place|Square|"
    r"Terrace|Gardens|Crescent|Quay|Row|Mews|Hill|Boulevard|Blvd\.|Parkway|Grove|Walk|Wharf|Yard)\b"
)


class StreetSeRecognizer:
    name = "STREET"

    def find(self, text: str) -> Iterable[Span]:
        for m in _STREET.finditer(text):
            yield Span(m.start(), m.end(), self.name, m.group(0), 0.9, "pattern")
        for m in _EN_STREET.finditer(text):
            yield Span(m.start(), m.end(), self.name, m.group(0), 0.85, "pattern")


# Large Swedish employers / brands that spaCy frequently misses or truncates.
# In a contract, CV or case file, who someone works for is identifying.
KNOWN_SWEDISH_COMPANIES = [
    "Volvo", "Volvo Cars", "Scania", "Ericsson", "IKEA", "Ikea", "H&M", "Spotify",
    "Klarna", "SEB", "Swedbank", "Handelsbanken", "Nordea", "Länsförsäkringar",
    "Telia", "Tele2", "Telenor", "Vattenfall", "ICA", "Ica", "Coop", "Axfood",
    "Skanska", "NCC", "Peab", "JM", "Saab", "SKF", "Atlas Copco", "Sandvik",
    "Electrolux", "Husqvarna", "Alfa Laval", "Essity", "SCA", "Boliden", "SSAB",
    "Investor", "Securitas", "Assa Abloy", "Hexagon", "Epiroc", "Getinge",
    "Elekta", "Northvolt", "PostNord", "Postnord", "Systembolaget", "Apoteket",
    "Folksam", "Trygg-Hansa", "Skandia", "Bonnier", "Arla", "Clas Ohlson",
    "Jula", "Biltema", "Stena", "Fortum", "Ellevio", "Sweco", "AFRY", "Afry",
    "Tietoevry", "Capio", "Samhall", "Riksbyggen", "HSB", "Akademiska Hus",
    "Vasakronan", "Castellum", "Fabege", "Atrium Ljungberg", "Kinnevik",
    "Lindex", "KappAhl", "Kappahl", "Åhléns", "Willys", "Hemköp", "Mathem",
    "Avanza", "Nordnet", "Collector", "Resurs Bank", "SBAB", "Söderberg & Partners",
    "Ramböll", "WSP", "Capgemini", "King", "Mojang", "Truecaller", "Einride", "Voi",
    "Sandvik Coromant", "Hemköp", "Sahlgrenska",
    # International employers frequent in Swedish CVs and contracts
    "Google", "Microsoft", "Amazon", "Apple", "Meta", "Facebook", "Netflix", "IBM", "Oracle",
    "SAP", "Cisco", "Intel", "Nvidia", "Samsung", "Nokia", "Siemens", "Bosch", "Philips",
    "Accenture", "Deloitte", "PwC", "KPMG", "EY", "Ernst & Young", "McKinsey", "BCG",
    "Bain", "Capgemini", "Sogeti", "Tieto", "CGI", "Cognizant", "Infosys", "TCS",
    "Vodafone", "Barclays", "HSBC", "Danske Bank", "DNB", "Unilever", "Nestlé", "Shell",
    "Maersk", "Novo Nordisk", "Equinor", "Uber", "Airbnb", "Salesforce",
]
_KNOWN_CO = re.compile(
    r"(?<![\w@.])(" + "|".join(re.escape(c) for c in sorted(KNOWN_SWEDISH_COMPANIES, key=len, reverse=True)) + r")"
    # Optional division/subsidiary words: "Volvo Personvagnar", "ICA Gruppen", "Telia Sverige"
    r"((?:[ \t]+(?:Personvagnar|Lastvagnar|Gruppen|Group|Sverige|Sweden|Nordic|Norden|"
    r"Maxi|Kvantum|Supermarket|Nära|Cars|Trucks|Bank|Fastigheter|Bostad|Energi|"
    r"Försäkring|Försäkringar|Liv|Pension|Construction|Holding|Konsult))*)"
    r"(?![\wåäöÅÄÖ])"
)


# Swedish sports clubs: "IFK Norrköping", "AIK", "Malmö FF", "IK Sirius".
_CLUB = re.compile(
    r"\b(?:(?:IFK|AIK|IK|IF|BK|GIF|HK|SK)[ \t]+[A-ZÅÄÖ][\wåäö]+|[A-ZÅÄÖ][\wåäö]+[ \t]+(?:FF|BK|IF|IK|HK|SK))\b"
)


class KnownCompanySeRecognizer:
    name = "COMPANY"

    def find(self, text: str) -> Iterable[Span]:
        for m in _KNOWN_CO.finditer(text):
            yield Span(m.start(), m.end(), self.name, m.group(0), 0.9, "pattern")
        for m in _CLUB.finditer(text):
            yield Span(m.start(), m.end(), self.name, m.group(0), 0.8, "pattern")
