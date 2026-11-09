"""Built-in blocklist of terms that NER sometimes tags as names/locations
but almost never are in a Swedish professional-document context.

Two kinds of rejections here:

1. `PERSON_BLOCKLIST` — lowercase-normalized terms that should never be
   treated as PERSON. Applied to single-token matches and to any match whose
   normalized form exactly matches.

2. `ROLE_WORDS` — job-title fragments that sometimes get pulled into a PERSON
   span by spaCy (e.g. "OWNER/FOUNDER"). If a detected span equals a role word
   (case-insensitive) or is a pure role-word with a trailing slash, drop it.

3. `COUNTRIES_AND_CONTINENTS` — geographic names the Swedish NER tags as LOC
   but that we keep as-is (mentioning "Somalia" in a CV isn't PII; replacing
   it with "Stationsgatan 1" is worse than leaving it).
"""
from __future__ import annotations

# Common English/Swedish nouns that spaCy's Swedish NER occasionally mis-tags
# as PERSON when they appear capitalized in CVs / job descriptions.
PERSON_BLOCKLIST: set[str] = {
    # Business / CV jargon
    "site", "sites", "turnover", "revenue", "profit", "budget", "scope",
    "scale", "scope", "mission", "vision", "agenda", "brief", "scope",
    "owner", "founder", "partner", "client", "customer", "supplier", "vendor",
    "team", "staff", "crew", "board", "executive", "management", "lead",
    "project", "program", "product", "portfolio", "stakeholder",
    # Roles
    "ceo", "cto", "cfo", "coo", "cio", "cmo", "vd", "chef", "grundare",
    "ägare", "direktör", "styrelseordförande", "konsult", "projektledare",
    "manager", "director", "president", "chair", "chairman", "chairwoman",
    "head", "officer", "analyst", "specialist", "engineer", "developer",
    # Tech / things often capitalized in CVs
    "java", "python", "linux", "windows", "macos", "excel", "word",
    "powerpoint", "office", "outlook", "slack", "teams", "docker", "kubernetes",
    "aws", "azure", "gcp", "react", "angular", "vue", "django",
    "cv", "lebenslauf", "resumé", "resume",
    # Sports / hobby terms that sometimes get flagged
    "barbell", "dumbbell", "crossfit", "yoga", "pilates",
    # Honorifics / titles
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor", "sir", "madam", "fru", "herr",
    "fröken", "doktor", "docent",
    # Weekdays and months (EN + SV)
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "måndag", "tisdag", "onsdag", "torsdag", "fredag", "lördag", "söndag",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "januari", "februari", "mars",
    "maj", "juni", "juli", "augusti", "oktober",
    # Pronouns (EN + SV) — capitalised at sentence start
    "she", "he", "they", "we", "i", "you", "it", "her", "his", "hon", "han", "hen", "de", "vi", "jag",
    # Languages / nationalities
    "swedish", "english", "german", "french", "spanish", "finnish", "norwegian", "danish",
    "arabic", "italian", "dutch", "polish", "russian", "chinese", "somali", "persian",
    "svenska", "engelska", "tyska", "franska", "spanska", "finska", "arabiska",
    "curriculum vitae", "cv", "personligt brev", "cover letter",
    # More tools that get tagged as names
    "jira", "tableau", "power bi", "sql", "confluence", "figma", "notion", "trello",
    "salesforce", "hubspot", "sap", "visma", "fortnox",
    # Conjunctions models sometimes glue between names
    "och", "and", "samt", "eller", "or",
    # Misc
    "mainly", "primarily", "ongoing", "recent", "previous", "current",
}


ROLE_WORDS: set[str] = {
    "ceo", "cto", "cfo", "coo", "cio", "cmo", "cpo",
    "owner", "founder", "co-founder", "cofounder", "vd", "chef",
    "grundare", "ägare", "direktör", "president", "vice president",
    "manager", "director", "head", "lead", "chief", "officer",
}


COUNTRIES_AND_CONTINENTS: set[str] = {
    # Swedish names
    "sverige", "norge", "danmark", "finland", "island", "tyskland", "frankrike",
    "spanien", "italien", "storbritannien", "england", "irland", "nederländerna",
    "belgien", "schweiz", "österrike", "portugal", "grekland", "polen",
    "tjeckien", "slovakien", "ungern", "rumänien", "bulgarien", "estland",
    "lettland", "litauen", "ryssland", "ukraina", "vitryssland",
    "usa", "kanada", "mexiko", "brasilien", "argentina", "chile",
    "kina", "japan", "indien", "sydkorea", "nordkorea", "thailand", "vietnam",
    "indonesien", "filippinerna", "malaysia", "singapore",
    "somalia", "etiopien", "kenya", "egypten", "sydafrika", "marocko",
    "irak", "iran", "syrien", "libanon", "israel", "saudiarabien",
    "australien", "nya zeeland",
    "europa", "asien", "afrika", "nordamerika", "sydamerika", "oceanien",
    # Regions / multi-country names
    "norden", "skandinavien", "nordics", "scandinavia", "baltikum", "eu", "europeiska unionen",
    "european union", "united kingdom", "kingdom", "uk", "great britain", "united states",
    "united states of america", "us", "the netherlands", "the united kingdom", "storbritannien",
    # English variants we might see in mixed-language CVs
    "sweden", "norway", "denmark", "germany", "france", "spain", "italy",
    "britain", "england", "netherlands", "belgium", "switzerland", "austria",
    "poland", "russia", "ukraine", "canada", "brazil", "china", "japan",
    "india", "korea", "thailand", "indonesia", "philippines", "iraq", "iran",
    "syria", "lebanon", "israel", "europe", "asia", "africa",
    # Notable islands that double as tech / other terms
    "java",
}


def normalize(value: str) -> str:
    return value.strip().lower()


def is_blocklisted_person(value: str) -> bool:
    v = normalize(value)
    if not v:
        return True
    if v in PERSON_BLOCKLIST:
        return True
    if v in ROLE_WORDS:
        return True
    # Strip trailing slashes / punctuation and re-check (handles "OWNER/").
    stripped = v.rstrip(" /-_.,:;")
    if stripped in PERSON_BLOCKLIST or stripped in ROLE_WORDS:
        return True
    return False


def is_country_or_continent(value: str) -> bool:
    v = normalize(value)
    return v in COUNTRIES_AND_CONTINENTS


# ORG-tagged terms that are products, platforms or public bodies rather than
# identifying the parties of a document.
COMPANY_BLOCKLIST: set[str] = {
    "teams", "microsoft teams", "zoom", "skype", "google meet", "meet", "webex",
    "linkedin", "facebook", "instagram", "youtube", "twitter", "tiktok", "github",
    "gitlab", "jira", "confluence", "sharepoint", "onedrive", "dropbox",
    "bankid", "mobilt bankid", "swish", "gdpr", "moms", "eu", "fn", "nato",
    "skatteverket", "bolagsverket", "försäkringskassan", "arbetsförmedlingen",
    "kronofogden", "polisen", "migrationsverket", "transportstyrelsen",
    "riksdagen", "regeringen", "imy", "integritetsskyddsmyndigheten", "ab", "hb", "kb",
}


COMPANY_BLOCKLIST |= {
    "socialtjänsten", "socialstyrelsen", "trafikverket", "skolverket", "boverket",
    "naturvårdsverket", "arbetsmiljöverket", "tullverket", "försvarsmakten",
    "socialnämnden", "kommunen", "regionen", "länsstyrelsen", "domstolsverket",
    "pensionsmyndigheten", "e-hälsomyndigheten", "csn", "1177", "vårdcentralen",
}

# Job-title words: a span made only of these is a role, not a company or person.
JOB_WORDS: set[str] = {
    "learning", "training", "service", "services", "improvement", "relationship", "client",
    "delivery", "unit", "programme", "coordinator", "administrative", "operations", "chair",
    "partner", "sponsor", "colleague", "supervisor", "trainer", "programme", "teacher",
    "data", "scientist", "engineer", "developer", "analyst", "manager", "consultant",
    "senior", "junior", "lead", "head", "chief", "officer", "director", "designer",
    "architect", "specialist", "coordinator", "assistant", "intern", "trainee",
    "product", "project", "program", "software", "backend", "frontend", "fullstack",
    "sales", "marketing", "finance", "hr", "operations", "support", "owner", "of",
    "utvecklare", "konsult", "chef", "projektledare", "säljare", "ekonom", "ingenjör",
}


def is_job_title(value: str) -> bool:
    words = [w for w in value.lower().replace(",", " ").split() if w]
    return bool(words) and all(w in JOB_WORDS for w in words)


# Swedish public bodies by name shape: "Trafikverket", "Kronofogdemyndigheten", "Arbetsmiljöinspektionen".
_AGENCY_SUFFIX = ("verket", "myndigheten", "inspektionen", "styrelsen", "nämnden", "kassan", "rådet")


def is_blocklisted_company(value: str) -> bool:
    v = normalize(value)
    if v in COMPANY_BLOCKLIST or is_job_title(v):
        return True
    return " " not in v and v.endswith(_AGENCY_SUFFIX)


# Field labels in CVs, forms and email headers. A model sometimes tags the
# label of a table cell or a "Label:" line as a name ("Arbetsadress", "Thesis").
LABEL_WORDS: set[str] = {
    # Swedish
    "arbetsadress", "bostadsadress", "postadress", "adress", "kontaktuppgifter", "kontakt",
    "personnummer", "födelsedatum", "telefon", "mobil", "e-post", "mejl", "arbetsmejl",
    "befattning", "relation", "referensspråk", "referens", "referenser", "profil", "portfolio",
    "körkort", "medborgarskap", "tilltalsnamn", "namn", "fullständigt", "uppgift", "fält",
    "utbildning", "arbetslivserfarenhet", "språk", "kompetenser", "examensarbete", "handledare",
    "kursledare", "från", "till", "kopia", "datum", "ämne", "bilaga", "meddelande",
    "sammanfattning", "övrigt", "underskrifter", "vänliga", "hälsningar", "hej",
    # English
    "title", "relationship", "email", "telephone", "phone", "mobile", "office", "address",
    "language", "languages", "thesis", "subject", "date", "from", "to", "cc", "bcc", "contact",
    "profile", "summary", "experience", "education", "skills", "references", "reference",
    "appendix", "correspondence", "name", "field", "value", "nationality", "citizenship",
    "curriculum", "vitae", "resume", "professional", "dear", "regards", "best", "kind",
    "sincerely", "note", "notes",
}


def is_label_word(value: str) -> bool:
    return normalize(value).rstrip(":") in LABEL_WORDS


# Currencies and amount units: never an entity ("SEK 12 million", "45 tkr").
CURRENCY_WORDS: set[str] = {
    "sek", "kr", "kronor", "kronur", "tkr", "mkr", "msek", "ksek", "eur", "euro", "euros", "usd",
    "dollar", "dollars", "gbp", "pund", "pounds", "nok", "dkk", "chf", "jpy", "cny", "isk", "pln",
}


def is_currency(value: str) -> bool:
    return normalize(value).strip(".") in CURRENCY_WORDS
