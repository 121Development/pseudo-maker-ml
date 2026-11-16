"""Export "golden" outputs that the .NET port must reproduce exactly.

    python scripts/export_golden.py OUT_DIR

Writes, for the gold/held-out sets, the synthetic CV and some edge-case texts:
  tokens.json   tokenizer ids, offsets and word ids (KB-BERT and Rampart)
  words.json    per-word model labels and scores (KB-BERT and Rampart)
  detect.json   final spans and ambiguous candidates per line, per document
  restore.json  cleaned text + mapping from this version, and the restored
                text and report expected from them

Setup: KB-BERT + Rampart + rules, no spaCy (the .NET setup). All offsets are
converted to UTF-16 code units, which is how .NET indexes strings.
"""
from __future__ import annotations

import dataclasses
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pseudo_maker import models, orchestrate  # noqa: E402
from pseudo_maker.detect.models.onnx_tagger import load_tagger  # noqa: E402
from pseudo_maker.detect.pipeline import (  # noqa: E402
    DetectionPipeline,
    filter_candidates,
    propagate_person_names,
)
from pseudo_maker.restore.restore import restore  # noqa: E402

EDGE = [
    "",
    "   ",
    "Mötet med 👩‍💼 Anna Ek och 🎉 Per Åkesson hölls i Västerås.",
    "Kontakt: Lena Holm, tfn 070-123 45 67.",
    "Kafé Ekdahl ägs av Bo Ek.",  # combining accent
    "Namn:\tKARIN LUNDQVIST\tPersonnummer: 19850612-1238",
    "## Erfarenhet\n",
    "| Namn | Företag |",
    "| Johan Berg | Nordström Bygg AB |",
    "**Referens:** Anna-Karin Lindqvist, *Volvo Cars*, Göteborg",
    "Dear Mr. O'Neill, please contact Siobhán Ní Bhriain at 14 Example Orchard Lane, Leeds LS1 4AB.",
    "Kenji Watanabe (Toyota) and Lucia Ferreira (Galp) agreed on the pilot.",
    "Mötet med Volvo Personvagnar hölls i Västerås. " * 60,
]


def u16(text: str, i: int) -> int:
    """Code-point offset → UTF-16 offset."""
    return len(text[:i].encode("utf-16-le")) // 2


def load_jsonl(name: str) -> list[str]:
    return [json.loads(line)["text"] for line in (ROOT / "tests/data" / name).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def documents() -> dict[str, list[str]]:
    cv = (ROOT / "tests/data/fake_cv_swedish_english.md").read_text(encoding="utf-8")
    return {
        "gold": load_jsonl("ner_gold.jsonl"),
        "heldout": load_jsonl("ner_heldout.jsonl"),
        "heldout2": load_jsonl("ner_heldout2.jsonl"),
        "cv": re.findall(r"[^\n]+", cv),
        "edge": EDGE,
    }


def pipeline() -> DetectionPipeline:
    p = DetectionPipeline(ner="kb", english_ner="auto")
    p.english_assist = None
    return p


def export_tokens_and_words(texts: list[str]) -> tuple[dict, dict]:
    tokens, words = {}, {}
    for name in (models.KB_NER, models.RAMPART):
        tagger = load_tagger(str(models.model_dir(name)))
        tok_out, word_out = [], []
        preds = tagger.tag_many(texts)
        for text, wp in zip(texts, preds):
            enc = tagger.tokenizer.encode(text, add_special_tokens=False)
            tok_out.append({
                "text": text,
                "ids": enc.ids,
                "offsets": [[u16(text, s), u16(text, e)] for s, e in enc.offsets],
                "word_ids": [-1 if w is None else w for w in enc.word_ids],
            })
            word_out.append({
                "text": text,
                "words": [[u16(text, w.start), u16(text, w.end), w.label, round(w.score, 5)] for w in wp],
            })
        tokens[name], words[name] = tok_out, word_out
    return tokens, words


def span_row(text: str, s) -> list:
    return [u16(text, s.start), u16(text, s.end), s.type, s.value, round(s.confidence, 5), s.source]


def cand_row(text: str, c) -> list:
    return [u16(text, c.start), u16(text, c.end), c.type, c.value, round(c.score, 5), c.reason]


def export_detect(p: DetectionPipeline, docs: dict[str, list[str]]) -> dict:
    out = {}
    for name, lines in docs.items():
        detected = p.detect_many(lines)
        per_line = propagate_person_names(lines, [d[0] for d in detected])
        out[name] = [
            {
                "text": text,
                "spans": [span_row(text, s) for s in spans],
                "candidates": [cand_row(text, c) for c in filter_candidates(spans, cands)],
            }
            for text, spans, (_, cands) in zip(lines, per_line, detected)
        ]
    return out


def llm_edit(text: str, entries: list[dict]) -> str:
    """A deterministic stand-in for an LLM rewrite: changed wording, case
    changes and a one-letter typo in some fakes, so fuzzy restore runs too."""
    persons = [e for e in entries if e["type"] == "PERSON" and len(e["fake"]) > 8]
    for i, e in enumerate(persons[:6]):
        fake = e["fake"]
        if i % 3 == 0:
            text = text.replace(fake, fake.upper(), 1)
        elif i % 3 == 1:
            text = text.replace(fake, fake[:-2] + fake[-1] + fake[-2], 1)  # swap last two letters
    return "Sammanfattning (omskriven):\n" + text.replace(" och ", " samt ")


def export_restore(p: DetectionPipeline, docs: dict[str, list[str]]) -> dict:
    orchestrate.make_pipeline = lambda **_: p  # same detection setup as above
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("gold", "heldout2", "cv"):
            src = tmp / f"{name}.txt"
            src.write_text("\n".join(docs[name]) + "\n", encoding="utf-8")
            orchestrate.clean_text(src, tmp / f"{name}_clean.txt", tmp / f"{name}_map.json", seed=42)
            cleaned = (tmp / f"{name}_clean.txt").read_text(encoding="utf-8")
            mapping = json.loads((tmp / f"{name}_map.json").read_text(encoding="utf-8"))
            cases = []
            for variant, text in (("unchanged", cleaned), ("llm", llm_edit(cleaned, mapping["entries"]))):
                restored, report = restore(text, mapping["entries"])
                cases.append({"variant": variant, "input": text, "restored": restored,
                              "report": dataclasses.asdict(report)})
            out[name] = {"original": src.read_text(encoding="utf-8"), "cleaned": cleaned,
                         "mapping": mapping, "cases": cases}
    return out


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = documents()
    texts = sorted({t for lines in docs.values() for t in lines})
    p = pipeline()
    tokens, words = export_tokens_and_words(texts)
    parts = {
        "tokens.json": tokens,
        "words.json": words,
        "detect.json": export_detect(p, docs),
        "restore.json": export_restore(p, docs),
    }
    for fname, data in parts.items():
        (out_dir / fname).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{fname}: {(out_dir / fname).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
