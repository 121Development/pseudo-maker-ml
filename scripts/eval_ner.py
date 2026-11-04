"""Compare NER setups on the hand-labelled gold set (tests/data/ner_gold.jsonl).

    python scripts/eval_ner.py [--configs a,b,...] [--show-errors CONFIG]

Metrics, per language (sv / en / mix) and overall:
  leak-free recall  share of gold entities whose every character is replaced
                    by *some* detection (what matters for anonymization)
  typed recall      share covered by a detection of the right kind
                    (PERSON / COMPANY / LOC=CITY+STREET)
  precision         share of name-like detections that overlap a gold entity
  ms/passage        mean detection time
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pseudo_maker.detect.pipeline import DetectionPipeline, propagate_person_names  # noqa: E402

GROUP = {"PERSON": "PERSON", "COMPANY": "COMPANY", "CITY": "LOC", "STREET": "LOC", "LOCATION": "LOC"}
NAME_LIKE = set(GROUP)


def load_gold(path: Path):
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        text = d["text"]
        occ = []
        for surface, typ in sorted(d["entities"], key=lambda e: -len(e[0])):
            for m in re.finditer(r"(?<![\w])" + re.escape(surface) + r"(?![\w])", text):
                if not any(m.start() >= s and m.end() <= e for s, e, _ in occ):
                    occ.append((m.start(), m.end(), typ))
        items.append((d["lang"], text, occ))
    return items


def pipeline_detector(**kw):
    p = DetectionPipeline(**kw)

    def run(text):
        return propagate_person_names([text], [p.detect(text)])[0]
    return run


def _no_assist():
    p = DetectionPipeline(ner="kb", english_ner="auto")
    p.english_assist = None
    return lambda text: propagate_person_names([text], [p.detect(text)])[0]


def model_only(kind):
    if kind == "kb":
        from pseudo_maker.detect.kb_ner import KbSwedishNerRecognizer as R
    elif kind == "rampart":
        from pseudo_maker.detect.rampart_ner import RampartRecognizer as R
    else:
        from pseudo_maker.detect.ner_swedish import SwedishNerRecognizer as R
    r = R()
    return lambda text: list(r.find(text))


CONFIGS = {
    "rules":               lambda: pipeline_detector(ner="off", english_ner="off"),
    "spacy+rules":         lambda: pipeline_detector(ner="spacy", english_ner="off"),
    "kb+rules":            lambda: pipeline_detector(ner="kb", english_ner="off"),
    "kb+rampart(auto)+rules": lambda: pipeline_detector(ner="kb", english_ner="auto"),
    "kb+rampart(all)+rules":  lambda: pipeline_detector(ner="kb", english_ner="on"),
    "kb+rampart(auto)+rules-noassist": lambda: _no_assist(),
    "spacy+rampart(auto)+rules": lambda: pipeline_detector(ner="spacy", english_ner="auto"),
    "spacy-only":          lambda: model_only("spacy"),
    "kb-only":             lambda: model_only("kb"),
    "rampart-only":        lambda: model_only("rampart"),
}


def evaluate(detect, gold, errors=None):
    stats = defaultdict(lambda: defaultdict(int))
    elapsed = 0.0
    detect(gold[0][1])  # warm-up (model load)
    for lang, text, occ in gold:
        t0 = time.perf_counter()
        spans = detect(text)
        elapsed += time.perf_counter() - t0
        covered = [False] * len(text)
        covered_by = [set() for _ in text]
        for s in spans:
            for i in range(s.start, s.end):
                covered[i] = True
                covered_by[i].add(GROUP.get(s.type, s.type))
        for key in (lang, "all"):
            for start, end, typ in occ:
                st = stats[key]
                st["gold"] += 1
                if all(covered[start:end]):
                    st["leakfree"] += 1
                elif errors is not None and key == "all":
                    errors.append(f"MISS  {text[start:end]!r:28} in: {text}")
                if all(typ in covered_by[i] for i in range(start, end)):
                    st["typed"] += 1
            for s in spans:
                if s.type not in NAME_LIKE:
                    continue
                st = stats[key]
                st["pred"] += 1
                if any(s.start < e and b < s.end for b, e, _ in occ):
                    st["tp"] += 1
                elif errors is not None and key == "all":
                    errors.append(f"FP    {s.value!r:28} ({s.type}) in: {text}")
    return stats, elapsed / len(gold) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=ROOT / "tests/data/ner_gold.jsonl")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--show-errors", default=None, help="print misses/false positives for one config")
    args = ap.parse_args()
    gold = load_gold(args.gold)

    header = f"{'config':40} {'lang':4} {'leak-free R':>11} {'typed R':>8} {'precision':>9} {'ms':>6}"
    print(header)
    print("-" * len(header))
    for name in args.configs.split(","):
        try:
            detect = CONFIGS[name]()
        except Exception as e:  # model missing etc.
            print(f"{name:40} skipped: {e}")
            continue
        errors = [] if name == args.show_errors else None
        stats, ms = evaluate(detect, gold, errors)
        for lang in ("sv", "en", "mix", "all"):
            st = stats[lang]
            lf = st["leakfree"] / st["gold"]
            ty = st["typed"] / st["gold"]
            pr = st["tp"] / st["pred"] if st["pred"] else 0.0
            print(f"{name if lang == 'sv' else '':40} {lang:4} {lf:11.1%} {ty:8.1%} {pr:9.1%} {ms if lang == 'all' else '':>6}"
                  if lang != "all" else f"{'':40} {lang:4} {lf:11.1%} {ty:8.1%} {pr:9.1%} {ms:6.1f}")
        print()
        if errors:
            print("\n".join(errors))
            print()


if __name__ == "__main__":
    main()
