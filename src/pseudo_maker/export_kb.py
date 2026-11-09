"""Export KBLab/bert-base-swedish-cased-ner to int8 ONNX for pseudo-maker.

Build-time only (needs the `export` extra: torch + transformers + onnx).
Runtime uses just onnxruntime + tokenizers.

    pseudo-maker fetch-models            (Rampart + this export)
    python -m pseudo_maker.export_kb [--out DIR] [--no-quantize]

Writes DIR/kb-ner/{model.onnx, tokenizer.json, config.json} and checks that
the exported model agrees with PyTorch on sample Swedish sentences.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HF_ID = "KBLab/bert-base-swedish-cased-ner"

SAMPLES = [
    "Anna-Karin Lindqvist arbetar på Nordström Bygg AB i Malmö.",
    "Per Åkessons kontor ligger på Drottninggatan 5 i Stockholm.",
    "Mötet med Volvo Personvagnar och Ica Gruppen hölls i Västerås.",
    "Handläggare Johanna Ek kontaktade Försäkringskassan om Mohammed Al-Hassan.",
    "Erik von Platen från Ericsson deltog via Teams den 15 januari.",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="models root (default: pseudo-maker cache)")
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args(argv)

    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from pseudo_maker import models

    dest = (args.out or models.cache_root()) / models.KB_NER
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_ID} …")
    tok = AutoTokenizer.from_pretrained(HF_ID, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(HF_ID).eval()

    # The KB tokenizer must stay cased and keep å/ä/ö.
    probe = tok.backend_tokenizer.normalizer.normalize_str("Åkesson ÄÖ")
    if probe != "Åkesson ÄÖ":
        print(f"Tokenizer normalizer changes text: {probe!r}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model_fp32.onnx"
        enc = tok(SAMPLES[0], return_tensors="pt")
        print("Exporting to ONNX …")
        torch.onnx.export(
            model,
            (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
            str(fp32),
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={n: {0: "batch", 1: "seq"} for n in ("input_ids", "attention_mask", "token_type_ids", "logits")},
            opset_version=17,
            dynamo=False,
        )
        target = dest / "model.onnx"
        if args.no_quantize:
            target.write_bytes(fp32.read_bytes())
        else:
            print("Quantizing to int8 …")
            quantize_dynamic(str(fp32), str(target), weight_type=QuantType.QInt8)

    tok.backend_tokenizer.save(str(dest / "tokenizer.json"))
    cfg = {
        "id2label": {str(k): v for k, v in model.config.id2label.items()},
        "source": HF_ID,
        "quantization": "none" if args.no_quantize else "dynamic-int8",
        # Dynamic int8 scales activations per call: batching would make results
        # depend on neighbouring paragraphs, so run one window per call.
        "batch_size": 16 if args.no_quantize else 1,
    }
    (dest / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    # Agreement check: ONNX vs PyTorch labels per token.
    sess = ort.InferenceSession(str(dest / "model.onnx"), providers=["CPUExecutionProvider"])
    same = total = 0
    for s in SAMPLES:
        e = tok(s, return_tensors="pt")
        with torch.no_grad():
            ref = model(**e).logits[0].argmax(-1).numpy()
        out = sess.run(None, {k: e[k].numpy() for k in ("input_ids", "attention_mask", "token_type_ids")})[0][0].argmax(-1)
        same += int((ref == out).sum())
        total += len(ref)
    agreement = same / total
    size_mb = (dest / "model.onnx").stat().st_size / 1e6
    print(f"Wrote {dest}  ({size_mb:.0f} MB)  token agreement with PyTorch: {agreement:.1%}")
    if agreement < 0.98:
        print("Agreement below 98% — consider --no-quantize", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
