"""Run a BERT-style token-classification ONNX model on raw text.

Shared by the KB-BERT (Swedish) and Rampart (English) recognizers. Returns
one prediction per *word* (label of its first sub-token), with character
offsets into the original string, so callers can build spans directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WordPrediction:
    start: int
    end: int
    label: str
    score: float


class OnnxTokenTagger:
    def __init__(self, model_dir: Path, *, window: int = 510, stride: int = 128):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.window = window
        self.stride = stride
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")

    def _run(self, ids: list[int]) -> np.ndarray:
        """Logits → probabilities for one window of sub-token ids (without CLS/SEP)."""
        full = np.array([[self.cls_id, *ids, self.sep_id]], dtype=np.int64)
        feeds = {"input_ids": full}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = np.ones_like(full)
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(full)
        logits = self.session.run(None, feeds)[0][0][1:-1]
        logits = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=-1, keepdims=True)

    def tag(self, text: str) -> list[WordPrediction]:
        if not text.strip():
            return []
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        ids, offsets, word_ids = enc.ids, enc.offsets, enc.word_ids
        n = len(ids)
        if n == 0:
            return []

        # Sliding windows; each token's prediction comes from the window where
        # it sits furthest from an edge (most context on both sides).
        probs = np.zeros((n, len(self.id2label)), dtype=np.float32)
        best_margin = np.full(n, -1)
        start = 0
        while True:
            end = min(start + self.window, n)
            p = self._run(ids[start:end])
            for i in range(start, end):
                margin = min(i - start, end - 1 - i)
                if margin > best_margin[i]:
                    best_margin[i] = margin
                    probs[i] = p[i - start]
            if end == n:
                break
            start = end - self.stride

        out: list[WordPrediction] = []
        seen_words: set[int] = set()
        word_end: dict[int, int] = {}
        for i, w in enumerate(word_ids):
            if w is not None:
                word_end[w] = offsets[i][1]
        for i, w in enumerate(word_ids):
            if w is None or w in seen_words:
                continue
            seen_words.add(w)
            k = int(probs[i].argmax())
            out.append(WordPrediction(offsets[i][0], word_end[w], self.id2label[k], float(probs[i][k])))
        return out


@lru_cache(maxsize=4)
def load_tagger(model_dir: str) -> OnnxTokenTagger:
    return OnnxTokenTagger(Path(model_dir))
