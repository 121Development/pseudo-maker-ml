"""Run a BERT-style token-classification ONNX model on raw text.

Shared by the KB-BERT (Swedish) and Rampart (English) recognizers. Returns
one prediction per *word* (label of its first sub-token), with character
offsets into the original string, so callers can build spans directly.
"""
from __future__ import annotations

import json
import os
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


def _session_options():
    """CPU session settings.

    Spinning is off: by default each ONNX session keeps its threads busy-waiting
    after a call, and with two models plus spaCy taking turns per paragraph the
    spinning pools starve each other (3x slower on a 14-core machine).
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    threads = int(os.environ.get("PSEUDO_MAKER_THREADS", "0")) or min(8, os.cpu_count() or 4)
    opts.intra_op_num_threads = threads
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


class OnnxTokenTagger:
    def __init__(self, model_dir: Path, *, window: int = 510, stride: int = 128, batch_size: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"), _session_options(), providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.window = window
        self.stride = stride
        # Models with dynamic int8 quantization (our KB-BERT export) compute one
        # activation scale per call, so other rows in a batch shift results
        # slightly (~0.7% of words). Run those one window per call: same input,
        # same output, whatever else is in the document. Weight-only quantized
        # models (Rampart) batch exactly.
        if batch_size is None:
            batch_size = int(cfg.get("batch_size", 1 if cfg.get("quantization") == "dynamic-int8"
                                     or str(cfg.get("source", "")).startswith("KBLab") else 16))
        self.batch_size = max(1, batch_size)
        self.cls_id = self.tokenizer.token_to_id("[CLS]")
        self.sep_id = self.tokenizer.token_to_id("[SEP]")
        self.pad_id = self.tokenizer.token_to_id("[PAD]") or 0

    def _run_batch(self, windows: list[list[int]]) -> list[np.ndarray]:
        """Probabilities for several windows of sub-token ids (without CLS/SEP), padded into one call."""
        width = max(len(w) for w in windows) + 2
        ids = np.full((len(windows), width), self.pad_id, dtype=np.int64)
        mask = np.zeros_like(ids)
        for r, w in enumerate(windows):
            row = [self.cls_id, *w, self.sep_id]
            ids[r, :len(row)] = row
            mask[r, :len(row)] = 1
        feeds = {"input_ids": ids}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = mask
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        logits = self.session.run(None, feeds)[0]
        out = []
        for r, w in enumerate(windows):
            lg = logits[r, 1:len(w) + 1]
            lg = lg - lg.max(axis=-1, keepdims=True)
            p = np.exp(lg)
            out.append(p / p.sum(axis=-1, keepdims=True))
        return out

    def tag(self, text: str) -> list[WordPrediction]:
        return self.tag_many([text])[0]

    def tag_many(self, texts: list[str]) -> list[list[WordPrediction]]:
        """Tag many texts with few model calls.

        Long texts are split into overlapping windows; all windows of all
        texts are sorted by length and run in padded batches, then mapped
        back. Each token keeps the prediction from the window where it sits
        furthest from an edge.
        """
        encs = [self.tokenizer.encode(t, add_special_tokens=False) if t.strip() else None for t in texts]
        windows: list[tuple[int, int, int]] = []  # (text index, start, end)
        for ti, enc in enumerate(encs):
            n = len(enc.ids) if enc else 0
            start = 0
            while n:
                end = min(start + self.window, n)
                windows.append((ti, start, end))
                if end == n:
                    break
                start = end - self.stride

        n_labels = len(self.id2label)
        probs = [np.zeros((len(e.ids), n_labels), dtype=np.float32) if e else None for e in encs]
        margins = [np.full(len(e.ids), -1) if e else None for e in encs]
        order = sorted(range(len(windows)), key=lambda k: windows[k][2] - windows[k][1])
        for b in range(0, len(order), self.batch_size):
            chunk = [windows[k] for k in order[b:b + self.batch_size]]
            results = self._run_batch([encs[ti].ids[s:e] for ti, s, e in chunk])
            for (ti, s, e), p in zip(chunk, results):
                for i in range(s, e):
                    margin = min(i - s, e - 1 - i)
                    if margin > margins[ti][i]:
                        margins[ti][i] = margin
                        probs[ti][i] = p[i - s]

        out: list[list[WordPrediction]] = []
        for enc, pr in zip(encs, probs):
            if enc is None or not len(enc.ids):
                out.append([])
                continue
            word_end: dict[int, int] = {}
            for i, w in enumerate(enc.word_ids):
                if w is not None:
                    word_end[w] = enc.offsets[i][1]
            words, seen = [], set()
            for i, w in enumerate(enc.word_ids):
                if w is None or w in seen:
                    continue
                seen.add(w)
                k = int(pr[i].argmax())
                words.append(WordPrediction(enc.offsets[i][0], word_end[w], self.id2label[k], float(pr[i][k])))
            out.append(words)
        return out


@lru_cache(maxsize=4)
def load_tagger(model_dir: str) -> OnnxTokenTagger:
    return OnnxTokenTagger(Path(model_dir))
