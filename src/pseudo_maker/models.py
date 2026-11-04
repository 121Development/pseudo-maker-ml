"""Locate (and, on request, fetch) the ONNX NER models.

Models never download at runtime. They are looked up in, in order:
  1. $PSEUDO_MAKER_MODELS_DIR/<name>
  2. the PyInstaller bundle (<_MEIPASS>/models/<name>)
  3. ~/.cache/pseudo-maker/models/<name>

Each model directory holds `model.onnx`, `tokenizer.json` and `config.json`
(with `id2label`). Run `pseudo-maker fetch-models` to populate the cache.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

KB_NER = "kb-ner"        # KBLab/bert-base-swedish-cased-ner, exported to int8 ONNX
RAMPART = "rampart"      # nationaldesignstudio/rampart (q4 ONNX)

KB_NER_HF_ID = "KBLab/bert-base-swedish-cased-ner"
RAMPART_HF_ID = "nationaldesignstudio/rampart"

_REQUIRED = ("model.onnx", "tokenizer.json", "config.json")


def cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pseudo-maker" / "models"


def _candidates(name: str) -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("PSEUDO_MAKER_MODELS_DIR")
    if env:
        out.append(Path(env) / name)
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        out.append(Path(bundle) / "models" / name)
    out.append(cache_root() / name)
    return out


def model_dir(name: str) -> Path | None:
    """Directory of an installed model, or None if it isn't installed."""
    for d in _candidates(name):
        if all((d / f).is_file() for f in _REQUIRED):
            return d
    return None


def available(name: str) -> bool:
    return model_dir(name) is not None


def runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def fetch_rampart(dest_root: Path | None = None) -> Path:
    """Download Rampart's q4 ONNX + tokenizer from Hugging Face."""
    import shutil

    from huggingface_hub import hf_hub_download

    dest = (dest_root or cache_root()) / RAMPART
    dest.mkdir(parents=True, exist_ok=True)
    for remote, local in (
        ("onnx/model_q4.onnx", "model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
        ("config.json", "config.json"),
        ("LICENSE", "LICENSE"),
    ):
        path = hf_hub_download(RAMPART_HF_ID, remote)
        shutil.copyfile(path, dest / local)
    return dest
