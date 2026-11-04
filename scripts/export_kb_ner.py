"""Thin wrapper: python scripts/export_kb_ner.py [--out DIR] [--no-quantize]."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pseudo_maker.export_kb import main  # noqa: E402

raise SystemExit(main())
