"""Typer CLI: clean / restore / detect."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import orchestrate
from .io import docx_io, text_io

app = typer.Typer(
    add_completion=False,
    help="Anonymize Swedish PII in documents for safe LLM use, then restore originals.",
)
console = Console()


SYSTEM_PROMPT_TEMPLATE = """\
The text below has been pre-processed: every personal name, company name,
address, phone number, ID number, and similar identifier has been replaced
with a realistic substitute. Treat the substitutes as if they were the real
entities. When you respond, please preserve the substituted names, numbers,
and identifiers verbatim — do not translate, paraphrase, or correct them.

---

Texten nedan har förbehandlats: alla person- och företagsnamn, adresser,
telefonnummer, person- och organisationsnummer och liknande uppgifter har
ersatts med realistiska påhittade värden. Behandla dem som om de vore de
verkliga. Återge namn, nummer och andra uppgifter exakt som de står när du
svarar – översätt, förkorta eller rätta dem inte. Svensk genitiv går bra
("Karin Bergs"), men ändra inte stavningen av själva namnet.
"""


def _default_out_path(input_path: Path, suffix: str) -> Path:
    """Produce e.g. `contract.docx` + `_CLEANED` → `contract_CLEANED.docx`."""
    stem = input_path.stem
    return input_path.with_name(f"{stem}_{suffix}{input_path.suffix}")


def _default_mapping_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_MAPPING.json")


def _default_prompt_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_PROMPT.txt")


def _default_audit_path(restored_path: Path) -> Path:
    stem = restored_path.stem
    # Strip trailing _RESTORED so the audit sits next to the cleaned/restored pair.
    if stem.endswith("_RESTORED"):
        stem = stem[: -len("_RESTORED")]
    return restored_path.with_name(f"{stem}_AUDIT.json")


@app.command()
def clean(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="Input .docx or .txt/.md file."),
    out: Path = typer.Option(None, "--out", "-o", help="Cleaned output path (default: <input>.cleaned.<ext>)."),
    mapping: Path = typer.Option(None, "--mapping", "-m", help="Mapping file path (default: <input>.mapping.json)."),
    system_prompt: Path = typer.Option(None, "--system-prompt", help="Where to write the suggested LLM system prompt."),
    no_ner: bool = typer.Option(False, "--no-ner", help="Skip all NER models (rules only). Same as --ner off."),
    ner: str = typer.Option(None, "--ner", help="Swedish NER: auto | kb | spacy | off (default: auto = KB-BERT if installed, else spaCy)."),
    english_ner: str = typer.Option(None, "--english-ner", help="English NER (Rampart): auto | on | off (default: auto = English-looking paragraphs)."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible fakes."),
):
    """Produce a cleaned version of the document and a mapping.json for restore."""
    out = out or _default_out_path(input_path, "CLEANED")
    mapping = mapping or _default_mapping_path(input_path)
    system_prompt = system_prompt or _default_prompt_path(input_path)

    if orchestrate.is_docx_path(input_path):
        mapper = orchestrate.clean_docx(input_path, out, mapping, use_ner=not no_ner, seed=seed, ner=ner, english_ner=english_ner)
    else:
        mapper = orchestrate.clean_text(input_path, out, mapping, use_ner=not no_ner, seed=seed, ner=ner, english_ner=english_ner)

    Path(system_prompt).write_text(SYSTEM_PROMPT_TEMPLATE, encoding="utf-8")

    entries = mapper.all_entries()
    table = Table(title=f"Cleaned: {input_path.name}", show_lines=False)
    table.add_column("Type")
    table.add_column("Count", justify="right")
    by_type: dict[str, int] = {}
    for e in entries:
        by_type[e.type] = by_type.get(e.type, 0) + 1
    for t in sorted(by_type):
        table.add_row(t, str(by_type[t]))
    console.print(table)
    console.print(f"[green]Wrote[/green] {out}")
    console.print(f"[green]Wrote[/green] {mapping}")
    console.print(f"[green]Wrote[/green] {system_prompt}")


@app.command()
def restore(
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="LLM output document (.docx or .txt/.md)."),
    mapping: Path = typer.Option(..., "--mapping", "-m", exists=True, readable=True, help="Mapping file from `clean`."),
    out: Path = typer.Option(None, "--out", "-o", help="Restored output path (default: <input>_RESTORED.<ext>)."),
    audit: Path = typer.Option(None, "--audit", help="Audit log path (default: <stem>_AUDIT.json next to the restored file)."),
    no_audit: bool = typer.Option(False, "--no-audit", help="Skip writing the audit log."),
    fuzzy_threshold: int = typer.Option(92, "--fuzzy-threshold", help="rapidfuzz score (0-100) above which fuzzy matches are accepted."),
):
    """Reverse a cleaned document by re-substituting originals using the mapping."""
    out = out or _default_out_path(input_path, "RESTORED")
    if orchestrate.is_docx_path(input_path):
        report = orchestrate.restore_docx(input_path, out, mapping, fuzzy_threshold=fuzzy_threshold)
    else:
        report = orchestrate.restore_text_file(input_path, out, mapping, fuzzy_threshold=fuzzy_threshold)

    console.print(f"[green]Restored[/green] {len(report.exact)} entities exactly.")
    if report.fuzzy:
        console.print(f"[yellow]Restored {len(report.fuzzy)} entities via fuzzy match (review recommended):[/yellow]")
        for hit in report.fuzzy:
            console.print(f"  '{hit['matched_text']}' → '{hit['original']}' (score {hit['score']})")
    if report.missing:
        console.print(f"[red]{len(report.missing)} entities NOT found in LLM output:[/red]")
        for hit in report.missing:
            console.print(f"  {hit['entry_id']} {hit['type']} '{hit['fake']}' (original: '{hit['original']}')")
    console.print(f"[green]Wrote[/green] {out}")

    if not no_audit:
        audit_path = audit or _default_audit_path(out)
        orchestrate.write_audit_log(
            report,
            audit_path,
            restored_document=out.name,
            mapping_file=Path(mapping).name,
        )
        console.print(f"[green]Wrote[/green] {audit_path}")


@app.command()
def detect(
    input_path: Path = typer.Argument(..., exists=True, readable=True),
    no_ner: bool = typer.Option(False, "--no-ner"),
    ner: str = typer.Option(None, "--ner", help="auto | kb | spacy | off"),
    english_ner: str = typer.Option(None, "--english-ner", help="auto | on | off"),
):
    """Dry-run: list detected entities without writing a cleaned document."""
    pipeline = orchestrate.make_pipeline(use_ner=not no_ner, ner=ner, english_ner=english_ner)
    if orchestrate.is_docx_path(input_path):
        _doc, handles = docx_io.load(input_path)
        text = "\n".join(h.full_text for h in handles)
    else:
        text = text_io.read_text(input_path)
    spans = pipeline.detect(text)

    table = Table(title=f"Detected entities in {input_path.name}")
    table.add_column("Type")
    table.add_column("Value")
    table.add_column("Source")
    table.add_column("Confidence", justify="right")
    for s in spans:
        table.add_row(s.type, s.value, s.source, f"{s.confidence:.2f}")
    console.print(table)
    console.print(f"[bold]Total:[/bold] {len(spans)} entities")


@app.command("fetch-models")
def fetch_models(
    dest: Path = typer.Option(None, "--dest", help="Models root (default: $PSEUDO_MAKER_MODELS_DIR or ~/.cache/pseudo-maker/models)."),
    skip_kb: bool = typer.Option(False, "--skip-kb", help="Only fetch Rampart."),
):
    """Download Rampart and build the KB-BERT ONNX model (KB needs the [export] extra)."""
    import os

    from . import models

    env_dir = os.environ.get("PSEUDO_MAKER_MODELS_DIR")
    root = dest or (Path(env_dir) if env_dir else models.cache_root())
    console.print(f"Rampart → {models.fetch_rampart(root)}")
    if skip_kb:
        return
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        console.print("[yellow]Building KB-BERT needs the export extra:[/] pip install '.[export]'")
        raise typer.Exit(1)
    from .export_kb import main as export_kb

    raise typer.Exit(export_kb(["--out", str(root)]))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (local-only by default)."),
    port: int = typer.Option(8765, "--port", "-p", help="Port for the web UI."),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not auto-open the browser."),
):
    """Start the local web UI (http://127.0.0.1:<port>)."""
    from .server.app import run
    run(host=host, port=port, open_browser=not no_browser)


if __name__ == "__main__":
    app()
