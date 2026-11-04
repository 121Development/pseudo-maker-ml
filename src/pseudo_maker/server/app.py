"""FastAPI web UI for pseudo-maker.

Runs on 127.0.0.1 only. Sessions are held in memory and written to per-session
temp directories. Uploaded files and generated outputs live there until the
session is reaped (on explicit cleanup or after SESSION_TTL_SECONDS).
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import orchestrate

SUPPORTED_EXTS = {".docx", ".txt", ".md"}
SESSION_TTL_SECONDS = 5 * 60          # 5 minutes (was 30)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024   # 25 MB per file


def _static_root() -> Path:
    # Works both in dev (src/pseudo_maker/server/static) and when PyInstaller
    # extracts data files to _MEIPASS (via --add-data).
    if hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS) / "pseudo_maker" / "server" / "static"
        if candidate.exists():
            return candidate
    return Path(__file__).parent / "static"


STATIC_ROOT = _static_root()


def _stem(name: str) -> str:
    return Path(name).stem


def _suffix(name: str) -> str:
    return Path(name).suffix


class SessionStore:
    """In-memory session registry with per-session temp dirs."""

    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self) -> tuple[str, Path]:
        sid = uuid.uuid4().hex
        work = Path(tempfile.mkdtemp(prefix=f"mcosmos_{sid}_"))
        self._sessions[sid] = {"id": sid, "work": work, "created": time.time(), "data": {}}
        return sid, work

    def get(self, sid: str) -> dict[str, Any]:
        s = self._sessions.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail="Unknown session")
        return s

    def put_data(self, sid: str, key: str, value: Any) -> None:
        self.get(sid)["data"][key] = value

    def get_data(self, sid: str, key: str) -> Any:
        return self.get(sid)["data"].get(key)

    def reap(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s["created"] > SESSION_TTL_SECONDS]
        for sid in expired:
            s = self._sessions.pop(sid)
            try:
                _rm_rf(s["work"])
            except Exception:
                pass


def _rm_rf(path: Path) -> None:
    if not path.exists():
        return
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for child in path.iterdir():
        _rm_rf(child)
    path.rmdir()


# Types a user can pick when adding a term by hand. CUSTOM → opaque placeholder.
ADDABLE_TYPES = {
    "PERSON", "COMPANY", "CITY", "STREET", "EMAIL", "PHONE", "ID_NUMBER",
    "PERSONNUMMER", "ORGNR", "DATE", "CASE_NUMBER", "CUSTOM",
}


def create_app() -> FastAPI:
    app = FastAPI(title="pseudo-maker", docs_url=None, redoc_url=None)
    store = SessionStore()

    if STATIC_ROOT.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        index_path = STATIC_ROOT / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=500, detail="index.html missing")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    async def _save_upload(file: UploadFile, dest: Path) -> None:
        written = 0
        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
                    )
                out.write(chunk)

    @app.post("/api/clean/upload")
    async def upload(file: UploadFile = File(...)):
        ext = _suffix(file.filename or "").lower()
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        sid, work = store.create()
        input_path = work / (file.filename or f"input{ext}")
        await _save_upload(file, input_path)

        session, _entities = await asyncio.to_thread(orchestrate.preview, input_path)
        store.put_data(sid, "session", session)
        store.put_data(sid, "original_stem", _stem(file.filename or "input"))
        store.put_data(sid, "original_ext", ext)

        return JSONResponse({
            "session_id": sid,
            "filename": file.filename,
            **_preview_state(session),
        })

    def _preview_state(session: orchestrate.CleanSession) -> dict:
        return {
            "entities": [e.as_dict() for e in orchestrate.entities(session)],
            "candidates": orchestrate.candidates(session),
            "document": orchestrate.document_view(session),
        }

    @app.post("/api/clean/add-term")
    async def add_term(
        session_id: str = Form(...),
        term: str = Form(...),
        type: str = Form("PERSON"),
    ):
        """Replace every occurrence of a user-supplied term (or an accepted candidate)."""
        s = store.get(session_id)
        session: orchestrate.CleanSession = s["data"]["session"]
        if type.upper() not in ADDABLE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported type: {type}")
        try:
            added = await asyncio.to_thread(orchestrate.add_term, session, term, type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse({"added": added, **_preview_state(session)})

    @app.post("/api/clean/set-fake")
    async def set_fake(
        session_id: str = Form(...),
        entity_id: str = Form(...),
        fake: str = Form(...),
    ):
        """Change an entity's fake now, so the document view shows it everywhere."""
        s = store.get(session_id)
        session: orchestrate.CleanSession = s["data"]["session"]
        try:
            orchestrate.set_fake(session, entity_id, fake)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown entity id: {entity_id}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return JSONResponse(_preview_state(session))

    @app.post("/api/clean/finalize")
    async def finalize(
        session_id: str = Form(...),
        excluded_ids: str = Form(""),  # comma-separated
        custom_fakes: str = Form(""),  # JSON object: {entity_id: new_fake}
    ):
        s = store.get(session_id)
        session: orchestrate.CleanSession = s["data"]["session"]
        excluded = {x.strip() for x in excluded_ids.split(",") if x.strip()}

        if custom_fakes.strip():
            try:
                overrides = json.loads(custom_fakes)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"custom_fakes must be JSON: {e}")
            if not isinstance(overrides, dict):
                raise HTTPException(status_code=400, detail="custom_fakes must be an object")
            for entity_id, new_fake in overrides.items():
                if entity_id in excluded:
                    continue
                if not isinstance(new_fake, str) or not new_fake.strip():
                    raise HTTPException(status_code=400, detail=f"Empty fake for {entity_id}")
                try:
                    session.mapper.override_fake(entity_id, new_fake)
                except KeyError:
                    raise HTTPException(status_code=400, detail=f"Unknown entity id: {entity_id}")

        stem = s["data"]["original_stem"]
        ext = s["data"]["original_ext"]
        cleaned_name = f"{stem}_CLEANED{ext}"
        mapping_name = f"{stem}_MAPPING.json"
        out_path = s["work"] / cleaned_name
        mapping_path = s["work"] / mapping_name

        await asyncio.to_thread(
            orchestrate.finalize,
            session,
            out_path,
            mapping_path,
            excluded_entity_ids=excluded,
        )
        return JSONResponse({
            "cleaned_url": f"/api/download/{session_id}/cleaned",
            "mapping_url": f"/api/download/{session_id}/mapping",
            "cleaned_name": cleaned_name,
            "mapping_name": mapping_name,
            "kept_entity_count": len([e for e in orchestrate.entities(session) if e.id not in excluded]),
        })

    @app.get("/api/download/{session_id}/{kind}")
    async def download(session_id: str, kind: str):
        s = store.get(session_id)
        work: Path = s["work"]
        stem = s["data"].get("original_stem", "output")
        ext = s["data"].get("original_ext", ".txt")
        if kind == "cleaned":
            path = work / f"{stem}_CLEANED{ext}"
            filename = path.name
        elif kind == "mapping":
            path = work / f"{stem}_MAPPING.json"
            filename = path.name
        elif kind == "restored":
            path = work / f"{stem}_RESTORED{ext}"
            filename = path.name
        elif kind == "audit":
            path = work / f"{stem}_AUDIT.json"
            filename = path.name
        else:
            raise HTTPException(status_code=400, detail=f"Unknown download kind: {kind}")
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not ready: {filename}")
        return FileResponse(path, filename=filename)

    @app.post("/api/restore")
    async def restore(
        document: UploadFile = File(...),
        mapping: UploadFile = File(...),
        fuzzy_threshold: int = Form(92),
    ):
        ext = _suffix(document.filename or "").lower()
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        if _suffix(mapping.filename or "").lower() != ".json":
            raise HTTPException(status_code=400, detail="Mapping must be .json")

        sid, work = store.create()
        doc_path = work / (document.filename or f"llm_output{ext}")
        map_path = work / (mapping.filename or "mapping.json")
        for upload_file, dest in ((document, doc_path), (mapping, map_path)):
            await _save_upload(upload_file, dest)

        # Derive the stem from the mapping's source_file if present.
        try:
            payload = json.loads(map_path.read_text(encoding="utf-8"))
            source = payload.get("source_file") or document.filename or "output"
        except Exception:
            source = document.filename or "output"
        stem = _stem(source)
        store.put_data(sid, "original_stem", stem)
        store.put_data(sid, "original_ext", ext)
        restored_name = f"{stem}_RESTORED{ext}"
        audit_name = f"{stem}_AUDIT.json"
        restored_path = work / restored_name
        audit_path = work / audit_name

        if ext == ".docx":
            report = await asyncio.to_thread(
                orchestrate.restore_docx,
                doc_path, restored_path, map_path,
                fuzzy_threshold=fuzzy_threshold,
            )
        else:
            report = await asyncio.to_thread(
                orchestrate.restore_text_file,
                doc_path, restored_path, map_path,
                fuzzy_threshold=fuzzy_threshold,
            )
        orchestrate.write_audit_log(
            report,
            audit_path,
            restored_document=restored_name,
            mapping_file=mapping.filename or "mapping.json",
        )
        return JSONResponse({
            "session_id": sid,
            "restored_url": f"/api/download/{sid}/restored",
            "restored_name": restored_name,
            "audit_url": f"/api/download/{sid}/audit",
            "audit_name": audit_name,
            "report": {
                "exact": report.exact,
                "fuzzy": report.fuzzy,
                "missing": report.missing,
            },
        })

    @app.post("/api/session/{session_id}/close")
    async def close_session(session_id: str):
        s = store.get(session_id)
        _rm_rf(s["work"])
        store._sessions.pop(session_id, None)
        return {"ok": True}

    @app.get("/api/health")
    async def health():
        store.reap()
        return {"ok": True, "sessions": len(store._sessions)}

    return app


def run(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True) -> None:
    import uvicorn
    import threading
    import webbrowser

    app = create_app()
    url = f"http://{host}:{port}"
    if open_browser:
        # Fire after a short delay so the server has time to bind.
        def _open():
            time.sleep(0.8)
            webbrowser.open(url)
        threading.Thread(target=_open, daemon=True).start()
    print(f"Pseudo Maker UI running at {url}   (Ctrl+C to stop)")
    # Access log ON (INFO) for operational visibility in container deployments.
    # Session IDs appear in URLs; sessions expire within minutes, so logging
    # them is acceptable for small-team LAN-only deployments.
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=True)
