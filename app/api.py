"""REST service (Table 5.4). Start with `veriscript serve` or `uvicorn app.api:app`.

Environment
  VERISCRIPT_CONFIG       path to a YAML config (default configs/default.yaml)
  VERISCRIPT_ADMIN_TOKEN  enables PUT /api/v1/config for requests with header X-Admin-Token
  VERISCRIPT_DEVICE       auto | cpu | cuda
"""

from __future__ import annotations

import csv
import io
import os
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from veriscript import __version__
from veriscript.cascade import CascadeDetector, summarize
from veriscript.config import load_config
from veriscript.io import SUPPORTED_SUFFIXES, ExtractionError, extract_bytes
from veriscript.result import DetectionResult

app = FastAPI(title="VeriScript", version=__version__,
              description="Hybrid heuristic-transformer detector of AI-generated text. "
                          "Results are probabilistic estimates and must be reviewed by a person.")

_state: dict = {"detector": None, "error": None}
_lock = threading.Lock()  # the models are shared; serialise inference
_jobs: dict[str, dict] = {}
MAX_BATCH = 5000


def detector() -> CascadeDetector:
    if _state["detector"] is None:
        with _lock:
            if _state["detector"] is None:
                try:
                    cfg = load_config(os.environ.get("VERISCRIPT_CONFIG"))
                    _state["detector"] = CascadeDetector.from_config(cfg, device=os.environ.get("VERISCRIPT_DEVICE"))
                except Exception as exc:  # report through /health
                    _state["error"] = str(exc)
                    raise HTTPException(503, f"models not available: {exc}")
    return _state["detector"]


class DetectRequest(BaseModel):
    text: str = Field(..., description="document text")
    mode: Optional[Literal["fast", "balanced", "thorough"]] = Field(None, description="operating point for this request")


class ConfigUpdate(BaseModel):
    operating_point: Optional[Literal["fast", "balanced", "thorough"]] = None
    t_low: Optional[float] = Field(None, ge=0, le=1)
    t_high: Optional[float] = Field(None, ge=0, le=1)


def _run(text: str, mode: Optional[str], doc_id=None) -> DetectionResult:
    det = detector()
    with _lock:
        prev = (det.gate, det.operating_point)
        try:
            if mode:
                det.set_operating_point(mode)
            return det.detect(text, doc_id)
        finally:
            det.gate, det.operating_point = prev


@app.post("/api/v1/detect")
def detect(req: DetectRequest):
    """Detect a single document; returns label, probability, route, latency and explanation."""
    return _run(req.text, req.mode).to_json()


@app.post("/api/v1/detect/file")
async def detect_file(file: UploadFile = File(...), mode: Optional[str] = None):
    """Detect a single uploaded .txt, .pdf or .docx file."""
    try:
        text = extract_bytes(await file.read(), file.filename or "upload.txt")
    except ExtractionError as exc:
        raise HTTPException(422, str(exc))
    return _run(text, mode, file.filename).to_json()


def _parse_batch(data: bytes, filename: str) -> list[tuple[str, Optional[str], Optional[str]]]:
    name = filename.lower()
    if name.endswith(".zip"):
        out = []
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                p = Path(info.filename)
                if info.is_dir() or p.suffix.lower() not in SUPPORTED_SUFFIXES or p.name.startswith("."):
                    continue
                try:
                    out.append((p.name, extract_bytes(z.read(info), p.name), None))
                except ExtractionError as exc:
                    out.append((p.name, None, str(exc)))
        return out
    if name.endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        if rows and "text" not in rows[0]:
            raise HTTPException(422, "CSV needs a 'text' column (and optionally 'id')")
        return [(r.get("id") or str(i), r["text"], None) for i, r in enumerate(rows)]
    raise HTTPException(422, "batch must be a .csv (id,text) or .zip of documents")


def _process(job_id: str, docs, mode: Optional[str]):
    job = _jobs[job_id]
    job["status"] = "running"
    try:
        det = detector()
        with _lock:
            prev = (det.gate, det.operating_point)
            try:
                if mode:
                    det.set_operating_point(mode)

                def prog(k, n):
                    job["progress"] = k / max(n, 1)

                res = det.detect_batch([(i, t) for i, t, e in docs if e is None], progress=prog)
            finally:
                det.gate, det.operating_point = prev
        res += [DetectionResult("rejected", None, None, {}, None, 0, i, note=e) for i, t, e in docs if e is not None]
        job["summary"] = summarize(res)
        job["results"] = [r.to_json() for r in res]
        job["status"] = "completed"
        job["progress"] = 1.0
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)


@app.post("/api/v1/batch")
async def submit_batch(background: BackgroundTasks, file: UploadFile = File(...), mode: Optional[str] = None):
    """Submit a batch (CSV with id,text columns, or ZIP of .txt/.pdf/.docx); returns a job id."""
    docs = _parse_batch(await file.read(), file.filename or "")
    if not docs:
        raise HTTPException(422, "no documents found")
    if len(docs) > MAX_BATCH:
        raise HTTPException(413, f"at most {MAX_BATCH} documents per batch")
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"id": job_id, "status": "queued", "n_documents": len(docs), "progress": 0.0}
    background.add_task(_process, job_id, docs, mode)
    return {"job_id": job_id, "status": "queued", "n_documents": len(docs)}


@app.get("/api/v1/batch/{job_id}")
def batch_status(job_id: str):
    """Job status and, when complete, the report."""
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@app.get("/api/v1/config")
def get_config():
    """Current operating point and thresholds."""
    return detector().config_json()


@app.put("/api/v1/config")
def put_config(upd: ConfigUpdate, x_admin_token: Optional[str] = Header(None)):
    """Change the operating point or set custom thresholds (administrator only)."""
    token = os.environ.get("VERISCRIPT_ADMIN_TOKEN")
    if not token or x_admin_token != token:
        raise HTTPException(403, "administrator token required")
    det = detector()
    with _lock:
        if upd.operating_point:
            det.set_operating_point(upd.operating_point)
        elif upd.t_low is not None and upd.t_high is not None:
            try:
                det.set_thresholds(upd.t_low, upd.t_high)
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        else:
            raise HTTPException(422, "give operating_point or both t_low and t_high")
    return det.config_json()


@app.get("/api/v1/health")
def health():
    """Service and model status."""
    det = _state["detector"]
    if det is None and _state["error"] is None:
        try:
            det = detector()
        except HTTPException:
            pass
    return {
        "status": "ok" if det is not None else "degraded",
        "version": __version__,
        "models_loaded": det is not None,
        "transformer_available": bool(det and det.trf and det.trf.available),
        "device": det.heu.extractor.device if det and det.heu else None,
        "error": _state["error"],
    }
