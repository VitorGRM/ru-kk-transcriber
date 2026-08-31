"""FastAPI application: upload → transcribe → edit → export.

Everything runs in this process on the local machine; nothing is uploaded
anywhere. Long jobs run on a worker thread and stream progress over SSE.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import cuda_setup, formats
from .audio import load_audio, probe_duration
from .config import (BUNDLE_DIR, DEFAULT_COMPUTE_TYPE, DEFAULT_DEVICE,
                     DEFAULT_LOCAL_MODEL, LANGUAGE_NAMES, MAX_UPLOAD_MB, MODEL_DIR,
                     UPLOAD_DIR)
from .engines.local_whisper import MODEL_CHOICES, ROUTING_MODES, CodeSwitchTranscriber
from .runtime import describe

app = FastAPI(title="RU/KK Mixed-Speech Transcriber")
STATIC_DIR = BUNDLE_DIR / "app" / "static"


# --------------------------------------------------------------------------
# Job registry
# --------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    filename: str
    path: Path
    duration: float = 0.0
    status: str = "idle"          # idle | running | done | error
    error: str = ""
    result: dict | None = None
    audio: object = None          # cached waveform, reused for manual re-decodes
    events: queue.Queue = field(default_factory=queue.Queue)
    settings: dict = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0


JOBS: dict[str, Job] = {}
_ENGINE: CodeSwitchTranscriber | None = None
_ENGINE_KEY: tuple | None = None
_ENGINE_LOCK = threading.Lock()
_WORK_LOCK = threading.Lock()     # one transcription at a time; the GPU is not shared


def get_engine(model: str, device: str, compute: str) -> CodeSwitchTranscriber:
    """Return a loaded engine, swapping models only when the request changes."""
    global _ENGINE, _ENGINE_KEY
    key = (model, device, compute)
    with _ENGINE_LOCK:
        if _ENGINE_KEY != key:
            if _ENGINE is not None:
                _ENGINE.unload()          # free VRAM before loading another model
            _ENGINE = CodeSwitchTranscriber(
                model_name=model, device=device, compute_type=compute,
                download_root=str(MODEL_DIR),
            )
            _ENGINE_KEY = key
        return _ENGINE


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/system")
def system_info() -> dict:
    hw = describe()
    cached = sorted(p.name for p in MODEL_DIR.glob("models--*")) if MODEL_DIR.exists() else []
    return {
        "hardware": hw,
        "models": MODEL_CHOICES,
        "routing_modes": ROUTING_MODES,
        "languages": LANGUAGE_NAMES,
        "defaults": {
            "model": DEFAULT_LOCAL_MODEL,
            "device": DEFAULT_DEVICE,
            "compute_type": DEFAULT_COMPUTE_TYPE,
        },
        "cached_models": cached,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename).suffix or ".bin"
    dest = UPLOAD_DIR / f"{job_id}{suffix}"

    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
            fh.write(chunk)

    duration = probe_duration(dest)
    JOBS[job_id] = Job(id=job_id, filename=file.filename, path=dest, duration=duration)
    return {"job_id": job_id, "filename": file.filename, "duration": duration,
            "size_bytes": size}


@app.post("/api/transcribe")
def start_transcription(
    job_id: str = Form(...),
    model: str = Form(DEFAULT_LOCAL_MODEL),
    device: str = Form(DEFAULT_DEVICE),
    compute_type: str = Form(DEFAULT_COMPUTE_TYPE),
    routing_mode: str = Form("balanced"),
    beam_size: int = Form(5),
    hotwords: str = Form(""),
    vad_aggressiveness: float = Form(0.5),
    boundary_gap_ms: int = Form(600),
    use_context: bool = Form(True),
) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    if job.status == "running":
        raise HTTPException(409, "This job is already running.")

    job.settings = {
        "model": model, "device": device, "compute_type": compute_type,
        "routing_mode": routing_mode, "beam_size": max(1, min(10, beam_size)),
        "hotwords": hotwords.strip(), "vad_aggressiveness": vad_aggressiveness,
        "boundary_gap_ms": max(150, min(2000, boundary_gap_ms)),
        "use_context": use_context,
    }
    job.status = "running"
    job.error = ""
    job.result = None
    job.started_at = time.time()
    job.events = queue.Queue()

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


def _run_job(job: Job) -> None:
    def emit(event: dict) -> None:
        job.events.put(event)

    try:
        with _WORK_LOCK:
            s = job.settings
            emit({"stage": "audio", "message": "Decoding audio…"})
            if job.audio is None:
                job.audio = load_audio(job.path)
            job.duration = len(job.audio) / 16_000

            engine = get_engine(s["model"], s["device"], s["compute_type"])
            emit({"stage": "load",
                  "message": f"Preparing {s['model']} — the first run downloads the "
                             "model, which can take a few minutes."})

            transcript = engine.transcribe(
                job.audio,
                mode=s["routing_mode"],
                beam_size=s["beam_size"],
                hotwords=s["hotwords"],
                vad_aggressiveness=s["vad_aggressiveness"],
                boundary_gap_s=s["boundary_gap_ms"] / 1000.0,
                use_context=s["use_context"],
                progress=emit,
            )

        payload = transcript.to_dict()
        payload["filename"] = job.filename
        payload["settings"] = job.settings
        job.result = payload
        job.status = "done"
        job.finished_at = time.time()
        payload["elapsed"] = round(job.finished_at - job.started_at, 1)
        emit({"stage": "done", "result": payload})

    except Exception as exc:                       # surfaced to the user verbatim
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        emit({"stage": "error", "message": job.error})


@app.get("/api/progress/{job_id}")
async def progress(job_id: str) -> StreamingResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")

    async def stream():
        loop = asyncio.get_running_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, job.events.get, True, 30)
            except queue.Empty:
                yield ": keep-alive\n\n"
                if job.status in ("done", "error"):
                    break
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("stage") in ("done", "error"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/result/{job_id}")
def result(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return {"status": job.status, "error": job.error, "result": job.result,
            "duration": job.duration, "filename": job.filename}


@app.post("/api/redecode")
def redecode(job_id: str = Form(...), start: float = Form(...), end: float = Form(...),
             language: str = Form(...), beam_size: int = Form(5),
             hotwords: str = Form("")) -> dict:
    """Re-transcribe one span under a language chosen by hand in the UI."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    if language not in LANGUAGE_NAMES:
        raise HTTPException(400, f"Language must be one of {list(LANGUAGE_NAMES)}.")

    with _WORK_LOCK:
        if job.audio is None:
            job.audio = load_audio(job.path)
        s = job.settings or {}
        engine = get_engine(s.get("model", DEFAULT_LOCAL_MODEL),
                            s.get("device", DEFAULT_DEVICE),
                            s.get("compute_type", DEFAULT_COMPUTE_TYPE))
        segments = engine.redecode(job.audio, start, end, language,
                                   beam_size=beam_size, hotwords=hotwords)
    return {"segments": [s.to_dict() for s in segments]}


@app.post("/api/export")
async def export(job_id: str = Form(...), fmt: str = Form("txt"),
                 segments: str = Form("")) -> Response:
    """Render an export. `segments` carries the user's edits back from the browser."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    if fmt not in formats.EXPORTERS:
        raise HTTPException(400, f"Unknown format: {fmt}")

    payload = dict(job.result or {"segments": []})
    if segments:
        try:
            payload["segments"] = json.loads(segments)
        except json.JSONDecodeError:
            raise HTTPException(400, "Edited segments were not valid JSON.")

    body = formats.render(fmt, payload)
    mime, ext = formats.EXPORTERS[fmt]
    stem = Path(job.filename).stem or "transcript"
    return Response(
        content=body.encode("utf-8"), media_type=mime,
        headers={"Content-Disposition": f'attachment; filename="{stem}.{ext}"'},
    )


@app.get("/api/gpu/status")
def gpu_status() -> dict:
    return cuda_setup.status()


_GPU_INSTALL = {"running": False, "events": None, "error": "", "done": False}


@app.post("/api/gpu/install")
def gpu_install() -> dict:
    """Fetch the CUDA runtime so the packaged app can use the GPU."""
    if _GPU_INSTALL["running"]:
        raise HTTPException(409, "The GPU runtime is already being installed.")
    if not cuda_setup.supported():
        raise HTTPException(
            400,
            "The automatic installer is Windows-only. On Linux or macOS run "
            "./setup.sh --gpu instead.",
        )

    _GPU_INSTALL.update(running=True, events=queue.Queue(), error="", done=False)

    def worker() -> None:
        try:
            cuda_setup.install(lambda e: _GPU_INSTALL["events"].put(e))
            _GPU_INSTALL["done"] = True
            # A model already loaded on the CPU has to go, so the next run picks
            # the GPU up.
            global _ENGINE, _ENGINE_KEY
            with _ENGINE_LOCK:
                if _ENGINE is not None:
                    _ENGINE.unload()
                _ENGINE, _ENGINE_KEY = None, None
        except Exception as exc:
            _GPU_INSTALL["error"] = f"{type(exc).__name__}: {exc}"
            _GPU_INSTALL["events"].put({"stage": "error", "message": _GPU_INSTALL["error"]})
        finally:
            _GPU_INSTALL["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "running"}


@app.get("/api/gpu/progress")
async def gpu_progress() -> StreamingResponse:
    events = _GPU_INSTALL["events"]
    if events is None:
        raise HTTPException(404, "No GPU installation has been started.")

    async def stream():
        loop = asyncio.get_running_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, events.get, True, 30)
            except queue.Empty:
                yield ": keep-alive\n\n"
                if not _GPU_INSTALL["running"]:
                    break
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("stage") in ("done", "error"):
                break

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/audio/{job_id}")
def audio_file(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None or not job.path.exists():
        raise HTTPException(404, "Unknown job.")
    return FileResponse(job.path, filename=job.filename)


@app.delete("/api/job/{job_id}")
def delete_job(job_id: str) -> dict:
    job = JOBS.pop(job_id, None)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    job.path.unlink(missing_ok=True)
    return {"deleted": job_id}


# If a previous session already downloaded the CUDA runtime, put it on the DLL
# search path now — it has to be there before CTranslate2 opens a GPU model.
cuda_setup.enable_dll_path()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _free_port(host: str, preferred: int, attempts: int = 20) -> int:
    """Return `preferred` if it is free, otherwise the next port that is.

    Port 8000 is a popular default, so refusing to start when something else
    already holds it would be needlessly obstructive.
    """
    import socket

    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(
        f"No free port between {preferred} and {preferred + attempts - 1}. "
        f"Set TA_PORT to choose one."
    )


def main() -> None:
    import uvicorn

    from .config import HOST, PORT

    port = _free_port(HOST, PORT)
    if port != PORT:
        print(f"\n  Port {PORT} is in use — starting on {port} instead.")
    print(f"\n  RU/KK Mixed-Speech Transcriber → http://{HOST}:{port}\n")
    uvicorn.run(app, host=HOST, port=port, log_level="info")


if __name__ == "__main__":
    main()
