"""
Example FastAPI application that combines a Teachable Machine image classifier
with an OpenAI GPT model to generate rich responses.

Required dependencies:
    pip install fastapi uvicorn tensorflow pillow openai python-multipart
"""
from __future__ import annotations

import atexit
import base64
import contextlib
import html
import io
import json
import logging
import os
import re
import secrets
import socket
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
import requests

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - only triggered when TF missing
    raise RuntimeError(
        "TensorFlow is required for this application. Install it with 'pip install tensorflow'."
    ) from exc

from openai import OpenAI

try:  # pragma: no cover - optional dependency for camera capture
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when OpenCV missing
    cv2 = None

try:  # pragma: no cover - optional dependency for network mode
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:  # pragma: no cover - handled at runtime when needed
    ServiceInfo = None  # type: ignore
    Zeroconf = None  # type: ignore

MODEL_PATH = os.getenv("TEACHABLE_MODEL_PATH", "./models/teachable_model")
DEFAULT_CLASS_NAMES: List[str] = ["class_1", "class_2", "class_3"]
GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4.1-mini")
BASE_DIR = Path(__file__).resolve().parent
SIMPLE_UI_PATH = BASE_DIR / "static" / "index.html"
CONFIG_UI_PATH = BASE_DIR / "static" / "config.html"
COMPLETIONS_UI_PATH = BASE_DIR / "static" / "completions.html"
SHARE_UI_PATH = BASE_DIR / "static" / "share.html"
TM_MODELS_DIR = BASE_DIR / "TM-models"
TM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TM_REGISTRY_PATH = TM_MODELS_DIR / "registry.json"
REQUIRED_TM_FILES = {"metadata.json", "model.json", "weights.bin"}
NETWORK_CONFIG_FILENAME = "network-config.json"
SETTINGS_PATH = BASE_DIR / "app-settings.json"
DOCS_DIR = BASE_DIR / "doc"
SHARE_STORE_DIR = BASE_DIR / "share-store"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
SHARE_STORE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_TYPE_ALIASES = {
    "trichome": "trichome",
    "trichomen": "trichome",
    "trichomen analyse": "trichome",
    "trichome analysis": "trichome",
    "trichomes": "trichome",
    "health": "health",
    "healthcare": "health",
    "gesundheit": "health",
}
NETWORK_DEFAULT_CONFIG = {
    "enabled": False,
    "hostname": "ottcolab.local",
    "port": 8000,
    "ip": None,
    "url": None,
}
STREAM_CAPTURE_INTERVAL_DEFAULT = 5.0
STREAM_BATCH_INTERVAL_DEFAULT = 30.0
STREAM_BUFFER_MAX = 24

_model_cache: Dict[str, tf.keras.Model] = {}
_client: OpenAI | None = None
_network_zeroconf: Zeroconf | None = None
_network_service: ServiceInfo | None = None
_network_runtime_config: Dict[str, Any] = {}
_app_settings: Dict[str, Any] = {}
_stream_lock = threading.Lock()

LOG_LEVEL = os.getenv("OTTC_LOG_LEVEL", "INFO").upper()
UPSTREAM_REPO_URL = os.getenv("OTTC_REPO_URL", "https://github.com/methoxy000/ottcops.git")
UPSTREAM_REPO_BRANCH = os.getenv("OTTC_REPO_BRANCH", "main")


def _build_logger() -> logging.Logger:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level)
    else:
        logging.getLogger().setLevel(level)
    return logging.getLogger("ottcouture.app")


logger = _build_logger()


@dataclass
class StreamJob:
    stream_id: str
    label: str
    source_url: str
    source_type: str
    analysis_mode: str
    prompt: str
    model_id: Optional[str]
    capture_interval: float = STREAM_CAPTURE_INTERVAL_DEFAULT
    batch_interval: float = STREAM_BATCH_INTERVAL_DEFAULT
    running: bool = False
    frames: List[bytes] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    last_capture_ts: float = 0.0  # epoch seconds for UI
    last_batch_ts: float = 0.0  # epoch seconds for UI
    last_capture_perf: float = 0.0  # perf counter for scheduling
    last_batch_perf: float = 0.0
    thread: Optional[threading.Thread] = None
    model_entry: Optional[Dict[str, Any]] = None
    video_capture: Any = None


class StreamManager:
    """Manage background capture jobs for snapshot/video sources."""

    def __init__(self) -> None:
        self.jobs: Dict[str, StreamJob] = {}
        self.lock = threading.Lock()

    def start_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stream_id = payload.get("stream_id") or slugify(payload.get("label") or "stream")
        with self.lock:
            if stream_id in self.jobs:
                raise HTTPException(status_code=400, detail="Stream-ID bereits vergeben.")
            job = StreamJob(
                stream_id=stream_id,
                label=payload.get("label") or stream_id,
                source_url=payload["source_url"],
                source_type=payload.get("source_type", "snapshot"),
                analysis_mode=payload.get("analysis_mode", "hybrid"),
                prompt=payload.get("prompt", ""),
                model_id=payload.get("model_id"),
                capture_interval=float(payload.get("capture_interval", STREAM_CAPTURE_INTERVAL_DEFAULT)),
                batch_interval=float(payload.get("batch_interval", STREAM_BATCH_INTERVAL_DEFAULT)),
            )
            job.model_entry = resolve_model_entry(job.model_id)
            job.running = True
            now_perf = time.perf_counter()
            now_epoch = time.time()
            job.last_batch_perf = now_perf
            job.last_capture_perf = now_perf
            job.last_batch_ts = now_epoch
            job.last_capture_ts = now_epoch
            if job.source_type == "video" and cv2 is None:
                raise HTTPException(status_code=400, detail="OpenCV fehlt für Videoquellen.")
            if job.source_type == "video" and cv2 is not None:
                job.video_capture = cv2.VideoCapture(job.source_url)
            thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
            job.thread = thread
            self.jobs[stream_id] = job
        thread.start()
        return self.serialize_job(job)

    def stop_job(self, stream_id: str) -> None:
        with self.lock:
            job = self.jobs.get(stream_id)
            if not job:
                raise HTTPException(status_code=404, detail="Stream nicht gefunden.")
            job.running = False
            if job.video_capture is not None and cv2 is not None:
                job.video_capture.release()
            self.jobs.pop(stream_id, None)

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [self.serialize_job(job) for job in self.jobs.values()]

    def get_job(self, stream_id: str) -> StreamJob:
        with self.lock:
            job = self.jobs.get(stream_id)
            if not job:
                raise HTTPException(status_code=404, detail="Stream nicht gefunden.")
            return job

    def trigger_now(self, stream_id: str) -> Dict[str, Any]:
        job = self.get_job(stream_id)
        self._process_batch(job)
        return self.serialize_job(job)

    def stop_all(self) -> None:
        with self.lock:
            ids = list(self.jobs.keys())
        for stream_id in ids:
            with contextlib.suppress(Exception):
                self.stop_job(stream_id)

    def serialize_job(self, job: StreamJob) -> Dict[str, Any]:
        try:
            model_meta = build_teachable_meta(job.model_entry or resolve_model_entry(job.model_id))
        except HTTPException:
            model_meta = {
                "id": job.model_id or "unknown",
                "name": "Unbekannt",
                "type": "n/a",
                "source": "registry",
            }
        return {
            "stream_id": job.stream_id,
            "label": job.label,
            "source_url": job.source_url,
            "source_type": job.source_type,
            "analysis_mode": job.analysis_mode,
            "capture_interval": job.capture_interval,
            "batch_interval": job.batch_interval,
            "last_capture_ts": job.last_capture_ts,
            "last_batch_ts": job.last_batch_ts,
            "running": job.running,
            "last_error": job.last_error,
            "last_result": job.last_result,
            "model": model_meta,
        }

    def _run_job(self, job: StreamJob) -> None:  # pragma: no cover - background thread
        while job.running:
            now_perf = time.perf_counter()
            try:
                if now_perf - job.last_capture_perf >= job.capture_interval:
                    frame = self._capture_frame(job)
                    if frame:
                        job.frames.append(frame)
                        if len(job.frames) > STREAM_BUFFER_MAX:
                            job.frames = job.frames[-STREAM_BUFFER_MAX:]
                        job.last_capture_perf = now_perf
                        job.last_capture_ts = time.time()
                if job.frames and now_perf - job.last_batch_perf >= job.batch_interval:
                    self._process_batch(job)
                    job.last_batch_perf = now_perf
                    job.last_batch_ts = time.time()
                time.sleep(0.5)
            except Exception as exc:
                job.last_error = str(exc)
                logger.exception("Stream %s Fehler: %s", job.stream_id, exc)
                time.sleep(2)

    def _capture_frame(self, job: StreamJob) -> Optional[bytes]:
        if job.source_type == "snapshot":
            response = requests.get(job.source_url, timeout=15)
            if response.status_code != 200:
                raise RuntimeError(f"Snapshot HTTP {response.status_code}")
            return response.content
        if job.source_type == "video":
            if cv2 is None:
                raise RuntimeError("OpenCV nicht verfügbar")
            if job.video_capture is None:
                job.video_capture = cv2.VideoCapture(job.source_url)
            ret, frame = job.video_capture.read()
            if not ret:
                job.video_capture.release()
                job.video_capture = cv2.VideoCapture(job.source_url)
                ret, frame = job.video_capture.read()
                if not ret:
                    raise RuntimeError("Konnte Videoframe nicht lesen")
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError("Frame konnte nicht encodiert werden")
            return encoded.tobytes()
        raise RuntimeError(f"Unbekannter source_type {job.source_type}")

    def _process_batch(self, job: StreamJob) -> None:
        if not job.frames:
            return
        frames = job.frames[:]
        job.frames.clear()
        model_entry = job.model_entry or resolve_model_entry(job.model_id)
        items: List[Dict[str, Any]] = []
        total_model_ms = 0.0
        total_llm_ms = 0.0
        for idx, frame in enumerate(frames, start=1):
            if job.analysis_mode == "ml":
                classification, timings = perform_ml_only(frame, model_entry)
                payload = build_ml_payload(classification, model_entry, timings)
                total_model_ms += timings["model_ms"]
                items.append({"image_id": f"{job.stream_id}-{idx}", "analysis": payload})
            else:
                prompt = job.prompt or "Automatisierter Stream-Report"
                classification, gpt_response, timings = perform_analysis(prompt, frame, model_entry)
                payload = build_analysis_payload(classification, gpt_response, model_entry, timings)
                total_model_ms += timings["model_ms"]
                total_llm_ms += timings["llm_ms"]
                items.append({"image_id": f"{job.stream_id}-{idx}", "analysis": payload})
        if not items:
            return
        summary_text = (
            summarize_ml_items(items)
            if job.analysis_mode == "ml"
            else summarize_batch(job.prompt or "Stream-Auswertung", items)
        )
        job.last_result = {
            "status": "ok",
            "summary": {"text": summary_text},
            "items": items,
            "analysis_mode": job.analysis_mode,
            "teachable_model": build_teachable_meta(model_entry),
            "debug": {
                "timings": {
                    "model_ms": round(total_model_ms, 2),
                    "llm_ms": round(total_llm_ms, 2),
                    "total_ms": round(total_model_ms + total_llm_ms, 2),
                }
            },
            "captured": datetime.utcnow().isoformat() + "Z",
        }


stream_manager = StreamManager()
atexit.register(stream_manager.stop_all)


def _run_git_command(args: Sequence[str]) -> tuple[int, str, str]:
    """Execute a git command and return (returncode, stdout, stderr)."""

    try:
        proc = subprocess.run(
            args,
            cwd=str(BASE_DIR),
            capture_output=True,
            check=False,
            text=True,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 1, "", "git executable not available"


def _get_local_commit_hash() -> Optional[str]:
    code, stdout, stderr = _run_git_command(["git", "rev-parse", "HEAD"])
    if code == 0:
        return stdout
    logger.debug("Konnte lokalen Commit nicht bestimmen: %s", stderr)
    return None


def _get_remote_commit_hash() -> Optional[str]:
    code, stdout, stderr = _run_git_command(
        ["git", "ls-remote", UPSTREAM_REPO_URL, UPSTREAM_REPO_BRANCH]
    )
    if code == 0 and stdout:
        return stdout.split()[0]
    logger.debug("Konnte Remote-Commit nicht bestimmen: %s", stderr)
    return None


def _prompt_user_for_update(remote_hash: str) -> bool:
    """Ask the operator whether an update should be pulled."""

    prompt = (
        f"Eine neue Version ({remote_hash[:7]}) ist verfügbar. Jetzt aktualisieren? [y/N]: "
    )
    try:
        response = input(prompt)
    except EOFError:
        logger.info("Keine interaktive Eingabe verfügbar – Update wird übersprungen.")
        return False
    return response.strip().lower() in {"y", "yes", "j", "ja"}


def _perform_git_update() -> None:
    logger.info("Starte git pull von %s (%s)...", UPSTREAM_REPO_URL, UPSTREAM_REPO_BRANCH)
    code, stdout, stderr = _run_git_command(
        ["git", "pull", UPSTREAM_REPO_URL, UPSTREAM_REPO_BRANCH]
    )
    if code != 0:
        logger.error("git pull fehlgeschlagen: %s", stderr or stdout)
    else:
        logger.info("Repository erfolgreich aktualisiert: %s", stdout)


def ensure_latest_code_checked_out() -> None:
    """Check GitHub for new commits and prompt for update before app start."""

    if os.getenv("OTTC_SKIP_UPDATE_CHECK", "0") in {"1", "true", "True"}:
        logger.info("Update-Check übersprungen (OTTC_SKIP_UPDATE_CHECK=1).")
        return

    local_hash = _get_local_commit_hash()
    remote_hash = _get_remote_commit_hash()

    if not local_hash or not remote_hash:
        logger.info("Update-Check konnte nicht abgeschlossen werden (fehlende Git-Infos).")
        return

    if local_hash == remote_hash:
        logger.info("OPENCORE Analyzer ist bereits auf dem neuesten Stand (%s).", local_hash[:7])
        return

    logger.info(
        "Neue Version erkannt. Lokal %s, Remote %s.", local_hash[:7], remote_hash[:7]
    )
    if _prompt_user_for_update(remote_hash):
        _perform_git_update()
    else:
        logger.info("Update abgelehnt – aktuelle Version bleibt aktiv.")


ensure_latest_code_checked_out()

app = FastAPI(
    title="Teachable Machine + GPT Analyzer",
    description=(
        "Upload an image plus a user prompt and combine the Teachable Machine "
        "classification with a GPT response. Interactive Swagger UI is available "
        "at /docs."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.mount("/doc", StaticFiles(directory=DOCS_DIR, html=True), name="doc")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class NetworkPayload(BaseModel):
    hostname: str
    port: int = 8000


class CompletionPayload(BaseModel):
    prompt: str


class SharePayload(BaseModel):
    payload: Dict[str, Any]


class StreamCreatePayload(BaseModel):
    label: Optional[str] = None
    source_url: str
    source_type: str = "snapshot"
    analysis_mode: str = "hybrid"
    prompt: Optional[str] = ""
    model_id: Optional[str] = None
    capture_interval: float = STREAM_CAPTURE_INTERVAL_DEFAULT
    batch_interval: float = STREAM_BATCH_INTERVAL_DEFAULT


def load_app_settings() -> Dict[str, Any]:
    if SETTINGS_PATH.is_file():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return {"default_model_id": data.get("default_model_id")}
        except json.JSONDecodeError:
            logger.warning("app-settings.json konnte nicht geparst werden, fallback auf Defaults.")
    return {"default_model_id": None}


def save_app_settings(data: Dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_network_config_path() -> Path:
    return BASE_DIR / NETWORK_CONFIG_FILENAME


def load_network_config_from_disk() -> Dict[str, Any]:
    config_path = get_network_config_path()
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            merged = {**NETWORK_DEFAULT_CONFIG, **data}
            return merged
        except json.JSONDecodeError:
            logger.warning("network-config.json konnte nicht geparst werden, fallback auf Defaults.")
    return dict(NETWORK_DEFAULT_CONFIG)


def save_network_config(data: Dict[str, Any]) -> None:
    config_path = get_network_config_path()
    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_bool_flag(value: Optional[str]) -> bool:
    """Normalize typical truthy inputs coming from form data."""

    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def generate_request_id() -> str:
    """Return a short unique identifier for correlating logs."""

    return uuid.uuid4().hex


def measure_elapsed_ms(start: float) -> float:
    """Return milliseconds between now and the provided timestamp."""

    return round((time.perf_counter() - start) * 1000, 2)


def build_teachable_meta(model_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a metadata block shared between responses."""

    return {
        "id": model_entry.get("id"),
        "name": model_entry.get("name"),
        "type": model_entry.get("type"),
        "source": model_entry.get("source"),
    }


def build_debug_payload(
    request_id: str,
    model_entry: Dict[str, Any],
    prompt: str,
    timings: Dict[str, float],
    batch_items: int = 1,
    debug_enabled: bool = False,
    error: str | None = None,
) -> Dict[str, Any]:
    """Compose a structured debug payload for the frontend panel."""

    payload = {
        "request_id": request_id,
        "model_version": f"{model_entry.get('source')}:{model_entry.get('id')}",
        "model_name": model_entry.get("name"),
        "model_type": model_entry.get("type"),
        "prompt_preview": prompt[:140],
        "timings": timings,
        "batch_items": batch_items,
        "debug_enabled": debug_enabled,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if error:
        payload["error"] = error
    return payload


def build_analysis_payload(
    classification: Dict[str, Any],
    gpt_response: str,
    model_entry: Dict[str, Any],
    timings: Dict[str, float],
) -> Dict[str, Any]:
    """Bundle classification, LLM output and metadata for a single asset."""

    return {
        "classification": classification,
        "gpt_response": gpt_response,
        "meta": GPTMeta(model=GPT_MODEL, success=True).dict(),
        "teachable_model": build_teachable_meta(model_entry),
        "timings": timings,
    }


def build_ml_payload(
    classification: Dict[str, Any], model_entry: Dict[str, Any], timings: Dict[str, float]
) -> Dict[str, Any]:
    """Return a payload describing only the Teachable Machine inference."""

    return {
        "classification": classification,
        "teachable_model": build_teachable_meta(model_entry),
        "timings": timings,
    }


def perform_analysis(
    prompt: str, image_bytes: bytes, model_entry: Dict[str, Any]
) -> tuple[Dict[str, Any], str, Dict[str, float]]:
    """Execute the TM classification and GPT call with timing data."""

    total_start = time.perf_counter()
    model_start = time.perf_counter()
    classification = classify_image(image_bytes, model_entry)
    model_ms = measure_elapsed_ms(model_start)
    llm_start = time.perf_counter()
    gpt_response = call_gpt_with_image_context(prompt, classification, image_bytes)
    llm_ms = measure_elapsed_ms(llm_start)
    timings = {
        "model_ms": model_ms,
        "llm_ms": llm_ms,
        "total_ms": measure_elapsed_ms(total_start),
    }
    return classification, gpt_response, timings


def perform_ml_only(image_bytes: bytes, model_entry: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Execute only the Teachable Machine inference."""

    total_start = time.perf_counter()
    classification = classify_image(image_bytes, model_entry)
    model_ms = measure_elapsed_ms(total_start)
    timings = {"model_ms": model_ms, "llm_ms": 0.0, "total_ms": model_ms}
    return classification, timings


def summarize_batch(prompt: str, items: List[Dict[str, Any]]) -> str:
    """Request a concise batch summary based on individual GPT responses."""

    context_lines = []
    for idx, item in enumerate(items, start=1):
        snippet = item["analysis"].get("gpt_response", "")
        context_lines.append(f"Bild {idx} ({item['image_id']}): {snippet}")
    client = get_openai_client()
    user_content = (
        "Original Prompt: "
        f"{prompt}\n" "Einzelresultate:\n" + "\n".join(context_lines)
        + "\nFormuliere eine strukturierte Zusammenfassung mit Hauptbefunden und Empfehlungen."
    )
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Erstelle präzise Reports zu Cannabis-Bildanalysen. Keine Medizin-Claims, nur Qualitätsbewertung.",
                },
                {"role": "user", "content": user_content},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - depends on provider
        logger.warning("Batch summary konnte nicht erstellt werden: %s", exc)
        fallback = " | ".join(context_lines)
        return f"Zusammenfassung (Fallback): {fallback[:1000]}"


def summarize_ml_items(items: List[Dict[str, Any]]) -> str:
    """Create a short summary purely from classification payloads."""

    summaries = []
    for idx, item in enumerate(items, start=1):
        classification = item.get("analysis", {}).get("classification")
        if not classification:
            continue
        label = classification.get("top_label") or "n/a"
        confidence = classification.get("top_confidence")
        if isinstance(confidence, (int, float)):
            conf_display = f"{confidence:.2%}"
        else:
            conf_display = str(confidence)
        summaries.append(f"Bild {idx}: {label} ({conf_display}).")
    if not summaries:
        return "Keine ML-Befunde verfügbar."
    joined = " ".join(summaries)
    return f"ML-Report: {joined}"


def get_share_file_path(share_id: str) -> Path:
    """Map a share ID to the on-disk JSON file."""

    if not re.fullmatch(r"[A-Za-z0-9_-]+", share_id):
        raise HTTPException(status_code=400, detail="Ungültige Share-ID.")
    return SHARE_STORE_DIR / f"{share_id}.json"


def save_share_payload(payload: Dict[str, Any]) -> str:
    """Persist the payload and return a newly generated share ID."""

    share_id = secrets.token_urlsafe(8)
    path = SHARE_STORE_DIR / f"{share_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return share_id


def load_share_payload_from_disk(share_id: str) -> Dict[str, Any]:
    """Load a stored payload by ID."""

    path = get_share_file_path(share_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Share-Link nicht gefunden.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupted files
        raise HTTPException(status_code=500, detail="Share-Daten beschädigt.") from exc


def load_share_ui_template(share_id: str) -> str:
    """Inject the share ID into the static HTML viewer."""

    if SHARE_UI_PATH.is_file():
        html_raw = SHARE_UI_PATH.read_text(encoding="utf-8")
    else:
        html_raw = "<html><body><h1>Share Viewer fehlt</h1></body></html>"
    return html_raw.replace("{{SHARE_ID}}", html.escape(share_id))


def determine_local_ip() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def sanitize_hostname(raw: str) -> str:
    if not raw:
        raise ValueError("Hostname darf nicht leer sein.")
    slug = re.sub(r"[^a-z0-9-]+", "-", raw.strip().lower()).strip("-")
    if not slug:
        slug = "ottcolab"
    if not slug.endswith(".local"):
        slug = f"{slug}.local"
    return slug


def require_zeroconf() -> None:
    if Zeroconf is None or ServiceInfo is None:
        raise RuntimeError(
            "Für den WiFi Broadcast muss das Python-Paket 'zeroconf' installiert sein. "
            "Installiere es via 'pip install zeroconf'."
        )


def deactivate_network_broadcast() -> None:
    global _network_zeroconf, _network_service
    if _network_zeroconf and _network_service:
        with contextlib.suppress(Exception):
            _network_zeroconf.unregister_service(_network_service)
        with contextlib.suppress(Exception):
            _network_zeroconf.close()
    _network_zeroconf = None
    _network_service = None


def activate_network_broadcast(hostname: str, port: int) -> Dict[str, Any]:
    require_zeroconf()
    if not (1 <= port <= 65535):
        raise ValueError("Port muss zwischen 1 und 65535 liegen.")
    sanitized = sanitize_hostname(hostname)
    ip_address = determine_local_ip()
    address_bytes = socket.inet_aton(ip_address)

    deactivate_network_broadcast()
    zeroconf_instance = Zeroconf()
    service_name = sanitized.replace(".local", "")
    service_info = ServiceInfo(
        "_http._tcp.local.",
        f"{service_name}._http._tcp.local.",
        addresses=[address_bytes],
        port=port,
        properties={"path": "/", "brand": "ottcouture.eu"},
    )
    zeroconf_instance.register_service(service_info)

    global _network_zeroconf, _network_service
    _network_zeroconf = zeroconf_instance
    _network_service = service_info

    return {
        "enabled": True,
        "hostname": sanitized,
        "port": port,
        "ip": ip_address,
        "url": f"http://{sanitized}:{port}",
    }


def get_network_status() -> Dict[str, Any]:
    return {
        "enabled": _network_runtime_config.get("enabled", False),
        "hostname": _network_runtime_config.get("hostname", NETWORK_DEFAULT_CONFIG["hostname"]),
        "port": _network_runtime_config.get("port", NETWORK_DEFAULT_CONFIG["port"]),
        "ip": _network_runtime_config.get("ip"),
        "url": _network_runtime_config.get("url"),
    }


def enable_network_mode(hostname: str, port: int) -> Dict[str, Any]:
    state = activate_network_broadcast(hostname, port)
    _network_runtime_config.update(state)
    save_network_config(_network_runtime_config)
    return get_network_status()


def disable_network_mode() -> Dict[str, Any]:
    deactivate_network_broadcast()
    _network_runtime_config.update(
        {"enabled": False, "ip": None, "url": None}
    )
    save_network_config(_network_runtime_config)
    return get_network_status()


_network_runtime_config = load_network_config_from_disk()
_app_settings = load_app_settings()

class GPTMeta(BaseModel):
    model: str
    success: bool


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "tm-model"


def normalize_model_type(raw: str) -> str:
    if not raw:
        raise HTTPException(status_code=400, detail="Modeltyp fehlt.")
    value = raw.strip().lower()
    normalized = MODEL_TYPE_ALIASES.get(value, value)
    if normalized not in {"trichome", "health"}:
        raise HTTPException(status_code=400, detail="Ungültiger Modeltyp.")
    return normalized


def load_tm_registry() -> List[Dict[str, Any]]:
    if TM_REGISTRY_PATH.is_file():
        try:
            return json.loads(TM_REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
    return []


def save_tm_registry(entries: List[Dict[str, Any]]) -> None:
    TM_REGISTRY_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


def build_unique_model_dir(slug: str) -> Path:
    candidate = TM_MODELS_DIR / slug
    counter = 1
    while candidate.exists():
        candidate = TM_MODELS_DIR / f"{slug}-{counter}"
        counter += 1
    return candidate


def find_model_root(temp_root: Path) -> Path:
    if all((temp_root / required).is_file() for required in REQUIRED_TM_FILES):
        return temp_root
    for metadata_file in temp_root.rglob("metadata.json"):
        candidate = metadata_file.parent
        if all((candidate / required).is_file() for required in REQUIRED_TM_FILES):
            return candidate
    raise HTTPException(status_code=400, detail="ZIP enthält kein gültiges Teachable Machine Modell.")


def list_tm_models() -> List[Dict[str, Any]]:
    default_id = _app_settings.get("default_model_id")
    enriched: List[Dict[str, Any]] = []
    for entry in load_tm_registry():
        enriched_entry = dict(entry)
        enriched_entry.setdefault("type", "trichome")
        enriched_entry["is_default"] = enriched_entry.get("id") == default_id
        enriched.append(enriched_entry)
    return enriched


def find_tm_entry(model_id: str) -> Optional[Dict[str, Any]]:
    for entry in load_tm_registry():
        if entry.get("id") == model_id:
            return entry
    return None


def set_default_tm_model(model_id: str | None) -> Optional[Dict[str, Any]]:
    if model_id is None:
        _app_settings["default_model_id"] = None
        save_app_settings(_app_settings)
        return None
    entry = find_tm_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Modell wurde nicht gefunden.")
    _app_settings["default_model_id"] = model_id
    save_app_settings(_app_settings)
    return entry


def load_tf_model(model_path: Path) -> tf.keras.Model:
    """Load and cache a Teachable Machine TensorFlow model."""
    resolved = str(model_path.resolve())
    model = _model_cache.get(resolved)
    if model is None:
        if not model_path.is_dir():
            raise RuntimeError(f"Model directory '{model_path}' not found.")
        model = tf.keras.models.load_model(model_path)
        _model_cache[resolved] = model
    return model


def get_openai_client() -> OpenAI:
    """Create and cache the OpenAI client using the API key from environment variables."""
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable must be set.")
        _client = OpenAI()
    return _client


def preprocess_image(image_bytes: bytes, input_shape: Sequence[int]) -> np.ndarray:
    """Resize and normalize the input image for the classifier."""
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream).convert("RGB")
    if len(input_shape) < 3:
        raise HTTPException(status_code=500, detail="Unexpected model input shape.")
    height, width = int(input_shape[1]), int(input_shape[2])
    image = image.resize((width, height))
    image_array = np.asarray(image).astype("float32") / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    return image_array


def resolve_model_entry(model_id: str | None = None) -> Dict[str, Any]:
    registry = load_tm_registry()
    candidate: Optional[Dict[str, Any]] = None
    force_builtin = model_id == "builtin"
    if model_id and not force_builtin:
        candidate = next((entry for entry in registry if entry.get("id") == model_id), None)
        if candidate is None:
            raise HTTPException(status_code=404, detail="Unbekanntes Teachable Machine Modell.")
    elif not force_builtin:
        default_id = _app_settings.get("default_model_id")
        if default_id:
            candidate = next((entry for entry in registry if entry.get("id") == default_id), None)
    if candidate is None and registry and not force_builtin:
        candidate = registry[0]

    if candidate:
        model_path = (BASE_DIR / candidate["path"]).resolve()
        if not model_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Der Modellordner '{candidate['path']}' fehlt. Bitte das ZIP erneut hochladen oder den Pfad korrigieren."
                ),
            )
        labels = candidate.get("metadata", {}).get("labels") if isinstance(candidate.get("metadata"), dict) else None
        if not isinstance(labels, list) or not labels:
            labels = DEFAULT_CLASS_NAMES
        else:
            labels = [str(label) for label in labels]
        return {
            "id": candidate.get("id"),
            "name": candidate.get("name", "TM Modell"),
            "type": candidate.get("type", "trichome"),
            "path": model_path,
            "labels": labels,
            "source": "registry",
        }

    model_path = Path(MODEL_PATH)
    if not model_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                "Kein Teachable Machine Modell gefunden. Bitte ein ZIP im Config-Bereich hochladen oder TEACHABLE_MODEL_PATH setzen."
            ),
        )
    return {
        "id": "builtin",
        "name": "OPENCORE Referenz",
        "type": "trichome",
        "path": model_path.resolve(),
        "labels": DEFAULT_CLASS_NAMES,
        "source": "builtin",
    }


def classify_image(image_bytes: bytes, model_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Teachable Machine model on the provided image bytes."""
    try:
        model = load_tf_model(model_entry["path"])
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    input_shape = model.input_shape
    if isinstance(input_shape, list):  # Some TF models have list of input shapes
        input_shape = input_shape[0]
    preprocessed = preprocess_image(image_bytes, input_shape)
    predictions = model.predict(preprocessed, verbose=0)[0]
    predictions = predictions.tolist()

    class_names = list(model_entry.get("labels", DEFAULT_CLASS_NAMES))
    if len(class_names) != len(predictions):
        adjusted = class_names[: len(predictions)]
        while len(adjusted) < len(predictions):
            adjusted.append(f"class_{len(adjusted) + 1}")
        class_names = adjusted

    labelled_predictions = [
        {"label": label, "confidence": float(conf)}
        for label, conf in zip(class_names, predictions)
    ]
    labelled_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    top_prediction = labelled_predictions[0]

    return {
        "top_label": top_prediction["label"],
        "top_confidence": top_prediction["confidence"],
        "all_predictions": labelled_predictions,
    }


def call_gpt_with_image_context(
    user_prompt: str, classification: Dict[str, Any], image_bytes: bytes
) -> str:
    """Send the classification context plus the user's prompt (and image) to GPT."""
    client = get_openai_client()
    distribution = ", ".join(
        f"{pred['label']}: {pred['confidence']:.2%}"
        for pred in classification["all_predictions"]
    )
    classification_summary = (
        f"Top label: {classification['top_label']} "
        f"(confidence: {classification['top_confidence']:.2%}). "
        f"Full distribution: {distribution}."
    )

    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_payload = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
    }

    messages = [
        {
            "role": "system",
            "content": "You are an assistant that combines Teachable Machine image classification "
            "results with user prompts to produce helpful insights.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Classification result from Teachable Machine: "
                        f"{classification_summary}\nUser prompt: {user_prompt}\n"
                        "Please analyze and provide a combined answer."
                    ),
                },
                image_payload,
            ],
        },
    ]

    try:
        response = client.chat.completions.create(model=GPT_MODEL, messages=messages)
    except Exception as exc:  # pragma: no cover - external service errors
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc

    return response.choices[0].message.content.strip()


def load_simple_ui() -> str:
    """Return the simple HTML UI for manual testing."""
    if SIMPLE_UI_PATH.is_file():
        return SIMPLE_UI_PATH.read_text(encoding="utf-8")
    return "<html><body><h1>Simple UI missing</h1><p>Please build static/index.html.</p></body></html>"


def load_config_ui() -> str:
    """Return the HTML for the configuration interface."""
    if CONFIG_UI_PATH.is_file():
        return CONFIG_UI_PATH.read_text(encoding="utf-8")
    return "<html><body><h1>Config UI missing</h1><p>Please build static/config.html.</p></body></html>"


def load_completions_ui() -> str:
    if COMPLETIONS_UI_PATH.is_file():
        return COMPLETIONS_UI_PATH.read_text(encoding="utf-8")
    return "<html><body><h1>Completions UI missing</h1></body></html>"


@app.get("/share/{share_id}", response_class=HTMLResponse, include_in_schema=False)
async def share_viewer(share_id: str) -> HTMLResponse:
    """Return the static share viewer with injected ID."""

    return HTMLResponse(load_share_ui_template(share_id))


@app.on_event("startup")
async def startup_network_mode() -> None:  # pragma: no cover - depends on runtime env
    if _network_runtime_config.get("enabled"):
        try:
            state = activate_network_broadcast(
                _network_runtime_config.get("hostname", NETWORK_DEFAULT_CONFIG["hostname"]),
                int(_network_runtime_config.get("port", NETWORK_DEFAULT_CONFIG["port"])),
            )
            _network_runtime_config.update(state)
        except Exception as exc:  # noqa: BLE001 - log and continue
            logger.warning("mDNS Broadcast konnte nicht aktiviert werden: %s", exc)


@app.on_event("shutdown")
async def shutdown_network_mode() -> None:  # pragma: no cover - depends on runtime env
    deactivate_network_broadcast()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_ui() -> HTMLResponse:
    """Serve a very small HTML helper to try the API without Swagger."""
    return HTMLResponse(load_simple_ui())


@app.get("/config", response_class=HTMLResponse, include_in_schema=False)
async def config_ui() -> HTMLResponse:
    """Serve a configuration helper for selecting local/self-hosted LLMs."""
    return HTMLResponse(load_config_ui())


@app.get("/completions", response_class=HTMLResponse, include_in_schema=False)
async def completions_ui() -> HTMLResponse:
    return HTMLResponse(load_completions_ui())


@app.get("/network/status", response_class=JSONResponse)
async def network_status() -> JSONResponse:
    """Expose the current mDNS broadcast status for the UI."""
    return JSONResponse({"status": get_network_status(), "mdns_available": Zeroconf is not None})


@app.post("/network/announce", response_class=JSONResponse)
async def network_announce(payload: NetworkPayload) -> JSONResponse:
    """Enable or refresh the WiFi broadcast hostname."""

    try:
        status = enable_network_mode(payload.hostname, payload.port)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"status": status, "mdns_available": Zeroconf is not None})


@app.delete("/network/announce", response_class=JSONResponse)
async def network_announce_stop() -> JSONResponse:
    """Disable the WiFi broadcast hostname."""

    status = disable_network_mode()
    return JSONResponse({"status": status, "mdns_available": Zeroconf is not None})


@app.get("/tm-models", response_class=JSONResponse)
async def tm_models() -> JSONResponse:
    """Return the registered Teachable Machine models."""
    return JSONResponse(
        {
            "models": list_tm_models(),
            "default_model_id": _app_settings.get("default_model_id"),
            "has_builtin": Path(MODEL_PATH).is_dir(),
        }
    )


@app.post("/tm-models/default/{model_id}", response_class=JSONResponse)
async def tm_models_set_default(model_id: str) -> JSONResponse:
    entry = set_default_tm_model(model_id)
    return JSONResponse({"default_model_id": _app_settings.get("default_model_id"), "model": entry})


@app.delete("/tm-models/default", response_class=JSONResponse)
async def tm_models_clear_default() -> JSONResponse:
    set_default_tm_model(None)
    return JSONResponse({"default_model_id": None})


@app.post("/tm-models/upload")
async def upload_tm_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    display_name: str = Form(...),
) -> JSONResponse:
    """Persist a zipped Teachable Machine export under TM-models."""

    normalized_type = normalize_model_type(model_type)
    if file.content_type not in {"application/zip", "application/x-zip-compressed", "multipart/form-data", "application/octet-stream"}:
        # Some browsers mislabel the upload, therefore we only warn if it's obviously not a zip.
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Bitte eine ZIP-Datei hochladen.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Leere Datei erhalten.")

    if not display_name or not display_name.strip():
        display_name = file.filename.rsplit(".", 1)[0] or "TM Modell"

    metadata_data: Dict[str, Any] | None = None
    target_dir: Path | None = None

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive, tempfile.TemporaryDirectory() as tmp_dir:
            archive.extractall(tmp_dir)
            temp_root = Path(tmp_dir)
            content_root = find_model_root(temp_root)
            missing = [req for req in REQUIRED_TM_FILES if not (content_root / req).is_file()]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Datei unvollständig. Folgende Bestandteile fehlen: {', '.join(missing)}.",
                )

            metadata_raw = (content_root / "metadata.json").read_text(encoding="utf-8")
            try:
                metadata_data = json.loads(metadata_raw)
            except json.JSONDecodeError as exc:  # pragma: no cover - depends on uploads
                raise HTTPException(status_code=400, detail="metadata.json ist nicht gültig JSON.") from exc
            slug = slugify(display_name)
            target_dir = build_unique_model_dir(slug)
            shutil.copytree(content_root, target_dir)
    except zipfile.BadZipFile as exc:  # pragma: no cover - depends on user uploads
        raise HTTPException(status_code=400, detail="Ungültige ZIP-Datei.") from exc

    if not metadata_data or target_dir is None:
        raise HTTPException(status_code=500, detail="Modell konnte nicht gespeichert werden.")

    registry = load_tm_registry()
    entry = {
        "id": target_dir.name,
        "name": display_name,
        "type": normalized_type,
        "path": str(target_dir.relative_to(BASE_DIR)),
        "metadata": metadata_data,
        "added": datetime.utcnow().isoformat() + "Z",
    }
    registry.append(entry)
    save_tm_registry(registry)

    if _app_settings.get("default_model_id") is None:
        _app_settings["default_model_id"] = entry["id"]
        save_app_settings(_app_settings)

    return JSONResponse({"message": "Modell gespeichert", "model": entry, "default_model_id": _app_settings.get("default_model_id")})


@app.post("/api/opencore/analyze-batch", response_class=JSONResponse)
async def analyze_batch(
    prompt: str = Form(default=""),
    model_id: Optional[str] = Form(default=None),
    debug: Optional[str] = Form(default=None),
    analysis_mode: str = Form(default="hybrid"),
    files: List[UploadFile] = File(..., alias="files[]"),
) -> JSONResponse:
    normalized_mode = (analysis_mode or "hybrid").lower()
    if normalized_mode not in {"hybrid", "ml"}:
        raise HTTPException(status_code=400, detail="Ungültiger Analysemodus.")
    if normalized_mode != "ml" and not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt ist erforderlich.")
    if not files:
        raise HTTPException(status_code=400, detail="Mindestens eine Datei hochladen.")

    request_id = generate_request_id()
    debug_flag = parse_bool_flag(debug)
    model_entry = resolve_model_entry(model_id)
    items: List[Dict[str, Any]] = []
    for idx, file in enumerate(files, start=1):
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Leere Datei übermittelt.")
        if normalized_mode == "ml":
            classification, timings = perform_ml_only(data, model_entry)
            analysis_payload = build_ml_payload(classification, model_entry, timings)
        else:
            classification, gpt_response, timings = perform_analysis(prompt, data, model_entry)
            analysis_payload = build_analysis_payload(classification, gpt_response, model_entry, timings)
        items.append(
            {
                "image_id": file.filename or f"image-{idx}",
                "analysis": analysis_payload,
            }
        )

    summary_text = (
        summarize_ml_items(items) if normalized_mode == "ml" else summarize_batch(prompt, items)
    )
    aggregated_timings = {
        "model_ms": round(sum(item["analysis"]["timings"]["model_ms"] for item in items), 2),
        "llm_ms": round(
            sum(item["analysis"]["timings"].get("llm_ms", 0.0) for item in items),
            2,
        ),
        "total_ms": round(sum(item["analysis"]["timings"]["total_ms"] for item in items), 2),
    }
    response_payload = {
        "status": "ok",
        "summary": {"text": summary_text},
        "items": items,
        "teachable_model": build_teachable_meta(model_entry),
        "analysis_mode": normalized_mode,
        "debug": build_debug_payload(
            request_id,
            model_entry,
            prompt if normalized_mode != "ml" else "ML-only Batch",
            aggregated_timings,
            batch_items=len(items),
            debug_enabled=debug_flag,
        ),
    }
    return JSONResponse(response_payload)


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    model_id: Optional[str] = Form(default=None),
    debug: Optional[str] = Form(default=None),
    analysis_mode: str = Form(default="hybrid"),
) -> JSONResponse:
    """Endpoint that classifies an image and optionally calls GPT."""
    if image is None:
        raise HTTPException(status_code=400, detail="Image file is required.")
    normalized_mode = (analysis_mode or "hybrid").lower()
    if normalized_mode not in {"hybrid", "ml"}:
        raise HTTPException(status_code=400, detail="Ungültiger Analysemodus.")
    if normalized_mode != "ml" and not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    request_id = generate_request_id()
    debug_flag = parse_bool_flag(debug)
    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc

    model_entry = resolve_model_entry(model_id)
    if normalized_mode == "ml":
        classification, timings = perform_ml_only(image_bytes, model_entry)
        analysis_payload = build_ml_payload(classification, model_entry, timings)
        debug_payload = build_debug_payload(
            request_id,
            model_entry,
            prompt or "ML-only",
            timings,
            batch_items=1,
            debug_enabled=debug_flag,
        )
    else:
        classification, gpt_response, timings = perform_analysis(prompt, image_bytes, model_entry)
        analysis_payload = build_analysis_payload(classification, gpt_response, model_entry, timings)
        debug_payload = build_debug_payload(
            request_id,
            model_entry,
            prompt,
            timings,
            batch_items=1,
            debug_enabled=debug_flag,
        )
    return JSONResponse(content={**analysis_payload, "analysis_mode": normalized_mode, "debug": debug_payload})


@app.post("/api/opencore/analyze-ml", response_class=JSONResponse)
async def analyze_ml_proxy(
    image: UploadFile = File(...),
    model_id: Optional[str] = Form(default=None),
    debug: Optional[str] = Form(default=None),
) -> JSONResponse:
    """Dedicated alias for ML-only calls."""

    return await analyze(image=image, prompt="", model_id=model_id, debug=debug, analysis_mode="ml")


@app.post("/api/opencore/share", response_class=JSONResponse)
async def create_share(payload: SharePayload) -> JSONResponse:
    if not isinstance(payload.payload, dict) or not payload.payload:
        raise HTTPException(status_code=400, detail="Share-Payload fehlt.")
    share_id = save_share_payload(payload.payload)
    return JSONResponse({"share_id": share_id, "url": f"/share/{share_id}"})


@app.get("/api/opencore/share/{share_id}", response_class=JSONResponse)
async def load_share(share_id: str) -> JSONResponse:
    data = load_share_payload_from_disk(share_id)
    return JSONResponse({"share_id": share_id, "payload": data})


@app.get("/api/opencore/streams", response_class=JSONResponse)
async def list_streams() -> JSONResponse:
    return JSONResponse({"streams": stream_manager.list_jobs()})


@app.post("/api/opencore/streams", response_class=JSONResponse)
async def create_stream(payload: StreamCreatePayload) -> JSONResponse:
    source_type = (payload.source_type or "snapshot").lower()
    analysis_mode = (payload.analysis_mode or "hybrid").lower()
    if source_type not in {"snapshot", "video"}:
        raise HTTPException(status_code=400, detail="source_type muss snapshot oder video sein.")
    if analysis_mode not in {"hybrid", "ml"}:
        raise HTTPException(status_code=400, detail="analysis_mode muss hybrid oder ml sein.")
    if analysis_mode != "ml" and not (payload.prompt or "").strip():
        raise HTTPException(status_code=400, detail="Prompt erforderlich für Hybrid-Modus.")
    if payload.capture_interval < 1 or payload.batch_interval < 5:
        raise HTTPException(status_code=400, detail="Intervalle zu klein.")
    job = stream_manager.start_job({**payload.dict(), "source_type": source_type, "analysis_mode": analysis_mode})
    return JSONResponse({"stream": job})


@app.get("/api/opencore/streams/{stream_id}", response_class=JSONResponse)
async def read_stream(stream_id: str) -> JSONResponse:
    job = stream_manager.get_job(stream_id)
    return JSONResponse({"stream": stream_manager.serialize_job(job)})


@app.delete("/api/opencore/streams/{stream_id}", response_class=JSONResponse)
async def delete_stream(stream_id: str) -> JSONResponse:
    stream_manager.stop_job(stream_id)
    return JSONResponse({"stream_id": stream_id, "status": "stopped"})


@app.post("/api/opencore/streams/{stream_id}/trigger", response_class=JSONResponse)
async def trigger_stream(stream_id: str) -> JSONResponse:
    data = stream_manager.trigger_now(stream_id)
    return JSONResponse({"stream": data})


OTTO_SYSTEM_PROMPT = (
    "Du bist OTTO, der Cultivation-Chatbot von ottcouture.eu. Beantworte Fragen zu Grow-Setups, "
    "Klima, Genetik und Betriebssicherheit sachlich, strukturiert und ohne medizinische Aussagen. "
    "Liefere klare Handlungsschritte und fasse Werte präzise zusammen."
)


@app.post("/api/completions", response_class=JSONResponse)
async def otto_completion(payload: CompletionPayload) -> JSONResponse:
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt darf nicht leer sein.")
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": OTTO_SYSTEM_PROMPT},
                {"role": "user", "content": payload.prompt.strip()},
            ],
        )
    except Exception as exc:  # pragma: no cover - external service errors
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {exc}") from exc
    answer = response.choices[0].message.content.strip()
    return JSONResponse({"response": answer, "model": GPT_MODEL})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
