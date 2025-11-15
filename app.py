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
import sys
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
from pydantic import BaseModel, Field
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
TFJS_REQUIRED_FILES = {"metadata.json", "model.json", "weights.bin"}
TFJS_REQUIRED_FILES_LOWER = {name.lower() for name in TFJS_REQUIRED_FILES}
KERAS_LABEL_FILE = "labels.txt"
KERAS_SUFFIXES = {".keras", ".h5", ".hdf5"}
KERAS_CUSTOM_OBJECTS: dict[str, Any] | None = None
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
DEFAULT_LLM_CONFIG = {
    "provider": "openai",
    "apiBase": "",
    "model": "",
    "apiKey": "",
    "vision": "yes",
    "systemPrompt": "",
}
LLM_ALLOWED_PROVIDERS = {"openai", "ollama", "lmstudio"}
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_HTTP_TIMEOUT = int(os.getenv("OTTC_LLM_TIMEOUT", "90"))
ANALYZER_SYSTEM_PROMPT = (
    "Du wertest Cannabisbilder für ottcouture.eu aus. Kombiniere Teachable-Machine-"
    "Klassifikationen mit dem Nutzerprompt, beschreibe Risiken und empfiehl klare,"
    "nicht-medizinische Maßnahmen."
)

_model_cache: Dict[str, Any] = {}
_client: OpenAI | None = None
_client_signature: tuple[str | None, str | None] | None = None
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


def get_keras_custom_objects() -> dict[str, Any]:
    """Return custom layer patches required for legacy Keras exports."""

    global KERAS_CUSTOM_OBJECTS
    if KERAS_CUSTOM_OBJECTS is None:

        class DepthwiseConv2DCompat(tf.keras.layers.DepthwiseConv2D):
            """DepthwiseConv2D that ignores the legacy 'groups' argument."""

            def __init__(self, *args: Any, groups: int | None = None, **kwargs: Any) -> None:
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        KERAS_CUSTOM_OBJECTS = {"DepthwiseConv2D": DepthwiseConv2DCompat}
    return KERAS_CUSTOM_OBJECTS


@dataclass
class StreamJob:
    stream_id: str
    label: str
    source_url: str
    source_type: str
    analysis_mode: str
    prompt: str
    model_id: Optional[str]
    llm_profile_id: Optional[str] = None
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
                llm_profile_id=payload.get("llm_profile_id") or None,
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
            "llm_profile_id": job.llm_profile_id,
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
        resolved_profile = job.llm_profile_id or get_active_llm_profile_id()
        llm_config = get_llm_config(resolved_profile)
        for idx, frame in enumerate(frames, start=1):
            if job.analysis_mode == "ml":
                classification, timings = perform_ml_only(frame, model_entry)
                payload = build_ml_payload(classification, model_entry, timings)
                total_model_ms += timings["model_ms"]
                items.append({"image_id": f"{job.stream_id}-{idx}", "analysis": payload})
            else:
                prompt = job.prompt or "Automatisierter Stream-Report"
                classification, gpt_response, timings = perform_analysis(prompt, frame, model_entry, llm_config)
                payload = build_analysis_payload(classification, gpt_response, model_entry, timings, llm_config)
                total_model_ms += timings["model_ms"]
                total_llm_ms += timings["llm_ms"]
                items.append({"image_id": f"{job.stream_id}-{idx}", "analysis": payload})
        if not items:
            return
        summary_text = (
            summarize_ml_items(items)
            if job.analysis_mode == "ml"
            else summarize_batch(job.prompt or "Stream-Auswertung", items, llm_config)
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
                },
                "llm_profile_id": resolved_profile,
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
    llm_profile_id: Optional[str] = None


class SharePayload(BaseModel):
    payload: Dict[str, Any]


class StreamCreatePayload(BaseModel):
    label: Optional[str] = None
    source_url: str
    source_type: str = "snapshot"
    analysis_mode: str = "hybrid"
    prompt: Optional[str] = ""
    model_id: Optional[str] = None
    llm_profile_id: Optional[str] = None
    capture_interval: float = STREAM_CAPTURE_INTERVAL_DEFAULT
    batch_interval: float = STREAM_BATCH_INTERVAL_DEFAULT


class LLMConfigPayload(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)
    profile_id: Optional[str] = None
    profile_name: Optional[str] = None
    make_active: bool = True


class LLMProfilePayload(BaseModel):
    profile_id: Optional[str] = None
    name: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    activate: bool = True


def normalize_llm_config(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize provider settings and merge them with defaults."""

    config = DEFAULT_LLM_CONFIG.copy()
    if isinstance(payload, dict):
        for key in config:
            if key not in payload:
                continue
            value = payload[key]
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
            config[key] = value

    provider = str(config.get("provider") or DEFAULT_LLM_CONFIG["provider"]).lower()
    if provider not in LLM_ALLOWED_PROVIDERS:
        provider = DEFAULT_LLM_CONFIG["provider"]
    config["provider"] = provider

    vision_raw = str(config.get("vision") or DEFAULT_LLM_CONFIG["vision"]).lower()
    config["vision"] = "manual" if vision_raw == "manual" else DEFAULT_LLM_CONFIG["vision"]

    for text_key in ("apiBase", "model", "apiKey", "systemPrompt"):
        config[text_key] = config.get(text_key) or ""

    return config


def normalize_llm_profile(entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ensure a profile dict always contains id, name and normalized config."""

    if not isinstance(entry, dict):
        return None
    config = normalize_llm_config(entry.get("config"))
    profile_id = str(entry.get("id") or entry.get("profile_id") or uuid.uuid4().hex)
    name = (entry.get("name") or f"Profil {profile_id[:6]}").strip()
    return {"id": profile_id, "name": name or profile_id, "config": config}


def normalize_llm_profiles(raw: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    profiles: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for entry in raw:
            normalized = normalize_llm_profile(entry)
            if normalized:
                profiles.append(normalized)
    return profiles


def load_app_settings() -> Dict[str, Any]:
    defaults = {
        "default_model_id": None,
        "llm_config": DEFAULT_LLM_CONFIG.copy(),
        "llm_profiles": [],
        "active_llm_profile": None,
    }
    if SETTINGS_PATH.is_file():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            defaults.update({k: v for k, v in data.items() if k != "llm_config"})
            defaults["llm_config"] = normalize_llm_config(data.get("llm_config"))
            defaults["llm_profiles"] = normalize_llm_profiles(data.get("llm_profiles"))
            defaults["active_llm_profile"] = data.get("active_llm_profile")
            if defaults["active_llm_profile"] and not any(
                p.get("id") == defaults["active_llm_profile"] for p in defaults["llm_profiles"]
            ):
                defaults["active_llm_profile"] = None
        except json.JSONDecodeError:
            logger.warning("app-settings.json konnte nicht geparst werden, fallback auf Defaults.")
    return defaults


def save_app_settings(data: Dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_llm_profiles() -> List[Dict[str, Any]]:
    profiles = normalize_llm_profiles(_app_settings.get("llm_profiles") or [])
    _app_settings["llm_profiles"] = profiles
    return profiles


def get_active_llm_profile_id() -> Optional[str]:
    return _app_settings.get("active_llm_profile")


def get_llm_config(profile_id: Optional[str] = None) -> Dict[str, Any]:
    profiles = get_llm_profiles()
    target_id = profile_id or get_active_llm_profile_id()
    if target_id:
        for profile in profiles:
            if profile.get("id") == target_id:
                return normalize_llm_config(profile.get("config"))
    return normalize_llm_config(_app_settings.get("llm_config"))


def set_active_llm_profile(profile_id: Optional[str]) -> Optional[str]:
    profiles = get_llm_profiles()
    if profile_id is None:
        _app_settings["active_llm_profile"] = None
    elif any(p.get("id") == profile_id for p in profiles):
        _app_settings["active_llm_profile"] = profile_id
    else:
        raise HTTPException(status_code=404, detail="Profil nicht gefunden.")
    save_app_settings(_app_settings)
    return _app_settings.get("active_llm_profile")


def upsert_llm_profile(
    config: Dict[str, Any], name: Optional[str] = None, profile_id: Optional[str] = None, activate: bool = True
) -> Dict[str, Any]:
    profiles = get_llm_profiles()
    normalized = normalize_llm_profile({"config": config, "id": profile_id, "name": name})
    if normalized is None:
        raise HTTPException(status_code=400, detail="Profil konnte nicht normalisiert werden.")
    updated = False
    for idx, profile in enumerate(profiles):
        if profile.get("id") == normalized["id"]:
            profiles[idx] = normalized
            updated = True
            break
    if not updated:
        profiles.append(normalized)
    _app_settings["llm_profiles"] = profiles
    if activate:
        _app_settings["active_llm_profile"] = normalized["id"]
    save_app_settings(_app_settings)
    return normalized


def delete_llm_profile(profile_id: str) -> List[Dict[str, Any]]:
    profiles = get_llm_profiles()
    filtered = [p for p in profiles if p.get("id") != profile_id]
    if len(filtered) == len(profiles):
        raise HTTPException(status_code=404, detail="Profil nicht gefunden.")
    _app_settings["llm_profiles"] = filtered
    if _app_settings.get("active_llm_profile") == profile_id:
        _app_settings["active_llm_profile"] = None
    save_app_settings(_app_settings)
    return filtered


def persist_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_llm_config(config)
    _app_settings["llm_config"] = normalized
    save_app_settings(_app_settings)
    return normalized


def reset_llm_config() -> Dict[str, Any]:
    return persist_llm_config(DEFAULT_LLM_CONFIG.copy())


def build_llm_settings_payload(message: Optional[str] = None) -> Dict[str, Any]:
    payload = {
        "config": get_llm_config(),
        "profiles": get_llm_profiles(),
        "active_profile_id": get_active_llm_profile_id(),
    }
    if message:
        payload["message"] = message
    return payload


def get_effective_model_name(config: Dict[str, Any], fallback: str = GPT_MODEL) -> str:
    candidate = (config.get("model") or fallback or GPT_MODEL).strip()
    return candidate or GPT_MODEL


def should_attach_vision(config: Dict[str, Any]) -> bool:
    return str(config.get("vision") or "yes").lower() != "manual"


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
                elif block.get("type") == "image_url":
                    parts.append("[Bildanhang]")
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_text_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        normalized.append({
            "role": message.get("role", "user"),
            "content": flatten_message_content(message.get("content", "")),
        })
    return normalized


def build_lmstudio_endpoint(base_url: str) -> str:
    base = (base_url or LMSTUDIO_BASE_URL).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_openai_backend(messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    api_base = (config.get("apiBase") or OPENAI_BASE_URL).strip() or OPENAI_BASE_URL
    api_key = (config.get("apiKey") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY oder Config-Key fehlt.")
    model_name = get_effective_model_name(config)
    client = get_openai_client(api_base, api_key)
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
    except Exception as exc:  # pragma: no cover - external call
        raise HTTPException(status_code=502, detail=f"OpenAI API Fehler: {exc}") from exc
    return response.choices[0].message.content.strip()


def call_ollama_backend(messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    base_url = (config.get("apiBase") or OLLAMA_BASE_URL).rstrip("/")
    model_name = get_effective_model_name(config, fallback="llama3.1")
    payload = {"model": model_name, "messages": normalize_text_messages(messages), "stream": False}
    try:
        response = requests.post(f"{base_url}/api/chat", json=payload, timeout=LLM_HTTP_TIMEOUT)
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=502, detail=f"Ollama nicht erreichbar: {exc}") from exc
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    data = response.json()
    content = data.get("message", {}).get("content") or data.get("response")
    if not content:
        raise HTTPException(status_code=502, detail="Ollama lieferte keine Antwort.")
    return str(content).strip()


def call_lmstudio_backend(messages: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    base_url = build_lmstudio_endpoint(config.get("apiBase") or LMSTUDIO_BASE_URL)
    model_name = get_effective_model_name(config, fallback="granite-vision-3b-q4")
    headers = {"Content-Type": "application/json"}
    api_key = (config.get("apiKey") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model_name, "messages": normalize_text_messages(messages), "stream": False}
    try:
        response = requests.post(base_url, json=payload, headers=headers, timeout=LLM_HTTP_TIMEOUT)
    except requests.RequestException as exc:  # pragma: no cover
        raise HTTPException(status_code=502, detail=f"LM Studio nicht erreichbar: {exc}") from exc
    if not response.ok:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail="LM Studio Antwort unvollständig.") from exc
    return str(content).strip()


def execute_llm_chat(messages: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> str:
    config = config or get_llm_config()
    provider = config.get("provider") or "openai"
    provider = provider.lower()
    if provider == "openai":
        return call_openai_backend(messages, config)
    if provider == "ollama":
        return call_ollama_backend(messages, config)
    if provider == "lmstudio":
        return call_lmstudio_backend(messages, config)
    raise HTTPException(status_code=400, detail=f"Unbekannter Provider: {provider}")


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
    llm_profile_id: Optional[str] = None,
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
    if llm_profile_id:
        payload["llm_profile_id"] = llm_profile_id
    if error:
        payload["error"] = error
    return payload


def build_analysis_payload(
    classification: Dict[str, Any],
    gpt_response: str,
    model_entry: Dict[str, Any],
    timings: Dict[str, float],
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bundle classification, LLM output and metadata for a single asset."""

    llm_config = llm_config or get_llm_config()
    return {
        "classification": classification,
        "gpt_response": gpt_response,
        "meta": GPTMeta(model=get_effective_model_name(llm_config), success=True).dict(),
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
    prompt: str, image_bytes: bytes, model_entry: Dict[str, Any], llm_config: Optional[Dict[str, Any]] = None
) -> tuple[Dict[str, Any], str, Dict[str, float]]:
    """Execute the TM classification and GPT call with timing data."""

    total_start = time.perf_counter()
    model_start = time.perf_counter()
    classification = classify_image(image_bytes, model_entry)
    model_ms = measure_elapsed_ms(model_start)
    llm_start = time.perf_counter()
    gpt_response = call_gpt_with_image_context(prompt, classification, image_bytes, llm_config)
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


def summarize_batch(prompt: str, items: List[Dict[str, Any]], llm_config: Optional[Dict[str, Any]] = None) -> str:
    """Request a concise batch summary based on individual GPT responses."""

    context_lines = []
    for idx, item in enumerate(items, start=1):
        snippet = item["analysis"].get("gpt_response", "")
        context_lines.append(f"Bild {idx} ({item['image_id']}): {snippet}")
    user_content = (
        "Original Prompt: "
        f"{prompt}\n" "Einzelresultate:\n" + "\n".join(context_lines)
        + "\nFormuliere eine strukturierte Zusammenfassung mit Hauptbefunden und Empfehlungen."
    )
    config = llm_config or get_llm_config()
    messages = [
        {
            "role": "system",
            "content": config.get("systemPrompt") or ANALYZER_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_content},
    ]
    try:
        return execute_llm_chat(messages, config)
    except HTTPException as exc:  # pragma: no cover - provider errors
        logger.warning("Batch summary konnte nicht erstellt werden: %s", exc.detail)
    except Exception as exc:  # pragma: no cover - unexpected failures
        logger.warning("Batch summary brach ab: %s", exc)
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


def prime_default_teachable_model() -> None:
    """Warm the default model once during startup for faster ML-only calls."""

    try:
        entry = resolve_model_entry(_app_settings.get("default_model_id"))
        ensure_model_ready(entry)
    except HTTPException as exc:
        logger.info("Kein Default-Modell zum Vorwärmen verfügbar: %s", exc.detail)
    except Exception as exc:  # pragma: no cover - depends on filesystem state
        logger.warning("Default-Modell konnte nicht vorgewärmt werden: %s", exc)

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


def _files_in_directory(directory: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    try:
        for child in directory.iterdir():
            if child.is_file():
                files[child.name.lower()] = child
    except FileNotFoundError:
        return {}
    return files


def _find_casefold_file(directory: Path, filename: str) -> Optional[Path]:
    target = filename.lower()
    return _files_in_directory(directory).get(target)


def _collect_keras_candidates(directory: Path, recursive: bool = False) -> List[Path]:
    """Return candidate Keras artifacts inside *directory* respecting casing."""

    candidates: List[Path] = []
    try:
        iterator = directory.rglob("*") if recursive else directory.iterdir()
    except FileNotFoundError:
        return candidates

    for item in iterator:
        if not item.is_file():
            continue
        if item.suffix.lower() in KERAS_SUFFIXES:
            candidates.append(item)
    return candidates


def _detect_bundle(candidate: Path) -> Optional[tuple[Path, str]]:
    if not candidate.is_dir():
        return None
    files_map = _files_in_directory(candidate)
    if all(req in files_map for req in TFJS_REQUIRED_FILES_LOWER):
        return candidate, "tfjs"
    keras_candidates = _collect_keras_candidates(candidate)
    if keras_candidates:
        return candidate, "keras_h5"
    if "saved_model.pb" in files_map:
        return candidate, "savedmodel"
    return None


def find_model_root(temp_root: Path) -> tuple[Path, str]:
    detected = _detect_bundle(temp_root)
    if detected:
        return detected
    for candidate in temp_root.rglob("*"):
        if not candidate.is_dir():
            continue
        detected = _detect_bundle(candidate)
        if detected:
            return detected

    raise HTTPException(status_code=400, detail="ZIP enthält kein gültiges Teachable Machine Modell.")


def _candidate_directories(content_root: Path) -> List[Path]:
    directories = [content_root]
    parent = content_root.parent
    if parent != content_root:
        directories.append(parent)
    return directories


def load_bundle_metadata(content_root: Path, bundle_type: str) -> Dict[str, Any]:
    """Extract metadata/labels for any supported Teachable Machine bundle."""

    metadata: Dict[str, Any] | None = None
    metadata_error: Exception | None = None
    for directory in _candidate_directories(content_root):
        metadata_file = _find_casefold_file(directory, "metadata.json")
        if not metadata_file:
            continue
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            break
        except json.JSONDecodeError as exc:
            metadata_error = exc
            continue

    if bundle_type == "tfjs" and metadata is None:
        if metadata_error:
            raise HTTPException(status_code=400, detail="metadata.json ist nicht gültig JSON.") from metadata_error
        raise HTTPException(status_code=400, detail="metadata.json fehlt im TFJS Export.")

    labels: List[str] | None = None
    if isinstance(metadata, dict):
        raw_labels = metadata.get("labels")
        if isinstance(raw_labels, list):
            normalized = [str(label).strip() for label in raw_labels if str(label).strip()]
            if normalized:
                labels = normalized

    for directory in _candidate_directories(content_root):
        labels_file = _find_casefold_file(directory, KERAS_LABEL_FILE)
        if not labels_file:
            continue
        raw = labels_file.read_text(encoding="utf-8")
        parsed = [line.strip() for line in raw.splitlines() if line.strip()]
        if parsed:
            labels = parsed
            break

    if not labels:
        raise HTTPException(
            status_code=400,
            detail="labels.txt oder Labels innerhalb der metadata.json werden benötigt.",
        )

    metadata = metadata or {}
    metadata["labels"] = labels
    metadata.setdefault("format", bundle_type)
    metadata.setdefault("source", "teachable_bundle")
    metadata.setdefault("generated", datetime.utcnow().isoformat() + "Z")
    return metadata


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


class SavedModelWrapper:
    """Thin wrapper that exposes predict() for TensorFlow SavedModels."""

    def __init__(self, model_path: Path) -> None:
        imported = tf.saved_model.load(str(model_path))
        signature = imported.signatures.get("serving_default")
        if signature is None and imported.signatures:
            # Fall back to the first available signature if serving_default is missing.
            signature = next(iter(imported.signatures.values()))
        if signature is None:
            raise RuntimeError("SavedModel enthält keinen validen serving_default-Endpunkt.")
        self._signature = signature
        _, kwargs = signature.structured_input_signature
        if not kwargs:
            raise RuntimeError("SavedModel Eingabesignatur ist leer.")
        self._input_spec = next(iter(kwargs.values()))
        self.input_shape = tuple(self._input_spec.shape)
        self.dtype = self._input_spec.dtype or tf.float32

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        tensor = tf.convert_to_tensor(batch, dtype=self.dtype)
        outputs = self._signature(tensor)
        if isinstance(outputs, dict):
            first = next(iter(outputs.values()))
        else:
            first = outputs
        return first.numpy()


def _discover_model_artifact(model_path: Path) -> tuple[str, Path]:
    """Return (type, path) tuple for the first usable artifact under model_path."""

    if model_path.is_file():
        suffix = model_path.suffix.lower()
        if suffix in KERAS_SUFFIXES:
            return ("keras", model_path)
        raise RuntimeError(
            "Unbekanntes Modellformat. Bitte ein .keras oder .h5 Modell bereitstellen oder ein SavedModel-Verzeichnis nutzen."
        )

    if not model_path.exists():
        raise RuntimeError(f"Model directory '{model_path}' not found.")

    saved_model_file = model_path / "saved_model.pb"
    if saved_model_file.is_file():
        return ("savedmodel", model_path)

    for nested_saved_model in model_path.rglob("saved_model.pb"):
        return ("savedmodel", nested_saved_model.parent)

    keras_candidates = _collect_keras_candidates(model_path)
    if not keras_candidates:
        keras_candidates = _collect_keras_candidates(model_path, recursive=True)

    if keras_candidates:
        keras_candidates.sort()
        return ("keras", keras_candidates[0])

    raise RuntimeError(
        "Modellordner enthält weder ein SavedModel noch eine .keras/.h5 Datei. Bitte Export überprüfen."
    )


def has_tfjs_bundle(model_dir: Path) -> bool:
    """Return True when the TFJS trio from Teachable Machine is still present."""

    files_map = _files_in_directory(model_dir)
    return all(required in files_map for required in TFJS_REQUIRED_FILES_LOWER)


def convert_tfjs_bundle(model_dir: Path) -> Path:
    """Convert a TFJS export into a SavedModel directory via tensorflowjs."""

    tfjs_model = _find_casefold_file(model_dir, "model.json")
    weights_file = _find_casefold_file(model_dir, "weights.bin")
    if tfjs_model is None or weights_file is None:
        raise RuntimeError("TFJS Export ist unvollständig (model.json/weights.bin fehlen).")

    converted_dir = model_dir / "converted-savedmodel"
    if converted_dir.exists():
        shutil.rmtree(converted_dir)
    converted_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "tensorflowjs.converters.converter",
        "--input_format=tfjs_layers_model",
        "--output_format=tf_saved_model",
        str(tfjs_model),
        str(converted_dir),
    ]
    try:
        completed = subprocess.run(  # noqa: S603 - intentional CLI call
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("TFJS Modell nach SavedModel konvertiert: %s", model_dir)
        if completed.stdout:
            logger.debug("tensorflowjs stdout: %s", completed.stdout.strip())
        if completed.stderr:
            logger.debug("tensorflowjs stderr: %s", completed.stderr.strip())
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "tensorflowjs ist nicht installiert. Bitte 'pip install tensorflowjs' ausführen."
        ) from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - converter errors
        raise RuntimeError(
            f"TFJS Konvertierung schlug fehl: {exc.stderr or exc.stdout or exc}"
        ) from exc

    return converted_dir


def ensure_model_artifact(model_path: Path) -> tuple[str, Path]:
    """Ensure we can find or build a loadable artifact for the given model."""

    try:
        return _discover_model_artifact(model_path)
    except RuntimeError as original_error:
        if has_tfjs_bundle(model_path):
            convert_tfjs_bundle(model_path)
            return _discover_model_artifact(model_path)
        raise original_error


def load_tf_model(model_path: Path) -> Any:
    """Load and cache Teachable Machine models across formats."""

    resolved = str(model_path.resolve())
    model = _model_cache.get(resolved)
    if model is not None:
        return model

    artifact_type, artifact_path = ensure_model_artifact(model_path)
    if artifact_type == "savedmodel":
        model = SavedModelWrapper(artifact_path)
    else:
        custom_objects = get_keras_custom_objects()
        try:
            with tf.keras.utils.custom_object_scope(custom_objects):
                model = tf.keras.models.load_model(artifact_path)
        except ValueError as exc:
            saved_model_file = artifact_path / "saved_model.pb"
            if saved_model_file.is_file():
                logger.info(
                    "Keras load_model() schlug fehl (%s). Fallback auf SavedModelWrapper.",
                    exc,
                )
                model = SavedModelWrapper(artifact_path)
            else:
                raise RuntimeError(
                    f"Keras Modell '{artifact_path}' konnte nicht geladen werden: {exc}"
                ) from exc

    _model_cache[resolved] = model
    logger.info("Teachable Machine Modell geladen: %s (Quelle: %s)", resolved, artifact_type)
    return model


def get_openai_client(api_base: str | None, api_key: str) -> OpenAI:
    """Create (and reuse) an OpenAI client for the given base URL and API key."""

    global _client, _client_signature
    fingerprint = (api_base, api_key)
    if _client is None or _client_signature != fingerprint:
        _client = OpenAI(api_key=api_key, base_url=api_base or None)
        _client_signature = fingerprint
    return _client


def preprocess_image(image_bytes: bytes, input_shape: Sequence[int]) -> np.ndarray:
    """Resize and normalize the input image for the classifier."""
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream).convert("RGB")
    if len(input_shape) < 3:
        raise HTTPException(status_code=500, detail="Unexpected model input shape.")
    height = int(input_shape[1] or 224)
    width = int(input_shape[2] or 224)
    image = image.resize((width, height))
    image_array = np.asarray(image).astype("float32") / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    return image_array


def extract_input_shape(model: Any) -> Sequence[int]:
    """Return a normalized input shape tuple for the loaded model."""

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if hasattr(input_shape, "as_list"):
        input_shape = tuple(input_shape.as_list())
    if isinstance(input_shape, tuple):
        return input_shape
    raise HTTPException(status_code=500, detail="Modell liefert keine Input-Shape zurück.")


def warmup_teachable_model(model: Any, model_entry: Dict[str, Any]) -> None:
    """Run a dummy inference once so the model is initialized."""

    if getattr(model, "_opencore_warmed", False):
        return
    try:
        input_shape = extract_input_shape(model)
        if len(input_shape) < 4:
            return
        height = int(input_shape[1] or 224)
        width = int(input_shape[2] or 224)
        channels = int(input_shape[3] or 3)
        dummy = np.zeros((1, height, width, channels), dtype="float32")
        model.predict(dummy, verbose=0)
        setattr(model, "_opencore_warmed", True)
        logger.info(
            "ML-Modell '%s' initialisiert (%sx%s, %s Kanäle).",
            model_entry.get("name"),
            height,
            width,
            channels,
        )
    except Exception as exc:  # pragma: no cover - depends on model internals
        logger.warning(
            "Warmup für Modell %s fehlgeschlagen: %s",
            model_entry.get("name"),
            exc,
        )


def ensure_model_ready(model_entry: Dict[str, Any]) -> Any:
    """Load (and warm) the TensorFlow model for the current request."""

    model = load_tf_model(model_entry["path"])
    warmup_teachable_model(model, model_entry)
    return model


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
        if not model_path.exists():
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
    if not model_path.exists():
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


prime_default_teachable_model()


def classify_image(image_bytes: bytes, model_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Run the Teachable Machine model on the provided image bytes."""
    try:
        model = ensure_model_ready(model_entry)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    input_shape = extract_input_shape(model)
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
    user_prompt: str, classification: Dict[str, Any], image_bytes: bytes, llm_config: Optional[Dict[str, Any]] = None
) -> str:
    """Send the classification context plus the user's prompt (and image) to the configured LLM."""

    config = llm_config or get_llm_config()
    distribution = ", ".join(
        f"{pred['label']}: {pred['confidence']:.2%}"
        for pred in classification["all_predictions"]
    )
    classification_summary = (
        f"Top label: {classification['top_label']} "
        f"(confidence: {classification['top_confidence']:.2%}). "
        f"Full distribution: {distribution}."
    )

    base_text = (
        "Classification result from Teachable Machine: "
        f"{classification_summary}\nUser prompt: {user_prompt}\n"
        "Bitte liefere eine strukturierte Analyse mit Handlungsempfehlungen."
    )

    attach_image = bool(image_bytes) and should_attach_vision(config)
    if attach_image:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        user_content: Any = [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    else:
        user_content = base_text

    messages = [
        {
            "role": "system",
            "content": config.get("systemPrompt") or ANALYZER_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_content},
    ]

    return execute_llm_chat(messages, config)


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


@app.get("/api/settings/llm", response_class=JSONResponse)
async def fetch_llm_settings() -> JSONResponse:
    """Return the persisted LLM/provider configuration for the config UI."""

    return JSONResponse(build_llm_settings_payload())


@app.post("/api/settings/llm", response_class=JSONResponse)
async def persist_llm_settings(payload: LLMConfigPayload) -> JSONResponse:
    """Persist the provider/system prompt configuration in app-settings.json."""

    if not payload.config:
        raise HTTPException(status_code=400, detail="Konfiguration fehlt.")
    if payload.profile_id or payload.profile_name:
        profile = upsert_llm_profile(payload.config, payload.profile_name, payload.profile_id, payload.make_active)
        return JSONResponse(
            {
                **build_llm_settings_payload("Profil gespeichert."),
                "profile": profile,
            }
        )
    config = persist_llm_config(payload.config)
    return JSONResponse({**build_llm_settings_payload("LLM-Konfiguration gespeichert."), "config": config})


@app.delete("/api/settings/llm", response_class=JSONResponse)
async def clear_llm_settings() -> JSONResponse:
    """Reset the provider configuration to defaults."""

    config = reset_llm_config()
    return JSONResponse({**build_llm_settings_payload("LLM-Konfiguration zurückgesetzt."), "config": config})


@app.get("/api/settings/llm/profiles", response_class=JSONResponse)
async def list_llm_profiles_endpoint() -> JSONResponse:
    """Expose all saved LLM profiles and the active selection."""

    return JSONResponse(build_llm_settings_payload())


@app.post("/api/settings/llm/profiles", response_class=JSONResponse)
async def upsert_llm_profile_endpoint(payload: LLMProfilePayload) -> JSONResponse:
    if not payload.config:
        raise HTTPException(status_code=400, detail="Profilkonfiguration fehlt.")
    profile = upsert_llm_profile(payload.config, payload.name, payload.profile_id, payload.activate)
    return JSONResponse({**build_llm_settings_payload("Profil gespeichert."), "profile": profile})


@app.post("/api/settings/llm/profiles/{profile_id}/activate", response_class=JSONResponse)
async def activate_llm_profile(profile_id: str) -> JSONResponse:
    set_active_llm_profile(profile_id)
    return JSONResponse(build_llm_settings_payload("Aktives Profil gesetzt."))


@app.delete("/api/settings/llm/profiles/{profile_id}", response_class=JSONResponse)
async def delete_llm_profile_endpoint(profile_id: str) -> JSONResponse:
    delete_llm_profile(profile_id)
    return JSONResponse(build_llm_settings_payload("Profil entfernt."))


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
            "has_builtin": Path(MODEL_PATH).exists(),
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
            content_root, bundle_type = find_model_root(temp_root)
            metadata_data = load_bundle_metadata(content_root, bundle_type)
            slug = slugify(display_name)
            target_dir = build_unique_model_dir(slug)
            shutil.copytree(content_root, target_dir)
            try:
                ensure_model_artifact(target_dir)
            except RuntimeError as exc:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    llm_profile_id: Optional[str] = Form(default=None),
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
    resolved_llm_profile = llm_profile_id or get_active_llm_profile_id()
    llm_config = get_llm_config(resolved_llm_profile)
    items: List[Dict[str, Any]] = []
    for idx, file in enumerate(files, start=1):
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Leere Datei übermittelt.")
        if normalized_mode == "ml":
            classification, timings = perform_ml_only(data, model_entry)
            analysis_payload = build_ml_payload(classification, model_entry, timings)
        else:
            classification, gpt_response, timings = perform_analysis(prompt, data, model_entry, llm_config)
            analysis_payload = build_analysis_payload(classification, gpt_response, model_entry, timings, llm_config)
        items.append(
            {
                "image_id": file.filename or f"image-{idx}",
                "analysis": analysis_payload,
            }
        )

    summary_text = (
        summarize_ml_items(items)
        if normalized_mode == "ml"
        else summarize_batch(prompt, items, llm_config)
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
            llm_profile_id=resolved_llm_profile,
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
    llm_profile_id: Optional[str] = Form(default=None),
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
    llm_config = get_llm_config(llm_profile_id)
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
            llm_profile_id=resolved_llm_profile,
        )
    else:
        classification, gpt_response, timings = perform_analysis(prompt, image_bytes, model_entry, llm_config)
        analysis_payload = build_analysis_payload(classification, gpt_response, model_entry, timings, llm_config)
        debug_payload = build_debug_payload(
            request_id,
            model_entry,
            prompt,
            timings,
            batch_items=1,
            debug_enabled=debug_flag,
            llm_profile_id=resolved_llm_profile,
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
    job = stream_manager.start_job(
        {
            **payload.dict(),
            "source_type": source_type,
            "analysis_mode": analysis_mode,
            "llm_profile_id": payload.llm_profile_id or None,
        }
    )
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
    messages = [
        {"role": "system", "content": OTTO_SYSTEM_PROMPT},
        {"role": "user", "content": payload.prompt.strip()},
    ]
    config = get_llm_config(payload.llm_profile_id)
    answer = execute_llm_chat(messages, config)
    return JSONResponse({"response": answer, "model": get_effective_model_name(config)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
