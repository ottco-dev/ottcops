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
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
import requests

try:  # pragma: no cover - optional dependency for MQTT sensors
    from paho.mqtt import client as mqtt_client  # type: ignore
except ImportError:  # pragma: no cover - gracefully degrade when MQTT is missing
    mqtt_client = None

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
from opencore.config import (
    ANALYZER_SYSTEM_PROMPT,
    BASE_DIR,
    COMPLETIONS_UI_PATH,
    CONFIG_UI_PATH,
    DEFAULT_CLASS_NAMES,
    DEFAULT_LLM_CONFIG,
    DOCS_DIR,
    GPT_MODEL,
    KERAS_LABEL_FILE,
    KERAS_SUFFIXES,
    LLM_ALLOWED_PROVIDERS,
    LLM_HTTP_TIMEOUT,
    LMSTUDIO_BASE_URL,
    LOG_LEVEL,
    MODEL_PATH,
    MODEL_TYPE_ALIASES,
    NETWORK_CONFIG_FILENAME,
    NETWORK_DEFAULT_CONFIG,
    MQTT_DEFAULT_CONFIG,
    MQTT_SENSOR_KINDS,
    OLLAMA_BASE_URL,
    OPENAI_BASE_URL,
    SETTINGS_PATH,
    SHARE_STORE_DIR,
    SHARE_UI_PATH,
    SIMPLE_UI_PATH,
    STREAM_BATCH_INTERVAL_DEFAULT,
    STREAM_BUFFER_MAX,
    STREAM_CAPTURE_INTERVAL_DEFAULT,
    TFJS_REQUIRED_FILES,
    TFJS_REQUIRED_FILES_LOWER,
    TM_MODELS_DIR,
    TM_REGISTRY_PATH,
    UPLOADS_DIR,
    UPLOAD_REGISTRY_PATH,
    UPSTREAM_REPO_BRANCH,
    UPSTREAM_REPO_URL,
)
from opencore.logging_utils import build_logger
from opencore.settings_store import (
    bootstrap_app_settings,
    build_llm_settings_payload,
    delete_llm_profile,
    get_active_llm_profile_id,
    get_default_model_id,
    get_llm_config,
    get_llm_profiles,
    get_mqtt_config,
    normalize_llm_config,
    persist_llm_config,
    persist_mqtt_config,
    reset_llm_config,
    set_active_llm_profile,
    set_default_model_id,
    upsert_llm_profile,
)
from opencore.update_check import ensure_latest_code_checked_out

KERAS_CUSTOM_OBJECTS: dict[str, Any] | None = None

_model_cache: Dict[str, Any] = {}
_client: OpenAI | None = None
_client_signature: tuple[str | None, str | None] | None = None
_network_zeroconf: Zeroconf | None = None
_network_service: ServiceInfo | None = None
_network_runtime_config: Dict[str, Any] = {}
_stream_lock = threading.Lock()


logger = build_logger()

app = FastAPI(title="OPENCORE Analyzer", docs_url=None, redoc_url=None)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/doc", StaticFiles(directory=DOCS_DIR, html=True), name="doc")
app.mount("/share-store", StaticFiles(directory=SHARE_STORE_DIR), name="share-store")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


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
    overlay_frame: Optional[bytes] = None


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
            if job.source_type in {"video", "usb"} and cv2 is None:
                raise HTTPException(status_code=400, detail="OpenCV fehlt für Videoquellen.")
            if job.source_type in {"video", "usb"} and cv2 is not None:
                source = job.source_url
                if job.source_type == "usb":
                    try:
                        source = int(job.source_url)
                    except ValueError:
                        source = job.source_url
                job.video_capture = cv2.VideoCapture(source)
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
            "overlay_available": bool(job.overlay_frame),
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
        if job.source_type == "usb":
            if cv2 is None:
                raise RuntimeError("OpenCV nicht verfügbar")
            if job.video_capture is None:
                source = job.source_url
                try:
                    source = int(job.source_url)
                except ValueError:
                    source = job.source_url
                job.video_capture = cv2.VideoCapture(source)
            ret, frame = job.video_capture.read()
            if not ret:
                job.video_capture.release()
                source = job.source_url
                try:
                    source = int(job.source_url)
                except ValueError:
                    source = job.source_url
                job.video_capture = cv2.VideoCapture(source)
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
            overlay_frame: Optional[bytes] = None
            overlay_text: Optional[str] = None
            prompt = job.prompt or "Automated stream report"
            payload, classification, overlay_text, timings = run_analysis_for_mode(
                prompt, frame, model_entry, llm_config, job.analysis_mode
            )
            total_model_ms += timings.get("model_ms", 0.0)
            total_llm_ms += timings.get("llm_ms", 0.0)
            items.append({"image_id": f"{job.stream_id}-{idx}", "analysis": payload})
            try:
                overlay_frame = build_overlay_frame(frame, classification, overlay_text)
            except Exception as exc:  # pragma: no cover - best-effort overlay
                logger.debug("Overlay konnte nicht erzeugt werden: %s", exc)
            if overlay_frame:
                job.overlay_frame = overlay_frame
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
ensure_latest_code_checked_out()


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


class MQTTConfigPayload(BaseModel):
    broker: str = ""
    port: int = 1883
    username: str = ""
    password: str = ""
    use_tls: bool = False
    sensors: List[Dict[str, Any]] = Field(default_factory=list)


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


def run_llm_connectivity_probe(config: Dict[str, Any]) -> Dict[str, Any]:
    """Send a lightweight ping to the configured LLM provider for reachability diagnostics."""

    normalized = normalize_llm_config(config)
    request_id = generate_request_id()
    model = get_effective_model_name(normalized)
    started = time.perf_counter()
    try:
        reply = execute_llm_chat(
            [
                {
                    "role": "system",
                    "content": "Connectivity check: antworte mit einem kurzen ok-Text.",
                },
                {"role": "user", "content": "Ping von ottcouture.eu config healthcheck."},
            ],
            normalized,
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=502, detail=f"LLM-Test fehlgeschlagen: {exc}") from exc
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    snippet = str(reply).strip()
    return {
        "request_id": request_id,
        "provider": normalized.get("provider") or "openai",
        "model": model,
        "elapsed_ms": elapsed_ms,
        "preview": snippet[:400],
    }


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


def ensure_mqtt_available() -> None:
    """Raise an HTTP error if the optional MQTT dependency is missing."""

    if mqtt_client is None:
        raise HTTPException(status_code=500, detail="Install paho-mqtt to use sensor polling.")


def sanitize_mqtt_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the MQTT config without leaking the password."""

    safe = dict(config)
    if safe.get("password"):
        safe["has_password"] = True
    safe.pop("password", None)
    return safe


def poll_mqtt_topics(config: Dict[str, Any], timeout: float = 5.0) -> List[Dict[str, Any]]:
    """Connect to the broker and collect the most recent payloads for configured topics."""

    ensure_mqtt_available()
    broker = (config.get("broker") or "").strip()
    port = int(config.get("port") or 1883)
    sensors = config.get("sensors") or []
    if not broker:
        raise HTTPException(status_code=400, detail="MQTT broker missing.")
    if not sensors:
        raise HTTPException(status_code=400, detail="No sensors configured.")

    readings: Dict[str, Dict[str, Any]] = {}
    client = mqtt_client.Client()
    if config.get("username"):
        client.username_pw_set(config.get("username"), config.get("password") or None)
    if config.get("use_tls"):
        client.tls_set()

    def on_message(client_obj, userdata, msg):  # type: ignore[unused-argument]
        for sensor in sensors:
            if msg.topic == sensor.get("topic"):
                try:
                    value = msg.payload.decode("utf-8")
                except Exception:
                    value = str(msg.payload)
                readings[sensor.get("id") or msg.topic] = {
                    "id": sensor.get("id"),
                    "label": sensor.get("label"),
                    "kind": sensor.get("kind"),
                    "topic": msg.topic,
                    "value": value,
                    "unit": sensor.get("unit"),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }

    client.on_message = on_message
    try:
        client.connect(broker, port, keepalive=60)
    except Exception as exc:  # pragma: no cover - network dependent
        raise HTTPException(status_code=400, detail=f"MQTT connection failed: {exc}") from exc
    for sensor in sensors:
        topic = sensor.get("topic")
        if topic:
            client.subscribe(topic)
    client.loop_start()
    start = time.time()
    try:
        while (time.time() - start) < timeout and len(readings) < len(sensors):
            client.loop(timeout=0.1)
            time.sleep(0.05)
    finally:
        client.loop_stop()
        client.disconnect()
    return list(readings.values())


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


def build_llm_only_payload(
    gpt_response: str, timings: Dict[str, float], llm_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Return a payload describing only the LLM vision step."""

    llm_config = llm_config or get_llm_config()
    return {
        "classification": {},
        "gpt_response": gpt_response,
        "meta": GPTMeta(model=get_effective_model_name(llm_config), success=True).dict(),
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


def perform_llm_only(
    prompt: str, image_bytes: bytes, llm_config: Optional[Dict[str, Any]] = None
) -> tuple[str, Dict[str, float]]:
    """Execute a pure LLM vision call without Teachable Machine context."""

    total_start = time.perf_counter()
    response = call_gpt_with_image_only(prompt, image_bytes, llm_config)
    timings = {"model_ms": 0.0, "llm_ms": measure_elapsed_ms(total_start), "total_ms": measure_elapsed_ms(total_start)}
    return response, timings


def run_analysis_for_mode(
    prompt: str,
    image_bytes: bytes,
    model_entry: Dict[str, Any],
    llm_config: Dict[str, Any],
    analysis_mode: str,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[str], Dict[str, float]]:
    """Execute ML, LLM, or Hybrid flows and return payload plus overlay text."""

    mode = (analysis_mode or "hybrid").lower()
    if mode == "ml":
        classification, timings = perform_ml_only(image_bytes, model_entry)
        payload = build_ml_payload(classification, model_entry, timings)
        overlay_text = f"{classification['top_label']} ({classification['top_confidence']:.1%})"
        return payload, classification, overlay_text, timings
    if mode == "llm":
        gpt_response, timings = perform_llm_only(prompt, image_bytes, llm_config)
        payload = build_llm_only_payload(gpt_response, timings, llm_config)
        return payload, None, gpt_response, timings
    classification, gpt_response, timings = perform_analysis(prompt, image_bytes, model_entry, llm_config)
    payload = build_analysis_payload(classification, gpt_response, model_entry, timings, llm_config)
    return payload, classification, gpt_response, timings


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
bootstrap_app_settings()


def prime_default_teachable_model() -> None:
    """Warm the default model once during startup for faster ML-only calls."""

    try:
        entry = resolve_model_entry(get_default_model_id())
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


def load_upload_registry() -> List[Dict[str, Any]]:
    """Load persisted upload metadata from disk."""

    if not UPLOAD_REGISTRY_PATH.exists():
        return []
    try:
        return json.loads(UPLOAD_REGISTRY_PATH.read_text("utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - depends on user edits
        return []


def save_upload_registry(entries: List[Dict[str, Any]]) -> None:
    """Persist upload metadata to disk."""

    UPLOAD_REGISTRY_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def normalize_upload_type(filename: str, content_type: str | None, explicit: str | None) -> str:
    """Return a normalized upload category based on type hints and file name."""

    if explicit in {"image", "video"}:
        return explicit
    lower_name = filename.lower()
    if content_type and content_type.startswith("video"):
        return "video"
    if lower_name.endswith((".mp4", ".mov", ".mkv", ".avi")):
        return "video"
    return "image"


def register_upload(file_path: Path, original_name: str, kind: str) -> Dict[str, Any]:
    """Append a new upload entry to the registry and return it."""

    registry = load_upload_registry()
    upload_id = f"upload-{uuid.uuid4().hex[:8]}"
    entry = {
        "id": upload_id,
        "name": original_name,
        "type": kind,
        "path": str(file_path.relative_to(BASE_DIR)),
        "created": datetime.utcnow().isoformat() + "Z",
        "size": file_path.stat().st_size,
        "overlay": None,
    }
    registry.append(entry)
    save_upload_registry(registry)
    return entry


def build_unique_upload_path(base_name: str, ext: str) -> Path:
    """Return a non-conflicting path inside the uploads directory."""

    counter = 1
    candidate = UPLOADS_DIR / f"{base_name}{ext}"
    while candidate.exists():
        candidate = UPLOADS_DIR / f"{base_name}-{counter}{ext}"
        counter += 1
    return candidate


def find_upload(upload_id: str) -> Dict[str, Any]:
    """Return a single upload entry or raise 404."""

    for item in load_upload_registry():
        if item.get("id") == upload_id:
            return item
    raise HTTPException(status_code=404, detail="Upload not found.")


def replace_upload_entry(upload_id: str, new_entry: Dict[str, Any]) -> None:
    """Update an upload entry in the registry."""

    registry = load_upload_registry()
    for idx, item in enumerate(registry):
        if item.get("id") == upload_id:
            registry[idx] = new_entry
            save_upload_registry(registry)
            return
    raise HTTPException(status_code=404, detail="Upload not found.")


def delete_upload_entry(upload_id: str) -> None:
    """Delete upload metadata and associated files."""

    registry = load_upload_registry()
    remaining: List[Dict[str, Any]] = []
    removed: Optional[Dict[str, Any]] = None
    for item in registry:
        if item.get("id") == upload_id:
            removed = item
            continue
        remaining.append(item)
    if removed is None:
        raise HTTPException(status_code=404, detail="Upload not found.")
    save_upload_registry(remaining)
    file_path = BASE_DIR / removed.get("path", "")
    if file_path.exists():
        file_path.unlink(missing_ok=True)
    overlay = removed.get("overlay")
    if overlay:
        overlay_path = BASE_DIR / overlay
        if overlay_path.exists():
            overlay_path.unlink(missing_ok=True)

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
    default_id = get_default_model_id()
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
        set_default_model_id(None)
        return None
    entry = find_tm_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Modell wurde nicht gefunden.")
    set_default_model_id(model_id)
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


def _wrap_text(text: str, width: int = 42) -> List[str]:
    """Simple word-wrapping helper for overlay annotations."""

    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def build_overlay_frame(
    image_bytes: bytes, classification: Optional[Dict[str, Any]], overlay_text: Optional[str]
) -> Optional[bytes]:
    """Render an overlay onto the captured frame for stream previews."""

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None

    label = classification.get("top_label") if classification else ""
    confidence = classification.get("top_confidence") if classification else None
    header = f"Top: {label}" if label else "Analysis"
    if confidence is not None:
        header += f" ({confidence:.1%})"
    body = overlay_text or ""

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_lines = [header]
    if body:
        for line in body.split("\n"):
            if not line.strip():
                continue
            text_lines.extend(_wrap_text(line.strip(), width=42))

    padding = 10
    line_height = font.getbbox("Hg")[3] - font.getbbox("Hg")[1] + 2
    block_height = padding * 2 + line_height * len(text_lines)
    block_width = max(draw.textlength(line, font=font) for line in text_lines) + padding * 2

    draw.rectangle([(5, 5), (5 + block_width, 5 + block_height)], fill=(0, 0, 0, 128))
    y = 5 + padding
    for line in text_lines:
        draw.text((5 + padding, y), line, fill=(0, 255, 127), font=font)
        y += line_height

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def extract_video_frames(video_path: Path, max_frames: int = 8, step_seconds: float = 1.5) -> List[bytes]:
    """Grab a handful of frames from a video file for analysis."""

    if cv2 is None:
        raise HTTPException(status_code=400, detail="OpenCV is required for video analysis.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open video file for reading.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step_frames = max(1, int(fps * step_seconds))
    frames: List[bytes] = []
    idx = 0
    success, frame = cap.read()
    while success and len(frames) < max_frames:
        if idx % step_frames == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append(buf.tobytes())
        idx += 1
        success, frame = cap.read()
    cap.release()
    if not frames:
        raise HTTPException(status_code=400, detail="Video contained no readable frames.")
    return frames


def build_overlay_video(frames: List[bytes], target_path: Path) -> Optional[str]:
    """Render overlay JPEG frames into a short MP4 for preview."""

    if cv2 is None or not frames:
        return None
    decoded = []
    for frame_bytes in frames:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        decoded.append(img)
    if not decoded:
        return None
    height, width = decoded[0].shape[:2]
    writer = cv2.VideoWriter(
        str(target_path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (width, height)
    )
    for img in decoded:
        writer.write(img)
    writer.release()
    return str(target_path.relative_to(BASE_DIR))


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
        default_id = get_default_model_id()
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


def call_gpt_with_image_only(
    user_prompt: str, image_bytes: bytes, llm_config: Optional[Dict[str, Any]] = None
) -> str:
    """Send only the user prompt plus image to the configured LLM."""

    config = llm_config or get_llm_config()
    attach_image = bool(image_bytes) and should_attach_vision(config)
    base_text = f"User prompt: {user_prompt}. Provide a concise, structured answer."
    if attach_image:
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        user_content: Any = [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    else:
        user_content = base_text

    messages = [
        {"role": "system", "content": config.get("systemPrompt") or ANALYZER_SYSTEM_PROMPT},
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


@app.post("/api/settings/llm/test", response_class=JSONResponse)
async def test_llm_settings(payload: LLMConfigPayload) -> JSONResponse:
    """Send a quick test completion against the selected or active LLM profile."""

    config = normalize_llm_config(payload.config) if payload.config else get_llm_config(payload.profile_id)
    probe = run_llm_connectivity_probe(config)
    return JSONResponse({"message": "LLM-Test erfolgreich", **probe})


@app.post("/api/settings/llm", response_class=JSONResponse)
async def persist_llm_settings(payload: LLMConfigPayload) -> JSONResponse:
    """Persist the provider/system prompt configuration in app-settings.json."""

    if not payload.config:
        raise HTTPException(status_code=400, detail="Konfiguration fehlt.")
    if (payload.profile_id is not None) or (payload.profile_name is not None):
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


@app.get("/api/mqtt/config", response_class=JSONResponse)
async def fetch_mqtt_config() -> JSONResponse:
    """Expose the MQTT broker + sensor settings (password is never returned)."""

    config = get_mqtt_config()
    return JSONResponse({"config": sanitize_mqtt_config(config), "kinds": sorted(MQTT_SENSOR_KINDS)})


@app.post("/api/mqtt/config", response_class=JSONResponse)
async def save_mqtt_config(payload: MQTTConfigPayload) -> JSONResponse:
    """Persist MQTT broker settings and sensor topics."""

    current = get_mqtt_config()
    body = payload.dict()
    if not body.get("password") and current.get("password"):
        body["password"] = current.get("password")
    saved = persist_mqtt_config(body)
    return JSONResponse({"config": sanitize_mqtt_config(saved), "message": "MQTT settings saved."})


@app.post("/api/mqtt/poll", response_class=JSONResponse)
async def poll_mqtt_values() -> JSONResponse:
    """Connect to the configured broker, subscribe to sensor topics, and return the most recent payloads."""

    readings = poll_mqtt_topics(get_mqtt_config())
    return JSONResponse({"values": readings, "count": len(readings)})


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
            "default_model_id": get_default_model_id(),
            "has_builtin": Path(MODEL_PATH).exists(),
        }
    )


@app.post("/tm-models/default/{model_id}", response_class=JSONResponse)
async def tm_models_set_default(model_id: str) -> JSONResponse:
    entry = set_default_tm_model(model_id)
    return JSONResponse({"default_model_id": get_default_model_id(), "model": entry})


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

    if get_default_model_id() is None:
        set_default_model_id(entry["id"])

    return JSONResponse({"message": "Modell gespeichert", "model": entry, "default_model_id": get_default_model_id()})


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
    if normalized_mode not in {"hybrid", "ml", "llm"}:
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
        analysis_payload, classification, gpt_text, timings = run_analysis_for_mode(
            prompt, data, model_entry, llm_config, normalized_mode
        )
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
    upload_id: Optional[str] = Form(default=None),
) -> JSONResponse:
    """Endpoint that classifies an image and optionally calls GPT."""
    if image is None and not upload_id:
        raise HTTPException(status_code=400, detail="Image file is required.")
    normalized_mode = (analysis_mode or "hybrid").lower()
    if normalized_mode not in {"hybrid", "ml", "llm"}:
        raise HTTPException(status_code=400, detail="Ungültiger Analysemodus.")
    if normalized_mode != "ml" and not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    request_id = generate_request_id()
    debug_flag = parse_bool_flag(debug)
    resolved_llm_profile = llm_profile_id or get_active_llm_profile_id()
    image_bytes: Optional[bytes] = None
    if image is not None:
        try:
            image_bytes = await image.read()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc
    if (image_bytes is None or not image_bytes) and upload_id:
        entry = find_upload(upload_id)
        entry_path = BASE_DIR / entry.get("path", "")
        if not entry_path.exists():
            raise HTTPException(status_code=404, detail="Upload file missing on disk.")
        if entry.get("type") != "image":
            raise HTTPException(status_code=400, detail="Upload is not an image.")
        image_bytes = entry_path.read_bytes()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image payload missing.")

    model_entry = resolve_model_entry(model_id)
    llm_config = get_llm_config(resolved_llm_profile)
    analysis_payload, classification, overlay_text, timings = run_analysis_for_mode(
        prompt, image_bytes, model_entry, llm_config, normalized_mode
    )
    debug_payload = build_debug_payload(
        request_id,
        model_entry,
        prompt or ("LLM-only" if normalized_mode == "llm" else "ML-only"),
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


@app.post("/api/opencore/analyze-video", response_class=JSONResponse)
async def analyze_video(
    file: UploadFile = File(default=None),
    upload_id: Optional[str] = Form(default=None),
    prompt: str = Form(default=""),
    model_id: Optional[str] = Form(default=None),
    analysis_mode: str = Form(default="hybrid"),
    llm_profile_id: Optional[str] = Form(default=None),
) -> JSONResponse:
    normalized_mode = (analysis_mode or "hybrid").lower()
    if normalized_mode not in {"hybrid", "ml", "llm"}:
        raise HTTPException(status_code=400, detail="Ungültiger Analysemodus.")
    if normalized_mode != "ml" and not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt ist erforderlich.")
    if file is None and not upload_id:
        raise HTTPException(status_code=400, detail="Video file is required.")

    entry: Optional[Dict[str, Any]] = None
    video_path: Optional[Path] = None
    if file is not None:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Leere Datei erhalten.")
        base_name = slugify(Path(file.filename).stem or "video")
        ext = Path(file.filename).suffix or ".mp4"
        target_path = build_unique_upload_path(base_name, ext)
        target_path.write_bytes(data)
        entry = register_upload(target_path, file.filename, "video")
        video_path = target_path
    elif upload_id:
        entry = find_upload(upload_id)
        if entry.get("type") != "video":
            raise HTTPException(status_code=400, detail="Upload ist kein Video.")
        video_path = BASE_DIR / entry.get("path", "")
    if video_path is None or not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found on disk.")

    request_id = generate_request_id()
    llm_profile = llm_profile_id or get_active_llm_profile_id()
    model_entry = resolve_model_entry(model_id)
    llm_config = get_llm_config(llm_profile)
    frames = extract_video_frames(video_path)
    overlay_frames: List[bytes] = []
    items: List[Dict[str, Any]] = []
    total_timings = {"model_ms": 0.0, "llm_ms": 0.0, "total_ms": 0.0}
    for idx, frame_bytes in enumerate(frames, start=1):
        payload, classification, overlay_text, timings = run_analysis_for_mode(
            prompt, frame_bytes, model_entry, llm_config, normalized_mode
        )
        total_timings["model_ms"] += timings.get("model_ms", 0.0)
        total_timings["llm_ms"] += timings.get("llm_ms", 0.0)
        total_timings["total_ms"] += timings.get("total_ms", 0.0)
        try:
            overlay_frame = build_overlay_frame(frame_bytes, classification, overlay_text)
            if overlay_frame:
                overlay_frames.append(overlay_frame)
        except Exception:
            pass
        items.append({"frame_id": f"frame-{idx}", "analysis": payload})

    summary_text = (
        summarize_ml_items(items)
        if normalized_mode == "ml"
        else summarize_batch(prompt, items, llm_config)
    )

    overlay_video: Optional[str] = None
    if overlay_frames:
        overlay_target = UPLOADS_DIR / f"{entry['id'] if entry else uuid.uuid4().hex}-overlay.mp4"
        overlay_video = build_overlay_video(overlay_frames, overlay_target)
        if entry is not None and overlay_video:
            entry["overlay"] = overlay_video
            replace_upload_entry(entry["id"], entry)

    response_payload = {
        "status": "ok",
        "summary": {"text": summary_text},
        "items": items,
        "analysis_mode": normalized_mode,
        "teachable_model": build_teachable_meta(model_entry),
        "debug": build_debug_payload(
            request_id,
            model_entry,
            prompt or ("LLM-only" if normalized_mode == "llm" else "ML-only"),
            total_timings,
            batch_items=len(items),
            debug_enabled=False,
            llm_profile_id=llm_profile,
        ),
        "overlay_video": overlay_video,
        "upload_id": entry.get("id") if entry else None,
    }
    return JSONResponse(response_payload)


@app.post("/api/opencore/share", response_class=JSONResponse)
async def create_share(payload: SharePayload) -> JSONResponse:
    if not isinstance(payload.payload, dict) or not payload.payload:
        raise HTTPException(status_code=400, detail="Share-Payload fehlt.")
    share_id = save_share_payload(payload.payload)
    return JSONResponse({"share_id": share_id, "url": f"/share/{share_id}"})


@app.get("/api/uploads", response_class=JSONResponse)
async def list_uploads(sort: str = "recent") -> JSONResponse:
    entries = load_upload_registry()
    if sort == "name":
        entries = sorted(entries, key=lambda x: x.get("name", "").lower())
    elif sort == "type":
        entries = sorted(entries, key=lambda x: (x.get("type", ""), x.get("name", "")))
    else:
        entries = sorted(entries, key=lambda x: x.get("created", ""), reverse=True)
    return JSONResponse({"uploads": entries})


@app.post("/api/uploads", response_class=JSONResponse)
async def upload_media(
    file: UploadFile = File(...),
    kind: str = Form(default="auto"),
    label: str = Form(default=""),
) -> JSONResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    detected_type = normalize_upload_type(file.filename, file.content_type, None if kind == "auto" else kind)
    base_name = slugify(label or Path(file.filename).stem or "upload")
    ext = Path(file.filename).suffix or (".mp4" if detected_type == "video" else ".jpg")
    target_path = build_unique_upload_path(base_name, ext)
    target_path.write_bytes(data)
    entry = register_upload(target_path, label or file.filename, detected_type)
    return JSONResponse({"upload": entry})


@app.post("/api/uploads/{upload_id}/rename", response_class=JSONResponse)
async def rename_upload(upload_id: str, new_name: str = Form(...)) -> JSONResponse:
    entry = find_upload(upload_id)
    if not new_name.strip():
        raise HTTPException(status_code=400, detail="Name is required.")
    old_path = BASE_DIR / entry.get("path", "")
    ext = old_path.suffix or (".mp4" if entry.get("type") == "video" else ".jpg")
    new_path = build_unique_upload_path(slugify(new_name), ext)
    if old_path.exists():
        shutil.move(str(old_path), new_path)
    entry["name"] = new_name.strip()
    entry["path"] = str(new_path.relative_to(BASE_DIR))
    replace_upload_entry(upload_id, entry)
    return JSONResponse({"upload": entry})


@app.delete("/api/uploads/{upload_id}", response_class=JSONResponse)
async def delete_upload(upload_id: str) -> JSONResponse:
    delete_upload_entry(upload_id)
    return JSONResponse({"message": "Upload deleted."})


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
    if source_type not in {"snapshot", "video", "usb"}:
        raise HTTPException(
            status_code=400, detail="source_type muss snapshot, video oder usb sein."
        )
    if analysis_mode not in {"hybrid", "ml", "llm"}:
        raise HTTPException(status_code=400, detail="analysis_mode muss hybrid, ml oder llm sein.")
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


@app.get("/api/opencore/streams/{stream_id}/overlay")
async def stream_overlay(stream_id: str) -> Response:
    job = stream_manager.get_job(stream_id)
    if not job.overlay_frame:
        raise HTTPException(status_code=404, detail="No overlay available yet.")
    return Response(content=job.overlay_frame, media_type="image/jpeg")


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
