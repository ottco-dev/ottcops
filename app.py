"""
Example FastAPI application that combines a Teachable Machine image classifier
with an OpenAI GPT model to generate rich responses.

Required dependencies:
    pip install fastapi uvicorn tensorflow pillow openai python-multipart
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import os
import re
import socket
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - only triggered when TF missing
    raise RuntimeError(
        "TensorFlow is required for this application. Install it with 'pip install tensorflow'."
    ) from exc

from openai import OpenAI

try:  # pragma: no cover - optional dependency for network mode
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:  # pragma: no cover - handled at runtime when needed
    ServiceInfo = None  # type: ignore
    Zeroconf = None  # type: ignore

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

MODEL_PATH = os.getenv("TEACHABLE_MODEL_PATH", "./models/teachable_model")
DEFAULT_CLASS_NAMES: List[str] = ["class_1", "class_2", "class_3"]
GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4.1-mini")
BASE_DIR = Path(__file__).resolve().parent
SIMPLE_UI_PATH = BASE_DIR / "static" / "index.html"
CONFIG_UI_PATH = BASE_DIR / "static" / "config.html"
COMPLETIONS_UI_PATH = BASE_DIR / "static" / "completions.html"
TM_MODELS_DIR = BASE_DIR / "TM-models"
TM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TM_REGISTRY_PATH = TM_MODELS_DIR / "registry.json"
REQUIRED_TM_FILES = {"metadata.json", "model.json", "weights.bin"}
NETWORK_CONFIG_FILENAME = "network-config.json"
SETTINGS_PATH = BASE_DIR / "app-settings.json"
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

_model_cache: Dict[str, tf.keras.Model] = {}
_client: OpenAI | None = None
_network_zeroconf: Zeroconf | None = None
_network_service: ServiceInfo | None = None
_network_runtime_config: Dict[str, Any] = {}
_app_settings: Dict[str, Any] = {}

LOG_LEVEL = os.getenv("OTTC_LOG_LEVEL", "INFO").upper()


def _build_logger() -> logging.Logger:
    level = getattr(logging, LOG_LEVEL, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level)
    else:
        logging.getLogger().setLevel(level)
    return logging.getLogger("ottcouture.app")


logger = _build_logger()


class NetworkPayload(BaseModel):
    hostname: str
    port: int = 8000


class CompletionPayload(BaseModel):
    prompt: str


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


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...), prompt: str = Form(...), model_id: Optional[str] = Form(default=None)
) -> JSONResponse:
    """Endpoint that classifies an image and enriches the result with GPT."""
    if image is None:
        raise HTTPException(status_code=400, detail="Image file is required.")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc

    model_entry = resolve_model_entry(model_id)
    classification = classify_image(image_bytes, model_entry)
    gpt_response = call_gpt_with_image_context(prompt, classification, image_bytes)

    result = {
        "classification": classification,
        "gpt_response": gpt_response,
        "meta": GPTMeta(model=GPT_MODEL, success=True).dict(),
        "teachable_model": {
            "id": model_entry.get("id"),
            "name": model_entry.get("name"),
            "type": model_entry.get("type"),
            "source": model_entry.get("source"),
        },
    }
    return JSONResponse(content=result)


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
