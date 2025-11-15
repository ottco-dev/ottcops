"""
Example FastAPI application that combines a Teachable Machine image classifier
with an OpenAI GPT model to generate rich responses.

Required dependencies:
    pip install fastapi uvicorn tensorflow pillow openai python-multipart
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
CLASS_NAMES: List[str] = ["class_1", "class_2", "class_3"]  # Replace with real labels from Teachable Machine.
GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-4.1-mini")
BASE_DIR = Path(__file__).resolve().parent
SIMPLE_UI_PATH = BASE_DIR / "static" / "index.html"
CONFIG_UI_PATH = BASE_DIR / "static" / "config.html"
TM_MODELS_DIR = BASE_DIR / "TM-models"
TM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TM_REGISTRY_PATH = TM_MODELS_DIR / "registry.json"
REQUIRED_TM_FILES = {"metadata.json", "model.json", "weights.bin"}
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

_model: tf.keras.Model | None = None
_client: OpenAI | None = None


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
    entries = load_tm_registry()
    for entry in entries:
        entry.setdefault("type", "trichome")
    return entries


def get_tf_model() -> tf.keras.Model:
    """Load and cache the Teachable Machine TensorFlow model."""
    global _model
    if _model is None:
        if not os.path.isdir(MODEL_PATH):
            raise RuntimeError(f"Model directory '{MODEL_PATH}' not found.")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


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


def classify_image(image_bytes: bytes) -> Dict[str, Any]:
    """Run the Teachable Machine model on the provided image bytes."""
    try:
        model = get_tf_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    input_shape = model.input_shape
    if isinstance(input_shape, list):  # Some TF models have list of input shapes
        input_shape = input_shape[0]
    preprocessed = preprocess_image(image_bytes, input_shape)
    predictions = model.predict(preprocessed, verbose=0)[0]
    predictions = predictions.tolist()

    if len(predictions) != len(CLASS_NAMES):
        raise HTTPException(
            status_code=500,
            detail="Model output size does not match number of class labels.",
        )

    labelled_predictions = [
        {"label": label, "confidence": float(conf)}
        for label, conf in zip(CLASS_NAMES, predictions)
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_ui() -> HTMLResponse:
    """Serve a very small HTML helper to try the API without Swagger."""
    return HTMLResponse(load_simple_ui())


@app.get("/config", response_class=HTMLResponse, include_in_schema=False)
async def config_ui() -> HTMLResponse:
    """Serve a configuration helper for selecting local/self-hosted LLMs."""
    return HTMLResponse(load_config_ui())


@app.get("/tm-models", response_class=JSONResponse)
async def tm_models() -> JSONResponse:
    """Return the registered Teachable Machine models."""
    return JSONResponse({"models": list_tm_models()})


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

    return JSONResponse({"message": "Modell gespeichert", "model": entry})


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), prompt: str = Form(...)) -> JSONResponse:
    """Endpoint that classifies an image and enriches the result with GPT."""
    if image is None:
        raise HTTPException(status_code=400, detail="Image file is required.")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    try:
        image_bytes = await image.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image: {exc}") from exc

    classification = classify_image(image_bytes)
    gpt_response = call_gpt_with_image_context(prompt, classification, image_bytes)

    result = {
        "classification": classification,
        "gpt_response": gpt_response,
        "meta": GPTMeta(model=GPT_MODEL, success=True).dict(),
    }
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
