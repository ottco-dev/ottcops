"""
Example FastAPI application that combines a Teachable Machine image classifier
with an OpenAI GPT model to generate rich responses.

Required dependencies:
    pip install fastapi uvicorn tensorflow pillow openai python-multipart
"""
from __future__ import annotations

import base64
import io
import os
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

_model: tf.keras.Model | None = None
_client: OpenAI | None = None


class GPTMeta(BaseModel):
    model: str
    success: bool


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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root_ui() -> HTMLResponse:
    """Serve a very small HTML helper to try the API without Swagger."""
    return HTMLResponse(load_simple_ui())


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
