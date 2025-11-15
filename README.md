# Teachable Machine + GPT API

Production-ready FastAPI application that accepts an uploaded image and a free-form
prompt, runs a Teachable Machine TensorFlow model, and combines the prediction with a
GPT response.

## Features
- ✅ TensorFlow/Keras model loading from `./models/teachable_model` (override via
  `TEACHABLE_MODEL_PATH`).
- ✅ Swagger UI at [`/docs`](http://localhost:8000/docs) and ReDoc at `/redoc`.
- ✅ Simple HTML UI (GET `/`) that sends multipart requests to `/analyze` without leaving
the browser.
- ✅ Pydantic-powered responses plus structured error handling for missing files,
  TensorFlow issues, and OpenAI failures.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | – | OpenAI API key with access to the selected GPT model. |
| `OPENAI_GPT_MODEL` | Optional | `gpt-4.1-mini` | Override the multimodal GPT model id. |
| `TEACHABLE_MODEL_PATH` | Optional | `./models/teachable_model` | Folder that contains the exported Teachable Machine SavedModel/Keras bundle. |

> ⚠️ The TensorFlow model directory must include the Teachable Machine labels. When you
> export a model, Teachable Machine writes a `labels.txt`; read it at boot and update
> the `CLASS_NAMES` list accordingly.

## Running the API
```bash
export OPENAI_API_KEY="sk-..."  # Required
uvicorn app:app --reload
```
Visit:
- `http://localhost:8000/docs` for Swagger UI
- `http://localhost:8000/` for the minimal HTML client

## Project Layout
```
.
├── app.py              # FastAPI service
├── requirements.txt    # Runtime dependencies
├── README.md           # Quickstart
├── DOKU.md             # Detailed German documentation
├── static/
│   └── index.html      # Simple browser UI
└── models/
    └── teachable_model # (Not tracked) place your exported Teachable Machine model
```

## API Contract
```
POST /analyze (multipart/form-data)
  - image: UploadFile
  - prompt: string
Response 200
{
  "classification": {
    "top_label": "class_1",
    "top_confidence": 0.97,
    "all_predictions": [
      {"label": "class_1", "confidence": 0.97},
      {"label": "class_2", "confidence": 0.02}
    ]
  },
  "gpt_response": "...",
  "meta": {"model": "gpt-4.1-mini", "success": true}
}
```

## Local Testing Without GPT
Set `OPENAI_API_KEY=dummy` and mock the OpenAI client (e.g., monkeypatch
`call_gpt_with_image_context`). TensorFlow predictions can be validated with sample
images via Swagger UI or the HTML form.

## License & Usage Rights
This project is released under the [GNU Affero General Public License v3.0](LICENSE).
OTTCOUTURE retains all rights to the OTTCOUTURE branding, and every modification or
derivative must remain open source, be distributed under the same AGPL terms, and
include clear attribution to OTTCOUTURE.
