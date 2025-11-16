# Project Documentation

This file describes how the FastAPI app is structured, how models are prepared, and how to configure the environment.

## 1. Requirements
- Python 3.10+
- Access to a trained Teachable Machine image classifier as a TensorFlow SavedModel or Keras model.
- OpenAI API key with access to a multimodal model (for example `gpt-4.1-mini`).
- For video streams: optional OpenCV (`opencv-python-headless`) and HTTP access to snapshot/RTSP sources.

## 2. Installation
1. Check out the repository and switch into the project directory.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 3. Models & Labels
- Primary location is `TM-models/`. The upload feature unzips bundles (TFJS with `metadata.json`, `model.json`, `weights.bin` **or** Keras packages with `keras_model.h5` and `labels.txt`) into a subfolder and writes an entry to `TM-models/registry.json`.
- If no entry is active, the analyzer falls back to `TEACHABLE_MODEL_PATH` (for example `./models/teachable_model`).
- Labels are pulled automatically from `metadata.json` or—when using Keras packages—from `labels.txt`. If data is missing, the service generates neutral names (`class_1`, `class_2`, …).

## 4. Configuration
| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | Required for cloud LLMs and the OTTO chat. |
| `OPENAI_GPT_MODEL` | Optional. Default is `gpt-4.1-mini`. |
| `TEACHABLE_MODEL_PATH` | Optional fallback path for a local TM model. |

Additional persistence files:
- `app-settings.json`: stores the default model and provider/prompt configuration from `/api/settings/llm`.
- `network-config.json`: remembers the mDNS status (hostname/port for `ottcolab.local`).

## 5. LLM Settings
- Managed via `/api/settings/llm` and the config console.
- Multiple profiles are supported (OpenAI, Ollama, LM Studio). Each stores base URL, model name, and system prompt.
- Profiles can be activated per run or set as the default for Analyzer, Streams, and OTTO chat.

## 6. Streams & Automation
- Stream jobs are created through the UI or `/api/opencore/streams*` endpoints.
- Jobs can snapshot cameras every few seconds and batch analyses roughly every 30 seconds.
- Results feed into the same JSON pipeline used by single-image requests.

## 7. Custom Modules
- Add Python modules under `opencore/` and include them from `app.py`.
- Mirror the response shape of existing endpoints (status + payload + optional debug) so the UI can consume them immediately.
- Front-end hooks belong in `static/index.html` or `static/config.html` with logic in `static/js/app.js`.

## 8. License & Usage
OPENCORE Analyzer is free for private individuals and developer testing. Cannabis Social Clubs (CSCs) and companies must contact <https://ottcouture.eu> to license commercial use. Feedback: otcdmin@outlook.com, Instagram @ottcouture.eu, Discord discord.gg/GMMSqePfPh.

## 9. Code reference & modular tips
- **Shared utils:** Common constants and logging live in `opencore/config.py` and `opencore/logging_utils.py`; import only what you need for new routers to keep changes isolated.
- **Settings store:** `opencore/settings_store.py` provides getters/setters for LLM profiles, MQTT brokers, and TM registry entries. Each helper is optional—reuse them in a new FastAPI router instead of reimplementing persistence.
- **Front-end helpers:** `static/js/app.js` is organized by feature (templates, uploads, streams, MQTT, exports). Each block is guarded by element checks so you can drop unused sections from HTML without breaking the rest of the page.
- **Adaptation pattern:** Mirror the existing request/response shape (`status`, `analysis`, `debug`) when adding endpoints. Doing so lets the analyzer UI, batch renderer, and exports pick up new functionality without additional wiring.
