# OTTCOUTURE Cannabis Vision OpenCore

The OTTCOPS analyzer from [ottcouture.eu](https://ottcouture.eu) combines Teachable Machine vision models with multimodal LLMs to produce structured JSON outputs. All branding and rights remain with ottcouture.eu; feedback or model submissions are welcome via **otcdmin@outlook.com**, Instagram **@ottcouture.eu**, or Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).

## Feature Highlights
- 🌿 **FastAPI core** with Analyzer UI, config deck, OTTO chat (`/completions`), and documented `/tm-models*` routes.
- 🧠 **Vision LLM switchboard** for OpenAI, Ollama, or LM Studio with system prompt presets, multiple profiles, and server-side persistence for the Analyzer, Streams, and OTTO.
- 🧪 **Teachable Machine depot** with ZIP uploads (TFJS: `metadata.json`/`model.json`/`weights.bin` or Keras: `keras_model.h5` + `labels.txt`), registry, and default selection for the Analyzer.
- 🧵 **Model routing**: the frontend lets you pick a TM slot per run; the choice is persisted in `app-settings.json` for reuse.
- 🤖 **OTTO grow chat** – dedicated screen for cultivation questions with a defined system prompt.
- 📡 **WiFi broadcast mode** (mDNS/zeroconf) for hostnames like `ottcolab.local` across your LAN.
- 📝 **Prompt templates + builder** with FAQ drag-and-drop tiles, sensor chips, and per-browser custom presets.
- 🗂️ **Batch analysis** with `/api/opencore/analyze-batch`, tabs per image, and an overall report.
- 🛠️ **Debug panel** with request ID, model version, and timings (UI toggle + `?debug=1`).
- 🔐 **API token mode**: custom base URL + token with code samples.
- 📤 **Export bundle**: JSON download, PDF report, and share links via `/api/opencore/share` + viewer (`/share/<id>`).
- 🧷 **ML-only analysis mode**: `analysis_mode=ml` returns Teachable Machine JSON without GPT.
- 📡 **MQTT sensor prompts**: broker-backed CO₂/PPFD/humidity/temperature/EC/pH values drop into prompts via drag-and-drop chips.
- 🎞️ **Video uploads + media library**: upload MP4s or reuse stored assets, generate overlay previews, and manage (sort/rename/delete) uploads.
- 🧮 **Selectable analysis pipelines**: choose ML-only, LLM-only, or Hybrid for manual runs, batch jobs, streams, and video clips.
- 🎥 **Stream orchestration**: snapshot/RTSP sources run as background jobs (5 s capture, 30 s batch) and produce automated reports.
- 🔄 **Launch update check**: on start the backend compares against `github.com/ottco-dev/ottcops` and offers an optional `git pull`.

## Usage & Licensing
- OPENCORE Analyzer is free to use only for private individuals and developer testing.
- Cannabis Social Clubs (CSCs) and companies—startups, MSOs, or service providers—must obtain a commercial license directly from **ottcouture.eu** before using this in any production or revenue context.
- Licensing & partnerships: **otcdmin@outlook.com**, Instagram **@ottcouture.eu**, Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).

## Installation (OTTCOUTURE style)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional if you want GPT calls
export OPENAI_API_KEY="sk-..."

# start the dev server
uvicorn app:app --reload
```

At startup the server automatically compares your checkout to `https://github.com/ottco-dev/ottcops`. If a newer commit exists you will be prompted in the console (“Update now?”). Reply `y` or `yes` to run `git pull`; any other response keeps the current version. Set `OTTC_SKIP_UPDATE_CHECK=1` to bypass this (e.g., in CI).

1. Analyzer UI: `http://localhost:8000/`
2. OTTO grow chat: `http://localhost:8000/completions`
3. Config hub + TM depot: `http://localhost:8000/config`
4. Discord crew & support: [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh)
5. Documentation (HTML): `http://localhost:8000/doc/index.html`

## Configuration
| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | for OpenAI flow | – | Key for GPT-4.1 mini or your preferred vision model. |
| `OPENAI_GPT_MODEL` | optional | `gpt-4.1-mini` | LLM ID for cloud vision. |
| `TEACHABLE_MODEL_PATH` | optional | `./models/teachable_model` | Alternative path to a legacy Teachable model. |

Provider/LLM configuration from the config hub is stored locally (`localStorage.cannabisLLMConfig`) and server-side via `/api/settings/llm`. Multiple profiles can be created, activated, or deleted through `/api/settings/llm/profiles`; selections appear in the Analyzer, Streams, and OTTO. Together with the default Teachable Machine model, values are written to `app-settings.json` so Analyzer, batch/stream endpoints, and OTTO share the same provider settings and survive restarts.

## WiFi broadcast (ottcolab.local)
1. Install dependencies (`zeroconf` ships in `requirements.txt`; run `pip install zeroconf` if you reuse an existing environment).
2. Start `uvicorn app:app --host 0.0.0.0 --port 8000` so the server is reachable on your LAN.
3. Open `http://localhost:8000/config` and scroll to “WiFi Broadcast & ottcolab.local”.
4. Set a hostname (we enforce `.local`) and confirm the port, then click “Enable broadcast”.
5. Devices on the same network can reach `http://ottcolab.local:8000/`. Feedback still welcome at **otcdmin@outlook.com**, Instagram **@ottcouture.eu**, or [Discord](https://discord.gg/GMMSqePfPh).

## Teachable Machine depot (`/TM-models`)
1. Export your Google Teachable Machine project as a **TensorFlow** bundle (`metadata.json`, `model.json`, `weights.bin`) or a **Keras (.h5) bundle** with `keras_model.h5` and `labels.txt`.
2. Open `http://localhost:8000/config` and use the “OTTCOUTURE Teachable Machine Depot” section.
3. After upload the model is stored under `/TM-models/<slug>` and registered in `TM-models/registry.json`.
4. The server converts TFJS exports into a TensorFlow SavedModel (`tensorflowjs` is declared as a dependency). Missing converters or broken bundles return clear errors.
5. The list view lets you mark any model as “Default in Analyzer”. The default is mirrored in `app-settings.json`.
6. If no community model is selected, the Analyzer falls back to `TEACHABLE_MODEL_PATH` (OPENCORE reference).

> Required files: either `metadata.json`, `model.json`, `weights.bin` **or** `keras_model.h5` plus `labels.txt`. Missing parts are rejected during upload.

## API routes
- `GET /` – Analyzer landing page with model selector
- `GET /config` – self-host configurator & TM depot
- `GET /completions` – OTTO grow chat UI
- `POST /analyze` – image + prompt + optional `model_id` + `analysis_mode`
- `POST /api/opencore/analyze-ml` – alias for ML-only calls (same as `/analyze` with `analysis_mode=ml`)
- `POST /api/opencore/analyze-video` – analyze MP4 uploads or library entries, with overlay generation
- `POST /api/opencore/analyze-batch` – multi-image analysis (FormData with `files[]`)
- `POST /api/opencore/share` & `GET /api/opencore/share/{id}` – JSON share service (`/share/{id}` serves the viewer)
- `GET/POST/DELETE /api/uploads*` – list, upload, rename, or delete stored media assets
- `POST /api/completions` – OTTO chat endpoint (`prompt` in JSON body)
- `GET/POST/DELETE /api/opencore/streams*` – manage snapshot/video streams including trigger endpoint
- `GET /tm-models` – registry + default information
- `POST /tm-models/upload` – ZIP upload (`file`, `model_type`, `display_name`)
- `POST /tm-models/default/{model_id}` – set default model
- `DELETE /tm-models/default` – clear default model
- `GET/POST/DELETE /api/settings/llm` – persist provider/prompt configurations
- `GET /network/status`, `POST /network/announce`, `DELETE /network/announce` – mDNS controls

## Documentation in `/doc`
All feature guides ship as static HTML pages served by FastAPI under `/doc`:

- `doc/prompts.html` – prompts & custom presets
- `doc/batch.html` – batch analysis with API examples
- `doc/debug.html` – debug panel
- `doc/api_token_mode.html` – professional mode
- `doc/ui.html` – UI extensions (drag & drop, theme, zoom, JSON fullscreen)
- `doc/export.html` – JSON/PDF/share export
- `doc/home_automation.html` – home-automation guide incl. curl, Python, Node-RED, Home Assistant
- `doc/streams.html` – video & snapshot streams with API calls
- `doc/models.html` – Teachable Machine (easy) and Label Studio/YOLO (pro) workflows
- `doc/raspberry.html` – Raspberry Pi mounting, camera setup, and edge scripting

## Project structure
```
.
├── app.py                # FastAPI service + TM depot + WiFi broadcast + OTTO endpoint
├── static/
│   ├── index.html        # Analyzer UI with model selector
│   ├── completions.html  # OTTO grow chat
│   └── config.html       # Self-host + TM depot console
├── TM-models/            # Versioned Teachable Machine bundles (ZIP uploads)
│   ├── README.md
│   └── registry.json     # maintained at runtime
├── app-settings.json     # Default model & provider settings (created if missing)
├── requirements.txt
├── README.md
└── LICENSE
```

## Feedback & rights
- Brand & rights: **ottcouture.eu** — this is released as OpenCore, but all branding remains with ottcouture.eu.
- Feedback: **otcdmin@outlook.com**, Instagram **@ottcouture.eu**, Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).
- License: [AGPL-3.0](LICENSE). Private use and developer testing are allowed; CSCs and companies must license commercial usage directly with ottcouture.eu.
