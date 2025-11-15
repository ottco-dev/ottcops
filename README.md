# OTTCOUTURE Cannabis Vision OpenCore

OTTCOPS ist der von [ottcouture.eu](https://ottcouture.eu) betriebene Analyzer für Cannabis-Vision. Er kombiniert Teachable-Machine-Modelle mit multimodalen LLMs und liefert strukturierte JSON-Outputs – sachlich, reproduzierbar und vollständig unter OTTCOUTURE-Rechten. Feedback oder neue Modelle gern an **otcdmin@outlook.com**, Instagram **@ottcouture.eu** oder Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).

## Feature Highlights
- 🌿 **FastAPI Core** mit Analyzer, Config Deck, OTTO-Chat (`/completions`) und dokumentierten `/tm-models*` Routen.
- 🧠 **Vision LLM Switchboard** für OpenAI, Ollama oder LM Studio inkl. System-Presetverwaltung, Mehrfach-Profilen und serverseitiger Persistenz für Analyzer, Streams und OTTO.
- 🧪 **Teachable-Machine-Depot** mit ZIP-Uploads (TFJS: metadata.json/model.json/weights.bin oder Keras: keras_model.h5 + labels.txt), Registry und Standardauswahl für den Analyzer.
- 🧵 **Model Routing**: Das Frontend kann pro Analyse den gewünschten TM-Slot wählen; die Einstellung wird zusätzlich serverseitig in `app-settings.json` persistiert.
- 🤖 **OTTO Grow Chat** – eigener Screen für kultivierungsrelevante Fragen mit definiertem System Prompt.
- 📡 **WiFi Broadcast Mode** (mDNS/zeroconf) für Hostnamen wie `ottcolab.local` im gesamten WLAN.
- 📝 **Prompt-Templates** inkl. lokaler Custom-Presets direkt im Analyzer.
- 🗂️ **Batch-Analyse** mit `/api/opencore/analyze-batch`, Tabs pro Bild und Gesamt-Report.
- 🛠️ **Debug-Panel** mit Request-ID, Modellversion und Timings (UI-Toggle + `?debug=1`).
- 🔐 **API-Token-Mode**: Eigene Base-URL + Token, inkl. Code-Beispielen.
- 📤 **Export-Paket**: JSON-Download, PDF-Report sowie Share-Links über `/api/opencore/share` + Viewer (`/share/<id>`).
- 🧷 **ML-only Analysemodus**: `analysis_mode=ml` liefert reine Teachable-Machine-JSONs ohne GPT-Laufzeit.
- 🎥 **Stream-Orchestrierung**: Snapshot/RTSP-Quellen laufen als Hintergrundjobs (5 s Capture, 30 s Batch) und liefern automatische Reports.
- 🔄 **Launch-Update-Check**: Bei jedem Start prüft das Backend gegen `github.com/methoxy000/ottcops` und bietet ein optionales `git pull` an.

## Nutzung & Lizenzpflicht
- Der OPENCORE Analyzer darf ohne weitere Freigabe ausschließlich von privaten Einzelnutzer:innen und Developer:innen zu Test- und Forschungszwecken betrieben werden.
- Cannabis Social Clubs (CSCs) und Unternehmen – egal ob Start-up, MSO oder Dienstleister – müssen vor Einsatz in kommerziellen Projekten direkt mit **ottcouture.eu** eine Lizenz vereinbaren.
- Kontakt für Lizenzen & Partnerschaften: **otcdmin@outlook.com**, Instagram **@ottcouture.eu**, Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).

## Installation im OTTCOUTURE Style
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional wenn du GPT Calls willst
export OPENAI_API_KEY="sk-..."

# Dev-Server starten
uvicorn app:app --reload
```

Beim Start führt der Server automatisch einen Git-Vergleich gegen `https://github.com/methoxy000/ottcops`. Wird ein neuer Commit gefunden, erscheint eine Konsolenabfrage („Jetzt aktualisieren?“). Die Eingabe `y` oder `yes` startet ein `git pull`, jede andere Antwort lässt die vorhandene Version aktiv. Setze `OTTC_SKIP_UPDATE_CHECK=1`, wenn der Check z. B. in CI-Pipelines übersprungen werden soll.

1. Analyzer UI: `http://localhost:8000/`
2. OTTO Grow Chat: `http://localhost:8000/completions`
3. Config Hub inkl. TM-Depot: `http://localhost:8000/config`
4. Discord Crew & Support: [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh)
5. Dokumentation (HTML): `http://localhost:8000/doc/index.html`

## Konfiguration
| Variable | Pflicht | Default | Beschreibung |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | bei OpenAI Flow | – | Key für GPT-4.1 mini oder dein bevorzugtes Vision Modell. |
| `OPENAI_GPT_MODEL` | optional | `gpt-4.1-mini` | LLM-ID für Cloud Vision. |
| `TEACHABLE_MODEL_PATH` | optional | `./models/teachable_model` | Alternativer Pfad zu einem Legacy-Teachable-Model. |

Die Provider-/LLM-Konfiguration aus dem Config Hub wird lokal (`localStorage.cannabisLLMConfig`) und serverseitig via `/api/settings/llm` gespeichert. Mehrere Profile lassen sich über `/api/settings/llm/profiles` anlegen, aktivieren oder löschen; die Auswahl erscheint im Analyzer, bei Streams und in OTTO. Gemeinsam mit dem Standard-Teachable-Machine-Modell landen die Werte in `app-settings.json`, damit Analyzer, Batch-/Stream-Endpunkte und der OTTO-Chat dieselbe Provider-Konfiguration verwenden und nach Neustarts synchron bleiben.

## WiFi Broadcast (ottcolab.local)
1. Installiere die Requirements (wir shippen `zeroconf`, wichtig für mDNS). Falls du ein bestehendes Environment nutzt, führe `pip install zeroconf` aus.
2. Starte `uvicorn app:app --host 0.0.0.0 --port 8000`, sodass der Server im WLAN erreichbar ist.
3. Öffne `http://localhost:8000/config`, scrolle zum Abschnitt „WiFi Broadcast & ottcolab.local“.
4. Hostname setzen (wir erzwingen `.local`) und den Port bestätigen, anschließend „Broadcast aktivieren“ anklicken.
5. Jetzt sollten Smartphones, Tablets und Desktop-Geräte im selben Netzwerk `http://ottcolab.local:8000/` aufrufen können. Feedback bitte weiterhin an **otcdmin@outlook.com**, Instagram **@ottcouture.eu** oder [Discord](https://discord.gg/GMMSqePfPh).

## Teachable Machine Depot (`/TM-models`)
1. Exportiere dein Google Teachable-Machine-Projekt als **TensorFlow** Paket (enthält `metadata.json`, `model.json`, `weights.bin`) oder als **Keras (.h5) Paket** mit `keras_model.h5` und `labels.txt`.
2. Öffne `http://localhost:8000/config` und nutze den Abschnitt „OTTCOUTURE Teachable Machine Depot“.
3. Nach dem Upload landet das Modell unter `/TM-models/<slug>` und wird in `TM-models/registry.json` geführt.
4. Der Server wandelt TFJS-Exporte automatisch in ein TensorFlow SavedModel um (`tensorflowjs` wird hierzu clientseitig mitgeliefert). Fehlende Konverter oder defekte Bundles führen zu einer klaren Fehlermeldung.
5. Die Listenansicht erlaubt pro Modell den Status „Standard im Analyzer“. Der Standard wird zusätzlich in `app-settings.json` notiert.
6. Wird kein Community-Modell ausgewählt, greift der Analyzer auf `TEACHABLE_MODEL_PATH` (OPENCORE Referenz) zurück.

> Pflichtdateien: entweder `metadata.json`, `model.json`, `weights.bin` **oder** `keras_model.h5` plus `labels.txt`. Fehlen Bestandteile, lehnt der Upload ab.

## API Routen
- `GET /` – Analyzer Landing Page mit Modellauswahl
- `GET /config` – Self-Host Konfigurator & TM-Depot
- `GET /completions` – OTTO Grow Chat UI
- `POST /analyze` – Bild + Prompt + optional `model_id` + `analysis_mode`
- `POST /api/opencore/analyze-ml` – Alias für ML-only Calls (identisch zu `/analyze` mit `analysis_mode=ml`)
- `POST /api/opencore/analyze-batch` – Multi-Bild-Analyse (FormData mit `files[]`)
- `POST /api/opencore/share` & `GET /api/opencore/share/{id}` – JSON-Share-Service (`/share/{id}` liefert Viewer)
- `POST /api/completions` – OTTO Chat Endpoint (`prompt` im JSON-Body)
- `GET/POST/DELETE /api/opencore/streams*` – Verwaltung der Snapshot/Video-Streams inkl. Trigger-Endpoint
- `GET /tm-models` – Registry + Defaultinformationen
- `POST /tm-models/upload` – ZIP Upload (`file`, `model_type`, `display_name`)
- `POST /tm-models/default/{model_id}` – setzt Standardmodell
- `DELETE /tm-models/default` – entfernt Standardmodell
- `GET/POST/DELETE /api/settings/llm` – persistiert Provider/Prompt-Konfigurationen im Backend
- `GET /network/status`, `POST /network/announce`, `DELETE /network/announce` – mDNS Steuerung

## Dokumentation im `/doc`-Verzeichnis

Alle geforderten Feature-Guides liegen als statische HTML-Seiten vor und werden über FastAPI unter `/doc` ausgeliefert:

- `doc/prompts.html` – Vorlagen & Custom-Presets
- `doc/batch.html` – Batch-Analyse mit API-Beispielen
- `doc/debug.html` – Debug-Panel
- `doc/api_token_mode.html` – Professional Mode
- `doc/ui.html` – UI-Erweiterungen (Drag&Drop, Theme, Zoom, JSON-Fullscreen)
- `doc/export.html` – JSON/PDF/Share-Export
- `doc/home_automation.html` – Home-Automation Guide inkl. curl, Python, Node-RED, Home Assistant
- `doc/streams.html` – Video- & Snapshot-Streams inkl. API-Aufrufen
- `doc/models.html` – Teachable-Machine (Easy) und Label-Studio/YOLO (Pro) Workflows
- `doc/raspberry.html` – Raspberry-Pi-Montage, Kamera-Setup und Edge-Scripting

## Projektstruktur
```
.
├── app.py                # FastAPI Service + TM Depot + WiFi Broadcast + OTTO Endpoint
├── static/
│   ├── index.html        # Analyzer UI inkl. Modellauswahl
│   ├── completions.html  # OTTO Grow Chat Oberfläche
│   └── config.html       # Self-Host + TM Depot Oberfläche
├── TM-models/            # Versionierte Teachable-Machine Bundles (ZIP-Uploads)
│   ├── README.md
│   └── registry.json     # wird zur Laufzeit gepflegt
├── app-settings.json     # Standardmodell (wird bei Bedarf erzeugt)
├── requirements.txt
├── README.md
└── LICENSE
```

## Feedback & Rechte
- Brand & Rechte: **ottcouture.eu** – wir veröffentlichen hier bewusst OpenCore, aber behalten sämtliche Markenrechte.
- Feedback: **otcdmin@outlook.com**, Instagram **@ottcouture.eu**, Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).
- Lizenz: [AGPL-3.0](LICENSE). Bitte alle Forks/Deployments wieder zur Community spiegeln und Credits lassen.
