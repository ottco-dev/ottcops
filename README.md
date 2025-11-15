# OTTCOUTURE Cannabis Vision OpenCore

OTTCOPS ist der von [ottcouture.eu](https://ottcouture.eu) betriebene Analyzer für Cannabis-Vision. Er kombiniert Teachable-Machine-Modelle mit multimodalen LLMs und liefert strukturierte JSON-Outputs – sachlich, reproduzierbar und vollständig unter OTTCOUTURE-Rechten. Feedback oder neue Modelle gern an **otcdmin@outlook.com**, Instagram **@ottcouture.eu** oder Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).

## Feature Highlights
- 🌿 **FastAPI Core** mit Analyzer, Config Deck, OTTO-Chat (`/completions`) und dokumentierten `/tm-models*` Routen.
- 🧠 **Vision LLM Switchboard** für OpenAI, Ollama oder LM Studio inkl. System-Presetverwaltung.
- 🧪 **Teachable-Machine-Depot** mit ZIP-Uploads (metadata.json, model.json, weights.bin), Registry und Standardauswahl für den Analyzer.
- 🧵 **Model Routing**: Das Frontend kann pro Analyse den gewünschten TM-Slot wählen; die Einstellung wird zusätzlich serverseitig in `app-settings.json` persistiert.
- 🤖 **OTTO Grow Chat** – eigener Screen für kultivierungsrelevante Fragen mit definiertem System Prompt.
- 📡 **WiFi Broadcast Mode** (mDNS/zeroconf) für Hostnamen wie `ottcolab.local` im gesamten WLAN.
OTTCOPS ist unser OpenCore-Playground für nerdige Cannabis Vision Flows, geboren bei [ottcouture.eu](https://ottcouture.eu) und veröffentlicht unter der AGPL. Wir mischen Teachable-Machine-Signale mit multimodalen LLMs, streamen rohe JSON-Outputs und behalten sämtliche Brand-Rechte bei OTTCOUTURE. Credits & Feedback bitte an **otcdmin@outlook.com** oder im Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).

## Feature Highlights
- 🌿 **FastAPI Core** mit `/analyze`, `/docs`, `/config` und den neuen `/tm-models*`-Routen.
- 🧠 **Vision LLM Switchboard**: OpenAI, Ollama oder LM Studio lassen sich live am `/config`-Frontend umstellen.
- 🧪 **Cannabis-Systemprompts & Lightweight-Modelle** für Trichome-Heatmaps, Terpen-Stacks und Glitch-Hunts.
- 📦 **Teachable-Machine-Depot**: ZIP-Uploads (metadata.json, model.json, weights.bin) landen versioniert unter `/TM-models` und werden typisiert (Trichomen vs. Health).
- 🛡️ **Brand Messaging** auf jeder Seite – ottcouture.eu Rechte, Kontaktwege, Discord-CTA.

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

1. Analyzer UI: `http://localhost:8000/`
2. OTTO Grow Chat: `http://localhost:8000/completions`
3. Config Hub inkl. TM-Depot: `http://localhost:8000/config`
4. Discord Crew & Support: [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh)
2. Config Hub inkl. TM-Depot: `http://localhost:8000/config`
3. Discord Crew & Support: [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh)

## Konfiguration
| Variable | Pflicht | Default | Beschreibung |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | bei OpenAI Flow | – | Key für GPT-4.1 mini oder dein bevorzugtes Vision Modell. |
| `OPENAI_GPT_MODEL` | optional | `gpt-4.1-mini` | LLM-ID für Cloud Vision. |
| `TEACHABLE_MODEL_PATH` | optional | `./models/teachable_model` | Alternativer Pfad zu einem Legacy-Teachable-Model. |

Alle UI-Einstellungen landen im Browser (`localStorage.cannabisLLMConfig`). Die Auswahl des Standard-Teachable-Machine-Modells speichert das Backend zusätzlich in `app-settings.json`, damit der Analyzer die Vorgabe auch nach einem Neustart nutzt.

## WiFi Broadcast (ottcolab.local)
1. Installiere die Requirements (wir shippen `zeroconf`, wichtig für mDNS). Falls du ein bestehendes Environment nutzt, führe `pip install zeroconf` aus.
2. Starte `uvicorn app:app --host 0.0.0.0 --port 8000`, sodass der Server im WLAN erreichbar ist.
3. Öffne `http://localhost:8000/config`, scrolle zum Abschnitt „WiFi Broadcast & ottcolab.local“.
4. Hostname setzen (wir erzwingen `.local`) und den Port bestätigen, anschließend „Broadcast aktivieren“ anklicken.
5. Jetzt sollten Smartphones, Tablets und Desktop-Geräte im selben Netzwerk `http://ottcolab.local:8000/` aufrufen können. Feedback bitte weiterhin an **otcdmin@outlook.com**, Instagram **@ottcouture.eu** oder [Discord](https://discord.gg/GMMSqePfPh).

## Teachable Machine Depot (`/TM-models`)
1. Exportiere dein Google Teachable-Machine-Projekt als **TensorFlow** Paket (enthält `metadata.json`, `model.json`, `weights.bin`).
2. Öffne `http://localhost:8000/config` und nutze den Abschnitt „OTTCOUTURE Teachable Machine Depot“.
3. Nach dem Upload landet das Modell unter `/TM-models/<slug>` und wird in `TM-models/registry.json` geführt.
4. Die Listenansicht erlaubt pro Modell den Status „Standard im Analyzer“. Der Standard wird zusätzlich in `app-settings.json` notiert.
5. Wird kein Community-Modell ausgewählt, greift der Analyzer auf `TEACHABLE_MODEL_PATH` (OPENCORE Referenz) zurück.

> Pflichtdateien: `metadata.json`, `model.json`, `weights.bin`. Fehlen Bestandteile, lehnt der Upload ab.

## API Routen
- `GET /` – Analyzer Landing Page mit Modellauswahl
- `GET /config` – Self-Host Konfigurator & TM-Depot
- `GET /completions` – OTTO Grow Chat UI
- `POST /analyze` – Bild + Prompt + optional `model_id`
- `POST /api/completions` – OTTO Chat Endpoint (`prompt` im JSON-Body)
- `GET /tm-models` – Registry + Defaultinformationen
- `POST /tm-models/upload` – ZIP Upload (`file`, `model_type`, `display_name`)
- `POST /tm-models/default/{model_id}` – setzt Standardmodell
- `DELETE /tm-models/default` – entfernt Standardmodell
- `GET /network/status`, `POST /network/announce`, `DELETE /network/announce` – mDNS Steuerung
Alle UI-Einstellungen landen im Browser (`localStorage.cannabisLLMConfig`). Für Self-Hosted Vision-LLMs (Ollama/LM Studio) kannst du Base URL, Model, Keys und unsere Cannabis-Systemprompts direkt übernehmen.

## Teachable Machine Depot (`/TM-models`)
1. Exportiere dein Google Teachable-Machine-Projekt als **TensorFlow** Paket (es enthält `metadata.json`, `model.json`, `weights.bin`).
2. Öffne `http://localhost:8000/config`, scrolle zum Abschnitt „OTTCOUTURE Teachable Machine Depot“.
3. Gib einen Modellnamen an, wähle den Typ:
   - `Trichomen Analyse` für Reifegrad/Qualitäts-Modelle.
   - `Health & Leaf Safety` für Symptom- oder Schadens-Detektoren.
4. Lade die ZIP-Datei hoch. Das Backend extrahiert sie nach `/TM-models/<slug>` und ergänzt `TM-models/registry.json`.
5. Zwei Starter-Slots liegen bereit: du kannst eigene Basismodelle im Repo-Verzeichnis `TM-models/` ablegen und mit dem Upload-Flow überschreiben.

> Wichtig: Jede ZIP muss mindestens `metadata.json`, `model.json` und `weights.bin` enthalten. Fehlende Dateien blocken wir bewusst, damit die Community nur valide Assets sieht.

## API Routen
- `GET /` – Analyzer Landing Page (brandet, Cannabis-Formular)
- `GET /config` – Self-Host Konfigurator & TM-Depot
- `POST /analyze` – Image + Prompt → TM Klassifikation + GPT Antwort
- `GET /tm-models` – Liefert registrierte TM-Modelle samt Metadaten
- `POST /tm-models/upload` – Erwartet `file`, `model_type`, `display_name`

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
├── app.py                # FastAPI Service + TM Depot Uploads
├── static/
│   ├── index.html        # Analyzer UI (OTTCOUTURE Style)
│   └── config.html       # Self-Host + TM Depot Oberfläche
├── TM-models/            # Versionierte Teachable Machine Bundles (ZIP-Uploads)
│   └── README.md         # Hinweise & Slots für Startermodelle
├── requirements.txt
├── README.md
└── LICENSE
```

## Feedback & Rechte
- Brand & Rechte: **ottcouture.eu** – wir veröffentlichen hier bewusst OpenCore, aber behalten sämtliche Markenrechte.
- Feedback: **otcdmin@outlook.com**, Instagram **@ottcouture.eu**, Discord [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh).
- Lizenz: [AGPL-3.0](LICENSE). Bitte alle Forks/Deployments wieder zur Community spiegeln und Credits lassen.
