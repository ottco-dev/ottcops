# OTTCOUTURE Cannabis Vision OpenCore

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
2. Config Hub inkl. TM-Depot: `http://localhost:8000/config`
3. Discord Crew & Support: [`discord.gg/GMMSqePfPh`](https://discord.gg/GMMSqePfPh)

## Konfiguration
| Variable | Pflicht | Default | Beschreibung |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | bei OpenAI Flow | – | Key für GPT-4.1 mini oder dein bevorzugtes Vision Modell. |
| `OPENAI_GPT_MODEL` | optional | `gpt-4.1-mini` | LLM-ID für Cloud Vision. |
| `TEACHABLE_MODEL_PATH` | optional | `./models/teachable_model` | Alternativer Pfad zu einem Legacy-Teachable-Model. |

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
