# Projekt-Dokumentation

Diese Datei beschreibt detailliert, wie die FastAPI-Anwendung aufgebaut ist, wie die
Modelle vorbereitet werden und wie die Umgebung zu konfigurieren ist.

## 1. Voraussetzungen
- Python 3.10+
- Zugriff auf einen trainierten Teachable Machine Bildklassifikator als TensorFlow
  SavedModel oder Keras-Modell.
- OpenAI API Key mit Zugriff auf ein multimodales Modell (z. B. `gpt-4.1-mini`).

## 2. Installation
1. Repository auschecken und in das Projektverzeichnis wechseln.
2. Virtuelle Umgebung erstellen:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```

## 3. Modelle & Labels
- Primärer Ablageort ist `TM-models/`. Das Upload-Feature entpackt ZIPs (mit `metadata.json`, `model.json`, `weights.bin`) in einen Unterordner und schreibt zusätzlich einen Eintrag in `TM-models/registry.json`.
- Falls kein Eintrag aktiv ist, nutzt der Analyzer `TEACHABLE_MODEL_PATH` (z. B. `./models/teachable_model`).
- Labels werden automatisch aus `metadata.json` gelesen. Fehlen Daten, erzeugt der Service neutrale Bezeichner (`class_1`, `class_2`, …).

## 4. Konfiguration
| Variable | Beschreibung |
| --- | --- |
| `OPENAI_API_KEY` | Pflicht für Cloud-LLMs und den OTTO-Chat. |
| `OPENAI_GPT_MODEL` | Optional. Default ist `gpt-4.1-mini`. |
| `TEACHABLE_MODEL_PATH` | Optionaler Fallback-Pfad für ein lokales TM-Modell. |

Weitere Persistenzdateien:
- `app-settings.json`: speichert das Standardmodell (wird vom `/tm-models/default` Endpunkt gepflegt).
- `network-config.json`: merkt sich den mDNS-Status (Hostname/Port für `ottcolab.local`).

## 5. Starten des Servers
```bash
export OPENAI_API_KEY="sk-..."
uvicorn app:app --host 0.0.0.0 --port 8000
```
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Simple HTML UI: http://localhost:8000/

## 6. HTML-Demo
- `static/index.html`: Analyzer UI mit Modell-Dropdown (ruft `/analyze`).
- `static/config.html`: Konsole für Provider, System-Prompts, TM-Depot und mDNS.
- `static/completions.html`: OTTO Grow Chat (ruft `/api/completions`).

## 7. Fehlerbehandlung
- **400**: fehlender Prompt oder Bild.
- **500**: TensorFlow- oder Preprocessing-Fehler.
- **502**: Probleme beim OpenAI-Aufruf.

## 8. Deployment-Hinweise
- TensorFlow ist speicherintensiv; setzen Sie passende Container-Limits.
- `OPENAI_API_KEY` niemals commiten, sondern per Secret Management bereitstellen.
- Optional: Gunicorn/Uvicorn-Worker über `gunicorn -k uvicorn.workers.UvicornWorker`.

## 9. Weiterentwicklung
- Optionale Auth-Schicht vor `/config`, `/tm-models/*` und `/api/completions`.
- Health-Checks für Teachable-Machine-Modelle (z. B. automatische Tests nach Upload).
- Persistente Provider-Konfiguration (derzeit nur im Browser-Storage).

## 10. Lizenz
- Die Anwendung steht unter der [GNU Affero General Public License v3.0](LICENSE).
- OTTCOUTURE behält sämtliche Rechte am Markennamen und Branding.
- Änderungen oder Erweiterungen müssen offen bleiben, wieder unter der AGPL
  veröffentlicht werden und eine eindeutige Attribution an OTTCOUTURE enthalten.
