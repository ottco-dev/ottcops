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
- Ablageort: `./models/teachable_model` (per `TEACHABLE_MODEL_PATH` überschreibbar).
- Muss die vom Export generierte `labels.txt` enthalten. Den Inhalt dieser Datei in die
  Konstante `CLASS_NAMES` in `app.py` übernehmen, damit die Vorhersagen korrekt benannt
  werden.
- Die Modell-Ordnerstruktur entspricht der Standardstruktur eines TensorFlow
  SavedModels (`assets/`, `variables/`, `saved_model.pb`).

## 4. Konfiguration
| Variable | Beschreibung |
| --- | --- |
| `OPENAI_API_KEY` | Pflichtvariable. API Key wird vom offiziellen OpenAI SDK gelesen. |
| `OPENAI_GPT_MODEL` | Optional. Default ist `gpt-4.1-mini`. |
| `TEACHABLE_MODEL_PATH` | Optional. Pfad zum Modellverzeichnis. |

## 5. Starten des Servers
```bash
export OPENAI_API_KEY="sk-..."
uvicorn app:app --host 0.0.0.0 --port 8000
```
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Simple HTML UI: http://localhost:8000/

## 6. HTML-Demo
`static/index.html` enthält ein schlichtes Formular, das `prompt` und `image` sammelt
und als `multipart/form-data` an `/analyze` sendet. Ergebnisse werden als JSON
angezeigt. Die Datei kann individuell angepasst werden.

## 7. Fehlerbehandlung
- **400**: fehlender Prompt oder Bild.
- **500**: TensorFlow- oder Preprocessing-Fehler.
- **502**: Probleme beim OpenAI-Aufruf.

## 8. Deployment-Hinweise
- TensorFlow ist speicherintensiv; setzen Sie passende Container-Limits.
- `OPENAI_API_KEY` niemals commiten, sondern per Secret Management bereitstellen.
- Optional: Gunicorn/Uvicorn-Worker über `gunicorn -k uvicorn.workers.UvicornWorker`.

## 9. Weiterentwicklung
- `CLASS_NAMES` aus einer `labels.txt` lesen statt hart zu kodieren.
- Modelldateien versionieren (z. B. via object storage) und bei Start synchronisieren.
- Authentifizierung vor den Endpunkten ergänzen.

## 10. Lizenz
- Die Anwendung steht unter der [GNU Affero General Public License v3.0](LICENSE).
- OTTCOUTURE behält sämtliche Rechte am Markennamen und Branding.
- Änderungen oder Erweiterungen müssen offen bleiben, wieder unter der AGPL
  veröffentlicht werden und eine eindeutige Attribution an OTTCOUTURE enthalten.
