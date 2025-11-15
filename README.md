# CALYX.IO OpenCore Component

Dies ist unsere erste OpenCore-Komponente für das CALYX.IO Projekt. Sie verbindet
ein Teachable-Machine-Modell mit GPT und erlaubt der Community, die Pipeline lokal
zu testen, zu erweitern und eigene Modelle einzubinden.

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

## Eigene Modelle trainieren

### 1. Datensätze erfassen und labeln
* Nutzt [Label Studio](https://labelstud.io/) oder ein anderes Open-Source-Tool.
* Für Trichome-Analysen empfehlen wir mindestens zwei Klassen, z. B. `reif`
  und `unreif`. Für Mangel-Erkennung können Kategorien wie `Stickstoffmangel`,
  `Calciummangel` und `gesund` hilfreich sein.
* Beim Labeln in Label Studio:
  - Verwendet hochauflösende Bilder, zoomt auf relevante Bildbereiche und setzt die
    Bounding Boxes knapp um die sichtbaren Trichome oder Symptome.
  - Achtet darauf, dass jedes Bild nur die passende Kategorie erhält. Im Zweifel
    ein zusätzliches Label `unsicher` anlegen, das später aus dem Trainingssatz
    gefiltert werden kann.
  - Dokumentiert Beispielbilder pro Klasse im Projekt, damit Team-Mitglieder
    konsistent labeln.

### 2. Daten exportieren
1. Export aus Label Studio als „YOLO“ oder „COCO“ Dataset.
2. Nutzt `label-studio-converter` oder `roboflow` nur, wenn ihr Metadaten benötigt.

### 3. Modell in Teachable Machine trainieren
1. Öffnet [Teachable Machine](https://teachablemachine.withgoogle.com/).
2. Importiert die gelabelten Bilder pro Klasse.
3. Für Trichom-Erkennung empfiehlt sich „Image Project“ → „Standard Image Model“. Für
   Mangel-Klassifikation könnt ihr das gleiche Template nutzen.
4. Startet das Training mit Standardparametern, testet direkt im Browser und exportiert
   anschließend als TensorFlow Keras.

### 4. Modell in CALYX.IO einbinden
1. Speichert den Export unter `./models/teachable_model` oder setzt
   `TEACHABLE_MODEL_PATH` auf euer Verzeichnis.
2. Aktualisiert die `labels.txt` entsprechend eurer Klassen.
3. Startet den Dienst (siehe "Running the API") und prüft eure Klasse im Web-UI.

### 5. Qualität sichern
* Nutzt ein dediziertes Validation-Set (mindestens 20 % der Daten).
* Vergleicht Predictions mit bekannten Beispielen und passt Labels an, wenn die
  Fehlklassifizierung auf inkonsistente Annotationen zurückgeht.
* Dokumentiert jede Iteration in DOKU.md oder einem separaten CHANGELOG, um das
  OpenCore-Prinzip transparent zu halten.
Set `OPENAI_API_KEY=dummy` and mock the OpenAI client (e.g., monkeypatch
`call_gpt_with_image_context`). TensorFlow predictions can be validated with sample
images via Swagger UI or the HTML form.

## License & Usage Rights
This project is released under the [GNU Affero General Public License v3.0](LICENSE).
OTTCOUTURE retains all rights to the OTTCOUTURE branding, and every modification or
derivative must remain open source, be distributed under the same AGPL terms, and
include clear attribution to OTTCOUTURE.
