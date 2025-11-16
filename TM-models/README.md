# OTTCOUTURE TM Slots

Hier landen alle hochgeladenen Teachable-Machine-ZIPs aus dem Konfigurator (`/config`). Jede ZIP muss entweder ein TFJS-Paket oder ein Keras-Paket enthalten:

- **TFJS:** `metadata.json`, `model.json`, `weights.bin`
- **Keras:** `keras_model.h5`, `labels.txt` (optional zusätzlich `metadata.json`)

Wir legen das entpackte Modell in einem eigenen Unterordner (Slug des angegebenen Namens) ab und dokumentieren es zusätzlich in `registry.json`. Über die Config-Seite lässt sich jedes Modell als Standard markieren (wir speichern die Auswahl serverseitig, damit der Analyzer automatisch darauf zugreift).

Community-Hinweis:
- Rechte & Branding: [ottcouture.eu](https://ottcouture.eu)
- Feedback: otcdmin@outlook.com · Instagram @ottcouture.eu · Discord https://discord.gg/GMMSqePfPh
