# AgriSage — Smart Crop Disease Detection

AgriSage is a local web application for preliminary crop-leaf disease screening. It includes a trained image-classification model, disease reference images and guidance, English/Hindi/Marathi interface support, and optional Gemini-powered image review and guidance.

## Quick setup (no Docker required)

### Windows

1. Install **Python 3.11 or 3.12** from [python.org](https://www.python.org/downloads/). During installation, enable **Add Python to PATH**.
2. Install [Visual Studio Code](https://code.visualstudio.com/) and the Microsoft Python extension.
3. Open this repository folder in VS Code.
4. Open **Terminal → New Terminal** and run:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
   ```

The first run creates a private `.venv`, installs the required Python packages, and creates `.env` from `.env.example`. It may take several minutes and needs an internet connection. Later runs reuse the installed environment.

### Linux / macOS

Install Python 3.11 or 3.12, open the project folder in VS Code, then run in its terminal:

```bash
bash ./setup.sh
```

When the server starts, open **http://127.0.0.1:5000** in your browser. Press **Ctrl+C** in the terminal to stop it. You can start it again using the same command.

## Optional Gemini setup

The app can use Gemini for independent image review and extra disease guidance. It also has built-in guidance for offline fallback. To enable Gemini, open the local `.env` file created on first run and set `GEMINI_API_KEY` to your own key. Keep `.env` private: it is ignored by Git and must never be committed or shared. Gemini usage depends on API availability and quota.

## What is included

- Flask web application, templates, and static assets
- Trained model and matching label files required for inference
- Disease reference gallery and offline guidance
- First-run setup scripts for Windows, Linux, and macOS
- Optional Gemini configuration example
- Render deployment configuration

Training datasets, temporary uploads, local databases, virtual environments, and secret keys are not included.

## Important limitations

Predictions are preliminary screening, not a guaranteed diagnosis. The classifier can make mistakes, especially on field photos or unfamiliar conditions. Confirm important crop-treatment decisions with a local KVK or agriculture officer.

## Project deployment

The hosted demo is deployed separately on Render. Local setup does not change the hosted website. See the Render dashboard available to the project owner for deployment settings.
