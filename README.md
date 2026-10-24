# Smart Crop Disease Detection

A local, CPU-based crop disease screening web app with English, Hindi and Marathi interface, a trained MobileNetV2 model, disease reference images, a disease knowledge library, and optional Gemini-generated guidance.

## Run with Docker (recommended)

**Target:** Linux with Docker Engine and Docker Compose. Extract the ZIP, open a terminal in the project folder, and run `docker compose up --build -d`. Then open **http://127.0.0.1:5000**. Check status with `docker compose ps`; view logs with `docker compose logs -f smart-crop-disease`; stop with `docker compose down`. The trained model and labels are supplied in `models/` and mounted into the container.

## Alternative: run from VS Code

Open the extracted folder in VS Code and press **Ctrl+Shift+B**, then select **Smart Crop: Run (automatic setup)** if prompted. The launcher creates a private `.venv`, installs `requirements.txt`, and creates `.env` from `.env.example`. Open **http://127.0.0.1:5000** and stop the task with **Ctrl+C**. Python 3.11 or 3.12 is required for this alternative.

## Gemini API and independent image review

The app uses Gemini Vision to independently review the actual uploaded photo against the classifier's proposed crop/disease. A model score—even a high one—is not treated as proof. If the reviewer disagrees, is uncertain, or cannot run, the app labels the result **not confirmed** and withholds disease-specific guidance. This second opinion is a safeguard, not ground truth; AI can also make mistakes.

Set `GEMINI_API_KEY` in your private `.env` file and ensure Gemini API quota is available. The key is excluded from the ZIP. `GEMINI_VISION_MODEL` optionally overrides the vision model (defaults to `gemini-3.1-flash-lite`). When the key is missing or the review fails, predictions remain unverified instead of being presented as confirmed diagnoses.

## What is included

- Flask app and templates
- Active 43-class MobileNetV2 model and matching labels (38 PlantVillage classes plus 5 cotton classes)
- Gemini Vision second-opinion check; conflicting/unavailable reviews withhold disease-specific guidance
- Docker Compose deployment and health check
- Disease reference gallery and UI styles
- Automatic first-run launcher and VS Code task
- Offline fallback guidance and SQLite cache
- Evaluation and reference documents

Training notebooks, raw datasets, experimental candidate models, temporary uploads, prediction-history databases, virtual environments, and secrets are excluded from the clean runtime package.

## Model limitations

The model predicts from its trained classes; it can misclassify field photos. Treat the result as preliminary screening, not a confirmed diagnosis. Confirm important decisions with a local KVK/agriculture officer. See `MODEL_EVALUATION.md` for evaluation results and limitations.
