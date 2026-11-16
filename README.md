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

Install Python 3.11 or 3.12, open the project folder in VS Code, then run:

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

## Free Render hosting for the 30-day college demo

AgriSage is configured for Render's **Free** web-service plan. It uses SQLite and does not need a separate paid database or persistent disk. Render Free has temporary storage, so accounts, scan history, and activity logs can reset after a restart, redeploy, or service spin-down. This is acceptable for the short college demonstration; do not treat the hosted demo as permanent production storage.

The two administrator accounts are recreated automatically whenever the app initializes a fresh database. Render stores only precomputed Werkzeug password hashes in its private Environment settings; the app never receives administrator passwords in plaintext. The app also uses hashed passwords for all farmer accounts.

Render may spin down while idle, so the first page load after inactivity can take longer. Uploaded images and SQLite records may be cleared when its temporary filesystem is reset.

The hosted demo is deployed separately on Render. Local setup does not change the hosted website until code is pushed to the connected GitHub branch.

## Optional Docker setup (Windows)

Docker is optional; the regular Python setup above works without it.

1. Install Docker Desktop for Windows with the WSL 2 backend. Windows virtualization must be enabled in the computer's BIOS/UEFI, and WSL 2 must be installed.
2. Open Docker Desktop and wait until its engine reports that it is running.
3. From the AgriSage project folder, open PowerShell and run:

   ```powershell
   docker compose -f compose.yaml up --build -d
   ```

4. Open **http://localhost:5000**.
5. View startup logs with `docker compose -f compose.yaml logs -f`.
6. Stop the app with `docker compose -f compose.yaml down`.

The Compose setup keeps the SQLite database and uploaded images in named Docker volumes. To create the two administrators after the first container start, run `docker compose -f compose.yaml exec agrisage python setup_admins.py`; the password prompts are hidden. To enable Gemini, add your own `GEMINI_API_KEY` to the project's local `.env` file before starting Compose. Never commit or share that file.

The first Docker build downloads the Python/TensorFlow base image and dependencies and can use several gigabytes of disk space. Keep adequate free disk space before building.

## Accounts, farmer dashboard, and administration

Disease detection requires a signed-in account. Farmers can register with a name, email, and password (minimum 8 characters), then sign in to use the scanner. Passwords are stored as Werkzeug password hashes, never as plaintext.

- **Farmer dashboard:** shows the signed-in farmer's account details and their own recent disease scans only.
- **Admin dashboard:** available at `/admin`; lists registered accounts, active/inactive status, scan totals, and the latest activity log. Only accounts with the server-side `admin` role can access it.
- **Admin activity log:** records account registration, successful and failed sign-in, sign-out, disease predictions, and account status changes. Activity logs are administrator-only.
- **Account controls:** administrators can activate or deactivate farmer accounts. Admin accounts cannot be changed through this control.
- **Account history:** each farmer gets a stable ID such as `AG-000123`. They can sign in with either their email address or account ID and password to retrieve their own saved scan history.

### Secure administrator setup

Administrator passwords are never placed in `.env`, source code, Docker configuration, or GitHub. To create or reset the two local administrators, run:

```powershell
.\.venv\Scripts\python.exe setup_admins.py
```

The script prompts privately for the two administrator email addresses and passwords (password input is not echoed). It hashes each password with Werkzeug before writing the account to the database. The default email prompts are `schinche735@gmail.com` and `shantanu056@gmail.com`; use different, unique passwords of at least 12 characters. Existing account IDs and linked scan history are preserved when an existing email is promoted.

The two administrator accounts have already been provisioned in the local development database on the connected Windows PC. Their passwords are stored as hashes in SQLite. A Windows DPAPI-protected recovery copy is excluded from Git. For Render, set `ADMIN_EMAIL`, `ADMIN_PASSWORD_HASH`, `ADMIN2_EMAIL`, and `ADMIN2_PASSWORD_HASH` in the private service Environment settings. The two hash values are generated locally from the DPAPI-protected passwords. Never put plaintext administrator passwords in environment variables, source code, or GitHub.

Existing databases are upgraded automatically with the `users`, `activity_logs`, and `prediction_history` tables when the app starts.
