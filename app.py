"""Smart Crop Disease Detection with local image classification and cached AI guidance."""
import json, os, re, sqlite3, threading, urllib.request, uuid, secrets, hmac
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from PIL import Image, UnidentifiedImageError
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from research import research_disease, offline_guidance
from vision_verification import verify_diagnosis_image

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
EXPANDED_MODEL_PATH = MODEL_DIR / "plantdisease_expanded_frozen_old_candidate.keras"
EXPANDED_LABELS_PATH = MODEL_DIR / "labels_plantdisease_expanded_frozen_old_candidate.json"
ALL_MODEL_PATH = MODEL_DIR / "plantdisease_all38_mnv2.keras"
ALL_LABELS_PATH = MODEL_DIR / "labels_plantdisease_all38.json"
TEN_MODEL_PATH = MODEL_DIR / "plantdisease_10_mnv2.keras"
TEN_LABELS_PATH = MODEL_DIR / "labels_plantdisease_10.json"
PV_MODEL_PATH = MODEL_DIR / "best_mnv2_pv_original.keras"
PV_LABELS_PATH = MODEL_DIR / "labels_pv_original.json"
UPLOAD_DIR = BASE / "static" / "uploads"
DB_PATH = BASE / "data" / "predictions.db"
MODEL_URL = "https://huggingface.co/ashikaasriarun/MobileNet-V2-PlantVillage/resolve/main/best_mnv2_pv_original.keras"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_urlsafe(48)
login_csrf_serializer = URLSafeTimedSerializer(app.secret_key, salt="agrisage-login-csrf")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("COOKIE_SECURE", "1" if os.environ.get("RENDER") else "0") == "1"


class TabScopedSessionCookieMiddleware:
    """Give each /w/<workspace>/ URL its own path-scoped Flask session cookie.

    The workspace ID is only a routing context, not an authentication secret.
    The HttpOnly session cookie remains the authentication credential and is
    sent only to that workspace path by the browser.
    """
    WORKSPACE_PATH = re.compile(r"^/w/([A-Za-z0-9_-]{32,64})(/.*)?$")

    def __init__(self, wrapped_app, flask_app):
        self.wrapped_app = wrapped_app
        self.flask_app = flask_app

    def cookie_name(self, workspace_id):
        prefix = "__Secure-" if self.flask_app.config.get("SESSION_COOKIE_SECURE") else ""
        return f"{prefix}AgriSage-{workspace_id}"

    @staticmethod
    def _cookie_pairs(header):
        pairs = []
        for part in (header or "").split(";"):
            if "=" not in part:
                continue
            name, value = part.strip().split("=", 1)
            if name:
                pairs.append((name, value))
        return pairs

    def __call__(self, environ, start_response):
        original_path = environ.get("PATH_INFO", "/")
        match = self.WORKSPACE_PATH.match(original_path)
        workspace_id = match.group(1) if match else None

        if workspace_id:
            prefix = f"/w/{workspace_id}"
            environ["agrisage_workspace_id"] = workspace_id
            environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + prefix
            environ["PATH_INFO"] = match.group(2) or "/"

            selected_name = self.cookie_name(workspace_id)
            selected_value = None
            filtered = []
            for name, value in self._cookie_pairs(environ.get("HTTP_COOKIE", "")):
                if name == selected_name:
                    selected_value = value
                elif name == "session" or name.startswith("AgriSage-") or name.startswith("__Secure-AgriSage-"):
                    continue
                else:
                    filtered.append((name, value))
            if selected_value is not None:
                filtered.append(("session", selected_value))
            environ["HTTP_COOKIE"] = "; ".join(f"{name}={value}" for name, value in filtered)

        def wrapped_start_response(status, headers, exc_info=None):
            new_workspace = environ.get("agrisage_new_workspace_id")
            target_workspace = workspace_id or new_workspace
            rewritten = []
            session_cookie_seen = False
            for header_name, header_value in headers:
                if header_name.lower() == "set-cookie" and header_value.lower().startswith("session=") and target_workspace:
                    session_cookie_seen = True
                    name = self.cookie_name(target_workspace)
                    value = re.sub(r"^session=", name + "=", header_value, count=1, flags=re.IGNORECASE)
                    scoped_path = f"/w/{target_workspace}"
                    if re.search(r";\s*path=", value, flags=re.IGNORECASE):
                        value = re.sub(r";\s*path=[^;]*", "; Path=" + scoped_path, value, count=1, flags=re.IGNORECASE)
                    else:
                        value += "; Path=" + scoped_path
                    rewritten.append((header_name, value))
                else:
                    rewritten.append((header_name, header_value))

            # Remove the old shared root cookie after the first successful
            # workspace login. Existing workspaces keep their own scoped cookies.
            if new_workspace and session_cookie_seen:
                deletion = "session=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"
                if self.flask_app.config.get("SESSION_COOKIE_SECURE"):
                    deletion += "; Secure"
                rewritten.append(("Set-Cookie", deletion))
            return start_response(status, rewritten, exc_info)

        return self.wrapped_app(environ, wrapped_start_response)


app.wsgi_app = TabScopedSessionCookieMiddleware(app.wsgi_app, app)


def csrf_token():
    # Root login/register forms use a signed, short-lived per-render token.
    # It does not depend on the shared root cookie, so two login pages opened
    # before either one is submitted remain independently valid.
    if not request.environ.get("agrisage_workspace_id"):
        return login_csrf_serializer.dumps(secrets.token_urlsafe(32))
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": csrf_token}


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        supplied = request.form.get("_csrf_token", "")
        if request.environ.get("agrisage_workspace_id"):
            expected = session.get("_csrf_token", "")
            valid = bool(expected and supplied and hmac.compare_digest(expected, supplied))
        else:
            try:
                login_csrf_serializer.loads(supplied, max_age=3600)
                valid = True
            except (BadSignature, SignatureExpired):
                # Compatibility for existing local tests and pre-workspace forms.
                expected = session.get("_csrf_token", "")
                valid = bool(expected and supplied and hmac.compare_digest(expected, supplied))
        if not valid:
            abort(400, description="The form expired or could not be verified. Refresh and try again.")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
for path in (UPLOAD_DIR, MODEL_DIR, DB_PATH.parent): path.mkdir(parents=True, exist_ok=True)
_model = None
_labels = None
_lock = threading.Lock()

# Ten diseases plus a healthy class for each of the three supported crops.
TARGET_LABELS = [
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]
CROP_NAMES = {
    "Corn_(maize)": {"en": "Maize / corn", "hi": "मक्का", "mr": "मका"},
    "Potato": {"en": "Potato", "hi": "आलू", "mr": "बटाटा"},
    "Tomato": {"en": "Tomato", "hi": "टमाटर", "mr": "टोमॅटो"},
    "Apple": {"en": "Apple", "hi": "सेब", "mr": "सफरचंद"},
    "Blueberry": {"en": "Blueberry", "hi": "ब्लूबेरी", "mr": "ब्लूबेरी"},
    "Cherry_(including_sour)": {"en": "Cherry", "hi": "चेरी", "mr": "चेरी"},
    "Grape": {"en": "Grape", "hi": "अंगूर", "mr": "द्राक्षे"},
    "Orange": {"en": "Orange", "hi": "संतरा", "mr": "संत्रे"},
    "Peach": {"en": "Peach", "hi": "आड़ू", "mr": "पीच"},
    "Pepper,_bell": {"en": "Bell pepper", "hi": "शिमला मिर्च", "mr": "ढोबळी मिरची"},
    "Raspberry": {"en": "Raspberry", "hi": "रास्पबेरी", "mr": "रास्पबेरी"},
    "Soybean": {"en": "Soybean", "hi": "सोयाबीन", "mr": "सोयाबीन"},
    "Wheat": {"en": "Wheat", "hi": "गेहूँ", "mr": "गहू"},
    "Tur": {"en": "Tur / pigeonpea", "hi": "अरहर / तूर", "mr": "तूर / तूरडाळ"},
    "Cotton": {"en": "Cotton", "hi": "कपास", "mr": "कापूस"},
    "Squash": {"en": "Squash (gourd)", "hi": "स्क्वैश (लौकी परिवार)", "mr": "स्क्वॅश (भोपळा वर्ग)"},
    "Strawberry": {"en": "Strawberry", "hi": "स्ट्रॉबेरी", "mr": "स्ट्रॉबेरी"},
}
# Soybean, wheat, and tur are available only when the expanded trained model is installed.
# crop_options_for_model filters these choices against the active model labels.
CROPS = {
    "auto": {"en": "Let the model choose", "hi": "मॉडल को चुनने दें", "mr": "मॉडेलला निवडू द्या"},
    **{key: value for key, value in CROP_NAMES.items() if key not in {"Blueberry", "Raspberry"}},
}
LANG = {"en": "English", "hi": "Hindi", "mr": "Marathi"}


class CompatibleConnection:
    """Keep the existing qmark SQL calls working with both SQLite and PostgreSQL."""
    def __init__(self, connection, backend):
        self.connection = connection
        self.backend = backend

    def execute(self, query, params=()):
        if self.backend == "postgres":
            query = query.replace("?", "%s")
        return self.connection.execute(query, params)

    def commit(self):
        return self.connection.commit()

    def rollback(self):
        return self.connection.rollback()

    def close(self):
        return self.connection.close()


@contextmanager
def db_connection():
    """Use Render PostgreSQL in production, with SQLite for local development."""
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if database_url:
        import psycopg
        from psycopg.rows import dict_row
        connection = CompatibleConnection(
            psycopg.connect(database_url, row_factory=dict_row), "postgres"
        )
    else:
        raw_connection = sqlite3.connect(DB_PATH)
        raw_connection.row_factory = sqlite3.Row
        connection = CompatibleConnection(raw_connection, "sqlite")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def database_backend():
    return "postgres" if os.environ.get("DATABASE_URL", "").strip() else "sqlite"


def init_db():
    postgres = database_backend() == "postgres"
    identity = "BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY" if postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
    email_column = "TEXT NOT NULL" if postgres else "TEXT NOT NULL COLLATE NOCASE UNIQUE"
    with db_connection() as con:
        con.execute(f"""CREATE TABLE IF NOT EXISTS disease_knowledge(
            id {identity},
            crop TEXT NOT NULL,
            disease TEXT NOT NULL,
            lang TEXT NOT NULL,
            description TEXT NOT NULL,
            prevention TEXT NOT NULL,
            sources_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(crop, disease, lang)
        )""")
        con.execute(f"""CREATE TABLE IF NOT EXISTS users(
            id {identity}, name TEXT NOT NULL,
            email {email_column}, password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user', is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL)""")
        con.execute(f"""CREATE TABLE IF NOT EXISTS activity_logs(
            id {identity}, user_id BIGINT,
            email TEXT, action TEXT NOT NULL, details TEXT, created_at TEXT NOT NULL)""")
        con.execute(f"""CREATE TABLE IF NOT EXISTS prediction_history(
            id {identity}, user_id BIGINT NOT NULL,
            crop TEXT, disease TEXT, confidence REAL, image_name TEXT, created_at TEXT NOT NULL)""")
        if postgres:
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_email_lower_idx ON users (LOWER(email))")

    # Recreate the configured admin accounts on a fresh database using
    # precomputed Werkzeug hashes only. Plaintext passwords are never read here.
    admin_pairs = (
        (os.environ.get("ADMIN_EMAIL", "").strip().lower(),
         os.environ.get("ADMIN_PASSWORD_HASH", "").strip()),
        (os.environ.get("ADMIN2_EMAIL", "").strip().lower(),
         os.environ.get("ADMIN2_PASSWORD_HASH", "").strip()),
    )
    for email, password_hash in admin_pairs:
        if not email or not password_hash:
            continue
        if not (password_hash.startswith("scrypt:") or password_hash.startswith("pbkdf2:")):
            raise RuntimeError("Admin bootstrap requires a Werkzeug password hash, never a plaintext password.")
        with db_connection() as con:
            existing = con.execute("SELECT id FROM users WHERE LOWER(email)=LOWER(?)", (email,)).fetchone()
            if existing:
                con.execute(
                    "UPDATE users SET password_hash=?,role='admin',is_active=1 WHERE id=?",
                    (password_hash, existing["id"]),
                )
            else:
                con.execute(
                    "INSERT INTO users(name,email,password_hash,role,is_active,created_at) VALUES(?,?,?,?,?,?)",
                    (email.split("@", 1)[0], email, password_hash, "admin", 1,
                     datetime.now().isoformat(timespec="seconds")),
                )


def current_user():
    # A workspace request reads only the session cookie scoped to that URL.
    # Root-level sessions are accepted only for migration compatibility; a
    # successful new login clears that legacy shared cookie.
    uid = session.get("user_id")
    if not uid:
        return None
    with db_connection() as con:
        row = con.execute("SELECT id,name,email,role,is_active,created_at FROM users WHERE id=?", (uid,)).fetchone()
    if not row or not row["is_active"]:
        session.clear()
        return None
    return dict(row)


def log_activity(user_id, email, action, details=""):
    with db_connection() as con:
        con.execute("INSERT INTO activity_logs(user_id,email,action,details,created_at) VALUES(?,?,?,?,?)",
            (user_id, email, action, details[:500], datetime.now().isoformat(timespec="seconds")))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect("/login")
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if not user:
            return redirect("/login")
        if user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapped


KNOWLEDGE_CACHE_DAYS = 90


def get_cached_guidance(crop, disease, lang):
    with db_connection() as con:
        row = con.execute(
            "SELECT description, prevention, sources_json, updated_at FROM disease_knowledge WHERE crop=? AND disease=? AND lang=?",
            (crop, disease, lang),
        ).fetchone()
    if not row:
        return None
    # Ignore the old generic "no familiar pattern" text that was incorrectly cached
    # for real disease classes. It is not a disease description.
    description_text = (row["description"] or "").strip().lower()
    obsolete_fallbacks = (
        "the classifier did not find a familiar disease pattern",
        "मॉडल को इस तस्वीर में कोई परिचित रोग पैटर्न नहीं मिला",
        "या प्रतिमेत मॉडेलला ओळखीचा रोग-पॅटर्न आढळला नाही",
    )
    if any(description_text.startswith(text) for text in obsolete_fallbacks):
        return None
    try:
        updated = datetime.fromisoformat(row["updated_at"])
        if (datetime.now() - updated).days >= KNOWLEDGE_CACHE_DAYS:
            return None
        sources = json.loads(row["sources_json"] or "[]")
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return {
        "status": "cached", "message": "", "description": row["description"],
        "prevention": row["prevention"], "sources": sources, "cached": True,
        "updated_at": row["updated_at"],
    }


def save_guidance(crop, disease, lang, guidance):
    # Cache reusable disease knowledge, including offline fallback guidance.
    if guidance.get("status") not in ("ok", "fallback") or not guidance.get("description") or not guidance.get("prevention"):
        return
    now = datetime.now().isoformat(timespec="seconds")
    params = (crop, disease, lang, guidance["description"], guidance["prevention"],
              json.dumps(guidance.get("sources", []), ensure_ascii=False), now)
    with db_connection() as con:
        if database_backend() == "mysql":
            con.execute(
                """INSERT INTO disease_knowledge(crop,disease,lang,description,prevention,sources_json,updated_at)
                   VALUES(?,?,?,?,?,?,?) ON DUPLICATE KEY UPDATE description=VALUES(description),
                   prevention=VALUES(prevention), sources_json=VALUES(sources_json), updated_at=VALUES(updated_at)""",
                params,
            )
        else:
            con.execute(
                """INSERT INTO disease_knowledge(crop,disease,lang,description,prevention,sources_json,updated_at)
                   VALUES(?,?,?,?,?,?,?) ON CONFLICT(crop,disease,lang) DO UPDATE SET
                   description=excluded.description, prevention=excluded.prevention,
                   sources_json=excluded.sources_json, updated_at=excluded.updated_at""",
                params,
            )


def disease_guidance(crop, disease, lang, fast=False):
    try:
        cached = get_cached_guidance(crop, disease, lang)
    except Exception:
        cached = None
    if cached:
        return cached
    if fast:
        # Avoid a network round-trip during a scan; use the local disease knowledge.
        fresh = offline_guidance(crop, disease, lang)
    else:
        fresh = research_disease(crop, disease, lang)
    if fresh.get("status") in ("ok", "fallback") and fresh.get("description") and fresh.get("prevention"):
        save_guidance(crop, disease, lang, fresh)
        fresh["cached"] = False
        fresh["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return fresh

def cleanup_old_uploads(max_age_hours=24):
    """Keep recent result images available, but don't retain a permanent image history."""
    cutoff = datetime.now().timestamp() - max_age_hours * 3600
    for path in UPLOAD_DIR.glob("*.jpg"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            app.logger.debug("Could not clean old upload %s", path.name)


def validate_plant_photo(image):
    """Conservative pre-screen for unusable or obviously unrelated uploads.

    This is a lightweight quality/vegetation heuristic, not a trained plant detector;
    ambiguous photos are not guaranteed to be caught.
    Returns (accepted, reason).
    """
    import numpy as np

    if image.width < 96 or image.height < 96:
        return False, "small"
    sample = image.convert("RGB").resize((128, 128))
    rgb = np.asarray(sample, dtype=np.uint8)
    gray = np.asarray(sample.convert("L"), dtype=np.float32)
    if float(gray.std()) < 9.0:
        return False, "detail"
    # Reject nearly textureless images and heavily blurred frames.
    edge_energy = float((np.abs(np.diff(gray, axis=0)).mean() +
                         np.abs(np.diff(gray, axis=1)).mean()) / 2.0)
    if edge_energy < 2.0:
        return False, "blur"

    hsv = np.asarray(sample.convert("HSV"), dtype=np.uint8)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # PIL hue is 0..255. This broad range covers many green, yellow and brown
    # plant tissues while avoiding a green-only rule that rejects diseased leaves.
    plant_tone = ((hue >= 4) & (hue <= 116) &
                  (saturation >= 35) & (value >= 25) & (value <= 250))
    if float(plant_tone.mean()) < 0.055:
        return False, "plant"
    return True, ""


def get_model():
    """Prefer the expanded model, then the focused model, then the original pretrained fallback."""
    global _model, _labels
    if _model is not None: return _model, _labels
    with _lock:
        if _model is not None: return _model, _labels
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        if EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists():
            model_path, labels_path, model_kind = EXPANDED_MODEL_PATH, EXPANDED_LABELS_PATH, "expanded55"
        elif ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists():
            model_path, labels_path, model_kind = ALL_MODEL_PATH, ALL_LABELS_PATH, "all38"
        elif TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists():
            model_path, labels_path, model_kind = TEN_MODEL_PATH, TEN_LABELS_PATH, "focused"
        else:
            model_path, labels_path, model_kind = PV_MODEL_PATH, PV_LABELS_PATH, "legacy"
        if not model_path.exists():
            tmp = model_path.with_suffix(".download")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(model_path)
        _labels = json.loads(labels_path.read_text(encoding="utf-8"))
        _model = tf.keras.models.load_model(str(model_path), compile=False)
        _model._crop_preprocess = (lambda x: x) if model_kind in ("expanded55", "all43", "all38", "focused") else preprocess_input
        _model._all_class_model = model_kind in ("expanded55", "all43", "all38")
        _model._ten_class_model = model_kind == "focused"
        _model._model_path = str(model_path)
        # Build the TensorFlow execution graph before the first visitor uploads a photo.
        import numpy as np
        _model(np.zeros((1, 160, 160, 3), dtype=np.float32), training=False)
    return _model, _labels


def pretty(label):
    raw_crop, _, disease = label.partition("___")
    crop = CROP_NAMES.get(raw_crop, {}).get("en", raw_crop.replace("_", " ").replace(",", ""))
    disease = disease.replace("_", " ").strip()
    if disease.lower() == "healthy": disease = "Healthy"
    return crop, disease


def label_allowed(label):
    if label in TARGET_LABELS:
        return True
    # Keep label validation aligned with whichever expanded classifier is active.
    for labels_path in (EXPANDED_LABELS_PATH, ALL_LABELS_PATH, TEN_LABELS_PATH, PV_LABELS_PATH):
        try:
            if labels_path.exists() and label in json.loads(labels_path.read_text(encoding="utf-8")):
                return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


def reference_image_name(label):
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"disease_images/plantvillage_{slug}.jpg"


def crop_options_for_model(lang):
    if EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists():
        labels_path = EXPANDED_LABELS_PATH
    elif ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists():
        labels_path = ALL_LABELS_PATH
    elif TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists():
        labels_path = TEN_LABELS_PATH
    else:
        labels_path = None
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path else TARGET_LABELS
    except (OSError, json.JSONDecodeError):
        labels = TARGET_LABELS
    supported = {label.partition("___")[0] for label in labels if "___" in label and not label.endswith("___healthy")}
    return [(key, value[lang]) for key, value in CROPS.items() if key == "auto" or key in supported]


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(name) < 2 or len(name) > 120 or "@" not in email or len(password) < 8:
            flash("Enter your name, a valid email and a password with at least 8 characters.", "error")
            return render_template("register.html")
        try:
            with db_connection() as con:
                con.execute("INSERT INTO users(name,email,password_hash,role,is_active,created_at) VALUES(?,?,?,?,?,?)",
                    (name, email, generate_password_hash(password), "user", 1, datetime.now().isoformat(timespec="seconds")))
                row = con.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            log_activity(row["id"], email, "registration", "New farmer account created")
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except Exception:
            app.logger.exception("Registration failed")
            flash("That email may already be registered. Please try another.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    # Account workspaces have their own path-scoped cookie. Account switching
    # always starts at the root login page so it creates a fresh workspace.
    if request.environ.get("agrisage_workspace_id"):
        return redirect("/login")
    active_user = None
    if request.method == "POST":
        identifier = request.form.get("email", "").strip()
        email = identifier.lower()
        password = request.form.get("password", "")
        with db_connection() as con:
            if re.fullmatch(r"AG-\d{6,}", identifier.upper()):
                account_id = int(identifier[3:])
                row = con.execute("SELECT * FROM users WHERE id=?", (account_id,)).fetchone()
            else:
                row = con.execute("SELECT * FROM users WHERE LOWER(email)=LOWER(?)", (email,)).fetchone()
        if row and row["is_active"] and check_password_hash(row["password_hash"], password):
            session.clear()
            session["user_id"] = row["id"]
            session["role"] = row["role"]
            # Each login gets a fresh URL workspace and a distinct HttpOnly
            # cookie whose Path is limited to that workspace.
            session["_csrf_token"] = secrets.token_urlsafe(32)
            session.permanent = True
            workspace_id = secrets.token_urlsafe(24)
            request.environ["agrisage_new_workspace_id"] = workspace_id
            log_activity(row["id"], row["email"], "login", "Successful sign-in")
            destination = url_for("admin_dashboard" if row["role"] == "admin" else "index")
            return redirect(f"/w/{workspace_id}{destination}")
        log_activity(row["id"] if row else None, email, "failed_login", "Invalid credentials or inactive account")
        flash("Email or password is incorrect.", "error")
    return render_template("login.html", active_user=None)


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    user = current_user()
    log_activity(user["id"], user["email"], "logout", "Signed out")
    session.clear()
    return redirect("/login")


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    if user["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("user_history"))


@app.route("/history")
@login_required
def user_history():
    user = current_user()
    if user["role"] == "admin":
        return redirect(url_for("admin_dashboard"))
    with db_connection() as con:
        scans = con.execute(
            "SELECT crop,disease,confidence,image_name,created_at FROM prediction_history WHERE user_id=? ORDER BY id DESC",
            (user["id"],),
        ).fetchall()
    return render_template("history.html", user=user, scans=scans)


@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    with db_connection() as con:
        stats = {
            "users": con.execute("SELECT COUNT(*) AS n FROM users WHERE role='user'").fetchone()["n"],
            "active": con.execute("SELECT COUNT(*) AS n FROM users WHERE role='user' AND is_active=1").fetchone()["n"],
            "scans": con.execute("SELECT COUNT(*) AS n FROM prediction_history").fetchone()["n"],
            "today": con.execute("SELECT COUNT(*) AS n FROM activity_logs WHERE created_at>=?", (datetime.now().date().isoformat(),)).fetchone()["n"],
        }
        users = con.execute("SELECT id,name,email,role,is_active,created_at FROM users ORDER BY id DESC LIMIT 100").fetchall()
        activities = con.execute("SELECT id,user_id,email,action,details,created_at FROM activity_logs ORDER BY id DESC LIMIT 100").fetchall()
    return render_template("admin_dashboard.html", stats=stats, users=users, activities=activities, admin=current_user())


@app.route("/admin/activity/<int:activity_id>/delete", methods=["POST"])
@admin_required
def admin_delete_activity(activity_id):
    """Allow an administrator to remove one specific activity-log entry."""
    with db_connection() as con:
        result = con.execute("DELETE FROM activity_logs WHERE id=?", (activity_id,))
    if result.rowcount:
        flash("Activity log entry removed.", "success")
    else:
        flash("That activity log entry was not found.", "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    """Permanently remove one farmer account and its account-specific records."""
    with db_connection() as con:
        row = con.execute(
            "SELECT id,email,role FROM users WHERE id=?", (user_id,)
        ).fetchone()
        if not row or row["role"] != "user":
            flash("That account cannot be removed.", "error")
            return redirect(url_for("admin_dashboard"))

        # Remove dependent records first because these tables may not have
        # database-level cascading foreign keys in existing deployments.
        con.execute("DELETE FROM prediction_history WHERE user_id=?", (user_id,))
        con.execute("DELETE FROM activity_logs WHERE user_id=?", (user_id,))
        con.execute("DELETE FROM users WHERE id=? AND role='user'", (user_id,))

    admin = current_user()
    if admin:
        log_activity(
            admin["id"], admin["email"], "user_deleted",
            f"Permanently removed farmer account {row['email']}"
        )
    flash("User account and its saved scan history were permanently removed.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:user_id>/status", methods=["POST"])
@admin_required
def admin_user_status(user_id):
    if user_id == session.get("user_id"):
        flash("You cannot deactivate your own administrator account.", "error")
    else:
        with db_connection() as con:
            row = con.execute("SELECT id,email,role,is_active FROM users WHERE id=?", (user_id,)).fetchone()
            if row and row["role"] != "admin":
                new_status = 0 if row["is_active"] else 1
                con.execute("UPDATE users SET is_active=? WHERE id=?", (new_status, user_id))
                changed_email = row["email"]
                changed_label = "active" if new_status else "inactive"
            else:
                changed_email = None
                changed_label = None
        if changed_email:
            log_activity(session["user_id"], current_user()["email"], "user_status_changed",
                f"User {changed_email} set to {changed_label}")
            flash("User status updated.", "success")
        else:
            flash("That account cannot be changed.", "error")
    return redirect(url_for("admin_dashboard"))


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    lang = request.args.get("lang", request.form.get("lang", "en"))
    if lang not in LANG: lang = "en"
    crop_choice = request.args.get("crop", request.form.get("crop", "auto"))
    crop_options = crop_options_for_model(lang)
    if crop_choice not in {key for key, _ in crop_options}: crop_choice = "auto"
    result = None
    if request.method == "POST":
        file = request.files.get("leaf_image")
        if not file or not file.filename:
            flash({"en":"Please choose a leaf image first.","hi":"कृपया पहले फसल की तस्वीर चुनें।","mr":"कृपया आधी पिकाचा फोटो निवडा."}[lang], "error")
            return redirect(url_for("index", lang=lang))
        cleanup_old_uploads()
        try:
            image = Image.open(file.stream).convert("RGB")
            accepted, reason = validate_plant_photo(image)
            if not accepted:
                message = {
                    "en": "This photo may not be a clear plant or leaf image. Please upload a sharp, well-lit photo showing the crop or leaf clearly.",
                    "hi": "यह तस्वीर स्पष्ट पौधे या पत्ती की तस्वीर नहीं लगती। कृपया फसल या पत्ती की साफ़ और अच्छी रोशनी वाली तस्वीर अपलोड करें।",
                    "mr": "हा फोटो स्पष्ट पिकाचा किंवा पानाचा फोटो वाटत नाही. कृपया पिकाचा किंवा पानाचा स्वच्छ आणि चांगल्या प्रकाशातील फोटो अपलोड करा.",
                }[lang]
                flash(message, "error")
                return redirect(url_for("index", lang=lang))
            image.thumbnail((1600, 1600))
            # Unique URLs prevent browsers from reusing the previous upload from cache.
            saved = f"scan_{uuid.uuid4().hex}.jpg"
            image.save(UPLOAD_DIR / saved, "JPEG", quality=88)
        except (UnidentifiedImageError, OSError, ValueError):
            flash({"en":"Unsupported image. Please upload a JPG, PNG, or WEBP leaf photo.","hi":"यह फ़ाइल समर्थित नहीं है। कृपया JPG, PNG या WEBP फसल की तस्वीर अपलोड करें।","mr":"ही फाइल समर्थित नाही. कृपया JPG, PNG किंवा WEBP पिकाचा फोटो अपलोड करा."}[lang], "error")
            return redirect(url_for("index", lang=lang))
        try:
            import numpy as np
            model, all_labels = get_model()
            arr = np.asarray(image.resize((160, 160)), dtype=np.float32)[None, ...]
            probs = model(model._crop_preprocess(arr), training=False).numpy()[0]
            allowed_indices = (
                [i for i, label in enumerate(all_labels) if label.partition("___")[0] in CROPS]
                if getattr(model, "_all_class_model", False)
                else [i for i, label in enumerate(all_labels) if label in TARGET_LABELS]
            )
            if not allowed_indices:
                raise RuntimeError("The loaded model does not contain any supported disease labels.")
            if crop_choice != "auto":
                crop_indices = [i for i in allowed_indices if all_labels[i].partition("___")[0] == crop_choice]
                if not crop_indices:
                    raise RuntimeError("The active model does not support the selected crop yet. Choose automatic crop detection or install the expanded model.")
                allowed_indices = crop_indices
            ranked = sorted(allowed_indices, key=lambda i: float(probs[i]), reverse=True)
            best = ranked[0]
            crop, disease = pretty(all_labels[best])
            # Keep the model's original softmax score. Renormalizing after crop/class
            # filtering can inflate a weak prediction into an apparently confident one.
            # This score is diagnostic, not a calibrated probability.
            score = float(probs[best])
            if score < 0.30:
                flash({
                    "en": "The model could not identify a supported crop disease confidently. Please upload a closer, clearer plant photo.",
                    "hi": "मॉडल किसी समर्थित फसल रोग की भरोसेमंद पहचान नहीं कर सका। कृपया पौधे की और स्पष्ट, पास से ली गई तस्वीर अपलोड करें।",
                    "mr": "मॉडेलला समर्थित पिकाच्या रोगाची खात्रीशीर ओळख पटली नाही. कृपया पिकाचा अधिक स्पष्ट आणि जवळून घेतलेला फोटो अपलोड करा.",
                }[lang], "error")
                return redirect(url_for("index", lang=lang))
            # Independently inspect the actual image. Softmax confidence alone is not
            # enough: confident false positives are common on out-of-domain field photos.
            verification = verify_diagnosis_image(UPLOAD_DIR / saved, crop, disease, lang)
            # Only an explicit contradiction or non-plant verdict should suppress the result.
            # Gemini can be uncertain or temporarily unavailable even when the classifier is right.
            confirmed = verification.get("verdict") not in {"disagree", "not_plant"}
            research = disease_guidance(crop, disease, lang, fast=True) if confirmed else {
                "status": "unverified", "message": "", "description": "", "prevention": "", "sources": []
            }
            # A real diagnosis must have matching disease-specific guidance. Never
            # display the old generic healthy/no-pattern paragraph for a disease class.
            if confirmed and (not research.get("description", "").strip() or not research.get("prevention", "").strip()):
                confirmed = False
                research = {"status": "unverified", "message": "", "description": "", "prevention": "", "sources": []}
            result = {
                "crop": crop,
                "disease": disease if confirmed else "Needs review — not confirmed",
                "predicted_disease": disease,
                "confidence": score,
                "image": saved, "research": research, "verification": verification,
                "unconfirmed": not confirmed,
                "low_confidence": score < 0.45,
                "model_name": "Crop image classifier",
                "healthy": disease.lower() == "healthy" and confirmed,
            }
            account = current_user()
            if account:
                try:
                    with db_connection() as con:
                        con.execute("INSERT INTO prediction_history(user_id,crop,disease,confidence,image_name,created_at) VALUES(?,?,?,?,?,?)",
                            (account["id"], crop, result["disease"], score, saved, datetime.now().isoformat(timespec="seconds")))
                    log_activity(account["id"], account["email"], "disease_prediction",
                        f"{crop}: {result['disease']} ({score:.3f})")
                except Exception:
                    app.logger.exception("Could not save account scan history; prediction remains available.")
        except Exception as exc:
            app.logger.exception("Prediction failed")
            flash({"en":"The disease model could not run. Please retry. Details: ","hi":"रोग पहचान मॉडल नहीं चल सका। कृपया फिर से प्रयास करें। विवरण: ","mr":"रोग ओळख मॉडेल चालू शकले नाही. कृपया पुन्हा प्रयत्न करा. तपशील: "}[lang] + str(exc)[:180], "error")
    return render_template("index.html", result=result, lang=lang, crop_choice=crop_choice, crop_options=crop_options, model_trained=(EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists()) or (ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists()) or (TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists()))


@app.route("/community")
def community():
    lang = request.args.get("lang", "en")
    if lang not in LANG:
        lang = "en"
    return render_template("community.html", lang=lang)


@app.route("/library")
def library():
    lang = request.args.get("lang", "en")
    if lang not in LANG:
        lang = "en"

    # Keep this catalogue aligned with the disease labels allowed by prediction.
    # Healthy leaf labels are intentionally not presented as diseases.
    image_files = {
        "Corn_(maize)___Common_rust_": "corn_common_rust.jpg",
        "Corn_(maize)___Northern_Leaf_Blight": "corn_northern_leaf_blight.jpg",
        "Potato___Early_blight": "potato_early_blight.jpg",
        "Potato___Late_blight": "potato_late_blight.jpg",
        "Tomato___Bacterial_spot": "tomato_bacterial_spot.jpg",
        "Tomato___Early_blight": "tomato_early_blight.jpg",
        "Tomato___Late_blight": "tomato_late_blight.jpg",
        "Tomato___Leaf_Mold": "tomato_leaf_mold.jpg",
        "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot.jpg",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato_yellow_leaf_curl_virus.jpg",
        "Cotton___Alternaria_leaf_spot": "plantvillage_cotton_alternaria_leaf_spot.jpg",
        "Cotton___Bacterial_blight": "plantvillage_cotton_bacterial_blight.jpg",
        "Cotton___Fusarium_wilt": "plantvillage_cotton_fusarium_wilt.jpg",
        "Cotton___Verticillium_wilt": "plantvillage_cotton_verticillium_wilt.jpg",
    }
    localized_names = {
        "Common rust": {"en": "Common rust", "hi": "कॉमन रस्ट", "mr": "कॉमन रस्ट"},
        "Northern Leaf Blight": {"en": "Northern Leaf Blight", "hi": "नॉर्दर्न लीफ ब्लाइट", "mr": "नॉर्दर्न लीफ ब्लाइट"},
        "Early blight": {"en": "Early blight", "hi": "अर्ली ब्लाइट", "mr": "अर्ली ब्लाइट"},
        "Late blight": {"en": "Late blight", "hi": "लेट ब्लाइट", "mr": "लेट ब्लाइट"},
        "Bacterial spot": {"en": "Bacterial spot", "hi": "बैक्टीरियल स्पॉट", "mr": "बॅक्टेरियल स्पॉट"},
        "Leaf Mold": {"en": "Leaf Mold", "hi": "लीफ मोल्ड", "mr": "लीफ मोल्ड"},
        "Septoria leaf spot": {"en": "Septoria leaf spot", "hi": "सेप्टोरिया लीफ स्पॉट", "mr": "सेप्टोरिया लीफ स्पॉट"},
        "Tomato Yellow Leaf Curl Virus": {"en": "Tomato Yellow Leaf Curl Virus", "hi": "टमाटर येलो लीफ कर्ल वायरस", "mr": "टोमॅटो येलो लीफ कर्ल व्हायरस"},
    }
    crop_labels = {
        "Maize / corn": {"en": "Maize / corn", "hi": "मक्का", "mr": "मका"},
        "Potato": {"en": "Potato", "hi": "आलू", "mr": "बटाटा"},
        "Tomato": {"en": "Tomato", "hi": "टमाटर", "mr": "टोमॅटो"},
    }
    active_labels = TARGET_LABELS
    if EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists():
        try:
            active_labels = json.loads(EXPANDED_LABELS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            active_labels = TARGET_LABELS
    elif ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists():
        try:
            active_labels = json.loads(ALL_LABELS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            active_labels = TARGET_LABELS
    catalog = []
    for label in active_labels:
        if label.endswith("___healthy"):
            continue
        image = "disease_images/" + image_files[label] if label in image_files else reference_image_name(label)
        if not (BASE / "static" / image).is_file():
            image = None
        crop, disease = pretty(label)
        cached = get_cached_guidance(crop, disease, lang)
        if not cached or not cached.get("description", "").strip() or not cached.get("prevention", "").strip():
            # Populate the library from verified local guidance even before a user
            # has requested AI research for this class. This performs no network call.
            fallback = offline_guidance(crop, disease, lang)
            if fallback.get("description", "").strip() and fallback.get("prevention", "").strip():
                cached = {**fallback, "cached": False}
        catalog.append({
            "crop": crop,
            "crop_label": crop_labels.get(crop, CROP_NAMES.get(label.partition("___")[0], {"en": crop, "hi": crop, "mr": crop})),
            "name": disease,
            "name_label": localized_names.get(disease, {"en": disease, "hi": disease, "mr": disease}),
            "image": image,
            "guidance": cached,
        })
    return render_template("library.html", catalog=catalog, lang=lang)



@app.route("/health")
def health():
    all43 = EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists()
    all38 = ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists()
    focused = TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists()
    labels_path = EXPANDED_LABELS_PATH if all43 else ALL_LABELS_PATH if all38 else TEN_LABELS_PATH if focused else None
    try:
        active_labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path and labels_path.exists() else TARGET_LABELS
    except (OSError, json.JSONDecodeError):
        active_labels = TARGET_LABELS
    if all43 or all38:
        active_labels = [label for label in active_labels if label.partition("___")[0] in CROPS]
    disease_count = sum(1 for label in active_labels if not label.endswith("___healthy"))
    healthy_count = sum(1 for label in active_labels if label.endswith("___healthy"))
    preference = "expanded55_incremental_mobilenetv2" if all43 else "all38_mobilenetv2" if all38 else "ten_disease_mobilenetv2" if focused else "existing_plantvillage_filtered"
    return {"app": "ok", "database": database_backend(), "model_file_present": (EXPANDED_MODEL_PATH.exists() and EXPANDED_LABELS_PATH.exists()) or (ALL_MODEL_PATH.exists() and ALL_LABELS_PATH.exists()) or (TEN_MODEL_PATH.exists() and TEN_LABELS_PATH.exists()) or PV_MODEL_PATH.exists(), "model_preference": preference, "target_disease_count": disease_count, "healthy_classes": healthy_count, "ai_role": "image_verification_and_description_and_prevention", "shared_disease_knowledge_cache": True, "knowledge_cache_refresh_days": KNOWLEDGE_CACHE_DAYS}


@app.errorhandler(403)
def forbidden(_):
    return render_template(
        "login.html",
        active_user=current_user(),
        auth_error="This account does not have permission to open that page. Sign in with an administrator account or switch accounts below.",
    ), 403


@app.errorhandler(400)
def bad_request(error):
    description = getattr(error, "description", "")
    if "form expired" in str(description).lower() or "could not be verified" in str(description).lower():
        return render_template(
            "login.html",
            active_user=current_user(),
            auth_error="This form or sign-in session expired. Refresh the page and try again.",
        ), 400
    return error


@app.errorhandler(413)
def too_large(_):
    lang = request.form.get("lang", request.args.get("lang", "en"))
    if lang not in LANG: lang = "en"
    flash({"en":"Please upload an image smaller than 8 MB.","hi":"कृपया 8 MB से छोटी तस्वीर अपलोड करें।","mr":"कृपया 8 MB पेक्षा लहान फोटो अपलोड करा."}[lang], "error")
    return redirect(url_for("index", lang=lang))


init_db()

def _preload_model_in_background():
    """Load and warm the classifier during service startup, not during a user's scan."""
    try:
        get_model()
        app.logger.info("Disease model loaded and warmed in background.")
    except Exception:
        app.logger.exception("Background model warm-up failed; it will retry on the first scan.")

# Render health checks can start immediately while TensorFlow initializes in the background.
if os.environ.get("AGROSAGE_SKIP_MODEL_WARMUP") != "1":
    threading.Thread(target=_preload_model_in_background, name="model-warmup", daemon=True).start()

if __name__ == "__main__": app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
