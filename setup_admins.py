"""Create or promote the two AgroSage administrators securely.

Run interactively from the project directory. Passwords are read with getpass,
hashed immediately, and never written to files or environment variables.
"""
import getpass
import os
import re
from datetime import datetime
from pathlib import Path

# Read only the database location from private local config; never load or print passwords.
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    for config_line in env_file.read_text(encoding="utf-8").splitlines():
        if config_line.strip().startswith("DATABASE_URL="):
            value = config_line.split("=", 1)[1].strip().strip('"').strip("'")
            if value:
                os.environ.setdefault("DATABASE_URL", value)

# Avoid loading TensorFlow/model weights while this one-time account utility runs.
os.environ["AGROSAGE_SKIP_MODEL_WARMUP"] = "1"

from app import db_connection, init_db  # noqa: E402
from werkzeug.security import generate_password_hash  # noqa: E402

DEFAULT_EMAILS = ("schinche735@gmail.com", "shantanu056@gmail.com")


def prompt_admin(index, default_email):
    email = input(f"Administrator {index} email [{default_email}]: ").strip().lower() or default_email
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise ValueError(f"Administrator {index} email address is invalid.")
    password = getpass.getpass(f"New password for {email} (hidden): ")
    confirmation = getpass.getpass("Confirm password (hidden): ")
    if len(password) < 12:
        raise ValueError("Use a unique password with at least 12 characters.")
    if password != confirmation:
        raise ValueError("The two password entries did not match.")
    return email, password


def main():
    init_db()
    first = prompt_admin(1, DEFAULT_EMAILS[0])
    second = prompt_admin(2, DEFAULT_EMAILS[1])
    if first[0] == second[0]:
        raise ValueError("The two administrator email addresses must be different.")
    if first[1] == second[1]:
        raise ValueError("Use different passwords for the two administrator accounts.")

    prepared = [
        (email, generate_password_hash(password))
        for email, password in (first, second)
    ]
    now = datetime.now().isoformat(timespec="seconds")
    with db_connection() as con:
        for email, password_hash in prepared:
            existing = con.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if existing:
                con.execute(
                    "UPDATE users SET password_hash=?,role='admin',is_active=1 WHERE id=?",
                    (password_hash, existing["id"]),
                )
            else:
                con.execute(
                    "INSERT INTO users(name,email,password_hash,role,is_active,created_at) VALUES(?,?,?,?,?,?)",
                    (email.split("@", 1)[0], email, password_hash, "admin", 1, now),
                )
    print("Both administrator accounts are ready. Only password hashes were saved.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyboardInterrupt) as error:
        print(f"Administrator setup stopped: {error}")
        raise SystemExit(1)
