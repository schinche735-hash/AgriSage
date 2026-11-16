import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as app_module
import setup_admins
from app import app
from werkzeug.security import check_password_hash, generate_password_hash


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(app_module, "DB_PATH", Path(self.temp_dir.name) / "auth.db")
        self.db_patch.start()
        app_module.init_db()
        with app_module.db_connection() as con:
            con.execute(
                "INSERT INTO users(name,email,password_hash,role,is_active,created_at) VALUES(?,?,?,?,?,?)",
                ("Farmer One", "farmer@example.com", generate_password_hash("farmer-password-123"), "user", 1, "2026-01-01T00:00:00"),
            )
            con.execute(
                "INSERT INTO users(name,email,password_hash,role,is_active,created_at) VALUES(?,?,?,?,?,?)",
                ("Admin One", "admin@example.com", generate_password_hash("admin-password-123"), "admin", 1, "2026-01-01T00:00:00"),
            )
        self.client = app.test_client()
        self.set_csrf(self.client)

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def set_csrf(client):
        with client.session_transaction() as sess:
            sess["_csrf_token"] = "auth-test-csrf"

    def test_registration_hashes_password_and_creates_farmer_role(self):
        response = self.client.post("/register", data={
            "_csrf_token": "auth-test-csrf",
            "name": "New Farmer",
            "email": "newfarmer@example.com",
            "password": "a-strong-farmer-password",
        })
        self.assertEqual(response.status_code, 302)
        with app_module.db_connection() as con:
            user = con.execute("SELECT * FROM users WHERE email=?", ("newfarmer@example.com",)).fetchone()
        self.assertEqual(user["role"], "user")
        self.assertTrue(check_password_hash(user["password_hash"], "a-strong-farmer-password"))
        self.assertNotEqual(user["password_hash"], "a-strong-farmer-password")
        self.assertNotIn("a-strong-farmer-password", user["password_hash"])

    def test_farmer_login_and_dashboard_are_account_scoped(self):
        client = app.test_client()
        self.set_csrf(client)
        response = client.post("/login", data={
            "_csrf_token": "auth-test-csrf",
            "email": "AG-000001",
            "password": "farmer-password-123",
        })
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard", response.headers["Location"])
        dashboard = client.get("/dashboard")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b"AG-000001", dashboard.data)
        self.assertEqual(client.get("/admin").status_code, 403)

    def test_admin_can_view_activity_but_farmer_cannot(self):
        farmer = app.test_client()
        self.set_csrf(farmer)
        with farmer.session_transaction() as sess:
            sess["user_id"] = 1
        self.assertEqual(farmer.get("/admin").status_code, 403)

        admin = app.test_client()
        self.set_csrf(admin)
        with admin.session_transaction() as sess:
            sess["user_id"] = 2
        response = admin.get("/admin")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"User activity log", response.data)

    def test_post_without_csrf_token_is_rejected(self):
        response = self.client.post("/login", data={"email": "farmer@example.com", "password": "bad"})
        self.assertEqual(response.status_code, 400)

    def test_hashed_admin_environment_bootstrap_creates_both_admins(self):
        first_hash = generate_password_hash("first-admin-password-strong")
        second_hash = generate_password_hash("second-admin-password-strong")
        environment = {
            "ADMIN_EMAIL": "schinche735@gmail.com",
            "ADMIN_PASSWORD_HASH": first_hash,
            "ADMIN2_EMAIL": "shantanu056@gmail.com",
            "ADMIN2_PASSWORD_HASH": second_hash,
        }
        with patch.dict("os.environ", environment, clear=False):
            app_module.init_db()
        with app_module.db_connection() as con:
            first = con.execute("SELECT * FROM users WHERE email=?", ("schinche735@gmail.com",)).fetchone()
            second = con.execute("SELECT * FROM users WHERE email=?", ("shantanu056@gmail.com",)).fetchone()
        self.assertEqual(first["role"], "admin")
        self.assertEqual(second["role"], "admin")
        self.assertTrue(check_password_hash(first["password_hash"], "first-admin-password-strong"))
        self.assertTrue(check_password_hash(second["password_hash"], "second-admin-password-strong"))

    def test_admin_bootstrap_rejects_plaintext_password_environment_value(self):
        with patch.dict("os.environ", {
            "ADMIN_EMAIL": "plain@example.com",
            "ADMIN_PASSWORD_HASH": "plaintext-password",
            "ADMIN2_EMAIL": "",
            "ADMIN2_PASSWORD_HASH": "",
        }, clear=False):
            with self.assertRaises(RuntimeError):
                app_module.init_db()

    def test_secure_admin_setup_promotes_two_accounts_and_preserves_existing_id(self):
        with app_module.db_connection() as con:
            con.execute(
                "INSERT INTO users(name,email,password_hash,role,is_active,created_at) VALUES(?,?,?,?,?,?)",
                ("Existing account", "schinche735@gmail.com", generate_password_hash("old-password-123"), "user", 1, "2025-01-01T00:00:00"),
            )
            original_id = con.execute(
                "SELECT id FROM users WHERE email=?", ("schinche735@gmail.com",)
            ).fetchone()["id"]

        with patch("builtins.input", side_effect=["", ""]), patch(
            "setup_admins.getpass.getpass",
            side_effect=[
                "new-first-admin-password-123", "new-first-admin-password-123",
                "new-second-admin-password-456", "new-second-admin-password-456",
            ],
        ):
            setup_admins.main()

        with app_module.db_connection() as con:
            first = con.execute("SELECT * FROM users WHERE email=?", ("schinche735@gmail.com",)).fetchone()
            second = con.execute("SELECT * FROM users WHERE email=?", ("shantanu056@gmail.com",)).fetchone()
        self.assertEqual(first["id"], original_id)
        self.assertEqual(first["role"], "admin")
        self.assertEqual(second["role"], "admin")
        self.assertTrue(check_password_hash(first["password_hash"], "new-first-admin-password-123"))
        self.assertTrue(check_password_hash(second["password_hash"], "new-second-admin-password-456"))
        self.assertNotEqual(first["password_hash"], "new-first-admin-password-123")
        self.assertNotEqual(second["password_hash"], "new-second-admin-password-456")


if __name__ == "__main__":
    unittest.main()
