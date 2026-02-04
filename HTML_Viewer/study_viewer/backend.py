from __future__ import annotations

import secrets
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, request, send_from_directory

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

HOST = "0.0.0.0"
PORT = 1337
PASSWORD = "guessthepdff!"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# In-memory tokens for this simple viewer session
TOKENS: Dict[str, bool] = {}


def _is_authorized(req) -> bool:
    token = req.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    return bool(token) and TOKENS.get(token, False)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/app.js")
def app_js():
    return send_from_directory(STATIC_DIR, "app.js")


@app.route("/styles.css")
def styles_css():
    return send_from_directory(STATIC_DIR, "styles.css")


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/auth", methods=["POST"])
def auth():
    data = request.get_json(silent=True) or {}
    if data.get("password") != PASSWORD:
        return jsonify({"ok": False}), 401
    token = secrets.token_urlsafe(24)
    TOKENS[token] = True
    return jsonify({"ok": True, "token": token})


@app.route("/api/study")
def study():
    if not _is_authorized(request):
        return jsonify({"error": "unauthorized"}), 401
    # Placeholder study payload. We will add actual image metadata later.
    return jsonify({
        "ok": True,
        "total_cases": 150,
        "cases": [f"case_{i:03d}" for i in range(1, 151)],
    })


if __name__ == "__main__":
    print("=" * 80)
    print("Study Viewer Backend")
    print("=" * 80)
    print(f"Starting server on http://localhost:{PORT}")
    print("Use the shared password to access the study view.")
    print("=" * 80)
    app.run(host=HOST, port=PORT, debug=False)
