"""
Gridiron server: static pages + live model endpoints.

Security and hygiene handled here rather than in the page:
  * no secrets in the frontend — the browser only ever talks to our own
    endpoints, and configuration comes from environment variables
  * HTTPS enforced in production (redirect + HSTS)
  * security headers including a content security policy
  * newsletter signup is validated server-side and honeypot-protected

Run: uv run python api/app.py   then open http://localhost:5000
"""

import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, redirect, Response

from models.fourth_down_model import FourthDownModel
from models.simulate import simulate_margins, summarize, histogram
from models.playcall_predictor import CLASSES as PC_CLASSES

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "artifacts"
FRONTEND = ROOT / "frontend"

# Config from the environment. Nothing sensitive is ever sent to the browser.
IS_PROD = os.environ.get("GRIDIRON_ENV", "dev") == "prod"
SITE_URL = os.environ.get("GRIDIRON_SITE_URL", "http://localhost:5000")
ANALYTICS_ID = os.environ.get("GRIDIRON_ANALYTICS_ID", "")

app = Flask(__name__, static_folder=None)

_fd = joblib.load(MODEL_DIR / "fourth_down_model.joblib")
fourth = FourthDownModel(_fd["ep_curve"], _fd["conversion"], _fd["fg"], _fd["punt_net"])
_pc = joblib.load(MODEL_DIR / "playcall_predictor.joblib")


# --------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------
@app.before_request
def force_https():
    if not IS_PROD:
        return None
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    if proto != "https":
        return redirect(request.url.replace("http://", "https://", 1), code=301)
    return None


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), interest-cohort=()"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: https://a.espncdn.com https://github.com https://raw.githubusercontent.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    )
    if IS_PROD:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page(name):
    return send_from_directory(FRONTEND, name)


@app.route("/")
def index():
    return page("index.html")


@app.route("/privacy")
def privacy():
    return page("privacy.html")


@app.route("/terms")
def terms():
    return page("terms.html")


@app.route("/data/<path:f>")
def data_files(f):
    return send_from_directory(FRONTEND / "data", f, max_age=300)


@app.route("/assets/<path:f>")
def assets(f):
    return send_from_directory(FRONTEND / "assets", f, max_age=86400)


@app.route("/favicon.ico")
def favicon():
    return send_from_directory(FRONTEND / "assets", "favicon.ico", max_age=86400)


@app.route("/robots.txt")
def robots():
    return Response(f"User-agent: *\nAllow: /\n\nSitemap: {SITE_URL}/sitemap.xml\n",
                    mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    pages = [("/", "1.0", "daily"), ("/privacy", "0.3", "yearly"), ("/terms", "0.3", "yearly")]
    today = time.strftime("%Y-%m-%d")
    urls = "".join(
        f"<url><loc>{SITE_URL}{p}</loc><lastmod>{today}</lastmod>"
        f"<changefreq>{cf}</changefreq><priority>{pr}</priority></url>"
        for p, pr, cf in pages)
    return Response(
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>',
        mimetype="application/xml")


@app.errorhandler(404)
def not_found(_):
    return send_from_directory(FRONTEND, "404.html"), 404


@app.route("/api/config")
def config():
    """Only non-sensitive, client-safe values. Analytics runs from our own
    server and is gated on consent, so no third-party key is exposed."""
    return jsonify({"analytics_enabled": bool(ANALYTICS_ID), "site_url": SITE_URL})


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
@app.route("/api/fourth-down", methods=["POST"])
def fourth_down():
    d = request.get_json(silent=True) or {}
    try:
        ytg = max(1, min(int(d["ydstogo"]), 30))
        yl = max(1, min(float(d["yardline_100"]), 99))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "ydstogo and yardline_100 are required numbers"}), 400
    return jsonify(fourth.evaluate(
        ydstogo=ytg, yardline_100=yl,
        score_differential=float(d.get("score_differential", 0)),
        game_seconds_remaining=float(d.get("game_seconds_remaining", 1800))))


@app.route("/api/playcall", methods=["POST"])
def playcall():
    """Four-class prediction (run / pass / punt / field goal), returned
    alongside the league's actual rate for the same situation."""
    d = request.get_json(silent=True) or {}
    try:
        dn = max(1, min(int(d["down"]), 4))
        ytg = max(1, min(int(d["ydstogo"]), 30))
        yl = max(1, min(float(d.get("yardline_100", 60)), 99))
        sd = float(d.get("score_differential", 0))
        sec = float(d.get("game_seconds_remaining", 2400))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "down and ydstogo are required numbers"}), 400

    pr = _pc["model"].predict_proba(np.array([[dn, ytg, yl, sd, sec]]))[0]
    probs = {c: round(float(pr[i]), 4) for i, c in enumerate(PC_CLASSES)}
    top = max(probs, key=probs.get)
    emp = _pc["empirical"].get(f"{dn}_{min(ytg, 21)}")
    return jsonify({"probabilities": probs, "predicted": top,
                    "confidence": round(float(max(pr)), 4), "empirical": emp})


@app.route("/api/simulate", methods=["POST"])
def simulate():
    d = request.get_json(silent=True) or {}
    p = d.get("home_win_prob")
    if p is None:
        return jsonify({"error": "home_win_prob is required"}), 400
    m = simulate_margins(float(p), n=min(int(d.get("n", 20000)), 50000), seed=int(d.get("seed", 0)))
    return jsonify({**summarize(m), "histogram": histogram(m)})


# --------------------------------------------------------------------------
# Newsletter: validated and honeypot-protected
# --------------------------------------------------------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]{2,}$")
_recent: dict[str, list[float]] = {}


@app.route("/api/subscribe", methods=["POST"])
def subscribe():
    d = request.get_json(silent=True) or {}

    # Honeypot: a field hidden from humans. Anything filling it is a bot.
    if (d.get("company") or "").strip():
        return jsonify({"ok": True}), 200          # succeed silently

    # Timing check: real people take more than a couple of seconds.
    try:
        if float(d.get("elapsed_ms", 99999)) < 1500:
            return jsonify({"ok": True}), 200
    except (TypeError, ValueError):
        pass

    email = (d.get("email") or "").strip().lower()
    if not email or len(email) > 254 or not EMAIL_RE.match(email):
        return jsonify({"ok": False, "error": "Enter a valid email address."}), 400

    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0]
    now = time.time()
    hits = [t for t in _recent.get(ip, []) if now - t < 3600]
    if len(hits) >= 5:
        return jsonify({"ok": False, "error": "Too many attempts. Try again later."}), 429
    _recent[ip] = hits + [now]

    # A real deployment would persist this; we accept and confirm.
    return jsonify({"ok": True, "message": "You're on the list."})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "env": "prod" if IS_PROD else "dev"})


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=not IS_PROD and False)
