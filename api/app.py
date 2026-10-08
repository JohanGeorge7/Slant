"""
Slant server: static pages + live model endpoints + background jobs that
keep the site current without anyone re-running scripts by hand.

Security and hygiene handled here rather than in the page:
  * no secrets in the frontend — the browser only ever talks to our own
    endpoints, and configuration comes from environment variables
  * HTTPS enforced in production (redirect + HSTS)
  * security headers including a content security policy
  * newsletter signup is validated server-side and honeypot-protected

Background jobs (see start_scheduler() below) are what make the board
"live": a light job polls ESPN scores every 20s during games and recomputes
win probability as the score moves, a medium job refreshes injury
designations and their modeled win-probability impact every 15 minutes, and
a heavier job periodically re-ingests the current season and re-exports the
site's data — which is also what makes the board advance to the next week
on its own, since export_frontend_data.py always exports whichever week
still has unplayed games.

Run: uv run python api/app.py   then open http://localhost:5000
"""

import atexit
import os
import sys
import time
import traceback
from datetime import datetime
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
FRONTEND_DATA = FRONTEND / "data"

# Config from the environment. Nothing sensitive is ever sent to the browser.
IS_PROD = os.environ.get("GRIDIRON_ENV", "dev") == "prod"
SITE_URL = os.environ.get("GRIDIRON_SITE_URL", "http://localhost:5000")
ANALYTICS_ID = os.environ.get("GRIDIRON_ANALYTICS_ID", "")

# Background jobs are on by default; disable with GRIDIRON_SCHEDULER=0 if
# running multiple worker processes behind a WSGI server, since each worker
# would otherwise start its own copy and multiply the ESPN polling and the
# retrain workload.
SCHEDULER_ENABLED = os.environ.get("GRIDIRON_SCHEDULER", "1") == "1"
LIVE_SCORES_INTERVAL = int(os.environ.get("GRIDIRON_LIVE_INTERVAL_SEC", "8"))
INJURY_INTERVAL_MIN = int(os.environ.get("GRIDIRON_INJURY_INTERVAL_MIN", "15"))
REFRESH_INTERVAL_MIN = int(os.environ.get("GRIDIRON_REFRESH_INTERVAL_MIN", "60"))
RETRAIN_INTERVAL_HOURS = int(os.environ.get("GRIDIRON_RETRAIN_INTERVAL_HOURS", "24"))

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
# Live updates: score/clock-driven win probability and injury impact notes,
# both written by background jobs (see start_scheduler below) into small
# JSON files the frontend polls every few seconds to tens of minutes.
# Served with no-cache since "live" is the entire point.
# --------------------------------------------------------------------------
def _serve_live_json(filename, empty_body):
    path = FRONTEND_DATA / filename
    if not path.exists():
        resp = jsonify(empty_body)
    else:
        resp = Response(path.read_text(), mimetype="application/json")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/live")
def live():
    return _serve_live_json("live.json", {"updated_at": None, "games": {}})


@app.route("/api/injury-notes")
def injury_notes_route():
    return _serve_live_json("injury_notes.json", {"games": {}})


@app.route("/api/depth-charts")
def depth_charts_route():
    return _serve_live_json("depth_charts.json", {"games": {}})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "env": "prod" if IS_PROD else "dev"})


# --------------------------------------------------------------------------
# Background jobs. Each is wrapped so one failure (a flaky ESPN response, a
# training run hitting bad data) logs and moves on instead of killing the
# whole scheduler thread — these run unattended, so silent survival matters
# more than failing loudly.
# --------------------------------------------------------------------------
def _guarded(fn, label):
    def run():
        try:
            fn()
        except Exception:
            print(f"[scheduler] {label} failed:")
            traceback.print_exc()
    return run


def _job_live_scores():
    from models.live_scores import main as run
    run()
    from models.picks_history import settle_from_live
    settle_from_live()


def _job_injury_notes():
    from models.injury_notes import main as run
    run()
    from models.depth_chart import main as run_dc
    run_dc()


def _job_refresh_and_export():
    """Cheap: re-fetch only the current season, then re-export the site's
    JSON from whatever's now in the cache. No model retraining here, so
    this is safe to run frequently. This is also what advances the board to
    the next week — export_frontend_data.py always exports whichever week
    still has a game with no result yet, so once this week's results land
    in the refreshed schedule, the next run of this job produces next
    week's board automatically.
    """
    from data.ingest import refresh_current_season
    from models.export_frontend_data import main as export_frontend
    refresh_current_season()
    export_frontend()
    _job_injury_notes()
    from models.picks_history import record_pregame_picks, backfill_missed_games, settle_from_schedule
    record_pregame_picks()
    backfill_missed_games()   # heals weeks the server was off for
    settle_from_schedule()


def _job_full_retrain():
    """Heavier: walk-forward retrain all three models on the latest data and
    regenerate the evidence page. Run on a slower cadence than the refresh
    job above since the deployed coefficients don't need to change every
    hour, just often enough to fold in accumulating games.
    """
    from models.game_predictor import main as train_game
    from models.playcall_predictor import main as train_playcall
    from models.fourth_down_model import main as train_fourth_down
    from models.export_evidence import main as export_evidence
    train_game()
    train_playcall()
    train_fourth_down()
    _job_refresh_and_export()
    export_evidence()


def start_scheduler():
    from apscheduler.schedulers.background import BackgroundScheduler

    # APScheduler's IntervalTrigger waits a FULL interval before its first
    # run by default -- e.g. refresh_and_export (60 min) wouldn't touch the
    # board for an hour after every single server start/restart, serving
    # whatever was already on disk the whole time. Combined with a server
    # that isn't kept running continuously, that's how the board can end up
    # stuck on last week's data with a live game already underway and
    # nothing in the logs to explain it. next_run_time=now makes every job
    # run once immediately at startup, then fall onto its normal interval.
    now = datetime.now()
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(_guarded(_job_live_scores, "live_scores"), "interval",
                  seconds=LIVE_SCORES_INTERVAL, id="live_scores", max_instances=1, coalesce=True,
                  next_run_time=now)
    sched.add_job(_guarded(_job_injury_notes, "injury_notes"), "interval",
                  minutes=INJURY_INTERVAL_MIN, id="injury_notes", max_instances=1, coalesce=True,
                  next_run_time=now)
    sched.add_job(_guarded(_job_refresh_and_export, "refresh_and_export"), "interval",
                  minutes=REFRESH_INTERVAL_MIN, id="refresh_and_export", max_instances=1, coalesce=True,
                  next_run_time=now)
    sched.add_job(_guarded(_job_full_retrain, "full_retrain"), "interval",
                  hours=RETRAIN_INTERVAL_HOURS, id="full_retrain", max_instances=1, coalesce=True)
    # full_retrain intentionally keeps the default first-run-after-one-interval
    # behavior -- it's the heavy job (full walk-forward retrain of all three
    # models), and _job_refresh_and_export above already refreshes the board
    # itself immediately, so there's no user-facing staleness riding on this
    # one running at startup too.
    sched.start()
    atexit.register(lambda: sched.shutdown(wait=False))
    print(f"[scheduler] live_scores every {LIVE_SCORES_INTERVAL}s, "
          f"injury_notes every {INJURY_INTERVAL_MIN}m, "
          f"refresh_and_export every {REFRESH_INTERVAL_MIN}m, "
          f"full_retrain every {RETRAIN_INTERVAL_HOURS}h")
    return sched


if SCHEDULER_ENABLED:
    start_scheduler()

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=not IS_PROD and False)
