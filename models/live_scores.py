"""
Live score + live win-probability poller. Reads the week's pregame
predictions (already exported by export_frontend_data.py), overlays
whatever ESPN's scoreboard reports right now, and writes a small
frontend/data/live.json that the page polls every ~15 seconds during game
windows — the same "check back for updates" idea as a fantasy app's live
scoring, applied to win probability instead of fantasy points.

Deliberately a separate, cheap file from predictions.json: the prediction
payload carries the full simulation/driver breakdown for every game and is
only worth regenerating when the model itself reruns, while this needs to
be refetched every few seconds during a live window without redoing any of
that work.

Run standalone for a manual check: uv run python models/live_scores.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.espn import get_json, to_nflverse
from models.live_winprob import live_win_prob

ROOT = Path(__file__).parent.parent
FRONTEND_DATA = ROOT / "frontend" / "data"

CLOCK_PER_QUARTER = 900.0  # 15 minutes


def _seconds_remaining(period: int, display_clock: str) -> float:
    try:
        mm, ss = display_clock.split(":")
        in_quarter = int(mm) * 60 + float(ss)
    except (ValueError, AttributeError):
        in_quarter = 0.0
    quarters_left = max(0, 4 - period)
    return in_quarter + quarters_left * CLOCK_PER_QUARTER


def fetch_scoreboard() -> list[dict]:
    """Every event on ESPN's current scoreboard, normalized to our team
    abbreviations. Empty list (rather than raising) if ESPN is unreachable,
    so a scheduler job never dies on a transient network hiccup.
    """
    try:
        data = get_json("scoreboard")
    except Exception as e:
        # Swallowed on purpose (a scheduler job must never crash on a
        # transient ESPN hiccup) but logged -- silently returning an empty
        # list here previously made a real, sustained outage (e.g. ESPN
        # blocking this client) indistinguishable from "no games right
        # now," which meant live win probability could silently stop
        # updating for hours with nothing in the logs to explain why.
        print(f"[live_scores] scoreboard fetch failed: {e}")
        return []

    events = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        status = comp["status"]
        state = status["type"]["state"]  # 'pre' | 'in' | 'post'
        period = status.get("period", 0)
        completed = bool(status["type"].get("completed"))

        teams = {}
        for c in comp["competitors"]:
            teams[c["homeAway"]] = {
                "abbr": to_nflverse(c["team"]["abbreviation"]),
                "score": int(c.get("score") or 0),
            }
        if "home" not in teams or "away" not in teams:
            continue

        events.append({
            "id": f"{teams['away']['abbr']}@{teams['home']['abbr']}",
            "state": state,
            "completed": completed,
            "period": period,
            "display_clock": status.get("displayClock", "0:00"),
            "detail": status["type"].get("shortDetail", ""),
            "home_score": teams["home"]["score"],
            "away_score": teams["away"]["score"],
        })
    return events


def build_live_snapshot(predictions: dict) -> dict:
    """Merge live ESPN state onto the week's pregame predictions. Games not
    found on the current scoreboard (bye, or the ESPN week boundary doesn't
    line up exactly with ours) are simply omitted rather than guessed at.
    """
    pregame = {g["id"]: g for g in predictions.get("games", [])}
    live_events = {e["id"]: e for e in fetch_scoreboard()}

    out = {}
    for gid, g in pregame.items():
        e = live_events.get(gid)
        if e is None:
            continue
        overtime = e["period"] >= 5
        seconds_remaining = 0.0 if e["completed"] else _seconds_remaining(e["period"], e["display_clock"])
        live_p = live_win_prob(g["home_win_prob"], e["home_score"], e["away_score"],
                                seconds_remaining, overtime=overtime)
        out[gid] = {
            "state": e["state"], "completed": e["completed"],
            "period": e["period"], "display_clock": e["display_clock"], "detail": e["detail"],
            "home_score": e["home_score"], "away_score": e["away_score"],
            "home_win_prob_live": round(live_p, 3),
            "away_win_prob_live": round(1 - live_p, 3),
        }
    return out


def main():
    pred_path = FRONTEND_DATA / "predictions.json"
    if not pred_path.exists():
        print("No predictions.json yet — run export_frontend_data.py first.")
        return
    predictions = json.loads(pred_path.read_text())
    snapshot = build_live_snapshot(predictions)
    out = {"updated_at": __import__("time").time(), "season": predictions.get("season"),
           "week": predictions.get("week"), "games": snapshot}
    (FRONTEND_DATA / "live.json").write_text(json.dumps(out, indent=1))
    print(f"live.json: {len(snapshot)} game(s) with live state")


if __name__ == "__main__":
    main()
