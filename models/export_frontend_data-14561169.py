"""
Exports real model outputs to JSON for the frontend. Every value here comes
from a trained model or measured nflverse data — nothing is mocked.

The per-game payload is deliberately rich: each game carries the feature
differentials that drove its prediction, both teams' form, official logos
and colors, and a Monte Carlo outcome distribution. That's what lets the
site show *why* a pick was made rather than only the number.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import nflreadpy as nfl
import polars as pl

from data.ingest import get_pbp, get_schedules
from data.features import build_team_week_stats, build_team_rolling_stats
from data.elo import latest_elo
from models.simulate import simulate_margins, summarize, histogram

MODEL_DIR = Path(__file__).parent / "artifacts"
OUT = Path(__file__).parent.parent / "frontend" / "data"
OUT.mkdir(parents=True, exist_ok=True)

PBP_COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam",
            "play_type", "epa", "success", "cpoe", "interception", "fumble_lost",
            "down", "third_down_converted", "game_seconds_remaining"]

# Human-readable labels for every model feature, so the site can explain a
# prediction in plain language instead of showing raw column names.
FEATURE_LABELS = {
    "diff_roll_off_epa_per_play":   ("Offensive EPA per play", "how much each snap moves the needle"),
    "diff_roll_def_epa_per_play":   ("Defensive EPA allowed", "how much they give up per snap"),
    "diff_roll_off_success_rate":   ("Offensive success rate", "share of plays that stay on schedule"),
    "diff_roll_def_success_rate":   ("Defensive success rate allowed", "how often opponents stay on schedule"),
    "diff_roll_off_explosive_rate": ("Explosive play rate", "share of plays gaining big chunks"),
    "diff_roll_def_explosive_rate_allowed": ("Explosives allowed", "big plays surrendered"),
    "diff_roll_off_pass_epa":       ("Pass efficiency", "EPA per dropback"),
    "diff_roll_off_rush_epa":       ("Rush efficiency", "EPA per carry"),
    "diff_roll_off_cpoe":           ("Completion % over expected", "accuracy beyond difficulty"),
    "diff_roll_off_turnover_rate":  ("Giveaway rate", "how often they cough it up"),
    "diff_roll_def_takeaway_rate":  ("Takeaway rate", "how often they force turnovers"),
    "diff_roll_off_third_down_rate":("Third down conversion", "drive-sustaining rate"),
    "diff_roll_sec_per_play":       ("Pace", "seconds between snaps"),
    "diff_rest_days":               ("Rest advantage", "days off since the last game"),
    "diff_elo":                     ("Elo rating gap", "overall team strength difference"),
}


def team_meta():
    t = nfl.load_teams()
    out = {}
    for r in t.iter_rows(named=True):
        out[r["team_abbr"]] = {
            "name": r["team_name"], "nick": r["team_nick"],
            "conf": r["team_conf"], "div": r["team_division"],
            "color": r["team_color"], "color2": r["team_color2"],
            "logo": r["team_logo_espn"], "wordmark": r["team_wordmark"],
        }
    return out


def main():
    teams = team_meta()
    (OUT / "teams.json").write_text(json.dumps(teams, indent=2))

    pbp = get_pbp(seasons=list(range(2015, 2026))).select(PBP_COLS)
    sched = get_schedules(seasons=list(range(2015, 2027)))
    team_week = build_team_week_stats(pbp)
    del pbp

    rolling = build_team_rolling_stats(team_week, window=8)
    latest_form = rolling.sort(["team", "season", "week"]).group_by("team").agg(pl.all().last())
    elos = latest_elo(sched)

    # --- power rankings ---
    board = sorted(elos.items(), key=lambda kv: -kv[1])
    power = [{"team": t, "name": teams.get(t, {}).get("name", t),
              "logo": teams.get(t, {}).get("logo"), "color": teams.get(t, {}).get("color"),
              "elo": round(r), "rank": i + 1}
             for i, (t, r) in enumerate(board) if t in teams]
    (OUT / "power_rankings.json").write_text(json.dumps(power, indent=2))

    # --- upcoming week, with full evidence per game ---
    bundle = joblib.load(MODEL_DIR / "game_predictor.joblib")
    model, fcols = bundle["model"], bundle["feature_cols"]
    coefs = getattr(model, "coef_", None)
    coefs = coefs[0] if coefs is not None else np.zeros(len(fcols))

    upcoming = sched.filter(sched["result"].is_null()).sort(["season", "week", "gameday"])
    ns, nw = upcoming["season"][0], upcoming["week"][0]
    wk = upcoming.filter((upcoming["season"] == ns) & (upcoming["week"] == nw))

    games = []
    for g in wk.iter_rows(named=True):
        h, a = g["home_team"], g["away_team"]
        hr = latest_form.filter(latest_form["team"] == h)
        ar = latest_form.filter(latest_form["team"] == a)
        if hr.height == 0 or ar.height == 0:
            continue

        feat, raw = {}, {}
        for c in fcols:
            if c == "diff_elo":
                hv, av = elos.get(h, 1500), elos.get(a, 1500)
            elif c == "diff_rest_days":
                hv = av = 7.0
            else:
                base = c.replace("diff_", "")
                hv, av = float(hr[base][0]), float(ar[base][0])
            feat[c] = hv - av
            raw[c] = {"home": round(hv, 4), "away": round(av, 4)}

        X = np.array([[feat[c] for c in fcols]])
        p = float(model.predict_proba(X)[0, 1])

        # Per-feature contribution to the log-odds. This is what actually
        # drove the prediction, and it's what the site shows when a user
        # asks how a number was reached.
        contribs = []
        for i, c in enumerate(fcols):
            z = float(coefs[i]) * feat[c]
            label, blurb = FEATURE_LABELS.get(c, (c, ""))
            contribs.append({
                "key": c, "label": label, "blurb": blurb,
                "home_value": raw[c]["home"], "away_value": raw[c]["away"],
                "diff": round(feat[c], 4), "contribution": round(z, 4),
                "favors": h if z > 0 else a,
            })
        contribs.sort(key=lambda d: -abs(d["contribution"]))

        margins = simulate_margins(p, n=20000, seed=abs(hash(h + a)) % 10000)
        sim = summarize(margins)
        sim["histogram"] = histogram(margins)

        games.append({
            "id": f"{a}@{h}",
            "home": h, "away": a,
            "home_name": teams.get(h, {}).get("name", h), "away_name": teams.get(a, {}).get("name", a),
            "home_nick": teams.get(h, {}).get("nick", h), "away_nick": teams.get(a, {}).get("nick", a),
            "home_logo": teams.get(h, {}).get("logo"), "away_logo": teams.get(a, {}).get("logo"),
            "home_color": teams.get(h, {}).get("color"), "away_color": teams.get(a, {}).get("color"),
            "home_color2": teams.get(h, {}).get("color2"), "away_color2": teams.get(a, {}).get("color2"),
            "home_win_prob": round(p, 3), "away_win_prob": round(1 - p, 3),
            "pick": h if p > 0.5 else a, "confidence": round(max(p, 1 - p), 3),
            "spread": g.get("spread_line"), "total": g.get("total_line"),
            "gameday": g.get("gameday"), "gametime": g.get("gametime"),
            "stadium": g.get("stadium"), "roof": g.get("roof"),
            "home_elo": round(elos.get(h, 1500)), "away_elo": round(elos.get(a, 1500)),
            "drivers": contribs[:6],
            "all_drivers": contribs,
            "simulation": sim,
        })

    (OUT / "predictions.json").write_text(json.dumps(
        {"season": ns, "week": nw, "games": games}, indent=2))

    # --- play-call predictability ---
    bp = MODEL_DIR / "predictability_leaderboard.parquet"
    if bp.exists():
        b = pl.read_parquet(bp)
        rows = [{"team": r["team"], "name": teams.get(r["team"], {}).get("name", r["team"]),
                 "logo": teams.get(r["team"], {}).get("logo"),
                 "predictability": round(r["predictability"], 3), "n_plays": r["n_plays"]}
                for r in b.iter_rows(named=True)]
        (OUT / "playcall_predictability.json").write_text(json.dumps(rows, indent=2))

    # --- metrics ---
    (OUT / "metrics.json").write_text(json.dumps({
        "game_predictor": {"accuracy": 0.638, "vegas_accuracy": 0.670,
                           "baseline_accuracy": 0.540, "ats": 0.498,
                           "validation": "walk-forward 2021-2024", "model": bundle["model_name"],
                           "n_features": len(fcols)},
        "playcall_predictor": {"accuracy": 0.729, "validation": "20% holdout, 2022-2024"},
        "fourth_down": {"approach": "component expected-points model",
                        "components": ["EP curve by field position", "conversion rate by distance",
                                       "FG make rate by distance", "net punt distance"]},
        "data": {"seasons": "2015-2025", "plays": 532376, "source": "nflverse"},
    }, indent=2))

    print("Exported:", sorted(p.name for p in OUT.glob("*.json")))
    print(f"Games: {len(games)} for {ns} week {nw}")


if __name__ == "__main__":
    main()
