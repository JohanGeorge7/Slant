"""
Exports real model outputs to JSON for the frontend. Every value here comes
from a trained model or measured nflverse data — nothing is mocked.

The per-game payload is deliberately rich: each game carries the feature
differentials that drove its prediction, both teams' form, official logos
and colors, and a Monte Carlo outcome distribution. That's what lets the
site show *why* a pick was made rather than only the number.

Run on a schedule (see api/app.py's background jobs) so the board and
metrics stay current — this is also what makes the board "auto-advance":
it always exports whichever week has games without a result yet, so once
last week's games are final in the schedule cache, the next call to this
script naturally produces next week's board.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import nflreadpy as nfl
import polars as pl
import joblib

from data.ingest import get_pbp, get_schedules, PBP_SEASONS, SCHEDULE_SEASONS, CURRENT_SEASON
from data.season_rating import season_power_rankings
from data.features import build_team_week_stats, build_team_rolling_stats
from data.elo import latest_elo
from data.qb_value import build_qb_week_value
from models.simulate import simulate_margins, summarize, histogram
from models.predict import build_matchup_features, predict_home_win_prob
from models.metrics import compute_metrics

MODEL_DIR = Path(__file__).parent / "artifacts"
OUT = Path(__file__).parent.parent / "frontend" / "data"
OUT.mkdir(parents=True, exist_ok=True)

PBP_COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam",
            "play_type", "epa", "success", "cpoe", "interception", "fumble_lost",
            "down", "ydstogo", "third_down_converted", "game_seconds_remaining",
            "passer_player_id", "passer_player_name"]

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
    "diff_roll_off_st_epa_per_play":("Special teams (kicking)", "field position value on kicks/punts"),
    "diff_roll_def_st_epa_per_play":("Special teams (return)", "field position value on returns"),
    "diff_roll_off_qb_value":       ("Starting QB value", "trailing EPA/dropback above a backup"),
    "diff_rest_days":               ("Rest advantage", "days off since the last game"),
    "diff_elo":                     ("Elo rating gap", "overall team strength difference"),
    "matchup_epa_edge":             ("Matchup-adjusted EPA edge", "offense vs THIS opponent's defense, not league average"),
    "matchup_success_edge":         ("Matchup-adjusted success edge", "efficiency edge against this specific opponent"),
    "ctx_is_dome":                  ("Indoors", "dome or closed roof — no wind or weather"),
    "ctx_temp":                     ("Temperature", "game-time forecast, degrees F"),
    "ctx_wind":                     ("Wind", "game-time forecast, mph"),
    "diff_roll_ngs_cpoe":           ("Completion % above expected", "QB accuracy beyond what the throw difficulty predicts"),
    "diff_roll_ngs_aggressiveness": ("Tight-window passing rate", "share of throws into tight coverage"),
    "diff_roll_ngs_time_to_throw":  ("Time to throw", "seconds from snap to release"),
    "diff_roll_ngs_separation":     ("Receiver separation", "average distance from the nearest defender at the catch point"),
    "diff_roll_ngs_yac_oe":         ("Yards after catch above expected", "after-catch production beyond what the play design predicts"),
    "diff_roll_ngs_rush_yoe":       ("Rush yards above expected", "rushing production beyond what the blocking and box count predict"),
    "diff_roll_ngs_stacked_box_rate": ("Stacked-box rate faced", "share of rushes against 8+ defenders in the box"),
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

    pbp = get_pbp(PBP_SEASONS).select(PBP_COLS)
    sched = get_schedules(SCHEDULE_SEASONS)
    team_week = build_team_week_stats(pbp)
    qb_week = build_qb_week_value(pbp)
    qb_week.write_parquet(MODEL_DIR / "qb_week_value.parquet")  # reused by models/injury_notes.py
    del pbp

    rolling = build_team_rolling_stats(team_week, window=8)
    latest_form = rolling.sort(["team", "season", "week"]).group_by("team").agg(pl.all().last())
    elos = latest_elo(sched)

    # --- power rankings: THIS season only (see data/season_rating.py). The
    # multi-season Elo stays as a model input and is kept on each row as
    # `elo`, but it is not what ranks the teams on the site.
    power = [{"team": d["team"], "name": teams.get(d["team"], {}).get("name", d["team"]),
              "logo": teams.get(d["team"], {}).get("logo"), "color": teams.get(d["team"], {}).get("color"),
              "rating": d["rating"], "rank": d["rank"], "rank_change": d["rank_change"],
              "record": f"{d['wins']}-{d['losses']}" + (f"-{d['ties']}" if d["ties"] else ""),
              "games": d["games"], "elo": round(elos.get(d["team"], 1500))}
             for d in season_power_rankings(sched, team_week, CURRENT_SEASON) if d["team"] in teams]
    if power and not any(p["games"] for p in power):
        # Before kickoff there are no 2026 games to rate from, so the only
        # honest ordering is last season's carried-over Elo, labelled as such.
        power.sort(key=lambda p: -p["elo"])
        for i, p in enumerate(power, 1):
            p["rank"], p["rating"], p["rank_change"] = i, p["elo"], 0
    (OUT / "power_rankings.json").write_text(json.dumps(power, indent=2))

    # --- upcoming week, with full evidence per game ---
    bundle = joblib.load(MODEL_DIR / "game_predictor.joblib")
    model, fcols = bundle["model"], bundle["feature_cols"]
    coefs = getattr(model, "coef_", None)
    coefs = coefs[0] if coefs is not None else np.zeros(len(fcols))

    # Which week to show: the earliest one with a still-unplayed game. But
    # the actual list of games for that week must come from the FULL
    # schedule, not from `upcoming` -- filtering to unplayed games first
    # and then picking that week's rows out of the already-filtered set
    # silently drops any game in that week that happened to finish before
    # this export ran (an early Thursday/Sunday game, say), so the board
    # would show a partial week (e.g. 6 of 16 games) instead of all of it,
    # and would only ever "complete" the week if every remaining game
    # happened to finish in the same export cycle -- otherwise it just sits
    # there missing games with no error, indistinguishable from working.
    upcoming = sched.filter(sched["result"].is_null()).sort(["season", "week", "gameday"])
    if upcoming.height == 0:
        print("No upcoming games found in the schedule cache — nothing to export for predictions.json.")
        games_out, ns, nw = [], None, None
    else:
        ns, nw = upcoming["season"][0], upcoming["week"][0]
        wk = sched.filter((sched["season"] == ns) & (sched["week"] == nw)).sort("gameday")

        games_out = []
        for g in wk.iter_rows(named=True):
            h, a = g["home_team"], g["away_team"]
            hr = latest_form.filter(latest_form["team"] == h)
            ar = latest_form.filter(latest_form["team"] == a)
            if hr.height == 0 or ar.height == 0:
                continue

            feat, raw = build_matchup_features(
                fcols, hr, ar, elos.get(h, 1500), elos.get(a, 1500),
                roof=g.get("roof"), temp=g.get("temp"), wind=g.get("wind"),
            )
            p = predict_home_win_prob(model, fcols, feat)

            # Per-feature contribution to the log-odds (logreg) or a
            # coefficient-free fallback for tree models: relative feature
            # value magnitude, still ranked but not a true log-odds push.
            contribs = []
            for i, c in enumerate(fcols):
                z = float(coefs[i]) * feat[c] if coefs is not None else feat[c]
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

            games_out.append({
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
                "temp": g.get("temp"), "wind": g.get("wind"),
                "home_elo": round(elos.get(h, 1500)), "away_elo": round(elos.get(a, 1500)),
                "drivers": contribs[:6],
                "all_drivers": contribs,
                "simulation": sim,
            })

    (OUT / "predictions.json").write_text(json.dumps(
        {"season": ns, "week": nw, "games": games_out}, indent=2))

    # --- play-call predictability ---
    bp = MODEL_DIR / "predictability_leaderboard.parquet"
    if bp.exists():
        b = pl.read_parquet(bp)
        rows = [{"team": r["team"], "name": teams.get(r["team"], {}).get("name", r["team"]),
                 "logo": teams.get(r["team"], {}).get("logo"),
                 "predictability": round(r["predictability"], 3), "n_plays": r["n_plays"]}
                for r in b.iter_rows(named=True)]
        (OUT / "playcall_predictability.json").write_text(json.dumps(rows, indent=2))

    # --- metrics: computed fresh, not hand-typed (see models/metrics.py) ---
    (OUT / "metrics.json").write_text(json.dumps(compute_metrics(), indent=2))

    print("Exported:", sorted(p.name for p in OUT.glob("*.json")))
    print(f"Games: {len(games_out)} for season {ns} week {nw}")


if __name__ == "__main__":
    main()
