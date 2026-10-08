"""
Quick terminal-only preview of the next unplayed week — no JSON written, no
simulation run, just the win probabilities. For the full site export (driver
breakdowns, Monte Carlo distributions, all the JSON files) use
export_frontend_data.py; this is a fast sanity check while iterating on the
model.

Shares its feature construction with export_frontend_data.py via
models/predict.py — an earlier version of this script built its own feature
dict with the wrong key names (home_roll_* instead of the diff_*/matchup_*/
ctx_* names the trained model actually expects) and would have raised a
KeyError the moment it was run against the current model; consolidating
onto one shared function means this can't drift out of sync again.

Run: uv run python models/predict_week.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import polars as pl

from data.ingest import get_pbp, get_schedules, PBP_SEASONS, SCHEDULE_SEASONS
from data.features import build_team_week_stats, build_team_rolling_stats
from data.elo import latest_elo
from models.predict import build_matchup_features, predict_home_win_prob

MODEL_DIR = Path(__file__).parent / "artifacts"


def main():
    needed_cols = ["game_id", "play_id", "season", "week", "posteam", "defteam",
                   "play_type", "epa", "success", "cpoe", "interception", "fumble_lost",
                   "down", "third_down_converted", "game_seconds_remaining",
                   "passer_player_id", "passer_player_name"]
    pbp = get_pbp(PBP_SEASONS).select(needed_cols)
    sched = get_schedules(SCHEDULE_SEASONS)

    team_week = build_team_week_stats(pbp)
    rolling = build_team_rolling_stats(team_week, window=8)
    elos = latest_elo(sched)

    # Most recent rolling row per team = current form heading into the next game.
    latest_form = (
        rolling.sort(["team", "season", "week"])
        .group_by("team")
        .agg(pl.all().last())
    )

    upcoming = sched.filter(sched["result"].is_null()).sort(["season", "week", "gameday"])
    if upcoming.height == 0:
        print("No upcoming games in the schedule cache.")
        return
    next_season, next_week = upcoming["season"][0], upcoming["week"][0]
    games = upcoming.filter((upcoming["season"] == next_season) & (upcoming["week"] == next_week))

    bundle = joblib.load(MODEL_DIR / "game_predictor.joblib")
    model, fcols = bundle["model"], bundle["feature_cols"]

    print(f"Predictions for season {next_season}, week {next_week}:\n")
    rows = []
    for g in games.iter_rows(named=True):
        home_row = latest_form.filter(latest_form["team"] == g["home_team"])
        away_row = latest_form.filter(latest_form["team"] == g["away_team"])
        if home_row.height == 0 or away_row.height == 0:
            print(f"  {g['away_team']} @ {g['home_team']}: SKIPPED (no historical form found)")
            continue

        feat, _ = build_matchup_features(
            fcols, home_row, away_row,
            elos.get(g["home_team"], 1500), elos.get(g["away_team"], 1500),
            roof=g.get("roof"), temp=g.get("temp"), wind=g.get("wind"),
        )
        prob_home_win = predict_home_win_prob(model, fcols, feat)

        spread = g.get("spread_line")
        pick = g["home_team"] if prob_home_win > 0.5 else g["away_team"]
        conf = max(prob_home_win, 1 - prob_home_win)

        print(f"  {g['away_team']:3s} @ {g['home_team']:3s}  "
              f"home_win_prob={prob_home_win:.3f}  pick={pick} ({conf:.1%})  "
              f"closing_spread={spread}")
        rows.append({**g, "home_win_prob": float(prob_home_win), "pick": pick})

    if rows:
        out = pl.DataFrame(rows)
        out_path = MODEL_DIR / f"predictions_{next_season}_wk{next_week}.parquet"
        out.write_parquet(out_path)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
