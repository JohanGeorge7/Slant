"""
Predict the next unplayed week of games using each team's most recent
trailing-form stats (carries over across the season boundary automatically,
since build_team_rolling_stats sorts by team/season/week and rolls over
`.over("team")` without resetting each season, so week 1 of a new season
uses the tail end of last season's form, exactly like a real analyst would).

Run: uv run python models/predict_week.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import polars as pl

from data.ingest import get_pbp, get_schedules
from data.features import build_team_week_stats, build_team_rolling_stats

MODEL_DIR = Path(__file__).parent / "artifacts"


def main():
    # Include the most recently completed season plus history for context.
    seasons = list(range(2015, 2026))
    needed_cols = ["game_id", "play_id", "season", "week", "posteam", "defteam",
                   "play_type", "epa", "game_seconds_remaining"]
    pbp = get_pbp(seasons=seasons).select(needed_cols)
    sched = get_schedules(seasons=seasons + [2026])
    del pbp

    pbp = get_pbp(seasons=seasons).select(needed_cols)
    team_week = build_team_week_stats(pbp)
    rolling = build_team_rolling_stats(team_week, window=6)

    # Most recent rolling row per team = current form heading into the next game.
    latest_form = (
        rolling.sort(["team", "season", "week"])
        .group_by("team")
        .agg(pl.all().last())
    )

    # Next unplayed week.
    upcoming = sched.filter(sched["result"].is_null()).sort(["season", "week", "gameday"])
    next_season = upcoming["season"][0]
    next_week = upcoming["week"][0]
    games = upcoming.filter((upcoming["season"] == next_season) & (upcoming["week"] == next_week))

    bundle = joblib.load(MODEL_DIR / "game_predictor.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]

    print(f"Predictions for season {next_season}, week {next_week}:\n")
    rows = []
    for g in games.iter_rows(named=True):
        home_row = latest_form.filter(latest_form["team"] == g["home_team"])
        away_row = latest_form.filter(latest_form["team"] == g["away_team"])
        if home_row.height == 0 or away_row.height == 0:
            print(f"  {g['away_team']} @ {g['home_team']}: SKIPPED (no historical form found)")
            continue

        # Rest days aren't meaningful across an offseason gap, use a neutral
        # default (matches a normal bye-adjusted week) rather than the
        # literal ~200-day gap since last season, which would be nonsense
        # input to a model trained on in-season rest values.
        feat = {
            "home_roll_off_epa_per_play": home_row["roll_off_epa_per_play"][0],
            "home_roll_def_epa_per_play": home_row["roll_def_epa_per_play"][0],
            "home_roll_sec_per_play": home_row["roll_sec_per_play"][0],
            "home_rest_days": 7.0,
            "away_roll_off_epa_per_play": away_row["roll_off_epa_per_play"][0],
            "away_roll_def_epa_per_play": away_row["roll_def_epa_per_play"][0],
            "away_roll_sec_per_play": away_row["roll_sec_per_play"][0],
            "away_rest_days": 7.0,
        }
        X = np.array([[feat[c] for c in feature_cols]])
        prob_home_win = model.predict_proba(X)[0, 1]

        spread = g.get("spread_line")
        pick = g["home_team"] if prob_home_win > 0.5 else g["away_team"]
        conf = max(prob_home_win, 1 - prob_home_win)

        print(f"  {g['away_team']:3s} @ {g['home_team']:3s}  "
              f"home_win_prob={prob_home_win:.3f}  pick={pick} ({conf:.1%})  "
              f"closing_spread={spread}")
        rows.append({**g, "home_win_prob": float(prob_home_win), "pick": pick})

    out = pl.DataFrame(rows)
    out_path = MODEL_DIR / f"predictions_{next_season}_wk{next_week}.parquet"
    out.write_parquet(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
