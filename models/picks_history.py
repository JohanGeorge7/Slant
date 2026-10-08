"""
Persistent, append-only record of what the model picked before each game's
kickoff, plus the eventual result -- the source of truth for the site's
track-record page and for the "locked in before kickoff" note shown in a
game's breakdown once it goes live.

predictions.json is NOT a reliable long-term record of this: per
export_frontend_data.py it only ever holds "whichever week still has
unplayed games," so a finished week's games disappear from it entirely
once the board rolls over to the next one. This file survives that
rollover -- it's written to, never wholesale replaced.

Two-speed settling, to avoid re-reading the (large, on-disk) schedule
parquet on every 8-second live-score tick:
  * settle_from_live() -- cheap, uses the live.json snapshot the live-score
    job already just fetched. Catches most results within ~8s of a game
    ending.
  * settle_from_schedule() -- the slower authoritative fallback from
    nflverse's schedule data, for anything settle_from_live() missed (e.g.
    the server was down when a game ended and it later rolled off ESPN's
    current-scoreboard window). Run on the hourly refresh cadence.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
FRONTEND_DATA = ROOT / "frontend" / "data"
HISTORY_PATH = FRONTEND_DATA / "picks_history.json"


def _load() -> dict:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return {"games": {}}


def _save(history: dict):
    HISTORY_PATH.write_text(json.dumps(history), encoding="utf-8")


def _new_entry(season, week, home, away, home_name, away_name, gameday, gametime,
               home_win_prob, pick, confidence, spread, **extra) -> dict:
    vegas_favorite = None
    if spread is not None and spread != 0:
        vegas_favorite = home if spread > 0 else away
    return {
        "season": season, "week": week,
        "home": home, "away": away,
        "home_name": home_name, "away_name": away_name,
        "gameday": gameday, "gametime": gametime,
        "pregame_home_win_prob": home_win_prob, "pregame_away_win_prob": round(1 - home_win_prob, 3),
        "pick": pick, "confidence": confidence,
        "spread": spread, "vegas_favorite": vegas_favorite,
        "home_score": None, "away_score": None,
        "winner": None, "correct": None, "beat_vegas": None,
        **extra,
    }


def record_pregame_picks():
    """Snapshot any not-yet-tracked game's pregame call.

    This used to skip a game entirely once its kickoff had passed, on the
    theory that recording "late" risked capturing an already-live number as
    if it were the pregame one. That theory was wrong: predictions.json's
    home_win_prob comes from predict_home_win_prob(), a pure function of
    pregame trailing stats -- it is never touched by live score/clock data
    (that only ever lives in live.json, which this function doesn't read).
    So the number is exactly as valid whether it's recorded a minute before
    kickoff or a day after the game ends.

    The guard's actual effect was much worse than the problem it prevented:
    combined with the scheduler's old startup-delay bug (see app.py), any
    game whose kickoff passed before record_pregame_picks() got around to
    running it was silently and *permanently* skipped -- never tracked,
    never settled, never shown as final on the board, with no error and no
    way to recover it after the week rolls over and it drops out of
    predictions.json for good. Recording it late is always correct; not
    recording it at all is a real bug.
    """
    pred_path = FRONTEND_DATA / "predictions.json"
    if not pred_path.exists():
        return
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    history = _load()
    changed = False
    for g in predictions.get("games", []):
        if g["id"] in history["games"]:
            continue
        history["games"][g["id"]] = _new_entry(
            predictions["season"], predictions["week"], g["home"], g["away"],
            g["home_name"], g["away_name"], g.get("gameday"), g.get("gametime"),
            g["home_win_prob"], g["pick"], g["confidence"], g.get("spread"))
        changed = True
    if changed:
        _save(history)


def backfill_missed_games() -> int:
    """Rebuild picks for any already-played game of the current season that
    was never recorded.

    Why this exists: record_pregame_picks() can only snapshot whatever is in
    predictions.json, and that file only ever holds "the earliest week with
    an unplayed game." If the server isn't running during a week (laptop
    closed, app not opened for days), that week is never the "earliest
    unplayed" at any moment the server is up -- on the next start the board
    jumps straight past it and its games are never recorded, which is how
    week 4 vanished from the track record entirely.

    Each missed game is predicted the way it would have been before kickoff:
      * team form  -- the trailing-8-game row for that exact week (the
        rolling table is already shifted so a week never sees itself)
      * Elo        -- the rating going into that game
      * the model  -- refit on PRIOR seasons only, so it has never seen any
        2026 result (the deployed model is trained through the latest games
        and would be grading its own homework)
    Entries are flagged reconstructed=True so the site can say so honestly.
    Returns the number of games added.
    """
    import joblib
    import polars as pl
    from data.ingest import get_schedules, get_pbp, CURRENT_SEASON, PBP_SEASONS
    from data.features import build_team_week_stats, build_team_rolling_stats
    from data.elo import compute_elo
    from models.game_predictor import build_game_table, build_models, _prep, PBP_COLS
    from models.predict import build_matchup_features, predict_home_win_prob

    history = _load()
    sched = get_schedules()
    played = (sched.filter((pl.col("season") == CURRENT_SEASON) & (pl.col("game_type") == "REG")
                           & pl.col("home_score").is_not_null()))
    missing = [g for g in played.iter_rows(named=True)
               if f"{g['away_team']}@{g['home_team']}" not in history["games"]]
    if not missing:
        return 0

    bundle = joblib.load(ROOT / "models" / "artifacts" / "game_predictor.joblib")
    fcols = bundle["feature_cols"]
    games = build_game_table()
    prior = sorted(s for s in games["season"].unique().to_list() if s < CURRENT_SEASON)
    X, y, w, _ = _prep(games, prior, fcols)
    model = build_models()[bundle["model_name"]]
    model.fit(X, y, sample_weight=w)
    del games

    pbp = get_pbp(PBP_SEASONS).select(PBP_COLS)
    rolling = build_team_rolling_stats(build_team_week_stats(pbp), window=8)
    del pbp
    rolling = rolling.filter(pl.col("season") == CURRENT_SEASON)
    elo = {(r["week"], r["home_team"], r["away_team"]): (r["home_elo_pre"], r["away_elo_pre"])
           for r in compute_elo(sched).filter(pl.col("season") == CURRENT_SEASON).iter_rows(named=True)}
    teams_path = FRONTEND_DATA / "teams.json"
    teams = json.loads(teams_path.read_text(encoding="utf-8")) if teams_path.exists() else {}

    added = 0
    for g in sorted(missing, key=lambda r: (r["week"], r["gameday"] or "")):
        h, a, wk = g["home_team"], g["away_team"], g["week"]
        hr = rolling.filter((pl.col("week") == wk) & (pl.col("team") == h))
        ar = rolling.filter((pl.col("week") == wk) & (pl.col("team") == a))
        if hr.height == 0 or ar.height == 0 or (wk, h, a) not in elo:
            print(f"[picks_history] cannot reconstruct {a}@{h} wk{wk}: missing form/Elo")
            continue
        h_elo, a_elo = elo[(wk, h, a)]
        feat, _ = build_matchup_features(fcols, hr, ar, h_elo, a_elo,
                                         roof=g.get("roof"), temp=g.get("temp"), wind=g.get("wind"))
        p = predict_home_win_prob(model, fcols, feat)
        entry = _new_entry(
            CURRENT_SEASON, wk, h, a,
            teams.get(h, {}).get("name", h), teams.get(a, {}).get("name", a),
            g.get("gameday"), g.get("gametime"),
            round(p, 3), h if p > 0.5 else a, round(max(p, 1 - p), 3), g.get("spread_line"),
            reconstructed=True)
        _apply_result(entry, int(g["home_score"]), int(g["away_score"]))
        history["games"][f"{a}@{h}"] = entry
        added += 1
    if added:
        _save(history)
        print(f"[picks_history] reconstructed {added} missed game(s)")
    return added


def _apply_result(g: dict, home_score: int, away_score: int):
    winner = g["home"] if home_score > away_score else (g["away"] if away_score > home_score else None)
    g["home_score"] = home_score
    g["away_score"] = away_score
    g["winner"] = winner
    g["correct"] = (winner == g["pick"]) if winner else None
    g["beat_vegas"] = (winner == g["pick"] and g["pick"] != g["vegas_favorite"]) \
        if (winner and g.get("vegas_favorite")) else None


def settle_from_live():
    live_path = FRONTEND_DATA / "live.json"
    if not live_path.exists():
        return
    live = json.loads(live_path.read_text(encoding="utf-8"))
    history = _load()
    changed = False
    for gid, e in live.get("games", {}).items():
        g = history["games"].get(gid)
        if not g or g["winner"] is not None or not e.get("completed"):
            continue
        _apply_result(g, e["home_score"], e["away_score"])
        changed = True
    if changed:
        _save(history)


def settle_from_schedule():
    import polars as pl
    from data.ingest import get_schedules, CURRENT_SEASON

    history = _load()
    pending = [gid for gid, g in history["games"].items() if g["winner"] is None]
    if not pending:
        return
    sched = get_schedules([CURRENT_SEASON])
    changed = False
    for gid in pending:
        g = history["games"][gid]
        row = sched.filter(
            (pl.col("home_team") == g["home"]) & (pl.col("away_team") == g["away"])
            & (pl.col("week") == g["week"])
        )
        if row.height == 0 or row["home_score"][0] is None:
            continue
        _apply_result(g, int(row["home_score"][0]), int(row["away_score"][0]))
        changed = True
    if changed:
        _save(history)


def main():
    record_pregame_picks()
    backfill_missed_games()
    settle_from_live()
    settle_from_schedule()
    history = _load()
    n = len(history["games"])
    settled = sum(1 for g in history["games"].values() if g["winner"] is not None)
    print(f"picks_history.json: {n} game(s) tracked, {settled} settled")


if __name__ == "__main__":
    main()
