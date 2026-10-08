"""
Turns a current injury designation into a plain-language note on a game
card with an actual model-derived win-probability shift — "Lamar Jackson
(QB) is Out — Ravens win probability drops from 61% to 55.8%" — instead of
either ignoring injuries entirely (the old behavior) or guessing a number.

METHOD FOR THE STARTING QB (the only position with a real per-player value
model behind it, see data/qb_value.py): predictions.json already stores
every feature's home/away value and diff for each game (`all_drivers`).
Reconstruct that exact feature vector, swap in the *backup's* trailing
value (or replacement level if the backup has no track record) for the
affected team's diff_roll_off_qb_value, and re-run the already-trained
game predictor. The resulting probability shift is a real output of the
model, not an eyeballed adjustment.

QUESTIONABLE is down-weighted (WEIGHT_QUESTIONABLE) rather than treated the
same as Out, because a "questionable" starter plays more often than not.

Other positions have no per-player value model (that would need pass-catcher
target share, O-line snap data, etc., well beyond what's built here), so
they get a small flagged heuristic nudge instead of a model-derived number,
and the note says so explicitly rather than implying more precision than
exists.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import polars as pl

from data.injuries import fetch_all_injuries, nflverse_player_key, OUT_STATUSES, QUESTIONABLE_STATUSES
from data.qb_value import current_starters, backup_value
from models.simulate import simulate_margins, summarize, histogram

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "artifacts"
FRONTEND_DATA = ROOT / "frontend" / "data"

WEIGHT_QUESTIONABLE = 0.4   # questionable players suit up more often than not
QB_FEATURE_KEY = "diff_roll_off_qb_value"


def _rebuild_feature_vector(all_drivers: list, fcols: list) -> np.ndarray:
    by_key = {d["key"]: d for d in all_drivers}
    return np.array([[by_key[c]["diff"] if c in by_key else 0.0 for c in fcols]])


def qb_note(game: dict, all_drivers: list, model, fcols: list,
            qb_week: pl.DataFrame, all_injuries: dict) -> dict | None:
    """Check both teams' starting QB status for one game; return the more
    severe note if either (or both) is affected, else None.
    """
    starters = current_starters(qb_week)
    notes = []

    for side, team, opp_team in (("home", game["home"], game["away"]),
                                  ("away", game["away"], game["home"])):
        row = starters.filter(starters["team"] == team)
        if row.height == 0:
            continue
        starter_name = row["passer_player_name"][0]
        starter_id = row["passer_player_id"][0]

        for inj in all_injuries.get(team, []):
            if inj["position"] != "QB":
                continue
            if nflverse_player_key(inj["short_name"]) != starter_name:
                continue
            status = inj["status"]
            if status in OUT_STATUSES:
                weight = 1.0
            elif status in QUESTIONABLE_STATUSES:
                weight = WEIGHT_QUESTIONABLE
            else:
                continue  # "Active" or an unrelated designation — no note

            repl = backup_value(qb_week, team, starter_id)
            starter_val = float(row["trailing_qb_value"][0])
            adjusted_val = starter_val + weight * (repl - starter_val)

            base_key = f"diff_roll_off_qb_value"
            by_key = {d["key"]: dict(d) for d in all_drivers}
            if base_key not in by_key:
                continue
            d = by_key[base_key]
            if side == "home":
                d["diff"] = adjusted_val - d["away_value"]
            else:
                d["diff"] = d["home_value"] - adjusted_val

            X = _rebuild_feature_vector(list(by_key.values()), fcols)
            new_home_p = float(model.predict_proba(X)[0, 1])
            base_home_p = game["home_win_prob"]
            team_p_before = base_home_p if side == "home" else 1 - base_home_p
            team_p_after = new_home_p if side == "home" else 1 - new_home_p

            delta_pts = (team_p_after - team_p_before) * 100
            notes.append({
                "team": team, "opponent": opp_team, "player": inj["name"],
                "position": "QB", "status": status,
                "source": "model",
                "win_prob_before": round(team_p_before, 3),
                "win_prob_after": round(team_p_after, 3),
                "win_prob_delta": round(team_p_after - team_p_before, 3),
                "home_win_prob_after": round(new_home_p, 3),
                "text": (f"{inj['name']} ({team}) is listed {status.lower()}. Model-adjusted: "
                         f"{team} win probability {'drops' if delta_pts < 0 else 'rises'} "
                         f"{abs(delta_pts):.1f} points, from {team_p_before*100:.1f}% to {team_p_after*100:.1f}%."),
            })

    if not notes:
        return None
    notes.sort(key=lambda n: abs(n["win_prob_delta"]), reverse=True)
    return notes[0]



def _qb_adjusted_from_baseline(baseline: dict, game: dict, model, fcols: list,
                                qb_week: pl.DataFrame, all_injuries: dict):
    """Rerun the trained model with each team's starting-QB value pulled
    toward their backup's, weighted by how likely the starter actually
    sits (Out = certain, Questionable = plays more often than not).
    Handles both teams independently so a game with injuries on both
    sides gets both adjustments, not just whichever is "more severe."

    Always computes from `baseline` -- the QB driver's untouched values
    from the last fresh export -- never from whatever's currently sitting
    in predictions.json, since this job reruns every 15 minutes as injury
    news develops through the week and that could already be adjusted from
    a previous run. Recomputing from a fixed baseline every time is what
    keeps repeated runs from compounding the adjustment on top of itself,
    and correctly reverts a game whose QB is later cleared to play.

    Returns (adjusted_home_win_prob, updated_qb_driver_dict) or
    (None, None) if neither team's starter has a live designation right now.
    """
    if baseline is None:
        return None, None
    starters = current_starters(qb_week)
    home_val, away_val = baseline["home_value"], baseline["away_value"]
    changed = False

    for side, team in (("home", game["home"]), ("away", game["away"])):
        row = starters.filter(starters["team"] == team)
        if row.height == 0:
            continue
        starter_name = row["passer_player_name"][0]
        starter_id = row["passer_player_id"][0]
        for inj in all_injuries.get(team, []):
            if inj["position"] != "QB":
                continue
            if nflverse_player_key(inj["short_name"]) != starter_name:
                continue
            status = inj["status"]
            if status in OUT_STATUSES:
                weight = 1.0
            elif status in QUESTIONABLE_STATUSES:
                weight = WEIGHT_QUESTIONABLE
            else:
                break
            repl = backup_value(qb_week, team, starter_id)
            starter_val = float(row["trailing_qb_value"][0])
            adjusted_val = starter_val + weight * (repl - starter_val)
            if side == "home":
                home_val = adjusted_val
            else:
                away_val = adjusted_val
            changed = True
            break

    if not changed:
        return None, None

    by_key = {d["key"]: dict(d) for d in game.get("all_drivers", [])}
    base_key = "diff_roll_off_qb_value"
    d = by_key[base_key]
    d["home_value"], d["away_value"] = home_val, away_val
    d["diff"] = home_val - away_val
    X = _rebuild_feature_vector(list(by_key.values()), fcols)
    new_home_p = float(model.predict_proba(X)[0, 1])

    coefs = getattr(model, "coef_", None)
    coef = float(coefs[0][fcols.index(base_key)]) if coefs is not None and base_key in fcols else None
    d["contribution"] = round(coef * d["diff"], 4) if coef is not None else d["diff"]
    d["favors"] = game["home"] if d["diff"] > 0 else game["away"]
    d["diff"] = round(d["diff"], 4)
    return new_home_p, d


def apply_qb_injury_adjustments():
    """The only injury signal here with a real, validated model behind it
    (starting-QB value, see data/qb_value.py) was previously computed but
    only ever written to injury_notes.json as a side note -- nothing the
    frontend actually displays reads that file, so a starting QB going
    Out never changed the win probability, pick, or confidence anyone
    actually sees. This rewrites predictions.json in place so an injured
    starting QB's impact is the prediction, not a footnote next to it.

    Deliberately QB-only and starters-only: a backup or third-stringer
    going down doesn't move a team's real odds much, and there's no
    validated per-player value model here for other positions the way
    there is for QB -- bolting a guessed adjustment onto the actual
    displayed number for those would risk making predictions worse, not
    better, which defeats the point.
    """
    pred_path = FRONTEND_DATA / "predictions.json"
    qb_path = MODEL_DIR / "qb_week_value.parquet"
    model_path = MODEL_DIR / "game_predictor.joblib"
    if not (pred_path.exists() and qb_path.exists() and model_path.exists()):
        return 0

    predictions = json.loads(pred_path.read_text())
    qb_week = pl.read_parquet(qb_path)
    bundle = joblib.load(model_path)
    model, fcols = bundle["model"], bundle["feature_cols"]
    all_injuries = fetch_all_injuries()

    n_touched = 0
    for game in predictions.get("games", []):
        base_key = "diff_roll_off_qb_value"
        qb_entry = next((d for d in game.get("all_drivers", []) if d["key"] == base_key), None)
        if qb_entry is None:
            continue
        # Captured once per fresh export (see export_frontend_data.py ->
        # _job_injury_notes ordering in api/app.py) and reused by every
        # later run of this same job within that export cycle. Stores the
        # FULL driver dict (not just the two raw values) so an unadjusted
        # revert restores diff/contribution/favors consistently too.
        baseline = game.setdefault("_qb_baseline", {
            "home_win_prob": game["home_win_prob"], "driver": dict(qb_entry),
        })
        baseline_vals = {"home_value": baseline["driver"]["home_value"],
                          "away_value": baseline["driver"]["away_value"]}

        new_p, qb_driver = _qb_adjusted_from_baseline(baseline_vals, game, model, fcols, qb_week, all_injuries)
        was_adjusted = new_p is not None
        if new_p is None:
            new_p = baseline["home_win_prob"]

        game["home_win_prob"] = round(new_p, 3)
        game["away_win_prob"] = round(1 - new_p, 3)
        game["pick"] = game["home"] if new_p > 0.5 else game["away"]
        game["confidence"] = round(max(new_p, 1 - new_p), 3)
        game["injury_adjusted"] = was_adjusted
        for d in game["all_drivers"]:
            if d["key"] == base_key:
                d.update(qb_driver if was_adjusted else baseline["driver"])
        game["drivers"] = sorted(game["all_drivers"], key=lambda x: -abs(x["contribution"]))[:6]
        margins = simulate_margins(new_p, n=20000, seed=abs(hash(game["home"] + game["away"])) % 10000)
        sim = summarize(margins)
        sim["histogram"] = histogram(margins)
        game["simulation"] = sim
        n_touched += 1

    if n_touched:
        pred_path.write_text(json.dumps(predictions, indent=2))
    return sum(1 for g in predictions.get("games", []) if g.get("injury_adjusted"))


def build_injury_notes() -> dict:
    pred_path = FRONTEND_DATA / "predictions.json"
    qb_path = MODEL_DIR / "qb_week_value.parquet"
    model_path = MODEL_DIR / "game_predictor.joblib"
    if not (pred_path.exists() and qb_path.exists() and model_path.exists()):
        return {"games": {}}

    predictions = json.loads(pred_path.read_text())
    qb_week = pl.read_parquet(qb_path)
    bundle = joblib.load(model_path)
    model, fcols = bundle["model"], bundle["feature_cols"]
    all_injuries = fetch_all_injuries()

    out = {}
    for game in predictions.get("games", []):
        top = qb_note(game, game.get("all_drivers", []), model, fcols, qb_week, all_injuries)
        if top:
            out[game["id"]] = [top]
    return {"games": out}


def main():
    result = build_injury_notes()
    (FRONTEND_DATA / "injury_notes.json").write_text(json.dumps(result, indent=1))
    n = sum(len(v) for v in result["games"].values())
    print(f"injury_notes.json: {n} note(s) across {len(result['games'])} game(s)")
    n_adjusted = apply_qb_injury_adjustments()
    print(f"predictions.json: {n_adjusted} game(s) with a starting-QB injury adjustment applied")


if __name__ == "__main__":
    main()
