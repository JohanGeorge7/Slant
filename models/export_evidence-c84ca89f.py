"""
Exports the data behind the charts: calibration, feature weights, the
expected-points curve, conversion and field-goal curves, and full
decision grids for the play-call and fourth-down models.

The point of these is to show the working. Every chart on the site is
generated from measured data or a trained model, not drawn by hand.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score, log_loss

from data.ingest import get_pbp, get_schedules
from data.features import build_team_week_stats, build_game_features
from data.elo import compute_elo
from models.game_predictor import PBP_COLS, get_feature_cols, build_models, _prep
from models.playcall_predictor import CLASSES as PC_CLASSES
from models.fourth_down_model import FourthDownModel, decision_is_feasible, fg_attempt_distance

OUT = Path(__file__).parent.parent / "frontend" / "data"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(__file__).parent / "artifacts"

FEATURE_LABELS = {
    "diff_roll_off_epa_per_play": "Offensive EPA/play",
    "diff_roll_def_epa_per_play": "Defensive EPA allowed",
    "diff_roll_off_success_rate": "Offensive success rate",
    "diff_roll_def_success_rate": "Defensive success allowed",
    "diff_roll_off_explosive_rate": "Explosive play rate",
    "diff_roll_def_explosive_rate_allowed": "Explosives allowed",
    "diff_roll_off_pass_epa": "Pass efficiency",
    "diff_roll_off_rush_epa": "Rush efficiency",
    "diff_roll_off_cpoe": "CPOE",
    "diff_roll_off_turnover_rate": "Giveaway rate",
    "diff_roll_def_takeaway_rate": "Takeaway rate",
    "diff_roll_off_third_down_rate": "Third down rate",
    "diff_roll_sec_per_play": "Pace",
    "diff_rest_days": "Rest advantage",
    "diff_elo": "Elo gap",
}


def game_evidence():
    """Walk-forward predictions on held-out seasons -> calibration curve,
    accuracy by season, and the model's learned feature weights.
    """
    pbp = get_pbp().select(PBP_COLS)
    sched = get_schedules()
    tw = build_team_week_stats(pbp)
    del pbp
    games = build_game_features(sched, tw, rolling_window=8, recency_half_life_seasons=4.0)
    elo = compute_elo(sched).with_columns(
        (pl.col("home_elo_pre") - pl.col("away_elo_pre")).alias("diff_elo")
    ).select(["season", "week", "home_team", "away_team", "diff_elo"])
    games = games.join(elo, on=["season", "week", "home_team", "away_team"], how="left").drop_nulls("diff_elo")

    fcols = get_feature_cols(games)
    seasons = sorted(games["season"].unique().to_list())
    test_seasons = seasons[-4:]

    all_p, all_y, by_season = [], [], []
    for ts in test_seasons:
        tr = [s for s in seasons if s < ts]
        if len(tr) < 3:
            continue
        Xtr, ytr, wtr, _ = _prep(games, tr, fcols)
        Xte, yte, _, sub = _prep(games, [ts], fcols)
        m = build_models()["logreg"]
        m.fit(Xtr, ytr, sample_weight=wtr)
        p = m.predict_proba(Xte)[:, 1]
        all_p.append(p); all_y.append(yte)
        # Vegas comparison for the same games
        spread = sub["spread_line"].to_numpy()
        vp = 1 / (1 + np.exp(-0.14 * spread))
        by_season.append({
            "season": int(ts), "n": int(len(yte)),
            "model": round(float(accuracy_score(yte, p > .5)), 4),
            "vegas": round(float(accuracy_score(yte, vp > .5)), 4),
            "baseline": round(float((yte == 1).mean()), 4),
            "ats": round(float(((p > .5) == sub["home_covers"].to_numpy()).mean()), 4),
        })

    p = np.concatenate(all_p); y = np.concatenate(all_y)

    # Calibration: bucket predictions, compare predicted vs actual win rate.
    bins = np.arange(0.05, 1.0, 0.1)
    cal = []
    for lo in np.arange(0.0, 1.0, 0.1):
        m_ = (p >= lo) & (p < lo + 0.1)
        if m_.sum() >= 20:
            cal.append({"bucket": round(float(lo + .05), 2),
                        "predicted": round(float(p[m_].mean()), 4),
                        "actual": round(float(y[m_].mean()), 4),
                        "n": int(m_.sum())})

    # Feature weights from a model fit on everything.
    Xa, ya, wa, _ = _prep(games, seasons, fcols)
    final = build_models()["logreg"]; final.fit(Xa, ya, sample_weight=wa)
    sd = Xa.std(axis=0)
    weights = sorted([
        {"feature": c, "label": FEATURE_LABELS.get(c, c),
         "weight": round(float(final.coef_[0][i]), 4),
         "std_impact": round(float(abs(final.coef_[0][i] * sd[i])), 4)}
        for i, c in enumerate(fcols)], key=lambda d: -d["std_impact"])

    return {"calibration": cal, "by_season": by_season, "weights": weights,
            "overall": {"accuracy": round(float(accuracy_score(y, p > .5)), 4),
                        "log_loss": round(float(log_loss(y, p)), 4), "n": int(len(y))}}


def playcall_evidence():
    b = joblib.load(MODEL_DIR / "playcall_predictor.joblib")
    clf, F = b["model"], b["feature_cols"]
    # Decision grid: for each down and distance, model probabilities at a
    # neutral game state. Powers the interactive heatmap.
    grid = []
    for dn in [1, 2, 3, 4]:
        for dist in range(1, 26):
            pr = clf.predict_proba(np.array([[dn, dist, 60, 0, 2400]]))[0]
            grid.append({"down": dn, "ydstogo": dist,
                         **{c: round(float(pr[i]), 4) for i, c in enumerate(PC_CLASSES)}})
    return {"grid": grid, "empirical": b["empirical"],
            "accuracy_4class": b["accuracy_4class"], "accuracy_runpass": b["accuracy_runpass"]}


def fourth_down_evidence():
    d = joblib.load(MODEL_DIR / "fourth_down_model.joblib")
    fd = FourthDownModel(d["ep_curve"], d["conversion"], d["fg"], d["punt_net"])
    ep_curve = [{"yardline_100": int(y), "ep": round(float(d["ep_curve"][y]), 3)} for y in range(1, 100)]
    conv = [{"ydstogo": k, "rate": round(v[0], 4), "n": v[1]} for k, v in sorted(d["conversion"].items())]
    fg = [{"distance": k, "rate": round(v[0], 4), "n": v[1]} for k, v in sorted(d["fg"].items())]
    # Decision grid across field position and distance.
    grid = []
    for yl in range(5, 100, 5):
        for dist in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
            r = fd.evaluate(dist, yl)
            grid.append({"yardline_100": yl, "ydstogo": dist,
                         "decision": r["recommended_decision"],
                         "margin": r["margin"], "strength": r["strength"],
                         "ep": r["expected_points"]})
    return {"ep_curve": ep_curve, "conversion": conv, "fg": fg, "grid": grid}


def main():
    ev = {"game": game_evidence(), "playcall": playcall_evidence(), "fourth_down": fourth_down_evidence()}
    (OUT / "evidence.json").write_text(json.dumps(ev, indent=1))
    g = ev["game"]
    print(f"calibration buckets: {len(g['calibration'])}")
    print(f"seasons: {[s['season'] for s in g['by_season']]}")
    print(f"top features: {[w['label'] for w in g['weights'][:4]]}")
    print(f"playcall grid: {len(ev['playcall']['grid'])} cells")
    print(f"4th down grid: {len(ev['fourth_down']['grid'])} cells, "
          f"ep curve {len(ev['fourth_down']['ep_curve'])} pts")
    print(f"wrote {(OUT/'evidence.json').stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
