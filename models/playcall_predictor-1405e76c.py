"""
Play-call predictor — four classes: run, pass, punt, field goal.

WHY THIS WAS REBUILT:
The first version trained only on run/pass plays, so it had never seen a
punt. Asked about 4th-and-20 it confidently answered "pass", which is
badly misleading: in ten seasons teams punt 74% of the time on 4th-and-11+
and pass only 5%. It also relied on FTN charting features (backfield count,
box count, motion) that the interface had to hardcode, and those hardcoded
values dominated the prediction — 2nd-and-3 came back "pass" when teams
actually run it 57% of the time.

This version predicts all four real options from situation alone: down,
distance, field position, score differential, and time. Those are always
available, need no hardcoded inputs, and cover every season from 2015.

Two models are trained:
  * a four-class model over every play (what will they DO)
  * a run/pass model over downs 1-3 only (given they're running a play,
    which is it) — this is what the predictability leaderboard uses, since
    predictability only means something where a real choice exists.

Run: uv run python models/playcall_predictor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from xgboost import XGBClassifier

from data.ingest import get_pbp

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = ["down", "ydstogo", "yardline_100", "score_differential", "game_seconds_remaining"]
CLASSES = ["run", "pass", "punt", "field_goal"]


def load_plays():
    pbp = get_pbp().select(
        ["season", "week", "posteam", "play_type", "down", "ydstogo", "yardline_100",
         "score_differential", "game_seconds_remaining"]
    )
    return pbp.filter(
        pl.col("down").is_not_null()
        & pl.col("play_type").is_in(CLASSES)
        & pl.col("yardline_100").is_not_null()
        & pl.col("score_differential").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
    )


def empirical_tables(plays: pl.DataFrame) -> dict:
    """League-wide rates straight from the data, so the interface can show
    'here is what teams actually did' next to the model's answer.
    """
    capped = plays.with_columns(pl.min_horizontal(pl.col("ydstogo"), 21).alias("d"))
    agg = (capped.group_by(["down", "d"])
                 .agg([(pl.col("play_type") == c).mean().alias(c) for c in CLASSES]
                      + [pl.len().alias("n")])
                 .filter(pl.col("n") >= 30)
                 .sort(["down", "d"]))
    out = {}
    for r in agg.iter_rows(named=True):
        out[f"{int(r['down'])}_{int(r['d'])}"] = {
            **{c: round(float(r[c]), 4) for c in CLASSES}, "n": int(r["n"])
        }
    return out


def main():
    plays = load_plays()
    print(f"Plays: {plays.height:,}")
    print(plays.group_by("play_type").len().sort("len", descending=True))

    y_all = plays["play_type"].to_numpy()
    X_all = plays.select(FEATURES).to_numpy()
    idx = {c: i for i, c in enumerate(CLASSES)}
    y_enc = np.array([idx[v] for v in y_all])

    Xtr, Xte, ytr, yte = train_test_split(X_all, y_enc, test_size=0.2,
                                          random_state=42, stratify=y_enc)
    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.06,
                        subsample=0.85, colsample_bytree=0.85,
                        objective="multi:softprob", num_class=4, eval_metric="mlogloss")
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)
    pred = prob.argmax(1)
    print(f"\n4-class holdout accuracy: {accuracy_score(yte, pred):.3f}  "
          f"log_loss: {log_loss(yte, prob, labels=list(range(4))):.3f}  (n={len(yte):,})")
    cm = confusion_matrix(yte, pred)
    print("\nconfusion (rows=actual, cols=predicted) " + " ".join(f"{c:>11}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c:11s} " + " ".join(f"{v:11d}" for v in cm[i]))

    # --- run/pass model on downs 1-3, for the predictability leaderboard ---
    rp = plays.filter((pl.col("down") <= 3) & pl.col("play_type").is_in(["run", "pass"]))
    Xr = rp.select(FEATURES).to_numpy()
    yr = (rp["play_type"] == "pass").to_numpy().astype(int)
    teams = rp["posteam"].to_numpy()
    Xr_tr, Xr_te, yr_tr, yr_te, _, t_te = train_test_split(
        Xr, yr, teams, test_size=0.2, random_state=42, stratify=yr)
    rpm = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.06,
                        subsample=0.85, colsample_bytree=0.85, eval_metric="logloss")
    rpm.fit(Xr_tr, yr_tr)
    p_pass = rpm.predict_proba(Xr_te)[:, 1]
    print(f"\nrun/pass (downs 1-3) accuracy: {accuracy_score(yr_te, p_pass > .5):.3f}  (n={len(yr_te):,})")

    conf = np.abs(p_pass - 0.5) * 2
    board = (pl.DataFrame({"team": t_te, "confidence": conf})
             .group_by("team").agg([pl.col("confidence").mean().alias("predictability"),
                                    pl.len().alias("n_plays")])
             .filter(pl.col("n_plays") >= 100).sort("predictability", descending=True))
    board.write_parquet(MODEL_DIR / "predictability_leaderboard.parquet")
    print("\nMost predictable:", board.head(3)["team"].to_list(),
          "| Least:", board.tail(3)["team"].to_list())

    tables = empirical_tables(plays)
    joblib.dump({
        "model": clf, "feature_cols": FEATURES, "classes": CLASSES,
        "runpass_model": rpm,
        "accuracy_4class": round(float(accuracy_score(yte, pred)), 4),
        "accuracy_runpass": round(float(accuracy_score(yr_te, p_pass > .5)), 4),
        "empirical": tables,
    }, MODEL_DIR / "playcall_predictor.joblib")
    print(f"\nSaved model + {len(tables)} empirical situation rows to {MODEL_DIR}")

    print("\n--- Sanity checks (cases the old model got wrong) ---")
    for dn, dist, yl, label in [
        (4, 20, 60, "4th & 20, own 40"),
        (4, 15, 70, "4th & 15, own 30"),
        (4, 2, 45, "4th & 2, midfield"),
        (4, 5, 25, "4th & 5, opp 25"),
        (2, 3, 55, "2nd & 3"),
        (2, 10, 55, "2nd & 10"),
        (3, 1, 50, "3rd & 1"),
        (3, 8, 50, "3rd & 8"),
        (1, 10, 75, "1st & 10"),
    ]:
        x = np.array([[dn, dist, yl, 0, 1800]])
        pr = clf.predict_proba(x)[0]
        top = CLASSES[int(pr.argmax())]
        parts = " ".join(f"{c}={pr[i]:.0%}" for i, c in enumerate(CLASSES) if pr[i] > .04)
        print(f"  {label:22s} -> {top:11s} ({parts})")


if __name__ == "__main__":
    main()
