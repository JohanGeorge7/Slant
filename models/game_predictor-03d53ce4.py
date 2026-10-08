"""
Weekly game predictor.

Predicts home-team win probability from trailing-form differential features
(home minus away rolling stats: EPA, success rate, explosive rate, pass/rush
EPA split, CPOE, turnover rate, third-down rate, pace) plus rest-day
difference.

Validation is walk-forward: for each recent test season, train only on
earlier seasons (recency-weighted) and predict that season cold. This is
the honest way to measure a time-series model, no future data ever leaks
into training. We compare three models against the naive "home team always
wins" baseline and against Vegas, and keep the best.

Run: uv run python models/game_predictor.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

from data.ingest import get_pbp, get_schedules
from data.features import build_team_week_stats, build_game_features
from data.elo import compute_elo

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

PBP_COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam",
            "play_type", "epa", "success", "cpoe", "interception", "fumble_lost",
            "down", "third_down_converted", "game_seconds_remaining"]


def get_feature_cols(games):
    return [c for c in games.columns if c.startswith("diff_")]


def _prep(games, seasons, feature_cols):
    sub = games.filter(games["season"].is_in(seasons))
    X = sub.select(feature_cols).to_numpy()
    y = sub["home_win"].to_numpy().astype(int)
    w = sub["sample_weight"].to_numpy()
    return X, y, w, sub


def build_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, C=0.5),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=300, max_depth=3, learning_rate=0.03,
            l2_regularization=1.0, random_state=42),
    }


def walk_forward(games, feature_cols, test_seasons):
    """For each test season, train on all prior seasons, predict cold."""
    all_seasons = sorted(games["season"].unique().to_list())
    results = {name: {"acc": [], "ll": [], "brier": []} for name in build_models()}
    results["vegas"] = {"acc": [], "ll": [], "brier": []}
    results["home_baseline"] = {"acc": []}

    for test_season in test_seasons:
        train_seasons = [s for s in all_seasons if s < test_season]
        if len(train_seasons) < 3:
            continue
        X_tr, y_tr, w_tr, _ = _prep(games, train_seasons, feature_cols)
        X_te, y_te, w_te, sub_te = _prep(games, [test_season], feature_cols)
        if len(y_te) == 0:
            continue

        for name, model in build_models().items():
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            p = model.predict_proba(X_te)[:, 1]
            results[name]["acc"].append(accuracy_score(y_te, p > 0.5))
            results[name]["ll"].append(log_loss(y_te, p, labels=[0, 1]))
            results[name]["brier"].append(brier_score_loss(y_te, p))

        # Vegas baseline: implied home win prob from the spread (rough logistic
        # map of spread to win prob, standard ~0.14 slope per point).
        spread = sub_te["spread_line"].to_numpy()
        vegas_p = 1 / (1 + np.exp(-0.14 * (-spread)))  # neg: home favored = neg spread here? see note
        # spread_line is negative when home is UNDERDOG, positive when favored,
        # so a positive spread should raise home win prob.
        vegas_p = 1 / (1 + np.exp(-0.14 * spread))
        results["vegas"]["acc"].append(accuracy_score(y_te, vegas_p > 0.5))
        results["vegas"]["ll"].append(log_loss(y_te, np.clip(vegas_p, 1e-6, 1-1e-6), labels=[0, 1]))
        results["vegas"]["brier"].append(brier_score_loss(y_te, vegas_p))

        results["home_baseline"]["acc"].append((y_te == 1).mean())

    return results


def main():
    pbp = get_pbp().select(PBP_COLS)
    sched = get_schedules()
    team_week = build_team_week_stats(pbp)
    del pbp
    games = build_game_features(sched, team_week, rolling_window=8, recency_half_life_seasons=4.0)

    # Add Elo differential as a feature (one of the strongest simple signals).
    elo = compute_elo(sched).with_columns(
        (pl.col("home_elo_pre") - pl.col("away_elo_pre")).alias("diff_elo")
    ).select(["season", "week", "home_team", "away_team", "diff_elo"])
    games = games.join(elo, on=["season", "week", "home_team", "away_team"], how="left").drop_nulls("diff_elo")

    feature_cols = get_feature_cols(games)

    print(f"Games with features: {games.height}")
    print(f"Feature count: {len(feature_cols)}")
    all_seasons = sorted(games["season"].unique().to_list())
    print(f"Seasons: {all_seasons}\n")

    # Walk-forward over the last 4 seasons we have data for.
    test_seasons = all_seasons[-4:]
    print(f"Walk-forward validation over {test_seasons}:\n")
    results = walk_forward(games, feature_cols, test_seasons)

    print(f"{'model':<16}{'accuracy':<12}{'log_loss':<12}{'brier':<10}")
    print("-" * 50)
    for name in ["home_baseline", "vegas", "logreg", "hist_gb"]:
        r = results[name]
        acc = np.mean(r["acc"])
        if "ll" in r and r["ll"]:
            print(f"{name:<16}{acc:<12.3f}{np.mean(r['ll']):<12.3f}{np.mean(r['brier']):<10.3f}")
        else:
            print(f"{name:<16}{acc:<12.3f}{'-':<12}{'-':<10}")

    # Pick best learned model by log loss (calibration matters more than raw acc).
    candidates = {n: np.mean(results[n]["ll"]) for n in build_models()}
    best_name = min(candidates, key=candidates.get)
    print(f"\nBest learned model: {best_name}")

    # Refit best model on ALL seasons for deployment.
    X_all, y_all, w_all, _ = _prep(games, all_seasons, feature_cols)
    final = build_models()[best_name]
    final.fit(X_all, y_all, sample_weight=w_all)

    joblib.dump({"model": final, "feature_cols": feature_cols, "model_name": best_name},
                MODEL_DIR / "game_predictor.joblib")

    # Honest ATS measurement, walk-forward, using the best model.
    ats = []
    for test_season in test_seasons:
        train_seasons = [s for s in all_seasons if s < test_season]
        if len(train_seasons) < 3:
            continue
        X_tr, y_tr, w_tr, _ = _prep(games, train_seasons, feature_cols)
        X_te, _, _, sub_te = _prep(games, [test_season], feature_cols)
        m = build_models()[best_name]
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        p = m.predict_proba(X_te)[:, 1]
        picks_home = p > 0.5
        covers = sub_te["home_covers"].to_numpy()
        ats.append((picks_home == covers).mean())
    print(f"\nATS (walk-forward, vs closing spread): {np.mean(ats):.3f} "
          f"[52.4% needed to beat -110 vig]")
    print(f"Saved model to {MODEL_DIR / 'game_predictor.joblib'}")


if __name__ == "__main__":
    main()
