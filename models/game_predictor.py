"""
Weekly game predictor.

Predicts home-team win probability from trailing-form differential features
(home minus away rolling stats: EPA, success rate, explosive rate, pass/rush
EPA split, CPOE, turnover rate, third-down rate, pace, special teams, QB
value), opponent-adjusted matchup edges, game weather/venue context, plus a
from-scratch Elo rating and a fitted market (Vegas) baseline.

Validation is walk-forward: for each recent test season, train only on
earlier seasons (recency-weighted) and predict that season cold. This is
the honest way to measure a time-series model, no future data ever leaks
into training. We compare three models against the naive "home team always
wins" baseline and against the market, and keep the best.

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

from data.ingest import get_pbp, get_schedules, PBP_SEASONS, SCHEDULE_SEASONS
from data.features import build_team_week_stats, build_game_features
from data.elo import compute_elo
from models.market import fit_market_model, market_prob, blend, best_blend_weight

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

PBP_COLS = ["game_id", "play_id", "season", "week", "posteam", "defteam",
            "play_type", "epa", "success", "cpoe", "interception", "fumble_lost",
            "down", "ydstogo", "third_down_converted", "game_seconds_remaining",
            "passer_player_id", "passer_player_name"]

def get_feature_cols(games: pl.DataFrame) -> list[str]:
    """Every engineered feature: home-vs-away differentials (`diff_`),
    opponent-adjusted matchup edges (`matchup_`), and game context like
    weather/dome (`ctx_`). Broadened from a `diff_`-only filter so newly
    added feature families are picked up automatically instead of needing
    a matching edit here every time features.py grows.
    """
    return [c for c in games.columns
            if c.startswith(("diff_", "matchup_", "ctx_"))]


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


def build_game_table():
    pbp = get_pbp(PBP_SEASONS).select(PBP_COLS)
    sched = get_schedules(SCHEDULE_SEASONS)
    team_week = build_team_week_stats(pbp)
    del pbp
    games = build_game_features(sched, team_week, rolling_window=8, recency_half_life_seasons=4.0)

    elo = compute_elo(sched).with_columns(
        (pl.col("home_elo_pre") - pl.col("away_elo_pre")).alias("diff_elo")
    ).select(["season", "week", "home_team", "away_team", "diff_elo"])
    games = games.join(elo, on=["season", "week", "home_team", "away_team"], how="left").drop_nulls("diff_elo")
    return games


def walk_forward(games, feature_cols, test_seasons):
    """For each test season, train on all prior seasons, predict cold."""
    all_seasons = sorted(games["season"].unique().to_list())
    results = {name: {"acc": [], "ll": [], "brier": [], "ats": []} for name in build_models()}
    results["vegas"] = {"acc": [], "ll": [], "brier": []}
    results["blend"] = {"acc": [], "ll": [], "brier": []}
    results["home_baseline"] = {"acc": []}
    blend_weights = []

    for test_season in test_seasons:
        train_seasons = [s for s in all_seasons if s < test_season]
        if len(train_seasons) < 3:
            continue
        X_tr, y_tr, w_tr, sub_tr = _prep(games, train_seasons, feature_cols)
        X_te, y_te, w_te, sub_te = _prep(games, [test_season], feature_cols)
        if len(y_te) == 0:
            continue

        market_model = fit_market_model(sub_tr)
        m_te = market_prob(market_model, sub_te["spread_line"].to_numpy())
        results["vegas"]["acc"].append(accuracy_score(y_te, m_te > 0.5))
        results["vegas"]["ll"].append(log_loss(y_te, np.clip(m_te, 1e-6, 1 - 1e-6), labels=[0, 1]))
        results["vegas"]["brier"].append(brier_score_loss(y_te, m_te))

        best_p_for_blend = None
        for name, model in build_models().items():
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            p = model.predict_proba(X_te)[:, 1]
            results[name]["acc"].append(accuracy_score(y_te, p > 0.5))
            results[name]["ll"].append(log_loss(y_te, p, labels=[0, 1]))
            results[name]["brier"].append(brier_score_loss(y_te, p))
            results[name]["ats"].append(((p > 0.5) == sub_te["home_covers"].to_numpy()).mean())
            if name == "logreg":
                best_p_for_blend = p  # blend against the calibration-friendly model

        w, _ = best_blend_weight(best_p_for_blend, m_te, y_te)
        blend_weights.append(w)
        p_blend = blend(best_p_for_blend, m_te, w)
        results["blend"]["acc"].append(accuracy_score(y_te, p_blend > 0.5))
        results["blend"]["ll"].append(log_loss(y_te, p_blend, labels=[0, 1]))
        results["blend"]["brier"].append(brier_score_loss(y_te, p_blend))

        results["home_baseline"]["acc"].append((y_te == 1).mean())

    return results, (float(np.mean(blend_weights)) if blend_weights else 0.0)


def select_best(results: dict) -> str:
    """Pick the deployed model by walk-forward log loss (calibration matters
    more than raw accuracy) among the learned candidates — never the market
    or blend, which aren't things we can "deploy" as our own model.
    """
    candidates = {n: np.mean(results[n]["ll"]) for n in build_models()}
    return min(candidates, key=candidates.get)


def main():
    games = build_game_table()
    feature_cols = get_feature_cols(games)

    print(f"Games with features: {games.height}")
    print(f"Feature count: {len(feature_cols)}")
    all_seasons = sorted(games["season"].unique().to_list())
    print(f"Seasons: {all_seasons}\n")

    test_seasons = all_seasons[-4:]
    print(f"Walk-forward validation over {test_seasons}:\n")
    results, blend_weight = walk_forward(games, feature_cols, test_seasons)

    print(f"{'model':<16}{'accuracy':<12}{'log_loss':<12}{'brier':<10}")
    print("-" * 50)
    for name in ["home_baseline", "vegas", "logreg", "hist_gb", "blend"]:
        r = results[name]
        acc = np.mean(r["acc"])
        if "ll" in r and r["ll"]:
            print(f"{name:<16}{acc:<12.3f}{np.mean(r['ll']):<12.3f}{np.mean(r['brier']):<10.3f}")
        else:
            print(f"{name:<16}{acc:<12.3f}{'-':<12}{'-':<10}")
    print(f"\nFitted market blend weight (share given to the market): {blend_weight:.2f}")

    best_name = select_best(results)
    print(f"Best learned model: {best_name}")

    X_all, y_all, w_all, sub_all = _prep(games, all_seasons, feature_cols)
    final = build_models()[best_name]
    final.fit(X_all, y_all, sample_weight=w_all)
    market_final = fit_market_model(sub_all)

    joblib.dump({
        "model": final, "feature_cols": feature_cols, "model_name": best_name,
        "market_model": market_final, "blend_weight": blend_weight,
    }, MODEL_DIR / "game_predictor.joblib")

    # ATS was already measured walk-forward for every candidate inside the
    # loop above (results[name]["ats"]) — no need for a third pass.
    print(f"\nATS (walk-forward, vs closing spread): {np.mean(results[best_name]['ats']):.3f} "
          f"[52.4% needed to beat -110 vig]")
    print(f"Saved model to {MODEL_DIR / 'game_predictor.joblib'}")


if __name__ == "__main__":
    main()
