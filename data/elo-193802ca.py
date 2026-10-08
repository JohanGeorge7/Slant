"""
NFL Elo ratings, computed from scratch off the schedule. Elo is one of the
single strongest simple predictors of NFL games (FiveThirtyEight built their
whole model around it), because it naturally captures overall team strength,
updates game by game, and carries over across seasons with a regression to
the mean each offseason.

We compute a pregame Elo for both teams for every game, so it can be used as
a feature without leakage (the rating only ever reflects games already
played).
"""

import polars as pl

BASE_RATING = 1500.0
K = 20.0                  # update speed
HOME_ADV = 55.0           # Elo points for home field (~2.2 pts on the spread)
SEASON_REGRESSION = 0.75  # each offseason, pull 25% back toward the mean


def compute_elo(schedules: pl.DataFrame) -> pl.DataFrame:
    """Returns one row per game with home_elo_pre and away_elo_pre (ratings
    BEFORE that game was played). Processes games in chronological order.
    """
    games = (
        schedules
        .filter(pl.col("game_type") == "REG")
        .filter(pl.col("home_score").is_not_null())
        .select(["season", "week", "gameday", "home_team", "away_team",
                 "home_score", "away_score"])
        .with_columns(pl.col("gameday").str.to_date())
        .sort(["gameday", "week"])
    )

    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    rows = []

    for g in games.iter_rows(named=True):
        home, away, season = g["home_team"], g["away_team"], g["season"]

        for team in (home, away):
            if team not in ratings:
                ratings[team] = BASE_RATING
                last_season[team] = season
            # Offseason regression to the mean.
            if season > last_season[team]:
                ratings[team] = BASE_RATING + SEASON_REGRESSION * (ratings[team] - BASE_RATING)
                last_season[team] = season

        home_pre = ratings[home]
        away_pre = ratings[away]

        rows.append({
            "season": season, "week": g["week"],
            "home_team": home, "away_team": away,
            "home_elo_pre": home_pre, "away_elo_pre": away_pre,
        })

        # Expected result and update.
        exp_home = 1.0 / (1.0 + 10 ** (-((home_pre + HOME_ADV) - away_pre) / 400))
        actual_home = 1.0 if g["home_score"] > g["away_score"] else (0.5 if g["home_score"] == g["away_score"] else 0.0)

        # Margin-of-victory multiplier (bigger wins move Elo more, dampened
        # for blowouts and for the favorite, standard 538 approach).
        margin = abs(g["home_score"] - g["away_score"])
        elo_diff = (home_pre + HOME_ADV) - away_pre if actual_home > 0.5 else away_pre - (home_pre + HOME_ADV)
        mov_mult = 2.2 / (0.001 * elo_diff + 2.2) * (margin + 1) ** 0.8 / 7.0
        mov_mult = max(mov_mult, 0.5)

        change = K * mov_mult * (actual_home - exp_home)
        ratings[home] += change
        ratings[away] -= change

    return pl.DataFrame(rows)


def latest_elo(schedules: pl.DataFrame) -> dict[str, float]:
    """Current Elo rating for every team after the last completed game,
    for predicting future/unplayed games.
    """
    elo_games = compute_elo(schedules)
    # Rebuild final ratings by replaying (compute_elo returns pregame only).
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    played = (
        schedules.filter(pl.col("game_type") == "REG")
        .filter(pl.col("home_score").is_not_null())
        .select(["season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"])
        .with_columns(pl.col("gameday").str.to_date())
        .sort(["gameday", "week"])
    )
    for g in played.iter_rows(named=True):
        home, away, season = g["home_team"], g["away_team"], g["season"]
        for team in (home, away):
            if team not in ratings:
                ratings[team] = BASE_RATING
                last_season[team] = season
            if season > last_season[team]:
                ratings[team] = BASE_RATING + SEASON_REGRESSION * (ratings[team] - BASE_RATING)
                last_season[team] = season
        home_pre, away_pre = ratings[home], ratings[away]
        exp_home = 1.0 / (1.0 + 10 ** (-((home_pre + HOME_ADV) - away_pre) / 400))
        actual_home = 1.0 if g["home_score"] > g["away_score"] else (0.5 if g["home_score"] == g["away_score"] else 0.0)
        margin = abs(g["home_score"] - g["away_score"])
        elo_diff = (home_pre + HOME_ADV) - away_pre if actual_home > 0.5 else away_pre - (home_pre + HOME_ADV)
        mov_mult = max(2.2 / (0.001 * elo_diff + 2.2) * (margin + 1) ** 0.8 / 7.0, 0.5)
        change = K * mov_mult * (actual_home - exp_home)
        ratings[home] += change
        ratings[away] -= change
    return ratings
