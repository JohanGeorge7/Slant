"""
Feature engineering shared across all three models. Each function takes the
cached raw tables from ingest.py and returns a model-ready dataframe.
"""

import polars as pl


# ---------------------------------------------------------------------------
# Game predictor: team-week EPA/pace/rest, joined into home-vs-away game rows
# ---------------------------------------------------------------------------

def build_team_week_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per-team per-week performance stats. Filters to normal plays (run/pass)
    for the rate stats so penalties, kneels, and spikes don't distort them.
    This is a SINGLE WEEK snapshot, not a rolling average, see
    build_team_rolling_stats() for the version used as a prediction feature.

    Produces a deliberately rich set of stable, predictive stats rather than
    EPA alone: success rate, explosive-play rate, pass EPA vs rush EPA split,
    CPOE, turnover rate, third-down conversion, and pace. These are the
    stats that public NFL models have found actually carry predictive signal
    week to week.
    """
    plays = pbp.filter(
        pl.col("play_type").is_in(["run", "pass"]) & pl.col("epa").is_not_null()
    )

    off = (
        plays.group_by(["season", "week", "posteam"])
        .agg([
            pl.col("epa").mean().alias("off_epa_per_play"),
            pl.col("success").mean().alias("off_success_rate"),
            (pl.col("epa") > 0.75).mean().alias("off_explosive_rate"),
            pl.col("epa").filter(pl.col("play_type") == "pass").mean().alias("off_pass_epa"),
            pl.col("epa").filter(pl.col("play_type") == "run").mean().alias("off_rush_epa"),
            pl.col("cpoe").mean().alias("off_cpoe"),
            (pl.col("interception").fill_null(0) + pl.col("fumble_lost").fill_null(0)).mean().alias("off_turnover_rate"),
            pl.col("play_id").count().alias("off_plays"),
        ])
        .rename({"posteam": "team"})
    )

    defn = (
        plays.group_by(["season", "week", "defteam"])
        .agg([
            pl.col("epa").mean().alias("def_epa_per_play"),
            pl.col("success").mean().alias("def_success_rate"),
            (pl.col("epa") > 0.75).mean().alias("def_explosive_rate_allowed"),
            (pl.col("interception").fill_null(0) + pl.col("fumble_lost").fill_null(0)).mean().alias("def_takeaway_rate"),
        ])
        .rename({"defteam": "team"})
    )

    # Third down conversion rate (offense), a stable drive-sustaining stat.
    third = (
        pbp.filter(pl.col("down") == 3)
        .group_by(["season", "week", "posteam"])
        .agg(pl.col("third_down_converted").fill_null(0).mean().alias("off_third_down_rate"))
        .rename({"posteam": "team"})
    )

    # Pace: seconds per play, using game_seconds_remaining deltas.
    pace = (
        plays.sort(["game_id", "posteam", "play_id"])
        .group_by(["season", "week", "posteam"])
        .agg(
            ((pl.col("game_seconds_remaining").max() - pl.col("game_seconds_remaining").min())
             / pl.col("play_id").count()).alias("sec_per_play")
        )
        .rename({"posteam": "team"})
    )

    return (
        off.join(defn, on=["season", "week", "team"], how="inner")
           .join(third, on=["season", "week", "team"], how="left")
           .join(pace, on=["season", "week", "team"], how="inner")
    )


def build_team_rolling_stats(team_week: pl.DataFrame, window: int = 8) -> pl.DataFrame:
    """Trailing N-game average of every team stat, computed BEFORE the current
    week (shift by 1, so week 9's features only ever see weeks 1-8). This is
    what feeds the game predictor at prediction time: it reflects current
    form, so a coaching change or scheme shift shows up in the next week's
    number regardless of how many years of history the model trained on.

    Rows with fewer than `window` prior games are kept but averaged over
    whatever history exists (min_samples=3), so early-season weeks still get
    a usable (if noisier) estimate instead of being dropped entirely.
    """
    stat_cols = [c for c in team_week.columns
                 if c not in ("season", "week", "team", "off_plays")]

    return (
        team_week.sort(["team", "season", "week"])
        .with_columns([
            pl.col(c)
              .shift(1)
              .rolling_mean(window_size=window, min_samples=3)
              .over("team")
              .alias(f"roll_{c}")
            for c in stat_cols
        ])
        .drop_nulls([f"roll_{c}" for c in stat_cols])
        .select(["season", "week", "team"] + [f"roll_{c}" for c in stat_cols])
    )


def build_game_features(schedules: pl.DataFrame, team_week: pl.DataFrame,
                        rolling_window: int = 8, recency_half_life_seasons: float = 3.0) -> pl.DataFrame:
    """One row per game with home/away trailing-form stats, rest days, the
    actual result and closing spread as labels, and a `sample_weight` column
    for recency-weighted training (exponential decay by season).

    Also emits `diff_*` columns (home minus away) for every rolling stat.
    Differentials are what actually drive win probability (it's the gap
    between teams that matters, not each team's raw level), and giving the
    model the difference directly tends to help a linear model most.

    Uses TRAILING form (see build_team_rolling_stats), not single-week or
    season-to-date stats.
    """
    team_week = build_team_rolling_stats(team_week, window=rolling_window)
    roll_cols = [c for c in team_week.columns if c.startswith("roll_")]

    # Rest days from the schedule gap between a team's consecutive games.
    sched_long = pl.concat([
        schedules.select(
            pl.col("season"), pl.col("week"), pl.col("gameday"),
            pl.col("home_team").alias("team")
        ),
        schedules.select(
            pl.col("season"), pl.col("week"), pl.col("gameday"),
            pl.col("away_team").alias("team")
        ),
    ]).with_columns(pl.col("gameday").str.to_date())

    rest = (
        sched_long.sort(["team", "season", "gameday"])
        .with_columns(
            (pl.col("gameday") - pl.col("gameday").shift(1).over("team")).dt.total_days()
            .clip(0, 21)  # cap offseason gaps so they aren't nonsense inputs
            .alias("rest_days")
        )
        .select(["season", "week", "team", "rest_days"])
    )

    feats = team_week.join(rest, on=["season", "week", "team"], how="left")

    home = feats.rename({c: f"home_{c}" for c in feats.columns if c not in ("season", "week", "team")}) \
                .rename({"team": "home_team"})
    away = feats.rename({c: f"away_{c}" for c in feats.columns if c not in ("season", "week", "team")}) \
                .rename({"team": "away_team"})

    games = (
        schedules
        .filter(pl.col("game_type") == "REG")
        .select(["season", "week", "home_team", "away_team", "home_score", "away_score", "spread_line"])
        .join(home, on=["season", "week", "home_team"], how="inner")
        .join(away, on=["season", "week", "away_team"], how="inner")
        .with_columns([
            (pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"diff_{c}")
            for c in roll_cols
        ])
        .with_columns(
            (pl.col("home_rest_days") - pl.col("away_rest_days")).alias("diff_rest_days"),
            (pl.col("home_score") > pl.col("away_score")).alias("home_win"),
            # Cover condition is (home_margin > spread_line), NOT margin +
            # spread_line > 0. In this dataset spread_line is NEGATIVE when
            # the home team is the underdog (e.g. a home +3 dog shows
            # spread_line = -3.0), so the naive "+" version silently mirrors
            # the label for every underdog game. Verified against real
            # results before trusting this.
            (pl.col("home_score") - pl.col("away_score") > pl.col("spread_line")).alias("home_covers"),
            (pl.col("home_score") - pl.col("away_score")).alias("home_margin"),
        )
        .drop_nulls()
    )
    max_season = games.select(pl.col("season").max()).item()
    decay = 0.5 ** (1.0 / recency_half_life_seasons)
    games = games.with_columns(
        (decay ** (max_season - pl.col("season"))).alias("sample_weight")
    )
    return games


# ---------------------------------------------------------------------------
# Play-call predictor: 2022+ only, pbp joined with FTN charting
# ---------------------------------------------------------------------------

def build_playcall_features(pbp: pl.DataFrame, ftn: pl.DataFrame) -> pl.DataFrame:
    """Play-level rows with down/distance/field position/score/time plus FTN's
    personnel proxies and no-huddle/motion indicators. Target is play_type
    (run/pass).

    Deliberately EXCLUDES is_play_action, is_rpo, n_pass_rushers, and
    n_blitzers even though FTN provides them: those are charted from film
    after the play happened and are near-definitionally tied to whether it
    was a pass (e.g. is_play_action is true on a pass 96% of the time),
    so including them leaks the answer and inflates accuracy artificially.
    Only genuinely pre-snap-available signals are kept.
    """
    plays = pbp.filter(
        pl.col("play_type").is_in(["run", "pass"])
        & pl.col("down").is_not_null()
    ).select([
        "game_id", "play_id", "posteam", "season", "week",
        "down", "ydstogo", "yardline_100", "score_differential",
        "game_seconds_remaining", "play_type",
    ])

    ftn_feats = ftn.select([
        "nflverse_game_id", "nflverse_play_id",
        "n_offense_backfield", "n_defense_box",
        "is_no_huddle", "is_motion",
    ]).rename({"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}) \
      .with_columns(pl.col("play_id").cast(pl.Float64))

    return plays.join(ftn_feats, on=["game_id", "play_id"], how="inner").drop_nulls()


# ---------------------------------------------------------------------------
# Fourth down model: state + actual decision, split by decision for the
# three separate expected-points models (go / punt / field goal)
# ---------------------------------------------------------------------------

def build_fourth_down_states(pbp: pl.DataFrame) -> pl.DataFrame:
    """Every 4th down play, labeled with the decision actually made and the
    resulting EPA (used as our expected-points proxy for that decision).
    Kept as one table; the model script splits it three ways by decision so
    each expected-points model only ever sees plays where its own decision
    was actually chosen (avoids the counterfactual problem, see README).
    """
    fourth = pbp.filter(pl.col("down") == 4).with_columns(
        pl.when(pl.col("play_type") == "punt").then(pl.lit("punt"))
          .when(pl.col("play_type") == "field_goal").then(pl.lit("field_goal"))
          .when(pl.col("play_type").is_in(["run", "pass"])).then(pl.lit("go"))
          .otherwise(pl.lit(None))
          .alias("decision")
    ).filter(
        pl.col("decision").is_not_null() & pl.col("epa").is_not_null()
    ).select([
        "game_id", "play_id", "posteam", "season", "week",
        "ydstogo", "yardline_100", "score_differential",
        "game_seconds_remaining", "decision", "epa",
    ])
    return fourth
