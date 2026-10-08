"""
Starting-QB value: how much better or worse a team's offense is expected to
be with its actual starter under center, versus a replacement-level passer.

WHY THIS EXISTS: quarterback is the single largest driver of NFL outcomes
that the original feature set didn't touch at all — team-level trailing EPA
already reflects whoever was playing, but doesn't separate "this team's
offense is good" from "this team's offense is good because of one player who
might not play this week." Tracking QB value on its own lets the game
predictor react to a starter change the moment it's known (see
models/injury_notes.py), instead of waiting 4-8 games for the team-level
rolling average to drift.

Method: identify each team's starter for a given week as whoever had the
most dropbacks, compute a trailing (shifted, so no leakage) EPA-per-dropback
for that player, and define replacement level as the season-wide EPA of
backup-usage passers (fewer than REPLACEMENT_MAX_DROPBACKS in that game).
"diff_qb_value" then feeds into the game predictor exactly like every other
diff_ feature, and the same per-player trailing numbers are reused live: if
a starter is ruled out, we substitute the presumptive backup's own trailing
value (or replacement level if they have none) and re-run the model to get
an honest, model-derived win-probability delta instead of a guessed number.
"""

import polars as pl

MIN_STARTER_DROPBACKS = 10      # per game, to count as "the starter that week"
REPLACEMENT_MAX_DROPBACKS = 10  # below this, a passer counts toward replacement level


def build_qb_week_value(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per team-week: the starter's identity and trailing (shifted)
    EPA-per-dropback, plus that season's replacement level. Designed to be
    left-joined into build_team_week_stats()'s output so it rolls through
    the same trailing-window machinery as every other stat.
    """
    passes = pbp.filter(
        (pl.col("play_type") == "pass")
        & pl.col("epa").is_not_null()
        & pl.col("passer_player_id").is_not_null()
    )

    per_qb_game = (
        passes.group_by(["season", "week", "posteam", "passer_player_id", "passer_player_name"])
        .agg([pl.col("epa").mean().alias("epa_per_dropback"), pl.len().alias("dropbacks")])
    )

    # Replacement level: season-wide average EPA/dropback among backup-usage
    # appearances (a passer who threw a handful of relief snaps), which is a
    # standard proxy for "what a team gets when its starter can't play."
    replacement = (
        per_qb_game.filter(pl.col("dropbacks") < REPLACEMENT_MAX_DROPBACKS)
        .group_by("season")
        .agg(pl.col("epa_per_dropback").mean().alias("replacement_epa"))
    )
    league_replacement = float(per_qb_game.filter(pl.col("dropbacks") < REPLACEMENT_MAX_DROPBACKS)["epa_per_dropback"].mean() or -0.15)

    starters = (
        per_qb_game.filter(pl.col("dropbacks") >= MIN_STARTER_DROPBACKS)
        .sort(["season", "week", "posteam", "dropbacks"], descending=[False, False, False, True])
        .group_by(["season", "week", "posteam"], maintain_order=True)
        .agg(pl.all().first())
        .rename({"posteam": "team"})
    )

    # Trailing (shifted) starter value: same "only see the past" discipline
    # as every other rolling feature. A new starter's rolling average starts
    # thin and blends in gradually, exactly like a coaching change would.
    starters = (
        starters.sort(["team", "season", "week"])
        .with_columns(
            pl.col("epa_per_dropback").shift(1).rolling_mean(window_size=8, min_samples=1).over("team")
            .alias("trailing_qb_value")
        )
        .join(replacement, on="season", how="left")
        .with_columns(pl.col("replacement_epa").fill_null(league_replacement))
    )

    return starters.select(["season", "week", "team", "passer_player_id", "passer_player_name",
                             "dropbacks", "trailing_qb_value", "replacement_epa"])


def attach_qb_value(team_week: pl.DataFrame, qb_week: pl.DataFrame) -> pl.DataFrame:
    """Left-join the starter's trailing value onto the team-week stat table
    as `off_qb_value`, so it flows through build_team_rolling_stats() and
    build_game_features() exactly like every other offensive stat. Missing
    weeks (bye, or too few qualifying passes) fall back to replacement level
    rather than leaving a null that would wipe out the whole row downstream.
    """
    league_replacement = float(qb_week["replacement_epa"].mean() or -0.15)
    joined = team_week.join(
        qb_week.select(["season", "week", "team", "trailing_qb_value"]),
        on=["season", "week", "team"], how="left",
    )
    return joined.with_columns(
        pl.col("trailing_qb_value").fill_null(league_replacement).alias("off_qb_value")
    ).drop("trailing_qb_value")


def current_starters(qb_week: pl.DataFrame) -> pl.DataFrame:
    """Each team's presumptive starter and trailing value — the baseline
    that injury_notes.py perturbs when a starter is reported out.

    Deliberately NOT "whoever started the most recent game": a team that
    already clinched its playoff seeding routinely rests its starter for a
    meaningless Week 18 finale, which would otherwise misidentify the
    backup as the real starter the moment a new season's injury report
    comes in. Instead this takes, within the most recent season on record,
    whichever passer logged the most total dropbacks for that team — one
    rested week doesn't change who the starter actually is — and reports
    that player's own most recent trailing value.
    """
    latest_season = qb_week["season"].max()
    season_rows = qb_week.filter(pl.col("season") == latest_season)
    workload = (season_rows.group_by(["team", "passer_player_id"])
                .agg(pl.col("dropbacks").sum().alias("season_dropbacks")))
    top = (workload.sort(["team", "season_dropbacks"], descending=[False, True])
           .group_by("team", maintain_order=True).agg(pl.all().first()))
    latest_value = (qb_week.sort(["team", "passer_player_id", "season", "week"])
                    .group_by(["team", "passer_player_id"]).agg(pl.all().last()))
    return top.join(latest_value, on=["team", "passer_player_id"], how="left")


def backup_value(qb_week: pl.DataFrame, team: str, exclude_player_id: str) -> float:
    """Best estimate of what a team gets from whoever plays behind the
    given starter: the most-recent trailing value of any OTHER passer
    who has taken meaningful snaps for this team, or replacement level if
    no such history exists (e.g. an unproven backup with no NFL passes).
    """
    others = (qb_week.filter((pl.col("team") == team) & (pl.col("passer_player_id") != exclude_player_id))
              .sort(["season", "week"]))
    if others.height == 0:
        return float(qb_week["replacement_epa"].mean() or -0.15)
    return float(others["trailing_qb_value"][-1])
