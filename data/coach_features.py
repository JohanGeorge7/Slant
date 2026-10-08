"""
A rolling, leakage-free version of the play-call predictability leaderboard
(models/playcall_predictor.py's board) for use as an actual model feature.

The leaderboard version fits one classifier on a random 80/20 split across
ALL seasons combined -- perfectly fine for "who's predictable overall,"
completely wrong to feed into a game prediction, because it means a Week 3
prediction could get influenced by how a team called plays in Week 15 of
that same season, which hasn't happened yet at prediction time. That's
lookahead bias -- information the model couldn't have actually had -- and it
would make backtested accuracy look better than what the model could really
achieve live.

This module only computes the raw, SINGLE-WEEK number (how run/pass-mixed a
team was in that one week's games, by situation) -- deliberately not
rolled/shifted here. build_team_week_stats() joins this in as just another
raw per-week stat, and build_team_rolling_stats() (data/features.py) already
turns every raw stat into a trailing, shifted average the same way for
everything else in this pipeline. Rolling it twice (once here, once there)
would double-smooth it and lag it further behind than intended -- one
rolling step, in one place, is the whole point of that shared function.

Situations are bucketed coarsely (down x short/medium/long distance, downs
1-3 only -- the downs where a real run/pass choice exists) rather than by
exact down-and-distance, since even a single week doesn't have enough plays
per team to estimate a narrow bucket reliably.
"""

import polars as pl

DIST_BANDS = [("short", 0, 3), ("medium", 4, 7), ("long", 8, 99)]


def _distance_band() -> pl.Expr:
    expr = pl.lit("long")
    for label, lo, hi in reversed(DIST_BANDS[:-1]):
        expr = pl.when(pl.col("ydstogo") <= hi).then(pl.lit(label)).otherwise(expr)
    return expr


def build_predictability_team_week(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (season, week, team): `coach_predictability`, 0
    (perfectly 50/50 run-pass split across situations that week) to 1
    (always the same call), weighted by how many plays back each situation
    bucket that week so a team's most common situations dominate the
    score, not a rarely-seen one with a noisy tiny sample.
    """
    rp = (
        pbp.filter(
            (pl.col("down") <= 3) & pl.col("play_type").is_in(["run", "pass"])
            & pl.col("down").is_not_null() & pl.col("ydstogo").is_not_null()
        )
        .rename({"posteam": "team"})
        .with_columns(_distance_band().alias("band"))
    )

    weekly = (
        rp.group_by(["season", "week", "team", "down", "band"])
        .agg([
            (pl.col("play_type") == "pass").sum().alias("pass_n"),
            pl.len().alias("play_n"),
        ])
        .with_columns(
            ((pl.col("pass_n") / pl.col("play_n")) - 0.5).abs().mul(2).alias("confidence")
        )
    )

    # Collapse the per-(down, band) rows to one predictability number per
    # team-week: a play-volume-weighted average of confidence, so a team's
    # most common situations (say, standard 1st-and-10) drive the score,
    # not a single 3rd-and-long they happened to face that week.
    return (
        weekly.group_by(["season", "week", "team"])
        .agg(
            ((pl.col("confidence") * pl.col("play_n")).sum() / pl.col("play_n").sum())
            .alias("coach_predictability")
        )
    )
