"""
Next Gen Stats (player-tracking data) rolled up to team-week level: real
per-play execution quality -- separation created, yards after catch beyond
what was expected, rushing yards beyond what the blocking/box count should
have allowed, a QB's actual accuracy above expectation -- rather than the
box-score outcome stats (EPA, success rate) already in build_team_week_stats.

The distinction matters: EPA measures what happened (points added), which a
team can accumulate through opponent mistakes, garbage time, or a bit of
luck. NGS measures how it happened, independent of whether the outcome was
good -- a team can win ugly (see: any bad win against a weak opponent) and
still show mediocre process numbers here, which a pure win/loss or EPA-based
signal can't distinguish.

Coverage starts in 2016 (nflreadpy raises for earlier seasons), one season
short of this app's own training window (2015-). Missing seasons/weeks are
left null here and filled with a league-median at the join site in
data/features.py, the same pattern already used for weather -- so a team's
2015 games aren't silently dropped from training for lacking a stat that
didn't exist yet.

Team abbreviations from this feed match ESPN's convention (LAR, WAS), not
nflverse's (LA, WSH) used everywhere else in this codebase -- remapped here
via the same crosswalk data/espn.py already maintains for that exact reason.
"""

import polars as pl

from data.espn import to_nflverse

FIRST_NGS_SEASON = 2016


def _team_col(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("team_abbr").map_elements(to_nflverse, return_dtype=pl.String).alias("team")
    )


def build_ngs_team_week_stats(seasons: list[int]) -> pl.DataFrame:
    """One row per (season, week, team) with process-quality features,
    volume-weighted where a team fielded multiple players in a stat category
    that week (e.g. two QBs) so the number reflects who actually did the
    work, not a naive average across a starter and a mop-up snap or two.
    """
    ngs_seasons = [s for s in seasons if s >= FIRST_NGS_SEASON]
    if not ngs_seasons:
        return pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64, "team": pl.String})

    import nflreadpy as nfl

    passing = (
        _team_col(nfl.load_nextgen_stats(seasons=ngs_seasons, stat_type="passing"))
        .filter(pl.col("attempts") > 0)
        .group_by(["season", "week", "team"])
        .agg([
            ((pl.col("completion_percentage_above_expectation") * pl.col("attempts")).sum()
             / pl.col("attempts").sum()).alias("ngs_cpoe"),
            ((pl.col("aggressiveness") * pl.col("attempts")).sum()
             / pl.col("attempts").sum()).alias("ngs_aggressiveness"),
            ((pl.col("avg_time_to_throw") * pl.col("attempts")).sum()
             / pl.col("attempts").sum()).alias("ngs_time_to_throw"),
        ])
    )

    receiving = (
        _team_col(nfl.load_nextgen_stats(seasons=ngs_seasons, stat_type="receiving"))
        .filter(pl.col("targets") > 0)
        .group_by(["season", "week", "team"])
        .agg([
            ((pl.col("avg_separation") * pl.col("targets")).sum()
             / pl.col("targets").sum()).alias("ngs_separation"),
            ((pl.col("avg_yac_above_expectation") * pl.col("receptions").clip(lower_bound=0)).sum()
             / pl.col("receptions").sum().clip(lower_bound=1)).alias("ngs_yac_oe"),
        ])
    )

    rushing = (
        _team_col(nfl.load_nextgen_stats(seasons=ngs_seasons, stat_type="rushing"))
        .filter(pl.col("rush_attempts") > 0)
        .group_by(["season", "week", "team"])
        .agg([
            ((pl.col("rush_yards_over_expected_per_att") * pl.col("rush_attempts")).sum()
             / pl.col("rush_attempts").sum()).alias("ngs_rush_yoe"),
            ((pl.col("percent_attempts_gte_eight_defenders") * pl.col("rush_attempts")).sum()
             / pl.col("rush_attempts").sum()).alias("ngs_stacked_box_rate"),
        ])
    )

    out = (
        passing.join(receiving, on=["season", "week", "team"], how="full", coalesce=True)
               .join(rushing, on=["season", "week", "team"], how="full", coalesce=True)
    )
    return out
