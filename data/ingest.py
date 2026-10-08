"""
Shared data ingestion layer.

Pulls play-by-play, schedules, team stats, and FTN charting from nflverse
via nflreadpy, and caches each **season** to its own local parquet file.
Historical seasons are frozen once played, so they're fetched once and
never touched again; only the in-progress season is ever re-downloaded.
That's what makes it cheap to keep the site's data fresh on a schedule
(see refresh_current_season(), used by api/app.py's background job)
instead of the old approach, which blew away and re-pulled the entire
multi-year cache just to pick up one new week.

All three models read from these cached files instead of re-downloading,
since they all need overlapping slices of the same underlying play-by-play
data.

Run this file directly to (re)build the full cache:
    uv run python data/ingest.py
"""

from datetime import date
from pathlib import Path

import nflreadpy as nfl
import polars as pl

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

FIRST_SEASON = 2015


def current_nfl_season(today: date | None = None) -> int:
    """The season label ('2026' for the 2026-27 season) for the season that
    is either in progress or about to start. The NFL year rolls over in
    March (free agency/draft cycle begins), so Jan/Feb still belong to the
    previous season's label.
    """
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


# Single source of truth for "how much history do we use", imported by every
# ingest/training/export script instead of each hardcoding its own range.
# This used to drift silently: ingest.py's own default stopped a season
# earlier than what export_frontend_data.py requested, and the play-call and
# fourth-down models used an even older range than either, so two of the
# three deployed models were quietly training without last season's games.
CURRENT_SEASON = current_nfl_season()
FTN_FIRST_SEASON = 2022

# All of these INCLUDE the in-progress season — see _load_seasons below for
# why that's safe: a season that has no data yet (or that this pinned
# nflreadpy version doesn't recognize yet) is skipped with a warning rather
# than crashing the pipeline, and is picked up automatically the moment
# real data (or a dependency bump) makes it available.
SCHEDULE_SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
PBP_SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
FTN_SEASONS = list(range(FTN_FIRST_SEASON, CURRENT_SEASON + 1))

# Backwards-compatible alias for scripts that only ever want fully-played
# seasons (e.g. anything computing season-level empirical rates).
COMPLETED_SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON))
SEASONS = PBP_SEASONS

# Franchises that relocated during our data window appear under their old
# city's abbreviation in seasons before the move, since nflverse records the
# abbreviation actually used that season. Left un-normalized, every
# consumer downstream (Elo, rolling team-form stats, QB value, ratings)
# silently treats the pre- and post-move team as two unrelated franchises:
# Elo would reset the LA Rams to a fresh 1500 in 2016 instead of carrying
# the St. Louis Rams' actual rating forward, and the ratings board would
# list six teams for three real franchises. Normalizing once here, at the
# shared ingestion boundary, means every model and export downstream of
# get_pbp()/get_schedules()/get_team_stats() sees only the current
# abbreviation without needing its own fix.
TEAM_ABBR_MAP = {
    "STL": "LA",    # Rams: St. Louis -> Los Angeles, 2016
    "SD": "LAC",    # Chargers: San Diego -> Los Angeles, 2017
    "OAK": "LV",    # Raiders: Oakland -> Las Vegas, 2020
}
_TEAM_COLS = ["team", "posteam", "defteam", "home_team", "away_team",
              "recent_team", "opponent_team"]


def _normalize_teams(df: pl.DataFrame) -> pl.DataFrame:
    present = [c for c in _TEAM_COLS if c in df.columns]
    if not present:
        return df
    return df.with_columns([pl.col(c).replace(TEAM_ABBR_MAP).alias(c) for c in present])


def _season_path(name: str, season: int) -> Path:
    return CACHE_DIR / f"{name}_{season}.parquet"


def _load_seasons(name: str, seasons: list[int], loader, current_season: int,
                   force_refresh: bool = False) -> pl.DataFrame:
    """Load one dataframe per season (from cache when present), refetching
    only `current_season` when force_refresh is set — every earlier season
    is immutable once played, so there's never a reason to re-download it.

    The current/in-progress season is allowed to simply not exist yet
    (before kickoff, or before an upstream data source/library adds
    support for it) — that's not an error, it just means "no rows to add
    yet", so we skip it with a note instead of failing the whole load.
    """
    frames = []
    for season in seasons:
        path = _season_path(name, season)
        refetch = force_refresh and season == current_season
        if path.exists() and not refetch:
            frames.append(pl.read_parquet(path))
            continue
        try:
            df = loader(seasons=[season])
        except Exception as e:
            if season == current_season:
                print(f"  [{name}] {season} not available yet ({type(e).__name__}); skipping for now.")
                continue
            raise
        df.write_parquet(path)
        print(f"  [{name}] cached {df.shape[0]:,} rows for {season}")
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def get_pbp(seasons: list[int] = None, force_refresh: bool = False) -> pl.DataFrame:
    """Play-by-play data. This is the core dataset: EPA, down/distance, field
    position, score differential, time remaining, and play type all live
    here. Every model pulls its features from this one table.
    """
    seasons = seasons if seasons is not None else PBP_SEASONS
    return _normalize_teams(_load_seasons("pbp", seasons, nfl.load_pbp, CURRENT_SEASON, force_refresh))


def get_schedules(seasons: list[int] = None, force_refresh: bool = False) -> pl.DataFrame:
    """Game-level schedule data: final scores, rest days, and closing spreads,
    used by the game predictor for both training labels and the ATS check.
    The current (in-progress) season's file is the one that keeps changing
    (new results, updated spreads) all season long, so it's the one that
    gets refreshed.
    """
    seasons = seasons if seasons is not None else SCHEDULE_SEASONS
    return _normalize_teams(_load_seasons("schedules", seasons, nfl.load_schedules, CURRENT_SEASON, force_refresh))


def get_team_stats(seasons: list[int] = None, force_refresh: bool = False) -> pl.DataFrame:
    """Pre-aggregated team-game stats from nflverse. Useful as a sanity check
    against our own EPA aggregation in features.py.
    """
    seasons = seasons if seasons is not None else PBP_SEASONS
    return _normalize_teams(_load_seasons("team_stats", seasons, nfl.load_team_stats, CURRENT_SEASON, force_refresh))


def get_ftn_charting(seasons: list[int] = None, force_refresh: bool = False) -> pl.DataFrame:
    """Play-level charting from FTN: personnel proxies (backfield count, box
    count), no-huddle, play action, RPO, motion, and blitz counts. Only
    available from 2022 onward. Used by the play-call predictor, joined to
    pbp on (nflverse_game_id, nflverse_play_id).
    """
    seasons = seasons if seasons is not None else FTN_SEASONS
    return _load_seasons("ftn_charting", seasons, nfl.load_ftn_charting, CURRENT_SEASON, force_refresh)


def refresh_current_season() -> dict:
    """Re-fetch only the in-progress season across every dataset (cheap: one
    season, not the full multi-year history) and return a small summary.
    This is what the scheduler calls periodically so the site picks up
    newly completed games without a manual re-ingest.
    """
    pbp = get_pbp(force_refresh=True)
    sched = get_schedules(force_refresh=True)
    stats = get_team_stats(force_refresh=True)
    ftn = get_ftn_charting(force_refresh=True)
    return {"pbp_rows": pbp.shape[0], "games": sched.shape[0],
            "team_stat_rows": stats.shape[0], "ftn_rows": ftn.shape[0]}


if __name__ == "__main__":
    pbp = get_pbp(force_refresh=True)
    sched = get_schedules(force_refresh=True)
    stats = get_team_stats(force_refresh=True)
    ftn = get_ftn_charting(force_refresh=True)

    print("\nSummary:")
    print(f"  pbp:          {pbp.shape}")
    print(f"  schedules:    {sched.shape}")
    print(f"  team_stats:   {stats.shape}")
    print(f"  ftn_charting: {ftn.shape}")
    print(f"  current season: {CURRENT_SEASON}")
