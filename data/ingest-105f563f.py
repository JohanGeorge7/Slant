"""
Shared data ingestion layer.

Pulls play-by-play, schedules, and team stats from nflverse via nflreadpy,
and caches each to a local parquet file, keyed by the season range so a
different season request never silently returns a stale cache from an
earlier call. All three models read from these cached files instead of
re-downloading, since they all need overlapping slices of the same
underlying play-by-play data.

Run this file directly to (re)build the full cache:
    uv run python data/ingest.py
"""

from pathlib import Path

import nflreadpy as nfl
import polars as pl

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Last 10 completed regular seasons. Adjust as seasons roll forward.
SEASONS = list(range(2015, 2025))

# FTN charting (personnel proxies, play action, RPO, motion, blitz counts)
# only exists from 2022 onward, so the play-call model uses this shorter
# window instead of the full 10 seasons. Recent coaching tendencies are
# more relevant to that model anyway.
FTN_SEASONS = list(range(2022, 2025))


def _cache_path(name: str, seasons: list[int]) -> Path:
    return CACHE_DIR / f"{name}_{seasons[0]}_{seasons[-1]}.parquet"


def get_pbp(seasons: list[int] = SEASONS, force_refresh: bool = False) -> pl.DataFrame:
    """Play-by-play data. This is the core dataset: EPA, down/distance, field
    position, score differential, time remaining, and play type all live
    here. Every model pulls its features from this one table.
    """
    path = _cache_path("pbp", seasons)
    if path.exists() and not force_refresh:
        return pl.read_parquet(path)

    print(f"Downloading play-by-play for {seasons[0]}-{seasons[-1]}...")
    pbp = nfl.load_pbp(seasons=seasons)
    pbp.write_parquet(path)
    print(f"Cached {pbp.shape[0]:,} plays to {path}")
    return pbp


def get_schedules(seasons: list[int] = SEASONS, force_refresh: bool = False) -> pl.DataFrame:
    """Game-level schedule data: final scores, rest days, and closing spreads,
    used by the game predictor for both training labels and the ATS check.
    """
    path = _cache_path("schedules", seasons)
    if path.exists() and not force_refresh:
        return pl.read_parquet(path)

    print("Downloading schedules...")
    sched = nfl.load_schedules(seasons=seasons)
    sched.write_parquet(path)
    print(f"Cached {sched.shape[0]:,} games to {path}")
    return sched


def get_team_stats(seasons: list[int] = SEASONS, force_refresh: bool = False) -> pl.DataFrame:
    """Pre-aggregated team-game stats from nflverse. Useful as a sanity check
    against our own EPA aggregation in features.py.
    """
    path = _cache_path("team_stats", seasons)
    if path.exists() and not force_refresh:
        return pl.read_parquet(path)

    print("Downloading team stats...")
    stats = nfl.load_team_stats(seasons=seasons)
    stats.write_parquet(path)
    print(f"Cached {stats.shape[0]:,} team-game rows to {path}")
    return stats


def get_ftn_charting(seasons: list[int] = FTN_SEASONS, force_refresh: bool = False) -> pl.DataFrame:
    """Play-level charting from FTN: personnel proxies (backfield count, box
    count), no-huddle, play action, RPO, motion, and blitz counts. Only
    available from 2022 onward. Used by the play-call predictor, joined to
    pbp on (nflverse_game_id, nflverse_play_id).
    """
    path = _cache_path("ftn_charting", seasons)
    if path.exists() and not force_refresh:
        return pl.read_parquet(path)

    print(f"Downloading FTN charting for {seasons[0]}-{seasons[-1]}...")
    ftn = nfl.load_ftn_charting(seasons=seasons)
    ftn.write_parquet(path)
    print(f"Cached {ftn.shape[0]:,} charted plays to {path}")
    return ftn


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
    print(f"  seasons:      {SEASONS[0]}-{SEASONS[-1]} (ftn: {FTN_SEASONS[0]}-{FTN_SEASONS[-1]})")
