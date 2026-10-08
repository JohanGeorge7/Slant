"""
Current injury designations from ESPN's league-wide injuries feed — updated
continuously through the week (not just at the old weekly-report cadence),
which is the closest thing to "real time" injury status actually exists for
an event that mostly happens in practice facilities, not on live TV.

This is intentionally scoped to the question the site needs answered: "is a
given team's starting QB out, doubtful, or questionable this week?" — see
models/injury_notes.py for how that turns into a win-probability note.
Other positions are read too (for a smaller heuristic adjustment) but the
QB case is the only one with a real model-derived number behind it, because
it's the only one we track individual trailing value for (data/qb_value.py).
"""

import re

import nflreadpy as nfl
import polars as pl

from data.espn import get_json, id_to_nflverse_abbr

OUT_STATUSES = {"Out", "Injured Reserve", "Doubtful", "Physically Unable to Perform", "Suspension"}
QUESTIONABLE_STATUSES = {"Questionable"}

_PLAYERCARD_ID_RE = re.compile(r"/id/(\d+)/")


def _athlete_espn_id(athlete: dict) -> str | None:
    """ESPN's injuries feed doesn't give a bare athlete id field, but every
    athlete has a "playercard" link whose URL embeds the same numeric id
    used everywhere else in ESPN's system (and matches nflreadpy's
    depth-chart `espn_id` column) -- e.g.
    https://www.espn.com/nfl/player/_/id/16716/aaron-donald -> "16716".
    """
    for link in athlete.get("links", []):
        if "playercard" in (link.get("rel") or []):
            m = _PLAYERCARD_ID_RE.search(link.get("href", ""))
            if m:
                return m.group(1)
    return None


def build_active_players() -> dict:
    """{nflverse_team_abbr: {espn_id, ...}} from the CURRENT season's depth
    charts (refreshed daily by ESPN), used to filter injury notes down to
    players who are actually on a real roster spot -- as opposed to every
    injured player at a tracked position regardless of relevance.

    This replaced an earlier version that inferred "active" from whether a
    player had recorded a stat in last season's play-by-play. That approach
    silently dropped two real categories of injury that matter most: rookies
    with zero prior-season history (a depth chart lists them the moment
    they're rostered, long before they've taken an NFL snap), and players
    who joined a team's rotation since the last completed season ended (a
    trade, a free-agent signing, a return from retirement) whose most recent
    on-field stats are with a different team or don't exist yet. Depth
    charts are refreshed continuously through the season, so today's chart
    reflects today's roster, not last year's box score.
    """
    from data.ingest import CURRENT_SEASON
    dc = nfl.load_depth_charts([CURRENT_SEASON])
    if dc.height == 0:
        return {}
    latest = dc.filter(pl.col("dt") == dc["dt"].max())
    active: dict = {}
    for team, espn_id in latest.select(["team", "espn_id"]).drop_nulls().iter_rows():
        active.setdefault(team, set()).add(str(espn_id))
    return active


def fetch_all_injuries() -> dict:
    """{nflverse_team_abbr: [{"name", "position", "status", "espn_id"}]} for
    every team with a current injury designation. Swallows network/shape
    errors so a scheduler job can call this every few minutes without ever
    crashing the app if ESPN is briefly unreachable — callers just see an
    empty result.
    """
    try:
        data = get_json("injuries")
    except Exception as e:
        # See live_scores.fetch_scoreboard for why this is logged, not just
        # swallowed -- a sustained ESPN failure should never look identical
        # to "no new injuries" in the server's own console.
        print(f"[injuries] fetch failed: {e}")
        return {}

    out = {}
    for team in data.get("injuries", []):
        abbr = id_to_nflverse_abbr(team.get("id"))
        rows = []
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete", {})
            status = (inj.get("status") or inj.get("type", {}).get("description") or "").strip()
            rows.append({
                "name": athlete.get("displayName", ""),
                "short_name": athlete.get("shortName", ""),
                "position": (athlete.get("position") or {}).get("abbreviation", ""),
                "status": status,
                "detail": (inj.get("details") or {}).get("type", ""),
                "date": inj.get("date"),
                "espn_id": _athlete_espn_id(athlete),
            })
        if abbr:
            out[abbr] = rows
    return out


SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def nflverse_player_key(short_name: str) -> str:
    """ESPN's shortName ("L. Jackson") to nflverse's player-name column
    format ("L.Jackson") used consistently across passer/receiver/rusher/
    tackle/etc. columns, so an injury row can be matched against pbp
    participation or data/qb_value.py's per-player history. Suffixes (Jr.,
    II, ...) are dropped since nflverse names generally omit them.
    Best-effort: rosters rarely have two same-position players sharing an
    initial+surname on the same team.
    """
    parts = short_name.replace(".", "").split()
    while len(parts) > 2 and parts[-1].lower() in SUFFIXES:
        parts = parts[:-1]
    if len(parts) < 2:
        return short_name
    return f"{parts[0][0]}.{parts[-1]}"


def team_injuries(all_injuries: dict, team: str, position: str = None) -> list:
    rows = all_injuries.get(team, [])
    if position:
        rows = [r for r in rows if r["position"] == position]
    return rows
