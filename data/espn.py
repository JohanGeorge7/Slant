"""
Small shared helper for ESPN's public (unauthenticated, no key required)
site API — used as the live-data source nflverse doesn't provide: real-time
scores/clock during games and continuously-updated injury designations.
nflverse pbp is the right source for everything historical (it's the
official, cleanly-labeled record), but it posts with a delay of hours to
days, so it can't drive anything described as "live."

Team abbreviations differ between the two systems for two franchises; this
crosswalk is the only place that needs to know that.
"""

import time

import requests

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
TIMEOUT = 8

ESPN_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}
NFLVERSE_TO_ESPN = {v: k for k, v in ESPN_TO_NFLVERSE.items()}

# ESPN's numeric team ids are stable and undocumented anywhere but the teams
# endpoint itself; the league-wide injuries feed only gives an id, not an
# abbreviation, so this is fetched once and pinned rather than hit on every
# poll. (Source: GET /apis/site/v2/sports/football/nfl/teams)
ESPN_ID_TO_ABBR = {
    "22": "ARI", "1": "ATL", "33": "BAL", "2": "BUF", "29": "CAR", "3": "CHI",
    "4": "CIN", "5": "CLE", "6": "DAL", "7": "DEN", "8": "DET", "9": "GB",
    "34": "HOU", "11": "IND", "30": "JAX", "12": "KC", "13": "LV", "24": "LAC",
    "14": "LAR", "15": "MIA", "16": "MIN", "17": "NE", "18": "NO", "19": "NYG",
    "20": "NYJ", "21": "PHI", "23": "PIT", "25": "SF", "26": "SEA", "27": "TB",
    "10": "TEN", "28": "WSH",
}


def to_nflverse(espn_abbr: str) -> str:
    return ESPN_TO_NFLVERSE.get(espn_abbr, espn_abbr)


def to_espn(nflverse_abbr: str) -> str:
    return NFLVERSE_TO_ESPN.get(nflverse_abbr, nflverse_abbr)


def id_to_nflverse_abbr(espn_team_id: str) -> str | None:
    abbr = ESPN_ID_TO_ABBR.get(str(espn_team_id))
    return to_nflverse(abbr) if abbr else None


# No custom User-Agent here on purpose -- confirmed by direct testing that
# ESPN's edge (almost certainly Cloudflare) 403s any brand-new, never-seen
# UA string on its very first request regardless of what it looks like (a
# fake browser string and a plain custom identifier both failed cold),
# while requests' own default UA and tools like curl succeeded immediately
# with zero warm-up. A custom identifier is normally the well-behaved
# choice for a polite API client, but here it's actively the thing that
# breaks on a fresh deploy or a rebrand (this exact bug: renaming the UA
# string from "gridiron-..." to "slant-..." silently broke all live data,
# since the old string had already been "seen" and the new one hadn't).
# Letting requests send its own default avoids ever hitting that cold-start
# wall again.


def get_json(path: str, retries: int = 2) -> dict:
    """A transient ESPN hiccup (a brief 5xx, a dropped connection) would
    otherwise just skip that poll cycle entirely -- fine most of the time
    at an 8s cadence, but Sunday's peak concurrent traffic (everyone's live
    win prob updating across a full slate of simultaneous games) is exactly
    when a blip is most likely and most noticeable. One quick retry costs
    nothing when things are healthy and covers the common transient case.

    A 4xx (403, 429, ...) is NOT retried -- that's a block or a rate limit,
    not a blip, and hammering it three times in under a second is more
    likely to look like abusive traffic than to somehow get through.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(f"{BASE}/{path.lstrip('/')}", timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if 400 <= e.response.status_code < 500:
                raise
            last_exc = e
        except requests.exceptions.RequestException as e:
            last_exc = e
        if attempt < retries:
            time.sleep(0.5 * (attempt + 1))
    raise last_exc


ATHLETE_URL = "https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{id}"


def get_athlete_name(espn_id: str) -> str | None:
    """Resolves a bare ESPN athlete id to a real display name -- a separate
    endpoint (different host) from the rest of this module, used as a
    fallback for the rare case where a feed hands us an id with no name
    attached (see models/depth_chart.py: nflverse's depth-chart mirror has
    occasionally shipped a row with a real espn_id but a missing name).
    Best-effort: returns None on any failure rather than raising, since a
    caller falling back to "no name available" is fine, but this should
    never be allowed to take down a whole depth-chart build over one
    player.
    """
    try:
        r = requests.get(ATHLETE_URL.format(id=espn_id), timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return (data.get("athlete") or data).get("displayName")
    except Exception:
        return None
