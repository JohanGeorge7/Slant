"""
Live depth chart for each of this week's teams, with current injury status
folded directly into it — replaces the separate injury-notes list in the
game modal, on the idea that "who's hurt" is only half the question; "who's
playing instead" is the other half, and ESPN's own depth chart already
answers both at once (an OUT player still shows up in their normal slot,
just flagged red, with whoever's next in line visible right below them).

Depth chart order comes from nflreadpy's load_depth_charts (an nflverse
mirror of ESPN's own depth charts, refreshed daily). Injury status is
matched onto it by ESPN player id, not name -- both this and
data/injuries.py's fetch_all_injuries() resolve to the same numeric ESPN
athlete id, which is a much more reliable join than string-matching names
across two different feeds.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

from data.espn import get_athlete_name
from data.injuries import fetch_all_injuries, OUT_STATUSES, QUESTIONABLE_STATUSES

ROOT = Path(__file__).parent.parent
FRONTEND_DATA = ROOT / "frontend" / "data"

OFFENSE_POS = {"QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT"}
DEFENSE_POS = {"LDE", "RDE", "LDT", "RDT", "NT", "LILB", "RILB", "MLB", "WLB", "SLB",
               "LCB", "RCB", "NB", "FS", "SS"}
# Everything else falls into Special Teams (H, KR, LS, P, PK, PR).

# Real depth-chart order (how ESPN's own chart lays a team out), not
# alphabetical -- alphabetical order buries QB behind C/FB/LG/LT, which
# reads as broken to anyone used to seeing a depth chart.
OFFENSE_ORDER = ["QB", "RB", "FB", "WR", "TE", "LT", "LG", "C", "RG", "RT"]
DEFENSE_ORDER = ["LDE", "LDT", "NT", "RDT", "RDE", "WLB", "LILB", "MLB", "RILB", "SLB",
                 "LCB", "SS", "FS", "RCB", "NB"]
ST_ORDER = ["PK", "P", "H", "LS", "KR", "PR"]
POS_ORDER = {pos: i for i, pos in enumerate(OFFENSE_ORDER + DEFENSE_ORDER + ST_ORDER)}

POS_LABEL = {
    "QB": "Quarterback", "RB": "Running Back", "FB": "Fullback", "WR": "Wide Receiver",
    "TE": "Tight End", "LT": "Left Tackle", "LG": "Left Guard", "C": "Center",
    "RG": "Right Guard", "RT": "Right Tackle",
    "LDE": "Left End", "RDE": "Right End", "LDT": "Left Tackle (DT)", "RDT": "Right Tackle (DT)",
    "NT": "Nose Tackle", "LILB": "Inside LB", "RILB": "Inside LB", "MLB": "Mike LB",
    "WLB": "Will LB", "SLB": "Sam LB", "LCB": "Left Corner", "RCB": "Right Corner",
    "NB": "Nickel Back", "FS": "Free Safety", "SS": "Strong Safety",
    "H": "Holder", "KR": "Kick Returner", "LS": "Long Snapper", "P": "Punter",
    "PK": "Kicker", "PR": "Punt Returner",
}


def _group_for(pos_abb: str) -> str:
    if pos_abb in OFFENSE_POS:
        return "Offense"
    if pos_abb in DEFENSE_POS:
        return "Defense"
    return "Special Teams"


def _status_for(espn_id: str | None, injury_by_id: dict) -> tuple[str, str | None]:
    """('active'|'questionable'|'out', display status label or None)."""
    if not espn_id:
        return "active", None
    inj = injury_by_id.get(espn_id)
    if not inj:
        return "active", None
    status = inj["status"]
    if status in OUT_STATUSES:
        return "out", status
    if status in QUESTIONABLE_STATUSES:
        return "questionable", status
    return "active", None


def _resolve_name(player_name: str | None, espn_id: str | None, name_cache: dict) -> str:
    """nflreadpy's depth-chart mirror has occasionally shipped a row with a
    real espn_id but a missing player_name (confirmed: 3 rows league-wide
    on a live pull, including PHI's own starting NT, Jordan Davis) --
    previously that null passed straight through to the frontend and
    rendered as the literal text "null" instead of a name. Never again:
    if the name's missing but we have an id, resolve it from ESPN directly
    (cached per id so a build only ever looks a given player up once);
    if even that fails, fall back to a clearly-labeled placeholder instead
    of leaking a raw null into the page.
    """
    if player_name:
        return player_name
    if not espn_id:
        return "Unlisted"
    if espn_id not in name_cache:
        name_cache[espn_id] = get_athlete_name(espn_id)
    return name_cache[espn_id] or "Unlisted"


def _team_chart(team: str, latest: pl.DataFrame, injury_by_id: dict, name_cache: dict) -> tuple[list, int]:
    rows = (latest.filter(pl.col("team") == team)
                   .sort(["pos_abb", "pos_rank"])
                   .select(["pos_abb", "pos_rank", "player_name", "espn_id"]))
    by_pos: dict[str, list] = {}
    out_count = 0
    for pos_abb, pos_rank, player_name, espn_id in rows.iter_rows():
        player_name = _resolve_name(player_name, espn_id, name_cache)
        status, label = _status_for(espn_id, injury_by_id)
        if status == "out":
            out_count += 1
        by_pos.setdefault(pos_abb, []).append({
            "rank": pos_rank, "name": player_name, "status": status, "status_label": label,
        })
    chart = [
        {"pos": pos, "label": POS_LABEL.get(pos, pos), "group": _group_for(pos),
         "slots": sorted(slots, key=lambda s: s["rank"])[:3]}
        for pos, slots in by_pos.items()
    ]
    group_order = {"Offense": 0, "Defense": 1, "Special Teams": 2}
    chart.sort(key=lambda r: (group_order[r["group"]], POS_ORDER.get(r["pos"], 99), r["pos"]))
    return chart, out_count


def build_depth_charts() -> dict:
    from data.ingest import CURRENT_SEASON
    import nflreadpy as nfl

    pred_path = FRONTEND_DATA / "predictions.json"
    if not pred_path.exists():
        return {"games": {}}
    predictions = json.loads(pred_path.read_text())

    dc = nfl.load_depth_charts([CURRENT_SEASON])
    if dc.height == 0:
        return {"games": {}}
    latest = dc.filter(pl.col("dt") == dc["dt"].max())

    all_injuries = fetch_all_injuries()
    injury_by_id = {
        inj["espn_id"]: inj
        for rows in all_injuries.values() for inj in rows
        if inj.get("espn_id")
    }

    out = {}
    chart_cache: dict[str, tuple] = {}
    name_cache: dict[str, str | None] = {}
    for game in predictions.get("games", []):
        sides = {}
        for side, team in (("home", game["home"]), ("away", game["away"])):
            if team not in chart_cache:
                chart_cache[team] = _team_chart(team, latest, injury_by_id, name_cache)
            chart, out_count = chart_cache[team]
            sides[side] = chart
            sides[f"{side}_out"] = out_count
        out[game["id"]] = sides
    return {"games": out}


def main():
    result = build_depth_charts()
    (FRONTEND_DATA / "depth_charts.json").write_text(json.dumps(result), encoding="utf-8")
    n_out = sum(v.get("home_out", 0) + v.get("away_out", 0) for v in result["games"].values())
    print(f"depth_charts.json: {len(result['games'])} game(s), {n_out} player(s) out")


if __name__ == "__main__":
    main()
