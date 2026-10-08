"""
In-season team ratings -- the "who's good THIS season" ranking shown on the
site. Deliberately separate from data/elo.py: Elo carries strength over from
past seasons, which is right for predicting games (it's one of the model's
inputs) but wrong for a ranking people read as "who's playing best right now."

Method: opponent-adjusted, shrunk, recency-weighted least squares.

  * Every completed regular-season game this year gives one "how dominant was
    the home team" number built from two things that agree on who played
    well but fail in different ways:
      - scoring margin (capped at +/-24 so garbage-time blowouts don't run
        away) -- what actually happened, but noisy and luck-prone
      - EPA per play margin (offense minus defense, from play-by-play) -- how
        well they actually moved the ball and stopped the other team, which
        strips out a lot of turnover/fumble-bounce luck. The quarterback's
        play and everyone else's show up here, since EPA is built from every
        snap.
    Each is divided by its historical standard deviation (prior seasons only)
    so they can be averaged on one scale.
  * Solve  game_score = rating_home - rating_away + home_field  for all 32
    ratings at once. That's the opponent adjustment: beating a strong team
    moves you more than beating a weak one, and a team's rating is
    consistent with everyone it played.
  * Ridge shrinkage (prior worth ~3 games of average play) keeps small
    samples from producing wild ratings in weeks 1-4 -- one 0-4 start
    shouldn't make a team look like the worst in football, but a month of
    bad play does pull it down hard.
  * Recent games count a bit more than early ones (half-life ~8 weeks), so
    the ranking follows hot and cold streaks.

Output is re-derived from scratch every time the site data is exported, so
the rankings update every week (and after every final) with no state kept.
"""

import numpy as np
import polars as pl

SCALE = 200.0          # rating units -> display points (1500 = average)
PRIOR_GAMES = 3.0      # ridge strength, in "games of average play"
HALF_LIFE_WEEKS = 8.0
MARGIN_CAP = 24
MIN_GAMES_FOR_MODEL = 1


def _game_scores(sched: pl.DataFrame, team_week: pl.DataFrame) -> pl.DataFrame:
    """One row per completed REG game with the raw margin and EPA margin
    from the home team's point of view."""
    net = team_week.select(["season", "week", "team",
                            (pl.col("off_epa_per_play") - pl.col("def_epa_per_play")).alias("net_epa")])
    g = (
        sched.filter((pl.col("game_type") == "REG") & pl.col("home_score").is_not_null())
        .select(["season", "week", "home_team", "away_team", "home_score", "away_score"])
        .join(net.rename({"team": "home_team", "net_epa": "h_net"}), on=["season", "week", "home_team"], how="left")
        .join(net.rename({"team": "away_team", "net_epa": "a_net"}), on=["season", "week", "away_team"], how="left")
        .with_columns([
            (pl.col("home_score") - pl.col("away_score")).clip(-MARGIN_CAP, MARGIN_CAP).alias("margin"),
            ((pl.col("h_net") - pl.col("a_net")) / 2).alias("epa_margin"),
        ])
    )
    return g


def compute_season_ratings(sched: pl.DataFrame, team_week: pl.DataFrame, season: int,
                           through_week: int = None) -> list[dict]:
    """Ratings for `season` using games up to and including `through_week`
    (all completed games when None). Returns a ranked list of dicts with
    team, rating, rank, wins, losses, ties, games."""
    g = _game_scores(sched, team_week)
    prior = g.filter(pl.col("season") < season)
    cur = g.filter(pl.col("season") == season)
    if through_week is not None:
        cur = cur.filter(pl.col("week") <= through_week)

    teams = sorted(set(sched.filter(pl.col("season") == season)["home_team"].to_list())
                   | set(sched.filter(pl.col("season") == season)["away_team"].to_list()))
    rec = {t: [0, 0, 0] for t in teams}
    for r in cur.iter_rows(named=True):
        h, a = r["home_team"], r["away_team"]
        d = r["home_score"] - r["away_score"]
        rec[h][0 if d > 0 else 1 if d < 0 else 2] += 1
        rec[a][0 if d < 0 else 1 if d > 0 else 2] += 1

    # Historical spread of each signal (prior seasons only) for putting them
    # on one scale. Falls back to sensible NFL constants if there's no history.
    sm = float(prior["margin"].std()) if prior.height > 100 else 12.0
    se = float(prior["epa_margin"].drop_nulls().std()) if prior.height > 100 else 0.22

    n = len(teams)
    if cur.height < MIN_GAMES_FOR_MODEL:
        return [{"team": t, "rating": 1500, "rank": i, "wins": 0, "losses": 0, "ties": 0, "games": 0}
                for i, t in enumerate(teams, 1)]

    idx = {t: i for i, t in enumerate(teams)}
    last_week = int(cur["week"].max())
    rows, ys, ws = [], [], []
    for r in cur.iter_rows(named=True):
        x = np.zeros(n + 1)
        x[idx[r["home_team"]]] = 1.0
        x[idx[r["away_team"]]] = -1.0
        x[n] = 1.0  # home-field term
        m = r["margin"] / sm
        e = r["epa_margin"]
        # If a game has no play-by-play yet, fall back to margin alone.
        y = m if e is None else 0.5 * m + 0.5 * (e / se)
        rows.append(x); ys.append(y)
        ws.append(0.5 ** ((last_week - r["week"]) / HALF_LIFE_WEEKS))
    X, y, w = np.array(rows), np.array(ys), np.array(ws)

    # Weighted ridge on the team ratings; home-field term unpenalized.
    lam = np.full(n + 1, PRIOR_GAMES)
    lam[n] = 0.0
    A = X.T @ (X * w[:, None]) + np.diag(lam)
    b = X.T @ (w * y)
    beta = np.linalg.solve(A + 1e-9 * np.eye(n + 1), b)
    r = beta[:n]
    r = r - r.mean()

    out = [{"team": t, "rating": int(round(1500 + SCALE * r[idx[t]])),
            "wins": rec[t][0], "losses": rec[t][1], "ties": rec[t][2],
            "games": sum(rec[t])} for t in teams]
    out.sort(key=lambda d: -d["rating"])
    for i, d in enumerate(out, 1):
        d["rank"] = i
    return out


def season_power_rankings(sched: pl.DataFrame, team_week: pl.DataFrame, season: int) -> list[dict]:
    """Current ranking plus each team's movement since the previous week."""
    played = sched.filter((pl.col("season") == season) & (pl.col("game_type") == "REG")
                          & pl.col("home_score").is_not_null())
    now = compute_season_ratings(sched, team_week, season)
    if played.height:
        wk = int(played["week"].max())
        before = {d["team"]: d["rank"] for d in compute_season_ratings(sched, team_week, season, wk - 1)} \
            if wk > 1 else {}
    else:
        before = {}
    for d in now:
        d["rank"] = d.get("rank", 0)
        d["rank_change"] = (before[d["team"]] - d["rank"]) if d["team"] in before and before else 0
    return now
