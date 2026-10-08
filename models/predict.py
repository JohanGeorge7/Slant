"""
Shared per-game feature construction for the deployed game predictor.

Used by both export_frontend_data.py (the full site export, with driver
breakdowns and simulations) and predict_week.py (a quick terminal-only
preview). Previously each script built this dict independently, and
predict_week.py's version used the wrong key names entirely (home_roll_*
instead of diff_*), so it would crash the moment `feature_cols` from the
trained model didn't match — it was never actually run after the feature
set moved on. Centralizing it here means there's exactly one place that has
to know how each feature column maps back to raw team/game values.
"""

import numpy as np

DOME_ROOFS = ("dome", "closed")
DEFAULT_REST_DAYS = 7.0   # neutral value across an offseason gap / when unknown
DEFAULT_OUTDOOR_TEMP = 60.0
DEFAULT_OUTDOOR_WIND = 7.0


def build_matchup_features(fcols: list[str], home_row, away_row,
                            home_elo: float, away_elo: float,
                            roof: str = None, temp: float = None, wind: float = None,
                            home_rest_days: float = DEFAULT_REST_DAYS,
                            away_rest_days: float = DEFAULT_REST_DAYS) -> tuple[dict, dict]:
    """Returns (feat, raw): `feat` is {column: model-input value} in the
    exact shape the trained model expects; `raw` is {column: {home, away}}
    for the site's "what drove this pick" display. `home_row`/`away_row`
    are single-row frames from the rolling team-stat table (columns named
    `roll_<stat>`, one row per team from build_team_rolling_stats' output).
    """
    is_dome = (roof or "") in DOME_ROOFS
    ctx_temp = 70.0 if is_dome else (temp if temp is not None else DEFAULT_OUTDOOR_TEMP)
    ctx_wind = 0.0 if is_dome else (wind if wind is not None else DEFAULT_OUTDOOR_WIND)

    feat, raw = {}, {}
    for c in fcols:
        if c == "diff_elo":
            hv, av = home_elo, away_elo
        elif c == "diff_rest_days":
            hv, av = home_rest_days, away_rest_days
        elif c == "matchup_epa_edge":
            ho, ao = float(home_row["roll_off_epa_per_play"][0]), float(away_row["roll_off_epa_per_play"][0])
            hd, ad = float(home_row["roll_def_epa_per_play"][0]), float(away_row["roll_def_epa_per_play"][0])
            feat[c] = (ho - ad) - (ao - hd)
            raw[c] = {"home": round(ho, 4), "away": round(ao, 4)}
            continue
        elif c == "matchup_success_edge":
            ho, ao = float(home_row["roll_off_success_rate"][0]), float(away_row["roll_off_success_rate"][0])
            hd, ad = float(home_row["roll_def_success_rate"][0]), float(away_row["roll_def_success_rate"][0])
            feat[c] = (ho - ad) - (ao - hd)
            raw[c] = {"home": round(ho, 4), "away": round(ao, 4)}
            continue
        elif c == "ctx_is_dome":
            feat[c] = float(is_dome)
            raw[c] = {"home": feat[c], "away": feat[c]}
            continue
        elif c == "ctx_temp":
            feat[c] = ctx_temp
            raw[c] = {"home": feat[c], "away": feat[c]}
            continue
        elif c == "ctx_wind":
            feat[c] = ctx_wind
            raw[c] = {"home": feat[c], "away": feat[c]}
            continue
        else:
            base = c.replace("diff_", "", 1)
            hv, av = float(home_row[base][0]), float(away_row[base][0])
        feat[c] = hv - av
        raw[c] = {"home": round(hv, 4), "away": round(av, 4)}

    return feat, raw


def predict_home_win_prob(model, fcols: list[str], feat: dict) -> float:
    X = np.array([[feat[c] for c in fcols]])
    return float(model.predict_proba(X)[0, 1])
