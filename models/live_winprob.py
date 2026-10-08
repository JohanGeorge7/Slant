"""
Live in-game win probability, updated from the pregame model as the score
and clock actually move — the same "explain the arithmetic" standard as the
rest of the site, applied to a live game instead of a Monday morning report.

METHOD (a Brownian bridge, the standard honest way to do this without
training a whole second model on play-by-play win-probability data that
arrives with a lag): simulate.py already treats a full game's final margin
as Normal(expected_margin, 13.5) around the pregame prediction. A Brownian
bridge treats the running score margin the same way *conditional on time
elapsed*: the remaining uncertainty shrinks toward zero as the clock runs
out, and the final margin is expected to land near a straight-line blend of
"where the score already is" and "where the pregame model expected it to
drift by the end."

    remaining_fraction   = 1 - elapsed_fraction
    drift_left           = pregame_expected_margin * remaining_fraction
    expected_final_margin = current_margin + drift_left
    remaining_stdev       = SIGMA * sqrt(remaining_fraction)
    live_win_prob         = P(final margin > 0) = Phi(expected_final_margin / remaining_stdev)

At kickoff (elapsed=0) this exactly reproduces the pregame probability. At
the final whistle (elapsed=1) it collapses to "whoever is ahead wins" (or a
coin flip if tied). In between it blends the two in a way that's explicit
about how much game is left, which is the actual reason a 1-point lead means
different things in the 2nd quarter versus with 30 seconds on the clock.
"""

import numpy as np
from scipy.stats import norm

from models.simulate import NFL_MARGIN_SIGMA

REGULATION_SECONDS = 3600.0


def live_win_prob(pregame_home_win_prob: float, home_score: int, away_score: int,
                   seconds_remaining: float, overtime: bool = False) -> float:
    """Home-team live win probability given the pregame model output and the
    current game state. `seconds_remaining` is remaining in regulation (0 if
    the game is over); overtime games are treated as effectively "almost no
    time left" plus whoever is ahead, since a tie can't stand.
    """
    p0 = float(np.clip(pregame_home_win_prob, 0.001, 0.999))
    pregame_expected_margin = NFL_MARGIN_SIGMA * norm.ppf(p0)
    current_margin = home_score - away_score

    if overtime:
        # Sudden death: a lead is close to decisive, a tie is a coin flip.
        if current_margin > 0:
            return 0.97
        if current_margin < 0:
            return 0.03
        return 0.5

    remaining_frac = float(np.clip(seconds_remaining / REGULATION_SECONDS, 0.0, 1.0))
    if remaining_frac <= 0:
        if current_margin > 0:
            return 0.99
        if current_margin < 0:
            return 0.01
        return 0.5

    drift_left = pregame_expected_margin * remaining_frac
    expected_final_margin = current_margin + drift_left
    remaining_stdev = NFL_MARGIN_SIGMA * np.sqrt(remaining_frac)
    return float(np.clip(norm.cdf(expected_final_margin / remaining_stdev), 0.01, 0.99))
