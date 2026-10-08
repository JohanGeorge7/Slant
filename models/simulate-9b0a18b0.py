"""
Monte Carlo game simulator. Turns the game model's win probability into a
full distribution of simulated final margins, which is what the funnel
visual on the site shows.

Method (principled, not decorative):
  1. The trained model gives P(home win) for a matchup.
  2. NFL final margins are well-approximated as normal around an expected
     margin, with a game-to-game standard deviation of ~13.5 points (a
     long-established empirical constant for NFL scoring).
  3. We invert the win probability to an expected margin: if P(win)=p, then
     expected_margin = sigma * inverse_normal_cdf(p), because
     P(margin > 0) = P(win).
  4. Simulate N games by drawing margins ~ Normal(expected_margin, sigma),
     rounding to whole points and nudging off exact ties (NFL games rarely
     tie). The spread of simulated margins is the outcome distribution.

This gives an honest uncertainty picture: a 60% favorite still loses a real
share of simulations, which is the whole point of showing a distribution
instead of a single number.
"""

import numpy as np
from scipy.stats import norm

NFL_MARGIN_SIGMA = 13.5


def simulate_margins(home_win_prob: float, n: int = 20000, seed: int = 0) -> np.ndarray:
    """Return N simulated final margins (home minus away points)."""
    rng = np.random.default_rng(seed)
    p = float(np.clip(home_win_prob, 0.001, 0.999))
    expected_margin = NFL_MARGIN_SIGMA * norm.ppf(p)
    margins = rng.normal(expected_margin, NFL_MARGIN_SIGMA, size=n)
    margins = np.round(margins)
    # NFL games almost never tie; nudge exact zeros toward the favorite.
    ties = margins == 0
    margins[ties] = 1.0 if expected_margin >= 0 else -1.0
    return margins


def summarize(margins: np.ndarray) -> dict:
    home_win_rate = float((margins > 0).mean())
    return {
        "expected_margin": round(float(margins.mean()), 1),
        "home_win_rate": round(home_win_rate, 3),
        "away_win_rate": round(1 - home_win_rate, 3),
        "median_margin": int(np.median(margins)),
        "p10_margin": int(np.percentile(margins, 10)),
        "p90_margin": int(np.percentile(margins, 90)),
        "n": int(len(margins)),
    }


def histogram(margins: np.ndarray, lo: int = -35, hi: int = 35, bin_width: int = 2) -> dict:
    """Binned counts for the frontend distribution chart."""
    edges = np.arange(lo, hi + bin_width, bin_width)
    counts, _ = np.histogram(np.clip(margins, lo, hi - 1), bins=edges)
    centers = [(int(edges[i]) + int(edges[i + 1])) / 2 for i in range(len(edges) - 1)]
    return {"centers": centers, "counts": counts.tolist(), "bin_width": bin_width}
