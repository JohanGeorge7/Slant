"""
Market (Vegas) win probability, fitted from real results instead of assumed.

The old code converted a closing spread straight to a win probability with a
hardcoded slope (`1 / (1 + exp(-0.14 * spread))`), a reasonable rule of thumb
but never actually checked against this dataset. Here the slope (and
intercept) are fit by logistic regression on the closing spread predicting
the real historical outcome, so "what does the market imply" is measured the
same honest way as everything else on the site.

Also provides a shrinkage blend of the model's probability with the market's:
the README is upfront that no public model reliably beats the closing line,
so leaning partway toward the market is a legitimate way to improve
calibration rather than a concession. The blend weight is picked by
walk-forward log loss, same discipline as model selection in
game_predictor.py.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def fit_market_model(games: pl.DataFrame) -> LogisticRegression:
    """Logistic regression of spread_line -> home_win, fit on ALL available
    seasons (the market's spread-to-probability relationship is stable
    across years, unlike team-specific form, so there's no walk-forward
    concern here — we're not predicting outcomes, just calibrating what a
    given spread has historically implied).
    """
    X = games.select("spread_line").to_numpy()
    y = games["home_win"].to_numpy().astype(int)
    m = LogisticRegression()
    m.fit(X, y)
    return m


def market_prob(model: LogisticRegression, spread_line) -> np.ndarray:
    x = np.asarray(spread_line, dtype=float).reshape(-1, 1)
    return model.predict_proba(x)[:, 1]


def blend(model_p: np.ndarray, market_p: np.ndarray, weight: float) -> np.ndarray:
    """Blend in log-odds space (not linear probability space), which is the
    correct way to average two calibrated probabilities — it keeps the
    result a proper probability and treats both sources symmetrically.
    `weight` is the share given to the market; 0 = pure model, 1 = pure market.
    """
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    return sigmoid((1 - weight) * logit(model_p) + weight * logit(market_p))


def best_blend_weight(model_p: np.ndarray, market_p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Grid-search the blend weight that minimizes log loss on held-out
    walk-forward predictions, so the blend is tuned the same honest way as
    model selection elsewhere — not just eyeballed at 50/50.
    """
    best_w, best_ll = 0.0, log_loss(y, np.clip(model_p, 1e-6, 1 - 1e-6), labels=[0, 1])
    for w in np.arange(0.05, 1.0, 0.05):
        p = blend(model_p, market_p, w)
        ll = log_loss(y, p, labels=[0, 1])
        if ll < best_ll:
            best_w, best_ll = float(w), ll
    return best_w, best_ll
