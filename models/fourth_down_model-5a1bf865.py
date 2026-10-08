"""
Fourth down decision model — expected points, built from components.

WHY NOT A NEURAL NET ON EPA:
An earlier version trained one network per decision to regress EPA directly.
Measured honestly, the field-goal net's error was identical to just
predicting the mean (MAE 1.0024 vs 1.0021), and the correlation between
field position and EPA on field goal attempts is 0.043 — essentially zero.
The reason is that EPA is *already* situation-adjusted (it's the change in
expected points), so regressing it on the same situation leaves little
signal, and coach selection bias contaminates what's left.

WHAT THIS DOES INSTEAD:
The approach real NFL analytics uses — decompose each decision into
measurable pieces and combine them:

  EP(go)   = P(convert) x EP(1st down at the new spot)
           + (1 - P(convert)) x -EP(opponent takes over here)

  EP(FG)   = P(make) x [3 + -EP(opponent after kickoff)]
           + (1 - P(make)) x -EP(opponent takes over at the kick spot)

  EP(punt) = -EP(opponent takes over after a net punt)

Every term is estimated empirically from ten seasons: the expected-points
curve by field position, fourth down conversion rates by distance, field
goal make rates by attempt distance, and net punt distance. That makes the
recommendation both more accurate and fully explainable — the site can show
the user exactly which numbers produced the answer.

Run: uv run python models/fourth_down_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import polars as pl

from data.ingest import get_pbp

MODEL_DIR = Path(__file__).parent / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)

# Feasibility limits from what actually occurs in 2015-2024 data.
# A field goal from yardline_100 = Y is a (Y + 17) yard attempt.
MAX_FG_YARDLINE = 49      # a 66-yard try, the NFL record
MIN_PUNT_YARDLINE = 28    # nobody punts from inside the opponent's 28
TOUCHBACK_YARDLINE = 75   # receiving team's own 25 => their yardline_100
KICKOFF_YARDLINE = 75     # after a made FG, opponent starts ~own 25


def fg_attempt_distance(yardline_100: float) -> int:
    return int(round(yardline_100)) + 17


def decision_is_feasible(decision: str, yardline_100: float) -> bool:
    if decision == "field_goal":
        return yardline_100 <= MAX_FG_YARDLINE
    if decision == "punt":
        return yardline_100 >= MIN_PUNT_YARDLINE
    return True


# ---------------------------------------------------------------------------
# Component estimation
# ---------------------------------------------------------------------------

def build_ep_curve(pbp: pl.DataFrame) -> np.ndarray:
    """Expected points for a 1st & 10 at each yardline_100 (1..99), smoothed.
    This is the backbone: every decision resolves into 'who has the ball,
    where', and this curve prices that.
    """
    first = pbp.filter(
        (pl.col("down") == 1) & (pl.col("ydstogo") == 10) & pl.col("ep").is_not_null()
    )
    agg = (first.group_by("yardline_100")
                .agg([pl.col("ep").mean().alias("ep"), pl.len().alias("n")])
                .sort("yardline_100"))
    ys = agg["yardline_100"].to_numpy()
    eps = agg["ep"].to_numpy()
    grid = np.arange(0, 101)
    curve = np.interp(grid, ys, eps)
    # light smoothing to remove sampling wobble
    k = np.array([1, 2, 3, 2, 1], dtype=float); k /= k.sum()
    curve = np.convolve(np.pad(curve, 2, mode="edge"), k, mode="valid")
    return curve


def build_conversion_curve(pbp: pl.DataFrame) -> dict:
    """P(convert a 4th down) by yards to go, empirically."""
    go = pbp.filter(
        (pl.col("down") == 4) & pl.col("play_type").is_in(["run", "pass"])
    ).with_columns(pl.col("fourth_down_converted").fill_null(0).alias("conv"))
    agg = (go.group_by("ydstogo")
             .agg([pl.col("conv").mean().alias("rate"), pl.len().alias("n")])
             .filter(pl.col("n") >= 25).sort("ydstogo"))
    d = {int(r["ydstogo"]): (float(r["rate"]), int(r["n"])) for r in agg.iter_rows(named=True)}
    return d


def build_fg_curve(pbp: pl.DataFrame) -> dict:
    """P(make a field goal) by attempt distance, empirically."""
    fg = pbp.filter(
        (pl.col("play_type") == "field_goal") & pl.col("yardline_100").is_not_null()
    ).with_columns(
        (pl.col("yardline_100") + 17).cast(pl.Int32).alias("dist"),
        (pl.col("field_goal_result") == "made").cast(pl.Int8).alias("made"),
    )
    agg = (fg.group_by("dist")
             .agg([pl.col("made").mean().alias("rate"), pl.len().alias("n")])
             .filter(pl.col("n") >= 15).sort("dist"))
    return {int(r["dist"]): (float(r["rate"]), int(r["n"])) for r in agg.iter_rows(named=True)}


def build_punt_net(pbp: pl.DataFrame) -> dict:
    """Average NET punt distance by field position bucket (gross kick minus
    return), which decides where the opponent actually starts.
    """
    p = pbp.filter(
        (pl.col("play_type") == "punt") & pl.col("kick_distance").is_not_null()
    ).with_columns(
        (pl.col("kick_distance") - pl.col("return_yards").fill_null(0)).alias("net"),
        (pl.col("yardline_100") / 10).floor().cast(pl.Int32).alias("bucket"),
    )
    agg = (p.group_by("bucket")
             .agg([pl.col("net").mean().alias("net"), pl.len().alias("n")])
             .filter(pl.col("n") >= 50).sort("bucket"))
    return {int(r["bucket"]): (float(r["net"]), int(r["n"])) for r in agg.iter_rows(named=True)}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class FourthDownModel:
    def __init__(self, ep_curve, conv, fg, punt_net):
        self.ep_curve = ep_curve
        self.conv = conv
        self.fg = fg
        self.punt_net = punt_net

    def ep_at(self, yardline_100: float) -> float:
        y = int(np.clip(round(yardline_100), 0, 100))
        return float(self.ep_curve[y])

    def opponent_ep(self, our_yardline_100: float) -> float:
        """Value TO US of the opponent taking over at a spot we describe in
        our own yardline_100 terms. Their distance to our end zone is
        100 - our_yardline_100, and their expected points count against us.
        """
        opp_yardline = 100 - np.clip(our_yardline_100, 0, 100)
        return -self.ep_at(opp_yardline)

    def p_convert(self, ydstogo: int) -> tuple:
        ydstogo = int(np.clip(ydstogo, 1, 30))
        if ydstogo in self.conv:
            return self.conv[ydstogo]
        keys = sorted(self.conv)
        if not keys:
            return (0.5, 0)
        nearest = min(keys, key=lambda k: abs(k - ydstogo))
        # long-distance extrapolation decays toward a floor
        rate = self.conv[nearest][0] * (0.9 ** max(0, ydstogo - nearest))
        return (float(np.clip(rate, 0.05, 0.95)), 0)

    def p_fg_make(self, dist: int) -> tuple:
        dist = int(dist)
        if dist in self.fg:
            return self.fg[dist]
        keys = sorted(self.fg)
        if not keys:
            return (0.5, 0)
        if dist < keys[0]:
            return (self.fg[keys[0]][0], 0)
        if dist > keys[-1]:
            # beyond observed range, decay hard
            return (max(0.02, self.fg[keys[-1]][0] * (0.85 ** (dist - keys[-1]))), 0)
        lo = max(k for k in keys if k <= dist)
        hi = min(k for k in keys if k >= dist)
        if lo == hi:
            return self.fg[lo]
        w = (dist - lo) / (hi - lo)
        return (self.fg[lo][0] * (1 - w) + self.fg[hi][0] * w, 0)

    def net_punt(self, yardline_100: float) -> float:
        b = int(np.clip(yardline_100 // 10, 0, 9))
        if b in self.punt_net:
            return self.punt_net[b][0]
        vals = [v[0] for v in self.punt_net.values()]
        return float(np.mean(vals)) if vals else 40.0

    def evaluate(self, ydstogo: int, yardline_100: float,
                 score_differential: float = 0, game_seconds_remaining: float = 1800) -> dict:
        """Return expected points for each feasible decision, plus the
        component numbers that produced them (for user-facing explanation).
        """
        results, explain = {}, {}

        # --- GO FOR IT ---
        p_conv, n_conv = self.p_convert(ydstogo)
        new_spot = max(1, yardline_100 - ydstogo)      # where we'd be after converting
        ep_success = self.ep_at(new_spot)
        ep_fail = self.opponent_ep(yardline_100)        # they take over right here
        ep_go = p_conv * ep_success + (1 - p_conv) * ep_fail
        results["go"] = ep_go
        explain["go"] = {
            "convert_prob": round(p_conv, 3), "sample_size": n_conv,
            "ep_if_converted": round(ep_success, 2),
            "ep_if_failed": round(ep_fail, 2),
            "formula": f"{p_conv:.2f} x {ep_success:+.2f} + {1-p_conv:.2f} x {ep_fail:+.2f}",
        }

        # --- FIELD GOAL ---
        if decision_is_feasible("field_goal", yardline_100):
            dist = fg_attempt_distance(yardline_100)
            p_make, n_fg = self.p_fg_make(dist)
            ep_made = 3.0 + self.opponent_ep(100 - KICKOFF_YARDLINE)
            miss_spot = min(99, yardline_100 + 7)       # they take over at the kick spot
            ep_miss = self.opponent_ep(miss_spot)
            ep_fg = p_make * ep_made + (1 - p_make) * ep_miss
            results["field_goal"] = ep_fg
            explain["field_goal"] = {
                "attempt_distance": dist, "make_prob": round(p_make, 3), "sample_size": n_fg,
                "ep_if_made": round(ep_made, 2), "ep_if_missed": round(ep_miss, 2),
                "formula": f"{p_make:.2f} x {ep_made:+.2f} + {1-p_make:.2f} x {ep_miss:+.2f}",
            }

        # --- PUNT ---
        if decision_is_feasible("punt", yardline_100):
            net = self.net_punt(yardline_100)
            landing = yardline_100 - net
            if landing <= 0:
                opp_start_our_terms = 100 - TOUCHBACK_YARDLINE   # touchback
                touchback = True
            else:
                opp_start_our_terms = landing
                touchback = False
            ep_punt = self.opponent_ep(opp_start_our_terms)
            results["punt"] = ep_punt
            explain["punt"] = {
                "net_distance": round(net, 1), "touchback": touchback,
                "opponent_starts_at_own": int(100 - (100 - opp_start_our_terms)) if not touchback else 25,
                "formula": f"opponent ball => {ep_punt:+.2f} to us",
            }

        best = max(results, key=results.get)

        # How decisive is the call? Two options within a fifth of a point is
        # a coin flip dressed up as a recommendation, and saying so is more
        # useful than projecting false confidence. Field position also
        # matters for trust: short-yardage attempts from deep in your own
        # territory are rare (39 in ten seasons), so the conversion rate
        # there is thinly sampled and we flag it rather than hide it.
        ordered = sorted(results.values(), reverse=True)
        margin = round(ordered[0] - ordered[1], 3) if len(ordered) > 1 else None
        if margin is None:
            strength = "only option"
        elif margin >= 0.75:
            strength = "clear"
        elif margin >= 0.25:
            strength = "moderate"
        else:
            strength = "close call"

        caveats = []
        if yardline_100 >= 80 and ydstogo <= 3:
            caveats.append("Short-yardage tries from inside your own 20 are rare "
                           "(39 in ten seasons), so this conversion rate is thinly sampled.")
        if margin is not None and margin < 0.25:
            caveats.append("The top two options are within a quarter of a point — "
                           "treat this as a toss-up, not a verdict.")

        return {
            "expected_points": {k: round(v, 3) for k, v in results.items()},
            "recommended_decision": best,
            "margin": margin,
            "strength": strength,
            "caveats": caveats,
            "explain": explain,
            "infeasible": [d for d in ("go", "punt", "field_goal") if d not in results],
            "fg_distance_if_attempted": fg_attempt_distance(yardline_100),
        }


def main():
    needed = ["down", "ydstogo", "yardline_100", "ep", "play_type",
              "field_goal_result", "fourth_down_converted", "kick_distance", "return_yards"]
    pbp = get_pbp().select(needed)

    ep_curve = build_ep_curve(pbp)
    conv = build_conversion_curve(pbp)
    fg = build_fg_curve(pbp)
    punt_net = build_punt_net(pbp)

    print(f"EP curve built over {len(ep_curve)} yardlines")
    print(f"Conversion rates for {len(conv)} distances, FG rates for {len(fg)} distances, "
          f"punt nets for {len(punt_net)} buckets\n")

    print("Sample conversion rates:")
    for d in [1, 2, 3, 5, 7, 10]:
        if d in conv:
            print(f"  4th & {d:2d}: {conv[d][0]:.1%} (n={conv[d][1]})")
    print("\nSample FG make rates:")
    for d in [25, 35, 45, 50, 55, 60]:
        if d in fg:
            print(f"  {d}yd: {fg[d][0]:.1%} (n={fg[d][1]})")

    model = FourthDownModel(ep_curve, conv, fg, punt_net)

    joblib.dump({
        "ep_curve": ep_curve, "conversion": conv, "fg": fg, "punt_net": punt_net,
        "max_fg_yardline": MAX_FG_YARDLINE, "min_punt_yardline": MIN_PUNT_YARDLINE,
    }, MODEL_DIR / "fourth_down_model.joblib")

    print("\n--- Sanity checks (previously broken cases) ---")
    for ytg, yl, label in [
        (2, 99, "own 1, 4th & 2"),
        (5, 80, "own 20, 4th & 5"),
        (2, 45, "midfield-ish, 4th & 2"),
        (1, 35, "opp 35, 4th & 1"),
        (10, 30, "opp 30, 4th & 10"),
        (3, 5, "opp 5, 4th & 3"),
        (15, 60, "own 40, 4th & 15"),
    ]:
        r = model.evaluate(ytg, yl)
        eps = ", ".join(f"{k}={v:+.2f}" for k, v in r["expected_points"].items())
        excl = f" [excluded: {', '.join(r['infeasible'])}]" if r["infeasible"] else ""
        print(f"  {label:24s} -> {r['recommended_decision']:11s} ({eps}){excl}")

    print(f"\nSaved model to {MODEL_DIR / 'fourth_down_model.joblib'}")


if __name__ == "__main__":
    main()
