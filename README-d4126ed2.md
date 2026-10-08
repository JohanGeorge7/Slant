# Gridiron — NFL prediction models

Three models over ten seasons of [nflverse](https://github.com/nflverse/nflreadpy)
play-by-play, with an interactive board as the front end. Every number on the
site traces to real data and a model validated on games it never trained on.

## Quick start

```bash
uv sync
uv run python api/app.py     # trained models ship with the repo
# open http://localhost:5000
```

Full rebuild from raw data:

```bash
uv run python data/ingest.py                  # cache ~10 seasons
uv run python models/game_predictor.py        # train + walk-forward validate
uv run python models/playcall_predictor.py
uv run python models/fourth_down_model.py
uv run python models/export_frontend_data.py  # real data -> frontend/data/*.json
uv run python api/app.py
```

## 1. Game predictor — 63.8% straight up

Logistic regression on home-minus-away trailing-form differentials (EPA,
success rate, explosive rate, pass/rush EPA, CPOE, turnovers, third downs,
pace, rest) plus a from-scratch Elo rating.

Validated **walk-forward**: for each of the last four seasons, train only on
earlier seasons and predict that season cold.

| | Accuracy | Log loss |
|---|---|---|
| Home-team baseline | 54.0% | — |
| **This model** | **63.8%** | 0.656 |
| Vegas closing line | 67.0% | 0.611 |

**Against the spread: 49.8%**, below the 52.4% break-even at -110. Stated
plainly on the site. No public model reliably beats a closing line; winner
prediction is the real output here.

## 2. Play-call predictor — 72.9%

XGBoost run/pass classifier on pre-snap state only. Post-snap charting tags
(play-action, RPO, pass-rush counts) are excluded because they're recorded
after the play and leak the label — they once inflated an earlier build to a
fake 96.7%. Also produces a per-offense predictability leaderboard from the
model's average confidence.

## 3. Fourth down — component expected points

**This was rebuilt after the first version proved unsound.** The original
trained one neural net per decision to regress EPA. Measured honestly, the
field-goal net's error matched a constant predictor (MAE 1.0024 vs 1.0021),
and the correlation between field position and EPA on field goals is 0.043 —
essentially zero, because EPA is already situation-adjusted. It also
extrapolated absurdly: at the opponent's 99 it recommended a *116-yard* field
goal.

The rebuild decomposes each option into measurable parts:

```
EP(go)   = P(convert) x EP(1st down at new spot)
         + (1 - P(convert)) x -EP(opponent ball here)
EP(FG)   = P(make | distance) x [3 + -EP(opponent after kickoff)]
         + (1 - P(make)) x -EP(opponent ball at the kick spot)
EP(punt) = -EP(opponent ball after a net punt)
```

Every term is measured from ten seasons: the expected-points curve by field
position, conversion rates by distance (67.1% on 4th-and-1, n=2565), field
goal make rates by distance (73.8% at 45 yards, n=302), and net punt distance
by field position. Physically impossible options are ruled out first — field
goals beyond the 49 (a 66-yard try, the record) and punts inside the
opponent's 28. The model also reports how decisive a call is and flags thinly
sampled situations rather than projecting false confidence.

## The interface

- **The board** is the centerpiece: every Week 1 game, sortable by kickoff,
  confidence, or upset picks. Each card opens a detail panel with the team
  logos and colors, a 20,000-run Monte Carlo distribution of final margins,
  the per-feature contributions that moved the pick, and market context.
- **"How this is calculated"** disclosures sit throughout, collapsed by
  default — the arithmetic is there for anyone who wants it without cluttering
  the page.
- Team colors and logos come from nflverse. Two real data problems are handled
  in code: some teams share a primary color (NE and SEA are both `#002244`),
  and some primaries are near-black and would be invisible on a dark page.

## Architecture

```
data/ingest.py      season-keyed parquet cache of nflverse data
data/features.py    shared feature engineering
data/elo.py         from-scratch Elo ratings
models/             one script per model, each self-validating
models/simulate.py  Monte Carlo margin simulation
api/app.py          Flask: frontend + live model endpoints
frontend/           single-page interface, real data only
```

## Bugs found and fixed (kept as a record)

- **Stale cache** — filenames were season-blind and silently served wrong data.
- **Play-call leakage** — post-snap features inflated accuracy to 96.7%.
- **Spread sign error** — `home_covers` had the sign flipped (home underdogs
  carry a negative `spread_line`), producing a fake 70% ATS.
- **Fourth-down extrapolation** — nets recommended a 116-yard field goal;
  replaced with the component model above.

Data: nflverse, CC-BY 4.0. FTN charting (play-call features) is 2022+;
everything else runs on 2015-2025.


## The site

Dark theme, board-first. Pages: `/`, `/privacy`, `/terms`, custom
404, plus `robots.txt` and a generated `sitemap.xml`.

**Interactive pieces**
- **The board** — every game, sortable, each card opens a panel with logos,
  a 20,000-run simulation, the per-feature contributions, and market context.
  Arrow keys move between games.
- **Play-calling heatmap** — every down and distance coloured by likely call.
  Click a square for the model split beside what teams actually did.
- **Fourth-down field map** — the recommended decision at every spot, with the
  measured conversion and field-goal curves next to it.
- **Evidence charts** — calibration, accuracy by season vs Vegas, feature
  weights. All from held-out seasons.
- **"How this is calculated"** disclosures throughout, collapsed by default.

**Production hygiene**
- No secrets in the frontend; config comes from environment variables
  (`GRIDIRON_ENV`, `GRIDIRON_SITE_URL`, `GRIDIRON_ANALYTICS_ID`)
- HTTPS redirect + HSTS when `GRIDIRON_ENV=prod`
- CSP, `X-Frame-Options`, `nosniff`, referrer and permissions policies
- Cookie consent banner; analytics off unless allowed, first-party only
- Meta titles, descriptions, Open Graph and Twitter cards, social preview image
- Favicon set and apple-touch-icon
- Alt text on every image, skip link, focus states, ARIA labels, reduced-motion
- Newsletter with client and server validation, honeypot field, timing check,
  and per-IP rate limiting
- Responsive down to 390px; contrast checked against the light background

## Play-call model, rebuilt

The first version trained only on run/pass plays, so it had never seen a punt.
Asked about 4th-and-20 it answered "pass" — in ten seasons teams punt 74% of
the time there and pass 5%. It also relied on FTN charting features the
interface had to hardcode, and those hardcoded values dominated: 2nd-and-3
came back "pass" when teams actually run it 57% of the time.

The rebuild predicts all four real options (run, pass, punt, field goal) from
situation alone. It is calibrated to three decimals against the data:

| Situation | Model | League actual |
|---|---|---|
| 4th & 20 | punt | punt 74% |
| 2nd & 3 | run 64% | run 57% (n=6,253) |
| 3rd & 1 | run 74% | run 74% (n=8,198) |

Every calculator now shows the league's actual rate beside the model's answer.


## Hero animation

Two subtle layers behind the headline, both real rather than decorative stock:

- **Logo marquee** — three rows of the 32 real NFL team logos drifting in
  alternating directions at different speeds (84s, 96s, 110s per cycle). Each
  row is duplicated so the `-50%` translate loops seamlessly.
- **Starfield** — drifting, slowly twinkling points, with an occasional
  streak across the top. Density scales to viewport area.

Both respect `prefers-reduced-motion`: the starfield paints one static frame
and the marquee animation is suppressed. A layered veil keeps contrast high
behind the headline while letting the logos read at the edges.
