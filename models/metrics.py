"""
Single source of truth for the headline numbers shown on the site
(accuracy, vegas accuracy, ATS, play-call/fourth-down summaries).

Previously these were hand-typed constants inside export_frontend_data.py,
completely disconnected from the walk-forward evaluation export_evidence.py
actually runs — retrain the model and the two files could quietly disagree,
exactly the kind of silent bug the project's own README calls out elsewhere.
Both export scripts now call compute_metrics() instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np

from data.ingest import PBP_SEASONS, get_pbp
from models.game_predictor import build_game_table, get_feature_cols, walk_forward, select_best

MODEL_DIR = Path(__file__).parent / "artifacts"


def compute_metrics(games=None, feature_cols=None) -> dict:
    if games is None:
        games = build_game_table()
    if feature_cols is None:
        feature_cols = get_feature_cols(games)

    all_seasons = sorted(games["season"].unique().to_list())
    test_seasons = all_seasons[-4:]
    results, blend_weight = walk_forward(games, feature_cols, test_seasons)
    best_name = select_best(results)

    game_metrics = {
        "accuracy": round(float(np.mean(results[best_name]["acc"])), 4),
        "log_loss": round(float(np.mean(results[best_name]["ll"])), 4),
        "vegas_accuracy": round(float(np.mean(results["vegas"]["acc"])), 4),
        "blend_accuracy": round(float(np.mean(results["blend"]["acc"])), 4),
        "baseline_accuracy": round(float(np.mean(results["home_baseline"]["acc"])), 4),
        "ats": round(float(np.mean(results[best_name]["ats"])), 4),
        "blend_weight": round(blend_weight, 3),
        "validation": f"walk-forward {test_seasons[0]}-{test_seasons[-1]}",
        "model": best_name,
        "n_features": len(feature_cols),
    }

    # Label with the seasons actually present in the data, not the
    # requested range — PBP_SEASONS optimistically includes the in-progress
    # season, which get_pbp() skips until real data for it exists (see
    # data/ingest.py), so the requested upper bound can be a season ahead
    # of what was actually trained/validated on.
    data_first, data_last = all_seasons[0], all_seasons[-1]

    playcall_path = MODEL_DIR / "playcall_predictor.joblib"
    if playcall_path.exists():
        pc = joblib.load(playcall_path)
        playcall_metrics = {
            "accuracy": pc["accuracy_4class"],
            "accuracy_runpass": pc["accuracy_runpass"],
            "validation": f"20% holdout, {data_first}-{data_last}",
        }
    else:
        playcall_metrics = {}

    fourth_down_metrics = {
        "approach": "component expected-points model",
        "components": ["EP curve by field position", "conversion rate by distance",
                       "FG make rate by distance", "net punt distance"],
    }

    n_plays = get_pbp(PBP_SEASONS).height
    return {
        "game_predictor": game_metrics,
        "playcall_predictor": playcall_metrics,
        "fourth_down": fourth_down_metrics,
        "data": {"seasons": f"{data_first}-{data_last}", "plays": n_plays, "source": "nflverse"},
    }
