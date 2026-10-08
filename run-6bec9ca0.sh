#!/bin/bash
# One-command setup and launch for the Gridiron NFL analytics app.
set -e
echo "Installing dependencies..."
uv sync
echo "Building data cache (first run downloads ~10 seasons, a few minutes)..."
uv run python data/ingest.py
echo "Training models..."
uv run python models/game_predictor.py
uv run python models/playcall_predictor.py
uv run python models/fourth_down_model.py
echo "Exporting real data for the frontend..."
uv run python models/export_frontend_data.py
echo ""
echo "Starting server at http://localhost:5000"
uv run python api/app.py
