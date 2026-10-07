#!/bin/bash
# ============================================================
#  AI-DRCS  —  Container Startup
#  Starts both the Rasa action server and the main Rasa server.
# ============================================================

set -e

echo "========================================"
echo "  AI-DRCS Rasa Server Starting..."
echo "========================================"

# Start the action server in the background on port 5055
echo "[1/2] Starting Rasa Action Server on port 5055..."
rasa run actions --port 5055 &
ACTION_PID=$!

# Give the action server 5 seconds to come up
sleep 5
echo "  ✓ Action server started (PID: $ACTION_PID)"

# Start the main Rasa server on port 5005
echo "[2/2] Starting Rasa NLU/Core Server on port 5005..."
rasa run \
  --enable-api \
  --cors "*" \
  --port 5005 \
  --endpoints endpoints.yml \
  --credentials credentials.yml \
  --model models/drcs_model.tar.gz \
  --log-file rasa.log \
  --debug

# If Rasa exits, also kill the action server
kill $ACTION_PID 2>/dev/null || true
