#!/usr/bin/env bash
# run_local.sh — Start the Email Triage server and run the baseline inference script.
# Usage: ./run_local.sh [task]
#   task: all | priority-classification | category-routing | full-triage-pipeline (default: all)

set -euo pipefail

TASK="${1:-all}"
PORT=7860
SERVER_PID=""

cleanup() {
  if [ -n "$SERVER_PID" ]; then
    echo "[run_local] Stopping server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[run_local] Installing dependencies..."
pip install -r requirements.txt -q

echo "[run_local] Starting environment server on port $PORT..."
uvicorn server.app:app --host 0.0.0.0 --port "$PORT" --workers 1 &
SERVER_PID=$!

# Wait for server to be ready
echo -n "[run_local] Waiting for server..."
for i in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo " ready!"
    break
  fi
  sleep 1
  echo -n "."
done

# Verify server is up
if ! curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
  echo ""
  echo "[run_local] ERROR: Server failed to start."
  exit 1
fi

echo "[run_local] Running inference (task=$TASK)..."
EMAIL_TRIAGE_TASK="$TASK" \
ENV_BASE_URL="http://localhost:$PORT" \
python inference.py

echo "[run_local] Done."
