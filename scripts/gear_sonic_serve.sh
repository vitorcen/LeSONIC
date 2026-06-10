#!/usr/bin/env bash
# Start the GR00T N1.7 ZMQ policy server (idempotent). Used by LAFAN.ipynb / live skill demos.
# The server holds the VLA; gear_sonic_live.sh (Isaac side) connects to it on $PORT.
#
#   bash scripts/gear_sonic_serve.sh                       # default = skills checkpoint-6000
#   GR00T_CKPT=outputs/gr00t_sonic_8k/checkpoint-8000 bash scripts/gear_sonic_serve.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_DIR="$REPO_ROOT/dependencies/Isaac-GR00T"
CKPT="${GR00T_CKPT:-$REPO_ROOT/outputs/gr00t_sonic_skills/checkpoint-6000}"
PORT="${GR00T_PORT:-5555}"
LOG="${SERVER_LOG:-/tmp/gr00t_server.log}"

if pgrep -f "[r]un_gr00t_server.*--port $PORT" >/dev/null 2>&1; then
  echo "[serve] GR00T server already up on port $PORT (reusing). To switch checkpoint: bash scripts/gear_sonic_stop.sh first."
  exit 0
fi
[[ -d "$CKPT" ]] || { echo "[serve] checkpoint missing: $CKPT"; exit 1; }
echo "[serve] starting GR00T server  model=$CKPT  port=$PORT  (~1 min to load shards)"
( cd "$GR00T_DIR" && COMPILE_ACTION_HEAD_DISABLE=1 .venv/bin/python \
    -m gr00t.eval.run_gr00t_server --model_path "$CKPT" \
    --embodiment_tag unitree_g1_sonic --port "$PORT" > "$LOG" 2>&1 & )
for _ in $(seq 1 150); do
  grep -q "Loading checkpoint shards: 100%" "$LOG" 2>/dev/null && { echo "[serve] ✅ ready on port $PORT"; exit 0; }
  pgrep -f "[r]un_gr00t_server.*--port $PORT" >/dev/null 2>&1 || { echo "[serve] ❌ server died:"; tail -15 "$LOG"; exit 1; }
  sleep 2
done
echo "[serve] ❌ timeout waiting for ready — check $LOG"; exit 1
