#!/usr/bin/env bash
# ONE-CLICK live prompt-sequenced demo: brings up the GR00T inference server (if not already
# running), then plays a demo flow in the Isaac viewer — a single session that loops one prompt
# after another (squat -> walk -> macarena ...) with NO GUI/server restart between motions.
#
# The two halves run in DIFFERENT envs (GR00T server = Isaac-GR00T/.venv; viewer = isaaclab
# conda), which is why a normal one-liner can't do it — this script orchestrates both.
#
# Usage:
#   bash scripts/gear_sonic_live_demo.sh @flow1        # discrete actions, looping
#   bash scripts/gear_sonic_live_demo.sh @flow2        # action -> walk -> action -> walk ... loop
#   bash scripts/gear_sonic_live_demo.sh list          # list named flows
#   bash scripts/gear_sonic_live_demo.sh squat,lunge,macarena   # ad-hoc sequence
# The GR00T server is left running for the next flow; free the GPU with: scripts/gear_sonic_stop.sh
# Camera: C = snap close to robot, F = focus toggle, R = reset.  (Click the viewport first.)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${GR00T_PORT:-5555}"
GR00T_DIR="$REPO_ROOT/dependencies/Isaac-GR00T"
GR00T_CKPT="${GR00T_CKPT:-$REPO_ROOT/outputs/gr00t_sonic_8k/checkpoint-8000}"
SERVER_LOG="${SERVER_LOG:-/tmp/groot_server.log}"
SEQ="${1:-@flow2}"

# `list` needs no server — forward straight to the sequence launcher.
[[ "$SEQ" == "list" ]] && exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" list

# 1) Ensure the GR00T inference server is up on $PORT (idempotent — reuse if already running).
if pgrep -f "[r]un_gr00t_server.*--port $PORT" >/dev/null 2>&1; then
  echo "[live-demo] GR00T server already running on :$PORT — reusing it."
else
  if [[ ! -d "$GR00T_CKPT" ]]; then
    echo "[live-demo] checkpoint not found: $GR00T_CKPT"
    echo "            train it (notebook 4.3) or set GR00T_CKPT=/path/to/checkpoint-XXXX"; exit 1
  fi
  echo "[live-demo] starting GR00T server (model=$GR00T_CKPT, port=$PORT) ..."
  ( cd "$GR00T_DIR" && COMPILE_ACTION_HEAD_DISABLE=1 nohup .venv/bin/python \
      -m gr00t.eval.run_gr00t_server --model_path "$GR00T_CKPT" \
      --embodiment_tag unitree_g1_sonic --port "$PORT" > "$SERVER_LOG" 2>&1 & )
  echo "[live-demo] loading the VLM (~1 min) — watching $SERVER_LOG ..."
  ready=0
  for _ in $(seq 1 48); do
    if grep -q "Loading checkpoint shards: 100%" "$SERVER_LOG" 2>/dev/null; then ready=1; break; fi
    if ! pgrep -f "[r]un_gr00t_server.*--port $PORT" >/dev/null 2>&1; then
      echo "[live-demo] server exited during load — last log:"; tail -15 "$SERVER_LOG"; exit 1
    fi
    sleep 5
  done
  [[ "$ready" == "1" ]] && echo "[live-demo] server ready." \
                        || echo "[live-demo] server not confirmed ready; the viewer will still wait+ping."
fi

# 2) Play the flow (opens the Isaac viewer; the injector pings the server on first step).
echo "[live-demo] launching viewer for '$SEQ' ..."
exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" "$SEQ"
