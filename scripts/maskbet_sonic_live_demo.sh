#!/usr/bin/env bash
# ONE-CLICK live MaskBeT-SONIC demo: bring up the MaskBeT token server (if not already
# running), then play a demo flow in the Isaac viewer. The Isaac side is the SAME machinery
# as the GR00T / StarVLA live demos (gear_sonic_sequence.sh + vla_live_injector) — the server
# just speaks the same ZMQ wire on its own port.
#
#   bash scripts/maskbet_sonic_live_demo.sh @flow3       # fight/run/dance loop, MaskBeT 25M
#   bash scripts/maskbet_sonic_live_demo.sh combat,circle
#   SONIC_MASKBET_DECODE=expected bash scripts/maskbet_sonic_live_demo.sh @flow3
# Server is left running for the next flow; stop with: pkill -f serve_maskbet_sonic
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${MASKBET_PORT:-5557}"
# base env zmq is broken (PYZMQ_DRAFT_API); the starvla_eval env has zmq+torch and loads the
# pure-torch MaskBeT state_dict fine.
ENV_BIN="${ENV_BIN:-$HOME/miniconda3/envs/starvla_eval_qwen35/bin}"
CKPT="${MASKBET_CKPT:-$REPO_ROOT/MaskBeT/outputs/flow3/ckpt_006000.pt}"
SERVER_LOG="${SERVER_LOG:-/tmp/maskbet_sonic_server.log}"
SEQ="${1:-@flow3}"

[[ "$SEQ" == "list" ]] && exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" list

# Decode: argmax is constructively on-grid (closed-loop default, snap is a no-op). expected
# is lower open-loop MSE (0.0090 vs 0.0193) but off-grid → the server snaps it back.
export SONIC_MASKBET_DECODE="${SONIC_MASKBET_DECODE:-argmax}"
export SONIC_MASKBET_TEMP="${SONIC_MASKBET_TEMP:-1.0}"

# 1) ensure the MaskBeT server is up on $PORT (idempotent)
if ps -eo cmd | grep -q "[s]erve_maskbet_sonic.py.*--port $PORT"; then
  echo "[live-demo] MaskBeT server already running on :$PORT — reusing it."
else
  [[ -f "$CKPT" ]] || { echo "[live-demo] ckpt not found: $CKPT (set MASKBET_CKPT=...)"; exit 1; }
  echo "[live-demo] starting MaskBeT server (ckpt=$(basename "$CKPT"), port=$PORT, decode=$SONIC_MASKBET_DECODE) ..."
  ( cd "$REPO_ROOT" && MASKBET_DIR="$REPO_ROOT/MaskBeT" nohup "$ENV_BIN/python" \
      "$REPO_ROOT/scripts/serve_maskbet_sonic.py" --ckpt "$CKPT" --port "$PORT" \
      --decode "$SONIC_MASKBET_DECODE" > "$SERVER_LOG" 2>&1 & )
  echo "[live-demo] loading the 25M model (GPF-prone box: auto-restarts on a crashed load) ..."
  ready=0
  for attempt in 1 2 3 4 5 6; do
    for _ in $(seq 1 24); do
      grep -q "SERVE_READY" "$SERVER_LOG" 2>/dev/null && { ready=1; break; }
      ps -eo cmd | grep -q "[s]erve_maskbet_sonic.py.*--port $PORT" || break  # died -> retry
      sleep 2
    done
    [[ "$ready" == "1" ]] && break
    echo "[live-demo] server load attempt $attempt died (kernel GPF burst?) — relaunching ..."
    ( cd "$REPO_ROOT" && MASKBET_DIR="$REPO_ROOT/MaskBeT" nohup "$ENV_BIN/python" \
        "$REPO_ROOT/scripts/serve_maskbet_sonic.py" --ckpt "$CKPT" --port "$PORT" \
        --decode "$SONIC_MASKBET_DECODE" > "$SERVER_LOG" 2>&1 & )
    sleep 3
  done
  [[ "$ready" == "1" ]] && echo "[live-demo] server ready." \
                        || { echo "[live-demo] server failed to come up — log:"; tail -15 "$SERVER_LOG"; exit 1; }
fi

# 2) play the flow against THIS server (bootstrap tokens just kickstart segment starts;
#    live tokens come from MaskBeT). Reuse the flow3 scaffolding + an existing pred dir.
export GR00T_PORT="$PORT"
export PKL="${PKL:-data/seg_flow3_all.pkl}"
export PRED_DIR="${PRED_DIR:-$REPO_ROOT/datasets/sonic_vla_pred_starvla_ce}"
echo "[live-demo] launching viewer for '$SEQ' (server :$PORT, bootstrap=$PRED_DIR) ..."
exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" "$SEQ"
