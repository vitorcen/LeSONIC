#!/usr/bin/env bash
# ONE-CLICK live StarVLA-CE demo: bring up the StarVLA SONIC inference server (if not
# already running), then play a demo flow in the Isaac viewer. The Isaac side is the SAME
# machinery as the GR00T live demo (gear_sonic_sequence.sh + vla_live_injector) — the
# server just speaks the same ZMQ wire on its own port.
#
#   bash scripts/starvla_sonic_live_demo.sh @flow3      # fight/run/dance loop, CE model
#   bash scripts/starvla_sonic_live_demo.sh combat,circle
# Server is left running for the next flow; stop with: pkill -f serve_starvla_sonic
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${STARVLA_PORT:-5556}"
ENV_BIN="${ENV_BIN:-$HOME/miniconda3/envs/starvla_eval_qwen35/bin}"
# Authoritative live default = CE v1 (per-dim independent CE) + expected-T0.5 decode:
# open-loop MSE64 0.0125, best of the StarVLA family. The masked / masked_hist heads
# are NEGATIVE results (worse than the constant-mean template — doc/sonic_starvla_swap_brainstorm.html
# §11.5); point STARVLA_CKPT at them only to reproduce the negative archive. This default
# also exercises the §11.6 serve-state-permutation fix (right_arm proprio was frozen before).
CKPT="${STARVLA_CKPT:-$REPO_ROOT/outputs/starvla/sonic_qwen3_5_4b_ce/checkpoints/steps_6000_pytorch_model.pt}"
SERVER_LOG="${SERVER_LOG:-/tmp/starvla_sonic_server.log}"
SEQ="${1:-@flow3}"

[[ "$SEQ" == "list" ]] && exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" list

# Decode mode for the CE head (doc §11.3 sweep): expected-value at T=0.5 cuts MSE64
# 28% vs argmax while keeping ~88% of GT amplitude; snap keeps the wire on-grid.
export SONIC_CE_DECODE="${SONIC_CE_DECODE:-expected}"
export SONIC_CE_TEMP="${SONIC_CE_TEMP:-0.5}"
export SONIC_CE_SNAP="${SONIC_CE_SNAP:-1}"

# 1) ensure the StarVLA server is up on $PORT (idempotent)
if ps -eo cmd | grep -q "[s]erve_starvla_sonic.py.*--port $PORT"; then
  echo "[live-demo] StarVLA server already running on :$PORT — reusing it."
else
  [[ -f "$CKPT" ]] || { echo "[live-demo] ckpt not found: $CKPT (set STARVLA_CKPT=...)"; exit 1; }
  # History model needs --proprio-history 3
  HIST_FLAG=""
  [[ "$CKPT" == *"_hist"* ]] && HIST_FLAG="--proprio-history 3"
  echo "[live-demo] starting StarVLA server (ckpt=$(basename "$CKPT"), port=$PORT) ..."
  ( nohup "$ENV_BIN/python" "$REPO_ROOT/scripts/serve_starvla_sonic.py" \
      --ckpt "$CKPT" --port "$PORT" $HIST_FLAG > "$SERVER_LOG" 2>&1 & )
  echo "[live-demo] loading the VLM (~1 min, GPF-prone box: auto-restarts on a crashed load) ..."
  ready=0
  for attempt in 1 2 3 4 5 6; do
    for _ in $(seq 1 36); do
      grep -q "SERVE_READY" "$SERVER_LOG" 2>/dev/null && { ready=1; break; }
      ps -eo cmd | grep -q "[s]erve_starvla_sonic.py.*--port $PORT" || break  # died -> retry
      sleep 5
    done
    [[ "$ready" == "1" ]] && break
    echo "[live-demo] server load attempt $attempt died (kernel GPF burst?) — relaunching ..."
    ( nohup "$ENV_BIN/python" "$REPO_ROOT/scripts/serve_starvla_sonic.py" \
        --ckpt "$CKPT" --port "$PORT" $HIST_FLAG > "$SERVER_LOG" 2>&1 & )
    sleep 5
  done
  [[ "$ready" == "1" ]] && echo "[live-demo] server ready." \
                        || { echo "[live-demo] server failed to come up — log:"; tail -15 "$SERVER_LOG"; exit 1; }
fi

# 2) play the flow against THIS server, with CE tokens as bootstrap + flow3 scaffolding
export GR00T_PORT="$PORT"
export PKL="${PKL:-data/seg_flow3_all.pkl}"
export PRED_DIR="${PRED_DIR:-$REPO_ROOT/datasets/sonic_vla_pred_starvla_ce}"
echo "[live-demo] launching viewer for '$SEQ' (server :$PORT, bootstrap=$PRED_DIR) ..."
exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" "$SEQ"
