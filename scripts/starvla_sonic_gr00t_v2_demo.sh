#!/usr/bin/env bash
# Live closed-loop GUI demo of the StarVLA QwenGR00T_v2 SONIC BonesSeed model.
# Serves the trained head over ZMQ and drives the GEAR-SONIC WBC in Isaac (the same
# gear_sonic_sequence.sh machinery the FlowDP / CE demos use — we just point
# GR00T_PORT at this server). Real-time inference: the robot's motion_token chunks
# are produced live by StarVLA each step, not replayed.
#
#   bash scripts/starvla_sonic_gr00t_v2_demo.sh @flow2     # dance->walk->...->macarena loop
#   bash scripts/starvla_sonic_gr00t_v2_demo.sh squat,kick,jump   # ad-hoc sequence
#   bash scripts/starvla_sonic_gr00t_v2_demo.sh list
#
# Notes:
#   * serve runs in starvla_eval_qwen35 (transformers 5.2 for Qwen3.5); STARVLA_DIR
#     = the local fork (has QwenGR00T_N17 + select_layer/truncate).
#   * --stats = BonesSeed meta/stats.json (NOT flow3): gravity is all-degenerate
#     (min==max=0) so the serve's _norm masks it to 0 = the training value -> no blow-up.
#   * single-frame (proprio_history=0, as trained) -> like FlowDP, amplitude may be
#     modest; this shows the live model honestly.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_BIN="${ENV_BIN:-$HOME/miniconda3/envs/starvla_eval_qwen35/bin}"
export STARVLA_DIR="${STARVLA_DIR:-$REPO_ROOT/dependencies/starVLA}"
PORT="${GR00T_PORT:-5556}"
# default = best of the select_layer sweep = sl14 (read@14, unfreeze 10-13, macro 0.0236, 7/7).
# Override with STARVLA_CKPT=.../sonic_qwen3_5_4b_gr00t_v2_sl{10,12,14,15_uf6}/checkpoints/steps_<N>_pytorch_model.pt
CKPT="${STARVLA_CKPT:-$REPO_ROOT/outputs/starvla/sonic_qwen3_5_4b_gr00t_v2_sl14/checkpoints/steps_6000_pytorch_model.pt}"
STATS="${STARVLA_STATS:-$REPO_ROOT/datasets/sonic_vla_lerobot/meta/stats.json}"
SEQ="${1:-@flow2}"
ACTION_HORIZON="${ACTION_HORIZON:-40}"
PRED_DIR="${PRED_DIR:-$REPO_ROOT/datasets/sonic_vla_gt}"   # GT tokens = clean per-segment bootstrap
SERVER_LOG="${SERVER_LOG:-/tmp/starvla_gr00t_v2_serve.log}"

[[ "$SEQ" == "list" ]] && exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" list

# 1) ensure the StarVLA server is up on $PORT (idempotent; [s] bracket avoids self-match).
if pgrep -f "[s]erve_starvla_sonic.py.*--port $PORT" >/dev/null 2>&1; then
  echo "[gr00t_v2-demo] StarVLA server already on :$PORT — reusing it."
else
  [[ -f "$CKPT" ]]  || { echo "[gr00t_v2-demo] ckpt not found: $CKPT (set STARVLA_CKPT=...)"; exit 1; }
  [[ -f "$STATS" ]] || { echo "[gr00t_v2-demo] stats not found: $STATS"; exit 1; }
  echo "[gr00t_v2-demo] starting server (ckpt=$(basename "$CKPT"), stats=$(basename "$STATS"), port=$PORT) ..."
  ( setsid nohup "$ENV_BIN/python" "$REPO_ROOT/scripts/serve_starvla_sonic.py" \
      --ckpt "$CKPT" --stats "$STATS" --port "$PORT" > "$SERVER_LOG" 2>&1 & )
  # wait for readiness (model load ~30-60s; kernel-6.17 mmap load can segfault -> retry)
  ready=0
  for i in $(seq 1 90); do
    if grep -q "SERVE_READY" "$SERVER_LOG" 2>/dev/null; then ready=1; break; fi
    if ! pgrep -f "[s]erve_starvla_sonic.py.*--port $PORT" >/dev/null 2>&1; then
      echo "[gr00t_v2-demo] server died during load (retry $i); see $SERVER_LOG"
      ( setsid nohup "$ENV_BIN/python" "$REPO_ROOT/scripts/serve_starvla_sonic.py" \
          --ckpt "$CKPT" --stats "$STATS" --port "$PORT" > "$SERVER_LOG" 2>&1 & )
    fi
    sleep 2
  done
  [[ "$ready" == "1" ]] && echo "[gr00t_v2-demo] server ready on :$PORT." \
                        || { echo "[gr00t_v2-demo] server not ready; tail $SERVER_LOG:"; tail -20 "$SERVER_LOG"; exit 1; }
fi

# 2) play the flow against THIS server (Isaac GUI). injector is wire-agnostic.
echo "[gr00t_v2-demo] launching viewer for '$SEQ' (GR00T_PORT=$PORT horizon=$ACTION_HORIZON bootstrap=$(basename "$PRED_DIR")) ..."
exec env GR00T_PORT="$PORT" ACTION_HORIZON="$ACTION_HORIZON" PRED_DIR="$PRED_DIR" \
     bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" "$SEQ"
