#!/usr/bin/env bash
# ONE-CLICK live prompt-sequenced demo driven by the FlowDP head (instead of GR00T):
# brings up the FlowDP ZMQ server (lerobot-v044 env), then plays a demo flow in the
# Isaac viewer — one session, looping prompt after prompt with NO GUI/server restart.
#
# FlowDP speaks the same wire as the GR00T server, so the Isaac side
# (gear_sonic_sequence.sh) is reused verbatim; we just point GR00T_PORT at FlowDP.
#
# Usage:
#   bash scripts/gear_sonic_flowdp_demo.sh @flow2          # squat->walk->...->macarena loop
#   bash scripts/gear_sonic_flowdp_demo.sh @flow1
#   bash scripts/gear_sonic_flowdp_demo.sh squat,lunge,macarena   # ad-hoc
# Pick the ckpt:  FLOWDP_CKPT=outputs/flowdp_sonic/checkpoints/015000/pretrained_model
# Free the GPU afterwards:  kill the server PID printed below (or scripts/gear_sonic_stop.sh).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${GR00T_PORT:-5557}"
PY="${PY:-$HOME/miniconda3/envs/lerobot-v044/bin/python}"
FLOWHEADS="${FLOWHEADS:-$REPO_ROOT/dependencies/FlowHeads}"
FLOWDP_CKPT="${FLOWDP_CKPT:-$REPO_ROOT/outputs/flowdp_sonic/checkpoints/last/pretrained_model}"
SERVER_LOG="${SERVER_LOG:-/tmp/flowdp_sonic_server.log}"
FLOW_STEPS="${FLOW_STEPS:-}"               # optional Euler-NFE override at serve
ACTION_HORIZON="${ACTION_HORIZON:-32}"     # = trained n_action_steps
SEQ="${1:-@flow2}"

[[ "$SEQ" == "list" ]] && exec bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" list

# 1) Ensure the FlowDP server is up on $PORT (idempotent; [s] bracket avoids self-match).
if pgrep -f "[s]erve_flowdp_sonic.*--port $PORT" >/dev/null 2>&1; then
  echo "[flowdp-demo] FlowDP server already running on :$PORT — reusing it."
else
  if [[ ! -d "$FLOWDP_CKPT" ]]; then
    echo "[flowdp-demo] checkpoint not found: $FLOWDP_CKPT"
    echo "             train it (scripts/sonic_flowdp_train.sh) or set FLOWDP_CKPT=..."; exit 1
  fi
  echo "[flowdp-demo] starting FlowDP server (ckpt=$FLOWDP_CKPT, port=$PORT) ..."
  fs_args=(); [[ -n "$FLOW_STEPS" ]] && fs_args=(--flow-steps "$FLOW_STEPS")
  ( cd "$REPO_ROOT" && PYTHONPATH="$FLOWHEADS:${PYTHONPATH:-}" HF_HUB_DISABLE_XET=1 \
      setsid nohup "$PY" scripts/serve_flowdp_sonic.py \
      --ckpt "$FLOWDP_CKPT" --port "$PORT" "${fs_args[@]}" > "$SERVER_LOG" 2>&1 & )
  echo "[flowdp-demo] loading FlowDP (~20s) — watching $SERVER_LOG ..."
  ready=0
  for _ in $(seq 1 36); do
    if grep -q "SERVE_READY" "$SERVER_LOG" 2>/dev/null; then ready=1; break; fi
    if ! pgrep -f "[s]erve_flowdp_sonic.*--port $PORT" >/dev/null 2>&1; then
      echo "[flowdp-demo] server exited during load — last log:"; tail -20 "$SERVER_LOG"; exit 1
    fi
    sleep 3
  done
  [[ "$ready" == "1" ]] && echo "[flowdp-demo] server ready on :$PORT." \
                        || { echo "[flowdp-demo] server not ready; last log:"; tail -20 "$SERVER_LOG"; exit 1; }
fi

# 2) Play the flow — reuse the GR00T sequence launcher, pointed at the FlowDP port.
echo "[flowdp-demo] launching viewer for '$SEQ' (GR00T_PORT=$PORT action_horizon=$ACTION_HORIZON) ..."
exec env GR00T_PORT="$PORT" ACTION_HORIZON="$ACTION_HORIZON" \
     bash "$REPO_ROOT/scripts/gear_sonic_sequence.sh" "$SEQ"
