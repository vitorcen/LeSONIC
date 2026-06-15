#!/usr/bin/env bash
# LeSONIC auto-eval watcher (the SONIC analog of LeIsaac's eval_watcher): polls a FlowDP
# training run's checkpoints, scores each NEW one with the open-loop token-MSE similarity
# metric, appends a CSV, and rebuilds the ranked markdown leaderboard — live, while
# training is still running. No Isaac needed (open-loop is forward-pass only), so it can
# share the GPU with training cheaply.
#
#   bash scripts/sonic_flowdp_eval_watcher.sh           # follows outputs/flowdp_sonic
# Stops when training logs "End of training" and every ckpt is scored, or after MAX_IDLE
# polls with no new ckpt.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWHEADS="${FLOWHEADS:-$REPO_ROOT/dependencies/FlowHeads}"
PY="${PY:-$HOME/miniconda3/envs/lerobot-v044/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/flowdp_sonic}"
CSV="${CSV:-$OUTPUT_DIR/openloop_eval.csv}"
LEADERBOARD="${LEADERBOARD:-$OUTPUT_DIR/leaderboard.md}"
EPOCH_STEPS="${EPOCH_STEPS:-59.6}"     # batch64, 3815 frames
TRAIN_LOG="${TRAIN_LOG:-/tmp/flowdp_sonic_train.log}"
POLL="${POLL:-30}"
MAX_IDLE="${MAX_IDLE:-120}"            # *POLL seconds of no-new-ckpt before giving up

export PYTHONPATH="$FLOWHEADS:${PYTHONPATH:-}"
export HF_HUB_DISABLE_XET=1
EVAL="$REPO_ROOT/scripts/sonic_flowdp_openloop_eval.py"

score_one() {  # $1 = ckpt pretrained_model dir; retry the kernel-6.17 import corruption
  local ck="$1" a
  for a in 1 2 3 4 5 6; do
    "$PY" "$EVAL" --ckpt "$ck" --csv "$CSV" --epoch-steps "$EPOCH_STEPS" --skip-existing \
        && return 0
    echo "[eval-watch] retry $a scoring $ck"; sleep 3
  done
  return 1
}

scored() {  # is this step already in the CSV?
  local step="$1"
  [ -f "$CSV" ] && cut -d, -f1 "$CSV" | grep -qx "$step"
}

echo "[eval-watch] $(date +%H:%M:%S) following $OUTPUT_DIR  csv=$CSV  board=$LEADERBOARD"
idle=0
while true; do
  new=0
  for ck in $(ls -d "$OUTPUT_DIR"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort -t/ -k7 -n); do
    [ -f "$ck/model.safetensors" ] || continue
    step="$(basename "$(dirname "$ck")")"; step="$((10#$step))"
    scored "$step" && continue
    echo "[eval-watch] scoring step $step ..."
    if score_one "$ck"; then
      new=1
      "$PY" "$EVAL" --leaderboard-from-csv "$CSV" --leaderboard "$LEADERBOARD" --epoch-steps "$EPOCH_STEPS" || true
    fi
  done
  if [ "$new" = "1" ]; then idle=0; else idle=$((idle+1)); fi

  # done? training finished AND nothing left unscored
  if grep -q "End of training" "$TRAIN_LOG" 2>/dev/null; then
    left=0
    for ck in $(ls -d "$OUTPUT_DIR"/checkpoints/[0-9]*/pretrained_model 2>/dev/null); do
      step="$(basename "$(dirname "$ck")")"; step="$((10#$step))"
      scored "$step" || left=1
    done
    [ "$left" = "0" ] && { echo "[eval-watch] training done + all ckpts scored — exit."; break; }
  fi
  [ "$idle" -ge "$MAX_IDLE" ] && { echo "[eval-watch] idle $((idle*POLL))s, giving up."; break; }
  sleep "$POLL"
done
echo "[eval-watch] final leaderboard:"; cat "$LEADERBOARD" 2>/dev/null || true
