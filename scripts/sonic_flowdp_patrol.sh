#!/usr/bin/env bash
# Self-healing patrol for the FlowDP-SONIC run. Pure bash (no torch) so it is immune to
# the kernel-6.17 heap corruption that crashes the python training. It supervises:
#   - the training watchdog (which per-crash resumes, but gives up after MAX_RETRIES);
#     if the watchdog process dies while training is unfinished, the patrol relaunches it
#     (a fresh retry budget) -> effectively unlimited recovery through corruption bursts.
#   - the open-loop eval watcher (relaunched if it dies).
# Exits when training reaches STEPS or logs "End of training".
#
#   setsid bash scripts/sonic_flowdp_patrol.sh &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/flowdp_sonic}"
STEPS="${STEPS:-6000}"
SAVE_FREQ="${SAVE_FREQ:-240}"
TRAIN_LOG="${TRAIN_LOG:-/tmp/flowdp_sonic_train.log}"
POLL="${POLL:-120}"

last_step() {
  local f="$OUTPUT_DIR/checkpoints/last/training_state/training_step.json"
  [ -f "$f" ] && grep -oE '[0-9]+' "$f" | head -1 || echo 0
}

echo "[patrol] $(date +%H:%M:%S) supervising watchdog+evalwatch | target=$STEPS poll=${POLL}s"
while true; do
  if grep -q "End of training" "$TRAIN_LOG" 2>/dev/null; then
    echo "[patrol] $(date +%H:%M:%S) End of training — stop."; break
  fi
  st="$(last_step)"
  if [ "$st" -ge "$STEPS" ] 2>/dev/null; then
    echo "[patrol] $(date +%H:%M:%S) last ckpt $st >= $STEPS — stop."; break
  fi

  if ! pgrep -f "[s]onic_flowdp_watchdog" >/dev/null 2>&1; then
    echo "[patrol] $(date +%H:%M:%S) watchdog DOWN at step $st — relaunching (fresh retries)"
    STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" setsid bash "$REPO_ROOT/scripts/sonic_flowdp_watchdog.sh" \
        < /dev/null >> /tmp/flowdp_sonic_watchdog.log 2>&1 &
  fi
  if ! pgrep -f "[s]onic_flowdp_eval_watcher" >/dev/null 2>&1; then
    echo "[patrol] $(date +%H:%M:%S) eval-watcher DOWN — relaunching"
    setsid bash "$REPO_ROOT/scripts/sonic_flowdp_eval_watcher.sh" \
        < /dev/null >> /tmp/flowdp_sonic_evalwatch.log 2>&1 &
  fi
  echo "[patrol] $(date +%H:%M:%S) ok: step=$st watchdog=$(pgrep -cf '[s]onic_flowdp_watchdog') evalwatch=$(pgrep -cf '[s]onic_flowdp_eval_watcher')"
  sleep "$POLL"
done
echo "[patrol] done. final leaderboard:"; cat "$OUTPUT_DIR/leaderboard.md" 2>/dev/null || true
